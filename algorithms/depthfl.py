"""
algorithms/depthfl.py
=====================
DepthFL: Depthwise Federated Learning (Kim et al., ICLR 2023) — official
baseline ported into the FedLab-ZMQ energy harness.

Faithful mechanics (paper Eq. 1 + Algorithm 1):
  * Each client k has a STATIC capacity level d_k ∈ {1..K} and trains the
    depth-d_k prefix of a multi-exit global model, with ALL classifiers ≤ d_k.
  * Local loss (Eq. 1):
        L = Σ_{i≤d} CE_i + (1/(d−1)) Σ_{i≤d} Σ_{j≠i} D_KL(p_j ‖ p_i)
    (mutual self-distillation among the on-path classifiers; the teacher side
    of every KL pair is detached, τ=1 as in the paper).
  * Aggregation: per-key UNIFORM mean over the clients whose prefix contains
    the key (shallow layers averaged over many, deep over few).
  * Optional FedDyn (paper default; `feddyn_alpha` > 0): client keeps a
    gradient state g_k on its prefix and minimises
        L − ⟨g_k, θ⟩ + (α/2)‖θ − θ^t‖²  ;  g_k ← g_k − α(θ̃ − θ^t)
    server keeps h and applies θ^{t+1} = mean − (1/α)h.
  * Inference: the paper reports the ENSEMBLE of all classifiers — use
    model `resnet8_ee_ens` in the config so the runner's standard evaluation
    measures exactly that.

Energy-harness adaptation (what "porting into our world" means):
  * The capacity level is assigned once (uniform split like the paper, or by
    initial-battery tier via `level_assign: battery`) and NEVER adapts —
    DepthFL is battery-blind by design; clients pay φ_{d_k}·F every round
    and die when the battery empties. Costs use the same measured-FLOPs
    machinery as every other algorithm in the harness (gate == bill).
"""

import gc
import math
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from hardware.flop_cost import round_compute_flops

from .base import AggregateResult, FLAlgorithm, register_algorithm
from .fedpart import _compute_group_flops, _derive_layer_groups
from .fedpart_be import _exit_flop_fractions


def _cumulative_exit_keys(model: nn.Module) -> list[list[str]]:
    """state_dict keys (params + buffers) of the depth-d prefix, per depth."""
    prefixes = model.EXIT_PREFIXES
    all_keys = list(model.state_dict().keys())
    out: list[list[str]] = []
    cum: list[str] = []
    for depth_prefixes in prefixes:
        cum = cum + list(depth_prefixes)
        keys = [
            k for k in all_keys
            if any(k == p or k.startswith(p + ".") for p in cum)
        ]
        out.append(keys)
    return out


@register_algorithm("depthfl")
class DepthFL(FLAlgorithm):
    """DepthFL (ICLR 2023): static depth-scaled submodels + mutual
    self-distillation + per-depth coverage aggregation (+ optional FedDyn)."""

    name = "depthfl"
    description = (
        "DepthFL (Kim et al., ICLR 2023): depth-pruned local models with "
        "multiple classifiers, mutual self-distillation, coverage "
        "aggregation, optional FedDyn. Battery-blind static baseline."
    )

    # ── client ──────────────────────────────────────────────────────────────
    def client_update(self, model, dataloader, state, config):
        device = config.get("device", "cpu")
        lr = float(config.get("lr", 0.003))
        local_epochs = int(config.get("local_epochs", 8))
        max_grad_norm = config.get("max_grad_norm", None)
        feddyn_alpha = float(config.get("feddyn_alpha", 0.0))
        num_exits = model.num_exits

        # ── static capacity level (assigned once, never adapts) ───────────
        if "depthfl_level" not in state.custom:
            assign = str(config.get("level_assign", "uniform"))
            if assign == "battery":
                lo = float(config.get("level_batt_min_j", 0.0))
                hi = float(config.get("level_batt_max_j", 0.0))
                span = max(hi - lo, 1e-9)
                tier = (state.battery_j - lo) / span
                lvl = min(num_exits, max(1, 1 + int(tier * num_exits)))
            else:  # uniform equal split, deterministic (paper default)
                lvl = (int(state.client_id) % num_exits) + 1
            state.custom["depthfl_level"] = lvl
        d = int(state.custom["depthfl_level"])

        # ── cached geometry + cost fractions (same machinery as FedSTEP) ──
        if "exit_keys" not in state.custom:
            state.custom["exit_keys"] = _cumulative_exit_keys(model)
            groups = _derive_layer_groups(model)
            gflops = _compute_group_flops(groups, model)
            # cumulative PARAM-name lists per depth (what _exit_flop_fractions
            # matches against the group param names)
            pnames = [n for n, _ in model.named_parameters()]
            exit_names, cum = [], []
            for pr in model.EXIT_PREFIXES:
                cum = cum + list(pr)
                exit_names.append(
                    [n for n in pnames
                     if any(n == p or n.startswith(p + ".") for p in cum)]
                )
            state.custom["exit_fractions"] = _exit_flop_fractions(
                exit_names, groups, gflops
            )
        fractions = state.custom["exit_fractions"]
        prefix_keys = state.custom["exit_keys"][d - 1]

        w_before = OrderedDict(
            {k: v.clone().cpu() for k, v in model.state_dict().items()
             if k in set(prefix_keys)}
        )
        w_global_flat = {k: v.to(device) for k, v in w_before.items()}

        model.train()
        model.to(device)

        # freeze everything beyond the prefix (never executed anyway)
        prefix_set = set(prefix_keys)
        trainable_params = []
        trainable_names = []
        for n, p in model.named_parameters():
            if n in prefix_set:
                p.requires_grad_(True)
                trainable_params.append(p)
                trainable_names.append(n)
            else:
                p.requires_grad_(False)

        optimizer_type = str(config.get("optimizer", "sgd")).lower()
        weight_decay = float(config.get("weight_decay", 1e-4))
        if optimizer_type == "adam":
            optimizer = optim.Adam(trainable_params, lr=lr,
                                   weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(trainable_params, lr=lr,
                                  momentum=float(config.get("momentum", 0.9)),
                                  weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        # FedDyn gradient state (prefix params only)
        g_state = None
        if feddyn_alpha > 0.0:
            g_state = state.custom.setdefault(
                "feddyn_g",
                {n: torch.zeros_like(p, device=device)
                 for n, p in model.named_parameters() if n in prefix_set},
            )
            named = dict(model.named_parameters())

        total_loss = total_ce = total_kd = 0.0
        num_batches = 0
        for _ in range(local_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outs = model.forward_exits_upto(x, d)

                ce = sum(criterion(outs[i], y) for i in range(1, d + 1))
                kd = torch.zeros((), device=device)
                if d >= 2:
                    logp = {i: torch.log_softmax(outs[i], dim=1)
                            for i in range(1, d + 1)}
                    p = {i: torch.softmax(outs[i], dim=1).detach()
                         for i in range(1, d + 1)}
                    for i in range(1, d + 1):
                        for j in range(1, d + 1):
                            if j == i:
                                continue
                            kd = kd + nn.functional.kl_div(
                                logp[i], p[j], reduction="batchmean"
                            )
                    kd = kd / (d - 1)
                loss = ce + kd

                if feddyn_alpha > 0.0:
                    lin = torch.zeros((), device=device)
                    prox = torch.zeros((), device=device)
                    for n in trainable_names:
                        theta = named[n]
                        lin = lin + torch.sum(g_state[n] * theta)
                        prox = prox + torch.sum(
                            (theta - w_global_flat[n]) ** 2
                        )
                    loss = loss - lin + 0.5 * feddyn_alpha * prox

                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, max_norm=max_grad_norm
                    )
                optimizer.step()
                total_loss += loss.item()
                total_ce += ce.item()
                total_kd += float(kd.detach())
                num_batches += 1

        if feddyn_alpha > 0.0:
            with torch.no_grad():
                for n in trainable_names:
                    g_state[n] -= feddyn_alpha * (
                        named[n].detach() - w_global_flat[n]
                    )

        # restore grads for the runner's next use of this model object
        for _, p in model.named_parameters():
            p.requires_grad_(True)

        current_sd = model.state_dict()
        delta = OrderedDict(
            {k: (w_before[k] - current_sd[k].cpu()).float()
             for k in w_before}
        )
        del w_before, current_sd, w_global_flat
        gc.collect()

        # ── energy accounting: full measured round × depth fraction ───────
        profile = config.get("device_profile")
        uplink_bytes = self.count_bytes(delta, sparse=False)
        full_bytes = self.count_bytes(model.state_dict(), sparse=False)
        if profile is not None:
            if "full_flops" not in state.custom:
                state.custom["full_flops"] = round_compute_flops(
                    model, [n for n, _ in model.named_parameters()],
                    config, profile, dataloader, local_epochs,
                )
            flops = state.custom["full_flops"] * fractions[d - 1]
            _bd = profile.round_energy_breakdown(
                flops, uplink_bytes, full_bytes,
                config.get("energy_scale_factor", 1.0),
                config.get("alpha_applies_to", "compute"),
            )
        else:
            _e = fractions[d - 1]
            _bd = {"compute": _e, "uplink": 0.0, "downlink": 0.0, "total": _e}
        energy_j = _bd["total"]
        state.battery_j = max(0.0, state.battery_j - energy_j)
        state.round_num += 1
        del optimizer

        metadata = {
            "client_id": state.client_id,
            "round_num": state.round_num,
            "beta_actual": 1.0,
            "battery_j_remaining": state.battery_j,
            "energy_j_consumed": energy_j,
            "energy_compute_j": _bd["compute"],
            "energy_uplink_j": _bd["uplink"],
            "energy_downlink_j": _bd["downlink"],
            "bytes_sent": uplink_bytes,
            "bytes_received": full_bytes,
            "local_loss": total_ce / max(num_batches, 1),
            "kd_loss": total_kd / max(num_batches, 1),
            "compression_ratio": uplink_bytes / max(full_bytes, 1),
            "exit_depth_k": d,
            "dataset_size": len(dataloader.dataset),
        }
        return dict(delta), metadata

    # ── server ──────────────────────────────────────────────────────────────
    def server_aggregate(self, global_model, client_updates, round_num, config):
        K = len(client_updates)
        feddyn_alpha = float(config.get("feddyn_alpha", 0.0))
        num_clients_total = int(config.get("num_clients", K))
        global_sd = global_model.state_dict()
        new_weights = OrderedDict(
            {k: v.clone().float() for k, v in global_sd.items()}
        )

        sums: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = defaultdict(int)
        for update, meta, _ in client_updates:
            if not update:
                continue
            for k, v in update.items():
                if k in sums:
                    sums[k] += v.float()
                else:
                    sums[k] = v.clone().float()
                counts[k] += 1

        if feddyn_alpha > 0.0:
            if not hasattr(self, "_h"):
                self._h = {k: torch.zeros_like(v) for k, v in new_weights.items()}
            for k, s_ in sums.items():
                # δ = θ^t − θ̃  ⇒  h ← h + (α/m)·Σδ
                self._h[k] += (feddyn_alpha / num_clients_total) * s_

        for k, s_ in sums.items():
            mean_delta = s_ / counts[k]
            new_weights[k] = new_weights[k] - mean_delta.to(new_weights[k].device)
            if feddyn_alpha > 0.0:
                new_weights[k] = new_weights[k] - (1.0 / feddyn_alpha) * (
                    self._h[k].to(new_weights[k].device)
                )
        del sums
        gc.collect()

        lvl_hist: dict[int, int] = defaultdict(int)
        for _, m, _ in client_updates:
            lvl_hist[int(m.get("exit_depth_k", 0))] += 1
        total_bytes = sum(m["bytes_sent"] for _, m, _ in client_updates)
        total_energy = sum(m["energy_j_consumed"] for _, m, _ in client_updates)

        return AggregateResult(
            new_weights=new_weights,
            metrics={
                "round": round_num,
                "total_bytes_sent": total_bytes,
                "total_energy_j": total_energy,
                "avg_beta": 1.0,
                "avg_battery_j": sum(s.battery_j for _, _, s in client_updates)
                / max(K, 1),
                "avg_local_loss": sum(m["local_loss"] for _, m, _ in client_updates)
                / max(K, 1),
                "participation_rate": 1.0,
                "jain_index": 1.0,
                "num_clients": K,
                "exit_histogram": dict(sorted(lvl_hist.items())),
                "exit_mode": "depthfl_static",
            },
        )

    def get_default_config(self):
        return {
            "lr": 0.003,
            "optimizer": "adam",
            "local_epochs": 8,
            "batch_size": 32,
            "device": "cpu",
            "device_profile": None,
            # level_assign: "uniform" (paper: equal client count per level)
            # or "battery" (initial-battery tier — our energy-world variant;
            # requires level_batt_min_j / level_batt_max_j).
            "level_assign": "uniform",
            "level_batt_min_j": 0.0,
            "level_batt_max_j": 0.0,
            # feddyn_alpha: 0 = FedAvg aggregation (DepthFL(FedAvg)+SD);
            # 0.1 = full DepthFL (paper default; use optimizer: sgd, lr 0.1).
            "feddyn_alpha": 0.0,
        }
