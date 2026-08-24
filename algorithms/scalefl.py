"""
algorithms/scalefl.py
=====================
ScaleFL: Resource-Adaptive Federated Learning with Heterogeneous Clients
(Ilhan et al., CVPR 2023) — official baseline ported into the FedLab-ZMQ
energy harness.

Faithful mechanics:
  * 2-D split config (§3.1): L complexity levels with target cost ratios
    r_l (halving heuristic: 0.25 / 0.5 / 1.0 for L=3). For each level a grid
    search picks the most UNIFORM (depth d, width w) pair whose parameter
    cost matches r_l within tolerance ε (Eq. 1). Width scaling keeps the
    upper-left submatrix of every hidden weight (§3.1.2, HeteroFL-style).
  * Local training (§3.3, Eq. 5): all exits of the local submodel trained
    jointly with depth-weighted CE + self-distillation from the DEEPEST
    local exit (detached teacher) at temperature τ:
        L = 1/(l(l+1)) Σ_{i=1..l} i · ( β·τ²·KL(exit_i ‖ exit_l) + CE_i )
  * Aggregation (§3.2, Eq. 2): per-ELEMENT coverage averaging — each weight
    element is averaged over exactly the clients whose slice contains it
    (nested upper-left shells).
  * Static level per client, equal split across levels (paper default);
    battery-tier assignment available for our energy-world variant.

Energy-harness adaptation: battery-blind — a level-l client pays the FLOPs
fraction of its (d, w) submodel every round (analytic w-scaling of measured
full-model FLOPs; gate == bill) and dies when the battery empties.
"""

import gc
import math
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from hardware.flop_cost import round_compute_flops
from models.registry import EarlyExitResNet8

from .base import AggregateResult, FLAlgorithm, register_algorithm
from .fedpart import _compute_group_flops, _derive_layer_groups
from .fedpart_be import _exit_flop_fractions


def _slice_to(t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Upper-left slice of `t` down to `shape` (per-dim prefix)."""
    if t.dim() == 0:
        return t.clone()
    return t[tuple(slice(0, s) for s in shape)].clone()


def _cumulative_exit_keys(model: nn.Module) -> list[list[str]]:
    prefixes = model.EXIT_PREFIXES
    all_keys = list(model.state_dict().keys())
    out, cum = [], []
    for depth_prefixes in prefixes:
        cum = cum + list(depth_prefixes)
        out.append(
            [k for k in all_keys
             if any(k == p or k.startswith(p + ".") for p in cum)]
        )
    return out


@register_algorithm("scalefl")
class ScaleFL(FLAlgorithm):
    """ScaleFL (CVPR 2023): 2-D depth×width submodels + self-distillation +
    per-element coverage aggregation. Battery-blind static baseline."""

    name = "scalefl"
    description = (
        "ScaleFL (Ilhan et al., CVPR 2023): 2-D (depth, width) split by "
        "budget ratios, upper-left weight slicing, exit self-distillation, "
        "per-element coverage aggregation."
    )

    # ── level table (server-side, computed once) ────────────────────────────
    def _compute_levels(self, model, config) -> list[dict]:
        ratios = list(config.get("scalefl_ratios", [0.25, 0.5, 1.0]))
        eps = float(config.get("scalefl_eps", 1.0))
        num_exits = model.num_exits
        num_classes = model.fc.out_features
        in_ch = next(model.stem.parameters()).shape[1]
        g_full = _derive_layer_groups(model)
        f_full = max(sum(_compute_group_flops(g_full, model)), 1e-9)

        def cost(d: int, w: float) -> float:
            """Analytic FLOPs fraction of the (d, w) submodel — the SAME
            quantity our energy accounting bills (paper §3.1 allows #FLOPS)."""
            local = EarlyExitResNet8(num_classes=num_classes,
                                     in_channels=in_ch, width_mult=w)
            g = _derive_layer_groups(local)
            gf = _compute_group_flops(g, local)
            pnames = [n for n, _ in local.named_parameters()]
            cum: list[str] = []
            for pr in local.EXIT_PREFIXES[:d]:
                cum = cum + list(pr)
            keep = set(
                n for n in pnames
                if any(n == p or n.startswith(p + ".") for p in cum)
            )
            prefix = sum(
                f for grp, f in zip(g, gf) if any(n in keep for n in grp)
            )
            del local
            return prefix / f_full

        levels: list[dict] = []
        widths = [round(0.25 + 0.05 * i, 2) for i in range(16)]  # 0.25..1.0
        for li, r in enumerate(ratios):
            if li == len(ratios) - 1:
                levels.append({"depth": num_exits, "width": 1.0, "ratio": 1.0})
                continue
            best, best_score = None, 1e9
            for d in range(1, num_exits + 1):
                for w in widths:
                    c = cost(d, w)
                    if abs(c / r - 1.0) <= eps:
                        # budget fit first, uniformity (Eq. 1) as tiebreak —
                        # on coarse 3-exit models pure uniformity degenerates.
                        score = abs(c / r - 1.0) + 0.1 * abs(d / num_exits - w)
                        if score < best_score:
                            best_score, best = score, {
                                "depth": d, "width": w, "ratio": round(c, 3)
                            }
            levels.append(best or {"depth": 1, "width": 1.0,
                                   "ratio": round(cost(1, 1.0), 3)})
        return levels

    # ── client ──────────────────────────────────────────────────────────────
    def client_update(self, model, dataloader, state, config):
        device = config.get("device", "cpu")
        lr = float(config.get("lr", 0.003))
        local_epochs = int(config.get("local_epochs", 8))
        max_grad_norm = config.get("max_grad_norm", None)
        tau = float(config.get("scalefl_temp", 3.0))
        beta = float(config.get("scalefl_beta", 0.1))

        if not hasattr(self, "_levels"):
            self._levels = self._compute_levels(model, config)
        levels = self._levels
        L = len(levels)

        if "scalefl_level" not in state.custom:
            assign = str(config.get("level_assign", "uniform"))
            if assign == "battery":
                lo = float(config.get("level_batt_min_j", 0.0))
                hi = float(config.get("level_batt_max_j", 0.0))
                span = max(hi - lo, 1e-9)
                tier = (state.battery_j - lo) / span
                lvl = min(L - 1, max(0, int(tier * L)))
            else:
                lvl = int(state.client_id) % L
            state.custom["scalefl_level"] = lvl
        lvl = int(state.custom["scalefl_level"])
        d, w = levels[lvl]["depth"], levels[lvl]["width"]

        num_classes = model.fc.out_features
        in_ch = next(model.stem.parameters()).shape[1]

        # ── build the width-scaled local model, load sliced global weights ─
        local = EarlyExitResNet8(num_classes=num_classes, in_channels=in_ch,
                                 width_mult=w)
        gsd = model.state_dict()
        lsd = local.state_dict()
        prefix_keys = _cumulative_exit_keys(local)[d - 1]
        prefix_set = set(prefix_keys)
        sliced = OrderedDict(
            {k: _slice_to(gsd[k], lsd[k].shape) for k in lsd}
        )
        local.load_state_dict(sliced)
        w_before = OrderedDict(
            {k: sliced[k].clone().cpu() for k in prefix_keys}
        )

        local.train()
        local.to(device)
        trainable_params = []
        for n, p in local.named_parameters():
            if n in prefix_set:
                p.requires_grad_(True)
                trainable_params.append(p)
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

        total_ce = total_kd = 0.0
        num_batches = 0
        norm = 1.0 / (d * (d + 1))
        for _ in range(local_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outs = local.forward_exits_upto(x, d)
                teacher = torch.softmax(outs[d].detach() / tau, dim=1)
                loss = torch.zeros((), device=device)
                ce_acc = kd_acc = 0.0
                for i in range(1, d + 1):
                    ce_i = criterion(outs[i], y)
                    ce_acc += float(ce_i)
                    term = ce_i
                    if i < d:
                        kd_i = nn.functional.kl_div(
                            torch.log_softmax(outs[i] / tau, dim=1),
                            teacher, reduction="batchmean",
                        ) * (tau * tau)
                        kd_acc += float(kd_i)
                        term = term + beta * kd_i
                    loss = loss + i * term
                loss = loss * norm
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, max_norm=max_grad_norm
                    )
                optimizer.step()
                total_ce += ce_acc / d
                total_kd += kd_acc / max(d - 1, 1)
                num_batches += 1

        current = local.state_dict()
        delta = OrderedDict(
            {k: (w_before[k] - current[k].cpu()).float()
             for k in prefix_keys}
        )
        del w_before, current, sliced, lsd
        gc.collect()

        # ── energy: measured full round × analytic (d, w) fraction ────────
        profile = config.get("device_profile")
        uplink_bytes = self.count_bytes(delta, sparse=False)
        full_bytes = self.count_bytes(gsd, sparse=False)
        cache_key = f"frac_l{lvl}"
        if cache_key not in state.custom:
            g_full = _derive_layer_groups(model)
            f_full = sum(_compute_group_flops(g_full, model))
            g_loc = _derive_layer_groups(local)
            gf_loc = _compute_group_flops(g_loc, local)
            pnames = [n for n, _ in local.named_parameters()]
            exit_names, cum = [], []
            for pr in local.EXIT_PREFIXES:
                cum = cum + list(pr)
                exit_names.append(
                    [n for n in pnames
                     if any(n == p or n.startswith(p + ".") for p in cum)]
                )
            fr_loc = _exit_flop_fractions(exit_names, g_loc, gf_loc)
            state.custom[cache_key] = (
                sum(gf_loc) * fr_loc[d - 1] / max(f_full, 1e-9)
            )
        frac = min(1.0, state.custom[cache_key])
        if profile is not None:
            if "full_flops" not in state.custom:
                state.custom["full_flops"] = round_compute_flops(
                    model, [n for n, _ in model.named_parameters()],
                    config, profile, dataloader, local_epochs,
                )
            _bd = profile.round_energy_breakdown(
                state.custom["full_flops"] * frac,
                uplink_bytes, full_bytes,
                config.get("energy_scale_factor", 1.0),
                config.get("alpha_applies_to", "compute"),
            )
        else:
            _bd = {"compute": frac, "uplink": 0.0, "downlink": 0.0,
                   "total": frac}
        energy_j = _bd["total"]
        state.battery_j = max(0.0, state.battery_j - energy_j)
        state.round_num += 1
        del optimizer, local
        gc.collect()

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
            "scalefl_width": w,
            "dataset_size": len(dataloader.dataset),
        }
        return dict(delta), metadata

    # ── server: per-element coverage averaging (Eq. 2) ──────────────────────
    def server_aggregate(self, global_model, client_updates, round_num, config):
        K = len(client_updates)
        global_sd = global_model.state_dict()
        new_weights = OrderedDict(
            {k: v.clone().float() for k, v in global_sd.items()}
        )

        sums: dict[str, torch.Tensor] = {}
        counts: dict[str, torch.Tensor] = {}
        for update, meta, _ in client_updates:
            if not update:
                continue
            for k, v in update.items():
                ref = new_weights[k]
                if k not in sums:
                    sums[k] = torch.zeros_like(ref)
                    counts[k] = torch.zeros_like(ref)
                region = tuple(slice(0, s) for s in v.shape) if v.dim() else ...
                sums[k][region] += v.float().to(ref.device)
                counts[k][region] += 1.0

        for k, s_ in sums.items():
            c = counts[k]
            mask = c > 0
            upd = torch.zeros_like(s_)
            upd[mask] = s_[mask] / c[mask]
            new_weights[k] = new_weights[k] - upd
        del sums, counts
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
                "exit_mode": "scalefl_static",
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
            # target cost-reduction ratios per level (halving heuristic,
            # paper §3.1: r_l = 0.5·r_{l+1}, last = full model).
            "scalefl_ratios": [0.25, 0.5, 1.0],
            "scalefl_eps": 1.0,
            # Eq. 5 hyperparameters (paper: τ=3, β=0.1 for images)
            "scalefl_temp": 3.0,
            "scalefl_beta": 0.1,
            # level_assign: "uniform" (paper) or "battery" (energy-world)
            "level_assign": "uniform",
            "level_batt_min_j": 0.0,
            "level_batt_max_j": 0.0,
        }
