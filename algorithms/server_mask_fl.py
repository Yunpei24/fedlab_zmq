"""
algorithms/server_mask_fl.py
=============================
Server-Mask FL: Battery-Adaptive Parameter-Level Masking with Gradient Importance

Nikiema & Amhoud, UM6P 2026

Core idea:
    Instead of assigning fixed layer groups (FedStep), the server computes a
    personalized parameter-level mask for each client based on historical gradient
    importance scores (EMA). Clients with low battery train fewer parameter tensors
    (small β_k), high-battery clients train more tensors (large β_k).

Key differences from FedStep:
    - Mask granularity: parameter tensors (e.g., conv1.weight, bn1.bias) vs layer groups
    - Mask assignment: continuous β_k(battery) vs discrete tiers
    - Mask selection: gradient-importance-based vs cost-based
    - Exploration: ε-greedy random tensor selection to prevent starvation

Convergence:
    Non-convex O(1/√T) under standard assumptions (L-smooth, bounded variance).
    Personalized masks reduce client drift vs random sparsification.

Energy signal:
    Battery level → β_k ∈ [β_min, β_max] → number of active parameter tensors
"""

import gc
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict, defaultdict
import random as rnd

from .base import FLAlgorithm, ClientState, AggregateResult, register_algorithm
from hardware.profiles import DeviceProfile
from hardware.flop_cost import round_compute_flops


@register_algorithm("server_mask")
class ServerMaskFL(FLAlgorithm):
    """
    Server-Mask FL: Gradient-importance-based parameter masking with battery-adaptive sparsity.

    Key properties:
      - Parameter-level masks (finer than layer groups)
      - Continuous β_k from battery level
      - Server-side importance tracking (EMA of gradient magnitudes)
      - ε-greedy exploration to prevent tensor starvation
      - Sparse aggregation (each client updates different tensors)
    """
    name = "server_mask"
    description = "Server-Mask FL: Battery-adaptive parameter masking with gradient importance"

    def client_update(self, model, dataloader, state, config):
        device = config.get("device", "cpu")
        lr = config.get("lr", 0.01)
        momentum = config.get("momentum", 0.9)
        weight_decay = config.get("weight_decay", 1e-4)
        local_epochs = config.get("local_epochs", 5)
        max_grad_norm = config.get("max_grad_norm", 10.0)
        warmup_rounds = config.get("warmup_rounds", 5)

        # ── Warmup vs masked training ─────────────────────────────────────────
        mask_names = state.custom.get("mask_names", None)
        is_warmup = (state.round_num < warmup_rounds) or (mask_names is None)

        # Build named-param list once (avoids repeated generator creation)
        named_params = list(model.named_parameters())
        trainable_param_names = {n for n, _ in named_params}
        bn_buffer_names = {
            n for n in model.state_dict()
            if n.endswith(('.running_mean', '.running_var'))
        }

        # ── Determine active params ───────────────────────────────────────────
        if is_warmup:
            active_param_names = trainable_param_names
        else:
            active_param_names = mask_names & trainable_param_names
            if not active_param_names:
                active_param_names = trainable_param_names

        # ── Snapshot only delta tensors (not full model) ──────────────────────
        # delta = active params + BN buffers (BN stats always aggregated)
        delta_names = active_param_names | bn_buffer_names
        sd = model.state_dict()
        w_before = {n: sd[n].clone().cpu() for n in delta_names}
        del sd

        model.train()
        model.to(device)

        # ── Apply mask: only flip requires_grad when it actually changes ──────
        # Avoids PyTorch autograd graph invalidation for unchanged tensors.
        if is_warmup:
            for _, param in named_params:
                if not param.requires_grad:
                    param.requires_grad_(True)
        else:
            for name, param in named_params:
                should_train = name in active_param_names
                if param.requires_grad != should_train:
                    param.requires_grad_(should_train)

        # ── Precompute active param lists once (used in optimizer + prox) ─────
        active_named_params = [(n, p) for n, p in named_params if n in active_param_names]
        active_params_only = [p for _, p in active_named_params]

        optimizer_type = config.get("optimizer", "sgd").lower()
        if optimizer_type == "adam":
            optimizer = optim.Adam(active_params_only, lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(
                active_params_only, lr=lr, momentum=momentum, weight_decay=weight_decay,
            )
        criterion = nn.CrossEntropyLoss()

        # ── Proximal reference (snapshot of active params at round start) ─────
        mu_weight = config.get("mu_weight", 0.0) if not is_warmup else 0.0
        if mu_weight > 0:
            w_global_ref = {n: p.detach().clone() for n, p in active_named_params}
        else:
            w_global_ref = {}

        # Precompute total elements once (used for beta and energy)
        total_elements = sum(p.numel() for _, p in named_params)

        # ── Local training ────────────────────────────────────────────────────
        total_loss, num_batches = 0.0, 0
        for _ in range(local_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                # Proximal term: loop only active params (not all named_params)
                if mu_weight > 0:
                    for n, p in active_named_params:
                        loss = loss + (mu_weight / 2.0) * (
                            p - w_global_ref[n].to(p.device)
                        ).pow(2).sum()
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(active_params_only, max_grad_norm)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        # ── Compute delta ─────────────────────────────────────────────────────
        current_sd = {n: v for n, v in model.state_dict().items() if n in delta_names}
        delta = OrderedDict()
        for name in delta_names:
            if name in w_before and name in current_sd:
                delta[name] = (w_before[name] - current_sd[name].cpu()).float()

        del w_before, current_sd, optimizer

        # ── Energy accounting ─────────────────────────────────────────────────
        active_elements = sum(p.numel() for _, p in active_named_params)
        beta_actual = active_elements / max(total_elements, 1)

        # Dense tensor transmission (4 bytes/elem — whole tensors, not sparse elements)
        uplink_bytes = sum(t.numel() * 4 for t in delta.values())
        downlink_bytes = total_elements * 4  # full model received from server

        profile: DeviceProfile = config.get("device_profile")
        if profile is not None:
            # Trainable set = the scattered tensors carrying gradients this round.
            # phi mode falls back to the legacy (1+beta)/2 formula via the
            # beta_fraction hint; measured/corrected capture the real backward
            # (which is near-full because grad_input must propagate through every
            # layer regardless of mask scatter — exactly the cost (1+beta)/2 misses).
            trainable_names = [n for n, _ in active_named_params]
            effective_flops = round_compute_flops(
                model, trainable_names, config,
                profile, dataloader, local_epochs,
                beta_fraction=beta_actual,
            )
            _bd = profile.round_energy_breakdown(
                effective_flops,
                uplink_bytes,
                downlink_bytes,
                config.get("energy_scale_factor", 1.0),
                config.get("alpha_applies_to", "compute"),
            )
        else:
            _e = (0.5 + 2.0 * beta_actual) * config.get("energy_scale_factor", 1.0)
            _bd = {"compute": _e, "uplink": 0.0, "downlink": 0.0, "total": _e}

        energy_j = _bd["total"]

        # ── Battery update ────────────────────────────────────────────────────
        state.battery_j = max(0.0, state.battery_j - energy_j)
        state.round_num += 1

        metadata = {
            "client_id": state.client_id,
            "round_num": state.round_num,
            "beta_actual": beta_actual,
            "battery_j_remaining": state.battery_j,
            "energy_j_consumed": energy_j,
            "energy_compute_j": _bd["compute"],
            "energy_uplink_j": _bd["uplink"],
            "energy_downlink_j": _bd["downlink"],
            "bytes_sent": uplink_bytes,
            "bytes_received": downlink_bytes,
            "local_loss": total_loss / max(num_batches, 1),
            "compression_ratio": beta_actual,
            "is_warmup": is_warmup,
            "active_params": list(active_param_names),
            "dataset_size": len(dataloader.dataset),
        }

        return dict(delta), metadata

    def server_aggregate(self, global_model, client_updates, round_num, config):
        K = len(client_updates)
        global_sd = global_model.state_dict()
        warmup_rounds = config.get("warmup_rounds", 5)
        is_warmup = round_num < warmup_rounds

        # ── Initialize server state ───────────────────────────────────────────
        if not hasattr(self, "_server_state"):
            self._server_state = {}

        # Cache static model structure (computed ONCE for entire experiment)
        if "_all_names" not in self._server_state:
            all_names = [n for n, _ in global_model.named_parameters()]
            param_sizes = {n: global_sd[n].numel() for n in all_names}
            total_elements = sum(param_sizes.values())
            head_names = {
                n for n in all_names
                if n.startswith(("fc.", "linear.", "classifier.", "head."))
            }
            self._server_state["_all_names"] = all_names
            self._server_state["_param_sizes"] = param_sizes
            self._server_state["_total_elements"] = total_elements
            self._server_state["_head_names"] = head_names
            self._server_state["_importance"] = defaultdict(
                lambda: {n: 1.0 for n in all_names}
            )
        else:
            all_names = self._server_state["_all_names"]
            param_sizes = self._server_state["_param_sizes"]
            total_elements = self._server_state["_total_elements"]
            head_names = self._server_state["_head_names"]

        # Auto-calibrate battery_max_j from fleet at round 0
        if "_battery_max_j" not in self._server_state:
            observed_max = max(s.battery_j for _, _, s in client_updates)
            config_override = config.get("battery_max_j")
            self._server_state["_battery_max_j"] = (
                float(config_override) if config_override is not None
                else observed_max / 0.95
            )

        # ── EMA importance update (skip BN buffers — not used for mask) ───────
        ema_alpha = config.get("ema_alpha", 0.3)
        one_minus_alpha = 1.0 - ema_alpha
        for update, meta, state in client_updates:
            cid = state.client_id
            imp = self._server_state["_importance"][cid]
            for name, delta in update.items():
                if name.endswith(('.running_mean', '.running_var')):
                    continue
                score = delta.float().pow(2).mean().item()
                imp[name] = one_minus_alpha * imp.get(name, 1.0) + ema_alpha * score

        # ── Aggregate sparse updates (dataset-size weighted) ──────────────────
        sizes = [m.get("dataset_size", 1) for _, m, _ in client_updates]
        total_n = max(sum(sizes), 1)

        param_weighted_sums = {}
        param_total_weights = {}
        bn_weighted_sums = {}

        for (update, meta, _), n_k in zip(client_updates, sizes):
            w_k = n_k / total_n
            for name, delta in update.items():
                delta_f = delta.float()
                if name.endswith((".running_mean", ".running_var")):
                    if name not in bn_weighted_sums:
                        bn_weighted_sums[name] = delta_f * w_k
                    else:
                        bn_weighted_sums[name] += delta_f * w_k
                else:
                    if name not in param_weighted_sums:
                        param_weighted_sums[name] = delta_f * n_k
                        param_total_weights[name] = n_k
                    else:
                        param_weighted_sums[name] += delta_f * n_k
                        param_total_weights[name] += n_k

        # server_lr=1.0 during warmup (standard FedAvg), scaled post-warmup
        server_lr = 1.0 if is_warmup else (config.get("server_lr") or 0.5)
        new_weights = OrderedDict()
        for name in global_sd:
            if name in param_weighted_sums:
                avg_delta = param_weighted_sums[name] / max(param_total_weights[name], 1)
                new_weights[name] = (
                    global_sd[name].float()
                    - server_lr * avg_delta.to(global_sd[name].device)
                )
            elif name in bn_weighted_sums:
                new_weights[name] = (
                    global_sd[name].float()
                    - bn_weighted_sums[name].to(global_sd[name].device)
                )
            else:
                new_weights[name] = global_sd[name].float()

        # ── Compute masks for NEXT round ──────────────────────────────────────
        if not is_warmup:
            beta_min = config.get("beta_min", 0.1)
            beta_max = config.get("beta_max", 0.5)
            battery_max_j = self._server_state["_battery_max_j"]
            epsilon_explore = config.get("epsilon_explore", 0.05)

            for update, meta, state in client_updates:
                if state.battery_j <= 0:
                    state.custom["mask_names"] = set()
                    continue

                beta_k = beta_min + (beta_max - beta_min) * min(
                    state.battery_j / battery_max_j, 1.0
                )
                element_budget = max(1, int(beta_k * total_elements))
                greedy_budget = max(1, int((1 - epsilon_explore) * element_budget))
                random_budget = element_budget - greedy_budget

                cid = state.client_id
                scores = self._server_state["_importance"][cid]

                # Greedy selection: highest-importance tensors first, within element budget
                sorted_names = sorted(all_names, key=lambda n: scores.get(n, 1.0), reverse=True)
                greedy_selected = set()
                used_elements = sum(param_sizes[n] for n in head_names if n in param_sizes)

                for name in sorted_names:
                    if name in head_names:
                        continue
                    sz = param_sizes[name]
                    if used_elements + sz <= greedy_budget:
                        greedy_selected.add(name)
                        used_elements += sz

                # ε-greedy random exploration on remaining budget
                random_selected = set()
                if random_budget > 0:
                    remaining = [
                        n for n in all_names
                        if n not in greedy_selected and n not in head_names
                    ]
                    r = rnd.Random(round_num * 10000 + state.client_id)
                    r.shuffle(remaining)
                    used_rand = 0
                    for name in remaining:
                        sz = param_sizes[name]
                        if used_rand + sz <= random_budget:
                            random_selected.add(name)
                            used_rand += sz

                state.custom["mask_names"] = greedy_selected | random_selected | head_names

        # ── Metrics ───────────────────────────────────────────────────────────
        total_bytes = sum(m["bytes_sent"] for _, m, _ in client_updates)
        total_energy = sum(m["energy_j_consumed"] for _, m, _ in client_updates)
        avg_beta = sum(m["beta_actual"] for _, m, _ in client_updates) / max(K, 1)
        avg_battery = sum(s.battery_j for _, _, s in client_updates) / max(K, 1)
        avg_loss = sum(m["local_loss"] for _, m, _ in client_updates) / max(K, 1)

        alive_clients = sum(1 for _, _, s in client_updates if s.battery_j > 0)
        participations = [1.0 if s.battery_j > 0 else 0.0 for _, _, s in client_updates]
        jain_index = (
            (sum(participations) ** 2) / (K * sum(p ** 2 for p in participations))
            if any(p > 0 for p in participations) else 0.0
        )

        del param_weighted_sums, param_total_weights, bn_weighted_sums

        # GC only every 10 rounds (explicit GC calls are expensive)
        if round_num % 10 == 0:
            gc.collect()

        return AggregateResult(
            new_weights=new_weights,
            metrics={
                "round": round_num,
                "total_bytes_sent": total_bytes,
                "total_energy_j": total_energy,
                "avg_beta": avg_beta,
                "avg_battery_j": avg_battery,
                "avg_local_loss": avg_loss,
                "participation_rate": alive_clients / max(K, 1),
                "jain_index": jain_index,
                "num_clients": K,
                "survival_ratio": alive_clients / max(K, 1),
            }
        )

    def get_default_config(self):
        return {
            "optimizer": "sgd",   # "sgd" | "adam"
            "lr": 0.01,
            "momentum": 0.9,      # SGD only (ignored when optimizer=adam)
            "weight_decay": 1e-4,
            "local_epochs": 5,
            "batch_size": 32,
            "warmup_rounds": 5,
            "beta_min": 0.1,
            "beta_max": 0.5,
            "ema_alpha": 0.3,
            "epsilon_explore": 0.05,
            "battery_max_j": None,
            "server_lr": 0.5,
            "mu_weight": 0.1,
            "energy_scale_factor": 1.0,
            "max_grad_norm": 10.0,
            "device": "cpu",
            "device_profile": None,
        }
