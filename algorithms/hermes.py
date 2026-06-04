"""
algorithms/hermes.py
====================
Hermes: An Efficient Federated Learning Framework for Heterogeneous Mobile Clients
via Adaptive Model Pruning

Li et al., MobiSys 2021

Implementation note
-------------------
The original Hermes uses channel-level pruning with hard structural removal.
Here we use **tensor-level magnitude-based masking** (same granularity as Server-Mask)
to avoid ResNet skip-connection complexity while preserving the core algorithmic idea:
battery-adaptive pruning based on weight magnitude (not gradient importance).

Core idea
---------
Battery level maps to β_k, which determines how many parameter tensors to keep active.
Tensors are ranked by Frobenius norm (||param||_F), and top β_k fraction by element budget
are selected. Head params are always active. No warmup needed since magnitude is available
from round 0 (but kept for fair comparison).

Key differences from Server-Mask
---------------------------------
  - Hermes: magnitude-based selection (||w||_F), no importance state
  - Server-Mask: gradient-importance-based selection (EMA of ||grad||²), importance tracked

Key differences from FedMask
-----------------------------
  - Hermes: tensor-level masks, magnitude-based, recomputed each round
  - FedMask: element-wise masks, gradient-importance-based, EMA tracking

Convergence
-----------
Non-convex O(1/√T) under L-smooth, bounded variance. Magnitude-based selection is more stable
than sparse gradient updates (no need for server_lr scaling).
"""

import gc
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

from .base import FLAlgorithm, ClientState, AggregateResult, register_algorithm
from hardware.profiles import DeviceProfile


@register_algorithm("hermes")
class Hermes(FLAlgorithm):
    """
    Hermes: Tensor-level magnitude-based masking with battery-adaptive sparsity.

    Key properties:
      - Tensor-level masks (same granularity as Server-Mask)
      - Magnitude-based selection (||param||_F, not gradient importance)
      - No importance state (masks recomputed each round from current weights)
      - Stable aggregation (no server_lr needed)
    """
    name = "hermes"
    description = "Hermes: Tensor-level magnitude-based masking with battery-adaptive sparsity"

    def client_update(self, model, dataloader, state, config):
        device = config.get("device", "cpu")
        lr = config.get("lr", 0.01)
        momentum = config.get("momentum", 0.9)
        weight_decay = config.get("weight_decay", 1e-4)
        local_epochs = config.get("local_epochs", 5)
        max_grad_norm = config.get("max_grad_norm", 10.0)
        warmup_rounds = config.get("warmup_rounds", 5)

        is_warmup = (state.round_num < warmup_rounds)

        named_params = list(model.named_parameters())
        param_sizes = {n: p.numel() for n, p in named_params}
        total_elements = sum(param_sizes.values())

        bn_buffer_names = {
            n for n in model.state_dict()
            if n.endswith(('.running_mean', '.running_var'))
        }

        head_names = {
            n for n, _ in named_params
            if n.startswith(("fc.", "linear.", "classifier.", "head."))
        }

        # Compute masks from battery level and magnitude
        if is_warmup:
            active_param_names = {n for n, _ in named_params}
            active_elements = total_elements
        else:
            beta_min = config.get("beta_min", 0.1)
            beta_max = config.get("beta_max", 0.5)
            battery_max_j = state.custom.get("_battery_max_j")
            if battery_max_j is None:
                battery_max_j = config.get("battery_max_j", 185400.0)
                state.custom["_battery_max_j"] = battery_max_j

            beta_k = beta_min + (beta_max - beta_min) * min(state.battery_j / battery_max_j, 1.0)
            element_budget = max(1, int(beta_k * total_elements))

            # Rank tensors by Frobenius norm (magnitude)
            scores = {n: p.norm(2).item() for n, p in named_params}

            sorted_names = sorted(scores.keys(), key=lambda n: scores[n], reverse=True)

            # Greedy selection by element budget
            active_param_names = set(head_names)
            used_elements = sum(param_sizes[n] for n in head_names if n in param_sizes)

            for n in sorted_names:
                if n in head_names:
                    continue
                sz = param_sizes[n]
                if used_elements + sz <= element_budget:
                    active_param_names.add(n)
                    used_elements += sz

            active_elements = used_elements

        beta_actual = active_elements / max(total_elements, 1)

        # Snapshot only delta tensors
        delta_names = active_param_names | bn_buffer_names
        sd = model.state_dict()
        w_before = {n: sd[n].clone().cpu() for n in delta_names}
        del sd

        model.train()
        model.to(device)

        # Apply mask: toggle requires_grad only when it changes
        if is_warmup:
            for _, param in named_params:
                if not param.requires_grad:
                    param.requires_grad_(True)
        else:
            for name, param in named_params:
                should_train = name in active_param_names
                if param.requires_grad != should_train:
                    param.requires_grad_(should_train)

        active_named_params = [(n, p) for n, p in named_params if n in active_param_names]
        active_params_only = [p for _, p in active_named_params]

        optimizer = optim.SGD(
            active_params_only,
            lr=lr, momentum=momentum, weight_decay=weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        total_loss, num_batches = 0.0, 0
        for _ in range(local_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(active_params_only, max_grad_norm)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        # Compute delta
        current_sd = {n: v for n, v in model.state_dict().items() if n in delta_names}
        delta = OrderedDict()
        for name in delta_names:
            if name in w_before and name in current_sd:
                delta[name] = (w_before[name] - current_sd[name].cpu()).float()

        del w_before, current_sd, optimizer

        # Dense tensor transmission (4 bytes/elem)
        uplink_bytes = sum(t.numel() * 4 for t in delta.values())
        downlink_bytes = total_elements * 4

        profile: DeviceProfile = config.get("device_profile")
        if profile is not None:
            full_flops = profile.flops_for_model(
                total_elements,
                dataloader.batch_size,
                local_epochs,
                len(dataloader.dataset),
            )
            effective_flops = (1.0 + beta_actual) * 0.5 * full_flops
            energy_j = profile.round_energy_j(effective_flops, uplink_bytes, downlink_bytes)
        else:
            energy_j = 0.5 + 2.0 * beta_actual

        energy_j *= config.get("energy_scale_factor", 1.0)

        state.battery_j = max(0.0, state.battery_j - energy_j)
        state.round_num += 1

        metadata = {
            "client_id": state.client_id,
            "round_num": state.round_num,
            "beta_actual": beta_actual,
            "battery_j_remaining": state.battery_j,
            "energy_j_consumed": energy_j,
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

        if not hasattr(self, "_server_state"):
            self._server_state = {}

        # Auto-calibrate battery_max_j at round 0
        if "_battery_max_j" not in self._server_state:
            observed_max = max(s.battery_j for _, _, s in client_updates)
            config_override = config.get("battery_max_j")
            battery_max_j = (
                float(config_override) if config_override is not None
                else observed_max / 0.95
            )
            self._server_state["_battery_max_j"] = battery_max_j
            # Propagate to all client states
            for _, _, s in client_updates:
                s.custom["_battery_max_j"] = battery_max_j

        # Aggregate sparse updates (dataset-size weighted)
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

        # No server_lr scaling (magnitude-based selection is stable)
        server_lr = 1.0
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

        # Metrics
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
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "local_epochs": 5,
            "batch_size": 32,
            "warmup_rounds": 5,
            "beta_min": 0.1,
            "beta_max": 0.5,
            "battery_max_j": None,
            "energy_scale_factor": 1.0,
            "max_grad_norm": 10.0,
            "device": "cpu",
            "device_profile": None,
        }
