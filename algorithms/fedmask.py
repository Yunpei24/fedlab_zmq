"""
algorithms/fedmask.py
=====================
FedMask: Joint Computation and Communication-Efficient Personalized FL via Heterogeneous Masking

Li et al., NeurIPS 2021

Core idea
---------
Each client maintains element-wise importance scores (EMA of |grad|) for ALL parameters.
Battery level maps to sparsity level β_k, which determines how many elements to keep active.
Binary masks are computed per-client by ranking all elements across all params by importance,
keeping the top β_k fraction. Head params (fc.*, linear.*) are always active.

Key properties
--------------
  - Element-wise masks (finest granularity, fully personalized)
  - Client-side importance tracking (EMA of |grad| per element)
  - Masks are NOT shared (stored in client state only)
  - Apply mask to weights before training and after backward
  - Sparse communication: only active elements transmitted (8 bytes/elem: index + value)
  - Server does element-wise sparse aggregation (average only from clients with non-zero delta)

Difference from Server-Mask
----------------------------
  - FedMask: element-wise masks, client-side importance, personalized (not shared)
  - Server-Mask: tensor-wise masks, server-side importance, server assigns masks

Convergence
-----------
Non-convex O(1/√T) under L-smooth, bounded variance. Personalized masks reduce client drift.
"""

import gc
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

from .base import FLAlgorithm, ClientState, AggregateResult, register_algorithm
from hardware.profiles import DeviceProfile


@register_algorithm("fedmask")
class FedMask(FLAlgorithm):
    """
    FedMask: Element-wise personalized masking with battery-adaptive sparsity.

    Key properties:
      - Element-wise masks (finer than tensor-level)
      - Client-side importance (EMA of |grad| per element)
      - Personalized masks (not shared across clients)
      - Sparse communication (index + value pairs)
    """
    name = "fedmask"
    description = "FedMask: Element-wise personalized masking with battery-adaptive sparsity"

    def client_update(self, model, dataloader, state, config):
        device = config.get("device", "cpu")
        lr = config.get("lr", 0.01)
        momentum = config.get("momentum", 0.9)
        weight_decay = config.get("weight_decay", 1e-4)
        local_epochs = config.get("local_epochs", 5)
        max_grad_norm = config.get("max_grad_norm", 10.0)
        warmup_rounds = config.get("warmup_rounds", 5)
        ema_alpha = config.get("ema_alpha", 0.3)

        is_warmup = (state.round_num < warmup_rounds)

        named_params = list(model.named_parameters())
        total_elements = sum(p.numel() for _, p in named_params)

        bn_buffer_names = {
            n for n in model.state_dict()
            if n.endswith(('.running_mean', '.running_var'))
        }

        head_names = {
            n for n, _ in named_params
            if n.startswith(("fc.", "linear.", "classifier.", "head."))
        }

        # Initialize importance scores (per element)
        if "importance" not in state.custom:
            state.custom["importance"] = {
                n: torch.ones_like(p, dtype=torch.float32)
                for n, p in named_params
            }

        importance = state.custom["importance"]

        # Compute masks from battery level
        if is_warmup:
            masks = {n: torch.ones_like(p, dtype=torch.float32) for n, p in named_params}
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

            # Vectorized ranking: concatenate all importance scores, find threshold
            all_scores = torch.cat([importance[n].view(-1) for n, _ in named_params])
            total = all_scores.numel()
            k_drop = total - element_budget
            if k_drop <= 0:
                threshold = all_scores.min().item() - 1.0
            else:
                threshold = torch.kthvalue(all_scores, k_drop + 1).values.item()

            masks = {}
            for n, p in named_params:
                masks[n] = (importance[n] >= threshold).float()

            # Head params always active
            for n in head_names:
                if n in masks:
                    masks[n].fill_(1.0)

            active_elements = int(sum(m.sum().item() for m in masks.values()))

        beta_actual = active_elements / max(total_elements, 1)

        # Snapshot before training
        sd = model.state_dict()
        w_before = {n: sd[n].clone().cpu() for n in list(sd.keys())}
        del sd

        model.train()
        model.to(device)

        # Precompute masks on device once (avoids repeated .to(device) per batch)
        masks_dev = {n: masks[n].to(device) for n in masks}

        # Apply masks to weights before training
        with torch.no_grad():
            for n, p in named_params:
                p.data *= masks_dev[n]

        optimizer = optim.SGD(
            [p for _, p in named_params],
            lr=lr, momentum=momentum, weight_decay=weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        all_params = [p for _, p in named_params]

        total_loss, num_batches = 0.0, 0
        for _ in range(local_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()

                # Zero out gradients of inactive elements (mask in gradient space)
                for n, p in named_params:
                    if p.grad is not None:
                        p.grad.mul_(masks_dev[n])

                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)

                optimizer.step()

                # Re-apply mask to weights (keep inactive elements at zero)
                with torch.no_grad():
                    for n, p in named_params:
                        p.data.mul_(masks_dev[n])

                total_loss += loss.item()
                num_batches += 1

        # Update importance EMA (only if not warmup)
        if not is_warmup:
            one_minus_alpha = 1.0 - ema_alpha
            for n, p in named_params:
                if p.grad is not None:
                    grad_abs = p.grad.abs().cpu().detach()
                    importance[n] = one_minus_alpha * importance[n] + ema_alpha * grad_abs

        # Compute delta (only active elements non-zero)
        current_sd = model.state_dict()
        delta = OrderedDict()
        for n in w_before:
            if n in current_sd:
                diff = (w_before[n] - current_sd[n].cpu()).float()
                if n in masks and not n.endswith(('.running_mean', '.running_var')):
                    diff = diff * masks[n]
                delta[n] = diff

        del w_before, current_sd, optimizer

        # Communication: sparse format (8 bytes/active element: 4 index + 4 value)
        # BN buffers are always dense (4 bytes/elem)
        uplink_bytes = 0
        for n, d in delta.items():
            if n.endswith(('.running_mean', '.running_var')):
                uplink_bytes += d.numel() * 4
            else:
                nnz = d.count_nonzero().item()
                uplink_bytes += nnz * 8

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

        compression_ratio = uplink_bytes / max(downlink_bytes, 1)

        metadata = {
            "client_id": state.client_id,
            "round_num": state.round_num,
            "beta_actual": beta_actual,
            "battery_j_remaining": state.battery_j,
            "energy_j_consumed": energy_j,
            "bytes_sent": uplink_bytes,
            "bytes_received": downlink_bytes,
            "local_loss": total_loss / max(num_batches, 1),
            "compression_ratio": compression_ratio,
            "is_warmup": is_warmup,
            "active_params": int(active_elements),
            "dataset_size": len(dataloader.dataset),
        }

        return dict(delta), metadata

    def server_aggregate(self, global_model, client_updates, round_num, config):
        K = len(client_updates)
        global_sd = global_model.state_dict()
        warmup_rounds = config.get("warmup_rounds", 5)
        is_warmup = round_num < warmup_rounds

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

        # Element-wise sparse aggregation (dataset-size weighted)
        sizes = [m.get("dataset_size", 1) for _, m, _ in client_updates]
        total_n = max(sum(sizes), 1)

        weighted_sums = {}
        weight_counts = {}

        for (update, meta, _), n_k in zip(client_updates, sizes):
            w_k = n_k / total_n
            for name, delta in update.items():
                delta_f = delta.float()
                if name.endswith((".running_mean", ".running_var")):
                    if name not in weighted_sums:
                        weighted_sums[name] = delta_f * w_k
                    else:
                        weighted_sums[name] += delta_f * w_k
                else:
                    # Element-wise sparse aggregation
                    mask = (delta_f.abs() > 0).float()
                    if name not in weighted_sums:
                        weighted_sums[name] = delta_f * n_k
                        weight_counts[name] = mask * n_k
                    else:
                        weighted_sums[name] += delta_f * n_k
                        weight_counts[name] += mask * n_k

        server_lr = 1.0
        new_weights = OrderedDict()
        for name in global_sd:
            if name in weighted_sums:
                if name.endswith((".running_mean", ".running_var")):
                    new_weights[name] = (
                        global_sd[name].float()
                        - weighted_sums[name].to(global_sd[name].device)
                    )
                else:
                    # Element-wise average (only where at least one client updated)
                    avg_delta = weighted_sums[name] / weight_counts[name].clamp(min=1e-6)
                    new_weights[name] = (
                        global_sd[name].float()
                        - server_lr * avg_delta.to(global_sd[name].device)
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

        del weighted_sums, weight_counts

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
            "ema_alpha": 0.3,
            "battery_max_j": None,
            "server_lr": 1.0,
            "energy_scale_factor": 1.0,
            "max_grad_norm": 10.0,
            "device": "cpu",
            "device_profile": None,
        }
