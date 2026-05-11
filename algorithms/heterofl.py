"""
algorithms/heterofl.py
======================
HeteroFL: Federated Learning with Heterogeneous Model Capacity
Diao et al., ICLR 2021
arXiv: 2010.01264

Problem solved
--------------
Clients have heterogeneous computational resources (CPU, memory, battery).
Standard FedAvg forces all clients to train the same full model, which is
infeasible for resource-constrained devices.

Key idea
--------
Assign each client a "capacity level" c ∈ {0.25, 0.5, 0.75, 1.0} based on
their available battery. Clients with lower capacity train only a subset
of the model layers (the first K layer groups), while higher-capacity
clients train all groups. The server aggregates: for each group g, average
only the deltas from clients who trained that group.

This layer-level approximation differs from the original channel-level
width scaling (HeteroFL trains narrower subnetworks by reducing channels).
Here we implement a simpler variant where capacity determines depth
(number of layer groups trained) instead of width.

Algorithm
---------
  1. Capacity assignment (battery-based):
       battery < 25% → c = 0.25 (train first 25% of groups)
       battery 25-50% → c = 0.5  (train first 50% of groups)
       battery 50-75% → c = 0.75 (train first 75% of groups)
       battery > 75%  → c = 1.0  (train all groups = FedAvg)

  2. Client:
       K = ceil(c × num_groups)
       Freeze all layers beyond the first K groups
       Train first K groups for E local epochs
       Upload deltas for K groups only

  3. Server:
       For each group g:
         if at least one client trained g:
           average deltas from all clients who trained g
           update global model group g
         else:
           keep group g unchanged

  4. Warmup (first W rounds):
       All clients train full model (c = 1.0) to initialize properly.

Convergence
-----------
Theorem 2 in the paper (non-convex):
    E[||∇F(w_T)||^2] ≤ O(1 / sqrt(T))
under L-smoothness, bounded variance, and heterogeneous client participation.

Communication cost:
  Average per client per round = (c × bytes_full_model)
  System-wide fairness measured by Jain index on delta bytes.

Energy cost:
  E_comp scales with c (fewer groups → fewer FLOPs)
  E_comm scales with c (fewer bytes uploaded)
"""

import gc
import math
from collections import OrderedDict, defaultdict
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .base import FLAlgorithm, ClientState, AggregateResult, register_algorithm
from .fedpart import _derive_layer_groups, _param_group_key


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _capacity_level_from_battery(battery_j: float, battery_max_j: float,
                                 thresholds: list[float],
                                 levels: list[float]) -> float:
    """
    Map battery level to capacity level c ∈ {0.25, 0.5, 0.75, 1.0}.

    Args:
        battery_j:     current battery in Joules
        battery_max_j: maximum battery capacity in Joules
        thresholds:    battery fraction thresholds [0.25, 0.50, 0.75]
        levels:        capacity levels [0.25, 0.50, 0.75, 1.0]

    Returns:
        capacity level c in (0, 1]
    """
    battery_fraction = battery_j / max(battery_max_j, 1.0)

    # thresholds = [0.25, 0.50, 0.75]
    # levels     = [0.25, 0.50, 0.75, 1.0]
    # battery < 0.25 → 0.25, [0.25, 0.50) → 0.50, [0.50, 0.75) → 0.75, ≥0.75 → 1.0
    for i, thresh in enumerate(thresholds):
        if battery_fraction < thresh:
            return levels[i]
    return levels[-1]


def _freeze_all_except_first_k_groups(
    model: nn.Module,
    groups: list[list[str]],
    k: int,
) -> None:
    """
    Freeze all parameters in groups [k, M) and enable gradient for groups [0, k).

    Args:
        model:  the model
        groups: list of parameter name lists (one list per group)
        k:      number of groups to keep trainable (first k groups)
                k <= 0 → all frozen (no-op)
                k >= len(groups) → all trainable (full update)
    """
    trainable_names = set()
    for i in range(min(k, len(groups))):
        trainable_names.update(groups[i])

    for name, param in model.named_parameters():
        param.requires_grad_(name in trainable_names)


def _restore_all_grad(model: nn.Module) -> None:
    """Re-enable gradients for all parameters."""
    for param in model.parameters():
        param.requires_grad_(True)


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm
# ─────────────────────────────────────────────────────────────────────────────

@register_algorithm("heterofl")
class HeteroFL(FLAlgorithm):
    """
    HeteroFL: battery-based capacity assignment with partial layer training.

    Clients with lower battery train fewer layer groups (early layers only).
    Server aggregates each group independently from clients who trained it.

    Registration key: "heterofl"
    """
    name = "heterofl"
    description = (
        "HeteroFL (Diao et al., ICLR 2021). Battery-based capacity assignment: "
        "clients with lower resources train fewer layer groups (early layers)."
    )

    # ── Client ────────────────────────────────────────────────────────────────

    def client_update(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        state: ClientState,
        config: dict,
    ) -> tuple[dict, dict]:
        """
        Local training step with capacity-based layer selection.

        1. Derive layer groups (cached in state.custom).
        2. Determine capacity level c from battery.
        3. K = ceil(c × num_groups) → train first K groups only.
        4. Freeze groups [K, M) and train groups [0, K).
        5. Upload deltas for groups [0, K) only.
        """
        device        = config.get("device", "cpu")
        lr            = config.get("lr", 0.01)
        momentum      = config.get("momentum", 0.9)
        weight_decay  = config.get("weight_decay", 1e-4)
        local_epochs  = config.get("local_epochs", 8)
        battery_max_j = config.get("battery_max_j", 185400.0)  # 10Ah @ 5.1V
        warmup_rounds = config.get("warmup_rounds", 5)

        model.train()
        model.to(device)

        # ── Derive layer groups (cached) ───────────────────────────────────────
        if "layer_groups" not in state.custom:
            groups = _derive_layer_groups(model)
            state.custom["layer_groups"] = groups
        groups: list[list[str]] = state.custom["layer_groups"]
        num_groups = len(groups)

        # ── Warmup: full model training (c = 1.0) ──────────────────────────────
        is_warmup = (state.round_num < warmup_rounds)

        if is_warmup:
            capacity = 1.0
            num_groups_trained = num_groups
        else:
            # ── Determine capacity level from battery ──────────────────────────
            thresholds = config.get("capacity_thresholds", [0.25, 0.50, 0.75])
            levels     = config.get("capacity_levels", [0.25, 0.50, 0.75, 1.0])
            capacity = _capacity_level_from_battery(
                state.battery_j, battery_max_j, thresholds, levels
            )
            num_groups_trained = max(1, math.ceil(capacity * num_groups))

        # ── Save weights before training ───────────────────────────────────────
        w_before = OrderedDict({
            k: v.clone().cpu() for k, v in model.state_dict().items()
        })

        # ── Freeze layers beyond first K groups ────────────────────────────────
        _freeze_all_except_first_k_groups(model, groups, num_groups_trained)

        # ── Local training ─────────────────────────────────────────────────────
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(
            trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        # Optional FedProx proximal term (μ ≥ 0)
        mu = config.get("mu", 0.0)

        total_loss, num_batches = 0.0, 0
        for _ in range(local_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)

                # FedProx regularization (optional)
                if mu > 0.0:
                    prox_loss = 0.0
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            prox_loss += (param - w_before[name].to(device)).pow(2).sum()
                    loss += (mu / 2.0) * prox_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        # ── Restore full gradient capability ────────────────────────────────────
        _restore_all_grad(model)

        # ── Compute delta: only for trained groups ──────────────────────────────
        # BN running stats (buffers) must always be included even for frozen
        # groups — they update on every forward pass regardless of requires_grad.
        # Omitting them leaves the global model with stale BN stats → divergence.
        current_sd = model.state_dict()
        trainable_keys = []
        for i in range(num_groups_trained):
            trainable_keys.extend(groups[i])

        bn_buffer_keys = [
            k for k in w_before
            if k.endswith(("running_mean", "running_var", "num_batches_tracked"))
        ]
        trainable_keys_set = set(trainable_keys)
        all_tx_keys = trainable_keys + [
            k for k in bn_buffer_keys if k not in trainable_keys_set
        ]
        partial_delta = OrderedDict({
            k: (w_before[k] - current_sd[k].cpu()).float()
            for k in all_tx_keys
        })

        # ── Communication / energy accounting ────────────────────────────────────
        uplink_bytes   = self.count_bytes(partial_delta, sparse=False)
        downlink_bytes = self.count_bytes(w_before, sparse=False)  # full model received
        # full model upload bytes = uplink_bytes / capacity (for compression ratio)
        _full_model_bytes = uplink_bytes / max(capacity, 1e-9)

        del w_before
        del current_sd
        gc.collect()

        profile = config.get("device_profile")
        if profile:
            # FLOPs scale with capacity (only K groups trained)
            num_params = sum(p.numel() for p in model.parameters())
            full_flops = profile.flops_for_model(
                num_params, dataloader.batch_size,
                local_epochs, len(dataloader.dataset)
            )
            # Forward pass: full model (need predictions)
            # Backward pass: only K groups (capacity fraction)
            forward_flops  = full_flops / 3.0
            backward_flops = full_flops * 2.0 / 3.0
            effective_flops = forward_flops + backward_flops * capacity
            energy_j = profile.round_energy_j(effective_flops, uplink_bytes, downlink_bytes)
        else:
            # Heuristic: scale base energy by capacity
            energy_j = 0.5 + 2.0 * capacity

        # ── Battery update (invariant: never negative) ───────────────────────────
        energy_j *= config.get("energy_scale_factor", 1.0)
        state.battery_j = max(0.0, state.battery_j - energy_j)
        state.round_num += 1

        del optimizer
        # ── Compression ratio relative to full model ─────────────────────────────
        compression_ratio = uplink_bytes / max(_full_model_bytes, 1)

        metadata = {
            # Mandatory fields
            "client_id":           state.client_id,
            "round_num":           state.round_num,
            "beta_actual":         1.0,  # no sparsification; full precision
            "battery_j_remaining": state.battery_j,
            "energy_j_consumed":   energy_j,
            "bytes_sent":          uplink_bytes,
            "bytes_received":      downlink_bytes,
            "local_loss":          total_loss / max(num_batches, 1),
            "compression_ratio":   compression_ratio,
            # HeteroFL-specific
            "capacity_level":      capacity,
            "num_groups_trained":  num_groups_trained,
            "is_warmup":           is_warmup,
            "delta_bytes":         uplink_bytes,  # for Jain fairness index
        }

        return dict(partial_delta), metadata

    # ── Server ─────────────────────────────────────────────────────────────────

    def server_aggregate(
        self,
        global_model: nn.Module,
        client_updates: list[tuple[dict, dict, ClientState]],
        round_num: int,
        config: dict,
    ) -> AggregateResult:
        """
        Aggregate partial deltas per layer group.

        For each group g:
          - Collect deltas from all clients who trained group g
          - Average those deltas
          - Update global model group g
        Groups not trained by any client remain unchanged.
        """
        K = len(client_updates)
        global_sd = global_model.state_dict()

        # ── Group clients by which groups they trained ──────────────────────────
        # group_idx → list of (delta_dict, metadata, state)
        group_updates: dict[str, list[torch.Tensor]] = defaultdict(list)

        for update, metadata, _ in client_updates:
            # Each client sends only keys for groups they trained
            for key, delta in update.items():
                group_updates[key].append(delta.float())

        # ── Average deltas per parameter key ─────────────────────────────────────
        averaged_delta = OrderedDict()
        for key, delta_list in group_updates.items():
            # Average across all clients who trained this key
            stacked = torch.stack(delta_list, dim=0)
            averaged_delta[key] = stacked.mean(dim=0)

        # ── Apply averaged deltas to global model ────────────────────────────────
        new_weights = OrderedDict()
        for k, v in global_sd.items():
            if k in averaged_delta:
                new_weights[k] = v.float() - averaged_delta[k].to(v.device)
            else:
                new_weights[k] = v.float()

        # ── Metrics ───────────────────────────────────────────────────────────────
        total_bytes  = sum(m["bytes_sent"] for _, m, _ in client_updates)
        total_energy = sum(m["energy_j_consumed"] for _, m, _ in client_updates)
        avg_battery  = sum(s.battery_j for _, _, s in client_updates) / K
        avg_loss     = sum(m["local_loss"] for _, m, _ in client_updates) / K
        avg_capacity = sum(m.get("capacity_level", 1.0) for _, m, _ in client_updates) / K

        # Jain fairness index on delta bytes
        bytes_list = [m.get("delta_bytes", 0) for _, m, _ in client_updates]
        bytes_sum  = sum(bytes_list)
        bytes_sq_sum = sum(b ** 2 for b in bytes_list)
        jain = (bytes_sum ** 2) / (K * bytes_sq_sum) if bytes_sq_sum > 0 else 1.0

        participations = [1.0 if s.battery_j > 0 else 0.0 for _, _, s in client_updates]

        return AggregateResult(
            new_weights=new_weights,
            metrics={
                "round":               round_num,
                "total_bytes_sent":    total_bytes,
                "total_energy_j":      total_energy,
                "avg_beta":            1.0,
                "avg_battery_j":       avg_battery,
                "avg_local_loss":      avg_loss,
                "compression_ratio":   avg_capacity,  # proxy: capacity ~ compression
                "participation_rate":  sum(participations) / K,
                "jain_index":          jain,
                "num_clients":         K,
                # HeteroFL-specific
                "avg_capacity_level":  avg_capacity,
            },
        )

    # ── Default configuration ──────────────────────────────────────────────────

    def get_default_config(self) -> dict:
        """
        Default hyperparameters for HeteroFL.

        warmup_rounds:       5      — initial full-model FedAvg rounds
        capacity_thresholds: [0.25, 0.50, 0.75]  — battery fraction thresholds
        capacity_levels:     [0.25, 0.50, 0.75, 1.0]  — corresponding capacity
        local_epochs:        8      — E, matching FedPart
        lr:                  0.01
        momentum:            0.9
        weight_decay:        1e-4
        mu:                  0.0    — FedProx proximal term (optional)
        battery_max_j:       185400.0  — 10Ah @ 5.1V (RPi4 powerbank)
        batch_size:          32
        device:              "cpu"
        device_profile:      None
        """
        return {
            "warmup_rounds":       5,
            "capacity_thresholds": [0.25, 0.50, 0.75],
            "capacity_levels":     [0.25, 0.50, 0.75, 1.0],
            "local_epochs":        8,
            "lr":                  0.01,
            "momentum":            0.9,
            "weight_decay":        1e-4,
            "mu":                  0.0,
            "battery_max_j":       185400.0,
            "batch_size":          32,
            "device":              "cpu",
            "device_profile":      None,
        }
