"""
algorithms/fjord.py
===================
FjORD: Fair and Accurate Federated Learning under Heterogeneous Targets
        with Ordered Dropout
Samuel et al., NeurIPS 2021
arXiv: 2102.11291

Problem solved
--------------
Clients have heterogeneous resource constraints (computation, memory, battery).
Standard FedAvg forces all clients to train the same large model, which is
infeasible for low-resource devices. Prior work (HeteroFL) allows different
model sizes but lacks the "ordered" structure needed for nested subnetwork
aggregation.

Key idea
--------
Ordered dropout maintains a nested structure: small models are strict subsets
of large models. All clients always train the cheapest layers (stem, early
layers) and progressively drop the most expensive layers based on capacity.

This ensures:
  1. Parameter sharing across heterogeneous clients (small models ⊂ large models)
  2. Fair aggregation: all clients contribute to early layers; only high-capacity
     clients contribute to expensive later layers
  3. Self-distillation: small models are regularized towards the full global model

Algorithm (layer-level approximation)
--------------------------------------
  1. Dropout rate p ∈ {0.25, 0.5, 0.75, 1.0} assigned by battery:
       battery < 25% → p = 0.25 (train only cheapest 25% of groups)
       battery 25-50% → p = 0.5
       battery 50-75% → p = 0.75
       battery > 75%  → p = 1.0 (train all groups)

  2. Cost ordering:
       Sort layer groups by parameter count (ascending).
       Groups = [stem, fc, layer1.pair1, ..., layer3.pair2]
                 ^ cheapest             most expensive ^

  3. Client (dropout rate p):
       K = ceil(p × num_groups)
       Train the K cheapest groups (by param count)
       Freeze the M - K most expensive groups
       Apply self-distillation: L2 regularization towards global model

  4. Server:
       For each group g:
         Average deltas from all clients who trained g
         Update global model group g

  5. Warmup (first W rounds):
       All clients train full model (p = 1.0).

Difference from HeteroFL
-------------------------
  HeteroFL: train first K groups (architectural order: stem → layer1 → layer2 → fc)
  FjORD:    train cheapest K groups (cost order: stem → fc → layer1 → layer2)

For ResNet-8, cost ordering (ascending param count):
  stem(448) < fc(650) < layer1.pair1(2368) < layer1.pair2(2368) <
  layer2.pair1(9408) < layer2.shortcut(4736) < layer2.pair2(18816) <
  layer3.shortcut(18944) < layer3.pair1(37632) < layer3.pair2(75264)

So p=0.25 trains {stem, fc} (2 cheapest) vs HeteroFL which trains {stem, layer1.pair1}.

Self-distillation (simplified)
-------------------------------
Original paper uses KD loss from full model logits to subnetwork logits.
We approximate this with L2 weight regularization towards the global model:
  loss = CE_loss + (mu_distill / 2) * sum(||w_k - w_global_k||^2)
This acts as a proximal constraint similar to FedProx but applied only to
the trainable (cheap) layers.

Convergence
-----------
Theorem 1 in the paper (non-convex):
    E[||∇F(w_T)||^2] ≤ O(1 / sqrt(T))
under L-smoothness, bounded gradient variance, and ordered dropout structure.

Communication cost:
  Average per client = (p × bytes_full_model)
  Ordered structure ensures fair contribution across capacities.

Energy cost:
  E_comp scales with p (fewer groups → fewer FLOPs)
  E_comm scales with p (fewer bytes uploaded)
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

def _dropout_rate_from_battery(battery_j: float, battery_max_j: float,
                               thresholds: list[float],
                               levels: list[float]) -> float:
    """
    Map battery level to dropout rate p ∈ {0.25, 0.5, 0.75, 1.0}.

    Same thresholds as HeteroFL but semantics differ:
      p = fraction of groups to train (ordered by cost ascending).
    """
    battery_fraction = battery_j / max(battery_max_j, 1.0)
    for i, thresh in enumerate(thresholds):
        if battery_fraction < thresh:
            return levels[i]
    return levels[-1]


def _group_cost_ordering(model: nn.Module, groups: list[list[str]]) -> list[int]:
    """
    Return indices of groups sorted by total parameter count (ascending).

    For each group g, compute sum of param.numel() over all params in g.
    Return sorted indices [i_0, i_1, ..., i_{M-1}] where
    cost[i_0] ≤ cost[i_1] ≤ ... ≤ cost[i_{M-1}].

    Example for ResNet-8:
      groups[0] = stem (448 params)
      groups[9] = fc (650 params)
      groups[1] = layer1.pair1 (2368 params)
      ...
    → ordering = [0, 9, 1, 2, ...]  (stem, fc, layer1.pair1, ...)
    """
    state_dict = model.state_dict()
    costs = []
    for i, group in enumerate(groups):
        total_params = sum(
            state_dict[k].numel() for k in group if k in state_dict
        )
        costs.append((total_params, i))
    costs.sort()  # sort by (cost, index)
    return [idx for _, idx in costs]


def _freeze_all_except_cheapest_k_groups(
    model: nn.Module,
    groups: list[list[str]],
    cost_order: list[int],
    k: int,
) -> None:
    """
    Freeze all parameters except the K cheapest groups.

    Args:
        model:      the model
        groups:     list of parameter name lists (one per group)
        cost_order: indices of groups sorted by cost ascending
        k:          number of cheapest groups to keep trainable
    """
    trainable_names = set()
    for i in range(min(k, len(cost_order))):
        group_idx = cost_order[i]
        trainable_names.update(groups[group_idx])

    for name, param in model.named_parameters():
        param.requires_grad_(name in trainable_names)


def _restore_all_grad(model: nn.Module) -> None:
    """Re-enable gradients for all parameters."""
    for param in model.parameters():
        param.requires_grad_(True)


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm
# ─────────────────────────────────────────────────────────────────────────────

@register_algorithm("fjord")
class FjORD(FLAlgorithm):
    """
    FjORD: Fair and Accurate FL with Ordered Dropout (Samuel et al., NeurIPS 2021).

    Clients train the cheapest K groups (by parameter count) based on battery.
    Ordered structure: small models ⊂ large models.
    Self-distillation regularizes subnetworks towards the global model.

    Registration key: "fjord"
    """
    name = "fjord"
    description = (
        "FjORD (Samuel et al., NeurIPS 2021). Ordered dropout: "
        "train cheapest K groups by parameter cost (battery-based p)."
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
        Local training step with ordered dropout.

        1. Derive layer groups + cost ordering (cached).
        2. Determine dropout rate p from battery.
        3. K = ceil(p × num_groups) → train K cheapest groups.
        4. Freeze expensive groups and train cheap groups.
        5. Apply self-distillation (L2 regularization towards global model).
        6. Upload deltas for K cheap groups only.
        """
        device        = config.get("device", "cpu")
        lr            = config.get("lr", 0.01)
        momentum      = config.get("momentum", 0.9)
        weight_decay  = config.get("weight_decay", 1e-4)
        local_epochs  = config.get("local_epochs", 8)
        battery_max_j = config.get("battery_max_j", 185400.0)
        warmup_rounds = config.get("warmup_rounds", 5)
        mu_distill    = config.get("mu_distill", 0.01)  # self-distillation coeff

        model.train()
        model.to(device)

        # ── Derive layer groups + cost ordering (cached) ─────────────────────────
        if "layer_groups" not in state.custom:
            groups = _derive_layer_groups(model)
            state.custom["layer_groups"] = groups
        groups: list[list[str]] = state.custom["layer_groups"]

        if "cost_ordering" not in state.custom:
            cost_order = _group_cost_ordering(model, groups)
            state.custom["cost_ordering"] = cost_order
        cost_order: list[int] = state.custom["cost_ordering"]

        num_groups = len(groups)

        # ── Warmup: full model training (p = 1.0) ────────────────────────────────
        is_warmup = (state.round_num < warmup_rounds)

        if is_warmup:
            dropout_rate = 1.0
            num_groups_trained = num_groups
        else:
            # ── Determine dropout rate from battery ───────────────────────────────
            thresholds = config.get("capacity_thresholds", [0.25, 0.50, 0.75])
            levels     = config.get("capacity_levels", [0.25, 0.50, 0.75, 1.0])
            dropout_rate = _dropout_rate_from_battery(
                state.battery_j, battery_max_j, thresholds, levels
            )
            num_groups_trained = max(1, math.ceil(dropout_rate * num_groups))

        # ── Save weights before training ───────────────────────────────────────────
        w_before = OrderedDict({
            k: v.clone().cpu() for k, v in model.state_dict().items()
        })

        # ── Freeze expensive groups, train cheap groups ─────────────────────────────
        _freeze_all_except_cheapest_k_groups(model, groups, cost_order, num_groups_trained)

        # ── Local training with self-distillation ───────────────────────────────────
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(
            trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        total_loss, num_batches = 0.0, 0
        for _ in range(local_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)

                # Self-distillation: L2 regularization towards global model
                # (Simplified approximation of KD loss from full model to subnetwork)
                if mu_distill > 0.0:
                    distill_loss = 0.0
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            distill_loss += (param - w_before[name].to(device)).pow(2).sum()
                    loss += (mu_distill / 2.0) * distill_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        # ── Restore full gradient capability ────────────────────────────────────────
        _restore_all_grad(model)

        # ── Compute delta: only for trained (cheap) groups ───────────────────────────
        # BN running stats (buffers) must always be included even for frozen
        # groups — they update on every forward pass regardless of requires_grad.
        # Omitting them leaves the global model with stale BN stats → divergence.
        current_sd = model.state_dict()
        trainable_keys = []
        for i in range(num_groups_trained):
            group_idx = cost_order[i]
            trainable_keys.extend(groups[group_idx])

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
        # ── Communication / energy accounting ────────────────────────────────────────
        uplink_bytes   = self.count_bytes(partial_delta, sparse=False)
        downlink_bytes = self.count_bytes(w_before, sparse=False)
        full_upload_bytes = self.count_bytes(w_before, sparse=False)

        del w_before
        del current_sd
        gc.collect()

        profile = config.get("device_profile")
        if profile:
            num_params = sum(p.numel() for p in model.parameters())
            full_flops = profile.flops_for_model(
                num_params, dataloader.batch_size,
                local_epochs, len(dataloader.dataset)
            )
            # Forward: full model (need predictions)
            # Backward: only K cheap groups (dropout_rate fraction)
            forward_flops  = full_flops / 3.0
            backward_flops = full_flops * 2.0 / 3.0
            effective_flops = forward_flops + backward_flops * dropout_rate
            energy_j = profile.round_energy_j(effective_flops, uplink_bytes, downlink_bytes)
        else:
            energy_j = 0.5 + 2.0 * dropout_rate

        # ── Battery update (invariant: never negative) ───────────────────────────────
        energy_j *= config.get("energy_scale_factor", 1.0)
        state.battery_j = max(0.0, state.battery_j - energy_j)
        state.round_num += 1

        del optimizer
        # ── Compression ratio ────────────────────────────────────────────────────────
        compression_ratio = uplink_bytes / max(full_upload_bytes, 1)

        metadata = {
            # Mandatory fields
            "client_id":           state.client_id,
            "round_num":           state.round_num,
            "beta_actual":         1.0,
            "battery_j_remaining": state.battery_j,
            "energy_j_consumed":   energy_j,
            "bytes_sent":          uplink_bytes,
            "bytes_received":      downlink_bytes,
            "local_loss":          total_loss / max(num_batches, 1),
            "compression_ratio":   compression_ratio,
            # FjORD-specific
            "dropout_rate":        dropout_rate,
            "num_groups_trained":  num_groups_trained,
            "is_warmup":           is_warmup,
            "delta_bytes":         uplink_bytes,
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
          - Average deltas from all clients who trained g
          - Update global model group g
        Groups not trained by any client remain unchanged.
        """
        K = len(client_updates)
        global_sd = global_model.state_dict()

        # ── Group deltas by parameter key ────────────────────────────────────────
        group_updates: dict[str, list[torch.Tensor]] = defaultdict(list)

        for update, metadata, _ in client_updates:
            for key, delta in update.items():
                group_updates[key].append(delta.float())

        # ── Average deltas per key ───────────────────────────────────────────────
        averaged_delta = OrderedDict()
        for key, delta_list in group_updates.items():
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
        avg_dropout  = sum(m.get("dropout_rate", 1.0) for _, m, _ in client_updates) / K

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
                "compression_ratio":   avg_dropout,  # proxy: dropout_rate ~ compression
                "participation_rate":  sum(participations) / K,
                "jain_index":          jain,
                "num_clients":         K,
                # FjORD-specific
                "avg_dropout_rate":    avg_dropout,
            },
        )

    # ── Default configuration ──────────────────────────────────────────────────

    def get_default_config(self) -> dict:
        """
        Default hyperparameters for FjORD.

        warmup_rounds:       5      — initial full-model FedAvg rounds
        capacity_thresholds: [0.25, 0.50, 0.75]  — battery thresholds
        capacity_levels:     [0.25, 0.50, 0.75, 1.0]  — dropout rates
        local_epochs:        8
        lr:                  0.01
        momentum:            0.9
        weight_decay:        1e-4
        mu_distill:          0.01   — self-distillation coefficient (L2 regularization)
        battery_max_j:       185400.0
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
            "mu_distill":          0.01,
            "battery_max_j":       185400.0,
            "batch_size":          32,
            "device":              "cpu",
            "device_profile":      None,
        }
