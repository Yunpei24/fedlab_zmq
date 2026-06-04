"""
algorithms/fed_od.py
====================
FedOD: Federated Object Detection with Battery-Aware Backbone / Head Split
J. Nikiema — UM6P, 2026

─────────────────────────────────────────────────────────────────────────────
Core Idea
─────────────────────────────────────────────────────────────────────────────
The model is split into two functional blocks:

  backbone  — early convolutional layers (feature extractor, transferable)
  head      — late layers + classifier (task-specific)

Clients are partitioned into two tiers each round based on their residual
battery ratio β = B_t^k / B_max:

  Tier 0  (β < beta_split) : train backbone only, head frozen
  Tier 1  (β ≥ beta_split) : train full model (backbone + head)

This mirrors the federated object detection setting where the backbone
(e.g. CSP-DarkNet in YOLOv8) dominates compute and its gradients are
universally useful across tasks, while the detection head is smaller and
only needs updates from energy-sufficient clients.

─────────────────────────────────────────────────────────────────────────────
Energy Model (per-round compute cost)
─────────────────────────────────────────────────────────────────────────────
Let α_b = backbone_fraction  (fraction of total parameter count in backbone).

  Tier 0 — backbone-only training:
    forward: full model (needed for loss computation)
    backward: backbone only  → grad_input stops at backbone output

    Effective energy ratio vs full training:
      ρ_backbone = (E_fwd + α_b × E_bwd) / (E_fwd + E_bwd)
               ≈ (1 + α_b) / 2             [E_fwd ≈ E_bwd / 2]

    For α_b = 0.70  →  ρ_backbone ≈ 0.85
    For YOLOv8-nano (α_b ≈ 0.92, d_h/d ≈ 8%):
      ρ_backbone ≈ 0.96  (very small head → save little compute)
      But uplink savings are large: only backbone params transmitted.

  Tier 1 — full training: ρ_full = 1.0

Communication cost:
  Tier 0: transmit only backbone delta  → α_b × model_size bytes
  Tier 1: transmit full delta           → model_size bytes

─────────────────────────────────────────────────────────────────────────────
Server Aggregation
─────────────────────────────────────────────────────────────────────────────
  backbone params: dataset-size-weighted average over ALL alive clients
  head params:     dataset-size-weighted average over Tier-1 clients only
                   (fallback to all clients if no Tier-1 client survived)
  BN buffers:      global weighted average over all alive clients
                   (all clients run full forward, so all BN stats are valid)

─────────────────────────────────────────────────────────────────────────────
Warmup Phase
─────────────────────────────────────────────────────────────────────────────
For rounds 0 … warmup_rounds-1: full FedAvg update (identical to FedAvg).
All clients train the complete model; no tier assignment.

─────────────────────────────────────────────────────────────────────────────
Convergence (informal sketch)
─────────────────────────────────────────────────────────────────────────────
Let K_b = alive backbone-only clients, K_f = alive full clients, K = K_b + K_f.

Backbone gradient: estimated from K clients → variance σ²/K (standard FedAvg).
Head gradient:     estimated from K_f clients only → variance σ²/K_f ≥ σ²/K.

As β_split → 0 (all clients train full model): reduces to FedAvg, O(1/√T).
As β_split → 1 (all clients train backbone only): head stagnates.

Optimal β_split* balances head quality vs fleet survival. A diminishing
schedule β_split(t) = β_split_0 × (1 - t/T) can anneal to FedAvg at end.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from .base import FLAlgorithm, ClientState, AggregateResult, register_algorithm
from .fedpart import _derive_layer_groups


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _energy_j(
    profile,
    model: nn.Module,
    active_fraction: float,
    bytes_sent: int,
    energy_scale_factor: float,
    config: dict,
) -> float:
    """Per-round energy in Joules. Uses DeviceProfile API when available."""
    try:
        from hardware.profiles import DeviceProfile
        if isinstance(profile, DeviceProfile):
            batch_size   = config.get("batch_size", 8)
            local_epochs = config.get("local_epochs", 3)
            dataset_size = config.get("_dataset_size", 500)
            num_params   = int(sum(p.numel() for p in model.parameters()) * active_fraction)
            full_flops   = profile.flops_for_model(num_params, batch_size, local_epochs, dataset_size)
            e_comp       = profile.compute_energy_j(full_flops)
            e_tx         = profile.comm_energy_j(bytes_sent, direction="uplink")
            return (e_comp + e_tx) * energy_scale_factor
    except Exception:
        pass
    # flat fallback
    return (2.33 * active_fraction + 1e-5 * bytes_sent) * energy_scale_factor


def _split_backbone_head(
    groups: list[list[str]],
    backbone_fraction: float,
) -> tuple[set[str], set[str]]:
    """
    Split model parameter keys into backbone and head sets.

    Args:
        groups:            List of layer groups (each group = list of param names).
        backbone_fraction: Fraction of groups assigned to backbone [0, 1].

    Returns:
        backbone_keys: set of parameter names belonging to the backbone.
        head_keys:     set of parameter names belonging to the head.
    """
    n_backbone = max(1, round(len(groups) * backbone_fraction))
    backbone_keys: set[str] = set()
    head_keys: set[str] = set()

    for i, group in enumerate(groups):
        if i < n_backbone:
            backbone_keys.update(group)
        else:
            head_keys.update(group)

    return backbone_keys, head_keys


def _bn_buffer_keys(model: nn.Module) -> set[str]:
    """Return names of all BatchNorm running-stat buffers (not parameters)."""
    bn_keys: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            prefix = (name + ".") if name else ""
            bn_keys.add(prefix + "running_mean")
            bn_keys.add(prefix + "running_var")
            bn_keys.add(prefix + "num_batches_tracked")
    return bn_keys


def _weighted_average(
    updates: list[tuple[dict, int]],
    param_names: set[str],
) -> dict[str, torch.Tensor]:
    """Dataset-size weighted average of deltas for selected param names."""
    total = sum(n for _, n in updates)
    if total == 0:
        return {}
    avg: dict[str, torch.Tensor] = {}
    for name in param_names:
        acc = None
        w_total = 0
        for delta, n in updates:
            if name not in delta:
                continue
            contrib = delta[name].float() * n
            acc = contrib if acc is None else acc + contrib
            w_total += n
        if acc is not None and w_total > 0:
            avg[name] = acc / w_total
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm
# ─────────────────────────────────────────────────────────────────────────────

@register_algorithm("fed_od")
class FedOD(FLAlgorithm):
    """
    FedOD — Battery-aware Backbone/Head Federated Learning.

    Clients with low residual battery train only the backbone (feature
    extractor).  Clients with sufficient battery train the full model.
    Server aggregates backbone from all clients, head from full-training
    clients only.
    """

    name = "fed_od"
    description = "FedOD: battery-aware backbone/head split for federated vision"

    # ── Client update ─────────────────────────────────────────────────────────

    def client_update(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        state: ClientState,
        config: dict,
    ) -> tuple[dict, dict]:

        device        = next(model.parameters()).device
        lr            = config.get("lr", 0.01)
        local_epochs  = config.get("local_epochs", 5)
        warmup_rounds = config.get("warmup_rounds", 5)
        backbone_fraction = config.get("backbone_fraction", 0.70)
        beta_split    = config.get("beta_split", 0.40)
        battery_max_j = config.get("battery_max_j", 13320.0)
        momentum      = config.get("momentum", 0.9)
        weight_decay  = config.get("weight_decay", 1e-4)
        optimizer_type = config.get("optimizer", "sgd").lower()
        energy_scale_factor = config.get("energy_scale_factor", 1.0)

        model_type    = config.get("model_type", "classification")  # "classification"|"detection"

        battery_ratio = state.battery_j / max(battery_max_j, 1.0)
        is_warmup     = state.round_num < warmup_rounds
        is_backbone_only = (not is_warmup) and (battery_ratio < beta_split)

        # ── Backbone / head key sets ────────────────────────────────────────
        if model_type == "detection" and hasattr(model, "backbone"):
            # Torchvision detection models: split by top-level module name.
            # SSDLite: backbone.* vs head.*, transform.*, anchor_generator.*
            backbone_keys = {n for n, _ in model.named_parameters()
                             if n.startswith("backbone.")}
            head_keys     = {n for n, _ in model.named_parameters()
                             if not n.startswith("backbone.")}
            # Compute actual backbone fraction from parameter counts
            n_backbone = sum(p.numel() for p in model.backbone.parameters())
            n_total    = sum(p.numel() for p in model.parameters())
            alpha_b    = n_backbone / max(n_total, 1)
        else:
            # Classification models: use layer-group heuristic
            if "groups" not in state.custom:
                state.custom["groups"] = _derive_layer_groups(model)
            backbone_keys, head_keys = _split_backbone_head(
                state.custom["groups"], backbone_fraction
            )
            alpha_b = backbone_fraction

        # ── Freeze head if backbone-only tier ──────────────────────────────
        if is_backbone_only:
            for name, param in model.named_parameters():
                param.requires_grad = (name in backbone_keys)
        else:
            for param in model.parameters():
                param.requires_grad = True

        # ── Save pre-training weights ───────────────────────────────────────
        w_before = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # ── Optimizer ──────────────────────────────────────────────────────
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if optimizer_type == "adam":
            optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(
                trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay
            )

        # ── Local training ─────────────────────────────────────────────────
        model.train()
        total_loss, n_batches = 0.0, 0

        if model_type == "detection":
            # Torchvision detection models: forward in train mode returns loss dict.
            # Dataloader returns (list[Tensor], list[dict]) per batch.
            for _ in range(local_epochs):
                for images, targets in dataloader:
                    images  = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    optimizer.zero_grad()
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches  += 1
        else:
            # Classification models: standard CrossEntropyLoss.
            criterion = nn.CrossEntropyLoss()
            for _ in range(local_epochs):
                for x, y in dataloader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(x), y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches  += 1

        # ── Compute delta (w_before − w_after, gradient direction) ─────────
        w_after = model.state_dict()
        delta = {k: (w_before[k] - w_after[k]).cpu() for k in w_before}

        # ── Zero-out head delta for backbone-only clients ──────────────────
        # Ensures server never accidentally applies stale head gradients.
        if is_backbone_only:
            for k in head_keys:
                if k in delta:
                    delta[k] = torch.zeros_like(delta[k])

        # ── Energy accounting ───────────────────────────────────────────────
        # Backbone-only: active fraction = (1 + α_b) / 2 (compute ratio)
        profile          = config.get("device_profile", {})
        active_fraction  = (1.0 + alpha_b) / 2.0 if is_backbone_only else 1.0
        model_size_bytes = self.compute_model_size_bytes(model)
        bytes_sent       = int(model_size_bytes * alpha_b) if is_backbone_only else model_size_bytes
        e_total          = _energy_j(profile, model, active_fraction, bytes_sent,
                                     energy_scale_factor, config)
        state.battery_j  = max(0.0, state.battery_j - e_total)

        # ── Persist tier info for server ────────────────────────────────────
        state.custom["tier"]     = 0 if is_backbone_only else 1
        state.custom["backbone_fraction"] = backbone_fraction

        metadata = {
            "bytes_sent":         bytes_sent,
            "energy_j":           e_total,
            "compression_ratio":  alpha_b if is_backbone_only else 1.0,
            "local_loss":         total_loss / max(n_batches, 1),
            "tier":               0 if is_backbone_only else 1,
            "battery_ratio":      battery_ratio,
            "is_backbone_only":   is_backbone_only,
            "is_warmup":          is_warmup,
        }

        return delta, metadata

    # ── Server aggregation ────────────────────────────────────────────────────

    def server_aggregate(
        self,
        global_model: nn.Module,
        client_updates: list[tuple[dict, dict, ClientState]],
        round_num: int,
        config: dict,
    ) -> AggregateResult:

        if not client_updates:
            return AggregateResult(
                new_weights=global_model.state_dict(),
                metrics={"error": "no_clients", "round": round_num},
            )

        warmup_rounds     = config.get("warmup_rounds", 5)
        backbone_fraction = config.get("backbone_fraction", 0.70)
        server_lr         = config.get("server_lr", 1.0)
        model_type        = config.get("model_type", "classification")
        is_warmup         = round_num < warmup_rounds

        # ── Determine backbone / head key sets ──────────────────────────────
        # Detection mode: use direct prefix matching (backbone.* vs everything else).
        # Classification mode: use layer-group heuristic.
        if model_type == "detection" and hasattr(global_model, "backbone"):
            all_param_keys = {n for n, _ in global_model.named_parameters()}
            backbone_keys  = {n for n in all_param_keys if n.startswith("backbone.")}
            head_keys      = all_param_keys - backbone_keys
        else:
            groups = None
            for _, _, cs in client_updates:
                groups = cs.custom.get("groups")
                if groups is not None:
                    break
            if groups is None:
                groups = _derive_layer_groups(global_model)
            backbone_keys, head_keys = _split_backbone_head(groups, backbone_fraction)

        bn_keys = _bn_buffer_keys(global_model)

        # ── Separate updates by tier ────────────────────────────────────────
        all_updates:  list[tuple[dict, int]] = []
        full_updates: list[tuple[dict, int]] = []

        for delta, _, client_state in client_updates:
            n = client_state.custom.get("dataset_size", 1)
            tier = client_state.custom.get("tier", 1)
            all_updates.append((delta, n))
            if tier == 1:
                full_updates.append((delta, n))

        # If no full client survived this round, fall back to all clients for head.
        head_source = full_updates if full_updates else all_updates

        # ── Weighted average of deltas per partition ────────────────────────
        backbone_avg = _weighted_average(all_updates,  backbone_keys)
        head_avg     = _weighted_average(head_source,  head_keys)
        bn_avg       = _weighted_average(all_updates,  bn_keys)

        # ── Apply deltas to global model ────────────────────────────────────
        # delta = w_before - w_after, so new = current - server_lr * avg_delta
        # This is equivalent to: new = weighted_avg(client_weights)  when server_lr=1
        new_weights = {k: v.clone() for k, v in global_model.state_dict().items()}

        for name, delta_val in backbone_avg.items():
            new_weights[name] = (
                new_weights[name] - server_lr * delta_val.to(new_weights[name].device)
            )
        for name, delta_val in head_avg.items():
            new_weights[name] = (
                new_weights[name] - server_lr * delta_val.to(new_weights[name].device)
            )
        # BN running stats: apply delta (same formula — avg_val is a delta here)
        for name, delta_val in bn_avg.items():
            if name.endswith("num_batches_tracked"):
                new_weights[name] = (
                    new_weights[name] - delta_val.long().to(new_weights[name].device)
                )
            else:
                new_weights[name] = (
                    new_weights[name] - server_lr * delta_val.to(new_weights[name].device)
                )

        # ── Round metrics ───────────────────────────────────────────────────
        n_backbone_only = sum(
            1 for _, meta, _ in client_updates if meta.get("is_backbone_only", False)
        )
        avg_energy = (
            sum(meta.get("energy_j", 0.0) for _, meta, _ in client_updates)
            / len(client_updates)
        )

        return AggregateResult(
            new_weights=new_weights,
            metrics={
                "round":            round_num,
                "n_clients":        len(client_updates),
                "n_backbone_only":  n_backbone_only,
                "n_full":           len(client_updates) - n_backbone_only,
                "head_fallback":    (len(full_updates) == 0),
                "avg_energy_j":     avg_energy,
                "is_warmup":        is_warmup,
            },
        )

    # ── Default config ────────────────────────────────────────────────────────

    def get_default_config(self) -> dict:
        return {
            # ── Model type ────────────────────────────────────────────────
            # "classification" : standard (x, y) dataloaders, CrossEntropyLoss
            # "detection"      : (images_list, targets_list) dataloaders,
            #                    torchvision detection model native loss
            "model_type":         "classification",

            # ── Optimizer ─────────────────────────────────────────────────
            "optimizer":          "sgd",    # "sgd" | "adam"
            "lr":                 0.01,
            "momentum":           0.9,      # SGD only (ignored for adam)
            "weight_decay":       1e-4,
            "local_epochs":       5,
            "batch_size":         32,

            # ── Backbone / head split ──────────────────────────────────────
            # For classification models: fraction of layer groups → backbone.
            #   ResNet-8 (10 groups): 0.70 → groups 0-6 backbone, 7-9 head.
            # For detection models (model_type="detection"): backbone_fraction
            #   is computed automatically from model.backbone parameter count.
            #   SSDLite320: backbone=88.1%, head=11.9% (auto-detected).
            "backbone_fraction":  0.70,

            # Battery ratio threshold: clients with B_t/B_max < beta_split
            # are assigned to Tier 0 (backbone-only training).
            # 0.40 = 40% of battery remaining → saves battery for late rounds.
            "beta_split":         0.40,

            # ── Battery / energy ──────────────────────────────────────────
            "battery_max_j":      13320.0,   # ESP32-S3 nominal: 1000mAh × 3.7V × 3.6
            "energy_scale_factor": 1.0,

            # ── Aggregation ────────────────────────────────────────────────
            "server_lr":          1.0,

            # ── Training phases ────────────────────────────────────────────
            "warmup_rounds":      5,         # rounds of full FedAvg before tier split
        }
