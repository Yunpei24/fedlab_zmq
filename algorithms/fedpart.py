"""
algorithms/fedpart.py
=====================
FedPart: Partial Network Updates for Federated Learning
------------------------------------------------------------------------
Paper : "Why Go Full? Elevating Federated Learning Through Partial
         Network Updates"
Authors: Haolin Wang, Xuefeng Liu, Jianwei Niu, Wenkai Guo, Shaojie Tang
Venue  : NeurIPS 2024  (arXiv: 2410.11559v3)
Source : https://github.com/FLAIR-Community/Fling

Problem solved
--------------
Standard FedAvg suffers from "layer mismatch": after parameter averaging,
the gradient update step sizes spike, indicating that the averaged layers
no longer cooperate well.  This slows convergence and degrades final
accuracy.

Key idea
--------
Each communication round trains and aggregates ONLY ONE group of layers
(a "layer group") while all other layers are frozen.  The frozen layers
act as anchors, constraining the trainable layer's update direction and
preventing mismatch.  Layer groups are visited sequentially from shallow
to deep, then the cycle repeats (multi-round cycle training).

Algorithm (from Section 3, paper):

  Phase 0 — Warmup (optional)
      Run W rounds of full FedAvg to initialise the model.

  Phase 1 — Partial Network Updates (PNU) — main loop
      Round t:
        1. Server determines the active layer group index:
               g = (t - W) // R  mod  M
           where R = rounds_per_layer, M = num_layer_groups.
        2. Server broadcasts the global model AND the group index g.
        3. Client freezes all layers except group g.
        4. Client runs E local epochs of SGD/Adam on the trainable subset.
        5. Client uploads ONLY the delta for layer group g.
        6. Server averages the partial deltas and updates ONLY group g.

  Training rounds R/L and training cycles C (Section 3.2, Table 5):
      R/L (rounds_per_layer): number of consecutive rounds each layer group
           is trained before rotating to the next group.
           "2 R/L" means each group gets 2 consecutive rounds.  Default: 2.
      C   (training_cycles) : total number of complete passes through all
           M layer groups.  One full cycle = M x R/L rounds.
           Total PNU rounds = C x M x R/L.
           After C complete cycles the algorithm keeps cycling (the fixed
           --rounds parameter determines the actual stopping point).
           Default: 5.

  Layer group selection strategies (Section 3.2, ablation Table 7):
      sequential (default) : group 0, 1, 2, ..., M-1, 0, 1, ...   best
      reverse              : M-1, M-2, ..., 0, M-1, ...
      random               : uniform random each round

  The binary mask S_t used in the paper's formula (Eq. 1):
      w_i^{t+1} = w_i^t - lr * S_t ⊙ ∇_w L(x | w_i^t)
  is implemented by freezing parameters (requires_grad=False) outside
  the active group, so gradients are never computed for frozen layers.

Layer partitioning
------------------
The paper partitions the model into M named parameter groups.  We derive
groups automatically from the model's named_parameters() by splitting on
top-level module names (e.g., "layer1", "layer2", "fc").  This is
model-agnostic and matches the paper's intent.

Convergence guarantee (Theorem 1, Appendix B)
---------------------------------------------
Under L-smoothness (Assumption 1), bounded gradient variance
(Assumption 2), and mask-invariance of gradient variance (Assumption 3):

    (1/T) Σ_{t=1}^T E[||S_t ⊙ ∇f(w̄^{t-1})||^2] = O(1 / sqrt(M·N·T))

where N = number of clients, T = total rounds, M = number of layer groups.
This is superior to FedAvg's O(1/sqrt(N·T)) by a factor of sqrt(M).

Communication cost: 1/M of FedAvg per partial round.
Computational cost: ~2/3 of FedAvg per cycle (Eq. 5 in paper).

Key differences from FedAvg in this implementation
---------------------------------------------------
* client_update() freezes all layers except the active group before
  local training; only the active group's delta is returned.
* server_aggregate() merges only the active group's parameters into the
  global model; other parameters are carried forward unchanged.
* The active group index is passed through config["active_group_idx"],
  set by the caller (server/orchestrator) based on the round schedule.
  During warmup rounds (round_num < warmup_rounds), active_group_idx=-1
  triggers full-network training identical to FedAvg.
* Layer groups are derived lazily from the model the first time
  client_update() is called; the result is stored in state.custom.
* Optimizer: SGD with momentum (momentum=0.9, weight_decay=1e-4).
  SGD with momentum is preferred over Adam in the current FL literature
  for better generalization and convergence stability under non-IID data.
* rounds_per_layer (R/L): consecutive rounds each group is trained before
  rotating.  "2 R/L" = 2 rounds per group per cycle.  Default: 2.
* training_cycles (C): total number of complete cycles through all M groups.
  Informational + used to compute expected total PNU rounds = C x M x R/L.
  Default: 5.
"""

import gc
import random
import re
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from hardware.flop_cost import compute_group_flops as _flopcost_compute_group_flops
from hardware.flop_cost import freeze_to_trainable as _flopcost_freeze_to_trainable
from hardware.flop_cost import restore_grad as _flopcost_restore_grad
from hardware.flop_cost import (
    round_compute_flops,
)

from .base import AggregateResult, ClientState, FLAlgorithm, register_algorithm

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _param_group_key(param_name: str) -> str:
    """
    Map a parameter name to its semantic group key.

    Rules (applied in order):
    1. Stem modules (stem.*, pre_conv.*, pre_bn.*) → "stem"
    2. Shortcut/downsample (any path containing that keyword) → "<parent>.shortcut"
    3. Named conv/bn pairs (leaf is convN or bnN) → "<parent>._pairN"
    4. FC / head → top-level name ("fc", "head", "classifier", "linear")
    5. Fallback → module path (everything except trailing weight/bias)

    This produces exactly 10 groups for ResNet-8 (matching Fig. 4 in the
    FedPart paper):
      stem | layer1.pair1 | layer1.pair2 |
      layer2.pair1 | layer2.pair2 | layer2.shortcut |
      layer3.pair1 | layer3.pair2 | layer3.shortcut | fc
    """
    parts = param_name.split(".")
    # Strip non-parameter trailing tokens
    if parts[-1] in (
        "weight",
        "bias",
        "running_mean",
        "running_var",
        "num_batches_tracked",
    ):
        parts = parts[:-1]
    if not parts:
        return param_name

    top = parts[0]

    # 1. Stem
    if top in ("stem", "pre_conv", "pre_bn"):
        return "stem"

    # 2. Shortcut / downsample — group ALL sub-params together
    for kw in ("shortcut", "downsample"):
        if kw in parts:
            idx = parts.index(kw)
            return ".".join(parts[: idx + 1])

    # 3. conv+bn pairing: leaf looks like conv1, bn1, norm1, etc.
    leaf = parts[-1]
    m = re.match(r"^(?:conv|bn|norm)(\d+)$", leaf, re.IGNORECASE)
    if m:
        parent = ".".join(parts[:-1]) if len(parts) > 1 else ""
        return f"{parent}._pair{m.group(1)}" if parent else f"_pair{m.group(1)}"

    # 4. FC / head
    if top in ("fc", "head", "classifier", "linear"):
        return top

    # 5. Fallback: use full module path
    return ".".join(parts)


def _derive_layer_groups(model: nn.Module) -> list[list[str]]:
    """
    Partition model parameters into semantic conv+BN groups (FedPart, Fig. 4).

    For ResNet-8 → exactly 10 groups:
      stem | layer1.pair1 | layer1.pair2 |
      layer2.pair1 | layer2.pair2 | layer2.shortcut |
      layer3.pair1 | layer3.pair2 | layer3.shortcut | fc

    Works generically for any ResNet-like architecture (our ResNet8/18/50,
    Fling ResNet with pre_conv/layers.X.0.convY naming, etc.).
    Shortcut/downsample layers are always grouped separately so that
    skip connections can be updated independently.
    """
    group_dict: dict[str, list[str]] = OrderedDict()
    for name, _ in model.named_parameters():
        key = _param_group_key(name)
        group_dict.setdefault(key, []).append(name)

    # Sort: stem first, fc/head last, all others by natural key order
    def _sort_key(k: str) -> tuple:
        if k == "stem":
            return (0, k)
        if k in ("fc", "head", "classifier", "linear"):
            return (2, k)
        return (1, k)

    sorted_keys = sorted(group_dict.keys(), key=_sort_key)
    return [group_dict[k] for k in sorted_keys]


def _compute_group_flops(
    groups: list[list[str]],
    model: nn.Module,
    input_shape: tuple = (1, 3, 32, 32),
) -> list[int]:
    """Backward-compatible re-export of hardware.flop_cost.compute_group_flops.

    The implementation lives in `hardware/flop_cost.py` so that every algorithm
    sees the same per-group FLOPs (the basis of both phi and corrected cost
    models).
    """
    return _flopcost_compute_group_flops(groups, model, input_shape)


def _active_group_idx_for_round(round_num: int, config: dict) -> int:
    """
    Compute which layer group is active at the given round.

    During warmup (round_num < warmup_rounds): returns -1 → full update.
    After warmup: cycles through groups sequentially / reverse / random
    using rounds_per_layer rounds per group per cycle.

    Inter-cycle rounds (inter_cycle_rounds > 0):
      After each complete cycle of M × R/L PNU rounds, insert
      inter_cycle_rounds full-network training rounds before the next cycle.
      These rounds also return -1, identical to warmup in behavior.

      Example (M=10, R/L=2, inter_cycle_rounds=5):
        cycle_length = 10 × 2 = 20 PNU rounds
        full_cycle   = 20 + 5 = 25 rounds total
        rounds 0–19 : PNU (groups 0–9, 2 rounds each)
        rounds 20–24: inter-cycle full training  ← returns -1
        rounds 25–44: PNU (next cycle)
        ...

      Set inter_cycle_rounds=0 (default) to disable → backward-compatible.

    Design note: the caller (server) can also inject active_group_idx
    directly via config to override scheduling (e.g., for experiments).
    """
    # Allow explicit override from server/config
    if "active_group_idx" in config and config["active_group_idx"] is not None:
        return int(config["active_group_idx"])

    warmup = config.get("warmup_rounds", 5)
    if round_num < warmup:
        return -1  # full-network warmup round

    num_groups = config.get("num_layer_groups", 10)
    rpl = config.get("rounds_per_layer", 2)
    strategy = config.get("layer_selection", "sequential")
    inter_cycle = config.get("inter_cycle_rounds", 0)  # 0 = disabled

    pnu_round = round_num - warmup  # index within PNU phase (0-based)

    if inter_cycle > 0:
        cycle_length = num_groups * rpl  # PNU rounds per cycle
        full_cycle = cycle_length + inter_cycle  # total rounds per cycle
        pos_in_cycle = pnu_round % full_cycle  # position within the current cycle

        if pos_in_cycle >= cycle_length:
            return -1  # inter-cycle full-network training round

        # Re-map position to within-cycle PNU index (strips inter-cycle offset)
        pnu_round = pos_in_cycle

    if strategy == "sequential":
        return (pnu_round // rpl) % num_groups
    elif strategy == "reverse":
        return (num_groups - 1) - (pnu_round // rpl) % num_groups
    elif strategy == "random":
        # Deterministic randomness: seed with round number for reproducibility
        rng = random.Random(round_num)
        return rng.randint(0, num_groups - 1)
    else:
        raise ValueError(
            f"Unknown layer_selection strategy: {strategy!r}. "
            "Choose from: 'sequential', 'reverse', 'random'."
        )


def _freeze_all_except_group(
    model: nn.Module,
    groups: list[list[str]],
    active_idx: int,
) -> None:
    """Backward-compatible wrapper around hardware.flop_cost.freeze_to_trainable.

    `active_idx == -1` => every parameter trainable (full-network update),
    consistent with the pre-refactor semantics.
    """
    if active_idx < 0:
        trainable = [n for n, _ in model.named_parameters()]
    else:
        trainable = list(groups[active_idx])
    _flopcost_freeze_to_trainable(model, trainable)


def _restore_all_grad(model: nn.Module) -> None:
    """Re-enable gradients for all parameters (post-training cleanup)."""
    _flopcost_restore_grad(model)


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm
# ─────────────────────────────────────────────────────────────────────────────


@register_algorithm("fedpart")
class FedPart(FLAlgorithm):
    """
    FedPart: Partial Network Updates (Wang et al., NeurIPS 2024).

    Trains and uploads only one layer group per round to eliminate layer
    mismatch in federated averaging.  Reduces communication by 1/M and
    computation by ~1/3 compared to full-network FedAvg.

    Registration key: "fedpart"
    """

    name = "fedpart"
    description = (
        "FedPart (Wang et al., NeurIPS 2024). Partial network updates: "
        "one layer group trained and aggregated per round to fix layer mismatch."
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
        Local training step.

        1. Derive layer groups if not yet cached in state.custom.
        2. Determine active group for this round.
        3. Freeze all layers except the active group.
        4. Run E local epochs of SGD (momentum) on the trainable subset.
        5. Compute delta only for the active group's parameters.
        6. Return (partial_delta, metadata).
        """
        device = config.get("device", "cpu")
        lr = config.get("lr", 0.01)
        momentum = config.get("momentum", 0.9)
        weight_decay = config.get("weight_decay", 1e-4)
        local_epochs = config.get("local_epochs", 8)  # paper uses E=8
        max_grad_norm = config.get("max_grad_norm", None)

        model.train()
        model.to(device)

        # ── Lazy derivation of layer groups (model-agnostic) ─────────────────
        if "layer_groups" not in state.custom:
            groups = _derive_layer_groups(model)
            state.custom["layer_groups"] = groups
            if config.get("verbose_groups", True) and state.client_id == 0:
                # Print layer group layout once (client 0 only — all clients share the same architecture)
                lines = []
                for i, g in enumerate(groups):
                    key = _param_group_key(g[0])
                    lines.append(f"    [{i:2d}] {key:<30s} ({len(g)} params)")
                print(
                    f"  [FedPart] {len(groups)} layer groups derived "
                    f"(ResNet architecture, shared across all clients)"
                )
                print("\n".join(lines))
        groups: list[list[str]] = state.custom["layer_groups"]

        # Always clamp num_layer_groups to available groups so the scheduler
        # never returns an index >= len(groups) (e.g. default 10 > resnet8's 4)
        config["num_layer_groups"] = min(
            config.get("num_layer_groups", len(groups)), len(groups)
        )
        num_groups = config["num_layer_groups"]

        # ── Determine active group ────────────────────────────────────────────
        active_idx = _active_group_idx_for_round(state.round_num, config)
        is_warmup = active_idx < 0

        # ── Save weights before local training ────────────────────────────────
        w_before = OrderedDict(
            {k: v.clone().cpu() for k, v in model.state_dict().items()}
        )

        # ── Freeze layers outside active group ───────────────────────────────
        _freeze_all_except_group(model, groups, active_idx)

        # ── Local training ────────────────────────────────────────────────────
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer_type = config.get("optimizer", "sgd").lower()
        if optimizer_type == "adam":
            optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
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
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        # ── Restore full gradient capability (cleanup) ────────────────────────
        _restore_all_grad(model)

        # ── Compute delta — ONLY for active group's parameters ───────────────
        #
        # Design note: during warmup (active_idx == -1) we send the full delta,
        # identical to FedAvg.  During PNU phase we send only the active group.
        # The server must know the active group to reconstruct the new global
        # model correctly (it receives active_group_idx in metadata).
        #
        # IMPORTANT — BatchNorm running statistics:
        # _derive_layer_groups is built from named_parameters(), which EXCLUDES
        # buffers (running_mean, running_var, num_batches_tracked).  These buffers
        # update on every forward pass regardless of requires_grad.  If they are
        # not included in the partial delta, the global model accumulates stale BN
        # statistics for all groups except the active one → catastrophic divergence.
        # Fix: always append ALL BN buffer keys to the transmitted delta.
        #
        current_sd = model.state_dict()
        bn_buffer_keys = [
            k
            for k in w_before
            if k.endswith(("running_mean", "running_var", "num_batches_tracked"))
        ]
        if is_warmup:
            # Full delta — all parameters + BN buffers (already included)
            partial_delta = OrderedDict(
                {k: (w_before[k] - current_sd[k].cpu()).float() for k in w_before}
            )
            transmitted_keys = list(w_before.keys())
        else:
            # Partial delta — active group's learnable params + ALL BN running stats.
            # BN stats are tiny (few hundred floats) but critical for normalisation.
            active_keys = groups[active_idx]
            active_keys_set = set(active_keys)
            all_tx_keys = list(active_keys) + [
                k for k in bn_buffer_keys if k not in active_keys_set
            ]
            partial_delta = OrderedDict(
                {k: (w_before[k] - current_sd[k].cpu()).float() for k in all_tx_keys}
            )
            transmitted_keys = all_tx_keys

        # ── Communication / energy accounting ─────────────────────────────────
        uplink_bytes = self.count_bytes(partial_delta, sparse=False)
        # full_model_bytes: size of the complete model (downlink = received from server,
        # full_upload_ref = reference for compression ratio computation).
        # Both are the same value; compute once before freeing w_before.
        full_model_bytes = self.count_bytes(w_before, sparse=False)
        downlink_bytes = full_model_bytes  # client receives the full global model
        full_upload_bytes_ref = full_model_bytes
        # w_before is now fully consumed (delta computed, sizes counted).
        # Free it immediately — it is a full state_dict clone and the dominant
        # per-round memory cost with large models.
        del w_before
        del current_sd
        gc.collect()

        profile = config.get("device_profile")

        # ── Compute FLOPs via the unified flop_cost dispatcher ───────────────
        # The algorithm's only job here is to declare which parameters carry
        # gradients during local training; flop_cost.round_compute_flops picks
        # the cost_model (phi / corrected / measured) and returns FLOPs.
        if not is_warmup and "group_flops" not in state.custom:
            input_shape = config.get("input_shape", (1, 3, 32, 32))
            state.custom["group_flops"] = _compute_group_flops(
                groups,
                model,
                input_shape,
            )

        if is_warmup:
            trainable_names = [n for n, _ in model.named_parameters()]
            _gf_hint = None
        else:
            trainable_names = list(groups[active_idx])
            _gf_hint = state.custom["group_flops"]

        if profile:
            effective_flops = round_compute_flops(
                model,
                trainable_names,
                config,
                profile,
                dataloader,
                local_epochs,
                groups=groups,
                active_group_idx=(-1 if is_warmup else active_idx),
                group_flops_analytic=_gf_hint,
            )
            energy_j = profile.round_energy_j(
                effective_flops, uplink_bytes, downlink_bytes
            )
        else:
            # Profile-less fallback retained for tests / unit smoke runs.
            if is_warmup:
                _frac = 1.0
            else:
                _gf = state.custom["group_flops"]
                _frac = _gf[active_idx] / sum(_gf)
            energy_j = 0.5 + 2.0 * _frac

        # ── Energy scale factor (calibration for experiments) ────────────────
        # Allows YAML configs to amplify per-round energy to produce
        # realistic dropout timelines in simulations with small models.
        energy_j *= config.get("energy_scale_factor", 1.0)

        # ── Battery update (invariant: never negative) ─────────────────────────
        state.battery_j = max(0.0, state.battery_j - energy_j)
        state.round_num += 1

        # ── Compression ratio relative to full-model upload ───────────────────
        # full_upload_bytes_ref was computed before w_before was freed above.
        compression_ratio = uplink_bytes / max(full_upload_bytes_ref, 1)

        metadata = {
            # Mandatory fields (framework invariant)
            "client_id": state.client_id,
            "round_num": state.round_num,
            "beta_actual": 1.0,  # no sparsification; full precision
            "battery_j_remaining": state.battery_j,
            "energy_j_consumed": energy_j,
            "bytes_sent": uplink_bytes,
            "bytes_received": downlink_bytes,
            "local_loss": total_loss / max(num_batches, 1),
            "compression_ratio": compression_ratio,
            # FedPart-specific
            "active_group_idx": active_idx,
            "is_warmup": is_warmup,
            "num_layer_groups": num_groups,
            "transmitted_keys": transmitted_keys,
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
        Average partial deltas and update ONLY the active group's parameters.

        During warmup (active_group_idx == -1): full FedAvg aggregation.
        During PNU phase: only the active group's parameters are updated;
        all other parameters in the global model are left unchanged.

        The active group index is read from the first client's metadata
        (all clients in the same round share the same active group).
        """
        K = len(client_updates)
        global_sd = global_model.state_dict()

        # ── Determine active group from client metadata ────────────────────────
        # All clients in a round train the same group; take index from first.
        first_meta = client_updates[0][1]
        active_idx = first_meta.get("active_group_idx", -1)
        is_warmup = first_meta.get("is_warmup", True)

        # Dataset-size weights for BN aggregation
        sizes = [m.get("dataset_size", 1) for _, m, _ in client_updates]
        total_n = max(sum(sizes), 1)

        # ── Average partial deltas (uniform) + weighted BN accumulation ───────
        agg: Optional[dict] = None
        agg_bn: dict = {}
        for (update, meta, _), n_k in zip(client_updates, sizes):
            w_k = n_k / total_n
            if agg is None:
                agg = {k: v.clone().float() for k, v in update.items()}
            else:
                for k in agg:
                    agg[k] += update[k].float()
            for k, v in update.items():
                if k.endswith((".running_mean", ".running_var")):
                    if k not in agg_bn:
                        agg_bn[k] = v.clone().float() * w_k
                    else:
                        agg_bn[k] += v.float() * w_k
        for k in agg:
            agg[k] /= K

        # ── Apply averaged delta to global model ───────────────────────────────
        # Deltas are always on CPU (computed via .cpu() in client_update).
        # Global model may be on any device (cpu/cuda/mps). Move delta to match.
        new_weights = OrderedDict()
        for k, v in global_sd.items():
            if k in agg:
                new_weights[k] = v.float() - agg[k].to(v.device)
            else:
                new_weights[k] = v.float()
        # Override BN running stats with dataset-size-weighted average
        for k, v in agg_bn.items():
            if k in new_weights:
                new_weights[k] = global_sd[k].float() - v.to(global_sd[k].device)
        # Free the accumulated gradient tensors — they are fully consumed.
        del agg
        gc.collect()

        # ── Metrics ───────────────────────────────────────────────────────────
        total_bytes = sum(m["bytes_sent"] for _, m, _ in client_updates)
        total_energy = sum(m["energy_j_consumed"] for _, m, _ in client_updates)
        avg_battery = sum(s.battery_j for _, _, s in client_updates) / K
        avg_loss = sum(m["local_loss"] for _, m, _ in client_updates) / K
        avg_cr = sum(m["compression_ratio"] for _, m, _ in client_updates) / K

        participations = [1.0 if s.battery_j > 0 else 0.0 for _, _, s in client_updates]
        p2_sum = sum(p**2 for p in participations)
        jain = (sum(participations) ** 2) / (K * p2_sum) if p2_sum > 0 else 0.0

        return AggregateResult(
            new_weights=new_weights,
            metrics={
                "round": round_num,
                "total_bytes_sent": total_bytes,
                "total_energy_j": total_energy,
                "avg_beta": 1.0,
                "avg_battery_j": avg_battery,
                "avg_local_loss": avg_loss,
                "compression_ratio": avg_cr,
                "participation_rate": sum(participations) / K,
                "jain_index": jain,
                "num_clients": K,
                # FedPart-specific metrics
                "active_group_idx": active_idx,
                "is_warmup_round": is_warmup,
            },
        )

    # ── Default configuration ──────────────────────────────────────────────────

    def get_default_config(self) -> dict:
        """
        Default hyperparameters matching the paper's experimental setup
        (Section 4, Table 1, Appendix F.1).

        lr              : 0.01   — SGD with momentum
        momentum        : 0.9    — SGD momentum
        weight_decay    : 1e-4   — L2 regularization
        local_epochs    : 8      — paper default (Section 4)
        batch_size      : 32     — standard
        warmup_rounds   : 5      — initial full FedAvg rounds (Table 6)
        rounds_per_layer: 2      — R/L: consecutive rounds per layer group
                                   before rotating ("2 R/L" in paper, Table 5)
        training_cycles : 5      — C: total complete cycles through all M
                                   groups.  Total PNU rounds = C × M × R/L.
                                   Actual stopping is controlled by --rounds.
        num_layer_groups: 10     — M, derived lazily from model at runtime;
                                   ResNet-8 has 10 groups (Appendix A)
        layer_selection : "sequential" — shallow-to-deep; best per Table 7
        active_group_idx: None   — None = compute from round schedule;
                                   set to int to override (server-side)
        device          : "cpu"
        device_profile  : None
        """
        return {
            "optimizer": "sgd",  # "sgd" | "adam"
            "lr": 0.01,
            "momentum": 0.9,  # SGD only (ignored when optimizer=adam)
            "weight_decay": 1e-4,
            "local_epochs": 8,
            "batch_size": 32,
            "warmup_rounds": 5,
            "rounds_per_layer": 2,  # R/L — consecutive rounds per group
            "training_cycles": 5,  # C  — total cycles through all groups
            "num_layer_groups": 10,  # M, overwritten at runtime from model
            "layer_selection": "sequential",
            "active_group_idx": None,  # None = use round-based schedule
            "device": "cpu",
            "device_profile": None,
        }
