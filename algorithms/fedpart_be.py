"""
algorithms/fedpart_be.py
========================
FedPartBE: Battery-Energy-aware Federated Partial Network Updates
Extension of FedPart (Wang et al., NeurIPS 2024) with energy-tier assignments.

Key innovations vs FedPart:
  1. Energy-tier layer assignment: sort clients by battery, match cheapest
     groups to lowest-battery clients.
  2. Representation-space proximal regularization: constrains the OUTPUT of
     the active group to stay close to global model output, preserving
     inter-layer compatibility.
  3. Staleness-aware layer priority: groups not updated recently get higher
     scheduling priority.
  4. Sequential server aggregation (Gauss-Seidel order): apply group deltas
     in architectural order (early layers first).
  5. Dynamic tier reorganization: server reassigns client→group mapping
     every round based on battery levels.

Algorithm:
  Round t < warmup_rounds: full FedAvg update (identical to FedPart).

  Round t >= warmup_rounds:
    Server:
      1. Receives battery levels {B_k} from client states
      2. Computes layer group costs: cost[g] = sum of param counts in group g
      3. Computes priority[g] = (staleness[g] + 1) / (cost[g] / max_cost)
      4. Sorts clients by B_k ascending → K tiers (K = num_tiers)
      5. Sorts groups by priority DESCENDING (most urgent group first)
      6. Assigns tier_i → group_i (tier 0 = lowest battery → highest priority)
      7. Broadcasts: per-client group assignment + global model

    Client k (assigned to group g_k):
      1. Receives global model + group assignment g_k
      2. Freezes all layers except group g_k
      3. Computes reference representation: h_ref = forward through group g_k
         using GLOBAL weights on a small reference batch
      4. Trains for local_epochs with loss:
           L = CE_loss
               + mu_weight * ||W_gk - W_gk_global||^2_F   (weight proximal)
               + mu_repr   * MSE(h_gk(x), h_ref(x))       (representation proximal)
      5. Uploads ONLY delta for group g_k

    Server aggregation (single-pass weighted accumulation, Jacobi-style):

      Step 1 — weighted accumulation (one pass over all client updates):
        For each active client k assigned to group g_k:
          group_weighted_sums[params of g_k] += delta_k * dataset_size_k
          group_sizes[g_k] += dataset_size_k

      Step 2 — normalize per group + update staleness + EMA grad norms:
        For g in range(num_groups):
          IF group g received at least one update:
            group_weighted_sums[g] /= group_sizes[g]     # dataset-size weighted mean
            ema_grad_norms[g] = (1-α)*ema_grad_norms[g] + α*||delta_g||  # α=0.3
            staleness[g] = 0
          ELSE:
            staleness[g] += 1

      Step 3 — fused GPU apply with server learning rate:
        global_model[params] -= server_lr * normalized_delta[params]
        server_lr = 1 / num_tiers   (auto, configurable)
        # server_lr prevents multi-group displacement divergence:
        # with num_tiers groups updated simultaneously, total model shift
        # would be num_tiers × per-group delta without this scaling.

      Step 4 — BN running stats (separate, global):
        BN running_mean / running_var / num_batches_tracked aggregated as
        dataset-size weighted mean across ALL active clients (not just per-group),
        because all clients do a full forward pass regardless of frozen groups.

Convergence: O(1/sqrt(T)) to a neighborhood of size O(M*sigma^2/K + tau_max^2).
  - Formally proved in convergence_proof.tex, Theorem thm:neighborhood.
  - Fixed step-size eta: converges to neighborhood eta*L*(M*sigma^2/K + L*G^2*tau_max^2).
    Neighborhood shrinks as K increases (more clients) or M decreases (fewer tiers).
    Optimal M: M* = (2*L*K*G^2*T_rot^2 / sigma^2)^(1/3) (Remark rem:neighborhood).
  - Diminishing step-size eta_t = c/sqrt(T): full O(1/sqrt(T)) convergence to zero.
  - FedAvg recovered exactly for M=1, G=1 (Corollary cor:fedavg_recovery).

Energy benefit: Low-battery clients train cheaper groups, extending lifetime.

Reference: FedPart (Wang et al., NeurIPS 2024, arXiv:2410.11559v3).
"""

import copy
import gc
import math
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from hardware.flop_cost import (
    compute_corrected_group_costs as _flopcost_compute_corrected_group_costs,
)
from hardware.flop_cost import freeze_to_trainable as _flopcost_freeze_to_trainable
from hardware.flop_cost import restore_grad as _flopcost_restore_grad
from hardware.flop_cost import (
    round_compute_flops,
)

from .base import AggregateResult, ClientState, FLAlgorithm, register_algorithm
from .fedpart import _compute_group_flops, _derive_layer_groups, _param_group_key

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# def _compute_group_costs(groups: list[list[str]], model: nn.Module) -> list[int]:
#     """
#     Compute cost (total param count) for each layer group.
#     Kept as fallback if FLOPs computation fails.
#     """
#     state_dict = model.state_dict()
#     costs = []
#     for group in groups:
#         cost = sum(state_dict[k].numel() for k in group if k in state_dict)
#         costs.append(max(cost, 1))
#     return costs


def _compute_corrected_group_costs(group_flops: list[float]) -> list[float]:
    """Backward-compatible re-export of compute_corrected_group_costs.

    Position-aware analytic cost (each group pays for its own backward plus
    half the FLOPs of every downstream group it must propagate grad_input
    through). The canonical implementation lives in `hardware/flop_cost.py`
    so that the cost model used for tier ASSIGNMENT (this function) and the
    cost model used for ENERGY ACCOUNTING (round_compute_flops) cannot drift
    apart again.
    """
    return _flopcost_compute_corrected_group_costs(group_flops)


def _assign_tiers_to_groups_v2(
    num_tiers: int,
    num_groups: int,
    costs: list[float],
    staleness: list[int],
    grad_norms: list[float],
    bucket_shift: int = 0,
    enforce_staleness_cap: bool = True,
) -> dict[int, int]:
    """
    Cost-bucketed, staleness-driven group selection.

    Fixes the perpetual-group-0 bug of the old priority-score assignment:
      The old formula (staleness+1)*norm_grad/norm_cost lets a cheap group
      with high gradient norm dominate regardless of staleness, because
      1/norm_cost for the stem layer (very cheap) overwhelms the staleness term.
      Result: Group 0 is trained every round → catastrophic divergence.

    New design:
      1. Sort groups by cost ascending.
      1b. Force the head group (last architectural group, e.g. fc layer) to the
          END of the sorted list, regardless of its FLOPs cost. This ensures it
          always falls into the most-expensive bucket (Tier num_tiers-1) and is
          trained exclusively by high-battery clients who better represent the full
          data distribution. Skipping this step causes classification head divergence
          in non-IID settings because the fc layer is cheap in FLOPs but critical
          to output quality.
      2. Partition into num_tiers cost-buckets:
           Tier 0 (lowest battery) → cheapest bucket
           Tier K-1 (highest battery) → most expensive bucket (always contains head)
      3. Within each bucket, select the group with the HIGHEST staleness.
         Ties broken by gradient norm / cost (secondary importance).

    Guarantees:
      - Tier 0 always trains a cheap group (energy-safe for low-battery clients)
      - Head group always trained by highest-battery tier (stable classification)
      - Perfect rotation: each group in a bucket gets trained in round-robin order
      - No starvation: staleness monotonically drives selection within each bucket
      - No perpetual lock-in: once a group is trained (staleness resets to 0),
        it has the LOWEST priority in its bucket and will not be picked again
        until all other groups in the bucket have been trained.

    Args:
        num_tiers:  number of energy tiers (K)
        num_groups: total number of layer groups (M)
        costs:      FLOPs (or param count) per group
        staleness:  rounds since each group was last updated
        grad_norms: EMA-smoothed L2 delta norm per group

    Returns:
        tier_assignment: dict mapping tier_idx → group_idx
    """
    # 1. Compute corrected costs that include downstream grad_input propagation.
    #    Raw FLOPs (costs[g]) give the WRONG energy ranking for sequential nets:
    #    stem appears cheap (small FLOPs) but backward must propagate grad_input
    #    through ALL n-1 downstream layers → actually the MOST expensive group.
    #    Corrected cost_p = gf[p] + 0.5 × Σ_{i>p} gf[i]
    corrected_costs = _compute_corrected_group_costs(costs)

    # Sort groups by corrected cost ascending (cheapest backward-corrected cost first).
    # With corrected costs, LATE layers (near output) are cheapest because they
    # have few downstream layers to propagate grad_input through.
    # EARLY layers (near input) are most expensive despite small raw FLOPs.
    cost_sorted = sorted(range(num_groups), key=lambda g: corrected_costs[g])

    # Force the head (last architectural group, fc) to the highest-battery tier.
    # Semantic reason: fc is the classification head. In non-IID settings, training
    # it with only low-battery clients (which may represent a biased data subset)
    # causes classification head divergence. High-battery clients see more diverse
    # data (they participate more rounds) → better head quality.
    # Note: with corrected costs, fc is ALREADY the cheapest group (no downstream).
    # Forcing it to tier K-1 preserves the semantic guarantee at the cost of giving
    # high-battery clients the cheapest group to train — acceptable because their
    # energy budget is not the binding constraint.
    head_group = num_groups - 1  # fc head is always last in architectural order
    if head_group in cost_sorted and cost_sorted.index(head_group) < num_groups - 1:
        cost_sorted.remove(head_group)
        cost_sorted.append(head_group)

    # 2. Partition into num_tiers cost-buckets
    bucket_size = max(1, math.ceil(num_groups / num_tiers))
    buckets = [
        cost_sorted[t * bucket_size : min((t + 1) * bucket_size, num_groups)]
        for t in range(num_tiers)
    ]
    # Handle edge case: num_tiers > num_groups → wrap around
    for t in range(num_tiers):
        if not buckets[t]:
            buckets[t] = [cost_sorted[t % num_groups]]

    # 3. Within each bucket, pick the group with highest staleness.
    #    Tiebreak: higher grad_norm / corrected_cost (importance per joule).
    #
    #    Cyclic bucket rotation (bucket_shift > 0):
    #      Tier t draws from bucket (t + bucket_shift) % num_tiers instead of bucket t.
    #      This rotates which energy stratum trains each cost bucket over time:
    #        Phase 0: tier 0 → cheapest bucket,  tier 2 → most expensive bucket (normal)
    #        Phase 1: tier 0 → middle bucket,    tier 1 → most expensive bucket, ...
    #        Phase 2: tier 0 → most expensive,   tier 1 → cheapest, ...
    #      Over a full rotation cycle (num_tiers phases), each group is trained
    #      by all energy strata → reduces the permanent data-distribution bias
    #      that occurs when cheap groups are ALWAYS trained by low-battery clients.
    #      Energy safety: if tier-0 receives an expensive group during rotation,
    #      the client-side energy gate falls back to the cheapest affordable group.
    tier_assignment = {}
    for t in range(num_tiers):
        source_bucket = buckets[(t + bucket_shift) % num_tiers]
        best_g = max(
            source_bucket,
            key=lambda g: (staleness[g], grad_norms[g] / max(corrected_costs[g], 1e-8)),
        )
        tier_assignment[t] = best_g

    # Hard staleness cap: enforce τ_max = ceil(G/M) - 1 deterministically.
    # If any group has reached the staleness cap it MUST be included in the
    # active set for this round, regardless of priority score.  Groups at the
    # cap are collected and substituted into tier slots one-by-one, cheapest
    # corrected-cost first (to preserve the energy-safety invariant as much
    # as possible).  Ties within overdue groups are broken by corrected cost
    # ascending so that the cheapest overdue group displaces the most
    # expensive already-assigned tier.
    if enforce_staleness_cap:
        tau_max = max(0, math.ceil(num_groups / num_tiers) - 1)
        already_covered = set(tier_assignment.values())
        overdue = [
            g
            for g in range(num_groups)
            if staleness[g] >= tau_max and g not in already_covered
        ]
        if overdue:
            # Sort overdue groups cheapest corrected cost first.
            overdue_sorted = sorted(overdue, key=lambda g: corrected_costs[g])
            # Find tiers not already serving an overdue group, sorted by how
            # expensive their currently assigned group is (most expensive first
            # so we displace the costliest, preserving energy safety).
            replaceable_tiers = sorted(
                [
                    t
                    for t in range(num_tiers)
                    if tier_assignment[t] not in overdue_sorted
                ],
                key=lambda t: -corrected_costs[tier_assignment[t]],
            )
            for g_overdue, t_replace in zip(overdue_sorted, replaceable_tiers):
                tier_assignment[t_replace] = g_overdue

    return tier_assignment


def _sequential_group_selection(
    pnu_round: int,
    num_tiers: int,
    num_groups: int,
    group_costs: list[float],
    window_step: int = 0,
) -> dict[int, int]:
    """
    Sliding-window sequential group selection (FedPartBESeq mode).

    At PNU round r, select M consecutive groups starting at position:
        window_start = (r * step) mod G
    where step defaults to M (non-overlapping windows) or 1 (overlapping).

    The M selected groups are then sorted by corrected cost and assigned to
    tiers: tier-0 (lowest battery) gets the cheapest group in the window,
    tier-M-1 (highest battery) gets the most expensive.

    Step choices
    ────────────
    step = 0 (auto) → defaults to num_tiers (non-overlapping, cleanest)
    step = 1        → overlapping windows (each group stays active for M rounds)
    step = M        → non-overlapping windows (each group visited once per G/M rounds)
    step = k        → any integer 1..G gives intermediate overlap

    Staleness properties
    ────────────────────
    step=1:  group g is absent for (G-M) rounds between visits → staleness ≤ G-M
    step=M:  group g is absent for (G/M - 1) windows → staleness ≤ (G-M)/M ≈ G/M
    Both are bounded under Assumption A.4 as long as step ≤ G.

    Returns
    ───────
    tier_assignment : dict[tier_idx → group_idx]
        tier 0 → cheapest group in window, tier M-1 → most expensive
    """
    effective_step = window_step if window_step > 0 else num_tiers

    # Clamp to valid range [1, G]
    effective_step = max(1, min(effective_step, num_groups))

    window_start = (pnu_round * effective_step) % num_groups
    window_groups = [(window_start + i) % num_groups for i in range(num_tiers)]

    # Sort window by corrected cost: tier-0 → cheapest, tier-M-1 → most expensive
    corrected_costs = _compute_corrected_group_costs(group_costs)
    window_groups_sorted = sorted(window_groups, key=lambda g: corrected_costs[g])

    return {t: window_groups_sorted[t] for t in range(num_tiers)}


def _client_to_tier(
    client_batteries: list[tuple[int, float]],
    num_tiers: int,
) -> dict[int, int]:
    """
    Assign clients to tiers based on battery level (equal quantile split).
    Legacy function — superseded by _battery_proportional_client_assignment.
    """
    sorted_clients = sorted(client_batteries, key=lambda x: x[1])
    client_tier_map = {}
    clients_per_tier = max(1, len(sorted_clients) // num_tiers)
    for i, (client_id, _) in enumerate(sorted_clients):
        tier_idx = min(i // clients_per_tier, num_tiers - 1)
        client_tier_map[client_id] = tier_idx
    return client_tier_map


def _battery_proportional_client_assignment(
    client_batteries: list[tuple[int, float]],
    group_costs: list[float],
    staleness: list[int],
    grad_norms: list[float],
    num_tiers: int,
    bucket_shift: int = 0,
    enforce_staleness_cap: bool = True,
) -> tuple[dict[int, int], dict[int, int]]:
    """
    Battery-proportional group assignment for maximum fleet lifetime.

    Principle
    ---------
    Assign client k a group whose cost is proportional to k's battery relative
    to the fleet mean:

        cost_assigned[k] / mean_selected_cost ≈ battery[k] / mean_battery

    When every client trains proportionally to their remaining energy, all
    clients drain at the same *relative* rate → the strongest client (last
    survivor) dies as late as possible → fleet lifetime is maximised.

    This corrects the flaw in equal-quantile tier assignment: there, the
    highest-battery clients are always assigned the most expensive groups and
    drain faster than their fair share, dying earlier than under FedPart's
    round-robin.

    Algorithm
    ---------
    1. Select num_tiers representative groups (one per cost-bucket), driven
       by staleness — same selection as _assign_tiers_to_groups_v2.
    2. For each client compute ideal_cost = mean_selected_cost × (battery / mean_battery).
       Assign the tier whose selected group cost is closest to ideal_cost.
    3. Coverage pass: ensure every tier has ≥ 1 client by reassigning the
       client least harmed by the move.

    Returns
    -------
    client_assignment : dict[client_id → group_idx]
    tier_to_group     : dict[tier_idx  → group_idx]  (for logging/metrics)
    """
    num_groups = len(group_costs)
    K = len(client_batteries)

    if K == 0:
        return {}, {}

    # ── Step 1: select one group per cost-bucket (staleness-driven) ──────────
    head_group = num_groups - 1
    cost_sorted = sorted(range(num_groups), key=lambda g: group_costs[g])
    # Force head (fc / classifier) into most-expensive bucket so it is always
    # trained by the highest-battery clients (best data coverage, non-IID safe).
    if head_group in cost_sorted and cost_sorted.index(head_group) < num_groups - 1:
        cost_sorted.remove(head_group)
        cost_sorted.append(head_group)

    bucket_size = max(1, math.ceil(num_groups / num_tiers))
    buckets: list[list[int]] = [
        cost_sorted[t * bucket_size : min((t + 1) * bucket_size, num_groups)]
        for t in range(num_tiers)
    ]
    for t in range(num_tiers):
        if not buckets[t]:
            buckets[t] = [cost_sorted[t % num_groups]]

    tier_to_group: dict[int, int] = {}
    for t in range(num_tiers):
        source_bucket = buckets[(t + bucket_shift) % num_tiers]
        best_g = max(
            source_bucket,
            key=lambda g: (staleness[g], grad_norms[g] / max(group_costs[g], 1e-8)),
        )
        tier_to_group[t] = best_g

    # Hard staleness cap: enforce τ_max = ceil(G/M) - 1 deterministically.
    # Mirror of the logic in _assign_tiers_to_groups_v2 — applied here so that
    # both assignment strategies honour the hard staleness guarantee.
    if enforce_staleness_cap:
        _corrected = _compute_corrected_group_costs(group_costs)
        _tau_max = max(0, math.ceil(num_groups / num_tiers) - 1)
        _covered = set(tier_to_group.values())
        _overdue = [
            g
            for g in range(num_groups)
            if staleness[g] >= _tau_max and g not in _covered
        ]
        if _overdue:
            _overdue_sorted = sorted(_overdue, key=lambda g: _corrected[g])
            _replaceable = sorted(
                [
                    t
                    for t in range(num_tiers)
                    if tier_to_group[t] not in _overdue_sorted
                ],
                key=lambda t: -_corrected[tier_to_group[t]],
            )
            for _go, _tr in zip(_overdue_sorted, _replaceable):
                tier_to_group[_tr] = _go

    # ── Step 2: proportional matching ────────────────────────────────────────
    mean_battery = sum(b for _, b in client_batteries) / K
    selected_costs = [group_costs[tier_to_group[t]] for t in range(num_tiers)]
    mean_sel_cost = sum(selected_costs) / num_tiers if num_tiers > 0 else 1.0

    bat_map: dict[int, float] = dict(client_batteries)
    client_assignment: dict[int, int] = {}
    for cid, bat in client_batteries:
        ideal = (
            mean_sel_cost * (bat / mean_battery) if mean_battery > 0 else mean_sel_cost
        )
        best_t = min(range(num_tiers), key=lambda t: abs(selected_costs[t] - ideal))
        client_assignment[cid] = tier_to_group[best_t]

    # ── Step 3: coverage — every tier must have ≥ 1 client ───────────────────
    covered = {
        t
        for t in range(num_tiers)
        if any(client_assignment[cid] == tier_to_group[t] for cid in client_assignment)
    }
    for t in range(num_tiers):
        if t in covered:
            continue
        # Reassign the client whose ideal cost is already closest to this tier
        best_cid = min(
            bat_map.keys(),
            key=lambda cid: abs(
                selected_costs[t]
                - (
                    mean_sel_cost * (bat_map[cid] / mean_battery)
                    if mean_battery > 0
                    else mean_sel_cost
                )
            ),
        )
        client_assignment[best_cid] = tier_to_group[t]
        covered.add(t)

    return client_assignment, tier_to_group


def _freeze_all_except_group(
    model: nn.Module,
    groups: list[list[str]],
    active_idx: int,
) -> None:
    """Backward-compatible wrapper around flop_cost.freeze_to_trainable."""
    if active_idx < 0:
        trainable = [n for n, _ in model.named_parameters()]
    else:
        trainable = list(groups[active_idx])
    _flopcost_freeze_to_trainable(model, trainable)


def _restore_all_grad(model: nn.Module) -> None:
    """Re-enable gradients for all parameters (post-training cleanup)."""
    _flopcost_restore_grad(model)


def _compute_reference_representation(
    model: nn.Module,
    global_weights: dict,
    active_group_idx: int,
    groups: list[list[str]],
    dataloader: torch.utils.data.DataLoader,
    device: str,
) -> torch.Tensor:
    """
    Compute reference output of the active group using global weights.

    Returns:
        h_ref: tensor of shape (batch_size, ...) containing the output of
               the active group with global weights on one reference batch.
    """
    # Create a copy of the model with global weights
    ref_model = copy.deepcopy(model)
    ref_model.load_state_dict(global_weights)
    ref_model.eval()
    ref_model.to(device)

    # Get one batch
    try:
        x_ref, _ = next(iter(dataloader))
    except StopIteration:
        del ref_model
        gc.collect()
        return None

    x_ref = x_ref.to(device)

    # Forward through the model up to and including the active group
    # For simplicity, we'll do a full forward pass and return intermediate
    # representation. A more sophisticated approach would use hooks.
    # Here we approximate by running the full model and capturing activations.

    # Since we don't have layer-wise access easily, we'll use a simpler
    # approach: run full forward and use the logits as proxy for now.
    # In production, you'd use register_forward_hook on the relevant modules.

    with torch.no_grad():
        h_ref = ref_model(x_ref)

    h_ref_cpu = h_ref.cpu()
    # Free the reference model and intermediate tensors immediately.
    del h_ref, x_ref, ref_model
    gc.collect()
    return h_ref_cpu


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm
# ─────────────────────────────────────────────────────────────────────────────


@register_algorithm("fedpart_be")
class FedPartBE(FLAlgorithm):
    """
    FedPartBE: Battery-Energy-aware Federated Partial Network Updates.

    Extension of FedPart with:
      - Energy-tier layer assignment (cheap groups to low-battery clients)
      - Representation-space proximal regularization
      - Staleness-aware layer priority
      - Sequential server aggregation (Gauss-Seidel order)

    Registration key: "fedpart_be"
    """

    name = "fedpart_be"
    description = (
        "FedPartBE: Battery-energy-aware partial network updates with "
        "energy-tier assignments and representation proximal regularization."
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
        Local training step with energy-aware group assignment.

        1. Derive layer groups if not yet cached.
        2. Get active group assignment from config (set by server).
        3. Freeze all layers except the active group.
        4. Compute reference representation with global weights.
        5. Train with CE + weight proximal + representation proximal loss.
        6. Return partial delta and metadata.
        """
        device = config.get("device", "cpu")
        lr = config.get("lr", 0.01)
        momentum = config.get("momentum", 0.9)
        weight_decay = config.get("weight_decay", 1e-4)
        local_epochs = config.get("local_epochs", 8)
        max_grad_norm = config.get("max_grad_norm", 10.0)
        mu_weight = config.get("mu_weight", 0.01)  # weight proximal coeff
        mu_repr = config.get("mu_repr", 0.1)  # representation proximal coeff
        warmup_rounds = config.get("warmup_rounds", 5)
        persist_optimizer = config.get("persist_optimizer", True)

        model.train()
        model.to(device)

        # ── Lazy derivation of layer groups ──────────────────────────────────
        if "layer_groups" not in state.custom:
            groups = _derive_layer_groups(model)
            state.custom["layer_groups"] = groups
            if config.get("verbose_groups", True) and state.client_id == 0:
                lines = []
                for i, g in enumerate(groups):
                    key = _param_group_key(g[0])
                    lines.append(f"    [{i:2d}] {key:<30s} ({len(g)} params)")
                print(
                    f"  [FedPartBE] {len(groups)} layer groups derived "
                    f"(shared across all clients)"
                )
                print("\n".join(lines))

        groups: list[list[str]] = state.custom["layer_groups"]
        num_groups = len(groups)

        # ── Determine active group ────────────────────────────────────────────
        # During warmup: active_idx = -1 (full FedAvg update).
        # During inter-cycle rounds: also active_idx = -1 (full training).
        # After warmup (PNU phase): server maps each client to its energy tier's group.
        # Energy gate below ensures that a client skips if battery < group cost.
        server_state = getattr(self, "_server_state", {})
        is_true_warmup = state.round_num < warmup_rounds
        is_inter_cycle = (not is_true_warmup) and server_state.get(
            "inter_cycle_round", False
        )
        is_warmup = is_true_warmup or is_inter_cycle

        if is_warmup:
            active_idx = -1
        else:
            # Per-client assignment: server maps each client to its energy tier's group
            client_assignment = server_state.get("client_group_assignment", {})
            active_idx = client_assignment.get(
                state.client_id,
                0,  # fallback: cheapest group if first PNU round has no assignment yet
            )
            active_idx = max(0, min(active_idx, num_groups - 1))

        # ── Energy gate — WARMUP phase ───────────────────────────────────────
        # Safety check: if battery < full-model training cost, skip this warmup
        # round without draining. Don't increment state.round_num so the client
        # retries warmup next time it is called.
        # Protects very-low-SOC clients (e.g. <5%) from burning their entire
        # budget during warmup before any PNU energy-tier protection activates.
        if is_warmup:
            _profile_wu = config.get("device_profile")
            if _profile_wu is not None:
                # Warmup = full training: trainable_names is the entire model.
                # The dispatcher returns the cost under the active cost_model.
                _trainable_wu = [n for n, _ in model.named_parameters()]
                _ff_wu = round_compute_flops(
                    model,
                    _trainable_wu,
                    config,
                    _profile_wu,
                    dataloader,
                    local_epochs,
                    groups=None,
                    active_group_idx=-1,
                )
                _fb_wu = (
                    sum(v.numel() for v in model.state_dict().values()) * 4
                )  # bytes for full model
                _wu_energy = _profile_wu.round_energy_breakdown(
                    _ff_wu,
                    _fb_wu,
                    _fb_wu,
                    config.get("energy_scale_factor", 1.0),
                    config.get("alpha_applies_to", "compute"),
                )[
                    "total"
                ]  # Energy for full model training (alpha on compute only)
            else:
                _wu_energy = 2.5 * config.get("energy_scale_factor", 1.0)

            if state.battery_j < _wu_energy:
                # Can't afford a full training round.
                # For true warmup rounds: force transition to PNU mode so the
                # client is not stuck in warmup forever (critical for wide-SOC
                # fleets where very-low-battery clients exhaust warmup budget).
                # For inter-cycle rounds: no state.round_num manipulation needed
                # (state.round_num >= warmup_rounds already; the server signals
                # inter-cycle via _server_state["inter_cycle_round"]).
                if is_true_warmup:
                    state.round_num = warmup_rounds
                return {}, {
                    "client_id": state.client_id,
                    "round_num": state.round_num,
                    "skipped": True,
                    "battery_j_remaining": state.battery_j,
                    "energy_j_consumed": 0.0,
                    "bytes_sent": 0,
                    "bytes_received": 0,
                    "local_loss": 0.0,
                    "compression_ratio": 0.0,
                    "beta_actual": 0.0,
                    "active_group_idx": -1,
                    "is_warmup": True,
                    "num_layer_groups": num_groups,
                    "dataset_size": len(dataloader.dataset),
                }

        # ── Energy gate (PNU phase only) ─────────────────────────────────────
        # Estimate cost of training the current group.  If battery < estimate,
        # skip this round without consuming energy — the client will participate
        # again when the rotation returns to a cheaper group.
        # This is the core FedPartBE innovation vs FedPart: selective participation
        # based on group cost, extending device lifetime.
        if not is_warmup:
            # Cache group FLOPs (architecture is fixed)
            if "group_flops" not in state.custom:
                input_shape = config.get("input_shape", (1, 3, 32, 32))
                state.custom["group_flops"] = _compute_group_flops(
                    groups, model, input_shape
                )
            gf = state.custom["group_flops"]
            total_gf = sum(gf) or 1.0

            # Energy estimate goes through the same flop_cost dispatcher as the
            # real accounting below — so the gate and the drain cannot drift
            # apart under any cost_model.
            profile = config.get("device_profile")
            if profile is not None:
                _trainable_est = list(groups[active_idx])
                eff_flops_est = round_compute_flops(
                    model,
                    _trainable_est,
                    config,
                    profile,
                    dataloader,
                    local_epochs,
                    groups=groups,
                    active_group_idx=active_idx,
                    group_flops_analytic=gf,
                )
                uplink_est = int(
                    sum(
                        model.state_dict()[k].numel()
                        for k in groups[active_idx]
                        if k in model.state_dict()
                    )
                    * 4
                )
                downlink_est = sum(v.numel() for v in model.state_dict().values()) * 4
                energy_est = profile.round_energy_breakdown(
                    eff_flops_est,
                    uplink_est,
                    downlink_est,
                    config.get("energy_scale_factor", 1.0),
                    config.get("alpha_applies_to", "compute"),
                )["total"]
            else:
                # Profile-less fallback — heuristic on raw phi (gate purpose).
                energy_est = (0.5 + 2.0 * (gf[active_idx] / total_gf)) * config.get(
                    "energy_scale_factor", 1.0
                )

            if state.battery_j < energy_est:
                # Energy gate override: group at staleness cap cannot be skipped.
                # If the assigned group has reached τ_max = ceil(G/M) - 1, the
                # hard staleness guarantee requires it to be trained this round.
                # Force the client to train the original group regardless of energy
                # cost; log a warning so the operator can detect chronic over-drain.
                _enforce_cap = config.get("enforce_staleness_cap", True)
                _srv_staleness = server_state.get("staleness", [])
                _n_tiers_srv = server_state.get("num_tiers", max(1, num_groups // 2))
                _tau_max_gate = max(0, math.ceil(num_groups / max(_n_tiers_srv, 1)) - 1)
                _original_group_at_cap = (
                    _enforce_cap
                    and len(_srv_staleness) > active_idx
                    and _srv_staleness[active_idx] >= _tau_max_gate
                )
                if _original_group_at_cap:
                    # Do NOT fire the gate for a group at the staleness cap.
                    # The client must train it even if under-battery; the energy
                    # accounting will drain as much as remains (clamped to 0).
                    import warnings

                    warnings.warn(
                        f"[FedPartBE] Client {state.client_id}: energy gate suppressed "
                        f"for group {active_idx} at staleness cap "
                        f"(staleness={_srv_staleness[active_idx]}, τ_max={_tau_max_gate}). "
                        f"Battery {state.battery_j:.1f}J < estimated {energy_est:.1f}J.",
                        stacklevel=2,
                    )
                    # Fall through to training — active_idx unchanged.
                else:
                    # Assigned group too expensive. Find the cheapest group under
                    # the SAME cost_model used for the actual drain — the gate
                    # must rank groups consistently with the accounting site.
                    min_group_idx = int(min(range(num_groups), key=lambda g: gf[g]))
                    if profile is not None:
                        _trainable_min = list(groups[min_group_idx])
                        min_eff_flops = round_compute_flops(
                            model,
                            _trainable_min,
                            config,
                            profile,
                            dataloader,
                            local_epochs,
                            groups=groups,
                            active_group_idx=min_group_idx,
                            group_flops_analytic=gf,
                        )
                        min_uplink_est = int(
                            sum(
                                model.state_dict()[k].numel()
                                for k in groups[min_group_idx]
                                if k in model.state_dict()
                            )
                            * 4
                        )
                        min_energy = profile.round_energy_breakdown(
                            min_eff_flops,
                            min_uplink_est,
                            downlink_est,
                            config.get("energy_scale_factor", 1.0),
                            config.get("alpha_applies_to", "compute"),
                        )["total"]
                    else:
                        min_energy = (
                            0.5 + 2.0 * (gf[min_group_idx] / total_gf)
                        ) * config.get("energy_scale_factor", 1.0)

                    if state.battery_j < min_energy:
                        # Below the floor for any group — permanently dead.
                        state.battery_j = 0.0
                        return {}, {
                            "client_id": state.client_id,
                            "round_num": state.round_num,
                            "skipped": True,
                            "battery_j_remaining": state.battery_j,
                            "energy_j_consumed": 0.0,
                            "bytes_sent": 0,
                            "bytes_received": 0,
                            "local_loss": 0.0,
                            "compression_ratio": 0.0,
                            "beta_actual": 0.0,
                            "active_group_idx": active_idx,
                            "is_warmup": False,
                            "num_layer_groups": num_groups,
                            "dataset_size": len(dataloader.dataset),
                        }
                    else:
                        # Can afford cheapest group — fall back to it instead of
                        # skipping. Prevents zombie state: a client that can't afford
                        # its assigned (expensive) group but has enough battery for a
                        # cheaper one should keep contributing, not become permanently
                        # stuck skipping while technically "alive".
                        active_idx = min_group_idx

        # ── Save weights before local training ───────────────────────────────
        w_before = OrderedDict(
            {k: v.clone().cpu() for k, v in model.state_dict().items()}
        )

        # Global weights for proximal terms — only needed post-warmup when mu > 0.
        # Skipped during warmup: avoids cloning the full model 30× per warmup round.
        w_global: dict = {}
        if not is_warmup and (mu_weight > 0 or mu_repr > 0):
            w_global = {k: v.clone().to(device) for k, v in model.state_dict().items()}

        # ── Freeze layers outside active group ───────────────────────────────
        _freeze_all_except_group(model, groups, active_idx)

        # ── repr proximal flag ───────────────────────────────────────────────
        # repr proximal computed via weight-swap (no deepcopy, correct x alignment).
        _do_repr = not is_warmup and mu_repr > 0 and w_global

        # ── Live references to active group parameters (weight proximal) ────────
        # named_parameters() returns (name, param) pairs where param is a live
        # tensor reference — no copy. Avoids calling model.state_dict() (which
        # copies all tensors) inside the per-batch training loop.
        active_param_refs: dict[str, torch.nn.Parameter] = {}
        if not is_warmup and (mu_weight > 0 or mu_repr > 0):
            active_names_set = set(groups[active_idx])
            active_param_refs = {
                name: param
                for name, param in model.named_parameters()
                if name in active_names_set
            }

        # Pre-allocate buffers to avoid memory allocation overhead during batch loop
        saved_active_buffers: dict[str, torch.Tensor] = {}
        if _do_repr:
            saved_active_buffers = {
                k: torch.empty_like(p.data) for k, p in active_param_refs.items()
            }

        # ── Local training ───────────────────────────────────────────────────
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer_type = config.get("optimizer", "sgd").lower()
        if optimizer_type == "adam":
            optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(
                trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        # Restore per-group optimizer state from previous visit (works for both
        # SGD momentum buffer and Adam exp_avg/exp_avg_sq tensors — state_dict
        # format is consistent within a run since optimizer_type is fixed).
        if persist_optimizer and not is_warmup:
            saved = state.custom.get("optimizer_states", {}).get(active_idx)
            if saved is not None:
                optimizer.load_state_dict(saved)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_weight_prox_loss = 0.0
        total_repr_prox_loss = 0.0
        num_batches = 0

        for epoch in range(local_epochs):
            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                # Forward pass
                out = model(x)
                ce_loss = criterion(out, y)

                # Weight proximal term: ||W_active - W_global||^2
                # Uses live parameter references — no state_dict() copy per batch.
                weight_prox_loss = 0.0
                if active_param_refs and mu_weight > 0:
                    for k, param in active_param_refs.items():
                        if k in w_global:
                            weight_prox_loss += torch.sum((param - w_global[k]) ** 2)
                    weight_prox_loss *= mu_weight / 2.0

                # Representation proximal term: MSE(f(x;W_local), f(x;W_global))
                # Weight-swap: temporarily restores active group to global weights,
                # runs forward on the SAME x (correct alignment), then restores.
                # Non-active params are frozen = already at global values, so only
                # the active group needs swapping. Eliminates deepcopy entirely.
                # Limited to final epoch: signal is strongest at maximum drift (epoch E-1).
                # At epoch 0, W_local == W_global exactly → repr_prox_loss ≈ 0 (useless).
                # Drift accumulates as δ(e) ∝ e²; epoch E-1 captures ~35% of total signal
                # at the same 6.25% overhead as the old epoch=0 approach.
                repr_prox_loss = 0.0
                # or epoch == 1
                if _do_repr and (epoch == local_epochs - 1):
                    with torch.no_grad():
                        for k, p in active_param_refs.items():
                            saved_active_buffers[k].copy_(p.data)
                            p.data.copy_(w_global[k])
                        h_ref_x = model(x).detach()
                        for k, p in active_param_refs.items():
                            p.data.copy_(saved_active_buffers[k])
                    repr_prox_loss = mu_repr * torch.mean((out - h_ref_x) ** 2)

                # Total loss
                loss = ce_loss + weight_prox_loss + repr_prox_loss

                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, max_norm=max_grad_norm
                    )
                optimizer.step()

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_weight_prox_loss += (
                    weight_prox_loss
                    if isinstance(weight_prox_loss, float)
                    else weight_prox_loss.item()
                )
                total_repr_prox_loss += (
                    repr_prox_loss
                    if isinstance(repr_prox_loss, float)
                    else repr_prox_loss.item()
                )
                num_batches += 1

        # Save per-group optimizer state for next visit to this group.
        if persist_optimizer and not is_warmup:
            state.custom.setdefault("optimizer_states", {})[
                active_idx
            ] = optimizer.state_dict()

        # ── Restore full gradient capability ─────────────────────────────────
        _restore_all_grad(model)

        # Free global weights (no ref_outputs to free — weight-swap leaves no residual).
        if w_global:
            del w_global
        gc.collect()

        # ── Compute delta — ONLY for active group's parameters ──────────────
        # BN running stats (running_mean, running_var, num_batches_tracked) are
        # buffers, not parameters → excluded from _derive_layer_groups.  They
        # update on every forward pass regardless of requires_grad.  Must always
        # be transmitted to prevent stale-BN divergence in the global model.
        current_sd = model.state_dict()
        bn_buffer_keys = [
            k
            for k in w_before
            if k.endswith(("running_mean", "running_var", "num_batches_tracked"))
        ]
        if is_warmup:
            # Full delta
            partial_delta = OrderedDict(
                {k: (w_before[k] - current_sd[k].cpu()).float() for k in w_before}
            )
        else:
            # Partial delta — active group's learnable params + ALL BN running stats
            active_keys = groups[active_idx]
            active_keys_set = set(active_keys)
            all_tx_keys = list(active_keys) + [
                k for k in bn_buffer_keys if k not in active_keys_set
            ]
            partial_delta = OrderedDict(
                {k: (w_before[k] - current_sd[k].cpu()).float() for k in all_tx_keys}
            )

        # ── Communication / energy accounting ────────────────────────────────
        uplink_bytes = self.count_bytes(partial_delta, sparse=False)
        # Count bytes from w_before before freeing it (compute once, reuse)
        full_model_bytes = self.count_bytes(w_before, sparse=False)
        downlink_bytes = full_model_bytes
        full_upload_bytes_ref = full_model_bytes
        del w_before
        del current_sd
        gc.collect()

        # Compute group FLOPs for metadata (use cached value if available)
        group_cost = 0
        if not is_warmup:
            gf = state.custom.get("group_flops")
            if gf is None:
                input_shape = config.get("input_shape", (1, 3, 32, 32))
                gf = _compute_group_flops(groups, model, input_shape)
                state.custom["group_flops"] = gf
            group_cost = gf[active_idx]

        profile = config.get("device_profile")
        # Trainable set declaration: warmup = full model, PNU = active group only.
        if is_warmup:
            _trainable_acc = [n for n, _ in model.named_parameters()]
            _gf_hint_acc = None
            _active_idx_hint = -1
        else:
            _trainable_acc = list(groups[active_idx])
            _gf_hint_acc = state.custom["group_flops"]
            _active_idx_hint = active_idx

        if profile:
            # Single dispatcher call — same cost_model as the gates above so
            # estimate and actual drain cannot diverge. No inline FLOP math.
            effective_flops = round_compute_flops(
                model,
                _trainable_acc,
                config,
                profile,
                dataloader,
                local_epochs,
                groups=groups,
                active_group_idx=_active_idx_hint,
                group_flops_analytic=_gf_hint_acc,
            )
            _bd = profile.round_energy_breakdown(
                effective_flops,
                uplink_bytes,
                downlink_bytes,
                config.get("energy_scale_factor", 1.0),
                config.get("alpha_applies_to", "compute"),
            )
        else:
            # Profile-less fallback retained for tests / smoke runs.
            if is_warmup:
                _active_fraction_hf = 1.0
            else:
                if "group_flops" not in state.custom:
                    input_shape = config.get("input_shape", (1, 3, 32, 32))
                    state.custom["group_flops"] = _compute_group_flops(
                        groups, model, input_shape
                    )
                _gf_acc = state.custom["group_flops"]
                _active_fraction_hf = _gf_acc[active_idx] / max(sum(_gf_acc), 1.0)
            _e = (0.5 + 2.0 * _active_fraction_hf) * config.get(
                "energy_scale_factor", 1.0
            )
            _bd = {"compute": _e, "uplink": 0.0, "downlink": 0.0, "total": _e}

        # energy_scale_factor (alpha) applied inside round_energy_breakdown —
        # compute term only by default (alpha = compute utilization gap).
        energy_j = _bd["total"]

        # ── Battery update ───────────────────────────────────────────────────
        state.battery_j = max(0.0, state.battery_j - energy_j)
        state.round_num += 1

        # ── Compression ratio ────────────────────────────────────────────────
        # full_upload_bytes_ref was computed before w_before was freed above.
        compression_ratio = uplink_bytes / max(full_upload_bytes_ref, 1)

        # ── Metadata ─────────────────────────────────────────────────────────
        metadata = {
            # Mandatory fields
            "client_id": state.client_id,
            "round_num": state.round_num,
            "beta_actual": 1.0,
            "battery_j_remaining": state.battery_j,
            "energy_j_consumed": energy_j,
            "energy_compute_j": _bd["compute"],
            "energy_uplink_j": _bd["uplink"],
            "energy_downlink_j": _bd["downlink"],
            "bytes_sent": uplink_bytes,
            "bytes_received": downlink_bytes,
            "local_loss": total_ce_loss / max(num_batches, 1),
            "compression_ratio": compression_ratio,
            # FedPartBE-specific
            "active_group_idx": active_idx,
            "tier_idx": config.get("fedpartbe_tier", -1),
            "battery_j": state.battery_j,
            "group_cost": group_cost,
            "delta_bytes": uplink_bytes,
            "weight_prox_loss": total_weight_prox_loss / max(num_batches, 1),
            "repr_prox_loss": total_repr_prox_loss / max(num_batches, 1),
            "is_warmup": is_warmup,
            "num_layer_groups": num_groups,
            "dataset_size": len(dataloader.dataset),
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
        Energy-tier-aware aggregation with sequential (Gauss-Seidel) ordering.

        During warmup: full FedAvg aggregation.
        After warmup:
          1. Compute tier assignments for next round
          2. Aggregate deltas sequentially by group (early layers first)
          3. Update staleness vector
          4. Return new weights and metrics
        """
        K = len(client_updates)
        global_sd = global_model.state_dict()
        warmup_rounds = config.get("warmup_rounds", 5)
        is_true_warmup = round_num < warmup_rounds

        # ── Inter-cycle detection ─────────────────────────────────────────────
        # After each complete PNU cycle (num_groups × rounds_per_layer rounds),
        # insert inter_cycle_rounds full-network training rounds before the next
        # cycle. inter_cycle_rounds=0 (default) disables this feature.
        inter_cycle_rounds = config.get("inter_cycle_rounds", 0)
        is_inter_cycle = False
        if inter_cycle_rounds > 0 and not is_true_warmup:
            _rpl = config.get("rounds_per_layer", 2)
            # num_groups is derived below from client metadata — use a temporary
            # estimate here; will be recomputed correctly after state init.
            _first_meta_tmp = client_updates[0][1]
            _ng_tmp = _first_meta_tmp.get("num_layer_groups", 10)
            _cycle_len = _ng_tmp * _rpl
            _full_cycle = _cycle_len + inter_cycle_rounds
            _pnu_round = round_num - warmup_rounds
            _pos = _pnu_round % _full_cycle
            is_inter_cycle = _pos >= _cycle_len

        # Signal clients about inter-cycle state (read in client_update)
        if not hasattr(self, "_server_state"):
            self._server_state = {}
        self._server_state["inter_cycle_round"] = is_inter_cycle

        is_warmup = is_true_warmup or is_inter_cycle

        # ── Initialize server state if needed ────────────────────────────────
        # (already initialised above for inter-cycle flag — this is a no-op on
        # the second and subsequent calls)
        if not hasattr(self, "_server_state"):
            self._server_state = {}  # pragma: no cover

        # Derive layer groups from first client's metadata
        first_meta = client_updates[0][1]
        num_groups = first_meta.get("num_layer_groups", 10)

        if "staleness" not in self._server_state:
            self._server_state["staleness"] = [0] * num_groups

        staleness = self._server_state["staleness"]

        # Ensure staleness vector has correct length
        if len(staleness) != num_groups:
            staleness = [0] * num_groups
            self._server_state["staleness"] = staleness

        # ── Derive layer groups + cache costs (shared by warmup and PNU) ─────
        if "layer_groups" not in self._server_state:
            self._server_state["layer_groups"] = _derive_layer_groups(global_model)
        groups = self._server_state["layer_groups"]
        # Reconcile num_groups with actual derived count
        num_groups = len(groups)
        while len(staleness) < num_groups:
            staleness.append(0)
        self._server_state["staleness"] = staleness

        if "group_flops" not in self._server_state:
            input_shape = config.get("input_shape", (1, 3, 32, 32))
            self._server_state["group_flops"] = _compute_group_flops(
                groups, global_model, input_shape
            )

        # key→group index map (cached after first call)
        if "key_to_group" not in self._server_state:
            key_to_group_map: dict[str, int] = {}
            for g_idx, group_keys in enumerate(groups):
                for k in group_keys:
                    key_to_group_map[k] = g_idx
            self._server_state["key_to_group"] = key_to_group_map

        if "ema_grad_norms" not in self._server_state:
            self._server_state["ema_grad_norms"] = [1.0] * num_groups

        # ── Warmup phase: full FedAvg ─────────────────────────────────────────
        if is_warmup:
            # Uniform average (1/K per client) — identical to FedPart warmup so
            # that warmup rounds are directly comparable across algorithms.
            K_wu = len(client_updates)

            agg = None
            for update, metadata, state in client_updates:
                if agg is None:
                    agg = {k: v.clone().float() for k, v in update.items()}
                else:
                    for k in update:
                        if k in agg:
                            agg[k] += update[k].float()
                        else:
                            agg[k] = update[k].float()
            for k in agg:
                agg[k] /= K_wu

            # Apply averaged aggregate
            new_weights = OrderedDict()
            for k in global_sd:
                if k in agg:
                    new_weights[k] = global_sd[k].float() - agg[k].to(
                        global_sd[k].device
                    )
                else:
                    new_weights[k] = global_sd[k].float()

        else:
            # ── Post-warmup: multi-tier Gauss-Seidel aggregation ─────────────
            #
            # True FedPartBE: num_tiers groups trained simultaneously each round.
            #   Tier 0 (lowest battery) → cheapest cost-bucket → cheap group
            #   Tier K-1 (highest battery) → most expensive bucket → head group
            #
            # Update frequency vs FedPart (10 groups, 5 tiers):
            #   FedPart:   each group updated every ~20 rounds (all 30 clients)
            #   FedPartBE: each group updated every ~2 rounds (~6 clients/group)
            #   → same total gradient budget, but 10× more frequent → lower staleness
            #
            # Gauss-Seidel: apply group deltas in ascending architectural order
            # (shallow → deep) so later groups see already-updated earlier layers.

            ema_grad_norms = self._server_state["ema_grad_norms"]
            ema_alpha = config.get("ema_alpha", 0.3)

            # server_lr: scales each group's applied delta to prevent multi-group
            # displacement divergence.  The correct denominator is M_t = the number
            # of groups ACTUALLY updated this round, not the configured num_tiers.
            # When clients die, some tiers may have no clients → fewer groups updated
            # → denominator shrinks → server_lr increases automatically.
            # auto (None in config): 1/M_t.  Override with float to fix (e.g. 1.0).
            _server_lr_override = config.get("server_lr", None)
            _active_groups_this_round = len(
                {
                    meta.get("active_group_idx")
                    for _, meta, _ in client_updates
                    if not meta.get("skipped", False)
                    and 0 <= meta.get("active_group_idx", -1) < num_groups
                }
            )
            server_lr = (
                float(_server_lr_override)
                if _server_lr_override is not None
                else 1.0 / max(_active_groups_this_round, 1)
            )

            # ── GPU-parallel aggregation ──────────────────────────────────────
            # Groups are disjoint parameter partitions → all group deltas can be
            # accumulated in a single pass and applied in one fused GPU operation.
            # This replaces the sequential for-g loop (which was Jacobi anyway —
            # all deltas were computed on the same w^{t-1}).
            key_to_group_map = self._server_state["key_to_group"]

            # Step 1: single-pass weighted accumulation over all client updates
            group_sizes: dict[int, float] = defaultdict(float)
            group_weighted_sums: dict[str, torch.Tensor] = {}
            active_groups: list[int] = []

            new_weights = OrderedDict({k: v.float() for k, v in global_sd.items()})

            for upd, meta, st in client_updates:
                if meta.get("skipped", False) or not upd:
                    continue
                g = meta.get("active_group_idx", -1)
                if not (0 <= g < num_groups):
                    continue
                n_k = float(meta.get("dataset_size", 1))
                group_sizes[g] += n_k
                for k, v in upd.items():
                    if key_to_group_map.get(k) != g:
                        continue  # BN stats handled separately below
                    vf = v.float()
                    if k not in group_weighted_sums:
                        group_weighted_sums[k] = vf.mul(n_k)
                    else:
                        group_weighted_sums[k].add_(vf, alpha=n_k)

            # Step 2: normalize per-group, update staleness + EMA
            for g in range(num_groups):
                if g not in group_sizes:
                    staleness[g] += 1
                    continue
                total = max(group_sizes[g], 1.0)
                norms = []
                for k in groups[g]:
                    if k in group_weighted_sums:
                        group_weighted_sums[k].div_(total)  # in-place normalize
                        norms.append(group_weighted_sums[k].norm().item())
                if norms:
                    avg_norm = sum(norms) / len(norms)
                    ema_grad_norms[g] = (1.0 - ema_alpha) * ema_grad_norms[
                        g
                    ] + ema_alpha * avg_norm
                staleness[g] = 0
                active_groups.append(g)

            # Step 3: fused GPU update — single kernel launch for all groups
            # torch._foreach_add_ processes a list of tensors in one CUDA/MPS op.
            keys_to_update = [k for k in group_weighted_sums if k in new_weights]
            if keys_to_update:
                _device = new_weights[keys_to_update[0]].device
                deltas_on_device = [
                    group_weighted_sums[k].to(_device) for k in keys_to_update
                ]
                current_vals = [new_weights[k] for k in keys_to_update]
                torch._foreach_add_(current_vals, deltas_on_device, alpha=-server_lr)
                for k, v in zip(keys_to_update, current_vals):
                    new_weights[k] = v
                del deltas_on_device

            del group_weighted_sums

            # ── BN running stats: global average across ALL active clients ────
            # All clients do a full forward pass -> all BN layers update regardless
            # of which group is frozen.  Aggregate globally to prevent stale-BN.
            all_active = [
                (u, m, s)
                for u, m, s in client_updates
                if not m.get("skipped", False) and u
            ]
            if all_active:
                bn_keys = [
                    k
                    for k in global_sd
                    if k.endswith(
                        ("running_mean", "running_var", "num_batches_tracked")
                    )
                ]
                if bn_keys:
                    sizes_bn = [m.get("dataset_size", 1) for _, m, _ in all_active]
                    total_bn = max(sum(sizes_bn), 1)
                    weights_bn = [s / total_bn for s in sizes_bn]
                    bn_agg: dict = {}
                    for (update, _, _), w_k in zip(all_active, weights_bn):
                        for k in bn_keys:
                            if k not in update:
                                continue
                            if k not in bn_agg:
                                bn_agg[k] = update[k].float() * w_k
                            else:
                                bn_agg[k] += update[k].float() * w_k
                    for k, delta in bn_agg.items():
                        if k in new_weights:
                            new_weights[k] = new_weights[k] - server_lr * delta.to(
                                new_weights[k].device
                            )
                    del bn_agg

            # ── Verbose round summary ─────────────────────────────────────────
            if config.get("verbose_groups", False):
                tier_map = self._server_state.get("tier_to_group", {})

                def _gname(g_idx: int) -> str:
                    k = groups[g_idx][0]
                    return k.replace(".weight", "").replace(".bias", "")[:14]

                # Count clients per active group from metadata
                _clients_per_group: dict[int, int] = defaultdict(int)
                for _, _m, _ in client_updates:
                    if not _m.get("skipped", False):
                        _cpg_idx = _m.get("active_group_idx", -1)
                        if 0 <= _cpg_idx < num_groups:
                            _clients_per_group[_cpg_idx] += 1

                parts = []
                for t in sorted(tier_map):
                    gg = tier_map[t]
                    n = _clients_per_group.get(gg, 0)
                    parts.append(f"T{t}→G{gg}[{_gname(gg)}]({n}c)")
                stale_str = ",".join(str(s) for s in staleness)
                print(
                    f"  [FedPartBE] R{round_num} | {' | '.join(parts)} | "
                    f"stale=[{stale_str}]"
                )

            gc.collect()

        # ── Compute next round's client→group assignment ─────────────────────
        # Done after BOTH warmup and PNU aggregation so the first PNU round
        # already has a valid per-client assignment when client_update is called.
        alive_clients_for_assign = [
            (st.client_id, st.battery_j)
            for _, _, st in client_updates
            if st.battery_j > 0
        ]
        if alive_clients_for_assign:
            _n_tiers_cfg = config.get("num_tiers", None)
            _n_tiers = max(
                1,
                min(
                    (
                        int(_n_tiers_cfg)
                        if _n_tiers_cfg is not None
                        else max(1, num_groups // 2)
                    ),
                    len(alive_clients_for_assign),
                    num_groups,
                ),
            )
            _strategy = config.get("assignment_strategy", "quantile")

            # ── Group selection mode ──────────────────────────────────────────
            # "staleness_aware" (default): staleness-priority + cyclic rotation
            # "sequential": sliding window — deterministic, no staleness tracking
            _group_selection = config.get("group_selection", "staleness_aware")

            # Cyclic bucket rotation (staleness_aware mode only).
            # rotation_period=0 disables rotation (backward-compatible default).
            #
            # Adaptive rotation (adaptive_rotation: true in config):
            #   T_rot_effective(r) = rotation_period × ceil(K_0 / K_alive(r))
            #
            # Motivation: as clients die, each round contributes less gradient
            # budget per group. Stretching the rotation period proportionally
            # keeps the cumulative gradient budget per phase constant:
            #   budget_phase = (K_alive/M) × E × B × T_rot_effective
            #                ≈ (K_0/M) × E × B × rotation_period  = const
            #
            # This prevents rotation transitions from firing during periods of
            # accelerated client death, eliminating the "compound event" drops
            # (transition + fleet shrinkage simultaneously).
            _rotation_period = config.get("rotation_period", 0)
            _adaptive_rotation = config.get("adaptive_rotation", False)

            if _rotation_period > 0 and _group_selection == "staleness_aware":
                if _adaptive_rotation:
                    # Initialise K_0 on the first PNU round (once only).
                    if "adaptive_rot_K0" not in self._server_state:
                        self._server_state["adaptive_rot_K0"] = max(
                            len(alive_clients_for_assign), 1
                        )
                    _K0 = self._server_state["adaptive_rot_K0"]
                    _K_alive = max(len(alive_clients_for_assign), 1)
                    import math

                    _rot_effective = _rotation_period * math.ceil(_K0 / _K_alive)
                else:
                    _rot_effective = _rotation_period

                _phase = ((round_num - warmup_rounds) // _rot_effective) % _n_tiers
            else:
                _rot_effective = _rotation_period  # for logging only
                _phase = 0

            # Store for metrics logging (read later in the Metrics block).
            self._server_state["_rot_effective_last"] = _rot_effective

            if _group_selection == "sequential":
                # ── Sequential sliding window (FedPartBESeq mode) ────────────
                # M consecutive groups starting at (pnu_round × step) mod G.
                # Tier assignment within the window: cheapest group → tier-0.
                # No staleness tracking needed — coverage guaranteed by design.
                _pnu_round_seq = max(0, round_num - warmup_rounds)
                _window_step = config.get("window_step", 0)  # 0 = auto (= M)
                _tier_to_group = _sequential_group_selection(
                    _pnu_round_seq,
                    _n_tiers,
                    num_groups,
                    self._server_state["group_flops"],
                    window_step=_window_step,
                )
                # Client→tier: always equal-quantile for sequential mode
                _client_tier_map = _client_to_tier(alive_clients_for_assign, _n_tiers)
                _client_assignment = {
                    cid: _tier_to_group[tier] for cid, tier in _client_tier_map.items()
                }

            elif _strategy == "proportional":
                # Battery-proportional: all clients drain at the same relative rate.
                # Best for wide battery spreads (SOC [5%, 95%]).
                # Fails for narrow spreads (SOC [5%, 15%]) — clients cluster at
                # the middle tier causing gradient imbalance.
                _client_assignment, _tier_to_group = (
                    _battery_proportional_client_assignment(
                        alive_clients_for_assign,
                        self._server_state["group_flops"],
                        self._server_state["staleness"],
                        self._server_state["ema_grad_norms"],
                        _n_tiers,
                        bucket_shift=_phase,
                        enforce_staleness_cap=config.get("enforce_staleness_cap", True),
                    )
                )
            else:
                # Equal-quantile (default): K/num_tiers clients per tier regardless
                # of battery distribution shape. Robust for any spread.
                _client_tier_map = _client_to_tier(alive_clients_for_assign, _n_tiers)
                _tier_to_group = _assign_tiers_to_groups_v2(
                    _n_tiers,
                    num_groups,
                    self._server_state["group_flops"],
                    self._server_state["staleness"],
                    self._server_state["ema_grad_norms"],
                    bucket_shift=_phase,
                    enforce_staleness_cap=config.get("enforce_staleness_cap", True),
                )
                _client_assignment = {
                    cid: _tier_to_group[tier] for cid, tier in _client_tier_map.items()
                }
            self._server_state["client_group_assignment"] = _client_assignment
            self._server_state["tier_to_group"] = _tier_to_group
            self._server_state["num_tiers"] = _n_tiers

            # NOTE: Dead-client group substitution was removed.
            # Root-cause of removal: the substitution fired every round due to
            # staleness-based group rotation — prev_covered and new_covered
            # always differed (each round picks fresh stale groups), so
            # "orphaned_groups" was non-empty even with zero client deaths.
            # Effect: highest-battery clients were perpetually reassigned to
            # cheap shallow groups, leaving the output head (deep layers) without
            # updates → accuracy collapsed after round 15.
            #
            # The staleness mechanism in _assign_tiers_to_groups_v2 already
            # handles true group coverage gaps: orphaned groups accumulate
            # staleness and are naturally prioritised in the next few rounds.
            # When alive clients drop below num_tiers, _n_tiers is reduced
            # and the single-tier assignment cycles through all groups via
            # staleness (perfect round-robin coverage with 1 client).

        # ── Server-side EMA on global model weights ──────────────────────────
        # Design: EMA as server momentum (FedAvgM-style).
        #
        # With ema_model_alpha > 0:
        #   w_ema_{t+1} = (1 - α) · w_ema_t + α · w_{t+1}^{brut}
        #
        # The EMA model (w_ema) is broadcast to clients next round instead of
        # the raw aggregated model. Clients compute gradients at w_ema_t and
        # the repr proximal anchors toward w_ema_t — both are coherent.
        #
        # IMPORTANT: use with mu_weight=0 to avoid anchoring clients toward a
        # stale model. The repr proximal (mu_repr) is safe — it anchors the
        # function output, not the weights, relative to the broadcast model.
        #
        # ema_model_alpha = 0  → disabled (default, backward-compatible)
        # ema_model_alpha = α  → broadcast w_ema; good range: [0.3, 0.7]
        #   α=0.3: heavy smoothing (memory ~3 rounds) — most stable
        #   α=0.5: moderate smoothing (memory ~2 rounds)
        #   α=1.0: no smoothing (w_ema = w_brut) — equivalent to disabled
        _ema_model_alpha = float(config.get("ema_model_alpha", 0.0))
        if is_warmup:
            _ema_model_alpha = (
                0.0  # Disable EMA during warmup to ensure full FedAvg steps
            )

        if _ema_model_alpha > 0.0:
            if "w_ema" not in self._server_state:
                # First round: initialise EMA from current aggregated weights
                self._server_state["w_ema"] = OrderedDict(
                    {k: v.clone() for k, v in new_weights.items()}
                )
            else:
                w_ema = self._server_state["w_ema"]
                # In-place EMA update: w_ema = (1-α)·w_ema + α·w_new
                for k in new_weights:
                    if k in w_ema:
                        w_ema[k].mul_(1.0 - _ema_model_alpha).add_(
                            new_weights[k].to(w_ema[k].device),
                            alpha=_ema_model_alpha,
                        )
                    else:
                        w_ema[k] = new_weights[k].clone()
            # Broadcast w_ema instead of w_brut
            broadcast_weights = self._server_state["w_ema"]
        else:
            broadcast_weights = new_weights

        # ── Metrics ──────────────────────────────────────────────────────────
        # Use all alive clients for battery/energy tracking.
        # Use only participating (non-skipped) clients for loss/bytes metrics.
        active_updates_metrics = (
            [
                (u, m, s)
                for u, m, s in client_updates
                if not m.get("skipped", False) and u
            ]
            if not is_warmup
            else client_updates
        )
        K_act = max(len(active_updates_metrics), 1)

        total_bytes = sum(m["bytes_sent"] for _, m, _ in client_updates)
        total_energy = sum(m["energy_j_consumed"] for _, m, _ in client_updates)
        avg_battery = sum(s.battery_j for _, _, s in client_updates) / K
        avg_loss = sum(m["local_loss"] for _, m, _ in active_updates_metrics) / K_act

        # Jain fairness index on bytes sent (all alive clients)
        bytes_sent = [m["bytes_sent"] for _, m, _ in client_updates]
        if sum(bytes_sent) > 0:
            jain = (sum(bytes_sent) ** 2) / (K * sum(b**2 for b in bytes_sent))
        else:
            jain = 1.0

        # Participation: clients that actually trained (not skipped, not dead)
        participations = [
            0.0 if m.get("skipped", False) else 1.0 for _, m, _ in client_updates
        ]

        # ── σ² — inter-client update variance (free: deltas already in memory) ──
        # During warmup: deltas are full-model → direct estimate of gradient diversity.
        # During PNU:    deltas are partial (each client updated only its assigned
        #                group) → compare only within tiers (same group updated).
        #
        # Normalisation: Δ_k ≈ E * lr * ∇f_k  →  σ²_grad ≈ σ²_update / (E * lr)²
        # This gives a round-by-round proxy for σ² without any extra pass.
        sigma2_update = None
        sigma2_gradient_approx = None
        try:
            _lr = float(config.get("lr", 0.01))
            _E = float(config.get("local_epochs", 1))
            _norm = max((_E * _lr) ** 2, 1e-12)

            if is_warmup:
                # Full deltas → compare all clients directly
                flat_deltas = [
                    torch.cat([v.float().cpu().reshape(-1) for v in u.values()])
                    for u, m, _ in client_updates
                    if u and not m.get("skipped", False)
                ]
            else:
                # PNU: group clients by their active group, compute within-group variance
                from collections import defaultdict as _dd

                _tier_to_group = self._server_state.get("tier_to_group", {})
                _group_deltas: dict = _dd(list)
                for u, m, _ in client_updates:
                    if not u or m.get("skipped", False):
                        continue
                    _grp = m.get("active_group", None)
                    if _grp is not None:
                        flat = torch.cat(
                            [v.float().cpu().reshape(-1) for v in u.values()]
                        )
                        _group_deltas[_grp].append(flat)
                # Compute within-group variance, average across groups
                _vars = []
                for _grp_deltas in _group_deltas.values():
                    if len(_grp_deltas) < 2:
                        continue
                    _d = torch.stack(_grp_deltas)
                    _mean = _d.mean(dim=0, keepdim=True)
                    _vars.append(float(((_d - _mean) ** 2).sum(dim=1).mean()))
                flat_deltas = []  # signal: use _vars directly
                if _vars:
                    sigma2_update = float(sum(_vars) / len(_vars))
                    sigma2_gradient_approx = sigma2_update / _norm

            if flat_deltas and len(flat_deltas) >= 2:
                _d = torch.stack(flat_deltas)
                _mean = _d.mean(dim=0, keepdim=True)
                sigma2_update = float(((_d - _mean) ** 2).sum(dim=1).mean())
                sigma2_gradient_approx = sigma2_update / _norm
        except Exception:
            pass  # never let σ² logging crash the training loop

        metrics = {
            "round": round_num,
            "total_bytes_sent": total_bytes,
            "total_energy_j": total_energy,
            "avg_beta": 1.0,
            "avg_battery_j": avg_battery,
            "avg_local_loss": avg_loss,
            "compression_ratio": sum(
                m["compression_ratio"] for _, m, _ in active_updates_metrics
            )
            / K_act,
            "participation_rate": sum(participations) / K,
            "jain_index": jain,
            "num_clients": K,
            "is_warmup_round": is_warmup,
            "sigma2_update": sigma2_update,
            "sigma2_gradient_approx": sigma2_gradient_approx,
        }

        # FedPartBE-specific metrics
        if not is_warmup:
            metrics["staleness_vector"] = staleness.copy()
            metrics["active_groups"] = active_groups
            metrics["tier_to_group"] = self._server_state.get("tier_to_group", {})
            metrics["num_tiers"] = self._server_state.get("num_tiers", 0)
            metrics["skipped_clients"] = sum(
                1 for _, m, _ in client_updates if m.get("skipped", False)
            )
            metrics["rotation_period_effective"] = self._server_state.get(
                "_rot_effective_last", config.get("rotation_period", 0)
            )
            if _ema_model_alpha > 0.0:
                metrics["ema_model_alpha"] = _ema_model_alpha
                metrics["ema_active"] = True

        return AggregateResult(new_weights=broadcast_weights, metrics=metrics)

    # ── Default configuration ──────────────────────────────────────────────────

    def get_default_config(self) -> dict:
        """
        Default hyperparameters for FedPartBE.

        warmup_rounds     : 5    — initial full FedAvg rounds
        num_tiers         : 3    — number of energy tiers (K)
        rounds_per_tier   : 1    — rounds each tier assignment persists
        mu_weight         : 0.01 — weight proximal regularization coefficient
        mu_repr           : 0.1  — representation proximal regularization coefficient
        local_epochs      : 8    — same as FedPart
        lr                : 0.01
        momentum          : 0.9
        weight_decay      : 1e-4
        batch_size        : 32
        verbose_groups    : True — print layer group derivation
        device            : "cpu"
        device_profile    : None
        """
        return {
            "optimizer": "sgd",  # "sgd" | "adam"
            "lr": 0.01,
            "momentum": 0.9,  # SGD only (ignored when optimizer=adam)
            "weight_decay": 1e-4,
            "local_epochs": 8,
            "batch_size": 32,
            "warmup_rounds": 5,
            # num_tiers: number of simultaneous layer groups trained per round.
            # None → auto = num_groups // 2 (e.g. 5 tiers for 10 groups).
            # Each tier trains a different cost-bucket → no energy-gate skips.
            "num_tiers": None,
            "server_lr": None,  # None = auto 1/num_tiers; float to override
            "assignment_strategy": "quantile",  # "quantile" | "proportional"
            # quantile     : K/num_tiers clients per tier — robust for any battery spread
            # proportional : cost_k ∝ battery_k — optimal for wide spreads (SOC [5%,95%])
            # ── Group selection mode ──────────────────────────────────────────────────
            # "group_selection": controls how the M groups are chosen each round
            #   "staleness_aware" (default): priority = staleness/cost + cyclic rotation
            #   "sequential"               : sliding window of M consecutive groups
            "group_selection": "staleness_aware",
            # adaptive_rotation: when True, stretches rotation_period proportionally
            # to fleet shrinkage to maintain a constant gradient budget per phase.
            # Formula: T_rot_eff(r) = rotation_period × ceil(K_0 / K_alive(r))
            # K_0 is captured on the first PNU round and never updated.
            # Effect: prevents rotation transitions from coinciding with fleet
            # attrition events (compound-drop elimination). Zero new params to tune.
            # Use with staleness_aware mode only; ignored for sequential.
            "adaptive_rotation": False,
            # window_step: step size for sequential mode (group_selection="sequential")
            #   0    → auto = num_tiers (non-overlapping windows, recommended)
            #   1    → overlapping (each group active for M consecutive rounds)
            #   M    → same as 0/auto
            #   k    → any integer 1..G
            # Ignored when group_selection="staleness_aware".
            "window_step": 0,
            "ema_alpha": 0.3,  # EMA smoothing for gradient-norm tracker
            # ema_model_alpha: server EMA on global weights (FedAvgM-style).
            #   0.0  → disabled (default, backward-compatible)
            #   >0   → w_ema_{t+1} = (1-α)·w_ema_t + α·w_brut_{t+1}; broadcast w_ema
            #   Recommended range [0.3, 0.7]. Use with mu_weight=0 to avoid gradient conflict.
            "ema_model_alpha": 0.0,
            "mu_weight": 0.01,  # weight proximal coefficient
            "mu_repr": 0.1,  # representation proximal coefficient
            "verbose_groups": False,
            "device": "cpu",
            "device_profile": None,
            # persist_optimizer: carry per-group SGD momentum buffer across rounds.
            # Each group's state is saved in state.custom["optimizer_states"][group_idx]
            # and restored on the next visit. Beneficial when rotation_period >= 2.
            "persist_optimizer": True,
            # enforce_staleness_cap: when True, any group whose staleness reaches
            # τ_max = ceil(G/M) - 1 is FORCED into the active set for the next
            # round, overriding normal priority-based selection.  Also prevents
            # the energy gate from skipping a group that is at the staleness cap.
            # Set to False only for ablation studies that intentionally allow
            # unbounded staleness growth.
            "enforce_staleness_cap": True,
        }
