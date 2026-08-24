"""
algorithms/fedpart_be.py
========================
FedStep: Battery-Energy-aware Federated Partial Network Updates.
Extension of FedPart (Wang et al., NeurIPS 2024) with energy-tier assignments.

Naming: the algorithm is registered as "fedstep" (paper name). The historical
key "fedpart_be" is kept as a backward-compatible alias so that old configs
and result folders remain readable. The module filename is unchanged on
purpose (import paths, helper re-exports and tests depend on it).

Two operating modes (config key "exit_mode"):

  exit_mode = "group" (default — historical FedPartBE behaviour)
    Every client runs the FULL forward pass and trains one assigned layer
    group (partial backward). Battery drives WHICH group a client trains.
    Limitation (measured in E3): with rotation active, every client pays the
    same average cost F/3 + (2F/3)·(Σφ_g/M) = F/2 over a rotation cycle, so
    the battery has no grip on the average energy spend — survival is
    insensitive to num_tiers / rotation_period.

  exit_mode = "early_exit" (FedStep-EE — requires model "resnet8_ee")
    Battery drives HOW DEEP a client computes. Client k picks the largest
    exit depth d whose estimated round energy fits its per-round budget:

        k_i = max{ d : E_round(d) ≤ beta_budget × battery_i }        (β-rule)

    Forward AND backward are truncated at the chosen exit (deeper blocks are
    never executed), so the round cost is genuinely proportional to the
    prefix FLOPs — the weakest clients pay ~40% of a full round instead of
    ~50% forever (ResNet-8), and the differentiation persists because no
    rotation is needed for cost fairness. Aggregation is per-key coverage-
    weighted averaging: each parameter is averaged over the clients whose
    depth reached it. Warmup trains all exits jointly (CE at final head +
    aux_loss_weight × CE at each aux head).
    Related work: BranchyNet (early exits), DepthFL (ICLR 2023 — depth from
    STATIC device capacity). FedStep-EE selects depth from the RESIDUAL
    battery every round (dynamic).

Key mechanisms vs FedPart (group mode):
  1. Energy-tier layer assignment: sort clients by battery, match cheapest
     groups to lowest-battery clients.
  2. Representation-space proximal regularization: constrains the OUTPUT of
     the active group to stay close to global model output, preserving
     inter-layer compatibility. (E3 ablation: effect within seed noise on
     CIFAR-10 α=1 — kept for non-IID regimes, disable with mu_repr=0.)
  3. Staleness-aware layer priority: groups not updated recently get higher
     scheduling priority.
  4. Dynamic tier reorganization: server reassigns client→group mapping
     every round based on battery levels.

Algorithm (group mode):
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
      3. Trains for local_epochs with loss:
           L = CE_loss
               + mu_weight * ||W_gk - W_gk_global||^2_F   (weight proximal)
               + mu_repr   * MSE(h_gk(x), h_ref(x))       (representation proximal,
                 h_ref computed by a weight-swap forward with global weights)
      4. Uploads ONLY delta for group g_k (+ BN running stats)

    Server aggregation (single-pass weighted accumulation, Jacobi-style —
    the deltas are all computed on the same w^{t-1}, accumulated in one pass
    and applied in one fused GPU op; there is no Gauss-Seidel ordering):

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

import gc
import math
import warnings
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
    Sliding-window sequential group selection (FedStepSeq mode).

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


# ─────────────────────────────────────────────────────────────────────────────
# Early-exit helpers (exit_mode = "early_exit")
# ─────────────────────────────────────────────────────────────────────────────


def _derive_exit_param_names(model: nn.Module) -> list[list[str]]:
    """
    Cumulative trainable parameter names per exit depth.

    Reads the model's EXIT_PREFIXES contract (see models.registry.
    EarlyExitResNet8): entry d-1 lists the name prefixes ADDED at depth d.
    Returns exits[d-1] = every param name trained when exiting at depth d
    (prefix of the network + that depth's head), cumulative over depths.
    """
    prefixes_per_depth = getattr(model, "EXIT_PREFIXES", None)
    if prefixes_per_depth is None:
        raise ValueError(
            "exit_mode='early_exit' requires a model with EXIT_PREFIXES / aux "
            "heads (e.g. model: resnet8_ee). Got a plain "
            f"{model.__class__.__name__}."
        )
    all_names = [n for n, _ in model.named_parameters()]
    exits: list[list[str]] = []
    cumulative: list[str] = []
    for depth_prefixes in prefixes_per_depth:
        cumulative = cumulative + [
            n
            for n in all_names
            if any(n.startswith(p + ".") or n == p for p in depth_prefixes)
        ]
        exits.append(list(cumulative))
    return exits


def _exit_flop_fractions(
    exit_names: list[list[str]],
    groups: list[list[str]],
    group_flops: list[float],
) -> list[float]:
    """
    Fraction of the FULL-model round FLOPs paid at each exit depth.

    In early-exit mode forward, backward and optimizer are ALL truncated at
    the exit, so round_flops(d) = full_round_flops × (prefix fwd FLOPs /
    total fwd FLOPs). Aux-head Linear layers are negligible (<0.1%) and any
    parameter not covered by the analytic groups (e.g. aux heads) contributes
    zero here, which is the conservative choice.
    """
    total = max(sum(group_flops), 1e-12)
    fractions = []
    for names in exit_names:
        name_set = set(names)
        prefix_flops = sum(
            gf
            for group, gf in zip(groups, group_flops)
            if any(k in name_set for k in group)
        )
        fractions.append(min(1.0, prefix_flops / total))
    if fractions:
        fractions[-1] = 1.0  # deepest exit is by definition the full model
    return fractions


def _local_label_set(dataloader) -> set[int]:
    """Classes present in this client's local dataset (cached by the caller).

    Fast paths read the label array directly (torchvision datasets expose
    `targets`; Subset wraps them with `indices`); the fallback does one pass
    over the dataloader (loads images once — acceptable, runs once per
    client lifetime).
    """
    ds = dataloader.dataset
    try:
        import numpy as _np
        t = getattr(ds, "targets", None)
        if t is None:
            t = getattr(ds, "labels", None)
        if t is not None:
            return set(int(v) for v in _np.asarray(t).reshape(-1).tolist())
        if hasattr(ds, "indices") and hasattr(ds, "dataset"):
            base = ds.dataset
            bt = getattr(base, "targets", getattr(base, "labels", None))
            if bt is not None:
                bt = _np.asarray(bt).reshape(-1)
                idx = _np.asarray(ds.indices)
                return set(int(v) for v in bt[idx].tolist())
    except Exception:  # pragma: no cover — fall through to the slow path
        pass
    seen: set[int] = set()
    for _, y in dataloader:
        seen.update(int(v) for v in y.reshape(-1).tolist())
    return seen


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm
# ─────────────────────────────────────────────────────────────────────────────


@register_algorithm("fedstep")       # primary name (applied last → cls.name)
@register_algorithm("fedpart_be")    # legacy alias: old configs keep working
class FedStep(FLAlgorithm):
    """
    FedStep: Battery-Energy-aware Federated Partial Network Updates.

    Extension of FedPart with:
      - Energy-tier layer assignment (cheap groups to low-battery clients)
      - Representation-space proximal regularization
      - Staleness-aware layer priority
      - Optional early-exit mode (exit_mode="early_exit"): battery drives the
        computation DEPTH via the β-rule — see module docstring.

    Registration keys: "fedstep" (primary), "fedpart_be" (legacy alias).
    """

    name = "fedstep"
    description = (
        "FedStep: Battery-energy-aware partial network updates with "
        "energy-tier assignments, representation proximal regularization, "
        "and an optional battery-driven early-exit mode."
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
        4. Train with CE + weight proximal + representation proximal loss
           (h_ref via weight-swap forward with global weights).
        5. Return partial delta and metadata.

        exit_mode="early_exit" dispatches to _client_update_early_exit.
        """
        if config.get("exit_mode", "group") == "early_exit":
            return self._client_update_early_exit(model, dataloader, state, config)

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
                    f"  [FedStep] {len(groups)} layer groups derived "
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
        # This is the core FedStep innovation vs FedPart: selective participation
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
                    warnings.warn(
                        f"[FedStep] Client {state.client_id}: energy gate suppressed "
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
                if _do_repr and (epoch == 1 or epoch == local_epochs - 1):
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
            # FedStep-specific
            "active_group_idx": active_idx,
            "tier_idx": config.get("fedstep_tier", config.get("fedpartbe_tier", -1)),
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

    # ── Client (early-exit mode) ──────────────────────────────────────────────

    def _skip_meta(self, state, num_exits: int, is_warmup: bool, ds: int) -> dict:
        """Metadata skeleton for a skipped early-exit round."""
        return {
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
            "exit_depth_k": 0,
            "is_warmup": is_warmup,
            "num_layer_groups": num_exits,
            "dataset_size": ds,
        }

    def _client_update_early_exit(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        state: ClientState,
        config: dict,
    ) -> tuple[dict, dict]:
        """
        FedStep-EE local step: the battery picks the computation DEPTH.

        β-rule (per client, per round, no server assignment needed):
            k = max{ d : E_round(d) ≤ beta_budget × battery }
        Floor: if even the shallowest exit violates the β budget but still
        fits the remaining battery, train depth 1 anyway (participation floor
        — a client that can pay should contribute rather than zombie-skip).
        If depth 1 exceeds the remaining battery → client is dead.

        Forward AND backward stop at the chosen exit (deeper blocks are never
        executed), so round cost = full_round_flops × prefix_fraction(k) —
        the same cached full-round FLOPs and fractions are reused for the
        gate and for the drain, so estimate and accounting cannot drift.

        Warmup rounds train the full network with a joint multi-exit loss
        (CE_final + aux_loss_weight · Σ CE_aux) so aux heads are usable from
        the first PNU round.
        """
        device = config.get("device", "cpu")
        lr = config.get("lr", 0.01)
        # lr_schedule="cosine" (default "constant"): anneal the LOCAL lr over
        # the server horizon, lr(t) = lr_min + (lr−lr_min)·½(1+cos(π·t/T)),
        # lr_min = lr·lr_min_ratio. Late-round client drift under non-IID is
        # step-size noise; annealing shrinks it where it hurts (the plateau).
        # Uses _server_round/_total_rounds injected by the harness; falls back
        # to state.round_num (participation count ≈ server round at f=1.0).
        if str(config.get("lr_schedule", "constant")) == "cosine":
            _t = int(config.get("_server_round", state.round_num))
            _T = max(1, int(config.get("_total_rounds", 200)))
            _ratio = float(config.get("lr_min_ratio", 0.1))
            lr = lr * (_ratio + (1.0 - _ratio) * 0.5 * (1.0 + math.cos(math.pi * min(_t, _T) / _T)))
        momentum = config.get("momentum", 0.9)
        weight_decay = config.get("weight_decay", 1e-4)
        local_epochs = config.get("local_epochs", 8)
        max_grad_norm = config.get("max_grad_norm", 10.0)
        mu_weight = config.get("mu_weight", 0.01)
        mu_repr = config.get("mu_repr", 0.1)
        warmup_rounds = config.get("warmup_rounds", 5)
        persist_optimizer = config.get("persist_optimizer", True)
        beta_budget = float(config.get("beta_budget", 0.05))
        aux_loss_weight = float(config.get("aux_loss_weight", 0.3))
        esf = config.get("energy_scale_factor", 1.0)
        alpha_to = config.get("alpha_applies_to", "compute")
        profile = config.get("device_profile")

        model.train()
        model.to(device)
        ds_size = len(dataloader.dataset)

        # ── Exit structure (cached: architecture is fixed) ───────────────────
        if "ee_exit_names" not in state.custom:
            exit_names = _derive_exit_param_names(model)
            if "layer_groups" not in state.custom:
                state.custom["layer_groups"] = _derive_layer_groups(model)
            groups = state.custom["layer_groups"]
            if "group_flops" not in state.custom:
                input_shape = config.get("input_shape", (1, 3, 32, 32))
                state.custom["group_flops"] = _compute_group_flops(
                    groups, model, input_shape
                )
            state.custom["ee_exit_names"] = exit_names
            # Models with identical repeated blocks (transformers) declare
            # exact analytic fractions via EXIT_FRACTIONS; CNNs fall back to
            # the per-group FLOPs mapping.
            _model_fracs = getattr(model, "EXIT_FRACTIONS", None)
            state.custom["ee_fractions"] = (
                [float(f) for f in _model_fracs]
                if _model_fracs is not None
                else _exit_flop_fractions(
                    exit_names, groups, state.custom["group_flops"]
                )
            )
            if config.get("verbose_groups", True) and state.client_id == 0:
                fr = ", ".join(
                    f"k={i + 1}:{f:.2f}"
                    for i, f in enumerate(state.custom["ee_fractions"])
                )
                print(f"  [FedStep-EE] exit cost fractions: {fr}")
        exit_names: list[list[str]] = state.custom["ee_exit_names"]
        fractions: list[float] = state.custom["ee_fractions"]
        num_exits = len(exit_names)

        # ── Prefix WIDTH: how much of the executed prefix is TRAINED ────────
        # ee_width_mode:
        #   "all"      — whole prefix (legacy EE): backward + uplink cover the
        #                full forward. Reproduces FedAvg cost at k=max.
        #   "fixed"    — ee_train_groups rotated groups inside the prefix (V3
        #                Hybrid). Cheapest, but STARVES the deep/head blocks
        #                (they exist only at k=max AND are rarely rotated to).
        #   "adaptive" — anti-starvation (V3.1): the last ee_exempt_deep groups
        #                (output side) are ALWAYS trained when in the prefix,
        #                then as many rotated shallow groups as the budget
        #                allows. Deep layers never starve; width follows the
        #                battery on the SAME deadline budget as the depth.
        # Back-compat: ee_train_groups 0→"all", ≥1→"fixed" when ee_width_mode
        # is unset (so the V2a / V3 runs reproduce exactly).
        _wm_cfg = config.get("ee_width_mode")
        ee_width_mode = str(
            _wm_cfg
            if _wm_cfg is not None
            else ("all" if int(config.get("ee_train_groups", 0)) == 0 else "fixed")
        )
        ee_fixed_w = max(1, int(config.get("ee_train_groups", 1)))
        ee_exempt_deep = int(config.get("ee_exempt_deep", 0))
        _emw = config.get("ee_max_width")
        ee_max_width = None if _emw is None else int(_emw)

        if ee_width_mode != "all" and "ee_prefix_groups" not in state.custom:
            _groups_all = state.custom["layer_groups"]
            _gf = state.custom["group_flops"]

            def _head_owner(g: list[str]):
                # aux_heads.<i> only receives gradient when exiting at depth i;
                # picking it at another depth gives a loss with no grad path.
                first = g[0]
                return (
                    int(first.split(".")[1])
                    if first.startswith("aux_heads.")
                    else None
                )

            _pg: dict[int, list[int]] = {}
            _pphi: dict[int, dict[int, float]] = {}
            for d in range(1, num_exits + 1):
                _nameset = set(exit_names[d - 1])
                _idxs = [
                    gi
                    for gi, g in enumerate(_groups_all)
                    if g
                    and all(k in _nameset for k in g)
                    and _head_owner(g) in (None, d)
                ]
                # Position-aware cost WITHIN the prefix: a group pays its own
                # backward plus half of every deeper prefix group it must
                # propagate grad_input through (same law as the global c_g).
                _cs = {}
                for _pos, gi in enumerate(_idxs):
                    _cs[gi] = _gf[gi] + 0.5 * sum(_gf[j] for j in _idxs[_pos + 1 :])
                _tot = max(sum(_cs.values()), 1e-12)
                _pg[d] = _idxs  # architectural order (shallow → deep/head)
                _pphi[d] = {gi: c / _tot for gi, c in _cs.items()}
            state.custom["ee_prefix_groups"] = _pg
            state.custom["ee_prefix_phi"] = _pphi
            state.custom.setdefault("ee_group_cursor", {})

        def _prefix_split(d: int):
            """(rotating shallow groups, always-trained exempt deep groups)."""
            idxs = state.custom["ee_prefix_groups"][d]
            if 0 < ee_exempt_deep < len(idxs):
                return idxs[:-ee_exempt_deep], idxs[-ee_exempt_deep:]
            return idxs, []  # exempt_deep 0, or covers the whole prefix

        def _rotated(d: int, n: int) -> list:
            """n shallow groups starting at this depth's rotation cursor."""
            shallow, _ = _prefix_split(d)
            if not shallow or n <= 0:
                return []
            cur = state.custom["ee_group_cursor"].get(d, 0)
            return [shallow[(cur + i) % len(shallow)] for i in range(min(n, len(shallow)))]

        is_warmup = state.round_num < warmup_rounds

        # ── Full-round FLOPs + downlink bytes (cached; gate == accounting) ───
        # Cache keyed on the CURRENT dataset size: FLOPs scale linearly with
        # the number of local steps, and under a class-arrival protocol (E7)
        # the visible dataset GROWS over rounds — a first-participation cache
        # silently undercharges every later round (pilot bug, 2026-08-05).
        # The φ fractions are ratios (scale-invariant): only the absolute
        # total needs refreshing.
        full_bytes = sum(v.numel() for v in model.state_dict().values()) * 4
        _cur_ds_size = len(dataloader.dataset)
        if profile is not None and (
            "ee_full_flops" not in state.custom
            or state.custom.get("ee_flops_ds_size") != _cur_ds_size
        ):
            _all_names = [n for n, _ in model.named_parameters()]
            state.custom["ee_full_flops"] = round_compute_flops(
                model,
                _all_names,
                config,
                profile,
                dataloader,
                local_epochs,
                groups=None,
                active_group_idx=-1,
            )
            state.custom["ee_flops_ds_size"] = _cur_ds_size

        def _eff_fraction(depth: int, g_set=None) -> float:
            """Fraction of the full-round FLOPs actually paid.

            g_set is None → whole prefix (frac). Otherwise the forward still
            covers the prefix (frac/3) but the backward only the trained
            groups: frac·(1/3 + 2/3·Σ_{g∈g_set} φ_g^prefix). Training every
            prefix group (Σφ=1) recovers the whole-prefix cost frac.
            """
            frac = fractions[depth - 1]
            if g_set is None:
                return frac
            phi = state.custom["ee_prefix_phi"][depth]
            s = sum(phi.get(g, 0.0) for g in g_set)
            return frac * (1.0 / 3.0 + (2.0 / 3.0) * s)

        def _round_energy(depth: int, uplink_bytes: int, g_set=None) -> float:
            """Estimated/charged energy for one round at `depth` (1-based)."""
            frac = _eff_fraction(depth, g_set)
            if profile is not None:
                return profile.round_energy_breakdown(
                    state.custom["ee_full_flops"] * frac,
                    uplink_bytes,
                    full_bytes,
                    esf,
                    alpha_to,
                )["total"]
            # Profile-less fallback (tests / smoke): full round ≈ 2.5 units,
            # scaled by the effective fraction — same convention as group mode.
            return 2.5 * frac * esf

        def _group_names(g_set):
            if g_set is None:
                return None
            gl = state.custom["layer_groups"]
            return [n for gi in g_set for n in gl[gi]]

        def _uplink_bytes_est(depth: int, g_set=None) -> int:
            sd = model.state_dict()
            keys = self._ee_tx_keys(
                exit_names, depth, sd, group_names=_group_names(g_set)
            )
            return int(sum(sd[k].numel() for k in keys) * 4)

        def _cost(d: int, g_set) -> float:
            return _round_energy(d, _uplink_bytes_est(d, g_set), g_set)

        def _select_width(d: int, budget: float):
            """Trained group set at depth d (None = whole prefix)."""
            if ee_width_mode == "all":
                return None
            shallow, exempt = _prefix_split(d)
            if ee_width_mode == "fixed":
                return _rotated(d, ee_fixed_w) + list(exempt)
            # adaptive: exempt always, then greedily fill the budget with
            # rotated shallow groups (width follows the battery), capped at
            # ee_max_width added groups (None = uncapped). The cap bounds the
            # backward + uplink cost even when the budget is generous — the
            # lever that closes the energy/comm gap vs FedPart.
            g_set = list(exempt) if exempt else _rotated(d, 1)
            _added = 0
            for g in _rotated(d, len(shallow)):  # full rotation order
                if g in g_set:
                    continue
                if ee_max_width is not None and _added >= ee_max_width:
                    break
                if _cost(d, g_set + [g]) <= budget:
                    g_set = g_set + [g]
                    _added += 1
                else:
                    break
            return g_set

        # ── Depth + width selection (β-rule) / warmup energy gate ─────────────
        g_set = None  # groups trained this round (None = whole executed prefix)
        # depth_mode: "dynamic" (default) — depth re-chosen from the residual
        #   battery every round (FedStep-EE). "static" — the DepthFL/ScaleFL
        #   baseline: depth is FROZEN once, by the client's initial-battery
        #   TIER (capacity), and never adapts to depletion. This is the
        #   decisive control: does adapting depth to the DEPLETING battery
        #   help over a fixed capacity-tier assignment (esp. under non-IID,
        #   where a static weak client is shallow forever → permanent bias)?
        depth_mode = str(config.get("depth_mode", "dynamic"))
        if is_warmup:
            wu_energy = _round_energy(num_exits, full_bytes)
            if state.battery_j < wu_energy:
                state.round_num = warmup_rounds  # exit warmup like group mode
                return {}, self._skip_meta(state, num_exits, True, ds_size)
            depth = num_exits
        elif depth_mode == "static":
            # Freeze depth by initial-battery tier at the first PNU round.
            if "ee_static_depth" not in state.custom:
                _bmin = float(config.get("static_batt_min_j", 0.0))
                _bmax = float(config.get("static_batt_max_j", 0.0))
                _cap = (
                    min(1.0, max(0.0, (state.battery_j - _bmin) / (_bmax - _bmin)))
                    if _bmax > _bmin
                    else 0.5
                )
                # lowest-capacity tier → depth 1, highest → num_exits.
                state.custom["ee_static_depth"] = max(
                    1, min(num_exits, 1 + int(round(_cap * (num_exits - 1))))
                )
            depth = state.custom["ee_static_depth"]
            # Width per configured mode; DepthFL trains the whole prefix
            # ("all" → g_set None). No re-selection, no energy gate: a static
            # client trains its fixed depth and drains to death if it must.
            g_set = _select_width(depth, float("inf"))
        else:
            # "fraction": budget = β × battery (greedy-deep — FedAvg-early).
            # "deadline": budget = battery / rounds_remaining (sustainable —
            # survival to the horizon guaranteed by construction). The SAME
            # budget governs depth (which prefix) and width (how much of it).
            if str(config.get("beta_mode", "fraction")) == "deadline":
                _T = int(config.get("deadline_rounds", 200))
                _remaining = max(_T - state.round_num, 1)
                budget = state.battery_j / _remaining
            else:
                budget = beta_budget * state.battery_j
            depth = 0
            for d in range(num_exits, 0, -1):
                gs = _select_width(d, budget)
                if _cost(d, gs) <= budget:
                    depth, g_set = d, gs
                    break
            if depth == 0:
                # Budget too small for any exit → participation floor at k=1,
                # unless even that exceeds the whole remaining battery (death).
                gs1 = _select_width(1, budget)
                if state.battery_j < _cost(1, gs1):
                    state.battery_j = 0.0
                    return {}, self._skip_meta(state, num_exits, False, ds_size)
                depth, g_set = 1, gs1

        # ── Freeze everything outside the trained set ─────────────────────────
        # Whole prefix (g_set None) or the selected groups within it.
        active_names = (
            _group_names(g_set) if g_set is not None else exit_names[depth - 1]
        )
        # Head-training plumbing (repairs V3.2 KD + enables ee_aux_ce): the
        # width groups exclude aux_heads.<d≠depth> (_head_owner filter, whose
        # "no grad path" assumption predates forward_exits_upto), so the
        # freeze below silently dropped every gradient the multi-exit forward
        # gave the shallow heads — ee_self_distill acted on the trunk only.
        # When a head-training mechanism is on, unfreeze (and transmit) the
        # on-path heads: a few hundred params, negligible bytes/energy.
        _aux_head_names: list[str] = []
        if (
            not is_warmup
            and depth >= 2
            and (
                float(config.get("ee_self_distill", 0.0)) > 0.0
                or float(config.get("ee_aux_ce", 0.0)) > 0.0
                or float(config.get("ee_global_kd", 0.0)) > 0.0
            )
        ):
            _seen = set(active_names)
            _aux_head_names = [
                n
                for n, _ in model.named_parameters()
                if n.startswith("aux_heads.")
                and int(n.split(".")[1]) < depth
                and n not in _seen
            ]
            active_names = list(active_names) + _aux_head_names
        _flopcost_freeze_to_trainable(model, active_names)

        # ── Proximal bookkeeping (same schedule as group mode) ───────────────
        w_before = OrderedDict(
            {k: v.clone().cpu() for k, v in model.state_dict().items()}
        )
        # ee_global_kd needs the same machinery as the anchor: a copy of the
        # global weights and the swap buffers (its teacher pass piggybacks the
        # anchor-epoch weight swap).
        _gkd_on = not is_warmup and float(config.get("ee_global_kd", 0.0)) > 0.0
        w_global: dict = {}
        if not is_warmup and (mu_weight > 0 or mu_repr > 0 or _gkd_on):
            w_global = {
                k: v.clone().to(device) for k, v in model.state_dict().items()
            }
        _do_repr = not is_warmup and mu_repr > 0 and bool(w_global)
        active_set = set(active_names)
        active_param_refs = (
            {n: p for n, p in model.named_parameters() if n in active_set}
            if (not is_warmup and (mu_weight > 0 or mu_repr > 0 or _gkd_on))
            else {}
        )
        saved_buf = (
            {k: torch.empty_like(p.data) for k, p in active_param_refs.items()}
            if (_do_repr or _gkd_on)
            else {}
        )

        # ── Optimizer (state persisted per depth) ────────────────────────────
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if config.get("optimizer", "sgd").lower() == "adam":
            optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(
                trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        # Optimizer state is keyed by the exact trained param set so a reload
        # only reuses momentum when the parameters match (they must, or Adam's
        # state_dict load would mismatch). With rotating widths the same key
        # recurs rarely, which is fine — persistence is an optimisation, not
        # a correctness requirement (the E1 runs use persist_optimizer=False).
        opt_key = (
            f"ee{depth}"
            if g_set is None
            else "ee%d:%s" % (depth, "-".join(map(str, sorted(g_set))))
        )
        if persist_optimizer and not is_warmup:
            saved = state.custom.get("optimizer_states", {}).get(opt_key)
            if saved is not None:
                optimizer.load_state_dict(saved)
        criterion = nn.CrossEntropyLoss()

        # ── Local training (truncated at `depth`) ────────────────────────────
        # V2a additions (all default-off, see get_default_config):
        #   repr_anchor="boundary" — anchor the GAP-pooled feature at the exit
        #     boundary (the interface deeper blocks consume) instead of the
        #     head logits, which are blind to most of that interface.
        #   mu_adaptive — FedDPA-style state-dependent strength: μ_eff grows
        #     with the measured local/global boundary divergence, so aligned
        #     clients pay ~μ0 and drifting clients are held harder.
        exit_k = None if (is_warmup or depth == num_exits) else depth
        repr_anchor = str(config.get("repr_anchor", "logits"))
        use_boundary = _do_repr and repr_anchor == "boundary"
        mu_adaptive = bool(config.get("mu_adaptive", False))
        mu_adapt_scale = float(config.get("mu_adapt_scale", 1.0))
        # V3.2 addition (default-off): deep→shallow self-distillation.
        #   ee_self_distill=λ>0 — at depth k≥2 the deepest ON-PATH exit
        #   teaches the shallower exits (KL on temperature-softened logits,
        #   teacher detached). DepthFL/ScaleFL-style self-distillation
        #   restricted to the client's truncated prefix: aligns shallow
        #   heads/trunk with deep semantics WITHOUT sharing one fc across
        #   depths (a shared head mixes shallow/deep feature statistics and
        #   invites gradient conflict — the reason DepthFL keeps heads
        #   separate). Complements ee_exempt_deep: exemption guarantees deep
        #   COVERAGE, KD propagates deep SEMANTICS shallow-ward.
        kd_weight = float(config.get("ee_self_distill", 0.0))
        kd_temp = float(config.get("ee_distill_temp", 3.0))
        do_kd = (not is_warmup) and kd_weight > 0.0 and depth >= 2
        # ee_aux_ce=w>0 — joint CE beyond warmup: every depth≥2 client adds
        # w·CE(true labels) at each of ITS on-path shallow heads, so heads
        # track the evolving trunk instead of freezing at their warmup state
        # (the starvation measured when the β-rule assigns nobody to a depth).
        # True-label supervision, unlike KD, cannot distill the local
        # teacher's class bias into the heads under label skew. Shares the
        # multi-exit forward and the unfreeze plumbing with ee_self_distill.
        aux_ce_weight = float(config.get("ee_aux_ce", 0.0))
        do_aux_ce = (not is_warmup) and aux_ce_weight > 0.0 and depth >= 2
        # ee_global_kd=λg>0 — GLOBAL-teacher distillation (FedGKD-style, per
        # exit): on anchor epochs, while the weights are swapped to the global
        # model for the anchor's reference pass, forward_exits_upto also yields
        # the GLOBAL logits at every on-path head; each local head d ≤ depth is
        # then distilled toward its global counterpart. Unlike ee_self_distill
        # (local deep head = teacher trained on this client's 1-2 classes,
        # which distills the local bias) the global teacher has seen ALL
        # classes through aggregation and is IDENTICAL across clients — a
        # consistency signal, not a bias injection. Works at depth 1 too.
        # Cost: free when the anchor is on (mu_repr>0 — the teacher pass
        # rides the anchor's swapped forward, active on the 2 anchor epochs
        # only). With mu_repr=0 it ADDS an uncharged no-grad forward on those
        # epochs (energy accounting has no gkd term) — pair it with the
        # anchor, as every production recipe does.
        gkd_weight = float(config.get("ee_global_kd", 0.0))
        do_gkd = (not is_warmup) and gkd_weight > 0.0
        gkd_ntd = bool(config.get("ee_gkd_ntd", False))
        # ee_global_kd_scope: "all" (default) distills every on-path head;
        # "final" only the deepest on-path head — the dual-teacher split
        # (local KD teaches heads 1..k-1, global KD teaches head k) that
        # gives each teacher the territory where it measured best.
        _gkd_students = (
            [depth]
            if str(config.get("ee_global_kd_scope", "all")) == "final"
            else list(range(1, depth + 1))
        )
        do_multi = do_kd or do_aux_ce or do_gkd
        # Teacher pass must not be dropout-noised (ViT variants have live
        # Dropout in train mode; resnet8_ee has none). Only Dropout modules
        # are toggled — BN/GN behavior is deliberately left untouched.
        _dropout_mods = (
            [m for m in model.modules() if isinstance(m, nn.Dropout)]
            if do_gkd
            else []
        )
        # FedRS-style restricted softmax (Li & Zhan, KDD 2021), default-off.
        #   fedrs_alpha=a<1 — in the LOCAL CE only, logits of classes ABSENT
        #   from this client's data are scaled by a, shrinking the gradient
        #   that pushes their classifier rows down (the label-skew head-bias
        #   mechanism: a 1-2-class client spends most of its CE gradient
        #   "destroying" the classes it never sees). Applied at every trained
        #   head (warmup joint loss included); inference, anchors and
        #   transmitted deltas are untouched. Natural complement of the
        #   exemption: exempt heads are ALWAYS trained, hence always exposed
        #   to the bias this flag removes.
        fedrs_alpha = float(config.get("fedrs_alpha", 1.0))
        _rs_scale = None
        if fedrs_alpha < 1.0:
            if "fedrs_present" not in state.custom:
                state.custom["fedrs_present"] = _local_label_set(dataloader)
            _rs_present = state.custom["fedrs_present"]

        def _rs_ce(logits, target):
            nonlocal _rs_scale
            if fedrs_alpha >= 1.0:
                return criterion(logits, target)
            if _rs_scale is None or _rs_scale.device != logits.device:
                s = torch.full((logits.shape[1],), fedrs_alpha,
                               device=logits.device)
                idx = [c for c in _rs_present if c < logits.shape[1]]
                s[idx] = 1.0
                _rs_scale = s
            return criterion(logits * _rs_scale, target)

        total_loss = total_ce = total_wprox = total_rprox = 0.0
        total_kd = 0.0
        mu_eff_sum = 0.0
        mu_eff_n = 0
        ce_hist: list[float] = []  # per-batch CE — feeds update_variance metric
        num_batches = 0
        for epoch in range(local_epochs):
            is_anchor_epoch = epoch == 1 or epoch == local_epochs - 1
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                f_loc = None
                if is_warmup:
                    outs = model.forward_all_exits(x)
                    final = outs[num_exits]
                    ce_loss = _rs_ce(final, y) + aux_loss_weight * sum(
                        _rs_ce(outs[d], y) for d in range(1, num_exits)
                    )
                    out = final
                else:
                    outs_kd = None
                    if do_multi:
                        # One truncated multi-exit forward (prefix ≤ depth):
                        # same traversal as forward(exit_k=depth), plus the
                        # tiny shallow heads (features already computed).
                        if use_boundary and is_anchor_epoch:
                            outs_kd, f_loc = model.forward_exits_upto(
                                x, depth, return_boundary=True
                            )
                        else:
                            outs_kd = model.forward_exits_upto(x, depth)
                        out = outs_kd[depth]
                    elif use_boundary and is_anchor_epoch:
                        out, f_loc = (
                            model(x, exit_k=exit_k, return_boundary=True)
                            if exit_k
                            else model(x, return_boundary=True)
                        )
                    else:
                        out = model(x, exit_k=exit_k) if exit_k else model(x)
                    ce_loss = _rs_ce(out, y)
                    if do_aux_ce:
                        ce_loss = ce_loss + aux_ce_weight * sum(
                            _rs_ce(outs_kd[d], y) for d in range(1, depth)
                        )

                wprox = 0.0
                if active_param_refs and mu_weight > 0:
                    for k, p in active_param_refs.items():
                        if k in w_global:
                            wprox += torch.sum((p - w_global[k]) ** 2)
                    wprox *= mu_weight / 2.0

                rprox = 0.0
                outs_glob = None
                if (_do_repr or do_gkd) and is_anchor_epoch:
                    with torch.no_grad():
                        # Swap active params to global. Frozen params never
                        # left their round-start (= global) values, so the
                        # whole model IS the global model during the swap.
                        for k, p in active_param_refs.items():
                            saved_buf[k].copy_(p.data)
                            p.data.copy_(w_global[k])
                        if do_gkd:
                            # One multi-exit global pass serves both the gkd
                            # teacher (outs_glob) and the anchor reference
                            # (f_glob / href) — same traversal cost as the
                            # single-exit pass it replaces. Dropout silenced
                            # for a deterministic teacher (review finding).
                            for _m in _dropout_mods:
                                _m.eval()
                            if use_boundary:
                                outs_glob, f_glob = model.forward_exits_upto(
                                    x, depth, return_boundary=True
                                )
                                f_glob = f_glob.detach()
                            else:
                                outs_glob = model.forward_exits_upto(x, depth)
                            href = outs_glob[depth]
                            for _m in _dropout_mods:
                                _m.train()
                        elif use_boundary:
                            _, f_glob = (
                                model(x, exit_k=exit_k, return_boundary=True)
                                if exit_k
                                else model(x, return_boundary=True)
                            )
                            f_glob = f_glob.detach()
                        else:
                            href = (
                                model(x, exit_k=exit_k) if exit_k else model(x)
                            ).detach()
                        for k, p in active_param_refs.items():
                            p.data.copy_(saved_buf[k])
                    if use_boundary:
                        mu_eff = mu_repr
                        if mu_adaptive:
                            _div = float(
                                (f_loc.detach() - f_glob).norm()
                                / (f_glob.norm() + 1e-8)
                            )
                            mu_eff = mu_repr * (1.0 + mu_adapt_scale * _div)
                        mu_eff_sum += mu_eff
                        mu_eff_n += 1
                        rprox = mu_eff * torch.mean((f_loc - f_glob) ** 2)
                    else:
                        rprox = mu_repr * torch.mean((out - href) ** 2)

                kd_loss = 0.0
                if do_kd:
                    # Deepest on-path exit = detached teacher; every
                    # shallower exit = student. KL over T-softened logits,
                    # ×T² (Hinton scaling), averaged over students so λ is
                    # comparable across depths.
                    t_soft = torch.softmax(
                        outs_kd[depth].detach() / kd_temp, dim=1
                    )
                    kd_loss = sum(
                        nn.functional.kl_div(
                            torch.log_softmax(outs_kd[d] / kd_temp, dim=1),
                            t_soft,
                            reduction="batchmean",
                        )
                        for d in range(1, depth)
                    ) * (kd_weight * kd_temp * kd_temp / (depth - 1))

                gkd_loss = 0.0
                if do_gkd and outs_glob is not None:
                    # Every on-path head (final included) distilled toward its
                    # GLOBAL counterpart. Same Hinton T² scaling as ee_self_
                    # distill; averaged over the `depth` students so λg is
                    # comparable across depths. Teacher logits are no-grad.
                    # ee_gkd_ntd (FedNTD, Lee et al. NeurIPS'22): mask the
                    # TRUE-class logit out of both softmaxes so the KL only
                    # preserves global knowledge over the not-true classes —
                    # CE keeps sole authority on the true class, and a
                    # mediocre teacher can no longer cap the heads on it.
                    if gkd_ntd:
                        _mask = torch.zeros_like(outs_kd[depth])
                        _mask.scatter_(1, y.unsqueeze(1), -1e9)
                        gkd_loss = sum(
                            nn.functional.kl_div(
                                torch.log_softmax(
                                    (outs_kd[d] + _mask) / kd_temp, dim=1
                                ),
                                torch.softmax(
                                    (outs_glob[d] + _mask) / kd_temp, dim=1
                                ),
                                reduction="batchmean",
                            )
                            for d in _gkd_students
                        ) * (gkd_weight * kd_temp * kd_temp / len(_gkd_students))
                    else:
                        gkd_loss = sum(
                            nn.functional.kl_div(
                                torch.log_softmax(outs_kd[d] / kd_temp, dim=1),
                                torch.softmax(outs_glob[d] / kd_temp, dim=1),
                                reduction="batchmean",
                            )
                            for d in _gkd_students
                        ) * (gkd_weight * kd_temp * kd_temp / len(_gkd_students))

                loss = ce_loss + wprox + rprox + kd_loss + gkd_loss
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, max_norm=max_grad_norm
                    )
                optimizer.step()

                total_loss += loss.item()
                _ce = ce_loss.item()
                total_ce += _ce
                ce_hist.append(_ce)
                total_wprox += wprox if isinstance(wprox, float) else wprox.item()
                total_rprox += rprox if isinstance(rprox, float) else rprox.item()
                total_kd += kd_loss if isinstance(kd_loss, float) else kd_loss.item()
                total_kd += (
                    gkd_loss if isinstance(gkd_loss, float) else gkd_loss.item()
                )
                num_batches += 1

        if persist_optimizer and not is_warmup:
            state.custom.setdefault("optimizer_states", {})[opt_key] = (
                optimizer.state_dict()
            )
        if g_set is not None:
            # Advance the client-local rotation by the number of SHALLOW
            # (non-exempt) groups trained, so the next visit continues the
            # round-robin over the rotating groups (exempt deep groups are
            # always trained and don't consume the cursor).
            _shallow_d, _ = _prefix_split(depth)
            _n_rot = len([g for g in g_set if g in set(_shallow_d)])
            if _n_rot:
                _cur = state.custom["ee_group_cursor"].get(depth, 0)
                state.custom["ee_group_cursor"][depth] = _cur + _n_rot
        _flopcost_restore_grad(model)
        if w_global:
            del w_global
        gc.collect()

        # ── Delta: trained params + BN buffers of the EXECUTED prefix ────────
        current_sd = model.state_dict()
        if is_warmup:
            tx_keys = list(w_before.keys())
        else:
            tx_keys = self._ee_tx_keys(
                exit_names, depth, w_before, group_names=_group_names(g_set)
            )
        # On-path aux heads trained via the head-training plumbing must reach
        # the server too — _ee_tx_keys follows the width groups, which exclude
        # them (same exclusion the freeze fix above compensates).
        for _n in _aux_head_names:
            if _n in w_before and _n not in tx_keys:
                tx_keys.append(_n)
        partial_delta = OrderedDict(
            {k: (w_before[k] - current_sd[k].cpu()).float() for k in tx_keys}
        )

        # ── intN delta quantization + error feedback (default-off) ───────────
        # uplink_quant_bits=8|4 → per-tensor symmetric intN with STOCHASTIC
        # rounding and ERROR FEEDBACK: the residual (delta − dequant(delta))
        # is carried in state.custom and added back next round, so the scheme
        # is unbiased over rounds (Karimireddy et al. 2019). Communication
        # drops ~4× (int8) / ~8× (int4: 2 values packed per byte) + one
        # float32 scale/tensor. Deltas quantize far better than weights; at
        # 4 bits (15 levels) EF is what keeps the scheme viable. The SERVER
        # receives the dequantized delta (we quantize→dequantize here and
        # transmit the reconstructed values), so aggregation is unchanged;
        # only the BYTE COUNT reflects the compressed payload.
        q_bits = int(config.get("uplink_quant_bits", 0))
        if q_bits in (4, 8) and not is_warmup:
            qmax = (1 << (q_bits - 1)) - 1  # 127 (int8) | 7 (int4)
            ef = state.custom.setdefault("ef_residual", {})
            n_scales = 0
            for k, d in partial_delta.items():
                d = d + ef.get(k, 0.0)  # re-inject last round's residual
                amax = d.abs().max()
                if amax < 1e-12:
                    ef[k] = d
                    partial_delta[k] = torch.zeros_like(d)
                    n_scales += 1
                    continue
                scale = amax / qmax
                noise = torch.rand_like(d) - 0.5  # stochastic rounding
                q = torch.clamp(torch.round(d / scale + noise), -qmax, qmax)
                deq = q * scale
                ef[k] = d - deq            # residual carried forward
                partial_delta[k] = deq     # server gets the reconstructed delta
                n_scales += 1
            # bytes: q_bits/8 per value (int4 packs 2/byte) + one float32 scale/tensor
            _nvals = sum(t.numel() for t in partial_delta.values())
            uplink_bytes = (_nvals * q_bits + 7) // 8 + 4 * n_scales
        else:
            uplink_bytes = self.count_bytes(partial_delta, sparse=False)

        # ── Accounting (same numbers as the gate → no drift) ─────────────────
        full_model_bytes = self.count_bytes(w_before, sparse=False)
        del w_before, current_sd
        gc.collect()

        _g_acct = g_set if not is_warmup else None
        if profile is not None:
            frac = _eff_fraction(depth, _g_acct)
            _bd = profile.round_energy_breakdown(
                state.custom["ee_full_flops"] * frac,
                uplink_bytes,
                full_model_bytes,
                esf,
                alpha_to,
            )
        else:
            e = _round_energy(depth, uplink_bytes, _g_acct)
            _bd = {"compute": e, "uplink": 0.0, "downlink": 0.0, "total": e}
        energy_j = _bd["total"]
        state.battery_j = max(0.0, state.battery_j - energy_j)
        state.round_num += 1

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
            "bytes_received": full_model_bytes,
            "local_loss": total_ce / max(num_batches, 1),
            "compression_ratio": uplink_bytes / max(full_model_bytes, 1),
            "active_group_idx": -1,  # server tier machinery unused in EE mode
            "exit_depth_k": depth,
            "exit_fraction": fractions[depth - 1],
            "ee_trained_groups": (sorted(g_set) if g_set is not None else []),
            "ee_width": (len(g_set) if g_set is not None else -1),
            "ee_eff_fraction": _eff_fraction(depth, _g_acct),
            "weight_prox_loss": total_wprox / max(num_batches, 1),
            "repr_prox_loss": total_rprox / max(num_batches, 1),
            "kd_loss": total_kd / max(num_batches, 1),
            # update_variance: squared coefficient of variation of the
            # per-batch CE losses this round — scale-free instability proxy
            # (FedBS-style), consumed by the var-aware server aggregation.
            "update_variance": (
                float(
                    (sum((c - total_ce / num_batches) ** 2 for c in ce_hist)
                     / num_batches)
                    / max((total_ce / num_batches) ** 2, 1e-12)
                )
                if num_batches > 1
                else 0.0
            ),
            "mu_eff_mean": (mu_eff_sum / mu_eff_n) if mu_eff_n else mu_repr,
            "repr_anchor": repr_anchor,
            "is_warmup": is_warmup,
            "num_layer_groups": num_exits,
            "dataset_size": ds_size,
        }
        return dict(partial_delta), metadata

    @staticmethod
    def _ee_tx_keys(
        exit_names: list[list[str]],
        depth: int,
        sd: dict,
        group_names: list[str] | None = None,
    ) -> list[str]:
        """Keys transmitted at `depth`: trained params + BN buffers of the
        EXECUTED prefix (the forward updates every prefix BN regardless of
        which params trained; deeper BN stats never moved — forward stopped).

        group_names (hybrid): restrict the trained params to that single
        group; the BN-buffer set still spans the whole executed prefix."""
        trained = group_names if group_names is not None else exit_names[depth - 1]
        params = [k for k in trained if k in sd]
        prefix_modules = {
            k.rsplit(".", 1)[0] for k in exit_names[depth - 1] if k in sd
        }
        bn = [
            k
            for k in sd
            if k.endswith(("running_mean", "running_var", "num_batches_tracked"))
            and k.rsplit(".", 1)[0] in prefix_modules
        ]
        seen = set(params)
        return params + [k for k in bn if k not in seen]

    # ── Server ─────────────────────────────────────────────────────────────────

    def server_aggregate(
        self,
        global_model: nn.Module,
        client_updates: list[tuple[dict, dict, ClientState]],
        round_num: int,
        config: dict,
    ) -> AggregateResult:
        """
        Energy-tier-aware aggregation (single-pass Jacobi, fused GPU apply).

        During warmup: full FedAvg aggregation.
        After warmup (group mode):
          1. Compute tier assignments for next round
          2. Aggregate deltas per group (weighted mean, one fused apply)
          3. Update staleness vector
          4. Return new weights and metrics

        exit_mode="early_exit" dispatches to _server_aggregate_early_exit
        (per-key coverage-weighted averaging — no tiers, no staleness).
        """
        if config.get("exit_mode", "group") == "early_exit":
            return self._server_aggregate_early_exit(
                global_model, client_updates, round_num, config
            )

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
            # True FedStep: num_tiers groups trained simultaneously each round.
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
                    f"  [FedStep] R{round_num} | {' | '.join(parts)} | "
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
                # ── Sequential sliding window (FedStepSeq mode) ────────────
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
                    _grp = m.get("active_group_idx", None)  # was "active_group": bug — σ² was never computed during PNU
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

        # FedStep-specific metrics
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

    # ── Server (early-exit mode) ──────────────────────────────────────────────

    def _server_aggregate_early_exit(
        self,
        global_model: nn.Module,
        client_updates: list[tuple[dict, dict, ClientState]],
        round_num: int,
        config: dict,
    ) -> AggregateResult:
        """
        FedStep-EE aggregation: per-key coverage-weighted averaging.

        Each parameter key is averaged over the clients whose exit depth
        reached it, weighted by dataset size:

            w[key] -= ee_server_lr × Σ_i n_i·δ_i[key] / Σ_i n_i[key]

        Shallow layers are averaged over (almost) the whole fleet; deep
        layers only over the clients that could afford them. Every key gets
        exactly ONE weighted-mean delta → ee_server_lr defaults to 1.0
        (there is no multi-group displacement issue here). No tiers, no
        staleness, no rotation: the client-side β-rule replaces assignment.
        Warmup rounds aggregate with the same per-key rule (all clients send
        the full model → it reduces to plain weighted FedAvg).
        """
        K = len(client_updates)
        global_sd = global_model.state_dict()
        warmup_rounds = config.get("warmup_rounds", 5)
        is_warmup = round_num < warmup_rounds
        server_lr = float(config.get("ee_server_lr", 1.0))

        active = [
            (u, m, s)
            for u, m, s in client_updates
            if not m.get("skipped", False) and u
        ]

        new_weights = OrderedDict({k: v.float() for k, v in global_sd.items()})
        sums: dict[str, torch.Tensor] = {}
        counts: dict[str, float] = defaultdict(float)
        # Variance-aware weighting (V2a, FedBS/FedPBS-style, default off):
        # w_i = n_i · exp(−τ · Var_i). Down-weights unstable clients on the
        # keys they share with the fleet (the prefix), protecting stem/layer1
        # from noisy shallow-client updates under severe non-IID.
        _var_aware = bool(config.get("var_aware_agg", False))
        _var_tau = float(config.get("var_agg_tau", 4.0))
        # bn_agg_mode="plain": BN running statistics are NOT gradients — they
        # must not go through the gradient machinery. Under severe non-IID this
        # is what destabilises training (measured: loss blow-ups, σ=18 across
        # seeds). In "plain" mode BN buffers are aggregated as a pure
        # data-weighted mean (no variance re-weighting, no server_lr, and
        # excluded from the EMA below); weights keep the standard treatment.
        _bn_plain = str(config.get("bn_agg_mode", "standard")).lower() == "plain"
        _is_bn = lambda k: k.endswith(  # noqa: E731
            ("running_mean", "running_var", "num_batches_tracked")
        )
        sums_bn: dict[str, torch.Tensor] = {}
        counts_bn: dict[str, float] = defaultdict(float)
        for upd, meta, _ in active:
            n_raw = float(meta.get("dataset_size", 1))
            n_k = n_raw
            if _var_aware and not is_warmup:
                _v = float(meta.get("update_variance", 0.0) or 0.0)
                n_k *= math.exp(-_var_tau * min(_v, 10.0))
            for k, v in upd.items():
                vf = v.float()
                if _bn_plain and _is_bn(k):
                    if k not in sums_bn:
                        sums_bn[k] = vf.mul(n_raw)
                    else:
                        sums_bn[k].add_(vf, alpha=n_raw)
                    counts_bn[k] += n_raw
                    continue
                if k not in sums:
                    sums[k] = vf.mul(n_k)
                else:
                    sums[k].add_(vf, alpha=n_k)
                counts[k] += n_k

        for k, s_ in sums.items():
            if k in new_weights and counts[k] > 0:
                new_weights[k] = new_weights[k] - server_lr * (
                    s_.div_(counts[k]).to(new_weights[k].device)
                )
        # BN buffers: plain data-weighted mean of the clients' local stats
        # (server_lr = 1 by construction, no variance re-weighting).
        for k, s_ in sums_bn.items():
            if k in new_weights and counts_bn[k] > 0:
                new_weights[k] = new_weights[k] - (
                    s_.div_(counts_bn[k]).to(new_weights[k].device)
                )
        del sums, sums_bn
        gc.collect()

        # ── Server-side EMA on global weights (FedAvgM-style) ─────────────────
        # Same contract as the group-mode block in server_aggregate: with
        # ema_model_alpha > 0, w_ema = (1-α)·w_ema + α·w_brut and the EMA
        # model is broadcast instead of the raw aggregate. Damps single-round
        # aggregation blow-ups (a destroyed aggregate weighs α, not 100%).
        # Disabled during warmup (full FedAvg steps) and by default (α=0).
        _ema_alpha = float(config.get("ema_model_alpha", 0.0))
        if is_warmup:
            _ema_alpha = 0.0
        if _ema_alpha > 0.0:
            if not hasattr(self, "_server_state"):
                self._server_state = {}
            if "w_ema_ee" not in self._server_state:
                self._server_state["w_ema_ee"] = OrderedDict(
                    {k: v.clone() for k, v in new_weights.items()}
                )
            else:
                w_ema = self._server_state["w_ema_ee"]
                for k in new_weights:
                    # bn_agg_mode="plain": BN buffers bypass the momentum —
                    # smoothing a running STATISTIC over rounds compounds the
                    # staleness instead of damping noise. Take them fresh.
                    if _bn_plain and _is_bn(k):
                        w_ema[k] = new_weights[k].clone()
                        continue
                    if k in w_ema:
                        w_ema[k].mul_(1.0 - _ema_alpha).add_(
                            new_weights[k].to(w_ema[k].device),
                            alpha=_ema_alpha,
                        )
                    else:
                        w_ema[k] = new_weights[k].clone()
            new_weights = self._server_state["w_ema_ee"]

        # ── Metrics ───────────────────────────────────────────────────────────
        K_act = max(len(active), 1)
        total_bytes = sum(m["bytes_sent"] for _, m, _ in client_updates)
        total_energy = sum(m["energy_j_consumed"] for _, m, _ in client_updates)
        avg_battery = sum(s.battery_j for _, _, s in client_updates) / max(K, 1)
        avg_loss = sum(m["local_loss"] for _, m, _ in active) / K_act
        bytes_sent = [m["bytes_sent"] for _, m, _ in client_updates]
        jain = (
            (sum(bytes_sent) ** 2) / (K * sum(b**2 for b in bytes_sent))
            if sum(bytes_sent) > 0
            else 1.0
        )
        exit_hist: dict[int, int] = defaultdict(int)
        for _, m, _ in active:
            exit_hist[int(m.get("exit_depth_k", 0))] += 1

        metrics = {
            "round": round_num,
            "total_bytes_sent": total_bytes,
            "total_energy_j": total_energy,
            "avg_beta": 1.0,
            "avg_battery_j": avg_battery,
            "avg_local_loss": avg_loss,
            "compression_ratio": sum(m["compression_ratio"] for _, m, _ in active)
            / K_act,
            "participation_rate": len(active) / max(K, 1),
            "jain_index": jain,
            "num_clients": K,
            "is_warmup_round": is_warmup,
            "exit_mode": "early_exit",
            "exit_histogram": dict(sorted(exit_hist.items())),
            "skipped_clients": sum(
                1 for _, m, _ in client_updates if m.get("skipped", False)
            ),
        }
        return AggregateResult(new_weights=new_weights, metrics=metrics)

    # ── Default configuration ──────────────────────────────────────────────────

    def get_default_config(self) -> dict:
        """
        Default hyperparameters for FedStep.

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
            # ── Early-exit mode (FedStep-EE) ──────────────────────────────────
            # exit_mode: "group" (historical FedPartBE behaviour, default) or
            # "early_exit" (battery drives the computation DEPTH; requires a
            # model with aux heads, e.g. model: resnet8_ee).
            "exit_mode": "group",
            # beta_budget (β): max fraction of the REMAINING battery a client
            # may spend on one round. The client picks the deepest exit whose
            # estimated round energy fits β × battery (β-rule). Smaller β →
            # earlier throttling to shallow exits → longer lifetime.
            "beta_budget": 0.05,
            # aux_loss_weight: weight of the auxiliary heads' CE in the warmup
            # joint loss (CE_final + aux_loss_weight × Σ CE_aux).
            "aux_loss_weight": 0.3,
            # ee_server_lr: scale applied to the per-key weighted-mean delta.
            # 1.0 is correct (each key receives exactly one mean delta).
            "ee_server_lr": 1.0,
            # ── Early-exit V2a (all default-off = pre-V2a behaviour) ─────────
            # beta_mode: "fraction" (budget = β×battery; greedy-deep — spends
            # like FedAvg while battery is abundant) or "deadline" (budget =
            # battery / rounds_remaining; sustainable spend over the horizon).
            "beta_mode": "fraction",
            "deadline_rounds": 200,
            # repr_anchor: "logits" (legacy MSE on head logits) or "boundary"
            # (MSE on the GAP-pooled feature at the exit boundary — the
            # interface deeper blocks actually consume).
            "repr_anchor": "logits",
            # mu_adaptive: μ_eff = mu_repr × (1 + mu_adapt_scale × div) where
            # div = ‖F_loc − F_glob‖/‖F_glob‖ (boundary anchor only).
            "mu_adaptive": False,
            "mu_adapt_scale": 1.0,
            # var_aware_agg: per-client aggregation weight n_i·exp(−τ·Var_i),
            # Var_i = CV² of per-batch CE losses (see update_variance).
            "var_aware_agg": False,
            "var_agg_tau": 4.0,
            # ── Hybrid width (V3 / V3.1) ─────────────────────────────────────
            # ee_width_mode: "all" (whole prefix, legacy EE) | "fixed" (V3:
            #   ee_train_groups rotated groups) | "adaptive" (V3.1: exempt deep
            #   groups always + greedily fill the budget with rotated shallow).
            #   Unset → derived from ee_train_groups (0→all, ≥1→fixed).
            "ee_width_mode": None,
            # ee_train_groups: rotated-group count for "fixed" (0 = whole
            # prefix). Backward+upload cost = frac·(1/3 + 2/3·Σφ_g^prefix):
            # equals FedPart at full depth+width, strictly cheaper otherwise.
            "ee_train_groups": 0,
            # ee_exempt_deep: number of output-side prefix groups ALWAYS
            # trained (never rationed) — prevents deep/head starvation, the
            # failure mode of naive width-1 hybrid. Used by "adaptive"/"fixed".
            "ee_exempt_deep": 0,
            # ee_max_width: cap on the number of rotated shallow groups the
            # "adaptive" mode may add beyond the exempt set (None = uncapped).
            # Bounds backward + uplink cost even under a generous budget —
            # the knob that closes the energy/comm gap vs FedPart.
            "ee_max_width": None,
            # ── depth_mode: dynamic (FedStep-EE) vs static (DepthFL baseline) ─
            # "static" freezes depth by the client's initial-battery tier and
            # never adapts to depletion (the decisive dynamic-vs-static
            # control). static_batt_{min,max}_j define the fleet battery range
            # used to map battery → capacity tier → fixed depth.
            "depth_mode": "dynamic",
            "static_batt_min_j": 0.0,
            "static_batt_max_j": 0.0,
            # ── V3.2: deep→shallow self-distillation (default-off) ───────────
            # ee_self_distill: weight λ of the KD loss. At depth k≥2 the
            # deepest on-path exit (detached teacher) teaches the shallower
            # exits via temperature-softened KL. Complements ee_exempt_deep:
            # exemption guarantees deep-block COVERAGE; KD propagates deep
            # SEMANTICS into the shallow heads/trunk (DepthFL-style, but
            # restricted to the client's truncated prefix — no full forward).
            "ee_self_distill": 0.0,
            # ee_distill_temp: softmax temperature T (KL scaled by T²).
            "ee_distill_temp": 3.0,
            # bn_agg_mode: "standard" (legacy: BN buffers go through the same
            # server_lr / variance-weighting / EMA as the weights) or "plain"
            # (BN buffers aggregated as a pure data-weighted mean, excluded
            # from server_lr, variance-weighting and EMA). BN running stats
            # are statistics, not gradients; under severe non-IID the legacy
            # path destabilises training.
            "bn_agg_mode": "standard",
            # uplink_quant_bits: 0 = off (dense float32) | 8 = int8 (~4×) |
            # 4 = int4 (~8×, 2 values/byte) delta quantization with stochastic
            # rounding + error feedback (residual in state.custom["ef_residual"]).
            "uplink_quant_bits": 0,
            # ── FedRS restricted softmax (default-off) ───────────────────────
            # fedrs_alpha: scale applied, in the LOCAL CE only, to the logits
            # of classes ABSENT from the client's data (Li & Zhan, KDD 2021).
            # 1.0 = off; 0.5 = paper-typical. Targets the label-skew head
            # bias (a 1-2-class client destroying the classes it never sees);
            # complements the exemption, whose always-trained heads are the
            # most exposed. Inference/deltas/anchors untouched.
            "fedrs_alpha": 1.0,
            # lr_schedule: "constant" | "cosine" — anneal local lr over the
            # server horizon (lr_min_ratio floor). See client_update (EE path).
            "lr_schedule": "constant",
            "lr_min_ratio": 0.1,
            # ee_aux_ce: w>0 — joint CE beyond warmup at every on-path shallow
            # head (w·CE(true labels), depth≥2 clients). Cures head starvation
            # when the β-rule assigns nobody to a depth; heads are unfrozen
            # and transmitted (plumbing shared with the repaired
            # ee_self_distill). 0 = off.
            "ee_aux_ce": 0.0,
            # ee_global_kd: λg>0 — GLOBAL-teacher per-exit distillation
            # (FedGKD-style): on anchor epochs each on-path head (final
            # included) is distilled toward the GLOBAL model's same head,
            # computed for free during the anchor weight swap. The teacher is
            # identical across clients and has seen all classes — the
            # bias-free alternative after aux-CE/local-KD both failed under
            # severe label skew. 0 = off.
            "ee_global_kd": 0.0,
            # ee_gkd_ntd: True — FedNTD masking on the global-KD term: the
            # true-class logit is removed from student AND teacher softmax, so
            # the KL only preserves not-true-class structure (CE keeps the
            # true class). Counters the head-capping observed with a mediocre
            # global teacher under severe skew. Only meaningful with
            # ee_global_kd>0.
            "ee_gkd_ntd": False,
        }
