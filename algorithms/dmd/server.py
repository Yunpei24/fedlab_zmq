"""Server aggregation and construction of the next frozen DMD context."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch

from algorithms.base import AggregateResult

from .config import DMDConfig
from .contracts import DMDClientReport, DMDRoundContext
from .metrics import class_reference_reliability
from .objectives import quadratic_margin_deficit
from .profiles import profile_from_wire
from .references import robust_margin_reference
from .tail_risk import weighted_upper_cvar


def _reports(client_updates) -> list[DMDClientReport]:
    reports: list[DMDClientReport] = []
    for _, metadata, _ in client_updates:
        value = metadata.get("dmd_client_report")
        if value is not None:
            reports.append(DMDClientReport.from_wire(value))
    return reports


def _dual_ascent_eta(
    deficits: torch.Tensor,
    weights: torch.Tensor,
    previous_eta: float | None,
    *,
    tail_mass: float,
    step: float,
    mean_deficit: float,
    empirical_eta: float,
) -> tuple[float, float]:
    """Move eta one Rockafellar-Uryasev dual step; return (eta, tail fraction).

    Minimising ``g(eta) = eta + E[D - eta]_+ / b`` over eta gives
    ``g'(eta) = 1 - P(D > eta) / b``, so a descent step is
    ``eta <- eta + lr * (P(D > eta) / b - 1)``: eta rises while more than a
    fraction ``b`` of the cohort sits above it and falls otherwise, converging
    on the ``1 - b`` quantile.  The step is scaled by the cohort mean deficit to
    stay dimensionless, and eta is floored at zero because deficits are.
    """

    above = (deficits > float(previous_eta if previous_eta is not None else 0.0)).to(
        weights.dtype
    )
    tail_fraction = float(torch.sum(weights * above))
    if previous_eta is None:
        # Nothing to continue from: seed on the exact order statistic, which is
        # the best single-round estimate available, then dual-ascend afterwards.
        return empirical_eta, tail_fraction
    scale = mean_deficit if mean_deficit > 0 else 1.0
    updated = previous_eta + step * scale * (tail_fraction / tail_mass - 1.0)
    # Clamp into the cohort support.  Outside [0, max D] the dual gradient
    # saturates at a constant (no client is above eta, or every client is), so
    # eta just ramps with no information -- which is how a large warmup deficit
    # can park the threshold far above every later deficit for dozens of rounds.
    # The RU minimiser always lies inside the support, so clamping cannot move
    # the fixed point; it only removes the uninformative excursion.
    ceiling = float(deficits.max())
    return float(min(max(0.0, updated), ceiling)), tail_fraction


def build_next_round_context(
    reports: list[DMDClientReport],
    *,
    source_round: int,
    config: DMDConfig,
    previous_eta: float | None = None,
) -> tuple[DMDRoundContext | None, dict[str, Any]]:
    """Build the one-round-stale reference and cohort tail audit state."""

    if not reports:
        return None, {"dmd_reference_published_classes": 0}
    profiles = torch.stack(
        [profile_from_wire(report.margins, report.counts).values for report in reports]
    )
    reference, support = robust_margin_reference(
        profiles,
        method=config.reference_method,
        trim_fraction=config.trim_fraction,
        min_clients=config.min_reference_clients,
    )
    if config.reference_mode == "fixed_zero":
        # A fixed scalar reference for every class.  ``margin_target`` defaults
        # to 0.0, which is the published "reference zero" behaviour: penalise a
        # class only once it is on the wrong side of the boundary.  A positive
        # target makes the criterion satisficing -- clear the boundary by this
        # much, then stop -- and is only meaningful in a bounded margin space.
        reference = torch.where(
            torch.isfinite(reference),
            torch.full_like(reference, config.margin_target),
            reference,
        )
    reliability = class_reference_reliability(
        support,
        num_clients=len(reports),
        min_clients=min(config.min_reference_clients, len(reports)),
    ).to(torch.float32)
    deficits = torch.stack(
        [
            quadratic_margin_deficit(
                profile_from_wire(report.margins, report.counts),
                reference,
                class_weight_mode=config.class_weight_mode,
            )
            for report in reports
        ]
    ).to(torch.float64)
    sizes = torch.tensor(
        [report.dataset_size for report in reports], dtype=torch.float64
    )
    weights = sizes / sizes.sum().clamp_min(1)
    mean_deficit = float(torch.sum(weights * deficits))
    tail = weighted_upper_cvar(deficits, weights, tail_mass=config.cvar_tail_mass)
    eta = float(tail.eta)
    eta_tail_fraction = float(torch.sum(weights * (deficits > eta).to(weights.dtype)))
    if config.cvar_eta_mode == "dual":
        eta, eta_tail_fraction = _dual_ascent_eta(
            deficits,
            weights,
            previous_eta,
            tail_mass=config.cvar_tail_mass,
            step=config.cvar_eta_lr,
            mean_deficit=mean_deficit,
            empirical_eta=eta,
        )
    context = DMDRoundContext(
        source_round=source_round,
        variant=config.variant,
        reference=tuple(
            float(value) if bool(torch.isfinite(value)) else None for value in reference
        ),
        reliability=tuple(float(value) for value in reliability),
        cohort_mean_deficit=mean_deficit,
        cvar_eta=eta,
        cvar_tail_mass=config.cvar_tail_mass,
    )
    audit = {
        "dmd_cvar_eta_mode": config.cvar_eta_mode,
        "dmd_cvar_eta_empirical": float(tail.eta),
        # Fraction of the cohort strictly above the published eta.  At the RU
        # optimum this tracks cvar_tail_mass; a value pinned at 0 means the
        # threshold is inert (the empirical order statistic collapsed onto the
        # cohort maximum), which is exactly what the dual mode is there to fix.
        "dmd_cvar_tail_fraction_above_eta": eta_tail_fraction,
        "dmd_reference_published_classes": int(torch.isfinite(reference).sum()),
        "dmd_reference_support": [int(value) for value in support],
        "dmd_reference_reliability": [float(value) for value in reliability],
        "dmd_deficit_mean": mean_deficit,
        "dmd_deficit_cvar": float(tail.cvar),
        # The PUBLISHED threshold, i.e. the one clients actually receive.  This
        # used to log ``tail.eta`` unconditionally, which silently reported the
        # empirical order statistic even when a different eta was published.
        "dmd_cvar_eta": eta,
        "dmd_tail_fraction": [float(value) for value in tail.tail_fraction],
        "dmd_tail_weights": [float(value) for value in tail.tail_weights],
    }
    return context, audit


def server_aggregate(
    global_model,
    client_updates,
    round_num: int,
    config: dict[str, Any],
    *,
    variant: str,
) -> AggregateResult:
    """Dataset-size-weighted FedAvg plus a next-round DMD control context."""

    if not client_updates:
        raise ValueError("DMD aggregation requires at least one client update")
    cfg = DMDConfig.from_mapping(config, variant=variant)
    global_state = global_model.state_dict()
    sizes = torch.tensor(
        [max(0, int(meta.get("dataset_size", 1))) for _, meta, _ in client_updates],
        dtype=torch.float64,
    )
    if float(sizes.sum()) <= 0:
        sizes.fill_(1)
    weights = sizes / sizes.sum()
    aggregated: dict[str, torch.Tensor] = {}
    for key in global_state:
        accumulator = torch.zeros_like(
            global_state[key], dtype=torch.float32, device="cpu"
        )
        for weight, (update, _, _) in zip(weights, client_updates):
            accumulator.add_(update[key].detach().cpu().float(), alpha=float(weight))
        aggregated[key] = accumulator
    new_weights = OrderedDict()
    for key, global_value in global_state.items():
        updated = global_value.detach().cpu().float() - aggregated[key]
        new_weights[key] = updated.to(
            dtype=global_value.dtype, device=global_value.device
        )

    # ``config`` carries the server state, so ``dmd_round_context`` here is the
    # context published one round ago: its eta is what the dual step continues.
    previous_context = config.get("dmd_round_context")
    if isinstance(previous_context, dict):
        previous_context = DMDRoundContext.from_wire(previous_context)
    previous_eta = (
        float(previous_context.cvar_eta)
        if isinstance(previous_context, DMDRoundContext)
        else None
    )
    context, audit = build_next_round_context(
        _reports(client_updates),
        source_round=round_num,
        config=cfg,
        previous_eta=previous_eta,
    )
    states = [state for _, _, state in client_updates]
    participations = [1.0 if state.battery_j > 0 else 0.0 for state in states]
    count = len(client_updates)
    jain = (
        sum(participations) ** 2
        / (count * sum(value * value for value in participations))
        if any(participations)
        else 0.0
    )
    metrics: dict[str, Any] = {
        "round": round_num,
        "total_bytes_sent": sum(meta["bytes_sent"] for _, meta, _ in client_updates),
        "total_energy_j": sum(
            meta["energy_j_consumed"] for _, meta, _ in client_updates
        ),
        "avg_beta": 1.0,
        "avg_battery_j": sum(state.battery_j for state in states) / count,
        "avg_local_loss": sum(meta["local_loss"] for _, meta, _ in client_updates)
        / count,
        "avg_local_ce": sum(meta.get("local_ce", 0.0) for _, meta, _ in client_updates)
        / count,
        "avg_local_dmd_addend": sum(
            meta.get("local_dmd_addend", 0.0) for _, meta, _ in client_updates
        )
        / count,
        "dmd_context_clients": sum(
            bool(meta.get("dmd_context_applied", False))
            for _, meta, _ in client_updates
        ),
        "participation_rate": sum(participations) / count,
        "jain_index": jain,
        "num_clients": count,
        **audit,
    }
    if context is not None:
        metrics["dmd_next_round_context"] = context.to_wire()
        metrics["_server_state_updates"] = {
            "dmd_round_context": context.to_wire()
        }
    return AggregateResult(new_weights=new_weights, metrics=metrics)


__all__ = ["build_next_round_context", "server_aggregate"]
