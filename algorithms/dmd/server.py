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


def build_next_round_context(
    reports: list[DMDClientReport],
    *,
    source_round: int,
    config: DMDConfig,
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
        reference = torch.where(
            torch.isfinite(reference), torch.zeros_like(reference), reference
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
    context = DMDRoundContext(
        source_round=source_round,
        variant=config.variant,
        reference=tuple(
            float(value) if bool(torch.isfinite(value)) else None for value in reference
        ),
        reliability=tuple(float(value) for value in reliability),
        cohort_mean_deficit=mean_deficit,
        cvar_eta=float(tail.eta),
        cvar_tail_mass=config.cvar_tail_mass,
    )
    audit = {
        "dmd_reference_published_classes": int(torch.isfinite(reference).sum()),
        "dmd_reference_support": [int(value) for value in support],
        "dmd_reference_reliability": [float(value) for value in reliability],
        "dmd_deficit_mean": mean_deficit,
        "dmd_deficit_cvar": float(tail.cvar),
        "dmd_cvar_eta": float(tail.eta),
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

    context, audit = build_next_round_context(
        _reports(client_updates), source_round=round_num, config=cfg
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
