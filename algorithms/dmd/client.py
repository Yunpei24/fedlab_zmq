"""Client-side local optimization and auditable DMD profile reporting."""

from __future__ import annotations

import gc
from collections import OrderedDict
from typing import Any

import torch
from torch import nn, optim

from hardware.flop_cost import round_compute_flops

from .config import DMDConfig
from .contracts import DMDClientReport, DMDRoundContext
from .objectives import deficit_distribution_objective, example_quadratic_dmd_loss
from .profiles import class_margin_profile, profile_to_wire


def parse_round_context(value: Any) -> DMDRoundContext | None:
    if value is None:
        return None
    if isinstance(value, DMDRoundContext):
        return value
    if isinstance(value, dict):
        return DMDRoundContext.from_wire(value)
    raise TypeError("dmd_round_context must be a mapping or DMDRoundContext")


def evaluate_margin_report(
    model: nn.Module,
    dataloader,
    *,
    device: str,
    num_classes: int,
    min_count: int,
    client_id: int,
    round_num: int,
    dataset_size: int,
    reference: torch.Tensor | None,
    class_weight_mode: str,
) -> DMDClientReport:
    """Evaluate the updated global/local model without changing its mode."""

    was_training = model.training
    model.eval()
    logits_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    with torch.no_grad():
        for features, targets in dataloader:
            logits_all.append(model(features.to(device)).detach().cpu())
            targets_all.append(targets.detach().cpu())
    if was_training:
        model.train()
    if not logits_all:
        raise ValueError("DMD profile dataloader must not be empty")
    profile = class_margin_profile(
        torch.cat(logits_all),
        torch.cat(targets_all),
        num_classes,
        min_count=min_count,
    )
    values, counts = profile_to_wire(profile)
    deficit = None
    if reference is not None:
        from .objectives import quadratic_margin_deficit

        deficit = float(
            quadratic_margin_deficit(
                profile, reference.detach().cpu(), class_weight_mode=class_weight_mode
            )
        )
    return DMDClientReport(
        client_id=client_id,
        round_num=round_num,
        margins=tuple(values),
        counts=tuple(counts),
        dataset_size=dataset_size,
        deficit=deficit,
    )


def client_update(
    algorithm,
    model: nn.Module,
    dataloader,
    state,
    config: dict[str, Any],
    *,
    variant: str,
) -> tuple[dict, dict]:
    """Train one DMD client and return dense delta plus scalar metadata."""

    cfg = DMDConfig.from_mapping(config, variant=variant)
    context = parse_round_context(config.get("dmd_round_context"))
    if context is not None and context.variant != variant:
        raise ValueError("DMD context variant does not match algorithm adapter")
    device = cfg.device
    w_before = OrderedDict(
        (key, value.detach().cpu().clone()) for key, value in model.state_dict().items()
    )
    model.to(device)
    model.train()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    reference = None
    reliability = None
    if context is not None:
        reference = context.reference_tensor(device=device, dtype=torch.float32)
        reliability = context.reliability_tensor(device=device, dtype=torch.float32)

    total_loss = 0.0
    total_ce = 0.0
    total_fair = 0.0
    num_batches = 0
    for _ in range(cfg.local_epochs):
        for features, targets in dataloader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(features)
            ce = criterion(logits, targets)
            fairness = ce * 0.0
            if reference is not None and reliability is not None:
                class_weights = (
                    None if cfg.reference_mode == "fixed_zero" else reliability
                )
                normalization_weights = (
                    None if class_weights is None else torch.ones_like(reliability)
                )
                deficit = example_quadratic_dmd_loss(
                    logits,
                    targets,
                    reference,
                    class_weights=class_weights,
                    normalization_class_weights=normalization_weights,
                )
                threshold = (
                    context.cvar_eta
                    if variant == "cvar"
                    else context.cohort_mean_deficit
                )
                fairness, _, _ = deficit_distribution_objective(
                    deficit,
                    threshold,
                    mean_mu=cfg.mean_mu,
                    dispersion_mu=cfg.dispersion_mu,
                    mode=variant,
                    cvar_tail_mass=cfg.cvar_tail_mass,
                )
            loss = ce + fairness
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            total_ce += float(ce.detach())
            total_fair += float(fairness.detach())
            num_batches += 1

    current = model.state_dict()
    delta = OrderedDict(
        (key, (w_before[key] - current[key].detach().cpu()).float()) for key in w_before
    )
    uplink_bytes = algorithm.count_bytes(delta, sparse=False)
    downlink_bytes = uplink_bytes
    profile = config.get("device_profile")
    if profile:
        trainable_names = [name for name, _ in model.named_parameters()]
        flops = round_compute_flops(
            model,
            trainable_names,
            config,
            profile,
            dataloader,
            cfg.local_epochs,
        )
        breakdown = profile.round_energy_breakdown(
            flops,
            uplink_bytes,
            downlink_bytes,
            config.get("energy_scale_factor", 1.0),
            config.get("alpha_applies_to", "compute"),
        )
    else:
        fallback = 2.5 * config.get("energy_scale_factor", 1.0)
        breakdown = {
            "compute": fallback,
            "uplink": 0.0,
            "downlink": 0.0,
            "total": fallback,
        }
    energy_j = float(breakdown["total"])
    state.battery_j = max(0.0, state.battery_j - energy_j)
    state.round_num += 1
    dataset_size = len(dataloader.dataset) if hasattr(dataloader, "dataset") else 1
    anchor = config.get("anchor_dataloader") or dataloader
    report = evaluate_margin_report(
        model,
        anchor,
        device=device,
        num_classes=cfg.num_classes,
        min_count=cfg.min_profile_count,
        client_id=state.client_id,
        round_num=state.round_num,
        dataset_size=dataset_size,
        reference=reference,
        class_weight_mode=cfg.class_weight_mode,
    )
    metadata = {
        "client_id": state.client_id,
        "round_num": state.round_num,
        "beta_actual": 1.0,
        "battery_j_remaining": state.battery_j,
        "energy_j_consumed": energy_j,
        "energy_compute_j": float(breakdown["compute"]),
        "energy_uplink_j": float(breakdown["uplink"]),
        "energy_downlink_j": float(breakdown["downlink"]),
        "bytes_sent": uplink_bytes,
        "bytes_received": downlink_bytes,
        "local_loss": total_loss / max(num_batches, 1),
        "local_ce": total_ce / max(num_batches, 1),
        "local_dmd_addend": total_fair / max(num_batches, 1),
        "compression_ratio": 1.0,
        "dataset_size": dataset_size,
        "dmd_context_applied": context is not None,
        "dmd_client_report": report.to_wire(),
    }
    del optimizer, current, w_before
    gc.collect()
    return dict(delta), metadata


__all__ = ["parse_round_context", "evaluate_margin_report", "client_update"]
