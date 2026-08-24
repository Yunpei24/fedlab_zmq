"""Differentiable decision-margin deficit primitives."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from ..contracts import MarginProfile, ObjectiveTerms
from ..profiles import true_class_margin


def normalized_class_weights(
    counts: Tensor,
    valid: Tensor,
    *,
    mode: Literal["uniform", "frequency"] = "uniform",
) -> Tensor:
    if counts.ndim != 1 or valid.shape != counts.shape:
        raise ValueError("counts and valid must be aligned vectors")
    if mode == "uniform":
        raw = valid.to(torch.float32)
    elif mode == "frequency":
        raw = counts.to(torch.float32) * valid.to(torch.float32)
    else:
        raise ValueError(f"unknown class-weight mode: {mode}")
    return raw / raw.sum() if float(raw.sum()) > 0 else torch.zeros_like(raw)


def quadratic_margin_deficit(
    profile: MarginProfile,
    reference: Tensor,
    *,
    class_weight_mode: Literal["uniform", "frequency"] = "uniform",
    class_reliability: Tensor | None = None,
    reliability_power: float = 1.0,
) -> Tensor:
    """Return a one-sided quadratic deficit over the observed classes.

    ``class_reliability`` directly attenuates uncertain class contributions;
    it is deliberately not renormalized away.  Multiplying all reliability
    values by one half therefore halves the complete deficit.
    """

    if reference.ndim != 1 or reference.shape != profile.values.shape:
        raise ValueError("reference and profile must have the same shape")
    if reliability_power < 0:
        raise ValueError("reliability_power must be non-negative")
    if class_reliability is not None:
        if class_reliability.shape != reference.shape:
            raise ValueError("class_reliability must align with reference")
        if not bool(torch.isfinite(class_reliability).all()) or bool(
            (class_reliability < 0).any()
        ):
            raise ValueError("class_reliability must be finite and non-negative")
    valid = profile.observed & torch.isfinite(reference)
    weights = normalized_class_weights(profile.counts, valid, mode=class_weight_mode)
    weights = weights.to(profile.values.device, profile.values.dtype)
    if class_reliability is not None:
        weights = weights * class_reliability.to(
            device=profile.values.device,
            dtype=profile.values.dtype,
        ).pow(reliability_power)
    profile_values = torch.where(
        valid, profile.values, torch.zeros_like(profile.values)
    )
    reference_values = torch.where(valid, reference, torch.zeros_like(reference))
    return 0.5 * torch.sum(
        weights * torch.relu(reference_values - profile_values).square()
    )


def linear_margin_deficit(
    profile: MarginProfile,
    reference: Tensor,
    *,
    class_weight_mode: Literal["uniform", "frequency"] = "uniform",
) -> Tensor:
    """Return the one-sided linear deficit ``sum_c rho_c [r_c-m_ic]_+``."""

    if reference.ndim != 1 or reference.shape != profile.values.shape:
        raise ValueError("reference and profile must have the same shape")
    valid = profile.observed & torch.isfinite(reference)
    weights = normalized_class_weights(profile.counts, valid, mode=class_weight_mode)
    weights = weights.to(profile.values.device, profile.values.dtype)
    profile_values = torch.where(
        valid, profile.values, torch.zeros_like(profile.values)
    )
    reference_values = torch.where(valid, reference, torch.zeros_like(reference))
    return torch.sum(weights * torch.relu(reference_values - profile_values))


def class_balanced_example_margin_deficit(
    margins: Tensor,
    targets: Tensor,
    reference: Tensor,
    *,
    class_reliability: Tensor | None = None,
    reliability_power: float = 1.0,
) -> Tensor:
    """Average quadratic sample deficits uniformly over observed classes.

    This is the exact full-dataset quantity estimated by inverse-frequency
    sample weights during local minibatch optimization.
    """

    if margins.ndim != 1 or targets.shape != margins.shape:
        raise ValueError("margins and targets must be aligned vectors")
    if reference.ndim != 1 or reference.numel() <= 1:
        raise ValueError("reference must contain one value per class")
    if reliability_power < 0:
        raise ValueError("reliability_power must be non-negative")
    targets = targets.to(device=margins.device, dtype=torch.long)
    if bool((targets < 0).any()) or bool((targets >= reference.numel()).any()):
        raise ValueError("targets fall outside the reference classes")
    if class_reliability is None:
        reliability = torch.ones_like(reference, dtype=margins.dtype)
    else:
        if class_reliability.shape != reference.shape:
            raise ValueError("class_reliability must align with reference")
        if not bool(torch.isfinite(class_reliability).all()) or bool(
            (class_reliability < 0).any()
        ):
            raise ValueError("class_reliability must be finite and non-negative")
        reliability = class_reliability.to(margins.device, margins.dtype)
    local_reference = reference.to(margins.device, margins.dtype)[targets]
    valid_examples = torch.isfinite(local_reference)
    class_terms: list[Tensor] = []
    for class_id in torch.unique(targets[valid_examples], sorted=True):
        selected = valid_examples & (targets == class_id)
        penalty = 0.5 * torch.relu(
            local_reference[selected] - margins[selected]
        ).square()
        class_terms.append(
            reliability[class_id].pow(reliability_power) * penalty.mean()
        )
    if not class_terms:
        return margins.sum() * 0.0
    return torch.stack(class_terms).mean()


def _weighted_penalty(
    penalties: Tensor,
    targets: Tensor,
    *,
    class_weights: Tensor | None,
    normalization_class_weights: Tensor | None,
    num_classes: int,
    logits: Tensor,
) -> Tensor:
    if class_weights is None:
        return penalties.mean()
    if class_weights.ndim != 1 or class_weights.shape[0] != num_classes:
        raise ValueError("class_weights must contain one value per class")
    weights = class_weights.to(logits.device, logits.dtype)[targets]
    denominator = weights
    if normalization_class_weights is not None:
        if normalization_class_weights.shape != class_weights.shape:
            raise ValueError("normalization_class_weights must align")
        denominator = normalization_class_weights.to(logits.device, logits.dtype)[
            targets
        ]
    if float(weights.sum()) <= 0 or float(denominator.sum()) <= 0:
        return logits.sum() * 0.0
    return torch.sum(weights * penalties) / denominator.sum()


def example_quadratic_dmd_loss(
    logits: Tensor,
    targets: Tensor,
    reference: Tensor,
    *,
    class_weights: Tensor | None = None,
    normalization_class_weights: Tensor | None = None,
) -> Tensor:
    """Differentiable per-example quadratic DMD penalty."""

    if reference.ndim != 1 or reference.shape[0] != logits.shape[1]:
        raise ValueError("reference must contain one value per class")
    targets = targets.to(logits.device, torch.long)
    margins = true_class_margin(logits, targets)
    local_reference = reference.to(logits.device, logits.dtype)[targets]
    valid = torch.isfinite(local_reference)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return _weighted_penalty(
        0.5 * torch.relu(local_reference[valid] - margins[valid]).square(),
        targets[valid],
        class_weights=class_weights,
        normalization_class_weights=normalization_class_weights,
        num_classes=logits.shape[1],
        logits=logits,
    )


def example_quadratic_standardized_dmd_loss(
    logits: Tensor,
    targets: Tensor,
    reference: Tensor,
    reference_scale: Tensor,
    *,
    class_weights: Tensor | None = None,
    normalization_class_weights: Tensor | None = None,
) -> Tensor:
    if reference.ndim != 1 or reference.shape[0] != logits.shape[1]:
        raise ValueError("reference must contain one value per class")
    if reference_scale.shape != reference.shape:
        raise ValueError("reference_scale must align with reference")
    targets = targets.to(logits.device, torch.long)
    margins = true_class_margin(logits, targets)
    local_reference = reference.to(logits.device, logits.dtype)[targets]
    local_scale = reference_scale.to(logits.device, logits.dtype)[targets]
    valid = (
        torch.isfinite(local_reference)
        & torch.isfinite(local_scale)
        & (local_scale > 0)
    )
    if not bool(valid.any()):
        return logits.sum() * 0.0
    penalties = (
        0.5
        * torch.relu(
            (local_reference[valid] - margins[valid]) / local_scale[valid]
        ).square()
    )
    return _weighted_penalty(
        penalties,
        targets[valid],
        class_weights=class_weights,
        normalization_class_weights=normalization_class_weights,
        num_classes=logits.shape[1],
        logits=logits,
    )


def example_linear_dmd_loss(
    logits: Tensor,
    targets: Tensor,
    reference: Tensor,
    *,
    class_weights: Tensor | None = None,
    normalization_class_weights: Tensor | None = None,
) -> Tensor:
    if reference.ndim != 1 or reference.shape[0] != logits.shape[1]:
        raise ValueError("reference must contain one value per class")
    targets = targets.to(logits.device, torch.long)
    margins = true_class_margin(logits, targets)
    local_reference = reference.to(logits.device, logits.dtype)[targets]
    valid = torch.isfinite(local_reference)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return _weighted_penalty(
        torch.relu(local_reference[valid] - margins[valid]),
        targets[valid],
        class_weights=class_weights,
        normalization_class_weights=normalization_class_weights,
        num_classes=logits.shape[1],
        logits=logits,
    )


def make_terms(mean_addend: Tensor, dispersion_addend: Tensor) -> ObjectiveTerms:
    return ObjectiveTerms(
        total=mean_addend + dispersion_addend,
        mean=mean_addend,
        dispersion=dispersion_addend,
    )


__all__ = [
    "normalized_class_weights",
    "quadratic_margin_deficit",
    "linear_margin_deficit",
    "class_balanced_example_margin_deficit",
    "example_quadratic_dmd_loss",
    "example_quadratic_standardized_dmd_loss",
    "example_linear_dmd_loss",
    "make_terms",
]
