"""Robust spatial and temporal references for decision-margin profiles."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import Tensor

from .contracts import TemporalMarginReference


def _weighted_median(values: Tensor, weights: Tensor) -> Tensor:
    if values.ndim != 1 or weights.shape != values.shape or values.numel() == 0:
        raise ValueError("values and weights must be aligned non-empty vectors")
    if bool((weights < 0).any()) or float(weights.sum()) <= 0:
        raise ValueError("weights must be non-negative with positive sum")
    order = torch.argsort(values)
    sorted_values = values[order]
    cumulative = torch.cumsum(weights[order], dim=0)
    index = int(torch.searchsorted(cumulative, weights.sum() / 2, right=False))
    return sorted_values[min(index, sorted_values.numel() - 1)]


def robust_margin_reference(
    profiles: Tensor,
    *,
    method: Literal["median", "trimmed_mean"] = "median",
    trim_fraction: float = 0.1,
    min_clients: int = 2,
) -> tuple[Tensor, Tensor]:
    """Return a class-wise robust reference and its raw client support."""

    if profiles.ndim != 2:
        raise ValueError("profiles must have shape [clients, classes]")
    if not 0.0 <= trim_fraction < 0.5 or min_clients <= 0:
        raise ValueError("invalid robust-reference configuration")
    reference = torch.full(
        (profiles.shape[1],), torch.nan, dtype=profiles.dtype, device=profiles.device
    )
    support = torch.zeros(profiles.shape[1], dtype=torch.long, device=profiles.device)
    for class_id in range(profiles.shape[1]):
        values = profiles[:, class_id]
        values = values[torch.isfinite(values)]
        support[class_id] = values.numel()
        if values.numel() < min_clients:
            continue
        if method == "median":
            reference[class_id] = values.median()
        elif method == "trimmed_mean":
            values = values.sort().values
            trim = int(np.floor(trim_fraction * values.numel()))
            reference[class_id] = values[trim:-trim].mean() if trim else values.mean()
        else:
            raise ValueError(f"unknown robust reference method: {method}")
    return reference, support


def temporal_robust_margin_reference(
    latest_profiles: Tensor,
    observed_rounds: Tensor,
    *,
    current_round: int,
    window: int,
    decay_gamma: float,
    min_effective_clients: float,
    target_effective_clients: float,
    publication_support: Literal["effective", "raw"] = "effective",
) -> TemporalMarginReference:
    """Build a recency-weighted median from the latest client observations."""

    if latest_profiles.ndim != 2 or observed_rounds.shape != latest_profiles.shape:
        raise ValueError("profile values and rounds must be aligned matrices")
    if current_round <= 0 or window <= 0 or decay_gamma < 0:
        raise ValueError("invalid temporal-reference configuration")
    if min_effective_clients <= 0 or target_effective_clients < min_effective_clients:
        raise ValueError("invalid effective-support thresholds")
    if publication_support not in {"effective", "raw"}:
        raise ValueError("publication_support must be effective or raw")

    profiles = latest_profiles.detach().to(dtype=torch.float64, device="cpu")
    rounds = observed_rounds.detach().to(dtype=torch.long, device="cpu")
    classes = profiles.shape[1]
    values = torch.full((classes,), torch.nan, dtype=torch.float64)
    scale = torch.full((classes,), torch.nan, dtype=torch.float64)
    raw_support = torch.zeros(classes, dtype=torch.long)
    effective_support = torch.zeros(classes, dtype=torch.float64)
    reliability = torch.zeros(classes, dtype=torch.float64)
    mean_age = torch.full((classes,), torch.nan, dtype=torch.float64)
    max_age = torch.full((classes,), torch.nan, dtype=torch.float64)
    ages = (current_round - 1) - rounds
    fresh = (
        torch.isfinite(profiles)
        & (rounds > 0)
        & (rounds < current_round)
        & (ages >= 0)
        & (ages < window)
    )
    for class_id in range(classes):
        valid = fresh[:, class_id]
        count = int(valid.sum())
        raw_support[class_id] = count
        if count == 0:
            continue
        class_values = profiles[valid, class_id]
        class_ages = ages[valid, class_id].to(torch.float64)
        weights = torch.exp(-decay_gamma * class_ages)
        weight_sum = weights.sum()
        n_eff = weight_sum.square() / weights.square().sum()
        effective_support[class_id] = n_eff
        mean_age[class_id] = torch.sum(weights * class_ages) / weight_sum
        max_age[class_id] = class_ages.max()
        support = float(n_eff) if publication_support == "effective" else count
        if support + 1e-12 < min_effective_clients:
            continue
        values[class_id] = _weighted_median(class_values, weights)
        mad = _weighted_median(torch.abs(class_values - values[class_id]), weights)
        scale[class_id] = 1.4826 * mad
        reliability[class_id] = min(1.0, float(n_eff) / target_effective_clients)
    dtype = latest_profiles.dtype
    return TemporalMarginReference(
        values=values.to(dtype=dtype),
        scale=scale.to(dtype=dtype),
        raw_support=raw_support,
        effective_support=effective_support.to(dtype=dtype),
        reliability=reliability.to(dtype=dtype),
        mean_age=mean_age.to(dtype=dtype),
        max_age=max_age.to(dtype=dtype),
    )


def temporal_pooled_location_scale(
    latest_profiles: Tensor,
    observed_rounds: Tensor,
    *,
    current_round: int,
    window: int,
    decay_gamma: float,
) -> tuple[Tensor, Tensor]:
    """Pool fresh client-class cells; intended only as a semantic control."""

    if latest_profiles.ndim != 2 or observed_rounds.shape != latest_profiles.shape:
        raise ValueError("profile values and rounds must be aligned matrices")
    if current_round <= 0 or window <= 0 or decay_gamma < 0:
        raise ValueError("invalid temporal pooling configuration")
    profiles = latest_profiles.detach().to(dtype=torch.float64, device="cpu")
    rounds = observed_rounds.detach().to(dtype=torch.long, device="cpu")
    ages = (current_round - 1) - rounds
    fresh = (
        torch.isfinite(profiles)
        & (rounds > 0)
        & (rounds < current_round)
        & (ages >= 0)
        & (ages < window)
    )
    if not bool(fresh.any()):
        nan = torch.tensor(float("nan"), dtype=latest_profiles.dtype)
        return nan, nan.clone()
    values = profiles[fresh]
    weights = torch.exp(-decay_gamma * ages[fresh].to(torch.float64))
    location = _weighted_median(values, weights)
    mad = _weighted_median(torch.abs(values - location), weights)
    return location.to(latest_profiles.dtype), (1.4826 * mad).to(latest_profiles.dtype)


def cyclically_permute_class_reference(
    reference: Tensor, reliability: Tensor, *, shift: int
) -> tuple[Tensor, Tensor]:
    """Destroy class identity while preserving the joint marginal values."""

    if reference.ndim != 1 or reference.shape != reliability.shape:
        raise ValueError("reference and reliability must be aligned vectors")
    normalized_shift = int(shift) % reference.numel()
    if reference.numel() < 2 or normalized_shift == 0:
        raise ValueError("shift must induce a non-identity class permutation")
    return torch.roll(reference, normalized_shift), torch.roll(
        reliability, normalized_shift
    )


def permute_class_reference(
    reference: Tensor, reliability: Tensor, permutation: Tensor
) -> tuple[Tensor, Tensor]:
    """Apply a class derangement to reference and reliability."""

    if reference.ndim != 1 or reference.shape != reliability.shape:
        raise ValueError("reference and reliability must be aligned vectors")
    if permutation.ndim != 1 or permutation.numel() != reference.numel():
        raise ValueError("permutation must contain one index per class")
    permutation = permutation.to(reference.device, torch.long)
    expected = torch.arange(reference.numel(), device=reference.device)
    if not torch.equal(permutation.sort().values, expected) or bool(
        (permutation == expected).any()
    ):
        raise ValueError("permutation must be a derangement")
    return reference[permutation], reliability[permutation.to(reliability.device)]


__all__ = [
    "robust_margin_reference",
    "temporal_robust_margin_reference",
    "temporal_pooled_location_scale",
    "cyclically_permute_class_reference",
    "permute_class_reference",
]
