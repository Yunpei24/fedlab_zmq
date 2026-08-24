"""Frozen compatibility core for the historical DMD research prototype.

New code should import the public modules in :mod:`algorithms.dmd`.  This file
preserves the exact pre-modularization implementation for research helpers
that have not yet become part of the deployable algorithm contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class MarginProfile:
    """Per-class mean margins and their local support."""

    values: Tensor
    counts: Tensor

    @property
    def observed(self) -> Tensor:
        return torch.isfinite(self.values) & (self.counts > 0)


@dataclass(frozen=True)
class TemporalMarginReference:
    """Class-wise temporal reference and its audit statistics.

    ``raw_support`` counts distinct clients, not observations.  Each
    client--class pair contributes only its latest profile in the configured
    window.  ``effective_support`` accounts for unequal recency weights.
    Unsupported classes have a ``NaN`` reference and zero reliability.
    """

    values: Tensor
    scale: Tensor
    raw_support: Tensor
    effective_support: Tensor
    reliability: Tensor
    mean_age: Tensor
    max_age: Tensor


@dataclass(frozen=True)
class WeightedUpperCvar:
    """Exact finite-cohort upper-tail CVaR audit state.

    ``tail_fraction[i]`` is the fraction of client ``i``'s probability mass
    allocated to the worst ``tail_mass`` of the weighted empirical
    distribution. ``tail_weights`` are normalized to sum to one and can be
    used to reconstruct ``cvar`` exactly. A fractional boundary client is
    necessary whenever a client's weight straddles the requested tail mass.
    """

    eta: Tensor
    cvar: Tensor
    tail_fraction: Tensor
    tail_weights: Tensor


def weighted_upper_cvar(
    values: Tensor,
    weights: Tensor,
    *,
    tail_mass: float,
) -> WeightedUpperCvar:
    """Return the exact upper-tail CVaR of a finite weighted cohort.

    The function allocates exactly ``tail_mass`` probability to the largest
    values, splitting the probability mass of the boundary client when
    needed. The returned ``eta`` is the weighted VaR threshold and satisfies

    ``CVaR_q(D) = eta + q^{-1} sum_i p_i [D_i - eta]_+``.

    Ties at ``eta`` may admit multiple valid allocations; the stable ordering
    used here only affects the audit fractions, never ``eta`` or ``cvar``.
    """

    if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape:
        raise ValueError("values and weights must be aligned vectors")
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("values must be finite and non-empty")
    if not bool(torch.isfinite(weights).all()) or not bool((weights >= 0).all()):
        raise ValueError("weights must be finite and non-negative")
    if weights.sum() <= 0:
        raise ValueError("weights must have positive mass")
    if not 0.0 < tail_mass <= 1.0:
        raise ValueError("tail_mass must lie in (0, 1]")

    normalized_weights = weights.to(dtype=torch.float64) / weights.sum()
    values64 = values.to(dtype=torch.float64)
    order = torch.argsort(values64, descending=True, stable=True)
    allocated_mass = torch.zeros_like(normalized_weights)
    remaining = float(tail_mass)
    eta_index = int(order[-1])
    tolerance = 1e-12
    for index_tensor in order:
        index = int(index_tensor)
        if remaining <= tolerance:
            break
        take = min(float(normalized_weights[index]), remaining)
        allocated_mass[index] = take
        remaining -= take
        eta_index = index
    if remaining > 1e-9:
        raise RuntimeError("failed to allocate the requested CVaR tail mass")

    tail_fraction = torch.where(
        normalized_weights > 0,
        allocated_mass / normalized_weights,
        torch.zeros_like(normalized_weights),
    )
    tail_weights = allocated_mass / float(tail_mass)
    eta = values64[eta_index]
    cvar = torch.sum(tail_weights * values64)
    return WeightedUpperCvar(
        eta=eta.to(dtype=values.dtype, device=values.device),
        cvar=cvar.to(dtype=values.dtype, device=values.device),
        tail_fraction=tail_fraction.to(dtype=values.dtype, device=values.device),
        tail_weights=tail_weights.to(dtype=values.dtype, device=values.device),
    )


def _weighted_median(values: Tensor, weights: Tensor) -> Tensor:
    """Return the lower recency-weighted median of finite scalar values."""

    if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape:
        raise ValueError("values and weights must be aligned vectors")
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("weighted median requires finite non-empty values")
    if not bool((weights > 0).all()):
        raise ValueError("weighted median requires strictly positive weights")
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = 0.5 * sorted_weights.sum()
    median_index = int(
        torch.searchsorted(sorted_weights.cumsum(0), cutoff, right=False)
    )
    return sorted_values[min(median_index, values.numel() - 1)]


def true_class_margin(logits: Tensor, targets: Tensor) -> Tensor:
    """Return ``z_y - max_{k != y} z_k`` for every observation.

    ``logits`` are the pre-softmax scores emitted by the final linear layer.
    They are not ReLU activations and must have shape ``[batch, classes]``.
    """

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must have shape [batch]")
    if logits.shape[1] < 2:
        raise ValueError("at least two classes are required")

    targets = targets.to(device=logits.device, dtype=torch.long)
    true_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    competitors = logits.clone()
    competitors.scatter_(1, targets.unsqueeze(1), -torch.inf)
    best_competitor = competitors.max(dim=1).values
    return true_logits - best_competitor


def class_margin_profile(
    logits: Tensor,
    targets: Tensor,
    num_classes: int,
    *,
    min_count: int = 1,
) -> MarginProfile:
    """Compute the mean decision margin for every locally observed class.

    Missing or insufficiently represented classes receive ``NaN``.  They are
    ignored by the robust reference instead of being confused with zero-margin
    classes.
    """

    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")
    if min_count <= 0:
        raise ValueError("min_count must be positive")

    margins = true_class_margin(logits, targets)
    targets = targets.to(device=logits.device, dtype=torch.long)
    sums = torch.zeros(num_classes, device=logits.device, dtype=logits.dtype)
    counts = torch.zeros(num_classes, device=logits.device, dtype=torch.long)
    sums.scatter_add_(0, targets, margins)
    counts.scatter_add_(0, targets, torch.ones_like(targets, dtype=torch.long))

    values = sums / counts.clamp_min(1).to(logits.dtype)
    values = values.masked_fill(counts < min_count, torch.nan)
    return MarginProfile(values=values, counts=counts)


def class_cross_entropy_profile(
    logits: Tensor,
    targets: Tensor,
    num_classes: int,
    *,
    min_count: int = 1,
) -> MarginProfile:
    """Compute the mean cross-entropy for every locally observed class.

    This profile is the class-wise loss control matched to DMD.  It preserves
    the same client--class granularity while changing only the scientific
    signal: high cross-entropy is harmful, whereas low decision margin is
    harmful. Missing or insufficiently represented classes receive ``NaN``.
    """

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must align with the batch dimension")
    if num_classes <= 1 or logits.shape[1] != num_classes:
        raise ValueError("num_classes must match logits and exceed one")
    if min_count <= 0:
        raise ValueError("min_count must be positive")

    targets = targets.to(device=logits.device, dtype=torch.long)
    losses = F.cross_entropy(logits, targets, reduction="none")
    sums = torch.zeros(num_classes, device=logits.device, dtype=logits.dtype)
    counts = torch.zeros(num_classes, device=logits.device, dtype=torch.long)
    sums.scatter_add_(0, targets, losses)
    counts.scatter_add_(0, targets, torch.ones_like(targets, dtype=torch.long))
    values = sums / counts.clamp_min(1).to(logits.dtype)
    values = values.masked_fill(counts < min_count, torch.nan)
    return MarginProfile(values=values, counts=counts)


def class_accuracy_profile(
    logits: Tensor,
    targets: Tensor,
    num_classes: int,
    *,
    min_count: int = 1,
) -> MarginProfile:
    """Compute accuracy conditional on each locally observed true class."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must align with the batch dimension")
    if num_classes <= 1 or logits.shape[1] != num_classes:
        raise ValueError("num_classes must match logits and exceed one")
    if min_count <= 0:
        raise ValueError("min_count must be positive")

    targets = targets.to(device=logits.device, dtype=torch.long)
    correct = (logits.argmax(dim=1) == targets).to(logits.dtype)
    sums = torch.zeros(num_classes, device=logits.device, dtype=logits.dtype)
    counts = torch.zeros(num_classes, device=logits.device, dtype=torch.long)
    sums.scatter_add_(0, targets, correct)
    counts.scatter_add_(0, targets, torch.ones_like(targets, dtype=torch.long))
    values = sums / counts.clamp_min(1).to(logits.dtype)
    values = values.masked_fill(counts < min_count, torch.nan)
    return MarginProfile(values=values, counts=counts)


def class_brier_profile(
    logits: Tensor,
    targets: Tensor,
    num_classes: int,
    *,
    min_count: int = 1,
) -> MarginProfile:
    """Compute the multiclass Brier score conditional on each true class.

    The per-example score is ``sum_k (p_k - 1[y=k])^2`` and therefore lies
    in ``[0, 2]``.  Lower values indicate better probabilistic predictions.
    """

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must align with the batch dimension")
    if num_classes <= 1 or logits.shape[1] != num_classes:
        raise ValueError("num_classes must match logits and exceed one")
    if min_count <= 0:
        raise ValueError("min_count must be positive")

    targets = targets.to(device=logits.device, dtype=torch.long)
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(targets, num_classes=num_classes).to(logits.dtype)
    scores = (probabilities - one_hot).square().sum(dim=1)
    sums = torch.zeros(num_classes, device=logits.device, dtype=logits.dtype)
    counts = torch.zeros(num_classes, device=logits.device, dtype=torch.long)
    sums.scatter_add_(0, targets, scores)
    counts.scatter_add_(0, targets, torch.ones_like(targets, dtype=torch.long))
    values = sums / counts.clamp_min(1).to(logits.dtype)
    values = values.masked_fill(counts < min_count, torch.nan)
    return MarginProfile(values=values, counts=counts)


def class_top_label_ece_profile(
    logits: Tensor,
    targets: Tensor,
    num_classes: int,
    *,
    bins: int = 10,
    min_count: int = 1,
) -> MarginProfile:
    """Compute top-label ECE within each locally observed true class.

    This is a true-class-conditional diagnostic: samples are first grouped by
    their ground-truth class, then top-label confidence is compared with
    correctness inside equal-width confidence bins.  It complements, but is
    not interchangeable with, a global classwise calibration error.
    """

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must align with the batch dimension")
    if num_classes <= 1 or logits.shape[1] != num_classes:
        raise ValueError("num_classes must match logits and exceed one")
    if bins <= 0:
        raise ValueError("bins must be positive")
    if min_count <= 0:
        raise ValueError("min_count must be positive")

    targets = targets.to(device=logits.device, dtype=torch.long)
    probabilities = logits.softmax(dim=1)
    confidence, predictions = probabilities.max(dim=1)
    correctness = (predictions == targets).to(logits.dtype)
    counts = torch.bincount(targets, minlength=num_classes)
    values = torch.full(
        (num_classes,),
        torch.nan,
        device=logits.device,
        dtype=logits.dtype,
    )
    boundaries = torch.linspace(
        0.0,
        1.0,
        bins + 1,
        device=logits.device,
        dtype=logits.dtype,
    )
    for class_id in range(num_classes):
        mask = targets == class_id
        class_count = int(mask.sum())
        if class_count < min_count:
            continue
        class_confidence = confidence[mask]
        class_correctness = correctness[mask]
        ece = logits.new_zeros(())
        for bin_id in range(bins):
            lower = boundaries[bin_id]
            upper = boundaries[bin_id + 1]
            if bin_id == 0:
                in_bin = (class_confidence >= lower) & (class_confidence <= upper)
            else:
                in_bin = (class_confidence > lower) & (class_confidence <= upper)
            if not bool(in_bin.any()):
                continue
            bin_weight = in_bin.to(logits.dtype).mean()
            gap = torch.abs(
                class_correctness[in_bin].mean() - class_confidence[in_bin].mean()
            )
            ece = ece + bin_weight * gap
        values[class_id] = ece
    return MarginProfile(values=values, counts=counts)


def robust_margin_reference(
    profiles: Tensor,
    *,
    method: Literal["median", "trimmed_mean"] = "median",
    trim_fraction: float = 0.1,
    min_clients: int = 2,
) -> tuple[Tensor, Tensor]:
    """Build a class-wise robust inter-client reference.

    Args:
        profiles: Tensor of shape ``[clients, classes]``. Missing local classes
            must be represented by ``NaN``.
        method: Coordinate-wise median or trimmed mean.
        trim_fraction: Fraction removed from each tail for ``trimmed_mean``.
        min_clients: Minimum class support required to publish a reference.

    Returns:
        ``(reference, support)`` where unsupported classes are ``NaN``.
    """

    if profiles.ndim != 2:
        raise ValueError("profiles must have shape [clients, classes]")
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must lie in [0, 0.5)")
    if min_clients <= 0:
        raise ValueError("min_clients must be positive")

    num_classes = profiles.shape[1]
    reference = torch.full(
        (num_classes,),
        torch.nan,
        dtype=profiles.dtype,
        device=profiles.device,
    )
    support = torch.zeros(num_classes, dtype=torch.long, device=profiles.device)

    for class_id in range(num_classes):
        values = profiles[:, class_id]
        values = values[torch.isfinite(values)]
        support[class_id] = values.numel()
        if values.numel() < min_clients:
            continue
        if method == "median":
            reference[class_id] = values.median()
        elif method == "trimmed_mean":
            sorted_values = values.sort().values
            trim = int(np.floor(trim_fraction * sorted_values.numel()))
            if trim > 0:
                sorted_values = sorted_values[trim:-trim]
            reference[class_id] = sorted_values.mean()
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
    """Build a recency-weighted median from latest client observations.

    Args:
        latest_profiles: Latest class-margin value for each client, shaped
            ``[clients, classes]``. Missing values must be ``NaN``.
        observed_rounds: Round in which each value was observed, with the same
            shape. Missing entries must be negative.
        current_round: Round about to be trained. Only observations from
            completed rounds strictly before it are eligible.
        window: Number of completed rounds retained. An observation from the
            immediately preceding round has age zero; freshness therefore
            requires ``0 <= age < window``.
        decay_gamma: Exponential age-decay coefficient.
        min_effective_clients: Minimum Kish effective sample size required to
            publish a class reference in ``effective`` mode. In ``raw`` mode,
            the same value is the minimum number of distinct clients.
        target_effective_clients: Support at which reliability reaches one.
        publication_support: ``effective`` preserves the original hard Kish
            threshold. ``raw`` publishes once enough distinct clients are
            available, while retaining Kish support as a continuous
            reliability score.

    The function intentionally does not average repeated observations from a
    frequent client. The caller overwrites the client--class cell whenever a
    fresher profile arrives.
    """

    if latest_profiles.ndim != 2:
        raise ValueError("latest_profiles must have shape [clients, classes]")
    if observed_rounds.shape != latest_profiles.shape:
        raise ValueError("observed_rounds must align with latest_profiles")
    if current_round <= 0:
        raise ValueError("current_round must be positive")
    if window <= 0:
        raise ValueError("window must be positive")
    if decay_gamma < 0:
        raise ValueError("decay_gamma must be non-negative")
    if min_effective_clients <= 0:
        raise ValueError("min_effective_clients must be positive")
    if target_effective_clients < min_effective_clients:
        raise ValueError(
            "target_effective_clients must be at least min_effective_clients"
        )
    if publication_support not in ("effective", "raw"):
        raise ValueError("publication_support must be 'effective' or 'raw'")

    profiles = latest_profiles.detach().to(dtype=torch.float64, device="cpu")
    rounds = observed_rounds.detach().to(dtype=torch.long, device="cpu")
    clients, classes = profiles.shape
    del clients
    values = torch.full((classes,), torch.nan, dtype=torch.float64)
    scale = torch.full((classes,), torch.nan, dtype=torch.float64)
    raw_support = torch.zeros(classes, dtype=torch.long)
    effective_support = torch.zeros(classes, dtype=torch.float64)
    reliability = torch.zeros(classes, dtype=torch.float64)
    mean_age = torch.full((classes,), torch.nan, dtype=torch.float64)
    max_age = torch.full((classes,), torch.nan, dtype=torch.float64)

    # The current round consumes the buffer stopped at t-1.  Hence a profile
    # uploaded in t-1 has age zero and a window W retains exactly W completed
    # rounds: t-W, ..., t-1.
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
        if publication_support == "effective":
            publish = float(n_eff) + 1e-12 >= min_effective_clients
        else:
            publish = count + 1e-12 >= min_effective_clients
        if not publish:
            continue

        values[class_id] = _weighted_median(class_values, weights)
        # A recency-weighted MAD supplies the class-specific uncertainty scale.
        # The Gaussian consistency factor makes it interpretable as a robust
        # standard-deviation estimate without assuming Gaussian client margins.
        mad = _weighted_median(
            torch.abs(class_values - values[class_id]),
            weights,
        )
        scale[class_id] = 1.4826 * mad
        reliability[class_id] = min(
            1.0,
            float(n_eff) / target_effective_clients,
        )

    return TemporalMarginReference(
        values=values.to(dtype=latest_profiles.dtype),
        scale=scale.to(dtype=latest_profiles.dtype),
        raw_support=raw_support,
        effective_support=effective_support.to(dtype=latest_profiles.dtype),
        reliability=reliability.to(dtype=latest_profiles.dtype),
        mean_age=mean_age.to(dtype=latest_profiles.dtype),
        max_age=max_age.to(dtype=latest_profiles.dtype),
    )


def temporal_pooled_location_scale(
    latest_profiles: Tensor,
    observed_rounds: Tensor,
    *,
    current_round: int,
    window: int,
    decay_gamma: float,
) -> tuple[Tensor, Tensor]:
    """Robust scalar location/scale after pooling all fresh client--class cells.

    This deliberately destroys class identity and is therefore a diagnostic
    control rather than a proposed fairness reference.  Recency weights are
    identical to :func:`temporal_robust_margin_reference`.
    """

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
    return (
        location.to(dtype=latest_profiles.dtype),
        (1.4826 * mad).to(dtype=latest_profiles.dtype),
    )


def normalized_class_weights(
    counts: Tensor,
    valid: Tensor,
    *,
    mode: Literal["uniform", "frequency"] = "uniform",
) -> Tensor:
    """Return class weights summing to one over valid local classes."""

    if counts.ndim != 1 or valid.ndim != 1 or counts.shape != valid.shape:
        raise ValueError("counts and valid must be one-dimensional and aligned")
    if mode == "uniform":
        raw = valid.to(torch.float32)
    elif mode == "frequency":
        raw = counts.to(torch.float32) * valid.to(torch.float32)
    else:
        raise ValueError(f"unknown class-weight mode: {mode}")
    total = raw.sum()
    if total <= 0:
        return torch.zeros_like(raw)
    return raw / total


def quadratic_margin_deficit(
    profile: MarginProfile,
    reference: Tensor,
    *,
    class_weight_mode: Literal["uniform", "frequency"] = "uniform",
) -> Tensor:
    """Compute ``0.5 * sum_c rho_c [r_c - m_ic]_+^2``.

    This one-sided quadratic shortfall is a fairness penalty, not a variance:
    it compares a client with a class-specific robust target rather than with
    the mean of the same random variable.
    """

    if reference.ndim != 1 or reference.shape != profile.values.shape:
        raise ValueError("reference and profile must have the same shape")
    valid = profile.observed & torch.isfinite(reference)
    weights = normalized_class_weights(
        profile.counts,
        valid,
        mode=class_weight_mode,
    ).to(device=profile.values.device, dtype=profile.values.dtype)
    # A zero weight does not neutralize NaN under IEEE arithmetic. Replace
    # both operands outside the jointly valid support so missing client classes
    # and unpublished references contribute exactly zero.
    safe_profile = torch.where(valid, profile.values, torch.zeros_like(profile.values))
    safe_reference = torch.where(valid, reference, torch.zeros_like(reference))
    shortfall = torch.relu(safe_reference - safe_profile)
    return 0.5 * torch.sum(weights * shortfall.square())


def linear_margin_deficit(
    profile: MarginProfile,
    reference: Tensor,
    *,
    class_weight_mode: Literal["uniform", "frequency"] = "uniform",
) -> Tensor:
    """Compute ``sum_c rho_c [r_c - m_ic]_+``.

    This is the one-sided linear counterpart of
    :func:`quadratic_margin_deficit`.  It applies a constant slope to every
    positive shortfall, whereas the quadratic penalty increases its slope
    proportionally to the deficit magnitude.
    """

    if reference.ndim != 1 or reference.shape != profile.values.shape:
        raise ValueError("reference and profile must have the same shape")
    valid = profile.observed & torch.isfinite(reference)
    weights = normalized_class_weights(
        profile.counts,
        valid,
        mode=class_weight_mode,
    ).to(device=profile.values.device, dtype=profile.values.dtype)
    safe_profile = torch.where(valid, profile.values, torch.zeros_like(profile.values))
    safe_reference = torch.where(valid, reference, torch.zeros_like(reference))
    return torch.sum(weights * torch.relu(safe_reference - safe_profile))


def example_quadratic_dmd_loss(
    logits: Tensor,
    targets: Tensor,
    reference: Tensor,
    *,
    class_weights: Tensor | None = None,
    normalization_class_weights: Tensor | None = None,
) -> Tensor:
    """Differentiable mini-batch DMD penalty used during local training.

    The reference is treated as fixed within the round.  The gradient flows
    through the logits and therefore through the current local model.
    """

    if reference.ndim != 1 or reference.shape[0] != logits.shape[1]:
        raise ValueError("reference must contain one value per class")
    margins = true_class_margin(logits, targets)
    targets = targets.to(device=logits.device, dtype=torch.long)
    local_reference = reference.to(logits.device)[targets]
    valid = torch.isfinite(local_reference)
    if not bool(valid.any()):
        return logits.sum() * 0.0

    penalties = 0.5 * torch.relu(local_reference[valid] - margins[valid]).square()
    if class_weights is None:
        return penalties.mean()

    if class_weights.ndim != 1 or class_weights.shape[0] != logits.shape[1]:
        raise ValueError("class_weights must contain one value per class")
    weights = class_weights.to(logits.device, logits.dtype)[targets[valid]]
    if normalization_class_weights is None:
        denominator_weights = weights
    else:
        if (
            normalization_class_weights.ndim != 1
            or normalization_class_weights.shape[0] != logits.shape[1]
        ):
            raise ValueError(
                "normalization_class_weights must contain one value per class"
            )
        denominator_weights = normalization_class_weights.to(
            logits.device,
            logits.dtype,
        )[targets[valid]]
    if weights.sum() <= 0 or denominator_weights.sum() <= 0:
        return logits.sum() * 0.0
    return torch.sum(weights * penalties) / denominator_weights.sum()


def example_quadratic_standardized_dmd_loss(
    logits: Tensor,
    targets: Tensor,
    reference: Tensor,
    reference_scale: Tensor,
    *,
    class_weights: Tensor | None = None,
    normalization_class_weights: Tensor | None = None,
) -> Tensor:
    """One-sided quadratic DMD after robust class-wise standardization.

    The penalty is

    ``0.5 * [(r_y - m(x,y;w)) / s_y]_+^2``.

    ``reference_scale`` must already include the numerical floor chosen by the
    experiment.  Standardization tests whether the previous semantic-null
    failure was caused by heterogeneous class-margin units rather than by an
    absence of useful class semantics.
    """

    if reference.ndim != 1 or reference.shape[0] != logits.shape[1]:
        raise ValueError("reference must contain one value per class")
    if reference_scale.shape != reference.shape:
        raise ValueError("reference_scale must align with reference")
    margins = true_class_margin(logits, targets)
    targets = targets.to(device=logits.device, dtype=torch.long)
    local_reference = reference.to(logits.device, logits.dtype)[targets]
    local_scale = reference_scale.to(logits.device, logits.dtype)[targets]
    valid = (
        torch.isfinite(local_reference)
        & torch.isfinite(local_scale)
        & (local_scale > 0)
    )
    if not bool(valid.any()):
        return logits.sum() * 0.0
    standardized = (local_reference[valid] - margins[valid]) / local_scale[valid]
    penalties = 0.5 * torch.relu(standardized).square()
    if class_weights is None:
        return penalties.mean()
    if class_weights.ndim != 1 or class_weights.shape[0] != logits.shape[1]:
        raise ValueError("class_weights must contain one value per class")
    weights = class_weights.to(logits.device, logits.dtype)[targets[valid]]
    if normalization_class_weights is None:
        denominator_weights = weights
    else:
        if normalization_class_weights.shape != reference.shape:
            raise ValueError("normalization_class_weights must align with reference")
        denominator_weights = normalization_class_weights.to(
            logits.device,
            logits.dtype,
        )[targets[valid]]
    if weights.sum() <= 0 or denominator_weights.sum() <= 0:
        return logits.sum() * 0.0
    return torch.sum(weights * penalties) / denominator_weights.sum()


def example_linear_dmd_loss(
    logits: Tensor,
    targets: Tensor,
    reference: Tensor,
    *,
    class_weights: Tensor | None = None,
    normalization_class_weights: Tensor | None = None,
) -> Tensor:
    """Differentiable ``[r_y - m(x,y;w)]_+`` mini-batch penalty."""

    if reference.ndim != 1 or reference.shape[0] != logits.shape[1]:
        raise ValueError("reference must contain one value per class")
    margins = true_class_margin(logits, targets)
    targets = targets.to(device=logits.device, dtype=torch.long)
    local_reference = reference.to(logits.device)[targets]
    valid = torch.isfinite(local_reference)
    if not bool(valid.any()):
        return logits.sum() * 0.0

    penalties = torch.relu(local_reference[valid] - margins[valid])
    if class_weights is None:
        return penalties.mean()

    if class_weights.ndim != 1 or class_weights.shape[0] != logits.shape[1]:
        raise ValueError("class_weights must contain one value per class")
    weights = class_weights.to(logits.device, logits.dtype)[targets[valid]]
    if normalization_class_weights is None:
        denominator_weights = weights
    else:
        if (
            normalization_class_weights.ndim != 1
            or normalization_class_weights.shape[0] != logits.shape[1]
        ):
            raise ValueError(
                "normalization_class_weights must contain one value per class"
            )
        denominator_weights = normalization_class_weights.to(
            logits.device,
            logits.dtype,
        )[targets[valid]]
    if weights.sum() <= 0 or denominator_weights.sum() <= 0:
        return logits.sum() * 0.0
    return torch.sum(weights * penalties) / denominator_weights.sum()


def example_quadratic_class_loss_excess(
    logits: Tensor,
    targets: Tensor,
    reference: Tensor,
    *,
    class_weights: Tensor | None = None,
    normalization_class_weights: Tensor | None = None,
) -> Tensor:
    """Return ``0.5 [CE(x,y)-ell_y^F]_+^2`` on a mini-batch.

    The temporal robust reference contains one cross-entropy target per class.
    This is deliberately the loss-space mirror of quadratic DMD: it uses the
    same class support, recency, reliability and one-sided quadratic form.  It
    therefore tests whether margins add value beyond class-wise granularity.
    """

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must align with the batch dimension")
    if reference.ndim != 1 or reference.shape[0] != logits.shape[1]:
        raise ValueError("reference must contain one value per class")

    targets = targets.to(device=logits.device, dtype=torch.long)
    per_example = F.cross_entropy(logits, targets, reduction="none")
    local_reference = reference.to(logits.device, logits.dtype)[targets]
    valid = torch.isfinite(local_reference)
    if not bool(valid.any()):
        return logits.sum() * 0.0

    penalties = (
        0.5 * torch.relu(per_example[valid] - local_reference[valid].detach()).square()
    )
    if class_weights is None:
        return penalties.mean()

    if class_weights.ndim != 1 or class_weights.shape[0] != logits.shape[1]:
        raise ValueError("class_weights must contain one value per class")
    weights = class_weights.to(logits.device, logits.dtype)[targets[valid]]
    if normalization_class_weights is None:
        denominator_weights = weights
    else:
        if (
            normalization_class_weights.ndim != 1
            or normalization_class_weights.shape[0] != logits.shape[1]
        ):
            raise ValueError(
                "normalization_class_weights must contain one value per class"
            )
        denominator_weights = normalization_class_weights.to(
            logits.device,
            logits.dtype,
        )[targets[valid]]
    if weights.sum() <= 0 or denominator_weights.sum() <= 0:
        return logits.sum() * 0.0
    return torch.sum(weights * penalties) / denominator_weights.sum()


def adaptive_hybrid_dmd_penalty(
    linear_penalty: Tensor,
    quadratic_penalty: Tensor,
    *,
    mixture: float | Tensor,
    linear_mu: float,
    quadratic_mu: float,
) -> Tensor:
    """Blend calibrated linear and quadratic DMD penalties.

    The mixture ``q`` is interpreted as an adaptive transition rather than an
    on/off multiplier:

    ``(1-q) * linear_mu * D_linear + q * quadratic_mu * D_quadratic``.

    Hence ``q=0`` exactly recovers calibrated linear DMD and ``q=1`` exactly
    recovers calibrated quadratic DMD.  The two coefficients remain distinct
    because the linear and quadratic deficits have different units/scales.
    """

    if linear_penalty.numel() != 1 or quadratic_penalty.numel() != 1:
        raise ValueError("linear_penalty and quadratic_penalty must be scalar")
    if linear_mu < 0 or quadratic_mu < 0:
        raise ValueError("DMD coefficients must be non-negative")
    q = torch.as_tensor(
        mixture,
        device=linear_penalty.device,
        dtype=linear_penalty.dtype,
    )
    if q.numel() != 1 or not bool(torch.isfinite(q)):
        raise ValueError("mixture must be a finite scalar")
    if not 0.0 <= float(q.detach().cpu()) <= 1.0:
        raise ValueError("mixture must lie in [0, 1]")
    return (1.0 - q) * linear_mu * linear_penalty + q * quadratic_mu * quadratic_penalty


def fedfair_loss_penalty(
    local_loss: Tensor, global_loss_reference: Tensor | float
) -> Tensor:
    """Return the FedFair-style penalty ``0.5 (F_i - F)^2``.

    ``global_loss_reference`` is treated as fixed for the current round.  The
    gradient therefore flows only through ``local_loss``, yielding the factor
    ``(1 + lambda * (F_i - F))`` when the penalty is added to the local loss.
    """

    if local_loss.numel() != 1:
        raise ValueError("local_loss must be scalar")
    reference = torch.as_tensor(
        global_loss_reference,
        device=local_loss.device,
        dtype=local_loss.dtype,
    )
    if reference.numel() != 1:
        raise ValueError("global_loss_reference must be scalar")
    return 0.5 * (local_loss - reference.detach()).square()


def one_sided_quadratic_loss_deficit(
    local_loss: Tensor,
    robust_loss_reference: Tensor | float,
) -> Tensor:
    """Return the one-sided loss-space control ``0.5 [F_i-r_F]_+^2``.

    This is the loss-space analogue of the quadratic DMD penalty.  It is used
    as a matched scientific control: both objectives use a robust temporal
    reference, a one-sided quadratic shortfall, and continuous reliability;
    only the signal space differs (scalar loss versus class-wise margin).

    ``robust_loss_reference`` is fixed during the local optimization step, so
    gradients flow only through ``local_loss``.
    """

    if local_loss.numel() != 1:
        raise ValueError("local_loss must be scalar")
    reference = torch.as_tensor(
        robust_loss_reference,
        device=local_loss.device,
        dtype=local_loss.dtype,
    )
    if reference.numel() != 1 or not bool(torch.isfinite(reference)):
        raise ValueError("robust_loss_reference must be a finite scalar")
    return 0.5 * torch.relu(local_loss - reference.detach()).square()


def deficit_distribution_objective(
    deficit: Tensor,
    mean_deficit_reference: Tensor | float,
    *,
    mean_mu: float,
    dispersion_mu: float,
    mode: Literal["mean", "variance", "upper_semivariance", "cvar"],
    cvar_tail_mass: float = 0.2,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return a client-level decision-deficit fairness addend.

    ``mean_deficit_reference`` is computed on the participating cohort at the
    start of the round and is deliberately detached during local training.
    With identical client weights in ``D_bar`` and server aggregation, the
    sum of the ``variance`` local gradients recovers the gradient of the
    weighted variance because the derivative through ``D_bar`` cancels.  The
    ``upper_semivariance`` arm is instead a round-wise frozen-target
    surrogate. For ``cvar``, the reference is the pre-round weighted VaR
    threshold ``eta_t`` and the Rockafellar--Uryasev representation is used
    with ``eta_t`` detached throughout local optimization. It is therefore a
    round-wise frozen-threshold surrogate: the exact cohort CVaR is recovered
    at the pre-round model, but the threshold is not re-optimized inside each
    local SGD step. These distinctions must remain explicit in reports.
    The three supported objectives are

    ``mean``
        ``mean_mu * D_i``;
    ``variance``
        ``mean_mu * D_i + dispersion_mu * (D_i - D_bar)^2``;
    ``upper_semivariance``
        ``mean_mu * D_i + dispersion_mu * [D_i - D_bar]_+^2``.
    ``cvar``
        ``mean_mu * D_i + dispersion_mu * (eta + [D_i-eta]_+/q)``.

    The upper-semivariance variant never pushes a below-average-deficit client
    upward merely to equalize the cohort.  The function returns the total,
    mean, and dispersion addends separately for auditable logging.
    """

    if deficit.numel() != 1:
        raise ValueError("deficit must be scalar")
    if mean_mu < 0 or dispersion_mu < 0:
        raise ValueError("deficit fairness coefficients must be non-negative")
    if not 0.0 < cvar_tail_mass <= 1.0:
        raise ValueError("cvar_tail_mass must lie in (0, 1]")
    reference = torch.as_tensor(
        mean_deficit_reference,
        device=deficit.device,
        dtype=deficit.dtype,
    )
    if reference.numel() != 1 or not bool(torch.isfinite(reference)):
        raise ValueError("mean_deficit_reference must be a finite scalar")
    centered = deficit - reference.detach()
    if mode == "mean":
        dispersion_penalty = deficit * 0.0
    elif mode == "variance":
        dispersion_penalty = centered.square()
    elif mode == "upper_semivariance":
        dispersion_penalty = torch.relu(centered).square()
    elif mode == "cvar":
        dispersion_penalty = reference.detach() + (
            torch.relu(centered) / cvar_tail_mass
        )
    else:
        raise ValueError(f"unknown deficit-distribution mode: {mode}")
    mean_addend = mean_mu * deficit
    dispersion_addend = dispersion_mu * dispersion_penalty
    return mean_addend + dispersion_addend, mean_addend, dispersion_addend


def cyclically_permute_class_reference(
    reference: Tensor,
    reliability: Tensor,
    *,
    shift: int,
) -> tuple[Tensor, Tensor]:
    """Destroy class semantics while preserving reference marginals exactly.

    A permutation over clients would leave a coordinate-wise median unchanged.
    The decisive negative control therefore permutes the *class coordinates*
    consumed by the local DMD objective.  The same permutation is applied to
    the reference and its reliability so their joint empirical distribution is
    preserved while class-to-class meaning is broken.
    """

    if reference.ndim != 1 or reliability.ndim != 1:
        raise ValueError("reference and reliability must be one-dimensional")
    if reference.shape != reliability.shape:
        raise ValueError("reference and reliability must have the same shape")
    if reference.numel() < 2:
        raise ValueError("at least two classes are required for permutation")
    normalized_shift = int(shift) % reference.numel()
    if normalized_shift == 0:
        raise ValueError("shift must induce a non-identity class permutation")
    return (
        torch.roll(reference, shifts=normalized_shift, dims=0),
        torch.roll(reliability, shifts=normalized_shift, dims=0),
    )


def permute_class_reference(
    reference: Tensor,
    reliability: Tensor,
    permutation: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply an arbitrary class derangement to reference and reliability.

    ``permutation[c]`` is the wrong reference consumed by local class ``c``.
    Requiring a derangement avoids partially preserving the very class
    semantics that the negative control is intended to destroy.
    """

    if reference.ndim != 1 or reliability.ndim != 1:
        raise ValueError("reference and reliability must be one-dimensional")
    if reference.shape != reliability.shape:
        raise ValueError("reference and reliability must have the same shape")
    if permutation.ndim != 1 or permutation.numel() != reference.numel():
        raise ValueError("permutation must contain one index per class")
    permutation = permutation.to(device=reference.device, dtype=torch.long)
    expected = torch.arange(reference.numel(), device=reference.device)
    if not torch.equal(permutation.sort().values, expected):
        raise ValueError("permutation must contain each class exactly once")
    if bool((permutation == expected).any()):
        raise ValueError("permutation must be a derangement without fixed points")
    reliability_permutation = permutation.to(reliability.device)
    return reference[permutation], reliability[reliability_permutation]


def fedfdp_fair_objective_penalty(
    logits: Tensor,
    targets: Tensor,
    global_loss_reference: Tensor | float,
) -> Tensor:
    """Return the sample-wise FedFDP fairness penalty without DP noise.

    FedFDP expands the FedFair objective at sample level before applying its
    fair-clipping and Gaussian mechanisms.  This helper isolates that fairness
    signal:

    ``0.5 * mean_j (ell_j(w) - F_global)^2``.

    It is deliberately *not* labelled as a full FedFDP implementation: there
    is no per-sample norm clipping, no Gaussian noise, and no private loss
    release here.  The ablation is useful for comparing the loss-space signal
    with DMD before privacy is introduced as a confounder.
    """

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must align with the batch dimension")
    reference = torch.as_tensor(
        global_loss_reference,
        device=logits.device,
        dtype=logits.dtype,
    )
    if reference.numel() != 1:
        raise ValueError("global_loss_reference must be scalar")
    per_example = F.cross_entropy(logits, targets, reduction="none")
    return 0.5 * (per_example - reference.detach()).square().mean()


def project_capped_simplex(probabilities: Tensor, cap: float) -> Tensor:
    """Project by proportional water filling onto ``p_i <= cap``.

    The operation preserves non-negativity and unit sum. It is intended as a
    simple influence cap for the prototype, not as a DP guarantee.
    """

    if probabilities.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    n = probabilities.numel()
    if n == 0:
        raise ValueError("at least one client is required")
    if cap <= 0 or cap * n < 1.0 - 1e-7:
        raise ValueError("cap must be positive and satisfy cap * n >= 1")

    p = probabilities.clamp_min(0)
    if p.sum() <= 0:
        p = torch.ones_like(p) / n
    else:
        p = p / p.sum()
    if bool((p <= cap + 1e-7).all()):
        return p

    result = torch.zeros_like(p)
    active = torch.ones(n, dtype=torch.bool, device=p.device)
    remaining_mass = torch.tensor(1.0, dtype=p.dtype, device=p.device)
    while bool(active.any()):
        active_values = p[active]
        scaled = active_values / active_values.sum() * remaining_mass
        over = scaled > cap
        active_indices = torch.where(active)[0]
        if not bool(over.any()):
            result[active_indices] = scaled
            break
        capped_indices = active_indices[over]
        result[capped_indices] = cap
        active[capped_indices] = False
        remaining_mass = remaining_mass - cap * capped_indices.numel()
    return result / result.sum()


def term_client_weights(
    losses: Tensor,
    *,
    base_weights: Tensor | None = None,
    tau: float,
    cap: float | None = None,
) -> Tensor:
    """Compute client-level TERM weights ``p_i exp(tau * F_i)``.

    The subtraction of the maximum log-weight is purely numerical and leaves
    the normalized distribution unchanged.  ``tau=0`` recovers the data-size
    weights.  An optional cap can be used as an influence-control ablation;
    the default experimental TERM baseline leaves it disabled.
    """

    if losses.ndim != 1 or losses.numel() == 0:
        raise ValueError("losses must be a non-empty one-dimensional tensor")
    if tau < 0:
        raise ValueError("tau must be non-negative")
    if base_weights is None:
        base = torch.ones_like(losses) / losses.numel()
    else:
        if base_weights.shape != losses.shape:
            raise ValueError("base_weights must align with losses")
        if bool((base_weights < 0).any()) or float(base_weights.sum()) <= 0:
            raise ValueError("base_weights must be non-negative with positive sum")
        base = base_weights.to(device=losses.device, dtype=losses.dtype)
        base = base / base.sum()
    log_weights = torch.log(base.clamp_min(torch.finfo(base.dtype).tiny)) + tau * losses
    raw = torch.exp(log_weights - log_weights.max())
    weights = raw / raw.sum()
    if cap is not None and cap < 1.0:
        weights = project_capped_simplex(weights, cap)
    return weights


def tilted_client_weights(
    deficits: Tensor,
    *,
    base_weights: Tensor | None = None,
    tau: float = 1.0,
    cap: float | None = None,
) -> Tensor:
    """Return ``pi_i proportional to p_i exp(tau * D_i)``."""

    if deficits.ndim != 1 or deficits.numel() == 0:
        raise ValueError("deficits must contain one scalar per client")
    if tau < 0:
        raise ValueError("tau must be non-negative for deficit tilting")
    n = deficits.numel()
    if base_weights is None:
        base_weights = torch.ones_like(deficits) / n
    if base_weights.shape != deficits.shape or bool((base_weights <= 0).any()):
        raise ValueError("base_weights must be positive and aligned with deficits")
    base_weights = base_weights / base_weights.sum()

    logits = torch.log(base_weights) + tau * deficits
    probabilities = torch.softmax(logits, dim=0)
    if cap is not None:
        probabilities = project_capped_simplex(probabilities, cap)
    return probabilities


def adaptive_fairness_intensity(
    disparity: float | Tensor,
    *,
    low: float,
    high: float,
) -> float:
    """Map an inter-client disparity signal to a factor in ``[0, 1]``.

    The initial adaptive DMD prototype uses the weighted variance of client
    anchor accuracies as the control signal. Below ``low`` the DMD penalty is
    switched off; above ``high`` its calibrated coefficient is fully active.
    Values in between are linearly interpolated. Temporal smoothing is kept
    in the experiment runner because its state must be checkpointed.
    """

    if high <= low:
        raise ValueError("high must be strictly greater than low")
    value = float(torch.as_tensor(disparity).detach().cpu())
    if not np.isfinite(value):
        raise ValueError("disparity must be finite")
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def reference_support_reliability(
    support: Tensor,
    *,
    num_clients: int,
    min_clients: int,
) -> float:
    """Return the usable client--class support fraction in ``[0, 1]``.

    For class ``c``, ``support[c]`` is the number of clients whose anchor set
    contains enough examples to contribute to the robust margin reference.
    A class below ``min_clients`` has no published reference and contributes
    zero.  Otherwise it contributes ``support[c] / num_clients``.  Averaging
    over all classes gives

    ``R = (1/C) sum_c 1[s_c >= s_min] s_c / n``.

    Hence ``R=1`` only when every client supports every class, while sparse or
    unpublished references reduce ``R``.  The score measures structural
    support, not honesty, calibration, or differential privacy.
    """

    usable = class_reference_reliability(
        support,
        num_clients=num_clients,
        min_clients=min_clients,
    )
    return float(usable.mean())


def class_reference_reliability(
    support: Tensor,
    *,
    num_clients: int,
    min_clients: int,
) -> Tensor:
    """Return one structural reliability score per class.

    For class ``c`` the score is

    ``R_c = 1[s_c >= s_min] * s_c / n``.

    Unlike :func:`reference_support_reliability`, this function preserves the
    class structure.  It can therefore attenuate only poorly supported class
    penalties instead of weakening the complete DMD objective.  The returned
    CPU tensor measures support, not honesty or statistical calibration.
    """

    if support.ndim != 1 or support.numel() == 0:
        raise ValueError("support must contain one value per class")
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if not 0 < min_clients <= num_clients:
        raise ValueError("min_clients must lie in [1, num_clients]")
    support_float = support.detach().to(dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(support_float).all()):
        raise ValueError("support must be finite")
    if bool((support_float < 0).any()) or bool((support_float > num_clients).any()):
        raise ValueError("support counts must lie in [0, num_clients]")
    return torch.where(
        support_float >= min_clients,
        support_float / num_clients,
        torch.zeros_like(support_float),
    )


def reliability_weighted_class_weights(
    base_weights: Tensor,
    reliability: Tensor,
    *,
    power: float,
) -> Tensor:
    """Renormalize class weights using class-specific reliability.

    ``rho_tilde_c = rho_c R_c**power / sum_k rho_k R_k**power``.

    Renormalization prevents a low average reliability from switching down the
    entire DMD objective.  If all supported classes have equal reliability,
    the original normalized class weights are recovered exactly.  Classes with
    zero reliability receive zero DMD influence.
    """

    if base_weights.ndim != 1 or reliability.ndim != 1:
        raise ValueError("base_weights and reliability must be one-dimensional")
    if base_weights.shape != reliability.shape or base_weights.numel() == 0:
        raise ValueError("base_weights and reliability must be aligned")
    if power < 0:
        raise ValueError("power must be non-negative")
    base = base_weights.detach().to(dtype=torch.float64, device="cpu")
    score = reliability.detach().to(dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(base).all()) or not bool(torch.isfinite(score).all()):
        raise ValueError("base_weights and reliability must be finite")
    if bool((base < 0).any()):
        raise ValueError("base_weights must be non-negative")
    if bool((score < 0).any()) or bool((score > 1).any()):
        raise ValueError("reliability must lie in [0, 1]")
    raw = base * score.pow(power)
    total = raw.sum()
    if total <= 0:
        return torch.zeros_like(raw, dtype=base_weights.dtype)
    return (raw / total).to(dtype=base_weights.dtype)


def reliability_adjusted_mixture(
    disparity_factor: float | Tensor,
    reliability: float | Tensor,
    *,
    power: float,
) -> float:
    """Attenuate the quadratic mixture using reference reliability.

    ``q_rel = q_disp * R**power``.  ``power=0`` recovers the original
    controller, ``power=1`` applies linear attenuation, and ``power=0.5`` is a
    softer square-root ablation.  This controls the linear--quadratic
    transition; it does not switch the entire fairness objective off.
    """

    if power < 0:
        raise ValueError("power must be non-negative")
    q = float(torch.as_tensor(disparity_factor).detach().cpu())
    score = float(torch.as_tensor(reliability).detach().cpu())
    if not np.isfinite(q) or not np.isfinite(score):
        raise ValueError("disparity_factor and reliability must be finite")
    if not 0.0 <= q <= 1.0 or not 0.0 <= score <= 1.0:
        raise ValueError("disparity_factor and reliability must lie in [0, 1]")
    return float(q * score**power)


def pearson_spearman(x: Tensor, y: Tensor) -> tuple[float, float]:
    """Return Pearson and Spearman correlations, ignoring non-finite pairs."""

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    x_np = x.detach().cpu().numpy().astype(np.float64, copy=False).ravel()
    y_np = y.detach().cpu().numpy().astype(np.float64, copy=False).ravel()
    valid = np.isfinite(x_np) & np.isfinite(y_np)
    x_np, y_np = x_np[valid], y_np[valid]
    if x_np.size < 2 or np.std(x_np) == 0 or np.std(y_np) == 0:
        return float("nan"), float("nan")

    pearson = float(np.corrcoef(x_np, y_np)[0, 1])
    x_rank = _average_ranks(x_np)
    y_rank = _average_ranks(y_np)
    spearman = float(np.corrcoef(x_rank, y_rank)[0, 1])
    return pearson, spearman


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank values using the average rank for ties."""

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks
