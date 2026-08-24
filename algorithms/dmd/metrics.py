"""DMD influence controls, reliability scores and analysis metrics."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def cohort_participation_mu_scale(
    survivors: int, *, reference_clients: int, mode: str
) -> float:
    """Scale a fairness coefficient by the surviving cohort fraction.

    ``reference_clients`` is the preregistered full cohort size, not the
    selected cohort size.  The function returns a dimensionless multiplier in
    ``[0, 1]``; multiplying it by ``mu_0`` gives the effective round value.
    """

    if reference_clients <= 0 or not 0 <= survivors <= reference_clients:
        raise ValueError("survivors must lie in [0, reference_clients]")
    ratio = survivors / reference_clients
    if mode == "sqrt":
        return float(np.sqrt(ratio))
    if mode == "linear":
        return float(ratio)
    raise ValueError("mode must be 'sqrt' or 'linear'")


def class_support_mu_scales(
    cohort_support: Tensor, reference_support: Tensor
) -> Tensor:
    """Return ``sqrt(h_c,t / h_c,ref)`` class scales clipped to ``[0, 1]``.

    A class absent from the static ten-client reference receives scale zero.
    Inputs are counts of clients whose fixed local anchor set contains the
    class, not counts of individual examples.
    """

    if (
        cohort_support.ndim != 1
        or cohort_support.shape != reference_support.shape
        or cohort_support.numel() == 0
    ):
        raise ValueError("support counts must be non-empty aligned vectors")
    cohort = cohort_support.detach().to(dtype=torch.float64, device="cpu")
    reference = reference_support.detach().to(dtype=torch.float64, device="cpu")
    if (
        not bool(torch.isfinite(cohort).all())
        or not bool(torch.isfinite(reference).all())
        or bool((cohort < 0).any())
        or bool((reference < 0).any())
        or bool((cohort > reference).any())
    ):
        raise ValueError("support counts must satisfy 0 <= cohort <= reference")
    ratio = torch.where(reference > 0, cohort / reference, torch.zeros_like(reference))
    return torch.sqrt(ratio.clamp(0.0, 1.0)).to(dtype=torch.float32)


def project_capped_simplex(probabilities: Tensor, cap: float) -> Tensor:
    """Proportional water-filling projection onto ``p_i <= cap``."""

    if probabilities.ndim != 1 or probabilities.numel() == 0:
        raise ValueError("probabilities must be a non-empty vector")
    n = probabilities.numel()
    if cap <= 0 or cap * n < 1.0 - 1e-7:
        raise ValueError("cap must satisfy cap * n >= 1")
    p = probabilities.clamp_min(0)
    p = p / p.sum() if float(p.sum()) > 0 else torch.ones_like(p) / n
    if bool((p <= cap + 1e-7).all()):
        return p
    result = torch.zeros_like(p)
    active = torch.ones(n, dtype=torch.bool, device=p.device)
    remaining = p.new_tensor(1.0)
    while bool(active.any()):
        active_indices = torch.where(active)[0]
        scaled = p[active] / p[active].sum() * remaining
        over = scaled > cap
        if not bool(over.any()):
            result[active_indices] = scaled
            break
        capped = active_indices[over]
        result[capped] = cap
        active[capped] = False
        remaining = remaining - cap * capped.numel()
    return result / result.sum()


def term_client_weights(
    losses: Tensor,
    *,
    base_weights: Tensor | None = None,
    tau: float,
    cap: float | None = None,
) -> Tensor:
    """Return normalized ``p_i exp(tau F_i)`` TERM weights."""

    if losses.ndim != 1 or losses.numel() == 0 or tau < 0:
        raise ValueError("losses must be a non-empty vector and tau non-negative")
    base = (
        torch.ones_like(losses) / losses.numel()
        if base_weights is None
        else base_weights
    )
    if base.shape != losses.shape or bool((base < 0).any()) or float(base.sum()) <= 0:
        raise ValueError("base_weights must be aligned and non-negative")
    base = base.to(losses) / base.sum()
    logits = torch.log(base.clamp_min(torch.finfo(base.dtype).tiny)) + tau * losses
    weights = torch.softmax(logits, dim=0)
    return (
        project_capped_simplex(weights, cap) if cap is not None and cap < 1 else weights
    )


def tilted_client_weights(
    deficits: Tensor,
    *,
    base_weights: Tensor | None = None,
    tau: float = 1.0,
    cap: float | None = None,
) -> Tensor:
    """Return normalized ``p_i exp(tau D_i)`` DMD weights."""

    if deficits.ndim != 1 or deficits.numel() == 0 or tau < 0:
        raise ValueError("deficits must be a non-empty vector and tau non-negative")
    base = (
        torch.ones_like(deficits) / deficits.numel()
        if base_weights is None
        else base_weights
    )
    if base.shape != deficits.shape or bool((base <= 0).any()):
        raise ValueError("base_weights must be positive and aligned")
    weights = torch.softmax(torch.log(base / base.sum()) + tau * deficits, dim=0)
    return project_capped_simplex(weights, cap) if cap is not None else weights


def adaptive_fairness_intensity(
    disparity: float | Tensor, *, low: float, high: float
) -> float:
    if high <= low:
        raise ValueError("high must exceed low")
    value = float(torch.as_tensor(disparity).detach().cpu())
    if not np.isfinite(value):
        raise ValueError("disparity must be finite")
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def class_reference_reliability(
    support: Tensor, *, num_clients: int, min_clients: int
) -> Tensor:
    if support.ndim != 1 or support.numel() == 0:
        raise ValueError("support must contain one value per class")
    if num_clients <= 0 or not 0 < min_clients <= num_clients:
        raise ValueError("invalid client-support configuration")
    values = support.detach().to(dtype=torch.float64, device="cpu")
    if (
        not bool(torch.isfinite(values).all())
        or bool((values < 0).any())
        or bool((values > num_clients).any())
    ):
        raise ValueError("support counts must lie in [0, num_clients]")
    return torch.where(
        values >= min_clients, values / num_clients, torch.zeros_like(values)
    )


def reference_support_reliability(
    support: Tensor, *, num_clients: int, min_clients: int
) -> float:
    return float(
        class_reference_reliability(
            support, num_clients=num_clients, min_clients=min_clients
        ).mean()
    )


def reliability_weighted_class_weights(
    base_weights: Tensor, reliability: Tensor, *, power: float
) -> Tensor:
    if base_weights.ndim != 1 or base_weights.shape != reliability.shape or power < 0:
        raise ValueError("base_weights and reliability must be aligned vectors")
    base = base_weights.detach().to(dtype=torch.float64, device="cpu")
    score = reliability.detach().to(dtype=torch.float64, device="cpu")
    if bool((base < 0).any()) or bool((score < 0).any()) or bool((score > 1).any()):
        raise ValueError("invalid weights or reliability")
    raw = base * score.pow(power)
    return (
        (raw / raw.sum()).to(base_weights.dtype)
        if float(raw.sum()) > 0
        else torch.zeros_like(base_weights)
    )


def reliability_adjusted_mixture(
    disparity_factor: float | Tensor,
    reliability: float | Tensor,
    *,
    power: float,
) -> float:
    if power < 0:
        raise ValueError("power must be non-negative")
    q = float(torch.as_tensor(disparity_factor))
    score = float(torch.as_tensor(reliability))
    if not 0 <= q <= 1 or not 0 <= score <= 1:
        raise ValueError("inputs must lie in [0, 1]")
    return float(q * score**power)


def pearson_spearman(x: Tensor, y: Tensor) -> tuple[float, float]:
    """Return Pearson and average-rank Spearman correlations."""

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    x_np = x.detach().cpu().numpy().astype(np.float64, copy=False).ravel()
    y_np = y.detach().cpu().numpy().astype(np.float64, copy=False).ravel()
    valid = np.isfinite(x_np) & np.isfinite(y_np)
    x_np, y_np = x_np[valid], y_np[valid]
    if x_np.size < 2 or np.std(x_np) == 0 or np.std(y_np) == 0:
        return float("nan"), float("nan")
    return float(np.corrcoef(x_np, y_np)[0, 1]), float(
        np.corrcoef(_average_ranks(x_np), _average_ranks(y_np))[0, 1]
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
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


__all__ = [
    "cohort_participation_mu_scale",
    "class_support_mu_scales",
    "project_capped_simplex",
    "term_client_weights",
    "tilted_client_weights",
    "adaptive_fairness_intensity",
    "class_reference_reliability",
    "reference_support_reliability",
    "reliability_weighted_class_weights",
    "reliability_adjusted_mixture",
    "pearson_spearman",
]
