"""Finite weighted tail-risk utilities used by DMD-Tail."""

from __future__ import annotations

import torch
from torch import Tensor

from .contracts import WeightedUpperCvar


def weighted_upper_cvar(
    values: Tensor,
    weights: Tensor,
    *,
    tail_mass: float,
) -> WeightedUpperCvar:
    """Return the exact upper-tail CVaR of a finite weighted cohort."""

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
    for index_tensor in order:
        index = int(index_tensor)
        if remaining <= 1e-12:
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
