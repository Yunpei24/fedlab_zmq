"""Frozen-threshold CVaR decision-deficit objective."""

import torch
from torch import Tensor

from ..contracts import ObjectiveTerms
from .base import make_terms


def tail_objective(
    deficit: Tensor,
    eta: Tensor | float,
    *,
    mean_mu: float,
    dispersion_mu: float,
    tail_mass: float,
) -> ObjectiveTerms:
    if deficit.numel() != 1 or min(mean_mu, dispersion_mu) < 0:
        raise ValueError("invalid DMD-Tail objective")
    if not 0.0 < tail_mass <= 1.0:
        raise ValueError("tail_mass must lie in (0, 1]")
    threshold = torch.as_tensor(
        eta, device=deficit.device, dtype=deficit.dtype
    ).detach()
    if threshold.numel() != 1 or not bool(torch.isfinite(threshold)):
        raise ValueError("eta must be finite and scalar")
    risk = threshold + torch.relu(deficit - threshold) / tail_mass
    return make_terms(mean_mu * deficit, dispersion_mu * risk)


__all__ = ["tail_objective"]
