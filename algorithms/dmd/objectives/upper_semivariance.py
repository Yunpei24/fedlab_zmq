"""Upper-semivariance decision-deficit objective."""

import torch
from torch import Tensor

from ..contracts import ObjectiveTerms
from .base import make_terms


def upper_semivariance_objective(
    deficit: Tensor,
    cohort_mean_deficit: Tensor | float,
    *,
    mean_mu: float,
    dispersion_mu: float,
) -> ObjectiveTerms:
    if deficit.numel() != 1 or min(mean_mu, dispersion_mu) < 0:
        raise ValueError("invalid upper-semivariance objective")
    reference = torch.as_tensor(
        cohort_mean_deficit, device=deficit.device, dtype=deficit.dtype
    ).detach()
    if reference.numel() != 1 or not bool(torch.isfinite(reference)):
        raise ValueError("cohort mean deficit must be finite and scalar")
    return make_terms(
        mean_mu * deficit,
        dispersion_mu * torch.relu(deficit - reference).square(),
    )


__all__ = ["upper_semivariance_objective"]
