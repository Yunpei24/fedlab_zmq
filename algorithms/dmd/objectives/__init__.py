"""Public objective API for DMD-Mean, DMD-USV and DMD-Tail."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from .base import (
    class_balanced_example_margin_deficit,
    example_linear_dmd_loss,
    example_quadratic_dmd_loss,
    example_quadratic_standardized_dmd_loss,
    linear_margin_deficit,
    normalized_class_weights,
    quadratic_margin_deficit,
)
from .mean import mean_objective
from .tail import tail_objective
from .upper_semivariance import upper_semivariance_objective


def deficit_distribution_objective(
    deficit: Tensor,
    mean_deficit_reference: Tensor | float,
    *,
    mean_mu: float,
    dispersion_mu: float,
    mode: Literal["mean", "variance", "upper_semivariance", "cvar"],
    cvar_tail_mass: float = 0.2,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compatibility API returning ``(total, mean, dispersion)``."""

    if mode == "mean":
        terms = mean_objective(deficit, mean_mu=mean_mu)
    elif mode == "upper_semivariance":
        terms = upper_semivariance_objective(
            deficit,
            mean_deficit_reference,
            mean_mu=mean_mu,
            dispersion_mu=dispersion_mu,
        )
    elif mode == "cvar":
        terms = tail_objective(
            deficit,
            mean_deficit_reference,
            mean_mu=mean_mu,
            dispersion_mu=dispersion_mu,
            tail_mass=cvar_tail_mass,
        )
    elif mode == "variance":
        centered = (
            deficit
            - torch.as_tensor(mean_deficit_reference)
            .to(device=deficit.device, dtype=deficit.dtype)
            .detach()
        )
        mean_term = mean_mu * deficit
        dispersion = dispersion_mu * centered.square()
        return mean_term + dispersion, mean_term, dispersion
    else:
        raise ValueError(f"unknown deficit-distribution mode: {mode}")
    return terms.total, terms.mean, terms.dispersion


__all__ = [
    "class_balanced_example_margin_deficit",
    "normalized_class_weights",
    "quadratic_margin_deficit",
    "linear_margin_deficit",
    "example_quadratic_dmd_loss",
    "example_quadratic_standardized_dmd_loss",
    "example_linear_dmd_loss",
    "mean_objective",
    "upper_semivariance_objective",
    "tail_objective",
    "deficit_distribution_objective",
]
