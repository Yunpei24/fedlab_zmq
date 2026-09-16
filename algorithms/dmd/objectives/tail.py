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
    # Rockafellar-Uryasev writes CVaR_b(D) = min_eta eta + [D-eta]_+ / b.  Here
    # eta is detached, so the leading ``eta`` is a constant: it contributes no
    # gradient, but it did add mu_V * eta to every logged ``local_dmd_addend``
    # even for clients strictly below the threshold, which made the loss traces
    # incomparable with the mean/USV variants.  We drop it; the optimisation is
    # unchanged and the CVaR level itself stays available as ``dmd_cvar_eta``
    # and ``dmd_deficit_cvar`` in the server audit.
    risk = torch.relu(deficit - threshold) / tail_mass
    return make_terms(mean_mu * deficit, dispersion_mu * risk)


__all__ = ["tail_objective"]
