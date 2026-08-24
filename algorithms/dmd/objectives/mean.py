"""Mean decision-deficit objective."""

from torch import Tensor

from ..contracts import ObjectiveTerms
from .base import make_terms


def mean_objective(deficit: Tensor, *, mean_mu: float) -> ObjectiveTerms:
    if deficit.numel() != 1 or mean_mu < 0:
        raise ValueError("deficit must be scalar and mean_mu non-negative")
    return make_terms(mean_mu * deficit, deficit * 0.0)


__all__ = ["mean_objective"]
