"""Robust aggregation rules used by the FAR reference experiments.

The public API deliberately works on ``list[dict[str, Tensor]]`` so it can be
used by every :class:`algorithms.base.FLAlgorithm` without flattening model
updates in the experiment runner.  Individual implementations flatten only
the floating-point tensors, aggregate them, and restore the original mapping.
"""

from .aggregators import (
    aggregate_updates,
    cmls,
    coordinate_median,
    geometric_median,
    nearest_neighbor_mixing,
    norm_based_screening,
    trimmed_mean,
)

__all__ = [
    "aggregate_updates",
    "cmls",
    "coordinate_median",
    "geometric_median",
    "nearest_neighbor_mixing",
    "norm_based_screening",
    "trimmed_mean",
]
