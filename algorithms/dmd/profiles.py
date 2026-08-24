"""Decision-space profiles computed locally from model logits."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .contracts import MarginProfile


def true_class_margin(logits: Tensor, targets: Tensor) -> Tensor:
    """Return ``z_y - max_{k != y} z_k`` for every observation."""

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
    return true_logits - competitors.max(dim=1).values


def _mean_profile(
    scores: Tensor,
    targets: Tensor,
    num_classes: int,
    *,
    min_count: int,
) -> MarginProfile:
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")
    if min_count <= 0:
        raise ValueError("min_count must be positive")
    targets = targets.to(device=scores.device, dtype=torch.long)
    sums = torch.zeros(num_classes, device=scores.device, dtype=scores.dtype)
    counts = torch.zeros(num_classes, device=scores.device, dtype=torch.long)
    sums.scatter_add_(0, targets, scores)
    counts.scatter_add_(0, targets, torch.ones_like(targets, dtype=torch.long))
    values = sums / counts.clamp_min(1).to(scores.dtype)
    return MarginProfile(
        values=values.masked_fill(counts < min_count, torch.nan),
        counts=counts,
    )


def class_margin_profile(
    logits: Tensor,
    targets: Tensor,
    num_classes: int,
    *,
    min_count: int = 1,
) -> MarginProfile:
    """Compute the mean decision margin for every locally observed class."""

    return _mean_profile(
        true_class_margin(logits, targets),
        targets,
        num_classes,
        min_count=min_count,
    )


def class_cross_entropy_profile(
    logits: Tensor,
    targets: Tensor,
    num_classes: int,
    *,
    min_count: int = 1,
) -> MarginProfile:
    """Compute mean cross-entropy conditional on each true class."""

    _validate_logits(logits, targets, num_classes)
    losses = F.cross_entropy(logits, targets.to(logits.device), reduction="none")
    return _mean_profile(losses, targets, num_classes, min_count=min_count)


def class_accuracy_profile(
    logits: Tensor,
    targets: Tensor,
    num_classes: int,
    *,
    min_count: int = 1,
) -> MarginProfile:
    """Compute accuracy conditional on each locally observed true class."""

    _validate_logits(logits, targets, num_classes)
    local_targets = targets.to(device=logits.device, dtype=torch.long)
    correct = (logits.argmax(dim=1) == local_targets).to(logits.dtype)
    return _mean_profile(correct, local_targets, num_classes, min_count=min_count)


def class_brier_profile(
    logits: Tensor,
    targets: Tensor,
    num_classes: int,
    *,
    min_count: int = 1,
) -> MarginProfile:
    """Compute multiclass Brier score conditional on each true class."""

    _validate_logits(logits, targets, num_classes)
    local_targets = targets.to(device=logits.device, dtype=torch.long)
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(local_targets, num_classes=num_classes).to(logits.dtype)
    scores = (probabilities - one_hot).square().sum(dim=1)
    return _mean_profile(scores, local_targets, num_classes, min_count=min_count)


def class_top_label_ece_profile(
    logits: Tensor,
    targets: Tensor,
    num_classes: int,
    *,
    bins: int = 10,
    min_count: int = 1,
) -> MarginProfile:
    """Compute top-label ECE within each locally observed true class."""

    _validate_logits(logits, targets, num_classes)
    if bins <= 0 or min_count <= 0:
        raise ValueError("bins and min_count must be positive")
    local_targets = targets.to(device=logits.device, dtype=torch.long)
    probabilities = logits.softmax(dim=1)
    confidence, predictions = probabilities.max(dim=1)
    correctness = (predictions == local_targets).to(logits.dtype)
    counts = torch.bincount(local_targets, minlength=num_classes)
    values = torch.full(
        (num_classes,), torch.nan, device=logits.device, dtype=logits.dtype
    )
    boundaries = torch.linspace(
        0.0, 1.0, bins + 1, device=logits.device, dtype=logits.dtype
    )
    for class_id in range(num_classes):
        mask = local_targets == class_id
        if int(mask.sum()) < min_count:
            continue
        class_confidence = confidence[mask]
        class_correctness = correctness[mask]
        ece = logits.new_zeros(())
        for bin_id in range(bins):
            lower, upper = boundaries[bin_id], boundaries[bin_id + 1]
            in_bin = (
                (class_confidence >= lower) & (class_confidence <= upper)
                if bin_id == 0
                else (class_confidence > lower) & (class_confidence <= upper)
            )
            if bool(in_bin.any()):
                ece = ece + in_bin.to(logits.dtype).mean() * torch.abs(
                    class_correctness[in_bin].mean() - class_confidence[in_bin].mean()
                )
        values[class_id] = ece
    return MarginProfile(values=values, counts=counts)


def profile_to_wire(profile: MarginProfile) -> tuple[list[float | None], list[int]]:
    """Convert a profile to primitives suitable for msgpack metadata."""

    values = [
        float(value) if bool(torch.isfinite(value)) else None
        for value in profile.values.detach().cpu()
    ]
    counts = [int(value) for value in profile.counts.detach().cpu()]
    return values, counts


def profile_from_wire(
    values: list[float | None] | tuple[float | None, ...],
    counts: list[int] | tuple[int, ...],
    *,
    dtype: torch.dtype = torch.float32,
) -> MarginProfile:
    """Reconstruct a CPU profile from serialized primitive values."""

    if len(values) != len(counts):
        raise ValueError("profile values and counts must align")
    return MarginProfile(
        values=torch.tensor(
            [float("nan") if value is None else value for value in values],
            dtype=dtype,
        ),
        counts=torch.tensor(counts, dtype=torch.long),
    )


def _validate_logits(logits: Tensor, targets: Tensor, num_classes: int) -> None:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must align with the batch dimension")
    if num_classes <= 1 or logits.shape[1] != num_classes:
        raise ValueError("num_classes must match logits and exceed one")


__all__ = [
    "MarginProfile",
    "true_class_margin",
    "class_margin_profile",
    "class_cross_entropy_profile",
    "class_accuracy_profile",
    "class_brier_profile",
    "class_top_label_ece_profile",
    "profile_to_wire",
    "profile_from_wire",
]
