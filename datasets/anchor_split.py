"""Deterministic train/anchor splits for algorithms with auxiliary reports.

The split is performed inside each already-partitioned client dataset.  Both
the ordinary training baselines and DMD therefore see the same reduced
training shard; only algorithms that need an auxiliary profile consume the
held-out anchor loader.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


def dataset_targets(dataset: Dataset) -> np.ndarray:
    """Return labels aligned with local positions, including nested Subsets."""

    if isinstance(dataset, Subset):
        parent = dataset_targets(dataset.dataset)
        return parent[np.asarray(dataset.indices, dtype=np.int64)]
    for name in ("targets", "labels", "y"):
        value = getattr(dataset, name, None)
        if value is not None:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            array = np.asarray(value).reshape(-1)
            if len(array) == len(dataset):
                return array.astype(np.int64, copy=False)
    # Generic datasets (including TensorDataset) do not expose a common label
    # attribute.  This fallback is setup-only and never enters the train loop.
    labels = []
    for index in range(len(dataset)):
        item = dataset[index]
        if not isinstance(item, Sequence) or len(item) < 2:
            raise TypeError("dataset items must expose (features, target)")
        target = item[1]
        labels.append(int(target.item()) if torch.is_tensor(target) else int(target))
    return np.asarray(labels, dtype=np.int64)


def _stratified_cap(
    positions: list[int], labels: np.ndarray, cap: int | None, rng: np.random.Generator
) -> list[int]:
    if cap is None or cap <= 0 or len(positions) <= cap:
        return sorted(positions)
    by_class: dict[int, list[int]] = defaultdict(list)
    for position in positions:
        by_class[int(labels[position])].append(position)
    chosen: list[int] = []
    # Preserve at least one example for as many observed classes as the cap
    # allows, then fill the remaining capacity without replacement.
    classes = list(by_class)
    rng.shuffle(classes)
    for class_id in classes[:cap]:
        bucket = by_class[class_id]
        chosen.append(bucket[int(rng.integers(len(bucket)))])
    chosen_set = set(chosen)
    remaining = [position for position in positions if position not in chosen_set]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, cap - len(chosen))])
    return sorted(chosen)


def split_train_anchor_loader(
    loader: DataLoader,
    *,
    anchor_fraction: float,
    seed: int,
    shuffle_seed: int | None = None,
    max_train_samples: int | None = None,
    max_anchor_samples: int | None = None,
    min_train_samples: int = 1,
    anchor_batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader | None]:
    """Split a client loader deterministically and approximately stratified."""

    if not 0.0 <= anchor_fraction < 1.0:
        raise ValueError("anchor_fraction must lie in [0, 1)")
    if anchor_fraction == 0.0:
        return loader, None
    dataset = loader.dataset
    size = len(dataset)
    if size <= min_train_samples:
        raise ValueError(
            f"client shard has {size} samples, not enough for min_train_samples="
            f"{min_train_samples} plus an anchor set"
        )
    labels = dataset_targets(dataset)
    rng = np.random.default_rng(seed)
    target_anchor = max(1, int(round(anchor_fraction * size)))
    if max_anchor_samples is not None and max_anchor_samples > 0:
        target_anchor = min(target_anchor, int(max_anchor_samples))
    target_anchor = min(target_anchor, size - min_train_samples)

    by_class: dict[int, list[int]] = defaultdict(list)
    for position, label in enumerate(labels):
        by_class[int(label)].append(position)
    eligible = [class_id for class_id, bucket in by_class.items() if len(bucket) >= 2]
    rng.shuffle(eligible)
    anchor: list[int] = []
    # One anchor per eligible class first, so rare classes are not silently
    # erased when the requested anchor budget is large enough.
    for class_id in eligible[:target_anchor]:
        bucket = by_class[class_id]
        selected = bucket[int(rng.integers(len(bucket)))]
        anchor.append(selected)
    anchor_set = set(anchor)
    remaining = [position for position in range(size) if position not in anchor_set]
    rng.shuffle(remaining)
    anchor.extend(remaining[: max(0, target_anchor - len(anchor))])
    anchor_set = set(anchor)
    train = [position for position in range(size) if position not in anchor_set]
    train = _stratified_cap(train, labels, max_train_samples, rng)
    anchor = _stratified_cap(anchor, labels, max_anchor_samples, rng)

    common = {
        "num_workers": int(getattr(loader, "num_workers", 0)),
        "pin_memory": bool(getattr(loader, "pin_memory", False)),
    }
    # ``seed`` fixes which samples belong to train and anchor.  ``shuffle_seed``
    # is intentionally separate so several training replicates can reuse the
    # exact same client partition/split while varying mini-batch order.
    train_generator = torch.Generator().manual_seed(
        seed + 1 if shuffle_seed is None else int(shuffle_seed)
    )
    train_loader = DataLoader(
        Subset(dataset, train),
        batch_size=loader.batch_size,
        shuffle=True,
        drop_last=False,
        generator=train_generator,
        **common,
    )
    anchor_loader = DataLoader(
        Subset(dataset, anchor),
        batch_size=anchor_batch_size or loader.batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, anchor_loader


__all__ = ["dataset_targets", "split_train_anchor_loader"]
