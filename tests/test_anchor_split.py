from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from datasets.anchor_split import split_train_anchor_loader


def test_split_is_disjoint_deterministic_and_preserves_rare_classes() -> None:
    x = torch.arange(60).float().unsqueeze(1)
    y = torch.tensor([0] * 30 + [1] * 20 + [2] * 10)
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)
    train_a, anchor_a = split_train_anchor_loader(
        loader,
        anchor_fraction=0.2,
        seed=42,
        max_train_samples=40,
        max_anchor_samples=12,
        min_train_samples=10,
    )
    train_b, anchor_b = split_train_anchor_loader(
        loader,
        anchor_fraction=0.2,
        seed=42,
        max_train_samples=40,
        max_anchor_samples=12,
        min_train_samples=10,
    )
    assert anchor_a is not None and anchor_b is not None
    train_ids_a = set(train_a.dataset.indices)
    anchor_ids_a = set(anchor_a.dataset.indices)
    assert train_ids_a.isdisjoint(anchor_ids_a)
    assert train_a.dataset.indices == train_b.dataset.indices
    assert anchor_a.dataset.indices == anchor_b.dataset.indices
    anchor_labels = {int(y[index]) for index in anchor_ids_a}
    assert anchor_labels == {0, 1, 2}


def test_split_seed_is_independent_from_training_shuffle_seed() -> None:
    x = torch.arange(80).float().unsqueeze(1)
    y = torch.tensor([0] * 40 + [1] * 40)
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)
    train_a, anchor_a = split_train_anchor_loader(
        loader,
        anchor_fraction=0.2,
        seed=101,
        shuffle_seed=201,
    )
    train_b, anchor_b = split_train_anchor_loader(
        loader,
        anchor_fraction=0.2,
        seed=101,
        shuffle_seed=202,
    )
    assert anchor_a is not None and anchor_b is not None
    assert train_a.dataset.indices == train_b.dataset.indices
    assert anchor_a.dataset.indices == anchor_b.dataset.indices
    assert train_a.generator is not None and train_b.generator is not None
    assert train_a.generator.initial_seed() == 201
    assert train_b.generator.initial_seed() == 202
