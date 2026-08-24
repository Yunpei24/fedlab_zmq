"""Lossless conversion between model-update dictionaries and flat vectors."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TensorLayout:
    """Shape information needed to reconstruct a flattened update."""

    keys: tuple[str, ...]
    shapes: tuple[torch.Size, ...]
    sizes: tuple[int, ...]
    dtypes: tuple[torch.dtype, ...]


def flatten_update(
    update: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, TensorLayout]:
    """Flatten floating tensors in deterministic key order.

    FL model deltas are expected to be floating point.  Integer buffers such
    as ``num_batches_tracked`` are skipped because Euclidean robust rules are
    not meaningful for them and PyTorch cannot average integer counters.
    """

    items = [
        (k, v.detach())
        for k, v in update.items()
        if v.is_floating_point() and not k.endswith("num_batches_tracked")
    ]
    if not items:
        raise ValueError("An update must contain at least one floating tensor")
    keys = tuple(k for k, _ in items)
    tensors = [v.reshape(-1).to(dtype=torch.float64, device="cpu") for _, v in items]
    layout = TensorLayout(
        keys=keys,
        shapes=tuple(v.shape for _, v in items),
        sizes=tuple(v.numel() for _, v in items),
        dtypes=tuple(v.dtype for _, v in items),
    )
    return torch.cat(tensors), layout


def unflatten_update(vector: torch.Tensor, layout: TensorLayout) -> OrderedDict:
    """Restore a vector produced by :func:`flatten_update`."""

    if vector.numel() != sum(layout.sizes):
        raise ValueError("Vector length does not match the update layout")
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    offset = 0
    for key, shape, size, dtype in zip(
        layout.keys, layout.shapes, layout.sizes, layout.dtypes
    ):
        out[key] = vector[offset : offset + size].reshape(shape).to(dtype=dtype)
        offset += size
    return out


def stack_updates(
    updates: list[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, TensorLayout]:
    """Return an ``(n_clients, n_parameters)`` matrix and common layout."""

    if not updates:
        raise ValueError("At least one update is required")
    first, layout = flatten_update(updates[0])
    rows = [first]
    for update in updates[1:]:
        row, other = flatten_update(update)
        if other.keys != layout.keys or other.shapes != layout.shapes:
            raise ValueError("All client updates must share keys and shapes")
        rows.append(row)
    return torch.stack(rows), layout
