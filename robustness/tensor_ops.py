"""Lossless conversion between model-update dictionaries and flat vectors."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

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


def score_subspace(
    vectors: torch.Tensor,
    layout: TensorLayout,
    *,
    mode: str = "full",
    dimension: int | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select a public, data-independent view used only to compute scores.

    Aggregation still applies the resulting scalar weights to the complete
    update vectors.  Keeping the selector public is important: a subspace
    chosen from the current private uploads would create an additional
    data-dependent mechanism and make both the causal interpretation and the
    privacy transcript harder to audit.

    Supported modes are:

    ``full``
        Use every floating coordinate.

    ``last_layer``
        Use every tensor belonging to the final parameter prefix in the
        model state (for example the final linear layer's weight and bias).

    ``public_coordinates``
        Use ``dimension`` coordinates selected once by a fixed public seed.
        This is a coordinate subspace, not a learned projection.
    """

    if vectors.ndim != 2:
        raise ValueError("score_subspace expects an (n_clients, dimension) matrix")
    total_dimension = int(vectors.shape[1])
    if total_dimension != sum(layout.sizes):
        raise ValueError("Vector dimension does not match TensorLayout")

    resolved = str(mode).strip().lower()
    if resolved == "full":
        selected = vectors
        selected_keys = list(layout.keys)
    elif resolved == "last_layer":
        if not layout.keys:
            raise ValueError("last_layer score space requires a non-empty layout")
        last_prefix = layout.keys[-1].rsplit(".", 1)[0]
        blocks: list[torch.Tensor] = []
        selected_keys = []
        offset = 0
        for key, size in zip(layout.keys, layout.sizes):
            if key == last_prefix or key.startswith(f"{last_prefix}."):
                blocks.append(vectors[:, offset : offset + size])
                selected_keys.append(key)
            offset += size
        if not blocks:
            raise ValueError(f"No tensors found for final prefix {last_prefix!r}")
        selected = torch.cat(blocks, dim=1)
    elif resolved == "public_coordinates":
        if dimension is None:
            raise ValueError("public_coordinates requires score_subspace_dimension")
        requested = int(dimension)
        if requested < 1 or requested > total_dimension:
            raise ValueError(
                "score_subspace_dimension must lie between 1 and the full dimension"
            )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        indices = torch.randperm(total_dimension, generator=generator)[:requested]
        indices = indices.sort().values.to(vectors.device)
        selected = vectors.index_select(1, indices)
        selected_keys = [f"public_coordinates(seed={int(seed)})"]
    else:
        raise ValueError(
            "score_subspace_mode must be full, last_layer or public_coordinates"
        )

    return selected, {
        "score_subspace_mode": resolved,
        "score_subspace_dimension": int(selected.shape[1]),
        "score_subspace_full_dimension": total_dimension,
        "score_subspace_fraction": float(selected.shape[1] / total_dimension),
        "score_subspace_seed": int(seed) if resolved == "public_coordinates" else None,
        "score_subspace_keys": selected_keys,
    }
