"""Reference Byzantine attacks used in FAR-style evaluations."""

from __future__ import annotations

from copy import deepcopy
from statistics import NormalDist

import torch

from robustness.tensor_ops import stack_updates, unflatten_update


def _malicious_ids(
    client_updates: list[tuple[dict, dict, object]], config: dict
) -> list[int]:
    """Resolve Byzantine positions deterministically.

    ``client_ids`` takes precedence.  Otherwise the first ``num_byzantine``
    participating clients are malicious.  Explicit IDs are recommended in
    paper experiments because client sampling changes tuple positions.
    """

    explicit = config.get("client_ids")
    if explicit is not None:
        wanted = {int(x) for x in explicit}
        return [
            pos
            for pos, (_, meta, state) in enumerate(client_updates)
            if int(meta.get("client_id", getattr(state, "client_id", -1))) in wanted
        ]
    count = min(int(config.get("num_byzantine", 0)), len(client_updates))
    return list(range(count))


def _direction(vectors: torch.Tensor, mode: str) -> torch.Tensor:
    mean = vectors.mean(dim=0)
    if mode == "negative_mean":
        return -mean / torch.linalg.vector_norm(mean).clamp_min(1e-12)
    if mode == "sign":
        return -torch.sign(mean)
    if mode == "std":
        std = vectors.std(dim=0, unbiased=False)
        return -std / torch.linalg.vector_norm(std).clamp_min(1e-12)
    raise ValueError(f"Unknown malicious direction: {mode}")


def _alie(vectors: torch.Tensor, num_byzantine: int, z: float | None) -> torch.Tensor:
    """A Little Is Enough: shift the honest mean by coordinate-wise std.

    If ``z`` is omitted, use the conventional normal-quantile heuristic based
    on the expected number of coordinates that remain inside the benign range.
    The sign is chosen opposite to the honest mean because FedLab deltas follow
    the gradient/descent convention ``w_t - w_local``.
    """

    n = vectors.shape[0] + num_byzantine
    if z is None:
        s = max(1, (n // 2 + 1) - num_byzantine)
        probability = max(
            0.5, min(1.0 - 1e-6, (n - num_byzantine - s) / max(n - num_byzantine, 1))
        )
        z = NormalDist().inv_cdf(probability)
    mean = vectors.mean(dim=0)
    std = vectors.std(dim=0, unbiased=False)
    return mean - float(z) * std * torch.sign(mean).masked_fill(mean == 0, 1.0)


def _binary_search_stealth(
    vectors: torch.Tensor,
    *,
    criterion: str,
    direction_mode: str,
    max_scale: float,
    steps: int,
) -> torch.Tensor:
    """Construct the Min-Max or Min-Sum optimized poisoning vector."""

    reference = vectors.mean(dim=0)
    direction = _direction(vectors, direction_mode)
    pairwise_sq = torch.cdist(vectors, vectors, p=2).square()
    if criterion == "minmax":
        threshold = pairwise_sq.max()
    elif criterion == "minsum":
        threshold = pairwise_sq.sum(dim=1).max()
    else:
        raise ValueError("criterion must be minmax or minsum")

    low, high = 0.0, float(max_scale)
    best = reference.clone()
    for _ in range(int(steps)):
        scale = (low + high) / 2.0
        candidate = reference + scale * direction
        candidate_sq = torch.linalg.vector_norm(vectors - candidate, dim=1).square()
        score = candidate_sq.max() if criterion == "minmax" else candidate_sq.sum()
        if score <= threshold:
            best = candidate
            low = scale
        else:
            high = scale
    return best


def apply_attack(
    updates: list[dict[str, torch.Tensor]],
    malicious_indices: list[int],
    *,
    name: str,
    scale: float = 1.0,
    z: float | None = None,
    direction: str = "negative_mean",
    search_steps: int = 40,
    max_scale: float = 1e4,
) -> list[dict[str, torch.Tensor]]:
    """Return attacked copies of model updates.

    The five supported attacks match the FAR evaluation vocabulary:

    * ``sign_flip``/``bf``: negate each malicious client's own update;
    * ``ipm``: submit a scaled ascent direction based on the honest mean;
    * ``alie``: benign mean plus a stealthy coordinate-wise deviation;
    * ``minmax`` and ``minsum``: maximize a malicious displacement subject to
      pairwise-distance stealth constraints.
    """

    if not malicious_indices or name.lower() in {"none", "na", "no_attack"}:
        return [{k: v.clone() for k, v in update.items()} for update in updates]
    if len(malicious_indices) >= len(updates):
        raise ValueError("At least one honest update is required to craft this attack")
    matrix, layout = stack_updates(updates)
    bad = set(malicious_indices)
    honest = matrix[[i for i in range(len(updates)) if i not in bad]]
    attacked = matrix.clone()
    normalized = name.lower().replace("-", "_")

    if normalized in {"bf", "bit_flip", "sign_flip", "signflip"}:
        for idx in bad:
            attacked[idx] = -float(scale) * attacked[idx]
    elif normalized == "ipm":
        malicious = -float(scale) * honest.mean(dim=0)
        for idx in bad:
            attacked[idx] = malicious
    elif normalized == "alie":
        malicious = _alie(honest, len(bad), z)
        for idx in bad:
            attacked[idx] = malicious
    elif normalized in {"minmax", "min_max", "minsum", "min_sum"}:
        criterion = "minmax" if "max" in normalized else "minsum"
        malicious = _binary_search_stealth(
            honest,
            criterion=criterion,
            direction_mode=direction,
            max_scale=max_scale,
            steps=search_steps,
        )
        for idx in bad:
            attacked[idx] = malicious
    else:
        raise ValueError(
            f"Unknown attack {name!r}. Available: alie, minmax, minsum, bf, ipm"
        )
    results = []
    for original, row in zip(updates, attacked):
        restored = {k: v.clone() for k, v in original.items()}
        restored.update(unflatten_update(row, layout))
        results.append(dict(restored))
    return results


def apply_configured_attack(
    client_updates: list[tuple[dict, dict, object]], config: dict | None
) -> list[tuple[dict, dict, object]]:
    """Apply an attack config while preserving metadata and client states."""

    if not config or not config.get("enabled", True):
        return client_updates
    name = str(config.get("name", "none"))
    positions = _malicious_ids(client_updates, config)
    attacked_dicts = apply_attack(
        [update for update, _, _ in client_updates],
        positions,
        name=name,
        scale=float(config.get("scale", 1.0)),
        z=config.get("z"),
        direction=str(config.get("direction", "negative_mean")),
        search_steps=int(config.get("search_steps", 40)),
        max_scale=float(config.get("max_scale", 1e4)),
    )
    result = []
    bad = set(positions)
    for pos, ((_, metadata, state), attacked) in enumerate(
        zip(client_updates, attacked_dicts)
    ):
        new_meta = dict(metadata)
        new_meta["is_byzantine"] = pos in bad
        new_meta["attack_name"] = name if pos in bad else "none"
        result.append((attacked, new_meta, state))
    return result
