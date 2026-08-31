"""Pedagogical implementations of the robust rules used around FAR.

References and terminology
--------------------------
``CM`` and ``trMean`` are coordinate-wise median and trimmed mean. ``NNM``
is nearest-neighbour mixing: every update is first replaced by the average of
its closest ``n-f`` updates. ``RFA`` is the smoothed Weiszfeld geometric
median. ``NBS`` screens the largest update norms. ``CMLS`` is the linear-
scalarisation extension of coordinate median: the robust reference is kept,
while original updates are reintroduced with inverse-distance penalties.

These functions implement research baselines, not a claim that any chosen
``f`` is valid for an unknown deployment.  Experiments must set the assumed
number of Byzantine clients explicitly.

The module also contains the two references used by the theoretical SC-FAR
study:

``centered_clipping``
    A one-step, public-anchor centered-clipping reference.  Unlike an
    iterative estimator whose initial point is computed from the current
    cohort, its replace-one stability follows directly from non-expansiveness
    of Euclidean projection.

``regularized_huber_reference``
    A deliberately finite-step ablation.  The public number of gradient
    iterations is part of the mechanism, so the stability certificate applies
    to the value that is actually returned, not merely to an ideal optimizer.
"""

from __future__ import annotations

import math

import torch

from .tensor_ops import stack_updates, unflatten_update


def _validate_vector_matrix(vectors: torch.Tensor) -> None:
    if vectors.ndim != 2 or vectors.shape[0] < 1:
        raise ValueError("vectors must have shape (n, d) with n >= 1")


def clip_l2(vector: torch.Tensor, radius: float) -> torch.Tensor:
    """Project one vector, or every row of a matrix, onto an L2 ball.

    The origin-centred projection is

    ``x * min(1, radius / ||x||_2)``.

    It is 1-Lipschitz.  This elementary property is the key ingredient in the
    replace-one stability proofs for both references below.
    """

    if radius <= 0:
        raise ValueError("radius must be positive")
    if vector.ndim == 1:
        norm = torch.linalg.vector_norm(vector)
        factor = (float(radius) / norm.clamp_min(1e-12)).clamp(max=1.0)
        return vector * factor
    if vector.ndim == 2:
        norms = torch.linalg.vector_norm(vector, dim=1, keepdim=True)
        factors = (float(radius) / norms.clamp_min(1e-12)).clamp(max=1.0)
        return vector * factors
    raise ValueError("clip_l2 expects a vector or a matrix of row vectors")


def centered_clipping(
    vectors: torch.Tensor,
    *,
    anchor: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    r"""One-step centered-clipping reference used by SC-FAR.

    .. math::

        F_{\mathrm{CC}}(U;r_0)
        = r_0 + \frac1n\sum_{i=1}^n
          \operatorname{Clip}_{\tau}(u_i-r_0).

    ``anchor`` must be fixed with respect to the *current* cohort.  In the
    SC-FAR implementation it is zero at round one and is subsequently derived
    only from the previous released (therefore already DP) aggregate.
    """

    _validate_vector_matrix(vectors)
    if tau <= 0:
        raise ValueError("tau must be positive")
    anchor = anchor.to(device=vectors.device, dtype=vectors.dtype).reshape(-1)
    if anchor.numel() != vectors.shape[1]:
        raise ValueError("anchor dimension must match the client vectors")
    centered = vectors - anchor
    return anchor + clip_l2(centered, tau).mean(dim=0)


def regularized_huber_reference(
    vectors: torch.Tensor,
    *,
    anchor: torch.Tensor,
    tau: float,
    gamma: float,
    num_steps: int,
    return_diagnostics: bool = False,
):
    r"""Finite-step regularized vector-Huber reference.

    The ideal objective is

    .. math::

        \frac1n\sum_i \rho_\tau(r-u_i)
        + \frac\gamma2\lVert r-r_0\rVert_2^2,

    whose gradient is

    .. math::

        \frac1n\sum_i\operatorname{Clip}_\tau(r-u_i)
        + \gamma(r-r_0).

    Starting at the public anchor, we execute exactly ``num_steps`` iterations
    with the public step size ``2 / (1 + 2*gamma)``.  For replace-one cohorts
    whose rows satisfy ``||u_i|| <= C``, the returned iterate has certificate

    ``delta_F <= 2*min(C,tau)/(gamma*n) * (1-rho**num_steps)``,

    where ``rho = 1/(1+2*gamma)``.  A tolerance-based early stopping rule is
    intentionally avoided because a fixed tolerance would add an error that
    need not decay as ``1/n``.
    """

    _validate_vector_matrix(vectors)
    if tau <= 0:
        raise ValueError("tau must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if num_steps < 1:
        raise ValueError("num_steps must be at least one")
    anchor = anchor.to(device=vectors.device, dtype=vectors.dtype).reshape(-1)
    if anchor.numel() != vectors.shape[1]:
        raise ValueError("anchor dimension must match the client vectors")

    point = anchor.clone()
    step_size = 2.0 / (1.0 + 2.0 * float(gamma))
    contraction = 1.0 / (1.0 + 2.0 * float(gamma))
    for _ in range(int(num_steps)):
        data_gradient = clip_l2(point[None, :] - vectors, tau).mean(dim=0)
        gradient = data_gradient + float(gamma) * (point - anchor)
        point = point - step_size * gradient

    if not return_diagnostics:
        return point
    final_data_gradient = clip_l2(point[None, :] - vectors, tau).mean(dim=0)
    final_gradient = final_data_gradient + float(gamma) * (point - anchor)
    return point, {
        "huber_step_size": float(step_size),
        "huber_contraction": float(contraction),
        "huber_gradient_residual": float(torch.linalg.vector_norm(final_gradient)),
        "huber_num_steps": int(num_steps),
    }


def coordinate_median(vectors: torch.Tensor) -> torch.Tensor:
    """Coordinate-wise median of an ``(n,d)`` matrix."""

    return torch.quantile(vectors, 0.5, dim=0, interpolation="midpoint")


def trimmed_mean(vectors: torch.Tensor, f: int) -> torch.Tensor:
    """Coordinate-wise mean after removing ``f`` values at each tail."""

    n = vectors.shape[0]
    if f < 0 or 2 * f >= n:
        raise ValueError(f"trimmed mean needs 0 <= 2f < n; got f={f}, n={n}")
    ordered = torch.sort(vectors, dim=0).values
    kept = ordered[f : n - f] if f else ordered
    return kept.mean(dim=0)


def nearest_neighbor_mixing(vectors: torch.Tensor, f: int) -> torch.Tensor:
    """NNM pre-aggregation from heterogeneous Byzantine-robust learning.

    For each client, average the ``n-f`` closest submitted updates, including
    the update itself.  A coordinate median or trimmed mean is then applied to
    these mixed vectors by the caller.
    """

    n = vectors.shape[0]
    keep = n - f
    if f < 0 or keep <= 0:
        raise ValueError(f"NNM needs 0 <= f < n; got f={f}, n={n}")
    distances = torch.cdist(vectors, vectors, p=2)
    neighbours = distances.topk(keep, largest=False, dim=1).indices
    return torch.stack([vectors[idx].mean(dim=0) for idx in neighbours])


def geometric_median(
    vectors: torch.Tensor,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    smoothing: float = 1e-8,
) -> torch.Tensor:
    """RFA/geometric median computed with a smoothed Weiszfeld iteration."""

    point = vectors.mean(dim=0)
    for _ in range(max_iter):
        distances = torch.linalg.vector_norm(vectors - point, dim=1).clamp_min(
            smoothing
        )
        weights = distances.reciprocal()
        candidate = (weights[:, None] * vectors).sum(dim=0) / weights.sum()
        if torch.linalg.vector_norm(candidate - point) <= tol:
            point = candidate
            break
        point = candidate
    return point


def norm_based_screening(
    vectors: torch.Tensor, screening_fraction: float
) -> torch.Tensor:
    """NBS: discard the largest norms and average the remaining updates."""

    if not 0.0 <= screening_fraction < 1.0:
        raise ValueError("screening_fraction must be in [0,1)")
    n = vectors.shape[0]
    keep = max(1, int(math.floor((1.0 - screening_fraction) * n)))
    indices = torch.linalg.vector_norm(vectors, dim=1).argsort()[:keep]
    return vectors[indices].mean(dim=0)


def cmls(
    vectors: torch.Tensor,
    *,
    alpha_trusted: float = 1.0,
    alpha_suspected: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Coordinate-Median Linear Scalarisation (CMLS).

    Coordinate median supplies a robust blended reference.  Each submitted
    vector is reintroduced with penalty

        ``alpha_suspected * min(1, 1 / ||g_i - g_ref||_2)``.

    The reference receives weight ``alpha_trusted`` and all weights are
    normalised before the final convex combination.  This follows the CMLS
    interpretation for a robust rule that returns a blended vector rather
    than a subset of trusted client indices.
    """

    if alpha_trusted <= 0 or not 0 <= alpha_suspected <= 1:
        raise ValueError("Need alpha_trusted>0 and alpha_suspected in [0,1]")
    reference = coordinate_median(vectors)
    distances = torch.linalg.vector_norm(vectors - reference, dim=1)
    penalties = alpha_suspected * torch.minimum(
        torch.ones_like(distances), distances.clamp_min(eps).reciprocal()
    )
    numerator = alpha_trusted * reference + (penalties[:, None] * vectors).sum(0)
    denominator = alpha_trusted + penalties.sum()
    return numerator / denominator


_ALIASES = {
    "cm": "coordinate_median",
    "median": "coordinate_median",
    "trmean": "trimmed_mean",
    "rfa": "geometric_median",
    "nbs": "norm_based_screening",
    "cm_nnm": "cm_nnm",
    "trmean_nnm": "trmean_nnm",
    "cm(nnm)": "cm_nnm",
    "trmean(nnm)": "trmean_nnm",
    "cmls": "cmls",
    "cc": "centered_clipping",
    "f_cc": "centered_clipping",
    "huber": "regularized_huber",
    "huber_regularized": "regularized_huber",
}


def aggregate_vectors(vectors: torch.Tensor, method: str, **kwargs) -> torch.Tensor:
    """Dispatch a robust rule by the names used in experiment YAML files."""

    method = _ALIASES.get(method.lower(), method.lower())
    f = int(kwargs.get("num_byzantine", kwargs.get("f", 0)))
    if method == "mean":
        return vectors.mean(dim=0)
    if method == "coordinate_median":
        return coordinate_median(vectors)
    if method == "trimmed_mean":
        return trimmed_mean(vectors, f=f)
    if method == "cm_nnm":
        return coordinate_median(nearest_neighbor_mixing(vectors, f=f))
    if method == "trmean_nnm":
        return trimmed_mean(nearest_neighbor_mixing(vectors, f=f), f=f)
    if method == "geometric_median":
        return geometric_median(
            vectors,
            max_iter=int(kwargs.get("max_iter", 100)),
            tol=float(kwargs.get("tol", 1e-6)),
            smoothing=float(kwargs.get("smoothing", 1e-8)),
        )
    if method == "norm_based_screening":
        fraction = kwargs.get("screening_fraction")
        if fraction is None:
            fraction = f / max(vectors.shape[0], 1)
        return norm_based_screening(vectors, float(fraction))
    if method == "cmls":
        return cmls(
            vectors,
            alpha_trusted=float(kwargs.get("alpha_trusted", 1.0)),
            alpha_suspected=float(kwargs.get("alpha_suspected", 1.0)),
        )
    if method == "centered_clipping":
        anchor = kwargs.get("anchor")
        tau = kwargs.get("tau")
        if anchor is None or tau is None:
            raise ValueError("centered_clipping requires anchor and tau")
        return centered_clipping(vectors, anchor=anchor, tau=float(tau))
    if method == "regularized_huber":
        anchor = kwargs.get("anchor")
        tau = kwargs.get("tau")
        if anchor is None or tau is None:
            raise ValueError("regularized_huber requires anchor and tau")
        return regularized_huber_reference(
            vectors,
            anchor=anchor,
            tau=float(tau),
            gamma=float(kwargs.get("gamma", 1.0)),
            num_steps=int(kwargs.get("num_steps", 10)),
        )
    raise ValueError(
        f"Unknown robust aggregator {method!r}. Available: mean, cm, trmean, "
        "cm_nnm, trmean_nnm, rfa, nbs, cmls, centered_clipping, "
        "regularized_huber"
    )


def aggregate_updates(
    updates: list[dict[str, torch.Tensor]], method: str, **kwargs
) -> dict[str, torch.Tensor]:
    """Apply a robust rule to model-update dictionaries."""

    vectors, layout = stack_updates(updates)
    result = aggregate_vectors(vectors, method, **kwargs)
    return dict(unflatten_update(result, layout))
