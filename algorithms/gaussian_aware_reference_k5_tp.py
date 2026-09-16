r"""Strictly transcript-past predictor primitives for G0g-K5-TP.

The functions in this module never accept a current-round upload, clean target,
noise draw, attack label, or privileged conditional oracle as an inference
input.  For a round ``t`` they construct six vector features from rounds
``t-4,...,t-1`` only and apply one set of scalar coefficients shared by every
coordinate.  The final vector is projected on the public ball ``B_G``.

Privileged K4c-CH targets are allowed only in the *offline fitting* routine.
They are labels for a synthetic development study, not runtime features.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from algorithms.gaussian_aware_reference_k4b import _stable_clip_rows

FEATURE_NAMES = (
    "rolling_accepted_mean",
    "accepted_mean_t_minus_1",
    "accepted_mean_delta_t_minus_1_t_minus_2",
    "fixed_denominator_direction_t_minus_1",
    "fixed_denominator_delta_t_minus_1_t_minus_2",
    "rolling_fixed_denominator_direction",
)


@torch.no_grad()
def _modified_gram_schmidt_rank(
    matrix: torch.Tensor,
) -> tuple[int, dict[str, Any]]:
    """Return a numerical column rank using MPS-supported primitives only.

    Every non-zero centered column is first normalized to unit Euclidean norm,
    then orthogonalized twice against the accepted basis.  The public relative
    threshold is ``max(rows, columns) * eps(dtype)``.  The double pass reduces
    the loss of orthogonality in float32 without calling QR/SVD/eigendecomposition,
    which are unavailable on MPS in the production environment.
    """

    if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2:
        raise ValueError("matrix must have shape (rows,columns)")
    if not matrix.is_floating_point() or not bool(torch.isfinite(matrix).all()):
        raise ValueError("matrix must be finite floating point")
    rows, columns = (int(value) for value in matrix.shape)
    if rows < 1 or columns < 1:
        raise ValueError("matrix must be non-empty")
    relative_tolerance = float(max(rows, columns)) * torch.finfo(matrix.dtype).eps
    basis: list[torch.Tensor] = []
    original_norms: list[float] = []
    residual_norms: list[float] = []
    for index in range(columns):
        column = matrix[:, index]
        original_norm = float(torch.linalg.vector_norm(column).item())
        original_norms.append(original_norm)
        if original_norm == 0.0:
            residual_norms.append(0.0)
            continue
        residual = column / original_norm
        for _ in range(2):
            for vector in basis:
                residual = residual - torch.dot(vector, residual) * vector
        residual_norm = float(torch.linalg.vector_norm(residual).item())
        residual_norms.append(residual_norm)
        if residual_norm > relative_tolerance:
            basis.append(residual / residual_norm)
    return len(basis), {
        "method": "two_pass_modified_gram_schmidt_on_unit_norm_columns",
        "relative_tolerance": relative_tolerance,
        "original_column_norms": original_norms,
        "normalized_residual_norms": residual_norms,
        "compute_device": str(matrix.device),
        "compute_dtype": str(matrix.dtype),
    }


@torch.no_grad()
def _condition_number_infinity(system: torch.Tensor) -> float:
    """Compute ``||A||_inf ||A^{-1}||_inf`` without an MPS eigensolver."""

    if not isinstance(system, torch.Tensor) or system.ndim != 2:
        raise ValueError("system must be a matrix")
    rows, columns = (int(value) for value in system.shape)
    if rows < 1 or rows != columns:
        raise ValueError("system must be non-empty and square")
    identity = torch.eye(rows, dtype=system.dtype, device=system.device)
    inverse = torch.linalg.solve(system, identity)
    system_norm = torch.max(torch.sum(torch.abs(system), dim=1))
    inverse_norm = torch.max(torch.sum(torch.abs(inverse), dim=1))
    value = float((system_norm * inverse_norm).item())
    if not math.isfinite(value):
        raise RuntimeError("regularized-system condition number is not finite")
    return value


def _finite_history(
    clipped_residual_history: torch.Tensor,
    accepted_gate_history: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    history = clipped_residual_history
    if not isinstance(history, torch.Tensor) or history.ndim != 3:
        raise ValueError("clipped_residual_history must have shape (L,n,d)")
    if not history.is_floating_point() or not bool(torch.isfinite(history).all()):
        raise ValueError("clipped_residual_history must be finite floating point")
    length, clients, dimension = (int(value) for value in history.shape)
    if length != 4 or clients < 1 or dimension < 1:
        raise ValueError("K5-TP requires exactly L=4 past rounds")
    gates = torch.as_tensor(
        accepted_gate_history, device=history.device, dtype=history.dtype
    )
    if gates.shape != (length, clients) or not bool(torch.isfinite(gates).all()):
        raise ValueError("accepted_gate_history must be finite with shape (4,n)")
    tolerance = 64.0 * torch.finfo(history.dtype).eps
    if bool((gates < -tolerance).any()) or bool((gates > 1.0 + tolerance).any()):
        raise ValueError("accepted gates must lie in [0,1]")
    return history, gates.clamp(0.0, 1.0), length, clients, dimension


@torch.no_grad()
def transcript_past_feature_dictionary(
    clipped_residual_history: torch.Tensor,
    accepted_gate_history: torch.Tensor,
    *,
    minimum_accepted_mass: float,
    influence_cap: float,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Return the fixed six-vector K5-TP dictionary for round ``t``.

    Rows of every input are ordered chronologically as ``t-4,...,t-1``.
    For a prior round ``r`` the accepted mean is

    .. math::

       m_r=\operatorname{Clip}_G\!\left(
       \frac{\sum_i g_{i,r}c_{i,r}}
       {\max\{m_{\min},\sum_i g_{i,r}\}}\right).

    Returned feature rows are the K4b rolling predictor, ``m_{t-1}``, its
    one-step difference, the fixed-denominator direction
    ``u_{t-1}=n^{-1}\sum_i g_{i,t-1}c_{i,t-1}``, its one-step difference,
    and the four-round mean of ``u_r``.  Every row is clipped at ``G``.
    """

    history, gates, _, clients, dimension = _finite_history(
        clipped_residual_history, accepted_gate_history
    )
    minimum = float(minimum_accepted_mass)
    cap = float(influence_cap)
    if not math.isfinite(minimum) or not 0.0 < minimum <= float(clients):
        raise ValueError("minimum_accepted_mass must lie in (0,n]")
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("influence_cap must be finite and positive")

    tolerance = 64.0 * torch.finfo(history.dtype).eps
    residual_norms = torch.linalg.vector_norm(history, dim=2)
    if bool((residual_norms > cap + tolerance * max(1.0, cap)).any()):
        raise ValueError("past residual rows must already be clipped at G")
    masses = gates.sum(dim=1)
    denominators = torch.maximum(masses, torch.full_like(masses, minimum))
    raw_means = torch.sum(gates[:, :, None] * history, dim=1) / denominators[:, None]
    _, means = _stable_clip_rows(raw_means, cap, name="K5-TP accepted means")
    # Input order is t-4,...,t-1.
    rolling_raw = means.mean(dim=0)
    _, rolling = _stable_clip_rows(
        rolling_raw[None, :], cap, name="K5-TP rolling accepted mean"
    )
    accepted_newest = means[-1]
    accepted_delta = means[-1] - means[-2]
    numerators = torch.sum(gates[:, :, None] * history, dim=1)
    fixed_directions = numerators / float(clients)
    newest_fixed = fixed_directions[-1]
    fixed_delta = fixed_directions[-1] - fixed_directions[-2]
    rolling_fixed = fixed_directions.mean(dim=0)
    _, features = _stable_clip_rows(
        torch.stack(
            (
                rolling[0],
                accepted_newest,
                accepted_delta,
                newest_fixed,
                fixed_delta,
                rolling_fixed,
            )
        ),
        cap,
        name="K5-TP feature dictionary",
    )
    if features.shape != (len(FEATURE_NAMES), dimension):
        raise RuntimeError("K5-TP feature schema changed unexpectedly")
    if not bool(torch.isfinite(features).all()):
        raise RuntimeError("K5-TP feature construction produced non-finite values")
    if not return_diagnostics:
        return features
    return features, {
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "history_length": 4,
        "most_recent_source_lag": 1,
        "oldest_source_lag": 4,
        "uses_current_round_input": False,
        "accepted_mass_chronological": masses.detach().cpu().tolist(),
        "denominator_chronological": denominators.detach().cpu().tolist(),
        "feature_norms": torch.linalg.vector_norm(features, dim=1)
        .detach()
        .cpu()
        .tolist(),
        "feature_cap": cap,
        "fixed_denominator_direction_norm": float(torch.linalg.vector_norm(features[3]).item()),
        "fixed_denominator_delta_norm": float(
            torch.linalg.vector_norm(features[4]).item()
        ),
    }


def _validate_fit_inputs(
    feature_tensor: torch.Tensor,
    targets: torch.Tensor,
    aggregate_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    features = feature_tensor
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise ValueError("feature_tensor must have shape (H,J,d)")
    if not features.is_floating_point() or not bool(torch.isfinite(features).all()):
        raise ValueError("feature_tensor must be finite floating point")
    histories, feature_count, dimension = (int(value) for value in features.shape)
    if feature_count < 1 or histories < 1 or dimension < 1:
        raise ValueError("feature_tensor must contain at least one feature")
    labels = torch.as_tensor(targets, device=features.device, dtype=features.dtype)
    weights = torch.as_tensor(
        aggregate_weights, device=features.device, dtype=features.dtype
    ).reshape(-1)
    if labels.shape != (histories, dimension) or not bool(torch.isfinite(labels).all()):
        raise ValueError("targets must be finite with shape (H,d)")
    if weights.shape != (histories,) or not bool(torch.isfinite(weights).all()):
        raise ValueError("aggregate_weights must be finite with shape (H,)")
    if bool((weights <= 0.0).any()):
        raise ValueError("aggregate_weights must be strictly positive")
    return features, labels, weights, histories, feature_count, dimension


@torch.no_grad()
def training_feature_scales(
    feature_tensor: torch.Tensor,
    *,
    rms_floor: float,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Compute feature-wise RMS scales from one fitting split only."""

    if not isinstance(feature_tensor, torch.Tensor) or feature_tensor.ndim != 3:
        raise ValueError("feature_tensor must have shape (H,J,d)")
    if not feature_tensor.is_floating_point() or not bool(
        torch.isfinite(feature_tensor).all()
    ):
        raise ValueError("feature_tensor must be finite floating point")
    floor = float(rms_floor)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("rms_floor must be finite and strictly positive")
    work = feature_tensor
    raw = torch.sqrt(torch.mean(work.square(), dim=(0, 2)))
    scales = torch.clamp(raw, min=floor)
    result = scales.to(dtype=feature_tensor.dtype)
    if not return_diagnostics:
        return result
    flattened = work.permute(0, 2, 1).reshape(-1, int(work.shape[1]))
    centered = flattened - flattened.mean(dim=0)
    rank, rank_diagnostics = _modified_gram_schmidt_rank(centered)
    centered_rms = torch.sqrt(
        torch.mean(centered.square(), dim=0)
    )
    return result, {
        "raw_rms": raw.detach().cpu().tolist(),
        "deployed_scales": scales.detach().cpu().tolist(),
        "rms_floor": floor,
        "floor_active_count": int(torch.sum(raw < floor).item()),
        "flattened_design_rank": rank,
        "feature_count": int(work.shape[1]),
        "minimum_centered_rms": float(torch.min(centered_rms).item()),
        "centered_rms": centered_rms.detach().cpu().tolist(),
        "rank_diagnostics": rank_diagnostics,
    }


@torch.no_grad()
def fit_shared_scalar_ridge(
    feature_tensor: torch.Tensor,
    targets: torch.Tensor,
    aggregate_weights: torch.Tensor,
    *,
    ridge_lambda: float,
    feature_scales: torch.Tensor | Sequence[float] | None = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Fit the six shared scalar coefficients with a normalized ridge loss.

    The optimized objective is

    .. math::

       \frac{\sum_h w_h\|V_h^\top\theta-y_h\|_2^2}
            {d\sum_h w_h}+\lambda\|\theta\|_2^2,

    with ``w_h=(R_h/n)^2`` supplied by the caller.  The normalization makes
    the preregistered lambda grid independent of the number of histories and
    ambient dimension.  The solve stays on the input device and in the input
    floating dtype; the production runner requires float32 on MPS and audits
    the regularized condition number and normal-equation residual.
    """

    features, labels, weights, histories, feature_count, dimension = (
        _validate_fit_inputs(feature_tensor, targets, aggregate_weights)
    )
    regularization = float(ridge_lambda)
    if not math.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("ridge_lambda must be finite and strictly positive")
    x = features
    y = labels
    w = weights
    if feature_scales is None:
        scales = torch.ones(feature_count, dtype=x.dtype, device=x.device)
    else:
        scales = torch.as_tensor(
            feature_scales, dtype=x.dtype, device=x.device
        ).reshape(-1)
        if scales.shape != (feature_count,) or not bool(torch.isfinite(scales).all()):
            raise ValueError("feature_scales must be finite with shape (J,)")
        if bool((scales <= 0.0).any()):
            raise ValueError("feature_scales must be strictly positive")
    x = x / scales[None, :, None]
    normalizer = float(dimension) * torch.sum(w)
    gram = torch.einsum("h,hjd,hkd->jk", w, x, x) / normalizer
    rhs = torch.einsum("h,hjd,hd->j", w, x, y) / normalizer
    system = gram + regularization * torch.eye(
        feature_count, dtype=x.dtype, device=x.device
    )
    coefficients = torch.linalg.solve(system, rhs)
    if not bool(torch.isfinite(coefficients).all()):
        raise RuntimeError("ridge solution is not finite")
    predictions = torch.einsum("j,hjd->hd", coefficients, x)
    residual = predictions - y
    weighted_mse = float(
        (torch.sum(w[:, None] * residual.square()) / normalizer).item()
    )
    penalty = float((regularization * torch.sum(coefficients.square())).item())
    residual_normal_equation = system @ coefficients - rhs
    residual_relative = float(
        (
            torch.linalg.vector_norm(residual_normal_equation)
            / torch.clamp(torch.linalg.vector_norm(rhs), min=torch.finfo(rhs.dtype).eps)
        ).item()
    )
    condition_number = _condition_number_infinity(system)
    result = coefficients
    if not return_diagnostics:
        return result
    return result, {
        "feature_names": list(FEATURE_NAMES[:feature_count]),
        "histories": histories,
        "dimension": dimension,
        "ridge_lambda": regularization,
        "weighted_mse_before_projection": weighted_mse,
        "ridge_penalty": penalty,
        "objective": weighted_mse + penalty,
        "coefficient_l2_norm": float(torch.linalg.vector_norm(coefficients).item()),
        "condition_number_regularized_system": condition_number,
        "condition_number_norm": "infinity_exact_via_solve",
        "normal_equation_relative_residual": residual_relative,
        "fit_dtype": str(features.dtype),
        "fit_device": str(features.device),
        "objective_normalization": "dimension_times_sum_aggregate_weights",
        "feature_scales": scales.detach().cpu().tolist(),
        "normalized_gram": gram.detach().cpu().tolist(),
        "normalized_rhs": rhs.detach().cpu().tolist(),
        "aggregate_weight_sum": float(torch.sum(w).item()),
    }


@torch.no_grad()
def transcript_past_predictor(
    features: torch.Tensor,
    coefficients: torch.Tensor | Sequence[float],
    *,
    influence_cap: float,
    feature_scales: torch.Tensor | Sequence[float] | None = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Apply frozen coefficients and project the predicted vector on ``B_G``."""

    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError("features must have shape (J,d)")
    feature_count = int(features.shape[0])
    if feature_count < 1:
        raise ValueError("features must contain at least one row")
    if not features.is_floating_point() or not bool(torch.isfinite(features).all()):
        raise ValueError("features must be finite floating point")
    theta = torch.as_tensor(
        coefficients, device=features.device, dtype=features.dtype
    ).reshape(-1)
    if theta.shape != (feature_count,) or not bool(torch.isfinite(theta).all()):
        raise ValueError("coefficients must be finite with shape (J,)")
    if feature_scales is None:
        scales = torch.ones_like(theta)
    else:
        scales = torch.as_tensor(
            feature_scales, device=features.device, dtype=features.dtype
        ).reshape(-1)
        if scales.shape != (feature_count,) or not bool(torch.isfinite(scales).all()):
            raise ValueError("feature_scales must be finite with shape (J,)")
        if bool((scales <= 0.0).any()):
            raise ValueError("feature_scales must be strictly positive")
    cap = float(influence_cap)
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("influence_cap must be finite and positive")
    raw = torch.einsum("j,jd->d", theta, features / scales[:, None])
    _, projected = _stable_clip_rows(raw[None, :], cap, name="K5-TP predictor")
    predictor = projected[0]
    if not return_diagnostics:
        return predictor
    return predictor, {
        "feature_names": list(FEATURE_NAMES[:feature_count]),
        "coefficients": theta.detach().cpu().tolist(),
        "feature_scales": scales.detach().cpu().tolist(),
        "raw_predictor_norm": float(torch.linalg.vector_norm(raw).item()),
        "predictor_norm": float(torch.linalg.vector_norm(predictor).item()),
        "predictor_cap": cap,
        "projection_active": bool(torch.linalg.vector_norm(raw).item() > cap),
        "observable_past_only": True,
        "uses_current_round_input": False,
    }


def forbidden_current_field_count(inference_payload: Mapping[str, Any]) -> int:
    """Audit a serialized inference payload for explicitly forbidden fields."""

    forbidden_fragments = (
        "current_upload",
        "current_clean",
        "current_target",
        "current_noise",
        "current_attack",
        "byzantine_label",
        "semi_oracle_target",
        "pointwise_oracle",
    )
    return sum(
        any(fragment in str(key).lower() for fragment in forbidden_fragments)
        for key in inference_payload
    )


__all__ = [
    "FEATURE_NAMES",
    "fit_shared_scalar_ridge",
    "forbidden_current_field_count",
    "training_feature_scales",
    "transcript_past_feature_dictionary",
    "transcript_past_predictor",
]
