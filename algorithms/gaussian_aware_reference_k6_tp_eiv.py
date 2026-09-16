r"""Prototype primitives for the K6 transcript-past EIV predictor.

K6 estimates one scalar coefficient from two rolling means built exclusively
from accepted, already-private uploads from rounds preceding the deployment
round.  The errors-in-variables (EIV) correction removes the *post-chain*
delta-method covariance terms from the two empirical moments before a
seed-balanced median-of-means reduction is applied.

This module deliberately does not define an end-to-end training algorithm or
an experimental protocol.  In particular, it does not infer a covariance
before clipping and reuse it after clipping.  ``post_chain_delta_covariance``
computes Jacobians of the caller-supplied transformation ``Phi`` on the input
device, then constructs

.. math::

   V_X = J_X\Sigma_XJ_X^\top,\qquad
   C_{XY}=J_X\Sigma_{XY}J_Y^\top.

The Jacobian is accompanied by a fail-closed numerical differentiability
audit so a clipping/projection boundary is not silently treated as a smooth
point.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch

from algorithms.gaussian_aware_reference_k4b import _stable_clip_rows

TensorTransform = Callable[[torch.Tensor], torch.Tensor]


def _validate_history(
    clipped_residual_history: torch.Tensor,
    accepted_gate_history: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    history = clipped_residual_history
    if not isinstance(history, torch.Tensor):
        raise TypeError("clipped_residual_history must be a torch.Tensor")
    if history.ndim != 3 or min(history.shape) < 1:
        raise ValueError("clipped_residual_history must have shape (L,n,d)")
    if not history.is_floating_point() or not bool(torch.isfinite(history).all()):
        raise ValueError("clipped_residual_history must be finite floating point")
    length, clients, dimension = (int(value) for value in history.shape)
    gates = torch.as_tensor(
        accepted_gate_history, device=history.device, dtype=history.dtype
    )
    if gates.shape != (length, clients) or not bool(torch.isfinite(gates).all()):
        raise ValueError("accepted_gate_history must be finite with shape (L,n)")
    tolerance = 64.0 * torch.finfo(history.dtype).eps
    if bool((gates < -tolerance).any()) or bool((gates > 1.0 + tolerance).any()):
        raise ValueError("accepted gates must lie in [0,1]")
    return history, gates.clamp(0.0, 1.0), length, clients, dimension


@torch.no_grad()
def pooled_accepted_direction(
    clipped_residual_history: torch.Tensor,
    accepted_gate_history: torch.Tensor,
    *,
    minimum_pooled_accepted_mass: float,
    influence_cap: float,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Pool accepted residuals over a past window before applying one floor.

    For a window of ``L`` rounds, this primitive computes

    .. math::

       Z=\operatorname{Proj}_{B_G}\left(
       \frac{\sum_{r=1}^L\sum_{i=1}^n g_{i,r}c_{i,r}}
            {\max\{m_{\mathrm{pool,min}},
                    \sum_{r=1}^L\sum_{i=1}^n g_{i,r}\}}\right).

    Since ``0 <= g <= 1``, every residual lies in ``B_G``, and the deployed
    denominator is at least the total accepted mass, the unprojected vector is
    already a sub-convex combination in exact arithmetic.  The final
    projection is retained as a numerical and interface-level certificate.
    """

    history, gates, length, clients, _ = _validate_history(
        clipped_residual_history, accepted_gate_history
    )
    minimum = float(minimum_pooled_accepted_mass)
    cap = float(influence_cap)
    maximum_mass = float(length * clients)
    if not math.isfinite(minimum) or not 0.0 < minimum <= maximum_mass:
        raise ValueError("minimum_pooled_accepted_mass must lie in (0,L*n]")
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("influence_cap must be finite and strictly positive")
    tolerance = 64.0 * torch.finfo(history.dtype).eps * max(1.0, cap)
    residual_norms = torch.linalg.vector_norm(history, dim=2)
    if not bool(torch.isfinite(residual_norms).all()) or bool(
        (residual_norms > cap + tolerance).any()
    ):
        raise ValueError("past residuals must already lie in the public ball B_G")

    total_mass = gates.sum()
    denominator = torch.maximum(
        total_mass,
        torch.as_tensor(minimum, device=history.device, dtype=history.dtype),
    )
    numerator = torch.sum(gates[:, :, None] * history, dim=(0, 1))
    raw = numerator / denominator
    raw_norm = torch.linalg.vector_norm(raw)
    _, bounded_matrix = _stable_clip_rows(
        raw[None, :], cap, name="K6 pooled accepted direction"
    )
    bounded = bounded_matrix[0]
    if not return_diagnostics:
        return bounded
    mass_float = float(total_mass.item())
    raw_norm_float = float(raw_norm.item())
    return bounded, {
        "formula": "Clip_G(sum_r_i(g_ri*c_ri)/max(m_pool_min,sum_r_i(g_ri)))",
        "history_length": length,
        "num_clients": clients,
        "total_accepted_mass": mass_float,
        "minimum_pooled_accepted_mass": minimum,
        "deployed_denominator": float(denominator.item()),
        "denominator_floor_active": mass_float < minimum,
        "raw_direction_norm": raw_norm_float,
        "direction_norm": float(torch.linalg.vector_norm(bounded).item()),
        "influence_cap": cap,
        "projection_active": raw_norm_float > cap,
        "subconvex_weight_sum": mass_float / float(denominator.item()),
        "uses_current_round_input": False,
    }


@torch.no_grad()
def strictly_past_pooled_accepted_direction_pair(
    clipped_residual_history: torch.Tensor,
    accepted_gate_history: torch.Tensor,
    *,
    window_length: int,
    window_shift: int = 1,
    minimum_pooled_accepted_mass: float,
    influence_cap: float,
    return_diagnostics: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]
):
    """Return two consecutive or separated strictly past pooled directions."""

    history, gates, length, _, _ = _validate_history(
        clipped_residual_history, accepted_gate_history
    )
    width = int(window_length)
    shift = int(window_shift)
    if width < 1:
        raise ValueError("window_length must be a positive integer")
    if shift < 1:
        raise ValueError("window_shift must be a positive integer")
    if length < width + shift:
        raise ValueError(
            "history requires at least window_length + window_shift past rounds"
        )
    newer_start = length - width
    newer_stop = length
    older_start = newer_start - shift
    older_stop = newer_stop - shift
    older, older_diagnostics = pooled_accepted_direction(
        history[older_start:older_stop],
        gates[older_start:older_stop],
        minimum_pooled_accepted_mass=minimum_pooled_accepted_mass,
        influence_cap=influence_cap,
        return_diagnostics=True,
    )
    newer, newer_diagnostics = pooled_accepted_direction(
        history[newer_start:newer_stop],
        gates[newer_start:newer_stop],
        minimum_pooled_accepted_mass=minimum_pooled_accepted_mass,
        influence_cap=influence_cap,
        return_diagnostics=True,
    )
    if not return_diagnostics:
        return older, newer
    overlap = max(0, width - shift)
    return (
        older,
        newer,
        {
            "uses_current_round_input": False,
            "caller_history_contract": "rows_are_t_minus_L_through_t_minus_1",
            "window_length": width,
            "window_shift": shift,
            "overlap_round_count": overlap,
            "windows_overlap": overlap > 0,
            "older_window_indices": list(range(older_start, older_stop)),
            "newer_window_indices": list(range(newer_start, newer_stop)),
            "newer_window_most_recent_source_lag": 1,
            "older_window_most_recent_source_lag": shift + 1,
            "oldest_source_lag": width + shift,
            "older_window": older_diagnostics,
            "newer_window": newer_diagnostics,
        },
    )


@torch.no_grad()
def strictly_past_rolling_accepted_mean_pair(
    clipped_residual_history: torch.Tensor,
    accepted_gate_history: torch.Tensor,
    *,
    window_length: int,
    window_shift: int = 1,
    minimum_accepted_mass: float,
    influence_cap: float,
    return_diagnostics: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]
):
    r"""Return an older/newer pair of strictly past rolling accepted means.

    Input rows are chronological and the final row must represent round
    ``t-1``.  For a window length ``W`` and shift ``h``, the newer window uses
    the final ``W`` rows and the older window ends ``h`` rows earlier.  Thus
    ``h=1`` gives two consecutive windows with ``W-1`` shared rounds, whereas
    ``h>=W`` gives non-overlapping windows.

    In each past round ``r`` the accepted mean is

    .. math::

       m_r=\operatorname{Proj}_{B_G}\left(
       \frac{\sum_i g_{i,r}c_{i,r}}
            {\max\{m_{\min},\sum_i g_{i,r}\}}\right).

    Both window averages are projected once more onto the same public ball.
    No current-round tensor is accepted by this API; the caller remains
    responsible for supplying a history whose newest row is truly ``t-1``.
    """

    history, gates, length, clients, _ = _validate_history(
        clipped_residual_history, accepted_gate_history
    )
    width = int(window_length)
    shift = int(window_shift)
    if width < 1:
        raise ValueError("window_length must be a positive integer")
    if shift < 1:
        raise ValueError("window_shift must be a positive integer")
    required = width + shift
    if length < required:
        raise ValueError(
            "history requires at least window_length + window_shift past rounds"
        )
    minimum = float(minimum_accepted_mass)
    cap = float(influence_cap)
    if not math.isfinite(minimum) or not 0.0 < minimum <= float(clients):
        raise ValueError("minimum_accepted_mass must lie in (0,n]")
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("influence_cap must be finite and strictly positive")
    tolerance = 64.0 * torch.finfo(history.dtype).eps * max(1.0, cap)
    norms = torch.linalg.vector_norm(history, dim=2)
    if not bool(torch.isfinite(norms).all()) or bool((norms > cap + tolerance).any()):
        raise ValueError("past residuals must already lie in the public ball B_G")

    masses = gates.sum(dim=1)
    denominators = torch.maximum(masses, torch.full_like(masses, minimum))
    raw_round_means = torch.sum(gates[:, :, None] * history, dim=1)
    raw_round_means = raw_round_means / denominators[:, None]
    _, round_means = _stable_clip_rows(
        raw_round_means, cap, name="K6 per-round accepted means"
    )

    newer_start = length - width
    newer_stop = length
    older_start = newer_start - shift
    older_stop = newer_stop - shift
    older_raw = round_means[older_start:older_stop].mean(dim=0, keepdim=True)
    newer_raw = round_means[newer_start:newer_stop].mean(dim=0, keepdim=True)
    _, older_matrix = _stable_clip_rows(
        older_raw, cap, name="K6 older rolling accepted mean"
    )
    _, newer_matrix = _stable_clip_rows(
        newer_raw, cap, name="K6 newer rolling accepted mean"
    )
    older = older_matrix[0]
    newer = newer_matrix[0]
    if not return_diagnostics:
        return older, newer
    overlap = max(0, width - shift)
    return (
        older,
        newer,
        {
            "uses_current_round_input": False,
            "caller_history_contract": "rows_are_t_minus_L_through_t_minus_1",
            "history_length_supplied": length,
            "window_length": width,
            "window_shift": shift,
            "overlap_round_count": overlap,
            "windows_overlap": overlap > 0,
            "older_window_indices": list(range(older_start, older_stop)),
            "newer_window_indices": list(range(newer_start, newer_stop)),
            "newer_window_most_recent_source_lag": 1,
            "older_window_most_recent_source_lag": shift + 1,
            "oldest_source_lag": width + shift,
            "accepted_mass_chronological": masses.detach().cpu().tolist(),
            "denominator_chronological": denominators.detach().cpu().tolist(),
            "influence_cap": cap,
        },
    )


def _broadcast_square_matrix(
    value: torch.Tensor | Sequence[float],
    *,
    leading_shape: torch.Size,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    matrix = torch.as_tensor(value, device=device, dtype=dtype)
    target_shape = (*leading_shape, dimension, dimension)
    if matrix.shape == (dimension, dimension):
        matrix = matrix.expand(target_shape)
    elif matrix.shape != target_shape:
        raise ValueError(
            f"{name} must have shape (d,d) or {target_shape}; "
            f"received {tuple(matrix.shape)}"
        )
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} must be finite")
    return matrix


@torch.no_grad()
def eiv_corrected_moments(
    older_predictors: torch.Tensor,
    newer_predictors: torch.Tensor,
    *,
    covariance_older: torch.Tensor | Sequence[float],
    cross_covariance_older_newer: torch.Tensor | Sequence[float],
    return_diagnostics: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]
):
    r"""Compute the per-observation EIV moments ``z`` and ``w``.

    For paired vectors ``X`` and ``Y`` this routine returns

    .. math::

       z=\langle X,Y\rangle-\operatorname{tr}(C_{XY}),\qquad
       w=\|X\|_2^2-\operatorname{tr}(V_X).

    The covariance arguments must describe the transformed pair itself (for
    example, values returned by :func:`post_chain_delta_covariance`).
    """

    older = older_predictors
    newer = newer_predictors
    if not isinstance(older, torch.Tensor) or not isinstance(newer, torch.Tensor):
        raise TypeError("older_predictors and newer_predictors must be tensors")
    if older.shape != newer.shape or older.ndim < 1 or older.shape[-1] < 1:
        raise ValueError("paired predictors must have identical shape (...,d)")
    if not older.is_floating_point() or newer.dtype != older.dtype:
        raise TypeError("paired predictors must share a floating-point dtype")
    if newer.device != older.device:
        raise ValueError("paired predictors must be on the same device")
    if not bool(torch.isfinite(older).all()) or not bool(torch.isfinite(newer).all()):
        raise ValueError("paired predictors must be finite")
    dimension = int(older.shape[-1])
    leading = older.shape[:-1]
    covariance = _broadcast_square_matrix(
        covariance_older,
        leading_shape=leading,
        dimension=dimension,
        device=older.device,
        dtype=older.dtype,
        name="covariance_older",
    )
    cross = _broadcast_square_matrix(
        cross_covariance_older_newer,
        leading_shape=leading,
        dimension=dimension,
        device=older.device,
        dtype=older.dtype,
        name="cross_covariance_older_newer",
    )
    inner = torch.sum(older * newer, dim=-1)
    squared_norm = torch.sum(older.square(), dim=-1)
    trace_covariance = torch.diagonal(covariance, dim1=-2, dim2=-1).sum(dim=-1)
    trace_cross = torch.diagonal(cross, dim1=-2, dim2=-1).sum(dim=-1)
    z_moment = inner - trace_cross
    w_moment = squared_norm - trace_covariance
    if not bool(torch.isfinite(z_moment).all()) or not bool(
        torch.isfinite(w_moment).all()
    ):
        raise RuntimeError("EIV correction produced non-finite moments")
    if not return_diagnostics:
        return z_moment, w_moment
    return (
        z_moment,
        w_moment,
        {
            "formula_z": "dot(X,Y)-trace(Cxy)",
            "formula_w": "squared_norm(X)-trace(Vx)",
            "covariance_stage_required": "post_chain",
            "uncorrected_inner_product": inner.detach().cpu().tolist(),
            "uncorrected_squared_norm": squared_norm.detach().cpu().tolist(),
            "trace_cross_covariance": trace_cross.detach().cpu().tolist(),
            "trace_older_covariance": trace_covariance.detach().cpu().tolist(),
        },
    )


@torch.no_grad()
def temporal_eiv_corrected_moments(
    older_predictors: torch.Tensor,
    newer_predictors: torch.Tensor,
    *,
    covariance_older: torch.Tensor | Sequence[float],
    covariance_newer: torch.Tensor | Sequence[float],
    cross_covariance_older_newer: torch.Tensor | Sequence[float],
    return_diagnostics: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]
):
    r"""Return corrected temporal alignment and both corrected energies.

    For paired rolling directions ``X`` and ``Y``, the three moments are

    .. math::

       a&=\langle X,Y\rangle-\operatorname{tr}(C_{XY}),\\
       b_X&=\|X\|_2^2-\operatorname{tr}(V_X),\\
       b_Y&=\|Y\|_2^2-\operatorname{tr}(V_Y).

    All covariance matrices must refer to the transformed rolling directions
    themselves, for example outputs of :func:`post_chain_delta_covariance`.
    """

    older = older_predictors
    newer = newer_predictors
    if not isinstance(older, torch.Tensor) or not isinstance(newer, torch.Tensor):
        raise TypeError("older_predictors and newer_predictors must be tensors")
    if older.shape != newer.shape or older.ndim < 1 or older.shape[-1] < 1:
        raise ValueError("paired predictors must have identical shape (...,d)")
    if not older.is_floating_point() or newer.dtype != older.dtype:
        raise TypeError("paired predictors must share a floating-point dtype")
    if newer.device != older.device:
        raise ValueError("paired predictors must be on the same device")
    if not bool(torch.isfinite(older).all()) or not bool(torch.isfinite(newer).all()):
        raise ValueError("paired predictors must be finite")
    dimension = int(older.shape[-1])
    leading = older.shape[:-1]
    covariance_x = _broadcast_square_matrix(
        covariance_older,
        leading_shape=leading,
        dimension=dimension,
        device=older.device,
        dtype=older.dtype,
        name="covariance_older",
    )
    covariance_y = _broadcast_square_matrix(
        covariance_newer,
        leading_shape=leading,
        dimension=dimension,
        device=older.device,
        dtype=older.dtype,
        name="covariance_newer",
    )
    cross = _broadcast_square_matrix(
        cross_covariance_older_newer,
        leading_shape=leading,
        dimension=dimension,
        device=older.device,
        dtype=older.dtype,
        name="cross_covariance_older_newer",
    )
    observed_alignment = torch.sum(older * newer, dim=-1)
    observed_energy_x = torch.sum(older.square(), dim=-1)
    observed_energy_y = torch.sum(newer.square(), dim=-1)
    trace_x = torch.diagonal(covariance_x, dim1=-2, dim2=-1).sum(dim=-1)
    trace_y = torch.diagonal(covariance_y, dim1=-2, dim2=-1).sum(dim=-1)
    trace_cross = torch.diagonal(cross, dim1=-2, dim2=-1).sum(dim=-1)
    alignment = observed_alignment - trace_cross
    energy_x = observed_energy_x - trace_x
    energy_y = observed_energy_y - trace_y
    if not bool(torch.isfinite(alignment).all()) or not bool(
        torch.isfinite(energy_x).all()
    ):
        raise RuntimeError("temporal EIV correction produced non-finite moments")
    if not bool(torch.isfinite(energy_y).all()):
        raise RuntimeError("temporal EIV correction produced non-finite moments")
    if not return_diagnostics:
        return alignment, energy_x, energy_y
    return (
        alignment,
        energy_x,
        energy_y,
        {
            "formula_alignment": "dot(X,Y)-trace(Cxy)",
            "formula_energy_x": "squared_norm(X)-trace(Vx)",
            "formula_energy_y": "squared_norm(Y)-trace(Vy)",
            "covariance_stage_required": "post_chain",
            "observed_alignment": observed_alignment.detach().cpu().tolist(),
            "observed_energy_x": observed_energy_x.detach().cpu().tolist(),
            "observed_energy_y": observed_energy_y.detach().cpu().tolist(),
            "trace_cross_covariance": trace_cross.detach().cpu().tolist(),
            "trace_covariance_x": trace_x.detach().cpu().tolist(),
            "trace_covariance_y": trace_y.detach().cpu().tolist(),
        },
    )


@torch.no_grad()
def temporal_eiv_confidence(
    corrected_alignment: torch.Tensor | float,
    corrected_energy_x: torch.Tensor | float,
    corrected_energy_y: torch.Tensor | float,
    *,
    b_min: float,
    tau_a: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Return a bounded EIV temporal confidence.

    .. math::

       a_\tau&=\max\{a-\tau_a,0\},\\
       \kappa&=\operatorname{clip}_{[0,1]}\left(
       \frac{a_\tau}{\sqrt{\max\{b_X,b_{\min}\}
                          \max\{b_Y,b_{\min}\}}}\right).

    ``b_min`` is a public positive energy floor.  It prevents division by an
    unstable, noise-corrected energy close to or below zero; its future value
    is deliberately not selected by this primitive.  ``tau_a`` is a public
    non-negative evidence threshold: non-positive or insufficient corrected
    alignment cannot activate the predictor.
    """

    floor = float(b_min)
    alignment_threshold = float(tau_a)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("b_min must be finite and strictly positive")
    if not math.isfinite(alignment_threshold) or alignment_threshold < 0.0:
        raise ValueError("tau_a must be finite and non-negative")
    tensor_values = tuple(
        value
        for value in (
            corrected_alignment,
            corrected_energy_x,
            corrected_energy_y,
        )
        if isinstance(value, torch.Tensor)
    )
    inferred_device = (
        device
        if device is not None
        else (tensor_values[0].device if tensor_values else None)
    )
    inferred_dtype = (
        dtype
        if dtype is not None
        else (tensor_values[0].dtype if tensor_values else torch.float32)
    )
    alignment = torch.as_tensor(
        corrected_alignment, device=inferred_device, dtype=inferred_dtype
    ).reshape(-1)
    energy_x = torch.as_tensor(
        corrected_energy_x, device=alignment.device, dtype=alignment.dtype
    ).reshape(-1)
    energy_y = torch.as_tensor(
        corrected_energy_y, device=alignment.device, dtype=alignment.dtype
    ).reshape(-1)
    if alignment.numel() != 1 or energy_x.numel() != 1 or energy_y.numel() != 1:
        raise ValueError("corrected alignment and energies must be scalar")
    alignment = alignment.reshape(())
    energy_x = energy_x.reshape(())
    energy_y = energy_y.reshape(())
    if not bool(torch.isfinite(alignment)) or not bool(torch.isfinite(energy_x)):
        raise ValueError("corrected alignment and energies must be finite")
    if not bool(torch.isfinite(energy_y)):
        raise ValueError("corrected alignment and energies must be finite")
    floor_tensor = torch.as_tensor(
        floor, device=alignment.device, dtype=alignment.dtype
    )
    deployed_x = torch.maximum(energy_x, floor_tensor)
    deployed_y = torch.maximum(energy_y, floor_tensor)
    denominator = torch.sqrt(deployed_x * deployed_y)
    threshold_tensor = torch.as_tensor(
        alignment_threshold, device=alignment.device, dtype=alignment.dtype
    )
    thresholded_alignment = torch.clamp(alignment - threshold_tensor, min=0.0)
    raw = thresholded_alignment / denominator
    confidence = raw.clamp(0.0, 1.0)
    if not return_diagnostics:
        return confidence
    raw_float = float(raw.item())
    return confidence, {
        "formula": ("clip(max(a-tau_a,0)/sqrt(max(bx,b_min)*max(by,b_min)),0,1)"),
        "corrected_alignment": float(alignment.item()),
        "tau_a": alignment_threshold,
        "thresholded_alignment": float(thresholded_alignment.item()),
        "corrected_energy_x": float(energy_x.item()),
        "corrected_energy_y": float(energy_y.item()),
        "b_min": floor,
        "deployed_energy_x": float(deployed_x.item()),
        "deployed_energy_y": float(deployed_y.item()),
        "deployed_denominator": float(denominator.item()),
        "raw_confidence": raw_float,
        "confidence": float(confidence.item()),
        "energy_x_floor_active": bool(float(energy_x.item()) < floor),
        "energy_y_floor_active": bool(float(energy_y.item()) < floor),
        "alignment_threshold_active": bool(thresholded_alignment == 0.0),
        "lower_projection_active": False,
        "upper_projection_active": raw_float > 1.0,
    }


@torch.no_grad()
def radial_confidence_predictor(
    newer_direction: torch.Tensor,
    confidence: torch.Tensor | float,
    *,
    influence_cap: float,
    minimum_direction_norm: float,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Return the bounded radial predictor driven by temporal confidence.

    With ``Y=newer_direction``, ``kappa=confidence``, ``G=influence_cap`` and
    ``r_min=minimum_direction_norm``, the rule is

    .. math::

       h_r(Y)&=\frac{Y}{\max\{\|Y\|_2,r_{\min}\}},\\
       p&=\operatorname{Proj}_{B_G}\left(G\kappa h_r(Y)\right).

    The regularized radial map is continuous and ``1/r_min``-Lipschitz because
    ``h_r(Y)=Proj_{B_1}(Y/r_min)`` and Euclidean projection is non-expansive.
    It also maps ``Y=0`` to zero without manufacturing an arbitrary direction.
    Neither ``G`` nor ``r_min`` is selected by this implementation.
    """

    direction = newer_direction
    if not isinstance(direction, torch.Tensor) or direction.ndim != 1:
        raise ValueError("newer_direction must be a one-dimensional tensor")
    if not direction.is_floating_point() or not bool(torch.isfinite(direction).all()):
        raise ValueError("newer_direction must be finite floating point")
    kappa = torch.as_tensor(
        confidence, device=direction.device, dtype=direction.dtype
    ).reshape(-1)
    if kappa.numel() != 1 or not bool(torch.isfinite(kappa)):
        raise ValueError("confidence must be one finite scalar")
    kappa = kappa.reshape(())
    tolerance = 64.0 * torch.finfo(direction.dtype).eps
    if bool(kappa < -tolerance) or bool(kappa > 1.0 + tolerance):
        raise ValueError("confidence must lie in [0,1]")
    kappa = kappa.clamp(0.0, 1.0)
    cap = float(influence_cap)
    threshold = float(minimum_direction_norm)
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("influence_cap must be finite and strictly positive")
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("minimum_direction_norm must be finite and strictly positive")
    direction_norm = torch.linalg.vector_norm(direction)
    radial_denominator = torch.maximum(
        direction_norm,
        torch.as_tensor(threshold, device=direction.device, dtype=direction.dtype),
    )
    norm_floor_active = bool(float(direction_norm.item()) < threshold)
    raw = cap * kappa * direction / radial_denominator
    raw_norm = float(torch.linalg.vector_norm(raw).item())
    _, predictor_matrix = _stable_clip_rows(
        raw[None, :], cap, name="K6 radial confidence predictor"
    )
    predictor = predictor_matrix[0]
    if not return_diagnostics:
        return predictor
    return predictor, {
        "formula": "Proj_B_G(G*kappa*Y/max(norm_Y,r_min))",
        "confidence": float(kappa.item()),
        "direction_norm": float(direction_norm.item()),
        "minimum_direction_norm": threshold,
        "radial_denominator": float(radial_denominator.item()),
        "norm_floor_active": norm_floor_active,
        "regularized_direction_lipschitz_bound": 1.0 / threshold,
        "raw_predictor_norm": raw_norm,
        "predictor_norm": float(torch.linalg.vector_norm(predictor).item()),
        "influence_cap": cap,
        "projection_active": raw_norm > cap,
    }


@torch.no_grad()
def seed_balanced_median_of_means(
    values: torch.Tensor,
    seed_ids: torch.Tensor | Sequence[int],
    *,
    num_blocks: int,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Reduce observations with deterministic, seed-balanced MOM blocks.

    Observations are averaged *within each seed first*, so a seed with more
    histories cannot dominate.  Sorted seed identifiers are then partitioned
    into contiguous blocks whose sizes differ by at most one.  ``num_blocks``
    must be odd, making the coordinate-wise median unique.
    """

    tensor = values
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1:
        raise ValueError("values must be a tensor with shape (H,...)")
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise ValueError("values must be finite floating point")
    identifiers = torch.as_tensor(seed_ids, device=tensor.device).reshape(-1)
    if identifiers.shape != (int(tensor.shape[0]),):
        raise ValueError("seed_ids must have one entry per observation")
    if identifiers.dtype.is_floating_point or identifiers.dtype == torch.bool:
        raise TypeError("seed_ids must use an integer dtype")
    unique = torch.unique(identifiers, sorted=True)
    seed_count = int(unique.numel())
    blocks = int(num_blocks)
    if blocks < 1 or blocks > seed_count or blocks % 2 == 0:
        raise ValueError("num_blocks must be odd and lie in [1,num_seeds]")

    seed_means = torch.stack(
        [tensor[identifiers == seed].mean(dim=0) for seed in unique], dim=0
    )
    base_size, remainder = divmod(seed_count, blocks)
    block_sizes = [base_size + (index < remainder) for index in range(blocks)]
    block_means: list[torch.Tensor] = []
    block_seed_ids: list[list[int]] = []
    start = 0
    for size in block_sizes:
        stop = start + int(size)
        block_means.append(seed_means[start:stop].mean(dim=0))
        block_seed_ids.append([int(value) for value in unique[start:stop].cpu()])
        start = stop
    stacked_blocks = torch.stack(block_means, dim=0)
    estimate = torch.median(stacked_blocks, dim=0).values
    if not return_diagnostics:
        return estimate
    return estimate, {
        "reduction": "seed_balanced_deterministic_contiguous_median_of_means",
        "num_observations": int(tensor.shape[0]),
        "num_seeds": seed_count,
        "num_blocks": blocks,
        "seed_ids_sorted": [int(value) for value in unique.cpu()],
        "block_seed_ids": block_seed_ids,
        "block_sizes": [int(size) for size in block_sizes],
        "seed_means": seed_means.detach().cpu().tolist(),
        "block_means": stacked_blocks.detach().cpu().tolist(),
    }


@torch.no_grad()
def projected_eiv_coefficient(
    corrected_cross_moment: torch.Tensor | float,
    corrected_second_moment: torch.Tensor | float,
    *,
    b_min: float,
    lambda0: float,
    beta_max: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Return ``beta=Proj_[0,beta_max](z/(max(w,b_min)+lambda0))``.

    ``b_min`` is a public positive floor on the corrected predictor energy and
    ``lambda0`` is a public non-negative ridge term.  Diagnostics distinguish
    activation of the denominator floor from the lower and upper projections.
    ``beta_max`` is public and strictly positive.  Its default of one is a
    conservative API default, not a scientific choice for a future protocol.
    """

    floor = float(b_min)
    ridge = float(lambda0)
    upper = float(beta_max)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("b_min must be finite and strictly positive")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("lambda0 must be finite and non-negative")
    if not math.isfinite(upper) or upper <= 0.0:
        raise ValueError("beta_max must be finite and strictly positive")
    inferred_device = device
    if inferred_device is None and isinstance(corrected_cross_moment, torch.Tensor):
        inferred_device = corrected_cross_moment.device
    if inferred_device is None and isinstance(corrected_second_moment, torch.Tensor):
        inferred_device = corrected_second_moment.device
    inferred_dtype = dtype
    if inferred_dtype is None:
        inferred_dtype = (
            corrected_cross_moment.dtype
            if isinstance(corrected_cross_moment, torch.Tensor)
            else torch.float32
        )
    z_value = torch.as_tensor(
        corrected_cross_moment, device=inferred_device, dtype=inferred_dtype
    )
    w_value = torch.as_tensor(
        corrected_second_moment, device=z_value.device, dtype=z_value.dtype
    )
    if z_value.numel() != 1 or w_value.numel() != 1:
        raise ValueError("corrected moments must be scalar")
    z_value = z_value.reshape(())
    w_value = w_value.reshape(())
    if not bool(torch.isfinite(z_value)) or not bool(torch.isfinite(w_value)):
        raise ValueError("corrected moments must be finite")
    floor_tensor = torch.as_tensor(floor, device=z_value.device, dtype=z_value.dtype)
    denominator = torch.maximum(w_value, floor_tensor) + ridge
    raw = z_value / denominator
    beta = raw.clamp(0.0, upper)
    if not return_diagnostics:
        return beta
    raw_float = float(raw.item())
    return beta, {
        "formula": "clip(z/(max(w,b_min)+lambda0),0,beta_max)",
        "corrected_cross_moment": float(z_value.item()),
        "corrected_second_moment": float(w_value.item()),
        "b_min": floor,
        "lambda0": ridge,
        "beta_max": upper,
        "deployed_denominator": float(denominator.item()),
        "raw_beta": raw_float,
        "beta": float(beta.item()),
        "denominator_floor_active": bool(float(w_value.item()) < floor),
        "lower_projection_active": raw_float < 0.0,
        "upper_projection_active": raw_float > upper,
    }


@torch.no_grad()
def projected_eiv_predictor(
    source_direction: torch.Tensor,
    beta: torch.Tensor | float,
    *,
    influence_cap: float,
    beta_max: float = 1.0,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Return ``Proj_{B_G}(beta * source_direction)`` on the input device."""

    source = source_direction
    if not isinstance(source, torch.Tensor) or source.ndim != 1:
        raise ValueError("source_direction must be a one-dimensional tensor")
    if not source.is_floating_point() or not bool(torch.isfinite(source).all()):
        raise ValueError("source_direction must be finite floating point")
    coefficient = torch.as_tensor(beta, device=source.device, dtype=source.dtype)
    if coefficient.numel() != 1 or not bool(torch.isfinite(coefficient)):
        raise ValueError("beta must be one finite scalar")
    coefficient = coefficient.reshape(())
    upper = float(beta_max)
    if not math.isfinite(upper) or upper <= 0.0:
        raise ValueError("beta_max must be finite and strictly positive")
    tolerance = 64.0 * torch.finfo(source.dtype).eps * max(1.0, upper)
    if bool(coefficient < -tolerance) or bool(coefficient > upper + tolerance):
        raise ValueError("beta must lie in [0,beta_max]")
    coefficient = coefficient.clamp(0.0, upper)
    raw = coefficient * source
    raw_norm = float(torch.linalg.vector_norm(raw).item())
    _, projected_matrix = _stable_clip_rows(
        raw[None, :], influence_cap, name="K6 EIV predictor"
    )
    predictor = projected_matrix[0]
    if not return_diagnostics:
        return predictor
    cap = float(influence_cap)
    return predictor, {
        "formula": "Proj_B_G(beta*source_direction)",
        "beta": float(coefficient.item()),
        "beta_max": upper,
        "source_norm": float(torch.linalg.vector_norm(source).item()),
        "raw_predictor_norm": raw_norm,
        "predictor_norm": float(torch.linalg.vector_norm(predictor).item()),
        "influence_cap": cap,
        "projection_active": raw_norm > cap,
    }


def _validate_linearization_point(point: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(point, torch.Tensor) or point.ndim != 1 or point.numel() < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional tensor")
    if not point.is_floating_point() or not bool(torch.isfinite(point).all()):
        raise ValueError(f"{name} must be finite floating point")
    return point.detach().clone().requires_grad_(True)


def _evaluate_transform(
    transform: TensorTransform,
    point: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    value = transform(point)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must return a torch.Tensor")
    if value.device != point.device or value.dtype != point.dtype:
        raise ValueError(f"{name} must preserve the point device and dtype")
    flattened = value.reshape(-1)
    if flattened.numel() < 1 or not bool(torch.isfinite(flattened).all()):
        raise ValueError(f"{name} must return a non-empty finite tensor")
    return flattened


def _exact_reverse_mode_jacobian(
    transform: TensorTransform,
    point: torch.Tensor,
    *,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    leaf = _validate_linearization_point(point, name=f"{name}_point")
    output = _evaluate_transform(transform, leaf, name=name)
    rows: list[torch.Tensor] = []
    for index in range(int(output.numel())):
        if not output[index].requires_grad:
            gradient = torch.zeros_like(leaf)
        else:
            gradient = torch.autograd.grad(
                output[index],
                leaf,
                retain_graph=index + 1 < int(output.numel()),
                allow_unused=True,
            )[0]
            if gradient is None:
                gradient = torch.zeros_like(leaf)
        rows.append(gradient)
    jacobian = torch.stack(rows, dim=0)
    if not bool(torch.isfinite(jacobian).all()):
        raise RuntimeError(f"{name} produced a non-finite Jacobian")
    return output.detach(), jacobian.detach()


@torch.no_grad()
def _audit_local_differentiability(
    transform: TensorTransform,
    point: torch.Tensor,
    output_at_point: torch.Tensor,
    jacobian: torch.Tensor,
    *,
    finite_difference_step: float | None,
    differentiability_rtol: float,
    differentiability_atol: float,
    name: str,
) -> dict[str, Any]:
    epsilon = float(
        finite_difference_step
        if finite_difference_step is not None
        else 8.0 * math.sqrt(torch.finfo(point.dtype).eps)
    )
    relative = float(differentiability_rtol)
    absolute = float(differentiability_atol)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("finite_difference_step must be finite and positive")
    if not math.isfinite(relative) or relative < 0.0:
        raise ValueError("differentiability_rtol must be finite and non-negative")
    if not math.isfinite(absolute) or absolute < 0.0:
        raise ValueError("differentiability_atol must be finite and non-negative")

    repeat = _evaluate_transform(transform, point.detach(), name=name)
    repeat_error = torch.max(torch.abs(repeat - output_at_point))
    scale = torch.maximum(torch.ones_like(output_at_point), torch.abs(output_at_point))
    repeat_tolerance = absolute + relative * torch.max(scale)
    if bool(repeat_error > repeat_tolerance):
        raise RuntimeError(f"{name} is not deterministic at the linearization point")

    maximum_mismatch = torch.zeros((), device=point.device, dtype=point.dtype)
    failing_coordinate: int | None = None
    for coordinate in range(int(point.numel())):
        step = epsilon * max(1.0, abs(float(point[coordinate].item())))
        perturbation = torch.zeros_like(point)
        perturbation[coordinate] = step
        forward_output = _evaluate_transform(
            transform, point.detach() + perturbation, name=name
        )
        backward_output = _evaluate_transform(
            transform, point.detach() - perturbation, name=name
        )
        forward = (forward_output - output_at_point) / step
        backward = (output_at_point - backward_output) / step
        derivative = jacobian[:, coordinate]
        local_scale = torch.maximum(
            torch.ones_like(derivative),
            torch.maximum(torch.abs(forward), torch.abs(backward)),
        )
        tolerance = absolute + relative * local_scale
        mismatch = torch.maximum(
            torch.abs(forward - derivative), torch.abs(backward - derivative)
        )
        coordinate_maximum = torch.max(mismatch / tolerance.clamp_min(1.0e-30))
        if bool(coordinate_maximum > maximum_mismatch):
            maximum_mismatch = coordinate_maximum
        if failing_coordinate is None and bool((mismatch > tolerance).any()):
            failing_coordinate = coordinate
    if failing_coordinate is not None:
        raise ValueError(
            f"{name} failed the differentiability audit at input coordinate "
            f"{failing_coordinate}; refuse post-chain delta covariance at a "
            "possible clipping/projection boundary"
        )
    return {
        "passed": True,
        "finite_difference_step_base": epsilon,
        "relative_tolerance": relative,
        "absolute_tolerance": absolute,
        "maximum_normalized_mismatch": float(maximum_mismatch.item()),
        "audited_input_coordinates": int(point.numel()),
    }


def _validate_covariance(
    covariance: torch.Tensor | Sequence[float],
    *,
    rows: int,
    columns: int,
    point: torch.Tensor,
    name: str,
    require_symmetric: bool,
    require_positive_semidefinite: bool,
) -> torch.Tensor:
    matrix = torch.as_tensor(covariance, device=point.device, dtype=point.dtype)
    if matrix.shape != (rows, columns) or not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} must be finite with shape ({rows},{columns})")
    if require_symmetric:
        tolerance = (
            128.0
            * torch.finfo(point.dtype).eps
            * max(1.0, float(torch.max(torch.abs(matrix)).item()))
        )
        if not bool(torch.allclose(matrix, matrix.T, atol=tolerance, rtol=0.0)):
            raise ValueError(f"{name} must be symmetric")
    if require_positive_semidefinite:
        scale = max(1.0, float(torch.max(torch.abs(matrix)).item()))
        tolerance = 256.0 * torch.finfo(point.dtype).eps * max(rows, columns) * scale
        identity = torch.eye(rows, device=point.device, dtype=point.dtype)
        _, information = torch.linalg.cholesky_ex(
            matrix + tolerance * identity, check_errors=False
        )
        if int(information.item()) != 0:
            raise ValueError(f"{name} must be positive semidefinite")
    return matrix


def post_chain_delta_covariance(
    phi_x: TensorTransform,
    phi_y: TensorTransform,
    *,
    point_x: torch.Tensor,
    point_y: torch.Tensor,
    covariance_x: torch.Tensor | Sequence[float],
    covariance_y: torch.Tensor | Sequence[float],
    shared_covariance_xy: torch.Tensor | Sequence[float],
    audit_differentiability: bool = True,
    finite_difference_step: float | None = None,
    differentiability_rtol: float = 2.0e-2,
    differentiability_atol: float = 2.0e-3,
    return_diagnostics: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]
):
    r"""Compute post-chain delta covariances using exact autograd Jacobians.

    ``phi_x`` and ``phi_y`` represent the complete differentiable chains from
    their primitive Gaussian inputs to the rolling vectors used by EIV.  The
    returned tensors are ``(Vx, Vy, Cxy)`` with

    ``Vx=Jx covariance_x Jx.T`` and
    ``Cxy=Jx shared_covariance_xy Jy.T``.

    The computation never changes device.  Consequently an MPS input causes
    both Jacobian and covariance propagation to execute on MPS, and an
    unsupported operation fails instead of silently substituting a CPU or a
    pre-clipping covariance.  By default, one-sided finite differences audit
    every input coordinate and reject possible non-differentiable boundaries.
    """

    if not callable(phi_x) or not callable(phi_y):
        raise TypeError("phi_x and phi_y must be callable")
    x_point = _validate_linearization_point(point_x, name="point_x").detach()
    y_point = _validate_linearization_point(point_y, name="point_y").detach()
    if x_point.device != y_point.device or x_point.dtype != y_point.dtype:
        raise ValueError("point_x and point_y must share device and dtype")
    output_x, jacobian_x = _exact_reverse_mode_jacobian(phi_x, x_point, name="phi_x")
    output_y, jacobian_y = _exact_reverse_mode_jacobian(phi_y, y_point, name="phi_y")
    if audit_differentiability:
        audit_x = _audit_local_differentiability(
            phi_x,
            x_point,
            output_x,
            jacobian_x,
            finite_difference_step=finite_difference_step,
            differentiability_rtol=differentiability_rtol,
            differentiability_atol=differentiability_atol,
            name="phi_x",
        )
        audit_y = _audit_local_differentiability(
            phi_y,
            y_point,
            output_y,
            jacobian_y,
            finite_difference_step=finite_difference_step,
            differentiability_rtol=differentiability_rtol,
            differentiability_atol=differentiability_atol,
            name="phi_y",
        )
    else:
        audit_x = {"passed": None, "skipped": True}
        audit_y = {"passed": None, "skipped": True}

    input_x = int(x_point.numel())
    input_y = int(y_point.numel())
    sigma_x = _validate_covariance(
        covariance_x,
        rows=input_x,
        columns=input_x,
        point=x_point,
        name="covariance_x",
        require_symmetric=True,
        require_positive_semidefinite=True,
    )
    sigma_y = _validate_covariance(
        covariance_y,
        rows=input_y,
        columns=input_y,
        point=y_point,
        name="covariance_y",
        require_symmetric=True,
        require_positive_semidefinite=True,
    )
    sigma_shared = _validate_covariance(
        shared_covariance_xy,
        rows=input_x,
        columns=input_y,
        point=x_point,
        name="shared_covariance_xy",
        require_symmetric=False,
        require_positive_semidefinite=False,
    )
    joint_covariance = torch.cat(
        (
            torch.cat((sigma_x, sigma_shared), dim=1),
            torch.cat((sigma_shared.T, sigma_y), dim=1),
        ),
        dim=0,
    )
    _validate_covariance(
        joint_covariance,
        rows=input_x + input_y,
        columns=input_x + input_y,
        point=torch.cat((x_point, y_point)),
        name="joint_input_covariance",
        require_symmetric=True,
        require_positive_semidefinite=True,
    )
    covariance_output_x = jacobian_x @ sigma_x @ jacobian_x.T
    covariance_output_y = jacobian_y @ sigma_y @ jacobian_y.T
    cross_covariance = jacobian_x @ sigma_shared @ jacobian_y.T
    covariance_output_x = 0.5 * (covariance_output_x + covariance_output_x.T)
    covariance_output_y = 0.5 * (covariance_output_y + covariance_output_y.T)
    if not bool(torch.isfinite(covariance_output_x).all()) or not bool(
        torch.isfinite(covariance_output_y).all()
    ):
        raise RuntimeError("post-chain marginal covariance is non-finite")
    if not bool(torch.isfinite(cross_covariance).all()):
        raise RuntimeError("post-chain cross-covariance is non-finite")
    _validate_covariance(
        covariance_output_x,
        rows=int(covariance_output_x.shape[0]),
        columns=int(covariance_output_x.shape[1]),
        point=output_x,
        name="post_chain_covariance_x",
        require_symmetric=True,
        require_positive_semidefinite=True,
    )
    _validate_covariance(
        covariance_output_y,
        rows=int(covariance_output_y.shape[0]),
        columns=int(covariance_output_y.shape[1]),
        point=output_y,
        name="post_chain_covariance_y",
        require_symmetric=True,
        require_positive_semidefinite=True,
    )
    result = (covariance_output_x, covariance_output_y, cross_covariance)
    if not return_diagnostics:
        return result
    return (
        *result,
        {
            "covariance_stage": "post_chain_delta_method",
            "jacobian_method": "exact_reverse_mode_autograd",
            "compute_device": str(x_point.device),
            "compute_dtype": str(x_point.dtype),
            "used_preclipping_covariance_substitute": False,
            "joint_input_covariance_psd_audited": True,
            "marginal_covariances_psd_audited": True,
            "phi_x_output_shape_flat": int(output_x.numel()),
            "phi_y_output_shape_flat": int(output_y.numel()),
            "jacobian_x": jacobian_x.detach().cpu().tolist(),
            "jacobian_y": jacobian_y.detach().cpu().tolist(),
            "differentiability_audit_x": audit_x,
            "differentiability_audit_y": audit_y,
        },
    )


__all__ = [
    "TensorTransform",
    "eiv_corrected_moments",
    "pooled_accepted_direction",
    "post_chain_delta_covariance",
    "projected_eiv_coefficient",
    "projected_eiv_predictor",
    "radial_confidence_predictor",
    "seed_balanced_median_of_means",
    "strictly_past_pooled_accepted_direction_pair",
    "strictly_past_rolling_accepted_mean_pair",
    "temporal_eiv_confidence",
    "temporal_eiv_corrected_moments",
]
