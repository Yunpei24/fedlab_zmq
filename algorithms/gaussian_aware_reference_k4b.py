r"""Past-only temporal-mixture primitives for the preregistered G0g-K4b screen.

K4 suppresses a client contribution when its causal history gate is small.
Because the aggregate keeps the public denominator ``n``, this rejection also
shrinks the whole update.  K4b keeps the same K2 current gate and the same K4
history gate.  Its primary mechanism is a *historical-confidence mixture*, also
called *full temporal missing-slot imputation*:

.. math::

   c_i=\operatorname{Clip}_G(X_i-a),\quad
   g_i=\min\{\gamma_i,h_i\},\quad
   \psi_i=g_i c_i+(1-h_i)p.

Here ``p`` is a vector computed from strictly earlier rounds.  Since
``g_i <= h_i`` and both ``c_i`` and ``p`` have norm at most ``G``, the
coefficient sum is at most one and therefore ``||psi_i|| <= G``.  The
aggregate remains ``a + sum_i psi_i / n``: gates are never used as a random
denominator.

The exact incremental-suppression ablation replaces ``1-h_i`` by
``gamma_i-g_i``.  It therefore imputes only the difference between the current
K2 gate and the joint K4 gate.  Both mechanisms reduce identically to K2 when
``h_i=1`` and both preserve the same per-slot cap ``G``.

This module deliberately lives beside, rather than inside, the frozen K4
implementation.  It imports K4 only to reproduce its current and temporal
gates exactly; no K4 source or result is modified.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from algorithms.gaussian_aware_reference import (
    TensorOrScalar,
    gaussian_aware_fixed_anchor_temporal_gated_reference,
)

FULL_TEMPORAL_MISSING_SLOT = "full_temporal_missing_slot_imputation"
INCREMENTAL_TEMPORAL_SUPPRESSION = "incremental_temporal_suppression_imputation"
IMPUTATION_MODES = (
    FULL_TEMPORAL_MISSING_SLOT,
    INCREMENTAL_TEMPORAL_SUPPRESSION,
)


def _validate_matrix(value: torch.Tensor, *, name: str) -> tuple[int, int]:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{name} must have shape (n,d) with n,d >= 1")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return int(value.shape[0]), int(value.shape[1])


def _stable_clip_rows(
    rows: torch.Tensor, radius: float, *, name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip rows without forming a potentially overflowing squared norm."""

    if rows.ndim != 2 or not bool(torch.isfinite(rows).all()):
        raise ValueError(f"{name} must be a finite matrix")
    cap = float(radius)
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("influence_cap must be finite and strictly positive")
    row_scale = rows.abs().amax(dim=1)
    safe_scale = torch.where(row_scale > 0.0, row_scale, torch.ones_like(row_scale))
    scaled = rows / safe_scale[:, None]
    unit_norm = torch.sqrt(torch.sum(scaled.square(), dim=1))
    norms = row_scale * unit_norm
    if not bool(torch.isfinite(norms).all()):
        raise ValueError(f"{name} norms are not representable in the input dtype")
    safe_unit = torch.where(unit_norm > 0.0, unit_norm, torch.ones_like(unit_norm))
    clipped_direction = scaled / safe_unit[:, None]
    clipped_candidate = clipped_direction * cap
    clipped = torch.where((norms > cap)[:, None], clipped_candidate, rows)
    if not bool(torch.isfinite(clipped).all()):
        raise ValueError(f"{name} clipping produced a non-finite value")
    return norms, clipped


@torch.no_grad()
def fixed_denominator_past_predictor(
    clipped_residual_history: torch.Tensor,
    accepted_gate_history: torch.Tensor,
    *,
    minimum_accepted_mass: float,
    influence_cap: float,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Construct the K4b predictor from strictly prior accepted residuals.

    ``clipped_residual_history`` has shape ``(L,n,d)`` and
    ``accepted_gate_history`` has shape ``(L,n)``.  For each prior round
    ``r`` this routine computes

    .. math::

       \widehat p_r=\operatorname{Clip}_G\!\left(
       \frac{\sum_i g_{i,r}c_{i,r}}
            {\max\{m_{\min},\sum_i g_{i,r}\}}\right),

    then returns ``Clip_G(mean_r p_hat_r)``.  The public floor
    ``m_min`` prevents rejected clients from creating a small, data-dependent
    denominator.  The caller owns the temporal contract; the runner supplies
    only rounds strictly earlier than the evaluated upload.
    """

    history = clipped_residual_history
    if not isinstance(history, torch.Tensor):
        raise TypeError("clipped_residual_history must be a torch.Tensor")
    if history.ndim != 3 or min(history.shape) < 1:
        raise ValueError("clipped_residual_history must have shape (L,n,d)")
    if not history.is_floating_point() or not bool(torch.isfinite(history).all()):
        raise ValueError("clipped_residual_history must be finite floating point")
    length, n, _ = (int(value) for value in history.shape)
    gates = torch.as_tensor(
        accepted_gate_history, device=history.device, dtype=history.dtype
    )
    if gates.shape != (length, n) or not bool(torch.isfinite(gates).all()):
        raise ValueError("accepted_gate_history must be finite with shape (L,n)")
    tolerance = 64.0 * torch.finfo(history.dtype).eps
    if bool((gates < -tolerance).any()) or bool((gates > 1.0 + tolerance).any()):
        raise ValueError("accepted gates must lie in [0,1]")
    gates = gates.clamp(0.0, 1.0)
    minimum = float(minimum_accepted_mass)
    if not math.isfinite(minimum) or not 0.0 < minimum <= float(n):
        raise ValueError("minimum_accepted_mass must lie in (0,n]")
    cap = float(influence_cap)
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("influence_cap must be finite and strictly positive")

    historical_norms = torch.linalg.vector_norm(history, dim=2)
    if not bool(torch.isfinite(historical_norms).all()):
        raise ValueError("historical clipped-residual norms must be finite")
    if bool((historical_norms > cap + tolerance * max(1.0, cap)).any()):
        raise ValueError("historical residuals must already be clipped at G")

    accepted_mass = gates.sum(dim=1)
    denominators = torch.maximum(
        accepted_mass,
        torch.full_like(accepted_mass, minimum),
    )
    numerators = torch.sum(gates[:, :, None] * history, dim=1)
    raw_round_predictors = numerators / denominators[:, None]
    _, round_predictors = _stable_clip_rows(
        raw_round_predictors, cap, name="per-round accepted-mean predictors"
    )
    raw_predictor = round_predictors.mean(dim=0, keepdim=True)
    _, predictor_matrix = _stable_clip_rows(
        raw_predictor, cap, name="past-window predictor"
    )
    predictor = predictor_matrix[0]
    if not return_diagnostics:
        return predictor
    return predictor, {
        "predictor_name": "fixed_denominator_strictly_past_accepted_mean",
        "history_length": length,
        "num_clients": n,
        "minimum_accepted_mass": minimum,
        "accepted_mass_by_round": accepted_mass.cpu().tolist(),
        "denominator_by_round": denominators.cpu().tolist(),
        "denominator_is_at_least_public_floor": bool(
            (denominators >= minimum - tolerance).all()
        ),
        "round_predictor_norms": torch.linalg.vector_norm(round_predictors, dim=1)
        .cpu()
        .tolist(),
        "predictor_norm": float(torch.linalg.vector_norm(predictor).item()),
        "predictor_cap": cap,
        "uses_strictly_past_inputs_only_if_caller_contract_holds": True,
        "gate_sum_used_as_aggregate_denominator": False,
    }


@torch.no_grad()
def pointwise_optimal_full_imputation_predictor(
    clipped_residuals: torch.Tensor,
    final_gates: torch.Tensor,
    history_gates: torch.Tensor,
    *,
    target_direction: torch.Tensor,
    influence_cap: float,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Return the pointwise-optimal predictor for the primary K4b mechanism.

    Conditional on the current clipped residuals ``c_i`` and gates ``g_i,h_i``,
    the primary reference error is minimized over ``p in B_G`` by

    .. math::

       p^*=\operatorname{Proj}_{B_G}\!\left(
       \frac{n\,v-\sum_i g_i c_i}{\sum_i(1-h_i)}\right),

    where ``v=target-anchor``.  If the missing-slot mass is zero, ``p*=0`` by
    convention because the aggregate is then independent of ``p``.  This
    helper is an evaluation oracle: callers must mark it non-deployable and
    must not make a privacy claim for it.
    """

    n, dimension = _validate_matrix(clipped_residuals, name="clipped_residuals")
    dtype = torch.float64 if clipped_residuals.dtype == torch.float64 else torch.float32
    device = clipped_residuals.device
    clipped = clipped_residuals.to(dtype=dtype)
    cap = float(influence_cap)
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("influence_cap must be finite and strictly positive")
    tolerance = 64.0 * torch.finfo(dtype).eps * max(1.0, cap)
    norms = torch.linalg.vector_norm(clipped, dim=1)
    if not bool(torch.isfinite(norms).all()) or bool((norms > cap + tolerance).any()):
        raise ValueError("clipped_residuals must be finite and lie in B_G")
    gates = torch.as_tensor(final_gates, device=device, dtype=dtype).reshape(-1)
    history = torch.as_tensor(history_gates, device=device, dtype=dtype).reshape(-1)
    target = torch.as_tensor(target_direction, device=device, dtype=dtype).reshape(-1)
    if gates.shape != (n,) or history.shape != (n,):
        raise ValueError("final_gates and history_gates must have shape (n,)")
    if target.shape != (dimension,) or not bool(torch.isfinite(target).all()):
        raise ValueError("target_direction must be a finite vector of dimension d")
    if not bool(torch.isfinite(gates).all()) or not bool(torch.isfinite(history).all()):
        raise ValueError("gates must be finite")
    scalar_tolerance = 64.0 * torch.finfo(dtype).eps
    if bool((gates < -scalar_tolerance).any()) or bool(
        (history < -scalar_tolerance).any()
    ):
        raise ValueError("gates must be non-negative")
    if bool((gates > history + scalar_tolerance).any()) or bool(
        (history > 1.0 + scalar_tolerance).any()
    ):
        raise ValueError("gates must satisfy 0 <= g <= h <= 1")
    gates = gates.clamp(0.0, 1.0)
    history = history.clamp(0.0, 1.0)
    missing_mass = torch.sum(1.0 - history)
    if float(missing_mass.item()) > scalar_tolerance:
        numerator = float(n) * target - torch.sum(gates[:, None] * clipped, dim=0)
        raw = numerator / missing_mass
        _, bounded_matrix = _stable_clip_rows(
            raw[None, :], cap, name="pointwise-optimal predictor"
        )
        bounded = bounded_matrix[0]
    else:
        raw = torch.zeros_like(target)
        bounded = torch.zeros_like(target)
    if not return_diagnostics:
        return bounded
    return bounded, {
        "predictor_name": "pointwise_optimal_full_temporal_missing_slot_oracle",
        "missing_slot_mass": float(missing_mass.item()),
        "raw_predictor_norm": float(torch.linalg.vector_norm(raw).item()),
        "predictor_norm": float(torch.linalg.vector_norm(bounded).item()),
        "predictor_cap": cap,
        "uses_current_target": True,
        "deployable": False,
        "privacy_claimed": False,
        "zero_when_missing_slot_mass_is_zero": True,
    }


@torch.no_grad()
def gaussian_aware_fixed_anchor_past_imputed_reference(
    vectors: torch.Tensor,
    *,
    anchor: torch.Tensor,
    statistical_radii: TensorOrScalar,
    temporal_standardized_history: torch.Tensor,
    enrollment_standardized_mean: torch.Tensor,
    enrollment_size: int,
    temporal_gate_inner_threshold: float,
    temporal_gate_outer_threshold: float,
    predictor: torch.Tensor,
    block_sizes: Sequence[int] | None = None,
    influence_cap: float = 1.0,
    current_gate_transition_width: float = 1.0,
    num_replacements_for_diagnostics: int = 1,
    predictor_role: str = "strictly_past_accepted_mean",
    imputation_mode: str = FULL_TEMPORAL_MISSING_SLOT,
    deployable: bool = True,
    privacy_claimed: bool = True,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Return the fixed-denominator K4b reference.

    K4 is evaluated first to reproduce its current gate ``gamma``, historical
    gate ``h`` and final gate ``g=min(gamma,h)`` exactly.  The caller-supplied
    predictor is clipped to the same public radius ``G``.  In the primary mode,
    each contribution is ``g*c + (1-h)*p``.  In the exact ablation it is
    ``g*c + (gamma-g)*p``.  The aggregate is divided by the public cohort size.
    The diagnostic flags make non-deployable oracle predictors explicit.
    """

    n, dimension = _validate_matrix(vectors, name="vectors")
    dtype = torch.float64 if vectors.dtype == torch.float64 else torch.float32
    work_vectors = vectors.to(dtype=dtype)
    work_anchor = torch.as_tensor(anchor, device=vectors.device, dtype=dtype).reshape(
        -1
    )
    if work_anchor.shape != (dimension,) or not bool(torch.isfinite(work_anchor).all()):
        raise ValueError("anchor must be a finite vector of dimension d")
    work_predictor = torch.as_tensor(
        predictor, device=vectors.device, dtype=dtype
    ).reshape(-1)
    if work_predictor.shape != (dimension,) or not bool(
        torch.isfinite(work_predictor).all()
    ):
        raise ValueError("predictor must be a finite vector of dimension d")
    cap = float(influence_cap)
    _, predictor_matrix = _stable_clip_rows(
        work_predictor[None, :], cap, name="K4b predictor"
    )
    bounded_predictor = predictor_matrix[0]

    _, k4_diagnostics = gaussian_aware_fixed_anchor_temporal_gated_reference(
        work_vectors,
        anchor=work_anchor,
        statistical_radii=statistical_radii,
        temporal_standardized_history=temporal_standardized_history,
        enrollment_standardized_mean=enrollment_standardized_mean,
        enrollment_size=enrollment_size,
        temporal_gate_inner_threshold=temporal_gate_inner_threshold,
        temporal_gate_outer_threshold=temporal_gate_outer_threshold,
        block_sizes=block_sizes,
        influence_cap=cap,
        current_gate_transition_width=current_gate_transition_width,
        num_replacements_for_diagnostics=num_replacements_for_diagnostics,
        return_diagnostics=True,
    )
    current_gates = torch.tensor(
        k4_diagnostics["current_gates_by_client"],
        device=vectors.device,
        dtype=dtype,
    )
    temporal_gates = torch.tensor(
        k4_diagnostics["temporal_gates_by_client"],
        device=vectors.device,
        dtype=dtype,
    )
    gates = torch.minimum(current_gates, temporal_gates)
    residuals = work_vectors - work_anchor[None, :]
    if not bool(torch.isfinite(residuals).all()):
        raise ValueError("vectors minus anchor overflowed in G0g-K4b")
    residual_norms, clipped = _stable_clip_rows(
        residuals, cap, name="G0g-K4b residuals"
    )
    mode = str(imputation_mode)
    if mode == FULL_TEMPORAL_MISSING_SLOT:
        imputation_coefficients = 1.0 - temporal_gates
        contribution_formula = "g*c_plus_(1-h)*p"
        mechanism_label = (
            "historical_confidence_mixture_full_temporal_missing_slot_imputation"
        )
    elif mode == INCREMENTAL_TEMPORAL_SUPPRESSION:
        imputation_coefficients = current_gates - gates
        contribution_formula = "g*c_plus_(gamma-g)*p"
        mechanism_label = "incremental_temporal_suppression_imputation"
    else:
        raise ValueError(f"Unknown imputation_mode: {mode!r}")
    coefficient_sums = gates + imputation_coefficients
    tolerance = 64.0 * torch.finfo(dtype).eps * max(1.0, cap)
    if bool((coefficient_sums > 1.0 + tolerance).any()):
        raise RuntimeError("K4b coefficient sum exceeded one")
    direct_current_contributions = gates[:, None] * clipped
    imputed_contributions = (
        imputation_coefficients[:, None] * bounded_predictor[None, :]
    )
    contributions = direct_current_contributions + imputed_contributions
    if not bool(torch.isfinite(contributions).all()):
        raise ValueError("G0g-K4b contributions must be finite")
    contribution_norms = torch.linalg.vector_norm(contributions, dim=1)
    direct_current_norms = torch.linalg.vector_norm(direct_current_contributions, dim=1)
    imputed_norms = torch.linalg.vector_norm(imputed_contributions, dim=1)
    if not bool(torch.isfinite(contribution_norms).all()):
        raise ValueError("G0g-K4b contribution norms must be finite")
    certificate = bool((contribution_norms <= cap + tolerance).all())
    if not certificate:
        raise RuntimeError("G0g-K4b client contribution exceeded the public cap")
    reference = work_anchor + contributions.mean(dim=0)
    if not bool(torch.isfinite(reference).all()):
        raise ValueError("G0g-K4b reference must be finite")
    if not return_diagnostics:
        return reference

    diagnostics = dict(k4_diagnostics)
    diagnostics.update(
        {
            "reference_name": f"g0g_k4b_{mode}",
            "mechanism_label": mechanism_label,
            "imputation_mode": mode,
            "predictor_role": str(predictor_role),
            "predictor_deployable": bool(deployable),
            "predictor_deployability": (
                "conditional_on_supplied_past_or_public_anchor"
                if deployable
                else "nondeployable_diagnostic"
            ),
            "privacy_claimed": bool(privacy_claimed),
            "predictor_is_clipped_at_common_cap": True,
            "predictor_norm": float(torch.linalg.vector_norm(bounded_predictor).item()),
            "predictor_by_dimension": bounded_predictor.cpu().tolist(),
            "imputation_coefficients_by_client": (
                imputation_coefficients.cpu().tolist()
            ),
            "imputation_coefficient_mean": float(imputation_coefficients.mean().item()),
            "coefficient_sums_by_client": coefficient_sums.cpu().tolist(),
            "coefficient_sum_max": float(coefficient_sums.max().item()),
            "contribution_formula": contribution_formula,
            "normalization_by_gate_sum": False,
            "aggregate_denominator": n,
            "direct_current_contribution_norms": direct_current_norms.cpu().tolist(),
            "imputed_contribution_norms": imputed_norms.cpu().tolist(),
            "total_slot_contribution_norms": contribution_norms.cpu().tolist(),
            "client_contribution_norms": contribution_norms.cpu().tolist(),
            "client_contribution_cap_respected": certificate,
            "global_cap_active_fraction": float(
                (residual_norms > cap).float().mean().item()
            ),
            "replace_one_bound": 2.0 * cap / float(n),
            "replace_one_bound_formula": (
                "2*G/n_conditional_on_fixed_past_predictor_history_anchor_and_covariance"
            ),
            "k4b_equals_k2_when_all_history_gates_are_one": True,
        }
    )
    return reference, diagnostics


__all__ = [
    "FULL_TEMPORAL_MISSING_SLOT",
    "IMPUTATION_MODES",
    "INCREMENTAL_TEMPORAL_SUPPRESSION",
    "fixed_denominator_past_predictor",
    "gaussian_aware_fixed_anchor_past_imputed_reference",
    "pointwise_optimal_full_imputation_predictor",
]
