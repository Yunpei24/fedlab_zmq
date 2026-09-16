r"""Current-randomness conditional semi-oracle primitives for G0g-K4c-CH.

This module does **not** define a deployable, transcript-only predictor.  It
estimates the best predictor in the K4b fixed-denominator imputation family for
squared Euclidean risk, conditional on the privileged sigma-field
``G_t^-``: one frozen observable past, the latent clean honest current target,
simulator-only honest/Byzantine labels, and the preregistered current noise and
attack generator.  Only fresh DP-noise and attack randomness are resampled
across children.  Construction and evaluation child streams must be
independent; that contract is enforced by the runner.

For a fixed past, let

``S = sum_i g_i c_i``, ``R = sum_i (1-h_i)`` and ``v = target - anchor``.

The K4b aggregate is ``anchor + (S + R p) / n``.  The missing-slot mass ``R``
is measurable from the observable frozen past, even though the semi-oracle is
conditioned on the larger ``G_t^-``.  The conditional-MSE minimizer over
``||p||_2 <= G`` is

    p_semi = Proj_{B_G}((n E[v | G_t^-] - E[S | G_t^-]) / R).

The pointwise K4b oracle is intentionally different: it observes the current
target and direct sum and can compensate the whole current reference error.
It is a nondeployable decision benchmark and must never be used to construct
or promote ``p_semi``.

After all validity and Monte-Carlo stability checks pass, failure of the
finite-``M`` semi-oracle mechanism is a preregistered stopping result for this
DGP and this implementation. It does not establish failure of the exact
conditional optimum: that would require analytic expectations or a formal
construction-error confidence bound. Success is only a DGP-specific
feasibility signal for a later, strictly transcript-only study; it is not
evidence that such a deployable predictor has already been constructed.
Neither outcome is a universal theorem for arbitrary adaptive Byzantine
strategies.

The compensated error is not identified as DP-noise error alone: it may also
contain attack, gate/clipping, and anchor-bias components.  The screen has no
no-noise/no-attack counterfactual with which to separate them.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from algorithms.gaussian_aware_reference_k4b import _stable_clip_rows


def _finite_matrix(
    value: torch.Tensor, *, name: str, rows: int | None = None
) -> tuple[int, int]:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{name} must have shape (m,d) with m,d >= 1")
    if rows is not None and int(value.shape[0]) != rows:
        raise ValueError(f"{name} must contain exactly {rows} children")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite floating point")
    return int(value.shape[0]), int(value.shape[1])


@torch.no_grad()
def current_randomness_conditional_mse_semi_oracle(
    construction_target_directions: torch.Tensor,
    construction_direct_sums: torch.Tensor,
    *,
    missing_slot_mass: float,
    num_clients: int,
    influence_cap: float,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Estimate the clean-state-privileged conditional-MSE semi-oracle.

    Both input matrices have one row per independent *construction* child.
    No evaluation child may occur in either matrix.  The function averages
    ``v`` and ``S`` separately and projects only once; averaging pointwise
    projected oracles would estimate a different object.
    """

    children, dimension = _finite_matrix(
        construction_target_directions,
        name="construction_target_directions",
    )
    direct_children, direct_dimension = _finite_matrix(
        construction_direct_sums,
        name="construction_direct_sums",
        rows=children,
    )
    if direct_children != children or direct_dimension != dimension:
        raise ValueError("construction matrices must have identical shape")
    if construction_direct_sums.device != construction_target_directions.device:
        raise ValueError("construction matrices must use the same device")
    n = int(num_clients)
    if n < 1:
        raise ValueError("num_clients must be positive")
    mass = float(missing_slot_mass)
    if not math.isfinite(mass) or not 0.0 <= mass <= float(n):
        raise ValueError("missing_slot_mass must lie in [0,n]")
    cap = float(influence_cap)
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("influence_cap must be finite and positive")

    work_dtype = (
        torch.float64
        if construction_target_directions.dtype == torch.float64
        else torch.float32
    )
    target = construction_target_directions.to(dtype=work_dtype)
    direct = construction_direct_sums.to(dtype=work_dtype)
    expected_target = target.mean(dim=0)
    expected_direct = direct.mean(dim=0)
    # This numerical zero-mass convention is shared with the frozen K4b pointwise
    # benchmark and the production eligibility rule.  It is deliberately not
    # multiplied by n: R is already a scalar sum of gate deficits.
    tolerance = 64.0 * torch.finfo(work_dtype).eps
    if mass <= tolerance:
        raw = torch.zeros_like(expected_target)
        predictor = torch.zeros_like(expected_target)
    else:
        raw = (float(n) * expected_target - expected_direct) / mass
        _, bounded = _stable_clip_rows(
            raw[None, :], cap, name="current-randomness conditional MSE semi-oracle"
        )
        predictor = bounded[0]
    if not bool(torch.isfinite(predictor).all()):
        raise RuntimeError(
            "current-randomness conditional MSE semi-oracle is not finite"
        )
    if not return_diagnostics:
        return predictor
    return predictor, {
        "predictor_name": (
            "current_randomness_conditional_clean_state_mse_semi_oracle"
        ),
        "construction_children": children,
        "dimension": dimension,
        "missing_slot_mass": mass,
        "num_clients": n,
        "expected_target_direction_norm": float(
            torch.linalg.vector_norm(expected_target).item()
        ),
        "expected_direct_sum_norm": float(
            torch.linalg.vector_norm(expected_direct).item()
        ),
        "raw_predictor_norm": float(torch.linalg.vector_norm(raw).item()),
        "predictor_norm": float(torch.linalg.vector_norm(predictor).item()),
        "predictor_cap": cap,
        "projection_applied_once_after_expectations": True,
        "pointwise_oracles_averaged": False,
        "construction_and_evaluation_independence_required_from_caller": True,
        "uses_current_evaluation_target": False,
        "uses_current_evaluation_noise": False,
        "conditions_on_latent_clean_current_state": True,
        "uses_simulator_honest_byzantine_labels": True,
        "knows_configured_current_noise_and_attack_law": True,
        "supports_arbitrary_adaptive_byzantine_behavior": False,
        "compensatory_headroom_components_separately_identified": False,
        "no_noise_no_attack_counterfactual_included": False,
        "possible_compensated_components": (
            "attack_gate_clipping_anchor_bias_dp_noise_attack_randomness"
        ),
        "conditioning_sigma_field": (
            "G_t_minus_observable_past_latent_clean_current_state_"
            "simulator_labels_and_configured_current_dgp"
        ),
        "measurable_with_respect_to_observable_past_only": False,
        "deployable": False,
        "privacy_claimed": False,
        "loss_optimized": "conditional_squared_l2_reference_error",
    }


@torch.no_grad()
def fixed_denominator_imputed_reference_from_sum(
    *,
    anchor: torch.Tensor,
    direct_sum: torch.Tensor,
    predictor: torch.Tensor,
    missing_slot_mass: float,
    num_clients: int,
    influence_cap: float,
) -> torch.Tensor:
    """Evaluate ``anchor + (direct_sum + R predictor) / n`` safely."""

    if not isinstance(anchor, torch.Tensor) or anchor.ndim != 1:
        raise ValueError("anchor must be a vector")
    if not anchor.is_floating_point() or not bool(torch.isfinite(anchor).all()):
        raise ValueError("anchor must be finite floating point")
    dtype = torch.float64 if anchor.dtype == torch.float64 else torch.float32
    direct = torch.as_tensor(direct_sum, device=anchor.device, dtype=dtype).reshape(-1)
    raw_predictor = torch.as_tensor(
        predictor, device=anchor.device, dtype=dtype
    ).reshape(-1)
    if direct.shape != anchor.shape or raw_predictor.shape != anchor.shape:
        raise ValueError("anchor, direct_sum and predictor must share dimension")
    if not bool(torch.isfinite(direct).all()) or not bool(
        torch.isfinite(raw_predictor).all()
    ):
        raise ValueError("direct_sum and predictor must be finite")
    n = int(num_clients)
    mass = float(missing_slot_mass)
    cap = float(influence_cap)
    if n < 1 or not math.isfinite(mass) or not 0.0 <= mass <= float(n):
        raise ValueError("invalid public cohort size or missing-slot mass")
    _, bounded = _stable_clip_rows(
        raw_predictor[None, :], cap, name="imputation predictor"
    )
    result = anchor.to(dtype=dtype) + (direct + mass * bounded[0]) / float(n)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("fixed-denominator imputed reference is not finite")
    return result


__all__ = [
    "current_randomness_conditional_mse_semi_oracle",
    "fixed_denominator_imputed_reference_from_sum",
]
