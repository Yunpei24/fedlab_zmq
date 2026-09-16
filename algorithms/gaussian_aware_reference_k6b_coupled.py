r"""Bounded magnitude--confidence coupling for the K6b predictor.

K6b's primary rule is a regularised temporal projection.  For two
strictly-past views, corrected alignment ``a``, older-view energy ``b_X`` and
newer direction ``Y`` it deploys

.. math::

   h_r(Y)=\frac{Y}{\max\{\lVert Y\rVert_2,r_{\min}\}},\qquad
   c_{PT}=\frac{(a-\tau_a)_+}{\sqrt{\max\{b_X,b_0\}}},

.. math::

   p_{PT}=\Pi_{B_G}\!\left(c_{PT}h_r(Y)\right).

The threshold, floors, regularised direction and final Euclidean projection
remove the hard singularities of the raw formula.  The previously announced
factorisation through the square-root magnitude
``min(G, sqrt(b_+))`` is continuous but is *not* globally Lipschitz at zero.
It is retained only as an ablation and algebraic-equivalence diagnostic: when
``b_Y>b_min``, ``sqrt(b_Y)<G`` and the radial floor is inactive, multiplying
that magnitude by the EIV cosine confidence cancels ``sqrt(b_Y)`` and gives
the temporal projection above.  All deployed constructions satisfy
``||p|| <= G`` and operate only on already private transcript quantities, so
they do not change the local-DP guarantee of the client mechanism.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch

from robustness.aggregators import clip_l2

MagnitudeRule = Literal["regularized", "sqrt"]


def _finite_scalar_like(
    value: torch.Tensor | float,
    *,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    scalar = torch.as_tensor(
        value, device=reference.device, dtype=reference.dtype
    ).reshape(-1)
    if scalar.numel() != 1 or not bool(torch.isfinite(scalar)):
        raise ValueError(f"{name} must be one finite scalar")
    return scalar.reshape(())


@torch.no_grad()
def regularized_radial_direction(
    newer_direction: torch.Tensor,
    *,
    minimum_direction_norm: float,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Return ``Y / max(||Y||, r_min)`` without a hard branch at zero."""

    value = newer_direction
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError("newer_direction must be a one-dimensional tensor")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("newer_direction must be finite floating point")
    radius = float(minimum_direction_norm)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("minimum_direction_norm must be finite and positive")
    norm = torch.linalg.vector_norm(value)
    denominator = torch.maximum(
        norm, torch.as_tensor(radius, device=value.device, dtype=value.dtype)
    )
    direction = value / denominator
    if not return_diagnostics:
        return direction
    return direction, {
        "formula": "Y/max(norm_Y,r_min)",
        "input_norm": float(norm.item()),
        "minimum_direction_norm": radius,
        "deployed_denominator": float(denominator.item()),
        "direction_norm": float(torch.linalg.vector_norm(direction).item()),
        "continuous": True,
        "global_lipschitz_bound": 1.0 / radius,
    }


@torch.no_grad()
def bounded_energy_magnitude(
    energy: torch.Tensor | float,
    *,
    influence_cap: float,
    b0: float,
    rule: MagnitudeRule = "regularized",
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Map a possibly negative energy estimate to a magnitude in ``[0,G]``.

    ``regularized`` is the K6b primary rule

    .. math:: m_{b_0}(b)=\min\{G,b_+/\sqrt{b_++b_0}\}.

    ``sqrt`` is the preregistered exploratory rule
    ``min(G, sqrt(b_+))``.  It is kept only as an ablation because its slope is
    unbounded as positive ``b`` approaches zero.
    """

    cap = float(influence_cap)
    floor = float(b0)
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("influence_cap must be finite and positive")
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("b0 must be finite and positive")
    if rule not in ("regularized", "sqrt"):
        raise ValueError("rule must be 'regularized' or 'sqrt'")
    inferred_dtype = dtype if dtype is not None else torch.float32
    if isinstance(energy, torch.Tensor):
        if dtype is None:
            inferred_dtype = energy.dtype
        if device is None:
            device = energy.device
    value = torch.as_tensor(energy, device=device, dtype=inferred_dtype).reshape(-1)
    if value.numel() != 1 or not bool(torch.isfinite(value)):
        raise ValueError("energy must be one finite scalar")
    value = value.reshape(())
    positive = torch.clamp(value, min=0.0)
    if rule == "regularized":
        raw = positive / torch.sqrt(
            positive + torch.as_tensor(floor, device=value.device, dtype=value.dtype)
        )
        lipschitz_bound: float | None = 1.0 / math.sqrt(floor)
    else:
        raw = torch.sqrt(positive)
        lipschitz_bound = None
    magnitude = torch.minimum(
        raw, torch.as_tensor(cap, device=value.device, dtype=value.dtype)
    )
    if not return_diagnostics:
        return magnitude
    return magnitude, {
        "formula": (
            "min(G,b_plus/sqrt(b_plus+b0))"
            if rule == "regularized"
            else "min(G,sqrt(b_plus))"
        ),
        "rule": rule,
        "input_energy": float(value.item()),
        "positive_energy": float(positive.item()),
        "b0": floor,
        "influence_cap": cap,
        "raw_magnitude": float(raw.item()),
        "magnitude": float(magnitude.item()),
        "cap_active": bool(float(raw.item()) > cap),
        "continuous": True,
        "globally_lipschitz_in_energy": rule == "regularized",
        "global_lipschitz_bound_in_energy": lipschitz_bound,
        "sqrt_ablation_holder_exponent": 0.5 if rule == "sqrt" else None,
    }


@torch.no_grad()
def coupled_magnitude_confidence_predictor(
    newer_direction: torch.Tensor,
    confidence: torch.Tensor | float,
    magnitude_energy: torch.Tensor | float,
    *,
    influence_cap: float,
    minimum_direction_norm: float,
    b0: float,
    magnitude_rule: MagnitudeRule = "regularized",
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Return ``p=m(b)*kappa*h_r(Y)`` with a public norm cap ``G``."""

    direction, direction_diagnostics = regularized_radial_direction(
        newer_direction,
        minimum_direction_norm=minimum_direction_norm,
        return_diagnostics=True,
    )
    kappa = _finite_scalar_like(
        confidence, reference=newer_direction, name="confidence"
    )
    tolerance = 64.0 * torch.finfo(newer_direction.dtype).eps
    if bool(kappa < -tolerance) or bool(kappa > 1.0 + tolerance):
        raise ValueError("confidence must lie in [0,1]")
    kappa = kappa.clamp(0.0, 1.0)
    magnitude, magnitude_diagnostics = bounded_energy_magnitude(
        magnitude_energy,
        influence_cap=influence_cap,
        b0=b0,
        rule=magnitude_rule,
        device=newer_direction.device,
        dtype=newer_direction.dtype,
        return_diagnostics=True,
    )
    predictor = magnitude * kappa * direction
    predictor_norm = torch.linalg.vector_norm(predictor)
    cap = float(influence_cap)
    if float(predictor_norm.item()) > cap + 128.0 * torch.finfo(
        predictor.dtype
    ).eps * max(1.0, cap):
        raise RuntimeError("K6b predictor violated its public norm cap")
    if not return_diagnostics:
        return predictor
    return predictor, {
        "formula": "magnitude(energy)*kappa*Y/max(norm_Y,r_min)",
        "confidence": float(kappa.item()),
        "magnitude": float(magnitude.item()),
        "predictor_norm": float(predictor_norm.item()),
        "influence_cap": cap,
        "norm_cap_certified": True,
        "local_dp_effect": "unchanged_post_processing_of_private_transcript",
        "direction": direction_diagnostics,
        "magnitude_diagnostics": magnitude_diagnostics,
    }


@torch.no_grad()
def temporal_projection_predictor(
    newer_direction: torch.Tensor,
    alignment: torch.Tensor | float,
    older_energy: torch.Tensor | float,
    *,
    alignment_threshold: float,
    energy_floor: float,
    influence_cap: float,
    minimum_direction_norm: float,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Return the primary K6b regularised temporal projection.

    .. math::

       p_{PT}=\Pi_{B_G}\left[
       \frac{(a-\tau_a)_+}{\sqrt{\max\{b_X,b_0\}}}
       \frac{Y}{\max\{\lVert Y\rVert_2,r_{\min}\}}
       \right].

    The public floor ``b0`` controls division by a near-zero corrected older
    energy, ``r_min`` removes the directional singularity at ``Y=0``, and the
    last projection gives the exact influence certificate ``||p_PT||<=G``.
    """

    direction, direction_diagnostics = regularized_radial_direction(
        newer_direction,
        minimum_direction_norm=minimum_direction_norm,
        return_diagnostics=True,
    )
    a = _finite_scalar_like(alignment, reference=newer_direction, name="alignment")
    bx = _finite_scalar_like(
        older_energy, reference=newer_direction, name="older_energy"
    )
    tau = float(alignment_threshold)
    b0 = float(energy_floor)
    cap = float(influence_cap)
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("alignment_threshold must be finite and non-negative")
    if not math.isfinite(b0) or b0 <= 0.0:
        raise ValueError("energy_floor must be finite and positive")
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("influence_cap must be finite and positive")
    thresholded = torch.clamp(
        a - torch.as_tensor(tau, device=a.device, dtype=a.dtype), min=0.0
    )
    deployed_energy = torch.maximum(
        bx, torch.as_tensor(b0, device=bx.device, dtype=bx.dtype)
    )
    coefficient = thresholded / torch.sqrt(deployed_energy)
    unprojected = coefficient * direction
    predictor = clip_l2(unprojected, cap)
    norm = torch.linalg.vector_norm(predictor)
    if float(norm.item()) > cap + 128.0 * torch.finfo(predictor.dtype).eps * max(
        1.0, cap
    ):
        raise RuntimeError("Temporal projection violated its public norm cap")
    if not return_diagnostics:
        return predictor
    return predictor, {
        "formula": (
            "Proj_B_G(max(a-tau_a,0)/sqrt(max(b_X,b0))" "*Y/max(norm_Y,r_min))"
        ),
        "alignment": float(a.item()),
        "alignment_threshold": tau,
        "thresholded_alignment": float(thresholded.item()),
        "older_energy": float(bx.item()),
        "energy_floor": b0,
        "deployed_older_energy": float(deployed_energy.item()),
        "energy_floor_active": bool(float(bx.item()) < b0),
        "unprojected_coefficient": float(coefficient.item()),
        "unprojected_norm": float(torch.linalg.vector_norm(unprojected).item()),
        "predictor_norm": float(norm.item()),
        "influence_cap": cap,
        "projection_active": bool(
            float(torch.linalg.vector_norm(unprojected).item()) > cap
        ),
        "norm_cap_certified": True,
        "local_dp_effect": "unchanged_post_processing_of_private_transcript",
        "direction": direction_diagnostics,
    }


__all__ = [
    "MagnitudeRule",
    "bounded_energy_magnitude",
    "coupled_magnitude_confidence_predictor",
    "regularized_radial_direction",
    "temporal_projection_predictor",
]
