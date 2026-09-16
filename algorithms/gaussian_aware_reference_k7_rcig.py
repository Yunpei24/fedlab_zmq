r"""Robust covariance-standardised temporal fusion for Gaussian-aware FAR.

The construction uses two strictly-past, already private and norm-bounded
reference views ``X`` (older) and ``Y`` (newer).  With public/attested noise
covariances ``V_X`` and ``V_Y`` and a public process covariance ``Q``, define

.. math::

   S = V_X + V_Y + Q + \lambda I,
   \qquad
   r = \sqrt{(Y-X)^\top S^{-1}(Y-X)}.

The trust placed in the newer view is the Huber radial weight

.. math::

   w_c(r) = \min\{1,c/\max(r,\varepsilon_r)\},

and the deployed predictor is

.. math::

   P = \Pi_{B_G}\bigl(X + w_c(r)(Y-X)\bigr).

Thus a statistically ordinary innovation gives exactly ``P=Y``.  An
innovation large relative to its expected DP-plus-drift covariance is shrunk
towards the older view.  Unlike the rejected K6/K6b scalar-amplitude rules,
this interpolation can change direction relative to ``Y``.

This module only post-processes private transcript quantities; it does not
change the client-side local-DP guarantee.  It does not by itself prove
Byzantine robustness: that requires assumptions ensuring the older view is
reliable and the covariance metadata is not adversarially understated.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch

from robustness.aggregators import clip_l2

CovarianceMode = Literal["full", "isotropic"]


def _check_vector(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional tensor")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite floating point")


def _check_covariance(
    value: torch.Tensor,
    *,
    dimension: int,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension})")
    if value.device != reference.device or value.dtype != reference.dtype:
        raise ValueError(f"{name} must share device and dtype with the views")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return 0.5 * (value + value.T)


@torch.no_grad()
def robust_covariance_innovation_fusion(
    older_view: torch.Tensor,
    newer_view: torch.Tensor,
    covariance_older: torch.Tensor,
    covariance_newer: torch.Tensor,
    *,
    process_variance: float,
    ridge: float,
    innovation_threshold: float,
    influence_cap: float,
    covariance_mode: CovarianceMode = "full",
    radial_epsilon: float = 1.0e-12,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Fuse two private views using a covariance-standardised Huber gate.

    ``covariance_mode='isotropic'`` replaces the innovation covariance by an
    isotropic matrix with the same average variance.  It is a required control
    for identifying whether covariance orientation adds value beyond a scalar
    noise scale.
    """

    _check_vector(older_view, name="older_view")
    _check_vector(newer_view, name="newer_view")
    if older_view.shape != newer_view.shape:
        raise ValueError("older_view and newer_view must have the same shape")
    if older_view.device != newer_view.device or older_view.dtype != newer_view.dtype:
        raise ValueError("older_view and newer_view must share device and dtype")
    dimension = int(older_view.numel())
    covariance_x = _check_covariance(
        covariance_older,
        dimension=dimension,
        reference=older_view,
        name="covariance_older",
    )
    covariance_y = _check_covariance(
        covariance_newer,
        dimension=dimension,
        reference=older_view,
        name="covariance_newer",
    )
    scalars = {
        "process_variance": float(process_variance),
        "ridge": float(ridge),
        "innovation_threshold": float(innovation_threshold),
        "influence_cap": float(influence_cap),
        "radial_epsilon": float(radial_epsilon),
    }
    if not all(math.isfinite(value) for value in scalars.values()):
        raise ValueError("all scalar parameters must be finite")
    if scalars["process_variance"] < 0.0:
        raise ValueError("process_variance must be non-negative")
    if scalars["ridge"] <= 0.0:
        raise ValueError("ridge must be positive")
    if scalars["innovation_threshold"] <= 0.0:
        raise ValueError("innovation_threshold must be positive")
    if scalars["influence_cap"] <= 0.0:
        raise ValueError("influence_cap must be positive")
    if scalars["radial_epsilon"] <= 0.0:
        raise ValueError("radial_epsilon must be positive")
    if covariance_mode not in ("full", "isotropic"):
        raise ValueError("covariance_mode must be 'full' or 'isotropic'")

    identity = torch.eye(dimension, device=older_view.device, dtype=older_view.dtype)
    innovation_covariance = (
        covariance_x + covariance_y + scalars["process_variance"] * identity
    )
    if covariance_mode == "isotropic":
        mean_variance = torch.trace(innovation_covariance) / dimension
        innovation_covariance = mean_variance * identity
    regularized_covariance = innovation_covariance + scalars["ridge"] * identity
    innovation = newer_view - older_view
    whitened = torch.linalg.solve(regularized_covariance, innovation[:, None]).reshape(
        -1
    )
    mahalanobis_squared = torch.dot(innovation, whitened).clamp_min(0.0)
    standardized_innovation = torch.sqrt(mahalanobis_squared)
    denominator = torch.clamp(
        standardized_innovation,
        min=scalars["radial_epsilon"],
    )
    trust = torch.minimum(
        torch.ones((), device=older_view.device, dtype=older_view.dtype),
        torch.as_tensor(
            scalars["innovation_threshold"],
            device=older_view.device,
            dtype=older_view.dtype,
        )
        / denominator,
    )
    fused_unprojected = older_view + trust * innovation
    predictor = clip_l2(fused_unprojected, scalars["influence_cap"])

    predictor_norm = torch.linalg.vector_norm(predictor)
    tolerance = (
        128.0 * torch.finfo(predictor.dtype).eps * max(1.0, scalars["influence_cap"])
    )
    if float(predictor_norm.item()) > scalars["influence_cap"] + tolerance:
        raise RuntimeError("RCIG predictor violated its public norm cap")
    if not return_diagnostics:
        return predictor

    # No eigendecomposition is needed for the certificate: covariance PSD and
    # ridge > 0 imply lambda_min >= ridge, while lambda_max <= trace.
    trace_upper_bound = float(torch.trace(regularized_covariance).item())
    minimum_eigenvalue_lower_bound = scalars["ridge"]
    return predictor, {
        "formula": "Proj_BG(X + min(1,c/max(r,eps))*(Y-X))",
        "covariance_mode": covariance_mode,
        "standardized_innovation": float(standardized_innovation.item()),
        "innovation_threshold": scalars["innovation_threshold"],
        "newer_view_trust": float(trust.item()),
        "gate_active": bool(
            float(standardized_innovation.item()) > scalars["innovation_threshold"]
        ),
        "innovation_norm": float(torch.linalg.vector_norm(innovation).item()),
        "older_norm": float(torch.linalg.vector_norm(older_view).item()),
        "newer_norm": float(torch.linalg.vector_norm(newer_view).item()),
        "predictor_norm": float(predictor_norm.item()),
        "projection_active": bool(
            float(torch.linalg.vector_norm(fused_unprojected).item())
            > scalars["influence_cap"]
        ),
        "process_variance": scalars["process_variance"],
        "ridge": scalars["ridge"],
        "minimum_regularized_covariance_eigenvalue_lower_bound": (
            minimum_eigenvalue_lower_bound
        ),
        "maximum_regularized_covariance_eigenvalue_upper_bound": trace_upper_bound,
        "condition_number_upper_bound": (
            trace_upper_bound / minimum_eigenvalue_lower_bound
        ),
        "norm_cap_certified": True,
        "ordinary_innovation_returns_identity_y": not bool(
            float(standardized_innovation.item()) > scalars["innovation_threshold"]
        ),
        "local_dp_effect": "unchanged_post_processing_of_private_transcript",
    }


@torch.no_grad()
def euclidean_innovation_fusion(
    older_view: torch.Tensor,
    newer_view: torch.Tensor,
    *,
    innovation_threshold: float,
    influence_cap: float,
    radial_epsilon: float = 1.0e-12,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Non-Gaussian-aware control using the raw Euclidean innovation norm."""

    _check_vector(older_view, name="older_view")
    _check_vector(newer_view, name="newer_view")
    if older_view.shape != newer_view.shape:
        raise ValueError("older_view and newer_view must have the same shape")
    threshold = float(innovation_threshold)
    cap = float(influence_cap)
    epsilon = float(radial_epsilon)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("innovation_threshold must be finite and positive")
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("influence_cap must be finite and positive")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("radial_epsilon must be finite and positive")
    innovation = newer_view - older_view
    innovation_norm = torch.linalg.vector_norm(innovation)
    trust = torch.minimum(
        torch.ones((), device=older_view.device, dtype=older_view.dtype),
        torch.as_tensor(threshold, device=older_view.device, dtype=older_view.dtype)
        / torch.clamp(innovation_norm, min=epsilon),
    )
    predictor = clip_l2(older_view + trust * innovation, cap)
    if not return_diagnostics:
        return predictor
    return predictor, {
        "formula": "Proj_BG(X + min(1,c_e/max(norm(Y-X),eps))*(Y-X))",
        "innovation_norm": float(innovation_norm.item()),
        "innovation_threshold": threshold,
        "newer_view_trust": float(trust.item()),
        "gate_active": bool(float(innovation_norm.item()) > threshold),
        "predictor_norm": float(torch.linalg.vector_norm(predictor).item()),
        "norm_cap_certified": True,
        "local_dp_effect": "unchanged_post_processing_of_private_transcript",
    }
