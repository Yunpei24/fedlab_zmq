r"""Five-feature transcript-past predictor for G0g-K5-TP-v2.

K5-v1 exposed six past-only vector features.  Its pre-evaluation fit audit
found that ``u_{t-1}`` was numerically redundant with ``m_{t-1}``: before the
public-ball projection, ``u_r=(D_r/n)m_r`` with
``D_r=max(m_min,sum_i g_{i,r})``, and the public floor was active through most
of the preregistered design.  V2 therefore removes ``u_{t-1}`` *before any
evaluation seed is opened*.  It retains the five non-redundant features and
the same strictly transcript-past inference contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from algorithms import gaussian_aware_reference_k5_tp as v1

FEATURE_NAMES = (
    "rolling_accepted_mean",
    "accepted_mean_t_minus_1",
    "accepted_mean_delta_t_minus_1_t_minus_2",
    "fixed_denominator_delta_t_minus_1_t_minus_2",
    "rolling_fixed_denominator_direction",
)
V1_RETAINED_INDICES = (0, 1, 2, 4, 5)
REMOVED_V1_FEATURE = "fixed_denominator_direction_t_minus_1"


@torch.no_grad()
def transcript_past_feature_dictionary(
    clipped_residual_history: torch.Tensor,
    accepted_gate_history: torch.Tensor,
    *,
    minimum_accepted_mass: float,
    influence_cap: float,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Return the fixed five-vector dictionary from rounds ``t-4,...,t-1``."""

    full, diagnostics = v1.transcript_past_feature_dictionary(
        clipped_residual_history,
        accepted_gate_history,
        minimum_accepted_mass=minimum_accepted_mass,
        influence_cap=influence_cap,
        return_diagnostics=True,
    )
    features = full[list(V1_RETAINED_INDICES)]
    if features.shape[0] != len(FEATURE_NAMES):
        raise RuntimeError("K5-TP-v2 feature schema changed unexpectedly")
    diagnostics = dict(diagnostics)
    diagnostics.update(
        {
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "removed_v1_feature": REMOVED_V1_FEATURE,
            "retained_v1_indices": list(V1_RETAINED_INDICES),
            "structural_relation_before_projection": (
                "u_r=(max(m_min,sum_i_g_i_r)/n)*m_r_unprojected"
            ),
        }
    )
    diagnostics.pop("fixed_denominator_direction_norm", None)
    return (features, diagnostics) if return_diagnostics else features


def training_feature_scales(
    feature_tensor: torch.Tensor,
    *,
    rms_floor: float,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Use the locked v1 per-coordinate RMS and MPS rank implementation."""

    result = v1.training_feature_scales(
        feature_tensor,
        rms_floor=rms_floor,
        return_diagnostics=return_diagnostics,
    )
    if not return_diagnostics:
        return result
    scales, diagnostics = result
    diagnostics = dict(diagnostics)
    diagnostics["feature_names"] = list(FEATURE_NAMES[: int(feature_tensor.shape[1])])
    return scales, diagnostics


def fit_shared_scalar_ridge(
    feature_tensor: torch.Tensor,
    targets: torch.Tensor,
    aggregate_weights: torch.Tensor,
    *,
    ridge_lambda: float,
    feature_scales: torch.Tensor | Sequence[float] | None = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Fit five shared scalar coefficients using the locked MPS ridge solver."""

    result = v1.fit_shared_scalar_ridge(
        feature_tensor,
        targets,
        aggregate_weights,
        ridge_lambda=ridge_lambda,
        feature_scales=feature_scales,
        return_diagnostics=return_diagnostics,
    )
    if not return_diagnostics:
        return result
    coefficients, diagnostics = result
    diagnostics = dict(diagnostics)
    diagnostics["feature_names"] = list(FEATURE_NAMES[: int(feature_tensor.shape[1])])
    return coefficients, diagnostics


def transcript_past_predictor(
    features: torch.Tensor,
    coefficients: torch.Tensor | Sequence[float],
    *,
    influence_cap: float,
    feature_scales: torch.Tensor | Sequence[float] | None = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Apply the frozen five-feature predictor and project it on ``B_G``."""

    result = v1.transcript_past_predictor(
        features,
        coefficients,
        influence_cap=influence_cap,
        feature_scales=feature_scales,
        return_diagnostics=return_diagnostics,
    )
    if not return_diagnostics:
        return result
    predictor, diagnostics = result
    diagnostics = dict(diagnostics)
    diagnostics["feature_names"] = list(FEATURE_NAMES[: int(features.shape[0])])
    return predictor, diagnostics


def forbidden_current_field_count(inference_payload: Mapping[str, Any]) -> int:
    """Reuse the locked allowlist audit for current-round inference fields."""

    return v1.forbidden_current_field_count(inference_payload)


__all__ = [
    "FEATURE_NAMES",
    "REMOVED_V1_FEATURE",
    "V1_RETAINED_INDICES",
    "fit_shared_scalar_ridge",
    "forbidden_current_field_count",
    "training_feature_scales",
    "transcript_past_feature_dictionary",
    "transcript_past_predictor",
]
