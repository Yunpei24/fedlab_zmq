from __future__ import annotations

import math

import torch

from algorithms.gaussian_aware_reference_k5_tp_v2 import (
    FEATURE_NAMES,
    REMOVED_V1_FEATURE,
    fit_shared_scalar_ridge,
    forbidden_current_field_count,
    training_feature_scales,
    transcript_past_feature_dictionary,
    transcript_past_predictor,
)


def _history() -> tuple[torch.Tensor, torch.Tensor]:
    residuals = torch.zeros(4, 3, 2, dtype=torch.float32)
    residuals[:, :, 0] = torch.tensor([0.01, 0.02, 0.03, 0.04])[:, None]
    return residuals, torch.ones(4, 3, dtype=torch.float32)


def test_five_feature_dictionary_removes_only_redundant_newest_u() -> None:
    residuals, gates = _history()
    features, diagnostics = transcript_past_feature_dictionary(
        residuals,
        gates,
        minimum_accepted_mass=2.0,
        influence_cap=0.13,
        return_diagnostics=True,
    )
    assert features.shape == (5, 2)
    assert tuple(diagnostics["feature_names"]) == FEATURE_NAMES
    assert diagnostics["removed_v1_feature"] == REMOVED_V1_FEATURE
    assert diagnostics["uses_current_round_input"] is False
    assert diagnostics["most_recent_source_lag"] == 1
    assert diagnostics["oldest_source_lag"] == 4
    assert torch.allclose(
        features[:, 0], torch.tensor([0.025, 0.04, 0.01, 0.01, 0.025])
    )


def test_five_feature_design_can_have_exact_rank_five() -> None:
    generator = torch.Generator().manual_seed(2027051001)
    features = torch.randn(30, 5, 8, generator=generator, dtype=torch.float32)
    scales, diagnostics = training_feature_scales(
        features, rms_floor=1.0e-8, return_diagnostics=True
    )
    assert scales.shape == (5,)
    assert diagnostics["flattened_design_rank"] == 5
    assert diagnostics["feature_count"] == 5
    assert tuple(diagnostics["feature_names"]) == FEATURE_NAMES


def test_ridge_and_predictor_use_exactly_five_coefficients() -> None:
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(60, 5, 7, generator=generator) * 0.01
    expected = torch.tensor([0.4, -0.2, 0.1, 0.3, 0.05])
    targets = torch.einsum("j,hjd->hd", expected, features)
    weights = torch.ones(60)
    scales = training_feature_scales(features, rms_floor=1.0e-8)
    coefficients = fit_shared_scalar_ridge(
        features,
        targets,
        weights,
        ridge_lambda=1.0e-6,
        feature_scales=scales,
    )
    assert coefficients.shape == (5,)
    predictor, diagnostics = transcript_past_predictor(
        features[0],
        coefficients * 1.0e6,
        influence_cap=0.13,
        feature_scales=scales,
        return_diagnostics=True,
    )
    assert math.isclose(float(torch.linalg.vector_norm(predictor)), 0.13, abs_tol=1e-6)
    assert diagnostics["projection_active"] is True


def test_current_round_inference_fields_are_rejected_by_allowlist_audit() -> None:
    safe = {
        "past_feature_dictionary_sha256": "a" * 64,
        "frozen_coefficients_sha256": "b" * 64,
    }
    assert forbidden_current_field_count(safe) == 0
    assert forbidden_current_field_count({**safe, "current_noise": []}) == 1
