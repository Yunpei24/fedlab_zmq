from __future__ import annotations

import math

import pytest
import torch

from algorithms.gaussian_aware_reference_k5_tp import (
    FEATURE_NAMES,
    _condition_number_infinity,
    _modified_gram_schmidt_rank,
    fit_shared_scalar_ridge,
    forbidden_current_field_count,
    training_feature_scales,
    transcript_past_feature_dictionary,
    transcript_past_predictor,
)


def _history() -> tuple[torch.Tensor, torch.Tensor]:
    residuals = torch.zeros(4, 3, 2, dtype=torch.float32)
    residuals[0, :, 0] = 0.01
    residuals[1, :, 0] = 0.02
    residuals[2, :, 0] = 0.03
    residuals[3, :, 0] = 0.04
    gates = torch.ones(4, 3, dtype=torch.float32)
    return residuals, gates


def test_feature_dictionary_is_newest_first_and_past_only() -> None:
    residuals, gates = _history()
    features, diagnostics = transcript_past_feature_dictionary(
        residuals,
        gates,
        minimum_accepted_mass=2.0,
        influence_cap=0.13,
        return_diagnostics=True,
    )
    assert features.shape == (len(FEATURE_NAMES), 2)
    assert torch.allclose(
        features[:, 0], torch.tensor([0.025, 0.04, 0.01, 0.04, 0.01, 0.025])
    )
    assert diagnostics["uses_current_round_input"] is False
    assert diagnostics["most_recent_source_lag"] == 1
    assert diagnostics["oldest_source_lag"] == 4


def test_feature_dictionary_uses_public_denominator_floor() -> None:
    residuals, gates = _history()
    gates[:] = 0.0
    gates[:, 0] = 1.0
    features, diagnostics = transcript_past_feature_dictionary(
        residuals,
        gates,
        minimum_accepted_mass=2.0,
        influence_cap=0.13,
        return_diagnostics=True,
    )
    assert diagnostics["denominator_chronological"] == [2.0] * 4
    assert torch.allclose(
        features[:, 0],
        torch.tensor([0.0125, 0.02, 0.005, 0.04 / 3.0, 0.01 / 3.0, 0.025 / 3.0]),
    )


def test_predictor_projection_enforces_public_cap() -> None:
    residuals, gates = _history()
    features = transcript_past_feature_dictionary(
        residuals,
        gates,
        minimum_accepted_mass=2.0,
        influence_cap=0.13,
    )
    predictor, diagnostics = transcript_past_predictor(
        features,
        torch.full((6,), 100.0),
        influence_cap=0.13,
        return_diagnostics=True,
    )
    assert math.isclose(float(torch.linalg.vector_norm(predictor)), 0.13, abs_tol=1e-6)
    assert diagnostics["projection_active"] is True
    assert diagnostics["uses_current_round_input"] is False


def test_ridge_recovers_shared_coefficients_without_projection() -> None:
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(40, 6, 5, generator=generator, dtype=torch.float32) * 0.02
    expected = torch.tensor([0.5, -0.2, 0.1, 0.3, 0.05, 0.0])
    targets = torch.einsum("j,hjd->hd", expected, features)
    weights = torch.linspace(0.1, 1.0, 40)
    actual = fit_shared_scalar_ridge(
        features,
        targets,
        weights,
        ridge_lambda=1.0e-8,
    )
    assert torch.allclose(actual, expected, atol=2.0e-4, rtol=2.0e-4)


def test_training_scales_are_rms_and_report_full_rank() -> None:
    generator = torch.Generator().manual_seed(11)
    features = torch.randn(30, 6, 8, generator=generator) * torch.arange(1, 7)[
        None, :, None
    ]
    scales, diagnostics = training_feature_scales(
        features, rms_floor=1.0e-8, return_diagnostics=True
    )
    expected = torch.sqrt(torch.mean(features.square(), dim=(0, 2)))
    assert torch.allclose(scales, expected)
    assert diagnostics["flattened_design_rank"] == 6
    assert diagnostics["floor_active_count"] == 0
    assert diagnostics["rank_diagnostics"]["method"] == (
        "two_pass_modified_gram_schmidt_on_unit_norm_columns"
    )


@pytest.mark.parametrize(
    ("matrix", "expected_rank"),
    [
        (torch.eye(6, dtype=torch.float32), 6),
        (
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 3.0, 4.0],
                    [2.0, 5.0, 7.0],
                ],
                dtype=torch.float32,
            ),
            2,
        ),
    ],
)
def test_mps_compatible_rank_matches_cpu_matrix_rank(
    matrix: torch.Tensor, expected_rank: int
) -> None:
    actual, diagnostics = _modified_gram_schmidt_rank(matrix)
    cpu_reference = int(torch.linalg.matrix_rank(matrix).item())
    assert actual == expected_rank == cpu_reference
    assert diagnostics["compute_device"] == "cpu"


def test_infinity_condition_number_matches_explicit_cpu_inverse() -> None:
    system = torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    expected = float(
        (
            torch.linalg.matrix_norm(system, ord=float("inf"))
            * torch.linalg.matrix_norm(torch.linalg.inv(system), ord=float("inf"))
        ).item()
    )
    assert _condition_number_infinity(system) == pytest.approx(expected)


def test_ridge_rejects_nonpositive_weights() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        fit_shared_scalar_ridge(
            torch.zeros(2, 6, 3),
            torch.zeros(2, 3),
            torch.tensor([1.0, 0.0]),
            ridge_lambda=0.1,
        )


def test_forbidden_current_field_audit() -> None:
    assert forbidden_current_field_count(
        {"past_features": [], "coefficients": [], "influence_cap": 0.13}
    ) == 0
    assert forbidden_current_field_count(
        {"past_features": [], "current_noise": [], "semi_oracle_target": []}
    ) == 2


@pytest.mark.parametrize("bad_lambda", [0.0, -1.0, float("inf"), float("nan")])
def test_ridge_rejects_invalid_regularization(bad_lambda: float) -> None:
    with pytest.raises(ValueError, match="ridge_lambda"):
        fit_shared_scalar_ridge(
            torch.zeros(2, 6, 3),
            torch.zeros(2, 3),
            torch.ones(2),
            ridge_lambda=bad_lambda,
        )
