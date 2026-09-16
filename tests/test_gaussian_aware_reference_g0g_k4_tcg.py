"""Unit tests for the G0g-K4 temporal causal gate primitives."""

from __future__ import annotations

import math

import pytest
import torch

from algorithms.gaussian_aware_reference import (
    gaussian_aware_fixed_anchor_scalar_gated_reference,
    gaussian_aware_fixed_anchor_temporal_gated_reference,
    gaussian_aware_temporal_standardized_messages,
)


def test_temporal_standardization_uses_only_public_dp_variance() -> None:
    vectors = torch.tensor([[2.0, 4.0, 6.0, 8.0]], dtype=torch.float64)
    result, diagnostics = gaussian_aware_temporal_standardized_messages(
        vectors,
        anchor=torch.zeros(4, dtype=torch.float64),
        noise_variances=torch.tensor([[4.0, 16.0]], dtype=torch.float64),
        block_sizes=[2, 2],
        variance_floor=1.0e-12,
        standardized_clip_norm=100.0,
        return_diagnostics=True,
    )
    expected = torch.tensor([[1.0, 2.0, 1.5, 2.0]], dtype=torch.float64)
    assert torch.allclose(result, expected, atol=1.0e-11, rtol=0.0)
    assert diagnostics["heterogeneity_in_whitener"] is False
    assert diagnostics["covariance_role"] == "public_authenticated_dp_covariance_only"


def test_temporal_standardization_rejects_nonpositive_dp_variance() -> None:
    with pytest.raises(ValueError, match="noise_variances must be strictly positive"):
        gaussian_aware_temporal_standardized_messages(
            torch.ones((2, 2), dtype=torch.float64),
            anchor=torch.zeros(2, dtype=torch.float64),
            noise_variances=torch.tensor([[1.0], [0.0]], dtype=torch.float64),
            block_sizes=[2],
        )


def test_temporal_standardization_clips_max_float_without_zero_or_nan() -> None:
    maximum = torch.finfo(torch.float32).max
    result, diagnostics = gaussian_aware_temporal_standardized_messages(
        torch.tensor([[maximum, 0.0]], dtype=torch.float32),
        anchor=torch.zeros(2, dtype=torch.float32),
        noise_variances=torch.ones((1, 1), dtype=torch.float32),
        block_sizes=[2],
        variance_floor=1.0e-12,
        standardized_clip_norm=4.0,
        return_diagnostics=True,
    )
    assert bool(torch.isfinite(result).all())
    assert float(result[0, 0]) == pytest.approx(4.0)
    assert float(torch.linalg.vector_norm(result, dim=1)[0]) == pytest.approx(4.0)
    assert diagnostics["stable_scaled_norm_and_clipping"] is True


def test_temporal_standardization_rejects_whitening_overflow_explicitly() -> None:
    info = torch.finfo(torch.float32)
    with pytest.raises(ValueError, match="temporal whitening overflowed"):
        gaussian_aware_temporal_standardized_messages(
            torch.tensor([[info.max]], dtype=torch.float32),
            anchor=torch.zeros(1, dtype=torch.float32),
            noise_variances=torch.tensor([[info.tiny]], dtype=torch.float32),
            block_sizes=[1],
            variance_floor=info.tiny,
            standardized_clip_norm=2.0,
        )


def test_temporal_standardization_rejects_variance_plus_ridge_overflow() -> None:
    maximum = torch.finfo(torch.float32).max
    with pytest.raises(ValueError, match="variance plus numerical ridge"):
        gaussian_aware_temporal_standardized_messages(
            torch.ones((1, 1), dtype=torch.float32),
            anchor=torch.zeros(1, dtype=torch.float32),
            noise_variances=torch.tensor([[maximum]], dtype=torch.float32),
            block_sizes=[1],
            variance_floor=maximum,
            standardized_clip_norm=2.0,
        )


def test_temporal_standardization_rejects_residual_subtraction_overflow() -> None:
    maximum = torch.finfo(torch.float32).max
    with pytest.raises(ValueError, match="vectors minus anchor overflowed"):
        gaussian_aware_temporal_standardized_messages(
            torch.tensor([[maximum]], dtype=torch.float32),
            anchor=torch.tensor([-maximum], dtype=torch.float32),
            noise_variances=torch.ones((1, 1), dtype=torch.float32),
            block_sizes=[1],
            variance_floor=1.0e-12,
            standardized_clip_norm=2.0,
        )


def test_all_one_temporal_gate_is_exactly_k2_aware() -> None:
    vectors = torch.tensor(
        [[0.05, 0.02], [-0.03, 0.01], [0.02, -0.04]], dtype=torch.float64
    )
    anchor = torch.zeros(2, dtype=torch.float64)
    radii = torch.ones((3, 1), dtype=torch.float64)
    history = torch.zeros((3, 4, 2), dtype=torch.float64)
    enrollment = torch.zeros((3, 2), dtype=torch.float64)
    k2 = gaussian_aware_fixed_anchor_scalar_gated_reference(
        vectors,
        anchor=anchor,
        statistical_radii=radii,
        block_sizes=[2],
        influence_cap=0.13,
        return_diagnostics=False,
    )
    k4, diagnostics = gaussian_aware_fixed_anchor_temporal_gated_reference(
        vectors,
        anchor=anchor,
        statistical_radii=radii,
        temporal_standardized_history=history,
        enrollment_standardized_mean=enrollment,
        enrollment_size=8,
        temporal_gate_inner_threshold=1.0,
        temporal_gate_outer_threshold=2.0,
        block_sizes=[2],
        influence_cap=0.13,
        return_diagnostics=True,
    )
    assert torch.equal(k4, k2)
    assert diagnostics["temporal_gates_by_client"] == [1.0, 1.0, 1.0]
    assert diagnostics["normalization_by_gate_sum"] is False


def test_temporal_gate_is_causal_with_respect_to_current_upload() -> None:
    history = torch.zeros((2, 4, 2), dtype=torch.float64)
    history[1] = 2.0
    enrollment = torch.zeros((2, 2), dtype=torch.float64)
    kwargs = {
        "anchor": torch.zeros(2, dtype=torch.float64),
        "statistical_radii": torch.ones((2, 1), dtype=torch.float64) * 100.0,
        "temporal_standardized_history": history,
        "enrollment_standardized_mean": enrollment,
        "enrollment_size": 8,
        "temporal_gate_inner_threshold": 1.0,
        "temporal_gate_outer_threshold": 3.0,
        "block_sizes": [2],
        "influence_cap": 0.13,
        "return_diagnostics": True,
    }
    _, left = gaussian_aware_fixed_anchor_temporal_gated_reference(
        torch.zeros((2, 2), dtype=torch.float64), **kwargs
    )
    _, right = gaussian_aware_fixed_anchor_temporal_gated_reference(
        torch.tensor([[100.0, -100.0], [-50.0, 75.0]], dtype=torch.float64),
        **kwargs,
    )
    assert left["temporal_statistics_by_client"] == right[
        "temporal_statistics_by_client"
    ]
    assert left["temporal_gates_by_client"] == right["temporal_gates_by_client"]
    assert left["gates_by_client"] != right["gates_by_client"]
    assert left["history_gate_is_past_measurable"] is True
    assert left["final_gate_is_past_measurable"] is False
    assert left["final_gate_depends_on_current_upload_via_current_gate"] is True


def test_temporal_scale_and_detection_gate_match_protocol() -> None:
    history = torch.zeros((1, 4, 2), dtype=torch.float64)
    history[:] = 1.0
    _, diagnostics = gaussian_aware_fixed_anchor_temporal_gated_reference(
        torch.zeros((1, 2), dtype=torch.float64),
        anchor=torch.zeros(2, dtype=torch.float64),
        statistical_radii=torch.ones((1, 1), dtype=torch.float64),
        temporal_standardized_history=history,
        enrollment_standardized_mean=torch.zeros((1, 2), dtype=torch.float64),
        enrollment_size=8,
        temporal_gate_inner_threshold=1.0,
        temporal_gate_outer_threshold=3.0,
        block_sizes=[2],
        influence_cap=0.13,
        return_diagnostics=True,
    )
    scale = math.sqrt(1.0 / 4.0 + 1.0 / 8.0)
    expected_statistic = math.sqrt(2.0) / scale
    expected_gate = max(0.0, min(1.0, (3.0 - expected_statistic) / 2.0))
    assert diagnostics["temporal_nominal_scale"] == pytest.approx(scale)
    assert diagnostics["temporal_scale_is_exact_z_score"] is False
    assert diagnostics["temporal_statistics_by_client"][0] == pytest.approx(
        expected_statistic
    )
    assert diagnostics["temporal_gates_by_client"][0] == pytest.approx(expected_gate)


def test_temporal_k4_clips_max_float_input_without_nan_or_silent_zero() -> None:
    maximum = torch.finfo(torch.float32).max
    reference, diagnostics = gaussian_aware_fixed_anchor_temporal_gated_reference(
        torch.tensor([[maximum, 0.0]], dtype=torch.float32),
        anchor=torch.zeros(2, dtype=torch.float32),
        statistical_radii=torch.tensor([[maximum]], dtype=torch.float32),
        temporal_standardized_history=torch.zeros((1, 4, 2), dtype=torch.float32),
        enrollment_standardized_mean=torch.zeros((1, 2), dtype=torch.float32),
        enrollment_size=8,
        temporal_gate_inner_threshold=1.0,
        temporal_gate_outer_threshold=2.0,
        block_sizes=[2],
        influence_cap=0.13,
        return_diagnostics=True,
    )
    assert bool(torch.isfinite(reference).all())
    assert float(reference[0]) == pytest.approx(0.13)
    assert float(reference[0]) != 0.0
    assert all(
        math.isfinite(float(value))
        for value in diagnostics["current_normalized_residuals_by_client"]
    )


def test_temporal_k4_rejects_normalized_residual_overflow_explicitly() -> None:
    info = torch.finfo(torch.float32)
    with pytest.raises(ValueError, match="normalized residuals overflowed"):
        gaussian_aware_fixed_anchor_temporal_gated_reference(
            torch.tensor([[info.max]], dtype=torch.float32),
            anchor=torch.zeros(1, dtype=torch.float32),
            statistical_radii=torch.tensor([[info.tiny]], dtype=torch.float32),
            temporal_standardized_history=torch.zeros(
                (1, 4, 1), dtype=torch.float32
            ),
            enrollment_standardized_mean=torch.zeros((1, 1), dtype=torch.float32),
            enrollment_size=8,
            temporal_gate_inner_threshold=1.0,
            temporal_gate_outer_threshold=2.0,
            block_sizes=[1],
            influence_cap=0.13,
        )


def test_current_round_replace_one_bound_is_attainable_conditionally() -> None:
    cap = 0.13
    vectors = torch.tensor([[10.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    neighbour = vectors.clone()
    neighbour[0] = torch.tensor([-10.0, 0.0], dtype=torch.float64)
    common = {
        "anchor": torch.zeros(2, dtype=torch.float64),
        "statistical_radii": torch.full((2, 1), 1.0e6, dtype=torch.float64),
        "temporal_standardized_history": torch.zeros(
            (2, 4, 2), dtype=torch.float64
        ),
        "enrollment_standardized_mean": torch.zeros((2, 2), dtype=torch.float64),
        "enrollment_size": 8,
        "temporal_gate_inner_threshold": 1.0,
        "temporal_gate_outer_threshold": 2.0,
        "block_sizes": [2],
        "influence_cap": cap,
    }
    left, diagnostics = gaussian_aware_fixed_anchor_temporal_gated_reference(
        vectors, return_diagnostics=True, **common
    )
    right = gaussian_aware_fixed_anchor_temporal_gated_reference(
        neighbour, return_diagnostics=False, **common
    )
    difference = torch.linalg.vector_norm(left - right).item()
    assert difference == pytest.approx(2.0 * cap / 2.0)
    assert diagnostics["replace_one_bound"] == pytest.approx(2.0 * cap / 2.0)
