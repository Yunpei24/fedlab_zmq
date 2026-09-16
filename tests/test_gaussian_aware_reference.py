"""Synthetic certificate tests for the Gaussian-aware Huber reference."""

from __future__ import annotations

import math

import pytest
import torch

from algorithms.gaussian_aware_reference import (
    gaussian_aware_huber_leave_one_out,
    gaussian_aware_huber_reference,
    gaussian_radial_thresholds,
    standardized_quadratic_scores,
)


def test_public_radial_threshold_scales_with_block_dimension():
    thresholds = gaussian_radial_thresholds(
        [1, 16, 64], tail_probability=0.01, dtype=torch.float64
    )
    assert torch.all(thresholds[1:] > thresholds[:-1])
    x = math.log(100.0)
    expected = math.sqrt(16.0 + 2.0 * math.sqrt(16.0 * x) + 2.0 * x)
    assert float(thresholds[1]) == pytest.approx(expected)


def test_central_quadratic_zone_preserves_equal_client_estimand():
    vectors = torch.tensor(
        [
            [0.20, -0.10, 0.10, 0.05],
            [0.00, 0.10, -0.10, 0.15],
            [0.10, 0.00, 0.05, -0.05],
        ],
        dtype=torch.float64,
    )
    anchor = torch.tensor([0.05, 0.05, 0.05, 0.05], dtype=torch.float64)
    common = dict(
        anchor=anchor,
        block_sizes=[2, 2],
        standardized_threshold=100.0,
        influence_cap=100.0,
        regularization=0.5,
        num_steps=60,
    )
    low_noise = gaussian_aware_huber_reference(
        vectors, noise_variances=torch.ones(3, 2), **common
    )
    heterogeneous_noise = gaussian_aware_huber_reference(
        vectors,
        noise_variances=torch.tensor(
            [[1.0, 9.0], [16.0, 1.0], [4.0, 25.0]], dtype=torch.float64
        ),
        **common,
    )
    expected = (vectors.mean(dim=0) + 0.5 * anchor) / 1.5
    assert torch.allclose(low_noise, expected, atol=1e-12, rtol=0.0)
    assert torch.allclose(heterogeneous_noise, expected, atol=1e-12, rtol=0.0)


def test_finite_solver_replace_one_certificate_is_respected():
    generator = torch.Generator().manual_seed(17)
    vectors = 0.2 * torch.randn(8, 4, generator=generator, dtype=torch.float64)
    replacement = vectors.clone()
    replacement[3] = torch.tensor([50.0, -80.0, 100.0, -30.0], dtype=torch.float64)
    variances = torch.linspace(0.04, 0.16, 8, dtype=torch.float64)[:, None].repeat(1, 2)
    kwargs = dict(
        anchor=torch.zeros(4, dtype=torch.float64),
        noise_variances=variances,
        block_sizes=[2, 2],
        standardized_threshold=None,
        null_tail_probability=0.01,
        influence_cap=[0.3, 0.4],
        regularization=0.7,
        num_steps=8,
    )
    first, diagnostics = gaussian_aware_huber_reference(
        vectors, return_diagnostics=True, **kwargs
    )
    second = gaussian_aware_huber_reference(replacement, **kwargs)
    observed = float(torch.linalg.vector_norm(first - second))
    assert observed <= diagnostics["finite_solver_replace_one_bound"] + 1e-12
    assert (
        diagnostics["finite_solver_replace_one_bound"]
        < diagnostics["exact_minimizer_replace_one_bound"]
    )
    assert diagnostics["certificate_requires_fixed_public_covariances"] is True


def test_covariance_limited_fraction_detects_active_and_masked_branches():
    vectors = torch.zeros(4, 4, dtype=torch.float64)
    common = dict(
        anchor=torch.zeros(4, dtype=torch.float64),
        noise_variances=torch.tensor(
            [[1e-4, 1e-4], [4e-4, 4e-4], [1e-4, 1e-4], [4e-4, 4e-4]],
            dtype=torch.float64,
        ),
        block_sizes=[2, 2],
        standardized_threshold=2.0,
        regularization=0.5,
        num_steps=2,
        return_diagnostics=True,
    )
    _, active = gaussian_aware_huber_reference(
        vectors, influence_cap=[1.0, 1.0], **common
    )
    _, masked = gaussian_aware_huber_reference(
        vectors, influence_cap=[1e-6, 1e-6], **common
    )
    assert active["fraction_covariance_limited_client_blocks"] == 1.0
    assert active["covariance_branch_active"] is True
    assert masked["fraction_covariance_limited_client_blocks"] == 0.0
    assert masked["covariance_branch_active"] is False


def test_covariance_limited_tail_detects_effective_huber_branch():
    vectors = torch.full((4, 4), 0.10, dtype=torch.float64)
    _, diagnostics = gaussian_aware_huber_reference(
        vectors,
        anchor=torch.zeros(4, dtype=torch.float64),
        noise_variances=torch.full((4, 2), 1e-4, dtype=torch.float64),
        block_sizes=[2, 2],
        standardized_threshold=2.0,
        influence_cap=[1.0, 1.0],
        regularization=2.0,
        num_steps=1,
        return_diagnostics=True,
    )
    assert diagnostics["fraction_covariance_limited_client_blocks"] == 1.0
    assert (
        diagnostics[
            "fraction_covariance_limited_and_huber_tail_client_blocks"
        ]
        == 1.0
    )
    assert diagnostics["covariance_branch_changes_influence_at_final_iterate"] is True


def test_leave_one_out_reference_is_independent_of_excluded_upload():
    vectors = torch.tensor(
        [[0.0, 0.0], [0.1, -0.1], [-0.2, 0.1], [0.05, 0.2]],
        dtype=torch.float64,
    )
    variances = torch.tensor([0.04, 0.09, 0.16, 0.25], dtype=torch.float64)
    kwargs = dict(
        anchor=torch.zeros(2, dtype=torch.float64),
        noise_variances=variances,
        influence_cap=0.4,
        regularization=0.5,
        num_steps=12,
    )
    references = gaussian_aware_huber_leave_one_out(vectors, **kwargs)
    changed = vectors.clone()
    changed[1] = torch.tensor([100.0, -100.0], dtype=torch.float64)
    changed_references = gaussian_aware_huber_leave_one_out(changed, **kwargs)
    assert references.shape == vectors.shape
    assert torch.equal(references[1], changed_references[1])

    manual = gaussian_aware_huber_reference(
        vectors[torch.tensor([True, False, True, True])],
        anchor=kwargs["anchor"],
        noise_variances=variances[torch.tensor([True, False, True, True])],
        influence_cap=kwargs["influence_cap"],
        regularization=kwargs["regularization"],
        num_steps=kwargs["num_steps"],
    )
    assert torch.equal(references[1], manual)


def test_standardized_quadratic_score_removes_scalar_noise_scale_under_null():
    vectors = torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float64)
    variances = torch.tensor([1.0, 4.0], dtype=torch.float64)
    scores, diagnostics = standardized_quadratic_scores(
        vectors,
        references=torch.zeros(2, dtype=torch.float64),
        noise_variances=variances,
        variance_floor=1e-15,
        return_diagnostics=True,
    )
    assert torch.allclose(
        diagnostics["quadratic_energy"], torch.tensor([2.0, 2.0], dtype=torch.float64)
    )
    assert torch.allclose(
        diagnostics["null_mean"], torch.tensor([2.0, 2.0], dtype=torch.float64)
    )
    assert torch.allclose(scores, torch.zeros_like(scores), atol=1e-14)


def test_standardized_score_is_empirically_pivotal_across_noise_tiers():
    generator = torch.Generator().manual_seed(123)
    cohort_per_tier = 10_000
    dimension = 8
    low = torch.randn(
        cohort_per_tier, dimension, generator=generator, dtype=torch.float64
    )
    high = 3.0 * torch.randn(
        cohort_per_tier, dimension, generator=generator, dtype=torch.float64
    )
    vectors = torch.cat((low, high))
    variances = torch.cat(
        (
            torch.ones(cohort_per_tier, dtype=torch.float64),
            torch.full((cohort_per_tier,), 9.0, dtype=torch.float64),
        )
    )
    scores = standardized_quadratic_scores(
        vectors,
        references=torch.zeros(dimension, dtype=torch.float64),
        noise_variances=variances,
        variance_floor=1e-15,
    )
    low_scores = scores[:cohort_per_tier]
    high_scores = scores[cohort_per_tier:]
    assert abs(float(low_scores.mean())) < 0.04
    assert abs(float(high_scores.mean())) < 0.04
    assert float(low_scores.std(unbiased=False)) == pytest.approx(1.0, abs=0.04)
    assert float(high_scores.std(unbiased=False)) == pytest.approx(1.0, abs=0.04)
    assert abs(float(low_scores.mean() - high_scores.mean())) < 0.05


def test_general_quadratic_moments_use_null_to_scoring_variance_ratio():
    vectors = torch.zeros(2, 3, dtype=torch.float64)
    scores, diagnostics = standardized_quadratic_scores(
        vectors,
        references=torch.zeros_like(vectors),
        noise_variances=torch.full((2,), 4.0, dtype=torch.float64),
        scoring_variances=torch.full((2,), 2.0, dtype=torch.float64),
        variance_floor=1e-15,
        return_diagnostics=True,
    )
    assert torch.allclose(
        diagnostics["null_mean"], torch.full((2,), 6.0, dtype=torch.float64)
    )
    assert torch.allclose(
        diagnostics["null_variance"], torch.full((2,), 24.0, dtype=torch.float64)
    )
    assert torch.allclose(scores, torch.full_like(scores, -6.0 / math.sqrt(24.0)))


def test_huber_reference_limits_a_large_byzantine_outlier():
    honest = torch.tensor(
        [
            [-0.04, 0.02],
            [0.03, -0.01],
            [0.01, 0.04],
            [-0.02, -0.03],
            [0.00, 0.01],
            [0.02, 0.00],
            [-0.01, 0.03],
            [0.01, -0.02],
        ],
        dtype=torch.float64,
    )
    byzantine = torch.tensor([[100.0, -100.0], [80.0, -120.0]], dtype=torch.float64)
    vectors = torch.cat((honest, byzantine), dim=0)
    reference, diagnostics = gaussian_aware_huber_reference(
        vectors,
        anchor=torch.zeros(2, dtype=torch.float64),
        noise_variances=torch.full((10,), 0.01, dtype=torch.float64),
        standardized_threshold=3.0,
        influence_cap=0.25,
        regularization=0.2,
        num_steps=25,
        return_diagnostics=True,
    )
    honest_centre = honest.mean(dim=0)
    mean_error = torch.linalg.vector_norm(vectors.mean(dim=0) - honest_centre)
    robust_error = torch.linalg.vector_norm(reference - honest_centre)
    assert robust_error < 0.02 * mean_error
    assert diagnostics["tail_fraction_client_blocks"] >= 0.2
    assert diagnostics["objective_final"] <= diagnostics["objective_initial"]


def test_float16_input_is_promoted_to_preserve_small_certificates():
    vectors = torch.tensor([[0.0, 0.1], [0.2, -0.1]], dtype=torch.float16)
    result = gaussian_aware_huber_reference(
        vectors,
        anchor=torch.zeros(2),
        noise_variances=torch.ones(2),
        regularization=1.0,
        influence_cap=1.0,
        num_steps=3,
    )
    assert result.dtype == torch.float32
    assert bool(torch.isfinite(result).all())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"block_sizes": [1, 2]}, "sum"),
        ({"noise_variances": [-1.0, 1.0]}, "non-negative"),
        ({"step_size": 10.0}, "contractive"),
    ],
)
def test_invalid_public_configuration_is_rejected(kwargs, match):
    base = dict(
        anchor=torch.zeros(2, dtype=torch.float64),
        noise_variances=torch.ones(2, dtype=torch.float64),
        regularization=0.5,
        num_steps=3,
    )
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        gaussian_aware_huber_reference(torch.zeros(2, 2, dtype=torch.float64), **base)


def test_n_equals_num_blocks_requires_unambiguous_variance_shape():
    vectors = torch.zeros(2, 2, dtype=torch.float64)
    common = dict(
        anchor=torch.zeros(2, dtype=torch.float64),
        block_sizes=[1, 1],
        regularization=0.5,
        num_steps=2,
    )
    with pytest.raises(ValueError, match="ambiguous"):
        gaussian_aware_huber_reference(
            vectors, noise_variances=torch.ones(2, dtype=torch.float64), **common
        )
    per_client = gaussian_aware_huber_reference(
        vectors,
        noise_variances=torch.ones(2, 1, dtype=torch.float64),
        **common,
    )
    per_block = gaussian_aware_huber_reference(
        vectors,
        noise_variances=torch.ones(1, 2, dtype=torch.float64),
        **common,
    )
    assert torch.equal(per_client, per_block)


def test_half_precision_rounding_cannot_break_reported_stability_bound():
    # This is a regression test for a counterexample where casting each
    # certified output back to float16 created a jump much larger than the
    # real-arithmetic replace-one bound.
    anchor = torch.tensor([1.00048828125], dtype=torch.float64)
    first = torch.tensor([[2.0], [2.0]], dtype=torch.float16)
    second = torch.tensor([[2.0], [0.0]], dtype=torch.float16)
    kwargs = dict(
        anchor=anchor,
        noise_variances=torch.ones(2, dtype=torch.float32),
        standardized_threshold=10.0,
        influence_cap=1e-5,
        regularization=1.0,
        num_steps=1,
    )
    first_reference, diagnostics = gaussian_aware_huber_reference(
        first, return_diagnostics=True, **kwargs
    )
    second_reference = gaussian_aware_huber_reference(second, **kwargs)
    observed = float(torch.linalg.vector_norm(first_reference - second_reference))
    assert first_reference.dtype == torch.float32
    assert observed <= diagnostics["finite_solver_replace_one_bound"] + 1e-7


def test_effective_variance_overflow_is_rejected():
    with pytest.raises(ValueError, match="overflowed"):
        gaussian_aware_huber_reference(
            torch.zeros(2, 1, dtype=torch.float32),
            anchor=torch.zeros(1, dtype=torch.float32),
            noise_variances=torch.full((2,), 3e38, dtype=torch.float32),
            heterogeneity_variances=3e38,
            influence_cap=1.0,
            regularization=1.0,
            num_steps=1,
        )


def test_float32_variance_floor_underflow_is_rejected():
    with pytest.raises(ValueError, match="underflows"):
        gaussian_aware_huber_reference(
            torch.zeros(2, 1, dtype=torch.float32),
            anchor=torch.zeros(1, dtype=torch.float32),
            noise_variances=torch.zeros(2, dtype=torch.float32),
            variance_floor=1e-50,
            influence_cap=1.0,
            regularization=1.0,
            num_steps=1,
        )
