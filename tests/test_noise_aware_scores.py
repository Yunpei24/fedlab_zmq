"""Unit tests for the public DP-noise-aware FAR score geometry."""

from __future__ import annotations

import math

import pytest
import torch

from algorithms.noise_aware_scores import (
    NULL_MC_THEIL_SEN_SEPARATE_TRUST_SCORE_MODE,
    NULL_MC_TIER_ROBUST_SEPARATE_TRUST_SCORE_MODE,
    ROBUST_DIRECT_SCORE_MODES,
    calibrated_independence_trust_scores,
    calibrated_null_energy_scores,
    debiased_distance_scores,
    directional_trust_scores,
    effective_upload_noise_variances,
    excess_energy_scores,
    lagged_descent_alignment_scores,
    leave_one_out_mean_residual_variances,
    mean_reference_residual_variances,
    mix_isotropic_covariance_variances,
    multi_krum_admissibility_trust_scores,
    null_mc_moment_scores,
    null_quantile_scores,
    orthogonalize_scores_against_public_scale,
    pairwise_independence_trust_scores,
    ranked_directional_peer_support_scores,
    robust_novelty_scores,
    separate_novelty_trust_scores,
    shrink_null_energy_moments,
    shrink_residual_variances,
    standardize_distances,
    temporal_projection_scores,
    theil_sen_orthogonalize_scores_against_public_scale,
    tierwise_midrank_scores,
)
from robustness.aggregators import centered_clipping, centered_clipping_leave_one_out


def test_homogeneous_covariance_proxy_is_exactly_neutral():
    distances = torch.tensor([0.1, 0.4, 0.9], dtype=torch.float64)
    scales = torch.ones(3, dtype=torch.float64)
    corrected, divisors, diagnostics = standardize_distances(
        distances,
        scales,
        mode="isotropic_dp_covariance_proxy",
    )
    assert torch.equal(corrected, distances)
    assert torch.equal(divisors, torch.ones_like(divisors))
    assert diagnostics["noise_score_public_scale_homogeneous"] is True
    assert diagnostics["noise_score_covariance_is_proxy"] is True


def test_multi_krum_trust_ranks_a_separated_byzantine_pair_last():
    honest = torch.tensor(
        [[1.00, 0.00], [1.02, 0.01], [0.98, -0.01], [1.01, -0.02], [0.99, 0.02]],
        dtype=torch.float64,
    )
    byzantine = torch.tensor([[-8.0, 0.0], [-8.0, 0.0]], dtype=torch.float64)
    trust, diagnostics = multi_krum_admissibility_trust_scores(
        torch.cat((honest, byzantine)), assumed_byzantine=2
    )
    assert diagnostics["noise_score_multi_krum_neighbour_count"] == 3
    assert float(trust[:5].min()) > float(trust[5:].max())
    assert torch.equal(trust[5:], torch.zeros(2, dtype=torch.float64))


def test_multi_krum_trust_rejects_an_invalid_resilience_regime():
    with pytest.raises(ValueError, match="n >= 2f\\+3"):
        multi_krum_admissibility_trust_scores(torch.zeros(6, 2), assumed_byzantine=2)


def test_covariance_proxy_reduces_a_large_noise_scale_residual():
    distances = torch.ones(3, dtype=torch.float64)
    scales = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float64)
    corrected, divisors, diagnostics = standardize_distances(
        distances,
        scales,
        mode="isotropic_dp_covariance_proxy",
    )
    assert corrected[-1] < corrected[0]
    assert divisors[-1] > divisors[0]
    assert math.isclose(float(divisors.mean()), 1.0, rel_tol=1e-12)
    assert diagnostics["noise_score_reference_variance_factor"] == pytest.approx(1 / 3)


def test_client_scale_ablation_and_covariance_proxy_are_distinct():
    distances = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
    scales = torch.tensor([1.0, 1.5, 2.0], dtype=torch.float64)
    client_only, _, _ = standardize_distances(
        distances, scales, mode="client_noise_scale"
    )
    covariance, _, _ = standardize_distances(
        distances, scales, mode="isotropic_dp_covariance_proxy"
    )
    assert not torch.allclose(client_only, covariance)


def test_noise_aware_mode_requires_public_scales():
    with pytest.raises(ValueError, match="public scale"):
        standardize_distances(
            torch.tensor([0.1, 0.2]),
            None,
            mode="isotropic_dp_covariance_proxy",
        )


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="noise_score_standardization"):
        standardize_distances(
            torch.tensor([0.1, 0.2]), torch.ones(2), mode="oracle_covariance"
        )


def test_mean_reference_residual_variance_matches_direct_formula():
    variances = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
    observed = mean_reference_residual_variances(variances)
    expected = torch.tensor(
        [
            (2 / 3) ** 2 * 1.0 + (4.0 + 9.0) / 9,
            (2 / 3) ** 2 * 4.0 + (1.0 + 9.0) / 9,
            (2 / 3) ** 2 * 9.0 + (1.0 + 4.0) / 9,
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(observed, expected)


def test_leave_one_out_residual_variance_matches_direct_formula():
    variances = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
    observed = leave_one_out_mean_residual_variances(variances)
    expected = torch.tensor(
        [1.0 + (4.0 + 9.0) / 4, 4.0 + (1.0 + 9.0) / 4, 9.0 + (1.0 + 4.0) / 4],
        dtype=torch.float64,
    )
    assert torch.allclose(observed, expected)


def test_excess_energy_removes_the_gaussian_dimension_floor():
    d = 100
    variances = torch.ones(3, dtype=torch.float64)
    pure_noise_distance = torch.full((3,), math.sqrt(d), dtype=torch.float64)
    scores, diagnostics = excess_energy_scores(
        pure_noise_distance,
        variances,
        score_dimension=d,
        z_clip=5.0,
    )
    assert torch.equal(scores, torch.zeros_like(scores))
    assert diagnostics["noise_score_noise_floor_subtracted"] is True

    clean_scores, _ = excess_energy_scores(
        torch.full((3,), math.sqrt(2 * d), dtype=torch.float64),
        variances,
        score_dimension=d,
        z_clip=5.0,
        subtract_noise_floor=False,
    )
    assert bool((clean_scores > 0).all())


def test_debiased_distance_removes_noise_floor_in_original_units():
    dimension = 100
    variances = torch.tensor([0.01, 0.04, 0.09], dtype=torch.float64)
    pure_noise_distances = torch.sqrt(dimension * variances)
    scores, diagnostics = debiased_distance_scores(
        pure_noise_distances,
        variances,
        score_dimension=dimension,
        distance_clip=2.0,
    )
    assert torch.allclose(scores, torch.zeros_like(scores), atol=1e-8)
    assert diagnostics["noise_score_noise_floor_subtracted"] is True
    assert diagnostics["noise_score_zero_rate"] == pytest.approx(1.0)


def test_debiased_distance_targets_same_absolute_signal_across_noise_scales():
    dimension = 64
    clean_distance = 0.6
    variances = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
    expected_noisy_distances = torch.sqrt(clean_distance**2 + dimension * variances)
    scores, _ = debiased_distance_scores(
        expected_noisy_distances,
        variances,
        score_dimension=dimension,
        distance_clip=1.2,
    )
    assert torch.allclose(
        scores, torch.full_like(scores, clean_distance / 1.2), atol=1e-12
    )

    # The standardized excess-energy score estimates signal-to-noise ratio,
    # so the same absolute signal intentionally yields different values.
    snr_scores, _ = excess_energy_scores(
        expected_noisy_distances,
        variances,
        score_dimension=dimension,
        z_clip=5.0,
    )
    assert snr_scores[0] > snr_scores[1] > snr_scores[2]


def test_temporal_projection_standardises_public_noise_variance():
    residuals = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    predictable = torch.tensor([[3.0, 0.0], [5.0, 0.0]], dtype=torch.float64)
    variances = torch.tensor([1.0, 4.0], dtype=torch.float64)
    scores, diagnostics = temporal_projection_scores(
        residuals,
        variances,
        predictable,
        lower_z=0.0,
        upper_z=2.0,
    )
    assert torch.allclose(scores, torch.tensor([0.5, 0.5], dtype=torch.float64))
    assert diagnostics["noise_score_temporal_history_coverage"] == pytest.approx(1.0)
    assert diagnostics["noise_score_temporal_is_postprocessing"] is True


def test_temporal_projection_is_uniform_without_past_direction():
    residuals = torch.tensor([[1.0, 0.0], [-2.0, 0.0]], dtype=torch.float64)
    scores, diagnostics = temporal_projection_scores(
        residuals,
        torch.ones(2, dtype=torch.float64),
        torch.zeros_like(residuals),
    )
    assert torch.equal(scores, torch.zeros_like(scores))
    assert diagnostics["noise_score_temporal_history_coverage"] == pytest.approx(0.0)


def test_debiased_distance_noise_free_oracle_is_bounded_raw_distance():
    distances = torch.tensor([0.2, 0.5, 1.5], dtype=torch.float64)
    variances = torch.ones_like(distances)
    scores, _ = debiased_distance_scores(
        distances,
        variances,
        score_dimension=10,
        distance_clip=1.0,
        subtract_noise_floor=False,
    )
    assert torch.equal(scores, torch.tensor([0.2, 0.5, 1.0], dtype=torch.float64))


def test_direct_score_modes_are_not_distance_standardizers():
    for mode in (
        "isotropic_dp_excess_energy",
        "isotropic_dp_debiased_distance",
    ):
        with pytest.raises(ValueError, match="produces bounded scores directly"):
            standardize_distances(torch.tensor([0.1, 0.2]), torch.ones(2), mode=mode)

    for mode in ROBUST_DIRECT_SCORE_MODES:
        with pytest.raises(ValueError, match="produces bounded scores directly"):
            standardize_distances(torch.tensor([0.1, 0.2]), torch.ones(2), mode=mode)


def test_ranked_peer_support_requires_more_peers_than_a_byzantine_block():
    # Three colluding copies cannot self-certify when f=3 and one additional
    # supporter is required.  The six honest messages share a direction.
    honest = torch.tensor(
        [[1.0, 0.02 * offset] for offset in (-2, -1, 0, 1, 2, 3)],
        dtype=torch.float64,
    )
    colluding = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    vectors = torch.cat((honest, colluding), dim=0)
    reference = torch.tensor([1.0, 0.0], dtype=torch.float64)
    support, diagnostics = ranked_directional_peer_support_scores(
        vectors,
        reference,
        assumed_byzantine=3,
        reference_reject_cosine=0.0,
        reference_full_support_cosine=0.5,
        peer_reject_cosine=0.0,
        peer_full_support_cosine=0.5,
        extra_honest_supporters=1,
    )
    assert bool((support[:6] > 0.95).all())
    assert torch.equal(support[6:], torch.zeros(3, dtype=torch.float64))
    assert diagnostics["noise_score_peer_support_required_rank"] == 4
    assert (
        diagnostics["noise_score_peer_support_is_identification_certificate"] is False
    )


def test_ranked_peer_support_is_bounded_and_accepts_leave_one_out_references():
    vectors = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.8, -0.1], [0.7, 0.2]],
        dtype=torch.float64,
    )
    references = vectors.mean(dim=0)[None, :].expand_as(vectors).clone()
    support, _ = ranked_directional_peer_support_scores(
        vectors,
        references,
        assumed_byzantine=1,
    )
    assert support.shape == (4,)
    assert bool(((support >= 0.0) & (support <= 1.0)).all())


def test_public_dpsgd_covariance_uses_steps_and_server_contraction():
    updates = []
    for client_id, sigma in enumerate((1.0, 2.0)):
        updates.append(
            (
                {"weight": torch.zeros(1)},
                {
                    "client_id": client_id,
                    "privacy_noise_multiplier": sigma,
                    "privacy_normalization_denominator": 10.0,
                    "model_steps": 4,
                },
                None,
            )
        )
    variances, diagnostics = effective_upload_noise_variances(
        updates,
        config={"lr": 0.5, "clip_norm": 2.0, "momentum": 0.0},
        server_clip_factors=torch.tensor([1.0, 0.5]),
        device="cpu",
        dtype=torch.float64,
    )
    # Before server contraction: [0.04, 0.16]. The second variance is
    # multiplied by 0.5^2, making both effective scalar proxies equal.
    assert torch.allclose(variances, torch.tensor([0.04, 0.04], dtype=torch.float64))
    assert diagnostics["noise_score_includes_server_contraction"] is True


def test_variance_shrinkage_has_declared_endpoints():
    variances = torch.tensor([1.0, 4.0, 10.0], dtype=torch.float64)
    assert torch.equal(shrink_residual_variances(variances, shrinkage=0.0), variances)
    assert torch.equal(
        shrink_residual_variances(variances, shrinkage=1.0),
        torch.full_like(variances, 5.0),
    )
    halfway = shrink_residual_variances(variances, shrinkage=0.5)
    assert torch.equal(halfway, torch.tensor([3.0, 4.5, 7.5], dtype=torch.float64))


def test_individual_covariance_weight_has_unambiguous_endpoints():
    variances = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64)
    pooled = mix_isotropic_covariance_variances(
        variances, individual_covariance_weight=0.0
    )
    individual = mix_isotropic_covariance_variances(
        variances, individual_covariance_weight=1.0
    )
    halfway = mix_isotropic_covariance_variances(
        variances, individual_covariance_weight=0.5
    )
    assert torch.equal(pooled, torch.full_like(variances, 3.0))
    assert torch.equal(individual, variances)
    assert torch.equal(halfway, torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))


def test_effective_null_moment_shrinkage_has_declared_endpoints():
    calibration = torch.tensor(
        [[0.0, 10.0], [2.0, 14.0], [4.0, 18.0]], dtype=torch.float64
    )
    pooled_mean, pooled_variance = shrink_null_energy_moments(
        calibration, individual_moment_weight=0.0
    )
    client_mean, client_variance = shrink_null_energy_moments(
        calibration, individual_moment_weight=1.0
    )
    assert torch.equal(pooled_mean, torch.full((2,), 8.0, dtype=torch.float64))
    assert torch.equal(client_mean, torch.tensor([2.0, 14.0], dtype=torch.float64))
    assert torch.allclose(
        pooled_variance,
        torch.full_like(pooled_variance, calibration.reshape(-1).var(unbiased=True)),
    )
    assert torch.equal(client_variance, calibration.var(dim=0, unbiased=True))


@pytest.mark.parametrize("mode", ["moment", "quantile"])
def test_effective_null_scores_are_bounded_and_deterministic(mode):
    observed = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
    calibration = torch.tensor(
        [
            [0.5, 2.0, 5.0],
            [0.8, 3.0, 6.0],
            [1.0, 4.0, 7.0],
            [1.2, 5.0, 8.0],
            [1.5, 6.0, 9.0],
        ]
        * 4,
        dtype=torch.float64,
    )
    first, diagnostics = calibrated_null_energy_scores(
        observed,
        calibration,
        mode=mode,
        individual_calibration_weight=0.5,
    )
    second, _ = calibrated_null_energy_scores(
        observed,
        calibration,
        mode=mode,
        individual_calibration_weight=0.5,
    )
    assert torch.equal(first, second)
    assert bool((first >= 0.0).all()) and bool((first <= 1.0).all())
    assert diagnostics["noise_score_individual_calibration_weight"] == 0.5


def test_empirical_null_quantile_score_is_monotone_in_observed_energy():
    calibration = torch.arange(20, dtype=torch.float64)[:, None].repeat(1, 3)
    observed = torch.tensor([2.0, 10.0, 19.0], dtype=torch.float64)
    scores, _ = calibrated_null_energy_scores(
        observed,
        calibration,
        mode="quantile",
        tail_probability=0.0,
        individual_calibration_weight=1.0,
    )
    assert scores[0] < scores[1] < scores[2]


def test_individual_moment_weight_changes_heterogeneous_null_scores():
    base = torch.linspace(1.0, 3.0, 20, dtype=torch.float64)
    calibration = torch.stack((base, 10.0 * base), dim=1)
    observed = torch.tensor([3.0, 30.0], dtype=torch.float64)
    pooled, _ = calibrated_null_energy_scores(
        observed,
        calibration,
        mode="moment",
        individual_calibration_weight=0.0,
    )
    individual, _ = calibrated_null_energy_scores(
        observed,
        calibration,
        mode="moment",
        individual_calibration_weight=1.0,
    )
    assert not torch.allclose(pooled, individual)


def test_null_quantile_score_is_invariant_to_variance_at_equal_null_quantile():
    dimension = 128
    variances = torch.tensor([0.01, 0.04, 0.16], dtype=torch.float64)
    # Equal chi-square energy T/d produces the same Wilson--Hilferty z-score.
    distances = torch.sqrt(1.5 * dimension * variances)
    scores, diagnostics = null_quantile_scores(
        distances,
        variances,
        score_dimension=dimension,
        lower_z=0.0,
        upper_z=4.0,
    )
    assert torch.allclose(scores, scores[:1].expand_as(scores), atol=1e-12)
    assert diagnostics["noise_score_null_is_approximation"] is True


def test_directional_trust_rejects_opposite_and_accepts_aligned_vectors():
    vectors = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    reference = torch.tensor([1.0, 0.0])
    trust, diagnostics = directional_trust_scores(
        vectors,
        reference,
        reject_cosine=-0.1,
        full_trust_cosine=0.25,
    )
    assert trust[0] == pytest.approx(1.0)
    assert trust[1] == pytest.approx(0.0)
    assert 0.0 < float(trust[2]) < 1.0
    assert diagnostics["noise_score_alignment_min"] == pytest.approx(-1.0)


def test_lagged_descent_alignment_orders_aligned_orthogonal_and_opposite():
    vectors = torch.tensor(
        [[1.0, 0.0], [0.25, 1.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float64,
    )
    scores, diagnostics = lagged_descent_alignment_scores(
        vectors,
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        reject_cosine=0.0,
        full_support_cosine=0.5,
    )
    assert scores[0] == pytest.approx(1.0)
    assert 0.0 < float(scores[1]) < 1.0
    assert scores[2] == pytest.approx(0.0)
    assert scores[3] == pytest.approx(0.0)
    assert diagnostics["noise_score_lagged_descent_direction_available"] is True
    assert (
        diagnostics["noise_score_lagged_descent_is_loss_decrease_certificate"] is False
    )
    assert diagnostics["noise_score_lagged_descent_is_private_postprocessing"] is True


def test_lagged_descent_alignment_is_neutral_for_zero_past_direction():
    scores, diagnostics = lagged_descent_alignment_scores(
        torch.eye(3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        neutral_score=0.4,
    )
    assert torch.equal(scores, torch.full((3,), 0.4, dtype=torch.float64))
    assert diagnostics["noise_score_lagged_descent_direction_available"] is False


def test_robust_novelty_score_is_bounded_and_respects_trust_floor():
    novelty = torch.tensor([1.0, 0.5, 0.0])
    trust = torch.tensor([0.0, 0.5, 1.0])
    combined = robust_novelty_scores(novelty, trust, trust_floor=0.2)
    assert torch.allclose(combined, torch.tensor([0.2, 0.3, 0.0]))


def test_pairwise_independence_trust_flags_exact_colluding_copies():
    vectors = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    variances = torch.full((4,), 0.1, dtype=torch.float64)
    trust, diagnostics = pairwise_independence_trust_scores(
        vectors,
        variances,
        low_separation=0.1,
        full_separation=0.5,
    )
    assert trust[0] == pytest.approx(0.0)
    assert trust[1] == pytest.approx(0.0)
    assert trust[2] > 0.0
    assert trust[3] > 0.0
    assert diagnostics["noise_score_nearest_standardized_separation_min"] == 0.0


def test_public_scale_orthogonalization_removes_linear_correlation():
    scales = torch.tensor([1.0, 1.0, 1.5, 1.5, 2.0, 2.0], dtype=torch.float64)
    scores = torch.tensor([0.1, 0.2, 0.4, 0.5, 0.8, 0.9], dtype=torch.float64)
    adjusted, diagnostics = orthogonalize_scores_against_public_scale(scores, scales)
    centered_adjusted = adjusted - adjusted.mean()
    centered_scales = scales - scales.mean()
    assert abs(float(centered_adjusted @ centered_scales)) <= 1e-12
    assert bool((adjusted >= 0).all()) and bool((adjusted <= 1).all())
    assert diagnostics["noise_score_scale_orthogonalization_applied"] is True


def test_public_scale_orthogonalization_is_neutral_for_homogeneous_noise():
    scores = torch.tensor([0.1, 0.4, 0.8], dtype=torch.float64)
    adjusted, diagnostics = orthogonalize_scores_against_public_scale(
        scores, torch.ones_like(scores)
    )
    assert torch.equal(adjusted, scores)
    assert diagnostics["noise_score_scale_orthogonalization_applied"] is False


def test_separate_novelty_and_trust_share_one_bounded_logit_budget():
    novelty = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    trust = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)
    combined, diagnostics = separate_novelty_trust_scores(
        novelty,
        trust,
        trust_fraction=0.25,
    )
    assert torch.allclose(combined, 0.75 * novelty + 0.25 * trust)
    assert bool((combined >= 0).all()) and bool((combined <= 1).all())
    assert diagnostics["noise_score_novelty_logit_fraction"] == pytest.approx(0.75)
    assert diagnostics["noise_score_trust_logit_fraction"] == pytest.approx(0.25)


def test_tierwise_midranks_have_exact_half_mean_and_zero_scale_covariance():
    scores = torch.tensor(
        [0.1, 0.5, 0.9, 0.2, 0.2, 0.8, 0.0, 0.4, 1.0],
        dtype=torch.float64,
    )
    scales = torch.tensor(
        [1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0],
        dtype=torch.float64,
    )
    ranks, diagnostics = tierwise_midrank_scores(scores, scales)
    for scale in torch.unique(scales):
        assert ranks[scales == scale].mean() == pytest.approx(0.5)
    covariance = ((ranks - ranks.mean()) * (scales - scales.mean())).mean()
    assert covariance == pytest.approx(0.0, abs=1e-15)
    assert diagnostics["noise_score_tier_mean_max_error"] == pytest.approx(0.0)
    assert diagnostics["noise_score_tier_midrank_scale_covariance"] == pytest.approx(
        0.0, abs=1e-15
    )


def test_tierwise_midranks_preserve_within_tier_order_and_ties():
    scores = torch.tensor([0.8, 0.1, 0.4, 0.4, 0.9], dtype=torch.float64)
    scales = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0], dtype=torch.float64)
    ranks, _ = tierwise_midrank_scores(scores, scales)
    assert ranks[1] < ranks[2] < ranks[0]
    assert ranks[3] < ranks[4]


def test_theil_sen_projection_recovers_slope_with_one_contaminated_score():
    scales = torch.tensor(
        [1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0],
        dtype=torch.float64,
    )
    scores = 0.1 + 0.2 * (scales - 1.0)
    scores[0] = 1.0
    adjusted, diagnostics = theil_sen_orthogonalize_scores_against_public_scale(
        scores,
        scales,
    )
    assert diagnostics["noise_score_robust_projection_slope"] == pytest.approx(0.2)
    assert bool((adjusted >= 0).all()) and bool((adjusted <= 1).all())


@pytest.mark.parametrize(
    "mode",
    [
        NULL_MC_THEIL_SEN_SEPARATE_TRUST_SCORE_MODE,
        NULL_MC_TIER_ROBUST_SEPARATE_TRUST_SCORE_MODE,
    ],
)
def test_generation5_modes_are_registered_as_robust_direct_scores(mode):
    assert mode in ROBUST_DIRECT_SCORE_MODES


def test_centered_clipping_leave_one_out_matches_explicit_recomputation():
    vectors = torch.tensor([[-2.0, 0.0], [0.5, 1.0], [2.0, -0.5]], dtype=torch.float64)
    anchor = torch.tensor([0.2, -0.1], dtype=torch.float64)
    observed = centered_clipping_leave_one_out(vectors, anchor=anchor, tau=0.8)
    expected = []
    for index in range(vectors.shape[0]):
        keep = torch.arange(vectors.shape[0]) != index
        expected.append(centered_clipping(vectors[keep], anchor=anchor, tau=0.8))
    assert torch.allclose(observed, torch.stack(expected), atol=1e-12)


def test_null_mc_moment_score_is_deterministic_and_bounded():
    distances = torch.tensor([0.1, 0.2, 0.4], dtype=torch.float64)
    variances = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)

    def mean_reference(vectors):
        return vectors.mean(dim=0)

    kwargs = {
        "score_dimension": 4,
        "reference_builder": mean_reference,
        "null_center": torch.zeros(4, dtype=torch.float64),
        "calibration_draws": 30,
        "calibration_seed": 71,
        "z_clip": 4.0,
    }
    first, diagnostics = null_mc_moment_scores(distances, variances, **kwargs)
    second, _ = null_mc_moment_scores(distances, variances, **kwargs)
    assert torch.equal(first, second)
    assert bool((first >= 0).all()) and bool((first <= 1).all())
    assert diagnostics["noise_score_null_mc_draws"] == 30


def test_calibrated_independence_trust_downweights_copied_uploads():
    vectors = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.5], [0.2, 1.4]],
        dtype=torch.float64,
    )
    variances = torch.full((4,), 0.1, dtype=torch.float64)
    trust, diagnostics = calibrated_independence_trust_scores(
        vectors,
        variances,
        calibration_draws=40,
        calibration_seed=19,
    )
    assert trust[0] == pytest.approx(0.05)
    assert trust[1] == pytest.approx(0.05)
    assert diagnostics["noise_score_independence_is_null_calibrated"] is True
