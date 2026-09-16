from __future__ import annotations

import math

import pytest
import torch

from algorithms.gaussian_aware_reference_k6_tp_eiv import (
    eiv_corrected_moments,
    pooled_accepted_direction,
    post_chain_delta_covariance,
    projected_eiv_coefficient,
    projected_eiv_predictor,
    radial_confidence_predictor,
    seed_balanced_median_of_means,
    strictly_past_pooled_accepted_direction_pair,
    strictly_past_rolling_accepted_mean_pair,
    temporal_eiv_confidence,
    temporal_eiv_corrected_moments,
)


def _scalar_history(length: int) -> tuple[torch.Tensor, torch.Tensor]:
    history = torch.arange(1, length + 1, dtype=torch.float32).reshape(length, 1, 1)
    history = history / 10.0
    gates = torch.ones(length, 1, dtype=torch.float32)
    return history, gates


def test_strictly_past_one_step_windows_overlap_as_declared() -> None:
    history, gates = _scalar_history(5)
    older, newer, diagnostics = strictly_past_rolling_accepted_mean_pair(
        history,
        gates,
        window_length=4,
        window_shift=1,
        minimum_accepted_mass=1.0,
        influence_cap=1.0,
        return_diagnostics=True,
    )
    assert torch.allclose(older, torch.tensor([0.25]))
    assert torch.allclose(newer, torch.tensor([0.35]))
    assert diagnostics["older_window_indices"] == [0, 1, 2, 3]
    assert diagnostics["newer_window_indices"] == [1, 2, 3, 4]
    assert diagnostics["overlap_round_count"] == 3
    assert diagnostics["newer_window_most_recent_source_lag"] == 1
    assert diagnostics["older_window_most_recent_source_lag"] == 2
    assert diagnostics["uses_current_round_input"] is False


def test_strictly_past_windows_can_be_non_overlapping() -> None:
    history, gates = _scalar_history(6)
    older, newer, diagnostics = strictly_past_rolling_accepted_mean_pair(
        history,
        gates,
        window_length=3,
        window_shift=3,
        minimum_accepted_mass=1.0,
        influence_cap=1.0,
        return_diagnostics=True,
    )
    assert torch.allclose(older, torch.tensor([0.2]))
    assert torch.allclose(newer, torch.tensor([0.5]))
    assert diagnostics["overlap_round_count"] == 0
    assert diagnostics["windows_overlap"] is False
    assert diagnostics["oldest_source_lag"] == 6


def test_accepted_mass_floor_and_public_ball_are_enforced() -> None:
    history = torch.tensor(
        [[[0.4], [0.2]], [[0.6], [0.4]], [[0.8], [0.6]]],
        dtype=torch.float32,
    )
    gates = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    older, newer, diagnostics = strictly_past_rolling_accepted_mean_pair(
        history,
        gates,
        window_length=2,
        window_shift=1,
        minimum_accepted_mass=2.0,
        influence_cap=1.0,
        return_diagnostics=True,
    )
    assert torch.allclose(older, torch.tensor([0.25]))
    assert torch.allclose(newer, torch.tensor([0.35]))
    assert diagnostics["denominator_chronological"] == [2.0, 2.0, 2.0]


def test_pooled_direction_applies_one_floor_after_pooling() -> None:
    history = torch.tensor([[[0.2, 0.0], [0.0, 0.2]], [[0.4, 0.0], [0.0, 0.4]]])
    gates = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    direction, diagnostics = pooled_accepted_direction(
        history,
        gates,
        minimum_pooled_accepted_mass=2.0,
        influence_cap=1.0,
        return_diagnostics=True,
    )
    assert torch.allclose(direction, torch.tensor([0.2, 0.4 / 3.0]))
    assert diagnostics["total_accepted_mass"] == 3.0
    assert diagnostics["deployed_denominator"] == 3.0
    assert diagnostics["denominator_floor_active"] is False
    assert diagnostics["subconvex_weight_sum"] == 1.0


def test_pooled_direction_handles_zero_and_low_mass_without_instability() -> None:
    history = torch.tensor([[[0.6, 0.8], [0.0, 1.0]]])
    zero, zero_diagnostics = pooled_accepted_direction(
        history,
        torch.zeros(1, 2),
        minimum_pooled_accepted_mass=1.5,
        influence_cap=1.0,
        return_diagnostics=True,
    )
    low, low_diagnostics = pooled_accepted_direction(
        history,
        torch.tensor([[0.5, 0.0]]),
        minimum_pooled_accepted_mass=1.5,
        influence_cap=1.0,
        return_diagnostics=True,
    )
    assert torch.equal(zero, torch.zeros(2))
    assert zero_diagnostics["denominator_floor_active"] is True
    assert zero_diagnostics["subconvex_weight_sum"] == 0.0
    assert torch.allclose(low, torch.tensor([0.2, 0.8 / 3.0]))
    assert low_diagnostics["denominator_floor_active"] is True
    assert math.isclose(low_diagnostics["subconvex_weight_sum"], 1.0 / 3.0)


def test_pooled_direction_respects_cap_and_pair_uses_consecutive_windows() -> None:
    history = torch.zeros(3, 2, 2)
    history[:, :, 0] = 1.0
    gates = torch.ones(3, 2)
    direction, diagnostics = pooled_accepted_direction(
        history,
        gates,
        minimum_pooled_accepted_mass=1.0,
        influence_cap=1.0,
        return_diagnostics=True,
    )
    assert torch.allclose(direction, torch.tensor([1.0, 0.0]))
    assert math.isclose(diagnostics["direction_norm"], 1.0)
    assert diagnostics["projection_active"] is False

    older, newer, pair_diagnostics = strictly_past_pooled_accepted_direction_pair(
        history,
        gates,
        window_length=2,
        window_shift=1,
        minimum_pooled_accepted_mass=1.0,
        influence_cap=1.0,
        return_diagnostics=True,
    )
    assert torch.equal(older, newer)
    assert pair_diagnostics["older_window_indices"] == [0, 1]
    assert pair_diagnostics["newer_window_indices"] == [1, 2]
    assert pair_diagnostics["overlap_round_count"] == 1


def test_eiv_moments_apply_exact_trace_corrections() -> None:
    older = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    newer = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    covariance = torch.tensor([[0.5, 0.0], [0.0, 0.25]])
    cross = torch.tensor([[0.2, 0.1], [0.0, 0.3]])
    z_moment, w_moment = eiv_corrected_moments(
        older,
        newer,
        covariance_older=covariance,
        cross_covariance_older_newer=cross,
    )
    assert torch.allclose(z_moment, torch.tensor([3.5, 10.5]))
    assert torch.allclose(w_moment, torch.tensor([4.25, 24.25]))


def test_temporal_eiv_moments_correct_shared_covariance_and_both_energies() -> None:
    older = torch.tensor([2.0, 1.0], dtype=torch.float64)
    newer = torch.tensor([1.0, 3.0], dtype=torch.float64)
    covariance_x = torch.diag(torch.tensor([0.4, 0.1], dtype=torch.float64))
    covariance_y = torch.diag(torch.tensor([0.2, 0.3], dtype=torch.float64))
    shared = torch.diag(torch.tensor([0.15, 0.05], dtype=torch.float64))
    alignment, energy_x, energy_y = temporal_eiv_corrected_moments(
        older,
        newer,
        covariance_older=covariance_x,
        covariance_newer=covariance_y,
        cross_covariance_older_newer=shared,
    )
    assert math.isclose(float(alignment), 4.8, abs_tol=1.0e-12)
    assert math.isclose(float(energy_x), 4.5, abs_tol=1.0e-12)
    assert math.isclose(float(energy_y), 9.5, abs_tol=1.0e-12)


def test_temporal_confidence_is_one_for_persistent_signal_without_noise() -> None:
    signal = torch.tensor([0.3, -0.4], dtype=torch.float64)
    zeros = torch.zeros(2, 2, dtype=torch.float64)
    alignment, energy_x, energy_y = temporal_eiv_corrected_moments(
        signal,
        signal,
        covariance_older=zeros,
        covariance_newer=zeros,
        cross_covariance_older_newer=zeros,
    )
    confidence = temporal_eiv_confidence(
        alignment, energy_x, energy_y, b_min=1.0e-6, tau_a=0.0
    )
    assert math.isclose(float(confidence), 1.0, abs_tol=1.0e-12)


def test_temporal_confidence_is_zero_for_zero_corrected_cross_moment() -> None:
    confidence, diagnostics = temporal_eiv_confidence(
        0.0,
        -0.1,
        0.0,
        b_min=0.25,
        tau_a=0.0,
        return_diagnostics=True,
    )
    assert float(confidence) == 0.0
    assert diagnostics["energy_x_floor_active"] is True
    assert diagnostics["energy_y_floor_active"] is True


def test_pure_shared_noise_is_removed_before_confidence() -> None:
    observed_noise = torch.tensor([1.0, 0.0], dtype=torch.float64)
    covariance = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))
    alignment, energy_x, energy_y = temporal_eiv_corrected_moments(
        observed_noise,
        observed_noise,
        covariance_older=covariance,
        covariance_newer=covariance,
        cross_covariance_older_newer=covariance,
    )
    assert float(alignment) == 0.0
    assert float(energy_x) == 0.0
    assert float(energy_y) == 0.0
    confidence = temporal_eiv_confidence(
        alignment, energy_x, energy_y, b_min=0.1, tau_a=0.0
    )
    assert float(confidence) == 0.0


def test_temporal_confidence_reports_threshold_and_upper_projection() -> None:
    lower, lower_diagnostics = temporal_eiv_confidence(
        -1.0, 1.0, 1.0, b_min=0.1, tau_a=0.0, return_diagnostics=True
    )
    upper, upper_diagnostics = temporal_eiv_confidence(
        2.0, 1.0, 1.0, b_min=0.1, tau_a=0.0, return_diagnostics=True
    )
    assert float(lower) == 0.0
    assert lower_diagnostics["alignment_threshold_active"] is True
    assert lower_diagnostics["lower_projection_active"] is False
    assert float(upper) == 1.0
    assert upper_diagnostics["upper_projection_active"] is True


@pytest.mark.parametrize("alignment", [-1.0, 0.0, 0.19, 0.2])
def test_temporal_confidence_does_not_activate_below_public_alignment_threshold(
    alignment: float,
) -> None:
    confidence, diagnostics = temporal_eiv_confidence(
        alignment,
        1.0,
        1.0,
        b_min=0.1,
        tau_a=0.2,
        return_diagnostics=True,
    )
    assert float(confidence) == 0.0
    assert diagnostics["thresholded_alignment"] == 0.0
    assert diagnostics["alignment_threshold_active"] is True


def test_radial_confidence_predictor_is_bounded_by_g() -> None:
    predictor, diagnostics = radial_confidence_predictor(
        torch.tensor([3.0, 4.0]),
        0.6,
        influence_cap=0.5,
        minimum_direction_norm=0.1,
        return_diagnostics=True,
    )
    assert torch.allclose(predictor, torch.tensor([0.18, 0.24]), atol=1.0e-7)
    assert math.isclose(float(torch.linalg.vector_norm(predictor)), 0.3, abs_tol=1.0e-7)
    assert diagnostics["norm_floor_active"] is False
    assert diagnostics["predictor_norm"] <= 0.5


def test_radial_confidence_predictor_shrinks_continuously_below_r_min() -> None:
    predictor, diagnostics = radial_confidence_predictor(
        torch.tensor([3.0e-4, 4.0e-4]),
        1.0,
        influence_cap=0.5,
        minimum_direction_norm=1.0e-3,
        return_diagnostics=True,
    )
    assert torch.allclose(predictor, torch.tensor([0.15, 0.2]), atol=1.0e-7)
    assert diagnostics["norm_floor_active"] is True
    assert diagnostics["regularized_direction_lipschitz_bound"] == 1000.0


def test_radial_confidence_predictor_is_continuous_at_r_min() -> None:
    threshold = 1.0
    epsilon = 1.0e-6
    below = radial_confidence_predictor(
        torch.tensor([threshold - epsilon], dtype=torch.float64),
        1.0,
        influence_cap=1.0,
        minimum_direction_norm=threshold,
    )
    at_boundary = radial_confidence_predictor(
        torch.tensor([threshold], dtype=torch.float64),
        1.0,
        influence_cap=1.0,
        minimum_direction_norm=threshold,
    )
    above = radial_confidence_predictor(
        torch.tensor([threshold + epsilon], dtype=torch.float64),
        1.0,
        influence_cap=1.0,
        minimum_direction_norm=threshold,
    )
    assert torch.linalg.vector_norm(below - at_boundary) <= epsilon + 1.0e-12
    assert torch.linalg.vector_norm(above - at_boundary) <= epsilon + 1.0e-12


def test_regularized_radial_map_obeys_one_over_r_min_lipschitz_bound() -> None:
    threshold = 0.5
    first = torch.tensor([0.49, 0.01], dtype=torch.float64)
    second = torch.tensor([0.51, -0.01], dtype=torch.float64)
    mapped_first = radial_confidence_predictor(
        first,
        1.0,
        influence_cap=1.0,
        minimum_direction_norm=threshold,
    )
    mapped_second = radial_confidence_predictor(
        second,
        1.0,
        influence_cap=1.0,
        minimum_direction_norm=threshold,
    )
    output_distance = torch.linalg.vector_norm(mapped_first - mapped_second)
    input_distance = torch.linalg.vector_norm(first - second)
    assert output_distance <= input_distance / threshold + 1.0e-12


def test_seed_balanced_mom_equalizes_unequal_history_counts() -> None:
    values = torch.tensor([0.0] * 20 + [3.0, 6.0, 9.0, 12.0])
    seed_ids = torch.tensor([10] * 20 + [20, 30, 40, 50])
    estimate, diagnostics = seed_balanced_median_of_means(
        values, seed_ids, num_blocks=5, return_diagnostics=True
    )
    assert float(estimate) == 6.0
    assert diagnostics["num_observations"] == 24
    assert diagnostics["num_seeds"] == 5
    assert diagnostics["block_seed_ids"] == [[10], [20], [30], [40], [50]]


def test_seed_balanced_mom_is_robust_to_one_outlying_block() -> None:
    values = torch.tensor([1.0, 1.2, 0.8, 1.1, 1000.0, 0.9, 1.0, 1.05, 0.95])
    seeds = torch.arange(9)
    first = seed_balanced_median_of_means(values, seeds, num_blocks=3)
    second = seed_balanced_median_of_means(values, seeds, num_blocks=3)
    assert torch.equal(first, second)
    assert math.isclose(float(first), 1.0, abs_tol=0.1)


@pytest.mark.parametrize(
    ("z_value", "w_value", "expected", "flag"),
    [
        (-1.0, 2.0, 0.0, "lower_projection_active"),
        (3.0, 1.0, 1.0, "upper_projection_active"),
        (0.25, -2.0, 0.5, "denominator_floor_active"),
    ],
)
def test_projected_eiv_coefficient_reports_floor_and_caps(
    z_value: float, w_value: float, expected: float, flag: str
) -> None:
    beta, diagnostics = projected_eiv_coefficient(
        z_value,
        w_value,
        b_min=0.25,
        lambda0=0.25,
        return_diagnostics=True,
    )
    assert math.isclose(float(beta), expected, abs_tol=1.0e-7)
    assert diagnostics[flag] is True


def test_eiv_denominator_floors_w_before_adding_lambda0() -> None:
    beta, diagnostics = projected_eiv_coefficient(
        0.5,
        0.1,
        b_min=1.0,
        lambda0=0.5,
        beta_max=4.0,
        return_diagnostics=True,
    )
    # Locked provisional formula: 0.5 / (max(0.1, 1.0) + 0.5) = 1/3.
    # The alternative max(w + lambda0, b_min) would incorrectly yield 1/2.
    assert math.isclose(float(beta), 1.0 / 3.0, rel_tol=1.0e-6)
    assert math.isclose(diagnostics["deployed_denominator"], 1.5, abs_tol=1.0e-7)
    assert diagnostics["denominator_floor_active"] is True


def test_public_beta_max_can_exceed_one_without_fixing_scientific_value() -> None:
    beta, diagnostics = projected_eiv_coefficient(
        10.0,
        2.0,
        b_min=0.1,
        lambda0=0.0,
        beta_max=4.5,
        return_diagnostics=True,
    )
    assert math.isclose(float(beta), 4.5, abs_tol=1.0e-7)
    assert diagnostics["upper_projection_active"] is True
    predictor = projected_eiv_predictor(
        torch.tensor([0.1, 0.0]),
        beta,
        beta_max=4.5,
        influence_cap=1.0,
    )
    assert torch.allclose(predictor, torch.tensor([0.45, 0.0]))


def test_predictor_is_projected_on_public_ball() -> None:
    predictor, diagnostics = projected_eiv_predictor(
        torch.tensor([3.0, 4.0]),
        0.5,
        influence_cap=1.0,
        return_diagnostics=True,
    )
    assert torch.allclose(predictor, torch.tensor([0.6, 0.8]), atol=1.0e-6)
    assert diagnostics["projection_active"] is True
    assert math.isclose(diagnostics["predictor_norm"], 1.0, abs_tol=1.0e-6)


def test_post_chain_covariance_matches_linear_analytic_formula() -> None:
    matrix_x = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float64)
    matrix_y = torch.tensor([[0.5, -0.25], [2.0, 1.0]], dtype=torch.float64)
    sigma_x = torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=torch.float64)
    sigma_y = torch.tensor([[1.5, -0.2], [-0.2, 0.8]], dtype=torch.float64)
    sigma_shared = torch.tensor([[0.4, 0.1], [-0.2, 0.3]], dtype=torch.float64)

    covariance_x, covariance_y, cross, diagnostics = post_chain_delta_covariance(
        lambda point: matrix_x @ point,
        lambda point: matrix_y @ point,
        point_x=torch.tensor([0.2, -0.1], dtype=torch.float64),
        point_y=torch.tensor([-0.3, 0.4], dtype=torch.float64),
        covariance_x=sigma_x,
        covariance_y=sigma_y,
        shared_covariance_xy=sigma_shared,
        return_diagnostics=True,
    )
    assert torch.allclose(covariance_x, matrix_x @ sigma_x @ matrix_x.T)
    assert torch.allclose(covariance_y, matrix_y @ sigma_y @ matrix_y.T)
    assert torch.allclose(cross, matrix_x @ sigma_shared @ matrix_y.T)
    assert diagnostics["used_preclipping_covariance_substitute"] is False
    assert diagnostics["differentiability_audit_x"]["passed"] is True
    assert torch.linalg.eigvalsh(covariance_x).min() >= -1.0e-12
    assert torch.isfinite(cross).all()


@pytest.mark.parametrize(
    ("older_row", "newer_row", "expected_cross"),
    [
        ([0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0], 0.25),
        ([0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5], 0.0),
    ],
)
def test_post_chain_cross_covariance_tracks_window_overlap(
    older_row: list[float], newer_row: list[float], expected_cross: float
) -> None:
    older_matrix = torch.tensor([older_row], dtype=torch.float64)
    newer_matrix = torch.tensor([newer_row], dtype=torch.float64)
    identity = torch.eye(4, dtype=torch.float64)
    _, _, cross = post_chain_delta_covariance(
        lambda point: older_matrix @ point,
        lambda point: newer_matrix @ point,
        point_x=torch.zeros(4, dtype=torch.float64),
        point_y=torch.zeros(4, dtype=torch.float64),
        covariance_x=identity,
        covariance_y=identity,
        # Both transforms index the same primitive round-noise vector.
        shared_covariance_xy=identity,
    )
    assert math.isclose(float(cross), expected_cross, abs_tol=1.0e-12)


def test_post_chain_covariance_fails_closed_at_clipping_boundary() -> None:
    with pytest.raises(ValueError, match="possible clipping/projection boundary"):
        post_chain_delta_covariance(
            lambda point: torch.clamp(point, min=-1.0, max=1.0),
            lambda point: point,
            point_x=torch.tensor([1.0], dtype=torch.float64),
            point_y=torch.tensor([0.0], dtype=torch.float64),
            covariance_x=torch.eye(1, dtype=torch.float64),
            covariance_y=torch.eye(1, dtype=torch.float64),
            shared_covariance_xy=torch.zeros(1, 1, dtype=torch.float64),
        )


def test_post_chain_covariance_rejects_indefinite_input_covariance() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        post_chain_delta_covariance(
            lambda point: point,
            lambda point: point,
            point_x=torch.zeros(2, dtype=torch.float64),
            point_y=torch.zeros(2, dtype=torch.float64),
            covariance_x=torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64),
            covariance_y=torch.eye(2, dtype=torch.float64),
            shared_covariance_xy=torch.zeros(2, 2, dtype=torch.float64),
        )


def test_post_chain_covariance_rejects_inconsistent_shared_covariance() -> None:
    with pytest.raises(ValueError, match="joint_input_covariance"):
        post_chain_delta_covariance(
            lambda point: point,
            lambda point: point,
            point_x=torch.zeros(2, dtype=torch.float64),
            point_y=torch.zeros(2, dtype=torch.float64),
            covariance_x=torch.eye(2, dtype=torch.float64),
            covariance_y=torch.eye(2, dtype=torch.float64),
            shared_covariance_xy=2.0 * torch.eye(2, dtype=torch.float64),
        )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable on this host"
)
def test_post_chain_covariance_stays_on_mps() -> None:
    device = torch.device("mps")
    matrix_x = torch.tensor([[1.0, 0.5], [-0.25, 2.0]], device=device)
    matrix_y = torch.tensor([[0.75, -0.5], [1.0, 0.25]], device=device)
    sigma = torch.eye(2, device=device)
    shared = 0.2 * torch.eye(2, device=device)
    covariance_x, covariance_y, cross, diagnostics = post_chain_delta_covariance(
        lambda point: matrix_x @ point,
        lambda point: matrix_y @ point,
        point_x=torch.tensor([0.1, 0.2], device=device),
        point_y=torch.tensor([-0.2, 0.3], device=device),
        covariance_x=sigma,
        covariance_y=sigma,
        shared_covariance_xy=shared,
        return_diagnostics=True,
    )
    assert covariance_x.device.type == "mps"
    assert covariance_y.device.type == "mps"
    assert cross.device.type == "mps"
    assert diagnostics["compute_device"] == "mps:0"
    assert torch.isfinite(covariance_x).all()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable on this host"
)
def test_temporal_confidence_and_radial_predictor_stay_on_mps() -> None:
    device = torch.device("mps")
    confidence = temporal_eiv_confidence(
        torch.tensor(1.0, device=device),
        torch.tensor(1.0, device=device),
        torch.tensor(1.0, device=device),
        b_min=0.1,
        tau_a=0.0,
    )
    predictor = radial_confidence_predictor(
        torch.tensor([3.0, 4.0], device=device),
        confidence,
        influence_cap=0.5,
        minimum_direction_norm=0.1,
    )
    assert confidence.device.type == "mps"
    assert predictor.device.type == "mps"
    assert torch.isfinite(predictor).all()
