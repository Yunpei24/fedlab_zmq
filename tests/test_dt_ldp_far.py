"""Causal and configuration tests for delayed-tilting local-DP FAR."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from algorithms.base import ClientState, get_algorithm
from algorithms.dt_ldp_far import (
    delayed_filtered_weights_from_scores,
    delayed_weights_from_scores,
    robust_anchor_perturbation,
    tilt_tau_max,
)
from algorithms.noise_aware_scores import (
    EFFECTIVE_NULL_MOMENT_SCORE_MODE,
    LOO_EXCESS_ROBUST_SCORE_MODE,
    NULL_MC_ROBUST_SCORE_MODE,
    NULL_MC_THEIL_SEN_SEPARATE_TRUST_SCORE_MODE,
    NULL_MC_TIER_ROBUST_SEPARATE_TRUST_SCORE_MODE,
    TEMPORAL_EMA_PROJECTION_SCORE_MODE,
    TEMPORAL_PREVIOUS_PROJECTION_SCORE_MODE,
    effective_null_fcc_loo_scores,
)
from robustness.tensor_ops import score_subspace, stack_updates


def _model():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    return model


def _tuples(values, ids=None, *, oracle_noise=None):
    if ids is None:
        ids = list(range(len(values)))
    result = []
    for index, (client_id, value) in enumerate(zip(ids, values)):
        metadata = {
            "client_id": int(client_id),
            "dataset_size": 4,
            "local_loss": 0.0,
            "bytes_sent": 4,
            "energy_j_consumed": 0.0,
            "privacy_epsilon": 2.0,
            "privacy_noise_multiplier": 1.2,
            "model_steps": 1,
        }
        if oracle_noise is not None:
            metadata["dtldp_realised_noise_norm_oracle"] = float(oracle_noise[index])
        result.append(
            (
                {"weight": torch.tensor([[float(value)]])},
                metadata,
                ClientState(client_id=int(client_id), battery_j=10.0),
            )
        )
    return result


def _config(**overrides):
    base = {
        "enable_dp": True,
        "server_clip_norm": 1.0,
        "distance_clip": 2.0,
        "tilt_tau": 0.2,
        "kappa_w": 2.0,
        "tilt_bound_policy": "clip",
        "dt_reference": "centered_clipping",
        "reference_clip_radius": 1.0,
        "anchor_update_rate": 0.1,
        "anchor_clip_norm": 1.0,
        "require_full_participation": True,
        "expected_num_clients": 3,
    }
    return {**base, **overrides}


def test_robust_anchor_perturbation_returns_anchor_for_uniform_weights():
    vectors = torch.tensor([[-1.0, 0.0], [0.2, 0.1], [2.0, -0.5]], dtype=torch.float64)
    anchor = torch.tensor([0.1, -0.1], dtype=torch.float64)
    weights = torch.full((3,), 1.0 / 3.0, dtype=torch.float64)

    output, metrics = robust_anchor_perturbation(
        vectors,
        weights,
        anchor,
        residual_radius=0.5,
        gain=0.25,
    )

    assert torch.allclose(output, anchor)
    assert metrics["dtldp_anchor_correction_norm"] == pytest.approx(0.0)
    assert metrics["dtldp_anchor_correction_certificate_respected"] is True


def test_robust_anchor_perturbation_respects_deterministic_certificate():
    vectors = torch.tensor([[-3.0, 0.0], [0.0, 2.0], [4.0, -1.0]], dtype=torch.float64)
    anchor = torch.zeros(2, dtype=torch.float64)
    weights = torch.tensor([0.05, 0.15, 0.80], dtype=torch.float64)

    output, metrics = robust_anchor_perturbation(
        vectors,
        weights,
        anchor,
        residual_radius=0.4,
        gain=0.25,
    )

    observed = float(torch.linalg.vector_norm(output - anchor).item())
    assert observed <= metrics["dtldp_anchor_correction_data_bound"] + 1e-12
    assert (
        metrics["dtldp_anchor_correction_data_bound"]
        <= metrics["dtldp_anchor_correction_universal_bound"] + 1e-12
    )
    assert metrics["dtldp_anchor_correction_universal_bound"] == pytest.approx(0.2)


def _current_dpfar_tuples(*, with_oracle=False):
    values = ((0.20, -0.10), (0.12, 0.05), (-0.15, 0.08), (0.40, -0.25))
    clean_values = ((0.10, -0.05), (0.08, 0.03), (-0.07, 0.04), (0.20, -0.10))
    result = []
    for client_id, (value, clean), scale in zip(
        range(4), zip(values, clean_values), (1.0, 1.3, 1.6, 2.0)
    ):
        metadata = {
            "client_id": client_id,
            "dataset_size": 20,
            "local_loss": 0.0,
            "bytes_sent": 8,
            "energy_j_consumed": 0.0,
            "privacy_epsilon": 4.0,
            "privacy_noise_multiplier_scale_public": scale,
            "privacy_noise_multiplier": 1.1 * scale,
            "privacy_normalization_denominator": 10.0,
            "model_steps": 2,
            "far_update_mode": "multi_epoch_delta",
        }
        if with_oracle:
            metadata["local_dp_noise_free_update_oracle"] = {
                "weight": torch.tensor([clean], dtype=torch.float32)
            }
        result.append(
            (
                {"weight": torch.tensor([value], dtype=torch.float32)},
                metadata,
                ClientState(client_id=client_id, battery_j=10.0),
            )
        )
    return result


def _current_dpfar_config(score_mode, **overrides):
    return {
        "lr": 0.01,
        "clip_norm": 1.0,
        "momentum": 0.0,
        "far_alpha": 0.2,
        "far_score_mode": "bounded_normalized",
        "far_distance_clip": 1.0,
        "far_server_clip_norm": 1.0,
        "kappa_w": 2.0,
        "robust_reference": "centered_clipping",
        "reference_clip_radius": 0.5,
        "anchor_update_rate": 0.1,
        "anchor_clip_norm": 1.0,
        "score_subspace_mode": "full",
        "noise_score_standardization": score_mode,
        "noise_score_null_mc_draws": 20,
        "noise_score_null_mc_seed": 71,
        **overrides,
    }


def test_registry_distinguishes_three_far_privacy_paths():
    assert type(get_algorithm("dpfar")).__name__ == "DPFAR"
    assert type(get_algorithm("scfar_dp")).__name__ == "SensitivityControlledFAR"
    assert type(get_algorithm("dt_ldp_far")).__name__ == "DTLDPFAR"


def test_current_dpfar_null_mc_robust_score_is_bounded_and_deterministic():
    config = _current_dpfar_config(NULL_MC_ROBUST_SCORE_MODE)
    first_model = torch.nn.Linear(2, 1, bias=False)
    second_model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        first_model.weight.zero_()
        second_model.weight.zero_()
    first = get_algorithm("dpfar").server_aggregate(
        first_model,
        _current_dpfar_tuples(),
        0,
        config,
    )
    second = get_algorithm("dpfar").server_aggregate(
        second_model,
        _current_dpfar_tuples(),
        0,
        config,
    )
    assert first.metrics["far_score_mode"] == "bounded_null_mc_robust"
    assert first.metrics["far_weight_cap_respected"] is True
    assert 0.0 <= first.metrics["far_score_min"] <= 1.0
    assert 0.0 <= first.metrics["far_score_max"] <= 1.0
    assert first.metrics["far_score_min"] == second.metrics["far_score_min"]
    assert first.metrics["far_score_max"] == second.metrics["far_score_max"]
    assert torch.equal(first.new_weights["weight"], second.new_weights["weight"])


@pytest.mark.parametrize(
    ("score_mode", "reported_mode"),
    [
        (
            NULL_MC_THEIL_SEN_SEPARATE_TRUST_SCORE_MODE,
            "bounded_null_mc_theil_sen_separate_trust",
        ),
        (
            NULL_MC_TIER_ROBUST_SEPARATE_TRUST_SCORE_MODE,
            "bounded_null_mc_tier_robust_separate_trust",
        ),
    ],
)
def test_current_dpfar_generation5_score_separates_channels_and_keeps_cap(
    score_mode,
    reported_mode,
):
    result = get_algorithm("dpfar").server_aggregate(
        torch.nn.Linear(2, 1, bias=False),
        _current_dpfar_tuples(with_oracle=True),
        0,
        _current_dpfar_config(
            score_mode,
            noise_score_trust_logit_fraction=0.25,
        ),
    )
    metrics = result.metrics
    assert metrics["far_score_mode"] == reported_mode
    assert metrics["far_weight_cap_respected"] is True
    assert metrics["far_noise_score_channels_separated"] is True
    assert metrics["far_noise_score_novelty_logit_fraction"] == pytest.approx(0.75)
    assert metrics["far_noise_score_trust_logit_fraction"] == pytest.approx(0.25)
    assert metrics["far_noise_score_combined_trust_rule"] == "minimum"
    assert 0.0 <= metrics["far_score_min"] <= metrics["far_score_max"] <= 1.0


@pytest.mark.parametrize(
    ("score_mode", "reported_mode"),
    [
        (
            TEMPORAL_PREVIOUS_PROJECTION_SCORE_MODE,
            "bounded_temporal_previous_projection",
        ),
        (
            TEMPORAL_EMA_PROJECTION_SCORE_MODE,
            "bounded_temporal_ema_projection",
        ),
    ],
)
def test_current_dpfar_temporal_score_uses_only_past_private_residuals(
    score_mode,
    reported_mode,
):
    algo = get_algorithm("dpfar")
    config = _current_dpfar_config(score_mode)
    first = algo.server_aggregate(
        torch.nn.Linear(2, 1, bias=False),
        _current_dpfar_tuples(with_oracle=True),
        0,
        config,
    )
    second = algo.server_aggregate(
        torch.nn.Linear(2, 1, bias=False),
        _current_dpfar_tuples(with_oracle=True),
        1,
        config,
    )
    assert first.metrics["far_score_mode"] == reported_mode
    assert first.metrics["far_noise_score_temporal_history_coverage"] == 0.0
    assert first.metrics["max_client_weight"] == pytest.approx(0.25)
    assert second.metrics["far_noise_score_temporal_history_coverage"] == 1.0
    assert second.metrics["far_weight_cap_respected"] is True
    assert (
        0.0 <= second.metrics["far_score_min"] <= second.metrics["far_score_max"] <= 1.0
    )


def test_current_dpfar_loo_robust_score_exposes_noise_free_oracle_metrics():
    result = get_algorithm("dpfar").server_aggregate(
        torch.nn.Linear(2, 1, bias=False),
        _current_dpfar_tuples(with_oracle=True),
        0,
        _current_dpfar_config(LOO_EXCESS_ROBUST_SCORE_MODE),
    )
    metrics = result.metrics
    assert metrics["far_score_mode"] == "bounded_loo_excess_robust"
    assert metrics["far_noise_score_residual_covariance_model"] == (
        "leave_one_out_mean_reference_proxy"
    )
    assert metrics["far_weight_cap_respected"] is True
    assert "far_noisy_clean_score_corr_oracle" in metrics
    assert "far_honest_noisy_clean_score_corr_oracle" in metrics
    assert 0.0 <= metrics["far_honest_clean_top_tail_recall_oracle"] <= 1.0


def test_delayed_algorithm_rejects_generation4_current_round_scores():
    with pytest.raises(ValueError, match="current-round DPFAR"):
        get_algorithm("dt_ldp_far").server_aggregate(
            _model(),
            _tuples([-0.2, 0.1, 0.9]),
            0,
            _config(noise_score_standardization=NULL_MC_ROBUST_SCORE_MODE),
        )


def test_effective_null_fcc_loo_score_is_bounded_and_rng_isolated():
    vectors = torch.tensor(
        [[0.20, -0.10], [0.12, 0.04], [-0.08, 0.05], [0.30, -0.20]],
        dtype=torch.float64,
    )
    variances = torch.tensor([0.01, 0.02, 0.04, 0.02], dtype=torch.float64)
    anchor = torch.zeros(2, dtype=torch.float64)
    first = effective_null_fcc_loo_scores(
        vectors,
        variances,
        anchor=anchor,
        reference_radius=0.2,
        calibration_draws=20,
        calibration_seed=71,
        mode="moment",
    )
    second = effective_null_fcc_loo_scores(
        vectors,
        variances,
        anchor=anchor,
        reference_radius=0.2,
        calibration_draws=20,
        calibration_seed=71,
        mode="moment",
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[2], second[2])
    assert bool(((first[0] >= 0.0) & (first[0] <= 1.0)).all())
    assert first[3]["noise_score_effective_null_training_rng_isolated"] is True


def test_dt_ldp_far_runs_effective_null_score_with_dual_trust():
    tuples = _tuples([-0.2, 0.1, 0.9])
    for item, scale in zip(tuples, (1.0, 1.5, 2.0)):
        item[1].update(
            {
                "privacy_noise_multiplier_scale_public": scale,
                "privacy_normalization_denominator": 4.0,
                "model_steps": 2,
            }
        )
    result = get_algorithm("dt_ldp_far").server_aggregate(
        _model(),
        tuples,
        0,
        _config(
            lr=0.01,
            clip_norm=1.0,
            momentum=0.0,
            noise_score_standardization=EFFECTIVE_NULL_MOMENT_SCORE_MODE,
            noise_score_null_mc_draws=20,
            dt_trust_profile="directional_independence",
        ),
    )
    metrics = result.metrics
    assert metrics["dtldp_noise_score_effective_null_mode"] == "moment"
    assert metrics["dtldp_noise_score_combined_trust_rule"] == "minimum"
    assert metrics["dtldp_weight_cap_respected"] is True
    assert 0.0 <= metrics["dtldp_current_score_min"]
    assert metrics["dtldp_current_score_max"] <= 1.0


def test_dt_ldp_far_runs_stage19_ranked_peer_support_with_public_f_bound():
    tuples = _tuples([-0.2, 0.1, 0.9])
    for item, scale in zip(tuples, (1.0, 1.5, 2.0)):
        item[1].update(
            {
                "privacy_noise_multiplier_scale_public": scale,
                "privacy_normalization_denominator": 4.0,
                "model_steps": 2,
            }
        )
    result = get_algorithm("dt_ldp_far").server_aggregate(
        _model(),
        tuples,
        0,
        _config(
            lr=0.01,
            clip_norm=1.0,
            momentum=0.0,
            noise_score_standardization=EFFECTIVE_NULL_MOMENT_SCORE_MODE,
            noise_score_null_mc_draws=20,
            dt_trust_profile="ranked_directional_peer_support",
            dt_support_assumed_byzantine=1,
            noise_score_trust_logit_fraction=0.5,
        ),
    )
    metrics = result.metrics
    assert metrics["dtldp_noise_score_peer_support_assumed_byzantine"] == 1
    assert metrics["dtldp_noise_score_peer_support_required_rank"] == 2
    assert (
        metrics["dtldp_noise_score_combined_trust_rule"]
        == "minimum_of_ranked_directional_peer_and_independence"
    )
    assert metrics["dtldp_weight_cap_respected"] is True


def test_dt_ldp_far_robust_anchor_mode_reports_and_respects_certificate():
    result = get_algorithm("dt_ldp_far").server_aggregate(
        _model(),
        _tuples([-1.0, 0.1, 0.9]),
        0,
        _config(
            dt_aggregate_mode="robust_anchor_perturbation",
            dt_aggregate_anchor="rfa",
            dt_anchor_residual_radius_fraction=0.5,
            dt_anchor_correction_gain=0.25,
        ),
    )

    metrics = result.metrics
    assert metrics["dtldp_aggregate_mode"] == "robust_anchor_perturbation"
    assert metrics["dtldp_aggregate_anchor"] == "rfa"
    assert metrics["dtldp_anchor_residual_radius"] == pytest.approx(0.5)
    assert metrics["dtldp_anchor_correction_universal_bound"] == pytest.approx(0.25)
    assert metrics["dtldp_anchor_correction_certificate_respected"] is True
    # The cold-start weights are uniform, so the certified FAR correction is zero.
    assert metrics["dtldp_anchor_correction_norm"] == pytest.approx(0.0, abs=1e-8)


def test_dt_ldp_far_lagged_descent_score_has_uniform_cold_start_then_uses_past():
    algo = get_algorithm("dt_ldp_far")
    config = _config(
        lr=0.01,
        clip_norm=1.0,
        momentum=0.0,
        noise_score_standardization=EFFECTIVE_NULL_MOMENT_SCORE_MODE,
        noise_score_null_mc_draws=20,
        dt_trust_profile="lagged_descent_alignment",
        noise_score_trust_logit_fraction=0.75,
        noise_score_lagged_descent_reject_cosine=0.0,
        noise_score_lagged_descent_full_cosine=0.5,
    )

    def prepared(values):
        tuples = _tuples(values)
        for item, scale in zip(tuples, (1.0, 1.5, 2.0)):
            item[1].update(
                {
                    "privacy_noise_multiplier_scale_public": scale,
                    "privacy_normalization_denominator": 4.0,
                    "model_steps": 2,
                }
            )
        return tuples

    cold = algo.server_aggregate(_model(), prepared([0.2, 0.4, 0.6]), 0, config)
    active = algo.server_aggregate(_model(), prepared([0.7, 0.2, -0.4]), 1, config)

    assert cold.metrics["dtldp_current_score_span"] == pytest.approx(0.0)
    assert cold.metrics["dtldp_noise_score_lagged_descent_cold_start_uniform"] is True
    assert (
        active.metrics["dtldp_noise_score_lagged_descent_direction_available"] is True
    )
    assert (
        active.metrics["dtldp_noise_score_combined_trust_rule"]
        == "lagged_robust_direction_alignment"
    )
    assert active.metrics["dtldp_current_score_span"] > 0.0
    assert active.metrics["dtldp_weight_cap_respected"] is True


def test_strict_temporal_reference_uses_prior_before_current_update():
    first = get_algorithm("dt_ldp_far")
    second = get_algorithm("dt_ldp_far")
    config = _config(
        dt_reference_time_mode="lagged_ema",
        anchor_update_rate=0.5,
        temporal_reference_step_radius=1.0,
    )
    common = _tuples([-0.6, 0.2, 0.8])
    first.server_aggregate(_model(), common, 0, config)
    second.server_aggregate(_model(), common, 0, config)
    left = first.server_aggregate(_model(), _tuples([-1.0, -0.5, 0.1]), 1, config)
    right = second.server_aggregate(_model(), _tuples([1.0, 0.8, 0.6]), 1, config)
    assert left.metrics["dtldp_reference_is_strictly_lagged"] is True
    assert left.metrics["dtldp_reference_norm"] == pytest.approx(
        right.metrics["dtldp_reference_norm"]
    )
    assert left.metrics["dtldp_reference_proposal_norm"] != pytest.approx(
        right.metrics["dtldp_reference_proposal_norm"]
    )


def test_temporal_reference_can_freeze_only_during_public_attack_window():
    algo = get_algorithm("dt_ldp_far")
    config = _config(
        dt_reference_time_mode="lagged_ema",
        anchor_update_rate=0.5,
        temporal_reference_step_radius=1.0,
        temporal_reference_freeze_during_attack=True,
        attack={
            "enabled": True,
            "name": "bf",
            "scale": 2.0,
            "num_byzantine": 1,
            "client_ids": [0],
            "active_round_start": 2,
            "active_round_end": 2,
        },
    )
    clean = algo.server_aggregate(_model(), _tuples([-0.6, 0.2, 0.8]), 0, config)
    attack = algo.server_aggregate(_model(), _tuples([-1.0, 0.1, 0.5]), 1, config)
    recovery = algo.server_aggregate(_model(), _tuples([-0.2, 0.4, 0.9]), 2, config)
    assert clean.metrics["dtldp_reference_attack_phase"] == "clean"
    assert clean.metrics["dtldp_reference_was_frozen"] is False
    assert attack.metrics["dtldp_reference_attack_phase"] == "attack"
    assert attack.metrics["dtldp_reference_was_frozen"] is True
    assert attack.metrics["dtldp_reference_state_update_norm"] == pytest.approx(0.0)
    assert recovery.metrics["dtldp_reference_attack_phase"] == "recovery"
    assert recovery.metrics["dtldp_reference_was_frozen"] is False
    assert recovery.metrics["dtldp_reference_state_update_norm"] > 0.0


def test_stage14_output_guard_is_reported_separately_from_delayed_weights():
    result = get_algorithm("dt_ldp_far").server_aggregate(
        _model(),
        _tuples([-1.0, -0.8, 1.0]),
        0,
        _config(
            dt_output_guard_radius_fraction=0.1,
            dt_guard_reference_clip_radius=0.2,
        ),
    )
    assert result.metrics["dtldp_output_guard_enabled"] is True
    assert result.metrics["dtldp_output_guard_radius"] == pytest.approx(0.1)
    assert result.metrics["dtldp_initial_weights_uniform"] is True


def test_public_score_subspaces_are_deterministic_and_keep_full_updates_intact():
    updates = [
        {
            "features.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3) + i,
            "classifier.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2) + i,
            "classifier.bias": torch.arange(2, dtype=torch.float32) + i,
        }
        for i in (0.0, 1.0, 2.0)
    ]
    vectors, layout = stack_updates(updates)
    last, last_meta = score_subspace(vectors, layout, mode="last_layer")
    assert vectors.shape == (3, 12)
    assert last.shape == (3, 6)
    assert last_meta["score_subspace_keys"] == [
        "classifier.weight",
        "classifier.bias",
    ]
    first, _ = score_subspace(
        vectors, layout, mode="public_coordinates", dimension=5, seed=17
    )
    second, _ = score_subspace(
        vectors, layout, mode="public_coordinates", dimension=5, seed=17
    )
    assert torch.equal(first, second)


def test_round_zero_weights_are_uniform():
    result = get_algorithm("dt_ldp_far").server_aggregate(
        _model(), _tuples([-0.8, 0.1, 0.9]), 0, _config()
    )
    assert result.metrics["dtldp_initial_weights_uniform"] is True
    assert math.isclose(result.metrics["max_client_weight"], 1 / 3, rel_tol=1e-7)
    assert math.isclose(result.metrics["min_client_weight"], 1 / 3, rel_tol=1e-7)
    assert result.metrics["dtldp_weights_source_round"] == -1
    assert set(result.metrics["dtldp_delayed_weights_by_client"]) == {"0", "1", "2"}
    assert all(
        math.isclose(value, 1 / 3, rel_tol=1e-7)
        for value in result.metrics["dtldp_delayed_weights_by_client"].values()
    )


def test_analytic_tau_cap_is_respected_without_weight_projection():
    n, kappa = 10, 2.0
    tau = tilt_tau_max(n, kappa)
    weights, _ = delayed_weights_from_scores(list(range(n)), {0: 1.0}, tau=tau)
    assert math.isclose(float(weights.max()), kappa / n, rel_tol=1e-12)


def test_filtered_tau_cap_uses_public_admissible_set_size():
    n, active, kappa = 25, 20, 2.0
    tau = tilt_tau_max(n, kappa, active_clients=active)
    scores = {client_id: 0.0 for client_id in range(n)}
    scores[0] = 1.0
    trust = {client_id: float(n - client_id) / n for client_id in range(n)}
    weights, _, eligible, active_filter = delayed_filtered_weights_from_scores(
        list(range(n)),
        scores,
        trust,
        tau=tau,
        max_excluded=n - active,
    )
    assert active_filter is True
    assert int(eligible.sum()) == active
    assert math.isclose(float(weights.max()), kappa / n, rel_tol=1e-12)
    assert bool((weights[~eligible] == 0.0).all())


def test_filtered_weights_are_cold_start_uniform_and_ties_use_client_id():
    client_ids = [30, 10, 20, 40]
    cold, _, cold_eligible, cold_active = delayed_filtered_weights_from_scores(
        client_ids,
        {},
        {},
        tau=0.2,
        max_excluded=1,
    )
    assert cold_active is False
    assert bool(cold_eligible.all())
    assert torch.allclose(cold, torch.full((4,), 0.25, dtype=torch.float64))

    trust = {client_id: 0.5 for client_id in client_ids}
    weights, _, eligible, active = delayed_filtered_weights_from_scores(
        client_ids,
        {client_id: 0.0 for client_id in client_ids},
        trust,
        tau=0.2,
        max_excluded=1,
    )
    assert active is True
    assert eligible.tolist() == [True, True, True, False]
    assert weights.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3, 0.0])


def test_delayed_admissibility_filter_uses_previous_round_trust_only():
    algo = get_algorithm("dt_ldp_far")
    config = _config(
        expected_num_clients=4,
        lr=0.01,
        clip_norm=1.0,
        momentum=0.0,
        noise_score_standardization=EFFECTIVE_NULL_MOMENT_SCORE_MODE,
        noise_score_null_mc_draws=20,
        dt_trust_profile="ranked_directional_peer_support_filter",
        dt_support_assumed_byzantine=1,
        dt_admissibility_filter_enabled=True,
        dt_admissibility_max_byzantine=1,
    )

    def prepared(values):
        tuples = _tuples(values)
        for item, scale in zip(tuples, (1.0, 1.2, 1.4, 1.6)):
            item[1].update(
                {
                    "privacy_noise_multiplier_scale_public": scale,
                    "privacy_normalization_denominator": 4.0,
                    "model_steps": 2,
                }
            )
        return tuples

    cold = algo.server_aggregate(_model(), prepared([0.2, 0.4, 0.6, 0.8]), 0, config)
    active = algo.server_aggregate(_model(), prepared([-0.7, 0.1, 0.5, 0.9]), 1, config)
    assert cold.metrics["dtldp_admissibility_filter_active"] is False
    assert cold.metrics["dtldp_admissible_client_count"] == 4
    assert active.metrics["dtldp_admissibility_filter_active"] is True
    assert active.metrics["dtldp_admissible_client_count"] == 3
    assert active.metrics["dtldp_weight_cap_respected"] is True
    assert active.metrics["dtldp_noise_score_trust_used_only_for_admissibility"] is True


def test_uncertified_diagnostic_reports_excess_without_claiming_clipping():
    algo = get_algorithm("dt_ldp_far")
    maximum = tilt_tau_max(3, 2.0)
    config = _config(
        tilt_tau=2.0 * maximum,
        tilt_bound_policy="diagnostic_only",
        allow_uncertified_tilt_diagnostic=True,
    )
    result = algo.server_aggregate(_model(), _tuples([-0.8, 0.1, 0.9]), 0, config)
    assert result.metrics["dtldp_tilt_tau_requested_exceeds_cap"] is True
    assert result.metrics["dtldp_tilt_tau_was_clipped"] is False
    assert result.metrics["dtldp_tilt_influence_certificate_claimed"] is False
    assert math.isclose(result.metrics["dtldp_tilt_tau"], 2.0 * maximum)


def test_raw_distance_dt_lane_keeps_untransformed_scores_and_public_2u_cap():
    algo = get_algorithm("dt_ldp_far")
    cfg = _config(
        dt_score_transform="raw_distance",
        noise_score_standardization="none",
        tilt_tau=0.2,
        server_clip_norm=1.0,
        anchor_clip_norm=1.0,
        reference_clip_radius=1.0,
    )
    first = algo.server_aggregate(_model(), _tuples([-0.8, 0.1, 0.9]), 0, cfg)
    second = algo.server_aggregate(_model(), _tuples([-0.6, 0.2, 0.8]), 1, cfg)
    assert first.metrics["dtldp_score_transform"] == "raw_distance"
    assert first.metrics["dtldp_score_is_unit_bounded"] is False
    assert first.metrics["dtldp_tilt_public_score_range"] == pytest.approx(2.0)
    assert first.metrics["dtldp_current_score_saturation_rate"] is None
    assert second.metrics["dtldp_previous_score_coverage"] == 1.0
    assert second.metrics["dtldp_weight_cap_respected"] is True


def test_delayed_weights_allow_raw_scores_only_when_bounds_are_explicit():
    raw_scores = {0: 0.1, 1: 1.4, 2: 1.8}
    with pytest.raises(ValueError, match="outside score_bounds"):
        delayed_weights_from_scores([0, 1, 2], raw_scores, tau=0.1)
    weights, scores = delayed_weights_from_scores(
        [0, 1, 2],
        raw_scores,
        tau=0.1,
        score_bounds=(0.0, 2.0),
    )
    assert torch.equal(scores, torch.tensor([0.1, 1.4, 1.8], dtype=torch.float64))
    assert float(weights.sum()) == pytest.approx(1.0)


def test_configured_two_round_delay_uses_exact_source_round():
    algo = get_algorithm("dt_ldp_far")
    cfg = _config(tilt_delay_rounds=2)
    round_zero = algo.server_aggregate(_model(), _tuples([-0.9, 0.1, 0.8]), 0, cfg)
    round_one = algo.server_aggregate(_model(), _tuples([-0.5, 0.2, 0.6]), 1, cfg)
    round_two = algo.server_aggregate(_model(), _tuples([-0.4, 0.3, 0.5]), 2, cfg)
    assert round_zero.metrics["dtldp_weights_source_round"] == -2
    assert round_one.metrics["dtldp_weights_source_round"] == -1
    assert round_one.metrics["dtldp_initial_weights_uniform"] is True
    assert round_two.metrics["dtldp_weights_source_round"] == 0
    assert round_two.metrics["dtldp_previous_score_coverage"] == 1.0


def test_current_fresh_updates_cannot_change_current_delayed_weights():
    first = get_algorithm("dt_ldp_far")
    second = get_algorithm("dt_ldp_far")
    common_round_zero = _tuples([-0.9, 0.15, 0.7])
    first.server_aggregate(_model(), common_round_zero, 0, _config())
    second.server_aggregate(_model(), common_round_zero, 0, _config())

    a = first.server_aggregate(_model(), _tuples([-1.0, -0.2, 0.4]), 1, _config())
    b = second.server_aggregate(_model(), _tuples([1.0, 0.95, -1.0]), 1, _config())
    # Current references, scores and aggregates differ.  Weight diagnostics
    # are identical because the applied weights came from the shared round 0.
    assert not math.isclose(
        a.metrics["dtldp_current_score_mean"],
        b.metrics["dtldp_current_score_mean"],
        rel_tol=1e-6,
    )
    for key in (
        "max_client_weight",
        "min_client_weight",
        "weight_entropy",
        "dtldp_weight_l2_squared",
        "dtldp_delayed_score_min",
        "dtldp_delayed_score_max",
    ):
        assert math.isclose(a.metrics[key], b.metrics[key], rel_tol=1e-12)


def test_delayed_scores_follow_client_identity_not_message_position():
    algo = get_algorithm("dt_ldp_far")
    algo.server_aggregate(
        _model(), _tuples([-0.9, 0.1, 0.8], ids=[10, 20, 30]), 0, _config()
    )
    stored = dict(algo._dt_previous_scores)
    reordered_ids = [30, 10, 20]
    values = [0.2, 0.2, 0.2]
    expected_weights, _ = delayed_weights_from_scores(
        reordered_ids,
        stored,
        tau=min(0.2, tilt_tau_max(3, 2.0)),
    )
    # Equal current vectors make the released aggregate independent of order,
    # but the internal delayed-score vector still proves ID alignment.
    result = algo.server_aggregate(
        _model(), _tuples(values, ids=reordered_ids), 1, _config()
    )
    assert math.isclose(
        result.metrics["dtldp_delayed_score_min"],
        min(stored.values()),
        rel_tol=1e-12,
    )
    assert math.isclose(
        result.metrics["max_client_weight"],
        float(expected_weights.max()),
        rel_tol=1e-7,
    )


def test_duplicate_client_identity_is_rejected():
    algo = get_algorithm("dt_ldp_far")
    try:
        algo.server_aggregate(
            _model(), _tuples([0.1, 0.2, 0.3], ids=[0, 0, 1]), 0, _config()
        )
    except ValueError as error:
        assert "unique client_id" in str(error)
    else:
        raise AssertionError("duplicate client IDs must be rejected")


def test_dt_ldp_far_uses_only_public_noise_scales_for_score_standardization():
    tuples = _tuples([-0.2, 0.1, 0.9])
    for item, scale in zip(tuples, (1.0, 1.0, 2.0)):
        item[1]["privacy_noise_multiplier_scale_public"] = scale
    result = get_algorithm("dt_ldp_far").server_aggregate(
        _model(),
        tuples,
        0,
        _config(noise_score_standardization="isotropic_dp_covariance_proxy"),
    )
    metrics = result.metrics
    assert metrics["dtldp_noise_score_standardization"] == (
        "isotropic_dp_covariance_proxy"
    )
    assert metrics["dtldp_noise_score_uses_public_parameters_only"] is True
    assert metrics["dtldp_noise_score_covariance_is_proxy"] is True
    assert metrics["dtldp_noise_score_divisor_max"] > 1.0
    assert metrics["dtldp_weight_cap_respected"] is True


def test_noise_aware_dt_ldp_far_rejects_missing_public_noise_scales():
    with torch.no_grad():
        try:
            get_algorithm("dt_ldp_far").server_aggregate(
                _model(),
                _tuples([-0.2, 0.1, 0.9]),
                0,
                _config(noise_score_standardization="isotropic_dp_covariance_proxy"),
            )
        except ValueError as error:
            assert "public per-client DP noise scale" in str(error)
        else:
            raise AssertionError("missing public DP noise scales must be rejected")


def test_dt_ldp_far_excess_energy_score_is_bounded_and_certified():
    tuples = _tuples([-0.2, 0.1, 0.9])
    for item, scale in zip(tuples, (1.0, 1.5, 2.0)):
        item[1].update(
            {
                "privacy_noise_multiplier_scale_public": scale,
                "privacy_noise_multiplier": 1.2 * scale,
                "privacy_normalization_denominator": 10.0,
                "model_steps": 2,
            }
        )
    result = get_algorithm("dt_ldp_far").server_aggregate(
        _model(),
        tuples,
        0,
        _config(
            lr=0.01,
            clip_norm=1.0,
            momentum=0.0,
            noise_score_standardization="isotropic_dp_excess_energy",
            noise_score_z_clip=5.0,
        ),
    )
    metrics = result.metrics
    assert metrics["dtldp_noise_score_standardization"] == (
        "isotropic_dp_excess_energy"
    )
    assert metrics["dtldp_noise_score_noise_floor_subtracted"] is True
    assert 0.0 <= metrics["dtldp_current_score_min"] <= 1.0
    assert 0.0 <= metrics["dtldp_current_score_max"] <= 1.0
    assert metrics["dtldp_weight_cap_respected"] is True


def test_dt_ldp_far_debiased_distance_score_is_bounded_and_certified():
    tuples = _tuples([-0.2, 0.1, 0.9])
    for item, scale in zip(tuples, (1.0, 1.5, 2.0)):
        item[1].update(
            {
                "privacy_noise_multiplier_scale_public": scale,
                "privacy_noise_multiplier": 1.2 * scale,
                "privacy_normalization_denominator": 10.0,
                "model_steps": 2,
            }
        )
    result = get_algorithm("dt_ldp_far").server_aggregate(
        _model(),
        tuples,
        0,
        _config(
            lr=0.01,
            clip_norm=1.0,
            momentum=0.0,
            distance_clip=2.0,
            noise_score_standardization="isotropic_dp_debiased_distance",
        ),
    )
    metrics = result.metrics
    assert metrics["dtldp_noise_score_standardization"] == (
        "isotropic_dp_debiased_distance"
    )
    assert metrics["dtldp_noise_score_noise_floor_subtracted"] is True
    assert metrics["dtldp_noise_score_distance_clip"] == 2.0
    assert 0.0 <= metrics["dtldp_current_score_min"] <= 1.0
    assert 0.0 <= metrics["dtldp_current_score_max"] <= 1.0
    assert metrics["dtldp_weight_cap_respected"] is True


def test_noise_aware_oracle_compares_noisy_and_noise_free_scores():
    tuples = _tuples([-0.2, 0.1, 0.9])
    clean_values = (-0.1, 0.0, 0.3)
    for item, scale, clean in zip(tuples, (1.0, 1.5, 2.0), clean_values):
        item[1]["privacy_noise_multiplier_scale_public"] = scale
        item[1]["dtldp_noise_free_update_oracle"] = {"weight": torch.tensor([[clean]])}
    result = get_algorithm("dt_ldp_far").server_aggregate(
        _model(),
        tuples,
        0,
        _config(
            enable_oracle_diagnostics=True,
            noise_score_standardization="isotropic_dp_covariance_proxy",
        ),
    )
    for key in (
        "dtldp_current_noisy_clean_score_corr_oracle",
        "dtldp_current_noisy_clean_score_mae_oracle",
        "dtldp_current_noisy_clean_score_rmse_oracle",
        "dtldp_current_noisy_clean_weight_l1_oracle",
        "dtldp_current_score_public_noise_scale_corr_oracle",
        "dtldp_clean_score_public_noise_scale_corr_oracle",
        "dtldp_current_noisy_raw_clean_score_corr_oracle",
        "dtldp_current_noisy_raw_clean_score_rmse_oracle",
        "dtldp_current_noisy_raw_clean_weight_l1_oracle",
        "dtldp_raw_clean_score_public_noise_scale_corr_oracle",
    ):
        assert key in result.metrics


def test_client_side_auxiliary_diagnostics_are_suppressed_by_default(monkeypatch):
    algo = get_algorithm("dt_ldp_far")

    def fake_update(model, dataloader, state, config):
        return {"weight": torch.zeros(1, 1)}, {
            "client_id": state.client_id,
            "round_num": 1,
            "local_loss": 1.5,
            "clip_rate": 0.4,
            "dp_noise_norm_mean": 2.0,
            "bytes_sent": 4,
            "bytes_received": 4,
            "energy_j_consumed": 0.0,
            "battery_j_remaining": state.battery_j,
            "compression_ratio": 1.0,
            "beta_actual": 1.0,
        }

    monkeypatch.setattr(algo, "_local_dp_update", fake_update)
    _, metadata = algo.client_update(
        _model(), None, ClientState(client_id=2, battery_j=10.0), {}
    )
    assert metadata["local_loss"] == 0.0
    assert metadata["local_loss_available"] is False
    assert "clip_rate" not in metadata
    assert "dp_noise_norm_mean" not in metadata
    assert "dtldp_local_loss_oracle" not in metadata


def test_reference_is_configurable_for_main_and_ablation_lanes():
    for reference in (
        "centered_clipping",
        "regularized_huber",
        "cm_nnm",
        "trmean_nnm",
        "rfa",
    ):
        algo = get_algorithm("dt_ldp_far")
        result = algo.server_aggregate(
            _model(),
            _tuples([-0.2, 0.0, 0.1]),
            0,
            _config(dt_reference=reference, num_byzantine=0),
        )
        assert result.metrics["dtldp_reference"] == reference


def test_native_far_supports_centered_clipping_with_causal_public_anchor():
    algo = get_algorithm("far")
    config = {
        "robust_reference": "centered_clipping",
        "reference_clip_radius": 0.5,
        "anchor_update_rate": 0.25,
        "anchor_clip_norm": 1.0,
        "far_alpha": 0.1,
        "far_update_mode": "multi_epoch_delta",
    }
    first = algo.server_aggregate(_model(), _tuples([-0.8, 0.1, 0.9]), 0, config)
    second = algo.server_aggregate(_model(), _tuples([-0.6, 0.2, 0.7]), 1, config)
    assert first.metrics["robust_reference"] == "centered_clipping"
    assert first.metrics["far_reference_anchor_norm"] == 0.0
    assert first.metrics["far_reference_clip_radius"] == 0.5
    assert second.metrics["far_reference_anchor_norm"] >= 0.0


def test_current_far_can_use_same_bounded_normalized_score_as_dt_lane():
    algo = get_algorithm("dpfar")
    config = {
        "robust_reference": "centered_clipping",
        "reference_clip_radius": 1.0,
        "anchor_update_rate": 0.0,
        "anchor_clip_norm": 1.0,
        "far_alpha": 0.3,
        "kappa_w": 2.0,
        "far_score_mode": "bounded_normalized",
        "far_distance_clip": 0.5,
        "far_server_clip_norm": 1.0,
        "far_update_mode": "multi_epoch_delta",
    }
    result = algo.server_aggregate(_model(), _tuples([-0.8, 0.1, 0.9]), 0, config)
    assert result.metrics["far_score_mode"] == "bounded_normalized"
    assert 0.0 <= result.metrics["far_score_min"] <= 1.0
    assert 0.0 <= result.metrics["far_score_max"] <= 1.0
    assert result.metrics["far_distance_clip"] == 0.5
    assert result.metrics["far_score_scale"] == 0.5
    assert result.metrics["far_tilt_influence_certificate_claimed"] is True
    assert result.metrics["far_weight_cap_respected"] is True
    assert result.metrics["far_noise_amplification_vs_uniform"] >= 1.0


def test_dt_geometry_diagnostics_detect_an_uninformative_saturated_score():
    result = get_algorithm("dt_ldp_far").server_aggregate(
        _model(),
        _tuples([-0.8, 0.1, 0.9]),
        0,
        _config(distance_clip=1e-6, pilot_minimum_score_span=0.02),
    )
    assert result.metrics["dtldp_current_score_all_saturated"] is True
    assert result.metrics["dtldp_current_score_span"] == 0.0
    assert result.metrics["dtldp_pilot_tilting_informative"] is False


def test_server_clip_utilization_uses_pre_clipping_upload_norms():
    algo = get_algorithm("dt_ldp_far")
    tuples = _tuples([2.0, 0.5, -0.25])
    tuples[0][1]["is_byzantine"] = True
    result = algo.server_aggregate(_model(), tuples, 0, _config(server_clip_norm=1.0))
    assert math.isclose(
        result.metrics["dtldp_upload_norm_pre_server_max"], 2.0, rel_tol=1e-7
    )
    assert math.isclose(
        result.metrics["dtldp_server_clip_utilization_max"], 2.0, rel_tol=1e-7
    )
    assert math.isclose(
        result.metrics["dtldp_server_clip_rate"], 1.0 / 3.0, rel_tol=1e-6
    )
    assert result.metrics["dtldp_server_clip_rate_byzantine_oracle"] == 1.0
    assert result.metrics["dtldp_server_clip_rate_honest_oracle"] == 0.0
    assert math.isclose(
        result.metrics["dtldp_upload_norm_post_server_max"], 1.0, rel_tol=1e-7
    )
    assert math.isclose(
        result.metrics["dtldp_byzantine_weighted_contribution_norm_oracle"],
        1.0 / 3.0,
        rel_tol=1e-6,
    )


def test_oracle_diagnostics_are_explicitly_namespaced():
    tuples = _tuples([-0.2, 0.0, 0.1], oracle_noise=[0.4, 0.7, 0.2])
    for index, (_, metadata, _) in enumerate(tuples):
        metadata["dtldp_local_loss_oracle"] = float(index + 1)
        metadata["dtldp_clip_rate_oracle"] = 0.1 * index
    result = get_algorithm("dt_ldp_far").server_aggregate(
        _model(), tuples, 0, _config(enable_oracle_diagnostics=True)
    )
    assert result.metrics["dtldp_local_loss_mean_oracle"] == 2.0
    assert math.isclose(result.metrics["dtldp_client_clip_rate_mean_oracle"], 0.1)
    assert "privacy_realised_noise_norm_mean_oracle" in result.metrics
    assert math.isclose(
        result.metrics["dtldp_realised_noise_norm_mean_oracle"], 1.3 / 3
    )


def test_last_layer_reference_oracle_uses_the_same_score_subspace():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 2),
    )
    tuples = []
    for client_id, value in enumerate((-0.2, 0.0, 0.1)):
        update = {
            key: torch.full_like(parameter, value)
            for key, parameter in model.state_dict().items()
        }
        clean_update = {
            key: torch.full_like(parameter, value / 2)
            for key, parameter in model.state_dict().items()
        }
        tuples.append(
            (
                update,
                {
                    "client_id": client_id,
                    "dataset_size": 4,
                    "local_loss": 0.0,
                    "bytes_sent": 4,
                    "energy_j_consumed": 0.0,
                    "privacy_epsilon": 2.0,
                    "privacy_noise_multiplier": 1.2,
                    "privacy_noise_multiplier_scale_public": 1.0,
                    "model_steps": 1,
                    "dtldp_noise_free_update_oracle": clean_update,
                },
                ClientState(client_id=client_id, battery_j=10.0),
            )
        )

    result = get_algorithm("dt_ldp_far").server_aggregate(
        model,
        tuples,
        0,
        _config(
            enable_oracle_diagnostics=True,
            score_subspace_mode="last_layer",
        ),
    )

    assert result.metrics["dtldp_score_subspace_mode"] == "last_layer"
    assert result.metrics["dtldp_score_subspace_dimension"] == 8
    assert "dtldp_reference_honest_center_error_oracle" in result.metrics


def test_client_private_oracles_require_separate_explicit_flag(monkeypatch):
    algo = get_algorithm("dt_ldp_far")

    def fake_update(model, dataloader, state, config):
        return {"weight": torch.zeros(1, 1)}, {
            "client_id": state.client_id,
            "round_num": 1,
            "local_loss": 1.5,
            "clip_rate": 0.4,
            "dp_noise_norm_mean": 2.0,
            "bytes_sent": 4,
            "bytes_received": 4,
            "energy_j_consumed": 0.0,
            "battery_j_remaining": state.battery_j,
            "compression_ratio": 1.0,
            "beta_actual": 1.0,
        }

    monkeypatch.setattr(algo, "_local_dp_update", fake_update)
    state = ClientState(client_id=0, battery_j=100.0)
    _, metadata = algo.client_update(
        _model(),
        None,
        state,
        _config(
            enable_oracle_diagnostics=True,
            enable_private_client_oracle_diagnostics=False,
            max_local_batches=1,
        ),
    )
    assert "dtldp_local_loss_oracle" not in metadata
    assert metadata["dtldp_transcript_policy"] == (
        "no_unaccounted_loss_clip_or_noise_release"
    )


def test_smoke_config_is_registered_and_full_participation():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs" / "dt_ldp_far" / "smoke_mnist.yaml").read_text()
    )
    assert config["training"]["algorithm"] == "dt_ldp_far"
    algo = config["training"]["algo_config"]
    clients = config["clients"]
    assert algo["require_full_participation"] is True
    assert algo["expected_num_clients"] == clients["num_clients"]
    assert clients["sample_fraction"] == 1.0
    assert clients["min_clients"] == clients["num_clients"]
    assert clients["dropout_rate"] == 0.0
    assert algo["tilt_delay_rounds"] == 1
    assert algo["sampling_scheme"] == "poisson"
    assert algo["privacy_adjacency"] == "add_remove"
    assert math.isclose(algo["privacy_sampling_rate_override"], 0.05)
    assert algo["poisson_steps_per_round"] == 1
    assert algo["privacy_public_dataset_size"] == 60001


def test_private_client_metadata_uses_public_capacity_not_real_cardinality():
    model = torch.nn.Linear(2, 2)
    loader = DataLoader(
        TensorDataset(torch.eye(2), torch.tensor([0, 1])),
        batch_size=2,
        shuffle=False,
    )
    config = _config(
        device="cpu",
        lr=0.01,
        local_epochs=1,
        batch_size=2,
        clip_norm=1.0,
        noise_multiplier=0.0,
        target_epsilon=None,
        sampling_scheme="poisson",
        privacy_adjacency="add_remove",
        privacy_sampling_rate_override=1.0,
        privacy_public_dataset_size=3,
        per_sample_backend="vectorized",
        enable_private_client_oracle_diagnostics=False,
    )
    _, metadata = get_algorithm("dt_ldp_far").client_update(
        model,
        loader,
        ClientState(client_id=0, battery_j=100.0),
        config,
    )
    assert len(loader.dataset) == 2
    assert metadata["dataset_size"] == 3
    assert metadata["privacy_normalization_denominator"] == 3.0
    assert metadata["privacy_adjacency"] == "add_remove"
    assert metadata["privacy_sampling_scheme"] == "poisson"


def test_fixed_without_replacement_metadata_and_accountant_are_explicit():
    model = torch.nn.Linear(2, 2)
    loader = DataLoader(
        TensorDataset(torch.eye(2), torch.tensor([0, 1])),
        batch_size=2,
        shuffle=False,
    )
    config = _config(
        device="cpu",
        lr=0.01,
        local_epochs=1,
        batch_size=2,
        clip_norm=1.0,
        noise_multiplier=2.0,
        target_epsilon=None,
        sampling_scheme="fixed_without_replacement",
        privacy_adjacency="replace_one",
        privacy_sampling_rate_override=0.5,
        privacy_public_dataset_size=2,
        fixed_batch_size=1,
        fixed_steps_per_round=1,
        per_sample_backend="vectorized",
        enable_private_client_oracle_diagnostics=False,
    )
    _, metadata = get_algorithm("dt_ldp_far").client_update(
        model,
        loader,
        ClientState(client_id=0, battery_j=100.0),
        config,
    )
    assert metadata["dataset_size"] == 2
    assert metadata["privacy_adjacency"] == "replace_one"
    assert metadata["privacy_sampling_scheme"] == "fixed_without_replacement"
    assert metadata["privacy_fixed_batch_size"] == 1
    assert metadata["privacy_normalization_denominator"] == 1.0
    assert metadata["privacy_sensitivity_multiplier"] == 2.0
    assert metadata["privacy_accounting_noise_multiplier"] == 1.0
    assert metadata["privacy_accounting_assumption"] == (
        "fixed_size_without_replacement_rdp_replace_one"
    )
