from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from algorithms.aggregate_control_replay import (
    diagnostic_control_candidates,
    evaluate_aggregate_control_replay,
)
from algorithms.aggregate_radial_control import (
    AggregateControlConfig,
    PredictableAggregateState,
    PredictableSnapshot,
    diagonal_mahalanobis_norm,
    euclidean_project_diagonal_ellipsoid,
    isotropic_clip_about_predictor,
    radial_ellipsoid_clip,
)
from algorithms.ldp_aggregation_role_ablation import cohort_rules
from scripts import run_aggregate_control_replay as replay_runner


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/ldp_gradient_far/aggregate_control_preregistered_v1.yaml"


def snapshot(predictor=(0.0, 0.0), variance=(1.0, 1.0)):
    return PredictableSnapshot(
        round_num=1,
        observations=1,
        ready=True,
        predictor=torch.tensor(predictor, dtype=torch.float64),
        variance=torch.tensor(variance, dtype=torch.float64),
    )


def test_isotropic_projection_and_inside_identity():
    predictor = torch.tensor([1.0, -1.0])
    inside = torch.tensor([1.3, -1.4])
    projected, gamma = isotropic_clip_about_predictor(
        inside, predictor, radius=1.0
    )
    assert gamma == 1.0
    assert torch.equal(projected, inside.to(torch.float64))

    outside = torch.tensor([4.0, 3.0])
    projected, gamma = isotropic_clip_about_predictor(
        outside, predictor, radius=2.0
    )
    assert 0.0 < gamma < 1.0
    assert torch.linalg.vector_norm(projected - predictor).item() == pytest.approx(2.0)


def test_inside_controls_return_exact_clone_despite_float64_cancellation():
    candidate = torch.tensor([1e-16, 0.2, 0.3], dtype=torch.float64)
    predictor = torch.tensor([1.0, 10.0, 0.1], dtype=torch.float64)
    variance = torch.ones(3, dtype=torch.float64)
    isotropic, gamma_i = isotropic_clip_about_predictor(
        candidate, predictor, radius=20.0
    )
    radial, gamma_r = radial_ellipsoid_clip(
        candidate, predictor, variance, radius=20.0
    )
    assert gamma_i == gamma_r == 1.0
    assert torch.equal(isotropic, candidate)
    assert torch.equal(radial, candidate)


def test_radial_has_mahalanobis_non_degradation_for_covered_target():
    predictor = torch.tensor([0.0, 0.0])
    variance = torch.tensor([9.0, 1.0])
    candidate = torch.tensor([9.0, 4.0])
    target = torch.tensor([1.5, 0.25])
    radius = 1.0
    assert diagonal_mahalanobis_norm(target - predictor, variance) < radius
    corrected, _ = radial_ellipsoid_clip(
        candidate, predictor, variance, radius=radius
    )
    raw_error = diagonal_mahalanobis_norm(candidate - target, variance)
    corrected_error = diagonal_mahalanobis_norm(corrected - target, variance)
    assert corrected_error <= raw_error + 1e-12
    assert diagonal_mahalanobis_norm(corrected - predictor, variance) == pytest.approx(
        radius
    )


def test_radial_can_worsen_euclidean_error_but_true_projection_does_not():
    # Required counterexample: radial clipping is not a Euclidean projection.
    variance = torch.tensor([100.0, 1.0])
    predictor = torch.zeros(2)
    target = torch.tensor([10.0, 0.0])  # lies on the ellipsoid
    candidate = torch.tensor([10.0, 1.0])
    radial, _ = radial_ellipsoid_clip(
        candidate, predictor, variance, radius=1.0
    )
    projected, multiplier, iterations = euclidean_project_diagonal_ellipsoid(
        candidate, predictor, variance, radius=1.0
    )
    raw_error_sq = float((candidate - target).square().sum())
    radial_error_sq = float((radial - target).square().sum())
    projected_error_sq = float((projected - target).square().sum())
    assert raw_error_sq == pytest.approx(1.0)
    assert radial_error_sq == pytest.approx(9.0786437627)
    assert radial_error_sq > raw_error_sq
    assert projected_error_sq <= raw_error_sq + 1e-10
    assert multiplier > 0.0 and iterations > 0
    assert diagonal_mahalanobis_norm(projected, variance) <= 1.0 + 1e-9


@pytest.mark.parametrize("variance_value", [1e-8, 1e-20])
def test_euclidean_projection_is_scale_accurate_for_tiny_variance(variance_value):
    variance = torch.tensor([variance_value], dtype=torch.float64)
    predictor = torch.zeros(1, dtype=torch.float64)
    target = torch.sqrt(variance)  # lies exactly on the radius-one ellipsoid
    candidate = target * (1.0 + 1e-6)
    projected, _, _ = euclidean_project_diagonal_ellipsoid(
        candidate, predictor, variance, radius=1.0, tolerance=1e-10
    )
    raw_error_sq = float((candidate - target).square().sum())
    projected_error_sq = float((projected - target).square().sum())
    assert diagonal_mahalanobis_norm(projected, variance) == pytest.approx(
        1.0, rel=1e-9, abs=1e-12
    )
    assert projected_error_sq <= raw_error_sq + 1e-30 * variance_value


@pytest.mark.parametrize(
    "candidate,predictor,variance,radius",
    [
        ([float("nan"), 0.0], [0.0, 0.0], [1.0, 1.0], 1.0),
        ([1.0, 0.0], [0.0, 0.0], [0.0, 1.0], 1.0),
        ([1.0, 0.0], [0.0, 0.0], [1.0, 1.0], 0.0),
    ],
)
def test_ellipsoid_controls_reject_invalid_inputs(
    candidate, predictor, variance, radius
):
    with pytest.raises(ValueError):
        radial_ellipsoid_clip(candidate, predictor, variance, radius=radius)
    with pytest.raises(ValueError):
        euclidean_project_diagonal_ellipsoid(
            candidate, predictor, variance, radius=radius
        )


def test_predictable_snapshot_is_current_independent_and_state_uses_uncorrected():
    cfg = AggregateControlConfig(
        predictor_rate=0.5,
        covariance_rate=0.25,
        initial_variance=2.0,
        variance_ridge=1e-8,
    )
    state_a = PredictableAggregateState(2, cfg)
    state_b = PredictableAggregateState(2, cfg)
    first = torch.tensor([1.0, -1.0])
    for state in (state_a, state_b):
        assert not state.snapshot(0).ready
        state.observe_uncorrected(first, 0)
    snap_a = state_a.snapshot(1)
    snap_b = state_b.snapshot(1)
    # Different hypothetical controls/currents cannot change the pre-current state.
    assert torch.equal(snap_a.predictor, snap_b.predictor)
    assert torch.equal(snap_a.variance, snap_b.variance)
    uncorrected = torch.tensor([3.0, 1.0])
    state_a.observe_uncorrected(uncorrected, 1)
    state_b.observe_uncorrected(uncorrected, 1)
    assert torch.equal(state_a.snapshot(2).predictor, state_b.snapshot(2).predictor)
    assert torch.equal(state_a.snapshot(2).variance, state_b.snapshot(2).variance)
    with pytest.raises(ValueError, match="non-sequential"):
        state_a.snapshot(4)


def test_repeated_contamination_can_move_the_common_predictor():
    cfg = AggregateControlConfig(
        predictor_rate=0.5,
        covariance_rate=0.2,
        initial_variance=1.0,
        variance_ridge=1e-8,
    )
    state = PredictableAggregateState(1, cfg)
    state.observe_uncorrected(torch.tensor([0.0]), 0)
    for round_num in range(1, 5):
        state.observe_uncorrected(torch.tensor([10.0]), round_num)
    assert state.snapshot(5).predictor.item() == pytest.approx(9.375)


def test_diagnostic_grid_is_finite_and_uses_sqrt_dimension_units():
    config = {
        "aggregate_control_ema_current_mixes": [0.2],
        "aggregate_control_isotropic_radius_multipliers": [0.5, 1.0, 2.0],
        "aggregate_control_mahalanobis_radius_multipliers": [0.5, 1.0, 2.0],
        "aggregate_control_projection_tolerance": 1e-10,
        "aggregate_control_projection_max_iterations": 160,
    }
    candidate = torch.arange(1.0, 5.0, dtype=torch.float64)
    controls = diagnostic_control_candidates(
        candidate,
        snapshot(predictor=(0.0, 0.0, 0.0, 0.0), variance=(1.0,) * 4),
        config=config,
        num_clients=10,
        server_clip_norm=16.0,
    )
    assert set(controls) == {
        "unchanged",
        "ema_mix_0p2",
        "isotropic_mult_0p5",
        "isotropic_mult_1",
        "isotropic_mult_2",
        "radial_mult_0p5",
        "radial_mult_1",
        "radial_mult_2",
        "euclidean_projection_mult_0p5",
        "euclidean_projection_mult_1",
        "euclidean_projection_mult_2",
    }
    assert controls["radial_mult_1"]["diagnostics"]["radius"] == pytest.approx(
        2.0
    )
    assert all(torch.isfinite(entry["vector"]).all() for entry in controls.values())


def test_offline_replay_evaluator_saves_oracle_sufficient_scalars_only():
    updates = [
        ({"w": torch.tensor([1.0, 0.0])}, {"client_id": 0}, object()),
        ({"w": torch.tensor([3.0, 0.0])}, {"client_id": 1}, object()),
    ]
    clean = {
        0: {"w": torch.tensor([0.5, 0.0])},
        1: {"w": torch.tensor([1.5, 0.0])},
    }
    vectors = torch.tensor([[1.0, 0.0], [3.0, 0.0]], dtype=torch.float64)
    rules = cohort_rules(vectors, torch.zeros(2, dtype=torch.float64), alpha=0.1)
    base = rules["aggregates"]["far_rfa"]
    snap = snapshot(predictor=(0.0, 0.0), variance=(1.0, 2.0))
    control_config = {
        "aggregate_control_ema_current_mixes": [0.2],
        "aggregate_control_isotropic_radius_multipliers": [1.0],
        "aggregate_control_mahalanobis_radius_multipliers": [1.0],
        "aggregate_control_projection_tolerance": 1e-10,
        "aggregate_control_projection_max_iterations": 160,
    }
    controls = diagnostic_control_candidates(
        base,
        snap,
        config=control_config,
        num_clients=2,
        server_clip_norm=10.0,
    )
    aggregation_payload = {
        "kind": "aggregation_role_v1",
        "round": 2,
        "client_ids": [0, 1],
        "server_clip_norm": 10.0,
        "deployed": "far_rfa",
        "requested_arm": "far_rfa",
        "ready": True,
        "vectors": vectors,
        "public_noise_scales": [1.0, 1.0],
        "rules": rules,
        "contains_clean_data": False,
        "contains_attack_labels": False,
        "contains_realised_noise": False,
    }
    payload = {
        "kind": "aggregate_control_replay_v1",
        "round": 2,
        "client_ids": [0, 1],
        "server_clip_norm": 10.0,
        "base_candidate": base,
        "predictor": snap.predictor,
        "variance": snap.variance,
        "predictor_ready": True,
        "controls": controls,
        "aggregation_payload": aggregation_payload,
        "contains_clean_data": False,
        "contains_attack_labels": False,
        "contains_realised_noise": False,
        "oracle_used_for_deployment": False,
        "calibration_status": "uncalibrated_diagnostic_grid_no_promotion",
    }
    metrics = evaluate_aggregate_control_replay(payload, clean, updates)
    assert metrics["aggregate_control_same_cohort_verified"]
    assert 0.0 <= metrics["aggregate_control_oracle_gamma_euclidean"] <= 1.0
    assert "aggregate_control_u_dot_predictor_minus_target" in metrics
    assert "aggregate_control_shadow_radial_mult_1_error_sq" in metrics
    assert not any(isinstance(value, torch.Tensor) for value in metrics.values())


def test_preregistered_seed_sets_and_first_phase_are_locked():
    matrix = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    registry = matrix["seed_registry"]
    sets = [set(registry[name]) for name in (
        "headroom_fit",
        "radius_calibration",
        "untouched_evaluation",
    )]
    assert all(len(values) == 3 for values in sets)
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert matrix["stages"]["headroom_fit"]["expected_runs"] == 6
    assert matrix["first_phase_execution"]["only_six_clean_runs"]
    assert matrix["fixed_training_contract"]["local_steps_per_round"] == 1
    assert matrix["fixed_training_contract"]["model_driver"] == (
        "uniform_rounds_1_to_12_then_far_rfa_rounds_13_to_40"
    )
    assert matrix["stages"]["radius_calibration"][
        "target_false_correction_rate"
    ] == 0.05
    margins = matrix["stages"]["untouched_evaluation"][
        "preliminary_clean_noninferiority_margins"
    ]
    assert margins["final_test_accuracy_drop_max_pp"] == 1.0
    assert margins["worst20_accuracy_drop_max_pp"] == 2.0
    assert margins["best20_worst20_gap_increase_max_pp"] == 2.0


def test_first_replay_runner_materializes_only_six_fit_tasks():
    matrix = replay_runner.load_matrix()
    tasks = replay_runner.headroom_tasks(matrix)
    assert len(tasks) == 6
    assert {task["seed"] for task in tasks} == {930101, 930102, 930103}
    assert {task["noise"] for task in tasks} == {
        "homogeneous",
        "heteroscedastic",
    }
    assert {task["scenario"] for task in tasks} == {"none"}
    closed = set(matrix["seed_registry"]["radius_calibration"]) | set(
        matrix["seed_registry"]["untouched_evaluation"]
    )
    assert not ({task["seed"] for task in tasks} & closed)


def test_resolved_replay_config_preserves_historical_driver_and_shadow_boundary():
    matrix = replay_runner.load_matrix()
    task = replay_runner.headroom_tasks(matrix)[0]
    config = replay_runner.resolved_config(
        matrix,
        task,
        {"matrix_sha256": "test-only-matrix-hash"},
    )
    algo = config["training"]["algo_config"]
    assert algo["aggregation_role_arm"] == "far_rfa"
    assert algo["rcig_n10_arm"] == "midpoint"
    assert algo["aggregate_control_model_driver"] == (
        "uniform_t1_t12_then_far_rfa"
    )
    assert algo["aggregate_control_replay_phase"] == "headroom_fit"
    assert algo["aggregate_control_oracle_never_selects_model"]
    assert algo["fixed_steps_per_round"] == 1
    assert algo["local_epochs"] == 1
    assert not algo["attack"]["enabled"]
