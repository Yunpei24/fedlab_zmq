"""Protocol-lock tests for the full-update SC-FAR-DP paper-1 matrices."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from privacy.rdp import RDPAccountant
from scripts.run_scfar_paper1 import (
    ROOT,
    alpha_max,
    expand_tasks,
    load_matrix,
    validate_matrix,
)
from scripts.validate_scfar_gaussian_accountant import (
    DEFAULT_ORDERS,
    direct_epsilon,
)

MATRIX_DIR = ROOT / "configs" / "scpfar" / "paper1"
MATRICES = {
    "s0_step1a_fmnist_health_dev.yaml": 5,
    "s0_step1a_fmnist_health_refine_dev.yaml": 4,
    "s0_step1b_fmnist_geometry_dev.yaml": 10,
    "s0_step1c_fmnist_score_calibration_dev.yaml": 7,
    "s0_step1d_fmnist_multiseed_score_tilt_screen.yaml": 39,
    "s0_step1e_fmnist_central_dp_raw_confirmation.yaml": 15,
    "s1_reference_tradeoff.yaml": 1908,
    "s2_full_update_ablations.yaml": 360,
    "s3_inclusion_attacks.yaml": 1224,
    "s4_central_dp.yaml": 720,
}


def test_s0_step1a_uses_disjoint_dev_seeds_and_expected_clipping_grid():
    document = load_matrix(MATRIX_DIR / "s0_step1a_fmnist_health_dev.yaml")
    tasks = expand_tasks(document, output_root=Path("/tmp/scfar-paper1-test"))
    assert len(tasks) == 5
    assert {task.partition_seed for task in tasks} == {104}
    assert {task.training_seed for task in tasks} == {17}

    native = [task for task in tasks if task.method_id == "fedavg"]
    clipped = [
        task for task in tasks if task.method_id == "central_dp_fedavg_exact"
    ]
    assert len(native) == 1
    assert native[0].config["training"]["algorithm"] == "fedavg"
    assert len(clipped) == 4
    assert {
        task.config["training"]["algo_config"]["user_clip_norm"]
        for task in clipped
    } == {1.0, 2.0, 4.0, 8.0}
    for task in clipped:
        algo = task.config["training"]["algo_config"]
        assert task.config["training"]["algorithm"] == "scfar_dp"
        assert algo["scfar_aggregation_rule"] == "uniform"
        assert algo["enable_central_dp"] is False
        assert algo["central_noise_multiplier"] == 0.0


def test_s0_step1a_refinement_uses_only_the_preregistered_interval():
    document = load_matrix(
        MATRIX_DIR / "s0_step1a_fmnist_health_refine_dev.yaml"
    )
    tasks = expand_tasks(document, output_root=Path("/tmp/scfar-paper1-test"))
    assert len(tasks) == 4
    assert {task.partition_seed for task in tasks} == {104}
    assert {task.training_seed for task in tasks} == {17}
    assert {
        task.config["training"]["algo_config"]["user_clip_norm"]
        for task in tasks
    } == {1.2, 1.4, 1.6, 1.8}
    for task in tasks:
        algo = task.config["training"]["algo_config"]
        assert task.method_id == "central_dp_fedavg_exact"
        assert algo["scfar_aggregation_rule"] == "uniform"
        assert algo["enable_central_dp"] is False


def test_s0_step1b_freezes_clip_and_uses_disjoint_development_seeds():
    document = load_matrix(MATRIX_DIR / "s0_step1b_fmnist_geometry_dev.yaml")
    tasks = expand_tasks(document, output_root=Path("/tmp/scfar-paper1-test"))
    assert len(tasks) == 10
    assert {task.partition_seed for task in tasks} == {105}
    assert {task.training_seed for task in tasks} == {19}
    assert {
        task.config["training"]["algo_config"]["user_clip_norm"]
        for task in tasks
    } == {1.4}
    tilted = [task for task in tasks if task.method_id == "scfar_no_dp"]
    assert len(tilted) == 9
    assert {task.anchor_id for task in tilted} == {
        "fixed_zero",
        "ema_release_0p1",
        "previous_release",
    }
    assert {
        task.config["training"]["algo_config"]["reference_clip_tau"]
        for task in tilted
    } == {0.35, 0.7, 1.4}
    for task in tilted:
        algo = task.config["training"]["algo_config"]
        assert algo["scfar_aggregation_rule"] == "controlled_tilt"
        assert algo["enable_central_dp"] is False


def test_s0_step1c_pairs_bounded_scores_with_raw_distance():
    document = load_matrix(
        MATRIX_DIR / "s0_step1c_fmnist_score_calibration_dev.yaml"
    )
    tasks = expand_tasks(document, output_root=Path("/tmp/scfar-paper1-test"))
    assert len(tasks) == 7
    assert {task.partition_seed for task in tasks} == {106}
    assert {task.training_seed for task in tasks} == {23}
    assert {
        task.config["training"]["algo_config"]["user_clip_norm"]
        for task in tasks
    } == {1.4}

    bounded = [task for task in tasks if task.method_id == "scfar_no_dp"]
    raw = [task for task in tasks if task.method_id == "far_raw_distance_fcc"]
    uniform = [task for task in tasks if task.method_id == "central_dp_fedavg_exact"]
    assert len(bounded) == 5
    assert len(raw) == len(uniform) == 1
    assert {task.distance_score_over_c for task in bounded} == {
        0.5,
        0.75,
        1.0,
        1.25,
        2.0,
    }
    assert sorted(
        task.config["training"]["algo_config"]["distance_clip"]
        for task in bounded
    ) == pytest.approx([0.7, 1.05, 1.4, 1.75, 2.8])
    assert raw[0].distance_score_over_c is None
    assert (
        raw[0].config["training"]["algo_config"]["scfar_aggregation_rule"]
        == "far_raw_distance"
    )
    assert raw[0].config["training"]["algo_config"]["enable_central_dp"] is False
    assert (
        raw[0].config["training"]["algo_config"]["far_alpha"]
        == bounded[0].config["training"]["algo_config"]["far_alpha"]
    )


def test_s0_step1d_uses_paired_seeds_and_matches_raw_logits():
    tasks = expand_tasks(
        load_matrix(MATRIX_DIR / "s0_step1d_fmnist_multiseed_score_tilt_screen.yaml"),
        output_root=Path("/tmp/scfar-paper1-test"),
    )
    assert len(tasks) == 39
    assert {(task.partition_seed, task.training_seed) for task in tasks} == {
        (101, 28),
        (102, 36),
        (103, 54),
    }
    matched = [
        task
        for task in tasks
        if task.method_id == "far_raw_distance_logit_matched_fcc"
    ]
    assert len(matched) == 12
    for task in matched:
        algo = task.config["training"]["algo_config"]
        bounded_alpha = (
            algo["alpha_fraction_of_max"]
            * alpha_max(task.config["clients"]["num_clients"], algo["kappa_w"])
        )
        d_score = task.distance_score_over_c * algo["user_clip_norm"]
        assert algo["far_alpha"] == pytest.approx(bounded_alpha / d_score)
        assert algo["raw_alpha_scaling"] == "matched_bounded_D_score"


def test_s0_step1e_raw_alpha_and_sensitivity_use_public_2c_range():
    tasks = expand_tasks(
        load_matrix(MATRIX_DIR / "s0_step1e_fmnist_central_dp_raw_confirmation.yaml"),
        output_root=Path("/tmp/scfar-paper1-test"),
    )
    raw = [
        task for task in tasks if task.method_id == "scfar_raw_distance_certified_fcc"
    ]
    assert len(raw) == 6
    for task in raw:
        algo = task.config["training"]["algo_config"]
        expected = alpha_max(25, algo["kappa_w"]) / (2.0 * algo["user_clip_norm"])
        assert algo["far_alpha"] == pytest.approx(expected)
        assert algo["sensitivity_mode"] == "proved_raw_distance_reference_bound"
        assert algo["enable_central_dp"] is True


def test_s0_step1g_effective_dimension_screen_is_public_masked_and_separate():
    document = load_matrix(
        MATRIX_DIR / "s0_step1g_fmnist_effective_dimension_nodp_screen.yaml"
    )
    assert validate_matrix(document) == []
    tasks = expand_tasks(document, output_root=Path("/tmp/scfar-paper1-test"))
    assert len(tasks) == 5
    assert {task.config["reproduction"]["protocol_id"] for task in tasks} == {
        "scfar_dp_effective_private_dimension_screen_v1"
    }
    assert {task.config["reproduction"]["execution_scope"] for task in tasks} == {
        "public_static_parameter_mask_utility_screen"
    }
    assert {task.config["training"]["algo_config"]["active_parameter_mode"] for task in tasks} == {
        "full",
        "classifier_head",
        "classifier_tail",
        "last_layer",
        "bias_only",
    }
    for task in tasks:
        assert task.config["training"]["algorithm"] == "fedavg"
        assert task.config["training"]["num_rounds"] == 40
        assert task.config["training"]["algo_config"]["enable_central_dp"] is False
        assert task.config["clients"]["num_clients"] == 25
        assert task.config["clients"]["sample_fraction"] == 1.0
        assert task.config["clients"]["dropout_rate"] == 0.0


def test_s0_step1h_promotes_only_step1g_masks_at_fixed_privacy_constants():
    document = load_matrix(
        MATRIX_DIR / "s0_step1h_fmnist_masked_dpfedavg_eps10_screen.yaml"
    )
    assert validate_matrix(document) == []
    tasks = expand_tasks(document, output_root=Path("/tmp/scfar-paper1-test"))
    assert len(tasks) == 6
    assert {task.privacy_id for task in tasks} == {"no_dp", "eps10"}
    assert {task.config["training"]["algo_config"]["active_parameter_mode"] for task in tasks} == {
        "classifier_head",
        "classifier_tail",
        "last_layer",
    }
    for task in tasks:
        algo = task.config["training"]["algo_config"]
        assert task.config["training"]["algorithm"] == "scfar_dp"
        assert task.config["training"]["num_rounds"] == 40
        assert algo["scfar_aggregation_rule"] == "uniform"
        assert algo["user_clip_norm"] == pytest.approx(1.4)
        assert algo["privacy_num_rounds"] == 40
        assert algo["target_epsilon"] in {None, 10.0}


def test_s0_step1i_preregisters_joint_mask_clip_calibration():
    document = load_matrix(
        MATRIX_DIR / "s0_step1i_fmnist_mask_clip_calibration.yaml"
    )
    assert validate_matrix(document) == []
    tasks = expand_tasks(document, output_root=Path("/tmp/scfar-paper1-test"))
    assert len(tasks) == 8
    assert {(task.partition_seed, task.training_seed) for task in tasks} == {
        (104, 17)
    }
    assert {task.privacy_id for task in tasks} == {"no_dp"}
    assert {task.config["training"]["num_rounds"] for task in tasks} == {20}
    assert {
        task.config["training"]["algo_config"]["active_parameter_mode"]
        for task in tasks
    } == {"classifier_tail", "last_layer"}
    assert {
        task.config["training"]["algo_config"]["user_clip_norm"]
        for task in tasks
    } == {0.7, 1.4, 2.8, 5.6}
    for task in tasks:
        algo = task.config["training"]["algo_config"]
        assert task.config["training"]["algorithm"] == "scfar_dp"
        assert algo["scfar_aggregation_rule"] == "uniform"
        assert algo["enable_central_dp"] is False
        assert algo["privacy_num_rounds"] == 20


def test_s0_step1ir_validates_one_quantile_clip_candidate_per_mask():
    document = load_matrix(
        MATRIX_DIR / "s0_step1ir_fmnist_mask_clip_refinement.yaml"
    )
    assert validate_matrix(document) == []
    tasks = expand_tasks(document, output_root=Path("/tmp/scfar-paper1-test"))
    assert len(tasks) == 2
    assert {(task.partition_seed, task.training_seed) for task in tasks} == {
        (104, 17)
    }
    assert {task.privacy_id for task in tasks} == {"no_dp"}
    assert {
        task.config["training"]["algo_config"]["active_parameter_mode"]
        for task in tasks
    } == {"classifier_tail", "last_layer"}
    for task in tasks:
        algo = task.config["training"]["algo_config"]
        assert task.config["training"]["num_rounds"] == 20
        assert algo["user_clip_norm"] > 0.0
        assert algo["privacy_num_rounds"] == 20
        assert algo["enable_central_dp"] is False


@pytest.mark.parametrize("rounds", [5, 10, 20])
def test_s0_step1j_pairs_privacy_at_each_preregistered_horizon(rounds: int):
    document = load_matrix(
        MATRIX_DIR / "s0_step1j_fmnist_joint_mask_clip_horizon.yaml"
    )
    assert validate_matrix(document) == []
    tasks = expand_tasks(
        document,
        output_root=Path("/tmp/scfar-paper1-test"),
        pilot_rounds=rounds,
    )
    assert len(tasks) == 2
    assert {task.privacy_id for task in tasks} == {"no_dp", "eps10"}
    assert {
        task.config["training"]["algo_config"]["active_parameter_mode"]
        for task in tasks
    } == {"classifier_tail"}
    for task in tasks:
        algo = task.config["training"]["algo_config"]
        assert task.config["training"]["num_rounds"] == rounds
        assert algo["privacy_num_rounds"] == rounds
        assert algo["user_clip_norm"] == pytest.approx(1.631041)
        assert algo["target_epsilon"] in {None, 10.0}


@pytest.mark.parametrize(("filename", "expected"), MATRICES.items())
def test_matrix_is_valid_unique_and_has_frozen_cardinality(filename: str, expected: int):
    document = load_matrix(MATRIX_DIR / filename)
    assert validate_matrix(document) == []
    tasks = expand_tasks(document, output_root=Path("/tmp/scfar-paper1-test"))
    assert len(tasks) == expected
    assert len({task.task_id for task in tasks}) == expected


@pytest.mark.parametrize("filename", MATRICES)
def test_every_paper1_task_is_full_update_full_participation(filename: str):
    document = load_matrix(MATRIX_DIR / filename)
    tasks = expand_tasks(document, output_root=Path("/tmp/scfar-paper1-test"))
    for task in tasks:
        cfg = task.config
        clients = cfg["clients"]
        algo = cfg["training"]["algo_config"]
        assert clients["sample_fraction"] == 1.0
        assert clients["min_clients"] == clients["num_clients"] == 25
        assert clients["dropout_rate"] == 0.0
        assert cfg["data"]["partition"] == "client_dirichlet_balanced"
        assert algo.get("active_parameter_mode", "full") == "full"
        assert not {"num_layer_groups", "layer_selection", "rounds_per_layer"}.intersection(algo)
        if cfg["training"]["algorithm"] == "scfar_dp":
            assert algo["alpha_bound_policy"] == "error"
            assert algo["privacy_num_rounds"] == cfg["training"]["num_rounds"]
            assert set(algo["honest_outlier_client_ids"]).isdisjoint(
                set(algo["attack"].get("client_ids", []))
            )


def test_s3_tilts_are_derived_inside_the_certified_region():
    document = load_matrix(MATRIX_DIR / "s3_inclusion_attacks.yaml")
    tasks = expand_tasks(document, output_root=Path("/tmp/scfar-paper1-test"))
    for task in tasks:
        algo = task.config["training"]["algo_config"]
        kappa = float(algo["kappa_w"])
        requested = float(algo["far_alpha"])
        maximum = alpha_max(25, kappa)
        assert 0.0 <= requested <= maximum + 1e-12
        assert math.isclose(
            requested,
            float(algo["alpha_fraction_of_max"]) * maximum,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


def test_s4_infinity_lane_really_disables_noise_and_finite_lanes_enable_it():
    document = load_matrix(MATRIX_DIR / "s4_central_dp.yaml")
    tasks = expand_tasks(document, output_root=Path("/tmp/scfar-paper1-test"))
    for task in tasks:
        algo = task.config["training"]["algo_config"]
        if task.privacy_id == "no_dp":
            assert algo["enable_central_dp"] is False
            assert algo["central_noise_multiplier"] == 0.0
            assert algo["target_epsilon"] is None
        else:
            assert algo["enable_central_dp"] is True
            assert float(algo["target_epsilon"]) in {1.0, 3.0, 6.0, 10.0}


def test_honest_outliers_are_preregistered_and_exclude_byzantines():
    document = load_matrix(MATRIX_DIR / "s3_inclusion_attacks.yaml")
    tasks = expand_tasks(document, output_root=Path("/tmp/scfar-paper1-test"))
    for task in tasks:
        algo = task.config["training"]["algo_config"]
        outliers = set(algo["honest_outlier_client_ids"])
        byzantines = set(algo["attack"].get("client_ids", []))
        assert len(outliers) == 4
        assert max(outliers) < 20
        assert outliers.isdisjoint(byzantines)


def test_direct_q_one_accountant_crosscheck_matches_the_ledger():
    sigma, steps, delta = 4.25, 100, 1e-5
    direct, direct_order, direct_rdp = direct_epsilon(
        noise_multiplier=sigma, steps=steps, delta=delta
    )
    accountant = RDPAccountant(orders=DEFAULT_ORDERS)
    accountant.add_gaussian(
        channel="central_model", noise_multiplier=sigma, steps=steps
    )
    epsilon, order = accountant.epsilon(delta)
    assert epsilon == pytest.approx(direct, abs=1e-12)
    assert order == direct_order
    assert accountant.total_rdp() == pytest.approx(direct_rdp, abs=1e-12)
