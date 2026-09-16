"""Validation tests for the executable DT-LDP-FAR E1--E8 protocol."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts.run_dt_ldp_far import (
    expand_tasks,
    is_complete,
    load_protocol,
    main,
    tilt_tau_max,
    validate_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "dt_ldp_far"


def _document(name: str):
    return load_protocol(CONFIG_ROOT / name)


def test_pilot_protocol_is_valid_and_has_frozen_cardinality(tmp_path):
    document = _document("pilot_e1_e8.yaml")
    assert validate_protocol(document) == []
    tasks = expand_tasks(document, output_root=tmp_path)
    assert len(tasks) == 37
    assert [task.index for task in tasks] == list(range(37))
    assert len({task.task_id for task in tasks}) == 37


def test_full_protocol_is_valid_and_bounded(tmp_path):
    document = _document("full_e1_e8.yaml")
    assert validate_protocol(document) == []
    tasks = expand_tasks(document, output_root=tmp_path)
    assert len(tasks) == 318
    assert {task.experiment_id for task in tasks} == {
        "E0_public_geometry_calibration",
        "E1_native_non_private",
        "E1_clipping_matched_no_noise",
        "E2_noise_decoupling",
        "E3_privacy_utility",
        "E4_byzantine",
        "E5_reference",
        "E6_staleness",
        "E7_poisson_steps",
        "E8_common_poisson",
        "E8_fedfdp_native",
        "E8_fedfdp_compute_matched",
    }


def test_decisive_protocols_are_valid_and_have_frozen_cardinality(tmp_path):
    expected = {
        "decisive_stage0_local_clip_calibration.yaml": 8,
        "decisive_stage0_geometry_refined_c4.yaml": 24,
        "decisive_stage0_geometry.yaml": 19,
        "decisive_stage1_alpha_delay_n25.yaml": 12,
        "decisive_stage1_references_attacks_n25.yaml": 30,
        "decisive_stage2_confirmatory_n25.yaml": 78,
        "decisive_stage2_references_attacks_n25.yaml": 24,
        "decisive_stage2_seed_hierarchy_n25.yaml": 9,
        "decisive_stage3_geometry_recalibration_n25.yaml": 12,
        "decisive_stage4_mechanistic_transfer_screen_n25.yaml": 26,
        "decisive_stage4_server_clip_ablation_n25.yaml": 24,
        "decisive_stage4_current_delay_references_n25.yaml": 72,
        "decisive_stage4_long_horizon_n25.yaml": 162,
        "decisive_stage5_end_to_end_stress_discovery_n25.yaml": 24,
        "decisive_stage9_fixed_without_replacement_n25.yaml": 36,
        "stage27_raw_distance_n25.yaml": 15,
    }
    for name, cardinality in expected.items():
        document = _document(name)
        assert validate_protocol(document) == []
        assert len(expand_tasks(document, output_root=tmp_path / name)) == cardinality


def test_current_and_delayed_controls_share_strict_randomness_pair_key(tmp_path):
    tasks = expand_tasks(
        _document("decisive_stage4_mechanistic_transfer_screen_n25.yaml"),
        output_root=tmp_path,
    )
    groups = {}
    for task in tasks:
        if task.experiment_id != "E11_alpha_stress_oracle_diagnostic_n25":
            continue
        key = (
            task.geometry_id,
            task.tilt_id,
            task.partition_seed,
            task.training_seed,
        )
        groups.setdefault(key, []).append(task)
    assert groups
    for paired_tasks in groups.values():
        assert {task.method_id for task in paired_tasks} == {
            "dp_far_current_matched_oracle",
            "dt_ldp_far_oracle",
        }
        pairing_keys = {
            task.config["reproduction"]["randomness_pair_key"] for task in paired_tasks
        }
        assert len(pairing_keys) == 1
        for task in paired_tasks:
            assert (
                task.config["reproduction"][
                    "private_client_oracles_make_run_non_private_diagnostic"
                ]
                is True
            )


def test_stage27_raw_distance_rescales_tilt_and_includes_slope_match(tmp_path):
    document = _document("stage27_raw_distance_n25.yaml")
    assert validate_protocol(document) == []
    tasks = expand_tasks(document, output_root=tmp_path)
    assert len(tasks) == 15
    assert {task.method_id for task in tasks} == {
        "dp_fedavg",
        "dp_far_current",
        "dt_ldp_far",
        "dt_ldp_far_raw_distance",
    }
    raw = [task for task in tasks if task.method_id == "dt_ldp_far_raw_distance"]
    assert len(raw) == 6
    for task in raw:
        algo = task.config["training"]["algo_config"]
        expected = tilt_tau_max(25, algo["kappa_w"]) / (2.0 * 0.42)
        assert algo["tilt_tau"] == pytest.approx(expected)
        assert algo["tilt_public_score_range"] == pytest.approx(0.84)
    slope_matched = [
        task for task in raw if task.tilt_id == "boundary_raw_slope_u42_d441"
    ]
    assert len(slope_matched) == 3
    bounded_alpha = tilt_tau_max(25, 2.0)
    assert slope_matched[0].config["training"]["algo_config"]["tilt_tau"] == pytest.approx(
        bounded_alpha / 0.441
    )


def test_stage22_is_preregistered_and_uniform_tilt_pairs_share_randomness(tmp_path):
    document = _document("stage22_lagged_descent_validation.yaml")
    assert validate_protocol(document) == []
    tasks = expand_tasks(document, output_root=tmp_path)
    assert len(tasks) == 24
    assert document["matrix"]["analysis_split"] == {
        "integration_screen_seed": 163,
        "confirmatory_seeds": [179, 193, 211],
        "screen_must_not_change_confirmatory_gates": True,
    }

    groups = {}
    for task in tasks:
        algo = task.config["training"]["algo_config"]
        assert algo["dt_trust_profile"] == "lagged_descent_alignment"
        assert algo["noise_score_trust_logit_fraction"] == 0.75
        assert algo["enable_private_client_oracle_diagnostics"] is False
        key = (task.training_seed, task.threat_id)
        groups.setdefault(key, []).append(task)
    assert len(groups) == 12
    for pair in groups.values():
        assert {task.tilt_id for task in pair} == {"uniform", "boundary"}
        assert (
            len({task.config["reproduction"]["randomness_pair_key"] for task in pair})
            == 1
        )


def test_stage23_filter_uses_its_active_support_for_the_tilt_cap(tmp_path):
    document = _document("stage23_robust_admissibility_validation.yaml")
    assert validate_protocol(document) == []
    tasks = expand_tasks(document, output_root=tmp_path)
    assert len(tasks) == 36

    filtered = [
        task for task in tasks if task.method_id == "dt_ldp_far_stage23_admissible_far"
    ]
    assert len(filtered) == 24
    for task in filtered:
        algo = task.config["training"]["algo_config"]
        n = task.config["clients"]["num_clients"]
        active = n - algo["dt_admissibility_max_byzantine"]
        maximum = tilt_tau_max(n, algo["kappa_w"], active_clients=active)
        assert math.isclose(
            algo["tilt_tau"],
            algo["tilt_tau_fraction_of_max"] * maximum,
            rel_tol=1e-12,
        )
        assert math.isclose(maximum, 0.5020919437972361, rel_tol=1e-12)


def test_stage24_multikrum_core_is_preregistered_and_certified(tmp_path):
    document = _document("stage24_multikrum_admissibility_validation.yaml")
    assert validate_protocol(document) == []
    tasks = expand_tasks(document, output_root=tmp_path)
    assert len(tasks) == 36
    assert document["matrix"]["analysis_split"] == {
        "integration_screen_seed": 307,
        "confirmatory_seeds": [311, 313, 317],
        "screen_must_not_change_confirmatory_gates": True,
    }

    candidate = [
        task for task in tasks if task.method_id == "dt_ldp_far_stage24_multikrum_far"
    ]
    assert len(candidate) == 24
    for task in candidate:
        algo = task.config["training"]["algo_config"]
        assert algo["dt_support_assumed_byzantine"] == 5
        assert algo["dt_admissibility_excluded_clients"] == 7
        n = task.config["clients"]["num_clients"]
        maximum = tilt_tau_max(n, algo["kappa_w"], active_clients=18)
        assert math.isclose(
            algo["tilt_tau"],
            algo["tilt_tau_fraction_of_max"] * maximum,
            rel_tol=1e-12,
        )


def test_stage25_robust_anchor_candidate_is_preregistered_and_bounded(tmp_path):
    document = _document("stage25_robust_anchor_containment_validation.yaml")
    assert validate_protocol(document) == []
    tasks = expand_tasks(document, output_root=tmp_path)
    assert len(tasks) == 36
    assert document["matrix"]["analysis_split"] == {
        "integration_screen_seed": 331,
        "confirmatory_seeds": [337, 347, 349],
        "screen_must_not_change_confirmatory_gates": True,
    }

    candidates = [
        task
        for task in tasks
        if task.method_id == "dt_ldp_far_stage25_robust_anchor_far"
    ]
    assert len(candidates) == 24
    for task in candidates:
        algo = task.config["training"]["algo_config"]
        assert algo["dt_aggregate_mode"] == "robust_anchor_perturbation"
        assert algo["dt_aggregate_anchor"] == "rfa"
        assert algo["dt_anchor_residual_radius_fraction"] == 1.0
        assert algo["dt_anchor_correction_gain"] == 0.25
        assert math.isclose(
            2.0
            * algo["dt_anchor_correction_gain"]
            * algo["dt_anchor_residual_radius_fraction"]
            * algo["server_clip_norm"],
            0.21,
            rel_tol=1e-12,
        )


def test_stage26a_anchor_screen_is_separate_from_locked_holdout(tmp_path):
    document = _document("stage26a_robust_anchor_selection_screen.yaml")
    assert validate_protocol(document) == []
    tasks = expand_tasks(document, output_root=tmp_path)
    assert len(tasks) == 15
    assert {task.training_seed for task in tasks} == {353}
    assert document["matrix"]["analysis_split"]["locked_confirmatory_seeds"] == [
        359,
        367,
        373,
    ]
    candidates = [
        task for task in tasks if task.method_id != "dt_ldp_far_stage19_peer_support"
    ]
    assert {
        task.config["training"]["algo_config"]["dt_aggregate_anchor"]
        for task in candidates
    } == {
        "coordinate_median",
        "trimmed_mean",
        "cm_nnm",
        "trmean_nnm",
    }
    for task in candidates:
        algo = task.config["training"]["algo_config"]
        assert algo["dt_aggregate_mode"] == "robust_anchor_perturbation"
        assert algo["dt_anchor_correction_gain"] == 0.25
        assert task.tilt_id == "uniform"


def test_stage26b_uses_only_selected_anchor_and_locked_holdout(tmp_path):
    document = _document("stage26b_trmean_nnm_anchor_confirmatory.yaml")
    assert validate_protocol(document) == []
    tasks = expand_tasks(document, output_root=tmp_path)
    assert len(tasks) == 27
    assert {task.training_seed for task in tasks} == {359, 367, 373}
    assert 353 not in {task.training_seed for task in tasks}
    candidates = [
        task
        for task in tasks
        if task.method_id == "dt_ldp_far_stage26_anchor_trmean_nnm"
    ]
    assert len(candidates) == 18
    assert {task.tilt_id for task in candidates} == {"uniform", "boundary"}
    for task in candidates:
        algo = task.config["training"]["algo_config"]
        assert algo["dt_aggregate_anchor"] == "trmean_nnm"
        assert algo["dt_aggregate_mode"] == "robust_anchor_perturbation"
        assert algo["dt_anchor_correction_gain"] == 0.25


def test_stage5_effective_noise_screen_is_oracle_only_and_strictly_paired(tmp_path):
    tasks = expand_tasks(
        _document("decisive_stage5_end_to_end_stress_discovery_n25.yaml"),
        output_root=tmp_path,
    )
    assert len(tasks) == 24
    groups = {}
    for task in tasks:
        algo = task.config["training"]["algo_config"]
        assert algo["enable_private_client_oracle_diagnostics"] is True
        assert algo["enable_noise_free_counterfactual_oracle"] is True
        assert (
            task.config["reproduction"][
                "private_client_oracles_make_run_non_private_diagnostic"
            ]
            is True
        )
        key = (
            task.experiment_id,
            task.privacy_id,
            task.geometry_id,
            task.partition_seed,
            task.training_seed,
        )
        groups.setdefault(key, []).append(task)
    assert len(groups) == 12
    for paired_tasks in groups.values():
        assert {task.method_id for task in paired_tasks} == {
            "dp_far_current_matched_oracle",
            "dt_ldp_far_oracle",
        }
        assert (
            len(
                {
                    task.config["reproduction"]["randomness_pair_key"]
                    for task in paired_tasks
                }
            )
            == 1
        )


def test_publishable_current_control_suppresses_unaccounted_diagnostics(tmp_path):
    tasks = expand_tasks(
        _document("decisive_stage4_long_horizon_n25.yaml"), output_root=tmp_path
    )
    current = [task for task in tasks if task.method_id == "dp_far_current_matched"]
    assert current
    assert all(
        task.config["training"]["algo_config"]["suppress_private_client_diagnostics"]
        is True
        for task in current
    )


def test_local_clip_calibration_is_an_explicit_non_private_oracle(tmp_path):
    tasks = expand_tasks(
        _document("decisive_stage0_local_clip_calibration.yaml"),
        output_root=tmp_path,
    )
    assert {task.config["training"]["algo_config"]["clip_norm"] for task in tasks} == {
        1.0,
        2.0,
        4.0,
        8.0,
    }
    assert {task.config["training"]["algo_config"]["enable_dp"] for task in tasks} == {
        False,
        True,
    }
    for task in tasks:
        algo = task.config["training"]["algo_config"]
        assert task.method_id == "dt_ldp_far_oracle"
        assert algo["enable_private_client_oracle_diagnostics"] is True
        assert algo["non_private_diagnostic_transcript"] is True
        assert algo["tilt_tau"] == 0.0
        assert algo["server_clip_norm"] == 10.0


def test_n25_decisive_tasks_use_public_capacity_and_full_participation(tmp_path):
    tasks = expand_tasks(
        _document("decisive_stage2_confirmatory_n25.yaml"), output_root=tmp_path
    )
    assert tasks
    for task in tasks:
        assert task.config["clients"]["num_clients"] == 25
        assert task.config["clients"]["min_clients"] == 25
        assert task.config["clients"]["sample_fraction"] == 1.0
        algo = task.config["training"]["algo_config"]
        assert algo["expected_num_clients"] == 25
        assert algo["privacy_public_dataset_size"] == 2401


def test_uncertified_alpha_stress_is_explicit_and_does_not_claim_cap(tmp_path):
    tasks = expand_tasks(
        _document("decisive_stage1_alpha_delay_n25.yaml"), output_root=tmp_path
    )
    stressed = [
        task
        for task in tasks
        if task.method_id == "dt_ldp_far" and "uncertified" in task.tilt_id
    ]
    assert len(stressed) == 2
    for task in stressed:
        algo = task.config["training"]["algo_config"]
        n = task.config["clients"]["num_clients"]
        assert algo["tilt_tau"] > tilt_tau_max(n, algo["kappa_w"])
        assert algo["tilt_bound_policy"] == "diagnostic_only"
        assert (
            task.config["reproduction"]["tilt_influence_certificate_claimed"] is False
        )
        assert (
            task.config["reproduction"][
                "uncertified_tilt_is_local_dp_postprocessing_diagnostic"
            ]
            is True
        )


def test_oracle_geometry_gate_is_explicitly_non_private_transcript(tmp_path):
    tasks = expand_tasks(
        _document("decisive_stage0_geometry.yaml"), output_root=tmp_path
    )
    oracle = next(task for task in tasks if task.method_id == "dt_ldp_far_oracle")
    algo = oracle.config["training"]["algo_config"]
    assert algo["enable_private_client_oracle_diagnostics"] is True
    assert algo["non_private_diagnostic_transcript"] is True
    assert (
        oracle.config["reproduction"][
            "private_client_oracles_make_run_non_private_diagnostic"
        ]
        is True
    )


def test_resolved_dt_task_enforces_causal_cap_and_full_participation(tmp_path):
    tasks = expand_tasks(_document("pilot_e1_e8.yaml"), output_root=tmp_path)
    task = next(task for task in tasks if task.method_id == "dt_ldp_far")
    config = task.config
    algo = config["training"]["algo_config"]
    clients = config["clients"]
    n = clients["num_clients"]
    assert algo["tilt_delay_rounds"] >= 1
    assert algo["require_full_participation"] is True
    assert algo["expected_num_clients"] == n
    assert clients["sample_fraction"] == 1.0
    assert clients["min_clients"] == n
    assert clients["dropout_rate"] == 0.0
    assert algo["tilt_tau"] <= tilt_tau_max(n, algo["kappa_w"])
    assert algo["privacy_num_rounds"] == config["training"]["num_rounds"]


def test_private_dt_tasks_use_add_remove_and_genuine_poisson_sampling(tmp_path):
    tasks = expand_tasks(_document("full_e1_e8.yaml"), output_root=tmp_path)
    private_dt = [
        task
        for task in tasks
        if task.config["training"]["algorithm"] == "dt_ldp_far"
        and task.config["training"]["algo_config"].get("enable_dp", True)
    ]
    assert private_dt
    for task in private_dt:
        algo = task.config["training"]["algo_config"]
        assert algo["sampling_scheme"] == "poisson"
        assert algo["privacy_adjacency"] == "add_remove"
        assert math.isclose(algo["privacy_sampling_rate_override"], 0.05)
        assert algo["privacy_public_dataset_size"] > 0
        if task.experiment_id != "E7_poisson_steps":
            assert algo["local_epochs"] == 5


def test_stage9_uses_fixed_without_replacement_replace_one_contract(tmp_path):
    tasks = expand_tasks(
        _document("decisive_stage9_fixed_without_replacement_n25.yaml"),
        output_root=tmp_path,
    )
    assert len(tasks) == 36
    groups = {}
    for task in tasks:
        algo = task.config["training"]["algo_config"]
        assert algo["sampling_scheme"] == "fixed_without_replacement"
        assert algo["privacy_adjacency"] == "replace_one"
        assert algo["privacy_public_dataset_size"] == 2400
        assert algo["fixed_batch_size"] == 120
        assert math.isclose(algo["privacy_sampling_rate_override"], 0.05)
        assert task.config["reproduction"]["local_steps_semantics"].startswith(
            "local_epochs axis denotes independent fixed-size WOR"
        )
        key = (
            task.reference_id,
            task.threat_id,
            task.partition_seed,
            task.training_seed,
        )
        groups.setdefault(key, []).append(task)
    assert len(groups) == 18
    for pair in groups.values():
        assert {task.method_id for task in pair} == {
            "dp_far_current_debiased_distance_oracle",
            "dt_ldp_far_debiased_distance_oracle",
        }
        assert (
            len({task.config["reproduction"]["randomness_pair_key"] for task in pair})
            == 1
        )


def test_private_client_oracles_are_confined_to_e2_diagnostics(tmp_path):
    tasks = expand_tasks(_document("full_e1_e8.yaml"), output_root=tmp_path)
    oracle_tasks = [
        task
        for task in tasks
        if task.config["training"]["algo_config"].get(
            "enable_private_client_oracle_diagnostics", False
        )
    ]
    assert oracle_tasks
    assert {task.experiment_id for task in oracle_tasks} == {"E2_noise_decoupling"}
    private_claim_tasks = [
        task
        for task in tasks
        if task.method_id == "dt_ldp_far"
        and task.experiment_id != "E2_noise_decoupling"
    ]
    assert private_claim_tasks
    assert all(
        not task.config["training"]["algo_config"].get(
            "enable_private_client_oracle_diagnostics", False
        )
        for task in private_claim_tasks
    )


def test_tilt_profiles_resolve_to_declared_fraction(tmp_path):
    tasks = expand_tasks(_document("full_e1_e8.yaml"), output_root=tmp_path)
    task = next(task for task in tasks if task.method_id == "dt_ldp_far")
    algo = task.config["training"]["algo_config"]
    maximum = tilt_tau_max(task.config["clients"]["num_clients"], algo["kappa_w"])
    assert math.isclose(
        algo["tilt_tau"], algo["tilt_tau_fraction_of_max"] * maximum, rel_tol=1e-12
    )


def test_geometry_profile_is_part_of_identity_and_resolved_config(tmp_path):
    tasks = expand_tasks(_document("pilot_e1_e8.yaml"), output_root=tmp_path)
    calibration = [
        task for task in tasks if task.experiment_id == "E0_public_geometry_calibration"
    ]
    assert len(calibration) == 6
    assert {task.geometry_id for task in calibration} == {
        "calibrated_d02",
        "calibrated_d05",
        "calibrated_d10",
    }
    for task in calibration:
        algo = task.config["training"]["algo_config"]
        assert task.geometry_id in task.task_id
        assert algo["experiment_geometry_profile"] == task.geometry_id
        assert algo["distance_clip"] in {0.02, 0.05, 0.10}


def test_no_noise_clipping_matched_lane_uses_one_poisson_training_path(tmp_path):
    tasks = expand_tasks(_document("pilot_e1_e8.yaml"), output_root=tmp_path)
    matched = [
        task for task in tasks if task.experiment_id == "E1_clipping_matched_no_noise"
    ]
    assert {task.config["training"]["algorithm"] for task in matched} == {
        "dpfedavg",
        "dpfar",
        "dt_ldp_far",
    }
    for task in matched:
        algo = task.config["training"]["algo_config"]
        assert algo["enable_dp"] is False
        assert algo["sampling_scheme"] == "poisson"
        assert algo["local_epochs"] == 5
        assert task.config["reproduction"]["comparison_lane"] == (
            "poisson_no_noise_clipping_matched"
        )


def test_fedfdp_lanes_are_not_mislabeled_as_common_poisson(tmp_path):
    tasks = expand_tasks(_document("pilot_e1_e8.yaml"), output_root=tmp_path)
    fedfdp = [
        task for task in tasks if task.config["training"]["algorithm"] == "fedfdp"
    ]
    assert len(fedfdp) == 2
    assert {task.experiment_id for task in fedfdp} == {
        "E8_fedfdp_native",
        "E8_fedfdp_compute_matched",
    }
    for task in fedfdp:
        algo = task.config["training"]["algo_config"]
        assert algo["sampling_scheme"] == "fixed_minibatch"
        assert algo["privacy_sampling_rate_override"] is None
        assert task.config["reproduction"]["privacy_label_policy"] == (
            "report_realised_epsilon_not_profile_name"
        )


def test_completion_requires_every_declared_round(tmp_path):
    task = expand_tasks(_document("pilot_e1_e8.yaml"), output_root=tmp_path)[0]
    metrics_dir = task.output_dir / "nested"
    metrics_dir.mkdir(parents=True)
    rounds = task.config["training"]["num_rounds"]
    (metrics_dir / "metrics.json").write_text(
        json.dumps({"rounds": [{}] * (rounds - 1)})
    )
    assert is_complete(task) is False
    (metrics_dir / "metrics.json").write_text(json.dumps({"rounds": [{}] * rounds}))
    assert is_complete(task) is True


def test_dry_run_writes_one_resolved_config_without_training(tmp_path, capsys):
    output = tmp_path / "output"
    return_code = main(
        [
            "--dry-run",
            "--matrix",
            str(CONFIG_ROOT / "pilot_e1_e8.yaml"),
            "--job-index",
            "0",
            "--output-root",
            str(output),
            "--data-root",
            str(tmp_path / "data"),
        ]
    )
    assert return_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task_index"] == 0
    assert Path(payload["resolved_config"]).exists()
    assert not list(output.glob("**/metrics.json"))


def test_short_run_mode_updates_training_and_accounting_horizon(tmp_path):
    tasks = expand_tasks(
        _document("pilot_e1_e8.yaml"), output_root=tmp_path, pilot_rounds=2
    )
    for task in tasks:
        assert task.config["training"]["num_rounds"] == 2
        assert task.config["training"]["algo_config"]["privacy_num_rounds"] == 2
        assert "not paper evidence" in task.config["reproduction"]["algorithmic_scope"]
