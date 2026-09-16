from __future__ import annotations

from pathlib import Path

from scripts.run_ldp_gradient_far_positioning import (
    load_campaign,
    resolved_config,
    tasks_for_phase,
)


ROOT = Path(__file__).resolve().parents[1]
V3_MATRIX = ROOT / "configs" / "ldp_gradient_far" / "positioning_v3.yaml"


def _configs(campaign, phase_id):
    return [
        resolved_config(campaign, task)
        for task in tasks_for_phase(campaign, phase_id)
    ]


def test_v3_has_independent_calibration_seeds_and_locked_counts():
    campaign = load_campaign(V3_MATRIX)
    counts = {
        phase["id"]: len(tasks_for_phase(campaign, phase["id"]))
        for phase in campaign.phases
    }
    assert counts == campaign.matrix["expected_task_counts"] == {
        "b3_local_clip_dp_confirmation": 18,
        "d_privacy_screen_v3": 24,
        "d2_noise_assignment_dev": 16,
        "d2_noise_assignment_confirmation": 48,
        "e_byzantine_identification_screen": 70,
        "f_confirmation_t20": 180,
        "f2_horizon_extension_t40": 36,
    }
    assert len(campaign.tasks) == 392
    randomness = campaign.matrix["randomness"]
    calibration = set(randomness["calibration_seeds"])
    development = set(randomness["development_seeds"])
    confirmation = set(randomness["confirmation_seeds"])
    assert calibration == {71, 83, 101}
    assert calibration.isdisjoint(development | confirmation)
    assert development.isdisjoint(confirmation)
    calibration_tasks = tasks_for_phase(campaign, "b3_local_clip_dp_confirmation")
    assert {task.phase_role for task in calibration_tasks} == {"calibration"}
    assert {task.seed for task in calibration_tasks} == calibration


def test_v3_inherits_stopped_v2_decision_without_copying_metrics():
    campaign = load_campaign(V3_MATRIX)
    lock = campaign.matrix["inherited_lock"]
    assert lock["source_v2_gate_decision"] == "stop"
    assert lock["inherited_candidates"] == {"n10": 8.0, "n25": 4.0}
    assert lock["metric_reuse"] == "none"
    assert lock["runtime_dependency_on_v1_or_v2"] == "none"
    assert campaign.phases[0]["depends_on"] == []


def test_v3_calibration_gate_is_exactly_n_specific_noninferiority():
    campaign = load_campaign(V3_MATRIX)
    phase = campaign.phases[0]
    criteria = {item["id"]: item for item in phase["gate_criteria"]}
    assert criteria["invalid_runs"]["op"] == "=="
    assert criteria["invalid_runs"]["threshold"] == 0
    assert criteria["max_abs_epsilon_error"]["op"] == "<="
    assert criteria["max_abs_epsilon_error"]["threshold"] == 0.05
    assert criteria["n10_c8_auc_deficit_to_best_alternative_pp"] == {
        "id": "n10_c8_auc_deficit_to_best_alternative_pp",
        "op": "<=",
        "threshold": 0.5,
        "description": "At n=10 C=8 is non-inferior to the best of C=4/C=16.",
    }
    assert criteria["n25_c4_auc_deficit_to_best_alternative_pp"] == {
        "id": "n25_c4_auc_deficit_to_best_alternative_pp",
        "op": "<=",
        "threshold": 0.5,
        "description": "At n=25 C=4 is non-inferior to the best of C=8/C=16.",
    }


def test_v3_calibration_tests_every_c_for_each_n_at_epsilon_four():
    campaign = load_campaign(V3_MATRIX)
    configs = _configs(campaign, "b3_local_clip_dp_confirmation")
    for n in (10, 25):
        subset = [cfg for cfg in configs if cfg["clients"]["num_clients"] == n]
        assert len(subset) == 9
        assert {cfg["training"]["algo_config"]["clip_norm"] for cfg in subset} == {
            4.0,
            8.0,
            16.0,
        }
        for cfg in subset:
            algo = cfg["training"]["algo_config"]
            assert algo["enable_dp"] is True
            assert algo["target_epsilon"] == 4.0
            assert algo["far_alpha"] == 0.0


def test_v3_all_post_gate_phases_use_n_specific_c_without_common_override():
    campaign = load_campaign(V3_MATRIX)
    post_gate_phases = [phase["id"] for phase in campaign.phases[1:]]
    for phase in campaign.phases[1:]:
        common_algo = (
            phase.get("common_overrides", {})
            .get("training", {})
            .get("algo_config", {})
        )
        assert "clip_norm" not in common_algo
    for phase_id in post_gate_phases:
        for cfg in _configs(campaign, phase_id):
            n = cfg["clients"]["num_clients"]
            expected_c = 8.0 if n == 10 else 4.0
            assert cfg["training"]["algo_config"]["clip_norm"] == expected_c


def test_v3_d2_integrated_axes_preserve_c_under_identity_and_reverse():
    campaign = load_campaign(V3_MATRIX)
    for phase_id in (
        "d2_noise_assignment_dev",
        "d2_noise_assignment_confirmation",
    ):
        tasks = tasks_for_phase(campaign, phase_id)
        lookup = {task.run_id: task for task in tasks}
        for task in tasks:
            cfg = resolved_config(campaign, task)
            n = cfg["clients"]["num_clients"]
            algo = cfg["training"]["algo_config"]
            assert algo["clip_norm"] == (8.0 if n == 10 else 4.0)
            if "_identity__" not in task.run_id:
                continue
            reverse_id = task.run_id.replace("_identity__", "_reverse__")
            reverse = resolved_config(campaign, lookup[reverse_id])
            identity_scales = algo["privacy_noise_multiplier_scale_by_client"]
            reverse_scales = reverse["training"]["algo_config"][
                "privacy_noise_multiplier_scale_by_client"
            ]
            assert list(reversed(identity_scales)) == reverse_scales


def test_v3_confirmation_keeps_four_references_ipm_and_t20_t40_split():
    campaign = load_campaign(V3_MATRIX)
    primary = _configs(campaign, "f_confirmation_t20")
    extension = _configs(campaign, "f2_horizon_extension_t40")
    assert len(primary) == 180
    assert {cfg["training"]["num_rounds"] for cfg in primary} == {20}
    assert {cfg["training"]["algo_config"]["attack"]["name"] for cfg in primary} == {
        "none",
        "bf",
        "ipm",
        "alie",
        "minmax",
    }
    assert {cfg["training"]["algo_config"]["robust_reference"] for cfg in primary} == {
        "coordinate_median",
        "trimmed_mean",
        "rfa",
        "centered_clipping",
    }
    assert len(extension) == 36
    assert {cfg["training"]["num_rounds"] for cfg in extension} == {40}
