from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.run_ldp_gradient_far_positioning import (
    approve_phase,
    campaign_config_hash,
    gate_decision,
    load_campaign,
    phase_config_hash,
    resolved_config,
    task_output_dir,
    tasks_for_phase,
)


ROOT = Path(__file__).resolve().parents[1]
V2_MATRIX = ROOT / "configs" / "ldp_gradient_far" / "positioning_v2.yaml"


def _configs(campaign, phase):
    return [resolved_config(campaign, task) for task in tasks_for_phase(campaign, phase)]


def _complete_phase(campaign, phase):
    for task in tasks_for_phase(campaign, phase):
        rounds = int(resolved_config(campaign, task)["training"]["num_rounds"])
        output = task_output_dir(campaign, task)
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(
            json.dumps(
                {
                    "summary": {"num_rounds": rounds},
                    "rounds": [{} for _ in range(rounds)],
                }
            ),
            encoding="utf-8",
        )


def test_v2_matrix_has_locked_380_task_sequential_design():
    campaign = load_campaign(V2_MATRIX)
    counts = {
        phase["id"]: len(tasks_for_phase(campaign, phase["id"]))
        for phase in campaign.phases
    }
    assert counts == campaign.matrix["expected_task_counts"] == {
        "b2_local_clip_dp_screen": 6,
        "d_privacy_screen_v2": 24,
        "d2_noise_assignment_dev": 16,
        "d2_noise_assignment_confirmation": 48,
        "e_byzantine_identification_screen": 70,
        "f_confirmation_t20": 180,
        "f2_horizon_extension_t40": 36,
    }
    assert len(campaign.tasks) == 380
    assert "a_activation_screen" not in counts
    assert "c_alpha_reference_screen" not in counts
    assert campaign.phases[0]["depends_on"] == []
    assert campaign.matrix["inherited_lock"]["metric_reuse"] == "none"


def test_v2_reaudits_local_clip_inside_epsilon_four_mechanism():
    campaign = load_campaign(V2_MATRIX)
    configs = _configs(campaign, "b2_local_clip_dp_screen")
    assert len(configs) == 6
    assert {cfg["clients"]["num_clients"] for cfg in configs} == {10, 25}
    assert {cfg["training"]["algo_config"]["clip_norm"] for cfg in configs} == {
        4.0,
        8.0,
        16.0,
    }
    for cfg in configs:
        algo = cfg["training"]["algo_config"]
        assert algo["enable_dp"] is True
        assert algo["target_epsilon"] == 4.0
        assert algo["far_alpha"] == 0.0
        assert cfg["training"]["num_rounds"] == 20


def test_v2_noise_tiers_are_identity_reverse_and_multiseed_confirmed():
    campaign = load_campaign(V2_MATRIX)
    dev = tasks_for_phase(campaign, "d2_noise_assignment_dev")
    confirm = tasks_for_phase(campaign, "d2_noise_assignment_confirmation")
    assert {task.seed for task in dev} == {137}
    assert {task.seed for task in confirm} == {28, 36, 54}

    for tasks in (dev, confirm):
        lookup = {task.run_id: resolved_config(campaign, task) for task in tasks}
        for task in tasks:
            if "_identity__" not in task.run_id:
                continue
            reverse_id = task.run_id.replace("_identity__", "_reverse__")
            assert reverse_id in lookup
            identity = task.overrides["training"]["algo_config"][
                "privacy_noise_multiplier_scale_by_client"
            ]
            reverse_task = next(item for item in tasks if item.run_id == reverse_id)
            reverse_scales = reverse_task.overrides["training"]["algo_config"][
                "privacy_noise_multiplier_scale_by_client"
            ]
            assert list(reversed(identity)) == reverse_scales
            assert sorted(identity) == sorted(reverse_scales)


def test_v2_attack_phase_identifies_attack_dp_and_tilting_effects():
    campaign = load_campaign(V2_MATRIX)
    tasks = tasks_for_phase(campaign, "e_byzantine_identification_screen")
    configs = [resolved_config(campaign, task) for task in tasks]
    assert len(configs) == 70
    assert {cfg["training"]["algo_config"]["attack"]["name"] for cfg in configs} == {
        "none",
        "bf",
        "ipm",
        "alie",
        "minmax",
    }
    assert {cfg["training"]["algo_config"]["robust_reference"] for cfg in configs} == {
        "coordinate_median",
        "trimmed_mean",
        "rfa",
        "centered_clipping",
    }
    assert any(not cfg["training"]["algo_config"]["enable_dp"] for cfg in configs)
    assert any(cfg["training"]["algo_config"]["far_alpha"] == 0.0 for cfg in configs)
    # Every attack/n pair contains a DP-vs-no-DP alpha-zero control and a
    # DP-vs-no-DP positive-FAR FCC control.
    for n in (10, 25):
        for attack in ("none", "bf", "ipm", "alie", "minmax"):
            cells = [
                cfg["training"]["algo_config"]
                for cfg in configs
                if cfg["clients"]["num_clients"] == n
                and cfg["training"]["algo_config"]["attack"]["name"] == attack
            ]
            signatures = {
                (
                    bool(algo["enable_dp"]),
                    float(algo["far_alpha"]),
                    str(algo["robust_reference"]),
                )
                for algo in cells
            }
            assert (True, 0.0, "centered_clipping") in signatures
            assert (False, 0.0, "centered_clipping") in signatures
            assert (True, 2.0, "centered_clipping") in signatures
            assert (False, 2.0, "centered_clipping") in signatures


def test_v2_primary_confirmation_is_t20_and_extension_is_separate_t40():
    campaign = load_campaign(V2_MATRIX)
    primary = _configs(campaign, "f_confirmation_t20")
    extension = _configs(campaign, "f2_horizon_extension_t40")
    assert len(primary) == 180 and {
        cfg["training"]["num_rounds"] for cfg in primary
    } == {20}
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
    assert len(extension) == 36 and {
        cfg["training"]["num_rounds"] for cfg in extension
    } == {40}


def test_v2_records_cross_n_cardinality_batch_confound_in_every_config():
    campaign = load_campaign(V2_MATRIX)
    for task in campaign.tasks:
        cfg = resolved_config(campaign, task)
        algo = cfg["training"]["algo_config"]
        n = cfg["clients"]["num_clients"]
        assert algo["positioning_global_train_size"] == 60_000
        assert algo["positioning_public_local_dataset_size"] == 60_000 // n
        assert algo["positioning_fixed_batch_size"] == (300 if n == 10 else 120)
        assert algo["positioning_fixed_batch_fraction"] == pytest.approx(0.05)
        assert "within_n" in algo["positioning_cross_n_scale_confound"]


def test_v2_gate_requires_numeric_evidence_hashes_config_and_refuses_overwrite(
    tmp_path,
):
    original = load_campaign(V2_MATRIX)
    campaign = replace(original, output_root=tmp_path / "results")
    phase = "b2_local_clip_dp_screen"
    _complete_phase(campaign, phase)

    with pytest.raises(ValueError, match="numeric evidence"):
        approve_phase(campaign, phase, "promote", "missing evidence")
    failing = {
        "complete_fraction": 1.0,
        "invalid_runs": 0,
        "max_abs_epsilon_error": 0.0,
        "c8_auc_gain_over_c4_pp": 0.1,
        "c8_auc_loss_vs_c16_pp": 0.1,
    }
    with pytest.raises(RuntimeError, match="criteria failed"):
        approve_phase(campaign, phase, "promote", "fails utility gate", failing)

    passing = {**failing, "c8_auc_gain_over_c4_pp": 0.8}
    path = approve_phase(campaign, phase, "promote", "registered evidence", passing)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["campaign_config_hash"] == campaign_config_hash(campaign)
    assert payload["phase_config_hash"] == phase_config_hash(campaign, phase)
    assert payload["immutable_record"] is True
    assert all(row["passed"] for row in payload["criterion_evaluations"])
    assert gate_decision(campaign, phase) == "promote"
    with pytest.raises(RuntimeError, match="cannot be overwritten"):
        approve_phase(campaign, phase, "stop", "second decision forbidden")

    altered_matrix = copy.deepcopy(campaign.matrix)
    altered_matrix["campaign_id"] = "changed_after_gate"
    stale = replace(campaign, matrix=altered_matrix)
    assert gate_decision(stale, phase) == "stale"
