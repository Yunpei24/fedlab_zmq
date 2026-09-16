from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_ldp_gradient_far_positioning import (
    DEFAULT_MATRIX,
    approve_phase,
    gate_decision,
    load_campaign,
    resolved_config,
    task_output_dir,
    tasks_for_phase,
    validate_requested_device,
)


def test_positioning_matrix_is_gated_paired_and_bounded():
    campaign = load_campaign(DEFAULT_MATRIX)
    counts = {
        phase["id"]: len(tasks_for_phase(campaign, phase["id"]))
        for phase in campaign.phases
    }
    assert counts == {
        "a_activation_screen": 8,
        "b_local_clip_screen": 6,
        "c_alpha_reference_screen": 56,
        "d_privacy_screen": 24,
        "d2_noise_heterogeneity_screen": 16,
        "e_byzantine_screen": 48,
        "e2_far_stealth_attack_screen": 16,
        "f_confirmation": 120,
    }
    assert len(campaign.tasks) == 294
    dev = set(campaign.matrix["randomness"]["development_seeds"])
    confirm = set(campaign.matrix["randomness"]["confirmation_seeds"])
    assert dev.isdisjoint(confirm)
    assert {task.seed for task in campaign.tasks if task.phase_role == "confirmation"} == confirm
    assert {task.seed for task in campaign.tasks if task.phase_role != "confirmation"} == dev


def test_positioning_matrix_covers_registered_scientific_factors():
    campaign = load_campaign(DEFAULT_MATRIX)
    configs = [resolved_config(campaign, task) for task in campaign.tasks]
    assert {cfg["data"]["dataset"] for cfg in configs} == {"mnist", "fashionmnist"}
    assert {cfg["model"]["architecture"] for cfg in configs} == {
        "lenet5_tanh",
        "lenet5_relu",
    }
    assert {cfg["clients"]["num_clients"] for cfg in configs} == {10, 25}
    assert {cfg["training"]["algo_config"]["clip_norm"] for cfg in configs} == {
        4.0,
        8.0,
        16.0,
    }
    assert {cfg["training"]["algo_config"]["far_alpha"] for cfg in configs} == {
        -5.0,
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        5.0,
    }
    assert {cfg["training"]["algo_config"]["robust_reference"] for cfg in configs} == {
        "coordinate_median",
        "trimmed_mean",
        "rfa",
        "centered_clipping",
    }
    assert {cfg["training"]["algo_config"]["attack"]["name"] for cfg in configs} == {
        "none",
        "bf",
        "ipm",
        "alie",
        "minmax",
        "minsum",
    }
    dp_eps = {
        float(cfg["training"]["algo_config"]["target_epsilon"])
        for cfg in configs
        if cfg["training"]["algo_config"]["enable_dp"]
    }
    assert dp_eps == {2.0, 4.0, 8.0}
    assert any(not cfg["training"]["algo_config"]["enable_dp"] for cfg in configs)

    heteroscedastic = [
        resolved_config(campaign, task)
        for task in tasks_for_phase(campaign, "d2_noise_heterogeneity_screen")
    ]
    assert len(heteroscedastic) == 16
    for cfg in heteroscedastic:
        algo = cfg["training"]["algo_config"]
        scales = algo["privacy_noise_multiplier_scale_by_client"]
        assert len(scales) == cfg["clients"]["num_clients"]
        assert min(scales) == 1
        assert max(scales) in {1.5, 2}
        assert algo["target_epsilon"] == 4.0
        assert algo["enable_oracle_diagnostics"] is True
    assert all(
        cfg["training"]["algo_config"].get("enable_oracle_diagnostics") is True
        for cfg in configs
        if cfg["training"]["algo_config"]["enable_dp"]
    )


def test_n10_n25_public_cardinality_and_attack_ids_are_consistent():
    campaign = load_campaign(DEFAULT_MATRIX)
    for task in campaign.tasks:
        config = resolved_config(campaign, task)
        algo = config["training"]["algo_config"]
        n = config["clients"]["num_clients"]
        assert algo["privacy_public_dataset_size"] == 60_000 // n
        assert algo["fixed_batch_size"] == int(0.05 * (60_000 // n))
        assert algo["batch_size"] == algo["fixed_batch_size"]
        assert algo["privacy_num_rounds"] == config["training"]["num_rounds"]
        if algo["attack"]["enabled"]:
            expected_f = 2 if n == 10 else 5
            assert algo["attack"]["client_ids"] == list(range(expected_f))


def test_stealth_attack_screen_is_theory_bridge_with_oracle_diagnostics():
    campaign = load_campaign(DEFAULT_MATRIX)
    configs = [
        resolved_config(campaign, task)
        for task in tasks_for_phase(campaign, "e2_far_stealth_attack_screen")
    ]
    assert len(configs) == 16
    assert {cfg["training"]["algo_config"]["attack"]["name"] for cfg in configs} == {
        "minmax",
        "minsum",
    }
    assert {cfg["clients"]["num_clients"] for cfg in configs} == {10, 25}
    assert {cfg["training"]["algo_config"]["far_alpha"] for cfg in configs} == {
        -2.0,
        2.0,
    }
    assert {
        cfg["training"]["algo_config"]["robust_reference"] for cfg in configs
    } == {"centered_clipping", "rfa"}
    assert all(
        cfg["training"]["algo_config"]["target_epsilon"] == 4.0
        and cfg["training"]["algo_config"]["enable_oracle_diagnostics"] is True
        for cfg in configs
    )


def test_runner_refuses_non_mps_devices():
    validate_requested_device("mps")
    with pytest.raises(ValueError, match="MPS-only"):
        validate_requested_device("cpu")
    with pytest.raises(ValueError, match="MPS-only"):
        validate_requested_device("cuda")


def test_gate_can_only_be_recorded_after_all_tasks_complete(tmp_path):
    campaign = load_campaign(DEFAULT_MATRIX)
    campaign = type(campaign)(
        matrix_path=campaign.matrix_path,
        matrix=campaign.matrix,
        base=campaign.base,
        output_root=tmp_path,
        tasks=campaign.tasks,
    )
    phase = "a_activation_screen"
    with pytest.raises(RuntimeError, match="incomplete"):
        approve_phase(campaign, phase, "promote", "premature")

    for task in tasks_for_phase(campaign, phase):
        output = task_output_dir(campaign, task)
        output.mkdir(parents=True)
        rounds = resolved_config(campaign, task)["training"]["num_rounds"]
        payload = {
            "summary": {"num_rounds": rounds},
            "rounds": [{} for _ in range(rounds)],
        }
        (output / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    path = approve_phase(campaign, phase, "promote", "Tanh retained a priori")
    assert path.exists()
    assert gate_decision(campaign, phase) == "promote"
