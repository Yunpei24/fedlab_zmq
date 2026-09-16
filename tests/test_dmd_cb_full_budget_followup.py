"""Non-training tests for the frozen factorial design and fail-closed gates."""

import copy
import json
import math
from pathlib import Path

import pytest

from scripts import run_dmd_cb_full_budget_followup as runner
from scripts.dmd_cb_followup_gate import statistics_for, check_rules


def groups(n=3):
    out = {}
    for noise in ("homogeneous", "heteroscedastic"):
        for mode in ("uniform", "direct_rfa", "far_rfa"):
            out[f"{noise}/{mode}"] = {
                key: statistics_for([value] * n)
                for key, value in {
                    "accuracy": 0.1,
                    "balanced_accuracy": 0.1,
                    "worst20": 0.4,
                    "gap": -0.1,
                    "loss": 0.0,
                    "variance": -1.0,
                }.items()
            }
    return out


def test_counts_unique_and_seed_separation():
    matrix = runner.load_matrix()
    for phase, count in zip(runner.PHASES, (18, 48, 96)):
        tasks = runner.tasks(matrix, phase)
        assert len(tasks) == count and len({x["id"] for x in tasks}) == count
    assert not set(matrix["phases"]["confirmation"]["seeds"]) & set(
        matrix["phases"]["ce_full_budget_screen"]["seeds"]
    )


def test_actual_pure_ce_control_and_unchanged_dmd_budget(tmp_path):
    m = runner.load_matrix()
    for phase in runner.PHASES:
        for task in runner.tasks(m, phase):
            cfg = runner.resolved(m, task, tmp_path / task["id"])
            a = cfg["training"]["algo_config"]
            dmd = task["objective"] == "dmd"
            assert a["dmd_histogram_epsilon"] == (0.25 if dmd else 0)
            assert a["target_epsilon"] == (3.75 if dmd else 4)
            assert a["privacy_num_rounds"] == cfg["training"]["num_rounds"] == 40
            assert a["far_alpha"] == (0.1 if task["mode"] == "far_rfa" else 0)
            assert a["dmd_pairing_seed"] == cfg["seed"] == cfg["data"]["partition_seed"]
            assert a["sampling_scheme"] == "fixed_without_replacement"
            assert a["privacy_adjacency"] == "replace_one"
            assert cfg["device"] == "mps"
            if phase == "attacks":
                assert (
                    a["attack"]["active_round_start"] == 17
                    and a["attack"]["active_round_end"] == 40
                )
                assert a["attack"]["client_ids"] == [0, 1, 2, 3, 4]
    assert runner.sigma("ce") < runner.sigma("dmd")
    assert runner.pilot.gradient_epsilon(40, runner.sigma("ce")) <= 4
    assert runner.pilot.gradient_epsilon(40, runner.sigma("dmd")) <= 3.75


def test_screen_can_pass_but_never_selects_a_lucky_far_arm():
    cfg = runner.load_matrix()["gates"]["screen"]
    data = groups()
    assert all(c["passed"] for c in check_rules(data, cfg, confirmation=False))
    data["heteroscedastic/uniform"]["worst20"] = statistics_for([0.0, 0.0, 0.0])
    data["heteroscedastic/far_rfa"]["worst20"] = statistics_for([10.0, 10.0, 10.0])
    assert not all(c["passed"] for c in check_rules(data, cfg, confirmation=False))


@pytest.mark.parametrize(
    "metric,values",
    [
        ("accuracy", [-0.3] * 3),
        ("balanced_accuracy", [-0.3] * 3),
        ("gap", [0.51] * 3),
        ("loss", [0.011] * 3),
        ("worst20", [1.0, -0.01, -0.01]),
    ],
)
def test_screen_conjunction_rejects_a_single_failed_criterion(metric, values):
    cfg = runner.load_matrix()["gates"]["screen"]
    data = groups()
    data["heteroscedastic/uniform"][metric] = statistics_for(values)
    assert not all(c["passed"] for c in check_rules(data, cfg, confirmation=False))


def test_confirmation_requires_both_uniform_and_direct_rfa():
    cfg = runner.load_matrix()["gates"]["confirmation"]
    data = groups(4)
    assert all(c["passed"] for c in check_rules(data, cfg, confirmation=True))
    data["heteroscedastic/direct_rfa"]["worst20"] = statistics_for(
        [0.8, 0.8, -0.8, -0.8]
    )
    assert not all(c["passed"] for c in check_rules(data, cfg, confirmation=True))


def test_gate_never_treats_rounds_as_seeds():
    for values in ([1, 2], [1] * 40, [1, math.nan, 2]):
        with pytest.raises(ValueError):
            statistics_for(values)
    result = statistics_for([1, 2, 3, 4])
    assert result["ci95_low"] < 2.5 < result["ci95_high"]


def test_status_does_not_create_results(tmp_path):
    m = runner.load_matrix()
    out = tmp_path / "absent"
    report = runner.status(out, m, {}, runner.PHASES[0])
    assert report["complete_valid"] == 0 and len(report["missing"]) == 18
    assert not out.exists()


def test_partially_created_run_is_invalid_not_missing(tmp_path):
    m = runner.load_matrix()
    phase = runner.PHASES[0]
    task = runner.tasks(m, phase)[0]
    (tmp_path / phase / "runs" / task["id"]).mkdir(parents=True)
    report = runner.status(tmp_path, m, {}, phase)
    assert len(report["invalid"]) == 1 and len(report["missing"]) == 17


def fake_recorded_run(tmp_path, scenario):
    m = runner.load_matrix()
    task = runner.tasks(m, "attacks")[0]
    task = {**task, "scenario": scenario}
    cfg = runner.resolved(m, task, tmp_path)
    a = cfg["training"]["algo_config"]
    rows = []
    for t in range(1, 41):
        active = t >= 17
        eps = runner.pilot.gradient_epsilon(t, a["noise_multiplier"])
        rows.append(
            {
                "round_num": t,
                "num_clients": 25,
                "num_selected": 25,
                "num_survivors": 25,
                "num_alive_clients": 25,
                "ldp_gradient_far_private_gradient_mps_fraction": 1.0,
                "ldp_gradient_far_private_compute_device": "mps",
                "privacy_sampling_scheme": "fixed_without_replacement",
                "privacy_adjacency": "replace_one",
                "dmd_server_mode": a["dmd_server_mode"],
                "dmd_mu": 0.0,
                "privacy_dmd_histogram_calls_per_client_max": 0,
                "privacy_dmd_histogram_calls_this_round": 0,
                "evaluated_client_ids_oracle": list(range(5, 25)),
                "attack_window_active": active,
                "attack_enabled": active,
                "attack_schedule_phase": "attack" if active else "clean",
                "attack_name": scenario if active else "none",
                "num_byzantine_oracle": 5 if active else 0,
                "byzantine_fraction_oracle": 0.2 if active else 0.0,
                "attack_scheduled_name": scenario,
                "far_attack_labels_visible_to_server_aggregate": False,
                "far_attack_config_visible_to_server_aggregate": False,
                "privacy_dmd_budget_composition": "gradient_rdp_only_no_histogram",
                "privacy_epsilon_max": eps,
                "privacy_epsilon_mean": eps,
                "privacy_gradient_epsilon_max": eps,
                "privacy_gradient_epsilon_mean": eps,
                "privacy_delta": 1e-5,
                "privacy_dmd_histogram_epsilon": 0.0,
                "privacy_gradient_target_epsilon": 4.0,
                "privacy_target_epsilon": 4.0,
                "privacy_model_noise_multiplier_min": a["noise_multiplier"],
                "privacy_model_noise_multiplier_max": a["noise_multiplier"],
                "test_accuracy": 0.5,
                "test_loss": 1.0,
                "client_accuracy_mean": 0.5,
                "mean_client_balanced_accuracy_pct": 50.0,
                "worst20_accuracy_pct": 25.0,
                "best20_worst20_gap_pct": 40.0,
                "client_accuracy_variance_pct2": 100.0,
            }
        )
    payload = {
        "algorithm": cfg["training"]["algorithm"],
        "config": {**a, "device": "mps"},
        "summary": {
            "num_rounds": 40,
            "num_clients": 25,
            "seed": cfg["seed"],
            "partition_seed": cfg["seed"],
            "dataset": "fashionmnist",
            "model": "lenet5_tanh",
        },
        "rounds": rows,
    }
    (tmp_path / "metrics.json").write_text(json.dumps(payload))
    (tmp_path / "runtime_imports.json").write_text(
        json.dumps(
            {
                "stage": "after_training",
                "device": "mps",
                "mps_fallback": 0,
                "algorithm": payload["algorithm"],
                "source_sha256": {"example.py": "abc"},
            }
        )
    )
    return cfg, payload, {"source_sha256": {"example.py": "abc"}}


@pytest.mark.parametrize("scenario", ["bf", "ipm", "alie"])
def test_valid_attack_schedule_and_server_boundary(tmp_path, scenario):
    cfg, payload, stamp = fake_recorded_run(tmp_path, scenario)
    runner.validate_result(tmp_path, cfg, stamp)


@pytest.mark.parametrize(
    "key,value",
    [
        ("attack_window_active", False),
        ("num_byzantine_oracle", 0),
        ("attack_name", "none"),
        ("attack_schedule_phase", "clean"),
        ("byzantine_fraction_oracle", 0.0),
        ("far_attack_labels_visible_to_server_aggregate", True),
        ("far_attack_config_visible_to_server_aggregate", True),
        ("privacy_dmd_histogram_calls_per_client_max", 1),
        ("ldp_gradient_far_private_compute_device", "cpu"),
    ],
)
def test_validation_rejects_silent_attack_or_privacy_failure(tmp_path, key, value):
    cfg, payload, stamp = fake_recorded_run(tmp_path, "bf")
    payload["rounds"][16][key] = value
    (tmp_path / "metrics.json").write_text(json.dumps(payload))
    with pytest.raises(RuntimeError):
        runner.validate_result(tmp_path, cfg, stamp)
