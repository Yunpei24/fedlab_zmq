from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import scripts.run_rcig_ldp_gradient_far_v2 as rcig_v2_runner

from scripts.run_rcig_ldp_gradient_far_v2 import (
    DEFAULT_MATRIX,
    _campaign_lock_payload,
    _exclusive_task_lock,
    _file_sha256,
    _post_run_device_audit,
    campaign_scientific_hash,
    ensure_campaign_lock,
    load_campaign,
    record_gate,
    resolved_config,
    run_task,
    task_is_complete,
    task_output_dir,
    tasks_for_phase,
    validate_requested_device,
)


def _campaign_at(tmp_path: Path):
    return replace(load_campaign(DEFAULT_MATRIX), output_root=tmp_path / "results")


def _write_calibration_gate(campaign) -> Path:
    return record_gate(
        campaign,
        "r0_dynamic_calibration",
        {
            "complete_fraction": 1.0,
            "invalid_runs": 0.0,
            "rcig_private_gradient_mps_fraction": 1.0,
            "server_cpu_float64_postprocess_fraction": 1.0,
            "strict_past_fraction": 1.0,
            "covariance_psd_fraction": 1.0,
            "nominal_covariance_provenance_fraction": 1.0,
            "finite_threshold_fraction": 1.0,
            "calibration_blocks_per_cell_min": 53.0,
            "simultaneous_distribution_free_confidence_lower": 1.0 - 6.0 * (0.90**53),
            "thresholds_by_regime_and_mode": {
                "homogeneous": {
                    "full": 3.0,
                    "isotropic": 2.0,
                    "euclidean": 1.0,
                },
                "heteroscedastic": {
                    "full": 6.0,
                    "isotropic": 5.0,
                    "euclidean": 4.0,
                },
            },
        },
    )


def test_v2_matrix_has_exact_counts_and_disjoint_seed_registries() -> None:
    campaign = load_campaign(DEFAULT_MATRIX)
    counts = {
        phase["id"]: len(tasks_for_phase(campaign, phase["id"]))
        for phase in campaign.phases
    }
    assert counts == {
        "r0_dynamic_calibration": 106,
        "r1_dynamic_null": 72,
        "r2_attack_mechanism": 144,
        "r3_e2e_confirmation": 384,
    }
    assert counts == campaign.matrix["expected_task_counts"]
    assert len(campaign.tasks) == 706
    registries = campaign.matrix["randomness"]
    names = [name for name in registries if name.endswith("_seeds")]
    seed_sets = {name: set(registries[name]) for name in names}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert seed_sets[left].isdisjoint(seed_sets[right])


def test_every_v2_arm_obeys_the_locked_private_gradient_contract() -> None:
    campaign = load_campaign(DEFAULT_MATRIX)
    for task in campaign.tasks:
        config = resolved_config(campaign, task, inject_threshold=False)
        algo = config["training"]["algo_config"]
        assert config["device"] == "mps"
        assert config["data"]["dataset"] == "fashionmnist"
        assert config["model"]["architecture"] == "lenet5_tanh"
        assert config["clients"]["num_clients"] == 25
        assert config["clients"]["sample_fraction"] == 1.0
        assert config["training"]["num_rounds"] == 40
        assert algo["privacy_public_dataset_size"] == 2400
        assert algo["fixed_batch_size"] == algo["batch_size"] == 120
        assert algo["clip_norm"] == 4.0
        assert algo["fixed_steps_per_round"] == 1
        assert algo["sampling_scheme"] == "fixed_without_replacement"
        assert algo["privacy_adjacency"] == "replace_one"
        assert algo["far_score_mode"] == "raw_distance"
        if (
            task.phase_id != "r3_e2e_confirmation"
            or task.axis_values["reference"] != "uniform"
        ):
            assert algo["far_alpha"] == 0.1
        assert algo["rcig_required_private_device"] == "mps"
        assert algo["rcig_covariance_registry"] == "authenticated_public_mechanism"
        assert algo["external_attack_diagnostics"] is True


def test_r2_materializes_deterministic_attacks_and_r3_has_no_recovery_arm() -> None:
    campaign = load_campaign(DEFAULT_MATRIX)
    r2 = tasks_for_phase(campaign, "r2_attack_mechanism")
    for task in r2:
        config = resolved_config(campaign, task, inject_threshold=False)
        attack = config["training"]["algo_config"]["attack"]
        assert attack["client_ids"] == [0, 1, 2, 3, 4]
        assert attack["num_byzantine"] == 5
        assert attack["active_round_end"] == 40
        expected_start = (
            17 if task.axis_values["schedule"] == "persistent_after_clean_warmup" else 1
        )
        assert attack["active_round_start"] == expected_start
        assert (
            attack["name"]
            == {"bf_x10": "bf", "ipm": "ipm", "alie": "alie"}[
                task.axis_values["attack"]
            ]
        )
    r3 = tasks_for_phase(campaign, "r3_e2e_confirmation")
    assert {task.axis_values["reference"] for task in r3} == {
        "uniform",
        "fcc",
        "rfa",
        "rcig_full",
    }
    for task in r3:
        config = resolved_config(campaign, task, inject_threshold=False)
        attack = config["training"]["algo_config"]["attack"]
        if attack["enabled"]:
            assert attack["active_round_start"] == 17
            assert attack["active_round_end"] == 40


def test_thresholds_are_injected_from_the_immutable_r0_artifact(
    tmp_path: Path,
) -> None:
    campaign = _campaign_at(tmp_path)
    artifact = _write_calibration_gate(campaign)
    task = next(
        task
        for task in tasks_for_phase(campaign, "r2_attack_mechanism")
        if task.axis_values["noise_regime"] == "heteroscedastic"
    )
    algo = resolved_config(campaign, task)["training"]["algo_config"]
    assert algo["rcig_innovation_threshold"] == 6.0
    assert algo["rcig_isotropic_innovation_threshold"] == 5.0
    assert algo["rcig_euclidean_innovation_threshold"] == 4.0
    assert algo["rcig_threshold_artifact_sha256"] == _file_sha256(artifact)


def test_corrupted_calibration_evidence_cannot_unlock_a_task(tmp_path: Path) -> None:
    campaign = _campaign_at(tmp_path)
    artifact = _write_calibration_gate(campaign)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["evidence"]["thresholds_by_regime_and_mode"]["homogeneous"]["full"] = 999.0
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    task = tasks_for_phase(campaign, "r1_dynamic_null")[0]
    with pytest.raises(RuntimeError, match="evidence hash"):
        resolved_config(campaign, task)


def test_campaign_lock_covers_all_scientifically_relevant_sources() -> None:
    payload = _campaign_lock_payload(load_campaign(DEFAULT_MATRIX))
    locked = payload["source_sha256"]
    assert {
        "algorithms/ldp_gradient_far.py",
        "algorithms/__init__.py",
        "algorithms/base.py",
        "algorithms/dp_references.py",
        "algorithms/fedavg.py",
        "algorithms/reference_utils.py",
        "algorithms/rcig_temporal_reference.py",
        "core/seeding.py",
        "datasets/partitioner.py",
        "datasets/registry.py",
        "models/registry.py",
        "privacy/local_dpsgd.py",
        "privacy/rdp.py",
        "metrics/client_fairness.py",
        "metrics/rcig_evaluation.py",
        "metrics/robustness.py",
        "robustness/aggregators.py",
        "robustness/tensor_ops.py",
        "attacks/byzantine.py",
        "attacks/__init__.py",
        "run_experiment.py",
    }.issubset(locked)
    assert all(len(value) == 64 for value in locked.values())


def test_device_audit_applies_to_non_rcig_arms_and_keeps_rcig_cpu_float64(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "ldp_gradient_far_private_gradient_mps_fraction": 1.0,
            "ldp_gradient_far_private_compute_device": "mps",
            "far_attack_labels_visible_to_server_aggregate": False,
            "far_attack_config_visible_to_server_aggregate": False,
            "far_external_attack_diagnostics": True,
            "far_external_attack_diagnostics_boundary": "posthoc_simulator_only",
        }
        for _ in range(40)
    ]
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"rounds": rows}), encoding="utf-8")
    baseline = {"training": {"algo_config": {"robust_reference": "centered_clipping"}}}
    _post_run_device_audit(metrics, baseline)
    rows[0]["ldp_gradient_far_private_compute_device"] = "cpu"
    metrics.write_text(json.dumps({"rounds": rows}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="generic private compute-device"):
        _post_run_device_audit(metrics, baseline)

    for row in rows:
        row.update(
            {
                "ldp_gradient_far_private_compute_device": "mps",
                "rcig_private_gradient_mps_fraction": 1.0,
                "rcig_private_gradient_compute_device": "mps",
                "rcig_server_aggregation_device": "cpu",
                "rcig_server_aggregation_dtype": "torch.float64",
            }
        )
    metrics.write_text(json.dumps({"rounds": rows}), encoding="utf-8")
    rcig = {"training": {"algo_config": {"robust_reference": "rcig_temporal"}}}
    _post_run_device_audit(metrics, rcig)
    rows[-1]["rcig_server_aggregation_dtype"] = "torch.float32"
    metrics.write_text(json.dumps({"rounds": rows}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="float64"):
        _post_run_device_audit(metrics, rcig)


def test_task_completion_validates_status_and_full_resolved_config(
    tmp_path: Path,
) -> None:
    campaign = _campaign_at(tmp_path)
    ensure_campaign_lock(campaign)
    task = next(
        task
        for task in tasks_for_phase(campaign, "r3_e2e_confirmation")
        if task.axis_values
        == {
            "noise_regime": "homogeneous",
            "reference": "uniform",
            "schedule": "none",
        }
    )
    output = task_output_dir(campaign, task)
    config = resolved_config(campaign, task, output)
    output.mkdir(parents=True)
    config_path = output / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    metrics_path = output / "experiment" / "metrics.json"
    metrics_path.parent.mkdir()
    metric_algo = dict(config["training"]["algo_config"])
    metric_algo["device"] = "mps"
    rounds = []
    for index in range(1, 41):
        rounds.append(
            {
                "round_num": index,
                "privacy_sampling_scheme": "fixed_without_replacement",
                "privacy_adjacency": "replace_one",
                "ldp_gradient_far_private_gradient_mps_fraction": 1.0,
                "ldp_gradient_far_private_compute_device": "mps",
                "far_attack_labels_visible_to_server_aggregate": False,
                "far_attack_config_visible_to_server_aggregate": False,
                "far_external_attack_diagnostics": True,
                "far_external_attack_diagnostics_boundary": ("posthoc_simulator_only"),
                "attack_window_active": False,
                "num_byzantine_oracle": 0,
                "test_accuracy": 0.60,
                "client_accuracy_mean": 0.59,
                "test_loss": 1.2,
                "client_accuracy_variance_pct2": 9.0,
                "worst20_accuracy_pct": 50.0,
                "best20_worst20_gap_pct": 12.0,
                "privacy_epsilon_max": 4.0,
                "privacy_delta": 1.0e-5,
                "privacy_model_noise_multiplier_min": 2.0,
                "privacy_model_noise_multiplier_max": 2.0,
                "far_server_clip_rate": 0.1,
                "far_max_weight": 0.04,
                "far_noise_amplification_vs_uniform": 1.0,
            }
        )
    metrics_path.write_text(
        json.dumps(
            {
                "algorithm": "ldp_gradient_far",
                "config": metric_algo,
                "summary": {
                    "num_rounds": 40,
                    "seed": task.seed,
                    "partition_seed": task.seed,
                    "dataset": "fashionmnist",
                    "model": "lenet5_tanh",
                    "num_clients": 25,
                },
                "rounds": rounds,
            }
        ),
        encoding="utf-8",
    )
    attack = config["training"]["algo_config"]["attack"]
    status = {
        "campaign_id": campaign.matrix["campaign_id"],
        "phase": task.phase_id,
        "run_id": task.run_id,
        "status": "completed",
        "device": "mps",
        "mps_fallback": 0,
        "server_postprocessing": "cpu_float64",
        "scientific_hash": campaign_scientific_hash(campaign),
        "resolved_config_sha256": _file_sha256(config_path),
        "metrics_sha256": _file_sha256(metrics_path),
        "pairing_seed_block": task.seed,
        "pairing_design_sha256": config["training"]["algo_config"][
            "rcig_pairing_design_sha256"
        ],
        "attack_ids": attack.get("client_ids", []),
        "attack_start": attack.get("active_round_start"),
        "attack_end": attack.get("active_round_end"),
    }
    status_path = output / "orchestration_status.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    assert task_is_complete(campaign, task)

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["rounds"][-1].pop("test_accuracy")
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    status["metrics_sha256"] = _file_sha256(metrics_path)
    status_path.write_text(json.dumps(status), encoding="utf-8")
    assert not task_is_complete(campaign, task)
    payload["rounds"][-1]["test_accuracy"] = 0.60
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    status["metrics_sha256"] = _file_sha256(metrics_path)
    status_path.write_text(json.dumps(status), encoding="utf-8")
    assert task_is_complete(campaign, task)

    status["mps_fallback"] = 1
    status_path.write_text(json.dumps(status), encoding="utf-8")
    assert not task_is_complete(campaign, task)
    status["mps_fallback"] = 0
    status_path.write_text(json.dumps(status), encoding="utf-8")
    config["seed"] += 1
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    status["resolved_config_sha256"] = _file_sha256(config_path)
    status_path.write_text(json.dumps(status), encoding="utf-8")
    assert not task_is_complete(campaign, task)


def test_r0_gate_is_fail_closed_and_immutable(tmp_path: Path) -> None:
    campaign = _campaign_at(tmp_path)
    criteria = campaign.phases[0]["gate_criteria"]
    evidence = {item["id"]: float(item["threshold"]) for item in criteria}
    evidence["finite_threshold_fraction"] = 0.5
    evidence["thresholds_by_regime_and_mode"] = {}
    path = record_gate(campaign, "r0_dynamic_calibration", evidence)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["decision"] == "stop"
    assert len(payload["evidence_sha256"]) == 64
    changed = dict(evidence)
    changed["finite_threshold_fraction"] = 1.0
    with pytest.raises(RuntimeError, match="immutable"):
        record_gate(campaign, "r0_dynamic_calibration", changed)


def test_device_policy_rejects_non_mps() -> None:
    validate_requested_device("mps")
    with pytest.raises(ValueError, match="CPU and CUDA"):
        validate_requested_device("cpu")


def test_direct_runner_bootstraps_repository_root_for_analyzer_import(
    tmp_path: Path,
) -> None:
    """Regress the direct-script import path used by ``--run-chain --resume``."""

    runner = Path(rcig_v2_runner.__file__).resolve()
    probe = (
        "import runpy; "
        f"runpy.run_path({str(runner)!r}); "
        "from scripts.analyze_rcig_ldp_gradient_far_v2 import "
        "_critical_protocol_errors; "
        "assert callable(_critical_protocol_errors)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_per_task_lock_prevents_duplicate_live_launcher(tmp_path: Path) -> None:
    campaign = _campaign_at(tmp_path)
    task = tasks_for_phase(campaign, "r0_dynamic_calibration")[0]
    with _exclusive_task_lock(campaign, task):
        with pytest.raises(RuntimeError, match="another v2 launcher owns"):
            with _exclusive_task_lock(campaign, task):
                pass


def test_run_task_materializes_running_status_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real status-building path before any expensive training."""

    campaign = _campaign_at(tmp_path)
    task = tasks_for_phase(campaign, "r0_dynamic_calibration")[0]

    def fake_training(*_args, **_kwargs) -> None:
        output = task_output_dir(campaign, task)
        (output / "metrics.json").write_text(
            json.dumps({"rounds": [{} for _ in range(40)]}), encoding="utf-8"
        )

    monkeypatch.setattr(rcig_v2_runner.subprocess, "run", fake_training)
    monkeypatch.setattr(
        rcig_v2_runner, "_post_run_device_audit", lambda *_args, **_kwargs: None
    )

    run_task(campaign, task, resume=True)
    status = json.loads(
        (task_output_dir(campaign, task) / "orchestration_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["status"] == "completed"
    assert status["pairing_design_sha256"]
