from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.analyze_rcig_ldp_gradient_far import (
    _cp_upper,
    collect_runs,
    evaluate_gate,
)
from scripts.run_rcig_ldp_gradient_far import (
    DEFAULT_MATRIX,
    campaign_scientific_hash,
    load_campaign,
    resolved_config,
    task_output_dir,
    tasks_for_phase,
)


def _install_calibration_gate(campaign) -> None:
    path = campaign.output_root / "_gates" / "r0_calibration_frozen.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "decision": "promote",
                "campaign_scientific_hash": campaign_scientific_hash(campaign),
                "evidence": {
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
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _payload(campaign, task, *, attacked: bool = False):
    inject = task.phase_id != "r0_calibration_frozen"
    config = resolved_config(campaign, task, inject_threshold=inject)
    algo = copy.deepcopy(config["training"]["algo_config"])
    algo["device"] = "mps"
    rounds = []
    for round_num in range(1, int(config["training"]["num_rounds"]) + 1):
        ready = round_num >= 13
        row = {
            "round_num": round_num,
            "privacy_sampling_scheme": "fixed_without_replacement",
            "privacy_adjacency": "replace_one",
            "rcig_private_gradient_mps_fraction": 1.0,
            "privacy_epsilon_max": 4.0,
            "test_accuracy": 0.60,
            "client_accuracy_mean": 0.59,
            "test_loss": 1.2,
            "client_accuracy_variance_pct2": 9.0,
            "worst20_accuracy_pct": 50.0,
            "best20_worst20_gap_pct": 12.0,
            "byzantine_weight_mass_oracle": 0.2,
            "rcig_history_ready": ready,
            "rcig_reference_mode": "full",
        }
        if ready:
            row.update(
                {
                    "rcig_reference_strictly_past": True,
                    "rcig_gate_source_round_min": round_num - 12,
                    "rcig_gate_source_round_max": round_num - 9,
                    "rcig_older_round_min": round_num - 8,
                    "rcig_older_round_max": round_num - 5,
                    "rcig_newer_round_min": round_num - 4,
                    "rcig_newer_round_max": round_num - 1,
                    "rcig_covariance_psd_certified": True,
                    "rcig_public_subspace_dimension": 64,
                    "rcig_full_innovation_stat": 2.0 + task.seed * 1.0e-6,
                    "rcig_isotropic_innovation_stat": 1.5 + task.seed * 1.0e-6,
                    "rcig_euclidean_innovation_stat": 1.0 + task.seed * 1.0e-6,
                    "rcig_full_gate_active": attacked,
                    "rcig_isotropic_gate_active": attacked,
                    "rcig_euclidean_gate_active": attacked,
                    "rcig_gate_active": attacked,
                    "rcig_full_error_to_clean_honest_center_oracle": (
                        0.50 if attacked else 0.90
                    ),
                    "rcig_isotropic_error_to_clean_honest_center_oracle": (
                        0.80 if attacked else 0.91
                    ),
                    "rcig_euclidean_error_to_clean_honest_center_oracle": (
                        0.51 if attacked else 0.92
                    ),
                    "rcig_reference_error_to_clean_honest_center_oracle": (
                        0.50 if attacked else 0.90
                    ),
                    "rcig_identity_new_error_to_clean_honest_center_oracle": 1.0,
                    "rcig_identity_old_error_to_clean_honest_center_oracle": 1.2,
                    "rcig_midpoint_error_to_clean_honest_center_oracle": 0.95,
                }
            )
        rounds.append(row)
    return {
        "algorithm": "ldp_gradient_far",
        "config": algo,
        "summary": {
            "num_rounds": config["training"]["num_rounds"],
            "seed": task.seed,
            "partition_seed": task.seed,
            "dataset": "fashionmnist",
            "model": "lenet5_tanh",
            "num_clients": 25,
        },
        "rounds": rounds,
    }


def _write(campaign, task, payload) -> None:
    output = task_output_dir(campaign, task)
    metrics = output / "experiment" / "metrics.json"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(json.dumps(payload), encoding="utf-8")
    (output / "orchestration_status.json").write_text(
        json.dumps({"status": "completed", "device": "mps"}), encoding="utf-8"
    )


def test_cp_upper_is_exact_enough_for_zero_of_36() -> None:
    expected = 1.0 - 0.025 ** (1.0 / 36)
    assert _cp_upper(0, 36) == pytest.approx(expected)
    assert _cp_upper(0, 36) < 0.10


def test_calibration_builds_six_finite_thresholds(tmp_path: Path) -> None:
    campaign = replace(load_campaign(DEFAULT_MATRIX), output_root=tmp_path / "results")
    for task in tasks_for_phase(campaign, "r0_calibration_frozen"):
        _write(campaign, task, _payload(campaign, task))
    evidence = evaluate_gate(campaign, "r0_calibration_frozen")
    assert evidence["complete_fraction"] == 1.0
    assert evidence["invalid_runs"] == 0.0
    assert evidence["strict_past_fraction"] == 1.0
    assert evidence["covariance_psd_fraction"] == 1.0
    assert evidence["finite_threshold_fraction"] == 1.0
    assert set(evidence["thresholds_by_regime_and_mode"]) == {
        "homogeneous",
        "heteroscedastic",
    }
    assert all(
        set(values) == {"full", "isotropic", "euclidean"}
        for values in evidence["thresholds_by_regime_and_mode"].values()
    )


def test_null_gate_uses_seed_level_false_activation_and_passes_zero_of_36(
    tmp_path: Path,
) -> None:
    campaign = replace(load_campaign(DEFAULT_MATRIX), output_root=tmp_path / "results")
    _install_calibration_gate(campaign)
    for task in tasks_for_phase(campaign, "r1_null_validation_frozen"):
        _write(campaign, task, _payload(campaign, task))
    evidence = evaluate_gate(campaign, "r1_null_validation_frozen")
    assert evidence["complete_fraction"] == 1.0
    assert evidence["max_cp97_5_false_activation_upper"] < 0.10
    assert evidence["max_no_attack_reference_loss_one_sided_upper"] < 0.02
    assert all(
        cell["trials"] == 36 for cell in evidence["false_activation_cells"].values()
    )


def test_mechanistic_gate_detects_paired_attacked_gain(tmp_path: Path) -> None:
    campaign = replace(load_campaign(DEFAULT_MATRIX), output_root=tmp_path / "results")
    _install_calibration_gate(campaign)
    for task in tasks_for_phase(campaign, "r2_attack_mechanism_frozen"):
        _write(campaign, task, _payload(campaign, task, attacked=True))
    evidence = evaluate_gate(campaign, "r2_attack_mechanism_frozen")
    assert evidence["complete_fraction"] == 1.0
    assert evidence["attacked_mse_gain_vs_identity_new_one_sided_low"] == pytest.approx(
        0.5
    )
    assert evidence["attack_gate_activation_rate"] == 1.0
    assert evidence["full_gain_vs_isotropic_one_sided_low"] > 0.0
    assert evidence["full_gain_vs_euclidean_one_sided_low"] > 0.0


def test_missing_rcig_counterfactual_is_invalid_not_imputed(tmp_path: Path) -> None:
    campaign = replace(load_campaign(DEFAULT_MATRIX), output_root=tmp_path / "results")
    task = tasks_for_phase(campaign, "r0_calibration_frozen")[0]
    payload = _payload(campaign, task)
    for row in payload["rounds"]:
        row.pop("rcig_euclidean_innovation_stat", None)
    _write(campaign, task, payload)
    record = collect_runs(campaign)[0]
    assert record.status == "invalid"
    assert "missing_rcig_euclidean_counterfactual" in record.reason
