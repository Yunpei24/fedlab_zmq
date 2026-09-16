from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from scripts.analyze_rcig_ldp_gradient_far_v2 import (
    ERROR_SUFFIX,
    _cp_upper,
    _mean_squared_error,
    _paired_deltas,
    _strict_past,
    analyze_campaign,
    collect_runs,
    evaluate_gate,
)
from scripts.run_rcig_ldp_gradient_far_v2 import (
    DEFAULT_MATRIX,
    _file_sha256,
    campaign_scientific_hash,
    ensure_campaign_lock,
    load_campaign,
    record_gate,
    resolved_config,
    task_output_dir,
    tasks_for_phase,
)


def _campaign_at(tmp_path: Path):
    return replace(load_campaign(DEFAULT_MATRIX), output_root=tmp_path / "results")


def _install_calibration_gate(campaign) -> None:
    record_gate(
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


def _payload(campaign, task, *, attacked_gate: bool = False) -> tuple[dict, dict]:
    config = resolved_config(
        campaign,
        task,
        inject_threshold=task.phase_id != "r0_dynamic_calibration",
    )
    algo = copy.deepcopy(config["training"]["algo_config"])
    algo["device"] = "mps"
    oracle_phase = task.phase_id in {"r1_dynamic_null", "r2_attack_mechanism"}
    rounds = []
    for round_num in range(1, 41):
        ready = round_num >= 13
        attack = algo["attack"]
        attack_active = bool(attack.get("enabled", False)) and int(
            attack["active_round_start"]
        ) <= round_num <= int(attack["active_round_end"])
        row = {
            "round_num": round_num,
            "privacy_sampling_scheme": "fixed_without_replacement",
            "privacy_adjacency": "replace_one",
            "ldp_gradient_far_private_gradient_mps_fraction": 1.0,
            "ldp_gradient_far_private_compute_device": "mps",
            "far_attack_labels_visible_to_server_aggregate": False,
            "far_attack_config_visible_to_server_aggregate": False,
            "far_external_attack_diagnostics": True,
            "far_external_attack_diagnostics_boundary": "posthoc_simulator_only",
            "privacy_epsilon_max": 4.0,
            "privacy_delta": 1.0e-5,
            "privacy_model_noise_multiplier_min": 2.0,
            "privacy_model_noise_multiplier_max": (
                4.0
                if task.axis_values.get("noise_regime") == "heteroscedastic"
                else 2.0
            ),
            "privacy_clip_rate_mean": 0.25,
            "test_accuracy": 0.60,
            "client_accuracy_mean": 0.59,
            "test_loss": 1.2,
            "client_accuracy_variance_pct2": 9.0,
            "worst20_accuracy_pct": 50.0,
            "best20_worst20_gap_pct": 12.0,
            "byzantine_weight_mass_oracle": 0.2 if attack_active else 0.0,
            "far_server_clip_rate": 0.1,
            "far_max_weight": 0.08,
            "far_noise_amplification_vs_uniform": 1.1,
            "attack_window_active": attack_active,
            "num_byzantine_oracle": 5 if attack_active else 0,
            "rcig_history_ready": ready,
            "rcig_reference_mode": "full",
            "rcig_private_gradient_mps_fraction": 1.0,
            "rcig_private_gradient_compute_device": "mps",
            "rcig_server_aggregation_device": "cpu",
            "rcig_server_aggregation_dtype": "torch.float64",
        }
        if ready:
            row.update(
                {
                    "rcig_reference_strictly_past": True,
                    "rcig_gate_source_round_min": round_num - 13,
                    "rcig_gate_source_round_max": round_num - 10,
                    "rcig_older_round_min": round_num - 9,
                    "rcig_older_round_max": round_num - 6,
                    "rcig_newer_round_min": round_num - 5,
                    "rcig_newer_round_max": round_num - 2,
                    "rcig_covariance_psd_certified": True,
                    "rcig_public_variance_provenance": (
                        "authenticated_public_mechanism"
                    ),
                    "rcig_public_variance_registry_verified": True,
                    "rcig_public_variance_source": (
                        "server_config_and_authenticated_client_id"
                    ),
                    "rcig_client_variance_metadata_used_for_construction": False,
                    "rcig_client_variance_metadata_consistency_checked": True,
                    "rcig_post_server_clip_covariance_is_delta_method_proxy": True,
                    "rcig_diagnostics_use_realised_noise": False,
                    "rcig_diagnostics_use_attack_labels": False,
                    "rcig_diagnostics_use_clean_gradients": False,
                    "rcig_full_innovation_stat": 2.0 + task.seed * 1.0e-7,
                    "rcig_isotropic_innovation_stat": 1.5 + task.seed * 1.0e-7,
                    "rcig_euclidean_innovation_stat": 1.0 + task.seed * 1.0e-7,
                    "rcig_full_gate_active": attacked_gate and attack_active,
                    "rcig_isotropic_gate_active": attacked_gate and attack_active,
                    "rcig_euclidean_gate_active": attacked_gate and attack_active,
                    "rcig_gate_active": attacked_gate and attack_active,
                    "rcig_max_covariance_anisotropy_ratio": 1.2,
                    "rcig_persistent_frozen_after_commit": (
                        attacked_gate and attack_active
                    ),
                }
            )
            if oracle_phase:
                attacked_error = task.phase_id == "r2_attack_mechanism"
                row.update(
                    {
                        f"rcig_full_{ERROR_SUFFIX}": (0.25 if attacked_error else 1.0),
                        f"rcig_isotropic_{ERROR_SUFFIX}": (
                            0.64 if attacked_error else 1.0
                        ),
                        f"rcig_euclidean_{ERROR_SUFFIX}": (
                            0.2601 if attacked_error else 1.0
                        ),
                        f"rcig_identity_new_{ERROR_SUFFIX}": 1.0,
                        f"rcig_reference_{ERROR_SUFFIX}": (
                            0.25 if attacked_error else 1.0
                        ),
                        "rcig_oracle_evaluation_boundary": "offline_simulator_only",
                        "rcig_oracle_was_visible_to_server_aggregate": False,
                        "rcig_oracle_metric_is_squared_l2": True,
                    }
                )
        rounds.append(row)
    payload = {
        "algorithm": "ldp_gradient_far",
        "config": algo,
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
    return config, payload


def _write(campaign, task, config: dict, payload: dict) -> None:
    output = task_output_dir(campaign, task)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    metrics_path = output / "experiment" / "metrics.json"
    metrics_path.parent.mkdir()
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
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
    (output / "orchestration_status.json").write_text(
        json.dumps(status), encoding="utf-8"
    )


def test_cp_upper_zero_of_36_matches_preregistered_bound() -> None:
    expected = 1.0 - 0.025 ** (1.0 / 36)
    assert _cp_upper(0, 36) == pytest.approx(expected)
    assert _cp_upper(0, 36) < 0.10


def test_mse_averages_canonical_squared_l2_errors_without_resquaring() -> None:
    rows = [{f"candidate_{ERROR_SUFFIX}": 9.0}, {f"candidate_{ERROR_SUFFIX}": 16.0}]
    assert _mean_squared_error(rows, "candidate") == pytest.approx(12.5)


def test_internal_round_causality_is_strict_and_off_by_one_safe() -> None:
    row = {
        "round_num": 13,
        "rcig_reference_strictly_past": True,
        "rcig_gate_source_round_min": 0,
        "rcig_gate_source_round_max": 3,
        "rcig_older_round_min": 4,
        "rcig_older_round_max": 7,
        "rcig_newer_round_min": 8,
        "rcig_newer_round_max": 11,
    }
    assert _strict_past(row)
    row["rcig_newer_round_max"] = 12
    assert not _strict_past(row)


def test_r0_uses_the_maximum_of_53_blocks_in_each_of_six_cells(
    tmp_path: Path,
) -> None:
    campaign = _campaign_at(tmp_path)
    ensure_campaign_lock(campaign)
    expected_maxima = {}
    for task in tasks_for_phase(campaign, "r0_dynamic_calibration"):
        config, payload = _payload(campaign, task)
        _write(campaign, task, config, payload)
        regime = task.axis_values["noise_regime"]
        expected_maxima.setdefault(regime, {})["full"] = max(
            expected_maxima.get(regime, {}).get("full", -1.0),
            2.0 + task.seed * 1.0e-7,
        )
        expected_maxima.setdefault(regime, {})["isotropic"] = max(
            expected_maxima.get(regime, {}).get("isotropic", -1.0),
            1.5 + task.seed * 1.0e-7,
        )
        expected_maxima.setdefault(regime, {})["euclidean"] = max(
            expected_maxima.get(regime, {}).get("euclidean", -1.0),
            1.0 + task.seed * 1.0e-7,
        )
    evidence = evaluate_gate(campaign, "r0_dynamic_calibration")
    assert evidence["complete_fraction"] == 1.0
    assert evidence["invalid_runs"] == 0.0
    assert evidence["calibration_blocks_per_cell_min"] == 53
    assert evidence["threshold_operator"].startswith("maximum")
    for regime, expected_modes in expected_maxima.items():
        for mode, expected_value in expected_modes.items():
            assert evidence["thresholds_by_regime_and_mode"][regime][
                mode
            ] == pytest.approx(expected_value)
    assert evidence["simultaneous_distribution_free_confidence_lower"] >= 0.975


def test_r1_uses_one_global_union_event_per_paired_seed_and_worst_regime_loss(
    tmp_path: Path,
) -> None:
    campaign = _campaign_at(tmp_path)
    ensure_campaign_lock(campaign)
    _install_calibration_gate(campaign)
    for task in tasks_for_phase(campaign, "r1_dynamic_null"):
        config, payload = _payload(campaign, task)
        _write(campaign, task, config, payload)
    evidence = evaluate_gate(campaign, "r1_dynamic_null")
    assert evidence["complete_fraction"] == 1.0
    assert evidence["invalid_runs"] == 0.0
    assert evidence["paired_seed_blocks"] == 36
    assert evidence["global_union_false_activation_count"] == 0
    assert evidence["global_union_cp97_5_upper"] == pytest.approx(
        1.0 - 0.025 ** (1.0 / 36)
    )
    assert evidence["no_attack_reference_loss_one_sided_97_5_upper"] == 0.0
    assert len(evidence["seed_block_details"]) == 36


def test_r2_aggregates_rounds_within_each_seed_cell_and_gates_all_six_cells(
    tmp_path: Path,
) -> None:
    campaign = _campaign_at(tmp_path)
    _install_calibration_gate(campaign)
    for task in tasks_for_phase(campaign, "r2_attack_mechanism"):
        config, payload = _payload(campaign, task, attacked_gate=True)
        _write(campaign, task, config, payload)
    evidence = evaluate_gate(campaign, "r2_attack_mechanism")
    assert evidence["complete_fraction"] == 1.0
    assert evidence["invalid_runs"] == 0.0
    assert evidence["primary_cell_count"] == 6
    assert evidence["min_primary_cell_seed_gain_successes"] == 12
    assert evidence["min_primary_cell_gate_successes"] == 12
    assert evidence["heteroscedastic_anisotropy_identifiable_fraction"] == 1.0
    assert evidence["min_heteroscedastic_full_vs_isotropic_ci_low"] > 0.0
    assert evidence["min_primary_full_vs_euclidean_ci_low"] >= -0.02
    assert evidence["rounds_are_not_inference_units"] is True
    assert evidence["mse_definition"].startswith("mean_of_squared_l2")
    assert all(cell["seed_count"] == 12 for cell in evidence["cell_details"].values())


def test_non_rcig_cpu_private_gradient_is_invalid_not_silently_resumed(
    tmp_path: Path,
) -> None:
    campaign = _campaign_at(tmp_path)
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
    config, payload = _payload(campaign, task)
    payload["rounds"][0]["ldp_gradient_far_private_compute_device"] = "cpu"
    _write(campaign, task, config, payload)
    record = next(record for record in collect_runs(campaign) if record.task == task)
    assert record.status == "invalid"
    assert "generic_private_device" in record.reason


def test_r3_pairing_uses_rcig_vs_rfa_and_reports_directional_signs() -> None:
    rows = []
    for seed in range(12):
        common = {
            "seed": seed,
            "noise_regime": "heteroscedastic",
            "schedule": "alie_persistent_after_clean_warmup",
        }
        rows.extend(
            [
                {
                    **common,
                    "reference": "rcig_full",
                    "test_accuracy_pct": 61.0,
                    "worst20_pct": 51.0,
                    "gap_pp": 11.0,
                },
                {
                    **common,
                    "reference": "rfa",
                    "test_accuracy_pct": 60.0,
                    "worst20_pct": 50.0,
                    "gap_pp": 12.0,
                },
            ]
        )
    result = _paired_deltas(
        rows,
        regime="heteroscedastic",
        schedule="alie_persistent_after_clean_warmup",
    )
    assert result is not None and result["n"] == 12
    for metric in ("test_accuracy_delta_pp", "worst20_delta_pp", "gap_reduction_pp"):
        assert result[metric]["mean_ci95"][0] == pytest.approx(1.0)
        assert result[metric]["signs"] == {"positive": 12, "negative": 0, "ties": 0}


def test_incomplete_campaign_report_never_evaluates_or_imputes_gates(
    tmp_path: Path,
) -> None:
    campaign = _campaign_at(tmp_path)
    ensure_campaign_lock(campaign)
    output = tmp_path / "analysis"
    report = output / "report.md"
    state = analyze_campaign(campaign, output_dir=output, report_path=report)
    assert state["counts"] == {
        "expected": 706,
        "complete": 0,
        "invalid": 0,
        "missing": 706,
    }
    assert all(
        phase.get("not_evaluated") is True for phase in state["phase_evidence"].values()
    )
    rendered = report.read_text(encoding="utf-8")
    assert "Une phase incomplète n'est jamais évaluée" in rendered
    assert "not_evaluated" in rendered
    assert "R0 — seuils de calibration figés" not in rendered
