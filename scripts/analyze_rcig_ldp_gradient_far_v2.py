#!/usr/bin/env python3
"""Gate and report the preregistered RCIG/LDP-gradient-FAR v2 campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_rcig_ldp_gradient_far_v2 import (  # noqa: E402
    DEFAULT_MATRIX,
    RCIG_MODES,
    RCIG_REFERENCES,
    RCIGV2Campaign,
    RCIGV2Task,
    _canonical_hash,
    _file_sha256,
    _verified_gate_payload,
    campaign_scientific_hash,
    gate_path,
    load_campaign,
    phase_definition,
    resolved_config,
    task_output_dir,
    verify_existing_campaign_lock,
)

DEFAULT_OUTPUT_DIR = ROOT / "output/analysis/rcig_ldp_gradient_far_v2"
DEFAULT_REPORT = ROOT / "output/analysis/RCIG_LDP_Gradient_FAR_End_to_End_Report_v2.md"
ERROR_SUFFIX = "squared_l2_error_to_clean_honest_center_oracle"


@dataclass
class RunRecord:
    task: RCIGV2Task
    status: str
    reason: str
    metrics_path: Path | None
    payload: dict[str, Any] | None
    expected: dict[str, Any]
    artifact_hashes: dict[str, str]

    @property
    def rounds(self) -> list[dict[str, Any]]:
        if self.payload is None:
            return []
        value = self.payload.get("rounds")
        return value if isinstance(value, list) else []


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool01(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    number = _finite(value)
    if number in {0.0, 1.0}:
        return number
    return None


def _config_value_equal(observed: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or expected is None:
        return observed is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        value = _finite(observed)
        return value is not None and math.isclose(
            value, float(expected), rel_tol=0.0, abs_tol=1e-12
        )
    return observed == expected


def _find_metrics(output_dir: Path) -> Path | None:
    paths = sorted(output_dir.glob("**/metrics.json"))
    return paths[0] if len(paths) == 1 else None


def _ready_rows(record: RunRecord) -> list[dict[str, Any]]:
    return [row for row in record.rounds if bool(row.get("rcig_history_ready", False))]


def _attack_active(expected: Mapping[str, Any], public_round: int) -> bool:
    attack = expected["training"]["algo_config"].get("attack", {})
    if not bool(attack.get("enabled", False)):
        return False
    return (
        int(attack["active_round_start"])
        <= public_round
        <= int(attack["active_round_end"])
    )


def _strict_past(row: Mapping[str, Any]) -> bool:
    """Validate internal zero-based RCIG fields against public one-based rows."""

    public_round = int(row.get("round_num", -1))
    deployment_internal = public_round - 1
    fields = [
        _finite(row.get("rcig_gate_source_round_min")),
        _finite(row.get("rcig_gate_source_round_max")),
        _finite(row.get("rcig_older_round_min")),
        _finite(row.get("rcig_older_round_max")),
        _finite(row.get("rcig_newer_round_min")),
        _finite(row.get("rcig_newer_round_max")),
    ]
    if (
        public_round < 1
        or _bool01(row.get("rcig_reference_strictly_past")) != 1.0
        or any(value is None for value in fields)
    ):
        return False
    gate_min, gate_max, old_min, old_max, new_min, new_max = fields
    assert all(value is not None for value in fields)
    return bool(
        gate_min
        <= gate_max
        < old_min
        <= old_max
        < new_min
        <= new_max
        < deployment_internal
    )


def _mode_value(row: Mapping[str, Any], mode: str, suffix: str) -> float | None:
    converter = _bool01 if suffix == "gate_active" else _finite
    direct = converter(row.get(f"rcig_{mode}_{suffix}"))
    if direct is not None:
        return direct
    if str(row.get("rcig_reference_mode", "")) == mode:
        aliases = {
            "innovation_stat": "rcig_innovation_stat",
            "gate_active": "rcig_gate_active",
            ERROR_SUFFIX: f"rcig_reference_{ERROR_SUFFIX}",
        }
        return converter(row.get(aliases.get(suffix, "")))
    return None


def _expected_with_placeholder(
    campaign: RCIGV2Campaign, task: RCIGV2Task
) -> dict[str, Any]:
    calibration = campaign.output_root / "_gates/r0_dynamic_calibration.json"
    inject = False
    if calibration.exists():
        try:
            inject = (
                json.loads(calibration.read_text(encoding="utf-8")).get("decision")
                == "promote"
            )
        except (OSError, json.JSONDecodeError):
            inject = False
    expected = resolved_config(campaign, task, inject_threshold=inject)
    algo = expected["training"]["algo_config"]
    reference = str(algo.get("robust_reference", "")).lower()
    if (
        reference in RCIG_REFERENCES
        and not bool(algo.get("rcig_calibration_mode", False))
        and not inject
    ):
        algo["rcig_innovation_threshold"] = 1.0
        algo["rcig_isotropic_innovation_threshold"] = 1.0
        algo["rcig_euclidean_innovation_threshold"] = 1.0
    return expected


def _critical_protocol_errors(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    task: RCIGV2Task,
    status: Mapping[str, Any],
    resolved_payload: Mapping[str, Any],
    config_path: Path,
    metrics_path: Path,
) -> list[str]:
    errors: list[str] = []
    rounds = payload.get("rounds")
    summary = payload.get("summary")
    algo_config = payload.get("config")
    if not isinstance(rounds, list):
        return ["rounds missing"]
    if not isinstance(summary, Mapping):
        return ["summary missing"]
    if not isinstance(algo_config, Mapping):
        return ["algorithm config missing"]
    expected_algo = expected["training"]["algo_config"]
    oracle_phase = task.phase_id in {"r1_dynamic_null", "r2_attack_mechanism"}
    checks = {
        "algorithm": payload.get("algorithm") == "ldp_gradient_far",
        "round_count": len(rounds) == 40,
        "summary_rounds": int(summary.get("num_rounds", -1)) == 40,
        "seed": int(summary.get("seed", -1)) == task.seed,
        "partition_seed": int(summary.get("partition_seed", -1)) == task.seed,
        "dataset": summary.get("dataset") == "fashionmnist",
        "model": summary.get("model") == "lenet5_tanh",
        "num_clients": int(summary.get("num_clients", -1)) == 25,
        "device": algo_config.get("device") == "mps",
        "reference": algo_config.get("robust_reference")
        == expected_algo.get("robust_reference"),
        "alpha": math.isclose(
            float(algo_config.get("far_alpha", math.nan)),
            float(expected_algo.get("far_alpha", math.nan)),
            abs_tol=1e-12,
        ),
        "status_device": status.get("device") == "mps",
        "status_no_fallback": int(status.get("mps_fallback", -1)) == 0,
        "status_campaign_id": status.get("campaign_id")
        == expected_algo.get("rcig_campaign_id"),
        "status_phase": status.get("phase") == task.phase_id,
        "status_run_id": status.get("run_id") == task.run_id,
        "status_pairing_seed": status.get("pairing_seed_block") == task.seed,
        "status_pairing_design": status.get("pairing_design_sha256")
        == expected_algo.get("rcig_pairing_design_sha256"),
        "status_server_postprocessing": status.get("server_postprocessing")
        == "cpu_float64",
        "status_scientific_hash": status.get("scientific_hash")
        == expected_algo.get("rcig_campaign_scientific_hash"),
        "resolved_hash": status.get("resolved_config_sha256")
        == _file_sha256(config_path),
        "metrics_hash": status.get("metrics_sha256") == _file_sha256(metrics_path),
        "resolved_config_exact": resolved_payload == expected,
        "oracle_scope": bool(algo_config.get("enable_oracle_diagnostics", False))
        == oracle_phase,
        "oracle_evaluation_only": bool(
            algo_config.get("rcig_oracle_evaluation_only", False)
        )
        == oracle_phase,
        "oracle_not_released": bool(
            algo_config.get("rcig_oracle_metrics_are_not_release", False)
        )
        == oracle_phase,
        "oracle_separation_required": bool(
            algo_config.get("rcig_oracle_separation_required", False)
        )
        == (str(expected_algo.get("robust_reference", "")).lower() in RCIG_REFERENCES),
    }
    errors.extend(name for name, passed in checks.items() if not passed)
    critical_algo_keys = (
        "batch_size",
        "fixed_batch_size",
        "privacy_public_dataset_size",
        "local_epochs",
        "fixed_steps_per_round",
        "sampling_scheme",
        "privacy_adjacency",
        "privacy_sampling_rate_override",
        "target_epsilon",
        "delta",
        "clip_norm",
        "far_server_lr",
        "far_server_clip_norm",
        "per_sample_backend",
        "enable_dp",
        "far_score_mode",
        "noise_score_standardization",
        "score_subspace_mode",
        "far_alpha",
        "kappa_w",
        "tilt_bound_policy",
        "robust_reference",
        "num_byzantine",
        "reference_clip_radius",
        "anchor_update_rate",
        "anchor_clip_norm",
        "rfa_max_iter",
        "rfa_tol",
        "expected_num_clients",
        "client_metrics_every",
        "fairness_tail_fraction",
        "suppress_private_client_diagnostics",
        "external_attack_diagnostics",
        "privacy_num_rounds",
        "privacy_noise_multiplier_scale_by_client",
        "enable_oracle_diagnostics",
        "rcig_oracle_evaluation_only",
        "rcig_oracle_metrics_are_not_release",
        "rcig_oracle_separation_required",
        "rcig_campaign_id",
        "rcig_campaign_phase",
        "rcig_campaign_variant",
        "rcig_campaign_scientific_hash",
        "rcig_pairing_seed_block",
        "rcig_pairing_design_sha256",
        "rcig_required_private_device",
        "rcig_required_server_postprocess_device",
        "rcig_required_server_postprocess_dtype",
        "rcig_mps_fallback",
        "rcig_covariance_registry",
        "attack",
    )
    if str(expected_algo.get("robust_reference", "")).lower() in RCIG_REFERENCES:
        critical_algo_keys += (
            "rcig_covariance_mode",
            "rcig_innovation_threshold",
            "rcig_isotropic_innovation_threshold",
            "rcig_euclidean_innovation_threshold",
            "rcig_gate_window",
            "rcig_old_window",
            "rcig_new_window",
            "rcig_public_subspace_dimension",
            "rcig_public_subspace_seed",
            "rcig_gate_inner_mad",
            "rcig_gate_outer_mad",
            "rcig_min_accepted_mass",
            "rcig_covariance_ridge",
            "rcig_process_variance",
            "rcig_warmup_policy",
            "rcig_persistent_policy",
            "rcig_reference_output_radius",
            "rcig_recovery_threshold_ratio",
            "rcig_recovery_patience",
            "rcig_threshold_artifact_sha256",
        )
    for key in critical_algo_keys:
        if not _config_value_equal(algo_config.get(key), expected_algo.get(key)):
            errors.append(f"algorithm_config_{key}")
    attack = expected_algo.get("attack", {})
    if status.get("attack_ids") != attack.get("client_ids", []):
        errors.append("status_attack_ids")
    if status.get("attack_start") != attack.get("active_round_start"):
        errors.append("status_attack_start")
    if status.get("attack_end") != attack.get("active_round_end"):
        errors.append("status_attack_end")

    for public_round, row in enumerate(rounds, start=1):
        if int(row.get("round_num", -1)) != public_round:
            errors.append(f"round_{public_round}_public_index")
            break
        if row.get("privacy_sampling_scheme") != "fixed_without_replacement":
            errors.append(f"round_{public_round}_sampling")
            break
        if row.get("privacy_adjacency") != "replace_one":
            errors.append(f"round_{public_round}_adjacency")
            break
        if _finite(row.get("ldp_gradient_far_private_gradient_mps_fraction")) != 1.0:
            errors.append(f"round_{public_round}_generic_private_mps_fraction")
            break
        if str(row.get("ldp_gradient_far_private_compute_device", "")) not in {
            "mps",
            "mps:0",
        }:
            errors.append(f"round_{public_round}_generic_private_device")
            break
        if _bool01(row.get("far_attack_labels_visible_to_server_aggregate")) != 0.0:
            errors.append(f"round_{public_round}_attack_oracle_leak")
            break
        if _bool01(row.get("far_attack_config_visible_to_server_aggregate")) != 0.0:
            errors.append(f"round_{public_round}_attack_config_leak")
            break
        if _bool01(row.get("far_external_attack_diagnostics")) != 1.0:
            errors.append(f"round_{public_round}_external_attack_diagnostics")
            break
        if row.get("far_external_attack_diagnostics_boundary") != (
            "posthoc_simulator_only"
        ):
            errors.append(f"round_{public_round}_attack_diagnostic_boundary")
            break
        active = _attack_active(expected, public_round)
        if bool(row.get("attack_window_active", False)) != active:
            errors.append(f"round_{public_round}_attack_schedule")
            break
        expected_byzantine = 5 if active else 0
        if int(row.get("num_byzantine_oracle", -1)) != expected_byzantine:
            errors.append(f"round_{public_round}_byzantine_count")
            break

    final_row = rounds[-1] if rounds else {}
    required_final_metrics = (
        "test_accuracy",
        "client_accuracy_mean",
        "test_loss",
        "client_accuracy_variance_pct2",
        "worst20_accuracy_pct",
        "best20_worst20_gap_pct",
        "privacy_epsilon_max",
        "privacy_delta",
        "privacy_model_noise_multiplier_min",
        "privacy_model_noise_multiplier_max",
        "far_server_clip_rate",
        "far_max_weight",
        "far_noise_amplification_vs_uniform",
    )
    missing_final = [
        key for key in required_final_metrics if _finite(final_row.get(key)) is None
    ]
    if missing_final:
        errors.append("missing_final_metrics:" + ",".join(missing_final))
    else:
        if not 0.0 <= float(final_row["test_accuracy"]) <= 1.0:
            errors.append("test_accuracy_out_of_range")
        if not 0.0 <= float(final_row["client_accuracy_mean"]) <= 1.0:
            errors.append("client_accuracy_out_of_range")
        if not 0.0 <= float(final_row["far_server_clip_rate"]) <= 1.0:
            errors.append("server_clip_rate_out_of_range")
        if not 0.0 < float(final_row["far_max_weight"]) <= 1.0:
            errors.append("far_max_weight_out_of_range")
        if float(final_row["far_noise_amplification_vs_uniform"]) < 1.0 - 1e-12:
            errors.append("far_concentration_below_uniform")
        if not math.isclose(
            float(final_row["privacy_delta"]), 1.0e-5, rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append("privacy_delta")
        noise_min = float(final_row["privacy_model_noise_multiplier_min"])
        noise_max = float(final_row["privacy_model_noise_multiplier_max"])
        if noise_min <= 0.0 or noise_max < noise_min:
            errors.append("noise_multiplier_range")

    attacked_rows = [
        row for row in rounds if _attack_active(expected, int(row.get("round_num", -1)))
    ]
    if attacked_rows and any(
        _finite(row.get("byzantine_weight_mass_oracle")) is None
        for row in attacked_rows
    ):
        errors.append("missing_attacked_byzantine_weight_mass")

    reference = str(expected_algo.get("robust_reference", "")).lower()
    if reference in RCIG_REFERENCES:
        if not rounds or not all(
            _finite(row.get("rcig_private_gradient_mps_fraction")) == 1.0
            and str(row.get("rcig_private_gradient_compute_device", ""))
            in {"mps", "mps:0"}
            and str(row.get("rcig_server_aggregation_device", "")) == "cpu"
            and str(row.get("rcig_server_aggregation_dtype", ""))
            in {"torch.float64", "float64"}
            for row in rounds
        ):
            errors.append("private_mps_or_server_cpu_float64")
        ready = [row for row in rounds if bool(row.get("rcig_history_ready", False))]
        if not ready:
            errors.append("no_rcig_ready_round")
        else:
            if not all(_strict_past(row) for row in ready):
                errors.append("rcig_not_strictly_past_internal_rounds")
            if not all(
                _bool01(row.get("rcig_covariance_psd_certified")) == 1.0
                for row in ready
            ):
                errors.append("rcig_covariance_not_psd")
            if not all(
                row.get("rcig_public_variance_provenance")
                == "authenticated_public_mechanism"
                and _bool01(row.get("rcig_public_variance_registry_verified")) == 1.0
                and row.get("rcig_public_variance_source")
                == "server_config_and_authenticated_client_id"
                and _bool01(
                    row.get("rcig_client_variance_metadata_used_for_construction")
                )
                == 0.0
                and _bool01(
                    row.get("rcig_client_variance_metadata_consistency_checked")
                )
                == 1.0
                and _bool01(
                    row.get("rcig_post_server_clip_covariance_is_delta_method_proxy")
                )
                == 1.0
                and _bool01(row.get("rcig_diagnostics_use_realised_noise")) == 0.0
                and _bool01(row.get("rcig_diagnostics_use_attack_labels")) == 0.0
                and _bool01(row.get("rcig_diagnostics_use_clean_gradients")) == 0.0
                for row in ready
            ):
                errors.append("rcig_covariance_provenance_or_oracle_leak")
            for mode in RCIG_MODES:
                required_suffixes = ["innovation_stat", "gate_active"]
                if task.phase_id in {"r1_dynamic_null", "r2_attack_mechanism"}:
                    required_suffixes.append(ERROR_SUFFIX)
                for suffix in required_suffixes:
                    if any(_mode_value(row, mode, suffix) is None for row in ready):
                        errors.append(f"missing_rcig_{mode}_{suffix}")
                        break
            if task.phase_id in {"r1_dynamic_null", "r2_attack_mechanism"} and any(
                _finite(row.get(f"rcig_identity_new_{ERROR_SUFFIX}")) is None
                for row in ready
            ):
                errors.append("missing_rcig_identity_new_oracle_error")
            if task.phase_id in {"r1_dynamic_null", "r2_attack_mechanism"} and any(
                _finite(row.get(f"rcig_reference_{ERROR_SUFFIX}")) is None
                for row in ready
            ):
                errors.append("missing_deployed_rcig_oracle_error")
            if task.phase_id in {"r1_dynamic_null", "r2_attack_mechanism"} and any(
                row.get("rcig_oracle_evaluation_boundary") != "offline_simulator_only"
                or _bool01(row.get("rcig_oracle_was_visible_to_server_aggregate"))
                != 0.0
                or _bool01(row.get("rcig_oracle_metric_is_squared_l2")) != 1.0
                for row in ready
            ):
                errors.append("rcig_oracle_not_separated_or_not_squared_l2")
            if any(
                _finite(row.get("rcig_max_covariance_anisotropy_ratio")) is None
                for row in ready
            ):
                errors.append("missing_rcig_anisotropy")
    final_epsilon = _finite(rounds[-1].get("privacy_epsilon_max")) if rounds else None
    if final_epsilon is None or abs(final_epsilon - 4.0) > 0.05:
        errors.append("final_epsilon")
    return errors


def collect_runs(campaign: RCIGV2Campaign) -> list[RunRecord]:
    records: list[RunRecord] = []
    for task in campaign.tasks:
        expected = _expected_with_placeholder(campaign, task)
        output_dir = task_output_dir(campaign, task)
        status_path = output_dir / "orchestration_status.json"
        config_path = output_dir / "resolved_config.yaml"
        metrics_path = _find_metrics(output_dir)
        hashes = {}
        if metrics_path is None or not status_path.exists() or not config_path.exists():
            records.append(
                RunRecord(
                    task,
                    "missing",
                    "artifact absent",
                    metrics_path,
                    None,
                    expected,
                    hashes,
                )
            )
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            resolved_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            hashes = {
                "metrics_sha256": _file_sha256(metrics_path),
                "resolved_config_sha256": _file_sha256(config_path),
                "status_sha256": _file_sha256(status_path),
            }
        except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
            records.append(
                RunRecord(
                    task, "invalid", str(exc), metrics_path, None, expected, hashes
                )
            )
            continue
        if status.get("status") != "completed":
            records.append(
                RunRecord(
                    task,
                    "invalid",
                    "orchestration not completed",
                    metrics_path,
                    payload,
                    expected,
                    hashes,
                )
            )
            continue
        errors = _critical_protocol_errors(
            payload,
            expected,
            task,
            status,
            resolved_payload,
            config_path,
            metrics_path,
        )
        records.append(
            RunRecord(
                task,
                "complete" if not errors else "invalid",
                "; ".join(errors),
                metrics_path,
                payload,
                expected,
                hashes,
            )
        )
    return records


def _phase_records(records: Sequence[RunRecord], phase_id: str) -> list[RunRecord]:
    return [record for record in records if record.task.phase_id == phase_id]


def _cp_upper(successes: int, trials: int, alpha: float = 0.025) -> float:
    """Exact one-sided Clopper-Pearson upper endpoint by bisection."""

    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    if successes == 0:
        return 1.0 - alpha ** (1.0 / trials)
    if successes == trials:
        return 1.0

    def cdf(probability: float) -> float:
        return sum(
            math.comb(trials, index)
            * probability**index
            * (1.0 - probability) ** (trials - index)
            for index in range(successes + 1)
        )

    low, high = 0.0, 1.0
    for _ in range(120):
        midpoint = (low + high) / 2.0
        if cdf(midpoint) > alpha:
            low = midpoint
        else:
            high = midpoint
    return high


_T975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    15: 2.131,
    20: 2.086,
    24: 2.064,
    30: 2.042,
    35: 2.030,
    40: 2.021,
    60: 2.000,
    120: 1.980,
}


def _t975(df: int) -> float:
    eligible = [key for key in _T975 if key <= df]
    return _T975[max(eligible)] if eligible else math.inf


def _interval95(values: Sequence[float]) -> tuple[float, float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return math.nan, -math.inf, math.inf
    mean = statistics.fmean(finite)
    if len(finite) == 1:
        return mean, -math.inf, math.inf
    margin = _t975(len(finite) - 1) * statistics.stdev(finite) / math.sqrt(len(finite))
    return mean, mean - margin, mean + margin


def _artifact_evidence(records: Sequence[RunRecord]) -> dict[str, Any]:
    manifest = [
        {
            "run_id": record.task.run_id,
            "seed": record.task.seed,
            **record.artifact_hashes,
        }
        for record in sorted(records, key=lambda item: item.task.run_id)
        if record.artifact_hashes
    ]
    return {
        "artifact_manifest_count": len(manifest),
        "artifact_manifest_sha256": _canonical_hash(manifest),
    }


def _common_evidence(records: Sequence[RunRecord]) -> dict[str, Any]:
    total = len(records)
    complete = [record for record in records if record.status == "complete"]
    rcig_rows = [
        row
        for record in complete
        if str(
            record.expected["training"]["algo_config"].get("robust_reference", "")
        ).lower()
        in RCIG_REFERENCES
        for row in record.rounds
    ]
    all_rows = [row for record in complete for row in record.rounds]
    evidence = {
        "complete_fraction": len(complete) / total if total else 0.0,
        "invalid_runs": float(sum(record.status == "invalid" for record in records)),
        "rcig_private_gradient_mps_fraction": (
            sum(
                _finite(row.get("ldp_gradient_far_private_gradient_mps_fraction"))
                == 1.0
                and str(row.get("ldp_gradient_far_private_compute_device", ""))
                in {"mps", "mps:0"}
                for row in all_rows
            )
            / len(all_rows)
            if all_rows
            else 0.0
        ),
        "server_cpu_float64_postprocess_fraction": (
            sum(
                str(row.get("rcig_server_aggregation_device", "")) == "cpu"
                and str(row.get("rcig_server_aggregation_dtype", ""))
                in {"torch.float64", "float64"}
                for row in rcig_rows
            )
            / len(rcig_rows)
            if rcig_rows
            else 0.0
        ),
    }
    evidence.update(_artifact_evidence(records))
    return evidence


def _causality_evidence(records: Sequence[RunRecord]) -> dict[str, float]:
    ready = [
        row
        for record in records
        if record.status == "complete"
        for row in _ready_rows(record)
    ]
    return {
        "strict_past_fraction": (
            sum(_strict_past(row) for row in ready) / len(ready) if ready else 0.0
        ),
        "covariance_psd_fraction": (
            sum(
                _bool01(row.get("rcig_covariance_psd_certified")) == 1.0
                for row in ready
            )
            / len(ready)
            if ready
            else 0.0
        ),
        "nominal_covariance_provenance_fraction": (
            sum(
                row.get("rcig_public_variance_provenance")
                == "authenticated_public_mechanism"
                and _bool01(row.get("rcig_public_variance_registry_verified")) == 1.0
                and row.get("rcig_public_variance_source")
                == "server_config_and_authenticated_client_id"
                and _bool01(
                    row.get("rcig_client_variance_metadata_used_for_construction")
                )
                == 0.0
                and _bool01(
                    row.get("rcig_client_variance_metadata_consistency_checked")
                )
                == 1.0
                and _bool01(
                    row.get("rcig_post_server_clip_covariance_is_delta_method_proxy")
                )
                == 1.0
                and _bool01(row.get("rcig_diagnostics_use_attack_labels")) == 0.0
                and _bool01(row.get("rcig_diagnostics_use_realised_noise")) == 0.0
                for row in ready
            )
            / len(ready)
            if ready
            else 0.0
        ),
    }


def _calibration_evidence(records: Sequence[RunRecord]) -> dict[str, Any]:
    evidence = _common_evidence(records)
    evidence.update(_causality_evidence(records))
    maxima: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        if record.status != "complete":
            continue
        regime = record.task.axis_values["noise_regime"]
        ready = _ready_rows(record)
        for mode in RCIG_MODES:
            values = [_mode_value(row, mode, "innovation_stat") for row in ready]
            finite = [value for value in values if value is not None]
            if finite:
                maxima[(regime, mode)].append(max(finite))
    thresholds: dict[str, dict[str, float]] = {}
    cell_counts: dict[str, int] = {}
    finite_cells = 0
    for regime in ("homogeneous", "heteroscedastic"):
        thresholds[regime] = {}
        for mode in RCIG_MODES:
            values = maxima[(regime, mode)]
            cell_counts[f"{regime}/{mode}"] = len(values)
            if values:
                threshold = max(values)
                thresholds[regime][mode] = threshold
                finite_cells += int(math.isfinite(threshold) and threshold > 0.0)
    evidence.update(
        {
            "finite_threshold_fraction": finite_cells / 6.0,
            "calibration_blocks_per_cell_min": float(
                min(cell_counts.values(), default=0)
            ),
            "simultaneous_distribution_free_confidence_lower": 1.0 - 6.0 * (0.90**53),
            "thresholds_by_regime_and_mode": thresholds,
            "threshold_statistical_unit_count": cell_counts,
            "threshold_operator": "maximum_of_seed_level_post_warmup_maxima",
            "population_null_coverage": 0.90,
            "bonferroni_cells": 6,
        }
    )
    return evidence


def _mean_squared_error(rows: Sequence[Mapping[str, Any]], prefix: str) -> float | None:
    """Average already-squared L2 errors; never square a norm ambiguously."""

    values = [_finite(row.get(f"{prefix}_{ERROR_SUFFIX}")) for row in rows]
    finite = [value for value in values if value is not None]
    return statistics.fmean(finite) if finite else None


def _null_evidence(records: Sequence[RunRecord]) -> dict[str, Any]:
    evidence = _common_evidence(records)
    evidence.update(_causality_evidence(records))
    by_seed: dict[int, list[RunRecord]] = defaultdict(list)
    for record in records:
        if record.status == "complete":
            by_seed[record.task.seed].append(record)
    global_events = 0
    worst_losses: list[float] = []
    details: dict[str, Any] = {}
    valid_blocks = 0
    for seed, block in sorted(by_seed.items()):
        if {record.task.axis_values["noise_regime"] for record in block} != {
            "homogeneous",
            "heteroscedastic",
        }:
            continue
        valid_blocks += 1
        event = False
        losses = []
        for record in block:
            ready = _ready_rows(record)
            event = event or any(
                (_mode_value(row, mode, "gate_active") or 0.0) >= 0.5
                for row in ready
                for mode in RCIG_MODES
            )
            full = _mean_squared_error(ready, "rcig_reference")
            identity = _mean_squared_error(ready, "rcig_identity_new")
            if full is not None and identity is not None:
                losses.append(full / max(identity, 1e-30) - 1.0)
        global_events += int(event)
        if len(losses) == 2:
            worst_losses.append(max(losses))
        details[str(seed)] = {
            "global_union_activation": event,
            "worst_regime_loss": max(losses) if losses else None,
        }
    mean, low, upper = _interval95(worst_losses)
    evidence.update(
        {
            "paired_seed_blocks": float(valid_blocks),
            "global_union_false_activation_count": float(global_events),
            "global_union_cp97_5_upper": (
                _cp_upper(global_events, valid_blocks) if valid_blocks else 1.0
            ),
            "no_attack_reference_loss_mean": mean,
            "no_attack_reference_loss_two_sided_95_low": low,
            "no_attack_reference_loss_one_sided_97_5_upper": upper,
            "seed_block_details": details,
        }
    )
    return evidence


def _mechanism_seed_summary(record: RunRecord) -> dict[str, float] | None:
    attacked = [
        row
        for row in _ready_rows(record)
        if _attack_active(record.expected, int(row["round_num"]))
    ]
    deployed = _mean_squared_error(attacked, "rcig_reference")
    full = _mean_squared_error(attacked, "rcig_full")
    identity = _mean_squared_error(attacked, "rcig_identity_new")
    isotropic = _mean_squared_error(attacked, "rcig_isotropic")
    euclidean = _mean_squared_error(attacked, "rcig_euclidean")
    gate = [_mode_value(row, "full", "gate_active") for row in attacked]
    gate_values = [value for value in gate if value is not None]
    anisotropy = [
        value
        for row in attacked
        if (value := _finite(row.get("rcig_max_covariance_anisotropy_ratio")))
        is not None
    ]
    if (
        None in {deployed, full, identity, isotropic, euclidean}
        or not gate_values
        or not anisotropy
    ):
        return None
    assert deployed is not None
    assert full is not None and identity is not None
    assert isotropic is not None and euclidean is not None
    return {
        "gain_vs_identity": 1.0 - deployed / max(identity, 1e-30),
        "gain_vs_isotropic": 1.0 - full / max(isotropic, 1e-30),
        "gain_vs_euclidean": 1.0 - full / max(euclidean, 1e-30),
        "gate_activation_rate": statistics.fmean(gate_values),
        "anisotropy_median": statistics.median(anisotropy),
        "attacked_ready_rounds": float(len(attacked)),
    }


def _mechanistic_evidence(records: Sequence[RunRecord]) -> dict[str, Any]:
    evidence = _common_evidence(records)
    evidence.update(_causality_evidence(records))
    grouped: dict[tuple[str, str, str], list[tuple[int, dict[str, float]]]] = (
        defaultdict(list)
    )
    for record in records:
        if record.status != "complete":
            continue
        summary = _mechanism_seed_summary(record)
        if summary is None:
            continue
        key = (
            record.task.axis_values["noise_regime"],
            record.task.axis_values["attack"],
            record.task.axis_values["schedule"],
        )
        grouped[key].append((record.task.seed, summary))

    primary = {
        key: values
        for key, values in grouped.items()
        if key[2] == "persistent_after_clean_warmup"
    }
    cell_details: dict[str, Any] = {}
    gain_successes = []
    gate_successes = []
    euclidean_lows = []
    isotropic_lows = []
    anisotropy_flags: list[bool] = []
    for key, values in sorted(grouped.items()):
        gains = [item[1]["gain_vs_identity"] for item in values]
        iso = [item[1]["gain_vs_isotropic"] for item in values]
        euclidean = [item[1]["gain_vs_euclidean"] for item in values]
        gate_rates = [item[1]["gate_activation_rate"] for item in values]
        anisotropy = [item[1]["anisotropy_median"] for item in values]
        gain_interval = _interval95(gains)
        iso_interval = _interval95(iso)
        euclidean_interval = _interval95(euclidean)
        name = "/".join(key)
        cell_details[name] = {
            "seed_count": len(values),
            "mse_gain_gt_10pct_seeds": sum(value > 0.10 for value in gains),
            "gate_rate_ge_50pct_seeds": sum(value >= 0.50 for value in gate_rates),
            "gain_vs_identity_mean_ci95": gain_interval,
            "gain_vs_isotropic_mean_ci95": iso_interval,
            "gain_vs_euclidean_mean_ci95": euclidean_interval,
            "anisotropy_identifiable_seeds": sum(value >= 1.10 for value in anisotropy),
            "seed_summaries": {str(seed): summary for seed, summary in values},
        }
        if key in primary:
            gain_successes.append(sum(value > 0.10 for value in gains))
            gate_successes.append(sum(value >= 0.50 for value in gate_rates))
            euclidean_lows.append(euclidean_interval[1])
            if key[0] == "heteroscedastic":
                anisotropy_flags.extend(value >= 1.10 for value in anisotropy)
                isotropic_lows.append(iso_interval[1])
    evidence.update(
        {
            "primary_cell_count": float(len(primary)),
            "min_primary_cell_seed_gain_successes": float(
                min(gain_successes, default=0)
            ),
            "min_primary_cell_gate_successes": float(min(gate_successes, default=0)),
            "min_primary_full_vs_euclidean_ci_low": min(
                euclidean_lows, default=-math.inf
            ),
            "heteroscedastic_anisotropy_identifiable_fraction": (
                sum(anisotropy_flags) / len(anisotropy_flags)
                if anisotropy_flags
                else 0.0
            ),
            "min_heteroscedastic_full_vs_isotropic_ci_low": min(
                isotropic_lows, default=-math.inf
            ),
            "mse_definition": "mean_of_squared_l2_norm_errors_within_seed_cell",
            "primary_mse_object": "deployed_freeze_hysteresis_reference",
            "mode_comparison_object": "instantaneous_pre_policy_counterfactual_candidates_on_same_transcript",
            "rounds_are_not_inference_units": True,
            "cell_details": cell_details,
        }
    )
    return evidence


def _final_metrics(record: RunRecord) -> dict[str, Any]:
    row = record.rounds[-1] if record.rounds else {}

    def pct_from_fraction(key: str) -> float | None:
        value = _finite(row.get(key))
        return None if value is None else 100.0 * value

    attacked_rows = [
        item
        for item in record.rounds
        if _attack_active(record.expected, int(item.get("round_num", -1)))
    ]
    byz = [_finite(item.get("byzantine_weight_mass_oracle")) for item in attacked_rows]
    byz = [value for value in byz if value is not None]
    gate = [_bool01(item.get("rcig_gate_active")) for item in attacked_rows]
    gate = [value for value in gate if value is not None]
    frozen = [
        _bool01(item.get("rcig_persistent_frozen_after_commit"))
        for item in attacked_rows
    ]
    frozen = [value for value in frozen if value is not None]
    return {
        "phase": record.task.phase_id,
        "seed": record.task.seed,
        **record.task.axis_values,
        "status": record.status,
        "reason": record.reason,
        "test_accuracy_pct": pct_from_fraction("test_accuracy"),
        "client_accuracy_pct": pct_from_fraction("client_accuracy_mean"),
        "test_loss": _finite(row.get("test_loss")),
        "variance_pp2": _finite(row.get("client_accuracy_variance_pct2")),
        "worst20_pct": _finite(row.get("worst20_accuracy_pct")),
        "gap_pp": _finite(row.get("best20_worst20_gap_pct")),
        "epsilon": _finite(row.get("privacy_epsilon_max")),
        "delta": _finite(row.get("privacy_delta")),
        "privacy_noise_multiplier_min": _finite(
            row.get("privacy_model_noise_multiplier_min")
        ),
        "privacy_noise_multiplier_max": _finite(
            row.get("privacy_model_noise_multiplier_max")
        ),
        "local_clip_norm": _finite(
            record.expected["training"]["algo_config"].get("clip_norm")
        ),
        "server_clip_norm": _finite(
            record.expected["training"]["algo_config"].get("far_server_clip_norm")
        ),
        "local_clip_rate": _finite(row.get("privacy_clip_rate_mean")),
        "server_clip_rate": _finite(row.get("far_server_clip_rate")),
        "far_max_weight": _finite(row.get("far_max_weight")),
        "far_weight_concentration": _finite(
            row.get("far_noise_amplification_vs_uniform")
        ),
        "byzantine_weight_mass_mean_attacked": statistics.fmean(byz) if byz else None,
        "rcig_gate_activation_rate_attacked": statistics.fmean(gate) if gate else None,
        "rcig_frozen_fraction_attacked": statistics.fmean(frozen) if frozen else None,
        "recovery_latency_rounds": None,
    }


def _sign_counts(values: Sequence[float]) -> dict[str, int]:
    tolerance = 1e-12
    return {
        "positive": sum(value > tolerance for value in values),
        "negative": sum(value < -tolerance for value in values),
        "ties": sum(abs(value) <= tolerance for value in values),
    }


def _paired_deltas(
    rows: Sequence[Mapping[str, Any]],
    *,
    regime: str,
    schedule: str,
) -> dict[str, Any] | None:
    selected = [
        row
        for row in rows
        if row.get("noise_regime") == regime and row.get("schedule") == schedule
    ]
    by_key = {(int(row["seed"]), str(row["reference"])): row for row in selected}
    seeds = sorted({int(row["seed"]) for row in selected})
    metrics = {
        "test_accuracy_delta_pp": ("test_accuracy_pct", 1.0),
        "worst20_delta_pp": ("worst20_pct", 1.0),
        "gap_reduction_pp": ("gap_pp", -1.0),
    }
    values: dict[str, list[float]] = {name: [] for name in metrics}
    paired_seeds = []
    for seed in seeds:
        rcig = by_key.get((seed, "rcig_full"))
        rfa = by_key.get((seed, "rfa"))
        if rcig is None or rfa is None:
            continue
        current: dict[str, float] = {}
        valid = True
        for name, (field, direction) in metrics.items():
            left, right = _finite(rcig.get(field)), _finite(rfa.get(field))
            if left is None or right is None:
                valid = False
                break
            current[name] = direction * (left - right)
        if valid:
            paired_seeds.append(seed)
            for name, value in current.items():
                values[name].append(value)
    if not paired_seeds:
        return None
    result: dict[str, Any] = {"paired_seeds": paired_seeds, "n": len(paired_seeds)}
    for name, numbers in values.items():
        result[name] = {
            "mean_ci95": _interval95(numbers),
            "signs": _sign_counts(numbers),
            "values_by_seed": dict(zip(map(str, paired_seeds), numbers)),
        }
    return result


def _confirmation_evidence(records: Sequence[RunRecord]) -> dict[str, Any]:
    evidence = _common_evidence(records)
    complete_rows = [
        _final_metrics(record) for record in records if record.status == "complete"
    ]
    epsilon_errors = [
        abs(value - 4.0)
        for row in complete_rows
        if (value := _finite(row.get("epsilon"))) is not None
    ]
    comparisons: dict[str, Any] = {}
    clean_lows = []
    attacked_test_means = []
    attacked_worst_means = []
    attacked_gap_means = []
    attacked_test_lows = []
    attacked_worst_lows = []
    attacked_gap_lows = []
    clean_count = 0
    attack_count = 0
    schedules = (
        "bf_x10_persistent_after_clean_warmup",
        "ipm_persistent_after_clean_warmup",
        "alie_persistent_after_clean_warmup",
    )
    for regime in ("homogeneous", "heteroscedastic"):
        clean = _paired_deltas(complete_rows, regime=regime, schedule="none")
        if clean is not None and clean["n"] == 12:
            comparisons[f"{regime}/none"] = clean
            clean_count += 1
            clean_lows.append(clean["test_accuracy_delta_pp"]["mean_ci95"][1])
        for schedule in schedules:
            comparison = _paired_deltas(complete_rows, regime=regime, schedule=schedule)
            if comparison is None or comparison["n"] != 12:
                continue
            comparisons[f"{regime}/{schedule}"] = comparison
            attack_count += 1
            attacked_test_means.append(
                comparison["test_accuracy_delta_pp"]["mean_ci95"][0]
            )
            attacked_test_lows.append(
                comparison["test_accuracy_delta_pp"]["mean_ci95"][1]
            )
            attacked_worst_means.append(comparison["worst20_delta_pp"]["mean_ci95"][0])
            attacked_worst_lows.append(comparison["worst20_delta_pp"]["mean_ci95"][1])
            attacked_gap_means.append(comparison["gap_reduction_pp"]["mean_ci95"][0])
            attacked_gap_lows.append(comparison["gap_reduction_pp"]["mean_ci95"][1])
    evidence.update(
        {
            "max_abs_epsilon_error": max(epsilon_errors, default=math.inf),
            "clean_noise_cell_count": float(clean_count),
            "min_clean_rcig_vs_rfa_test_accuracy_ci_low_pp": min(
                clean_lows, default=-math.inf
            ),
            "attacked_primary_cell_count": float(attack_count),
            "min_attacked_rcig_vs_rfa_test_accuracy_mean_delta_pp": min(
                attacked_test_means, default=-math.inf
            ),
            "min_attacked_rcig_vs_rfa_worst20_mean_delta_pp": min(
                attacked_worst_means, default=-math.inf
            ),
            "min_attacked_rcig_vs_rfa_gap_mean_reduction_pp": min(
                attacked_gap_means, default=-math.inf
            ),
            "min_attacked_rcig_vs_rfa_test_accuracy_ci_low_pp": min(
                attacked_test_lows, default=-math.inf
            ),
            "min_attacked_rcig_vs_rfa_worst20_ci_low_pp": min(
                attacked_worst_lows, default=-math.inf
            ),
            "min_attacked_rcig_vs_rfa_gap_reduction_ci_low_pp": min(
                attacked_gap_lows, default=-math.inf
            ),
            "primary_comparator": "rcig_full_minus_rfa",
            "paired_comparisons": comparisons,
            "recovery_latency_identifiable": False,
            "recovery_latency_reason": "all attacked R3 schedules remain active through round 40",
        }
    )
    return evidence


def evaluate_gate(campaign: RCIGV2Campaign, phase_id: str) -> dict[str, Any]:
    verify_existing_campaign_lock(campaign)
    records = _phase_records(collect_runs(campaign), phase_id)
    if phase_id == "r0_dynamic_calibration":
        return _calibration_evidence(records)
    if phase_id == "r1_dynamic_null":
        return _null_evidence(records)
    if phase_id == "r2_attack_mechanism":
        return _mechanistic_evidence(records)
    if phase_id == "r3_e2e_confirmation":
        return _confirmation_evidence(records)
    raise ValueError(f"phase {phase_id!r} has no preregistered v2 evaluator")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean_sd(values: Iterable[Any]) -> str:
    finite = [value for item in values if (value := _finite(item)) is not None]
    if not finite:
        return "—"
    if len(finite) == 1:
        return f"{finite[0]:.3f}"
    return f"{statistics.fmean(finite):.3f} ± {statistics.stdev(finite):.3f}"


def _fmt_number(value: Any, *, digits: int = 3) -> str:
    number = _finite(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _fmt_interval(value: Any, *, digits: int = 3) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return "—"
    mean, low, high = (_finite(item) for item in value)
    if mean is None or low is None or high is None:
        return "—"
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _gate_preview(
    campaign: RCIGV2Campaign, phase_id: str, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute a preregistered decision and audit any immutable artifact."""

    criteria = phase_definition(campaign, phase_id).get("gate_criteria", [])
    evaluations = []
    for criterion in criteria:
        identifier = str(criterion["id"])
        value = evidence.get(identifier)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"non-numeric evidence for {phase_id}/{identifier}")
        observed = float(value)
        if not math.isfinite(observed):
            raise ValueError(f"non-finite evidence for {phase_id}/{identifier}")
        operator = str(criterion["op"])
        threshold = float(criterion["threshold"])
        if operator == "==":
            passed = math.isclose(observed, threshold, abs_tol=1e-12)
        elif operator == "<=":
            passed = observed <= threshold
        elif operator == ">=":
            passed = observed >= threshold
        else:  # pragma: no cover - matrix validation rejects this first
            raise ValueError(f"unsupported gate operator {operator!r}")
        evaluations.append(
            {
                "id": identifier,
                "observed": observed,
                "op": operator,
                "threshold": threshold,
                "passed": passed,
            }
        )
    decision = (
        "promote"
        if evaluations and all(row["passed"] for row in evaluations)
        else "stop"
    )
    artifact = gate_path(campaign, phase_id)
    artifact_status = "not_recorded"
    artifact_sha256 = None
    if artifact.exists():
        artifact_sha256 = _file_sha256(artifact)
        try:
            payload = _verified_gate_payload(campaign, phase_id)
            if payload.get("evidence_sha256") != _canonical_hash(dict(evidence)):
                artifact_status = "verified_but_stale_against_current_artifacts"
            elif payload.get("decision") != decision:
                artifact_status = "verified_but_decision_mismatch"
            else:
                artifact_status = "verified_and_current"
        except (OSError, json.JSONDecodeError, RuntimeError, TypeError, ValueError):
            artifact_status = "invalid"
    return {
        "decision_if_recorded_now": decision,
        "evaluations": evaluations,
        "gate_artifact_status": artifact_status,
        "gate_artifact_sha256": artifact_sha256,
    }


def analyze_campaign(
    campaign: RCIGV2Campaign,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    verify_existing_campaign_lock(campaign)
    records = collect_runs(campaign)
    run_rows = [_final_metrics(record) for record in records]
    _write_csv(output_dir / "run_level.csv", run_rows)
    phase_rows = []
    phase_evidence: dict[str, Any] = {}
    gate_rows: list[dict[str, Any]] = []
    evaluators = {
        "r0_dynamic_calibration": _calibration_evidence,
        "r1_dynamic_null": _null_evidence,
        "r2_attack_mechanism": _mechanistic_evidence,
        "r3_e2e_confirmation": _confirmation_evidence,
    }
    for phase in campaign.phases:
        phase_id = str(phase["id"])
        cell = _phase_records(records, phase_id)
        counts = {
            "phase": phase_id,
            "expected": len(cell),
            "complete": sum(row.status == "complete" for row in cell),
            "invalid": sum(row.status == "invalid" for row in cell),
            "missing": sum(row.status == "missing" for row in cell),
        }
        if counts["complete"] == counts["expected"] and not (
            counts["invalid"] or counts["missing"]
        ):
            evidence = evaluators[phase_id](cell)
            preview = _gate_preview(campaign, phase_id, evidence)
            phase_evidence[phase_id] = {"evidence": evidence, **preview}
            counts["decision"] = preview["decision_if_recorded_now"]
            counts["gate_artifact"] = preview["gate_artifact_status"]
            for evaluation in preview["evaluations"]:
                gate_rows.append({"phase": phase_id, **evaluation})
        else:
            phase_evidence[phase_id] = {
                "not_evaluated": True,
                "reason": "phase incomplete or contains invalid artifacts",
            }
            counts["decision"] = "not_evaluated"
            counts["gate_artifact"] = (
                "present_but_not_usable"
                if gate_path(campaign, phase_id).exists()
                else "not_recorded"
            )
        phase_rows.append(counts)
    _write_csv(output_dir / "phase_status.csv", phase_rows)
    if gate_rows:
        _write_csv(output_dir / "gate_criteria.csv", gate_rows)
    else:
        (output_dir / "gate_criteria.csv").write_text("", encoding="utf-8")
    e2e = [
        row
        for row in run_rows
        if row["phase"] == "r3_e2e_confirmation" and row["status"] == "complete"
    ]
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in e2e:
        groups[
            (
                str(row.get("noise_regime")),
                str(row.get("reference")),
                str(row.get("schedule")),
            )
        ].append(row)
    summaries = []
    for (regime, reference, schedule), cell in sorted(groups.items()):
        summaries.append(
            {
                "noise_regime": regime,
                "reference": reference,
                "schedule": schedule,
                "n": len(cell),
                "test_accuracy_pct": _mean_sd(row["test_accuracy_pct"] for row in cell),
                "client_accuracy_pct": _mean_sd(
                    row["client_accuracy_pct"] for row in cell
                ),
                "test_loss": _mean_sd(row["test_loss"] for row in cell),
                "variance_pp2": _mean_sd(row["variance_pp2"] for row in cell),
                "worst20_pct": _mean_sd(row["worst20_pct"] for row in cell),
                "gap_pp": _mean_sd(row["gap_pp"] for row in cell),
                "epsilon": _mean_sd(row["epsilon"] for row in cell),
                "delta": _mean_sd(row["delta"] for row in cell),
                "noise_multiplier_min": _mean_sd(
                    row["privacy_noise_multiplier_min"] for row in cell
                ),
                "noise_multiplier_max": _mean_sd(
                    row["privacy_noise_multiplier_max"] for row in cell
                ),
                "local_clip_norm": _mean_sd(row["local_clip_norm"] for row in cell),
                "server_clip_norm": _mean_sd(row["server_clip_norm"] for row in cell),
                "far_max_weight": _mean_sd(row["far_max_weight"] for row in cell),
                "far_weight_concentration": _mean_sd(
                    row["far_weight_concentration"] for row in cell
                ),
                "local_clip_rate": _mean_sd(row["local_clip_rate"] for row in cell),
                "server_clip_rate": _mean_sd(row["server_clip_rate"] for row in cell),
                "byzantine_weight_mass": _mean_sd(
                    row["byzantine_weight_mass_mean_attacked"] for row in cell
                ),
                "rcig_gate_activation": _mean_sd(
                    row["rcig_gate_activation_rate_attacked"] for row in cell
                ),
                "rcig_frozen_fraction": _mean_sd(
                    row["rcig_frozen_fraction_attacked"] for row in cell
                ),
                "recovery_latency_rounds": "not_identifiable",
            }
        )
    _write_csv(output_dir / "end_to_end_summary.csv", summaries)
    state = {
        "campaign_id": campaign.matrix["campaign_id"],
        "campaign_scientific_hash": campaign_scientific_hash(campaign),
        "counts": {
            "expected": len(records),
            "complete": sum(record.status == "complete" for record in records),
            "invalid": sum(record.status == "invalid" for record in records),
            "missing": sum(record.status == "missing" for record in records),
        },
        "phase_status": phase_rows,
        "phase_evidence": phase_evidence,
        "end_to_end_summary": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# RCIG dans LDP-Gradient-FAR — rapport v2",
        "",
        f"Hash scientifique : `{campaign_scientific_hash(campaign)}`.",
        "",
        "## État fail-closed",
        "",
        "Une phase incomplète n'est jamais évaluée et aucune valeur manquante n'est imputée.",
        "",
        "| Phase | Attendus | Complets | Invalides | Manquants | Décision recalculée | Artefact de gate |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in phase_rows:
        lines.append(
            f"| {row['phase']} | {row['expected']} | {row['complete']} | {row['invalid']} | {row['missing']} | "
            f"{row['decision']} | {row['gate_artifact']} |"
        )

    invalid = [record for record in records if record.status == "invalid"]
    if invalid:
        lines.extend(
            [
                "",
                "### Artefacts invalides",
                "",
                "| Phase | Run | Motif |",
                "|---|---|---|",
            ]
        )
        for record in invalid:
            reason = record.reason.replace("|", "\\|")
            lines.append(
                f"| {record.task.phase_id} | {record.task.run_id} | {reason} |"
            )

    r0 = phase_evidence.get("r0_dynamic_calibration", {})
    if not r0.get("not_evaluated"):
        evidence = r0["evidence"]
        lines.extend(
            [
                "",
                "## R0 — seuils de calibration figés",
                "",
                "Le seuil de chaque cellule est le maximum des maxima post-warmup de 53 seeds. Ce n'est pas un quantile empirique. La borne simultanée distribution-free est "
                f"`{_fmt_number(evidence['simultaneous_distribution_free_confidence_lower'], digits=6)}`.",
                "",
                "| Régime | Mode | Seuil maximum | Blocs |",
                "|---|---|---:|---:|",
            ]
        )
        thresholds = evidence["thresholds_by_regime_and_mode"]
        counts = evidence["threshold_statistical_unit_count"]
        for regime in ("homogeneous", "heteroscedastic"):
            for mode in RCIG_MODES:
                lines.append(
                    f"| {regime} | {mode} | {_fmt_number(thresholds[regime][mode], digits=6)} | "
                    f"{counts[f'{regime}/{mode}']} |"
                )

    r1 = phase_evidence.get("r1_dynamic_null", {})
    if not r1.get("not_evaluated"):
        evidence = r1["evidence"]
        lines.extend(
            [
                "",
                "## R1 — validation nulle indépendante",
                "",
                "L'événement est l'union globale des activations sur les six cellules d'une seed. La perte de référence est calculée par seed, puis maximisée sur les deux régimes avant l'intervalle : aucun tour n'est traité comme une répétition indépendante.",
                "",
                "| Quantité | Valeur |",
                "|---|---:|",
                f"| Blocs appariés | {int(evidence['paired_seed_blocks'])} |",
                f"| Activations union-globales | {int(evidence['global_union_false_activation_count'])} |",
                f"| Borne CP unilatérale 97,5 % | {_fmt_number(evidence['global_union_cp97_5_upper'], digits=6)} |",
                f"| Perte moyenne de référence | {_fmt_number(evidence['no_attack_reference_loss_mean'], digits=6)} |",
                f"| Borne supérieure unilatérale 97,5 % de la perte | {_fmt_number(evidence['no_attack_reference_loss_one_sided_97_5_upper'], digits=6)} |",
            ]
        )

    r2 = phase_evidence.get("r2_attack_mechanism", {})
    if not r2.get("not_evaluated"):
        evidence = r2["evidence"]
        lines.extend(
            [
                "",
                "## R2 — mécanisme sur gradients privés et attaques persistantes",
                "",
                "Le gain primaire compare la référence effectivement déployée après `freeze_hysteresis` à la vue nouvelle. Les comparaisons full/isotropic/euclidean sont des candidats instantanés pré-politique sur le même transcript, pas trois trajectoires d'entraînement.",
                "",
                "| Cellule | Seeds | Gain >10 % | Gate ≥50 % | Gain déployé vs nouvelle, IC95 | Full vs isotropic, IC95 | Full vs euclidean, IC95 | Anisotropie identifiable |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, detail in sorted(evidence["cell_details"].items()):
            lines.append(
                f"| {name} | {detail['seed_count']} | {detail['mse_gain_gt_10pct_seeds']} | "
                f"{detail['gate_rate_ge_50pct_seeds']} | {_fmt_interval(detail['gain_vs_identity_mean_ci95'])} | "
                f"{_fmt_interval(detail['gain_vs_isotropic_mean_ci95'])} | "
                f"{_fmt_interval(detail['gain_vs_euclidean_mean_ci95'])} | "
                f"{detail['anisotropy_identifiable_seeds']} |"
            )
        lines.extend(
            [
                "",
                f"Fraction identifiable dans les cellules hétéroscédastiques primaires : `{_fmt_number(evidence['heteroscedastic_anisotropy_identifiable_fraction'], digits=4)}`. Le claim full > isotropic est interdit si ce diagnostic échoue.",
                "Les intervalles et critères R2 sont cellule par cellule. Ils ne contrôlent pas une FWER globale sur toute la famille ; tout verdict mécanistique est conditionnel au protocole préenregistré.",
            ]
        )

    r3 = phase_evidence.get("r3_e2e_confirmation", {})
    if not r3.get("not_evaluated"):
        evidence = r3["evidence"]
        lines.extend(
            [
                "",
                "## R3 — comparaison primaire appariée RCIG-full vs RFA",
                "",
                "Les deltas positifs favorisent RCIG : accuracy et Worst-20 plus élevées, ou gap plus faible. Les intervalles et signes utilisent les 12 seeds appariées de chaque cellule. Les IC95 sont cellule par cellule et ne forment pas une région simultanée ; le verdict reste empirique et conditionnel, sans claim de contrôle FWER global.",
                "",
                "| Cellule | n | Δ Test Acc., IC95 (pp) | signes +/−/= | Δ Worst-20, IC95 (pp) | signes +/−/= | Réduction gap, IC95 (pp) | signes +/−/= |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, comparison in sorted(evidence["paired_comparisons"].items()):
            metric_cells = []
            for metric in (
                "test_accuracy_delta_pp",
                "worst20_delta_pp",
                "gap_reduction_pp",
            ):
                value = comparison[metric]
                signs = value["signs"]
                metric_cells.extend(
                    [
                        _fmt_interval(value["mean_ci95"]),
                        f"{signs['positive']}/{signs['negative']}/{signs['ties']}",
                    ]
                )
            lines.append(
                f"| {name} | {comparison['n']} | " + " | ".join(metric_cells) + " |"
            )
    lines.extend(
        [
            "",
            "## Résultats end-to-end disponibles",
            "",
            "Chaque cellule est une moyenne ± écart-type entre seeds indépendantes.",
            "",
            "| Bruit | Référence | Calendrier | n | Test Acc. | Client Acc. | Worst-20 | Gap | Var. pp² |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['noise_regime']} | {row['reference']} | {row['schedule']} | {row['n']} | "
            f"{row['test_accuracy_pct']} | {row['client_accuracy_pct']} | {row['worst20_pct']} | "
            f"{row['gap_pp']} | {row['variance_pp2']} |"
        )
    lines.extend(
        [
            "",
            "### Confidentialité, clipping et pondération",
            "",
            "Les taux de clipping sont des fractions. La masse byzantine et l'activation RCIG ne sont définies que pendant les calendriers attaqués. Le temps de récupération est non identifiable, puisque l'attaque ne s'arrête pas avant le tour 40.",
            "",
            "| Bruit | Référence | Calendrier | epsilon | delta | sigma min–max | C local | taux clip local | U serveur | taux clip serveur | poids max | concentration | masse byz. | gate RCIG | gel RCIG |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['noise_regime']} | {row['reference']} | {row['schedule']} | "
            f"{row['epsilon']} | {row['delta']} | {row['noise_multiplier_min']}–{row['noise_multiplier_max']} | "
            f"{row['local_clip_norm']} | {row['local_clip_rate']} | {row['server_clip_norm']} | "
            f"{row['server_clip_rate']} | {row['far_max_weight']} | {row['far_weight_concentration']} | "
            f"{row['byzantine_weight_mass']} | {row['rcig_gate_activation']} | {row['rcig_frozen_fraction']} |"
        )
    lines.extend(
        [
            "",
            "## Lecture obligatoire",
            "",
            "Le calcul du gradient privé est audité sur MPS. L'agrégation serveur est un post-traitement CPU float64 audité séparément. La variance nominale RCIG est reconstruite depuis la configuration immuable et le registre serveur des IDs ; les champs client ne servent qu'à un contrôle de cohérence. Les erreurs de référence suffixées `oracle` sont des métriques d'évaluation et ne pilotent jamais RCIG.",
            "",
            "Les labels d'attaque sont retirés avant `server_aggregate` dans tous les bras. La masse byzantine est jointe aux poids uniquement post-hoc dans le harness et le payload interne est retiré avant persistance.",
            "",
            "Les attaques confirmatoires restent actives jusqu'au tour 40. Le temps de récupération après retrait de l'attaque n'est donc pas identifiable dans R3 et aucune valeur n'est imputée.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--evaluate-gate")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    campaign = load_campaign(args.matrix)
    if args.evaluate_gate:
        evidence = evaluate_gate(campaign, args.evaluate_gate)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / f"{args.evaluate_gate}_gate_evidence.json"
        path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print(path)
        return
    state = analyze_campaign(
        campaign, output_dir=args.output_dir, report_path=args.report
    )
    print(json.dumps(state["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
