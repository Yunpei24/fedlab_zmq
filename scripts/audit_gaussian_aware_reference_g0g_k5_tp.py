#!/usr/bin/env python3
"""Independent post-run audit for the locked G0g-K5-TP experiment.

This module deliberately does not import the K5 runner or call its summary
functions.  It reconstructs lambda selection, ridge solutions, seed-level
estimands, confidence intervals and gates from persisted sufficient
statistics and CSV rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / (
    "configs/ldp_gradient_far/"
    "gaussian_aware_reference_g0g_k5_transcript_predictor.yaml"
)
DEFAULT_LOCK = ROOT / (
    "configs/ldp_gradient_far/"
    "gaussian_aware_reference_g0g_k5_transcript_predictor.lock.json"
)
DEFAULT_RESULTS = ROOT / (
    "results/ldp_gradient_far/"
    "gaussian_aware_reference_g0g_k5_transcript_predictor_mps_v1"
)

K2 = "g0g_k2"
K4 = "g0g_k4_temporal_causal_gate"
K4B = "g0g_k4b_rolling_past_imputation"
K5_1D = "g0g_k5_tp_one_dimensional_control"
K5 = "g0g_k5_tp_shared_scalar_ridge"
K4C = "g0g_k4c_ch_privileged_benchmark"
POINTWISE = "g0g_k4b_pointwise_optimal_oracle"
CANDIDATES = (K2, K4, K4B, K5_1D, K5, K4C, POINTWISE)
FEATURE_NAMES = (
    "rolling_accepted_mean",
    "accepted_mean_t_minus_1",
    "accepted_mean_delta_t_minus_1_t_minus_2",
    "fixed_denominator_direction_t_minus_1",
    "fixed_denominator_delta_t_minus_1_t_minus_2",
    "rolling_fixed_denominator_direction",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _ci(values: Sequence[float], t_critical: float) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    mean = statistics.fmean(numbers)
    sd = statistics.stdev(numbers)
    half = float(t_critical) * sd / math.sqrt(len(numbers))
    return {"n": len(numbers), "mean": mean, "low": mean - half, "high": mean + half}


def _close(left: float, right: float, tolerance: float = 2e-6) -> bool:
    return math.isfinite(left) and math.isfinite(right) and math.isclose(
        left, right, rel_tol=tolerance, abs_tol=tolerance
    )


def _select_lambda(rows: Sequence[Mapping[str, Any]], tolerance: float) -> float:
    values: list[tuple[float, float]] = []
    for row in rows:
        seed_mse = [float(value) for value in row["calibration_seed_mse"]]
        recomputed = statistics.fmean(seed_mse)
        recorded = float(row["equal_seed_mean_projected_aggregate_mse"])
        if not _close(recomputed, recorded, tolerance=1e-10):
            raise RuntimeError("Calibration row mean is not the equal-seed mean")
        values.append((float(row["ridge_lambda"]), recomputed))
    minimum = min(score for _, score in values)
    return max(value for value, score in values if score <= minimum + tolerance)


def _solve_sufficient_statistics(
    diagnostics: Mapping[str, Any], device: torch.device
) -> torch.Tensor:
    gram = torch.tensor(diagnostics["normalized_gram"], dtype=torch.float32, device=device)
    rhs = torch.tensor(diagnostics["normalized_rhs"], dtype=torch.float32, device=device)
    ridge_lambda = float(diagnostics["ridge_lambda"])
    system = gram + ridge_lambda * torch.eye(
        int(gram.shape[0]), dtype=torch.float32, device=device
    )
    return torch.linalg.solve(system, rhs)


def _expected_history_ids(config: Mapping[str, Any], split: str) -> set[str]:
    seed_key = {
        "train": "train_outer_seeds",
        "calibration": "calibration_outer_seeds",
        "evaluation": "evaluation_outer_seeds",
    }[split]
    result: set[str] = set()
    for seed in config["randomness"][seed_key]:
        for regime in config["privacy_noise"]["regimes"]:
            for permutation in regime["permutations"]:
                for geometry in config["cohort"]["honest_outliers"]["geometries"]:
                    for dynamics in config["honest_dynamics"]["names"]:
                        for threat in config["threats"]["names"]:
                            for round_index in config["temporal"]["assessment_rounds"]:
                                result.add(
                                    "|".join(
                                        (
                                            split,
                                            str(seed),
                                            str(regime["name"]),
                                            str(permutation),
                                            str(geometry),
                                            str(dynamics),
                                            str(threat),
                                            str(round_index),
                                        )
                                    )
                                )
    return result


def _verify_lock(config: Mapping[str, Any], lock_path: Path) -> dict[str, Any]:
    lock = _read_json(lock_path)
    if lock["campaign_id"] != config["campaign_id"] or lock["schema_version"] != 1:
        raise RuntimeError("Independent audit found a lock identity mismatch")
    mismatches: list[str] = []
    for section in ("locked_files", "dependencies"):
        for relative, expected in lock[section].items():
            path = ROOT / relative
            if not path.is_file() or _sha256(path) != str(expected):
                mismatches.append(relative)
    return {
        "pass": not mismatches,
        "mismatches": mismatches,
        "lock_sha256": _sha256(lock_path),
    }


def _audit_fit(
    config: Mapping[str, Any], results: Path, device: torch.device, lock_audit: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = _read_json(results / "manifest.json")
    decision = _read_json(results / "fit_decision.json")
    predictor = _read_json(results / "frozen_predictor.json")
    calibration = _read_json(results / "lambda_calibration.json")
    design = _read_json(results / "feature_design_diagnostics.json")
    registry = _read_json(results / "fit_rng_registry.json")
    fit_rows = _read_csv(results / "fit_history_rows.csv")
    predictor_sha = _sha256(results / "frozen_predictor.json")
    tolerance = float(calibration["tie_absolute_tolerance"])
    selected = _select_lambda(calibration["primary"], tolerance)
    selected_1d = _select_lambda(calibration["one_dimensional"], tolerance)
    recomputed_theta = _solve_sufficient_statistics(design["final_fit"], device)
    recomputed_theta_1d = _solve_sufficient_statistics(design["final_1d_fit"], device)
    stored_theta = torch.tensor(predictor["coefficients"], dtype=torch.float32, device=device)
    stored_theta_1d = torch.tensor(
        [predictor["one_dimensional_control"]["coefficient"]],
        dtype=torch.float32,
        device=device,
    )
    fit_generated = [
        int(value)
        for key in ("train_target", "calibration_target", "calibration_evaluation")
        for value in registry[key]
    ]
    train_ids = [row["history_id"] for row in fit_rows if row["split"] == "train"]
    calibration_ids = [
        row["history_id"] for row in fit_rows if row["split"] == "calibration"
    ]
    balanced_train = design["weight_audit"]["balanced_train_sum_by_seed"]
    balanced_refit = design["weight_audit"]["balanced_train_plus_calibration_sum_by_seed"]
    checks = {
        "lock_integrity": bool(lock_audit["pass"]),
        "manifest_fit_complete": manifest["status"] in {
            "fit_completed_evaluation_locked",
            "completed_development",
        },
        "fit_decision_valid": bool(decision["all_validity_checks_pass"]),
        "predictor_hash_consistent": predictor_sha
        == manifest["frozen_predictor_sha256"]
        == decision["frozen_predictor_sha256"],
        "config_hash_consistent": predictor["config_sha256"] == _sha256(DEFAULT_CONFIG),
        "lock_hash_consistent": predictor["lock_sha256"] == lock_audit["lock_sha256"],
        "feature_schema_exact": tuple(predictor["feature_names"]) == FEATURE_NAMES,
        "selected_lambda_reproduced": selected
        == float(predictor["selected_lambda"])
        == float(decision["selected_lambda"]),
        "selected_lambda_1d_reproduced": selected_1d
        == float(predictor["one_dimensional_control"]["selected_lambda"])
        == float(decision["selected_lambda_one_dimensional"]),
        "final_coefficients_reproduced": bool(
            torch.allclose(recomputed_theta, stored_theta, atol=2e-6, rtol=2e-6)
        ),
        "final_1d_coefficient_reproduced": bool(
            torch.allclose(recomputed_theta_1d, stored_theta_1d, atol=2e-6, rtol=2e-6)
        ),
        "train_matrix_exact": len(train_ids) == len(set(train_ids))
        and set(train_ids) == _expected_history_ids(config, "train"),
        "calibration_matrix_exact": len(calibration_ids) == len(set(calibration_ids))
        and set(calibration_ids) == _expected_history_ids(config, "calibration"),
        "strict_past_fit_rows": all(
            int(row["feature_max_source_round"]) <= int(row["assessment_round"]) - 1
            for row in fit_rows
        ),
        "forbidden_current_fields_zero": all(
            int(row["inference_payload_forbidden_current_fields"]) == 0
            for row in fit_rows
        ),
        "fit_rng_counts_exact": len(registry["train_target"])
        == int(config["gates"]["train_target_child_seeds_exact"])
        and len(registry["calibration_target"])
        == int(config["gates"]["calibration_target_child_seeds_exact"])
        and len(registry["calibration_evaluation"])
        == int(config["gates"]["calibration_evaluation_child_seeds_exact"]),
        "fit_rng_unique": len(fit_generated) == len(set(fit_generated)),
        "evaluation_rng_absent": registry["evaluation_target"] == []
        and registry["evaluation"] == [],
        "holdout_rng_absent": registry["holdout"] == [],
        "seed_balancing_train": all(_close(float(value), 1.0, 1e-6) for value in balanced_train.values()),
        "seed_balancing_refit": all(_close(float(value), 1.0, 1e-6) for value in balanced_refit.values()),
        "rank_exact": int(design["train"]["flattened_design_rank"]) == 6
        and int(design["train_plus_calibration"]["flattened_design_rank"]) == 6,
        "rank_method_exact": design["train"]["rank_diagnostics"]["method"]
        == "two_pass_modified_gram_schmidt_on_unit_norm_columns"
        and design["train_plus_calibration"]["rank_diagnostics"]["method"]
        == "two_pass_modified_gram_schmidt_on_unit_norm_columns",
        "rank_computed_on_mps": torch.device(
            design["train"]["rank_diagnostics"]["compute_device"]
        ).type
        == "mps"
        and torch.device(
            design["train_plus_calibration"]["rank_diagnostics"]["compute_device"]
        ).type
        == "mps",
        "rms_floor_inactive": int(design["train"]["floor_active_count"]) == 0
        and int(design["train_plus_calibration"]["floor_active_count"]) == 0,
        "mps_float32_fit": torch.device(design["final_fit"]["fit_device"]).type == "mps"
        and design["final_fit"]["fit_dtype"] == "torch.float32"
        and design["final_fit"]["condition_number_norm"]
        == "infinity_exact_via_solve"
        and design["final_1d_fit"]["condition_number_norm"]
        == "infinity_exact_via_solve",
        "evaluation_not_generated_before_freeze": int(
            decision["evaluation_trajectory_count_generated"]
        )
        == 0,
        "holdout_closed": manifest["holdout_opened"] is False,
    }
    return {
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "selected_lambda_recomputed": selected,
        "selected_lambda_1d_recomputed": selected_1d,
        "max_coefficient_abs_difference": float(
            torch.max(torch.abs(recomputed_theta - stored_theta)).item()
        ),
        "one_dimensional_coefficient_abs_difference": float(
            torch.max(torch.abs(recomputed_theta_1d - stored_theta_1d)).item()
        ),
    }


def _recompute_evaluation(
    config: Mapping[str, Any], history_rows: Sequence[Mapping[str, str]], child_rows: Sequence[Mapping[str, str]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float | int]]]:
    by_history: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    meta = {row["history_id"]: row for row in history_rows}
    for row in child_rows:
        by_history[row["history_id"]][row["candidate"]].append(
            float(row["squared_reference_error"])
        )
    seed_rows: list[dict[str, Any]] = []
    for seed in config["randomness"]["evaluation_outer_seeds"]:
        ids = [key for key, row in meta.items() if int(row["seed"]) == int(seed)]
        sums: dict[str, float] = defaultdict(float)
        regimes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        split_sq = 0.0
        for history_id in ids:
            for candidate in CANDIDATES:
                value = statistics.fmean(by_history[history_id][candidate])
                sums[candidate] += value
                regimes[meta[history_id]["noise_regime"]][candidate] += value
            split_sq += float(meta[history_id]["privileged_target_split_aggregate_scale_distance"]) ** 2
        denominator = sums[K4] - sums[K4C]
        seed_rows.append(
            {
                "seed": int(seed),
                "histories": len(ids),
                "k4_integrated_mse": sums[K4],
                "k4b_integrated_mse": sums[K4B],
                "one_dimensional_integrated_mse": sums[K5_1D],
                "k5_integrated_mse": sums[K5],
                "k4c_integrated_mse": sums[K4C],
                "pointwise_integrated_mse": sums[POINTWISE],
                "gain_vs_k4b": (sums[K4B] - sums[K5]) / sums[K4B],
                "gain_vs_k4": (sums[K4] - sums[K5]) / sums[K4],
                "gain_vs_one_dimensional": (sums[K5_1D] - sums[K5]) / sums[K5_1D],
                "ch_capture_fraction": (sums[K4] - sums[K5]) / denominator,
                "k4_minus_k4c_positive": denominator > 0.0,
                "homogeneous_gain_vs_k4b": (
                    regimes["homogeneous"][K4B] - regimes["homogeneous"][K5]
                )
                / regimes["homogeneous"][K4B],
                "heteroscedastic_gain_vs_k4b": (
                    regimes["heteroscedastic"][K4B] - regimes["heteroscedastic"][K5]
                )
                / regimes["heteroscedastic"][K4B],
                "privileged_target_split_disagreement_mse_ratio": split_sq / sums[K4],
            }
        )
    tcrit = float(config["statistical_analysis"]["t_critical_df11"])
    cis = {
        key: _ci([float(row[key]) for row in seed_rows], tcrit)
        for key in (
            "gain_vs_k4b",
            "gain_vs_k4",
            "gain_vs_one_dimensional",
            "ch_capture_fraction",
            "homogeneous_gain_vs_k4b",
            "heteroscedastic_gain_vs_k4b",
        )
    }
    return seed_rows, cis


def _audit_evaluation(
    config: Mapping[str, Any], results: Path, fit_audit: Mapping[str, Any]
) -> dict[str, Any]:
    directory = results / "evaluation"
    history_rows = _read_csv(directory / "history_rows.csv")
    child_rows = _read_csv(directory / "evaluation_child_rows.csv")
    stored_seed_rows = _read_csv(directory / "seed_summary.csv")
    replace_rows = _read_csv(directory / "replace_one_audit.csv")
    registry = _read_json(directory / "evaluation_rng_registry.json")
    stored_decision = _read_json(directory / "decision.json")
    manifest = _read_json(directory / "manifest.json")
    recomputed_seed_rows, cis = _recompute_evaluation(config, history_rows, child_rows)
    seed_differences: list[float] = []
    stored_by_seed = {int(row["seed"]): row for row in stored_seed_rows}
    for row in recomputed_seed_rows:
        stored = stored_by_seed[int(row["seed"])]
        for key, value in row.items():
            if key in {"seed", "histories", "k4_minus_k4c_positive"}:
                continue
            seed_differences.append(abs(float(value) - float(stored[key])))
    expected_histories = _expected_history_ids(config, "evaluation")
    observed_histories = [row["history_id"] for row in history_rows]
    children = int(config["nested_monte_carlo"]["evaluation_children"])
    observed_child_keys = [
        (row["history_id"], row["candidate"], int(row["evaluation_child"]))
        for row in child_rows
    ]
    expected_child_keys = {
        (history_id, candidate, child)
        for history_id in expected_histories
        for candidate in CANDIDATES
        for child in range(children)
    }
    target_rng = [int(value) for value in registry["evaluation_target"]]
    evaluation_rng = [int(value) for value in registry["evaluation"]]
    fit_registry = _read_json(results / "fit_rng_registry.json")
    fit_rng = [
        int(value)
        for key in ("train_target", "calibration_target", "calibration_evaluation")
        for value in fit_registry[key]
    ]
    gates = config["gates"]
    scientific = {
        "gain_vs_k4b_mean": float(cis["gain_vs_k4b"]["mean"]) >= float(gates["primary_gain_vs_k4b_mean_min"]),
        "gain_vs_k4b_ci": float(cis["gain_vs_k4b"]["low"]) > float(gates["primary_gain_vs_k4b_ci95_low_strictly_greater_than"]),
        "gain_vs_k4_mean": float(cis["gain_vs_k4"]["mean"]) >= float(gates["primary_gain_vs_k4_mean_min"]),
        "gain_vs_k4_ci": float(cis["gain_vs_k4"]["low"]) > float(gates["primary_gain_vs_k4_ci95_low_strictly_greater_than"]),
        "gain_vs_one_dimensional_mean": float(cis["gain_vs_one_dimensional"]["mean"]) >= float(gates["primary_gain_vs_one_dimensional_mean_min"]),
        "gain_vs_one_dimensional_ci": float(cis["gain_vs_one_dimensional"]["low"]) > float(gates["primary_gain_vs_one_dimensional_ci95_low_strictly_greater_than"]),
        "capture_mean": float(cis["ch_capture_fraction"]["mean"]) >= float(gates["ch_capture_fraction_mean_min"]),
        "capture_ci": float(cis["ch_capture_fraction"]["low"]) > float(gates["ch_capture_fraction_ci95_low_strictly_greater_than"]),
        "homogeneous_gain_ci": float(cis["homogeneous_gain_vs_k4b"]["low"]) > float(gates["homogeneous_gain_vs_k4b_ci95_low_strictly_greater_than"]),
        "heteroscedastic_gain_ci": float(cis["heteroscedastic_gain_vs_k4b"]["low"]) > float(gates["heteroscedastic_gain_vs_k4b_ci95_low_strictly_greater_than"]),
    }
    stored_cis = stored_decision["confidence_intervals"]
    ci_differences: list[float] = []
    ci_schema_exact = set(stored_cis) == set(cis)
    if ci_schema_exact:
        for contrast, recomputed in cis.items():
            stored = stored_cis[contrast]
            if set(stored) != {"n", "mean", "low", "high"}:
                ci_schema_exact = False
                break
            if int(stored["n"]) != int(recomputed["n"]):
                ci_schema_exact = False
                break
            ci_differences.extend(
                abs(float(stored[key]) - float(recomputed[key]))
                for key in ("mean", "low", "high")
            )
    validity = {
        "fit_audit_pass": bool(fit_audit["all_checks_pass"]),
        "matrix_history_exact": len(observed_histories) == len(set(observed_histories))
        and set(observed_histories) == expected_histories,
        "matrix_child_exact": len(observed_child_keys) == len(expected_child_keys)
        and set(observed_child_keys) == expected_child_keys,
        "seed_summary_exact": max(seed_differences, default=0.0) <= 2e-6,
        "confidence_intervals_exact": ci_schema_exact
        and max(ci_differences, default=0.0) <= 2e-6,
        "target_rng_count_exact": len(target_rng) == int(gates["evaluation_target_child_seeds_exact"]),
        "evaluation_rng_count_exact": len(evaluation_rng) == int(gates["evaluation_child_seeds_exact"]),
        "all_rng_unique": len(fit_rng + target_rng + evaluation_rng)
        == len(set(fit_rng + target_rng + evaluation_rng)),
        "replace_one_count_exact": len(replace_rows) == int(gates["replace_one_exact_trials"]),
        "replace_one_no_violation": all(
            float(row["observed_difference"])
            <= float(row["theoretical_bound"]) + 1e-6
            for row in replace_rows
        ),
        "replace_one_bound_exact": all(_close(float(row["theoretical_bound"]), float(gates["replace_one_bound"]), 1e-8) for row in replace_rows),
        "formula_checks": max(float(row["k4_manual_formula_error"]) for row in child_rows) <= float(gates["k4_manual_formula_abs_error_max"])
        and max(float(row["k4b_reproduction_error"]) for row in child_rows) <= float(gates["k4b_reproduction_abs_error_max"])
        and max(float(row["fixed_denominator_formula_error"]) for row in child_rows) <= float(gates["fixed_denominator_formula_abs_error_max"]),
        "contribution_cap_numeric": max(
            float(row["max_slot_contribution_norm"])
            for row in child_rows
            if row["candidate"] == K5
        )
        <= float(config["references"]["total_client_influence_cap"])
        + 64.0 * torch.finfo(torch.float32).eps,
        "strict_past": all(int(row["feature_max_source_round"]) <= int(row["assessment_round"]) - 1 for row in history_rows),
        "forbidden_current_fields_zero": all(int(row["forbidden_current_inference_field_count"]) == 0 for row in history_rows),
        "predictor_hash_stable": manifest["frozen_predictor_sha256"] == _sha256(results / "frozen_predictor.json"),
        "holdout_closed": manifest["holdout_opened"] is False and registry["holdout"] == [],
        "positive_headroom": sum(bool(row["k4_minus_k4c_positive"]) for row in recomputed_seed_rows) == 12,
        "mc_stability": max(float(row["privileged_target_split_disagreement_mse_ratio"]) for row in recomputed_seed_rows) <= float(gates["privileged_target_split_disagreement_mse_ratio_max"]),
    }
    validity_pass = all(validity.values())
    science_pass = all(scientific.values()) if validity_pass else False
    decision = (
        "authorize_end_to_end_development_screen"
        if validity_pass and science_pass
        else "stop_this_linear_transcript_predictor_instance"
        if validity_pass
        else "invalid_or_inconclusive_screen"
    )
    decision_differences = {
        "decision": decision != stored_decision["decision"],
        "validity_pass": validity_pass != bool(stored_decision["validity_pass"]),
        "scientific_checks_pass": science_pass
        != bool(stored_decision["scientific_checks_pass"]),
    }
    return {
        "validity_checks": validity,
        "scientific_checks": scientific,
        "confidence_intervals": cis,
        "decision_recomputed": decision,
        "decision_differences": decision_differences,
        "all_checks_pass": validity_pass
        and not any(decision_differences.values())
        and scientific == stored_decision["scientific_checks"],
        "max_seed_summary_abs_difference": max(seed_differences, default=0.0),
        "max_confidence_interval_abs_difference": max(
            ci_differences, default=0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--phase", choices=("fit", "evaluation"), default="evaluation")
    parser.add_argument("--device", choices=("mps",), default="mps")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    lock_path = args.lock if args.lock.is_absolute() else ROOT / args.lock
    results = args.results if args.results.is_absolute() else ROOT / args.results
    if config_path.resolve() != DEFAULT_CONFIG.resolve():
        raise RuntimeError("The independent audit accepts only the locked K5 config")
    if lock_path.resolve() != DEFAULT_LOCK.resolve():
        raise RuntimeError("The independent audit accepts only the locked K5 lock")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not torch.backends.mps.is_available():
        raise RuntimeError("The independent K5 audit is locked to MPS")
    device = torch.device("mps")
    lock_audit = _verify_lock(config, lock_path)
    fit_audit = _audit_fit(config, results, device, lock_audit)
    result: dict[str, Any] = {
        "campaign_id": config["campaign_id"],
        "audit_device": "mps",
        "lock_audit": lock_audit,
        "fit_audit": fit_audit,
        "holdout_opened": False,
    }
    if args.phase == "evaluation":
        result["evaluation_audit"] = _audit_evaluation(config, results, fit_audit)
        result["all_checks_pass"] = bool(result["evaluation_audit"]["all_checks_pass"])
        output = results / "evaluation" / "independent_postrun_audit.json"
    else:
        result["all_checks_pass"] = bool(fit_audit["all_checks_pass"])
        output = results / "independent_fit_audit.json"
    _write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["all_checks_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
