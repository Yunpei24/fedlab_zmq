#!/usr/bin/env python3
"""Independently recompute the K6 radial development-prescreen decision.

The audit intentionally does not import the prescreen implementation.  It
rejoins the persisted K5 controls and radial child rows, recomputes the
seed-level estimands and Student intervals, and checks hashes, counts, caps,
causality fields, and the registered Boolean decision.
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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / (
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k5_tp_v2_mps_v2"
)
DEFAULT_SCREEN = ROOT / ("output/analysis/gaussian_aware_g0g_k6_radial_prescreen")

K4 = "g0g_k4_temporal_causal_gate"
K4B = "g0g_k4b_rolling_past_imputation"
K5_1D = "g0g_k5_tp_v2_one_dimensional_control"
K4C = "g0g_k4c_ch_privileged_benchmark"
POINTWISE = "g0g_k4b_pointwise_optimal_oracle"
RADIAL = "g0g_k6_unmodulated_radial_development"
CONTROLS = (K4, K4B, K5_1D, K4C, POINTWISE)

EXPECTED_SEEDS = 12
EXPECTED_HISTORIES = 576
EXPECTED_CHILDREN = 64
EXPECTED_CHILD_ROWS = EXPECTED_HISTORIES * EXPECTED_CHILDREN
TOLERANCE = 2.0e-12

# These hashes extend the chain of custody after the scientific run.  They are
# deliberately absent from the preregistered manifest and must never be
# described as preregistered evidence.
POST_RUN_SCIENTIFIC_OUTPUTS = {
    "decision": "decision.json",
    "diagnostics": "diagnostics.json",
    "manifest": "manifest.json",
    "radial_child_rows": "radial_child_rows.csv",
    "radial_history_rows": "radial_history_rows.csv",
    "seed_summary": "seed_summary.csv",
    "stratified_seed_summary": "stratified_seed_summary.csv",
}
POST_RUN_IMPLEMENTATION_FILES = {
    "k6_algorithm_module": "algorithms/gaussian_aware_reference_k6_tp_eiv.py",
    "k6_algorithm_tests": "tests/test_gaussian_aware_reference_g0g_k6_tp_eiv.py",
    "prescreen_runner": "scripts/analyze_gaussian_aware_k6_radial_prescreen.py",
    "prescreen_tests": "tests/test_analyze_gaussian_aware_k6_radial_prescreen.py",
    "independent_audit": "scripts/audit_gaussian_aware_k6_radial_prescreen.py",
    "report_generator": "scripts/report_gaussian_aware_k6_radial_prescreen.py",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _post_run_provenance_extension(screen: Path) -> dict[str, Any]:
    """Hash the persisted evidence and code without rewriting preregistration.

    The independent audit file itself is intentionally not among the output
    artifacts: a file cannot contain a stable cryptographic hash of itself.
    Its implementation is nevertheless hashed in ``implementation_sha256``.
    """

    scientific_paths = {
        name: screen / relative
        for name, relative in POST_RUN_SCIENTIFIC_OUTPUTS.items()
    }
    implementation_paths = {
        name: ROOT / relative
        for name, relative in POST_RUN_IMPLEMENTATION_FILES.items()
    }
    missing = [
        str(path)
        for path in (*scientific_paths.values(), *implementation_paths.values())
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Post-run provenance inputs are incomplete: " + ", ".join(missing)
        )
    return {
        "status": "post_run_only_not_preregistered",
        "interpretation": (
            "These hashes were recorded by the independent audit after the "
            "scientific run. They strengthen traceability but were not present "
            "in, and do not modify, the preregistered manifest."
        ),
        "scientific_output_sha256": {
            name: _sha256(path) for name, path in scientific_paths.items()
        },
        "implementation_sha256": {
            name: _sha256(path) for name, path in implementation_paths.items()
        },
        "excluded_self_referential_output": "independent_postrun_audit.json",
    }


def _finite(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{name} is not finite")
    return number


def _metadata(row: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        str(row[name])
        for name in (
            "seed",
            "noise_regime",
            "noise_permutation",
            "outlier_geometry",
            "honest_dynamics",
            "threat",
            "assessment_round",
            "evaluation_child_seed",
        )
    )


def _load_source_controls(
    path: Path,
) -> tuple[
    dict[tuple[str, int, str], float],
    dict[tuple[str, int], tuple[str, ...]],
    dict[str, int],
]:
    values: dict[tuple[str, int, str], float] = {}
    metadata: dict[tuple[str, int], tuple[str, ...]] = {}
    all_candidate_counts: dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            candidate = str(row["candidate"])
            all_candidate_counts[candidate] += 1
            if candidate not in CONTROLS:
                continue
            key = (str(row["history_id"]), int(row["evaluation_child"]))
            full_key = (*key, candidate)
            if full_key in values:
                raise RuntimeError("Duplicate persisted K5 control row")
            error = _finite(row["squared_reference_error"], name="source squared error")
            if error < 0.0:
                raise RuntimeError("Negative persisted K5 squared error")
            values[full_key] = error
            observed_metadata = _metadata(row)
            previous = metadata.setdefault(key, observed_metadata)
            if previous != observed_metadata:
                raise RuntimeError("K5 controls disagree on child metadata")
    return values, metadata, dict(all_candidate_counts)


def _load_radial(
    path: Path,
    source_metadata: Mapping[tuple[str, int], tuple[str, ...]],
) -> tuple[
    dict[tuple[str, int, str], float],
    dict[tuple[str, int], tuple[str, ...]],
]:
    values: dict[tuple[str, int, str], float] = {}
    metadata: dict[tuple[str, int], tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["candidate"] != RADIAL:
                raise RuntimeError("Unexpected candidate in radial CSV")
            key = (str(row["history_id"]), int(row["evaluation_child"]))
            full_key = (*key, RADIAL)
            if full_key in values:
                raise RuntimeError("Duplicate radial child row")
            error = _finite(row["squared_reference_error"], name="radial error")
            if error < 0.0:
                raise RuntimeError("Negative radial squared error")
            values[full_key] = error
            observed_metadata = _metadata(row)
            metadata[key] = observed_metadata
            if source_metadata.get(key) != observed_metadata:
                raise RuntimeError("Radial and K5 child metadata are not paired")
            for boolean_name in (
                "radial_predictor_fixed_before_current_child",
                "predictor_observable_past_only",
                "contribution_cap_respected",
            ):
                if str(row[boolean_name]) != "True":
                    raise RuntimeError(f"Radial row violates {boolean_name}")
            if str(row["normalization_by_gate_sum"]) != "False":
                raise RuntimeError("Radial row normalized by the private gate sum")
    return values, metadata


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise RuntimeError("Cannot average an empty group")
    return statistics.fmean(values)


def _estimands(means: Mapping[str, float]) -> dict[str, float]:
    headroom = means[K4] - means[K4C]
    if min(means.values()) <= 0.0 or headroom <= 0.0:
        raise RuntimeError("Invalid positive-denominator assumption")
    return {
        "gain_vs_k4b": 1.0 - means[RADIAL] / means[K4B],
        "gain_vs_k4": 1.0 - means[RADIAL] / means[K4],
        "relative_loss_vs_k5_1d": means[RADIAL] / means[K5_1D] - 1.0,
        "ch_capture_fraction": (means[K4] - means[RADIAL]) / headroom,
        "relative_k4_k4c_headroom": headroom / means[K4],
        "remaining_mse_above_k4c_fraction_of_headroom": (means[RADIAL] - means[K4C])
        / headroom,
        "relative_loss_vs_pointwise": means[RADIAL] / means[POINTWISE] - 1.0,
    }


def _student_ci(values: Sequence[float], t_critical: float) -> dict[str, float | int]:
    if len(values) != EXPECTED_SEEDS:
        raise RuntimeError("Student interval does not contain twelve seed units")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    half_width = t_critical * standard_deviation / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "sd": standard_deviation,
        "low": mean - half_width,
        "high": mean + half_width,
    }


def _close(left: Any, right: Any, *, tolerance: float = TOLERANCE) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def _mapping_close(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    for key, value in left.items():
        other = right[key]
        if isinstance(value, Mapping):
            if not isinstance(other, Mapping) or not _mapping_close(value, other):
                return False
        elif isinstance(value, bool):
            if type(other) is not bool or value is not other:
                return False
        elif isinstance(value, (int, float)):
            if not _close(value, other):
                return False
        elif value != other:
            return False
    return True


def audit(source: Path, screen: Path) -> dict[str, Any]:
    decision_path = screen / "decision.json"
    manifest_path = screen / "manifest.json"
    diagnostics_path = screen / "diagnostics.json"
    radial_child_path = screen / "radial_child_rows.csv"
    radial_history_path = screen / "radial_history_rows.csv"
    seed_summary_path = screen / "seed_summary.csv"
    stratified_summary_path = screen / "stratified_seed_summary.csv"
    required = (
        decision_path,
        manifest_path,
        diagnostics_path,
        radial_child_path,
        radial_history_path,
        seed_summary_path,
        stratified_summary_path,
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("The radial prescreen artifact set is incomplete")

    decision = _read_json(decision_path)
    manifest = _read_json(manifest_path)
    diagnostics = _read_json(diagnostics_path)
    resolved = _read_json(source / "resolved_config.json")
    source_values, source_metadata, candidate_counts = _load_source_controls(
        source / "evaluation/evaluation_child_rows.csv"
    )
    radial_values, radial_metadata = _load_radial(radial_child_path, source_metadata)

    child_keys = set(source_metadata)
    seeds = sorted({int(metadata[0]) for metadata in source_metadata.values()})
    histories = {history_id for history_id, _ in child_keys}
    complete_matrix = (
        len(seeds) == EXPECTED_SEEDS
        and len(histories) == EXPECTED_HISTORIES
        and len(child_keys) == EXPECTED_CHILD_ROWS
        and len(radial_values) == EXPECTED_CHILD_ROWS
        and set(radial_metadata) == child_keys
        and all(
            (*key, candidate) in source_values
            for key in child_keys
            for candidate in CONTROLS
        )
    )

    candidate_values = {**source_values, **radial_values}
    seed_estimands: dict[int, dict[str, float]] = {}
    stratum_estimands: dict[tuple[str, str, int], dict[str, float]] = {}
    for seed in seeds:
        keys = [key for key in child_keys if int(source_metadata[key][0]) == seed]
        means = {
            candidate: _mean([candidate_values[(*key, candidate)] for key in keys])
            for candidate in (*CONTROLS, RADIAL)
        }
        seed_estimands[seed] = _estimands(means)
        for stratum_name, metadata_index in (("noise_regime", 1), ("threat", 5)):
            levels = sorted({source_metadata[key][metadata_index] for key in keys})
            for level in levels:
                subset = [
                    key for key in keys if source_metadata[key][metadata_index] == level
                ]
                stratum_means = {
                    candidate: _mean(
                        [candidate_values[(*key, candidate)] for key in subset]
                    )
                    for candidate in (*CONTROLS, RADIAL)
                }
                stratum_estimands[(stratum_name, level, seed)] = _estimands(
                    stratum_means
                )

    t_critical = float(resolved["statistical_analysis"]["t_critical_df11"])
    recomputed_ci = {
        estimand: _student_ci(
            [seed_estimands[seed][estimand] for seed in seeds], t_critical
        )
        for estimand in next(iter(seed_estimands.values()))
    }
    recomputed_strata: dict[str, dict[str, dict[str, float | int]]] = {}
    for stratum_name in ("noise_regime", "threat"):
        levels = sorted(
            {level for name, level, _ in stratum_estimands if name == stratum_name}
        )
        recomputed_strata[stratum_name] = {
            level: _student_ci(
                [
                    stratum_estimands[(stratum_name, level, seed)]["gain_vs_k4b"]
                    for seed in seeds
                ],
                t_critical,
            )
            for level in levels
        }

    gates = decision["scientific_gates_frozen_in_source_before_execution"]
    homogeneous = recomputed_strata["noise_regime"]["homogeneous"]
    heteroscedastic = recomputed_strata["noise_regime"]["heteroscedastic"]
    recomputed_scientific_checks = {
        "gain_vs_k4b_mean": recomputed_ci["gain_vs_k4b"]["mean"]
        >= gates["gain_vs_k4b_mean_min"],
        "gain_vs_k4b_ci": recomputed_ci["gain_vs_k4b"]["low"]
        > gates["gain_vs_k4b_ci95_low_strictly_greater_than"],
        "gain_vs_k4_mean": recomputed_ci["gain_vs_k4"]["mean"]
        >= gates["gain_vs_k4_mean_min"],
        "gain_vs_k4_ci": recomputed_ci["gain_vs_k4"]["low"]
        > gates["gain_vs_k4_ci95_low_strictly_greater_than"],
        "noninferior_to_k5_1d": recomputed_ci["relative_loss_vs_k5_1d"]["high"]
        < gates["relative_loss_vs_k5_1d_ci95_high_strictly_less_than"],
        "homogeneous_gain_vs_k4b_ci": homogeneous["low"]
        > gates["homogeneous_gain_vs_k4b_ci95_low_strictly_greater_than"],
        "heteroscedastic_gain_vs_k4b_ci": heteroscedastic["low"]
        > gates["heteroscedastic_gain_vs_k4b_ci95_low_strictly_greater_than"],
    }

    with radial_history_path.open("r", encoding="utf-8", newline="") as handle:
        history_rows = list(csv.DictReader(handle))
    cap_ok = len(history_rows) == EXPECTED_HISTORIES and all(
        _finite(row["radial_predictor_norm"], name="predictor norm")
        <= _finite(row["influence_cap"], name="influence cap") + 1.0e-6
        for row in history_rows
    )
    causal_ok = all(
        int(row["feature_max_source_round"]) < int(row["assessment_round"])
        and row["uses_current_round_input"] == "False"
        and int(row["forbidden_current_inference_field_count"]) == 0
        for row in history_rows
    )

    source_hashes = {
        name: _sha256(source / relative)
        for name, relative in {
            "root_manifest": "manifest.json",
            "resolved_config": "resolved_config.json",
            "frozen_predictor": "frozen_predictor.json",
            "evaluation_manifest": "evaluation/manifest.json",
            "evaluation_decision": "evaluation/decision.json",
            "independent_postrun_audit": ("evaluation/independent_postrun_audit.json"),
            "evaluation_rng_registry": ("evaluation/evaluation_rng_registry.json"),
            "history_rows": "evaluation/history_rows.csv",
            "evaluation_child_rows": "evaluation/evaluation_child_rows.csv",
        }.items()
    }
    screen_script_hash = _sha256(
        ROOT / "scripts/analyze_gaussian_aware_k6_radial_prescreen.py"
    )
    checks = {
        "complete_paired_matrix": complete_matrix,
        "source_candidate_counts_exact": len(candidate_counts) == 7
        and all(value == EXPECTED_CHILD_ROWS for value in candidate_counts.values()),
        "source_hashes_exact": source_hashes
        == manifest["source_hashes"]
        == diagnostics["source_hashes_before"]
        == diagnostics["source_hashes_after"],
        "screen_script_hash_exact": screen_script_hash
        == manifest["screen_script_sha256"]
        == diagnostics["screen_script_sha256"],
        "seed_estimands_match": _mapping_close(
            {
                str(row["seed"]): {
                    key: _finite(row[key], name=f"seed summary {key}")
                    for key in seed_estimands[int(row["seed"])]
                }
                for row in csv.DictReader(
                    seed_summary_path.open("r", encoding="utf-8", newline="")
                )
            },
            {str(seed): estimands for seed, estimands in seed_estimands.items()},
        ),
        "confidence_intervals_match": _mapping_close(
            recomputed_ci, decision["confidence_intervals"]
        ),
        "stratified_intervals_match": _mapping_close(
            recomputed_strata,
            decision["stratified_gain_vs_k4b_confidence_intervals"],
        ),
        "scientific_checks_match": recomputed_scientific_checks
        == decision["scientific_checks"],
        "scientific_checks_all_pass": all(recomputed_scientific_checks.values()),
        "predictor_cap_respected": cap_ok,
        "strictly_past": causal_ok,
        "mps_no_fallback_recorded": manifest["device"] == "mps"
        and decision["validity_checks"]["mps_fallback_disabled"] is True,
        "holdout_closed": manifest["holdout_opened"] is False
        and decision["holdout_opened"] is False
        and decision["authorizes_holdout_or_promotion"] is False,
        "registered_decision_exact": decision["validity_pass"] is True
        and decision["scientific_checks_pass"] is True
        and decision["all_gates_pass"] is True
        and decision["decision"] == "advance_radial_confidence_k6_development",
    }
    result = {
        "audit": "independent_k6_radial_prescreen_postrun_v1",
        "all_checks_pass": all(checks.values()),
        "checks": checks,
        "counts": {
            "outer_seeds": len(seeds),
            "histories": len(histories),
            "children": len(child_keys),
            "history_rows": len(history_rows),
        },
        "recomputed_confidence_intervals": recomputed_ci,
        "recomputed_stratified_gain_vs_k4b": recomputed_strata,
        "recomputed_scientific_checks": recomputed_scientific_checks,
        "source_hashes": source_hashes,
        "screen_script_sha256": screen_script_hash,
        "post_run_provenance_extension": _post_run_provenance_extension(screen),
        "scope": "development_only_reusing_consumed_k5_evaluation_seeds",
        "holdout_opened": False,
        "authorizes_holdout_or_promotion": False,
    }
    if not result["all_checks_pass"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("Independent radial audit failed: " + ", ".join(failed))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--screen", type=Path, default=DEFAULT_SCREEN)
    args = parser.parse_args()
    result = audit(args.source, args.screen)
    output = args.screen / "independent_postrun_audit.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
