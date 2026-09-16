#!/usr/bin/env python3
"""Run the development-only K6 radial-predictor prescreen on MPS.

This screen intentionally reuses the twelve *already consumed* K5-v2
evaluation outer seeds.  It is therefore a mechanistic development analysis,
not a new confirmation set, and it can never authorize opening the reserved
holdout or promoting a method.  Current-round children are regenerated with
the locked K5-v2 RNG helpers solely to obtain an exactly paired error for the
new radial predictor.

The predictor itself is computed before any current-round child is generated:

    p_radial = G Y / max(||Y||_2, r_min),

where Y is K4b's rolling accepted mean from rounds at most t-1, G is the
existing public influence cap, and r_min is a public numerical constant.  No
label, clean current upload, current noise, current attack, or privileged
target enters the predictor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.gaussian_aware_reference_k6_tp_eiv import (  # noqa: E402
    radial_confidence_predictor,
)
from scripts import run_gaussian_aware_reference_g0g_k5_tp_v2 as k5  # noqa: E402
from scripts import run_gaussian_aware_reference_oracle as oracle  # noqa: E402

SCREEN_ID = "gaussian_aware_g0g_k6_radial_prescreen_development_v1"
RADIAL = "g0g_k6_unmodulated_radial_development"

DEFAULT_CONFIG = ROOT / (
    "configs/ldp_gradient_far/k5_v2/" "gaussian_aware_reference_g0g_k5_tp_v2.yaml"
)
DEFAULT_K5_RESULTS = ROOT / (
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k5_tp_v2_mps_v2"
)
DEFAULT_OUTPUT = ROOT / ("output/analysis/gaussian_aware_g0g_k6_radial_prescreen")

# Frozen in source before executing the new radial candidate.  These are
# development-screen thresholds, not retrospectively chosen summaries.
PREREGISTERED_SCIENTIFIC_GATES: dict[str, float] = {
    "gain_vs_k4b_mean_min": 0.10,
    "gain_vs_k4b_ci95_low_strictly_greater_than": 0.0,
    "gain_vs_k4_mean_min": 0.30,
    "gain_vs_k4_ci95_low_strictly_greater_than": 0.20,
    "relative_loss_vs_k5_1d_ci95_high_strictly_less_than": 0.05,
    "homogeneous_gain_vs_k4b_ci95_low_strictly_greater_than": 0.0,
    "heteroscedastic_gain_vs_k4b_ci95_low_strictly_greater_than": 0.0,
}
RADIAL_MINIMUM_DIRECTION_NORM = 1.0e-8
SOURCE_RECOMPUTATION_ABS_TOLERANCE = 2.0e-7
REFERENCE_REPRODUCTION_ABS_TOLERANCE = 1.0e-6
EXPECTED_OUTER_SEEDS = 12

CONTROL_CANDIDATES = (k5.K4, k5.K4B, k5.K5_1D, k5.K4C, k5.POINTWISE)
ALL_SOURCE_CANDIDATES = tuple(k5.CANDIDATES)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Refusing to serialize a non-finite number")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _all_true(value: Any, *, name: str) -> bool:
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError(f"{name} must be a non-empty mapping")
    if any(type(item) is not bool for item in value.values()):
        raise RuntimeError(f"{name} must contain JSON booleans only")
    return all(value.values())


def _as_finite_float(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{name} must be finite")
    return number


def _ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise ValueError("Ratio operands must be finite")
    if denominator <= 0.0:
        raise ValueError("Ratio denominator must be strictly positive")
    return numerator / denominator


def _student_ci(values: Sequence[float], t_critical: float) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    if len(numbers) != EXPECTED_OUTER_SEEDS or not all(
        math.isfinite(value) for value in numbers
    ):
        raise ValueError("A prescreen interval requires exactly 12 finite seed values")
    mean = statistics.fmean(numbers)
    standard_deviation = statistics.stdev(numbers)
    half_width = float(t_critical) * standard_deviation / math.sqrt(len(numbers))
    return {
        "n": len(numbers),
        "mean": mean,
        "sd": standard_deviation,
        "low": mean - half_width,
        "high": mean + half_width,
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    numbers = sorted(float(value) for value in values)
    if not numbers or not 0.0 <= probability <= 1.0:
        raise ValueError("Quantile input is empty or probability is invalid")
    position = probability * (len(numbers) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return numbers[lower] * (1.0 - fraction) + numbers[upper] * fraction


def _distribution_summary(values: Sequence[float]) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    if not numbers or not all(math.isfinite(value) for value in numbers):
        raise ValueError("Distribution summary requires finite values")
    return {
        "n": len(numbers),
        "minimum": min(numbers),
        "q05": _quantile(numbers, 0.05),
        "median": statistics.median(numbers),
        "mean": statistics.fmean(numbers),
        "q95": _quantile(numbers, 0.95),
        "maximum": max(numbers),
    }


def _source_paths(results: Path) -> dict[str, Path]:
    return {
        "root_manifest": results / "manifest.json",
        "resolved_config": results / "resolved_config.json",
        "frozen_predictor": results / "frozen_predictor.json",
        "evaluation_manifest": results / "evaluation/manifest.json",
        "evaluation_decision": results / "evaluation/decision.json",
        "independent_postrun_audit": (
            results / "evaluation/independent_postrun_audit.json"
        ),
        "evaluation_rng_registry": (
            results / "evaluation/evaluation_rng_registry.json"
        ),
        "history_rows": results / "evaluation/history_rows.csv",
        "evaluation_child_rows": (results / "evaluation/evaluation_child_rows.csv"),
    }


def _load_history_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = k5._read_csv(path)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        history_id = str(row.get("history_id", ""))
        if not history_id or history_id in result:
            raise RuntimeError("K5 history rows contain an empty or duplicate ID")
        result[history_id] = row
    return result


def _load_control_rows(
    path: Path,
    *,
    expected_histories: int,
    children: int,
) -> tuple[
    dict[tuple[str, int], dict[str, float]],
    dict[tuple[str, int], dict[str, str]],
    dict[str, int],
]:
    controls: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    child_metadata: dict[tuple[str, int], dict[str, str]] = {}
    masks: dict[tuple[str, int], int] = defaultdict(int)
    counts: Counter[str] = Counter()
    candidate_bits = {
        candidate: 1 << index for index, candidate in enumerate(ALL_SOURCE_CANDIDATES)
    }
    expected_mask = (1 << len(ALL_SOURCE_CANDIDATES)) - 1
    total_rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "history_id",
            "seed",
            "noise_regime",
            "noise_permutation",
            "outlier_geometry",
            "honest_dynamics",
            "threat",
            "assessment_round",
            "evaluation_child",
            "evaluation_child_seed",
            "candidate",
            "squared_reference_error",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError("K5 child CSV schema is incomplete")
        for row in reader:
            total_rows += 1
            candidate = str(row["candidate"])
            if candidate not in candidate_bits:
                raise RuntimeError(f"Unexpected K5 source candidate: {candidate}")
            history_id = str(row["history_id"])
            child = int(row["evaluation_child"])
            if not 0 <= child < children:
                raise RuntimeError("K5 source child index is outside the locked range")
            key = (history_id, child)
            bit = candidate_bits[candidate]
            if masks[key] & bit:
                raise RuntimeError("Duplicate K5 candidate row for one paired child")
            masks[key] |= bit
            counts[candidate] += 1
            error = _as_finite_float(
                row["squared_reference_error"], name="source squared error"
            )
            if error < 0.0:
                raise RuntimeError("K5 source squared error cannot be negative")
            if candidate in CONTROL_CANDIDATES:
                controls[key][candidate] = error
            metadata = {
                field: str(row[field])
                for field in (
                    "seed",
                    "noise_regime",
                    "noise_permutation",
                    "outlier_geometry",
                    "honest_dynamics",
                    "threat",
                    "assessment_round",
                    "evaluation_child_seed",
                )
            }
            previous = child_metadata.setdefault(key, metadata)
            if previous != metadata:
                raise RuntimeError(
                    "K5 candidate rows disagree on paired-child metadata"
                )

    expected_child_keys = expected_histories * children
    expected_rows_per_candidate = expected_child_keys
    if total_rows != expected_rows_per_candidate * len(ALL_SOURCE_CANDIDATES):
        raise RuntimeError("K5 source child-row count is not exact")
    if len(masks) != expected_child_keys or any(
        value != expected_mask for value in masks.values()
    ):
        raise RuntimeError("K5 source candidate matrix is incomplete")
    if set(counts) != set(ALL_SOURCE_CANDIDATES) or any(
        counts[candidate] != expected_rows_per_candidate
        for candidate in ALL_SOURCE_CANDIDATES
    ):
        raise RuntimeError("K5 source candidate counts are not exact")
    if any(set(value) != set(CONTROL_CANDIDATES) for value in controls.values()):
        raise RuntimeError("K5 source controls are incomplete")
    return dict(controls), child_metadata, dict(counts)


def _expected_evaluation_rng(
    config: Mapping[str, Any], seeds: Sequence[int]
) -> tuple[dict[str, Any], dict[tuple[str, int], int]]:
    children = int(config["nested_monte_carlo"]["evaluation_children"])
    target_children = int(
        config["nested_monte_carlo"]["evaluation_target_construction_children"]
    )
    streams = {
        "train_target": [],
        "calibration_target": [],
        "calibration_evaluation": [],
        "evaluation_target": k5._expected_rng_stream_records(
            config,
            split="evaluation",
            seeds=seeds,
            stream="evaluation_target",
            children=target_children,
        ),
        "evaluation": k5._expected_rng_stream_records(
            config,
            split="evaluation",
            seeds=seeds,
            stream="evaluation",
            children=children,
        ),
        "holdout": [],
    }
    expected_registry = k5._rng_registry(streams)
    seed_map: dict[tuple[str, int], int] = {}
    for record in streams["evaluation"]:
        history_id = str(record["history_id"])
        for child, seed in enumerate(record["child_seeds"]):
            key = (history_id, child)
            if key in seed_map:
                raise RuntimeError("Duplicate expected K5 evaluation RNG key")
            seed_map[key] = int(seed)
    return expected_registry, seed_map


def _validate_source(config_path: Path, results: Path) -> tuple[
    dict[str, Any],
    dict[str, dict[str, str]],
    dict[tuple[str, int], dict[str, float]],
    dict[tuple[str, int], dict[str, str]],
    dict[tuple[str, int], int],
    dict[str, str],
    dict[str, Any],
]:
    paths = _source_paths(results)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing K5-v2 source artifacts: " + ", ".join(missing))

    root_manifest = _read_json(paths["root_manifest"])
    evaluation_manifest = _read_json(paths["evaluation_manifest"])
    decision = _read_json(paths["evaluation_decision"])
    audit = _read_json(paths["independent_postrun_audit"])
    resolved = _read_json(paths["resolved_config"])
    predictor = _read_json(paths["frozen_predictor"])
    persisted_rng = _read_json(paths["evaluation_rng_registry"])
    config, amendment = k5._load_amended_config(config_path)
    k5._validate_config(config, amendment)

    if k5._canonical_hash(config) != k5._canonical_hash(resolved):
        raise RuntimeError("Resolved K5 configuration differs from the locked loader")
    if root_manifest.get("campaign_id") != k5.CAMPAIGN_ID:
        raise RuntimeError("Unexpected K5-v2 source campaign")
    if (
        root_manifest.get("status") != "completed_development"
        or evaluation_manifest.get("status") != "completed_development"
    ):
        raise RuntimeError("K5-v2 source evaluation is not complete")
    if (
        root_manifest.get("device") != "mps"
        or evaluation_manifest.get("device") != "mps"
        or audit.get("audit_device") != "mps"
    ):
        raise RuntimeError("K5-v2 source or independent audit was not MPS-only")
    if any(
        value is not False
        for value in (
            root_manifest.get("holdout_opened"),
            evaluation_manifest.get("holdout_opened"),
            decision.get("holdout_opened"),
            audit.get("holdout_opened"),
            predictor.get("holdout_opened"),
        )
    ):
        raise RuntimeError("Reserved K5 holdout is not closed")
    if audit.get("all_checks_pass") is not True or not _all_true(
        audit.get("checks"), name="independent audit checks"
    ):
        raise RuntimeError("Independent K5 post-run audit did not pass")
    evaluation_audit = audit.get("evaluation_audit")
    if (
        not isinstance(evaluation_audit, Mapping)
        or evaluation_audit.get("pass") is not True
        or evaluation_audit.get("all_checks_pass") is not True
    ):
        raise RuntimeError("Independent K5 evaluation recomputation did not pass")
    if decision.get("validity_pass") is not True or not _all_true(
        decision.get("validity_checks"), name="K5 decision validity checks"
    ):
        raise RuntimeError("K5-v2 source evaluation is invalid")

    source_hashes = {name: _sha256(path) for name, path in paths.items()}
    resolved_hash = source_hashes["resolved_config"]
    predictor_hash = source_hashes["frozen_predictor"]
    if not (
        root_manifest.get("resolved_config_sha256")
        == predictor.get("resolved_config_sha256")
        == audit.get("config_hashes", {}).get("resolved_config_sha256")
        == resolved_hash
    ):
        raise RuntimeError("K5 resolved-configuration SHA-256 does not reconcile")
    if not (
        root_manifest.get("frozen_predictor_sha256")
        == evaluation_manifest.get("frozen_predictor_sha256")
        == predictor_hash
    ):
        raise RuntimeError("K5 frozen-predictor SHA-256 does not reconcile")
    audited_hashes = audit.get("audited_artifact_sha256")
    if not isinstance(audited_hashes, Mapping) or not (
        audited_hashes.get("manifest.json") == source_hashes["root_manifest"]
        and audited_hashes.get("frozen_predictor.json") == predictor_hash
    ):
        raise RuntimeError("Independent audit hashes do not match K5 source files")

    seeds = [int(value) for value in config["randomness"]["evaluation_outer_seeds"]]
    if len(seeds) != EXPECTED_OUTER_SEEDS or len(set(seeds)) != EXPECTED_OUTER_SEEDS:
        raise RuntimeError("Expected twelve distinct consumed K5 evaluation seeds")
    other_seed_sets = [
        {int(value) for value in config["randomness"][name]}
        for name in (
            "train_outer_seeds",
            "calibration_outer_seeds",
            "reserved_holdout_seeds",
        )
    ]
    if any(set(seeds) & values for values in other_seed_sets):
        raise RuntimeError("K5 evaluation seeds overlap another registered split")

    expected_rng, expected_child_seeds = _expected_evaluation_rng(config, seeds)
    expected_streams = expected_rng["streams"]
    if not k5._validate_rng_registry(persisted_rng, expected_streams):
        raise RuntimeError("K5 evaluation RNG registry is not exact")
    if persisted_rng.get("payload_sha256") != evaluation_manifest.get(
        "evaluation_rng_registry_structured_sha256"
    ):
        raise RuntimeError("K5 evaluation RNG hash does not match its manifest")

    histories = _load_history_rows(paths["history_rows"])
    expected_histories = int(config["gates"]["evaluation_histories_exact"])
    if len(histories) != expected_histories:
        raise RuntimeError("K5 history-row count is not exact")
    controls, child_metadata, candidate_counts = _load_control_rows(
        paths["evaluation_child_rows"],
        expected_histories=expected_histories,
        children=int(config["nested_monte_carlo"]["evaluation_children"]),
    )
    if set(child_metadata) != set(expected_child_seeds):
        raise RuntimeError("K5 persisted and expected RNG child keys differ")
    for key, expected_seed in expected_child_seeds.items():
        if int(child_metadata[key]["evaluation_child_seed"]) != expected_seed:
            raise RuntimeError("K5 persisted child seed differs from its registry")

    source_diagnostics = {
        "candidate_counts": candidate_counts,
        "evaluation_outer_seeds": seeds,
        "expected_history_count": expected_histories,
        "expected_children_per_history": int(
            config["nested_monte_carlo"]["evaluation_children"]
        ),
        "evaluation_rng_payload_sha256": persisted_rng["payload_sha256"],
        "source_k5_decision": decision["decision"],
        "source_k5_scientific_checks_pass": decision["scientific_checks_pass"],
        "source_k5_validity_pass": decision["validity_pass"],
    }
    return (
        config,
        histories,
        controls,
        child_metadata,
        expected_child_seeds,
        source_hashes,
        source_diagnostics,
    )


def _metadata_matches(context: Mapping[str, Any], persisted: Mapping[str, str]) -> bool:
    cell = context["cell"]
    expected = {
        "seed": str(int(cell["seed"])),
        "noise_regime": str(cell["regime"]["name"]),
        "noise_permutation": str(cell["permutation"]),
        "outlier_geometry": str(cell["geometry"]),
        "honest_dynamics": str(cell["dynamics"]),
        "threat": str(cell["threat"]),
        "assessment_round": str(int(context["round_index"])),
    }
    return all(str(persisted.get(key)) == value for key, value in expected.items())


def _integrated_row(
    *,
    seed: int,
    history_ids: Sequence[str],
    history_means: Mapping[tuple[str, str], float],
) -> dict[str, Any]:
    sums = {
        candidate: sum(
            history_means[(history_id, candidate)] for history_id in history_ids
        )
        for candidate in (*CONTROL_CANDIDATES, RADIAL)
    }
    headroom = sums[k5.K4] - sums[k5.K4C]
    if headroom <= 0.0:
        raise RuntimeError("K4-to-K4c headroom must be strictly positive")
    return {
        "seed": int(seed),
        "histories": len(history_ids),
        "k4_integrated_mse": sums[k5.K4],
        "k4b_integrated_mse": sums[k5.K4B],
        "k5_1d_integrated_mse": sums[k5.K5_1D],
        "radial_integrated_mse": sums[RADIAL],
        "k4c_integrated_mse": sums[k5.K4C],
        "pointwise_integrated_mse": sums[k5.POINTWISE],
        "gain_vs_k4b": _ratio(sums[k5.K4B] - sums[RADIAL], sums[k5.K4B]),
        "gain_vs_k4": _ratio(sums[k5.K4] - sums[RADIAL], sums[k5.K4]),
        "relative_loss_vs_k5_1d": _ratio(sums[RADIAL] - sums[k5.K5_1D], sums[k5.K5_1D]),
        "ch_capture_fraction": _ratio(sums[k5.K4] - sums[RADIAL], headroom),
        "relative_k4_k4c_headroom": _ratio(headroom, sums[k5.K4]),
        "remaining_mse_above_k4c_fraction_of_headroom": _ratio(
            sums[RADIAL] - sums[k5.K4C], headroom
        ),
        "relative_loss_vs_pointwise": _ratio(
            sums[RADIAL] - sums[k5.POINTWISE], sums[k5.POINTWISE]
        ),
    }


def _summaries(
    *,
    config: Mapping[str, Any],
    histories: Mapping[str, Mapping[str, str]],
    control_errors: Mapping[tuple[str, int], Mapping[str, float]],
    radial_child_rows: Sequence[Mapping[str, Any]],
    radial_history_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, float | int]],
    dict[str, dict[str, dict[str, float | int]]],
    dict[str, Any],
]:
    children = int(config["nested_monte_carlo"]["evaluation_children"])
    by_history_candidate: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (history_id, _), candidate_errors in control_errors.items():
        for candidate, error in candidate_errors.items():
            by_history_candidate[(history_id, candidate)].append(float(error))
    for row in radial_child_rows:
        by_history_candidate[(str(row["history_id"]), RADIAL)].append(
            float(row["squared_reference_error"])
        )
    expected_candidates = (*CONTROL_CANDIDATES, RADIAL)
    for history_id in histories:
        for candidate in expected_candidates:
            values = by_history_candidate[(history_id, candidate)]
            if len(values) != children or not all(
                math.isfinite(value) for value in values
            ):
                raise RuntimeError("Paired child matrix is incomplete or non-finite")
    history_means = {
        key: statistics.fmean(values) for key, values in by_history_candidate.items()
    }
    seeds = [int(value) for value in config["randomness"]["evaluation_outer_seeds"]]
    seed_rows: list[dict[str, Any]] = []
    stratified_rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_ids = [
            history_id
            for history_id, row in histories.items()
            if int(row["seed"]) == seed
        ]
        seed_rows.append(
            _integrated_row(
                seed=seed, history_ids=seed_ids, history_means=history_means
            )
        )
        for stratum_type, field in (
            ("noise_regime", "noise_regime"),
            ("threat", "threat"),
        ):
            values = sorted({str(histories[key][field]) for key in seed_ids})
            for value in values:
                subset = [
                    key for key in seed_ids if str(histories[key][field]) == value
                ]
                row = _integrated_row(
                    seed=seed, history_ids=subset, history_means=history_means
                )
                stratified_rows.append(
                    {"stratum_type": stratum_type, "stratum_value": value, **row}
                )

    t_critical = float(config["statistical_analysis"]["t_critical_df11"])
    overall_ci_keys = (
        "gain_vs_k4b",
        "gain_vs_k4",
        "relative_loss_vs_k5_1d",
        "ch_capture_fraction",
        "relative_k4_k4c_headroom",
        "remaining_mse_above_k4c_fraction_of_headroom",
        "relative_loss_vs_pointwise",
    )
    confidence_intervals = {
        key: _student_ci([float(row[key]) for row in seed_rows], t_critical)
        for key in overall_ci_keys
    }
    stratified_intervals: dict[str, dict[str, dict[str, float | int]]] = {}
    for stratum_type in ("noise_regime", "threat"):
        values = sorted(
            {
                str(row["stratum_value"])
                for row in stratified_rows
                if row["stratum_type"] == stratum_type
            }
        )
        stratified_intervals[stratum_type] = {}
        for value in values:
            rows = [
                row
                for row in stratified_rows
                if row["stratum_type"] == stratum_type and row["stratum_value"] == value
            ]
            if len(rows) != EXPECTED_OUTER_SEEDS:
                raise RuntimeError("Each stratum must contain all twelve outer seeds")
            stratified_intervals[stratum_type][value] = _student_ci(
                [float(row["gain_vs_k4b"]) for row in rows], t_critical
            )

    norm_summary = {
        "rolling_direction_norm": _distribution_summary(
            [float(row["rolling_direction_norm"]) for row in radial_history_rows]
        ),
        "radial_predictor_norm": _distribution_summary(
            [float(row["radial_predictor_norm"]) for row in radial_history_rows]
        ),
        "radial_over_rolling_norm": _distribution_summary(
            [float(row["radial_over_rolling_norm"]) for row in radial_history_rows]
        ),
        "norm_floor_active_fraction": statistics.fmean(
            float(bool(row["norm_floor_active"])) for row in radial_history_rows
        ),
        "at_public_cap_fraction": statistics.fmean(
            float(bool(row["at_public_cap"])) for row in radial_history_rows
        ),
        "projection_active_fraction": statistics.fmean(
            float(bool(row["projection_active"])) for row in radial_history_rows
        ),
    }
    return (
        seed_rows,
        stratified_rows,
        confidence_intervals,
        stratified_intervals,
        norm_summary,
    )


def _scientific_decision(
    confidence_intervals: Mapping[str, Mapping[str, float | int]],
    stratified_intervals: Mapping[str, Mapping[str, Mapping[str, float | int]]],
) -> tuple[dict[str, bool], bool, str]:
    gates = PREREGISTERED_SCIENTIFIC_GATES
    homogeneous = stratified_intervals["noise_regime"]["homogeneous"]
    heteroscedastic = stratified_intervals["noise_regime"]["heteroscedastic"]
    checks = {
        "gain_vs_k4b_mean": float(confidence_intervals["gain_vs_k4b"]["mean"])
        >= gates["gain_vs_k4b_mean_min"],
        "gain_vs_k4b_ci": float(confidence_intervals["gain_vs_k4b"]["low"])
        > gates["gain_vs_k4b_ci95_low_strictly_greater_than"],
        "gain_vs_k4_mean": float(confidence_intervals["gain_vs_k4"]["mean"])
        >= gates["gain_vs_k4_mean_min"],
        "gain_vs_k4_ci": float(confidence_intervals["gain_vs_k4"]["low"])
        > gates["gain_vs_k4_ci95_low_strictly_greater_than"],
        "noninferior_to_k5_1d": float(
            confidence_intervals["relative_loss_vs_k5_1d"]["high"]
        )
        < gates["relative_loss_vs_k5_1d_ci95_high_strictly_less_than"],
        "homogeneous_gain_vs_k4b_ci": float(homogeneous["low"])
        > gates["homogeneous_gain_vs_k4b_ci95_low_strictly_greater_than"],
        "heteroscedastic_gain_vs_k4b_ci": float(heteroscedastic["low"])
        > gates["heteroscedastic_gain_vs_k4b_ci95_low_strictly_greater_than"],
    }
    passed = all(checks.values())
    decision = (
        "advance_radial_confidence_k6_development"
        if passed
        else "stop_unmodulated_radial_predictor_instance"
    )
    return checks, passed, decision


@torch.no_grad()
def run(config_path: Path, k5_results: Path, output: Path) -> dict[str, Any]:
    """Execute the paired radial prescreen and write a fail-closed audit trail."""

    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError(
            "Set PYTORCH_ENABLE_MPS_FALLBACK=0; this screen refuses silent CPU fallback"
        )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite a prescreen directory: {output}")
    temporary_output = output.with_name(output.name + ".tmp")
    if temporary_output.exists():
        raise FileExistsError(
            f"A prior incomplete temporary directory exists: {temporary_output}"
        )

    (
        config,
        histories,
        control_errors,
        child_metadata,
        expected_child_seeds,
        source_hashes_before,
        source_diagnostics,
    ) = _validate_source(config_path, k5_results)
    oracle._configure_runtime("mps")
    if oracle._RUNTIME_DEVICE.type != "mps" or oracle._RUNTIME_DTYPE != torch.float32:
        raise RuntimeError("The radial prescreen requires MPS float32")

    k2_calibration, temporal_calibration, _ = k5.v1._load_calibrations(config)
    seeds = [int(value) for value in config["randomness"]["evaluation_outer_seeds"]]
    children = int(config["nested_monte_carlo"]["evaluation_children"])
    cap = float(config["references"]["total_client_influence_cap"])
    radial_history_rows: list[dict[str, Any]] = []
    radial_child_rows: list[dict[str, Any]] = []
    context_ids: set[str] = set()
    regenerated_child_seeds: list[int] = []
    expected_regenerated_child_seeds: list[int] = []
    maximum_k4_mse_error = 0.0
    maximum_k4b_mse_error = 0.0
    maximum_k4b_reference_error = 0.0
    maximum_slot_contribution_norm = 0.0
    all_contributions_bounded = True
    progress_seed: int | None = None
    progress_completed = 0

    for context in k5._snapshot_contexts(
        config,
        k2_calibration,
        temporal_calibration,
        split="evaluation",
        seeds=seeds,
    ):
        current_seed = int(context["cell"]["seed"])
        if progress_seed is not None and current_seed != progress_seed:
            progress_completed += 1
            print(
                json.dumps(
                    {
                        "screen_id": SCREEN_ID,
                        "phase": "paired_radial_development_prescreen",
                        "outer_seed": progress_seed,
                        "outer_seeds_completed": progress_completed,
                        "outer_seeds_total": len(seeds),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        progress_seed = current_seed
        history_id = str(context["history_id"])
        if history_id in context_ids or history_id not in histories:
            raise RuntimeError("Regenerated K5 history ID is duplicate or unexpected")
        context_ids.add(history_id)
        persisted_history = histories[history_id]
        if not _metadata_matches(context, persisted_history):
            raise RuntimeError("Regenerated history metadata differs from K5 source")
        if k5._tensor_hash(context["feature"]) != persisted_history["feature_hash"]:
            raise RuntimeError("Regenerated K5 past feature hash changed")
        expected_rounds = ",".join(str(value) for value in context["source_rounds"])
        if (
            expected_rounds != persisted_history["feature_source_rounds"]
            or int(context["feature_max_source_round"])
            > int(context["round_index"]) - 1
        ):
            raise RuntimeError("Radial predictor source is not strictly past-only")

        rolling = context["feature"][0]
        if rolling.device.type != "mps" or rolling.dtype != torch.float32:
            raise RuntimeError("Radial input did not remain MPS float32")
        inference_payload = {
            "rolling_accepted_mean_t_minus_4_through_t_minus_1": (
                k5._tensor_hash(rolling)
            ),
            "public_influence_cap": cap,
            "public_minimum_direction_norm": RADIAL_MINIMUM_DIRECTION_NORM,
        }
        if k5.forbidden_current_field_count(inference_payload) != 0:
            raise RuntimeError("Radial inference payload contains a forbidden field")
        radial, radial_diagnostics = radial_confidence_predictor(
            rolling,
            1.0,
            influence_cap=cap,
            minimum_direction_norm=RADIAL_MINIMUM_DIRECTION_NORM,
            return_diagnostics=True,
        )
        radial_norm = float(radial_diagnostics["predictor_norm"])
        rolling_norm = float(radial_diagnostics["direction_norm"])
        radial_denominator = float(radial_diagnostics["radial_denominator"])
        cap_tolerance = 128.0 * torch.finfo(torch.float32).eps * max(1.0, cap)
        if radial.device.type != "mps" or radial.dtype != torch.float32:
            raise RuntimeError("Radial predictor did not remain MPS float32")
        if radial_norm > cap + cap_tolerance:
            raise RuntimeError("Radial predictor violates the public influence cap")
        radial_history_rows.append(
            {
                "history_id": history_id,
                "seed": current_seed,
                "noise_regime": str(context["cell"]["regime"]["name"]),
                "noise_permutation": str(context["cell"]["permutation"]),
                "outlier_geometry": str(context["cell"]["geometry"]),
                "honest_dynamics": str(context["cell"]["dynamics"]),
                "threat": str(context["cell"]["threat"]),
                "assessment_round": int(context["round_index"]),
                "feature_max_source_round": int(context["feature_max_source_round"]),
                "rolling_direction_hash": k5._tensor_hash(rolling),
                "radial_predictor_hash": k5._tensor_hash(radial),
                "rolling_direction_norm": rolling_norm,
                "radial_predictor_norm": radial_norm,
                "radial_over_rolling_norm": _ratio(radial_norm, radial_denominator),
                "minimum_direction_norm": RADIAL_MINIMUM_DIRECTION_NORM,
                "norm_floor_active": bool(radial_diagnostics["norm_floor_active"]),
                "projection_active": bool(radial_diagnostics["projection_active"]),
                "at_public_cap": abs(radial_norm - cap) <= cap_tolerance,
                "influence_cap": cap,
                "uses_current_round_input": False,
                "forbidden_current_inference_field_count": 0,
            }
        )

        aware_radii = k5.k4._radii(
            config,
            context["components"]["variances"],
            k2_calibration,
            regime_name=str(context["cell"]["regime"]["name"]),
            blind=False,
        )
        for child in range(children):
            key = (history_id, child)
            if key not in expected_child_seeds or key not in control_errors:
                raise RuntimeError("Regenerated child key is not in the K5 source")
            values, vectors, child_seed = k5.v1._child_values(
                config,
                k2_calibration,
                temporal_calibration,
                context,
                stream="evaluation",
                child=child,
            )
            regenerated_child_seeds.append(int(child_seed))
            expected_regenerated_child_seeds.append(expected_child_seeds[key])
            if int(child_seed) != expected_child_seeds[key]:
                raise RuntimeError("Regenerated child seed differs from K5 registry")
            if not _metadata_matches(context, child_metadata[key]):
                raise RuntimeError("Regenerated child metadata differs from K5 source")
            if values["target"].device.type != "mps":
                raise RuntimeError("Current-child scoring target did not remain on MPS")

            fixed_k4b = k5.v1._fixed_reference(context, values, rolling, config)
            reproduced_k4b, _ = k5.k4b._k4b_reference(
                vectors,
                anchor=context["components"]["anchor"],
                aware_radii=aware_radii,
                history=context["history"],
                enrollment_mean=context["enrollment_mean"],
                predictor=rolling,
                predictor_role="k6_radial_prescreen_k4b_reproduction",
                imputation_mode=k5.FULL_TEMPORAL_MISSING_SLOT,
                deployable=False,
                privacy_claimed=True,
                temporal_calibration=temporal_calibration,
                config=config,
            )
            k4b_reference_error = float(
                torch.linalg.vector_norm(fixed_k4b - reproduced_k4b).item()
            )
            maximum_k4b_reference_error = max(
                maximum_k4b_reference_error, k4b_reference_error
            )
            k4_mse = float(
                torch.sum((values["k4_reference"] - values["target"]).square()).item()
            )
            k4b_mse = float(torch.sum((fixed_k4b - values["target"]).square()).item())
            maximum_k4_mse_error = max(
                maximum_k4_mse_error,
                abs(k4_mse - control_errors[key][k5.K4]),
            )
            maximum_k4b_mse_error = max(
                maximum_k4b_mse_error,
                abs(k4b_mse - control_errors[key][k5.K4B]),
            )
            radial_reference = k5.v1._fixed_reference(context, values, radial, config)
            radial_mse = float(
                torch.sum((radial_reference - values["target"]).square()).item()
            )
            contributions = (
                values["gates"][:, None] * values["clipped"]
                + (1.0 - values["history_gates"])[:, None] * radial[None, :]
            )
            contribution_norms = torch.linalg.vector_norm(contributions, dim=1)
            child_max_contribution = float(torch.max(contribution_norms).item())
            maximum_slot_contribution_norm = max(
                maximum_slot_contribution_norm, child_max_contribution
            )
            child_cap_ok = bool((contribution_norms <= cap + cap_tolerance).all())
            all_contributions_bounded = all_contributions_bounded and child_cap_ok
            radial_child_rows.append(
                {
                    "history_id": history_id,
                    "seed": current_seed,
                    "noise_regime": str(context["cell"]["regime"]["name"]),
                    "noise_permutation": str(context["cell"]["permutation"]),
                    "outlier_geometry": str(context["cell"]["geometry"]),
                    "honest_dynamics": str(context["cell"]["dynamics"]),
                    "threat": str(context["cell"]["threat"]),
                    "assessment_round": int(context["round_index"]),
                    "evaluation_child": child,
                    "evaluation_child_seed": int(child_seed),
                    "candidate": RADIAL,
                    "squared_reference_error": radial_mse,
                    "reference_error_l2_descriptive": math.sqrt(radial_mse),
                    "radial_predictor_hash": k5._tensor_hash(radial),
                    "radial_predictor_fixed_before_current_child": True,
                    "predictor_observable_past_only": True,
                    "fixed_denominator_n": int(config["cohort"]["num_clients"]),
                    "normalization_by_gate_sum": False,
                    "contribution_cap_respected": child_cap_ok,
                    "max_slot_contribution_norm": child_max_contribution,
                    "k4_source_mse_abs_error": abs(k4_mse - control_errors[key][k5.K4]),
                    "k4b_source_mse_abs_error": abs(
                        k4b_mse - control_errors[key][k5.K4B]
                    ),
                    "k4b_reference_reproduction_error": k4b_reference_error,
                }
            )

    if progress_seed is not None:
        progress_completed += 1
        print(
            json.dumps(
                {
                    "screen_id": SCREEN_ID,
                    "phase": "paired_radial_development_prescreen",
                    "outer_seed": progress_seed,
                    "outer_seeds_completed": progress_completed,
                    "outer_seeds_total": len(seeds),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    torch.mps.synchronize()

    expected_histories = int(config["gates"]["evaluation_histories_exact"])
    expected_children = expected_histories * children
    regenerated_rng_exact = (
        len(regenerated_child_seeds) == expected_children
        and regenerated_child_seeds == expected_regenerated_child_seeds
    )
    regenerated_rng_count_unique = len(regenerated_child_seeds) == len(
        set(regenerated_child_seeds)
    )
    source_hashes_after = {
        name: _sha256(path) for name, path in _source_paths(k5_results).items()
    }
    validity_checks = {
        "device_mps": oracle._RUNTIME_DEVICE.type == "mps",
        "dtype_float32": oracle._RUNTIME_DTYPE == torch.float32,
        "mps_fallback_disabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0",
        "source_hashes_unchanged": source_hashes_before == source_hashes_after,
        "source_k5_validity_passed": source_diagnostics["source_k5_validity_pass"]
        is True,
        "holdout_not_opened": True,
        "development_uses_exactly_consumed_k5_evaluation_outer_seeds": len(seeds)
        == EXPECTED_OUTER_SEEDS,
        "history_matrix_exact": len(context_ids) == expected_histories
        and context_ids == set(histories),
        "radial_history_rows_exact": len(radial_history_rows) == expected_histories,
        "radial_child_rows_exact": len(radial_child_rows) == expected_children,
        "regenerated_child_seed_count_exact": len(regenerated_child_seeds)
        == expected_children,
        "regenerated_child_seed_unique": regenerated_rng_count_unique,
        "regenerated_rng_sequence_exact": regenerated_rng_exact,
        "k4_source_mse_recomputed": maximum_k4_mse_error
        <= SOURCE_RECOMPUTATION_ABS_TOLERANCE,
        "k4b_source_mse_recomputed": maximum_k4b_mse_error
        <= SOURCE_RECOMPUTATION_ABS_TOLERANCE,
        "k4b_reference_reproduced": maximum_k4b_reference_error
        <= REFERENCE_REPRODUCTION_ABS_TOLERANCE,
        "strict_past_predictor": all(
            int(row["feature_max_source_round"]) < int(row["assessment_round"])
            and row["uses_current_round_input"] is False
            and int(row["forbidden_current_inference_field_count"]) == 0
            for row in radial_history_rows
        ),
        "predictor_cap_respected": all(
            float(row["radial_predictor_norm"]) <= cap + 1.0e-6
            for row in radial_history_rows
        ),
        "slot_contribution_cap_respected": all_contributions_bounded,
        "all_metrics_finite": all(
            math.isfinite(float(row["squared_reference_error"]))
            and float(row["squared_reference_error"]) >= 0.0
            for row in radial_child_rows
        ),
    }
    if not all(validity_checks.values()):
        failed = [key for key, value in validity_checks.items() if not value]
        raise RuntimeError("Radial prescreen validity failure: " + ", ".join(failed))

    (
        seed_rows,
        stratified_rows,
        confidence_intervals,
        stratified_intervals,
        norm_summary,
    ) = _summaries(
        config=config,
        histories=histories,
        control_errors=control_errors,
        radial_child_rows=radial_child_rows,
        radial_history_rows=radial_history_rows,
    )
    scientific_checks, scientific_pass, scientific_decision = _scientific_decision(
        confidence_intervals, stratified_intervals
    )
    decision = {
        "screen_id": SCREEN_ID,
        "scope": "post_k5_development_only_reusing_consumed_evaluation_seeds",
        "candidate": RADIAL,
        "formula": "Proj_B_G(G*Y/max(norm_Y,r_min)); kappa fixed to one",
        "predictor_source": "K4b rolling accepted mean through t-1",
        "influence_cap": cap,
        "minimum_direction_norm": RADIAL_MINIMUM_DIRECTION_NORM,
        "scientific_gates_frozen_in_source_before_execution": (
            PREREGISTERED_SCIENTIFIC_GATES
        ),
        "validity_checks": validity_checks,
        "validity_pass": True,
        "scientific_checks": scientific_checks,
        "scientific_checks_pass": scientific_pass,
        "all_gates_pass": scientific_pass,
        "decision": scientific_decision,
        "confidence_intervals": confidence_intervals,
        "stratified_gain_vs_k4b_confidence_intervals": stratified_intervals,
        "norm_and_cap_diagnostics": norm_summary,
        "holdout_opened": False,
        "authorizes_holdout_or_promotion": False,
        "interpretation_limit": (
            "This development screen can reject or motivate a separately locked K6 "
            "confirmation. Reused K5 evaluation seeds forbid confirmatory claims."
        ),
    }
    diagnostics = {
        "source": source_diagnostics,
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "screen_script_sha256": _sha256(Path(__file__)),
        "maximum_k4_source_mse_abs_error": maximum_k4_mse_error,
        "maximum_k4b_source_mse_abs_error": maximum_k4b_mse_error,
        "maximum_k4b_reference_reproduction_error": maximum_k4b_reference_error,
        "maximum_slot_contribution_norm": maximum_slot_contribution_norm,
        "evaluation_child_seed_count": len(regenerated_child_seeds),
        "evaluation_child_seed_unique_count": len(set(regenerated_child_seeds)),
        "current_round_target_role": (
            "used_only_after_predictor_freeze_for_paired_scoring"
        ),
        "reserved_holdout_seed_count_generated": 0,
    }

    temporary_output.mkdir(parents=True)
    _write_csv(temporary_output / "radial_history_rows.csv", radial_history_rows)
    _write_csv(temporary_output / "radial_child_rows.csv", radial_child_rows)
    _write_csv(temporary_output / "seed_summary.csv", seed_rows)
    _write_csv(temporary_output / "stratified_seed_summary.csv", stratified_rows)
    _write_json(temporary_output / "decision.json", decision)
    _write_json(temporary_output / "diagnostics.json", diagnostics)
    manifest = {
        "screen_id": SCREEN_ID,
        "status": "completed_development_prescreen",
        "device": "mps",
        "dtype": "torch.float32",
        "source_campaign": k5.CAMPAIGN_ID,
        "source_scope": "already_consumed_k5_evaluation_seeds",
        "source_hashes": source_hashes_before,
        "screen_script_sha256": diagnostics["screen_script_sha256"],
        "outer_seeds": seeds,
        "histories": len(radial_history_rows),
        "children": len(radial_child_rows),
        "decision": scientific_decision,
        "validity_pass": True,
        "scientific_checks_pass": scientific_pass,
        "all_gates_pass": scientific_pass,
        "holdout_opened": False,
        "authorizes_holdout_or_promotion": False,
    }
    _write_json(temporary_output / "manifest.json", manifest)
    temporary_output.replace(output)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--k5-results", type=Path, default=DEFAULT_K5_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.config, args.k5_results, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
