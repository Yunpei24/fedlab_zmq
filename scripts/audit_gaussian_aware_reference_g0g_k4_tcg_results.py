#!/usr/bin/env python3
"""Independently audit the raw G0g-K4-TCG development artifacts.

This module deliberately does not import the experiment runner.  It rebuilds
the preregistered matrix, trajectory summaries, paired seed contrasts and gate
decisions from the CSV files plus the frozen YAML contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "g0g_k4_temporal_causal_gate"
COUNTERFACTUAL = "g0g_k4_no_compromise_counterfactual_control"
AWARE = "g0g_k2"
K3 = "g0g_k3_dual_gate"
FCC = "fcc"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_bool(series: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    lowered = series.astype(str).str.strip().str.lower()
    valid = lowered.isin(("true", "false"))
    if not bool(valid.all()):
        examples = sorted(set(series.loc[~valid].astype(str)))[:5]
        raise ValueError(f"{name} contains non-Boolean values: {examples}")
    return lowered.eq("true")


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def _finite_std(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.std(ddof=1)) if finite.size > 1 else 0.0


def _ci95(values: Sequence[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"n": 0, "mean": float("nan"), "low": float("nan"), "high": float("nan")}
    mean = float(statistics.fmean(finite))
    if len(finite) == 1:
        return {"n": 1, "mean": mean, "low": float("nan"), "high": float("nan")}
    critical = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }.get(len(finite), 1.96)
    half = critical * statistics.stdev(finite) / math.sqrt(len(finite))
    return {"n": len(finite), "mean": mean, "low": mean - half, "high": mean + half}


def _allclose(left: pd.Series, right: pd.Series, *, atol: float = 1.0e-11) -> bool:
    return bool(
        np.allclose(
            pd.to_numeric(left, errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(right, errors="coerce").to_numpy(dtype=float),
            rtol=1.0e-10,
            atol=atol,
            equal_nan=True,
        )
    )


def _max_abs(left: pd.Series, right: pd.Series) -> float:
    difference = np.abs(
        pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
        - pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    )
    finite = difference[np.isfinite(difference)]
    return float(finite.max()) if finite.size else 0.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _compare_tree(
    expected: Any,
    observed: Any,
    *,
    path: str = "",
    differences: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if differences is None:
        differences = []
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        for key in sorted(set(expected) | set(observed)):
            child = f"{path}.{key}" if path else str(key)
            if key not in expected or key not in observed:
                differences.append(
                    {"path": child, "recomputed": expected.get(key), "reported": observed.get(key)}
                )
            else:
                _compare_tree(expected[key], observed[key], path=child, differences=differences)
        return differences
    if isinstance(expected, bool) or isinstance(observed, bool):
        same = type(expected) is type(observed) and expected == observed
    elif isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        left = float(expected)
        right = float(observed)
        same = (math.isnan(left) and math.isnan(right)) or math.isclose(
            left, right, rel_tol=1.0e-10, abs_tol=1.0e-11
        )
    else:
        same = expected == observed
    if not same:
        differences.append({"path": path, "recomputed": expected, "reported": observed})
    return differences


class Audit:
    def __init__(self) -> None:
        self.checks: dict[str, bool] = {}
        self.details: dict[str, Any] = {}

    def check(self, name: str, condition: Any, detail: Any | None = None) -> bool:
        passed = bool(condition)
        self.checks[name] = passed
        if detail is not None:
            self.details[name] = detail
        return passed


def _expected_matrix(config: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str]]]:
    noise_cells = [
        (str(regime["name"]), str(permutation))
        for regime in config["privacy_noise"]["regimes"]
        for permutation in regime["permutations"]
    ]
    expected: dict[str, dict[str, Any]] = {}
    for seed, (regime, permutation), geometry, dynamics in product(
        [int(value) for value in config["randomness"]["development_seeds"]],
        noise_cells,
        [str(value) for value in config["cohort"]["honest_outliers"]["geometries"]],
        [str(value) for value in config["honest_dynamics"]["names"]],
    ):
        cells = [
            (str(threat), str(schedule))
            for threat, schedule in product(
                config["threats"]["names"], config["threats"]["schedules"]
            )
        ]
        cells.append(
            (
                str(config["no_compromise_control"]["threat_name"]),
                str(config["no_compromise_control"]["schedule_name"]),
            )
        )
        for threat, schedule in cells:
            identifier = "|".join(
                (str(seed), regime, permutation, geometry, dynamics, threat, schedule)
            )
            expected[identifier] = {
                "seed": seed,
                "noise_regime": regime,
                "noise_permutation": permutation,
                "outlier_geometry": geometry,
                "honest_dynamics": dynamics,
                "threat": threat,
                "schedule": schedule,
            }
    return expected, noise_cells


def _axis_matches(
    frame: pd.DataFrame,
    expected: Mapping[str, Mapping[str, Any]],
    fields: Sequence[str],
) -> bool:
    identifiers_valid = frame["trajectory_id"].isin(expected)
    if not bool(identifiers_valid.all()):
        return False
    for field in fields:
        mapped = frame["trajectory_id"].map(lambda value: expected[str(value)][field])
        if field == "seed":
            if not bool(pd.to_numeric(frame[field], errors="coerce").eq(mapped).all()):
                return False
        elif not bool(frame[field].astype(str).eq(mapped.astype(str)).all()):
            return False
    return True


def _phase(round_number: pd.Series, temporal: Mapping[str, Any]) -> pd.Series:
    values = np.select(
        [
            round_number.le(int(temporal["enrollment_rounds"])),
            round_number.lt(int(temporal["attack_start_round"])),
            round_number.le(int(temporal["attack_end_round"])),
        ],
        ["enrollment", "monitoring", "attack"],
        default="recovery",
    )
    return pd.Series(values, index=round_number.index)


def _attack_active(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.Series:
    rounds = pd.to_numeric(frame["round"], errors="coerce").astype(int)
    start = int(config["temporal"]["attack_start_round"])
    end = int(config["temporal"]["attack_end_round"])
    in_window = rounds.between(start, end)
    persistent = frame["schedule"].eq("persistent") & in_window
    intermittent = frame["schedule"].eq("intermittent_2_on_1_off") & in_window
    pattern = [bool(value) for value in config["threats"]["intermittent_pattern"]]
    pattern_active = rounds.map(lambda value: pattern[(int(value) - start) % len(pattern)])
    return persistent | (intermittent & pattern_active)


def _completeness(
    round_rows: pd.DataFrame,
    client_rows: pd.DataFrame,
    trajectory_rows: pd.DataFrame,
    replace_rows: pd.DataFrame,
    expected: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    candidates: Sequence[str],
    noise_cells: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    expected_ids = set(expected)
    total_rounds = int(config["temporal"]["total_rounds"])
    first_gate = int(config["temporal"]["first_temporal_gate_round"])
    n = int(config["cohort"]["num_clients"])

    round_valid = (
        round_rows["trajectory_id"].isin(expected_ids)
        & round_rows["candidate"].isin(candidates)
        & pd.to_numeric(round_rows["round"], errors="coerce").between(1, total_rounds)
    )
    trajectory_valid = trajectory_rows["trajectory_id"].isin(expected_ids) & trajectory_rows[
        "candidate"
    ].isin(candidates)
    client_valid = (
        client_rows["trajectory_id"].isin(expected_ids)
        & pd.to_numeric(client_rows["round"], errors="coerce").between(first_gate, total_rounds)
        & pd.to_numeric(client_rows["client"], errors="coerce").between(0, n - 1)
    )

    def counts(frame: pd.DataFrame, valid: pd.Series, keys: list[str], expected_count: int) -> dict[str, int]:
        valid_frame = frame.loc[valid, keys]
        unique = int(len(valid_frame.drop_duplicates()))
        valid_count = int(valid.sum())
        return {
            "observed": int(len(frame)),
            "expected": int(expected_count),
            "missing_unique_keys": int(expected_count - unique),
            "duplicate_keys": int(valid_count - unique),
            "unexpected_rows": int((~valid).sum()),
        }

    expected_round = len(expected_ids) * len(candidates) * total_rounds
    expected_trajectory = len(expected_ids) * len(candidates)
    expected_client = len(expected_ids) * (total_rounds - first_gate + 1) * n
    round_counts = counts(
        round_rows, round_valid, ["trajectory_id", "candidate", "round"], expected_round
    )
    trajectory_counts = counts(
        trajectory_rows,
        trajectory_valid,
        ["trajectory_id", "candidate"],
        expected_trajectory,
    )
    client_counts = counts(
        client_rows, client_valid, ["trajectory_id", "round", "client"], expected_client
    )

    trials = int(config["randomness"]["replace_one_trials_per_seed_noise_cell"])
    expected_replace = {
        (int(seed), regime, permutation, trial)
        for seed, (regime, permutation), trial in product(
            config["randomness"]["development_seeds"], noise_cells, range(trials)
        )
    }
    observed_keys = list(
        zip(
            pd.to_numeric(replace_rows["seed"], errors="coerce").astype(int),
            replace_rows["noise_regime"].astype(str),
            replace_rows["noise_permutation"].astype(str),
            pd.to_numeric(replace_rows["trial"], errors="coerce").astype(int),
        )
    )
    valid_replace = pd.Series([key in expected_replace for key in observed_keys], index=replace_rows.index)
    valid_key_count = len(set(key for key in observed_keys if key in expected_replace))
    replace_counts = {
        "observed": int(len(replace_rows)),
        "expected": int(len(expected_replace)),
        "missing_unique_keys": int(len(expected_replace) - valid_key_count),
        "duplicate_keys": int(valid_replace.sum() - valid_key_count),
        "unexpected_rows": int((~valid_replace).sum()),
    }
    unique_fraction = min(
        (expected_round - round_counts["missing_unique_keys"]) / expected_round,
        (expected_trajectory - trajectory_counts["missing_unique_keys"]) / expected_trajectory,
        (expected_client - client_counts["missing_unique_keys"]) / expected_client,
        (len(expected_replace) - replace_counts["missing_unique_keys"]) / len(expected_replace),
    )
    exact = unique_fraction == 1.0 and not any(
        item
        for block in (round_counts, trajectory_counts, client_counts, replace_counts)
        for key, item in block.items()
        if key in {"duplicate_keys", "unexpected_rows"}
    )
    return {
        "exact_identifier_completeness": bool(exact),
        "complete_fraction": float(unique_fraction),
        "expected_trajectory_ids": len(expected_ids),
        "missing_trajectory_ids": len(expected_ids - set(trajectory_rows["trajectory_id"].astype(str))),
        "round_rows": round_counts,
        "trajectory_rows": trajectory_counts,
        "client_rows": client_counts,
        "replace_one_rows": replace_counts,
    }


def _rebuild_trajectories(round_rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["trajectory_id", "candidate"]
    metadata = [
        "seed",
        "noise_regime",
        "noise_permutation",
        "outlier_geometry",
        "honest_dynamics",
        "threat",
        "schedule",
    ]
    result = round_rows.groupby(keys, sort=False)[metadata].first()

    def phase_mean(mask: pd.Series, value: str) -> pd.Series:
        return round_rows.loc[mask].groupby(keys, sort=False)[value].mean()

    result["enrollment_auc"] = phase_mean(round_rows["phase"].eq("enrollment"), "reference_error")
    result["monitoring_auc"] = phase_mean(round_rows["phase"].eq("monitoring"), "reference_error")
    result["attack_auc"] = phase_mean(round_rows["phase"].eq("attack"), "reference_error")
    result["active_attack_auc"] = phase_mean(
        round_rows["phase"].eq("attack") & round_rows["attack_active_bool"],
        "reference_error",
    )
    result["recovery_auc"] = phase_mean(round_rows["phase"].eq("recovery"), "reference_error")
    result["post_enrollment_auc"] = phase_mean(
        round_rows["phase"].isin(("attack", "recovery")), "reference_error"
    )
    result["attack_byzantine_contribution_mass"] = phase_mean(
        round_rows["phase"].eq("attack"), "byzantine_contribution_mass"
    )
    return result.reset_index()


def _event_summaries(client_rows: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    byzantine = client_rows.loc[client_rows["latent_byzantine_bool"]].copy()
    wide = byzantine.pivot(
        index=["trajectory_id", "client"], columns="round", values="temporal_gate"
    )
    threshold_detection = float(config["temporal"]["detection_gate_threshold"])
    detection_end = pd.Series(np.nan, index=wide.index, dtype=float)
    for end_round in range(
        int(config["temporal"]["first_eligible_detection_round"]),
        int(config["temporal"]["detection_deadline_round"]) + 1,
    ):
        event = wide[end_round - 1].le(threshold_detection) & wide[end_round].le(
            threshold_detection
        )
        detection_end.loc[detection_end.isna() & event] = float(end_round)

    threshold_recovery = float(config["temporal"]["recovery_gate_threshold"])
    recovery_end = pd.Series(np.nan, index=wide.index, dtype=float)
    for end_round in range(
        int(config["temporal"]["first_fully_clean_window_round"]) + 1,
        int(config["temporal"]["recovery_deadline_round"]) + 1,
    ):
        event = wide[end_round - 1].ge(threshold_recovery) & wide[end_round].ge(
            threshold_recovery
        )
        recovery_end.loc[recovery_end.isna() & event] = float(end_round)

    events = pd.DataFrame(
        {
            "detection_end": detection_end,
            "recovery_end": recovery_end,
        }
    ).reset_index()
    grouped = events.groupby("trajectory_id", sort=False)
    return grouped.agg(
        detection_rate_within_deadline=("detection_end", lambda values: float(values.notna().mean())),
        mean_detection_end_round=("detection_end", "mean"),
        recovery_rate_within_deadline=("recovery_end", lambda values: float(values.notna().mean())),
        mean_recovery_end_round=("recovery_end", "mean"),
        detection_end_15=("detection_end", lambda values: int(values.eq(15.0).sum())),
        detection_end_16=("detection_end", lambda values: int(values.eq(16.0).sum())),
        detection_end_17=("detection_end", lambda values: int(values.eq(17.0).sum())),
        recovery_end_30=("recovery_end", lambda values: int(values.eq(30.0).sum())),
        latent_clients=("client", "size"),
    ).reset_index()


def _paired_seed_contrast(
    rows: pd.DataFrame,
    mask: pd.Series,
    competitor: str,
    metric: str,
) -> tuple[list[float], dict[str, Any]]:
    selected = rows.loc[mask & rows["candidate"].isin((PRIMARY, competitor))]
    pivot = selected.pivot(index=["trajectory_id", "seed"], columns="candidate", values=metric)
    complete = pivot[[PRIMARY, competitor]].notna().all(axis=1)
    deltas = pivot.loc[complete, PRIMARY] - pivot.loc[complete, competitor]
    per_seed = deltas.groupby(level="seed").mean().sort_index()
    evidence = {
        "competitor": competitor,
        "metric": metric,
        "paired_cells": int(complete.sum()),
        "incomplete_cells": int((~complete).sum()),
        "per_seed": {str(int(seed)): float(value) for seed, value in per_seed.items()},
    }
    return [float(value) for value in per_seed], evidence


def _ratio(
    rows: pd.DataFrame,
    mask: pd.Series,
    candidate: str,
    baseline: str,
    metric: str,
) -> float:
    candidate_mean = _finite_mean(rows.loc[mask & rows["candidate"].eq(candidate), metric])
    baseline_mean = _finite_mean(rows.loc[mask & rows["candidate"].eq(baseline), metric])
    return candidate_mean / max(baseline_mean, 1.0e-12)


def _calibration_audit(
    rows: pd.DataFrame,
    calibration: Mapping[str, Any],
    config: Mapping[str, Any],
    noise_cells: Sequence[tuple[str, str]],
    audit: Audit,
) -> tuple[float, float, dict[str, Any]]:
    settings = config["temporal_calibration"]
    expected_contexts = {
        "|".join((regime, permutation, str(geometry), str(dynamics)))
        for (regime, permutation), geometry, dynamics in product(
            noise_cells,
            config["cohort"]["honest_outliers"]["geometries"],
            config["honest_dynamics"]["names"],
        )
    }
    expected_keys = {
        (context, int(root), int(draw))
        for context, root, draw in product(
            expected_contexts,
            settings["roots"],
            range(int(settings["trajectories_per_root_per_context"])),
        )
    }
    observed_keys = list(
        zip(
            rows["context"].astype(str),
            pd.to_numeric(rows["root_seed"], errors="coerce").astype(int),
            pd.to_numeric(rows["draw"], errors="coerce").astype(int),
        )
    )
    audit.check("calibration_row_count", len(rows) == len(expected_keys), {"observed": len(rows), "expected": len(expected_keys)})
    audit.check("calibration_exact_keys", set(observed_keys) == expected_keys and len(observed_keys) == len(set(observed_keys)))
    values = pd.to_numeric(rows["maximum_temporal_statistic"], errors="coerce")
    audit.check("calibration_statistics_finite_nonnegative", bool(np.isfinite(values).all() and values.ge(0.0).all()))
    audit.check("calibration_trajectory_seeds_unique", rows["trajectory_seed"].nunique() == len(rows))

    c0_rank = int(settings["c0_order_statistic_one_indexed"])
    c1_rank = int(settings["c1_order_statistic_one_indexed"])
    recomputed: dict[str, dict[str, float | int]] = {}
    for context, group in rows.groupby("context", sort=True):
        ordered = np.sort(pd.to_numeric(group["maximum_temporal_statistic"]).to_numpy(dtype=float))
        recomputed[str(context)] = {
            "trajectory_count": int(len(ordered)),
            "c0_order_statistic": float(ordered[c0_rank - 1]),
            "c1_order_statistic": float(ordered[c1_rank - 1]),
            "c0_rank_one_indexed": c0_rank,
            "c1_rank_one_indexed": c1_rank,
        }
    reported_contexts = {str(row["context"]): row for row in calibration["contexts"]}
    context_differences: list[dict[str, Any]] = []
    for context in sorted(expected_contexts):
        expected_row = {"context": context, **recomputed.get(context, {})}
        _compare_tree(expected_row, reported_contexts.get(context, {}), path=context, differences=context_differences)
    audit.check("calibration_context_order_statistics_match_json", not context_differences, context_differences[:20])
    deployed_c0 = max(float(row["c0_order_statistic"]) for row in recomputed.values())
    deployed_c1 = max(float(row["c1_order_statistic"]) for row in recomputed.values())
    audit.check(
        "calibration_deployed_thresholds_match_json",
        math.isclose(deployed_c0, float(calibration["deployed_c0"]), rel_tol=1e-10, abs_tol=1e-11)
        and math.isclose(deployed_c1, float(calibration["deployed_c1"]), rel_tol=1e-10, abs_tol=1e-11)
        and deployed_c1 > deployed_c0,
        {"deployed_c0": deployed_c0, "deployed_c1": deployed_c1},
    )
    audit.check("calibration_holdout_closed", calibration.get("holdout_opened") is False)
    return deployed_c0, deployed_c1, {"contexts": recomputed}


def audit_results(config_path: Path, results_dir: Path, output_path: Path) -> dict[str, Any]:
    required = {
        "manifest": results_dir / "manifest.json",
        "calibration": results_dir / "temporal_calibration.json",
        "calibration_rows": results_dir / "temporal_calibration_trajectories.csv",
        "round_rows": results_dir / "development_round_rows.csv",
        "client_rows": results_dir / "development_client_rows.csv",
        "trajectory_rows": results_dir / "development_trajectory_rows.csv",
        "replace_rows": results_dir / "replace_one_audit.csv",
        "summary": results_dir / "summary.csv",
        "decision": results_dir / "decision.json",
        "k2_provenance": results_dir / "k2_calibration_provenance.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete result directory; missing: " + ", ".join(missing))

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = _read_json(required["manifest"])
    calibration = _read_json(required["calibration"])
    decision = _read_json(required["decision"])
    k2_provenance = _read_json(required["k2_provenance"])
    if manifest.get("status") != "completed_development":
        raise RuntimeError(f"Result directory is not complete: status={manifest.get('status')!r}")

    calibration_rows = pd.read_csv(required["calibration_rows"], low_memory=False)
    round_rows = pd.read_csv(required["round_rows"], low_memory=False)
    client_rows = pd.read_csv(required["client_rows"], low_memory=False)
    trajectory_rows = pd.read_csv(required["trajectory_rows"], low_memory=False)
    replace_rows = pd.read_csv(required["replace_rows"], low_memory=False)
    summary = pd.read_csv(required["summary"], low_memory=False)

    candidates = [str(value) for value in config["candidates"]["names"]]
    expected, noise_cells = _expected_matrix(config)
    audit = Audit()

    bool_columns = {
        "round": [
            "attack_active",
            "temporal_gate_all_one",
            "contribution_cap_respected",
            "normalization_by_gate_sum",
            "all_finite",
        ],
        "client": [
            "attack_active",
            "latent_byzantine",
            "active_byzantine",
            "honest_outlier",
            "honest_regular",
        ],
        "replace": ["same_past", "same_covariances", "same_anchor", "violation"],
    }
    for column in bool_columns["round"]:
        round_rows[f"{column}_bool"] = _as_bool(round_rows[column], name=f"round.{column}")
    for column in bool_columns["client"]:
        client_rows[f"{column}_bool"] = _as_bool(client_rows[column], name=f"client.{column}")
    for column in bool_columns["replace"]:
        replace_rows[f"{column}_bool"] = _as_bool(replace_rows[column], name=f"replace.{column}")

    expected_counts = {
        "calibration_rows": 5120,
        "development_trajectories": 900,
        "round_rows": 194400,
        "trajectory_rows": 5400,
        "client_rows": 540000,
        "replace_one_rows": 100,
        "summary_rows": 108,
    }
    audit.check("frozen_matrix_dimensions", len(expected) == 900 and len(noise_cells) == 5, {"trajectories": len(expected), "noise_cells": len(noise_cells)})

    axis_fields = [
        "seed",
        "noise_regime",
        "noise_permutation",
        "outlier_geometry",
        "honest_dynamics",
        "threat",
        "schedule",
    ]
    audit.check("round_axis_matches_trajectory_id", _axis_matches(round_rows, expected, axis_fields))
    audit.check("client_axis_matches_trajectory_id", _axis_matches(client_rows, expected, axis_fields))
    audit.check("trajectory_axis_matches_trajectory_id", _axis_matches(trajectory_rows, expected, axis_fields))

    completeness = _completeness(
        round_rows,
        client_rows,
        trajectory_rows,
        replace_rows,
        expected,
        config,
        candidates,
        noise_cells,
    )
    audit.check("exact_identifier_completeness", completeness["exact_identifier_completeness"], completeness)
    audit.check(
        "raw_row_counts",
        len(calibration_rows) == expected_counts["calibration_rows"]
        and len(round_rows) == expected_counts["round_rows"]
        and len(trajectory_rows) == expected_counts["trajectory_rows"]
        and len(client_rows) == expected_counts["client_rows"]
        and len(replace_rows) == expected_counts["replace_one_rows"]
        and len(summary) == expected_counts["summary_rows"],
        {
            "calibration_rows": len(calibration_rows),
            "round_rows": len(round_rows),
            "trajectory_rows": len(trajectory_rows),
            "client_rows": len(client_rows),
            "replace_one_rows": len(replace_rows),
            "summary_rows": len(summary),
        },
    )
    round_pair_sizes = round_rows.groupby("pairing_id").agg(size=("candidate", "size"), candidates=("candidate", "nunique"))
    audit.check(
        "round_candidate_pairing_complete",
        bool(round_pair_sizes["size"].eq(len(candidates)).all() and round_pair_sizes["candidates"].eq(len(candidates)).all()),
    )
    audit.check(
        "pairing_id_exact",
        bool(
            round_rows["pairing_id"].astype(str).eq(
                round_rows["trajectory_id"].astype(str) + "|" + round_rows["round"].astype(int).astype(str)
            ).all()
        ),
    )
    audit.check(
        "phase_labels_exact",
        bool(round_rows["phase"].astype(str).eq(_phase(round_rows["round"].astype(int), config["temporal"])).all()),
    )
    audit.check(
        "attack_schedule_exact_round_rows",
        bool(round_rows["attack_active_bool"].eq(_attack_active(round_rows, config)).all()),
    )
    audit.check(
        "attack_schedule_exact_client_rows",
        bool(client_rows["attack_active_bool"].eq(_attack_active(client_rows, config)).all()),
    )

    deployed_c0, deployed_c1, calibration_evidence = _calibration_audit(
        calibration_rows, calibration, config, noise_cells, audit
    )
    audit.check(
        "manifest_temporal_thresholds_match",
        math.isclose(float(manifest["temporal_c0"]), deployed_c0, rel_tol=1e-10, abs_tol=1e-11)
        and math.isclose(float(manifest["temporal_c1"]), deployed_c1, rel_tol=1e-10, abs_tol=1e-11),
    )

    core_round_fields = ["reference_error", "difference_to_k2_aware"]
    certified_round_fields = [
        "current_gate_mean",
        "final_gate_mean",
        "byzantine_contribution_mass",
        "max_client_contribution_norm",
    ]
    core_finite = all(np.isfinite(pd.to_numeric(round_rows[field], errors="coerce")).all() for field in core_round_fields)
    certified_mask = round_rows["candidate"].ne(FCC)
    certified_finite = all(
        np.isfinite(pd.to_numeric(round_rows.loc[certified_mask, field], errors="coerce")).all()
        for field in certified_round_fields
    )
    temporal_candidates = round_rows["candidate"].isin((PRIMARY, COUNTERFACTUAL))
    temporal_finite = np.isfinite(
        pd.to_numeric(
            round_rows.loc[temporal_candidates, "temporal_gate_mean"],
            errors="coerce",
        )
    ).all()
    audit.check(
        "round_required_metrics_finite",
        core_finite and certified_finite and temporal_finite,
    )
    audit.check("round_all_finite_certificate_true", bool(round_rows["all_finite_bool"].all()))
    audit.check(
        "client_metrics_finite",
        all(
            np.isfinite(pd.to_numeric(client_rows[field], errors="coerce")).all()
            for field in (
                "public_noise_tier",
                "temporal_statistic",
                "current_gate",
                "temporal_gate",
                "counterfactual_temporal_statistic",
                "counterfactual_temporal_gate",
                "final_gate",
                "gate_drop_vs_k2_aware",
                "contribution_norm",
            )
        ),
    )
    gate_columns = ["current_gate", "temporal_gate", "counterfactual_temporal_gate", "final_gate"]
    audit.check(
        "client_gates_in_unit_interval",
        all(pd.to_numeric(client_rows[field]).between(-1e-7, 1.0 + 1e-7).all() for field in gate_columns),
    )
    audit.check(
        "client_role_flags_consistent",
        bool(
            (~(client_rows["honest_regular_bool"] & client_rows["honest_outlier_bool"])).all()
            and (~(client_rows["honest_regular_bool"] & client_rows["latent_byzantine_bool"])).all()
            and (~(client_rows["honest_outlier_bool"] & client_rows["latent_byzantine_bool"])).all()
            and client_rows["active_byzantine_bool"].eq(
                client_rows["latent_byzantine_bool"] & client_rows["attack_active_bool"]
            ).all()
        ),
    )

    rebuilt = _rebuild_trajectories(round_rows)
    event_summary = _event_summaries(client_rows, config)
    audit.check(
        "five_latent_byzantine_clients_per_trajectory",
        bool(event_summary["latent_clients"].eq(int(config["cohort"]["num_byzantine"])).all()),
    )
    rebuilt = rebuilt.merge(event_summary, on="trajectory_id", how="left", validate="many_to_one")
    non_primary = rebuilt["candidate"].ne(PRIMARY)
    rebuilt.loc[non_primary, [
        "detection_rate_within_deadline",
        "mean_detection_end_round",
        "recovery_rate_within_deadline",
        "mean_recovery_end_round",
    ]] = np.nan

    comparison = trajectory_rows.merge(
        rebuilt,
        on=["trajectory_id", "candidate"],
        suffixes=("_reported", "_rebuilt"),
        validate="one_to_one",
    )
    summary_fields = [
        "enrollment_auc",
        "monitoring_auc",
        "attack_auc",
        "active_attack_auc",
        "recovery_auc",
        "post_enrollment_auc",
        "attack_byzantine_contribution_mass",
        "detection_rate_within_deadline",
        "mean_detection_end_round",
        "recovery_rate_within_deadline",
        "mean_recovery_end_round",
    ]
    trajectory_field_differences = {
        field: _max_abs(comparison[f"{field}_reported"], comparison[f"{field}_rebuilt"])
        for field in summary_fields
    }
    audit.check(
        "trajectory_summaries_rebuilt_from_round_and_client_rows",
        all(
            _allclose(comparison[f"{field}_reported"], comparison[f"{field}_rebuilt"])
            for field in summary_fields
        ),
        trajectory_field_differences,
    )

    trajectory_numeric_required = [
        "enrollment_auc",
        "monitoring_auc",
        "attack_auc",
        "recovery_auc",
        "post_enrollment_auc",
    ]
    audit.check(
        "trajectory_required_metrics_finite",
        all(np.isfinite(pd.to_numeric(rebuilt[field], errors="coerce")).all() for field in trajectory_numeric_required)
        and np.isfinite(pd.to_numeric(rebuilt.loc[rebuilt["schedule"].ne("no_compromise"), "active_attack_auc"], errors="coerce")).all()
        and np.isfinite(pd.to_numeric(rebuilt.loc[rebuilt["candidate"].ne(FCC), "attack_byzantine_contribution_mass"], errors="coerce")).all(),
    )

    separated = set(str(value) for value in config["threats"]["separated_for_primary_gate"])
    persistent_separated = rebuilt["schedule"].eq("persistent") & rebuilt["threat"].isin(separated)
    byzantine_high = (
        persistent_separated
        & rebuilt["noise_regime"].eq("heteroscedastic")
        & rebuilt["noise_permutation"].eq("byzantine_high")
    )
    intermittent = rebuilt["schedule"].eq("intermittent_2_on_1_off")
    persistent_alie = rebuilt["schedule"].eq("persistent") & rebuilt["threat"].eq("alie")
    no_compromise = rebuilt["schedule"].eq("no_compromise") & rebuilt["threat"].eq("none")

    main_values, main_evidence = _paired_seed_contrast(rebuilt, persistent_separated, AWARE, "attack_auc")
    high_values, high_evidence = _paired_seed_contrast(rebuilt, byzantine_high, AWARE, "attack_auc")
    counterfactual_values, counterfactual_evidence = _paired_seed_contrast(
        rebuilt, byzantine_high, COUNTERFACTUAL, "attack_auc"
    )
    audit.check(
        "paired_contrasts_have_all_five_seeds",
        len(main_values) == len(high_values) == len(counterfactual_values) == 5
        and main_evidence["incomplete_cells"] == 0
        and high_evidence["incomplete_cells"] == 0
        and counterfactual_evidence["incomplete_cells"] == 0,
        {
            "persistent_separated": main_evidence,
            "byzantine_high": high_evidence,
            "counterfactual": counterfactual_evidence,
        },
    )
    main_ci = _ci95(main_values)
    high_ci = _ci95(high_values)
    counterfactual_ci = _ci95(counterfactual_values)

    main_ratio = _ratio(rebuilt, persistent_separated, PRIMARY, AWARE, "attack_auc")
    high_ratio = _ratio(rebuilt, byzantine_high, PRIMARY, AWARE, "attack_auc")
    counterfactual_ratio = _ratio(
        rebuilt, byzantine_high, PRIMARY, COUNTERFACTUAL, "attack_auc"
    )
    k3_ratio = _ratio(rebuilt, persistent_separated, PRIMARY, K3, "attack_auc")
    intermittent_ratio = _ratio(rebuilt, intermittent, PRIMARY, AWARE, "attack_auc")
    intermittent_active_ratio = _ratio(
        rebuilt, intermittent, PRIMARY, AWARE, "active_attack_auc"
    )
    alie_ratio = _ratio(rebuilt, persistent_alie, PRIMARY, AWARE, "attack_auc")
    no_compromise_ratio = _ratio(
        rebuilt, no_compromise, PRIMARY, AWARE, "post_enrollment_auc"
    )
    mass_ratio = _ratio(
        rebuilt,
        persistent_separated,
        PRIMARY,
        AWARE,
        "attack_byzantine_contribution_mass",
    )

    representative = client_rows.loc[
        client_rows["threat"].eq(str(config["threats"]["names"][0]))
        & client_rows["schedule"].eq("persistent")
    ]
    regular = representative.loc[representative["honest_regular_bool"]]
    regular_trigger = pd.to_numeric(regular["temporal_gate"]).lt(1.0 - 1.0e-7)
    client_round_rate = float(regular_trigger.mean())
    trajectory_max = regular.groupby("trajectory_id")["temporal_statistic"].max()
    trajectory_rate = float(trajectory_max.gt(deployed_c0).mean())
    heteroscedastic_regular = regular.loc[regular["noise_regime"].eq("heteroscedastic")]
    tier_rates: dict[str, float] = {}
    tier_gate_means: dict[str, float] = {}
    for tier in (1.0, 1.5, 2.0):
        tier_rows = heteroscedastic_regular.loc[
            np.isclose(pd.to_numeric(heteroscedastic_regular["public_noise_tier"]), tier, atol=1e-7)
        ]
        if len(tier_rows):
            tier_rates[str(tier)] = float(
                pd.to_numeric(tier_rows["temporal_gate"]).lt(1.0 - 1.0e-7).mean()
            )
            tier_gate_means[str(tier)] = float(pd.to_numeric(tier_rows["temporal_gate"]).mean())
    false_triggers = {
        "client_round_rate": client_round_rate,
        "trajectory_rate": trajectory_rate,
        "tier_rates": tier_rates,
        "tier_gate_means": tier_gate_means,
        "tier_rate_gap": max(tier_rates.values()) - min(tier_rates.values()),
        "tier_gate_mean_gap": max(tier_gate_means.values()) - min(tier_gate_means.values()),
        "representative_client_rounds": int(len(regular)),
        "representative_trajectories": int(trajectory_max.size),
    }
    expected_regular = (
        len(config["randomness"]["development_seeds"])
        * len(noise_cells)
        * len(config["cohort"]["honest_outliers"]["geometries"])
        * len(config["honest_dynamics"]["names"])
        * (int(config["temporal"]["total_rounds"]) - int(config["temporal"]["first_temporal_gate_round"]) + 1)
        * (
            int(config["cohort"]["num_clients"])
            - int(config["cohort"]["num_byzantine"])
            - int(config["cohort"]["honest_outliers"]["count"])
        )
    )
    audit.check(
        "false_trigger_population_exact",
        len(regular) == expected_regular == 36000 and trajectory_max.size == 100,
        {"client_rounds": len(regular), "trajectories": trajectory_max.size},
    )
    outlier_rows = representative.loc[
        representative["honest_outlier_bool"] & ~representative["latent_byzantine_bool"]
    ]
    outlier_gate_drop = float(pd.to_numeric(outlier_rows["gate_drop_vs_k2_aware"]).mean())
    audit.check("honest_outlier_population_exact", len(outlier_rows) == 12000, {"rows": len(outlier_rows)})

    primary_persistent_ids = set(
        rebuilt.loc[persistent_separated & rebuilt["candidate"].eq(PRIMARY), "trajectory_id"]
    )
    decision_events = event_summary.loc[event_summary["trajectory_id"].isin(primary_persistent_ids)]
    detection_rate = float(decision_events["detection_rate_within_deadline"].mean())
    recovery_rate = float(decision_events["recovery_rate_within_deadline"].mean())
    detection_end_counts = {
        "15": int(decision_events["detection_end_15"].sum()),
        "16": int(decision_events["detection_end_16"].sum()),
        "17": int(decision_events["detection_end_17"].sum()),
    }
    recovered_at_30 = int(decision_events["recovery_end_30"].sum())
    detected_total = sum(detection_end_counts.values())
    detection_mean_end = (
        sum(int(round_number) * count for round_number, count in detection_end_counts.items())
        / detected_total
        if detected_total
        else float("nan")
    )
    audit.check(
        "detection_recovery_population_exact",
        len(decision_events) == 300
        and int(decision_events["latent_clients"].sum()) == 1500,
        {
            "trajectories": len(decision_events),
            "trajectory_clients": int(decision_events["latent_clients"].sum()),
            "detection_end_counts": detection_end_counts,
            "detection_mean_end_round": detection_mean_end,
            "recovery_end_30": recovered_at_30,
        },
    )

    certified = round_rows.loc[round_rows["candidate"].ne(FCC)]
    cap_violations = int((~certified["contribution_cap_respected_bool"]).sum())
    normalization_violations = int(certified["normalization_by_gate_sum_bool"].sum())
    finite_fraction = float(round_rows["all_finite_bool"].mean())
    identity = round_rows.loc[
        round_rows["candidate"].eq(PRIMARY) & round_rows["temporal_gate_all_one_bool"]
    ]
    identity_max = float(pd.to_numeric(identity["difference_to_k2_aware"]).max())
    audit.check("identity_population_nonempty", len(identity) >= len(expected) * 12, {"rows": len(identity)})

    replace_numeric = [
        "observed_replace_one_difference",
        "theoretical_replace_one_bound",
        "ratio_observed_to_bound",
    ]
    audit.check(
        "replace_one_required_metrics_finite",
        all(np.isfinite(pd.to_numeric(replace_rows[field], errors="coerce")).all() for field in replace_numeric),
    )
    configured_bound = 2.0 * float(config["references"]["total_client_influence_cap"]) / float(
        config["cohort"]["num_clients"]
    )
    recomputed_violation = pd.to_numeric(replace_rows["observed_replace_one_difference"]).gt(
        pd.to_numeric(replace_rows["theoretical_replace_one_bound"]) + 1.0e-6
    )
    audit.check(
        "replace_one_formula_and_fixed_past",
        bool(
            np.allclose(pd.to_numeric(replace_rows["theoretical_replace_one_bound"]), configured_bound, rtol=0, atol=1e-12)
            and np.allclose(
                pd.to_numeric(replace_rows["ratio_observed_to_bound"]),
                pd.to_numeric(replace_rows["observed_replace_one_difference"])
                / pd.to_numeric(replace_rows["theoretical_replace_one_bound"]),
                rtol=1e-10,
                atol=1e-11,
            )
            and replace_rows["violation_bool"].eq(recomputed_violation).all()
            and replace_rows["same_past_bool"].all()
            and replace_rows["same_covariances_bool"].all()
            and replace_rows["same_anchor_bool"].all()
            and replace_rows["round"].astype(int).eq(13).all()
        ),
        {"configured_bound": configured_bound},
    )
    replace_violations = int(recomputed_violation.sum())
    replace_max_ratio = float(pd.to_numeric(replace_rows["ratio_observed_to_bound"]).max())

    clean_trajectory_ids = {
        identifier
        for identifier, row in expected.items()
        if row["threat"] == "none" and row["schedule"] == "no_compromise"
    }
    clean_round = round_rows.loc[round_rows["trajectory_id"].isin(clean_trajectory_ids)]
    clean_client = client_rows.loc[client_rows["trajectory_id"].isin(clean_trajectory_ids)]
    audit.check(
        "no_compromise_population_and_attack_flag",
        len(clean_trajectory_ids) == 100
        and len(clean_round) == 100 * 36 * len(candidates)
        and len(clean_client) == 100 * 24 * int(config["cohort"]["num_clients"])
        and not bool(clean_round["attack_active_bool"].any())
        and not bool(clean_client["attack_active_bool"].any()),
    )
    clean_pair = clean_round.loc[
        clean_round["candidate"].isin((PRIMARY, COUNTERFACTUAL))
    ].pivot(index="pairing_id", columns="candidate", values=[
        "reference_error",
        "difference_to_k2_aware",
        "temporal_gate_mean",
        "current_gate_mean",
        "final_gate_mean",
        "byzantine_contribution_mass",
        "max_client_contribution_norm",
    ])
    clean_output_max = max(
        float(np.nanmax(np.abs(clean_pair[(field, PRIMARY)] - clean_pair[(field, COUNTERFACTUAL)])))
        for field in (
            "reference_error",
            "difference_to_k2_aware",
            "temporal_gate_mean",
            "current_gate_mean",
            "final_gate_mean",
            "byzantine_contribution_mass",
            "max_client_contribution_norm",
        )
    )
    clean_client_stat_max = _max_abs(
        clean_client["temporal_statistic"], clean_client["counterfactual_temporal_statistic"]
    )
    clean_client_gate_max = _max_abs(
        clean_client["temporal_gate"], clean_client["counterfactual_temporal_gate"]
    )
    audit.check(
        "no_compromise_primary_equals_counterfactual",
        clean_output_max <= 1e-11 and clean_client_stat_max <= 1e-11 and clean_client_gate_max <= 1e-11,
        {
            "round_output_max_abs_difference": clean_output_max,
            "client_statistic_max_abs_difference": clean_client_stat_max,
            "client_gate_max_abs_difference": clean_client_gate_max,
        },
    )
    cf_group_keys = [
        "seed",
        "noise_regime",
        "noise_permutation",
        "outlier_geometry",
        "honest_dynamics",
        "round",
        "client",
    ]
    cf_invariance = client_rows.groupby(cf_group_keys).agg(
        stat_min=("counterfactual_temporal_statistic", "min"),
        stat_max=("counterfactual_temporal_statistic", "max"),
        gate_min=("counterfactual_temporal_gate", "min"),
        gate_max=("counterfactual_temporal_gate", "max"),
        rows=("trajectory_id", "size"),
    )
    cf_stat_range = float((cf_invariance["stat_max"] - cf_invariance["stat_min"]).max())
    cf_gate_range = float((cf_invariance["gate_max"] - cf_invariance["gate_min"]).max())
    audit.check(
        "counterfactual_history_common_random_numbers",
        cf_stat_range <= 1e-11
        and cf_gate_range <= 1e-11
        and bool(cf_invariance["rows"].eq(9).all()),
        {"max_statistic_range": cf_stat_range, "max_gate_range": cf_gate_range, "rows_per_group": sorted(cf_invariance["rows"].unique().tolist())},
    )
    causal_reset = client_rows.loc[client_rows["round"].isin((13, 29))]
    audit.check(
        "causal_history_matches_counterfactual_at_rounds_13_and_29",
        _max_abs(causal_reset["temporal_statistic"], causal_reset["counterfactual_temporal_statistic"]) <= 1e-11
        and _max_abs(causal_reset["temporal_gate"], causal_reset["counterfactual_temporal_gate"]) <= 1e-11,
    )

    gates = config["gates"]
    main_gain = 1.0 - main_ratio
    high_gain = 1.0 - high_ratio
    counterfactual_gain = 1.0 - counterfactual_ratio
    mass_reduction = 1.0 - mass_ratio
    control_ratio = max(intermittent_ratio, alie_ratio)
    observed = {
        "device": str(manifest["device"]),
        "development_trajectories": int(len(expected)),
        "development_round_rows": int(len(round_rows)),
        "expected_round_rows": completeness["round_rows"]["expected"],
        "development_trajectory_rows": int(len(trajectory_rows)),
        "expected_trajectory_rows": completeness["trajectory_rows"]["expected"],
        "development_client_rows": int(len(client_rows)),
        "expected_client_rows": completeness["client_rows"]["expected"],
        "complete_fraction": float(completeness["complete_fraction"]),
        "identifier_completeness": completeness,
        "finite_metric_fraction": finite_fraction,
        "contribution_cap_violations": cap_violations,
        "gate_sum_normalization_violations": normalization_violations,
        "replace_one_trials": int(len(replace_rows)),
        "replace_one_violations": replace_violations,
        "replace_one_max_ratio_to_bound": replace_max_ratio,
        "k4_k2_identity_max_abs_error": identity_max,
        "false_triggers": false_triggers,
        "honest_outlier_gate_drop_vs_k2_aware": outlier_gate_drop,
        "no_compromise_post_enrollment_error_ratio_to_k2_aware": no_compromise_ratio,
        "persistent_separated_attack_auc_gain_vs_k2_aware": main_gain,
        "persistent_separated_attack_auc_difference_seed_ci95": main_ci,
        "byzantine_high_attack_auc_gain_vs_k2_aware": high_gain,
        "byzantine_high_attack_auc_difference_seed_ci95": high_ci,
        "persistent_separated_byzantine_mass_reduction_vs_k2_aware": mass_reduction,
        "persistent_separated_detection_rate_within_deadline": detection_rate,
        "persistent_separated_recovery_rate_within_deadline": recovery_rate,
        "persistent_separated_attack_auc_ratio_to_k3": k3_ratio,
        "byzantine_high_attack_auc_gain_vs_counterfactual_history": counterfactual_gain,
        "byzantine_high_attack_auc_difference_vs_counterfactual_seed_ci95": counterfactual_ci,
        "intermittent_or_alie_attack_auc_ratio_to_k2_aware": control_ratio,
        "intermittent_attack_auc_ratio_to_k2_aware": intermittent_ratio,
        "intermittent_active_attack_auc_ratio_to_k2_aware": intermittent_active_ratio,
        "persistent_alie_attack_auc_ratio_to_k2_aware": alie_ratio,
    }
    recomputed_checks = {
        "complete": bool(completeness["exact_identifier_completeness"])
        and completeness["complete_fraction"] >= float(gates["complete_fraction_min"]),
        "finite": finite_fraction >= float(gates["finite_metric_fraction_min"]),
        "production_device": str(manifest["device"]) == str(gates["production_device_required"]),
        "contribution_cap": cap_violations <= int(gates["contribution_cap_violation_max"]),
        "no_gate_sum_normalization": normalization_violations == 0,
        "replace_one": replace_violations <= int(gates["replace_one_violation_max"]),
        "k4_k2_identity_when_temporal_gate_one": identity_max
        <= float(gates["k4_k2_identity_when_temporal_gate_one_abs_error_max"]),
        "regular_client_round_false_trigger": client_round_rate
        <= float(gates["regular_client_round_false_trigger_rate_max"]),
        "regular_trajectory_false_trigger": trajectory_rate
        <= float(gates["regular_trajectory_false_trigger_rate_max"]),
        "pooled_regular_false_trigger_tier_gap": false_triggers["tier_rate_gap"]
        <= float(gates["pooled_regular_false_trigger_tier_gap_max"]),
        "pooled_regular_gate_mean_tier_gap": false_triggers["tier_gate_mean_gap"]
        <= float(gates["pooled_regular_gate_mean_tier_gap_max"]),
        "honest_outlier_gate_drop": outlier_gate_drop
        <= float(gates["honest_outlier_gate_drop_vs_k2_aware_max"]),
        "no_compromise_post_enrollment_noninferiority": no_compromise_ratio
        <= float(gates["no_compromise_post_enrollment_error_ratio_to_k2_aware_max"]),
        "persistent_separated_gain": main_gain
        >= float(gates["persistent_separated_attack_auc_gain_vs_k2_aware_min"]),
        "persistent_separated_ci": float(main_ci["high"])
        <= float(gates["persistent_separated_attack_auc_difference_ci95_high_max"]),
        "byzantine_high_gain": high_gain
        >= float(gates["byzantine_high_attack_auc_gain_vs_k2_aware_min"]),
        "byzantine_high_ci": float(high_ci["high"])
        <= float(gates["byzantine_high_attack_auc_difference_ci95_high_max"]),
        "byzantine_mass_reduction": mass_reduction
        >= float(gates["persistent_separated_byzantine_mass_reduction_vs_k2_aware_min"]),
        "detection": detection_rate
        >= float(gates["persistent_separated_detection_rate_within_deadline_min"]),
        "recovery": recovery_rate
        >= float(gates["persistent_separated_recovery_rate_within_deadline_min"]),
        "noninferiority_vs_k3": k3_ratio
        <= float(gates["persistent_separated_attack_auc_ratio_to_k3_max"]),
        "gain_vs_counterfactual_history": counterfactual_gain
        >= float(gates["byzantine_high_attack_auc_gain_vs_counterfactual_history_min"]),
        "ci_vs_counterfactual_history": float(counterfactual_ci["high"])
        <= float(gates["byzantine_high_attack_auc_difference_vs_counterfactual_ci95_high_max"]),
        "intermittent_or_alie_noninferiority": control_ratio
        <= float(gates["intermittent_or_alie_attack_auc_ratio_to_k2_aware_max"]),
    }
    recomputed_decision = {
        "decision": "promote_to_holdout" if all(recomputed_checks.values()) else "stop_after_development",
        "all_gates_pass": bool(all(recomputed_checks.values())),
        "checks": recomputed_checks,
        "observed": observed,
        "holdout_opened": False,
    }
    decision_differences = _compare_tree(recomputed_decision, decision)
    audit.check("decision_json_matches_independent_recalculation", not decision_differences, decision_differences[:100])
    audit.check(
        "gate_registry_shape",
        len(gates) == 23 and len(recomputed_checks) == 24 and "no_gate_sum_normalization" in recomputed_checks,
        {"yaml_thresholds": len(gates), "decision_checks": len(recomputed_checks), "extra_hard_invariant": "no_gate_sum_normalization"},
    )

    summary_group_keys = ["candidate", "noise_regime", "threat", "schedule"]
    rebuilt_summary = (
        rebuilt.groupby(summary_group_keys, sort=True)
        .agg(
            n_trajectories=("trajectory_id", "size"),
            monitoring_auc_mean=("monitoring_auc", "mean"),
            attack_auc_mean=("attack_auc", "mean"),
            attack_auc_std=("attack_auc", "std"),
            active_attack_auc_mean=("active_attack_auc", "mean"),
            recovery_auc_mean=("recovery_auc", "mean"),
            post_enrollment_auc_mean=("post_enrollment_auc", "mean"),
            attack_byzantine_contribution_mass_mean=("attack_byzantine_contribution_mass", "mean"),
        )
        .reset_index()
    )
    summary_compare = summary.merge(
        rebuilt_summary,
        on=summary_group_keys,
        suffixes=("_reported", "_rebuilt"),
        validate="one_to_one",
    )
    summary_metrics = [
        "n_trajectories",
        "monitoring_auc_mean",
        "attack_auc_mean",
        "attack_auc_std",
        "active_attack_auc_mean",
        "recovery_auc_mean",
        "post_enrollment_auc_mean",
        "attack_byzantine_contribution_mass_mean",
    ]
    audit.check(
        "summary_csv_rebuilt_from_raw_rows",
        len(summary_compare) == len(summary) == len(rebuilt_summary)
        and all(
            _allclose(summary_compare[f"{field}_reported"], summary_compare[f"{field}_rebuilt"])
            for field in summary_metrics
        ),
    )

    source_files = {
        "runner": ROOT / "scripts/run_gaussian_aware_reference_g0g_k4_tcg.py",
        "algorithm": ROOT / "algorithms/gaussian_aware_reference.py",
        "shared_k1_runner": ROOT / "scripts/run_gaussian_aware_reference_g0g_k1.py",
        "shared_k2_runner": ROOT / "scripts/run_gaussian_aware_reference_g0g_k2.py",
        "shared_k3_runner": ROOT / "scripts/run_gaussian_aware_reference_g0g_k3.py",
        "oracle_runner": ROOT / "scripts/run_gaussian_aware_reference_oracle.py",
        "robust_aggregators": ROOT / "robustness/aggregators.py",
    }
    observed_hashes = {key: _sha256(path) for key, path in source_files.items()}
    audit.check("config_hash_matches_manifest", _sha256(config_path) == manifest.get("config_sha256"))
    audit.check("source_hashes_match_manifest", observed_hashes == manifest.get("source_sha256"), {"observed": observed_hashes, "manifest": manifest.get("source_sha256")})
    frozen_path = ROOT / str(config["frozen_calibration"]["path"])
    frozen_hash = _sha256(frozen_path)
    audit.check(
        "frozen_k2_calibration_hash",
        frozen_hash == str(config["frozen_calibration"]["sha256"])
        and k2_provenance.get("sha256_verified") is True
        and k2_provenance.get("observed_sha256") == frozen_hash,
        {"sha256": frozen_hash},
    )
    development_seeds = set(int(value) for value in config["randomness"]["development_seeds"])
    holdout_seeds = set(int(value) for value in config["randomness"]["holdout_seeds"])
    raw_seed_sets = {
        "round": set(round_rows["seed"].astype(int)),
        "client": set(client_rows["seed"].astype(int)),
        "trajectory": set(trajectory_rows["seed"].astype(int)),
        "replace_one": set(replace_rows["seed"].astype(int)),
    }
    audit.check(
        "holdout_remains_closed_and_absent",
        all(values == development_seeds for values in raw_seed_sets.values())
        and not any(values & holdout_seeds for values in raw_seed_sets.values())
        and manifest.get("holdout_opened") is False
        and decision.get("holdout_opened") is False,
        {"development_seeds": sorted(development_seeds), "holdout_seeds": sorted(holdout_seeds)},
    )
    audit.check(
        "manifest_completed_on_mps",
        manifest.get("status") == "completed_development"
        and manifest.get("device") == "mps"
        and manifest.get("dtype") == "torch.float32"
        and manifest.get("mps_required_without_fallback") is True,
    )

    failed_audit_checks = [name for name, passed in audit.checks.items() if not passed]
    failed_scientific_gates = [name for name, passed in recomputed_checks.items() if not passed]
    payload = {
        "audit_kind": "independent_raw_csv_recalculation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.resolve()),
        "results_dir": str(results_dir.resolve()),
        "audit_verdict": "PASS" if not failed_audit_checks else "FAIL",
        "scientific_decision": recomputed_decision["decision"],
        "all_scientific_checks_pass": recomputed_decision["all_gates_pass"],
        "failed_audit_checks": failed_audit_checks,
        "failed_scientific_checks": failed_scientific_gates,
        "audit_checks": audit.checks,
        "audit_details": audit.details,
        "frozen_gate_thresholds": dict(gates),
        "expected_counts": expected_counts,
        "temporal_calibration": {
            "deployed_c0": deployed_c0,
            "deployed_c1": deployed_c1,
            **calibration_evidence,
        },
        "paired_seed_evidence": {
            "persistent_separated_vs_k2_aware": {**main_evidence, "ci95": main_ci},
            "byzantine_high_vs_k2_aware": {**high_evidence, "ci95": high_ci},
            "byzantine_high_vs_counterfactual": {**counterfactual_evidence, "ci95": counterfactual_ci},
        },
        "recomputed_decision": recomputed_decision,
        "reported_decision_difference_count": len(decision_differences),
        "reported_decision_differences": decision_differences[:100],
        "raw_file_sha256": {key: _sha256(path) for key, path in required.items()},
    }
    safe_payload = _json_safe(payload)
    output_path.write_text(
        json.dumps(safe_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return safe_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k4_tcg.yaml",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results/ldp_gradient_far/gaussian_aware_reference_g0g_k4_tcg_mps_v1",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    output = args.output.resolve() if args.output else results_dir / "independent_audit.json"
    result = audit_results(args.config.resolve(), results_dir, output)
    observed = result["recomputed_decision"]["observed"]
    print(f"Audit verdict: {result['audit_verdict']}")
    print(f"Scientific decision: {result['scientific_decision']}")
    print(f"Failed scientific checks: {result['failed_scientific_checks'] or 'none'}")
    print(
        "Main gain / CI95 high / detection / recovery: "
        f"{observed['persistent_separated_attack_auc_gain_vs_k2_aware']:.6f} / "
        f"{observed['persistent_separated_attack_auc_difference_seed_ci95']['high']:.6f} / "
        f"{observed['persistent_separated_detection_rate_within_deadline']:.6f} / "
        f"{observed['persistent_separated_recovery_rate_within_deadline']:.6f}"
    )
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
