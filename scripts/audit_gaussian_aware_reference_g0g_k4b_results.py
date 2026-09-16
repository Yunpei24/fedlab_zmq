#!/usr/bin/env python3
"""Independently audit completed G0g-K4b raw development artifacts.

The auditor deliberately does not import the K4b experiment runner.  It first
reads *only* ``manifest.json`` and refuses to inspect any CSV while the run is
not ``completed_development``.  Once complete, it reconstructs the frozen
matrix, trajectory summaries, seed-level contrasts, log-ratio confidence
intervals, component masses, certificates, and scientific decision from raw
CSV files and compares them with ``decision.json``.
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
DEFAULT_CONFIG = ROOT / (
    "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k4b_past_imputation.yaml"
)
DEFAULT_RESULTS = ROOT / (
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k4b_past_imputation_mps_v1"
)

CAMPAIGN_ID = "gaussian_aware_reference_g0g_k4b_past_imputation_mps_v1"
PUBLISHED_LOCK_SHA256 = (
    "a7f349492336ba9d198ece99b4cd5c4ed4ba080f74aff984c4e9d0b1c4c8b587"
)
FCC = "fcc"
K2 = "g0g_k2"
K3 = "g0g_k3_dual_gate"
K4 = "g0g_k4_temporal_causal_gate"
PRIMARY = "g0g_k4b_full_temporal_missing_slot_imputation"
DELTA = "g0g_k4b_incremental_temporal_suppression_imputation"
STATIC = "g0g_k4b_static_clean_window_full_imputation"
ORACLE = "g0g_k4b_pointwise_optimal_oracle"
CANDIDATES = (FCC, K2, K3, K4, PRIMARY, DELTA, STATIC, ORACLE)
CERTIFIED = (K2, K3, K4, PRIMARY, DELTA, STATIC, ORACLE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _guard_completed(results_dir: Path) -> dict[str, Any]:
    """Read only the manifest and stop before any raw-artifact access."""

    manifest_path = results_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing K4b manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    status = manifest.get("status")
    if status != "completed_development":
        raise RuntimeError(
            "K4b results are not complete; raw artifacts were not opened: "
            f"status={status!r}"
        )
    if manifest.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError(f"Unexpected K4b campaign: {manifest.get('campaign_id')!r}")
    return manifest


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


def _allclose(left: pd.Series, right: pd.Series, *, atol: float = 2.0e-9) -> bool:
    return bool(
        np.allclose(
            pd.to_numeric(left, errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(right, errors="coerce").to_numpy(dtype=float),
            rtol=2.0e-8,
            atol=atol,
            equal_nan=True,
        )
    )


def _float32_reduction_close(
    left: pd.Series,
    right: pd.Series,
    *,
    max_terms: int,
    quotient: bool = False,
) -> bool:
    """Compare CSV reconstructions with a recorded float32 reduction.

    Client contribution norms are exported individually and pandas rebuilds
    their sums in float64.  The runner records the same totals after a
    ``torch.float32`` reduction on MPS.  Requiring float64-level agreement
    therefore produces false failures.  For non-negative component masses,
    the standard forward-error bound for a sum of at most ``max_terms`` terms
    is ``gamma_n * sum(x_i)`` with

        gamma_n = n u / (1 - n u),  u = eps(float32) / 2.

    A mass share additionally contains a numerator reduction, a denominator
    reduction and a division, hence the factor three.  The tolerance remains
    around a few parts per million for the frozen n=25 screen; it is local to
    this float32-reduction audit and does not relax any scientific gate.
    """

    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    if left_values.shape != right_values.shape:
        return False
    if not bool(np.isfinite(left_values).all() and np.isfinite(right_values).all()):
        return False
    unit_roundoff = float(np.finfo(np.float32).eps) / 2.0
    terms = max(1, int(max_terms))
    gamma_n = terms * unit_roundoff / (1.0 - terms * unit_roundoff)
    operation_factor = 3.0 if quotient else 1.0
    scale = np.maximum(np.abs(left_values), np.abs(right_values))
    tolerance = 2.0e-9 + operation_factor * gamma_n * scale
    return bool(np.less_equal(np.abs(left_values - right_values), tolerance).all())


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _compare_tree(
    recomputed: Any,
    reported: Any,
    *,
    path: str = "",
    differences: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if differences is None:
        differences = []
    if isinstance(recomputed, Mapping) and isinstance(reported, Mapping):
        for key in sorted(set(recomputed) | set(reported)):
            child = f"{path}.{key}" if path else str(key)
            if key not in recomputed or key not in reported:
                differences.append(
                    {
                        "path": child,
                        "recomputed": recomputed.get(key),
                        "reported": reported.get(key),
                    }
                )
            else:
                _compare_tree(
                    recomputed[key],
                    reported[key],
                    path=child,
                    differences=differences,
                )
        return differences
    if isinstance(recomputed, bool) or isinstance(reported, bool):
        same = type(recomputed) is type(reported) and recomputed == reported
    elif isinstance(recomputed, (int, float)) and isinstance(reported, (int, float)):
        left, right = float(recomputed), float(reported)
        same = (math.isnan(left) and math.isnan(right)) or math.isclose(
            left, right, rel_tol=2.0e-8, abs_tol=2.0e-9
        )
    else:
        same = recomputed == reported
    if not same:
        differences.append(
            {"path": path, "recomputed": recomputed, "reported": reported}
        )
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


def _noise_cells(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        (str(regime["name"]), str(permutation))
        for regime in config["privacy_noise"]["regimes"]
        for permutation in regime["permutations"]
    ]


def _expected_matrix(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    attack_cells = list(
        product(config["threats"]["names"], config["threats"]["schedules"])
    )
    clean = (
        str(config["no_compromise_control"]["threat_name"]),
        str(config["no_compromise_control"]["schedule_name"]),
    )
    for seed, (regime, permutation), geometry, dynamics in product(
        [int(value) for value in config["randomness"]["development_seeds"]],
        _noise_cells(config),
        [str(value) for value in config["cohort"]["honest_outliers"]["geometries"]],
        [str(value) for value in config["honest_dynamics"]["names"]],
    ):
        for threat, schedule in [*attack_cells, clean]:
            values = {
                "seed": seed,
                "noise_regime": regime,
                "noise_permutation": permutation,
                "outlier_geometry": geometry,
                "honest_dynamics": dynamics,
                "threat": str(threat),
                "schedule": str(schedule),
            }
            identifier = "|".join(str(values[key]) for key in values)
            if identifier in expected:
                raise RuntimeError(f"Duplicate expected trajectory: {identifier}")
            expected[identifier] = values
    return expected


def _axis_matches(
    frame: pd.DataFrame,
    expected: Mapping[str, Mapping[str, Any]],
    fields: Sequence[str],
) -> bool:
    identifiers = frame["trajectory_id"].astype(str)
    if not bool(identifiers.isin(expected).all()):
        return False
    for field in fields:
        mapped = identifiers.map(lambda value: expected[value][field])
        if field == "seed":
            if not bool(pd.to_numeric(frame[field], errors="coerce").eq(mapped).all()):
                return False
        elif not bool(frame[field].astype(str).eq(mapped.astype(str)).all()):
            return False
    return True


def _unique_coverage(
    frame: pd.DataFrame,
    *,
    keys: list[str],
    valid: pd.Series,
    expected_count: int,
) -> tuple[float, int]:
    valid_rows = frame.loc[valid, keys]
    unique = int(len(valid_rows.drop_duplicates()))
    duplicates = int(len(valid_rows) - unique)
    return min(unique / float(expected_count), 1.0), duplicates


def _completeness(
    rounds: pd.DataFrame,
    clients: pd.DataFrame,
    trajectories: pd.DataFrame,
    replacements: pd.DataFrame,
    config: Mapping[str, Any],
    expected: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_ids = set(expected)
    total_rounds = int(config["temporal"]["total_rounds"])
    first_gate = int(config["temporal"]["first_temporal_gate_round"])
    n = int(config["cohort"]["num_clients"])
    expected_rounds = len(expected_ids) * len(CANDIDATES) * total_rounds
    expected_trajectories = len(expected_ids) * len(CANDIDATES)
    expected_clients = len(expected_ids) * (total_rounds - first_gate + 1) * n

    round_valid = (
        rounds["trajectory_id"].astype(str).isin(expected_ids)
        & rounds["candidate"].astype(str).isin(CANDIDATES)
        & pd.to_numeric(rounds["round"], errors="coerce").between(1, total_rounds)
    )
    trajectory_valid = trajectories["trajectory_id"].astype(str).isin(
        expected_ids
    ) & trajectories["candidate"].astype(str).isin(CANDIDATES)
    client_valid = (
        clients["trajectory_id"].astype(str).isin(expected_ids)
        & pd.to_numeric(clients["round"], errors="coerce").between(
            first_gate, total_rounds
        )
        & pd.to_numeric(clients["client"], errors="coerce").between(0, n - 1)
    )
    round_fraction, round_duplicates = _unique_coverage(
        rounds,
        keys=["trajectory_id", "candidate", "round"],
        valid=round_valid,
        expected_count=expected_rounds,
    )
    trajectory_fraction, trajectory_duplicates = _unique_coverage(
        trajectories,
        keys=["trajectory_id", "candidate"],
        valid=trajectory_valid,
        expected_count=expected_trajectories,
    )
    client_fraction, client_duplicates = _unique_coverage(
        clients,
        keys=["trajectory_id", "round", "client"],
        valid=client_valid,
        expected_count=expected_clients,
    )

    trials = int(config["randomness"]["replace_one_trials_per_seed_noise_cell"])
    expected_replace = {
        (
            "clean_first_causal_gate",
            candidate,
            int(seed),
            regime,
            permutation,
            trial,
        )
        for seed, (regime, permutation), candidate, trial in product(
            config["randomness"]["development_seeds"],
            _noise_cells(config),
            (PRIMARY, DELTA),
            range(trials),
        )
    }
    expected_replace |= {
        (
            "persistent_bitflip_round17_h_below_one",
            candidate,
            int(seed),
            "heteroscedastic",
            "byzantine_high",
            trial,
        )
        for seed, candidate, trial in product(
            config["randomness"]["development_seeds"],
            (PRIMARY, DELTA),
            range(trials),
        )
    }
    observed_replace = [
        (
            str(row.audit_scenario),
            str(row.candidate),
            int(row.seed),
            str(row.noise_regime),
            str(row.noise_permutation),
            int(row.trial),
        )
        for row in replacements.itertuples(index=False)
    ]
    observed_replace_set = set(observed_replace)
    replace_fraction = len(observed_replace_set & expected_replace) / float(
        len(expected_replace)
    )
    replace_duplicates = len(observed_replace) - len(observed_replace_set)
    fractions = [round_fraction, trajectory_fraction, client_fraction, replace_fraction]
    duplicates = {
        "round": round_duplicates,
        "trajectory": trajectory_duplicates,
        "client": client_duplicates,
        "replace": replace_duplicates,
    }
    exact = bool(
        min(fractions) == 1.0
        and not any(duplicates.values())
        and bool(round_valid.all())
        and bool(trajectory_valid.all())
        and bool(client_valid.all())
        and observed_replace_set == expected_replace
    )
    return {
        "exact_identifier_completeness": exact,
        "complete_fraction": min(fractions),
        "expected_trajectory_ids": len(expected_ids),
        "round_rows_expected": expected_rounds,
        "trajectory_rows_expected": expected_trajectories,
        "client_rows_expected": expected_clients,
        "replace_rows_expected": len(expected_replace),
        "duplicate_counts": duplicates,
    }


def _phase(rounds: pd.Series, config: Mapping[str, Any]) -> pd.Series:
    temporal = config["temporal"]
    values = np.select(
        [
            rounds.le(int(temporal["enrollment_rounds"])),
            rounds.lt(int(temporal["attack_start_round"])),
            rounds.le(int(temporal["attack_end_round"])),
        ],
        ["enrollment", "monitoring", "attack"],
        default="recovery",
    )
    return pd.Series(values, index=rounds.index)


def _attack_active(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.Series:
    rounds = pd.to_numeric(frame["round"], errors="coerce").astype(int)
    start = int(config["temporal"]["attack_start_round"])
    end = int(config["temporal"]["attack_end_round"])
    window = rounds.between(start, end)
    persistent = frame["schedule"].astype(str).eq("persistent") & window
    intermittent = frame["schedule"].astype(str).eq("intermittent_2_on_1_off") & window
    pattern = [bool(value) for value in config["threats"]["intermittent_pattern"]]
    pattern_active = rounds.map(
        lambda value: pattern[(int(value) - start) % len(pattern)]
    )
    return persistent | (intermittent & pattern_active)


def _event_rates(clients: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    latent = clients.loc[clients["latent_byzantine_bool"]].copy()
    wide = latent.pivot(
        index=["trajectory_id", "client"], columns="round", values="temporal_gate"
    )
    detection = pd.Series(False, index=wide.index)
    for end in range(
        int(config["temporal"]["first_eligible_detection_round"]),
        int(config["temporal"]["detection_deadline_round"]) + 1,
    ):
        detection |= wide[end - 1].le(
            float(config["temporal"]["detection_gate_threshold"])
        ) & wide[end].le(float(config["temporal"]["detection_gate_threshold"]))
    recovery = pd.Series(False, index=wide.index)
    for end in range(
        int(config["temporal"]["first_fully_clean_window_round"]) + 1,
        int(config["temporal"]["recovery_deadline_round"]) + 1,
    ):
        recovery |= wide[end - 1].ge(
            float(config["temporal"]["recovery_gate_threshold"])
        ) & wide[end].ge(float(config["temporal"]["recovery_gate_threshold"]))
    return (
        pd.DataFrame({"detection": detection, "recovery": recovery})
        .groupby(level="trajectory_id")
        .mean()
        .rename(
            columns={
                "detection": "detection_rate_within_deadline",
                "recovery": "recovery_rate_within_deadline",
            }
        )
        .reset_index()
    )


def _rebuild_trajectories(
    rounds: pd.DataFrame, clients: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
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
    result = rounds.groupby(keys, sort=False)[metadata].first()

    def add_mean(name: str, mask: pd.Series, source: str) -> None:
        result[name] = rounds.loc[mask].groupby(keys, sort=False)[source].mean()

    add_mean("enrollment_auc", rounds["phase"].eq("enrollment"), "reference_error")
    add_mean("monitoring_auc", rounds["phase"].eq("monitoring"), "reference_error")
    add_mean("attack_auc", rounds["phase"].eq("attack"), "reference_error")
    add_mean(
        "active_attack_auc",
        rounds["phase"].eq("attack") & rounds["attack_active_bool"],
        "reference_error",
    )
    add_mean("recovery_auc", rounds["phase"].eq("recovery"), "reference_error")
    add_mean(
        "post_enrollment_auc",
        rounds["phase"].isin(("attack", "recovery")),
        "reference_error",
    )
    attack = rounds["phase"].eq("attack")
    add_mean(
        "attack_byzantine_direct_current_mass_share",
        attack,
        "byzantine_direct_current_mass_share",
    )
    add_mean(
        "attack_byzantine_imputed_mass_share",
        attack,
        "byzantine_imputed_mass_share",
    )
    add_mean(
        "attack_byzantine_total_slot_mass_share_descriptive",
        attack,
        "byzantine_total_slot_mass_share",
    )
    add_mean(
        "attack_predictor_error_vs_current_honest_clipped_direction",
        attack,
        "predictor_error_vs_current_honest_clipped_direction",
    )
    result = result.reset_index()
    result["deployable"] = False
    result["privacy_claimed"] = result["candidate"].ne(ORACLE)
    events = _event_rates(clients, config)
    result = result.merge(
        events, on="trajectory_id", how="left", validate="many_to_one"
    )
    non_primary = result["candidate"].ne(PRIMARY)
    result.loc[
        non_primary,
        ["detection_rate_within_deadline", "recovery_rate_within_deadline"],
    ] = np.nan
    return result


def _component_mass_audit(
    rounds: pd.DataFrame, clients: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    keys = ["trajectory_id", "round"]
    details: dict[str, Any] = {}
    all_ok = True
    cap = float(config["references"]["total_client_influence_cap"])
    for candidate, prefix in ((PRIMARY, "primary"), (DELTA, "delta")):
        values = clients.copy()
        maximum_terms = int(values.groupby(keys, sort=False).size().max())
        for component in ("direct_current", "imputed", "total_slot"):
            column = f"{prefix}_{component}_contribution_norm"
            values[column] = pd.to_numeric(values[column], errors="coerce")
            values[f"{component}_byzantine"] = values[column].where(
                values["latent_byzantine_bool"], 0.0
            )
        grouped = values.groupby(keys, sort=False).agg(
            direct_total=(f"{prefix}_direct_current_contribution_norm", "sum"),
            direct_byzantine=("direct_current_byzantine", "sum"),
            imputed_total=(f"{prefix}_imputed_contribution_norm", "sum"),
            imputed_byzantine=("imputed_byzantine", "sum"),
            total_total=(f"{prefix}_total_slot_contribution_norm", "sum"),
            total_byzantine=("total_slot_byzantine", "sum"),
            max_total=(f"{prefix}_total_slot_contribution_norm", "max"),
        )
        for stem in ("direct", "imputed", "total"):
            grouped[f"{stem}_share"] = np.where(
                grouped[f"{stem}_total"] > 0.0,
                grouped[f"{stem}_byzantine"] / grouped[f"{stem}_total"],
                0.0,
            )
        reported = rounds.loc[
            rounds["candidate"].eq(candidate)
            & rounds["round"].ge(int(config["temporal"]["first_temporal_gate_round"]))
        ].set_index(keys)
        grouped = grouped.sort_index()
        reported = reported.sort_index()
        aligned = grouped.index.equals(reported.index)
        checks = {
            "exact_round_alignment": aligned,
            "direct_total": _float32_reduction_close(
                grouped["direct_total"],
                reported["direct_current_mass_total"],
                max_terms=maximum_terms,
            ),
            "imputed_total": _float32_reduction_close(
                grouped["imputed_total"],
                reported["imputed_mass_total"],
                max_terms=maximum_terms,
            ),
            "total_slot_total": _float32_reduction_close(
                grouped["total_total"],
                reported["total_slot_mass_total"],
                max_terms=maximum_terms,
            ),
            "direct_share": _float32_reduction_close(
                grouped["direct_share"],
                reported["byzantine_direct_current_mass_share"],
                max_terms=maximum_terms,
                quotient=True,
            ),
            "imputed_share": _float32_reduction_close(
                grouped["imputed_share"],
                reported["byzantine_imputed_mass_share"],
                max_terms=maximum_terms,
                quotient=True,
            ),
            "total_slot_share": _float32_reduction_close(
                grouped["total_share"],
                reported["byzantine_total_slot_mass_share"],
                max_terms=maximum_terms,
                quotient=True,
            ),
            "maximum_slot": _allclose(
                grouped["max_total"], reported["max_client_contribution_norm"]
            ),
            "cap": bool(grouped["max_total"].le(cap + 1.0e-6).all()),
        }
        details[candidate] = checks
        all_ok = all_ok and all(checks.values())
    return all_ok, details


def _ratio(
    rows: pd.DataFrame,
    mask: pd.Series,
    *,
    candidate: str,
    baseline: str,
    metric: str,
) -> float:
    numerator = _finite_mean(rows.loc[mask & rows["candidate"].eq(candidate), metric])
    denominator = _finite_mean(rows.loc[mask & rows["candidate"].eq(baseline), metric])
    return numerator / max(denominator, 1.0e-12)


def _paired_seed_differences(
    rows: pd.DataFrame,
    mask: pd.Series,
    *,
    candidate: str,
    baseline: str,
    metric: str,
) -> tuple[list[float], dict[str, Any]]:
    selected = rows.loc[mask & rows["candidate"].isin((candidate, baseline))]
    cell = selected.pivot(
        index=["trajectory_id", "seed"], columns="candidate", values=metric
    )
    complete = cell[[candidate, baseline]].notna().all(axis=1)
    deltas = cell.loc[complete, candidate] - cell.loc[complete, baseline]
    per_seed = deltas.groupby(level="seed").mean().sort_index()
    return [float(value) for value in per_seed], {
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "paired_cells": int(complete.sum()),
        "incomplete_cells": int((~complete).sum()),
        "differences_by_seed": {
            str(int(seed)): float(value) for seed, value in per_seed.items()
        },
    }


def _paired_seed_log_ratios(
    rows: pd.DataFrame,
    mask: pd.Series,
    *,
    candidate: str,
    baseline: str,
    metric: str,
) -> dict[str, Any]:
    selected = rows.loc[mask & rows["candidate"].isin((candidate, baseline))]
    cell = selected.pivot(
        index=["trajectory_id", "seed"], columns="candidate", values=metric
    )
    complete = cell[[candidate, baseline]].notna().all(axis=1)
    if not bool((cell.loc[complete, [candidate, baseline]] > 0.0).all().all()):
        raise ValueError(f"Non-positive value in {metric} log-ratio")
    seed_means = (
        selected.groupby(["seed", "candidate"], sort=True)[metric].mean().unstack()
    )
    seed_means = seed_means.dropna(subset=[candidate, baseline])
    ratios = seed_means[candidate] / seed_means[baseline]
    logs = np.log(ratios)
    ci = _ci95([float(value) for value in logs])
    pooled = _ratio(
        rows,
        mask,
        candidate=candidate,
        baseline=baseline,
        metric=metric,
    )
    return {
        "n_seed_pairs": int(len(logs)),
        "seed_ids": [int(seed) for seed in logs.index],
        "ratios_by_seed": {
            str(int(seed)): float(value) for seed, value in ratios.items()
        },
        "log_ratios_by_seed": {
            str(int(seed)): float(value) for seed, value in logs.items()
        },
        "log_ratio_ci95": ci,
        "exp_log_ratio_ci95_high": math.exp(float(ci["high"])),
        "geometric_mean_ratio": math.exp(_finite_mean(logs)),
        "pooled_ratio_descriptive": pooled,
        "paired_cells": int(complete.sum()),
        "incomplete_cells": int((~complete).sum()),
    }


def _false_triggers(clients: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    representative = clients.loc[
        clients["threat"].eq(str(config["threats"]["names"][0]))
        & clients["schedule"].eq("persistent")
    ]
    selected = representative.loc[representative["honest_regular_bool"]]
    triggered = selected["temporal_gate"].lt(1.0 - 1.0e-7)
    trajectory_max = selected.groupby("trajectory_id")["temporal_statistic"].max()
    c0 = float(config["frozen_calibrations"]["k4_temporal"]["deployed_c0"])
    hetero = selected.loc[selected["noise_regime"].eq("heteroscedastic")]
    tiers = sorted(
        {
            float(value)
            for regime in config["privacy_noise"]["regimes"]
            if regime["name"] == "heteroscedastic"
            for value in regime["client_std_multipliers"]
        }
    )
    tier_rates: dict[str, float] = {}
    tier_gate_means: dict[str, float] = {}
    for tier in tiers:
        rows = hetero.loc[np.isclose(hetero["public_noise_tier"], tier)]
        tier_rates[str(tier)] = float(rows["temporal_gate"].lt(1.0 - 1.0e-7).mean())
        tier_gate_means[str(tier)] = float(rows["temporal_gate"].mean())
    return {
        "client_round_rate": float(triggered.mean()),
        "trajectory_rate": float(trajectory_max.gt(c0).mean()),
        "tier_rates": tier_rates,
        "tier_gate_means": tier_gate_means,
        "tier_rate_gap": max(tier_rates.values()) - min(tier_rates.values()),
        "tier_gate_mean_gap": max(tier_gate_means.values())
        - min(tier_gate_means.values()),
        "representative_client_rounds": len(selected),
        "representative_trajectories": len(trajectory_max),
    }


def _recompute_decision(
    rounds: pd.DataFrame,
    clients: pd.DataFrame,
    trajectories: pd.DataFrame,
    replacements: pd.DataFrame,
    config: Mapping[str, Any],
    completeness: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    gates = config["gates"]
    separated = set(
        str(value) for value in config["threats"]["separated_for_primary_gate"]
    )
    headroom = set(str(value) for value in config["threats"]["oracle_headroom_threats"])
    persistent_headroom = trajectories["schedule"].eq("persistent") & trajectories[
        "threat"
    ].isin(headroom)
    persistent_separated = trajectories["schedule"].eq("persistent") & trajectories[
        "threat"
    ].isin(separated)
    persistent_ipm = trajectories["schedule"].eq("persistent") & trajectories[
        "threat"
    ].eq("ipm")
    persistent_alie = trajectories["schedule"].eq("persistent") & trajectories[
        "threat"
    ].eq("alie")
    intermittent = trajectories["schedule"].eq("intermittent_2_on_1_off")
    clean = trajectories["schedule"].eq("no_compromise") & trajectories["threat"].eq(
        "none"
    )

    oracle_gain = 1.0 - _ratio(
        trajectories,
        persistent_headroom,
        candidate=ORACLE,
        baseline=K4,
        metric="attack_auc",
    )
    oracle_diffs, oracle_evidence = _paired_seed_differences(
        trajectories,
        persistent_headroom,
        candidate=ORACLE,
        baseline=K4,
        metric="attack_auc",
    )
    oracle_ci = _ci95(oracle_diffs)
    primary_gain = 1.0 - _ratio(
        trajectories,
        persistent_headroom,
        candidate=PRIMARY,
        baseline=K4,
        metric="attack_auc",
    )
    primary_diffs, primary_evidence = _paired_seed_differences(
        trajectories,
        persistent_headroom,
        candidate=PRIMARY,
        baseline=K4,
        metric="attack_auc",
    )
    primary_ci = _ci95(primary_diffs)
    k2_gain = 1.0 - _ratio(
        trajectories,
        persistent_separated,
        candidate=PRIMARY,
        baseline=K2,
        metric="attack_auc",
    )
    k2_diffs, k2_evidence = _paired_seed_differences(
        trajectories,
        persistent_separated,
        candidate=PRIMARY,
        baseline=K2,
        metric="attack_auc",
    )
    k2_ci = _ci95(k2_diffs)
    mean_k4 = _finite_mean(
        trajectories.loc[
            persistent_headroom & trajectories["candidate"].eq(K4), "attack_auc"
        ]
    )
    mean_primary = _finite_mean(
        trajectories.loc[
            persistent_headroom & trajectories["candidate"].eq(PRIMARY), "attack_auc"
        ]
    )
    mean_delta = _finite_mean(
        trajectories.loc[
            persistent_headroom & trajectories["candidate"].eq(DELTA), "attack_auc"
        ]
    )
    mean_oracle = _finite_mean(
        trajectories.loc[
            persistent_headroom & trajectories["candidate"].eq(ORACLE), "attack_auc"
        ]
    )
    oracle_headroom = mean_k4 - mean_oracle
    primary_captured = mean_k4 - mean_primary
    delta_captured = mean_k4 - mean_delta
    capture_fraction = (
        primary_captured / oracle_headroom if oracle_headroom > 0.0 else float("-inf")
    )
    delta_capture = (
        delta_captured / oracle_headroom if oracle_headroom > 0.0 else float("-inf")
    )
    ipm_log = _paired_seed_log_ratios(
        trajectories,
        persistent_ipm,
        candidate=PRIMARY,
        baseline=K4,
        metric="attack_auc",
    )
    alie_log = _paired_seed_log_ratios(
        trajectories,
        persistent_alie,
        candidate=PRIMARY,
        baseline=K4,
        metric="attack_auc",
    )
    intermittent_log = _paired_seed_log_ratios(
        trajectories,
        intermittent,
        candidate=PRIMARY,
        baseline=K4,
        metric="attack_auc",
    )
    clean_log = _paired_seed_log_ratios(
        trajectories,
        clean,
        candidate=PRIMARY,
        baseline=K2,
        metric="post_enrollment_auc",
    )
    mass_reduction = 1.0 - _ratio(
        trajectories,
        persistent_separated,
        candidate=PRIMARY,
        baseline=K2,
        metric="attack_byzantine_direct_current_mass_share",
    )
    delta_gain_k4 = 1.0 - _ratio(
        trajectories,
        persistent_headroom,
        candidate=DELTA,
        baseline=K4,
        metric="attack_auc",
    )
    delta_gain_k2 = 1.0 - _ratio(
        trajectories,
        persistent_separated,
        candidate=DELTA,
        baseline=K2,
        metric="attack_auc",
    )
    delta_mass_reduction = 1.0 - _ratio(
        trajectories,
        persistent_separated,
        candidate=DELTA,
        baseline=K2,
        metric="attack_byzantine_direct_current_mass_share",
    )

    false_triggers = _false_triggers(clients, config)
    representative = clients.loc[
        clients["threat"].eq(str(config["threats"]["names"][0]))
        & clients["schedule"].eq("persistent")
    ]
    outlier_drop = _finite_mean(
        representative.loc[
            representative["honest_outlier_bool"]
            & ~representative["latent_byzantine_bool"],
            "gate_drop_vs_k2_aware",
        ]
    )
    primary_persistent = trajectories.loc[
        persistent_separated & trajectories["candidate"].eq(PRIMARY)
    ]
    detection = _finite_mean(primary_persistent["detection_rate_within_deadline"])
    recovery = _finite_mean(primary_persistent["recovery_rate_within_deadline"])

    certified = rounds.loc[rounds["candidate"].isin(CERTIFIED)]
    cap_violations = int((~certified["contribution_cap_respected_bool"]).sum())
    normalization_violations = int(certified["normalization_by_gate_sum_bool"].sum())
    causal_violations = int(
        (
            ~rounds.loc[
                rounds["candidate"].isin((PRIMARY, DELTA, STATIC)),
                "predictor_is_strictly_past_bool",
            ]
        ).sum()
    )
    feedback_violations = int(
        rounds.loc[
            rounds["candidate"].isin((PRIMARY, DELTA, STATIC)),
            "predictor_feedback_from_k4b_output_bool",
        ].sum()
    )
    first_gate = int(config["temporal"]["first_temporal_gate_round"])
    oracle_rows = rounds.loc[rounds["candidate"].eq(ORACLE)]
    oracle_expected_use = oracle_rows["round"].ge(first_gate)
    oracle_label_violations = int(
        (
            ~(
                ~oracle_rows["deployable_bool"]
                & ~oracle_rows["privacy_claimed_bool"]
                & oracle_rows["uses_oracle_target_bool"].eq(oracle_expected_use)
            )
        ).sum()
    )
    replace_violations = int(replacements["violation_bool"].sum())
    k4_identity = float(
        rounds.loc[
            rounds["candidate"].eq(K4) & rounds["history_gate_all_one_bool"],
            "difference_to_k2",
        ].max()
    )
    k4_reproduction = float(
        rounds.loc[rounds["candidate"].eq(K4), "k4_comparator_reproduction_error"].max()
    )
    k4b_identity = float(
        rounds.loc[
            rounds["candidate"].isin((PRIMARY, DELTA))
            & rounds["history_gate_all_one_bool"],
            "difference_to_k2",
        ].max()
    )
    finite_fraction = float(rounds["all_finite_bool"].mean())
    predictor_primary = _finite_mean(
        trajectories.loc[
            trajectories["candidate"].eq(PRIMARY),
            "attack_predictor_error_vs_current_honest_clipped_direction",
        ]
    )
    predictor_delta = _finite_mean(
        trajectories.loc[
            trajectories["candidate"].eq(DELTA),
            "attack_predictor_error_vs_current_honest_clipped_direction",
        ]
    )

    observed = {
        "device": str(manifest["device"]),
        "development_trajectories": int(completeness["expected_trajectory_ids"]),
        "development_round_rows": len(rounds),
        "development_client_rows": len(clients),
        "development_trajectory_rows": len(trajectories),
        "completeness": dict(completeness),
        "finite_metric_fraction": finite_fraction,
        "contribution_cap_violations": cap_violations,
        "gate_sum_normalization_violations": normalization_violations,
        "causal_predictor_violations": causal_violations,
        "predictor_feedback_violations": feedback_violations,
        "oracle_metadata_violations": oracle_label_violations,
        "replace_one_trials": len(replacements),
        "replace_one_violations": replace_violations,
        "replace_one_max_ratio_to_bound": float(
            replacements["ratio_observed_to_bound"].max()
        ),
        "k4_before_or_all_one_history_identity_max_abs_error": k4_identity,
        "k4_comparator_reproduction_max_abs_error": k4_reproduction,
        "k4b_k2_identity_max_abs_error": k4b_identity,
        "false_triggers": false_triggers,
        "honest_outlier_gate_drop_vs_k2_aware": outlier_drop,
        "oracle_bf_mr_persistent_gain_vs_k4": oracle_gain,
        "oracle_bf_mr_persistent_seed_difference_raw_count": len(oracle_diffs),
        "oracle_bf_mr_persistent_seed_difference_count": int(oracle_ci["n"]),
        "oracle_bf_mr_persistent_difference_seed_ci95": oracle_ci,
        "primary_bf_mr_persistent_gain_vs_k4": primary_gain,
        "primary_bf_mr_persistent_seed_difference_raw_count": len(primary_diffs),
        "primary_bf_mr_persistent_seed_difference_count": int(primary_ci["n"]),
        "primary_bf_mr_persistent_difference_seed_ci95": primary_ci,
        "primary_persistent_separated_gain_vs_k2": k2_gain,
        "primary_persistent_separated_seed_difference_raw_count": len(k2_diffs),
        "primary_persistent_separated_seed_difference_count": int(k2_ci["n"]),
        "primary_persistent_separated_difference_seed_ci95": k2_ci,
        "oracle_headroom_absolute": oracle_headroom,
        "primary_captured_headroom_absolute": primary_captured,
        "primary_oracle_headroom_capture_fraction": capture_fraction,
        "delta_oracle_headroom_capture_fraction_descriptive": delta_capture,
        "primary_persistent_ipm_attack_auc_log_ratio": {
            key: value
            for key, value in ipm_log.items()
            if key not in {"paired_cells", "incomplete_cells"}
        },
        "primary_persistent_alie_attack_auc_log_ratio": {
            key: value
            for key, value in alie_log.items()
            if key not in {"paired_cells", "incomplete_cells"}
        },
        "primary_intermittent_attack_auc_log_ratio": {
            key: value
            for key, value in intermittent_log.items()
            if key not in {"paired_cells", "incomplete_cells"}
        },
        "clean_primary_post_enrollment_auc_log_ratio": {
            key: value
            for key, value in clean_log.items()
            if key not in {"paired_cells", "incomplete_cells"}
        },
        "persistent_separated_byzantine_mass_reduction_vs_k2": mass_reduction,
        "delta_bf_mr_persistent_gain_vs_k4_descriptive": delta_gain_k4,
        "delta_persistent_separated_gain_vs_k2_descriptive": delta_gain_k2,
        "delta_persistent_separated_direct_mass_reduction_vs_k2_descriptive": delta_mass_reduction,
        "primary_predictor_error_vs_honest_clipped_direction": predictor_primary,
        "delta_predictor_error_vs_honest_clipped_direction": predictor_delta,
        "historical_contamination_separately_identified": False,
        "persistent_separated_detection_rate_within_deadline": detection,
        "persistent_separated_recovery_rate_within_deadline": recovery,
    }
    checks = {
        "complete": bool(completeness["exact_identifier_completeness"])
        and float(completeness["complete_fraction"])
        >= float(gates["complete_fraction_min"]),
        "finite": finite_fraction >= float(gates["finite_metric_fraction_min"]),
        "production_device": str(manifest["device"])
        == str(gates["production_device_required"]),
        "contribution_cap": cap_violations
        <= int(gates["contribution_cap_violation_max"]),
        "no_gate_sum_normalization": normalization_violations == 0,
        "causal_predictor": causal_violations == 0,
        "no_predictor_feedback": feedback_violations == 0,
        "oracle_metadata": oracle_label_violations == 0,
        "replace_one": replace_violations <= int(gates["replace_one_violation_max"]),
        "k4_comparator_reproduction": k4_reproduction
        <= float(gates["k4_comparator_reproduction_abs_error_max"]),
        "k4b_k2_identity_when_history_one": k4b_identity
        <= float(gates["k4b_k2_identity_when_history_gate_one_abs_error_max"]),
        "regular_client_round_false_trigger": false_triggers["client_round_rate"]
        <= float(gates["regular_client_round_false_trigger_rate_max"]),
        "regular_trajectory_false_trigger": false_triggers["trajectory_rate"]
        <= float(gates["regular_trajectory_false_trigger_rate_max"]),
        "pooled_regular_false_trigger_tier_gap": false_triggers["tier_rate_gap"]
        <= float(gates["pooled_regular_false_trigger_tier_gap_max"]),
        "pooled_regular_gate_mean_tier_gap": false_triggers["tier_gate_mean_gap"]
        <= float(gates["pooled_regular_gate_mean_tier_gap_max"]),
        "honest_outlier_gate_drop": outlier_drop
        <= float(gates["honest_outlier_gate_drop_vs_k2_aware_max"]),
        "byzantine_mass_reduction": mass_reduction
        >= float(
            gates["persistent_separated_byzantine_mass_reduction_vs_k2_aware_min"]
        ),
        "detection": detection
        >= float(gates["persistent_separated_detection_rate_within_deadline_min"]),
        "recovery": recovery
        >= float(gates["persistent_separated_recovery_rate_within_deadline_min"]),
        "oracle_headroom_gain": oracle_gain
        >= float(gates["oracle_bf_mr_persistent_gain_vs_k4_min"]),
        "oracle_effectiveness_seed_count": int(oracle_ci["n"]) == 5,
        "oracle_headroom_ci": float(oracle_ci["high"])
        <= float(gates["oracle_bf_mr_persistent_difference_ci95_high_max"]),
        "primary_gain_vs_k4": primary_gain
        >= float(gates["primary_bf_mr_persistent_gain_vs_k4_min"]),
        "primary_vs_k4_effectiveness_seed_count": int(primary_ci["n"]) == 5,
        "primary_ci_vs_k4": float(primary_ci["high"])
        <= float(gates["primary_bf_mr_persistent_difference_ci95_high_max"]),
        "primary_gain_vs_k2": k2_gain
        >= float(gates["primary_persistent_separated_gain_vs_k2_min"]),
        "primary_vs_k2_effectiveness_seed_count": int(k2_ci["n"]) == 5,
        "primary_ci_vs_k2": float(k2_ci["high"])
        <= float(gates["primary_persistent_separated_difference_ci95_high_max"]),
        "headroom_capture": capture_fraction
        >= float(gates["primary_oracle_headroom_capture_fraction_min"]),
        "ipm_noninferiority_vs_k4": ipm_log["n_seed_pairs"] == 5
        and ipm_log["exp_log_ratio_ci95_high"]
        <= float(gates["primary_persistent_ipm_exp_log_ratio_ci95_high_to_k4_max"]),
        "alie_noninferiority_vs_k4": alie_log["n_seed_pairs"] == 5
        and alie_log["exp_log_ratio_ci95_high"]
        <= float(gates["primary_persistent_alie_exp_log_ratio_ci95_high_to_k4_max"]),
        "intermittent_noninferiority_vs_k4": intermittent_log["n_seed_pairs"] == 5
        and intermittent_log["exp_log_ratio_ci95_high"]
        <= float(gates["primary_intermittent_exp_log_ratio_ci95_high_to_k4_max"]),
        "clean_noninferiority_vs_k2": clean_log["n_seed_pairs"] == 5
        and clean_log["exp_log_ratio_ci95_high"]
        <= float(
            gates["clean_primary_post_enrollment_exp_log_ratio_ci95_high_to_k2_max"]
        ),
    }
    oracle_pass = bool(
        checks["oracle_headroom_gain"]
        and checks["oracle_effectiveness_seed_count"]
        and checks["oracle_headroom_ci"]
    )
    decision = {
        "decision": "promote_to_separate_holdout_runner"
        if all(checks.values())
        else "stop_after_development",
        "all_gates_pass": all(checks.values()),
        "oracle_headroom_mechanism_viable": oracle_pass,
        "stop_mechanism_if_oracle_headroom_fails": not oracle_pass,
        "checks": checks,
        "observed": observed,
        "holdout_opened": False,
    }
    evidence = {
        "oracle_vs_k4": {**oracle_evidence, "ci95": oracle_ci},
        "primary_vs_k4": {**primary_evidence, "ci95": primary_ci},
        "primary_vs_k2": {**k2_evidence, "ci95": k2_ci},
        "noninferiority": {
            "ipm": ipm_log,
            "alie": alie_log,
            "intermittent": intermittent_log,
            "clean": clean_log,
        },
    }
    return decision, evidence


def _summary_from_trajectories(trajectories: pd.DataFrame) -> pd.DataFrame:
    keys = ["candidate", "noise_regime", "threat", "schedule"]
    rows: list[dict[str, Any]] = []
    for values, group in trajectories.groupby(keys, sort=True):
        candidate, regime, threat, schedule = values
        rows.append(
            {
                "candidate": candidate,
                "noise_regime": regime,
                "threat": threat,
                "schedule": schedule,
                "n_trajectories": len(group),
                "deployable": False,
                "end_to_end_deployability_claimed": False,
                "screen_conditioned_on_semi_oracle_anchor": True,
                "privacy_claimed": candidate != ORACLE,
                "attack_auc_mean": _finite_mean(group["attack_auc"]),
                "attack_auc_std": _finite_std(group["attack_auc"]),
                "active_attack_auc_mean": _finite_mean(group["active_attack_auc"]),
                "recovery_auc_mean": _finite_mean(group["recovery_auc"]),
                "post_enrollment_auc_mean": _finite_mean(group["post_enrollment_auc"]),
                "attack_byzantine_direct_current_mass_share_mean": _finite_mean(
                    group["attack_byzantine_direct_current_mass_share"]
                ),
                "attack_byzantine_imputed_mass_share_mean": _finite_mean(
                    group["attack_byzantine_imputed_mass_share"]
                ),
                "attack_byzantine_total_slot_mass_share_mean_descriptive": _finite_mean(
                    group["attack_byzantine_total_slot_mass_share_descriptive"]
                ),
                "attack_predictor_error_vs_honest_clipped_direction_mean": (
                    _finite_mean(
                        group[
                            "attack_predictor_error_vs_current_honest_clipped_direction"
                        ]
                    )
                    if candidate in {PRIMARY, DELTA, STATIC, ORACLE}
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def audit_results(
    config_path: Path,
    results_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest = _guard_completed(results_dir)
    required = {
        "decision": results_dir / "decision.json",
        "rounds": results_dir / "development_round_rows.csv",
        "clients": results_dir / "development_client_rows.csv",
        "trajectories": results_dir / "development_trajectory_rows.csv",
        "replace_one": results_dir / "replace_one_audit.csv",
        "summary": results_dir / "summary.csv",
        "calibration_provenance": results_dir / "frozen_calibration_provenance.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Completed K4b directory is incomplete: " + ", ".join(missing)
        )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    decision_reported = _read_json(required["decision"])
    calibration_provenance = _read_json(required["calibration_provenance"])
    rounds = pd.read_csv(required["rounds"], low_memory=False)
    clients = pd.read_csv(required["clients"], low_memory=False)
    trajectories = pd.read_csv(required["trajectories"], low_memory=False)
    replacements = pd.read_csv(required["replace_one"], low_memory=False)
    summary = pd.read_csv(required["summary"], low_memory=False)

    required_columns = {
        "rounds": {
            "trajectory_id",
            "candidate",
            "round",
            "phase",
            "reference_error",
            "difference_to_k2",
            "difference_to_k4",
            "attack_active",
            "all_finite",
            "contribution_cap_respected",
            "normalization_by_gate_sum",
            "predictor_is_strictly_past",
            "predictor_feedback_from_k4b_output",
            "deployable",
            "privacy_claimed",
            "uses_oracle_target",
            "history_gate_all_one",
            "k4_comparator_reproduction_error",
            "byzantine_direct_current_mass_share",
            "byzantine_imputed_mass_share",
            "byzantine_total_slot_mass_share",
            "direct_current_mass_total",
            "imputed_mass_total",
            "total_slot_mass_total",
            "max_client_contribution_norm",
            "predictor_error_vs_current_honest_clipped_direction",
        },
        "clients": {
            "trajectory_id",
            "round",
            "client",
            "temporal_statistic",
            "temporal_gate",
            "current_gate",
            "final_gate",
            "latent_byzantine",
            "active_byzantine",
            "honest_outlier",
            "honest_regular",
            "public_noise_tier",
            "gate_drop_vs_k2_aware",
            "primary_direct_current_contribution_norm",
            "primary_imputed_contribution_norm",
            "primary_total_slot_contribution_norm",
            "delta_direct_current_contribution_norm",
            "delta_imputed_contribution_norm",
            "delta_total_slot_contribution_norm",
        },
    }
    for name, columns in required_columns.items():
        frame = rounds if name == "rounds" else clients
        absent = sorted(columns - set(frame.columns))
        if absent:
            raise ValueError(f"{name} missing columns: {absent}")

    bool_columns = {
        "rounds": [
            "attack_active",
            "all_finite",
            "contribution_cap_respected",
            "normalization_by_gate_sum",
            "predictor_is_strictly_past",
            "predictor_feedback_from_k4b_output",
            "deployable",
            "privacy_claimed",
            "uses_oracle_target",
            "history_gate_all_one",
        ],
        "clients": [
            "attack_active",
            "latent_byzantine",
            "active_byzantine",
            "honest_outlier",
            "honest_regular",
        ],
        "replace": [
            "same_past",
            "same_predictor",
            "same_history_gate",
            "same_covariances",
            "same_anchor",
            "violation",
            "history_gate_has_suppression",
        ],
    }
    for column in bool_columns["rounds"]:
        rounds[f"{column}_bool"] = _as_bool(rounds[column], name=f"rounds.{column}")
    for column in bool_columns["clients"]:
        clients[f"{column}_bool"] = _as_bool(clients[column], name=f"clients.{column}")
    for column in bool_columns["replace"]:
        replacements[f"{column}_bool"] = _as_bool(
            replacements[column], name=f"replace.{column}"
        )

    expected = _expected_matrix(config)
    audit = Audit()
    audit.check("matrix_has_900_trajectories", len(expected) == 900)
    fields = [
        "seed",
        "noise_regime",
        "noise_permutation",
        "outlier_geometry",
        "honest_dynamics",
        "threat",
        "schedule",
    ]
    audit.check("round_axes_match_ids", _axis_matches(rounds, expected, fields))
    audit.check("client_axes_match_ids", _axis_matches(clients, expected, fields))
    audit.check(
        "trajectory_axes_match_ids", _axis_matches(trajectories, expected, fields)
    )
    completeness = _completeness(
        rounds, clients, trajectories, replacements, config, expected
    )
    audit.check(
        "exact_identifier_completeness",
        completeness["exact_identifier_completeness"],
        completeness,
    )
    audit.check(
        "pairing_ids_and_crn_complete",
        bool(
            rounds["pairing_id"]
            .astype(str)
            .eq(
                rounds["trajectory_id"].astype(str)
                + "|"
                + rounds["round"].astype(int).astype(str)
            )
            .all()
            and rounds["crn_key"].astype(str).eq(rounds["pairing_id"].astype(str)).all()
            and rounds.groupby("pairing_id")["candidate"]
            .nunique()
            .eq(len(CANDIDATES))
            .all()
        ),
    )
    numeric_round = pd.to_numeric(rounds["round"], errors="coerce").astype(int)
    audit.check(
        "phase_labels_exact",
        bool(rounds["phase"].eq(_phase(numeric_round, config)).all()),
    )
    audit.check(
        "round_attack_schedule_exact",
        bool(rounds["attack_active_bool"].eq(_attack_active(rounds, config)).all()),
    )
    audit.check(
        "client_attack_schedule_exact",
        bool(clients["attack_active_bool"].eq(_attack_active(clients, config)).all()),
    )

    core_numeric_fields = [
        "reference_error",
        "difference_to_k2",
        "difference_to_k4",
        "imputation_coefficient_mean",
        "predictor_norm",
        "k4_comparator_reproduction_error",
    ]
    certified_numeric_fields = [
        "current_gate_mean",
        "final_gate_mean",
        "byzantine_direct_current_mass_share",
        "byzantine_imputed_mass_share",
        "byzantine_total_slot_mass_share",
        "direct_current_mass_total",
        "imputed_mass_total",
        "total_slot_mass_total",
        "max_client_contribution_norm",
    ]
    certified_rows = rounds.loc[rounds["candidate"].isin(CERTIFIED)]
    audit.check(
        "round_required_metrics_finite",
        all(
            np.isfinite(pd.to_numeric(rounds[field], errors="coerce")).all()
            for field in core_numeric_fields
        )
        and all(
            np.isfinite(pd.to_numeric(certified_rows[field], errors="coerce")).all()
            for field in certified_numeric_fields
        ),
    )
    first_gate = int(config["temporal"]["first_temporal_gate_round"])
    predictor_candidates = {PRIMARY, DELTA, STATIC, ORACLE}
    predictor_rows = rounds.loc[
        rounds["candidate"].isin(predictor_candidates) & rounds["round"].ge(first_gate)
    ]
    audit.check(
        "predictor_errors_finite_for_predictor_candidates",
        np.isfinite(
            pd.to_numeric(
                predictor_rows["predictor_error_vs_current_honest_clipped_direction"],
                errors="coerce",
            )
        ).all(),
    )
    audit.check("round_all_finite_flags_true", bool(rounds["all_finite_bool"].all()))
    rolling = rounds.loc[
        rounds["candidate"].isin((PRIMARY, DELTA)) & rounds["round"].ge(first_gate)
    ]
    static = rounds.loc[rounds["candidate"].eq(STATIC) & rounds["round"].ge(first_gate)]
    rolling_source = pd.to_numeric(
        rolling["predictor_max_source_round"], errors="coerce"
    )
    static_source = pd.to_numeric(static["predictor_max_source_round"], errors="coerce")
    audit.check(
        "predictor_source_rounds_are_exactly_causal",
        bool(
            rolling_source.eq(
                pd.to_numeric(rolling["round"], errors="coerce") - 1
            ).all()
            and static_source.eq(12).all()
            and rounds.loc[
                rounds["candidate"].isin((PRIMARY, DELTA, STATIC))
                & rounds["round"].lt(first_gate),
                "predictor_max_source_round",
            ]
            .isna()
            .all()
        ),
    )
    influence_cap = float(config["references"]["total_client_influence_cap"])
    certified_maximum = pd.to_numeric(
        certified_rows["max_client_contribution_norm"], errors="coerce"
    )
    component_shares = certified_rows[
        [
            "byzantine_direct_current_mass_share",
            "byzantine_imputed_mass_share",
            "byzantine_total_slot_mass_share",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    audit.check(
        "numeric_contribution_cap_and_mass_share_ranges",
        bool(
            certified_maximum.le(influence_cap + 1.0e-6).all()
            and component_shares.ge(-1.0e-9).all().all()
            and component_shares.le(1.0 + 1.0e-9).all().all()
        ),
        {"influence_cap": influence_cap},
    )
    component_ok, component_detail = _component_mass_audit(rounds, clients, config)
    audit.check(
        "component_masses_rebuilt_from_client_rows", component_ok, component_detail
    )

    rebuilt = _rebuild_trajectories(rounds, clients, config)
    merge_keys = ["trajectory_id", "candidate"]
    compared = trajectories.merge(
        rebuilt,
        on=merge_keys,
        suffixes=("_reported", "_rebuilt"),
        validate="one_to_one",
    )
    trajectory_metrics = [
        "enrollment_auc",
        "monitoring_auc",
        "attack_auc",
        "active_attack_auc",
        "recovery_auc",
        "post_enrollment_auc",
        "attack_byzantine_direct_current_mass_share",
        "attack_byzantine_imputed_mass_share",
        "attack_byzantine_total_slot_mass_share_descriptive",
        "attack_predictor_error_vs_current_honest_clipped_direction",
        "detection_rate_within_deadline",
        "recovery_rate_within_deadline",
    ]
    audit.check(
        "trajectory_metrics_rebuilt_from_round_and_client_rows",
        len(compared) == len(trajectories) == len(rebuilt)
        and all(
            _allclose(compared[f"{field}_reported"], compared[f"{field}_rebuilt"])
            for field in trajectory_metrics
        ),
    )

    for column in (
        "theoretical_replace_one_bound",
        "observed_replace_one_difference",
        "ratio_observed_to_bound",
        "history_gate_min",
    ):
        replacements[column] = pd.to_numeric(replacements[column], errors="coerce")
    replace_bound = (
        2.0
        * float(config["references"]["total_client_influence_cap"])
        / float(config["cohort"]["num_clients"])
    )
    recomputed_replace_violation = replacements["observed_replace_one_difference"].gt(
        replacements["theoretical_replace_one_bound"] + 1.0e-6
    )
    attacked_replace = replacements["audit_scenario"].eq(
        "persistent_bitflip_round17_h_below_one"
    )
    audit.check(
        "replace_one_current_certificate_rebuilt",
        bool(
            np.allclose(
                replacements["theoretical_replace_one_bound"],
                replace_bound,
                atol=1.0e-12,
                rtol=0.0,
            )
            and np.allclose(
                replacements["ratio_observed_to_bound"],
                replacements["observed_replace_one_difference"]
                / replacements["theoretical_replace_one_bound"],
                atol=2.0e-9,
                rtol=2.0e-8,
            )
            and replacements["violation_bool"].eq(recomputed_replace_violation).all()
            and replacements["same_past_bool"].all()
            and replacements["same_predictor_bool"].all()
            and replacements["same_history_gate_bool"].all()
            and replacements["same_covariances_bool"].all()
            and replacements["same_anchor_bool"].all()
            and replacements.loc[
                attacked_replace, "history_gate_has_suppression_bool"
            ].all()
            and replacements.loc[attacked_replace, "history_gate_min"]
            .lt(1.0 - 1.0e-7)
            .all()
        ),
        {"bound": replace_bound, "attacked_rows": int(attacked_replace.sum())},
    )

    decision_recomputed, paired_evidence = _recompute_decision(
        rounds, clients, rebuilt, replacements, config, completeness, manifest
    )
    decision_differences = _compare_tree(decision_recomputed, decision_reported)
    audit.check(
        "decision_json_matches_independent_recalculation",
        not decision_differences,
        decision_differences[:100],
    )

    rebuilt_summary = _summary_from_trajectories(rebuilt)
    summary_keys = ["candidate", "noise_regime", "threat", "schedule"]
    summary_compare = summary.merge(
        rebuilt_summary,
        on=summary_keys,
        suffixes=("_reported", "_rebuilt"),
        validate="one_to_one",
    )
    summary_metrics = [
        "n_trajectories",
        "attack_auc_mean",
        "attack_auc_std",
        "active_attack_auc_mean",
        "recovery_auc_mean",
        "post_enrollment_auc_mean",
        "attack_byzantine_direct_current_mass_share_mean",
        "attack_byzantine_imputed_mass_share_mean",
        "attack_byzantine_total_slot_mass_share_mean_descriptive",
        "attack_predictor_error_vs_honest_clipped_direction_mean",
    ]
    audit.check(
        "summary_csv_rebuilt_from_raw_rows",
        len(summary_compare) == len(summary) == len(rebuilt_summary)
        and all(
            _allclose(
                summary_compare[f"{field}_reported"],
                summary_compare[f"{field}_rebuilt"],
            )
            for field in summary_metrics
        ),
    )

    source_hashes = {
        "runner": _sha256(
            ROOT / "scripts/run_gaussian_aware_reference_g0g_k4b_past_imputation.py"
        ),
        "k4b_primitive": _sha256(ROOT / "algorithms/gaussian_aware_reference_k4b.py"),
        **{
            str(relative): _sha256(ROOT / str(relative))
            for relative in config["frozen_source_hashes"]
        },
    }
    lock_path = ROOT / str(config["preregistration_lock"]["path"])
    lock_registry = _read_json(lock_path)
    registry_entries = {
        **lock_registry.get("locked_files", {}),
        **lock_registry.get("dependencies", {}),
    }
    registry_hashes_match = bool(registry_entries) and all(
        (ROOT / str(relative)).is_file()
        and _sha256(ROOT / str(relative)) == str(expected_hash)
        for relative, expected_hash in registry_entries.items()
    )
    audit.check(
        "published_lock_and_all_registry_hashes_match",
        _sha256(lock_path) == PUBLISHED_LOCK_SHA256 and registry_hashes_match,
        {
            "published_lock_sha256": PUBLISHED_LOCK_SHA256,
            "observed_lock_sha256": _sha256(lock_path),
            "registry_entries": len(registry_entries),
        },
    )
    audit.check(
        "config_hash_matches_manifest",
        _sha256(config_path) == manifest.get("config_sha256"),
    )
    audit.check(
        "source_hashes_match_manifest", source_hashes == manifest.get("source_sha256")
    )
    audit.check(
        "lock_hash_matches_manifest",
        _sha256(lock_path) == manifest.get("preregistration_lock", {}).get("sha256")
        and manifest.get("preregistration_lock", {}).get("verified") is True,
    )
    provenance_ok = calibration_provenance.get("recalibrated") is False
    for name, entry in config["frozen_calibrations"].items():
        if name == "reuse_thresholds_exactly" or name == "recalibrate":
            continue
        observed_hash = _sha256(ROOT / str(entry["path"]))
        reported = calibration_provenance.get(name, {})
        provenance_ok = provenance_ok and (
            observed_hash == str(entry["sha256"])
            and reported.get("observed_sha256") == observed_hash
            and reported.get("sha256_verified") is True
        )
    audit.check("frozen_calibrations_match_provenance", provenance_ok)
    development = set(int(value) for value in config["randomness"]["development_seeds"])
    holdout = set(int(value) for value in config["randomness"]["holdout_seeds"])
    raw_seed_sets = {
        "rounds": set(rounds["seed"].astype(int)),
        "clients": set(clients["seed"].astype(int)),
        "trajectories": set(trajectories["seed"].astype(int)),
        "replace_one": set(replacements["seed"].astype(int)),
    }
    audit.check(
        "holdout_closed_and_absent",
        all(values == development for values in raw_seed_sets.values())
        and not any(values & holdout for values in raw_seed_sets.values())
        and manifest.get("holdout_opened") is False
        and decision_reported.get("holdout_opened") is False,
    )
    audit.check(
        "manifest_scope_and_device",
        manifest.get("device") == "mps"
        and manifest.get("dtype") == "torch.float32"
        and manifest.get("screen_conditioned_on_semi_oracle_anchor") is True
        and manifest.get("end_to_end_deployability_claimed") is False,
    )

    failed = [name for name, passed in audit.checks.items() if not passed]
    failed_scientific = [
        name for name, passed in decision_recomputed["checks"].items() if not passed
    ]
    payload = {
        "audit_kind": "independent_k4b_raw_csv_recalculation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.resolve()),
        "results_dir": str(results_dir.resolve()),
        "audit_verdict": "PASS" if not failed else "FAIL",
        "failed_audit_checks": failed,
        "audit_checks": audit.checks,
        "audit_details": audit.details,
        "scientific_decision": decision_recomputed["decision"],
        "all_scientific_checks_pass": decision_recomputed["all_gates_pass"],
        "failed_scientific_checks": failed_scientific,
        "recomputed_decision": decision_recomputed,
        "reported_decision_difference_count": len(decision_differences),
        "reported_decision_differences": decision_differences[:100],
        "paired_seed_evidence": paired_evidence,
        "component_mass_evidence": component_detail,
        "expected_counts": {
            "development_trajectories": len(expected),
            "round_rows": int(completeness["round_rows_expected"]),
            "client_rows": int(completeness["client_rows_expected"]),
            "trajectory_rows": int(completeness["trajectory_rows_expected"]),
            "replace_one_rows": int(completeness["replace_rows_expected"]),
            "summary_rows": len(rebuilt_summary),
        },
        "raw_file_sha256": {
            "manifest": _sha256(results_dir / "manifest.json"),
            **{name: _sha256(path) for name, path in required.items()},
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.results_dir / "independent_audit.json"
    payload = audit_results(
        args.config.resolve(), args.results_dir.resolve(), output.resolve()
    )
    print(
        f"K4b independent audit: {payload['audit_verdict']} — "
        f"scientific decision={payload['scientific_decision']}"
    )
    print(f"Saved {output}")
    if payload["audit_verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
