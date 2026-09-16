#!/usr/bin/env python3
"""Independently audit completed G0g-K4c-CH development artifacts.

This module deliberately does not import the K4c experiment runner.  It first
opens only ``manifest.json`` and refuses to inspect the configuration, lock, or
raw CSV files unless the manifest says ``completed_development``.  It then
reconstructs the exact matrix, seed-level P/Q/C and regime contrasts,
Student-t intervals, preregistered validity/scientific gates, certificates, and
the three-state decision directly from the raw artifacts.
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

import yaml

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "gaussian_aware_reference_g0g_k4c_causal_headroom_mps_v1"
DEFAULT_CONFIG = ROOT / (
    "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k4c_causal_headroom.yaml"
)
DEFAULT_LOCK = ROOT / (
    "configs/ldp_gradient_far/"
    "gaussian_aware_reference_g0g_k4c_causal_headroom.lock.json"
)
DEFAULT_RESULTS = ROOT / (
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k4c_causal_headroom_mps_v1"
)

K2 = "g0g_k2"
K4 = "g0g_k4_temporal_causal_gate"
K4B = "g0g_k4b_rolling_past_imputation"
SEMI_ORACLE = "g0g_k4c_ch_current_randomness_conditional_mse_semi_oracle"
POINTWISE = "g0g_k4b_pointwise_optimal_oracle"
CANDIDATES = (K2, K4, K4B, SEMI_ORACLE, POINTWISE)

LOCKED_PATHS = {
    "algorithms/gaussian_aware_reference_k4c_ch.py",
    "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k4c_causal_headroom.yaml",
    "scripts/audit_gaussian_aware_reference_g0g_k4c_ch.py",
    "scripts/run_gaussian_aware_reference_g0g_k4c_causal_headroom.py",
    "tests/test_audit_gaussian_aware_reference_g0g_k4c_ch.py",
    "tests/test_gaussian_aware_reference_g0g_k4c_ch.py",
    "tests/test_run_gaussian_aware_reference_g0g_k4c_causal_headroom.py",
    "output/analysis/Gaussian_Aware_G0g_K4c_Causal_Headroom_Protocol_PreRun.md",
}
DEPENDENCY_PATHS = {
    "algorithms/gaussian_aware_reference.py",
    "algorithms/gaussian_aware_reference_k4b.py",
    "scripts/run_gaussian_aware_reference_g0g_k1.py",
    "scripts/run_gaussian_aware_reference_g0g_k2.py",
    "scripts/run_gaussian_aware_reference_g0g_k4_tcg.py",
    "scripts/run_gaussian_aware_reference_g0g_k4b_past_imputation.py",
    "scripts/run_gaussian_aware_reference_oracle.py",
    "robustness/aggregators.py",
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k2_mps_v1/calibration.json",
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k4_tcg_mps_v1/temporal_calibration.json",
}

CHILD_FINITE_FIELDS = (
    "squared_reference_error",
    "reference_error_l2_descriptive",
    "pointwise_excess_mse_over_candidate",
    "missing_slot_mass",
    "predictor_norm",
    "k4_manual_formula_error",
    "frozen_k4_comparator_error_descriptive",
    "k4b_comparator_reproduction_error",
    "fixed_denominator_formula_error",
)
HISTORY_FINITE_FIELDS = (
    "missing_slot_mass",
    "rolling_predictor_norm",
    "semi_oracle_predictor_norm",
    "semi_oracle_raw_predictor_norm",
    "split_a_b_predictor_distance",
    "split_a_b_aggregate_scale_distance",
)
REPLACE_FINITE_FIELDS = (
    "observed_difference",
    "theoretical_bound",
    "ratio_to_bound",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _guard_completed(results_dir: Path) -> dict[str, Any]:
    """Read only the manifest and stop before any other artifact access."""

    manifest_path = results_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing K4c-CH manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "completed_development":
        raise RuntimeError(
            "K4c-CH results are not complete; config, lock, and raw artifacts "
            f"were not opened: status={manifest.get('status')!r}"
        )
    if manifest.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError(
            f"Unexpected K4c-CH campaign: {manifest.get('campaign_id')!r}"
        )
    return manifest


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing K4c-CH raw artifact: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"Not a Boolean value: {value!r}")


def _safe_bool(value: Any) -> bool:
    try:
        return _as_bool(value)
    except (TypeError, ValueError):
        return False


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _all_finite(row: Mapping[str, Any], fields: Sequence[str]) -> bool:
    return all(math.isfinite(_safe_float(row.get(field))) for field in fields)


def _strict_mean(values: Sequence[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        return float("nan")
    return statistics.fmean(values)


def _strict_max(values: Sequence[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        return float("nan")
    return max(values)


def _ci(values: Sequence[float], t_critical: float) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) < 2:
        return {
            "n": len(finite),
            "mean": _strict_mean(finite),
            "sd": float("nan"),
            "low": float("nan"),
            "high": float("nan"),
        }
    mean = statistics.fmean(finite)
    sd = statistics.stdev(finite)
    half = float(t_critical) * sd / math.sqrt(len(finite))
    return {
        "n": len(finite),
        "mean": mean,
        "sd": sd,
        "low": mean - half,
        "high": mean + half,
    }


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def _noise_cells(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        (str(regime["name"]), str(permutation))
        for regime in config["privacy_noise"]["regimes"]
        for permutation in regime["permutations"]
    ]


def _expected_histories(
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    for seed in config["randomness"]["development_outer_seeds"]:
        for regime, permutation in _noise_cells(config):
            for geometry in config["cohort"]["honest_outliers"]["geometries"]:
                for dynamics in config["honest_dynamics"]["names"]:
                    for threat in config["threats"]["names"]:
                        for round_index in config["temporal"]["assessment_rounds"]:
                            values = {
                                "seed": int(seed),
                                "noise_regime": str(regime),
                                "noise_permutation": str(permutation),
                                "outlier_geometry": str(geometry),
                                "honest_dynamics": str(dynamics),
                                "threat": str(threat),
                                "assessment_round": int(round_index),
                            }
                            identifier = "|".join(
                                str(value) for value in values.values()
                            )
                            if identifier in expected:
                                raise RuntimeError(
                                    f"Duplicate expected history: {identifier}"
                                )
                            expected[identifier] = values
    return expected


def _parse_seed_registry(value: Any) -> list[int]:
    text = str(value).strip()
    if not text:
        return []
    try:
        return [int(token) for token in text.split(",")]
    except ValueError:
        return []


def _matrix_audit(
    config: Mapping[str, Any],
    histories: Sequence[Mapping[str, Any]],
    children: Sequence[Mapping[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    expected = _expected_histories(config)
    observed_ids = [str(row.get("history_id", "")) for row in histories]
    history_ids_exact = len(observed_ids) == len(expected) and set(observed_ids) == set(
        expected
    )
    history_metadata_exact = history_ids_exact
    if history_metadata_exact:
        for row in histories:
            specification = expected[str(row["history_id"])]
            for field, value in specification.items():
                observed = (
                    _safe_int(row.get(field))
                    if field
                    in {
                        "seed",
                        "assessment_round",
                    }
                    else str(row.get(field, ""))
                )
                if observed != value:
                    history_metadata_exact = False
                    break
            if not history_metadata_exact:
                break

    evaluation_count = int(config["nested_monte_carlo"]["evaluation_children"])
    expected_child_keys = {
        (history_id, child, candidate)
        for history_id in expected
        for child in range(evaluation_count)
        for candidate in CANDIDATES
    }
    observed_child_keys = [
        (
            str(row.get("history_id", "")),
            _safe_int(row.get("evaluation_child")),
            str(row.get("candidate", "")),
        )
        for row in children
    ]
    child_keys_exact = (
        len(observed_child_keys) == len(expected_child_keys)
        and set(observed_child_keys) == expected_child_keys
    )

    child_metadata_exact = child_keys_exact
    evaluation_seed_exact = child_keys_exact
    crn_key_exact = child_keys_exact
    if child_keys_exact:
        tag = str(config["nested_monte_carlo"]["evaluation_stream_tag"])
        for row in children:
            history_id = str(row["history_id"])
            specification = expected[history_id]
            child = _safe_int(row["evaluation_child"])
            for field, value in specification.items():
                observed = (
                    _safe_int(row.get(field))
                    if field
                    in {
                        "seed",
                        "assessment_round",
                    }
                    else str(row.get(field, ""))
                )
                if observed != value:
                    child_metadata_exact = False
                    break
            expected_seed = _stable_seed(
                tag,
                specification["seed"],
                specification["noise_regime"],
                specification["noise_permutation"],
                specification["outlier_geometry"],
                specification["honest_dynamics"],
                specification["threat"],
                specification["assessment_round"],
                child,
            )
            evaluation_seed_exact &= (
                _safe_int(row.get("evaluation_child_seed")) == expected_seed
            )
            crn_key_exact &= str(row.get("crn_key", "")) == (
                f"{history_id}|evaluation|{child}"
            )

    construction_tag = str(config["nested_monte_carlo"]["construction_stream_tag"])
    evaluation_tag = str(config["nested_monte_carlo"]["evaluation_stream_tag"])
    construction_count = int(config["nested_monte_carlo"]["construction_children"])
    rng_disjoint = True
    declared_overlap_zero = True
    declared_child_counts_exact = True
    recorded_registries_exact = True
    recorded_construction: list[int] = []
    recorded_evaluation: list[int] = []
    for history_id, specification in expected.items():
        construction_list = [
            _stable_seed(
                construction_tag,
                specification["seed"],
                specification["noise_regime"],
                specification["noise_permutation"],
                specification["outlier_geometry"],
                specification["honest_dynamics"],
                specification["threat"],
                specification["assessment_round"],
                child,
            )
            for child in range(construction_count)
        ]
        evaluation_list = [
            _stable_seed(
                evaluation_tag,
                specification["seed"],
                specification["noise_regime"],
                specification["noise_permutation"],
                specification["outlier_geometry"],
                specification["honest_dynamics"],
                specification["threat"],
                specification["assessment_round"],
                child,
            )
            for child in range(evaluation_count)
        ]
        construction = set(construction_list)
        evaluation = set(evaluation_list)
        rng_disjoint &= not bool(construction & evaluation)
        matching = [
            row for row in histories if str(row.get("history_id", "")) == history_id
        ]
        if len(matching) != 1:
            declared_overlap_zero = False
            declared_child_counts_exact = False
            continue
        row = matching[0]
        reported_construction = _parse_seed_registry(
            row.get("construction_child_seed_registry")
        )
        reported_evaluation = _parse_seed_registry(
            row.get("evaluation_child_seed_registry")
        )
        recorded_construction.extend(reported_construction)
        recorded_evaluation.extend(reported_evaluation)
        construction_payload = ",".join(str(value) for value in reported_construction)
        evaluation_payload = ",".join(str(value) for value in reported_evaluation)
        recorded_registries_exact &= (
            reported_construction == construction_list
            and reported_evaluation == evaluation_list
            and str(row.get("construction_child_seed_registry_sha256", ""))
            == hashlib.sha256(construction_payload.encode("utf-8")).hexdigest()
            and str(row.get("evaluation_child_seed_registry_sha256", ""))
            == hashlib.sha256(evaluation_payload.encode("utf-8")).hexdigest()
            and _safe_int(row.get("construction_child_seed_unique_count"))
            == len(set(reported_construction))
            and _safe_int(row.get("evaluation_child_seed_unique_count"))
            == len(set(reported_evaluation))
        )
        declared_overlap_zero &= (
            _safe_int(row.get("construction_evaluation_seed_overlap")) == 0
        )
        declared_child_counts_exact &= (
            _safe_int(row.get("construction_children")) == construction_count
            and _safe_int(row.get("evaluation_children")) == evaluation_count
        )

    expected_construction_total = int(
        config["statistical_analysis"]["construction_child_seeds_total"]
    )
    expected_evaluation_total = int(
        config["statistical_analysis"]["evaluation_child_seeds_total"]
    )
    global_registries_exact = (
        len(recorded_construction) == expected_construction_total
        and len(set(recorded_construction)) == expected_construction_total
        and len(recorded_evaluation) == expected_evaluation_total
        and len(set(recorded_evaluation)) == expected_evaluation_total
        and not (set(recorded_construction) & set(recorded_evaluation))
    )

    checks = {
        "history_ids_exact": history_ids_exact,
        "history_metadata_exact": history_metadata_exact,
        "child_keys_exact": child_keys_exact,
        "child_metadata_exact": child_metadata_exact,
        "evaluation_seed_exact": evaluation_seed_exact,
        "crn_key_exact": crn_key_exact,
        "derived_rng_namespaces_disjoint": rng_disjoint,
        "declared_rng_overlap_zero": declared_overlap_zero,
        "declared_child_counts_exact": declared_child_counts_exact,
        "recorded_seed_registries_exact": recorded_registries_exact,
        "global_seed_registries_unique_and_disjoint": global_registries_exact,
    }
    return all(checks.values()), {
        "checks": checks,
        "expected_histories": len(expected),
        "observed_history_rows": len(histories),
        "expected_child_rows": len(expected_child_keys),
        "observed_child_rows": len(children),
        "unique_history_ids": len(set(observed_ids)),
        "unique_child_keys": len(set(observed_child_keys)),
        "recorded_construction_child_seeds": len(recorded_construction),
        "unique_recorded_construction_child_seeds": len(set(recorded_construction)),
        "recorded_evaluation_child_seeds": len(recorded_evaluation),
        "unique_recorded_evaluation_child_seeds": len(set(recorded_evaluation)),
        "global_recorded_stream_overlap": len(
            set(recorded_construction) & set(recorded_evaluation)
        ),
    }


def _recompute_seed_rows(
    config: Mapping[str, Any],
    histories: Sequence[Mapping[str, Any]],
    children: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rebuild P/Q/C and regime contrasts without importing the runner."""

    history_meta = {str(row["history_id"]): row for row in histories}
    accumulators: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in children:
        accumulators[(str(row["history_id"]), str(row["candidate"]))].append(
            _safe_float(row.get("squared_reference_error"))
        )
    history_means = {key: _strict_mean(values) for key, values in accumulators.items()}
    result: list[dict[str, Any]] = []
    for configured_seed in config["randomness"]["development_outer_seeds"]:
        seed = int(configured_seed)
        ids = [
            history_id
            for history_id, row in history_meta.items()
            if _safe_int(row.get("seed")) == seed
        ]
        eligible = [
            history_id
            for history_id in ids
            if _safe_bool(history_meta[history_id].get("eligible_R_positive"))
        ]
        eligible_noise_cell_counts = {
            f"{regime}|{permutation}": sum(
                str(history_meta[history_id].get("noise_regime")) == regime
                and str(history_meta[history_id].get("noise_permutation"))
                == permutation
                for history_id in eligible
            )
            for regime, permutation in _noise_cells(config)
        }
        eligible_noise_cell_composition_exact = all(
            count
            == int(config["gates"]["eligible_histories_per_seed_noise_cell_exact"])
            for count in eligible_noise_cell_counts.values()
        )
        candidate_sums = {
            candidate: math.fsum(
                history_means.get((history_id, candidate), float("nan"))
                for history_id in eligible
            )
            for candidate in CANDIDATES
        }
        regime_sums: dict[str, dict[str, float]] = {}
        for regime in ("homogeneous", "heteroscedastic"):
            regime_ids = [
                history_id
                for history_id in eligible
                if str(history_meta[history_id].get("noise_regime")) == regime
            ]
            regime_sums[regime] = {
                candidate: math.fsum(
                    history_means.get((history_id, candidate), float("nan"))
                    for history_id in regime_ids
                )
                for candidate in CANDIDATES
            }

        k4_sum = candidate_sums[K4]
        k4b_sum = candidate_sums[K4B]
        semi_sum_mse = candidate_sums[SEMI_ORACLE]
        point_sum_mse = candidate_sums[POINTWISE]
        point_headroom = k4_sum - point_sum_mse
        semi_headroom = k4_sum - semi_sum_mse
        mse_denominators = [k4_sum, k4b_sum]
        point_denominators = [point_headroom]
        regime_statistics: dict[str, float] = {}
        for regime in ("homogeneous", "heteroscedastic"):
            sums = regime_sums[regime]
            regime_k4 = sums[K4]
            regime_point = regime_k4 - sums[POINTWISE]
            regime_semi = regime_k4 - sums[SEMI_ORACLE]
            mse_denominators.append(regime_k4)
            point_denominators.append(regime_point)
            regime_statistics.update(
                {
                    f"{regime}_pointwise_relative_mse_headroom_vs_k4": (
                        regime_point / regime_k4 if regime_k4 > 0.0 else float("nan")
                    ),
                    f"{regime}_semi_oracle_relative_mse_gain_vs_k4": (
                        regime_semi / regime_k4 if regime_k4 > 0.0 else float("nan")
                    ),
                    f"{regime}_capture_fraction_descriptive": (
                        regime_semi / regime_point
                        if regime_point > 0.0
                        else float("nan")
                    ),
                }
            )
        split_squared = math.fsum(
            _safe_float(
                history_meta[history_id].get("split_a_b_aggregate_scale_distance")
            )
            ** 2
            for history_id in eligible
        )
        per_history_semi_headroom = [
            history_means.get((history_id, K4), float("nan"))
            - history_means.get((history_id, SEMI_ORACLE), float("nan"))
            for history_id in eligible
        ]
        per_history_point_headroom = [
            history_means.get((history_id, K4), float("nan"))
            - history_means.get((history_id, POINTWISE), float("nan"))
            for history_id in eligible
        ]
        per_history_difference = [
            history_means.get((history_id, SEMI_ORACLE), float("nan"))
            - history_means.get((history_id, K4B), float("nan"))
            for history_id in eligible
        ]
        result.append(
            {
                "seed": seed,
                "eligible_histories": len(eligible),
                "total_histories": len(ids),
                "eligible_histories_by_noise_cell": json.dumps(
                    eligible_noise_cell_counts,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "eligible_noise_cell_composition_exact": (
                    eligible_noise_cell_composition_exact
                ),
                "capture_fraction": (
                    semi_headroom / point_headroom
                    if point_headroom > 0.0
                    else float("nan")
                ),
                "pointwise_relative_mse_headroom_vs_k4": (
                    point_headroom / k4_sum if k4_sum > 0.0 else float("nan")
                ),
                "semi_oracle_relative_mse_gain_vs_k4": (
                    semi_headroom / k4_sum if k4_sum > 0.0 else float("nan")
                ),
                "semi_oracle_headroom_mse": _strict_mean(per_history_semi_headroom),
                "pointwise_headroom_mse": _strict_mean(per_history_point_headroom),
                "semi_oracle_minus_k4b_mse": _strict_mean(per_history_difference),
                "k4_two_snapshot_integrated_mse": k4_sum,
                "k4b_two_snapshot_integrated_mse": k4b_sum,
                "semi_oracle_two_snapshot_integrated_mse": semi_sum_mse,
                "pointwise_two_snapshot_integrated_mse": point_sum_mse,
                "semi_oracle_relative_mse_gain_vs_k4b": (
                    (k4b_sum - semi_sum_mse) / k4b_sum
                    if k4b_sum > 0.0
                    else float("nan")
                ),
                "positive_finite_mse_denominators": all(
                    math.isfinite(value) and value > 0.0 for value in mse_denominators
                ),
                "positive_finite_pointwise_headroom_denominators": all(
                    math.isfinite(value) and value > 0.0 for value in point_denominators
                ),
                "construction_split_disagreement_mse_ratio": (
                    split_squared / k4_sum if k4_sum > 0.0 else float("nan")
                ),
                **regime_statistics,
            }
        )
    return result


def _replace_coverage(
    config: Mapping[str, Any], replacements: Sequence[Mapping[str, Any]]
) -> tuple[bool, dict[str, Any]]:
    n = int(config["cohort"]["num_clients"])
    expected = {
        (
            int(seed),
            regime,
            permutation,
            client,
            client,
            "|".join(
                (
                    str(seed),
                    regime,
                    permutation,
                    "aligned",
                    "stationary",
                    "bitflip_x10",
                    "17",
                )
            ),
        )
        for seed in config["randomness"]["development_outer_seeds"]
        for regime, permutation in _noise_cells(config)
        for client in range(n)
    }
    observed = [
        (
            _safe_int(row.get("seed")),
            str(row.get("noise_regime", "")),
            str(row.get("noise_permutation", "")),
            _safe_int(row.get("trial")),
            _safe_int(row.get("replaced_client")),
            str(row.get("history_id", "")),
        )
        for row in replacements
    ]
    bound = 2.0 * float(config["references"]["total_client_influence_cap"]) / n
    structural = (
        len(observed) == len(expected)
        and set(observed) == expected
        and all(_safe_bool(row.get("same_past")) for row in replacements)
        and all(_safe_bool(row.get("same_predictor")) for row in replacements)
        and all(_safe_bool(row.get("same_history_gate")) for row in replacements)
        and all(
            not _safe_bool(row.get("end_to_end_semi_oracle_sensitivity_claimed"))
            for row in replacements
        )
        and all(
            str(row.get("certificate_scope", ""))
            == (
                "evaluation_map_only_with_fixed_anchor_past_history_"
                "radii_and_semi_oracle_predictor"
            )
            for row in replacements
        )
    )
    numeric = all(_all_finite(row, REPLACE_FINITE_FIELDS) for row in replacements)
    calculations = numeric and all(
        math.isclose(_safe_float(row.get("theoretical_bound")), bound, abs_tol=1.0e-9)
        and math.isclose(
            _safe_float(row.get("ratio_to_bound")),
            _safe_float(row.get("observed_difference")) / bound,
            rel_tol=2.0e-7,
            abs_tol=2.0e-9,
        )
        and _safe_bool(row.get("violation"))
        == (_safe_float(row.get("observed_difference")) > bound + 1.0e-6)
        for row in replacements
    )
    return structural and calculations, {
        "expected_rows": len(expected),
        "observed_rows": len(replacements),
        "unique_keys": len(set(observed)),
        "all_slots": sorted(
            {_safe_int(row.get("replaced_client")) for row in replacements}
        ),
        "structural": structural,
        "numeric_and_recomputed": calculations,
        "conditional_bound": bound,
    }


def _decision_status(
    validity_checks: Mapping[str, bool], scientific_checks: Mapping[str, bool]
) -> str:
    if not all(validity_checks.values()):
        return "invalid_or_inconclusive_screen"
    if not all(scientific_checks.values()):
        return "stop_missing_slot_imputation_branch"
    return "authorize_transcript_only_predictor_study"


def _recompute_decision(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    histories: Sequence[Mapping[str, Any]],
    children: Sequence[Mapping[str, Any]],
    replacements: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    *,
    matrix_exact: bool,
    replacement_exact: bool,
) -> dict[str, Any]:
    tcrit = float(config["statistical_analysis"]["t_critical_df11"])
    capture_ci = _ci([float(row["capture_fraction"]) for row in seed_rows], tcrit)
    point_ci = _ci(
        [float(row["pointwise_relative_mse_headroom_vs_k4"]) for row in seed_rows],
        tcrit,
    )
    semi_ci = _ci(
        [float(row["semi_oracle_relative_mse_gain_vs_k4"]) for row in seed_rows],
        tcrit,
    )
    headroom_ci = _ci(
        [float(row["semi_oracle_headroom_mse"]) for row in seed_rows], tcrit
    )
    difference_ci = _ci(
        [float(row["semi_oracle_minus_k4b_mse"]) for row in seed_rows], tcrit
    )
    k4b_gain_ci = _ci(
        [float(row["semi_oracle_relative_mse_gain_vs_k4b"]) for row in seed_rows],
        tcrit,
    )
    homo_q_ci = _ci(
        [
            float(row["homogeneous_semi_oracle_relative_mse_gain_vs_k4"])
            for row in seed_rows
        ],
        tcrit,
    )
    hetero_q_ci = _ci(
        [
            float(row["heteroscedastic_semi_oracle_relative_mse_gain_vs_k4"])
            for row in seed_rows
        ],
        tcrit,
    )
    homo_c_ci = _ci(
        [float(row["homogeneous_capture_fraction_descriptive"]) for row in seed_rows],
        tcrit,
    )
    hetero_c_ci = _ci(
        [
            float(row["heteroscedastic_capture_fraction_descriptive"])
            for row in seed_rows
        ],
        tcrit,
    )
    expected_histories = int(config["statistical_analysis"]["frozen_histories_total"])
    expected_children = (
        expected_histories
        * int(config["nested_monte_carlo"]["evaluation_children"])
        * len(CANDIDATES)
    )
    raw_child_finite = [_all_finite(row, CHILD_FINITE_FIELDS) for row in children]
    declared_child_finite = [_safe_bool(row.get("all_finite")) for row in children]
    finite_fraction = _strict_mean(
        [
            1.0 if raw and declared else 0.0
            for raw, declared in zip(raw_child_finite, declared_child_finite)
        ]
    )
    eligible_fraction = _strict_mean(
        [
            1.0 if _safe_bool(row.get("eligible_R_positive")) else 0.0
            for row in histories
        ]
    )
    declared_per_history_overlap = sum(
        _safe_int(row.get("construction_evaluation_seed_overlap"), 1)
        for row in histories
    )
    construction_seed_registry = [
        seed
        for row in histories
        for seed in _parse_seed_registry(row.get("construction_child_seed_registry"))
    ]
    evaluation_seed_registry = [
        seed
        for row in histories
        for seed in _parse_seed_registry(row.get("evaluation_child_seed_registry"))
    ]
    construction_seed_duplicates = len(construction_seed_registry) - len(
        set(construction_seed_registry)
    )
    evaluation_seed_duplicates = len(evaluation_seed_registry) - len(
        set(evaluation_seed_registry)
    )
    global_seed_overlap = len(
        set(construction_seed_registry) & set(evaluation_seed_registry)
    )
    predictor_variation = 0
    for history in histories:
        history_id = str(history.get("history_id", ""))
        hashes = {
            str(row.get("semi_oracle_predictor_hash", ""))
            for row in children
            if str(row.get("history_id", "")) == history_id
            and str(row.get("candidate", "")) == SEMI_ORACLE
        }
        predictor_variation = max(predictor_variation, max(len(hashes) - 1, 0))
    cap_violations = sum(
        not _safe_bool(row.get("contribution_cap_respected"))
        for row in children
        if str(row.get("candidate", "")) == SEMI_ORACLE
    )
    replacement_violations = sum(
        _safe_bool(row.get("violation")) for row in replacements
    )
    gate_sum_violations = sum(
        _safe_bool(row.get("normalization_by_gate_sum")) for row in children
    )
    denominator_violations = sum(
        _safe_int(row.get("fixed_denominator_n"))
        != int(config["cohort"]["num_clients"])
        or not math.isfinite(_safe_float(row.get("fixed_denominator_formula_error")))
        or _safe_float(row.get("fixed_denominator_formula_error"))
        > float(config["gates"]["fixed_denominator_formula_abs_error_max"])
        for row in children
    )
    k4_manual_formula = _strict_max(
        [_safe_float(row.get("k4_manual_formula_error")) for row in children]
    )
    frozen_k4_comparator = _strict_max(
        [
            _safe_float(row.get("frozen_k4_comparator_error_descriptive"))
            for row in children
        ]
    )
    k4b_reproduction = _strict_max(
        [_safe_float(row.get("k4b_comparator_reproduction_error")) for row in children]
    )
    dominance = _strict_max(
        [
            _safe_float(row.get("pointwise_excess_mse_over_candidate"))
            for row in children
            if str(row.get("candidate", "")) in {K4, K4B, SEMI_ORACLE}
        ]
    )
    split_ratios = [
        float(row["construction_split_disagreement_mse_ratio"]) for row in seed_rows
    ]
    cap_saturation_rate = _strict_mean(
        [
            1.0 if _safe_bool(row.get("semi_oracle_projection_active")) else 0.0
            for row in histories
        ]
    )
    positive_mse = sum(
        _safe_bool(row["positive_finite_mse_denominators"]) for row in seed_rows
    )
    positive_point = sum(
        _safe_bool(row["positive_finite_pointwise_headroom_denominators"])
        for row in seed_rows
    )
    eligible_noise_cell_composition_count = sum(
        _safe_bool(row["eligible_noise_cell_composition_exact"]) for row in seed_rows
    )
    pooled_k4b = math.fsum(
        float(row["k4b_two_snapshot_integrated_mse"]) for row in seed_rows
    )
    pooled_semi = math.fsum(
        float(row["semi_oracle_two_snapshot_integrated_mse"]) for row in seed_rows
    )
    observed = {
        "frozen_histories": len(histories),
        "evaluation_child_rows": len(children),
        "expected_frozen_histories": expected_histories,
        "expected_evaluation_child_rows": expected_children,
        "complete_fraction": min(
            len(histories) / expected_histories,
            len(children) / expected_children,
        ),
        "matrix_exact": bool(matrix_exact),
        "finite_metric_fraction": finite_fraction,
        "construction_evaluation_seed_overlap": global_seed_overlap,
        "declared_per_history_construction_evaluation_seed_overlap": (
            declared_per_history_overlap
        ),
        "construction_child_seed_count": len(construction_seed_registry),
        "evaluation_child_seed_count": len(evaluation_seed_registry),
        "construction_child_seed_duplicates": construction_seed_duplicates,
        "evaluation_child_seed_duplicates": evaluation_seed_duplicates,
        "eligible_R_positive_history_fraction": eligible_fraction,
        "eligible_noise_cell_composition_seed_count": (
            eligible_noise_cell_composition_count
        ),
        "semi_oracle_capture_fraction_seed_ci95": capture_ci,
        "pointwise_relative_mse_headroom_vs_k4_seed_ci95": point_ci,
        "semi_oracle_relative_mse_gain_vs_k4_seed_ci95": semi_ci,
        "semi_oracle_headroom_mse_seed_ci95": headroom_ci,
        "semi_oracle_minus_k4b_mse_seed_ci95": difference_ci,
        "semi_oracle_relative_mse_gain_vs_k4b_seed_ci95": k4b_gain_ci,
        "semi_oracle_relative_mse_gain_vs_k4b_pooled_descriptive": (
            (pooled_k4b - pooled_semi) / pooled_k4b
            if pooled_k4b > 0.0
            else float("nan")
        ),
        "homogeneous_semi_oracle_relative_mse_gain_vs_k4_seed_ci95": homo_q_ci,
        "heteroscedastic_semi_oracle_relative_mse_gain_vs_k4_seed_ci95": hetero_q_ci,
        "homogeneous_capture_fraction_seed_ci95_descriptive": homo_c_ci,
        "heteroscedastic_capture_fraction_seed_ci95_descriptive": hetero_c_ci,
        "positive_finite_mse_denominator_count": positive_mse,
        "positive_finite_pointwise_headroom_denominator_count": positive_point,
        "construction_split_disagreement_mse_ratio_mean": _strict_mean(split_ratios),
        "construction_split_disagreement_mse_ratio_max": _strict_max(split_ratios),
        "semi_oracle_projection_cap_saturation_rate": cap_saturation_rate,
        "predictor_variation_across_evaluation_children": predictor_variation,
        "k4_manual_formula_abs_error_max": k4_manual_formula,
        "frozen_k4_comparator_abs_error_max_descriptive": frozen_k4_comparator,
        "k4b_comparator_reproduction_abs_error_max": k4b_reproduction,
        "gate_sum_normalization_violations": gate_sum_violations,
        "fixed_denominator_violations": denominator_violations,
        "pointwise_oracle_dominance_excess_mse_max": dominance,
        "contribution_cap_violations": cap_violations,
        "replace_one_trials": len(replacements),
        "replace_one_exact_coverage": bool(replacement_exact),
        "replace_one_violations": replacement_violations,
        "replace_one_max_ratio_to_bound": _strict_max(
            [_safe_float(row.get("ratio_to_bound")) for row in replacements]
        ),
        "pointwise_oracle_used_in_construction": sum(
            _safe_bool(row.get("pointwise_oracle_used_in_construction"))
            for row in children
        ),
        "ordinary_l2_error_used_for_gate": False,
        "device": str(manifest.get("device")),
    }
    gates = config["gates"]
    exact_seed_count = int(gates["exact_seed_capture_count"])
    validity = {
        "complete": observed["complete_fraction"]
        >= float(gates["complete_fraction_min"]),
        "matrix_exact": bool(observed["matrix_exact"]),
        "finite": observed["finite_metric_fraction"]
        >= float(gates["finite_metric_fraction_min"]),
        "production_device": observed["device"]
        == str(gates["production_device_required"]),
        "construction_evaluation_independent": (
            global_seed_overlap
            <= int(gates["construction_evaluation_seed_overlap_max"])
            and declared_per_history_overlap
            <= int(gates["construction_evaluation_seed_overlap_max"])
        ),
        "rng_global_unique": (
            len(construction_seed_registry)
            == int(gates["construction_child_seed_exact_count"])
            and len(evaluation_seed_registry)
            == int(gates["evaluation_child_seed_exact_count"])
            and construction_seed_duplicates
            <= int(gates["global_child_seed_duplicate_max"])
            and evaluation_seed_duplicates
            <= int(gates["global_child_seed_duplicate_max"])
        ),
        "predictor_fixed_across_evaluation_children": predictor_variation
        <= float(gates["predictor_variation_across_evaluation_children_max"]),
        "pointwise_not_used_in_construction": observed[
            "pointwise_oracle_used_in_construction"
        ]
        <= int(gates["pointwise_oracle_used_in_construction_max"]),
        "exact_outer_seed_count": len(seed_rows) == exact_seed_count,
        "positive_finite_mse_denominators": (
            not bool(gates["positive_finite_mse_denominators_required"])
            or positive_mse == exact_seed_count
        ),
        "construction_mc_stability": observed[
            "construction_split_disagreement_mse_ratio_max"
        ]
        <= float(gates["construction_split_disagreement_mse_ratio_max"]),
        "contribution_cap": cap_violations
        <= int(gates["contribution_cap_violation_max"]),
        "replace_one": replacement_violations
        <= int(gates["replace_one_violation_max"]),
        "replace_one_complete": (
            len(replacements) == int(gates["replace_one_exact_trials"])
            and replacement_exact
        ),
        "k4_manual_formula": k4_manual_formula
        <= float(gates["k4_manual_formula_abs_error_max"]),
        "k4b_reproduction": k4b_reproduction
        <= float(gates["k4b_comparator_reproduction_abs_error_max"]),
        "no_gate_sum_normalization": gate_sum_violations
        <= int(gates["gate_sum_normalization_violation_max"]),
        "fixed_denominator": denominator_violations
        <= int(gates["fixed_denominator_violation_max"]),
        "pointwise_oracle_dominance": dominance
        <= float(gates["pointwise_oracle_dominance_excess_mse_max"]),
    }
    intervals = (
        capture_ci,
        point_ci,
        semi_ci,
        difference_ci,
        k4b_gain_ci,
        homo_q_ci,
        hetero_q_ci,
    )
    scientific = {
        "eligible_R": eligible_fraction
        >= float(gates["eligible_R_positive_history_fraction_min"]),
        "eligible_noise_cell_composition": (
            eligible_noise_cell_composition_count == exact_seed_count
        ),
        "positive_finite_pointwise_headroom_denominators": (
            not bool(gates["positive_finite_pointwise_headroom_denominators_required"])
            or positive_point == exact_seed_count
        ),
        "exact_scientific_seed_count": all(
            int(interval["n"]) == exact_seed_count for interval in intervals
        ),
        "material_pointwise_headroom_mean": float(point_ci["mean"])
        >= float(gates["pointwise_relative_mse_headroom_vs_k4_mean_min"]),
        "material_pointwise_headroom_ci": float(point_ci["low"])
        > float(
            gates[
                "pointwise_relative_mse_headroom_vs_k4_seed_ci95_low_strictly_greater_than"
            ]
        ),
        "material_semi_oracle_gain_vs_k4_mean": float(semi_ci["mean"])
        >= float(gates["semi_oracle_relative_mse_gain_vs_k4_mean_min"]),
        "material_semi_oracle_gain_vs_k4_ci": float(semi_ci["low"])
        > float(
            gates[
                "semi_oracle_relative_mse_gain_vs_k4_seed_ci95_low_strictly_greater_than"
            ]
        ),
        "capture_mean": float(capture_ci["mean"])
        >= float(gates["semi_oracle_capture_fraction_mean_min"]),
        "capture_ci_low": float(capture_ci["low"])
        > float(gates["semi_oracle_capture_fraction_ci95_low_strictly_greater_than"]),
        "gain_vs_k4b": float(k4b_gain_ci["mean"])
        >= float(gates["semi_oracle_relative_mse_gain_vs_k4b_min"]),
        "gain_vs_k4b_ci": float(k4b_gain_ci["low"])
        >= float(gates["semi_oracle_relative_mse_gain_vs_k4b_seed_ci95_low_min"]),
        "ci_vs_k4b": float(difference_ci["high"])
        <= float(gates["semi_oracle_minus_k4b_mse_seed_ci95_high_max"]),
        "homogeneous_gain_vs_k4_ci": float(homo_q_ci["low"])
        > float(
            gates[
                "homogeneous_semi_oracle_relative_mse_gain_vs_k4_seed_ci95_low_strictly_greater_than"
            ]
        ),
        "heteroscedastic_gain_vs_k4_ci": float(hetero_q_ci["low"])
        > float(
            gates[
                "heteroscedastic_semi_oracle_relative_mse_gain_vs_k4_seed_ci95_low_strictly_greater_than"
            ]
        ),
    }
    classification = config["statistical_analysis"]["gate_classification"]
    if tuple(classification["validity"]) != tuple(validity):
        raise RuntimeError("Configured validity-gate order differs from raw audit")
    if tuple(classification["scientific"]) != tuple(scientific):
        raise RuntimeError("Configured scientific-gate order differs from raw audit")
    validity_pass = all(validity.values())
    scientific_pass = all(scientific.values())
    checks = {**validity, **scientific}
    return {
        "decision": _decision_status(validity, scientific),
        "all_gates_pass": validity_pass and scientific_pass,
        "checks": checks,
        "validity_checks": validity,
        "scientific_checks": scientific,
        "validity_pass": validity_pass,
        "scientific_checks_pass": scientific_pass if validity_pass else None,
        "observed": observed,
        "pointwise_oracle_role": (
            "nondeployable_decision_benchmark_headroom_denominator_only"
        ),
        "semi_oracle_role": (
            "dgp_specific_finite_monte_carlo_mechanism_conditioned_on_observable_"
            "past_latent_clean_current_state_simulator_labels_and_configured_"
            "current_noise_attack_law"
        ),
        "pass_does_not_authorize_promotion_or_holdout": True,
        "failure_interpretation": (
            "no_scientific_inference_when_invalid_or_inconclusive_otherwise_"
            "the_preregistered_finite_mc_privileged_mechanism_fails_required_"
            "gates_for_this_dgp_and_stops_this_fixed_denominator_branch"
        ),
        "success_interpretation": (
            "headroom_exists_but_a_separate_observable_past_only_predictor_"
            "must_still_be_constructed_and_tested"
        ),
        "universal_impossibility_claimed": False,
        "exact_conditional_optimum_failure_claimed": False,
        "arbitrary_adaptive_attack_claimed": False,
        "loss_used_for_all_scientific_gates": "squared_l2_reference_error",
        "holdout_opened": False,
    }


def _same_scalar(left: Any, right: Any) -> bool:
    if left is None or right is None:
        if left is None and right is None:
            return True
        other = right if left is None else left
        return isinstance(other, float) and not math.isfinite(other)
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        first, second = float(left), float(right)
        if math.isnan(first) and math.isnan(second):
            return True
        return math.isclose(first, second, rel_tol=2.0e-7, abs_tol=2.0e-9)
    return left == right


def _compare_tree(
    left: Any,
    right: Any,
    *,
    path: str = "",
    differences: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if differences is None:
        differences = []
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                differences.append(
                    {
                        "path": child,
                        "recomputed": left.get(key),
                        "reported": right.get(key),
                    }
                )
            else:
                _compare_tree(
                    left[key], right[key], path=child, differences=differences
                )
        return differences
    if not _same_scalar(left, right):
        differences.append({"path": path, "recomputed": left, "reported": right})
    return differences


def _compare_seed_rows(
    recomputed: Sequence[Mapping[str, Any]],
    reported: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reported_by_seed = {_safe_int(row.get("seed")): row for row in reported}
    differences: list[dict[str, Any]] = []
    for row in recomputed:
        seed = int(row["seed"])
        other = reported_by_seed.get(seed)
        if other is None:
            differences.append({"seed": seed, "field": "row", "reported": None})
            continue
        for field, expected in row.items():
            raw = other.get(field)
            if isinstance(expected, bool):
                observed: Any = _safe_bool(raw)
            elif isinstance(expected, int):
                observed = _safe_int(raw)
            elif isinstance(expected, str):
                observed = str(raw)
            else:
                observed = _safe_float(raw)
            if not _same_scalar(expected, observed):
                differences.append(
                    {
                        "seed": seed,
                        "field": field,
                        "recomputed": expected,
                        "reported": observed,
                    }
                )
    if len(recomputed) != len(reported):
        differences.append(
            {
                "field": "row_count",
                "recomputed": len(recomputed),
                "reported": len(reported),
            }
        )
    return differences


def _verify_lock(
    config: Mapping[str, Any],
    config_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if config_path.resolve() != DEFAULT_CONFIG.resolve():
        raise RuntimeError("Post-run audit accepts only the preregistered config")
    declared = Path(str(config["preregistration_lock"]["path"]))
    lock_path = declared if declared.is_absolute() else ROOT / declared
    if lock_path.resolve() != DEFAULT_LOCK.resolve() or not lock_path.is_file():
        raise RuntimeError(f"Unexpected or missing K4c-CH lock: {lock_path}")
    lock = _read_json(lock_path)
    expected_schema = {
        "schema_version",
        "campaign_id",
        "locked_files",
        "dependencies",
        "lock_file_self_hash_embedded",
        "publication_requirement",
    }
    if set(lock) != expected_schema or lock.get("schema_version") != 1:
        raise RuntimeError("K4c-CH lock schema mismatch")
    if lock.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError("K4c-CH lock campaign mismatch")
    policy_exact = (
        lock.get("lock_file_self_hash_embedded") is False
        and lock.get("publication_requirement")
        == "publish_this_lock_file_sha256_in_research_log_or_chat_before_run"
    )
    if set(lock.get("locked_files", {})) != LOCKED_PATHS:
        raise RuntimeError("K4c-CH locked-file registry mismatch")
    if set(lock.get("dependencies", {})) != DEPENDENCY_PATHS:
        raise RuntimeError("K4c-CH dependency registry mismatch")
    hashes_match = True
    mismatches: list[str] = []
    for relative, expected_hash in {
        **lock["locked_files"],
        **lock["dependencies"],
    }.items():
        path = ROOT / relative
        if not path.is_file() or _sha256(path) != str(expected_hash):
            hashes_match = False
            mismatches.append(relative)
    manifest_lock = manifest.get("preregistration_lock", {})
    publication = manifest.get("lock_publication_attestation", {})
    lock_sha = _sha256(lock_path)
    manifest_matches = (
        isinstance(manifest_lock, Mapping)
        and manifest_lock.get("verified") is True
        and Path(str(manifest_lock.get("path", ""))).resolve() == lock_path.resolve()
        and manifest_lock.get("sha256") == lock_sha
        and _safe_int(manifest_lock.get("locked_files")) == len(LOCKED_PATHS)
        and _safe_int(manifest_lock.get("dependencies")) == len(DEPENDENCY_PATHS)
        and manifest.get("config_sha256") == _sha256(config_path)
        and isinstance(publication, Mapping)
        and publication.get("procedural_external_publication_attested") is True
        and publication.get("published_lock_sha256") == lock_sha
        and publication.get("machine_verifies_external_log_itself") is False
    )
    return {
        "passed": hashes_match and manifest_matches and policy_exact,
        "path": str(lock_path.resolve()),
        "sha256": lock_sha,
        "hashes_match": hashes_match,
        "manifest_matches": manifest_matches,
        "publication_policy_exact": policy_exact,
        "mismatches": mismatches,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def audit_results(
    config_path: Path,
    results_dir: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Audit a completed screen; completion is checked before every other read."""

    manifest = _guard_completed(results_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError("Unexpected K4c-CH configuration")
    lock_audit = _verify_lock(config, config_path, manifest)

    histories = _read_csv(results_dir / "frozen_history_rows.csv")
    children = _read_csv(results_dir / "evaluation_child_rows.csv")
    reported_seeds = _read_csv(results_dir / "seed_summary.csv")
    replacements = _read_csv(results_dir / "replace_one_audit.csv")
    reported_decision = _read_json(results_dir / "decision.json")

    matrix_exact, matrix_details = _matrix_audit(config, histories, children)
    replacement_exact, replacement_details = _replace_coverage(config, replacements)
    recomputed_seeds = _recompute_seed_rows(config, histories, children)
    recomputed_decision = _recompute_decision(
        config,
        manifest,
        histories,
        children,
        replacements,
        recomputed_seeds,
        matrix_exact=matrix_exact,
        replacement_exact=replacement_exact,
    )
    seed_differences = _compare_seed_rows(recomputed_seeds, reported_seeds)
    decision_differences = _compare_tree(recomputed_decision, reported_decision)

    child_flags_match = all(
        _safe_bool(row.get("all_finite")) == _all_finite(row, CHILD_FINITE_FIELDS)
        for row in children
    )
    history_finite = all(_all_finite(row, HISTORY_FINITE_FIELDS) for row in histories)
    replacement_finite = all(
        _all_finite(row, REPLACE_FINITE_FIELDS) for row in replacements
    )
    float32_eps = 1.1920928955078125e-7
    history_scope_exact = all(
        not _safe_bool(row.get("semi_oracle_uses_evaluation_target"))
        and not _safe_bool(row.get("semi_oracle_uses_evaluation_noise"))
        and not _safe_bool(row.get("semi_oracle_uses_k4b_feedback"))
        and _safe_bool(row.get("semi_oracle_conditions_on_latent_clean_current_state"))
        and _safe_bool(row.get("semi_oracle_uses_simulator_honest_byzantine_labels"))
        and _safe_bool(row.get("semi_oracle_knows_configured_current_noise_attack_dgp"))
        and not _safe_bool(row.get("arbitrary_adaptive_byzantine_behavior_supported"))
        and not _safe_bool(row.get("semi_oracle_is_observable_past_measurable"))
        and _safe_bool(row.get("projection_applied_once_after_expectations"))
        and not _safe_bool(row.get("pointwise_oracles_averaged"))
        and _safe_bool(row.get("eligible_R_positive"))
        == (_safe_float(row.get("missing_slot_mass")) > 64.0 * float32_eps)
        for row in histories
    )
    child_scope_exact = all(
        _safe_bool(row.get("semi_oracle_predictor_fixed_across_evaluation_children"))
        and _safe_bool(row.get("pointwise_oracle"))
        == (str(row.get("candidate")) == POINTWISE)
        and _safe_bool(row.get("construction_stream_read"))
        == (str(row.get("candidate")) == SEMI_ORACLE)
        and _safe_bool(row.get("pointwise_oracle_used_in_construction")) is False
        and _safe_bool(row.get("conditions_on_latent_clean_current_state"))
        == (str(row.get("candidate")) == SEMI_ORACLE)
        for row in children
    )
    scope_exact = (
        manifest.get("device") == "mps"
        and manifest.get("dtype") == "torch.float32"
        and manifest.get("development_only") is True
        and manifest.get("holdout_opened") is False
        and manifest.get("observable_past_only_predictor_constructed") is False
        and manifest.get("pointwise_oracle_role")
        == "nondeployable_decision_benchmark_headroom_denominator_only"
        and manifest.get("semi_oracle_is_finite_monte_carlo_approximation") is True
    )
    manifest_decision_exact = (
        manifest.get("development_decision") == recomputed_decision["decision"]
        and manifest.get("all_gates_pass") == recomputed_decision["all_gates_pass"]
        and _safe_int(manifest.get("frozen_histories")) == len(histories)
        and _safe_int(manifest.get("evaluation_child_rows")) == len(children)
    )
    checks = {
        "immutable_lock_and_manifest_hashes": bool(lock_audit["passed"]),
        "mps_development_scope": scope_exact,
        "exact_matrix_and_rng_contract": matrix_exact,
        "raw_numeric_finiteness_and_child_flags": (
            child_flags_match and history_finite and replacement_finite
        ),
        "raw_scope_and_role_flags": history_scope_exact and child_scope_exact,
        "seed_summary_reproduced": not seed_differences,
        "decision_reproduced": not decision_differences,
        "replace_one_balanced_and_recomputed": replacement_exact,
        "manifest_decision_matches_raw": manifest_decision_exact,
    }
    result = {
        "audit_status": "passed" if all(checks.values()) else "failed",
        "all_checks_pass": all(checks.values()),
        "checks": checks,
        "lock_audit": lock_audit,
        "matrix_audit": matrix_details,
        "replace_one_audit": replacement_details,
        "seed_summary_differences": seed_differences,
        "decision_differences": decision_differences,
        "recomputed_seed_summary": recomputed_seeds,
        "recomputed_decision": recomputed_decision,
        "scientific_interpretation_allowed": (
            all(checks.values())
            and recomputed_decision["decision"] != "invalid_or_inconclusive_screen"
        ),
        "holdout_opened": False,
        "pointwise_role": (
            "nondeployable_decision_benchmark_headroom_denominator_only"
        ),
        "finite_mc_exact_optimum_claimed": False,
    }
    if output_path is not None:
        _atomic_json(output_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = args.config if args.config.is_absolute() else ROOT / args.config
    results = args.results if args.results.is_absolute() else ROOT / args.results
    output = args.output
    if output is None:
        output = results / "independent_postrun_audit.json"
    elif not output.is_absolute():
        output = ROOT / output
    result = audit_results(config, results, output)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0 if result["all_checks_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
