#!/usr/bin/env python3
"""Apply the pre-registered Stage-8 gate to frozen-update score trials."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml

METRICS = (
    "score_clean_geometry_corr",
    "score_noise_scale_corr",
    "weight_l1_to_clean_target",
    "honest_outlier_top5_recall",
    "score_span",
    "weight_concentration",
)


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _aggregate_trials(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    keys = ("seed", "honest_outliers", "noise_permutation", "reference", "score_mode")
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)

    result: list[dict[str, Any]] = []
    for key, trials in sorted(groups.items()):
        item: dict[str, Any] = dict(zip(keys, key))
        item["seed"] = int(item["seed"])
        item["honest_outliers"] = item["honest_outliers"].lower() == "true"
        item["evaluation_draws"] = len(trials)
        for metric in METRICS:
            values = [value for row in trials if (value := _float(row.get(metric))) is not None]
            item[f"{metric}_median"] = median(values) if values else None
            item[f"{metric}_mean"] = mean(values) if values else None
        result.append(item)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _paired(
    summaries: list[dict[str, Any]], baseline_mode: str, candidate_mode: str
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summaries:
        key = (
            row["seed"],
            row["honest_outliers"],
            row["noise_permutation"],
            row["reference"],
        )
        groups[key][row["score_mode"]] = row

    result: list[dict[str, Any]] = []
    for key, modes in sorted(groups.items()):
        if baseline_mode not in modes or candidate_mode not in modes:
            continue
        baseline = modes[baseline_mode]
        candidate = modes[candidate_mode]
        corr_delta = (
            candidate["score_clean_geometry_corr_median"]
            - baseline["score_clean_geometry_corr_median"]
        )
        weight_delta = (
            candidate["weight_l1_to_clean_target_median"]
            - baseline["weight_l1_to_clean_target_median"]
        )
        baseline_recall = baseline["honest_outlier_top5_recall_median"]
        candidate_recall = candidate["honest_outlier_top5_recall_median"]
        recall_delta = (
            candidate_recall - baseline_recall
            if baseline_recall is not None and candidate_recall is not None
            else None
        )
        corr_improved = corr_delta > 0.0
        weight_improved = weight_delta < 0.0
        recall_not_worse = recall_delta is None or recall_delta >= -1e-12
        result.append(
            {
                "seed": key[0],
                "honest_outliers": key[1],
                "noise_permutation": key[2],
                "reference": key[3],
                "delta_clean_correlation": corr_delta,
                "delta_weight_l1": weight_delta,
                "delta_top5_recall": recall_delta,
                "clean_correlation_improved": corr_improved,
                "weight_l1_improved": weight_improved,
                "top5_recall_not_worse": recall_not_worse,
                "all_three": corr_improved and weight_improved and recall_not_worse,
            }
        )
    return result


def _fraction(rows: list[dict[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/dt_ldp_far/decisive/stage8_frozen_debiased_distance"),
    )
    parser.add_argument(
        "--gate-config",
        type=Path,
        default=Path("configs/dt_ldp_far/stage8_frozen_debiased_distance_gate.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/analysis/dt_ldp_far_stage8_frozen_debiased_distance"),
    )
    args = parser.parse_args()

    gate = yaml.safe_load(args.gate_config.read_text(encoding="utf-8"))
    with (args.results_dir / "frozen_noise_trials.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        trials = list(csv.DictReader(handle))
    summaries = _aggregate_trials(trials)
    paired = _paired(summaries, gate["baseline_mode"], gate["candidate_mode"])
    outlier_pairs = [row for row in paired if row["honest_outliers"]]

    candidate_null = [
        row
        for row in summaries
        if not row["honest_outliers"] and row["score_mode"] == gate["candidate_mode"]
    ]
    null_abs_corr = [
        abs(row["score_noise_scale_corr_median"])
        for row in candidate_null
        if row["score_noise_scale_corr_median"] is not None
    ]
    null_concentration = [
        row["weight_concentration_mean"] for row in candidate_null
    ]
    null_span = [row["score_span_mean"] for row in candidate_null]

    observed = {
        "honest_outlier_pair_count": len(outlier_pairs),
        "fraction_clean_correlation_improved": _fraction(
            outlier_pairs, "clean_correlation_improved"
        ),
        "fraction_weight_l1_improved": _fraction(outlier_pairs, "weight_l1_improved"),
        "fraction_top5_recall_not_worse": _fraction(
            outlier_pairs, "top5_recall_not_worse"
        ),
        "fraction_all_three": _fraction(outlier_pairs, "all_three"),
        "noise_only_median_absolute_noise_scale_correlation": median(null_abs_corr),
        "noise_only_mean_weight_concentration": mean(null_concentration),
        "noise_only_mean_score_span": mean(null_span),
    }
    honest_thresholds = gate["honest_outlier_gate"]
    null_thresholds = gate["noise_only_gate"]
    checks = {
        "clean_correlation": observed["fraction_clean_correlation_improved"]
        >= honest_thresholds["min_fraction_clean_correlation_improved"],
        "weight_l1": observed["fraction_weight_l1_improved"]
        >= honest_thresholds["min_fraction_weight_l1_improved"],
        "top5_recall": observed["fraction_top5_recall_not_worse"]
        >= honest_thresholds["min_fraction_top5_recall_not_worse"],
        "joint_outlier_recovery": observed["fraction_all_three"]
        >= honest_thresholds["min_fraction_all_three"],
        "null_noise_independence": observed[
            "noise_only_median_absolute_noise_scale_correlation"
        ]
        <= null_thresholds["max_median_absolute_noise_scale_correlation"],
        "null_weight_concentration": observed["noise_only_mean_weight_concentration"]
        <= null_thresholds["max_mean_weight_concentration"],
        "null_score_non_degeneracy": observed["noise_only_mean_score_span"]
        >= null_thresholds["min_mean_score_span"],
    }
    status = {
        "status": "completed",
        "baseline_mode": gate["baseline_mode"],
        "candidate_mode": gate["candidate_mode"],
        "num_trials": len(trials),
        "num_stratum_summaries": len(summaries),
        "observed": observed,
        "checks": checks,
        "promote_to_end_to_end": all(checks.values()),
        "gate_config": str(args.gate_config),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "stratum_summary.csv", summaries)
    _write_csv(args.output_dir / "paired_candidate_vs_raw.csv", paired)
    (args.output_dir / "gate_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
