#!/usr/bin/env python3
"""Analyze the pre-registered n=25 end-to-end DT-LDP-FAR stress screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import yaml

EXPECTED_RUNS = 24
EXPECTED_SEEDS = 3


def _value(value: object, default: float) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return float(default)


def _numeric(rounds: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(row[key])
        for row in rounds
        if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))
    ]


def _median(rounds: list[dict[str, Any]], key: str) -> float | None:
    values = _numeric(rounds, key)
    return median(values) if values else None


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _nearest(path: Path, filename: str) -> Path:
    for parent in path.parents:
        candidate = parent / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {filename} above {path}")


def _load_run(metrics_path: Path) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(
        _nearest(metrics_path, "resolved_config.yaml").read_text(encoding="utf-8")
    )
    axes = config["reproduction"]["axes"]
    all_rounds = payload.get("rounds", [])
    # Round one is excluded because delayed weights are necessarily uniform.
    rounds = [
        row for row in all_rounds if int(row.get("round", row.get("round_num", 0))) > 1
    ]
    if len(all_rounds) != int(config["training"]["num_rounds"]):
        raise ValueError(f"Incomplete run: {metrics_path}")
    if not rounds:
        raise ValueError(f"No post-initialization round: {metrics_path}")
    method = str(axes["method"])
    current = method.startswith("dp_far_current")
    if current:
        keys = {
            "score_span": "far_score_span",
            "logit_span": "far_logit_range",
            "saturation": "far_score_saturation_rate",
            "max_weight": "max_client_weight",
            "concentration": "far_noise_amplification_vs_uniform",
            "raw_corr": "far_weight_effective_noise_corr_oracle",
            "normalized_corr": "far_weight_normalized_effective_noise_corr_oracle",
            "fixed_noise_error": "far_fixed_weight_fresh_noise_sq_error_oracle",
            "total_noise_error": "far_total_fresh_noise_sq_error_oracle",
            "reweight_error": "far_reweighting_component_sq_norm_oracle",
        }
    else:
        keys = {
            "score_span": "dtldp_current_score_span",
            "logit_span": "dtldp_current_logit_span",
            "saturation": "dtldp_current_score_saturation_rate",
            "max_weight": "dtldp_current_max_client_weight",
            "concentration": "dtldp_current_noise_amplification_vs_uniform",
            "raw_corr": "dtldp_current_weight_effective_noise_corr_oracle",
            "normalized_corr": (
                "dtldp_current_weight_normalized_effective_noise_corr_oracle"
            ),
            "fixed_noise_error": (
                "dtldp_current_fixed_weight_fresh_noise_sq_error_oracle"
            ),
            "delayed_noise_error": (
                "dtldp_delayed_fixed_weight_fresh_noise_sq_error_oracle"
            ),
            "current_delay_ratio": (
                "dtldp_current_vs_delayed_fresh_noise_sq_error_ratio_oracle"
            ),
        }
    final = all_rounds[-1]
    raw_noise_key = (
        "privacy_realised_noise_norm_mean_oracle"
        if current
        else "dtldp_realised_noise_norm_mean_oracle"
    )
    row = {
        "experiment": str(axes["scenario"]),
        "experiment_id": str(config["reproduction"].get("experiment_id", "")),
        "method": method,
        "variant": "current" if current else "delayed",
        "privacy": str(axes["privacy"]),
        "geometry": str(axes["geometry"]),
        "tilt": str(axes["tilt"]),
        "partition_seed": int(axes["partition_seed"]),
        "training_seed": int(axes["training_seed"]),
        "pair_key": str(config["reproduction"]["randomness_pair_key"]),
        "score_subspace_mode": str(
            config["training"]["algo_config"].get("score_subspace_mode", "full")
        ),
        "score_subspace_dimension": int(
            _median(
                rounds,
                (
                    "far_score_subspace_dimension"
                    if current
                    else "dtldp_score_subspace_dimension"
                ),
            )
            or 0
        ),
        "score_span_median": _median(rounds, keys["score_span"]),
        "logit_span_median": _median(rounds, keys["logit_span"]),
        "saturation_p90": _quantile(_numeric(rounds, keys["saturation"]), 0.90),
        "max_weight_median": _median(rounds, keys["max_weight"]),
        "concentration_median": _median(rounds, keys["concentration"]),
        "effective_noise_corr_median": _median(rounds, keys["raw_corr"]),
        "normalized_effective_noise_corr_median": _median(
            rounds, keys["normalized_corr"]
        ),
        "fixed_weight_noise_error_median": _median(rounds, keys["fixed_noise_error"]),
        "total_noise_error_median": (
            _median(rounds, keys["total_noise_error"]) if current else None
        ),
        "reweighting_error_median": (
            _median(rounds, keys["reweight_error"]) if current else None
        ),
        "delayed_noise_error_median": (
            _median(rounds, keys["delayed_noise_error"]) if not current else None
        ),
        "current_delay_noise_ratio_median": (
            _median(rounds, keys["current_delay_ratio"]) if not current else None
        ),
        "epsilon_final": final.get("privacy_epsilon_max"),
        "test_accuracy_final": final.get("test_accuracy"),
        "worst20_final": final.get("worst20_accuracy_pct"),
        "gap_final": final.get("best20_worst20_gap_pct"),
        "raw_noise_sequence": _numeric(all_rounds, raw_noise_key),
        "metrics_path": str(metrics_path),
    }
    return row


def _pair_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        key = (
            run["privacy"],
            run["geometry"],
            run["tilt"],
            run["partition_seed"],
            run["training_seed"],
        )
        grouped[key][run["variant"]] = run
    paired: list[dict[str, Any]] = []
    for key, variants in sorted(grouped.items()):
        if set(variants) != {"current", "delayed"}:
            raise ValueError(
                f"Incomplete current/delayed pair for {key}: {set(variants)}"
            )
        current, delayed = variants["current"], variants["delayed"]
        raw_sequences_match = len(current["raw_noise_sequence"]) == len(
            delayed["raw_noise_sequence"]
        ) and all(
            math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
            for left, right in zip(
                current["raw_noise_sequence"], delayed["raw_noise_sequence"]
            )
        )
        checks = {
            "pair_key": current["pair_key"] == delayed["pair_key"],
            "noise_sequence": raw_sequences_match,
            "logit_span": _value(current["logit_span_median"], 0.0) >= 1.0,
            "positive_raw_corr": _value(
                current["effective_noise_corr_median"], -math.inf
            )
            > 0.0,
            "positive_normalized_corr": _value(
                current["normalized_effective_noise_corr_median"], -math.inf
            )
            > 0.0,
            "weight_concentration": _value(current["concentration_median"], 0.0)
            >= 1.05,
            "limited_saturation": _value(current["saturation_p90"], 1.0) <= 0.25,
        }
        paired.append(
            {
                "privacy": key[0],
                "geometry": key[1],
                "tilt": key[2],
                "partition_seed": key[3],
                "training_seed": key[4],
                "score_subspace_mode": current["score_subspace_mode"],
                "score_subspace_dimension": current["score_subspace_dimension"],
                "pair_key_matches": checks["pair_key"],
                "noise_sequence_matches": checks["noise_sequence"],
                "logit_span_median": current["logit_span_median"],
                "effective_noise_corr_median": current["effective_noise_corr_median"],
                "normalized_effective_noise_corr_median": current[
                    "normalized_effective_noise_corr_median"
                ],
                "concentration_median": current["concentration_median"],
                "saturation_p90": current["saturation_p90"],
                "within_delayed_run_current_delay_noise_ratio_median": delayed[
                    "current_delay_noise_ratio_median"
                ],
                "delta_test_delayed_minus_current": float(
                    delayed["test_accuracy_final"]
                )
                - float(current["test_accuracy_final"]),
                "core_gate_pass": all(checks.values()),
                "failed_checks": ";".join(
                    name for name, passed in checks.items() if not passed
                ),
            }
        )
    return paired


def _candidate_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[(row["privacy"], row["geometry"], row["tilt"])].append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        if len(rows) != EXPECTED_SEEDS:
            raise ValueError(f"Candidate {key} has {len(rows)}/{EXPECTED_SEEDS} seeds")

        def avg(field: str) -> float | None:
            values = [
                float(row[field])
                for row in rows
                if isinstance(row.get(field), (int, float))
                and math.isfinite(float(row[field]))
            ]
            return mean(values) if values else None

        output.append(
            {
                "privacy": key[0],
                "geometry": key[1],
                "tilt": key[2],
                "score_subspace_mode": rows[0]["score_subspace_mode"],
                "score_subspace_dimension": rows[0]["score_subspace_dimension"],
                "seeds_passing_core_gate": sum(
                    bool(row["core_gate_pass"]) for row in rows
                ),
                "all_seeds_pass": all(row["core_gate_pass"] for row in rows),
                "mean_logit_span": avg("logit_span_median"),
                "mean_effective_noise_corr": avg("effective_noise_corr_median"),
                "mean_normalized_effective_noise_corr": avg(
                    "normalized_effective_noise_corr_median"
                ),
                "mean_concentration": avg("concentration_median"),
                "mean_current_delay_noise_ratio": avg(
                    "within_delayed_run_current_delay_noise_ratio_median"
                ),
                "mean_delta_test_delayed_minus_current": avg(
                    "delta_test_delayed_minus_current"
                ),
                "sd_delta_test_delayed_minus_current": stdev(
                    float(row["delta_test_delayed_minus_current"]) for row in rows
                ),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "campaign_root",
        nargs="?",
        type=Path,
        default=Path(
            "results/dt_ldp_far/decisive/"
            "dt_ldp_far_decisive_stage5_end_to_end_stress_discovery_n25_v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/analysis/dt_ldp_far_n25_stage5_stress"),
    )
    args = parser.parse_args()
    metric_paths = sorted(args.campaign_root.glob("**/metrics.json"))
    if len(metric_paths) != EXPECTED_RUNS:
        raise SystemExit(
            f"Stage-5 screen incomplete: {len(metric_paths)}/{EXPECTED_RUNS} runs"
        )
    runs = [_load_run(path) for path in metric_paths]
    pairs = _pair_rows(runs)
    candidates = _candidate_rows(pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    serializable_runs = [
        {key: value for key, value in row.items() if key != "raw_noise_sequence"}
        for row in runs
    ]
    _write_csv(args.output_dir / "run_metrics.csv", serializable_runs)
    _write_csv(args.output_dir / "paired_seed_gate.csv", pairs)
    _write_csv(args.output_dir / "candidate_summary.csv", candidates)
    promoted = [row for row in candidates if row["all_seeds_pass"]]
    summary = {
        "schema_version": 1,
        "status": "promoted_to_confirmation" if promoted else "no_multi_seed_gate",
        "privacy_claim": "none; simulation-only counterfactual oracle",
        "runs": len(runs),
        "pairs": len(pairs),
        "candidate_cells": len(candidates),
        "multi_seed_promoted_cells": len(promoted),
        "promoted_cells": promoted,
        "selection_uses_accuracy": False,
        "gate": {
            "burn_in_rounds": 1,
            "median_logit_span_min": 1.0,
            "median_current_effective_noise_corr_strictly_positive": True,
            "median_current_normalized_effective_noise_corr_strictly_positive": True,
            "median_concentration_min": 1.05,
            "score_saturation_p90_max": 0.25,
            "all_three_seeds_required": True,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if promoted else 2


if __name__ == "__main__":
    raise SystemExit(main())
