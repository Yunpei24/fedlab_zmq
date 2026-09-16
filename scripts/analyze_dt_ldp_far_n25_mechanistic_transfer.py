#!/usr/bin/env python3
"""Gate the n=25 end-to-end transfer of the paired mechanism stress.

The input campaign is explicitly an oracle diagnostic, not a publishable LDP
transcript.  It tests whether a synthetic mechanism effect transfers to real
Fashion-MNIST client updates before any 80/120-round campaign is launched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import yaml


def _nearest_config(metrics_path: Path) -> Path:
    for parent in metrics_path.parents:
        candidate = parent / "resolved_config.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No resolved_config.yaml above {metrics_path}")


def _numeric(rounds: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rounds:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


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


def _load_run(metrics_path: Path) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(_nearest_config(metrics_path).read_text(encoding="utf-8"))
    axes = config["reproduction"]["axes"]
    rounds = [
        row
        for row in metrics.get("rounds", [])
        if int(row.get("round", row.get("round_num", 0))) > 0
    ]
    if not rounds:
        raise ValueError(f"No informative round in {metrics_path}")
    method = str(axes["method"])
    current = method.startswith("dp_far_current")
    prefix = "far" if current else "dtldp"
    span_key = "far_score_span" if current else "dtldp_current_score_span"
    logit_key = "far_logit_range" if current else "dtldp_current_logit_span"
    saturation_key = (
        "far_score_saturation_rate"
        if current
        else "dtldp_current_score_saturation_rate"
    )
    concentration_key = (
        "far_noise_amplification_vs_uniform"
        if current
        else "dtldp_noise_amplification_vs_uniform"
    )
    correlation_key = (
        "far_weight_dp_noise_corr_oracle"
        if current
        else "dtldp_delayed_weight_noise_corr_oracle"
    )
    final = rounds[-1]
    noise_sequence = _numeric(
        rounds,
        (
            "privacy_realised_noise_norm_mean_oracle"
            if current
            else "dtldp_realised_noise_norm_mean_oracle"
        ),
    )
    return {
        "method": method,
        "variant": "current" if current else "delayed",
        "geometry": str(axes["geometry"]),
        "tilt": str(axes["tilt"]),
        "partition_seed": int(axes["partition_seed"]),
        "training_seed": int(axes["training_seed"]),
        "randomness_pair_key": str(config["reproduction"]["randomness_pair_key"]),
        "alpha": float(
            config["training"]["algo_config"].get(
                "far_alpha", config["training"]["algo_config"].get("tilt_tau", 0.0)
            )
        ),
        "score_span_median": _median(rounds, span_key),
        "logit_span_median": _median(rounds, logit_key),
        "saturation_p90": _quantile(_numeric(rounds, saturation_key), 0.90),
        "max_weight_median": _median(rounds, "max_client_weight"),
        "concentration_median": _median(rounds, concentration_key),
        "weight_noise_corr_median": _median(rounds, correlation_key),
        "test_accuracy_final": float(final["test_accuracy"]),
        "worst20_final": float(final["worst20_accuracy_pct"]),
        "gap_final": float(final["best20_worst20_gap_pct"]),
        "noise_sequence": noise_sequence,
        "metric_path": str(metrics_path),
        "metric_prefix": prefix,
    }


def _close_sequences(left: list[float], right: list[float]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
        for a, b in zip(left, right)
    )


def _value(value: object, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) else float(default)


def _pair_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for run in runs:
        if run["tilt"] == "uniform":
            continue
        key = (
            run["geometry"],
            run["tilt"],
            run["partition_seed"],
            run["training_seed"],
        )
        groups.setdefault(key, {})[run["variant"]] = run

    rows: list[dict[str, Any]] = []
    for key, variants in sorted(groups.items()):
        if set(variants) != {"current", "delayed"}:
            raise ValueError(f"Incomplete current/delayed pair for {key}: {set(variants)}")
        current = variants["current"]
        delayed = variants["delayed"]
        pair_key_matches = current["randomness_pair_key"] == delayed["randomness_pair_key"]
        noise_draws_match = _close_sequences(
            current["noise_sequence"], delayed["noise_sequence"]
        )
        checks = {
            "strict_pair_key": pair_key_matches,
            "strict_noise_draws": noise_draws_match,
            "score_span": _value(current["score_span_median"], 0.0) >= 0.15,
            "logit_span": _value(current["logit_span_median"], 0.0) >= 1.0,
            "limited_saturation": _value(current["saturation_p90"], 1.0) <= 0.25,
            "maximum_weight": _value(current["max_weight_median"], 0.0)
            >= 1.25 / 25,
            "weight_concentration": _value(current["concentration_median"], 0.0)
            >= 1.03,
            "current_self_selection": _value(
                current["weight_noise_corr_median"], 0.0
            )
            >= 0.10,
            "delayed_decoupling": abs(
                _value(delayed["weight_noise_corr_median"], 0.0)
            )
            <= 0.15,
        }
        failed = [name for name, passed in checks.items() if not passed]
        rows.append(
            {
                "geometry": key[0],
                "tilt": key[1],
                "partition_seed": key[2],
                "training_seed": key[3],
                "alpha": current["alpha"],
                "pair_key_matches": pair_key_matches,
                "noise_draws_match": noise_draws_match,
                "current_score_span_median": current["score_span_median"],
                "current_logit_span_median": current["logit_span_median"],
                "current_saturation_p90": current["saturation_p90"],
                "current_max_weight_median": current["max_weight_median"],
                "current_concentration_median": current["concentration_median"],
                "current_weight_noise_corr_median": current[
                    "weight_noise_corr_median"
                ],
                "delayed_weight_noise_corr_median": delayed[
                    "weight_noise_corr_median"
                ],
                "delta_test_delayed_minus_current": (
                    delayed["test_accuracy_final"] - current["test_accuracy_final"]
                ),
                "delta_worst20_delayed_minus_current": (
                    delayed["worst20_final"] - current["worst20_final"]
                ),
                "delta_gap_delayed_minus_current": (
                    delayed["gap_final"] - current["gap_final"]
                ),
                "promoted": not failed,
                "failed_checks": ";".join(failed),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "campaign_root",
        nargs="?",
        type=Path,
        default=Path(
            "results/dt_ldp_far/decisive/"
            "dt_ldp_far_decisive_stage4_mechanistic_transfer_screen_n25_v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/analysis/dt_ldp_far_n25_mechanistic_transfer_v1"),
    )
    args = parser.parse_args()
    metric_paths = sorted(args.campaign_root.glob("**/metrics.json"))
    if len(metric_paths) != 26:
        raise SystemExit(
            f"Screen is incomplete: found {len(metric_paths)}/26 metrics.json files"
        )
    runs = [_load_run(path) for path in metric_paths]
    pair_rows = _pair_rows(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "transfer_gate.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    promoted = [row for row in pair_rows if row["promoted"]]
    promoted.sort(
        key=lambda row: (
            _value(row["current_logit_span_median"], 0.0),
            _value(row["current_concentration_median"], 0.0),
        ),
        reverse=True,
    )
    summary = {
        "schema_version": 1,
        "status": "promoted" if promoted else "stopped_no_transfer_cell",
        "privacy_claim": "none; oracle diagnostic transfer gate",
        "runs": len(runs),
        "paired_cells": len(pair_rows),
        "promoted_cells": len(promoted),
        "selected_cell": promoted[0] if promoted else None,
        "gate": {
            "score_span_median_min": 0.15,
            "logit_span_median_min": 1.0,
            "saturation_p90_max": 0.25,
            "max_weight_median_min": 1.25 / 25,
            "concentration_median_min": 1.03,
            "current_weight_noise_corr_median_min": 0.10,
            "delayed_weight_noise_corr_abs_median_max": 0.15,
            "strict_randomness_pair_key_required": True,
            "strict_noise_norm_sequence_match_required": True,
        },
        "outputs": {"csv": str(csv_path)},
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if promoted else 2


if __name__ == "__main__":
    raise SystemExit(main())
