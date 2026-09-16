#!/usr/bin/env python3
"""Evaluate the refined public geometry gate for DT-LDP-FAR.

The operating gate detects a usable score geometry without requiring clean
uploads to be clipped artificially.  A stricter stress flag identifies
profiles whose bounded weights are sufficiently non-uniform for a sensitive
current-versus-delayed comparison.  Both use only server-visible
post-processing of already locally-private uploads.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _values(rounds: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rounds if row.get(key) is not None]


def _median(rounds: list[dict[str, Any]], key: str) -> float | None:
    values = _values(rounds, key)
    return statistics.median(values) if values else None


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _fraction(values: list[float], predicate) -> float | None:
    return sum(bool(predicate(value)) for value in values) / len(values) if values else None


def _manifest_near(metrics_path: Path) -> dict[str, Any]:
    for parent in metrics_path.parents:
        candidate = parent / "dt_ldp_far_task_manifest.json"
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Missing task manifest near {metrics_path}")


def evaluate(
    metrics_path: Path,
    *,
    minimum_median_score_span: float = 0.15,
    persistent_score_span_threshold: float = 0.20,
    minimum_persistent_score_span_fraction: float = 0.60,
) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rounds = list(payload.get("rounds", []))
    if len(rounds) < 2:
        raise ValueError(f"Need at least two rounds in {metrics_path}")
    # Round zero applies uniform delayed weights and is excluded from the
    # applied-weight criteria. The remaining current scores produce the next
    # round's weights and are still valid public diagnostics.
    informative = rounds[1:]
    manifest = _manifest_near(metrics_path)
    task_id = str(manifest["task_id"])
    parts = task_id.split("__")
    geometry = parts[7]
    method = parts[2]
    n = int(informative[-1].get("num_clients", 10))

    clip_rates = _values(informative, "dtldp_server_clip_rate")
    utilizations = _values(informative, "dtldp_server_clip_utilization_max")
    spans = _values(informative, "dtldp_current_score_span")
    saturations = _values(informative, "dtldp_current_score_saturation_rate")
    max_weights = _values(informative, "max_client_weight")
    amplifications = _values(informative, "dtldp_noise_amplification_vs_uniform")
    cap_margins = [
        float(row["dtldp_weight_cap"]) - float(row["max_client_weight"])
        for row in informative
        if row.get("dtldp_weight_cap") is not None
        and row.get("max_client_weight") is not None
    ]

    median_clip = statistics.median(clip_rates) if clip_rates else None
    median_utilization = statistics.median(utilizations) if utilizations else None
    median_span = statistics.median(spans) if spans else None
    median_max_weight = statistics.median(max_weights) if max_weights else None
    median_amplification = (
        statistics.median(amplifications) if amplifications else None
    )
    p90_saturation = _quantile(saturations, 0.90)
    rounds_low_clip = _fraction(clip_rates, lambda value: value <= 0.50)
    rounds_span_ge_020 = _fraction(spans, lambda value: value >= 0.20)
    rounds_persistent = _fraction(
        spans, lambda value: value >= persistent_score_span_threshold
    )
    minimum_cap_margin = min(cap_margins) if cap_margins else None

    public_only = method != "dt_ldp_far_oracle"
    operating_gate = bool(
        public_only
        and median_clip is not None
        and median_clip <= 0.30
        and rounds_low_clip is not None
        and rounds_low_clip >= 0.80
        and median_utilization is not None
        and 0.50 <= median_utilization <= 1.25
        and median_span is not None
        and median_span >= minimum_median_score_span
        and rounds_persistent is not None
        and rounds_persistent >= minimum_persistent_score_span_fraction
        and p90_saturation is not None
        and p90_saturation <= 0.50
        and minimum_cap_margin is not None
        and minimum_cap_margin >= -1e-8
    )
    stress_informative = bool(
        operating_gate
        and median_span is not None
        and median_span >= 0.30
        and median_max_weight is not None
        and median_amplification is not None
        and (
            median_max_weight >= 1.25 / n
            or median_amplification >= 1.03
        )
    )
    return {
        "task_id": task_id,
        "geometry": geometry,
        "median_server_clip_rate": median_clip,
        "fraction_rounds_server_clip_le_50pct": rounds_low_clip,
        "median_server_clip_utilization_max": median_utilization,
        "median_score_span": median_span,
        # Keep the original diagnostic for comparisons with the strict n=10
        # gate, while recording the exact threshold used by this invocation.
        "fraction_rounds_score_span_ge_0_20": rounds_span_ge_020,
        "persistent_score_span_threshold": persistent_score_span_threshold,
        "fraction_rounds_score_span_ge_gate_threshold": rounds_persistent,
        "minimum_median_score_span": minimum_median_score_span,
        "minimum_persistent_score_span_fraction": (
            minimum_persistent_score_span_fraction
        ),
        "p90_score_saturation_rate": p90_saturation,
        "median_max_weight": median_max_weight,
        "median_noise_amplification": median_amplification,
        "minimum_weight_cap_margin": minimum_cap_margin,
        "operating_gate_pass": operating_gate,
        "stress_informative": stress_informative,
        "metrics_path": str(metrics_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-median-score-span", type=float, default=0.15)
    parser.add_argument(
        "--persistent-score-span-threshold", type=float, default=0.20
    )
    parser.add_argument(
        "--minimum-persistent-score-span-fraction", type=float, default=0.60
    )
    args = parser.parse_args()
    rows = [
        evaluate(
            path,
            minimum_median_score_span=args.minimum_median_score_span,
            persistent_score_span_threshold=args.persistent_score_span_threshold,
            minimum_persistent_score_span_fraction=(
                args.minimum_persistent_score_span_fraction
            ),
        )
        for path in sorted(args.root.glob("**/metrics.json"))
    ]
    rows.sort(
        key=lambda row: (
            not row["operating_gate_pass"],
            not row["stress_informative"],
            -(row["median_score_span"] or 0.0),
            row["p90_score_saturation_rate"] or 0.0,
        )
    )
    output = args.output or args.root / "refined_geometry_gate_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(
        json.dumps(
            {
                "runs": len(rows),
                "operating_pass": sum(row["operating_gate_pass"] for row in rows),
                "stress_informative": sum(row["stress_informative"] for row in rows),
                "ranked": rows,
            },
            indent=2,
        )
    )
    return 0 if any(row["operating_gate_pass"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
