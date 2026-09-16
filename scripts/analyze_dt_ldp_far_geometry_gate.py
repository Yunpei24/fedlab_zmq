#!/usr/bin/env python3
"""Evaluate the public geometry gate before 25-client DT-LDP-FAR runs.

The gate deliberately uses only server-visible quantities derived from
already-private uploads.  The separately labelled oracle task is reported but
never used to certify or select the publishable private mechanism.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def _median(rounds: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rounds if row.get(key) is not None]
    return statistics.median(values) if values else None


def _minimum(rounds: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rounds if row.get(key) is not None]
    return min(values) if values else None


def _identity(metrics_path: Path) -> dict[str, Any]:
    manifest_path = next(
        iter(metrics_path.parents[1].glob("dt_ldp_far_task_manifest.json")), None
    )
    if manifest_path is None:
        manifest_path = metrics_path.parents[1] / "dt_ldp_far_task_manifest.json"
    if not manifest_path.exists():
        # The framework adds one run-directory below the protocol task folder.
        manifest_path = metrics_path.parents[2] / "dt_ldp_far_task_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing task manifest near {metrics_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def evaluate(metrics_path: Path) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rounds = list(payload.get("rounds", []))
    if not rounds:
        raise ValueError(f"No rounds in {metrics_path}")
    # Round zero is necessarily uniform because delayed scores do not exist.
    informative = rounds[1:] if len(rounds) > 1 else rounds
    manifest = _identity(metrics_path)
    task_id = str(manifest["task_id"])
    geometry = task_id.split("__")[7]
    method = task_id.split("__")[2]
    n = int(informative[-1].get("num_clients", 10))
    clip_rate = _median(informative, "dtldp_server_clip_rate")
    score_span = _median(informative, "dtldp_current_score_span")
    saturation = _median(informative, "dtldp_current_score_saturation_rate")
    max_weight = _median(informative, "max_client_weight")
    amplification = _median(
        informative, "dtldp_noise_amplification_vs_uniform"
    )
    cap_margin = _minimum(
        [
            {
                "margin": float(row["dtldp_weight_cap"])
                - float(row["max_client_weight"])
            }
            for row in informative
            if row.get("dtldp_weight_cap") is not None
            and row.get("max_client_weight") is not None
        ],
        "margin",
    )
    public_only = method != "dt_ldp_far_oracle"
    gate_pass = bool(
        public_only
        and clip_rate is not None
        and 0.05 <= clip_rate <= 0.95
        and score_span is not None
        and score_span >= 0.30
        and saturation is not None
        and saturation <= 0.20
        and max_weight is not None
        and amplification is not None
        and (max_weight >= 1.25 / n or amplification >= 1.03)
        and cap_margin is not None
        and cap_margin >= -1e-8
    )
    return {
        "task_id": task_id,
        "method": method,
        "geometry": geometry,
        "public_selection_eligible": public_only,
        "median_server_clip_rate": clip_rate,
        "median_score_span": score_span,
        "median_score_saturation_rate": saturation,
        "median_max_weight": max_weight,
        "median_noise_amplification": amplification,
        "minimum_weight_cap_margin": cap_margin,
        "gate_pass": gate_pass,
        "metrics_path": str(metrics_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.root.glob("**/metrics.json"))
    rows = [evaluate(path) for path in paths]
    rows.sort(
        key=lambda row: (
            not row["gate_pass"],
            -(row["median_score_span"] or 0.0),
            abs((row["median_server_clip_rate"] or 0.0) - 0.5),
        )
    )
    output = args.output or args.root / "geometry_gate_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"runs": len(rows), "passing": sum(r["gate_pass"] for r in rows), "ranked": rows}, indent=2))
    return 0 if any(row["gate_pass"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
