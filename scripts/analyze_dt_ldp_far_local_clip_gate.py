#!/usr/bin/env python3
"""Summarize the explicitly non-private local clipping calibration.

The per-example clipping rate is a private client-side diagnostic.  This
script is therefore restricted to runs whose manifests explicitly label the
transcript as non-private.  It ranks a pre-registered C grid but does not turn
the observed rate into a deployable data-dependent clipping rule.
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


def _manifest_near(metrics_path: Path) -> dict[str, Any]:
    for parent in metrics_path.parents:
        candidate = parent / "dt_ldp_far_task_manifest.json"
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Missing task manifest near {metrics_path}")


def _resolved_config_near(metrics_path: Path) -> dict[str, Any]:
    for parent in metrics_path.parents:
        candidate = parent / "resolved_config.yaml"
        if candidate.exists():
            import yaml

            return yaml.safe_load(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Missing resolved config near {metrics_path}")


def evaluate(metrics_path: Path) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rounds = list(payload.get("rounds", []))
    if not rounds:
        raise ValueError(f"No rounds in {metrics_path}")
    manifest = _manifest_near(metrics_path)
    config = _resolved_config_near(metrics_path)
    algo = config["training"]["algo_config"]
    if not bool(algo.get("non_private_diagnostic_transcript", False)):
        raise ValueError(f"Refusing unlabelled private oracle: {metrics_path}")

    local_clip = _median(rounds, "dtldp_client_clip_rate_mean_oracle")
    server_clip = _median(rounds[1:] or rounds, "dtldp_server_clip_rate")
    final = rounds[-1]
    # This is only a broad non-degeneracy screen.  Utility and the magnitude
    # of the DP noise must still be compared before C is frozen.
    nondegenerate = local_clip is not None and 0.05 <= local_clip <= 0.95
    return {
        "task_id": manifest["task_id"],
        "clip_norm_C": float(algo["clip_norm"]),
        "dp_enabled": bool(algo.get("enable_dp", True)),
        "privacy_profile": config["reproduction"]["axes"]["privacy"],
        "median_local_clip_rate_oracle": local_clip,
        "median_server_clip_rate": server_clip,
        "final_test_accuracy": final.get("test_accuracy"),
        "final_test_loss": final.get("test_loss"),
        "final_privacy_epsilon": final.get("privacy_epsilon_max"),
        "final_noise_multiplier": final.get("privacy_model_noise_multiplier_mean"),
        "broad_local_clip_nondegeneracy": nondegenerate,
        "metrics_path": str(metrics_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [evaluate(path) for path in sorted(args.root.glob("**/metrics.json"))]
    rows.sort(
        key=lambda row: (
            row["clip_norm_C"],
            row["dp_enabled"],
            abs((row["median_local_clip_rate_oracle"] or 0.0) - 0.5),
        )
    )
    output = args.output or args.root / "local_clip_gate_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"runs": len(rows), "ranked": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
