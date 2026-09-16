#!/usr/bin/env python3
"""Analyze paired raw/noise-aware DT-LDP-FAR end-to-end screens."""

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

EXPECTED_RUNS = 84


def _nearest(path: Path, filename: str) -> Path:
    for parent in path.parents:
        candidate = parent / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {filename} above {path}")


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))
    ]
    return median(values) if values else None


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = yaml.safe_load(
        _nearest(path, "resolved_config.yaml").read_text(encoding="utf-8")
    )
    rounds = payload.get("rounds", [])
    expected = int(config["training"]["num_rounds"])
    if len(rounds) != expected:
        raise ValueError(f"Incomplete run {len(rounds)}/{expected}: {path}")
    axes = config["reproduction"]["axes"]
    algo = config["training"]["algo_config"]
    method = str(axes["method"])
    current = method.startswith("dp_far_current")
    prefix = "far" if current else "dtldp_current"
    general = "far" if current else "dtldp"
    final = rounds[-1]
    return {
        "method": method,
        "timing": "current" if current else "delayed",
        "reference": str(axes["reference"]),
        "threat": str(axes["threat"]),
        "score_mode": str(algo.get("noise_score_standardization", "none")),
        "partition_seed": int(axes["partition_seed"]),
        "training_seed": int(axes["training_seed"]),
        "pair_key": str(config["reproduction"]["randomness_pair_key"]),
        "epsilon_final": final.get("privacy_epsilon_max"),
        "score_clean_corr_median": _median(
            rounds, f"{prefix}_noisy_clean_score_corr_oracle"
        ),
        "score_clean_rmse_median": _median(
            rounds, f"{prefix}_noisy_clean_score_rmse_oracle"
        ),
        "raw_clean_corr_median": _median(
            rounds, f"{prefix}_noisy_raw_clean_score_corr_oracle"
        ),
        "weight_clean_l1_median": _median(
            rounds, f"{prefix}_noisy_clean_weight_l1_oracle"
        ),
        "score_public_scale_corr_median": _median(
            rounds, f"{prefix}_score_public_noise_scale_corr_oracle"
        ),
        "score_span_median": _median(rounds, f"{prefix}_score_span"),
        "weight_concentration_median": _median(
            rounds,
            (
                "far_noise_amplification_vs_uniform"
                if current
                else "dtldp_current_noise_amplification_vs_uniform"
            ),
        ),
        "byzantine_weight_mass_median": _median(
            rounds, "byzantine_weight_mass_oracle"
        ),
        "reference_honest_error_median": _median(
            rounds, f"{general}_reference_honest_center_error_oracle"
        ),
        "weight_cap_all_rounds": all(
            bool(
                row.get(
                    "far_weight_cap_respected"
                    if current
                    else "dtldp_weight_cap_respected"
                )
            )
            for row in rounds
        ),
        "test_accuracy_final": final.get("test_accuracy"),
        "worst20_final": final.get("worst20_accuracy_pct"),
        "gap_final": final.get("best20_worst20_gap_pct"),
        "metrics_path": str(path),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summaries(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        groups[
            (row["timing"], row["reference"], row["threat"], row["score_mode"])
        ].append(row)
    fields = (
        "score_clean_corr_median",
        "score_clean_rmse_median",
        "raw_clean_corr_median",
        "weight_clean_l1_median",
        "score_public_scale_corr_median",
        "score_span_median",
        "weight_concentration_median",
        "byzantine_weight_mass_median",
        "reference_honest_error_median",
        "test_accuracy_final",
        "worst20_final",
        "gap_final",
    )
    result = []
    for key, rows in sorted(groups.items()):
        record: dict[str, Any] = {
            "timing": key[0],
            "reference": key[1],
            "threat": key[2],
            "score_mode": key[3],
            "n": len(rows),
        }
        for field in fields:
            values = [float(row[field]) for row in rows if row[field] is not None]
            record[f"{field}_mean"] = mean(values) if values else None
            record[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0
        result.append(record)
    return result


def _paired(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in runs:
        key = (
            row["timing"],
            row["reference"],
            row["threat"],
            row["partition_seed"],
            row["training_seed"],
        )
        groups[key][row["score_mode"]] = row
    result = []
    for key, modes in sorted(groups.items()):
        raw = modes.get("none")
        if raw is None:
            continue
        for name, candidate in modes.items():
            if name == "none":
                continue
            result.append(
                {
                    "timing": key[0],
                    "reference": key[1],
                    "threat": key[2],
                    "partition_seed": key[3],
                    "training_seed": key[4],
                    "score_mode": name,
                    "pair_key_matches": raw["pair_key"] == candidate["pair_key"],
                    "delta_score_clean_corr": (
                        candidate["score_clean_corr_median"]
                        - raw["score_clean_corr_median"]
                    ),
                    "score_rmse_ratio": (
                        candidate["score_clean_rmse_median"]
                        / max(raw["score_clean_rmse_median"], 1e-15)
                    ),
                    "weight_l1_ratio": (
                        candidate["weight_clean_l1_median"]
                        / max(raw["weight_clean_l1_median"], 1e-15)
                    ),
                    "delta_abs_scale_corr": abs(
                        candidate["score_public_scale_corr_median"]
                    )
                    - abs(raw["score_public_scale_corr_median"]),
                    "delta_test_accuracy_pp": (
                        candidate["test_accuracy_final"]
                        - raw["test_accuracy_final"]
                    ),
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(
            "results/dt_ldp_far/decisive/"
            "dt_ldp_far_decisive_stage7_excess_energy_rfa_n25_v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/analysis/dt_ldp_far_n25_stage7_excess_energy_rfa"),
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--expected-runs", type=int, default=EXPECTED_RUNS)
    args = parser.parse_args()
    paths = sorted(args.results_root.rglob("metrics.json"))
    if len(paths) != args.expected_runs and not args.allow_partial:
        raise SystemExit(
            f"End-to-end screen incomplete: {len(paths)}/{args.expected_runs}"
        )
    runs = [_load(path) for path in paths]
    summaries = _summaries(runs)
    paired = _paired(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "run_metrics.csv", runs)
    _write_csv(args.output_dir / "summary_by_method_reference_threat.csv", summaries)
    _write_csv(args.output_dir / "paired_score_comparisons.csv", paired)
    status = {
        "completed_runs": len(runs),
        "expected_runs": args.expected_runs,
        "complete": len(runs) == args.expected_runs,
        "all_weight_caps_respected": all(row["weight_cap_all_rounds"] for row in runs),
        "all_available_pairs_match": all(
            row["pair_key_matches"] for row in paired
        ),
        "decision_guardrail": (
            "Promote a candidate only if it improves clean-score and "
            "clean-weight recovery on a majority of paired seeds without "
            "increasing dependence on the public noise tier; accuracy is secondary."
        ),
    }
    (args.output_dir / "status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
