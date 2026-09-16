#!/usr/bin/env python3
"""Analyze the pre-registered Stage-6 public noise-aware FAR score campaign."""

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

EXPECTED_RUNS = 30


def _nearest(path: Path, filename: str) -> Path:
    for parent in path.parents:
        candidate = parent / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {filename} above {path}")


def _numeric(rounds: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(row[key])
        for row in rounds
        if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))
    ]


def _median(rounds: list[dict[str, Any]], key: str) -> float | None:
    values = _numeric(rounds, key)
    return median(values) if values else None


def _load_run(metrics_path: Path) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(
        _nearest(metrics_path, "resolved_config.yaml").read_text(encoding="utf-8")
    )
    expected_rounds = int(config["training"]["num_rounds"])
    all_rounds = payload.get("rounds", [])
    if len(all_rounds) != expected_rounds:
        raise ValueError(
            f"Incomplete run ({len(all_rounds)}/{expected_rounds}): {metrics_path}"
        )
    # Round zero is retained for score-fidelity diagnostics.  DT's current
    # score exists at round zero even though its applied weights are uniform.
    rounds = all_rounds
    axes = config["reproduction"]["axes"]
    algo = config["training"]["algo_config"]
    method = str(axes["method"])
    current = method.startswith("dp_far_current")
    prefix = "far" if current else "dtldp_current"
    general_prefix = "far" if current else "dtldp"

    keys = {
        "score_span": f"{prefix}_score_span",
        "saturation": f"{prefix}_score_saturation_rate",
        "concentration": (
            "far_noise_amplification_vs_uniform"
            if current
            else "dtldp_current_noise_amplification_vs_uniform"
        ),
        "max_weight": (
            "max_client_weight" if current else "dtldp_current_max_client_weight"
        ),
        "score_clean_corr": (
            "far_noisy_raw_clean_score_corr_oracle"
            if current
            else "dtldp_current_noisy_raw_clean_score_corr_oracle"
        ),
        "score_clean_mae": (
            "far_noisy_raw_clean_score_mae_oracle"
            if current
            else "dtldp_current_noisy_raw_clean_score_mae_oracle"
        ),
        "score_clean_rmse": (
            "far_noisy_raw_clean_score_rmse_oracle"
            if current
            else "dtldp_current_noisy_raw_clean_score_rmse_oracle"
        ),
        "weight_clean_l1": (
            "far_noisy_raw_clean_weight_l1_oracle"
            if current
            else "dtldp_current_noisy_raw_clean_weight_l1_oracle"
        ),
        "score_scale_corr": (
            "far_score_public_noise_scale_corr_oracle"
            if current
            else "dtldp_current_score_public_noise_scale_corr_oracle"
        ),
        "raw_clean_scale_corr": (
            "far_raw_clean_score_public_noise_scale_corr_oracle"
            if current
            else "dtldp_raw_clean_score_public_noise_scale_corr_oracle"
        ),
        "cap": (
            "far_weight_cap_respected" if current else "dtldp_weight_cap_respected"
        ),
    }
    final = all_rounds[-1]
    score_scale_corr = _median(rounds, keys["score_scale_corr"])
    raw_clean_scale_corr = _median(rounds, keys["raw_clean_scale_corr"])
    excess_scale_corr = (
        abs(score_scale_corr - raw_clean_scale_corr)
        if score_scale_corr is not None and raw_clean_scale_corr is not None
        else None
    )
    return {
        "method": method,
        "timing": "current" if current else "delayed",
        "score_standardization": str(algo.get("noise_score_standardization", "none")),
        "privacy": str(axes["privacy"]),
        "partition_seed": int(axes["partition_seed"]),
        "training_seed": int(axes["training_seed"]),
        "pair_key": str(config["reproduction"]["randomness_pair_key"]),
        "epsilon_final": final.get("privacy_epsilon_max"),
        "score_span_median": _median(rounds, keys["score_span"]),
        "score_saturation_median": _median(rounds, keys["saturation"]),
        "weight_concentration_median": _median(rounds, keys["concentration"]),
        "max_weight_median": _median(rounds, keys["max_weight"]),
        "score_raw_clean_corr_median": _median(rounds, keys["score_clean_corr"]),
        "score_raw_clean_mae_median": _median(rounds, keys["score_clean_mae"]),
        "score_raw_clean_rmse_median": _median(rounds, keys["score_clean_rmse"]),
        "weight_raw_clean_l1_median": _median(rounds, keys["weight_clean_l1"]),
        "score_public_scale_corr_median": score_scale_corr,
        "raw_clean_score_public_scale_corr_median": raw_clean_scale_corr,
        "excess_public_scale_corr_median": excess_scale_corr,
        "weight_cap_respected_all_rounds": all(
            bool(row.get(keys["cap"])) for row in rounds
        ),
        "test_accuracy_final": final.get("test_accuracy"),
        "client_accuracy_final": final.get("client_mean_accuracy_pct"),
        "variance_pp2_final": final.get("client_accuracy_variance_pp2"),
        "worst20_final": final.get("worst20_accuracy_pct"),
        "gap_final": final.get("best20_worst20_gap_pct"),
        "score_span_sequence": _numeric(rounds, keys["score_span"]),
        "test_accuracy_sequence": _numeric(rounds, "test_accuracy"),
        "metrics_path": str(metrics_path),
    }


def _ratio(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value / max(baseline, 1e-15)


def _paired_comparisons(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, int], dict[str, dict[str, Any]]] = defaultdict(
        dict
    )
    for row in runs:
        key = (
            row["privacy"],
            row["timing"],
            row["partition_seed"],
            row["training_seed"],
        )
        groups[key][row["score_standardization"]] = row
    comparisons = []
    for key, modes in sorted(groups.items()):
        raw = modes.get("none")
        if raw is None:
            raise ValueError(f"Missing raw score arm for {key}")
        for mode, row in sorted(modes.items()):
            if mode == "none":
                continue
            comparisons.append(
                {
                    "privacy": key[0],
                    "timing": key[1],
                    "partition_seed": key[2],
                    "training_seed": key[3],
                    "score_standardization": mode,
                    "pair_key_matches": row["pair_key"] == raw["pair_key"],
                    "score_rmse_ratio_vs_raw": _ratio(
                        row["score_raw_clean_rmse_median"],
                        raw["score_raw_clean_rmse_median"],
                    ),
                    "weight_l1_ratio_vs_raw": _ratio(
                        row["weight_raw_clean_l1_median"],
                        raw["weight_raw_clean_l1_median"],
                    ),
                    "excess_scale_corr_ratio_vs_raw": _ratio(
                        row["excess_public_scale_corr_median"],
                        raw["excess_public_scale_corr_median"],
                    ),
                    "delta_score_clean_corr": (
                        row["score_raw_clean_corr_median"]
                        - raw["score_raw_clean_corr_median"]
                    ),
                    "delta_test_accuracy_pp": (
                        row["test_accuracy_final"] - raw["test_accuracy_final"]
                    ),
                }
            )
    return comparisons


def _negative_control(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in runs if row["privacy"] == "eps4_c4"]
    groups: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        groups[(row["timing"], row["partition_seed"], row["training_seed"])][
            row["score_standardization"]
        ] = row
    checks = []
    for key, modes in sorted(groups.items()):
        raw = modes.get("none")
        corrected = modes.get("isotropic_dp_covariance_proxy")
        if raw is None or corrected is None:
            raise ValueError(f"Incomplete homogeneous negative control: {key}")
        span_delta = max(
            (
                abs(a - b)
                for a, b in zip(
                    raw["score_span_sequence"], corrected["score_span_sequence"]
                )
            ),
            default=0.0,
        )
        accuracy_delta = max(
            (
                abs(a - b)
                for a, b in zip(
                    raw["test_accuracy_sequence"],
                    corrected["test_accuracy_sequence"],
                )
            ),
            default=0.0,
        )
        checks.append(
            {
                "timing": key[0],
                "partition_seed": key[1],
                "training_seed": key[2],
                "pair_key_matches": raw["pair_key"] == corrected["pair_key"],
                "max_abs_score_span_delta": span_delta,
                "max_abs_test_accuracy_delta": accuracy_delta,
                "exact_negative_control": span_delta <= 1e-12
                and accuracy_delta <= 1e-12,
            }
        )
    return checks


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    scalar_rows = [
        {key: value for key, value in row.items() if not isinstance(value, list)}
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)


def _summary_table(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        groups[(row["privacy"], row["timing"], row["score_standardization"])].append(
            row
        )
    result = []
    fields = (
        "score_raw_clean_corr_median",
        "score_raw_clean_rmse_median",
        "weight_raw_clean_l1_median",
        "excess_public_scale_corr_median",
        "test_accuracy_final",
        "worst20_final",
        "gap_final",
    )
    for key, rows in sorted(groups.items()):
        record: dict[str, Any] = {
            "privacy": key[0],
            "timing": key[1],
            "score_standardization": key[2],
            "n": len(rows),
        }
        for field in fields:
            values = [float(row[field]) for row in rows if row[field] is not None]
            record[f"{field}_mean"] = mean(values) if values else None
            record[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0
        result.append(record)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(
            "results/dt_ldp_far/decisive/"
            "dt_ldp_far_decisive_stage6_noise_aware_score_n25_v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/analysis/dt_ldp_far_n25_stage6_noise_aware"),
    )
    args = parser.parse_args()
    paths = sorted(args.results_root.rglob("metrics.json"))
    if len(paths) != EXPECTED_RUNS:
        raise SystemExit(
            f"Stage 6 incomplete: {len(paths)}/{EXPECTED_RUNS} metrics.json"
        )
    runs = [_load_run(path) for path in paths]
    comparisons = _paired_comparisons(runs)
    negative = _negative_control(runs)
    summaries = _summary_table(runs)

    hetero_cov = [
        row
        for row in comparisons
        if row["privacy"] == "eps4_c4_hetero_noise_n25"
        and row["score_standardization"] == "isotropic_dp_covariance_proxy"
    ]
    criteria = {
        "all_randomness_pairs_match": all(
            row["pair_key_matches"] for row in comparisons
        ),
        "homogeneous_negative_control_exact": all(
            row["exact_negative_control"] for row in negative
        ),
        "covariance_proxy_reduces_score_rmse_majority": sum(
            row["score_rmse_ratio_vs_raw"] < 1.0 for row in hetero_cov
        )
        >= 4,
        "covariance_proxy_reduces_weight_l1_majority": sum(
            row["weight_l1_ratio_vs_raw"] < 1.0 for row in hetero_cov
        )
        >= 4,
        "covariance_proxy_halves_excess_scale_corr_majority": sum(
            row["excess_scale_corr_ratio_vs_raw"] <= 0.5 for row in hetero_cov
        )
        >= 4,
        "all_weight_caps_respected": all(
            row["weight_cap_respected_all_rounds"] for row in runs
        ),
    }
    criteria["primary_hypothesis_supported"] = all(criteria.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "run_metrics.csv", runs)
    _write_csv(args.output_dir / "paired_comparisons.csv", comparisons)
    _write_csv(args.output_dir / "homogeneous_negative_control.csv", negative)
    _write_csv(args.output_dir / "summary_by_method.csv", summaries)
    (args.output_dir / "verdict.json").write_text(
        json.dumps(
            {
                "completed_runs": len(runs),
                "criteria": criteria,
                "interpretation_guardrail": (
                    "This diagnostic campaign tests a public isotropic covariance "
                    "proxy. It does not establish the exact post-DP-SGD/post-clipping "
                    "upload covariance and does not select by final accuracy."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(criteria, indent=2))


if __name__ == "__main__":
    main()
