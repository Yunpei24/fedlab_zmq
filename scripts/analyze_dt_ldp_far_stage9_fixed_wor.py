#!/usr/bin/env python3
"""Analyze the Stage-9 fixed-size-without-replacement ablation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

from analyze_dt_ldp_far_stage7 import _load as _load_stage7
from analyze_dt_ldp_far_stage7 import _summaries, _write_csv


def _finite_median(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return median(values) if values else None


def _load(path: Path) -> dict:
    """Load the common Stage-7 fields and Stage-9 privacy diagnostics."""

    row = _load_stage7(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rounds = payload["rounds"]
    prefix = "far" if row["timing"] == "current" else "dtldp"
    final = rounds[-1]
    row.update(
        {
            "server_clip_rate_median": _finite_median(
                rounds, f"{prefix}_server_clip_rate"
            ),
            "server_clip_rate_honest_median": _finite_median(
                rounds, f"{prefix}_server_clip_rate_honest_oracle"
            ),
            "server_clip_rate_byzantine_median": _finite_median(
                rounds, f"{prefix}_server_clip_rate_byzantine_oracle"
            ),
            "privacy_sampling_scheme": final.get("privacy_sampling_scheme"),
            "privacy_adjacency": final.get("privacy_adjacency"),
            "privacy_accounting_assumption": final.get("privacy_accounting_assumption"),
            "privacy_noise_multiplier_min_final": final.get(
                "privacy_model_noise_multiplier_min"
            ),
            "privacy_noise_multiplier_mean_final": final.get(
                "privacy_model_noise_multiplier_mean"
            ),
            "privacy_noise_multiplier_max_final": final.get(
                "privacy_model_noise_multiplier_max"
            ),
            "privacy_steps_per_round_final": final.get("privacy_model_steps_mean"),
        }
    )
    return row


def _summaries_stage9(runs: list[dict]) -> list[dict]:
    summaries = _summaries(runs)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in runs:
        grouped[
            (row["timing"], row["reference"], row["threat"], row["score_mode"])
        ].append(row)
    extra_fields = (
        "server_clip_rate_median",
        "server_clip_rate_honest_median",
        "server_clip_rate_byzantine_median",
        "privacy_noise_multiplier_min_final",
        "privacy_noise_multiplier_mean_final",
        "privacy_noise_multiplier_max_final",
    )
    for summary in summaries:
        key = (
            summary["timing"],
            summary["reference"],
            summary["threat"],
            summary["score_mode"],
        )
        rows = grouped[key]
        for field in extra_fields:
            values = [float(row[field]) for row in rows if row[field] is not None]
            summary[f"{field}_mean"] = mean(values) if values else None
            summary[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0
    return summaries


def _paired_timing(runs: list[dict]) -> list[dict]:
    groups: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in runs:
        key = (
            row["reference"],
            row["threat"],
            row["partition_seed"],
            row["training_seed"],
        )
        groups[key][row["timing"]] = row
    result = []
    for key, pair in sorted(groups.items()):
        if set(pair) != {"current", "delayed"}:
            continue
        current, delayed = pair["current"], pair["delayed"]
        result.append(
            {
                "reference": key[0],
                "threat": key[1],
                "partition_seed": key[2],
                "training_seed": key[3],
                "pair_key_matches": current["pair_key"] == delayed["pair_key"],
                "delta_test_accuracy_pp": 100.0
                * (delayed["test_accuracy_final"] - current["test_accuracy_final"]),
                "delta_worst20_pp": delayed["worst20_final"] - current["worst20_final"],
                "delta_gap_pp": delayed["gap_final"] - current["gap_final"],
                "delta_clean_score_corr": delayed["score_clean_corr_median"]
                - current["score_clean_corr_median"],
                "delta_clean_weight_l1": delayed["weight_clean_l1_median"]
                - current["weight_clean_l1_median"],
                "delta_weight_concentration": delayed["weight_concentration_median"]
                - current["weight_concentration_median"],
                "delta_byzantine_weight_mass": delayed["byzantine_weight_mass_median"]
                - current["byzantine_weight_mass_median"],
                "delta_server_clip_rate": delayed["server_clip_rate_median"]
                - current["server_clip_rate_median"],
            }
        )
    return result


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fixed_vs_poisson(runs: list[dict], poisson_csv: Path) -> list[dict]:
    poisson_rows = [
        row
        for row in _read_csv(poisson_csv)
        if row["score_mode"] == "isotropic_dp_debiased_distance"
    ]
    poisson = {
        (
            row["timing"],
            row["reference"],
            row["threat"],
            int(row["partition_seed"]),
            int(row["training_seed"]),
        ): row
        for row in poisson_rows
    }
    result = []
    for fixed in runs:
        key = (
            fixed["timing"],
            fixed["reference"],
            fixed["threat"],
            fixed["partition_seed"],
            fixed["training_seed"],
        )
        baseline = poisson.get(key)
        if baseline is None:
            continue
        result.append(
            {
                "timing": key[0],
                "reference": key[1],
                "threat": key[2],
                "partition_seed": key[3],
                "training_seed": key[4],
                "comparison_is_not_same_adjacency": True,
                "fixed_replace_one_epsilon": fixed["epsilon_final"],
                "poisson_add_remove_epsilon": float(baseline["epsilon_final"]),
                "delta_test_accuracy_pp": 100.0
                * (
                    fixed["test_accuracy_final"]
                    - float(baseline["test_accuracy_final"])
                ),
                "delta_worst20_pp": fixed["worst20_final"]
                - float(baseline["worst20_final"]),
                "delta_gap_pp": fixed["gap_final"] - float(baseline["gap_final"]),
                "delta_clean_score_corr": fixed["score_clean_corr_median"]
                - float(baseline["score_clean_corr_median"]),
                "delta_clean_weight_l1": fixed["weight_clean_l1_median"]
                - float(baseline["weight_clean_l1_median"]),
                "fixed_server_clip_rate_median": fixed["server_clip_rate_median"],
                "poisson_server_clip_rate_median": _load(
                    Path(baseline["metrics_path"])
                )["server_clip_rate_median"],
                "fixed_noise_multiplier_min": fixed[
                    "privacy_noise_multiplier_min_final"
                ],
                "poisson_noise_multiplier_min": _load(Path(baseline["metrics_path"]))[
                    "privacy_noise_multiplier_min_final"
                ],
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(
            "results/dt_ldp_far/decisive_stage9_fixed_wor_mps/"
            "dt_ldp_far_decisive_stage9_fixed_wor_n25_v1"
        ),
    )
    parser.add_argument(
        "--poisson-run-csv",
        type=Path,
        default=Path(
            "output/analysis/"
            "dt_ldp_far_n25_stage8_debiased_distance_end_to_end/run_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/analysis/dt_ldp_far_n25_stage9_fixed_wor"),
    )
    parser.add_argument("--expected-runs", type=int, default=36)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    paths = sorted(args.results_root.rglob("metrics.json"))
    if len(paths) != args.expected_runs and not args.allow_partial:
        raise SystemExit(f"Stage 9 incomplete: {len(paths)}/{args.expected_runs}")
    runs = [_load(path) for path in paths]
    timing_pairs = _paired_timing(runs)
    sampler_pairs = _fixed_vs_poisson(runs, args.poisson_run_csv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "run_metrics.csv", runs)
    _write_csv(
        args.output_dir / "summary_by_timing_reference_threat.csv",
        _summaries_stage9(runs),
    )
    _write_csv(args.output_dir / "paired_delayed_minus_current.csv", timing_pairs)
    _write_csv(args.output_dir / "fixed_wor_minus_poisson.csv", sampler_pairs)
    status = {
        "completed_runs": len(runs),
        "expected_runs": args.expected_runs,
        "complete": len(runs) == args.expected_runs,
        "current_delayed_pairs": len(timing_pairs),
        "all_current_delayed_pair_keys_match": bool(timing_pairs)
        and all(row["pair_key_matches"] for row in timing_pairs),
        "fixed_wor_poisson_seed_matched_comparisons": len(sampler_pairs),
        "sampler_comparison_warning": (
            "Fixed WOR uses public-size replace-one adjacency and 2C sensitivity; "
            "Poisson uses add/remove adjacency and C sensitivity. Epsilon values "
            "are numerically matched but the neighboring relations differ."
        ),
    }
    (args.output_dir / "status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
