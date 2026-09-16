#!/usr/bin/env python3
"""Analyze the frozen Stage-12 generation-5 score screen."""

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


EXPECTED_RUNS = 32
NO_TRUST = "dp_far_current_tier_rank_no_trust_oracle"
TIER_FAIRNESS = "dp_far_current_tier_rank_fairness_oracle"
TIER_BALANCED = "dp_far_current_tier_rank_balanced_oracle"
THEIL_SEN = "dp_far_current_theil_sen_fairness_oracle"

PROFILE_LABELS = {
    NO_TRUST: "tier-rank sans confiance",
    TIER_FAIRNESS: "tier-rank 75/25",
    TIER_BALANCED: "tier-rank 50/50",
    THEIL_SEN: "Theil-Sen 75/25",
}


def _nearest(path: Path, filename: str) -> Path:
    for parent in path.parents:
        candidate = parent / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {filename} above {path}")


def _finite(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))
    ]


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _finite(rows, key)
    return median(values) if values else None


def _maximum(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _finite(rows, key)
    return max(values) if values else None


def _pct(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    value = float(value)
    return 100.0 * value if abs(value) <= 1.0 else value


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = yaml.safe_load(
        _nearest(path, "resolved_config.yaml").read_text(encoding="utf-8")
    )
    rounds = payload.get("rounds", [])
    expected_rounds = int(config["training"]["num_rounds"])
    if len(rounds) != expected_rounds:
        raise ValueError(f"Incomplete run {len(rounds)}/{expected_rounds}: {path}")
    axes = config["reproduction"]["axes"]
    algo = config["training"]["algo_config"]
    final = rounds[-1]
    return {
        "method": str(axes["method"]),
        "profile": PROFILE_LABELS[str(axes["method"])],
        "reference": str(axes["reference"]),
        "threat": str(axes["threat"]),
        "partition_seed": int(axes["partition_seed"]),
        "training_seed": int(axes["training_seed"]),
        "pair_key": str(config["reproduction"]["randomness_pair_key"]),
        "device": str(
            payload.get("summary", {})
            .get("algo_config", {})
            .get("device", config.get("training", {}).get("device", "unknown"))
        ),
        "num_rounds": len(rounds),
        "epsilon_max": final.get("privacy_epsilon_max"),
        "epsilon_mean": final.get("privacy_epsilon_mean"),
        "delta": final.get("privacy_delta"),
        "privacy_steps_per_round": final.get("privacy_model_steps_mean"),
        "privacy_sampling_scheme": final.get("privacy_sampling_scheme"),
        "privacy_adjacency": final.get("privacy_adjacency"),
        "noise_multiplier_min": final.get("privacy_model_noise_multiplier_min"),
        "noise_multiplier_mean": final.get("privacy_model_noise_multiplier_mean"),
        "noise_multiplier_max": final.get("privacy_model_noise_multiplier_max"),
        "local_clip_rate_median": _median(rounds, "privacy_clip_rate_mean"),
        "server_clip_rate_median": _median(rounds, "far_server_clip_rate"),
        "honest_score_clean_corr_median": _median(
            rounds, "far_honest_noisy_clean_score_corr_oracle"
        ),
        "honest_score_clean_rmse_median": _median(
            rounds, "far_honest_noisy_clean_score_rmse_oracle"
        ),
        "honest_top20_recall_median": _median(
            rounds, "far_honest_clean_top_tail_recall_oracle"
        ),
        "honest_score_public_scale_corr_median": _median(
            rounds, "far_honest_score_public_noise_scale_corr_oracle"
        ),
        "honest_weight_clean_l1_median": _median(
            rounds, "far_honest_noisy_clean_weight_l1_oracle"
        ),
        "score_span_median": _median(rounds, "far_score_span"),
        "weight_concentration_median": _median(
            rounds, "far_noise_amplification_vs_uniform"
        ),
        "byzantine_weight_mass_median": _median(rounds, "byzantine_weight_mass_oracle"),
        "byzantine_weight_mass_max": _maximum(rounds, "byzantine_weight_mass_oracle"),
        "max_client_weight": _maximum(rounds, "max_client_weight"),
        "weight_cap": final.get("far_weight_cap"),
        "weight_cap_all_rounds": all(
            bool(row.get("far_weight_cap_respected")) for row in rounds
        ),
        "tier_mean_max_error_median": _median(
            rounds, "far_noise_score_tier_mean_max_error"
        ),
        "tier_scale_covariance_median": _median(
            rounds, "far_noise_score_tier_midrank_scale_covariance"
        ),
        "channels_separated_all_rounds": all(
            bool(row.get("far_noise_score_channels_separated")) for row in rounds
        ),
        "novelty_fraction_median": _median(
            rounds, "far_noise_score_novelty_logit_fraction"
        ),
        "trust_fraction_median": _median(
            rounds, "far_noise_score_trust_logit_fraction"
        ),
        "test_accuracy_final_pp": _pct(final.get("test_accuracy")),
        "client_accuracy_final_pp": _pct(final.get("client_accuracy_mean")),
        "client_accuracy_variance_final_pp2": final.get(
            "client_accuracy_variance_pct2"
        ),
        "worst20_final_pp": _pct(final.get("worst20_accuracy")),
        "gap_final_pp": _pct(final.get("best20_worst20_gap")),
        "metrics_path": str(path),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _informative(row: dict[str, Any]) -> bool:
    corr = row["honest_score_clean_corr_median"]
    recall = row["honest_top20_recall_median"]
    noise_corr = row["honest_score_public_scale_corr_median"]
    return bool(
        corr is not None
        and corr >= 0.50
        and recall is not None
        and recall >= 0.60
        and noise_corr is not None
        and abs(noise_corr) <= 0.10
        and row["weight_cap_all_rounds"]
    )


def _profile_gates(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        by_method[row["method"]].append(row)

    no_trust = {(row["reference"], row["threat"]): row for row in by_method[NO_TRUST]}
    output = []
    for method, rows in sorted(by_method.items()):
        attacked = [row for row in rows if row["threat"] != "none"]
        no_attack = [row for row in rows if row["threat"] == "none"]
        bit_flip = [row for row in rows if row["threat"].startswith("bf")]
        trust_cells = []
        if method in {TIER_FAIRNESS, TIER_BALANCED}:
            for row in attacked:
                baseline = no_trust[(row["reference"], row["threat"])]
                reduction = float(baseline["byzantine_weight_mass_median"]) - float(
                    row["byzantine_weight_mass_median"]
                )
                recall_loss = float(baseline["honest_top20_recall_median"]) - float(
                    row["honest_top20_recall_median"]
                )
                trust_cells.append(reduction >= 0.02 and recall_loss <= 0.05)
        elif method == THEIL_SEN:
            trust_cells = []  # No matched Theil-Sen no-trust control was frozen.

        record = {
            "method": method,
            "profile": PROFILE_LABELS[method],
            "available_cells": len(rows),
            "informative_cells": sum(_informative(row) for row in rows),
            "no_attack_cells_informative": sum(_informative(row) for row in no_attack),
            "safe_attacked_cells": sum(
                float(row["byzantine_weight_mass_median"]) <= 0.20 + 1e-12
                for row in attacked
            ),
            "safe_bit_flip_cells": sum(
                float(row["byzantine_weight_mass_median"]) <= 0.20 + 1e-12
                for row in bit_flip
            ),
            "trust_criterion_cells": sum(trust_cells),
            "trust_criterion_cells_available": len(trust_cells),
            "all_caps_respected": all(row["weight_cap_all_rounds"] for row in rows),
        }
        record["promoted"] = bool(
            record["available_cells"] == 8
            and record["informative_cells"] >= 6
            and record["no_attack_cells_informative"] == 2
            and record["safe_attacked_cells"] >= 5
            and record["safe_bit_flip_cells"] == 2
            and (
                method == NO_TRUST
                or (
                    record["trust_criterion_cells_available"] == 6
                    and record["trust_criterion_cells"] == 6
                )
            )
        )
        output.append(record)
    return output


def _mean_by_profile(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "honest_score_clean_corr_median",
        "honest_top20_recall_median",
        "honest_score_clean_rmse_median",
        "honest_weight_clean_l1_median",
        "weight_concentration_median",
        "test_accuracy_final_pp",
        "client_accuracy_final_pp",
        "client_accuracy_variance_final_pp2",
        "worst20_final_pp",
        "gap_final_pp",
    )
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        by_method[row["method"]].append(row)
    output = []
    for method, rows in sorted(by_method.items()):
        record: dict[str, Any] = {
            "method": method,
            "profile": PROFILE_LABELS[method],
            "cells": len(rows),
        }
        for field in fields:
            values = [float(row[field]) for row in rows if row[field] is not None]
            record[f"{field}_mean"] = mean(values)
        noise_corrs = [
            abs(float(row["honest_score_public_scale_corr_median"])) for row in rows
        ]
        record["honest_score_public_scale_abs_corr_mean"] = mean(noise_corrs)
        attacked = [row for row in rows if row["threat"] != "none"]
        record["attacked_byzantine_weight_mass_mean"] = mean(
            float(row["byzantine_weight_mass_median"]) for row in attacked
        )
        output.append(record)
    return output


def _stage11_pairing(
    runs: list[dict[str, Any]], stage11_root: Path
) -> list[dict[str, Any]]:
    stage11_paths = sorted(stage11_root.rglob("metrics.json"))
    stage11 = [_load_stage11(path) for path in stage11_paths]
    controls = {
        (row["reference"], row["threat"]): row
        for row in stage11
        if row["method"] == "dp_far_current_null_mc_robust_oracle"
    }
    fields = (
        "honest_score_clean_corr_median",
        "honest_top20_recall_median",
        "honest_score_public_scale_abs_corr",
        "honest_score_clean_rmse_median",
        "honest_weight_clean_l1_median",
        "weight_concentration_median",
        "byzantine_weight_mass_median",
        "test_accuracy_final_pp",
        "client_accuracy_variance_final_pp2",
        "worst20_final_pp",
        "gap_final_pp",
    )
    output = []
    for row in runs:
        control = controls[(row["reference"], row["threat"])]
        current = dict(row)
        current["honest_score_public_scale_abs_corr"] = abs(
            float(row["honest_score_public_scale_corr_median"])
        )
        record: dict[str, Any] = {
            "method": row["method"],
            "profile": row["profile"],
            "reference": row["reference"],
            "threat": row["threat"],
            "pair_key_matches": row["pair_key"] == control["pair_key"],
        }
        for field in fields:
            left, right = current[field], control[field]
            record[f"delta_{field}"] = (
                float(left) - float(right)
                if left is not None and right is not None
                else None
            )
        output.append(record)
    return output


def _load_stage11(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = yaml.safe_load(
        _nearest(path, "resolved_config.yaml").read_text(encoding="utf-8")
    )
    axes = config["reproduction"]["axes"]
    rounds = payload["rounds"]
    final = rounds[-1]
    scale_corr = _median(rounds, "far_honest_score_public_noise_scale_corr_oracle")
    return {
        "method": str(axes["method"]),
        "reference": str(axes["reference"]),
        "threat": str(axes["threat"]),
        "pair_key": str(config["reproduction"]["randomness_pair_key"]),
        "honest_score_clean_corr_median": _median(
            rounds, "far_honest_noisy_clean_score_corr_oracle"
        ),
        "honest_top20_recall_median": _median(
            rounds, "far_honest_clean_top_tail_recall_oracle"
        ),
        "honest_score_public_scale_abs_corr": (
            abs(float(scale_corr)) if scale_corr is not None else None
        ),
        "honest_score_clean_rmse_median": _median(
            rounds, "far_honest_noisy_clean_score_rmse_oracle"
        ),
        "honest_weight_clean_l1_median": _median(
            rounds, "far_honest_noisy_clean_weight_l1_oracle"
        ),
        "weight_concentration_median": _median(
            rounds, "far_noise_amplification_vs_uniform"
        ),
        "byzantine_weight_mass_median": _median(rounds, "byzantine_weight_mass_oracle"),
        "test_accuracy_final_pp": _pct(final.get("test_accuracy")),
        "client_accuracy_variance_final_pp2": final.get(
            "client_accuracy_variance_pct2"
        ),
        "worst20_final_pp": _pct(final.get("worst20_accuracy")),
        "gap_final_pp": _pct(final.get("best20_worst20_gap")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(
            "results/dt_ldp_far/decisive/"
            "dt_ldp_far_decisive_stage12_generation5_tier_rank_n25_screen_v1"
        ),
    )
    parser.add_argument(
        "--stage11-root",
        type=Path,
        default=Path(
            "results/dt_ldp_far/decisive/"
            "dt_ldp_far_decisive_stage11_generation4_scores_n25_screen_v2"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/analysis/dt_ldp_far_n25_stage12_generation5"),
    )
    args = parser.parse_args()

    paths = sorted(args.results_root.rglob("metrics.json"))
    if len(paths) != EXPECTED_RUNS:
        raise SystemExit(f"Stage 12 incomplete: {len(paths)}/{EXPECTED_RUNS}")
    runs = [_load(path) for path in paths]
    gates = _profile_gates(runs)
    summaries = _mean_by_profile(runs)
    paired = _stage11_pairing(runs, args.stage11_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "run_metrics.csv", runs)
    _write_csv(args.output_dir / "profile_summary.csv", summaries)
    _write_csv(args.output_dir / "profile_gates.csv", gates)
    _write_csv(args.output_dir / "paired_vs_stage11_null_mc.csv", paired)

    status = {
        "completed_runs": len(runs),
        "expected_runs": EXPECTED_RUNS,
        "complete": len(runs) == EXPECTED_RUNS,
        "devices": sorted({row["device"] for row in runs}),
        "all_runs_have_six_rounds": all(row["num_rounds"] == 6 for row in runs),
        "all_weight_caps_respected": all(row["weight_cap_all_rounds"] for row in runs),
        "all_stage11_pair_keys_match": all(row["pair_key_matches"] for row in paired),
        "gates": gates,
    }
    (args.output_dir / "status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
