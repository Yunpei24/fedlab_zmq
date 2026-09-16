#!/usr/bin/env python3
"""Analyze the frozen Stage-11 generation-4 score screen."""

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

EXPECTED_RUNS = 32
PRIMARY_METHOD = "dp_far_current_null_mc_robust_oracle"
LOO_METHOD = "dp_far_current_loo_excess_robust_oracle"
RAW_METHOD = "dp_far_current_matched_oracle"
ENERGY_METHOD = "dp_far_current_excess_energy_oracle"


def _nearest(path: Path, filename: str) -> Path:
    for parent in path.parents:
        candidate = parent / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {filename} above {path}")


def _finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float))
        and math.isfinite(float(row[key]))
    ]


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _finite_values(rows, key)
    return median(values) if values else None


def _maximum(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _finite_values(rows, key)
    return max(values) if values else None


def _pct(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    result = float(value)
    return 100.0 * result if abs(result) <= 1.0 else result


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
    final = rounds[-1]
    is_far = method.startswith("dp_far_current")
    return {
        "method": method,
        "reference": str(axes["reference"]),
        "threat": str(axes["threat"]),
        "score_mode": str(algo.get("noise_score_standardization", "uniform")),
        "partition_seed": int(axes["partition_seed"]),
        "training_seed": int(axes["training_seed"]),
        "pair_key": str(config["reproduction"]["randomness_pair_key"]),
        "epsilon_final": final.get("privacy_epsilon_max"),
        "honest_score_clean_corr_median": (
            _median(rounds, "far_honest_noisy_clean_score_corr_oracle")
            if is_far
            else None
        ),
        "honest_score_clean_rmse_median": (
            _median(rounds, "far_honest_noisy_clean_score_rmse_oracle")
            if is_far
            else None
        ),
        "honest_top20_recall_median": (
            _median(rounds, "far_honest_clean_top_tail_recall_oracle")
            if is_far
            else None
        ),
        "honest_score_public_scale_corr_median": (
            _median(rounds, "far_honest_score_public_noise_scale_corr_oracle")
            if is_far
            else None
        ),
        "global_score_clean_corr_median": (
            _median(rounds, "far_noisy_clean_score_corr_oracle")
            if is_far
            else None
        ),
        "honest_weight_clean_l1_median": (
            _median(rounds, "far_honest_noisy_clean_weight_l1_oracle")
            if is_far
            else None
        ),
        "score_span_median": _median(rounds, "far_score_span") if is_far else None,
        "score_saturation_median": (
            _median(rounds, "far_score_saturation_rate") if is_far else None
        ),
        "weight_concentration_median": (
            _median(rounds, "far_noise_amplification_vs_uniform")
            if is_far
            else 1.0
        ),
        "byzantine_weight_mass_median": _median(
            rounds, "byzantine_weight_mass_oracle"
        ),
        "byzantine_weight_mass_max": _maximum(
            rounds, "byzantine_weight_mass_oracle"
        ),
        "reference_honest_error_median": (
            _median(rounds, "far_reference_honest_center_error_oracle")
            if is_far
            else None
        ),
        "weight_cap_all_rounds": (
            all(bool(row.get("far_weight_cap_respected")) for row in rounds)
            if is_far
            else True
        ),
        "test_accuracy_final_pp": _pct(final.get("test_accuracy")),
        "client_accuracy_final_pp": _pct(final.get("client_accuracy_mean")),
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


def _summaries(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        groups[(row["method"], row["reference"], row["threat"])].append(row)
    fields = (
        "honest_score_clean_corr_median",
        "honest_score_clean_rmse_median",
        "honest_top20_recall_median",
        "honest_score_public_scale_corr_median",
        "honest_weight_clean_l1_median",
        "score_span_median",
        "score_saturation_median",
        "weight_concentration_median",
        "byzantine_weight_mass_median",
        "reference_honest_error_median",
        "test_accuracy_final_pp",
        "client_accuracy_final_pp",
        "worst20_final_pp",
        "gap_final_pp",
    )
    output = []
    for (method, reference, threat), rows in sorted(groups.items()):
        record: dict[str, Any] = {
            "method": method,
            "reference": reference,
            "threat": threat,
            "n": len(rows),
        }
        for field in fields:
            values = [float(row[field]) for row in rows if row[field] is not None]
            record[f"{field}_mean"] = mean(values) if values else None
            record[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0
        output.append(record)
    return output


def _paired(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in runs:
        groups[
            (
                row["reference"],
                row["threat"],
                row["partition_seed"],
                row["training_seed"],
            )
        ][row["method"]] = row
    fields = (
        "honest_score_clean_corr_median",
        "honest_score_clean_rmse_median",
        "honest_top20_recall_median",
        "honest_score_public_scale_corr_median",
        "honest_weight_clean_l1_median",
        "weight_concentration_median",
        "byzantine_weight_mass_median",
        "test_accuracy_final_pp",
        "worst20_final_pp",
        "gap_final_pp",
    )
    output = []
    for key, methods in sorted(groups.items()):
        for candidate_name in (PRIMARY_METHOD, LOO_METHOD):
            candidate = methods.get(candidate_name)
            if candidate is None:
                continue
            for control_name in (RAW_METHOD, ENERGY_METHOD):
                control = methods.get(control_name)
                if control is None:
                    continue
                row: dict[str, Any] = {
                    "candidate": candidate_name,
                    "control": control_name,
                    "reference": key[0],
                    "threat": key[1],
                    "partition_seed": key[2],
                    "training_seed": key[3],
                    "pair_key_matches": candidate["pair_key"] == control["pair_key"],
                }
                for field in fields:
                    left, right = candidate[field], control[field]
                    row[f"delta_{field}"] = (
                        float(left) - float(right)
                        if left is not None and right is not None
                        else None
                    )
                output.append(row)
    return output


def _gate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [row for row in runs if row["method"] == PRIMARY_METHOD]
    cells = []
    for row in primary:
        corr = row["honest_score_clean_corr_median"]
        recall = row["honest_top20_recall_median"]
        scale_corr = row["honest_score_public_scale_corr_median"]
        informative = bool(
            corr is not None
            and corr >= 0.50
            and recall is not None
            and recall >= 0.60
            and scale_corr is not None
            and abs(scale_corr) <= 0.10
            and row["weight_cap_all_rounds"]
        )
        cells.append(
            {
                "reference": row["reference"],
                "threat": row["threat"],
                "informative": informative,
                "honest_score_clean_corr_median": corr,
                "honest_top20_recall_median": recall,
                "honest_score_public_scale_corr_median": scale_corr,
                "weight_cap_all_rounds": row["weight_cap_all_rounds"],
            }
        )
    attacked = [row for row in primary if row["threat"] != "none"]
    safe_byzantine_cells = sum(
        row["byzantine_weight_mass_median"] is not None
        and row["byzantine_weight_mass_median"] <= 0.20 + 1e-12
        for row in attacked
    )
    informative_cells = sum(cell["informative"] for cell in cells)
    complete = len(primary) == 8
    promoted = bool(
        complete
        and informative_cells >= 6
        and len(attacked) == 6
        and safe_byzantine_cells >= 5
    )
    return {
        "primary_cells_available": len(primary),
        "primary_cells_expected": 8,
        "informative_cells": informative_cells,
        "informative_cells_required": 6,
        "attacked_cells_at_or_below_uniform_byzantine_mass": safe_byzantine_cells,
        "attacked_cells_expected": 6,
        "safe_byzantine_cells_required": 5,
        "promote_to_multiseed_confirmation": promoted,
        "cells": cells,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _report(path: Path, runs: list[dict[str, Any]], gate: dict[str, Any], *, complete: bool) -> None:
    lines = [
        "# Stage 11 — Résultats end-to-end des scores de génération 4",
        "",
        (
            "**Statut : campagne complète.**"
            if complete
            else f"**Statut provisoire : {len(runs)}/{EXPECTED_RUNS} runs complets.**"
        ),
        "",
        "Les nombres ci-dessous sont des observations. Les oracles sans bruit servent uniquement à l'audit expérimental et ne constituent pas un transcript DP publiable.",
        "",
        "| Méthode | Référence | Menace | Corr. score propre honnête | Rappel top-20 honnête | |Corr. score–bruit| | Masse byzantine | Test Acc. (pp) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(runs, key=lambda item: (item["method"], item["reference"], item["threat"])):
        scale = row["honest_score_public_scale_corr_median"]
        lines.append(
            "| {method} | {reference} | {threat} | {corr} | {recall} | {scale} | {byz} | {acc} |".format(
                method=row["method"],
                reference=row["reference"],
                threat=row["threat"],
                corr=_fmt(row["honest_score_clean_corr_median"]),
                recall=_fmt(row["honest_top20_recall_median"]),
                scale=_fmt(abs(scale) if scale is not None else None),
                byz=_fmt(row["byzantine_weight_mass_median"]),
                acc=_fmt(row["test_accuracy_final_pp"], 2),
            )
        )
    lines.extend(
        [
            "",
            "## Gate préenregistré",
            "",
            f"- Cellules principales informatives : **{gate['informative_cells']}/{gate['primary_cells_available']}** ; seuil de promotion : 6/8.",
            f"- Cellules attaquées dont la masse byzantine médiane ne dépasse pas 0,20 : **{gate['attacked_cells_at_or_below_uniform_byzantine_mass']}/{max(0, gate['attacked_cells_expected'] if complete else len([r for r in runs if r['method'] == PRIMARY_METHOD and r['threat'] != 'none']))}** ; seuil : 5/6.",
            f"- Décision automatique : **{'PROMOUVOIR' if gate['promote_to_multiseed_confirmation'] else 'NE PAS PROMOUVOIR À CE STADE'}**.",
            "",
            "L'accuracy à six tours est descriptive et n'intervient pas dans cette décision.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(
            "results/dt_ldp_far/decisive/"
            "dt_ldp_far_decisive_stage11_generation4_scores_n25_screen_v2"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/analysis/dt_ldp_far_n25_stage11_generation4"),
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--expected-runs", type=int, default=EXPECTED_RUNS)
    args = parser.parse_args()
    paths = sorted(args.results_root.rglob("metrics.json"))
    if len(paths) != args.expected_runs and not args.allow_partial:
        raise SystemExit(f"Stage 11 incomplete: {len(paths)}/{args.expected_runs}")
    runs = [_load(path) for path in paths]
    summaries = _summaries(runs)
    paired = _paired(runs)
    gate = _gate(runs)
    complete = len(runs) == args.expected_runs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "run_metrics.csv", runs)
    _write_csv(args.output_dir / "summary_by_method_reference_threat.csv", summaries)
    _write_csv(args.output_dir / "paired_score_comparisons.csv", paired)
    status = {
        "completed_runs": len(runs),
        "expected_runs": args.expected_runs,
        "complete": complete,
        "all_available_randomness_pairs_match": all(
            row["pair_key_matches"] for row in paired
        ),
        "all_available_far_weight_caps_respected": all(
            row["weight_cap_all_rounds"]
            for row in runs
            if row["method"].startswith("dp_far_current")
        ),
        "gate": gate,
    }
    (args.output_dir / "status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    _report(
        args.output_dir / "DT_LDP_FAR_Stage11_Generation4_Results.md",
        runs,
        gate,
        complete=complete,
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
