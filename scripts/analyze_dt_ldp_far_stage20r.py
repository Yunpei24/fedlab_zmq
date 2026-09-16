#!/usr/bin/env python3
"""Evaluate the independent Stage-20R end-to-end mechanism replication."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_dt_ldp_far_stage14b_fmnist_temporal import _collect, _fmt

DEFAULT_RESULTS = ROOT / "results/dt_ldp_far/stage20r_fmnist_mechanism_replication_v1"
DEFAULT_CONFIG = ROOT / "configs/dt_ldp_far/stage20r_fmnist_mechanism_replication.yaml"
DEFAULT_REPORT = ROOT / "output/analysis/DT_LDP_FAR_Stage20R_Independent_Replication.md"

THREATS = ("none", "ipm20_n25", "bf20_n25_s10")
ATTACKS = THREATS[1:]


def _paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (int(row["training_seed"]), str(row["threat"]), str(row["tilt"])): row
        for row in rows
        if row["method"] == "dt_ldp_far_stage19_peer_support"
        and row["reference"] == "rfa"
    }
    seeds = sorted({key[0] for key in indexed})
    paired: list[dict[str, Any]] = []
    for seed in seeds:
        for threat in THREATS:
            try:
                uniform = indexed[(seed, threat, "uniform")]
                candidate = indexed[(seed, threat, "boundary")]
            except KeyError as exc:
                raise ValueError(f"Missing paired cell {exc.args[0]}") from exc
            uniform_mass = uniform["median_byzantine_mass"]
            candidate_mass = candidate["median_byzantine_mass"]
            paired.append(
                {
                    "seed": seed,
                    "threat": threat,
                    "uniform_test_accuracy_pct": uniform["test_accuracy_pct"],
                    "candidate_test_accuracy_pct": candidate["test_accuracy_pct"],
                    "accuracy_difference_pp": candidate["test_accuracy_pct"]
                    - uniform["test_accuracy_pct"],
                    "uniform_worst20_pct": uniform["worst20_pct"],
                    "candidate_worst20_pct": candidate["worst20_pct"],
                    "worst20_difference_pp": candidate["worst20_pct"]
                    - uniform["worst20_pct"],
                    "uniform_byzantine_mass": uniform_mass,
                    "candidate_byzantine_mass": candidate_mass,
                    "byzantine_mass_difference": (
                        None
                        if uniform_mass is None or candidate_mass is None
                        else float(candidate_mass) - float(uniform_mass)
                    ),
                    "uniform_epsilon": uniform["epsilon_max"],
                    "candidate_epsilon": candidate["epsilon_max"],
                    "candidate_weight_cap_respected": candidate[
                        "weight_cap_respected_all"
                    ],
                }
            )
    if len(paired) != 6:
        raise ValueError(f"Expected six paired cells, got {len(paired)}")
    return paired


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(results: Path, config_path: Path, report_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows = _collect(results)
    if len(rows) != int(config["expected_tasks"]):
        raise ValueError(
            f"Stage 20R incomplete: {len(rows)}/{config['expected_tasks']} runs"
        )
    paired = _paired_rows(rows)
    clean = [row for row in paired if row["threat"] == "none"]
    attacked = [row for row in paired if row["threat"] in ATTACKS]
    gates = config["gates"]

    observations = {
        "no_attack_test_accuracy_differences_pp": [
            row["accuracy_difference_pp"] for row in clean
        ],
        "no_attack_test_accuracy_difference_mean_pp": statistics.fmean(
            row["accuracy_difference_pp"] for row in clean
        ),
        "no_attack_worst20_difference_mean_pp": statistics.fmean(
            row["worst20_difference_pp"] for row in clean
        ),
        "separated_test_accuracy_differences_pp": [
            row["accuracy_difference_pp"] for row in attacked
        ],
        "separated_test_accuracy_gain_mean_pp": statistics.fmean(
            row["accuracy_difference_pp"] for row in attacked
        ),
        "separated_positive_accuracy_gain_rate": statistics.fmean(
            row["accuracy_difference_pp"] > 0.0 for row in attacked
        ),
        "separated_candidate_byzantine_mass_mean": statistics.fmean(
            float(row["candidate_byzantine_mass"]) for row in attacked
        ),
        "separated_byzantine_mass_not_above_uniform_rate": statistics.fmean(
            float(row["candidate_byzantine_mass"])
            <= float(row["uniform_byzantine_mass"]) + 1e-12
            for row in attacked
        ),
        "privacy_epsilon_difference_max": max(
            abs(float(row["candidate_epsilon"]) - float(row["uniform_epsilon"]))
            for row in paired
        ),
    }
    checks = {
        "no_attack_accuracy_mean": observations[
            "no_attack_test_accuracy_difference_mean_pp"
        ]
        >= float(gates["no_attack_test_accuracy_difference_mean_min_pp"]),
        "no_attack_accuracy_each": min(
            observations["no_attack_test_accuracy_differences_pp"]
        )
        >= float(gates["no_attack_test_accuracy_difference_each_min_pp"]),
        "no_attack_worst20_mean": observations["no_attack_worst20_difference_mean_pp"]
        >= float(gates["no_attack_worst20_difference_mean_min_pp"]),
        "separated_accuracy_gain": observations["separated_test_accuracy_gain_mean_pp"]
        >= float(gates["separated_test_accuracy_gain_mean_min_pp"]),
        "separated_positive_gain_rate": observations[
            "separated_positive_accuracy_gain_rate"
        ]
        >= float(gates["separated_positive_accuracy_gain_rate_min"]),
        "byzantine_mass_mean": observations["separated_candidate_byzantine_mass_mean"]
        <= float(gates["separated_byzantine_weight_mass_mean_max"]),
        "byzantine_mass_paired_rate": observations[
            "separated_byzantine_mass_not_above_uniform_rate"
        ]
        >= float(gates["separated_byzantine_mass_not_above_uniform_rate_min"]),
        "privacy": observations["privacy_epsilon_difference_max"]
        <= float(gates["privacy_epsilon_difference_max"]),
        "weight_cap": all(
            bool(row["candidate_weight_cap_respected"]) for row in paired
        ),
    }
    passed = all(checks.values())
    decision = {
        "completed_runs": len(rows),
        "independent_seeds": sorted({row["seed"] for row in paired}),
        "thresholds": gates,
        "observations": observations,
        "checks": checks,
        "end_to_end_candidate_validated": passed,
        "negative_validation": not passed,
    }
    results.mkdir(parents=True, exist_ok=True)
    _write_csv(results / "stage20r_paired_results.csv", paired)
    decision_path = results / "stage20r_decision.json"
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Stage 20R — Réplication indépendante du verdict end-to-end",
        "",
        "Cette réplication utilise deux nouvelles seeds qui n'ont pas servi à "
        "sélectionner le candidat. Les différences sont strictement appariées : "
        "candidat à alpha_max moins la même chaîne à alpha = 0.",
        "",
        "| Seed | Menace | Acc. uniforme | Acc. candidat | Diff. | Worst-20 diff. | Masse byz. uniforme | Masse byz. candidat |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            "| {seed} | {threat} | {uniform} % | {candidate} % | {diff} pp | {worst} pp | {uniform_mass} | {candidate_mass} |".format(
                seed=row["seed"],
                threat=row["threat"],
                uniform=_fmt(row["uniform_test_accuracy_pct"], 2),
                candidate=_fmt(row["candidate_test_accuracy_pct"], 2),
                diff=_fmt(row["accuracy_difference_pp"], 2),
                worst=_fmt(row["worst20_difference_pp"], 2),
                uniform_mass=_fmt(row["uniform_byzantine_mass"], 4),
                candidate_mass=_fmt(row["candidate_byzantine_mass"], 4),
            )
        )
    lines.extend(["", "## Gates préenregistrés", ""])
    for name, value in checks.items():
        lines.append(f"- `{name}` : **{'pass' if value else 'fail'}**.")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            (
                "Le candidat est validé end-to-end sur les critères gelés."
                if passed
                else "Le candidat est réfuté dans cette configuration end-to-end : "
                "la validation vectorielle du Stage 19 ne se transfère pas selon les critères gelés."
            ),
            "",
            f"Décision machine : `{decision_path.resolve()}`",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    analyze(args.results.resolve(), args.config.resolve(), args.report.resolve())


if __name__ == "__main__":
    main()
