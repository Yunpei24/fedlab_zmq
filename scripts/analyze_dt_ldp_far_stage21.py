#!/usr/bin/env python3
"""Evaluate the pre-registered Stage-21 IPM-specific confirmation."""

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

DEFAULT_RESULTS = ROOT / "results/dt_ldp_far/stage21_ipm_specific_confirmatory_v1"
DEFAULT_CONFIG = ROOT / "configs/dt_ldp_far/stage21_ipm_specific_confirmatory.yaml"
DEFAULT_REPORT = ROOT / "output/analysis/DT_LDP_FAR_Stage21_IPM_Confirmatory.md"


def _pair(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (int(row["training_seed"]), str(row["threat"]), str(row["tilt"])): row
        for row in rows
        if row["method"] == "dt_ldp_far_stage19_peer_support"
        and row["reference"] == "rfa"
    }
    seeds = sorted({key[0] for key in indexed})
    paired: list[dict[str, Any]] = []
    for seed in seeds:
        for threat in ("none", "ipm20_n25"):
            try:
                uniform = indexed[(seed, threat, "uniform")]
                candidate = indexed[(seed, threat, "boundary")]
            except KeyError as exc:
                raise ValueError(f"Missing paired cell {exc.args[0]}") from exc
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
                    "uniform_byzantine_mass": uniform["median_byzantine_mass"],
                    "candidate_byzantine_mass": candidate["median_byzantine_mass"],
                    "candidate_concentration": candidate["median_concentration"],
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
            f"Stage 21 incomplete: {len(rows)}/{config['expected_tasks']} runs"
        )
    paired = _pair(rows)
    clean = [row for row in paired if row["threat"] == "none"]
    ipm = [row for row in paired if row["threat"] == "ipm20_n25"]
    gates = config["gates"]

    observations = {
        "no_attack_test_accuracy_differences_pp": [
            row["accuracy_difference_pp"] for row in clean
        ],
        "no_attack_test_accuracy_difference_mean_pp": statistics.fmean(
            row["accuracy_difference_pp"] for row in clean
        ),
        "ipm_test_accuracy_differences_pp": [
            row["accuracy_difference_pp"] for row in ipm
        ],
        "ipm_test_accuracy_gain_mean_pp": statistics.fmean(
            row["accuracy_difference_pp"] for row in ipm
        ),
        "ipm_positive_accuracy_gain_rate": statistics.fmean(
            row["accuracy_difference_pp"] > 0.0 for row in ipm
        ),
        "ipm_worst20_difference_mean_pp": statistics.fmean(
            row["worst20_difference_pp"] for row in ipm
        ),
        "ipm_candidate_byzantine_masses": [
            row["candidate_byzantine_mass"] for row in ipm
        ],
        "ipm_candidate_byzantine_mass_mean": statistics.fmean(
            float(row["candidate_byzantine_mass"]) for row in ipm
        ),
        "ipm_candidate_concentration_median": statistics.median(
            float(row["candidate_concentration"]) for row in ipm
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
        "ipm_accuracy_gain": observations["ipm_test_accuracy_gain_mean_pp"]
        >= float(gates["ipm_test_accuracy_gain_mean_min_pp"]),
        "ipm_positive_gain_rate": observations["ipm_positive_accuracy_gain_rate"]
        >= float(gates["ipm_positive_accuracy_gain_rate_min"]),
        "ipm_worst20": observations["ipm_worst20_difference_mean_pp"]
        >= float(gates["ipm_worst20_difference_mean_min_pp"]),
        "ipm_byzantine_mass_mean": observations["ipm_candidate_byzantine_mass_mean"]
        <= float(gates["ipm_byzantine_weight_mass_mean_max"]),
        "ipm_byzantine_mass_each": max(
            float(value) for value in observations["ipm_candidate_byzantine_masses"]
        )
        <= float(gates["ipm_byzantine_weight_mass_each_max"]),
        "ipm_nonuniformity": observations["ipm_candidate_concentration_median"]
        >= float(gates["ipm_candidate_concentration_median_min"]),
        "privacy": observations["privacy_epsilon_difference_max"]
        <= float(gates["privacy_epsilon_difference_max"]),
        "weight_cap": all(
            bool(row["candidate_weight_cap_respected"]) for row in paired
        ),
    }
    validated = all(checks.values())
    decision = {
        "completed_runs": len(rows),
        "confirmatory_seeds": sorted({row["seed"] for row in paired}),
        "thresholds": gates,
        "observations": observations,
        "checks": checks,
        "ipm_specific_hypothesis_validated": validated,
        "claim_scope": (
            "conditional_ipm_specific_not_universal_byzantine_robustness"
            if validated
            else "ipm_specific_hypothesis_not_confirmed"
        ),
    }
    results.mkdir(parents=True, exist_ok=True)
    _write_csv(results / "stage21_paired_results.csv", paired)
    decision_path = results / "stage21_decision.json"
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Stage 21 — Confirmation indépendante de l'hypothèse IPM",
        "",
        "Cette expérience confirme ou réfute sur trois seeds inédites une "
        "hypothèse générée par les Stages 20C/20R. Elle porte uniquement sur "
        "IPM à 20 % et ne revendique aucune robustesse byzantine universelle.",
        "",
        "| Seed | Menace | Acc. uniforme | Acc. candidat | Diff. | Worst-20 diff. | Masse byz. candidat | Concentration |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            "| {seed} | {threat} | {uniform} % | {candidate} % | {diff} pp | {worst} pp | {mass} | {concentration} |".format(
                seed=row["seed"],
                threat=row["threat"],
                uniform=_fmt(row["uniform_test_accuracy_pct"], 2),
                candidate=_fmt(row["candidate_test_accuracy_pct"], 2),
                diff=_fmt(row["accuracy_difference_pp"], 2),
                worst=_fmt(row["worst20_difference_pp"], 2),
                mass=_fmt(row["candidate_byzantine_mass"], 4),
                concentration=_fmt(row["candidate_concentration"], 4),
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
                "L'hypothèse IPM spécifique est confirmée selon tous les critères gelés."
                if validated
                else "L'hypothèse IPM spécifique n'est pas confirmée selon tous les critères gelés."
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
