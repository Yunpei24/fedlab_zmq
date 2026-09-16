#!/usr/bin/env python3
"""Evaluate the frozen Stage-20A operational gates and write its decision."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_dt_ldp_far_stage14b_fmnist_temporal import _collect

DEFAULT_RESULTS = ROOT / "results/dt_ldp_far/stage20a_fmnist_rfa_support_screen_v1"
DEFAULT_CONFIG = ROOT / "configs/dt_ldp_far/stage20a_fmnist_rfa_support_screen.yaml"
DEFAULT_REPORT = ROOT / "output/analysis/DT_LDP_FAR_Stage20A_FashionMNIST_Screen.md"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _row(rows: list[dict[str, Any]], method: str, reference: str, threat: str):
    selected = [
        row
        for row in rows
        if row["method"] == method
        and row["reference"] == reference
        and row["threat"] == threat
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one row for {method}/{reference}/{threat}, got {len(selected)}"
        )
    return selected[0]


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}".replace(".", ",")


def analyze(results: Path, config_path: Path, report_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows = _collect(results)
    if len(rows) != int(config["expected_tasks"]):
        raise ValueError(
            f"Stage 20A is incomplete: {len(rows)}/{config['expected_tasks']} metrics"
        )
    candidate_method = "dt_ldp_far_stage19_peer_support"
    baseline_method = "dp_fedavg"
    threats = ["none", "ipm20_n25", "bf20_n25_s10"]
    candidate = {
        threat: _row(rows, candidate_method, "rfa", threat) for threat in threats
    }
    baseline = {
        threat: _row(rows, baseline_method, "fcc", threat) for threat in threats
    }
    fcc_candidate = {
        threat: _row(rows, candidate_method, "fcc", threat) for threat in threats
    }
    gates = config["gates"]
    separated = threats[1:]
    epsilon_differences = [
        abs(
            float(candidate[threat]["epsilon_max"])
            - float(baseline[threat]["epsilon_max"])
        )
        for threat in threats
    ]
    rfa_errors = [
        float(candidate[threat]["median_reference_error"])
        for threat in threats
        if candidate[threat]["median_reference_error"] is not None
    ]
    fcc_errors = [
        float(fcc_candidate[threat]["median_reference_error"])
        for threat in threats
        if fcc_candidate[threat]["median_reference_error"] is not None
    ]
    observations = {
        "no_attack_test_accuracy_difference_pp": float(
            candidate["none"]["test_accuracy_pct"]
            - baseline["none"]["test_accuracy_pct"]
        ),
        "no_attack_worst20_difference_pp": float(
            candidate["none"]["worst20_pct"] - baseline["none"]["worst20_pct"]
        ),
        "separated_test_accuracy_differences_pp": {
            threat: float(
                candidate[threat]["test_accuracy_pct"]
                - baseline[threat]["test_accuracy_pct"]
            )
            for threat in separated
        },
        "separated_test_accuracy_difference_mean_pp": float(
            statistics.fmean(
                candidate[threat]["test_accuracy_pct"]
                - baseline[threat]["test_accuracy_pct"]
                for threat in separated
            )
        ),
        "separated_byzantine_weight_mass_median": {
            threat: candidate[threat]["median_byzantine_mass"] for threat in separated
        },
        "privacy_epsilon_difference_max": float(max(epsilon_differences)),
        "rfa_reference_error_mean": (
            float(statistics.fmean(rfa_errors)) if rfa_errors else None
        ),
        "fcc_reference_error_mean": (
            float(statistics.fmean(fcc_errors)) if fcc_errors else None
        ),
    }
    checks = {
        "no_attack_accuracy": candidate["none"]["test_accuracy_pct"]
        >= baseline["none"]["test_accuracy_pct"]
        - float(gates["no_attack_test_accuracy_drop_vs_dp_fedavg_max_pp"]),
        "no_attack_worst20": candidate["none"]["worst20_pct"]
        >= baseline["none"]["worst20_pct"]
        - float(gates["no_attack_worst20_drop_vs_dp_fedavg_max_pp"]),
        "separated_accuracy": statistics.fmean(
            candidate[threat]["test_accuracy_pct"]
            - baseline[threat]["test_accuracy_pct"]
            for threat in separated
        )
        >= -float(gates["separated_test_accuracy_drop_vs_dp_fedavg_mean_max_pp"]),
        "byzantine_mass": all(
            candidate[threat]["median_byzantine_mass"] is not None
            and float(candidate[threat]["median_byzantine_mass"])
            <= float(gates["separated_byzantine_weight_mass_median_max"])
            for threat in separated
        ),
        "privacy": max(epsilon_differences)
        <= float(gates["privacy_epsilon_difference_max"]),
        "weight_cap": all(
            bool(row["weight_cap_respected_all"]) for row in candidate.values()
        ),
        "rfa_reference": bool(rfa_errors)
        and bool(fcc_errors)
        and statistics.fmean(rfa_errors)
        <= statistics.fmean(fcc_errors)
        + float(gates["rfa_reference_error_excess_over_fcc_max"]),
    }
    passed = all(checks.values())
    decision = {
        "config": str(config_path.resolve()),
        "completed_runs": len(rows),
        "thresholds": gates,
        "observations": observations,
        "checks": checks,
        "screen_passed": passed,
        "stage20b_authorized": passed,
    }
    output_dir = results
    _write_csv(output_dir / "stage20a_analysis.csv", rows)
    (output_dir / "stage20a_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Stage 20A — Screen Fashion-MNIST du candidat confirmé",
        "",
        "## Portée",
        "",
        "Ce screen d'une seed vérifie l'intégration et l'absence d'échec "
        "end-to-end évident. Il ne constitue pas le résultat final d'accuracy. "
        "Les seuils ont été figés avant tout résultat du candidat.",
        "",
        "| Méthode | F | Menace | Test Acc. | Worst-20 | Gap | Masse byz. médiane | Poids max médian | epsilon |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {reference} | {threat} | {acc} % | {worst} % | {gap} pp | {byz} | {weight} | {epsilon} |".format(
                method=row["method"],
                reference=str(row["reference"]).upper(),
                threat=row["threat"],
                acc=_fmt(row["test_accuracy_pct"], 2),
                worst=_fmt(row["worst20_pct"], 2),
                gap=_fmt(row["gap_pp"], 2),
                byz=_fmt(row["median_byzantine_mass"], 4),
                weight=_fmt(row["median_max_weight"], 4),
                epsilon=_fmt(row["epsilon_max"], 4),
            )
        )
    lines.extend(["", "## Gates opérationnels", ""])
    for name, value in checks.items():
        lines.append(f"- `{name}` : **{'pass' if value else 'fail'}**.")
    lines.extend(
        [
            "",
            "## Décision",
            "",
            (
                "Stage 20A passe. La confirmation multi-seeds Stage 20B est autorisée."
                if passed
                else "Stage 20A échoue. Stage 20B ne doit pas être lancé."
            ),
            "",
            f"Décision machine : `{(output_dir / 'stage20a_decision.json').resolve()}`",
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
