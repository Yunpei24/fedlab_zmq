#!/usr/bin/env python3
"""Evaluate the Stage-20C same-pipeline uniform-weight control."""

from __future__ import annotations

import argparse
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

DEFAULT_UNIFORM = ROOT / "results/dt_ldp_far/stage20c_fmnist_mechanism_control_v1"
DEFAULT_CANDIDATE = ROOT / "results/dt_ldp_far/stage20a_fmnist_rfa_support_screen_v1"
DEFAULT_CONFIG = ROOT / "configs/dt_ldp_far/stage20c_fmnist_mechanism_control.yaml"
DEFAULT_REPORT = ROOT / "output/analysis/DT_LDP_FAR_Stage20C_Mechanism_Control.md"


def _index(rows: list[dict[str, Any]], tilt: str) -> dict[str, dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["method"] == "dt_ldp_far_stage19_peer_support"
        and row["reference"] == "rfa"
        and row["tilt"] == tilt
    ]
    indexed = {str(row["threat"]): row for row in selected}
    expected = {"none", "ipm20_n25", "bf20_n25_s10"}
    if set(indexed) != expected:
        raise ValueError(f"Incomplete {tilt} arm: {sorted(indexed)}")
    return indexed


def analyze(
    uniform_root: Path, candidate_root: Path, config_path: Path, report_path: Path
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    uniform = _index(_collect(uniform_root), "uniform")
    candidate = _index(_collect(candidate_root), "boundary")
    attacks = ["ipm20_n25", "bf20_n25_s10"]

    accuracy_differences = {
        threat: float(
            candidate[threat]["test_accuracy_pct"]
            - uniform[threat]["test_accuracy_pct"]
        )
        for threat in uniform
    }
    worst20_difference = float(
        candidate["none"]["worst20_pct"] - uniform["none"]["worst20_pct"]
    )
    byzantine_mass_differences = {
        threat: float(
            candidate[threat]["median_byzantine_mass"]
            - uniform[threat]["median_byzantine_mass"]
        )
        for threat in attacks
    }
    attack_gain_mean = float(
        statistics.fmean(accuracy_differences[threat] for threat in attacks)
    )
    byzantine_mass_gain_mean = float(
        statistics.fmean(-byzantine_mass_differences[threat] for threat in attacks)
    )
    epsilon_difference_max = max(
        abs(
            float(candidate[threat]["epsilon_max"])
            - float(uniform[threat]["epsilon_max"])
        )
        for threat in uniform
    )
    gates = config["gates"]
    checks = {
        "no_attack_accuracy": accuracy_differences["none"]
        >= -float(gates["no_attack_test_accuracy_drop_vs_uniform_max_pp"]),
        "no_attack_worst20": worst20_difference
        >= -float(gates["no_attack_worst20_drop_vs_uniform_max_pp"]),
        "separated_accuracy_gain": attack_gain_mean
        >= float(gates["separated_test_accuracy_gain_vs_uniform_mean_min_pp"]),
        "byzantine_mass_not_above_uniform": all(
            byzantine_mass_differences[threat] <= 1e-12 for threat in attacks
        ),
        "byzantine_mass_gain": byzantine_mass_gain_mean
        >= float(gates["separated_byzantine_mass_gain_mean_min"]),
        "privacy": epsilon_difference_max
        <= float(gates["privacy_epsilon_difference_max"]),
        "weight_cap": all(
            bool(row["weight_cap_respected_all"]) for row in candidate.values()
        ),
    }
    observations = {
        "test_accuracy_difference_pp_boundary_minus_uniform": accuracy_differences,
        "no_attack_worst20_difference_pp_boundary_minus_uniform": worst20_difference,
        "separated_test_accuracy_gain_mean_pp": attack_gain_mean,
        "byzantine_mass_difference_boundary_minus_uniform": byzantine_mass_differences,
        "byzantine_mass_gain_mean": byzantine_mass_gain_mean,
        "privacy_epsilon_difference_max": epsilon_difference_max,
    }
    passed = all(checks.values())
    decision = {
        "completed_runs": 6,
        "checks": checks,
        "observations": observations,
        "thresholds": gates,
        "mechanism_screen_passed": passed,
        "interpretation": (
            "acceptable_clean_cost_and_reproducible_attack_benefit"
            if passed
            else "candidate_not_end_to_end_acceptable_on_seed_71"
        ),
    }
    uniform_root.mkdir(parents=True, exist_ok=True)
    decision_path = uniform_root / "stage20c_decision.json"
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Stage 20C — Contrôle mécanistique strictement apparié",
        "",
        "Les deux bras utilisent la même chaîne DT-LDP-FAR, le même clipping "
        "serveur, la même référence RFA, le même score et les mêmes seeds. Seul "
        "le tilt public change : `uniform` correspond à alpha = 0 et `boundary` "
        "à alpha = alpha_max.",
        "",
        "| Menace | Acc. uniforme | Acc. candidat | Diff. candidat−uniforme | Masse byz. uniforme | Masse byz. candidat |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for threat in ["none", *attacks]:
        lines.append(
            "| {threat} | {uniform_acc} % | {candidate_acc} % | {difference} pp | {uniform_mass} | {candidate_mass} |".format(
                threat=threat,
                uniform_acc=_fmt(uniform[threat]["test_accuracy_pct"], 2),
                candidate_acc=_fmt(candidate[threat]["test_accuracy_pct"], 2),
                difference=_fmt(accuracy_differences[threat], 2),
                uniform_mass=_fmt(uniform[threat]["median_byzantine_mass"], 4),
                candidate_mass=_fmt(candidate[threat]["median_byzantine_mass"], 4),
            )
        )
    lines.extend(["", "## Critères gelés", ""])
    for name, passed_check in checks.items():
        lines.append(f"- `{name}` : **{'pass' if passed_check else 'fail'}**.")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            (
                "Le candidat satisfait le screen mécanistique."
                if passed
                else "Le candidat ne satisfait pas le screen mécanistique sur la seed 71. "
                "La réplication indépendante Stage 20R doit déterminer si ce rejet est reproductible."
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
    parser.add_argument("--uniform-results", type=Path, default=DEFAULT_UNIFORM)
    parser.add_argument("--candidate-results", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    analyze(
        args.uniform_results.resolve(),
        args.candidate_results.resolve(),
        args.config.resolve(),
        args.report.resolve(),
    )


if __name__ == "__main__":
    main()
