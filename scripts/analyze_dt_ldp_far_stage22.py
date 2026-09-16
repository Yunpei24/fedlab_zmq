#!/usr/bin/env python3
"""Evaluate the pre-registered Stage-22 lagged-descent holdout."""

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

DEFAULT_RESULTS = ROOT / "results/dt_ldp_far/stage22_lagged_descent_validation_v1"
DEFAULT_CONFIG = ROOT / "configs/dt_ldp_far/stage22_lagged_descent_validation.yaml"
DEFAULT_REPORT = (
    ROOT / "output/analysis/DT_LDP_FAR_Stage22_Lagged_Descent_Validation.md"
)


def _task_root(metrics_path: Path) -> Path:
    for parent in metrics_path.parents:
        if (parent / "dt_ldp_far_task_manifest.json").exists():
            return parent
    raise FileNotFoundError(f"No task manifest above {metrics_path}")


def _median(rounds: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rounds if row.get(key) is not None]
    return float(statistics.median(values)) if values else None


def _augment(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        metrics_path = Path(row["metrics_path"])
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        rounds = payload["rounds"]
        active_rounds = [
            item
            for item in rounds
            if item.get("dtldp_noise_score_lagged_descent_direction_available", False)
        ]
        resolved = yaml.safe_load(
            (_task_root(metrics_path) / "resolved_config.yaml").read_text(
                encoding="utf-8"
            )
        )
        row["randomness_pair_key"] = resolved["reproduction"]["randomness_pair_key"]
        row["median_lagged_trust_honest"] = _median(
            active_rounds, "dtldp_lagged_descent_trust_honest_oracle"
        )
        row["median_lagged_trust_byzantine"] = _median(
            active_rounds, "dtldp_lagged_descent_trust_byzantine_oracle"
        )
        row["median_lagged_trust_margin"] = _median(
            active_rounds, "dtldp_lagged_descent_trust_margin_oracle"
        )
        row["lagged_direction_active_rounds"] = len(active_rounds)


def _pair(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["method"] == "dt_ldp_far_stage22_lagged_descent"
        and row["reference"] == "rfa"
    ]
    indexed = {
        (int(row["training_seed"]), str(row["threat"]), str(row["tilt"])): row
        for row in selected
    }
    seeds = sorted({key[0] for key in indexed})
    paired: list[dict[str, Any]] = []
    for seed in seeds:
        for threat in ("none", "ipm20_n25", "bf20_n25_s10"):
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
                    "candidate_trust_honest": candidate["median_lagged_trust_honest"],
                    "candidate_trust_byzantine": candidate[
                        "median_lagged_trust_byzantine"
                    ],
                    "candidate_trust_margin": candidate["median_lagged_trust_margin"],
                    "candidate_active_direction_rounds": candidate[
                        "lagged_direction_active_rounds"
                    ],
                    "uniform_epsilon": uniform["epsilon_max"],
                    "candidate_epsilon": candidate["epsilon_max"],
                    "candidate_weight_cap_respected": candidate[
                        "weight_cap_respected_all"
                    ],
                    "uniform_randomness_pair_key": uniform["randomness_pair_key"],
                    "candidate_randomness_pair_key": candidate["randomness_pair_key"],
                    "randomness_pair_key_matches": uniform["randomness_pair_key"]
                    == candidate["randomness_pair_key"],
                }
            )
    if len(paired) != 12:
        raise ValueError(f"Expected twelve paired cells, got {len(paired)}")
    return paired


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


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
            f"Stage 22 incomplete: {len(rows)}/{config['expected_tasks']} runs"
        )
    _augment(rows)
    paired = _pair(rows)
    screen_seed = int(config["analysis_split"]["integration_screen_seed"])
    confirmatory_seeds = {
        int(seed) for seed in config["analysis_split"]["confirmatory_seeds"]
    }
    screen = [row for row in paired if row["seed"] == screen_seed]
    holdout = [row for row in paired if row["seed"] in confirmatory_seeds]
    if len(screen) != 3 or len(holdout) != 9:
        raise ValueError("Stage-22 screen/holdout split is incomplete")

    clean = [row for row in holdout if row["threat"] == "none"]
    attacked = [row for row in holdout if row["threat"] != "none"]
    by_attack = {
        attack: [row for row in attacked if row["threat"] == attack]
        for attack in ("ipm20_n25", "bf20_n25_s10")
    }
    gates = config["gates"]

    observations = {
        "screen_accuracy_differences_pp": {
            row["threat"]: row["accuracy_difference_pp"] for row in screen
        },
        "no_attack_accuracy_differences_pp": [
            row["accuracy_difference_pp"] for row in clean
        ],
        "no_attack_accuracy_difference_mean_pp": _mean(clean, "accuracy_difference_pp"),
        "separated_accuracy_differences_pp": [
            row["accuracy_difference_pp"] for row in attacked
        ],
        "separated_accuracy_gain_mean_pp": _mean(attacked, "accuracy_difference_pp"),
        "separated_positive_accuracy_gain_rate": statistics.fmean(
            row["accuracy_difference_pp"] > 0.0 for row in attacked
        ),
        "separated_worst20_difference_mean_pp": _mean(
            attacked, "worst20_difference_pp"
        ),
        "accuracy_gain_mean_by_attack_pp": {
            attack: _mean(group, "accuracy_difference_pp")
            for attack, group in by_attack.items()
        },
        "candidate_byzantine_mass_mean_by_attack": {
            attack: _mean(group, "candidate_byzantine_mass")
            for attack, group in by_attack.items()
        },
        "lagged_descent_trust_margin_mean_by_attack": {
            attack: _mean(group, "candidate_trust_margin")
            for attack, group in by_attack.items()
        },
        "candidate_concentration_median": statistics.median(
            float(row["candidate_concentration"]) for row in attacked
        ),
        "privacy_epsilon_difference_max": max(
            abs(float(row["candidate_epsilon"]) - float(row["uniform_epsilon"]))
            for row in paired
        ),
        "strict_randomness_pair_rate": statistics.fmean(
            bool(row["randomness_pair_key_matches"]) for row in paired
        ),
    }

    checks = {
        "no_attack_accuracy_mean": observations["no_attack_accuracy_difference_mean_pp"]
        >= float(gates["no_attack_test_accuracy_difference_mean_min_pp"]),
        "no_attack_accuracy_each": min(
            observations["no_attack_accuracy_differences_pp"]
        )
        >= float(gates["no_attack_test_accuracy_difference_each_min_pp"]),
        "separated_accuracy_gain": observations["separated_accuracy_gain_mean_pp"]
        >= float(gates["separated_test_accuracy_gain_mean_min_pp"]),
        "separated_accuracy_each_attack": min(
            observations["accuracy_gain_mean_by_attack_pp"].values()
        )
        >= float(gates["separated_test_accuracy_gain_each_attack_mean_min_pp"]),
        "separated_positive_gain_rate": observations[
            "separated_positive_accuracy_gain_rate"
        ]
        >= float(gates["separated_positive_accuracy_gain_rate_min"]),
        "separated_worst20": observations["separated_worst20_difference_mean_pp"]
        >= float(gates["separated_worst20_difference_mean_min_pp"]),
        "byzantine_mass_each_attack": max(
            observations["candidate_byzantine_mass_mean_by_attack"].values()
        )
        <= float(gates["byzantine_weight_mass_each_attack_mean_max"]),
        "lagged_descent_trust_margin_each_attack": min(
            observations["lagged_descent_trust_margin_mean_by_attack"].values()
        )
        >= float(gates["lagged_descent_trust_margin_each_attack_mean_min"]),
        "candidate_nonuniformity": observations["candidate_concentration_median"]
        >= float(gates["candidate_concentration_median_min"]),
        "privacy": observations["privacy_epsilon_difference_max"]
        <= float(gates["privacy_epsilon_difference_max"]),
        "weight_cap": all(
            bool(row["candidate_weight_cap_respected"]) for row in paired
        ),
        "strict_randomness_pairing": observations["strict_randomness_pair_rate"] == 1.0,
    }
    validated = all(checks.values())
    decision = {
        "completed_runs": len(rows),
        "integration_screen_seed": screen_seed,
        "confirmatory_seeds": sorted(confirmatory_seeds),
        "thresholds": gates,
        "observations": observations,
        "checks": checks,
        "lagged_descent_hypothesis_validated": validated,
        "claim_scope": (
            "conditional_lagged_robust_descent_alignment"
            if validated
            else "lagged_descent_hypothesis_not_confirmed"
        ),
    }
    results.mkdir(parents=True, exist_ok=True)
    _write_csv(results / "stage22_paired_results.csv", paired)
    decision_path = results / "stage22_decision.json"
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Stage 22 — Validation de l'alignement avec la descente retardée",
        "",
        "Le score combine 25 % de nouveauté covariance-calibrée et 75 % "
        "d'alignement avec la référence RFA du tour précédent. La seed 163 "
        "est un screen d'intégration ; seules les seeds 179, 193 et 211 "
        "déterminent le verdict confirmatoire.",
        "",
        "| Bloc | Seed | Menace | Acc. uniforme | Acc. candidat | Diff. | Worst-20 diff. | Masse byz. | Marge trust H−B | Concentration |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            "| {block} | {seed} | {threat} | {uniform} % | {candidate} % | "
            "{diff} pp | {worst} pp | {mass} | {margin} | {concentration} |".format(
                block="screen" if row["seed"] == screen_seed else "holdout",
                seed=row["seed"],
                threat=row["threat"],
                uniform=_fmt(row["uniform_test_accuracy_pct"], 2),
                candidate=_fmt(row["candidate_test_accuracy_pct"], 2),
                diff=_fmt(row["accuracy_difference_pp"], 2),
                worst=_fmt(row["worst20_difference_pp"], 2),
                mass=_fmt(row["candidate_byzantine_mass"], 4),
                margin=_fmt(row["candidate_trust_margin"], 4),
                concentration=_fmt(row["candidate_concentration"], 4),
            )
        )
    lines.extend(["", "## Gates confirmatoires", ""])
    for name, value in checks.items():
        lines.append(f"- `{name}` : **{'pass' if value else 'fail'}**.")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            (
                "L'hypothèse d'alignement avec la descente retardée est "
                "confirmée sous tous les critères préenregistrés."
                if validated
                else "L'hypothèse d'alignement avec la descente retardée "
                "n'est pas confirmée sous tous les critères préenregistrés."
            ),
            "",
            "Cette décision ne constitue pas une preuve de diminution de la "
            "loss ni une garantie byzantine universelle.",
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
