#!/usr/bin/env python3
"""Evaluate the pre-registered Stage-23 robust-admissibility holdout."""

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

DEFAULT_RESULTS = ROOT / "results/dt_ldp_far/stage23_robust_admissibility_validation_v1"
DEFAULT_CONFIG = (
    ROOT / "configs/dt_ldp_far/stage23_robust_admissibility_validation.yaml"
)
DEFAULT_REPORT = (
    ROOT / "output/analysis/DT_LDP_FAR_Stage23_Robust_Admissibility_Validation.md"
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
        active = [
            item
            for item in rounds
            if item.get("dtldp_admissibility_filter_active", False)
        ]
        resolved = yaml.safe_load(
            (_task_root(metrics_path) / "resolved_config.yaml").read_text(
                encoding="utf-8"
            )
        )
        row["randomness_pair_key"] = resolved["reproduction"]["randomness_pair_key"]
        row["median_admissible_byzantine_count"] = _median(
            active, "dtldp_admissible_byzantine_count_oracle"
        )
        row["median_admissibility_trust_margin"] = _median(
            active, "dtldp_admissibility_trust_margin_oracle"
        )
        num_byzantine = max(
            (int(item.get("num_byzantine_oracle", 0)) for item in active), default=0
        )
        row["median_byzantine_eligibility_rate"] = (
            row["median_admissible_byzantine_count"] / num_byzantine
            if num_byzantine and row["median_admissible_byzantine_count"] is not None
            else 0.0
        )
        row["filter_active_rounds"] = len(active)


def _triplets(
    rows: list[dict[str, Any]],
    *,
    control_method: str,
    candidate_method: str,
    stage_label: str,
) -> list[dict[str, Any]]:
    full = {
        (int(row["training_seed"]), str(row["threat"])): row
        for row in rows
        if row["method"] == control_method and row["tilt"] == "uniform"
    }
    filtered = {
        (int(row["training_seed"]), str(row["threat"]), str(row["tilt"])): row
        for row in rows
        if row["method"] == candidate_method
    }
    seeds = sorted({key[0] for key in full})
    result: list[dict[str, Any]] = []
    for seed in seeds:
        for threat in ("none", "ipm20_n25", "bf20_n25_s10"):
            key = (seed, threat)
            try:
                control = full[key]
                filter_uniform = filtered[(seed, threat, "uniform")]
                candidate = filtered[(seed, threat, "boundary")]
            except KeyError as exc:
                raise ValueError(f"Missing {stage_label} cell {exc.args[0]}") from exc
            keys = {
                control["randomness_pair_key"],
                filter_uniform["randomness_pair_key"],
                candidate["randomness_pair_key"],
            }
            result.append(
                {
                    "seed": seed,
                    "threat": threat,
                    "full_uniform_accuracy_pct": control["test_accuracy_pct"],
                    "filtered_uniform_accuracy_pct": filter_uniform[
                        "test_accuracy_pct"
                    ],
                    "candidate_accuracy_pct": candidate["test_accuracy_pct"],
                    "integrated_accuracy_difference_pp": candidate["test_accuracy_pct"]
                    - control["test_accuracy_pct"],
                    "filter_accuracy_difference_pp": filter_uniform["test_accuracy_pct"]
                    - control["test_accuracy_pct"],
                    "tilt_accuracy_difference_pp": candidate["test_accuracy_pct"]
                    - filter_uniform["test_accuracy_pct"],
                    "integrated_worst20_difference_pp": candidate["worst20_pct"]
                    - control["worst20_pct"],
                    "tilt_worst20_difference_pp": candidate["worst20_pct"]
                    - filter_uniform["worst20_pct"],
                    "candidate_byzantine_mass": candidate["median_byzantine_mass"],
                    "candidate_byzantine_eligibility_rate": candidate[
                        "median_byzantine_eligibility_rate"
                    ],
                    "candidate_trust_margin": candidate[
                        "median_admissibility_trust_margin"
                    ],
                    "candidate_concentration": candidate["median_concentration"],
                    "candidate_filter_active_rounds": candidate["filter_active_rounds"],
                    "control_epsilon": control["epsilon_max"],
                    "filtered_uniform_epsilon": filter_uniform["epsilon_max"],
                    "candidate_epsilon": candidate["epsilon_max"],
                    "candidate_weight_cap_respected": candidate[
                        "weight_cap_respected_all"
                    ],
                    "randomness_pair_key": candidate["randomness_pair_key"],
                    "randomness_pair_key_matches": len(keys) == 1,
                }
            )
    if len(result) != 12:
        raise ValueError(f"Expected twelve {stage_label} triplets, got {len(result)}")
    return result


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(results: Path, config_path: Path, report_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    analysis_config = config.get("analysis", {})
    stage_label = str(analysis_config.get("stage_label", "Stage 23"))
    artifact_prefix = str(analysis_config.get("artifact_prefix", "stage23"))
    control_method = str(
        analysis_config.get("control_method", "dt_ldp_far_stage19_peer_support")
    )
    candidate_method = str(
        analysis_config.get("candidate_method", "dt_ldp_far_stage23_admissible_far")
    )
    rows = _collect(results)
    if len(rows) != int(config["expected_tasks"]):
        raise ValueError(
            f"{stage_label} incomplete: {len(rows)}/{config['expected_tasks']} runs"
        )
    _augment(rows)
    triplets = _triplets(
        rows,
        control_method=control_method,
        candidate_method=candidate_method,
        stage_label=stage_label,
    )
    screen_seed = int(config["analysis_split"]["integration_screen_seed"])
    holdout_seeds = {
        int(seed) for seed in config["analysis_split"]["confirmatory_seeds"]
    }
    screen = [row for row in triplets if row["seed"] == screen_seed]
    holdout = [row for row in triplets if row["seed"] in holdout_seeds]
    if len(screen) != 3 or len(holdout) != 9:
        raise ValueError(f"{stage_label} screen/holdout split is incomplete")
    clean = [row for row in holdout if row["threat"] == "none"]
    attacked = [row for row in holdout if row["threat"] != "none"]
    by_attack = {
        attack: [row for row in attacked if row["threat"] == attack]
        for attack in ("ipm20_n25", "bf20_n25_s10")
    }
    gates = config["gates"]
    observations = {
        "screen_integrated_accuracy_differences_pp": {
            row["threat"]: row["integrated_accuracy_difference_pp"] for row in screen
        },
        "integrated_clean_accuracy_mean_pp": _mean(
            clean, "integrated_accuracy_difference_pp"
        ),
        "integrated_clean_accuracy_each_pp": [
            row["integrated_accuracy_difference_pp"] for row in clean
        ],
        "integrated_attacked_accuracy_gain_mean_pp": _mean(
            attacked, "integrated_accuracy_difference_pp"
        ),
        "integrated_attacked_accuracy_gain_by_attack_pp": {
            attack: _mean(group, "integrated_accuracy_difference_pp")
            for attack, group in by_attack.items()
        },
        "integrated_attacked_positive_gain_rate": statistics.fmean(
            row["integrated_accuracy_difference_pp"] > 0.0 for row in attacked
        ),
        "integrated_attacked_worst20_difference_mean_pp": _mean(
            attacked, "integrated_worst20_difference_pp"
        ),
        "tilt_clean_accuracy_mean_pp": _mean(clean, "tilt_accuracy_difference_pp"),
        "tilt_attacked_accuracy_gain_mean_pp": _mean(
            attacked, "tilt_accuracy_difference_pp"
        ),
        "tilt_attacked_positive_gain_rate": statistics.fmean(
            row["tilt_accuracy_difference_pp"] > 0.0 for row in attacked
        ),
        "candidate_concentration_median": statistics.median(
            float(row["candidate_concentration"]) for row in attacked
        ),
        "candidate_byzantine_mass_mean_by_attack": {
            attack: _mean(group, "candidate_byzantine_mass")
            for attack, group in by_attack.items()
        },
        "candidate_byzantine_eligibility_rate_mean_by_attack": {
            attack: _mean(group, "candidate_byzantine_eligibility_rate")
            for attack, group in by_attack.items()
        },
        "privacy_epsilon_difference_max": max(
            max(
                abs(float(row["candidate_epsilon"]) - float(row["control_epsilon"])),
                abs(
                    float(row["candidate_epsilon"])
                    - float(row["filtered_uniform_epsilon"])
                ),
            )
            for row in triplets
        ),
        "strict_randomness_pair_rate": statistics.fmean(
            bool(row["randomness_pair_key_matches"]) for row in triplets
        ),
    }
    checks = {
        "integrated_clean_mean": observations["integrated_clean_accuracy_mean_pp"]
        >= float(gates["integrated_no_attack_accuracy_mean_min_pp"]),
        "integrated_clean_each": min(observations["integrated_clean_accuracy_each_pp"])
        >= float(gates["integrated_no_attack_accuracy_each_min_pp"]),
        "integrated_attacked_gain": observations[
            "integrated_attacked_accuracy_gain_mean_pp"
        ]
        >= float(gates["integrated_attacked_accuracy_gain_mean_min_pp"]),
        "integrated_each_attack": min(
            observations["integrated_attacked_accuracy_gain_by_attack_pp"].values()
        )
        >= float(gates["integrated_attacked_accuracy_gain_each_attack_mean_min_pp"]),
        "integrated_positive_rate": observations[
            "integrated_attacked_positive_gain_rate"
        ]
        >= float(gates["integrated_attacked_positive_gain_rate_min"]),
        "integrated_worst20": observations[
            "integrated_attacked_worst20_difference_mean_pp"
        ]
        >= float(gates["integrated_attacked_worst20_difference_mean_min_pp"]),
        "tilt_clean_mean": observations["tilt_clean_accuracy_mean_pp"]
        >= float(gates["tilt_no_attack_accuracy_mean_min_pp"]),
        "tilt_attacked_gain": observations["tilt_attacked_accuracy_gain_mean_pp"]
        >= float(gates["tilt_attacked_accuracy_gain_mean_min_pp"]),
        "tilt_positive_rate": observations["tilt_attacked_positive_gain_rate"]
        >= float(gates["tilt_attacked_positive_gain_rate_min"]),
        "candidate_nonuniformity": observations["candidate_concentration_median"]
        >= float(gates["candidate_concentration_median_min"]),
        "byzantine_mass_each_attack": max(
            observations["candidate_byzantine_mass_mean_by_attack"].values()
        )
        <= float(gates["candidate_byzantine_weight_mass_each_attack_mean_max"]),
        "byzantine_eligibility_each_attack": max(
            observations["candidate_byzantine_eligibility_rate_mean_by_attack"].values()
        )
        <= float(gates["candidate_byzantine_eligibility_rate_each_attack_mean_max"]),
        "privacy": observations["privacy_epsilon_difference_max"]
        <= float(gates["privacy_epsilon_difference_max"]),
        "weight_cap": all(
            bool(row["candidate_weight_cap_respected"]) for row in triplets
        ),
        "strict_randomness_pairing": observations["strict_randomness_pair_rate"] == 1.0,
    }
    integrated_checks = [name for name in checks if name.startswith("integrated_")] + [
        "byzantine_mass_each_attack",
        "byzantine_eligibility_each_attack",
        "privacy",
        "weight_cap",
        "strict_randomness_pairing",
    ]
    tilt_checks = [
        "tilt_clean_mean",
        "tilt_attacked_gain",
        "tilt_positive_rate",
        "candidate_nonuniformity",
    ]
    integrated_validated = all(checks[name] for name in integrated_checks)
    tilt_validated = all(checks[name] for name in tilt_checks)
    fully_validated = integrated_validated and tilt_validated
    decision = {
        "completed_runs": len(rows),
        "integration_screen_seed": screen_seed,
        "confirmatory_seeds": sorted(holdout_seeds),
        "thresholds": gates,
        "observations": observations,
        "checks": checks,
        "integrated_robust_admissibility_validated": integrated_validated,
        "far_tilt_increment_validated": tilt_validated,
        f"{artifact_prefix}_fully_validated": fully_validated,
        "claim_scope": (
            "conditional_delayed_robust_admissibility_far"
            if fully_validated
            else (
                "integrated_filter_only_not_far"
                if integrated_validated
                else "robust_admissibility_hypothesis_not_confirmed"
            )
        ),
    }
    results.mkdir(parents=True, exist_ok=True)
    _write_csv(results / f"{artifact_prefix}_triplet_results.csv", triplets)
    decision_path = results / f"{artifact_prefix}_decision.json"
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        f"# {stage_label} — Validation de l'admissibilité robuste retardée",
        "",
        "Le test compare trois bras strictement appariés : uniforme complet, "
        f"filtre uniforme, puis filtre avec tilt FAR. La seed {screen_seed} est un "
        "screen d'intégration ; seules les seeds "
        f"{', '.join(str(seed) for seed in sorted(holdout_seeds))} décident.",
        "",
        "| Bloc | Seed | Menace | Uniforme complet | Filtre uniforme | Filtre+FAR | Candidat−complet | FAR−filtre | Masse byz. | Taux byz. admissible |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in triplets:
        lines.append(
            "| {block} | {seed} | {threat} | {full} % | {filtered} % | "
            "{candidate} % | {integrated} pp | {tilt} pp | {mass} | "
            "{eligible} |".format(
                block="screen" if row["seed"] == screen_seed else "holdout",
                seed=row["seed"],
                threat=row["threat"],
                full=_fmt(row["full_uniform_accuracy_pct"], 2),
                filtered=_fmt(row["filtered_uniform_accuracy_pct"], 2),
                candidate=_fmt(row["candidate_accuracy_pct"], 2),
                integrated=_fmt(row["integrated_accuracy_difference_pp"], 2),
                tilt=_fmt(row["tilt_accuracy_difference_pp"], 2),
                mass=_fmt(row["candidate_byzantine_mass"], 4),
                eligible=_fmt(row["candidate_byzantine_eligibility_rate"], 3),
            )
        )
    lines.extend(["", "## Gates confirmatoires", ""])
    for name, passed in checks.items():
        lines.append(f"- `{name}` : **{'pass' if passed else 'fail'}**.")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            (
                "Le mécanisme intégré et la contribution incrémentale du tilt FAR "
                "sont confirmés sous tous les critères préenregistrés."
                if fully_validated
                else (
                    "Le filtre intégré est confirmé, mais la contribution "
                    "incrémentale du tilt FAR ne l'est pas."
                    if integrated_validated
                    else "L'hypothèse d'admissibilité robuste n'est pas confirmée."
                )
            ),
            "",
            "La portée reste conditionnelle aux deux attaques, au modèle, au "
            "niveau de confidentialité et aux trois seeds du holdout.",
            "",
            f"Décision machine : `{decision_path}`",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    decision = analyze(args.results, args.config, args.report)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
