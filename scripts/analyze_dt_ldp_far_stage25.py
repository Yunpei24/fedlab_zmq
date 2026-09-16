#!/usr/bin/env python3
"""Evaluate the preregistered Stage-25 robust-anchor holdout."""

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
from scripts.analyze_dt_ldp_far_stage23 import _task_root

DEFAULT_RESULTS = (
    ROOT / "results/dt_ldp_far/stage25_robust_anchor_containment_validation_v1"
)
DEFAULT_CONFIG = (
    ROOT / "configs/dt_ldp_far/stage25_robust_anchor_containment_validation.yaml"
)
DEFAULT_REPORT = (
    ROOT / "output/analysis/DT_LDP_FAR_Stage25_Robust_Anchor_Containment_Validation.md"
)


def _median(rounds: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rounds if row.get(key) is not None]
    return float(statistics.median(values)) if values else None


def _augment(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        metrics_path = Path(row["metrics_path"])
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        rounds = payload["rounds"]
        resolved = yaml.safe_load(
            (_task_root(metrics_path) / "resolved_config.yaml").read_text(
                encoding="utf-8"
            )
        )
        row["randomness_pair_key"] = resolved["reproduction"]["randomness_pair_key"]
        row["median_correction_norm"] = _median(rounds, "dtldp_anchor_correction_norm")
        row["median_correction_data_bound"] = _median(
            rounds, "dtldp_anchor_correction_data_bound"
        )
        row["correction_universal_bound"] = max(
            (
                float(item["dtldp_anchor_correction_universal_bound"])
                for item in rounds
                if item.get("dtldp_anchor_correction_universal_bound") is not None
            ),
            default=None,
        )
        row["correction_certificate_all"] = all(
            bool(item.get("dtldp_anchor_correction_certificate_respected", False))
            for item in rounds
        )
        row["aggregate_mode_all"] = all(
            item.get("dtldp_aggregate_mode") == "robust_anchor_perturbation"
            for item in rounds
        )


def _triplets(
    rows: list[dict[str, Any]], *, control_method: str, candidate_method: str
) -> list[dict[str, Any]]:
    full = {
        (int(row["training_seed"]), str(row["threat"])): row
        for row in rows
        if row["method"] == control_method and row["tilt"] == "uniform"
    }
    anchored = {
        (int(row["training_seed"]), str(row["threat"]), str(row["tilt"])): row
        for row in rows
        if row["method"] == candidate_method
    }
    result: list[dict[str, Any]] = []
    for seed in sorted({key[0] for key in full}):
        for threat in ("none", "ipm20_n25", "bf20_n25_s10"):
            try:
                control = full[(seed, threat)]
                anchor = anchored[(seed, threat, "uniform")]
                candidate = anchored[(seed, threat, "boundary")]
            except KeyError as exc:
                raise ValueError(f"Missing Stage-25 cell {exc.args[0]}") from exc
            pair_keys = {
                control["randomness_pair_key"],
                anchor["randomness_pair_key"],
                candidate["randomness_pair_key"],
            }
            result.append(
                {
                    "seed": seed,
                    "threat": threat,
                    "full_uniform_accuracy_pct": control["test_accuracy_pct"],
                    "anchor_accuracy_pct": anchor["test_accuracy_pct"],
                    "candidate_accuracy_pct": candidate["test_accuracy_pct"],
                    "anchor_vs_full_accuracy_pp": anchor["test_accuracy_pct"]
                    - control["test_accuracy_pct"],
                    "candidate_vs_full_accuracy_pp": candidate["test_accuracy_pct"]
                    - control["test_accuracy_pct"],
                    "correction_vs_anchor_accuracy_pp": candidate["test_accuracy_pct"]
                    - anchor["test_accuracy_pct"],
                    "anchor_vs_full_worst20_pp": anchor["worst20_pct"]
                    - control["worst20_pct"],
                    "candidate_vs_full_worst20_pp": candidate["worst20_pct"]
                    - control["worst20_pct"],
                    "correction_vs_anchor_worst20_pp": candidate["worst20_pct"]
                    - anchor["worst20_pct"],
                    "candidate_byzantine_mass": candidate["median_byzantine_mass"],
                    "candidate_concentration": candidate["median_concentration"],
                    "candidate_correction_norm": candidate["median_correction_norm"],
                    "candidate_correction_data_bound": candidate[
                        "median_correction_data_bound"
                    ],
                    "candidate_correction_universal_bound": candidate[
                        "correction_universal_bound"
                    ],
                    "candidate_correction_certificate_all": candidate[
                        "correction_certificate_all"
                    ],
                    "candidate_aggregate_mode_all": candidate["aggregate_mode_all"],
                    "control_epsilon": control["epsilon_max"],
                    "anchor_epsilon": anchor["epsilon_max"],
                    "candidate_epsilon": candidate["epsilon_max"],
                    "candidate_weight_cap_respected": candidate[
                        "weight_cap_respected_all"
                    ],
                    "randomness_pair_key_matches": len(pair_keys) == 1,
                }
            )
    if len(result) != 12:
        raise ValueError(f"Expected twelve Stage-25 triplets, got {len(result)}")
    return result


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _positive_rate(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) > 0.0 for row in rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(results: Path, config_path: Path, report_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows = _collect(results)
    if len(rows) != int(config["expected_tasks"]):
        raise ValueError(
            f"Stage 25 incomplete: {len(rows)}/{config['expected_tasks']} runs"
        )
    _augment(rows)
    analysis = config["analysis"]
    triplets = _triplets(
        rows,
        control_method=str(analysis["control_method"]),
        candidate_method=str(analysis["candidate_method"]),
    )
    screen_seed = int(config["analysis_split"]["integration_screen_seed"])
    holdout_seeds = {
        int(seed) for seed in config["analysis_split"]["confirmatory_seeds"]
    }
    screen = [row for row in triplets if row["seed"] == screen_seed]
    holdout = [row for row in triplets if row["seed"] in holdout_seeds]
    if len(screen) != 3 or len(holdout) != 9:
        raise ValueError("Stage-25 screen/holdout split is incomplete")
    clean = [row for row in holdout if row["threat"] == "none"]
    attacked = [row for row in holdout if row["threat"] != "none"]
    by_attack = {
        attack: [row for row in attacked if row["threat"] == attack]
        for attack in ("ipm20_n25", "bf20_n25_s10")
    }
    gates = config["gates"]
    observations = {
        "screen_candidate_vs_full_accuracy_pp": {
            row["threat"]: row["candidate_vs_full_accuracy_pp"] for row in screen
        },
        "anchor_clean_accuracy_mean_pp": _mean(clean, "anchor_vs_full_accuracy_pp"),
        "anchor_clean_accuracy_each_pp": [
            row["anchor_vs_full_accuracy_pp"] for row in clean
        ],
        "anchor_attacked_accuracy_gain_mean_pp": _mean(
            attacked, "anchor_vs_full_accuracy_pp"
        ),
        "anchor_attacked_accuracy_gain_by_attack_pp": {
            attack: _mean(group, "anchor_vs_full_accuracy_pp")
            for attack, group in by_attack.items()
        },
        "anchor_attacked_positive_gain_rate": _positive_rate(
            attacked, "anchor_vs_full_accuracy_pp"
        ),
        "anchor_attacked_worst20_difference_mean_pp": _mean(
            attacked, "anchor_vs_full_worst20_pp"
        ),
        "candidate_clean_accuracy_mean_pp": _mean(
            clean, "candidate_vs_full_accuracy_pp"
        ),
        "candidate_clean_accuracy_each_pp": [
            row["candidate_vs_full_accuracy_pp"] for row in clean
        ],
        "candidate_attacked_accuracy_gain_mean_pp": _mean(
            attacked, "candidate_vs_full_accuracy_pp"
        ),
        "candidate_attacked_accuracy_gain_by_attack_pp": {
            attack: _mean(group, "candidate_vs_full_accuracy_pp")
            for attack, group in by_attack.items()
        },
        "candidate_attacked_positive_gain_rate": _positive_rate(
            attacked, "candidate_vs_full_accuracy_pp"
        ),
        "candidate_attacked_worst20_difference_mean_pp": _mean(
            attacked, "candidate_vs_full_worst20_pp"
        ),
        "correction_clean_accuracy_mean_pp": _mean(
            clean, "correction_vs_anchor_accuracy_pp"
        ),
        "correction_attacked_accuracy_gain_mean_pp": _mean(
            attacked, "correction_vs_anchor_accuracy_pp"
        ),
        "correction_attacked_positive_gain_rate": _positive_rate(
            attacked, "correction_vs_anchor_accuracy_pp"
        ),
        "correction_attacked_worst20_difference_mean_pp": _mean(
            attacked, "correction_vs_anchor_worst20_pp"
        ),
        "candidate_concentration_median": statistics.median(
            float(row["candidate_concentration"]) for row in attacked
        ),
        "candidate_correction_norm_median": statistics.median(
            float(row["candidate_correction_norm"]) for row in attacked
        ),
        "candidate_correction_universal_bound": max(
            float(row["candidate_correction_universal_bound"]) for row in holdout
        ),
        "privacy_epsilon_difference_max": max(
            max(
                abs(float(row["candidate_epsilon"]) - float(row["control_epsilon"])),
                abs(float(row["candidate_epsilon"]) - float(row["anchor_epsilon"])),
            )
            for row in triplets
        ),
        "strict_randomness_pair_rate": statistics.fmean(
            bool(row["randomness_pair_key_matches"]) for row in triplets
        ),
    }
    checks = {
        "anchor_clean_mean": observations["anchor_clean_accuracy_mean_pp"]
        >= float(gates["anchor_no_attack_accuracy_mean_min_pp"]),
        "anchor_clean_each": min(observations["anchor_clean_accuracy_each_pp"])
        >= float(gates["anchor_no_attack_accuracy_each_min_pp"]),
        "anchor_attacked_gain": observations["anchor_attacked_accuracy_gain_mean_pp"]
        >= float(gates["anchor_attacked_accuracy_gain_mean_min_pp"]),
        "anchor_each_attack": min(
            observations["anchor_attacked_accuracy_gain_by_attack_pp"].values()
        )
        >= float(gates["anchor_attacked_accuracy_gain_each_attack_mean_min_pp"]),
        "anchor_positive_rate": observations["anchor_attacked_positive_gain_rate"]
        >= float(gates["anchor_attacked_positive_gain_rate_min"]),
        "anchor_worst20": observations["anchor_attacked_worst20_difference_mean_pp"]
        >= float(gates["anchor_attacked_worst20_difference_mean_min_pp"]),
        "candidate_clean_mean": observations["candidate_clean_accuracy_mean_pp"]
        >= float(gates["candidate_no_attack_accuracy_mean_min_pp"]),
        "candidate_clean_each": min(observations["candidate_clean_accuracy_each_pp"])
        >= float(gates["candidate_no_attack_accuracy_each_min_pp"]),
        "candidate_attacked_gain": observations[
            "candidate_attacked_accuracy_gain_mean_pp"
        ]
        >= float(gates["candidate_attacked_accuracy_gain_mean_min_pp"]),
        "candidate_each_attack": min(
            observations["candidate_attacked_accuracy_gain_by_attack_pp"].values()
        )
        >= float(gates["candidate_attacked_accuracy_gain_each_attack_mean_min_pp"]),
        "candidate_positive_rate": observations["candidate_attacked_positive_gain_rate"]
        >= float(gates["candidate_attacked_positive_gain_rate_min"]),
        "candidate_worst20": observations[
            "candidate_attacked_worst20_difference_mean_pp"
        ]
        >= float(gates["candidate_attacked_worst20_difference_mean_min_pp"]),
        "correction_clean_mean": observations["correction_clean_accuracy_mean_pp"]
        >= float(gates["correction_no_attack_accuracy_mean_min_pp"]),
        "correction_attacked_gain": observations[
            "correction_attacked_accuracy_gain_mean_pp"
        ]
        >= float(gates["correction_attacked_accuracy_gain_mean_min_pp"]),
        "correction_positive_rate": observations[
            "correction_attacked_positive_gain_rate"
        ]
        >= float(gates["correction_attacked_positive_gain_rate_min"]),
        "correction_worst20": observations[
            "correction_attacked_worst20_difference_mean_pp"
        ]
        >= float(gates["correction_attacked_worst20_difference_mean_min_pp"]),
        "candidate_nonuniformity": observations["candidate_concentration_median"]
        >= float(gates["candidate_concentration_median_min"]),
        "correction_engaged": observations["candidate_correction_norm_median"]
        >= float(gates["candidate_correction_norm_median_min"]),
        "correction_certificate": all(
            bool(row["candidate_correction_certificate_all"])
            and bool(row["candidate_aggregate_mode_all"])
            for row in triplets
        ),
        "privacy": observations["privacy_epsilon_difference_max"]
        <= float(gates["privacy_epsilon_difference_max"]),
        "weight_cap": all(
            bool(row["candidate_weight_cap_respected"]) for row in triplets
        ),
        "strict_randomness_pairing": observations["strict_randomness_pair_rate"] == 1.0,
    }
    anchor_checks = [name for name in checks if name.startswith("anchor_")] + [
        "privacy",
        "strict_randomness_pairing",
    ]
    candidate_checks = [name for name in checks if name.startswith("candidate_")] + [
        "correction_certificate",
        "privacy",
        "weight_cap",
        "strict_randomness_pairing",
    ]
    correction_checks = [name for name in checks if name.startswith("correction_")] + [
        "candidate_nonuniformity",
        "privacy",
        "weight_cap",
        "strict_randomness_pairing",
    ]
    anchor_validated = all(checks[name] for name in anchor_checks)
    candidate_validated = all(checks[name] for name in candidate_checks)
    correction_validated = all(checks[name] for name in correction_checks)
    fully_validated = anchor_validated and candidate_validated and correction_validated
    decision = {
        "completed_runs": len(rows),
        "integration_screen_seed": screen_seed,
        "confirmatory_seeds": sorted(holdout_seeds),
        "thresholds": gates,
        "observations": observations,
        "checks": checks,
        "robust_anchor_baseline_validated": anchor_validated,
        "bounded_far_candidate_validated": candidate_validated,
        "far_correction_increment_validated": correction_validated,
        "stage25_fully_validated": fully_validated,
        "claim_scope": (
            "conditional_rfa_anchored_bounded_far"
            if fully_validated
            else (
                "rfa_anchor_only"
                if anchor_validated
                else "robust_anchor_containment_hypothesis_not_confirmed"
            )
        ),
    }
    results.mkdir(parents=True, exist_ok=True)
    _write_csv(results / "stage25_triplet_results.csv", triplets)
    decision_path = results / "stage25_decision.json"
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Stage 25 — Validation de l'agrégat robuste ancré",
        "",
        "Trois bras strictement appariés sont comparés : moyenne uniforme, "
        "ancre RFA seule (alpha=0), puis ancre RFA avec correction FAR bornée. "
        f"La seed {screen_seed} sert uniquement à l'intégration ; les seeds "
        f"{', '.join(str(seed) for seed in sorted(holdout_seeds))} décident.",
        "",
        "| Bloc | Seed | Menace | Moyenne | RFA | RFA+FAR | RFA−moy. | Candidat−moy. | FAR−RFA | Masse byz. | Correction / borne |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in triplets:
        lines.append(
            "| {block} | {seed} | {threat} | {full} % | {anchor} % | "
            "{candidate} % | {anchor_diff} pp | {candidate_diff} pp | "
            "{far_diff} pp | {mass} | {correction} / {bound} |".format(
                block="screen" if row["seed"] == screen_seed else "holdout",
                seed=row["seed"],
                threat=row["threat"],
                full=_fmt(row["full_uniform_accuracy_pct"], 2),
                anchor=_fmt(row["anchor_accuracy_pct"], 2),
                candidate=_fmt(row["candidate_accuracy_pct"], 2),
                anchor_diff=_fmt(row["anchor_vs_full_accuracy_pp"], 2),
                candidate_diff=_fmt(row["candidate_vs_full_accuracy_pp"], 2),
                far_diff=_fmt(row["correction_vs_anchor_accuracy_pp"], 2),
                mass=_fmt(row["candidate_byzantine_mass"], 4),
                correction=_fmt(row["candidate_correction_norm"], 5),
                bound=_fmt(row["candidate_correction_universal_bound"], 3),
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
                "L'ancrage RFA et la correction FAR bornée sont validés dans le "
                "périmètre préenregistré."
                if fully_validated
                else (
                    "Seule l'ancre RFA est validée ; la contribution FAR "
                    "incrémentale ne l'est pas."
                    if anchor_validated
                    else "L'hypothèse d'ancrage robuste n'est pas confirmée."
                )
            ),
            "",
            "Le certificat de confinement est déterministe ; les conclusions "
            "d'utilité restent conditionnelles à Fashion-MNIST, aux deux attaques, "
            "au budget de confidentialité et aux trois seeds de holdout.",
            "",
            f"Décision machine : `{decision_path.resolve().relative_to(ROOT)}`",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    decision = analyze(args.results, args.config, args.report)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
