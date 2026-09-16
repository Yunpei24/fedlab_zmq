#!/usr/bin/env python3
"""Evaluate the locked Stage-26B trMean(NNM) holdout."""

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
from scripts.analyze_dt_ldp_far_stage25 import _augment

DEFAULT_RESULTS = ROOT / "results/dt_ldp_far/stage26b_trmean_nnm_anchor_confirmatory_v1"
DEFAULT_CONFIG = (
    ROOT / "configs/dt_ldp_far/stage26b_trmean_nnm_anchor_confirmatory.yaml"
)
DEFAULT_REPORT = ROOT / "output/analysis/DT_LDP_FAR_Stage26B_trMean_NNM_Confirmatory.md"


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(statistics.fmean(float(row[key]) for row in rows))


def _positive_rate(rows: list[dict[str, Any]], key: str) -> float:
    return float(statistics.fmean(float(row[key]) > 0.0 for row in rows))


def _triplets(
    rows: list[dict[str, Any]], control_method: str, candidate_method: str
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
    for seed, threat in sorted(full):
        control = full[(seed, threat)]
        anchor = anchored[(seed, threat, "uniform")]
        candidate = anchored[(seed, threat, "boundary")]
        pair_keys = {
            control["randomness_pair_key"],
            anchor["randomness_pair_key"],
            candidate["randomness_pair_key"],
        }
        result.append(
            {
                "seed": seed,
                "threat": threat,
                "full_accuracy_pct": control["test_accuracy_pct"],
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
                "candidate_weight_cap_respected": candidate["weight_cap_respected_all"],
                "randomness_pair_key_matches": len(pair_keys) == 1,
            }
        )
    if len(result) != 9:
        raise ValueError(f"Expected nine confirmatory triplets, got {len(result)}")
    return result


def analyze(results: Path, config_path: Path, report_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selection_path = ROOT / config["analysis"]["selection_record"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["selected_method"] != config["analysis"]["candidate_method"]:
        raise ValueError(
            "Stage-26B candidate does not match the frozen Stage-26A selection"
        )
    rows = _collect(results)
    if len(rows) != int(config["expected_tasks"]):
        raise ValueError(
            f"Stage 26B incomplete: {len(rows)}/{config['expected_tasks']} runs"
        )
    _augment(rows)
    triplets = _triplets(
        rows,
        config["analysis"]["control_method"],
        config["analysis"]["candidate_method"],
    )
    expected_seeds = {
        int(seed) for seed in config["analysis_split"]["confirmatory_seeds"]
    }
    if {int(row["seed"]) for row in triplets} != expected_seeds:
        raise ValueError("Unexpected seed in Stage-26B holdout")
    clean = [row for row in triplets if row["threat"] == "none"]
    attacked = [row for row in triplets if row["threat"] != "none"]
    by_attack = {
        name: [row for row in attacked if row["threat"] == name]
        for name in ("ipm20_n25", "bf20_n25_s10")
    }
    gates = config["gates"]
    observations = {
        "anchor_clean_accuracy_mean_pp": _mean(clean, "anchor_vs_full_accuracy_pp"),
        "anchor_clean_accuracy_each_pp": [
            row["anchor_vs_full_accuracy_pp"] for row in clean
        ],
        "anchor_attacked_accuracy_gain_mean_pp": _mean(
            attacked, "anchor_vs_full_accuracy_pp"
        ),
        "anchor_attacked_accuracy_gain_by_attack_pp": {
            name: _mean(group, "anchor_vs_full_accuracy_pp")
            for name, group in by_attack.items()
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
        "candidate_attacked_accuracy_gain_mean_pp": _mean(
            attacked, "candidate_vs_full_accuracy_pp"
        ),
        "candidate_attacked_accuracy_gain_by_attack_pp": {
            name: _mean(group, "candidate_vs_full_accuracy_pp")
            for name, group in by_attack.items()
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
        "candidate_concentration_median": float(
            statistics.median(float(row["candidate_concentration"]) for row in attacked)
        ),
        "candidate_correction_norm_median": float(
            statistics.median(
                float(row["candidate_correction_norm"]) for row in attacked
            )
        ),
        "candidate_correction_universal_bound": max(
            float(row["candidate_correction_universal_bound"]) for row in triplets
        ),
        "privacy_epsilon_difference_max": max(
            max(
                abs(float(row["candidate_epsilon"]) - float(row["control_epsilon"])),
                abs(float(row["candidate_epsilon"]) - float(row["anchor_epsilon"])),
            )
            for row in triplets
        ),
        "strict_randomness_pair_rate": float(
            statistics.fmean(
                bool(row["randomness_pair_key_matches"]) for row in triplets
            )
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
    anchor_names = [name for name in checks if name.startswith("anchor_")] + [
        "privacy",
        "strict_randomness_pairing",
    ]
    correction_names = [name for name in checks if name.startswith("correction_")] + [
        "candidate_nonuniformity",
        "privacy",
        "weight_cap",
        "strict_randomness_pairing",
    ]
    anchor_validated = all(checks[name] for name in anchor_names)
    correction_validated = all(checks[name] for name in correction_names)
    fully_validated = anchor_validated and correction_validated
    decision = {
        "completed_runs": len(rows),
        "confirmatory_seeds": sorted(expected_seeds),
        "selected_anchor": config["analysis"]["selected_anchor"],
        "thresholds": gates,
        "observations": observations,
        "checks": checks,
        "robust_anchor_baseline_validated": anchor_validated,
        "bounded_far_correction_validated": correction_validated,
        "stage26_fully_validated": fully_validated,
        "claim_scope": (
            "conditional_trmean_nnm_anchored_bounded_far"
            if fully_validated
            else (
                "trmean_nnm_anchor_only"
                if anchor_validated
                else "stage26_hypothesis_not_confirmed"
            )
        ),
    }
    results.mkdir(parents=True, exist_ok=True)
    csv_path = results / "stage26b_triplet_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(triplets[0]))
        writer.writeheader()
        writer.writerows(triplets)
    decision_path = results / "stage26b_decision.json"
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Stage 26B — Confirmation indépendante de trMean(NNM)",
        "",
        "Les seeds 359, 367 et 373 constituent le holdout verrouillé avant l'écran Stage 26A.",
        "",
        "| Seed | Menace | Moyenne | Ancre | Ancre+FAR | Ancre−moy. | Candidat−moy. | FAR−ancre | Masse byz. | Correction / borne |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in triplets:
        lines.append(
            "| {seed} | {threat} | {full} % | {anchor} % | {candidate} % | {adiff} pp | {cdiff} pp | {fdiff} pp | {mass} | {corr} / {bound} |".format(
                seed=row["seed"],
                threat=row["threat"],
                full=_fmt(row["full_accuracy_pct"], 2),
                anchor=_fmt(row["anchor_accuracy_pct"], 2),
                candidate=_fmt(row["candidate_accuracy_pct"], 2),
                adiff=_fmt(row["anchor_vs_full_accuracy_pp"], 2),
                cdiff=_fmt(row["candidate_vs_full_accuracy_pp"], 2),
                fdiff=_fmt(row["correction_vs_anchor_accuracy_pp"], 2),
                mass=_fmt(row["candidate_byzantine_mass"], 4),
                corr=_fmt(row["candidate_correction_norm"], 5),
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
                "L'ancre trMean(NNM) et la correction FAR bornée sont validées dans le périmètre préenregistré."
                if fully_validated
                else (
                    "Seule l'ancre trMean(NNM) est validée ; la correction FAR incrémentale ne l'est pas."
                    if anchor_validated
                    else "L'hypothèse Stage 26 n'est pas confirmée sur le holdout."
                )
            ),
            "",
            "Le certificat de confinement est déterministe ; le claim d'utilité reste conditionnel au protocole évalué.",
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
    print(
        json.dumps(
            analyze(args.results, args.config, args.report), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
