#!/usr/bin/env python3
"""Apply the preregistered Stage-26A robust-anchor selection rule."""

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
from scripts.analyze_dt_ldp_far_stage25 import _augment
from scripts.analyze_dt_ldp_far_stage23 import _task_root

DEFAULT_RESULTS = ROOT / "results/dt_ldp_far/stage26a_robust_anchor_selection_screen_v1"
DEFAULT_CONFIG = (
    ROOT / "configs/dt_ldp_far/stage26a_robust_anchor_selection_screen.yaml"
)
DEFAULT_REPORT = ROOT / "output/analysis/DT_LDP_FAR_Stage26A_Robust_Anchor_Selection.md"


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values))


def analyze(results: Path, config_path: Path, report_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows = _collect(results)
    if len(rows) != int(config["expected_tasks"]):
        raise ValueError(f"Expected {config['expected_tasks']} runs, found {len(rows)}")
    _augment(rows)
    analysis = config["analysis"]
    control_method = analysis["control_method"]
    candidate_methods = list(analysis["candidate_methods"])
    seed = int(config["analysis_split"]["selection_screen_seed"])
    threats = ("none", "ipm20_n25", "bf20_n25_s10")
    indexed = {
        (str(row["method"]), str(row["threat"])): row
        for row in rows
        if int(row["training_seed"]) == seed
    }
    controls = {threat: indexed[(control_method, threat)] for threat in threats}
    summaries: list[dict[str, Any]] = []
    clean_floor = float(analysis["clean_accuracy_gain_min_pp"])
    for method in candidate_methods:
        cells = {threat: indexed[(method, threat)] for threat in threats}
        gains = {
            threat: float(cells[threat]["test_accuracy_pct"])
            - float(controls[threat]["test_accuracy_pct"])
            for threat in threats
        }
        attacked = [gains["ipm20_n25"], gains["bf20_n25_s10"]]
        summaries.append(
            {
                "method": method,
                "anchor": (
                    cells["none"]["dtldp_aggregate_anchor"]
                    if "dtldp_aggregate_anchor" in cells["none"]
                    else yaml.safe_load(
                        (
                            _task_root(Path(cells["none"]["metrics_path"]))
                            / "resolved_config.yaml"
                        ).read_text(encoding="utf-8")
                    )["training"]["algo_config"]["dt_aggregate_anchor"]
                ),
                "clean_accuracy_gain_pp": gains["none"],
                "ipm_accuracy_gain_pp": gains["ipm20_n25"],
                "bitflip_accuracy_gain_pp": gains["bf20_n25_s10"],
                "minimum_attacked_accuracy_gain_pp": min(attacked),
                "mean_attacked_accuracy_gain_pp": _mean(attacked),
                "eligible": gains["none"] >= clean_floor,
            }
        )
    eligible = [row for row in summaries if row["eligible"]]
    if not eligible:
        raise ValueError("No anchor satisfies the preregistered clean-utility floor")
    selected = max(
        eligible,
        key=lambda row: (
            row["minimum_attacked_accuracy_gain_pp"],
            row["mean_attacked_accuracy_gain_pp"],
            row["clean_accuracy_gain_pp"],
        ),
    )
    decision = {
        "stage26a_screen_completed": True,
        "claim_scope": "selection_screen_only_no_confirmatory_claim",
        "selection_seed": seed,
        "locked_confirmatory_seeds": config["analysis_split"][
            "locked_confirmatory_seeds"
        ],
        "selected_method": selected["method"],
        "selected_anchor": selected["anchor"],
        "selection_rule": analysis["selection_order"],
        "clean_accuracy_gain_min_pp": clean_floor,
        "candidates": summaries,
    }
    results.mkdir(parents=True, exist_ok=True)
    decision_path = results / "stage26a_selection.json"
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Stage 26A — Écran de sélection de l'ancre robuste",
        "",
        "La seed 353 est exploratoire. Les seeds 359, 367 et 373 n'ont pas été exécutées.",
        "",
        "| Ancre | Gain propre | Gain IPM | Gain Bit-Flip | Pire gain attaqué | Gain attaqué moyen | Éligible |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summaries:
        lines.append(
            "| {anchor} | {clean} pp | {ipm} pp | {bf} pp | {worst} pp | {mean} pp | {eligible} |".format(
                anchor=row["anchor"],
                clean=_fmt(row["clean_accuracy_gain_pp"], 2),
                ipm=_fmt(row["ipm_accuracy_gain_pp"], 2),
                bf=_fmt(row["bitflip_accuracy_gain_pp"], 2),
                worst=_fmt(row["minimum_attacked_accuracy_gain_pp"], 2),
                mean=_fmt(row["mean_attacked_accuracy_gain_pp"], 2),
                eligible="oui" if row["eligible"] else "non",
            )
        )
    lines.extend(
        [
            "",
            f"Ancre sélectionnée mécaniquement : **{selected['anchor']}** (`{selected['method']}`).",
            "",
            "Ce choix n'est pas une validation. Il fixe uniquement la candidate qui sera évaluée sur le holdout indépendant.",
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
