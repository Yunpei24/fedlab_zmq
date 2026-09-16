#!/usr/bin/env python3
"""Run Stage 19 and require the frozen RFA candidate to pass both structures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_dt_ldp_far_stage18_conditional_byzantine_holdout import (  # noqa: E402
    run as run_stage18_engine,
)


def run(
    config_path: Path,
    output_dir: Path,
    report_path: Path,
    *,
    calibration_draws_override: int | None = None,
    null_holdout_draws_override: int | None = None,
    signal_draws_override: int | None = None,
) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    engine_report = output_dir / "engine_report.md"
    run_stage18_engine(
        config_path,
        output_dir,
        engine_report,
        calibration_draws_override=calibration_draws_override,
        null_holdout_draws_override=null_holdout_draws_override,
        signal_draws_override=signal_draws_override,
    )
    summary = pd.read_csv(output_dir / "summary.csv")
    decision = config["decision"]
    selected = summary[
        summary["profile"].eq(str(decision["primary_profile"]))
        & summary["robust_reference"].eq(str(decision["primary_reference"]))
        & summary["minority_structure"].isin(
            [str(value) for value in decision["required_structures"]]
        )
    ].copy()
    required = set(str(value) for value in decision["required_structures"])
    observed = set(selected["minority_structure"])
    if observed != required:
        raise ValueError("Stage 19 did not produce every required structure")
    confirmed = bool(selected["passes_full_conditional_gates"].astype(bool).all())
    rows = selected.to_dict(orient="records")
    machine = {
        "config": str(config_path.resolve()),
        "candidate": {
            "profile": str(decision["primary_profile"]),
            "robust_reference": str(decision["primary_reference"]),
        },
        "required_structures": rows,
        "confirmed": confirmed,
        "fashion_mnist_authorized": confirmed,
    }
    (output_dir / "confirmatory_decision.json").write_text(
        json.dumps(machine, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Stage 19 — Confirmation indépendante du candidat RFA",
        "",
        "## Candidat figé après Stage 18",
        "",
        "- Référence : **RFA**.",
        "- Nouveauté : énergie covariance cross-fittée.",
        "- Soutien : 50 % de la plage de logits.",
        "- Cap : `max_i omega_i <= 2/25 = 0,08`.",
        "- Seeds : 431, 457, 479, 503, 521, 547 et 569, absentes des stages précédents.",
        "- Attaques du claim : IPM et Bit-Flip, amplitudes 0,50, 1,00 et 1,50.",
        "",
        "Le candidat n'est confirmé que si les outliers dispersés **et** le "
        "groupe cohérent de taille `f` passent simultanément les seuils inchangés.",
        "",
        "## Résultats confirmatoires",
        "",
        "| Structure | corr. propre | rappel propre | gain masse outliers | masse byz. pire | rappel attaqué pire | erreur/uniforme moy. | erreur/uniforme pire | F compétitive | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in rows:
        lines.append(
            "| {structure} | {corr:.3f} | {recall:.3f} | {gain:+.3f} | {byz:.3f} | {attacked:.3f} | {mean:.3f} | {worst:.3f} | {ref} | {passed} |".format(
                structure=row["minority_structure"],
                corr=row["clean_geometry_correlation_mean"],
                recall=row["honest_outlier_recall_mean"],
                gain=row["honest_outlier_weight_mass_gain_over_uniform"],
                byz=row["separated_byzantine_weight_mass_worst_group"],
                attacked=row["separated_global_honest_outlier_recall_worst_group"],
                mean=row["separated_aggregate_error_mean_ratio_to_uniform"],
                worst=row["separated_aggregate_error_worst_group_ratio_to_uniform"],
                ref="oui" if bool(row["reference_is_competitive"]) else "non",
                passed=("oui" if bool(row["passes_full_conditional_gates"]) else "non"),
            )
        )
    lines.extend(["", "## Verdict", ""])
    if confirmed:
        lines.extend(
            [
                "**Candidat confirmé sur holdout indépendant.**",
                "",
                "La validation est synthétique et conditionnelle aux attaques "
                "séparées ; elle autorise le test end-to-end Fashion-MNIST. "
                "Elle ne constitue ni un détecteur Byzantine universel ni une "
                "garantie contre ALIE/l'attaque adaptative alignée.",
            ]
        )
    else:
        failed = [
            str(row["minority_structure"])
            for row in rows
            if not bool(row["passes_full_conditional_gates"])
        ]
        lines.extend(
            [
                "**Candidat non confirmé.**",
                "",
                "Structures en échec : " + ", ".join(failed) + ".",
                "",
                "Le candidat doit être rejeté sans ajuster ses paramètres sur "
                "ce holdout.",
            ]
        )
    lines.extend(
        [
            "",
            "## Traçabilité",
            "",
            f"- Configuration : `{config_path.resolve()}`",
            f"- Détails : `{(output_dir / 'signal_detail.csv').resolve()}`",
            f"- Synthèse complète : `{(output_dir / 'summary.csv').resolve()}`",
            f"- Décision : `{(output_dir / 'confirmatory_decision.json').resolve()}`",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(machine, indent=2, sort_keys=True))
    return machine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/dt_ldp_far/stage19_rfa_crossfit_support_confirmatory.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/dt_ldp_far/score_quality_stage19_rfa_confirmatory_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "output/analysis/DT_LDP_FAR_Stage19_RFA_Confirmatory.md",
    )
    parser.add_argument("--calibration-draws", type=int)
    parser.add_argument("--null-holdout-draws", type=int)
    parser.add_argument("--signal-draws", type=int)
    args = parser.parse_args()
    run(
        args.config.resolve(),
        args.output_dir.resolve(),
        args.report.resolve(),
        calibration_draws_override=args.calibration_draws,
        null_holdout_draws_override=args.null_holdout_draws,
        signal_draws_override=args.signal_draws,
    )


if __name__ == "__main__":
    main()
