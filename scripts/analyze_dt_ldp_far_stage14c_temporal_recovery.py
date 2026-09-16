#!/usr/bin/env python3
"""Analyse the paired clean -> attack -> recovery temporal-reference screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/dt_ldp_far/stage14c_temporal_poison_recovery_screen_v1"
DEFAULT_REPORT = (
    ROOT / "output/analysis/DT_LDP_FAR_Stage14C_Temporal_Poison_Recovery.md"
)

METHOD_LABELS = {
    "dt_ldp_far_stage14_moment_current_trust_guard": "FCC courant",
    "dt_ldp_far_stage14_moment_lagged_trust_guard": "FCC temporel",
    "dt_ldp_far_stage14_moment_lagged_frozen_attack_trust_guard": (
        "FCC temporel gelé pendant l'attaque"
    ),
}


def _task_root(metrics_path: Path) -> Path:
    for parent in metrics_path.parents:
        if (parent / "dt_ldp_far_task_manifest.json").exists():
            return parent
    raise FileNotFoundError(f"No task manifest above {metrics_path}")


def _values(rounds: list[dict[str, Any]], phase: str, key: str) -> list[float]:
    return [
        float(row[key])
        for row in rounds
        if row.get("attack_schedule_phase") == phase and row.get(key) is not None
    ]


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return float(ordered[index])


def _recovery_time(
    recovery_errors: list[float], threshold: float, *, consecutive: int = 2
) -> int | None:
    for index in range(len(recovery_errors) - consecutive + 1):
        if all(value <= threshold for value in recovery_errors[index : index + consecutive]):
            return index + 1
    return None


def _collect(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(input_root.glob("**/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        rounds = payload.get("rounds", [])
        if len(rounds) != 20:
            continue
        task_root = _task_root(metrics_path)
        resolved = yaml.safe_load(
            (task_root / "resolved_config.yaml").read_text(encoding="utf-8")
        )
        axes = resolved["reproduction"]["axes"]
        phase_counts = {
            phase: sum(row.get("attack_schedule_phase") == phase for row in rounds)
            for phase in ("clean", "attack", "recovery")
        }
        if phase_counts != {"clean": 6, "attack": 6, "recovery": 8}:
            raise ValueError(f"Invalid phase counts for {metrics_path}: {phase_counts}")

        key = "dtldp_reference_honest_center_error_oracle"
        clean_errors = _values(rounds, "clean", key)
        attack_errors = _values(rounds, "attack", key)
        recovery_errors = _values(rounds, "recovery", key)
        clean_median = _median(clean_errors)
        clean_p90 = _quantile(clean_errors, 0.90)
        # Pre-registered recovery envelope: 25% above the largest typical
        # clean-phase reference error.  Two consecutive rounds are required.
        threshold = max(1e-12, 1.25 * clean_p90)
        time_to_recover = _recovery_time(recovery_errors, threshold)
        excess_auc = sum(max(0.0, value - threshold) for value in recovery_errors)
        final = rounds[-1]
        rows.append(
            {
                **axes,
                "method_label": METHOD_LABELS.get(axes["method"], axes["method"]),
                "metrics_path": str(metrics_path),
                "clean_end_test_acc_pct": 100.0 * float(rounds[5]["test_accuracy"]),
                "attack_end_test_acc_pct": 100.0 * float(rounds[11]["test_accuracy"]),
                "recovery_end_test_acc_pct": 100.0 * float(final["test_accuracy"]),
                "recovery_end_worst20_pct": float(final["worst20_accuracy_pct"]),
                "recovery_end_gap_pp": float(final["best20_worst20_gap_pct"]),
                "clean_reference_error_median": clean_median,
                "clean_reference_error_p90": clean_p90,
                "attack_reference_error_peak": max(attack_errors),
                "attack_reference_error_end": attack_errors[-1],
                "reference_poisoning_excess": max(attack_errors) - clean_median,
                "recovery_reference_error_median": _median(recovery_errors),
                "recovery_reference_error_final": recovery_errors[-1],
                "recovery_threshold": threshold,
                "recovery_time_rounds": time_to_recover,
                "recovery_excess_error_auc": excess_auc,
                "attack_byzantine_mass_median": _median(
                    _values(rounds, "attack", "byzantine_weight_mass_oracle")
                ),
                "attack_reference_update_median": _median(
                    _values(rounds, "attack", "dtldp_reference_state_update_norm")
                ),
                "attack_reference_frozen_fraction": statistics.mean(
                    float(bool(row.get("dtldp_reference_was_frozen", False)))
                    for row in rounds
                    if row.get("attack_schedule_phase") == "attack"
                ),
                "epsilon_max": float(final["privacy_epsilon_max"]),
                "delta": float(final["privacy_delta"]),
                "weight_cap_respected_all": all(
                    bool(row.get("dtldp_weight_cap_respected", False)) for row in rounds
                ),
            }
        )
    return rows


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "non récupérée"
    if isinstance(value, bool):
        return "oui" if value else "non"
    return f"{float(value):.{digits}f}".replace(".", ",")


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def _write(rows: list[dict[str, Any]], report: Path) -> None:
    if not rows:
        raise ValueError("No complete Stage-14C run found")
    report.parent.mkdir(parents=True, exist_ok=True)
    csv_path = report.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    performance = []
    dynamics = []
    for row in rows:
        performance.append(
            [
                row["method_label"],
                row["threat"],
                _fmt(row["clean_end_test_acc_pct"], 2),
                _fmt(row["attack_end_test_acc_pct"], 2),
                _fmt(row["recovery_end_test_acc_pct"], 2),
                _fmt(row["recovery_end_worst20_pct"], 2),
                _fmt(row["recovery_end_gap_pp"], 2),
            ]
        )
        dynamics.append(
            [
                row["method_label"],
                row["threat"],
                _fmt(row["clean_reference_error_median"], 4),
                _fmt(row["attack_reference_error_peak"], 4),
                _fmt(row["reference_poisoning_excess"], 4),
                _fmt(row["recovery_reference_error_final"], 4),
                _fmt(row["recovery_time_rounds"], 0),
                _fmt(row["recovery_excess_error_auc"], 4),
                _fmt(row["attack_byzantine_mass_median"], 4),
                _fmt(row["attack_reference_frozen_fraction"], 2),
            ]
        )

    complete = len(rows) == 9
    cap_ok = all(row["weight_cap_respected_all"] for row in rows)
    lines = [
        "# Stage 14C — Référence FCC temporelle : empoisonnement et récupération",
        "",
        "## Protocole",
        "",
        f"- Runs complets : **{len(rows)}/9**.",
        "- Fashion-MNIST/LeNet-5, 25 clients, seed 28, 20 rounds, 10 pas "
        "Poisson par round.",
        "- Rounds 1–6 : propres ; 7–12 : attaque ; 13–20 : récupération.",
        "- Comparaison appariée : FCC courant, FCC temporel et FCC temporel "
        "gelé pendant l'attaque.",
        "- Confidentialité locale add/remove : epsilon=4, delta=10^-5 ; "
        "la référence est un post-traitement des uploads privés.",
        "",
        "Le retour à l'enveloppe propre exige deux rounds consécutifs avec une "
        "erreur de référence inférieure à 1,25 fois le p90 de la phase propre. "
        "Ce critère est évaluatif et n'est jamais utilisé par l'algorithme.",
        "",
        "## Utilité par phase",
        "",
        _table(
            [
                "Méthode",
                "Menace",
                "Acc. fin propre",
                "Acc. fin attaque",
                "Acc. fin récupération",
                "Worst-20 final",
                "Gap final",
            ],
            performance,
        ),
        "",
        "## Dynamique de la référence",
        "",
        _table(
            [
                "Méthode",
                "Menace",
                "Erreur propre",
                "Pic attaque",
                "Excès poison",
                "Erreur finale",
                "Temps récup. (rounds)",
                "AUC excès récup.",
                "Masse byz.",
                "Fraction gelée",
            ],
            dynamics,
        ),
        "",
        "## Contrôles",
        "",
        f"- Cap analytique des poids respecté partout : **{'oui' if cap_ok else 'non'}**.",
        "- L'identité des Byzantins et le centre honnête sont des oracles "
        "d'évaluation, jamais des entrées de l'algorithme.",
        "- Une conclusion confirmatoire exige les seeds 28, 36 et 54.",
        "",
        "## Décision",
        "",
    ]
    if not complete:
        lines.append("Campagne incomplète : aucun verdict n'est rendu.")
    else:
        by_method = {
            method: [row for row in rows if row["method"] == method]
            for method in METHOD_LABELS
        }

        def mean_for(method: str, key: str) -> float:
            return statistics.mean(float(row[key]) for row in by_method[method])

        current = "dt_ldp_far_stage14_moment_current_trust_guard"
        temporal = "dt_ldp_far_stage14_moment_lagged_trust_guard"
        frozen = "dt_ldp_far_stage14_moment_lagged_frozen_attack_trust_guard"
        temporal_acc_delta = mean_for(
            temporal, "recovery_end_test_acc_pct"
        ) - mean_for(current, "recovery_end_test_acc_pct")
        frozen_acc_delta = mean_for(
            frozen, "recovery_end_test_acc_pct"
        ) - mean_for(current, "recovery_end_test_acc_pct")
        temporal_mass_delta = mean_for(
            temporal, "attack_byzantine_mass_median"
        ) - mean_for(current, "attack_byzantine_mass_median")
        current_clean_error = mean_for(current, "clean_reference_error_median")
        temporal_clean_error = mean_for(temporal, "clean_reference_error_median")
        all_recovered_immediately = all(
            row["recovery_time_rounds"] == 1
            and row["recovery_excess_error_auc"] <= 1e-12
            for row in rows
        )
        lines.extend(
            [
                "**Observation.** Aucun empoisonnement persistant n'est observé : "
                + (
                    "les neuf bras récupèrent dès le premier round et leur AUC "
                    "d'excès est nulle."
                    if all_recovered_immediately
                    else "au moins un bras ne récupère pas immédiatement."
                ),
                "",
                "La référence temporelle diminue en moyenne la masse byzantine "
                f"de {_fmt(-temporal_mass_delta, 4)} par rapport à FCC courant, "
                "mais son erreur de référence propre moyenne passe de "
                f"{_fmt(current_clean_error, 4)} à {_fmt(temporal_clean_error, 4)}. "
                "Son accuracy finale moyenne varie de "
                f"{_fmt(temporal_acc_delta, 2)} point ; le contrôle gelé varie de "
                f"{_fmt(frozen_acc_delta, 2)} point.",
                "",
                "**Décision.** Ce screen ne soutient ni la promotion de FCC "
                "temporel ni le gel pendant l'attaque. Dans ce régime, la mémoire "
                "n'apporte pas de bénéfice d'utilité ou de récupération mesurable. "
                "Une confirmation multi-seed n'est donc pas lancée automatiquement. "
                "La piste ne devrait être rouverte qu'avec un stress pré-enregistré "
                "capable de produire un déplacement persistant de la référence "
                "courante, sans sélectionner le stress après observation de "
                "l'accuracy.",
            ]
        )
    lines.extend(
        [
            "",
            "## Traçabilité",
            "",
            f"- Résultats : `{DEFAULT_INPUT}`",
            f"- Tableau machine : `{csv_path}`",
            "- Matrice : "
            "`configs/dt_ldp_far/decisive_stage14c_temporal_poison_recovery_screen.yaml`",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    rows = _collect(args.input_root.resolve())
    _write(rows, args.report.resolve())
    print(f"runs={len(rows)} report={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
