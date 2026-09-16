#!/usr/bin/env python3
"""Summarise the exploratory Stage-14B Fashion-MNIST transfer screen."""

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
DEFAULT_INPUT = ROOT / "results/dt_ldp_far/stage14b_fmnist_temporal_reference_screen_v1"
DEFAULT_REPORT = (
    ROOT / "output/analysis/DT_LDP_FAR_Stage14B_FashionMNIST_Temporal_Reference.md"
)


METHOD_LABELS = {
    "dp_fedavg": "DP-FedAvg",
    "dt_ldp_far_stage14_moment_current_raw": "Moment / F courant / brut",
    "dt_ldp_far_stage14_moment_current_trust_raw": (
        "Moment / F courant / confiance / brut"
    ),
    "dt_ldp_far_stage14_moment_current_trust_guard": (
        "Moment / F courant / confiance / garde"
    ),
    "dt_ldp_far_stage14_quantile_current_trust_guard": (
        "Quantile / F courant / confiance / garde"
    ),
    "dt_ldp_far_stage14_moment_lagged_trust_guard": (
        "Moment / H(t-1) / confiance / garde"
    ),
}


def _median(rounds: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rounds if row.get(key) is not None]
    return float(statistics.median(values)) if values else None


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    number = float(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}".replace(".", ",")


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered = [[str(cell) for cell in row] for row in rows]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rendered)
    return "\n".join(lines)


def _task_root(metrics_path: Path) -> Path:
    for parent in metrics_path.parents:
        if (parent / "dt_ldp_far_task_manifest.json").exists():
            return parent
    raise FileNotFoundError(f"No task manifest above {metrics_path}")


def _collect(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(input_root.glob("**/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        rounds = payload.get("rounds", [])
        if not rounds:
            continue
        task_root = _task_root(metrics_path)
        resolved = yaml.safe_load(
            (task_root / "resolved_config.yaml").read_text(encoding="utf-8")
        )
        axes = resolved["reproduction"]["axes"]
        final = rounds[-1]
        row = {
            **axes,
            "algorithm": payload.get("algorithm"),
            "num_rounds": len(rounds),
            "metrics_path": str(metrics_path),
            "test_accuracy_pct": 100.0 * float(final["test_accuracy"]),
            "client_accuracy_pct": 100.0 * float(final["client_accuracy_mean"]),
            "variance_pp2": float(final["client_accuracy_variance_pct2"]),
            "worst20_pct": float(final["worst20_accuracy_pct"]),
            "gap_pp": float(final["best20_worst20_gap_pct"]),
            "epsilon_max": final.get("privacy_epsilon_max"),
            "delta": final.get("privacy_delta"),
            "noise_multiplier_min": final.get("privacy_model_noise_multiplier_min"),
            "noise_multiplier_max": final.get("privacy_model_noise_multiplier_max"),
            "median_score_span": _median(rounds, "dtldp_current_score_span"),
            "median_max_weight": _median(rounds, "max_client_weight"),
            "median_weight_entropy": _median(rounds, "weight_entropy"),
            "median_concentration": _median(
                rounds, "dtldp_noise_amplification_vs_uniform"
            ),
            "median_byzantine_mass": _median(rounds, "byzantine_weight_mass_oracle"),
            "median_reference_error": _median(
                rounds, "dtldp_reference_honest_center_error_oracle"
            ),
            "median_reference_drift": _median(rounds, "dtldp_reference_drift"),
            "median_reference_state_update": _median(
                rounds, "dtldp_reference_state_update_norm"
            ),
            "median_guard_active": _median(rounds, "dtldp_output_guard_active"),
            "median_guard_shift": _median(rounds, "dtldp_output_guard_shift_norm"),
            "median_guard_error_reduction": _median(
                rounds, "dtldp_guard_error_reduction_vs_raw_oracle"
            ),
            "median_released_honest_error": _median(
                rounds, "dtldp_released_aggregate_honest_center_error_oracle"
            ),
            "median_raw_honest_error": _median(
                rounds, "dtldp_raw_aggregate_honest_center_error_oracle"
            ),
            "median_novelty": _median(
                rounds, "dtldp_noise_score_effective_null_novelty_mean"
            ),
            "median_trust": _median(rounds, "dtldp_noise_score_combined_trust_mean"),
            "weight_cap_respected_all": all(
                bool(round_row.get("dtldp_weight_cap_respected", True))
                for round_row in rounds
            ),
        }
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No completed runs were found")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _paired(
    rows: list[dict[str, Any]], left_method: str, right_method: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    keys = ("reference", "threat", "partition_seed", "training_seed")
    left = {
        tuple(row[key] for key in keys): row
        for row in rows
        if row["method"] == left_method
    }
    right = {
        tuple(row[key] for key in keys): row
        for row in rows
        if row["method"] == right_method
    }
    return [(left[key], right[key]) for key in sorted(left.keys() & right.keys())]


def _make_report(rows: list[dict[str, Any]], report: Path) -> None:
    final_rows = []
    for row in rows:
        final_rows.append(
            [
                METHOD_LABELS.get(row["method"], row["method"]),
                str(row["reference"]).upper(),
                row["threat"],
                _fmt(row["test_accuracy_pct"], 2),
                _fmt(row["client_accuracy_pct"], 2),
                _fmt(row["variance_pp2"], 2),
                _fmt(row["worst20_pct"], 2),
                _fmt(row["gap_pp"], 2),
                _fmt(row["epsilon_max"], 4),
            ]
        )

    diagnostic_rows = []
    for row in rows:
        if row["method"] == "dp_fedavg":
            continue
        diagnostic_rows.append(
            [
                METHOD_LABELS.get(row["method"], row["method"]),
                str(row["reference"]).upper(),
                row["threat"],
                _fmt(row["median_score_span"]),
                _fmt(row["median_max_weight"], 4),
                _fmt(row["median_concentration"]),
                _fmt(row["median_weight_entropy"]),
                _fmt(row["median_byzantine_mass"], 4),
                _fmt(row["median_reference_error"], 4),
                _fmt(row["median_guard_active"], 2),
                _fmt(row["median_guard_error_reduction"], 3),
            ]
        )

    temporal_pairs = _paired(
        rows,
        "dt_ldp_far_stage14_moment_current_trust_guard",
        "dt_ldp_far_stage14_moment_lagged_trust_guard",
    )
    temporal_rows = []
    for current, lagged in temporal_pairs:
        temporal_rows.append(
            [
                str(current["reference"]).upper(),
                current["threat"],
                _fmt(lagged["test_accuracy_pct"] - current["test_accuracy_pct"], 2),
                _fmt(lagged["worst20_pct"] - current["worst20_pct"], 2),
                _fmt(lagged["gap_pp"] - current["gap_pp"], 2),
                _fmt(
                    (lagged["median_byzantine_mass"] or 0.0)
                    - (current["median_byzantine_mass"] or 0.0),
                    4,
                ),
                _fmt(
                    (lagged["median_reference_error"] or 0.0)
                    - (current["median_reference_error"] or 0.0),
                    4,
                ),
            ]
        )

    completed = len(rows)
    expected = 32
    cap_ok = all(row["weight_cap_respected_all"] for row in rows)
    body = [
        "# Stage 14B — Transfert Fashion-MNIST et référence temporelle",
        "",
        "## Statut et portée",
        "",
        f"- Runs complets : **{completed}/{expected}**.",
        "- Écran exploratoire : Fashion-MNIST, LeNet-5, 25 clients, seed 28, "
        "6 rounds, 10 pas Poisson par round, q=0,05, C=4.",
        "- Confidentialité : sample-level local DP, add/remove, avec "
        "hétéroscédasticité publique des multiplicateurs de bruit.",
        "- La campagne a été demandée malgré l'absence de survivant complet au "
        "pré-audit vectoriel 14B. Les accuracies sont donc descriptives et ne "
        "servent pas à modifier rétrospectivement les gates.",
        "",
        "La référence temporelle testée est stricte : au tour t, le score utilise "
        "H(t-1), qui ne dépend que des uploads privés antérieurs. Une proposition "
        "robuste est ensuite calculée sur le tour t et ne met à jour H(t) qu'après "
        "le calcul du score. Cela est distinct du retard d'un tour des poids, "
        "présent dans toutes les variantes DT-LDP-FAR de la matrice.",
        "",
        "## Métriques finales",
        "",
        _table(
            [
                "Méthode",
                "F",
                "Menace",
                "Test Acc. (%)",
                "Client Acc. (%)",
                "Var. (pp²)",
                "Worst-20 (%)",
                "Gap (pp)",
                "epsilon max",
            ],
            final_rows,
        ),
        "",
        "## Diagnostics mécanistiques — médiane sur les rounds",
        "",
        _table(
            [
                "Méthode",
                "F",
                "Menace",
                "Span score",
                "Poids max",
                "n sum(w²)",
                "Entropie",
                "Masse byz.",
                "Erreur F",
                "Garde active",
                "Réduction erreur garde",
            ],
            diagnostic_rows,
        ),
        "",
        "`Masse byz.` et les erreurs par rapport au centre honnête sont des oracles "
        "d'évaluation : ils ne sont jamais utilisés par l'algorithme.",
        "",
        "## Comparaison appariée : référence temporelle moins référence courante",
        "",
        _table(
            [
                "F",
                "Menace",
                "Delta Test Acc. (pp)",
                "Delta Worst-20 (pp)",
                "Delta Gap (pp)",
                "Delta masse byz.",
                "Delta erreur F",
            ],
            temporal_rows,
        ),
        "",
        "Les deltas favorables sont : positifs pour Test Acc. et Worst-20, "
        "négatifs pour Gap, masse byzantine et erreur de référence.",
        "",
        "## Contrôles de cohérence",
        "",
        f"- Cap analytique des poids respecté dans tous les runs disponibles : "
        f"**{'oui' if cap_ok else 'non'}**.",
        "- La calibration nulle emploie un générateur public séparé : elle ne "
        "modifie pas les tirages du DP-SGD.",
        "- F_score vit dans 256 coordonnées publiques ; F_rob de la garde est "
        "recalculée séparément dans l'espace complet de l'update.",
        "- La garde est un post-traitement d'uploads déjà localement privés : "
        "elle n'ajoute aucun coût epsilon, mais peut introduire du biais d'utilité.",
        "",
        "## Décision scientifique",
        "",
    ]
    if completed < expected:
        body.extend(
            [
                "La campagne est incomplète. Aucun verdict comparatif n'est rendu "
                "avant d'obtenir les 32 cellules prévues.",
            ]
        )
    else:
        body.extend(
            [
                "Ce screen à une seed et six rounds peut éliminer une variante "
                "manifestement défaillante, mais il ne suffit pas à promouvoir "
                "une méthode. Une référence temporelle n'est retenue que si son "
                "avantage a le même signe sous plusieurs menaces, puis se confirme "
                "sur les seeds 28, 36 et 54 à 20 rounds.",
                "",
                "Une campagne propre/attaque/récupération reste nécessaire pour "
                "mesurer l'avantage spécifiquement temporel : dérive de H, temps "
                "de récupération et éventuel empoisonnement persistant. Une "
                "attaque stationnaire seule ne sépare pas ces mécanismes.",
            ]
        )
    body.extend(
        [
            "",
            "## Traçabilité",
            "",
            f"- Résultats détaillés : `{DEFAULT_INPUT}`",
            f"- Tableau machine : `{report.with_suffix('.csv')}`",
            "- Matrice : "
            "`configs/dt_ldp_far/decisive_stage14b_fmnist_temporal_reference_screen.yaml`",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(body) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    rows = _collect(args.input_root.resolve())
    _write_csv(args.report.with_suffix(".csv"), rows)
    _make_report(rows, args.report.resolve())
    print(f"runs={len(rows)} report={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
