#!/usr/bin/env python3
"""Analyze the frozen Stage-13A temporal-persistence score screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml

EXPECTED_RUNS = 8
PREVIOUS = "dp_far_current_temporal_previous_oracle"
EMA = "dp_far_current_temporal_ema_oracle"
LABELS = {
    PREVIOUS: "projection sur le tour précédent",
    EMA: "projection sur l'EMA du passé",
}


def _nearest(path: Path, filename: str) -> Path:
    for parent in path.parents:
        candidate = parent / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {filename} above {path}")


def _finite(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))
    ]


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _finite(rows, key)
    return median(values) if values else None


def _maximum(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _finite(rows, key)
    return max(values) if values else None


def _pct(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    value = float(value)
    return 100.0 * value if abs(value) <= 1.0 else value


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = yaml.safe_load(
        _nearest(path, "resolved_config.yaml").read_text(encoding="utf-8")
    )
    rounds = payload.get("rounds", [])
    expected_rounds = int(config["training"]["num_rounds"])
    if len(rounds) != expected_rounds:
        raise ValueError(f"Incomplete run {len(rounds)}/{expected_rounds}: {path}")
    if expected_rounds < 2:
        raise ValueError("Stage 13A requires at least two rounds")
    axes = config["reproduction"]["axes"]
    method = str(axes["method"])
    if method not in LABELS:
        raise ValueError(f"Unexpected Stage-13A method: {method}")
    # Round one is the intentionally uniform warm-up.  Only rounds with a
    # predictable direction enter the score-quality gate.
    score_rounds = rounds[1:]
    final = rounds[-1]
    return {
        "method": method,
        "profile": LABELS[method],
        "reference": str(axes["reference"]),
        "threat": str(axes["threat"]),
        "partition_seed": int(axes["partition_seed"]),
        "training_seed": int(axes["training_seed"]),
        "pair_key": str(config["reproduction"]["randomness_pair_key"]),
        "num_rounds": len(rounds),
        "score_rounds": len(score_rounds),
        "epsilon_max": final.get("privacy_epsilon_max"),
        "epsilon_mean": final.get("privacy_epsilon_mean"),
        "delta": final.get("privacy_delta"),
        "privacy_sampling_scheme": final.get("privacy_sampling_scheme"),
        "privacy_adjacency": final.get("privacy_adjacency"),
        "history_coverage_min": min(
            _finite(score_rounds, "far_noise_score_temporal_history_coverage")
        ),
        "honest_score_clean_corr_median": _median(
            score_rounds, "far_honest_noisy_clean_score_corr_oracle"
        ),
        "honest_top20_recall_median": _median(
            score_rounds, "far_honest_clean_top_tail_recall_oracle"
        ),
        "honest_score_public_scale_corr_median": _median(
            score_rounds, "far_honest_score_public_noise_scale_corr_oracle"
        ),
        "honest_score_clean_rmse_median": _median(
            score_rounds, "far_honest_noisy_clean_score_rmse_oracle"
        ),
        "score_span_median": _median(score_rounds, "far_score_span"),
        "score_zero_rate_median": _median(
            score_rounds, "far_noise_score_temporal_zero_rate"
        ),
        "score_saturation_rate_median": _median(
            score_rounds, "far_noise_score_temporal_saturation_rate"
        ),
        "weight_concentration_median": _median(
            score_rounds, "far_noise_amplification_vs_uniform"
        ),
        "byzantine_weight_mass_median": _median(
            score_rounds, "byzantine_weight_mass_oracle"
        ),
        "max_client_weight": _maximum(rounds, "max_client_weight"),
        "weight_cap": final.get("far_weight_cap"),
        "weight_cap_all_rounds": all(
            bool(row.get("far_weight_cap_respected")) for row in rounds
        ),
        "test_accuracy_final_pp": _pct(final.get("test_accuracy")),
        "client_accuracy_final_pp": _pct(final.get("client_accuracy_mean")),
        "client_accuracy_variance_final_pp2": final.get(
            "client_accuracy_variance_pct2"
        ),
        "worst20_final_pp": _pct(final.get("worst20_accuracy")),
        "gap_final_pp": _pct(final.get("best20_worst20_gap")),
        "metrics_path": str(path),
    }


def _informative(row: dict[str, Any]) -> bool:
    corr = row["honest_score_clean_corr_median"]
    recall = row["honest_top20_recall_median"]
    noise_corr = row["honest_score_public_scale_corr_median"]
    return bool(
        row["history_coverage_min"] >= 1.0 - 1e-12
        and corr is not None
        and corr >= 0.50
        and recall is not None
        and recall >= 0.60
        and noise_corr is not None
        and abs(noise_corr) <= 0.10
        and row["weight_cap_all_rounds"]
    )


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "oui" if value else "non"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _profile_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[row["method"]].append(row)
    output = []
    for method in (PREVIOUS, EMA):
        rows = grouped[method]
        passed = sum(_informative(row) for row in rows)
        no_attack_passed = any(
            row["threat"] == "none" and _informative(row) for row in rows
        )
        output.append(
            {
                "method": method,
                "profile": LABELS[method],
                "cells_passed": passed,
                "cells_total": len(rows),
                "no_attack_passed": no_attack_passed,
                "promoted": passed >= 3 and no_attack_passed,
                "mean_clean_corr": mean(
                    row["honest_score_clean_corr_median"] for row in rows
                ),
                "mean_top20_recall": mean(
                    row["honest_top20_recall_median"] for row in rows
                ),
                "mean_abs_noise_corr": mean(
                    abs(row["honest_score_public_scale_corr_median"])
                    for row in rows
                ),
                "mean_concentration": mean(
                    row["weight_concentration_median"] for row in rows
                ),
                "mean_byzantine_mass_attacked": mean(
                    row["byzantine_weight_mass_median"]
                    for row in rows
                    if row["threat"] != "none"
                ),
            }
        )
    return output


def _write_report(path: Path, runs: list[dict[str, Any]], profiles: list[dict[str, Any]]) -> None:
    lines = [
        "# Stage 13A — Analyse du score de persistance temporelle",
        "",
        f"Campagne complète : **{len(runs)}/{EXPECTED_RUNS} runs valides**.",
        "Le premier tour uniforme est exclu des métriques de qualité du score.",
        "",
        "## Résultats par menace",
        "",
        "| Profil | Menace | Corr. propre | Rappel top-20 | Corr. bruit (valeur absolue) | Span | Concentration | Masse byz. | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in sorted(runs, key=lambda item: (item["method"], item["threat"])):
        lines.append(
            "| {profile} | {threat} | {corr} | {recall} | {noise} | {span} | {conc} | {byz} | {gate} |".format(
                profile=row["profile"],
                threat=row["threat"],
                corr=_fmt(row["honest_score_clean_corr_median"]),
                recall=_fmt(row["honest_top20_recall_median"]),
                noise=_fmt(abs(row["honest_score_public_scale_corr_median"])),
                span=_fmt(row["score_span_median"]),
                conc=_fmt(row["weight_concentration_median"]),
                byz=_fmt(row["byzantine_weight_mass_median"]),
                gate="oui" if _informative(row) else "non",
            )
        )
    lines.extend(
        [
            "",
            "## Décision préenregistrée",
            "",
            "| Profil | Cellules | Sans attaque | Corr. propre moyenne | Rappel moyen | Corr. bruit absolue moyenne | Promotion |",
            "|---|---:|:---:|---:|---:|---:|:---:|",
        ]
    )
    for row in profiles:
        lines.append(
            "| {profile} | {passed}/{total} | {none} | {corr} | {recall} | {noise} | {promoted} |".format(
                profile=row["profile"],
                passed=row["cells_passed"],
                total=row["cells_total"],
                none="oui" if row["no_attack_passed"] else "non",
                corr=_fmt(row["mean_clean_corr"]),
                recall=_fmt(row["mean_top20_recall"]),
                noise=_fmt(row["mean_abs_noise_corr"]),
                promoted="oui" if row["promoted"] else "non",
            )
        )
    promoted = [row["profile"] for row in profiles if row["promoted"]]
    if promoted:
        decision = (
            "Au moins un score franchit le gate : " + ", ".join(promoted)
            + ". Le Stage 13B peut ajouter une pénalité byzantine unilatérale."
        )
    else:
        decision = (
            "Aucun score ne franchit le gate. L'hypothèse de persistance "
            "directionnelle n'est pas soutenue dans ce protocole ; aucune "
            "confirmation multi-seeds ni sélection par accuracy ne doit être lancée."
        )
    lines.extend(
        [
            "",
            decision,
            "",
            "## Confidentialité et bornes",
            "",
            "Les deux profils utilisent le même mécanisme local-DP add/remove avec vrai Poisson.",
            "Le score temporel est un post-traitement d'uploads déjà privés et de paramètres publics.",
            "Il ne consomme donc pas d'epsilon supplémentaire. Le cap analytique des poids reste 2/25 = 0,08.",
            "Les oracles propres présents dans ces runs servent uniquement à l'audit de simulation et ne sont pas déployables.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(
            "results/dt_ldp_far/decisive/"
            "dt_ldp_far_decisive_stage13a_generation6_temporal_score_n25_screen_v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/analysis/dt_ldp_far_n25_stage13a_generation6"),
    )
    args = parser.parse_args()
    paths = sorted(args.results_root.rglob("metrics.json"))
    if len(paths) != EXPECTED_RUNS:
        raise SystemExit(f"Stage 13A incomplete: {len(paths)}/{EXPECTED_RUNS} metrics.json")
    runs = [_load(path) for path in paths]
    profiles = _profile_rows(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "run_metrics.csv", runs)
    _write_csv(args.output_dir / "profile_gates.csv", profiles)
    _write_report(args.output_dir / "DT_LDP_FAR_Stage13A_Analyse.md", runs, profiles)
    status = {
        "complete": True,
        "valid_runs": len(runs),
        "expected_runs": EXPECTED_RUNS,
        "promoted_profiles": [row["method"] for row in profiles if row["promoted"]],
    }
    (args.output_dir / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
