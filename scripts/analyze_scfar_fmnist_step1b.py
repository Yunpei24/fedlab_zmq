#!/usr/bin/env python3
"""Analyze the preregistered Fashion-MNIST SC-FAR geometry screen."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


EXPECTED_RUNS = 10
UNIFORM_QMAX = 1.0 / 25.0


def _finite_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    return values


def _median(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = _finite_values(rows, key)
    return statistics.median(values) if values else None


def _percentile(rows: Iterable[dict[str, Any]], key: str, q: float) -> float | None:
    values = sorted(_finite_values(rows, key))
    if not values:
        return None
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    position = q * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n.d."
    return f"{value:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class RunSummary:
    task_id: str
    method: str
    anchor: str
    tau_over_c: float
    test_accuracy_pct: float
    worst20_pct: float
    gap_pct: float
    variance_pct2: float
    median_user_clip: float | None
    median_score_span: float | None
    score_saturation_p90: float | None
    median_qmax: float | None
    median_concentration: float | None
    median_entropy: float | None
    median_reference_error: float | None
    median_anchor_drift: float | None


def _load_run(manifest_path: Path) -> RunSummary:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete manifest: {manifest_path}")
    metrics_paths = sorted(manifest_path.parent.glob("**/metrics.json"))
    if len(metrics_paths) != 1:
        raise ValueError(
            f"expected one metrics.json below {manifest_path.parent}, found {len(metrics_paths)}"
        )
    payload = json.loads(metrics_paths[0].read_text(encoding="utf-8"))
    rounds = payload.get("rounds", [])
    config = manifest["config"]
    declared_rounds = int(config["training"]["num_rounds"])
    if len(rounds) != declared_rounds:
        raise ValueError(
            f"{manifest_path}: {len(rounds)} rounds, expected {declared_rounds}"
        )
    final = rounds[-1]
    axes = config["reproduction"]["matrix_axes"]
    return RunSummary(
        task_id=str(manifest["task_id"]),
        method=str(axes["method"]),
        anchor=str(axes["anchor"]),
        tau_over_c=float(axes["tau_over_c"]),
        test_accuracy_pct=100.0 * float(final["test_accuracy"]),
        worst20_pct=float(final["worst20_accuracy_pct"]),
        gap_pct=float(final["best20_worst20_gap_pct"]),
        variance_pct2=float(final["client_accuracy_variance_pct2"]),
        median_user_clip=_median(rounds, "scfar_user_clip_rate"),
        median_score_span=_median(rounds, "scfar_score_span"),
        score_saturation_p90=_percentile(rounds, "scfar_score_saturation_rate", 0.9),
        median_qmax=_median(rounds, "max_client_weight"),
        median_concentration=_median(rounds, "scfar_weight_quadratic_concentration"),
        median_entropy=_median(rounds, "weight_entropy"),
        median_reference_error=_median(rounds, "scfar_reference_error_honest_mean"),
        median_anchor_drift=_median(rounds, "scfar_reference_anchor_drift"),
    )


def load_summaries(root: Path) -> list[RunSummary]:
    manifests = sorted(root.glob("**/scfar_paper1_task_manifest.json"))
    if len(manifests) != EXPECTED_RUNS:
        raise ValueError(f"found {len(manifests)} manifests below {root}; expected {EXPECTED_RUNS}")
    summaries = [_load_run(path) for path in manifests]
    if len({summary.task_id for summary in summaries}) != EXPECTED_RUNS:
        raise ValueError("duplicate task identifiers in geometry screen")
    return summaries


def build_report(summaries: list[RunSummary]) -> str:
    controls = [row for row in summaries if row.method == "central_dp_fedavg_exact"]
    if len(controls) != 1:
        raise ValueError(f"expected one uniform control, found {len(controls)}")
    baseline = controls[0]
    tilted = sorted(
        (row for row in summaries if row.method == "scfar_no_dp"),
        key=lambda row: (row.anchor, row.tau_over_c),
    )
    if len(tilted) != 9:
        raise ValueError(f"expected nine tilted runs, found {len(tilted)}")

    lines = [
        "# SC-FAR-DP full-update — Étape 1B : écran géométrique propre",
        "",
        "## Protocole",
        "",
        "Écran de développement sans bruit DP : Fashion-MNIST / LeNet-5, 25 clients, "
        "participation complète, 20 tours, 2 époques locales, "
        "partition `client_dirichlet_balanced` avec \\(\\beta=0{,}1\\), "
        "\\(C=1{,}4\\), graine de partition 105 et graine d'entraînement 19.",
        "",
        "## Résultats",
        "",
        "| Ancre | \\(\\tau/C\\) | Test Acc. | Worst-20 | Gap | Var. (pp²) | Clip médian | Span médian | Saturation p90 | \\(nq_{\\max}\\) médian | \\(n\\sum_i q_i^2\\) médian | Entropie médiane |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| uniforme | — | {_fmt(baseline.test_accuracy_pct, 2)} % | {_fmt(baseline.worst20_pct, 2)} % | {_fmt(baseline.gap_pct, 2)} | {_fmt(baseline.variance_pct2, 2)} | {_fmt(100.0 * baseline.median_user_clip if baseline.median_user_clip is not None else None, 1)} % | — | — | 1,000 | 1,000 | {_fmt(baseline.median_entropy)} |",
    ]
    for row in tilted:
        lines.append(
            f"| `{row.anchor}` | {_fmt(row.tau_over_c, 2)} | "
            f"{_fmt(row.test_accuracy_pct, 2)} % | {_fmt(row.worst20_pct, 2)} % | "
            f"{_fmt(row.gap_pct, 2)} | {_fmt(row.variance_pct2, 2)} | "
            f"{_fmt(100.0 * row.median_user_clip if row.median_user_clip is not None else None, 1)} % | "
            f"{_fmt(row.median_score_span)} | "
            f"{_fmt(100.0 * row.score_saturation_p90 if row.score_saturation_p90 is not None else None, 1)} % | "
            f"{_fmt(25.0 * row.median_qmax if row.median_qmax is not None else None)} | "
            f"{_fmt(row.median_concentration)} | {_fmt(row.median_entropy)} |"
        )

    lines.extend(
        [
            "",
            "## Gates de développement",
            "",
            "Pour chaque configuration inclinée, les diagnostics sont :",
            "",
            "- utilité : Test Accuracy à moins de 3 points du contrôle uniforme ;",
            "- score non dégénéré : span médian au moins égal à 0,30 ;",
            "- absence de saturation excessive : 90e percentile du taux de saturation au plus égal à 20 % ;",
            "- repondération active : médiane de \\(nq_{\\max}\\) au moins égale à 1,25 et médiane de \\(n\\sum_iq_i^2\\) au moins égale à 1,03.",
            "",
            "| Ancre | \\(\\tau/C\\) | Utilité | Span | Saturation | Poids actifs | Gate géométrique partiel |",
            "|---|---:|:---:|:---:|:---:|:---:|:---:|",
        ]
    )
    for row in tilted:
        utility = row.test_accuracy_pct >= baseline.test_accuracy_pct - 3.0
        span = row.median_score_span is not None and row.median_score_span >= 0.30
        saturation = row.score_saturation_p90 is not None and row.score_saturation_p90 <= 0.20
        active = (
            row.median_qmax is not None
            and row.median_concentration is not None
            and row.median_qmax >= 1.25 * UNIFORM_QMAX
            and row.median_concentration >= 1.03
        )
        passed = utility and span and saturation and active
        mark = lambda value: "oui" if value else "non"
        lines.append(
            f"| `{row.anchor}` | {_fmt(row.tau_over_c, 2)} | {mark(utility)} | "
            f"{mark(span)} | {mark(saturation)} | {mark(active)} | **{mark(passed)}** |"
        )
    lines.extend(
        [
            "",
            "Ce gate reste **partiel** : l'inclusion d'honest outliers n'est pas testée dans cette campagne propre. Elle appartient à l'étape d'attaque/inclusion confirmatoire.",
            "",
            "## Règle de décision",
            "",
            "Le bruit gaussien central ne doit être activé que si au moins une configuration conserve l'utilité et produit simultanément une géométrie non dégénérée, non saturée et des poids effectivement non uniformes. Dans le cas contraire, il faut recalibrer publiquement \\(D_{\\mathrm{score}}\\), \\(\\tau\\) ou l'ancre avant tout résultat de confidentialité.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = load_summaries(args.input_root)
    report = build_report(summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
