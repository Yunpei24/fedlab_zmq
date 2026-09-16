#!/usr/bin/env python3
"""Analyze the paired bounded-score and raw-distance SC-FAR screen."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_RUNS = 7
N_CLIENTS = 25


def _values(rounds: list[dict[str, Any]], key: str) -> list[float]:
    result = []
    for row in rounds:
        value = row.get(key)
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            result.append(number)
    return result


def _median(rounds: list[dict[str, Any]], key: str) -> float | None:
    values = _values(rounds, key)
    return statistics.median(values) if values else None


def _percentile(rounds: list[dict[str, Any]], key: str, q: float) -> float | None:
    values = sorted(_values(rounds, key))
    if not values:
        return None
    position = q * (len(values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n.a."
    return f"{value:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class Summary:
    method: str
    transform: str
    dscore_over_c: float | None
    alpha: float
    test_accuracy_pct: float
    worst20_pct: float
    gap_pct: float
    variance_pct2: float
    clip_rate: float | None
    score_span: float | None
    logit_span: float | None
    saturation_p90: float | None
    nqmax: float | None
    concentration: float | None
    entropy: float | None
    sensitivity: float | None
    sensitivity_mode: str
    kappa_source: str


def _load(manifest_path: Path) -> Summary:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete run: {manifest_path}")
    metric_paths = sorted(manifest_path.parent.glob("**/metrics.json"))
    if len(metric_paths) != 1:
        raise ValueError(f"expected one metrics file below {manifest_path.parent}")
    payload = json.loads(metric_paths[0].read_text(encoding="utf-8"))
    rounds = payload.get("rounds", [])
    config = manifest["config"]
    if len(rounds) != int(config["training"]["num_rounds"]):
        raise ValueError(f"incomplete metrics: {metric_paths[0]}")
    axes = config["reproduction"]["matrix_axes"]
    final = rounds[-1]
    return Summary(
        method=str(axes["method"]),
        transform=str(final.get("scfar_score_transform", "unknown")),
        dscore_over_c=(
            float(axes["distance_score_over_c"])
            if axes.get("distance_score_over_c") is not None
            else None
        ),
        alpha=float(final.get("scfar_effective_alpha", 0.0)),
        test_accuracy_pct=100.0 * float(final["test_accuracy"]),
        worst20_pct=float(final["worst20_accuracy_pct"]),
        gap_pct=float(final["best20_worst20_gap_pct"]),
        variance_pct2=float(final["client_accuracy_variance_pct2"]),
        clip_rate=_median(rounds, "scfar_user_clip_rate"),
        score_span=_median(rounds, "scfar_score_span"),
        logit_span=_median(rounds, "scfar_logit_span"),
        saturation_p90=_percentile(rounds, "scfar_score_saturation_rate", 0.9),
        nqmax=(
            N_CLIENTS * _median(rounds, "max_client_weight")
            if _median(rounds, "max_client_weight") is not None
            else None
        ),
        concentration=_median(rounds, "scfar_weight_quadratic_concentration"),
        entropy=_median(rounds, "weight_entropy"),
        sensitivity=(
            float(final["scfar_sensitivity"])
            if final.get("scfar_sensitivity") is not None
            else None
        ),
        sensitivity_mode=str(final.get("scfar_sensitivity_mode", "n.a.")),
        kappa_source=str(final.get("scfar_certified_kappa_source", "n.a.")),
    )


def load(root: Path) -> list[Summary]:
    manifests = sorted(root.glob("**/scfar_paper1_task_manifest.json"))
    if len(manifests) != EXPECTED_RUNS:
        raise ValueError(f"found {len(manifests)} manifests; expected {EXPECTED_RUNS}")
    return [_load(path) for path in manifests]


def report(rows: list[Summary]) -> str:
    uniform = [row for row in rows if row.method == "central_dp_fedavg_exact"]
    bounded = sorted(
        (row for row in rows if row.method == "scfar_no_dp"),
        key=lambda row: float(row.dscore_over_c or 0.0),
    )
    raw = [row for row in rows if row.method == "far_raw_distance_fcc"]
    if len(uniform) != 1 or len(bounded) != 5 or len(raw) != 1:
        raise ValueError("unexpected Step 1C method cardinalities")
    control = uniform[0]
    ordered = [control, *bounded, raw[0]]
    tilted_alpha = bounded[0].alpha
    lines = [
        "# SC-FAR-DP full-update — Étape 1C : calibration de score",
        "",
        "## Protocole apparié",
        "",
        "Fashion-MNIST / LeNet-5, 25 clients, 20 tours, 2 époques locales, "
        "participation complète, aucune attaque, \\(C=1{,}4\\), "
        "\\(F_{\\mathrm{CC}}\\), ancre `previous_release`, \\(\\tau/C=1\\), "
        "graine de partition 106 et graine d'entraînement 23.",
        "",
        "Le tilt est fixé à la borne publique associée à \\(n=25\\) et "
        "\\(\\kappa_w=2\\) :",
        "",
        "\\[",
        "\\alpha=\\alpha_{\\max}="
        "\\log\\!\\left(\\frac{2(25-1)}{25-2}\\right)="
        f"{tilted_alpha:.6f}.",
        "\\]",
        "",
        "Le score borné utilise \\(s_i=\\min(d_i/D_{\\mathrm{score}},1)\\). "
        "L'ablation brute utilise directement \\(s_i=d_i=\\lVert X_i-r\\rVert_2\\). "
        "Tous les bras inclinés emploient le même \\(\\alpha\\) numérique. Il s'agit "
        "d'un écran de géométrie **sans bruit DP**, et non encore d'une comparaison "
        "d'utilité privée.",
        "",
        "## Résultats",
        "",
        "| Variante | \\(D_{\\mathrm{score}}/C\\) | Test Acc. | Worst-20 | Gap | Var. (pp²) | Clip médian | Span médian | Logit-span médian | Saturation p90 | \\(nq_{\\max}\\) médian | Concentration | Entropie | Sensibilité utilisée |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        label = (
            "uniforme"
            if row.method == "central_dp_fedavg_exact"
            else "score borné"
            if row.method == "scfar_no_dp"
            else "distance brute"
        )
        saturation = (
            f"{_fmt(100.0 * row.saturation_p90, 1)} %"
            if row.saturation_p90 is not None
            else "n.a."
        )
        lines.append(
            f"| {label} | {_fmt(row.dscore_over_c, 2)} | "
            f"{_fmt(row.test_accuracy_pct, 2)} % | {_fmt(row.worst20_pct, 2)} % | "
            f"{_fmt(row.gap_pct, 2)} | {_fmt(row.variance_pct2, 2)} | "
            f"{_fmt(100.0 * row.clip_rate if row.clip_rate is not None else None, 1)} % | "
            f"{_fmt(row.score_span)} | {_fmt(row.logit_span)} | "
            f"{saturation} | "
            f"{_fmt(row.nqmax)} | {_fmt(row.concentration)} | {_fmt(row.entropy)} | "
            f"{_fmt(row.sensitivity)} (`{row.sensitivity_mode}`) |"
        )
    lines.extend(
        [
            "",
            "## Différences appariées par rapport à l'agrégation uniforme",
            "",
            "Une différence positive de Worst-20 est favorable. Une différence négative "
            "de gap ou de variance est favorable.",
            "",
            "| Variante | \\(\\Delta\\) Test Acc. | \\(\\Delta\\) Worst-20 | "
            "\\(\\Delta\\) Gap | \\(\\Delta\\) Var. (pp²) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in [*bounded, raw[0]]:
        label = (
            f"score borné, \\(D_{{\\mathrm{{score}}}}/C={_fmt(row.dscore_over_c, 2)}\\)"
            if row.method == "scfar_no_dp"
            else "distance brute"
        )
        lines.append(
            f"| {label} | {_fmt(row.test_accuracy_pct - control.test_accuracy_pct, 2)} | "
            f"{_fmt(row.worst20_pct - control.worst20_pct, 2)} | "
            f"{_fmt(row.gap_pct - control.gap_pct, 2)} | "
            f"{_fmt(row.variance_pct2 - control.variance_pct2, 2)} |"
        )
    lines.extend(
        [
            "",
            "## Gates du score borné",
            "",
            "Ces seuils sont des critères de diagnostic préenregistrés, et non des "
            "conditions nécessaires de SC-FAR-DP ni des hypothèses reprises du papier FAR. "
            "Ils servent uniquement à détecter une repondération devenue pratiquement "
            "uniforme après le contrôle de sensibilité. Les conclusions doivent donc aussi "
            "être rapportées comme fonctions continues du span et de la concentration, sans "
            "transformer les seuils choisis en vérités théoriques.",
            "",
            "Une configuration bornée est informative si elle reste à moins de 3 points du contrôle, "
            "a un span médian au moins égal à 0,30, une saturation p90 au plus égale à 20 %, "
            "\\(nq_{\\max}\\geq1{,}25\\) et \\(n\\sum_iq_i^2\\geq1{,}03\\).",
            "",
            "| \\(D_{\\mathrm{score}}/C\\) | Utilité | Span | Saturation | Poids actifs | Gate |",
            "|---:|:---:|:---:|:---:|:---:|:---:|",
        ]
    )
    for row in bounded:
        utility = row.test_accuracy_pct >= control.test_accuracy_pct - 3.0
        span = row.score_span is not None and row.score_span >= 0.30
        saturation = row.saturation_p90 is not None and row.saturation_p90 <= 0.20
        active = (
            row.nqmax is not None
            and row.concentration is not None
            and row.nqmax >= 1.25
            and row.concentration >= 1.03
        )
        mark = lambda value: "oui" if value else "non"
        lines.append(
            f"| {_fmt(row.dscore_over_c, 2)} | {mark(utility)} | {mark(span)} | "
            f"{mark(saturation)} | {mark(active)} | **{mark(utility and span and saturation and active)}** |"
        )
    lines.extend(
        [
            "",
            "## Interprétation obligatoire de la distance brute",
            "",
            "Sur cette seed, la distance brute est plus discriminante : elle atteint "
            "\\(nq_{\\max}=1{,}280\\), améliore Worst-20 de 2,05 points et réduit le gap "
            "de 2,80 points. Sa concentration quadratique reste toutefois à 1,016, sous le "
            "gate préenregistré de 1,03.",
            "",
            "Elle ne bénéficie pas du certificat de conception "
            "\\(q_i\\leq\\kappa_w/n=0{,}08\\) dérivé pour un score dans \\([0,1]\\). "
            "Le clipping implique bien une plage publique conservatrice pour les distances, "
            "mais le présent mécanisme n'en déduit pas un nouveau cap \\(\\kappa_w=2\\). Il "
            "utilise donc la sensibilité globale sûre \\(2C=2{,}8\\). À multiplicateur "
            "gaussien identique, cela demanderait environ 5,1 fois l'écart-type de bruit du "
            "bras borné \\(D_{\\mathrm{score}}/C=1\\), dont la sensibilité vaut 0,554.",
            "",
            "La comparaison à \\(\\alpha\\) numérique identique ne sépare pas complètement "
            "l'effet de la transformation de celui de l'échelle des logits : la distance brute "
            "a un logit-span médian de 0,505, contre 0,349 pour "
            "\\(D_{\\mathrm{score}}/C=1\\). Une comparaison mécanistique définitive devra "
            "aussi apparier le logit-span ou la concentration effective.",
            "",
            "## Décision de l'étape 1C",
            "",
            "1. \\(D_{\\mathrm{score}}/C=0{,}5\\) est rejeté : la saturation rend les poids "
            "pratiquement uniformes.",
            "2. Les réglages bornés \\(D_{\\mathrm{score}}/C\\in\\{1,1{,}25\\}\\) "
            "offrent la géométrie la plus saine, mais aucun ne passe encore le gate d'activité "
            "des poids avec \\(\\kappa_w=2\\).",
            "3. La distance brute donne un signal de fairness intéressant, mais ne constitue "
            "pas le mécanisme SC-FAR-DP principal : son coût de sensibilité annulerait "
            "probablement ce gain une fois le bruit central ajouté.",
            "4. Ces résultats proviennent d'une seule paire de seeds et de 20 tours, sans "
            "attaque et sans DP. Ils sélectionnent des candidats de développement ; ils ne "
            "valident pas encore une contribution expérimentale.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = report(load(args.input_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
