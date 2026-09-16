#!/usr/bin/env python3
"""Create presentation-ready figures and reports for the K6 radial prescreen."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib as mpl  # noqa: E402

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_RESULTS = ROOT / "output/analysis/gaussian_aware_g0g_k6_radial_prescreen"
DEFAULT_FIGURES = ROOT / "output/figures/gaussian_aware_g0g_k6_radial_prescreen"
DEFAULT_REPORT = ROOT / (
    "output/analysis/Gaussian_Aware_G0g_K6_Radial_Prescreen_Report.md"
)
DEFAULT_BRIEF = ROOT / ("output/analysis/Gaussian_Aware_G0g_K6_Radial_Monday_Brief.md")
T_CRITICAL_DF11 = 2.2009851600916406
EXPECTED_SEEDS = 12


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _ci(series: pd.Series) -> dict[str, float]:
    values = series.to_numpy(dtype=float)
    if len(values) != EXPECTED_SEEDS or not np.isfinite(values).all():
        raise RuntimeError("Each interval requires twelve finite outer seeds")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    half = T_CRITICAL_DF11 * sd / math.sqrt(len(values))
    return {"mean": mean, "sd": sd, "low": mean - half, "high": mean + half}


def _validate(results: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    decision = _read_json(results / "decision.json")
    manifest = _read_json(results / "manifest.json")
    audit = _read_json(results / "independent_postrun_audit.json")
    if (
        manifest.get("status") != "completed_development_prescreen"
        or manifest.get("device") != "mps"
        or manifest.get("holdout_opened") is not False
    ):
        raise RuntimeError(
            "The radial prescreen is incomplete, non-MPS, or opened holdout"
        )
    if audit.get("all_checks_pass") is not True or not all(
        audit.get("checks", {}).values()
    ):
        raise RuntimeError("The independent radial audit did not pass")
    if (
        decision.get("validity_pass") is not True
        or decision.get("scientific_checks_pass") is not True
        or decision.get("decision") != "advance_radial_confidence_k6_development"
    ):
        raise RuntimeError("Unexpected registered radial decision")
    seeds = pd.read_csv(results / "seed_summary.csv").sort_values("seed")
    strata = pd.read_csv(results / "stratified_seed_summary.csv")
    if len(seeds) != EXPECTED_SEEDS or seeds["seed"].nunique() != EXPECTED_SEEDS:
        raise RuntimeError("Expected twelve distinct seed summaries")
    if len(strata) != 48:
        raise RuntimeError("Expected 48 stratified seed summaries")
    numeric = seeds.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Non-finite value in seed summary")
    return seeds, strata, {"decision": decision, "manifest": manifest, "audit": audit}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def _save(fig: plt.Figure, output: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def _plot_registered(seeds: pd.DataFrame, output: Path) -> None:
    specifications = [
        ("gain_vs_k4b", "Radial vs K4b", 0.10),
        ("gain_vs_k4", "Radial vs K4", 0.30),
        ("ch_capture_fraction", "Headroom K4c capturé", 0.40),
    ]
    intervals = [_ci(seeds[column]) for column, _, _ in specifications]
    y = np.arange(len(specifications))
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    for index, ((_, label, threshold), interval) in enumerate(
        zip(specifications, intervals, strict=True)
    ):
        mean = 100.0 * interval["mean"]
        low = 100.0 * interval["low"]
        high = 100.0 * interval["high"]
        ax.errorbar(
            mean,
            index,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            capsize=5,
            linewidth=2.2,
            markersize=8,
            color="#087e8b",
        )
        ax.scatter(
            100.0 * threshold,
            index,
            marker="D",
            facecolor="white",
            edgecolor="#c8553d",
            s=55,
            zorder=4,
        )
        ax.text(high + 1.4, index, f"{mean:.2f}%", va="center", fontweight="bold")
    ax.set_yticks(y, [item[1] for item in specifications])
    ax.set_xlabel("Gain relatif ou fraction de headroom (%)")
    ax.set_title("Pré-écran radial : trois critères d'efficacité préenregistrés")
    ax.legend(
        handles=[
            plt.Line2D([], [], marker="o", color="#087e8b", label="Moyenne ± IC95"),
            plt.Line2D(
                [],
                [],
                marker="D",
                color="#c8553d",
                markerfacecolor="white",
                linestyle="None",
                label="Seuil préenregistré",
            ),
        ],
        loc="center left",
        frameon=False,
    )
    ax.set_xlim(left=0.0)
    fig.tight_layout()
    _save(fig, output, "01_registered_efficacy_gates")


def _plot_mse(seeds: pd.DataFrame, output: Path) -> None:
    columns = [
        ("k4_integrated_mse", "K4"),
        ("k4b_integrated_mse", "K4b"),
        ("k5_1d_integrated_mse", "K5-1D\nprivilégié"),
        ("radial_integrated_mse", "Radial-Y"),
        ("k4c_integrated_mse", "K4c\nprivilégié"),
        ("pointwise_integrated_mse", "Oracle\npointwise"),
    ]
    intervals = [_ci(seeds[column]) for column, _ in columns]
    means = np.array([item["mean"] for item in intervals])
    lows = np.array([item["low"] for item in intervals])
    highs = np.array([item["high"] for item in intervals])
    colors = ["#8d99ae", "#577590", "#f9c74f", "#087e8b", "#43aa8b", "#90be6d"]
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    bars = ax.bar(
        [label for _, label in columns],
        means,
        color=colors,
        yerr=np.vstack([means - lows, highs - means]),
        capsize=4,
    )
    for bar, mean in zip(bars, means, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0012,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("MSE intégrée de la référence (plus faible = mieux)")
    ax.set_title("La remise à l'amplitude G reproduit et dépasse K5-1D")
    fig.tight_layout()
    _save(fig, output, "02_integrated_reference_mse")


def _plot_pairing(seeds: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.3, 5.7))
    x = seeds["k5_1d_integrated_mse"].to_numpy(dtype=float)
    y = seeds["radial_integrated_mse"].to_numpy(dtype=float)
    lower = min(float(x.min()), float(y.min())) * 0.96
    upper = max(float(x.max()), float(y.max())) * 1.04
    ax.plot([lower, upper], [lower, upper], "--", color="#6c757d", label="égalité")
    ax.scatter(x, y, s=62, color="#087e8b", edgecolor="white", linewidth=0.7)
    for _, row in seeds.iterrows():
        ax.annotate(
            str(int(row["seed"]))[-3:],
            (row["k5_1d_integrated_mse"], row["radial_integrated_mse"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7.5,
        )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("MSE K5-1D supervisé")
    ax.set_ylabel("MSE radial-Y sans apprentissage")
    ax.set_title("Comparaison appariée par seed")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, output, "03_radial_vs_k5_1d_by_seed")


def _plot_strata(strata: pd.DataFrame, output: Path) -> None:
    rows: list[tuple[str, dict[str, float]]] = []
    labels = {
        ("noise_regime", "homogeneous"): "Bruit homogène",
        ("noise_regime", "heteroscedastic"): "Bruit hétéroscédastique",
        ("threat", "bitflip_x10"): "Bit-Flip ×10",
        ("threat", "model_replacement"): "Model replacement",
    }
    for key, label in labels.items():
        subset = strata[
            (strata["stratum_type"] == key[0]) & (strata["stratum_value"] == key[1])
        ]
        rows.append((label, _ci(subset["gain_vs_k4b"])))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    y = np.arange(len(rows))
    for index, (label, interval) in enumerate(rows):
        mean, low, high = (100.0 * interval[name] for name in ("mean", "low", "high"))
        ax.errorbar(
            mean,
            index,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            capsize=5,
            linewidth=2.1,
            markersize=8,
            color="#087e8b" if index < 2 else "#f3722c",
        )
        ax.text(high + 1.1, index, f"{mean:.2f}%", va="center", fontweight="bold")
    ax.set_yticks(y, [label for label, _ in rows])
    ax.set_xlabel("Gain radial-Y vs K4b (%)")
    ax.set_title("Le gain reste positif dans chaque régime et chaque attaque")
    ax.set_xlim(left=0.0)
    fig.tight_layout()
    _save(fig, output, "04_gain_by_noise_and_threat")


def _mean_sd(frame: pd.DataFrame, column: str, scale: float = 1.0) -> str:
    values = frame[column].to_numpy(dtype=float) * scale
    return f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"


def _pct_interval(context: MappingLike, key: str) -> str:
    value = context[key]
    return (
        f"{100.0 * value['mean']:.2f} % "
        f"[{100.0 * value['low']:.2f} % ; {100.0 * value['high']:.2f} %]"
    )


MappingLike = dict[str, dict[str, float]]


def _write_reports(
    seeds: pd.DataFrame,
    strata: pd.DataFrame,
    context: dict[str, Any],
    report: Path,
    brief: Path,
) -> None:
    decision = context["decision"]
    intervals: MappingLike = decision["confidence_intervals"]
    method_rows = [
        ("K4", "k4_integrated_mse"),
        ("K4b", "k4b_integrated_mse"),
        ("K5-1D supervisé", "k5_1d_integrated_mse"),
        ("Radial-Y", "radial_integrated_mse"),
        ("K4c privilégié", "k4c_integrated_mse"),
        ("Oracle pointwise", "pointwise_integrated_mse"),
    ]
    table = "\n".join(
        f"| {name} | {_mean_sd(seeds, column)} |" for name, column in method_rows
    )
    radial_wins = int(
        (seeds["radial_integrated_mse"] < seeds["k5_1d_integrated_mse"]).sum()
    )
    regime = decision["stratified_gain_vs_k4b_confidence_intervals"]["noise_regime"]
    threat = decision["stratified_gain_vs_k4b_confidence_intervals"]["threat"]
    norm = decision["norm_and_cap_diagnostics"]
    report_text = rf"""# G0g–K6 : résultat du pré-écran radial

## Verdict

Le pré-écran de développement est **positif** : les sept critères fixés avant
le run passent. La règle

\[
p_t^{{\mathrm{{rad}}}}=
\Pi_{{B_G}}\!\left(
G\,\frac{{Y_t}}{{\max\{{\|Y_t\|_2,r_{{\min}}\}}}}
\right),
\qquad G=0{{,}}13,
\]

réduit la MSE de référence de {_pct_interval(intervals, 'gain_vs_k4b')} face à
K4b et de {_pct_interval(intervals, 'gain_vs_k4')} face à K4. Elle est aussi
meilleure que le contrôle K5-1D de
{-100.0 * intervals['relative_loss_vs_k5_1d']['mean']:.2f} % en moyenne ;
l'intervalle du gain est
[{ -100.0 * intervals['relative_loss_vs_k5_1d']['high']:.2f} % ;
{ -100.0 * intervals['relative_loss_vs_k5_1d']['low']:.2f} %].

Ce résultat **n'est pas encore la validation de K6 Gaussian-aware**. Il valide
le mécanisme préalable : une direction passée remise à une amplitude bornée
explique l'essentiel du gain du prédicteur supervisé K5-1D. Les seeds sont les
seeds d'évaluation K5 déjà consommées ; elles sont réutilisées ici comme
données de développement. Le holdout reste fermé.

![Gates préenregistrés](../figures/gaussian_aware_g0g_k6_radial_prescreen/01_registered_efficacy_gates.png)

## Protocole et validité

- 12 seeds externes, unité statistique des intervalles ;
- 576 histoires et 36 864 enfants Monte-Carlo appariés ;
- MPS en `float32`, avec fallback CPU désactivé ;
- prédicteur construit au plus tard à \(t-1\), avant chaque enfant du tour courant ;
- recomputation exacte de K4 et K4b : erreur MSE maximale nulle ;
- erreur maximale de reproduction vectorielle K4b :
  7,05×10⁻⁹ ;
- cap des contributions respecté ;
- audit post-run indépendant : tous les contrôles passent ;
- aucune seed holdout générée.

## MSE intégrée par méthode

| Méthode | Moyenne ± écart-type entre seeds |
|---|---:|
{table}

![MSE intégrée](../figures/gaussian_aware_g0g_k6_radial_prescreen/02_integrated_reference_mse.png)

Radial-Y bat K5-1D sur {radial_wins}/12 seeds. La fraction moyenne du headroom
K4→K4c capturée est {_pct_interval(intervals, 'ch_capture_fraction')}.

![Comparaison appariée](../figures/gaussian_aware_g0g_k6_radial_prescreen/03_radial_vs_k5_1d_by_seed.png)

## Homogène, hétéroscédastique et attaques

Le gain radial-Y face à K4b reste positif dans chaque strate préenregistrée :

| Strate | Gain moyen | IC95 |
|---|---:|---:|
| Bruit homogène | {100.0 * regime['homogeneous']['mean']:.2f} % | [{100.0 * regime['homogeneous']['low']:.2f} % ; {100.0 * regime['homogeneous']['high']:.2f} %] |
| Bruit hétéroscédastique | {100.0 * regime['heteroscedastic']['mean']:.2f} % | [{100.0 * regime['heteroscedastic']['low']:.2f} % ; {100.0 * regime['heteroscedastic']['high']:.2f} %] |
| Bit-Flip ×10 | {100.0 * threat['bitflip_x10']['mean']:.2f} % | [{100.0 * threat['bitflip_x10']['low']:.2f} % ; {100.0 * threat['bitflip_x10']['high']:.2f} %] |
| Model replacement | {100.0 * threat['model_replacement']['mean']:.2f} % | [{100.0 * threat['model_replacement']['low']:.2f} % ; {100.0 * threat['model_replacement']['high']:.2f} %] |

![Strates](../figures/gaussian_aware_g0g_k6_radial_prescreen/04_gain_by_noise_and_threat.png)

## Ce que le résultat enseigne

La norme passée n'est que {norm['rolling_direction_norm']['mean']:.4f} en
moyenne, tandis que radial-Y vaut pratiquement toujours \(G=0,13\). Le facteur
radial médian vaut {norm['radial_over_rolling_norm']['median']:.2f}, proche du
coefficient 4,527 appris par K5-1D. Le prédicteur radial est au cap dans
{100.0 * norm['at_public_cap_fraction']:.0f} % des 576 histoires.

L'interprétation défendable est donc : **le signal passé donne surtout une
direction utile ; K5-1D apprenait principalement à restaurer son amplitude.**
Les quatre variables supplémentaires de K5-5D étaient redondantes et le
plancher de masse n'expliquait pas cette amplification.

## Limites

1. Le pré-écran contient uniquement les deux menaces évaluées par K5 ; il ne
   démontre pas encore la sécurité bénigne sans attaque.
2. L'amplitude est toujours au cap. Une telle règle peut sur-corriger les cas
   où la direction passée est obsolète ou contaminée.
3. Radial-Y n'est pas noise-aware : aucun moment de covariance n'est encore
   utilisé.
4. Un attaquant persistant peut produire une direction temporellement stable ;
   répétabilité ne signifie pas honnêteté.
5. Les résultats portent sur la MSE synthétique de référence, pas encore sur
   l'accuracy, Worst-20 ou le gap end-to-end.

## Décision suivante

Le résultat autorise la construction K6 complète sur **de nouvelles seeds**.
K6 devra apprendre à réduire l'amplitude lorsque l'accord entre deux fenêtres
passées ne dépasse pas ce que le bruit DP peut expliquer. Le contraste primaire
sera K6 corrigé contre radial-Y : sans gain ou bénéfice de sécurité face à ce
contrôle, la composante Gaussian-aware ne sera pas justifiée.
"""
    report.write_text(report_text, encoding="utf-8")

    brief_text = rf"""# Résultat à présenter lundi — K6 radial

## Message en une phrase

Le pré-écran MPS montre que la direction privée passée est très informative et
que son **amplitude**, plus que cinq features temporelles, expliquait le gain de
K5 : radial-Y gagne {100.0 * intervals['gain_vs_k4b']['mean']:.2f} % contre K4b
et { -100.0 * intervals['relative_loss_vs_k5_1d']['mean']:.2f} % contre K5-1D.

## Chiffres à montrer

| Résultat | Moyenne | IC95 |
|---|---:|---:|
| Gain radial-Y vs K4b | {100.0 * intervals['gain_vs_k4b']['mean']:.2f} % | [{100.0 * intervals['gain_vs_k4b']['low']:.2f} % ; {100.0 * intervals['gain_vs_k4b']['high']:.2f} %] |
| Gain radial-Y vs K4 | {100.0 * intervals['gain_vs_k4']['mean']:.2f} % | [{100.0 * intervals['gain_vs_k4']['low']:.2f} % ; {100.0 * intervals['gain_vs_k4']['high']:.2f} %] |
| Gain radial-Y vs K5-1D | {-100.0 * intervals['relative_loss_vs_k5_1d']['mean']:.2f} % | [{-100.0 * intervals['relative_loss_vs_k5_1d']['high']:.2f} % ; {-100.0 * intervals['relative_loss_vs_k5_1d']['low']:.2f} %] |
| Headroom K4c capturé | {100.0 * intervals['ch_capture_fraction']['mean']:.2f} % | [{100.0 * intervals['ch_capture_fraction']['low']:.2f} % ; {100.0 * intervals['ch_capture_fraction']['high']:.2f} %] |

![Résultat principal](../figures/gaussian_aware_g0g_k6_radial_prescreen/01_registered_efficacy_gates.png)

## Formulation orale

> K5-v2 nous avait dit qu'une moyenne passée suffit, mais son coefficient
> appris de 4,53 restait difficile à interpréter. Nous avons testé sans
> apprentissage la même direction remise directement au cap public. Sur douze
> seeds, 576 histoires et 36 864 enfants appariés, elle réduit la MSE de 59,3 %
> face à K4b et fait 3,0 % mieux que le contrôle appris. Cela montre que le
> passé prédit surtout une direction et que l'amplitude manquante était le
> mécanisme central. Ce n'est pas encore notre résultat Gaussian-aware : la
> prochaine étape doit montrer qu'un accord corrigé de la covariance DP sait
> diminuer cette amplitude quand la direction est due au bruit ou devient
> obsolète.

## Précaution à dire explicitement

Ce résultat est un pré-écran de développement sur des seeds déjà consommées.
Il justifie K6 sur de nouvelles seeds ; il ne valide pas encore un algorithme
end-to-end et n'ouvre pas le holdout.
"""
    brief.write_text(brief_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--brief", type=Path, default=DEFAULT_BRIEF)
    args = parser.parse_args()
    seeds, strata, context = _validate(args.results)
    args.figures.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _style()
    _plot_registered(seeds, args.figures)
    _plot_mse(seeds, args.figures)
    _plot_pairing(seeds, args.figures)
    _plot_strata(strata, args.figures)
    _write_reports(seeds, strata, context, args.report, args.brief)
    summary = {
        "report": str(args.report),
        "brief": str(args.brief),
        "figures": sorted(str(path) for path in args.figures.glob("*.png")),
        "decision": context["decision"]["decision"],
        "independent_audit_pass": context["audit"]["all_checks_pass"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
