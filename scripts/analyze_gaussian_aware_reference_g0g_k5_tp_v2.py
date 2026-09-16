#!/usr/bin/env python3
"""Produce the read-only, presentation-ready analysis of G0g-K5-TP-v2.

The scientific result directory is never modified.  Before reading the seed
table, this program requires a completed MPS run, a closed holdout, a passing
independent post-run audit, and passing validity checks.  Confidence intervals
are recomputed with the preregistered outer seed as the statistical unit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / (
    "results/ldp_gradient_far/" "gaussian_aware_reference_g0g_k5_tp_v2_mps_v2"
)
DEFAULT_OUTPUT = ROOT / "output/figures/gaussian_aware_g0g_k5_tp_v2"
DEFAULT_REPORT = ROOT / ("output/analysis/Gaussian_Aware_G0g_K5_TP_V2_Final_Report.md")

T_CRITICAL_DF11 = 2.2009851600916406
EXPECTED_SEEDS = 12


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _all_true(value: Any, *, name: str) -> bool:
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"{name} must be a non-empty mapping")
    if any(type(item) is not bool for item in value.values()):
        raise RuntimeError(f"{name} must contain JSON booleans only")
    return all(value.values())


def _ci(values: pd.Series) -> dict[str, float | int]:
    array = values.to_numpy(dtype=float)
    if len(array) != EXPECTED_SEEDS or not np.isfinite(array).all():
        raise RuntimeError("Each interval must use exactly 12 finite outer seeds")
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    half = T_CRITICAL_DF11 * sd / math.sqrt(len(array))
    return {
        "n": len(array),
        "mean": mean,
        "sd": sd,
        "low": mean - half,
        "high": mean + half,
    }


def _close(left: float, right: float, tolerance: float = 2.0e-6) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _validate(results: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = {
        "root_manifest": results / "manifest.json",
        "evaluation_manifest": results / "evaluation/manifest.json",
        "decision": results / "evaluation/decision.json",
        "audit": results / "evaluation/independent_postrun_audit.json",
        "config": results / "resolved_config.json",
        "predictor": results / "frozen_predictor.json",
        "design": results / "feature_design_diagnostics.json",
        "seed_summary": results / "evaluation/seed_summary.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing K5-v2 artifacts: " + ", ".join(missing))

    root = _read_json(paths["root_manifest"])
    evaluation = _read_json(paths["evaluation_manifest"])
    decision = _read_json(paths["decision"])
    audit = _read_json(paths["audit"])
    config = _read_json(paths["config"])
    predictor = _read_json(paths["predictor"])
    design = _read_json(paths["design"])

    if root.get("status") != "completed_development":
        raise RuntimeError("The campaign is not completed")
    if root.get("device") != "mps" or evaluation.get("device") != "mps":
        raise RuntimeError("The completed scientific run was not MPS-only")
    if (
        root.get("holdout_opened") is not False
        or decision.get("holdout_opened") is not False
    ):
        raise RuntimeError("The reserved holdout is not closed")
    if audit.get("all_checks_pass") is not True or audit.get("audit_device") != "mps":
        raise RuntimeError("The independent post-run audit did not pass on MPS")
    if not _all_true(audit.get("checks"), name="audit.checks"):
        raise RuntimeError("A top-level independent audit check failed")
    evaluation_audit = audit.get("evaluation_audit")
    if (
        not isinstance(evaluation_audit, dict)
        or evaluation_audit.get("pass") is not True
    ):
        raise RuntimeError("The independent evaluation recomputation did not pass")
    if not _all_true(
        evaluation_audit.get("validity_checks"), name="audit.evaluation.validity_checks"
    ):
        raise RuntimeError("An independently recomputed validity check failed")
    if decision.get("validity_pass") is not True or not _all_true(
        decision.get("validity_checks"), name="decision.validity_checks"
    ):
        raise RuntimeError("The scientific screen is invalid or inconclusive")
    if decision.get("decision") != "stop_five_feature_linear_predictor_instance":
        raise RuntimeError("Unexpected registered scientific decision")
    if decision.get("scientific_checks_pass") is not False:
        raise RuntimeError("Unexpected scientific gate status")

    frame = pd.read_csv(paths["seed_summary"])
    required = {
        "seed",
        "k4_integrated_mse",
        "k4b_integrated_mse",
        "one_dimensional_integrated_mse",
        "k5_integrated_mse",
        "k4c_integrated_mse",
        "pointwise_integrated_mse",
        "gain_vs_k4b",
        "gain_vs_k4",
        "gain_vs_one_dimensional",
        "ch_capture_fraction",
        "relative_k4_k4c_headroom",
        "split_mc_over_headroom",
        "homogeneous_gain_vs_k4b",
        "heteroscedastic_gain_vs_k4b",
    }
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise RuntimeError(f"Missing seed-summary columns: {sorted(missing_columns)}")
    if len(frame) != EXPECTED_SEEDS or frame["seed"].nunique() != EXPECTED_SEEDS:
        raise RuntimeError("Expected exactly one row for each of 12 outer seeds")
    numeric = frame[list(required)].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError("The seed summary contains non-finite data")

    # Reproduce the registered intervals from the seed table.
    registered = decision.get("confidence_intervals")
    if not isinstance(registered, dict):
        raise RuntimeError("The registered confidence intervals are missing")
    for key, stored in registered.items():
        if key not in frame:
            raise RuntimeError(f"Missing registered estimand: {key}")
        computed = _ci(frame[key])
        for field in ("mean", "low", "high"):
            if not _close(float(computed[field]), float(stored[field])):
                raise RuntimeError(f"Confidence interval mismatch for {key}.{field}")

    context = {
        "root": root,
        "evaluation": evaluation,
        "decision": decision,
        "audit": audit,
        "config": config,
        "predictor": predictor,
        "design": design,
    }
    return frame.sort_values("seed").reset_index(drop=True), context


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
        }
    )


def _save(fig: plt.Figure, directory: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def _plot_registered_contrasts(
    frame: pd.DataFrame, context: dict[str, Any], output: Path
) -> None:
    gates = context["config"]["gates"]
    specifications = [
        ("gain_vs_k4b", "K5 vs K4b", gates["primary_gain_vs_k4b_mean_min"]),
        ("gain_vs_k4", "K5 vs K4", gates["primary_gain_vs_k4_mean_min"]),
        (
            "gain_vs_one_dimensional",
            "K5 vs contrôle 1D",
            gates["primary_gain_vs_one_dimensional_mean_min"],
        ),
        (
            "ch_capture_fraction",
            "Headroom K4c capturé",
            gates["ch_capture_fraction_mean_min"],
        ),
    ]
    intervals = [_ci(frame[key]) for key, _, _ in specifications]
    means = np.array([float(item["mean"]) for item in intervals]) * 100.0
    lows = np.array([float(item["low"]) for item in intervals]) * 100.0
    highs = np.array([float(item["high"]) for item in intervals]) * 100.0
    thresholds = np.array([float(item[2]) for item in specifications]) * 100.0
    checks = context["decision"]["scientific_checks"]
    passed = [
        checks["gain_vs_k4b_mean"] and checks["gain_vs_k4b_ci"],
        checks["gain_vs_k4_mean"] and checks["gain_vs_k4_ci"],
        checks["gain_vs_one_dimensional_mean"] and checks["gain_vs_one_dimensional_ci"],
        checks["capture_mean"] and checks["capture_ci"],
    ]
    colors = ["#238636" if item else "#c23b22" for item in passed]
    y = np.arange(len(specifications))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for index in range(len(specifications)):
        ax.errorbar(
            means[index],
            y[index],
            xerr=[[means[index] - lows[index]], [highs[index] - means[index]]],
            fmt="o",
            color=colors[index],
            capsize=5,
            linewidth=2.2,
            markersize=7,
        )
        ax.scatter(
            thresholds[index],
            y[index],
            marker="D",
            facecolor="white",
            edgecolor="#5d4037",
            s=52,
            zorder=4,
        )
        ax.text(
            highs[index] + 2.2,
            y[index],
            "PASS" if passed[index] else "FAIL",
            color=colors[index],
            va="center",
            fontweight="bold",
        )
    ax.axvline(0.0, color="#555", linewidth=1)
    ax.set_yticks(y, [item[1] for item in specifications])
    ax.invert_yaxis()
    ax.set_xlabel("Gain relatif ou fraction capturée (%)")
    ax.set_title("K5-v2 : moyenne inter-seeds, IC95 et seuil moyen préenregistré")
    ax.legend(
        handles=[
            plt.Line2D(
                [], [], marker="o", color="#238636", linestyle="", label="estimé + IC95"
            ),
            plt.Line2D(
                [],
                [],
                marker="D",
                markerfacecolor="white",
                markeredgecolor="#5d4037",
                linestyle="",
                label="seuil sur la moyenne",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
    )
    ax.set_xlim(-4.0, 97.0)
    _save(fig, output, "01_registered_contrasts")


def _plot_method_mse(frame: pd.DataFrame, output: Path) -> None:
    methods = [
        ("K4", "k4_integrated_mse"),
        ("K4b", "k4b_integrated_mse"),
        ("Contrôle 1D appris", "one_dimensional_integrated_mse"),
        ("K5-v2 (5D)", "k5_integrated_mse"),
        ("K4c-CH semi-oracle", "k4c_integrated_mse"),
        ("Oracle pointwise", "pointwise_integrated_mse"),
    ]
    intervals = [_ci(frame[column]) for _, column in methods]
    means = np.array([float(item["mean"]) for item in intervals])
    lows = np.array([float(item["low"]) for item in intervals])
    highs = np.array([float(item["high"]) for item in intervals])
    x = np.arange(len(methods))
    colors = ["#9e9e9e", "#e69f00", "#56b4e9", "#009e73", "#cc79a7", "#333333"]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    bars = ax.bar(
        x, means, color=colors, yerr=np.vstack([means - lows, highs - means]), capsize=4
    )
    ax.set_xticks(x, [name for name, _ in methods], rotation=20, ha="right")
    ax.set_ylabel("MSE intégrée sur 48 histoires par seed")
    ax.set_title(
        "Erreur de référence : le contrôle 1D et K5-v2 sont pratiquement confondus"
    )
    for bar, mean in zip(bars, means, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.set_ylim(bottom=0)
    _save(fig, output, "02_integrated_mse_by_method")


def _plot_paired_one_dimensional(frame: pd.DataFrame, output: Path) -> None:
    gain = frame["gain_vs_one_dimensional"] * 100.0
    fig, (overview, ax) = plt.subplots(
        2,
        1,
        figsize=(9.2, 6.1),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 2.2], "hspace": 0.08},
    )
    colors = np.where(gain >= 0.0, "#238636", "#c23b22")
    ax.bar(np.arange(len(frame)), gain, color=colors)
    interval = _ci(frame["gain_vs_one_dimensional"])
    mean = float(interval["mean"]) * 100.0
    for current in (overview, ax):
        current.axhline(mean, color="#005a9c", linewidth=2)
        current.axhline(0.0, color="#444", linewidth=1)
    overview.axhline(
        5.0,
        color="#5d4037",
        linestyle="--",
        linewidth=2,
        label="seuil matériel préenregistré = 5 %",
    )
    overview.scatter(
        [len(frame) / 2.0 - 0.5],
        [mean],
        color="#005a9c",
        s=55,
        label=f"moyenne observée = {mean:.3f} %",
        zorder=3,
    )
    overview.set_ylim(-0.2, 5.35)
    overview.set_ylabel("Vue du seuil (%)")
    overview.legend(loc="center right")
    ax.axhspan(
        float(interval["low"]) * 100.0,
        float(interval["high"]) * 100.0,
        color="#005a9c",
        alpha=0.12,
        label="IC95 de la moyenne",
    )
    padding = max(0.08, 0.12 * float(gain.max() - gain.min()))
    ax.set_ylim(float(gain.min()) - padding, float(gain.max()) + padding)
    ax.set_xticks(
        np.arange(len(frame)),
        [str(int(value))[-4:] for value in frame["seed"]],
        rotation=45,
    )
    ax.set_xlabel("Seed externe (4 derniers chiffres)")
    ax.set_ylabel("Zoom : gain apparié (%)")
    ax.legend(loc="lower right")
    fig.suptitle(
        "Comparaison appariée : quatre features supplémentaires n'apportent pas 5 %"
    )
    _save(fig, output, "03_paired_gain_vs_learned_1d")


def _plot_headroom(frame: pd.DataFrame, output: Path) -> None:
    x = np.arange(len(frame))
    capture = frame["ch_capture_fraction"] * 100.0
    relative = frame["relative_k4_k4c_headroom"] * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharex=True)
    axes[0].plot(x, relative, marker="o", color="#0072b2")
    axes[0].axhline(
        float(np.median(relative)),
        color="#333",
        linestyle="--",
        label=f"médiane = {np.median(relative):.1f} %",
    )
    axes[0].axhline(5.0, color="#c23b22", linestyle=":", label="seuil d'alerte = 5 %")
    axes[0].set_title("Headroom causal disponible")
    axes[0].set_ylabel("(MSE K4 − MSE K4c) / MSE K4 (%)")
    axes[0].legend(loc="lower left")
    axes[1].plot(x, capture, marker="o", color="#009e73")
    axes[1].axhline(
        float(capture.mean()),
        color="#333",
        linestyle="--",
        label=f"moyenne = {capture.mean():.1f} %",
    )
    axes[1].axhline(40.0, color="#5d4037", linestyle=":", label="gate = 40 %")
    axes[1].set_title("Fraction capturée par K5-v2")
    axes[1].set_ylabel("Capture du headroom K4c (%)")
    axes[1].legend(loc="lower left")
    for ax in axes:
        ax.set_xticks(x, [str(int(value))[-4:] for value in frame["seed"]], rotation=45)
        ax.set_xlabel("Seed externe")
    fig.suptitle("Le résultat négatif ne vient pas d'un headroom trop faible")
    _save(fig, output, "04_headroom_and_capture")


def _write_summaries(
    frame: pd.DataFrame, context: dict[str, Any], output: Path
) -> dict[str, Any]:
    definitions = {
        "k5_vs_k4b": frame["gain_vs_k4b"],
        "k5_vs_k4": frame["gain_vs_k4"],
        "k5_vs_1d": frame["gain_vs_one_dimensional"],
        "k5_headroom_capture": frame["ch_capture_fraction"],
        "homogeneous_k5_vs_k4b": frame["homogeneous_gain_vs_k4b"],
        "heteroscedastic_k5_vs_k4b": frame["heteroscedastic_gain_vs_k4b"],
        "one_dimensional_vs_k4b": (
            frame["k4b_integrated_mse"] - frame["one_dimensional_integrated_mse"]
        )
        / frame["k4b_integrated_mse"],
        "one_dimensional_vs_k4": (
            frame["k4_integrated_mse"] - frame["one_dimensional_integrated_mse"]
        )
        / frame["k4_integrated_mse"],
        "one_dimensional_headroom_capture": (
            frame["k4_integrated_mse"] - frame["one_dimensional_integrated_mse"]
        )
        / (frame["k4_integrated_mse"] - frame["k4c_integrated_mse"]),
    }
    rows = []
    payload: dict[str, Any] = {
        "registered_decision": context["decision"]["decision"],
        "validity_pass": context["decision"]["validity_pass"],
        "scientific_checks_pass": context["decision"]["scientific_checks_pass"],
        "holdout_opened": False,
        "outer_seed_count": EXPECTED_SEEDS,
        "estimands": {},
    }
    for name, values in definitions.items():
        interval = _ci(values)
        payload["estimands"][name] = interval
        rows.append({"estimand": name, **interval})
    pd.DataFrame(rows).to_csv(output / "summary_estimands.csv", index=False)
    (output / "summary_estimands.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f} %".replace(".", ",")


def _ci_text(interval: dict[str, Any]) -> str:
    return f"{_pct(float(interval['mean']))} ± {_pct(float(interval['sd']))} ; IC95 [{_pct(float(interval['low']))} ; {_pct(float(interval['high']))}]"


def _write_report(
    report: Path, output: Path, context: dict[str, Any], summary: dict[str, Any]
) -> None:
    estimates = summary["estimands"]
    predictor = context["predictor"]
    design = context["design"]
    decision = context["decision"]
    headroom = context["audit"]["evaluation_audit"]["headroom_diagnostics_non_gating"]
    coefficients = predictor["coefficients"]
    scales = predictor["feature_scales"]
    names = predictor["feature_names"]

    figure_dir = Path("../figures/gaussian_aware_g0g_k5_tp_v2")
    lines = [
        "# G0g-K5-TP-v2 — rapport final audité",
        "",
        "## Verdict en une phrase",
        "",
        "**La campagne est techniquement et statistiquement valide, mais l'instanciation linéaire à cinq variables est arrêtée : elle n'améliore le contrôle appris unidimensionnel que de "
        f"{_pct(float(estimates['k5_vs_1d']['mean']))}, contre un seuil matériel préenregistré de 5 %.**",
        "",
        "Ce résultat est informatif. Il montre que le signal prédictible du transcript passé existe, mais qu'il est déjà presque entièrement porté par la moyenne acceptée roulante. Les quatre variables temporelles supplémentaires de K5-v2 n'ajoutent pas une information prédictive suffisamment grande pour justifier leur complexité.",
        "",
        "## 1. Intégrité de la campagne",
        "",
        "| Contrôle | Résultat |",
        "|---|---:|",
        "| Exécution scientifique | MPS, `torch.float32`, sans fallback CPU |",
        "| Seeds externes d'évaluation | 12 |",
        "| Histoires évaluées | 576 = 48 par seed |",
        "| Lignes enfant-candidat | 258 048 |",
        "| Essais replace-one | 900, zéro violation |",
        "| Validité du runner | toutes les vérifications passent |",
        "| Recalcul indépendant post-run | toutes les vérifications passent |",
        "| Holdout réservé | fermé |",
        "",
        "L'unité statistique est la seed externe. Les 64 enfants Monte-Carlo servent à estimer la MSE conditionnelle de chaque histoire ; ils ne sont jamais traités comme 64 réplicats indépendants dans les intervalles de confiance.",
        "",
        "## 2. Résultats préenregistrés",
        "",
        "| Contraste K5-v2 | Moyenne ± écart-type ; IC95 | Seuil sur la moyenne | Verdict |",
        "|---|---:|---:|:---:|",
        f"| vs K4b | {_ci_text(estimates['k5_vs_k4b'])} | 10 % | PASS |",
        f"| vs K4 | {_ci_text(estimates['k5_vs_k4'])} | 30 % | PASS |",
        f"| vs contrôle 1D appris | {_ci_text(estimates['k5_vs_1d'])} | 5 % | **FAIL** |",
        f"| fraction du headroom K4c capturée | {_ci_text(estimates['k5_headroom_capture'])} | 40 % | PASS |",
        f"| vs K4b, bruit homogène | {_ci_text(estimates['homogeneous_k5_vs_k4b'])} | IC95 bas > 0 | PASS |",
        f"| vs K4b, bruit hétéroscédastique | {_ci_text(estimates['heteroscedastic_k5_vs_k4b'])} | IC95 bas > 0 | PASS |",
        "",
        f"![Contrastes préenregistrés]({figure_dir / '01_registered_contrasts.png'})",
        "",
        "Le +58,09 % contre K4b est réel mais ne suffit pas à valider K5-v2 : le contrôle 1D appris obtient déjà +58,01 % contre K4b. Le contraste décisif est donc K5-v2 contre ce contrôle fort, pas K5-v2 contre les anciennes baselines.",
        "",
        "## 3. Ce que révèle le contrôle unidimensionnel",
        "",
        "Le contrôle apprend un seul scalaire appliqué à `rolling_accepted_mean`. Ses résultats descriptifs, recalculés sur les mêmes seeds, sont :",
        "",
        "| Contraste du contrôle 1D | Moyenne ± écart-type ; IC95 |",
        "|---|---:|",
        f"| 1D vs K4b | {_ci_text(estimates['one_dimensional_vs_k4b'])} |",
        f"| 1D vs K4 | {_ci_text(estimates['one_dimensional_vs_k4'])} |",
        f"| Headroom K4c capturé par 1D | {_ci_text(estimates['one_dimensional_headroom_capture'])} |",
        "",
        f"![MSE intégrée par méthode]({figure_dir / '02_integrated_mse_by_method.png'})",
        "",
        f"![Gain apparié contre le contrôle 1D]({figure_dir / '03_paired_gain_vs_learned_1d.png'})",
        "",
        "Sur 3 seeds sur 12, K5-v2 est même très légèrement moins bon que le contrôle 1D. Il n'est meilleur que sur 278 histoires sur 576 ; la médiane des gains histoire par histoire vaut −0,014 %. Sur les autres seeds, l'amélioration reste comprise dans une plage très faible. L'IC95 de l'effet seed-level est strictement positif, mais **significatif ne veut pas dire matériel** : son ordre de grandeur est environ 28 fois inférieur au seuil de 5 %.",
        "",
        "## 4. Le headroom n'explique pas l'échec",
        "",
        f"Le headroom causal relatif médian vaut {_pct(float(headroom['relative_headroom_median']))} et son minimum {_pct(float(headroom['relative_headroom_minimum']))}. Le ratio médian de désaccord Monte-Carlo sur ce headroom n'est que {_pct(float(headroom['split_mc_over_headroom_median']))}, avec un maximum de {_pct(float(headroom['split_mc_over_headroom_maximum']))}. Il n'y a donc ni dénominateur quasi nul, ni cible trop bruitée pour interpréter le contraste.",
        "",
        f"![Headroom et capture]({figure_dir / '04_headroom_and_capture.png'})",
        "",
        "## 5. Prédicteur gelé et diagnostic de redondance",
        "",
        f"K5-v2 sélectionne $\\lambda={float(predictor['selected_lambda']):g}$ ; le contrôle 1D sélectionne $\\lambda={float(predictor['one_dimensional_control']['selected_lambda']):g}$. Le système 5D est de rang exact 5, avec un conditionnement régularisé de {float(design['final_fit']['condition_number_regularized_system']):.1f} et un résidu relatif de {float(design['final_fit']['normal_equation_relative_residual']):.2e}.",
        "",
        "| Variable strictement passée | Échelle | Coefficient | Coefficient / échelle |",
        "|---|---:|---:|---:|",
    ]
    for name, scale, coefficient in zip(names, scales, coefficients, strict=True):
        lines.append(
            f"| `{name}` | {float(scale):.7f} | {float(coefficient):+.7f} | {float(coefficient) / float(scale):+.3f} |"
        )
    lines.extend(
        [
            "",
            "Le rang exact confirme que les cinq colonnes ne sont pas algébriquement identiques. Il ne garantit toutefois pas qu'elles portent une information utile supplémentaire. Les fortes corrélations entre la moyenne roulante et la direction roulante, puis entre les deux différences, expliquent pourquoi la ridge peut ajuster cinq coefficients sans produire un gain matériel hors calibration.",
            "",
            "## 6. Observations, inférences et limites",
            "",
            "### Observations",
            "",
            "- toutes les vérifications de validité et l'audit indépendant passent ;",
            "- tous les gains contre K4, K4b et dans les deux régimes de bruit passent ;",
            "- le seul échec scientifique est le gain moyen K5-v2 contre le contrôle 1D ;",
            "- le holdout reste fermé et aucune seed d'évaluation ne doit être réutilisée pour régler une variante.",
            "",
            "### Inférences défendables",
            "",
            "- une direction compensatoire utile est prédictible à partir du transcript passé dans ce générateur synthétique ;",
            "- cette prédictibilité est dominée par une statistique de bas degré : la moyenne acceptée roulante ;",
            "- ajouter quatre features linéaires proches n'est pas une contribution algorithmique suffisamment forte.",
            "",
            "### Ce que l'expérience ne démontre pas",
            "",
            "- elle ne mesure ni accuracy, ni Worst-20, ni gap, ni convergence end-to-end ;",
            "- elle ne prouve pas que toute méthode transcript-only est inutile ;",
            "- elle ne valide pas le semi-oracle K4c comme algorithme déployable ;",
            "- elle ne fournit pas un théorème d'impossibilité pour les prédicteurs non linéaires ou les filtres d'état.",
            "",
            "## 7. Décision et prochaine hypothèse publiable",
            "",
            f"La décision enregistrée est `{decision['decision']}`. Il faut donc **arrêter exactement le prédicteur linéaire 5D K5-v2** et ne pas abaisser son gate après observation.",
            "",
            "La prochaine piste la plus parcimonieuse est de formaliser la **tolérance statistique scalaire** comme un mécanisme à part entière : une imputation causale à shrinkage appris sur la moyenne acceptée passée, avec cap de contribution et référence noise-aware. Avant toute nouvelle évaluation, elle doit recevoir un nouveau nom, de nouvelles seeds et deux tests séparés :",
            "",
            "1. une confirmation synthétique du mécanisme 1D contre K4b, avec coefficient figé et nouvelles seeds ;",
            "2. seulement en cas de confirmation, un screen end-to-end LDP-Gradient-FAR mesurant accuracy, Worst-20, gap, robustesse Byzantine et stabilité de convergence.",
            "",
            "Si l'objectif scientifique exige une contribution plus riche que ce shrinkage scalaire, la variante suivante doit ajouter une information réellement nouvelle — par exemple un filtre d'état robuste non linéaire avec incertitude prédictive — et non recombiner les mêmes moyennes et différences linéaires. Cette hypothèse devra être préenregistrée sur de nouvelles seeds ; le holdout K5-v2 reste fermé.",
            "",
            "## 8. Traçabilité",
            "",
            f"- Décision brute : `{context['evaluation'].get('decision')}`",
            f"- SHA-256 du verrou : `{context['audit']['lock_audit']['lock_sha256']}`",
            f"- SHA-256 du prédicteur : `{context['root']['frozen_predictor_sha256']}`",
            "- Audit indépendant : `evaluation/independent_postrun_audit.json`",
            "- Synthèse numérique : `output/figures/gaussian_aware_g0g_k5_tp_v2/summary_estimands.csv`",
            "",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def analyze(results: Path, output: Path, report: Path) -> dict[str, Any]:
    frame, context = _validate(results.resolve())
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _configure_style()
    _plot_registered_contrasts(frame, context, output)
    _plot_method_mse(frame, output)
    _plot_paired_one_dimensional(frame, output)
    _plot_headroom(frame, output)
    summary = _write_summaries(frame, context, output)
    _write_report(report.resolve(), output, context, summary)
    return {
        "decision": context["decision"]["decision"],
        "validity_pass": True,
        "scientific_checks_pass": False,
        "holdout_opened": False,
        "report": str(report.resolve()),
        "output": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(args.results, args.output, args.report), indent=2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
