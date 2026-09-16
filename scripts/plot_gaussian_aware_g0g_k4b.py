#!/usr/bin/env python3
"""Create readable seed-level figures for a completed G0g-K4b screen.

Every plotted quantity is averaged across contexts *within each seed* before
the mean and sample standard deviation are computed across the five seeds.
The script reads only ``manifest.json`` until the campaign reports
``completed_development`` and requires a passing independent raw-CSV audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / (
    "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k4b_past_imputation.yaml"
)
DEFAULT_RESULTS = ROOT / (
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k4b_past_imputation_mps_v1"
)
DEFAULT_OUTPUT = ROOT / "output/figures/gaussian_aware_g0g_k4b"

CAMPAIGN_ID = "gaussian_aware_reference_g0g_k4b_past_imputation_mps_v1"
K2 = "g0g_k2"
K4 = "g0g_k4_temporal_causal_gate"
PRIMARY = "g0g_k4b_full_temporal_missing_slot_imputation"
DELTA = "g0g_k4b_incremental_temporal_suppression_imputation"
ORACLE = "g0g_k4b_pointwise_optimal_oracle"

THREATS = ("alie", "ipm", "bitflip_x10", "model_replacement")
THREAT_LABELS = {
    "alie": "ALIE",
    "ipm": "IPM",
    "bitflip_x10": "Bit-Flip ×10",
    "model_replacement": "Model replacement",
}
CANDIDATE_LABELS = {
    K2: "K2 — gate courant",
    K4: "K4 — gate temporel",
    PRIMARY: "K4b-B — mélange historique",
    DELTA: "K4b-A — imputation incrémentale",
    ORACLE: "Oracle B — non déployable",
}
CANDIDATE_SHORT = {
    K2: "K2",
    K4: "K4",
    PRIMARY: "K4b-B",
    DELTA: "K4b-A",
    ORACLE: "Oracle B",
}
COLORS = {
    K2: "#E69F00",
    K4: "#0072B2",
    PRIMARY: "#009E73",
    DELTA: "#CC79A7",
    ORACLE: "#4D4D4D",
}


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linestyle": "--",
            "lines.linewidth": 2.0,
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guard_completed(results_dir: Path) -> dict[str, Any]:
    """Read only the manifest before deciding whether raw results may open."""

    path = results_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing K4b manifest: {path}")
    manifest = _read_json(path)
    if manifest.get("status") != "completed_development":
        raise RuntimeError(
            "K4b is not complete; figures did not open raw results: "
            f"status={manifest.get('status')!r}"
        )
    if manifest.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError(f"Unexpected campaign: {manifest.get('campaign_id')!r}")
    return manifest


def _as_bool(series: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    lowered = series.astype(str).str.strip().str.lower()
    if not bool(lowered.isin(("true", "false")).all()):
        raise ValueError(f"{name} is not Boolean")
    return lowered.eq("true")


def _validate_sources(
    results_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    manifest = _guard_completed(results_dir)
    required = {
        "decision": results_dir / "decision.json",
        "audit": results_dir / "independent_audit.json",
        "rounds": results_dir / "development_round_rows.csv",
        "clients": results_dir / "development_client_rows.csv",
        "trajectories": results_dir / "development_trajectory_rows.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Completed K4b directory is incomplete: " + ", ".join(missing)
        )
    decision = _read_json(required["decision"])
    audit = _read_json(required["audit"])
    if manifest.get("device") != "mps" or manifest.get("dtype") != "torch.float32":
        raise RuntimeError("Figures require the completed MPS/float32 screen")
    if manifest.get("holdout_opened") or decision.get("holdout_opened"):
        raise RuntimeError("Reserved holdout must remain closed")
    if manifest.get("end_to_end_deployability_claimed") is not False:
        raise RuntimeError("This semi-oracle-conditioned screen is not end-to-end")
    if (
        audit.get("audit_verdict") != "PASS"
        or audit.get("failed_audit_checks")
        or audit.get("reported_decision_difference_count") != 0
    ):
        raise RuntimeError("Independent K4b raw-CSV audit must pass before plotting")
    audited_hashes = audit.get("raw_file_sha256", {})
    for name in ("decision", "rounds", "clients", "trajectories"):
        if _sha256(required[name]) != audited_hashes.get(name):
            raise RuntimeError(f"{name} changed after the independent K4b audit")
    rounds = pd.read_csv(required["rounds"], low_memory=False)
    clients = pd.read_csv(required["clients"], low_memory=False)
    trajectories = pd.read_csv(required["trajectories"], low_memory=False)
    expected = audit["expected_counts"]
    for name, frame in (
        ("round_rows", rounds),
        ("client_rows", clients),
        ("trajectory_rows", trajectories),
    ):
        if len(frame) != int(expected[name]):
            raise RuntimeError(
                f"{name}: expected {expected[name]}, observed {len(frame)}"
            )
    seeds = sorted(int(value) for value in trajectories["seed"].unique())
    if len(seeds) != 5:
        raise RuntimeError(f"Expected five development seeds, got {seeds}")
    clients["latent_byzantine_bool"] = _as_bool(
        clients["latent_byzantine"], name="clients.latent_byzantine"
    )
    clients["honest_outlier_bool"] = _as_bool(
        clients["honest_outlier"], name="clients.honest_outlier"
    )
    clients["honest_regular_bool"] = _as_bool(
        clients["honest_regular"], name="clients.honest_regular"
    )
    return manifest, decision, audit, rounds, clients, trajectories


def _seed_values(
    frame: pd.DataFrame,
    *,
    value: str,
    axes: list[str],
) -> pd.DataFrame:
    """Average all contexts within each seed and retain five independent units."""

    result = (
        frame.groupby(["seed", *axes], observed=True, sort=True)[value]
        .mean()
        .rename("value")
        .reset_index()
    )
    counts = result.groupby(axes, observed=True)["seed"].nunique()
    if not bool(counts.eq(5).all()):
        raise RuntimeError(
            f"{value}: at least one estimand does not contain five seeds"
        )
    return result


def _mean_sd(seed_values: pd.DataFrame, *, axes: list[str]) -> pd.DataFrame:
    result = (
        seed_values.groupby(axes, observed=True, sort=True)["value"]
        .agg(mean="mean", sd="std", n_seeds="count")
        .reset_index()
    )
    if not bool(result["n_seeds"].eq(5).all()):
        raise RuntimeError("Inter-seed summary is incomplete")
    result["sd"] = result["sd"].fillna(0.0)
    return result


def _phase_background(ax: plt.Axes) -> None:
    phases = [
        (0.5, 8.5, "Enrôlement", "#D9EAF7"),
        (8.5, 12.5, "Monitoring", "#ECECEC"),
        (12.5, 24.5, "Attaque", "#FADBD8"),
        (24.5, 36.5, "Récupération", "#D5F5E3"),
    ]
    for left, right, label, color in phases:
        ax.axvspan(left, right, color=color, alpha=0.45, zorder=0)
        ax.text(
            (left + right) / 2,
            0.985,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.5,
            fontweight="bold",
        )


def plot_attack_effects(trajectories: pd.DataFrame, output_dir: Path) -> Path:
    candidates = (K2, K4, PRIMARY, DELTA, ORACLE)
    selected = trajectories.loc[
        trajectories["schedule"].eq("persistent")
        & trajectories["threat"].isin(THREATS)
        & trajectories["candidate"].isin(candidates)
    ]
    seeds = _seed_values(selected, value="attack_auc", axes=["threat", "candidate"])
    summary = _mean_sd(seeds, axes=["threat", "candidate"])
    order = list(candidates)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), sharey=True)
    for ax, threat in zip(axes.flat, THREATS, strict=True):
        data = seeds.loc[seeds["threat"].eq(threat)]
        stats = summary.loc[summary["threat"].eq(threat)].set_index("candidate")
        x = np.arange(len(order), dtype=float)
        pivot = data.pivot(index="seed", columns="candidate", values="value")[order]
        for _, row in pivot.iterrows():
            ax.plot(x, row.to_numpy(float), color="#B6B6B6", alpha=0.62, linewidth=0.9)
            ax.scatter(x, row.to_numpy(float), color="#B6B6B6", s=14, alpha=0.75)
        for position, candidate in enumerate(order):
            ax.errorbar(
                position,
                stats.loc[candidate, "mean"],
                yerr=stats.loc[candidate, "sd"],
                fmt="o",
                markersize=7,
                capsize=4,
                color=COLORS[candidate],
                label=CANDIDATE_LABELS[candidate],
                zorder=4,
            )
        ax.set_xticks(x, [CANDIDATE_SHORT[value] for value in order], rotation=18)
        ax.set_title(THREAT_LABELS[threat])
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    fig.supylabel(
        "Erreur moyenne pendant l’attaque (plus faible = meilleur)",
        x=0.015,
        fontsize=11,
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "K4b face à K2, K4 et à l’oracle B — attaques persistantes",
        fontsize=15,
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        "Traits gris : cinq seeds appariées. Points colorés : moyenne ± 1 SD inter-seeds, "
        "après moyenne intra-seed sur bruit, permutation, géométrie et dynamique. "
        "Oracle B : diagnostic non déployable.",
        ha="center",
        fontsize=8.7,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.895,
        bottom=0.185,
        hspace=0.34,
        wspace=0.10,
    )
    path = output_dir / "01_k4b_effect_by_persistent_attack.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    seeds.to_csv(output_dir / "01_k4b_effect_seed_values.csv", index=False)
    return path


def plot_temporal_error(rounds: pd.DataFrame, output_dir: Path) -> Path:
    candidates = (K2, K4, PRIMARY, ORACLE)
    selected = rounds.loc[
        rounds["schedule"].eq("persistent")
        & rounds["threat"].isin(THREATS)
        & rounds["candidate"].isin(candidates)
    ]
    seeds = _seed_values(
        selected,
        value="reference_error",
        axes=["threat", "round", "candidate"],
    )
    summary = _mean_sd(seeds, axes=["threat", "round", "candidate"])
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.9), sharex=True, sharey=True)
    for ax, threat in zip(axes.flat, THREATS, strict=True):
        _phase_background(ax)
        for candidate in candidates:
            data = summary.loc[
                summary["threat"].eq(threat) & summary["candidate"].eq(candidate)
            ].sort_values("round")
            x = data["round"].to_numpy(float)
            mean = data["mean"].to_numpy(float)
            sd = data["sd"].to_numpy(float)
            ax.plot(x, mean, color=COLORS[candidate], label=CANDIDATE_LABELS[candidate])
            ax.fill_between(
                x, mean - sd, mean + sd, color=COLORS[candidate], alpha=0.10
            )
        ax.axvline(13, color="#8B0000", linestyle=":", linewidth=1.1)
        ax.axvline(25, color="#006400", linestyle=":", linewidth=1.1)
        ax.set_title(THREAT_LABELS[threat])
        ax.set_xlim(1, 36)
        # The vertical lines retain the exact phase boundaries (13 and 25);
        # sparse tick labels prevent the adjacent 12/13 and 24/25 labels from
        # colliding in the rendered figure.
        ax.set_xticks([1, 8, 13, 18, 24, 30, 36])
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    fig.supylabel(
        "Erreur de référence (plus faible = meilleur)",
        x=0.015,
        fontsize=11,
    )
    fig.supxlabel("Tour", y=0.125, fontsize=11)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.064),
        ncol=4,
        frameon=False,
        fontsize=8.6,
    )
    fig.suptitle(
        "Trajectoire temporelle de l’erreur de référence", fontsize=15, y=0.985
    )
    fig.text(
        0.5,
        0.018,
        "Moyenne ± 1 SD sur cinq seeds; les contextes sont d’abord moyennés dans chaque seed. "
        "Ce screen synthétique est conditionné à une ancre semi-oracle.",
        ha="center",
        fontsize=8.7,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.895,
        bottom=0.20,
        hspace=0.29,
        wspace=0.10,
    )
    path = output_dir / "02_k4b_temporal_reference_error.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    seeds.to_csv(output_dir / "02_k4b_temporal_seed_values.csv", index=False)
    return path


def plot_component_masses(trajectories: pd.DataFrame, output_dir: Path) -> Path:
    candidates_by_metric = {
        "attack_byzantine_direct_current_mass_share": (K2, K4, PRIMARY, DELTA),
        "attack_byzantine_imputed_mass_share": (PRIMARY, DELTA),
        "attack_byzantine_total_slot_mass_share_descriptive": (K2, K4, PRIMARY, DELTA),
    }
    titles = {
        "attack_byzantine_direct_current_mass_share": "Masse directe courante  gᵢcᵢ\n(seule composante utilisée par le gate)",
        "attack_byzantine_imputed_mass_share": "Masse imputée\n(prédicteur passé)",
        "attack_byzantine_total_slot_mass_share_descriptive": "Masse totale du slot\n(descriptive, non additive)",
    }
    # The imputed-share estimand lives near 50–91 %, whereas the direct and
    # total-slot shares remain below about 15 %. Sharing the y-axis therefore
    # clipped the entire middle panel. Each panel now has an honest scale for
    # its own estimand; no data or aggregation is changed.
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.6), sharey=False)
    seed_exports: list[pd.DataFrame] = []
    offsets = np.linspace(-0.24, 0.24, 4)
    for ax, (metric, candidates) in zip(
        axes, candidates_by_metric.items(), strict=True
    ):
        selected = trajectories.loc[
            trajectories["schedule"].eq("persistent")
            & trajectories["threat"].isin(THREATS)
            & trajectories["candidate"].isin(candidates)
        ]
        seeds = _seed_values(selected, value=metric, axes=["threat", "candidate"])
        seeds["metric"] = metric
        seed_exports.append(seeds)
        summary = _mean_sd(seeds, axes=["threat", "candidate"])
        x = np.arange(len(THREATS), dtype=float)
        used_offsets = offsets[: len(candidates)]
        if len(candidates) == 2:
            used_offsets = np.array([-0.10, 0.10])
        for offset, candidate in zip(used_offsets, candidates, strict=True):
            data = summary.loc[summary["candidate"].eq(candidate)].set_index("threat")
            means = np.array([data.loc[threat, "mean"] for threat in THREATS])
            sd = np.array([data.loc[threat, "sd"] for threat in THREATS])
            ax.errorbar(
                x + offset,
                means,
                yerr=sd,
                fmt="o",
                capsize=3,
                markersize=6,
                color=COLORS[candidate],
                label=CANDIDATE_LABELS[candidate],
            )
        ax.set_xticks(x, [THREAT_LABELS[value] for value in THREATS], rotation=22)
        ax.set_ylim(bottom=0.0)
        if metric == "attack_byzantine_imputed_mass_share":
            ax.set_ylim(0.0, 1.0)
        else:
            upper = float((summary["mean"] + summary["sd"]).max())
            ax.set_ylim(0.0, max(0.05, 1.12 * upper))
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0))
        ax.set_title(titles[metric])
        ax.legend(frameon=False, fontsize=8)
    fig.supylabel(
        "Part de la composante attribuée aux cinq slots byzantins latents",
        x=0.012,
        fontsize=11,
    )
    fig.suptitle(
        "Décomposition des masses K4b pendant les attaques persistantes",
        fontsize=15,
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        "Moyenne ± 1 SD inter-seeds après agrégation intra-seed. Les trois parts ne s’additionnent "
        "pas : chaque panneau a son propre dénominateur et sa propre échelle; la masse totale est "
        "la norme de la somme vectorielle.",
        ha="center",
        fontsize=8.7,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.84,
        bottom=0.23,
        wspace=0.25,
    )
    path = output_dir / "03_k4b_direct_imputed_total_masses.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    pd.concat(seed_exports, ignore_index=True).to_csv(
        output_dir / "03_k4b_mass_seed_values.csv", index=False
    )
    return path


def _gate_table_rows(
    decision: dict[str, Any], config: dict[str, Any]
) -> list[tuple[str, str, str, bool]]:
    observed, checks, gates = decision["observed"], decision["checks"], config["gates"]

    def percentage(value: float) -> str:
        return f"{100.0 * float(value):.2f} %"

    return [
        (
            "Oracle B : gain vs K4 (BF/MR persistants)",
            percentage(observed["oracle_bf_mr_persistent_gain_vs_k4"]),
            f"≥ {percentage(gates['oracle_bf_mr_persistent_gain_vs_k4_min'])}",
            checks["oracle_headroom_gain"],
        ),
        (
            "Oracle B − K4 : IC95 haut; cinq seeds",
            f"{observed['oracle_bf_mr_persistent_difference_seed_ci95']['high']:.6f}; n={observed['oracle_bf_mr_persistent_seed_difference_count']}",
            "≤ 0; n=5",
            checks["oracle_headroom_ci"] and checks["oracle_effectiveness_seed_count"],
        ),
        (
            "K4b-B : gain vs K4 (BF/MR persistants)",
            percentage(observed["primary_bf_mr_persistent_gain_vs_k4"]),
            f"≥ {percentage(gates['primary_bf_mr_persistent_gain_vs_k4_min'])}",
            checks["primary_gain_vs_k4"],
        ),
        (
            "K4b-B − K4 : IC95 haut; cinq seeds",
            f"{observed['primary_bf_mr_persistent_difference_seed_ci95']['high']:.6f}; n={observed['primary_bf_mr_persistent_seed_difference_count']}",
            "≤ 0; n=5",
            checks["primary_ci_vs_k4"]
            and checks["primary_vs_k4_effectiveness_seed_count"],
        ),
        (
            "K4b-B : gain vs K2 (attaques séparées)",
            percentage(observed["primary_persistent_separated_gain_vs_k2"]),
            f"≥ {percentage(gates['primary_persistent_separated_gain_vs_k2_min'])}",
            checks["primary_gain_vs_k2"],
        ),
        (
            "K4b-B − K2 : IC95 haut; cinq seeds",
            f"{observed['primary_persistent_separated_difference_seed_ci95']['high']:.6f}; n={observed['primary_persistent_separated_seed_difference_count']}",
            "≤ 0; n=5",
            checks["primary_ci_vs_k2"]
            and checks["primary_vs_k2_effectiveness_seed_count"],
        ),
        (
            "Headroom K4→oracle capturé par K4b-B",
            percentage(observed["primary_oracle_headroom_capture_fraction"]),
            f"≥ {percentage(gates['primary_oracle_headroom_capture_fraction_min'])}",
            checks["headroom_capture"],
        ),
        *[
            (
                label,
                f"{observed[key]['exp_log_ratio_ci95_high']:.4f}",
                "≤ 1.02; n=5",
                checks[check],
            )
            for label, key, check in (
                (
                    "Non-infériorité IPM : exp(IC95 haut log-ratio)",
                    "primary_persistent_ipm_attack_auc_log_ratio",
                    "ipm_noninferiority_vs_k4",
                ),
                (
                    "Non-infériorité ALIE : exp(IC95 haut log-ratio)",
                    "primary_persistent_alie_attack_auc_log_ratio",
                    "alie_noninferiority_vs_k4",
                ),
                (
                    "Non-infériorité intermittente : exp(IC95 haut)",
                    "primary_intermittent_attack_auc_log_ratio",
                    "intermittent_noninferiority_vs_k4",
                ),
                (
                    "Non-infériorité propre : exp(IC95 haut)",
                    "clean_primary_post_enrollment_auc_log_ratio",
                    "clean_noninferiority_vs_k2",
                ),
            )
        ],
        (
            "Réduction de masse Byzantine directe vs K2",
            percentage(observed["persistent_separated_byzantine_mass_reduction_vs_k2"]),
            f"≥ {percentage(gates['persistent_separated_byzantine_mass_reduction_vs_k2_aware_min'])}",
            checks["byzantine_mass_reduction"],
        ),
        (
            "Détection dans le délai",
            percentage(observed["persistent_separated_detection_rate_within_deadline"]),
            f"≥ {percentage(gates['persistent_separated_detection_rate_within_deadline_min'])}",
            checks["detection"],
        ),
        (
            "Récupération dans le délai",
            percentage(observed["persistent_separated_recovery_rate_within_deadline"]),
            f"≥ {percentage(gates['persistent_separated_recovery_rate_within_deadline_min'])}",
            checks["recovery"],
        ),
        (
            "Faux triggers client-tour",
            percentage(observed["false_triggers"]["client_round_rate"]),
            f"≤ {percentage(gates['regular_client_round_false_trigger_rate_max'])}",
            checks["regular_client_round_false_trigger"],
        ),
        (
            "Faux triggers trajectoire",
            percentage(observed["false_triggers"]["trajectory_rate"]),
            f"≤ {percentage(gates['regular_trajectory_false_trigger_rate_max'])}",
            checks["regular_trajectory_false_trigger"],
        ),
        (
            "Baisse du gate des outliers honnêtes",
            f"{observed['honest_outlier_gate_drop_vs_k2_aware']:.4f}",
            f"≤ {gates['honest_outlier_gate_drop_vs_k2_aware_max']:.2f}",
            checks["honest_outlier_gate_drop"],
        ),
        (
            "Violations du cap / replace-one courant",
            f"{observed['contribution_cap_violations']} / {observed['replace_one_violations']}",
            "0 / 0",
            checks["contribution_cap"] and checks["replace_one"],
        ),
        (
            "Complétude / métriques finies / MPS",
            f"{observed['completeness']['complete_fraction']:.3f} / {observed['finite_metric_fraction']:.3f} / {observed['device']}",
            "1 / 1 / mps",
            checks["complete"] and checks["finite"] and checks["production_device"],
        ),
    ]


def plot_gate_table(
    decision: dict[str, Any], config: dict[str, Any], output_dir: Path
) -> Path:
    rows = _gate_table_rows(decision, config)
    cells = [
        [label, value, threshold, "PASS" if passed else "FAIL"]
        for label, value, threshold, passed in rows
    ]
    colors = []
    for index, (_, _, _, passed) in enumerate(rows):
        neutral = "#F7F7F7" if index % 2 == 0 else "#FFFFFF"
        colors.append([neutral, neutral, neutral, "#D5F5E3" if passed else "#FADBD8"])
    fig, ax = plt.subplots(figsize=(15.2, 10.3))
    ax.axis("off")
    table = ax.table(
        cellText=cells,
        cellColours=colors,
        colLabels=["Critère préenregistré", "Observé", "Seuil", "Verdict"],
        colColours=["#DCE6F1"] * 4,
        cellLoc="left",
        colLoc="left",
        colWidths=[0.52, 0.23, 0.14, 0.09],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1.0, 1.65)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.55)
        if row == 0:
            cell.set_text_props(fontweight="bold")
        elif column == 3:
            cell.set_text_props(fontweight="bold", ha="center")
    verdict = decision["decision"]
    ax.set_title(
        "G0g-K4b — tableau des gates de développement",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    fig.text(
        0.04,
        0.03,
        f"Décision recalculée et auditée : {verdict}. Holdout fermé. "
        "Screen de référence synthétique conditionné à une ancre semi-oracle; aucun claim end-to-end.",
        fontsize=9.2,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.96))
    path = output_dir / "04_k4b_preregistered_gate_table.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows, columns=["criterion", "observed", "threshold", "passed"]).to_csv(
        output_dir / "04_k4b_preregistered_gate_table.csv", index=False
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_style()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest, decision, _, rounds, _, trajectories = _validate_sources(
        args.results_dir.resolve()
    )
    if _sha256(args.config.resolve()) != manifest.get("config_sha256"):
        raise RuntimeError("Figure config differs from the completed K4b manifest")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths: Iterable[Path] = (
        plot_attack_effects(trajectories, args.output_dir),
        plot_temporal_error(rounds, args.output_dir),
        plot_component_masses(trajectories, args.output_dir),
        plot_gate_table(decision, config, args.output_dir),
    )
    print(
        "Validated completed K4b development source: "
        f"status={manifest['status']}, device={manifest['device']}, seeds=5, "
        "holdout_opened=false"
    )
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
