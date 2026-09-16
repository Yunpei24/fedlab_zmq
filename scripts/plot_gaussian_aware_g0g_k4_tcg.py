#!/usr/bin/env python3
"""Create seed-level scientific figures for the G0g-K4-TCG development screen.

The script deliberately treats the five development seeds as the independent
replication units.  Contexts, clients, and rounds are averaged *within* seed
before dispersion is computed across seeds; they are never counted as
independent replicates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

DEFAULT_RESULTS = Path(
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k4_tcg_mps_v1"
)
DEFAULT_CONFIG = Path(
    "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k4_tcg.yaml"
)
DEFAULT_OUTPUT = Path("output/figures/gaussian_aware_g0g_k4_tcg")

PRIMARY = "g0g_k4_temporal_causal_gate"
COUNTERFACTUAL = "g0g_k4_no_compromise_counterfactual_control"
K2 = "g0g_k2"
K3 = "g0g_k3_dual_gate"
FCC = "fcc"

SEPARABLE_THREATS = ("ipm", "bitflip_x10", "model_replacement")
THREAT_LABELS = {
    "ipm": "IPM",
    "bitflip_x10": "Bit-Flip ×10",
    "model_replacement": "Model replacement",
}
CANDIDATE_LABELS = {
    PRIMARY: "K4-TCG",
    K2: "K2 covariance-aware",
    K3: "K3 dual-gate",
    FCC: "FCC",
}
CANDIDATE_COLORS = {
    PRIMARY: "#0072B2",
    K2: "#E69F00",
    K3: "#009E73",
    FCC: "#777777",
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
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "lines.linewidth": 2.0,
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_sources(
    results_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_files = {
        "manifest": results_dir / "manifest.json",
        "decision": results_dir / "decision.json",
        "audit": results_dir / "independent_audit.json",
        "rounds": results_dir / "development_round_rows.csv",
        "clients": results_dir / "development_client_rows.csv",
        "trajectories": results_dir / "development_trajectory_rows.csv",
    }
    missing = [str(path) for path in required_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required G0g-K4 files: {missing}")

    manifest = _read_json(required_files["manifest"])
    decision = _read_json(required_files["decision"])
    audit = _read_json(required_files["audit"])
    expected = audit["expected_counts"]

    if manifest.get("status") != "completed_development":
        raise RuntimeError(f"Unexpected campaign status: {manifest.get('status')!r}")
    if manifest.get("device") != "mps":
        raise RuntimeError(
            f"Figures require the MPS campaign, got {manifest.get('device')!r}"
        )
    if manifest.get("holdout_opened") or decision.get("holdout_opened"):
        raise RuntimeError("The development holdout must remain closed")
    if audit.get("audit_verdict") != "PASS" or audit.get("failed_audit_checks"):
        raise RuntimeError("The independent raw-CSV integrity audit did not pass")
    if not decision["checks"].get("complete") or not decision["checks"].get("finite"):
        raise RuntimeError(
            "The development rows are incomplete or contain non-finite metrics"
        )

    rounds = pd.read_csv(required_files["rounds"])
    clients = pd.read_csv(required_files["clients"])
    trajectories = pd.read_csv(required_files["trajectories"])
    observed_counts = {
        "round_rows": len(rounds),
        "client_rows": len(clients),
        "trajectory_rows": len(trajectories),
    }
    for key, observed in observed_counts.items():
        if observed != int(expected[key]):
            raise RuntimeError(f"{key}: expected {expected[key]}, observed {observed}")

    duplicate_specs = {
        "rounds": (rounds, ["trajectory_id", "candidate", "round"]),
        "clients": (clients, ["trajectory_id", "round", "client"]),
        "trajectories": (trajectories, ["trajectory_id", "candidate"]),
    }
    for name, (frame, keys) in duplicate_specs.items():
        duplicate_count = int(frame.duplicated(keys).sum())
        if duplicate_count:
            raise RuntimeError(f"{name}: {duplicate_count} duplicate identifier rows")
    if rounds["all_finite"].astype(bool).mean() != 1.0:
        raise RuntimeError("Non-finite round output found")

    seeds = sorted(rounds["seed"].unique().tolist())
    if len(seeds) != 5:
        raise RuntimeError(f"Expected five development seeds, observed {seeds}")
    return manifest, decision, rounds, clients, trajectories


def _phase_background(ax: plt.Axes, *, compact: bool = False) -> None:
    phases = [
        (0.5, 8.5, "Enrôlement" if not compact else "Enrôl.", "#D9EAF7"),
        (8.5, 12.5, "Monitoring" if not compact else "Monit.", "#ECECEC"),
        (12.5, 24.5, "Attaque", "#FADBD8"),
        (24.5, 36.5, "Récupération" if not compact else "Récup.", "#D5F5E3"),
    ]
    for left, right, label, color in phases:
        ax.axvspan(left, right, color=color, alpha=0.52, zorder=0)
        ax.text(
            (left + right) / 2,
            0.985,
            label,
            ha="center",
            va="top",
            fontsize=8 if compact else 8.5,
            fontweight="bold",
            color="#333333",
            transform=ax.get_xaxis_transform(),
        )


def _seed_summary(
    frame: pd.DataFrame,
    *,
    value: str,
    inner_groups: list[str],
    outer_groups: list[str],
) -> pd.DataFrame:
    """Average contexts within seed, then report mean and SD across seeds."""

    per_seed = frame.groupby(["seed", *inner_groups], observed=True)[value].mean()
    summary = (
        per_seed.groupby(outer_groups, observed=True)
        .agg(mean="mean", sd="std", n_seeds="count")
        .reset_index()
    )
    if not (summary["n_seeds"] == 5).all():
        raise RuntimeError(
            "At least one plotted estimand is missing a development seed"
        )
    summary["sd"] = summary["sd"].fillna(0.0)
    return summary


def plot_temporal_gates(clients: pd.DataFrame, output_dir: Path) -> Path:
    selected = clients[
        (clients["schedule"] == "persistent")
        & clients["threat"].isin(SEPARABLE_THREATS)
    ].copy()
    selected["role"] = np.select(
        [
            selected["latent_byzantine"].astype(bool),
            selected["honest_outlier"].astype(bool),
            selected["honest_regular"].astype(bool),
        ],
        ["Client compromis", "Outlier honnête", "Client honnête régulier"],
        default="Autre",
    )
    if (selected["role"] == "Autre").any():
        raise RuntimeError("Unclassified client role in temporal-gate plot")

    summary = _seed_summary(
        selected,
        value="temporal_gate",
        inner_groups=["round", "role"],
        outer_groups=["round", "role"],
    )
    counter = _seed_summary(
        selected[selected["latent_byzantine"].astype(bool)],
        value="counterfactual_temporal_gate",
        inner_groups=["round"],
        outer_groups=["round"],
    )

    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    _phase_background(ax)
    role_styles = {
        "Client honnête régulier": ("#009E73", "o"),
        "Outlier honnête": ("#CC79A7", "s"),
        "Client compromis": ("#D55E00", "^"),
    }
    for role, (color, marker) in role_styles.items():
        data = summary[summary["role"] == role].sort_values("round")
        x = data["round"].to_numpy(dtype=float)
        y = data["mean"].to_numpy(dtype=float)
        sd = data["sd"].to_numpy(dtype=float)
        ax.plot(x, y, color=color, marker=marker, markevery=3, label=f"K4 — {role}")
        ax.fill_between(
            x, np.clip(y - sd, 0, 1), np.clip(y + sd, 0, 1), color=color, alpha=0.13
        )

    x = counter["round"].to_numpy(dtype=float)
    y = counter["mean"].to_numpy(dtype=float)
    sd = counter["sd"].to_numpy(dtype=float)
    ax.plot(
        x,
        y,
        color="#4D4D4D",
        linestyle=(0, (4, 2)),
        linewidth=1.8,
        label="Contrôle oracle sans compromis — mêmes clients/bruits",
    )
    ax.fill_between(
        x, np.clip(y - sd, 0, 1), np.clip(y + sd, 0, 1), color="#4D4D4D", alpha=0.10
    )

    ax.axvline(13, color="#8B0000", linestyle=":", linewidth=1.3)
    ax.axvline(29, color="#006400", linestyle=":", linewidth=1.3)
    ax.annotate(
        "1er gate causal",
        xy=(13, 1.0),
        xytext=(14.0, 0.86),
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    ax.annotate(
        "fenêtre redevenue propre",
        xy=(29, 1.0),
        xytext=(25.2, 0.70),
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    ax.set_xlim(1, 36)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xticks([1, 4, 8, 9, 12, 13, 15, 17, 20, 24, 25, 29, 30, 36])
    ax.set_xlabel("Tour")
    ax.set_ylabel("Gate historique (1 = aucune atténuation)")
    ax.set_title(
        "K4-TCG : détection temporelle et récupération sous attaques séparables persistantes"
    )
    ax.legend(loc="lower right", frameon=True, framealpha=0.94, ncol=2)
    fig.text(
        0.01,
        0.01,
        "Moyenne ± 1 écart-type sur 5 seeds, après moyenne intra-seed sur les contextes. "
        "Le gate final min(K2 courant, gate historique) n'est pas tracé ici.",
        fontsize=8.5,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    path = output_dir / "01_temporal_gate_by_client_role.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_reference_error(rounds: pd.DataFrame, output_dir: Path) -> Path:
    candidates = [PRIMARY, K2, K3, FCC]
    selected = rounds[
        (rounds["schedule"] == "persistent")
        & rounds["threat"].isin(SEPARABLE_THREATS)
        & rounds["candidate"].isin(candidates)
    ].copy()

    per_seed = (
        selected.groupby(["seed", "threat", "round", "candidate"], observed=True)[
            "reference_error"
        ]
        .mean()
        .rename("value")
        .reset_index()
    )
    summary = (
        per_seed.groupby(["threat", "round", "candidate"], observed=True)["value"]
        .agg(mean="mean", sd="std", n_seeds="count")
        .reset_index()
    )
    if not (summary["n_seeds"] == 5).all():
        raise RuntimeError("Reference-error curve missing at least one seed")

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4), sharex=True, sharey=True)
    for ax, threat in zip(axes, SEPARABLE_THREATS, strict=True):
        _phase_background(ax, compact=True)
        for candidate in candidates:
            data = summary[
                (summary["threat"] == threat) & (summary["candidate"] == candidate)
            ].sort_values("round")
            x = data["round"].to_numpy(dtype=float)
            y = data["mean"].to_numpy(dtype=float)
            sd = data["sd"].to_numpy(dtype=float)
            color = CANDIDATE_COLORS[candidate]
            ax.plot(x, y, color=color, label=CANDIDATE_LABELS[candidate])
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.10)
        ax.axvline(13, color="#8B0000", linestyle=":", linewidth=1.0)
        ax.axvline(25, color="#006400", linestyle=":", linewidth=1.0)
        ax.set_title(THREAT_LABELS[threat])
        ax.set_xlim(1, 36)
        ax.set_xticks([1, 8, 12, 13, 17, 24, 25, 29, 36])
        ax.set_xlabel("Tour")
    axes[0].set_ylabel(r"Erreur de référence $\|\widehat{F}_t-\mu_t^{\mathcal{H}}\|_2$")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.035),
    )
    fig.suptitle(
        "Erreur de référence par tour — attaques séparables persistantes",
        y=1.04,
        fontsize=14,
    )
    fig.text(
        0.01,
        0.01,
        "Moyenne ± 1 écart-type sur 5 seeds; chaque seed moyenne d'abord tous ses contextes "
        "de bruit, géométrie et dynamique honnête.",
        fontsize=8.5,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.98))
    path = output_dir / "02_reference_error_persistent_separable.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_key_effects(
    decision: dict[str, Any], config: dict[str, Any], output_dir: Path
) -> Path:
    observed = decision["observed"]
    gates = config["gates"]
    gain_labels = [
        "Gain AUC\npersistant séparé",
        "Gain AUC\nByzantine-high",
        "Gain vs contrôle\ncontrefactuel",
        "Réduction masse\nbyzantine",
    ]
    gain_values = (
        np.array(
            [
                observed["persistent_separated_attack_auc_gain_vs_k2_aware"],
                observed["byzantine_high_attack_auc_gain_vs_k2_aware"],
                observed["byzantine_high_attack_auc_gain_vs_counterfactual_history"],
                observed["persistent_separated_byzantine_mass_reduction_vs_k2_aware"],
            ]
        )
        * 100
    )
    gain_thresholds = (
        np.array(
            [
                gates["persistent_separated_attack_auc_gain_vs_k2_aware_min"],
                gates["byzantine_high_attack_auc_gain_vs_k2_aware_min"],
                gates["byzantine_high_attack_auc_gain_vs_counterfactual_history_min"],
                gates["persistent_separated_byzantine_mass_reduction_vs_k2_aware_min"],
            ]
        )
        * 100
    )
    gain_pass = gain_values >= gain_thresholds

    rate_labels = [
        "Détection\n≤ tour 17",
        "Récupération\n≤ tour 30",
        "Faux triggers\nclient-tour",
        "Faux triggers\ntrajectoire",
    ]
    rate_values = (
        np.array(
            [
                observed["persistent_separated_detection_rate_within_deadline"],
                observed["persistent_separated_recovery_rate_within_deadline"],
                observed["false_triggers"]["client_round_rate"],
                observed["false_triggers"]["trajectory_rate"],
            ]
        )
        * 100
    )
    rate_thresholds = (
        np.array(
            [
                gates["persistent_separated_detection_rate_within_deadline_min"],
                gates["persistent_separated_recovery_rate_within_deadline_min"],
                gates["regular_client_round_false_trigger_rate_max"],
                gates["regular_trajectory_false_trigger_rate_max"],
            ]
        )
        * 100
    )
    rate_pass = np.array(
        [
            rate_values[0] >= rate_thresholds[0],
            rate_values[1] >= rate_thresholds[1],
            rate_values[2] <= rate_thresholds[2],
            rate_values[3] <= rate_thresholds[3],
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    for ax, labels, values, thresholds, passed, title in [
        (
            axes[0],
            gain_labels,
            gain_values,
            gain_thresholds,
            gain_pass,
            "Effets relatifs (plus élevé = meilleur)",
        ),
        (
            axes[1],
            rate_labels,
            rate_values,
            rate_thresholds,
            rate_pass,
            "Détection, récupération et faux triggers",
        ),
    ]:
        x = np.arange(len(labels))
        colors = ["#009E73" if ok else "#D55E00" for ok in passed]
        bars = ax.bar(x, values, color=colors, alpha=0.88, width=0.64)
        ax.scatter(
            x,
            thresholds,
            marker="_",
            s=420,
            linewidths=3,
            color="#111111",
            zorder=4,
            label="Seuil préenregistré",
        )
        vertical_scale = max(values.max(), thresholds.max())
        for bar, value, threshold, ok in zip(
            bars, values, thresholds, passed, strict=True
        ):
            label_y = bar.get_height() + vertical_scale * 0.025
            if abs(label_y - threshold) < vertical_scale * 0.06:
                label_y = max(bar.get_height(), threshold) + vertical_scale * 0.04
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{value:.2f}%\n{'PASS' if ok else 'FAIL'}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        ax.set_xticks(x, labels)
        ax.set_ylabel("Pourcentage")
        ax.set_title(title)
        ax.set_ylim(0, max(values.max(), thresholds.max()) * 1.22 + 1)
        ax.legend(loc="upper left", frameon=False)

    fig.suptitle(
        "G0g-K4-TCG : résultats scientifiques clés du développement",
        fontsize=14,
        y=1.02,
    )
    fig.text(
        0.01,
        0.005,
        "Vert = critère préenregistré satisfait; orange = échec. Le gain persistant séparé "
        "(3,77 %) est inférieur au seuil de promotion de 5 % : le holdout reste fermé.",
        fontsize=8.7,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    path = output_dir / "03_key_effects_and_operating_rates.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gate_table(
    decision: dict[str, Any], config: dict[str, Any], output_dir: Path
) -> Path:
    observed = decision["observed"]
    gates = config["gates"]
    rows = [
        (
            "Gain AUC, attaques séparables persistantes",
            f"{100 * observed['persistent_separated_attack_auc_gain_vs_k2_aware']:.2f} %",
            f"≥ {100 * gates['persistent_separated_attack_auc_gain_vs_k2_aware_min']:.1f} %",
            decision["checks"]["persistent_separated_gain"],
        ),
        (
            "IC95 seed-level : borne haute de K4 − K2",
            f"{observed['persistent_separated_attack_auc_difference_seed_ci95']['high']:.6f}",
            "≤ 0",
            decision["checks"]["persistent_separated_ci"],
        ),
        (
            "Gain AUC, bruit Byzantine-high",
            f"{100 * observed['byzantine_high_attack_auc_gain_vs_k2_aware']:.2f} %",
            f"≥ {100 * gates['byzantine_high_attack_auc_gain_vs_k2_aware_min']:.1f} %",
            decision["checks"]["byzantine_high_gain"],
        ),
        (
            "Réduction masse de contribution byzantine",
            f"{100 * observed['persistent_separated_byzantine_mass_reduction_vs_k2_aware']:.2f} %",
            f"≥ {100 * gates['persistent_separated_byzantine_mass_reduction_vs_k2_aware_min']:.1f} %",
            decision["checks"]["byzantine_mass_reduction"],
        ),
        (
            "Détection dans le délai (≤ tour 17)",
            f"{100 * observed['persistent_separated_detection_rate_within_deadline']:.2f} %",
            f"≥ {100 * gates['persistent_separated_detection_rate_within_deadline_min']:.1f} %",
            decision["checks"]["detection"],
        ),
        (
            "Récupération dans le délai (≤ tour 30)",
            f"{100 * observed['persistent_separated_recovery_rate_within_deadline']:.2f} %",
            f"≥ {100 * gates['persistent_separated_recovery_rate_within_deadline_min']:.1f} %",
            decision["checks"]["recovery"],
        ),
        (
            "Faux triggers, clients-tours honnêtes réguliers",
            f"{100 * observed['false_triggers']['client_round_rate']:.4f} %",
            f"≤ {100 * gates['regular_client_round_false_trigger_rate_max']:.1f} %",
            decision["checks"]["regular_client_round_false_trigger"],
        ),
        (
            "Faux triggers, trajectoires honnêtes régulières",
            f"{100 * observed['false_triggers']['trajectory_rate']:.2f} %",
            f"≤ {100 * gates['regular_trajectory_false_trigger_rate_max']:.1f} %",
            decision["checks"]["regular_trajectory_false_trigger"],
        ),
        (
            "Ratio erreur propre post-enrôlement K4/K2",
            f"{observed['no_compromise_post_enrollment_error_ratio_to_k2_aware']:.6f}",
            f"≤ {gates['no_compromise_post_enrollment_error_ratio_to_k2_aware_max']:.2f}",
            decision["checks"]["no_compromise_post_enrollment_noninferiority"],
        ),
        (
            "Maximum replace-one observé / borne 2G/n",
            f"{100 * observed['replace_one_max_ratio_to_bound']:.2f} %",
            "≤ 100 %",
            decision["checks"]["replace_one"],
        ),
    ]

    cell_text = [
        [label, value, threshold, "PASS" if passed else "FAIL"]
        for label, value, threshold, passed in rows
    ]
    cell_colors: list[list[str]] = []
    for index, (_, _, _, passed) in enumerate(rows):
        neutral = "#F7F7F7" if index % 2 == 0 else "#FFFFFF"
        verdict = "#D5F5E3" if passed else "#FADBD8"
        cell_colors.append([neutral, neutral, neutral, verdict])

    fig, ax = plt.subplots(figsize=(13.2, 6.3))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        colLabels=["Critère préenregistré", "Observé", "Seuil", "Verdict"],
        colColours=["#DCE6F1"] * 4,
        cellLoc="left",
        colLoc="left",
        colWidths=[0.53, 0.18, 0.14, 0.11],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.72)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(fontweight="bold")
        elif col == 3:
            cell.set_text_props(fontweight="bold", ha="center")

    ax.set_title(
        "G0g-K4-TCG — tableau de décision (développement uniquement)",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )
    fig.text(
        0.05,
        0.035,
        "Décision finale : STOP après développement. Un seul critère scientifique échoue; "
        "l'intégrité, les certificats et les autres critères passent. Le holdout n'a pas été ouvert.",
        fontsize=9.5,
        color="#333333",
    )
    fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.96))
    path = output_dir / "04_preregistered_gate_summary_table.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
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
    manifest, decision, rounds, clients, _ = _validate_sources(args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_temporal_gates(clients, args.output_dir),
        plot_reference_error(rounds, args.output_dir),
        plot_key_effects(decision, config, args.output_dir),
        plot_gate_table(decision, config, args.output_dir),
    ]
    print(
        "Validated complete G0g-K4-TCG development source: "
        f"status={manifest['status']}, device={manifest['device']}, "
        "seeds=5, holdout_opened=false"
    )
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
