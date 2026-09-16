#!/usr/bin/env python3
"""Generate the post-hoc report for the frozen Gaussian-aware K6 M0 screen.

This script never recomputes a scientific result.  It first verifies every
artifact and source hash recorded in the frozen scientific manifest, and only
then reads the CSV/JSON outputs used to build tables and figures.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
RESULTS = (
    ROOT
    / "results"
    / "ldp_gradient_far"
    / "gaussian_aware_reference_g0g_k6_tp_eiv_microbench_mps_v1"
)
FIGURES = ROOT / "output" / "figures" / "gaussian_aware_g0g_k6_m0"
REPORT = ROOT / "output" / "analysis" / "Gaussian_Aware_G0g_K6_M0_Report.md"
BRIEF = ROOT / "output" / "analysis" / "Gaussian_Aware_G0g_K6_M0_Monday_Brief.md"
REPORT_MANIFEST = FIGURES / "report_manifest.json"

# Keep the plotting cache writable and outside the scientific result tree.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fedlab_gaussian_aware_k6_m0_mpl")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CANDIDATE_LABELS = {
    "radial_mse_mean": "Radial-Y",
    "k6_mse_mean": "K6 corrigé EIV",
    "k6_uncorrected_mse_mean": "K6 non corrigé",
}
REGIME_LABELS = {
    "homogeneous": "Bruit homogène",
    "heteroscedastic": "Bruit hétéroscédastique",
}
THREAT_LABELS = {
    "none": "Sans attaque",
    "bitflip_x10": "Bit-Flip ×10",
    "model_replacement": "Model replacement",
}
REGIME_ORDER = ("homogeneous", "heteroscedastic")
THREAT_ORDER = ("none", "bitflip_x10", "model_replacement")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_frozen_manifest() -> tuple[dict[str, Any], dict[str, str]]:
    """Verify the frozen manifest before any scientific artifact is parsed."""

    manifest_path = RESULTS / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing frozen manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []

    for relative, expected in manifest.get("artifact_sha256", {}).items():
        path = RESULTS / relative
        if not path.is_file():
            mismatches.append(f"missing artifact: {relative}")
            continue
        observed = _sha256(path)
        if observed != expected:
            mismatches.append(
                f"artifact {relative}: expected {expected}, observed {observed}"
            )

    for relative, expected in manifest.get("source_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file():
            mismatches.append(f"missing source: {relative}")
            continue
        observed = _sha256(path)
        if observed != expected:
            mismatches.append(
                f"source {relative}: expected {expected}, observed {observed}"
            )

    if mismatches:
        joined = "\n - ".join(mismatches)
        raise RuntimeError(f"Frozen-manifest verification failed:\n - {joined}")
    return manifest, {
        "manifest.json": _sha256(manifest_path),
        **{
            relative: str(expected)
            for relative, expected in manifest["artifact_sha256"].items()
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pct(value: float, digits: int = 2, signed: bool = False) -> str:
    sign = "+" if signed else ""
    return f"{value * 100:{sign}.{digits}f} %".replace(".", ",")


def _decimal(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _sci(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}e}".replace(".", ",")


def _ci_text(interval: Mapping[str, Any], invert: bool = False) -> str:
    mean = float(interval["mean"])
    low = float(interval["low"])
    high = float(interval["high"])
    if invert:
        mean, low, high = -mean, -high, -low
    return f"{_pct(mean, signed=True)} [{_pct(low)} ; {_pct(high)}]"


def _student_ci_seed(values: Sequence[float]) -> dict[str, float]:
    """Return the two-sided 95% Student interval over 12 outer seeds."""

    numbers = [float(value) for value in values]
    if len(numbers) != 12:
        raise RuntimeError("Expected exactly 12 independent outer-seed values")
    mean = statistics.fmean(numbers)
    half_width = 2.200985160091638 * statistics.stdev(numbers) / math.sqrt(12.0)
    return {"mean": mean, "low": mean - half_width, "high": mean + half_width}


def _summary_lookup(
    summary_rows: Sequence[Mapping[str, str]], regime: str, threat: str
) -> Mapping[str, str]:
    matches = [
        row
        for row in summary_rows
        if row["noise_regime"] == regime and row["threat"] == threat
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one summary row for {regime}/{threat}")
    return matches[0]


def _write_mse_figure(summary_rows: Sequence[Mapping[str, str]]) -> Path:
    path = FIGURES / "mse_by_regime_and_threat.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    colors = ("#8b95a5", "#136f8a", "#e18727")
    metrics = tuple(CANDIDATE_LABELS)
    x = np.arange(len(THREAT_ORDER), dtype=float)
    width = 0.23
    for axis, regime in zip(axes, REGIME_ORDER, strict=True):
        for index, (metric, color) in enumerate(zip(metrics, colors, strict=True)):
            values = [
                float(_summary_lookup(summary_rows, regime, threat)[metric]) * 1e4
                for threat in THREAT_ORDER
            ]
            axis.bar(
                x + (index - 1) * width,
                values,
                width,
                label=CANDIDATE_LABELS[metric],
                color=color,
            )
        axis.set_title(REGIME_LABELS[regime], fontweight="bold")
        axis.set_xticks(x, [THREAT_LABELS[item] for item in THREAT_ORDER])
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(r"MSE moyenne ($\times 10^{-4}$) — plus bas = meilleur")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "M0 — erreur du prédicteur selon le bruit et la menace",
        y=0.98,
        fontsize=14,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_contrast_figure(intervals: Mapping[str, Any]) -> Path:
    path = FIGURES / "seed_level_contrasts_95ci.png"
    pooled = intervals["pooled_equal_weight_noise_regimes_within_seed"]
    specifications = (
        (
            "K6 corrigé vs Radial-Y\n(cellules attaquées)",
            pooled["k6_corrected_gain_vs_radial_attacked"],
            False,
            "#136f8a",
        ),
        (
            "K6 corrigé vs non corrigé\n(toutes cellules)",
            pooled["k6_corrected_gain_vs_uncorrected_all_cells"],
            False,
            "#c4493d",
        ),
        (
            "K6 corrigé vs Radial-Y\n(sans attaque)",
            pooled["k6_corrected_benign_loss_vs_radial"],
            True,
            "#3a875f",
        ),
    )
    fig, axis = plt.subplots(figsize=(10.2, 4.7))
    for y, (label, interval, invert, color) in enumerate(specifications):
        mean = float(interval["mean"])
        low = float(interval["low"])
        high = float(interval["high"])
        if invert:
            mean, low, high = -mean, -high, -low
        mean, low, high = (100 * mean, 100 * low, 100 * high)
        axis.errorbar(
            mean,
            y,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            markersize=8,
            capsize=5,
            linewidth=2,
            color=color,
        )
        axis.text(high + 0.7, y, f"{mean:+.2f} %", va="center", color=color)
    axis.axvline(0, color="#27364a", linewidth=1.2)
    axis.set_yticks(range(len(specifications)), [item[0] for item in specifications])
    axis.invert_yaxis()
    axis.set_xlabel("Gain relatif de MSE (%, plus haut = meilleur pour K6 corrigé)")
    axis.set_title("Contrastes appariés par seed — moyenne et IC de Student à 95 %")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_paired_figure(seed_rows: Sequence[Mapping[str, str]]) -> Path:
    path = FIGURES / "corrected_vs_uncorrected_attacked_seed_pairs.png"
    fig, axis = plt.subplots(figsize=(6.7, 6.1))
    colors = {"homogeneous": "#136f8a", "heteroscedastic": "#e18727"}
    all_values: list[float] = []
    for regime in REGIME_ORDER:
        rows = [row for row in seed_rows if row["noise_regime"] == regime]
        x = np.asarray(
            [float(row["k6_uncorrected_attacked_mse_mean"]) * 1e4 for row in rows]
        )
        y = np.asarray(
            [float(row["k6_corrected_attacked_mse_mean"]) * 1e4 for row in rows]
        )
        all_values.extend(x.tolist())
        all_values.extend(y.tolist())
        axis.scatter(
            x,
            y,
            s=54,
            alpha=0.82,
            color=colors[regime],
            label=REGIME_LABELS[regime],
        )
    lower, upper = min(all_values) * 0.94, max(all_values) * 1.05
    axis.plot([lower, upper], [lower, upper], "--", color="#27364a", label="Égalité")
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"MSE K6 non corrigé ($\times 10^{-4}$)")
    axis.set_ylabel(r"MSE K6 corrigé EIV ($\times 10^{-4}$)")
    axis.set_title("Cellules attaquées : comparaison appariée par seed")
    axis.text(
        0.04,
        0.94,
        "Au-dessus de la diagonale :\nla correction EIV est moins bonne",
        transform=axis.transAxes,
        va="top",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "#ccd3dc", "alpha": 0.9},
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_confidence_figure(summary_rows: Sequence[Mapping[str, str]]) -> Path:
    path = FIGURES / "temporal_confidence_by_cell.png"
    labels = [
        f"{REGIME_LABELS[regime]}\n{THREAT_LABELS[threat]}"
        for regime in REGIME_ORDER
        for threat in THREAT_ORDER
    ]
    corrected = [
        float(_summary_lookup(summary_rows, regime, threat)["confidence_mean"])
        for regime in REGIME_ORDER
        for threat in THREAT_ORDER
    ]
    uncorrected = [
        float(
            _summary_lookup(summary_rows, regime, threat)["uncorrected_confidence_mean"]
        )
        for regime in REGIME_ORDER
        for threat in THREAT_ORDER
    ]
    x = np.arange(len(labels), dtype=float)
    fig, axis = plt.subplots(figsize=(11.2, 4.8))
    axis.bar(x - 0.18, corrected, 0.36, color="#136f8a", label="K6 corrigé EIV")
    axis.bar(x + 0.18, uncorrected, 0.36, color="#e18727", label="K6 non corrigé")
    axis.set_ylim(0.82, 1.0)
    axis.set_ylabel("Confiance temporelle moyenne")
    axis.set_xticks(x, labels, rotation=14, ha="right")
    axis.set_title(
        "La correction EIV augmente la confiance, mais pas la précision finale"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _magnitude_ablation_rows(
    evaluation_rows: Sequence[Mapping[str, str]], protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconstruct the pre-announced, post-opening magnitude ablation exactly."""

    cap = float(protocol["geometry"]["residual_influence_cap"])
    dimension = int(protocol["cohort"]["dimension"])
    reconstructed: list[dict[str, Any]] = []
    radial_identity_errors: list[float] = []
    radial_norm_errors: list[float] = []
    for row in evaluation_rows:
        radial_mse = float(row["radial_mse"])
        target_norm = float(row["target_norm"])
        confidence = float(row["confidence"])
        corrected_energy_y = float(row["corrected_energy_y"])
        raw_energy_y = float(row["raw_energy_y"])
        radial_norm = float(row["radial_predictor_norm"])

        # If u is the unit radial direction and z=<u,target>, then
        # d*MSE(G*u,target)=G^2-2Gz+||target||^2.  This recovers z from
        # quantities already frozen in evaluation_rows.csv.
        radial_dot_target = (
            cap * cap + target_norm * target_norm - dimension * radial_mse
        ) / (2.0 * cap)
        amplitude = confidence * min(cap, math.sqrt(max(corrected_energy_y, 0.0)))
        magnitude_mse = (
            amplitude * amplitude
            - 2.0 * amplitude * radial_dot_target
            + target_norm * target_norm
        ) / dimension
        if magnitude_mse < -1.0e-12:
            raise RuntimeError("Reconstructed magnitude-ablation MSE is negative")
        magnitude_mse = max(magnitude_mse, 0.0)
        identity_amplitude = min(cap, math.sqrt(max(raw_energy_y, 0.0)))
        identity_mse = (
            identity_amplitude * identity_amplitude
            - 2.0 * identity_amplitude * radial_dot_target
            + target_norm * target_norm
        ) / dimension
        if identity_mse < -1.0e-12:
            raise RuntimeError("Reconstructed identity-Y MSE is negative")
        identity_mse = max(identity_mse, 0.0)
        radial_identity = (
            cap * cap - 2.0 * cap * radial_dot_target + target_norm * target_norm
        ) / dimension
        radial_identity_errors.append(abs(radial_identity - radial_mse))
        radial_norm_errors.append(abs(radial_norm - cap))
        reconstructed.append(
            {
                "seed": int(row["seed"]),
                "noise_regime": row["noise_regime"],
                "threat": row["threat"],
                "radial_mse": radial_mse,
                "magnitude_ablation_mse": magnitude_mse,
                "identity_y_mse": identity_mse,
                "relative_gain_vs_radial": (
                    (radial_mse - magnitude_mse) / max(radial_mse, 1.0e-15)
                ),
                "reconstructed_amplitude": amplitude,
                "identity_y_amplitude": identity_amplitude,
                "relative_gain_vs_identity_y": (
                    (identity_mse - magnitude_mse) / max(identity_mse, 1.0e-15)
                ),
                "reconstructed_radial_dot_target": radial_dot_target,
            }
        )
    verification = {
        "formula_preannounced_in_protocol_section_6": True,
        "contrast_and_gates_preregistered": False,
        "computed_after_evaluation_opened": True,
        "maximum_radial_identity_absolute_error": max(radial_identity_errors),
        "maximum_radial_norm_vs_cap_absolute_error": max(radial_norm_errors),
        "all_reconstructed_amplitudes_within_cap": all(
            0.0 <= row["reconstructed_amplitude"] <= cap + 1.0e-12
            for row in reconstructed
        ),
        "all_identity_y_amplitudes_within_cap": all(
            0.0 <= row["identity_y_amplitude"] <= cap + 1.0e-12 for row in reconstructed
        ),
        "row_count": len(reconstructed),
    }
    if verification["maximum_radial_identity_absolute_error"] > 1.0e-12:
        raise RuntimeError("Magnitude-ablation algebraic identity check failed")
    if verification["maximum_radial_norm_vs_cap_absolute_error"] > 1.0e-6:
        raise RuntimeError("Radial baseline did not saturate the registered cap")
    if not verification["all_reconstructed_amplitudes_within_cap"]:
        raise RuntimeError("Magnitude-ablation amplitude exceeded its public cap")
    if not verification["all_identity_y_amplitudes_within_cap"]:
        raise RuntimeError("Identity-Y amplitude exceeded its public cap")
    return reconstructed, verification


def _magnitude_ablation_table(rows: Sequence[Mapping[str, Any]]) -> str:
    result: list[str] = []
    for regime in REGIME_ORDER:
        for threat in THREAT_ORDER:
            cell = [
                row
                for row in rows
                if row["noise_regime"] == regime and row["threat"] == threat
            ]
            result.append(
                "| "
                + " | ".join(
                    (
                        REGIME_LABELS[regime],
                        THREAT_LABELS[threat],
                        _sci(
                            statistics.fmean(float(row["radial_mse"]) for row in cell)
                        ),
                        _sci(
                            statistics.fmean(
                                float(row["magnitude_ablation_mse"]) for row in cell
                            )
                        ),
                        _sci(
                            statistics.fmean(
                                float(row["identity_y_mse"]) for row in cell
                            )
                        ),
                        _pct(
                            statistics.fmean(
                                float(row["relative_gain_vs_radial"]) for row in cell
                            ),
                            signed=True,
                        ),
                        _pct(
                            statistics.fmean(
                                float(row["relative_gain_vs_identity_y"])
                                for row in cell
                            ),
                            signed=True,
                        ),
                    )
                )
                + " |"
            )
    return "\n".join(result)


def _identity_y_contrast_intervals(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compare reconstructed K6b with Identity-Y at the outer-seed level."""

    result: dict[str, dict[str, dict[str, float]]] = {}
    for regime in REGIME_ORDER:
        by_subset: dict[str, dict[str, float]] = {}
        for subset in ("all", "attacked", "benign"):
            seed_gains: list[float] = []
            for seed in sorted({int(row["seed"]) for row in rows}):
                cell = [
                    row
                    for row in rows
                    if int(row["seed"]) == seed
                    and row["noise_regime"] == regime
                    and (
                        subset == "all"
                        or (subset == "attacked" and row["threat"] != "none")
                        or (subset == "benign" and row["threat"] == "none")
                    )
                ]
                expected = 3 if subset == "all" else (2 if subset == "attacked" else 1)
                if len(cell) != expected:
                    raise RuntimeError("Incomplete seed cell for Identity-Y contrast")
                identity_total = sum(float(row["identity_y_mse"]) for row in cell)
                candidate_total = sum(
                    float(row["magnitude_ablation_mse"]) for row in cell
                )
                seed_gains.append(
                    (identity_total - candidate_total) / max(identity_total, 1.0e-15)
                )
            by_subset[subset] = _student_ci_seed(seed_gains)
        result[regime] = by_subset
    return result


def _write_magnitude_ablation_figure(
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    path = FIGURES / "exploratory_magnitude_ablation.png"
    cell_order = [
        (regime, threat) for regime in REGIME_ORDER for threat in THREAT_ORDER
    ]
    labels = [
        f"{REGIME_LABELS[regime]}\n{THREAT_LABELS[threat]}"
        for regime, threat in cell_order
    ]
    fig, axis = plt.subplots(figsize=(11.3, 5.2))
    rng = np.random.default_rng(20260911)
    for index, (regime, threat) in enumerate(cell_order):
        values = np.asarray(
            [
                100.0 * float(row["relative_gain_vs_radial"])
                for row in rows
                if row["noise_regime"] == regime and row["threat"] == threat
            ]
        )
        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        color = "#136f8a" if regime == "homogeneous" else "#e18727"
        axis.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=28,
            alpha=0.58,
            color=color,
        )
        axis.scatter(
            [index],
            [float(np.mean(values))],
            marker="D",
            s=72,
            color="#172b3a",
            zorder=4,
        )
        axis.text(
            index,
            float(np.mean(values)) + 4.0,
            f"{np.mean(values):.1f} %",
            ha="center",
            fontsize=9,
        )
    axis.axhline(0, color="#c4493d", linewidth=1.3)
    axis.set_xticks(range(len(labels)), labels, rotation=15, ha="right")
    axis.set_ylabel("Gain de MSE vs Radial-Y (%)")
    axis.set_title(
        "Ablation de magnitude K6b — descriptive, reconstruite après ouverture"
    )
    axis.grid(axis="y", alpha=0.24)
    axis.text(
        0.01,
        0.02,
        "Points : seeds individuelles ; losange noir : moyenne.\n"
        "Formule préannoncée, mais contraste et gates non préenregistrés.",
        transform=axis.transAxes,
        fontsize=9,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#ccd3dc", "alpha": 0.92},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_identity_y_figure(
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Show the decisive K6b-versus-recent-view control at seed level."""

    path = FIGURES / "k6b_vs_identity_y_seed_contrasts.png"
    fig, axis = plt.subplots(figsize=(8.8, 4.8))
    rng = np.random.default_rng(20260912)
    colors = {"homogeneous": "#136f8a", "heteroscedastic": "#e18727"}
    for index, regime in enumerate(REGIME_ORDER):
        seed_gains: list[float] = []
        for seed in sorted({int(row["seed"]) for row in rows}):
            cell = [
                row
                for row in rows
                if int(row["seed"]) == seed
                and row["noise_regime"] == regime
                and row["threat"] != "none"
            ]
            if len(cell) != 2:
                raise RuntimeError("Incomplete attacked cell for Identity-Y figure")
            identity_total = sum(float(row["identity_y_mse"]) for row in cell)
            candidate_total = sum(float(row["magnitude_ablation_mse"]) for row in cell)
            seed_gains.append(
                100.0
                * (identity_total - candidate_total)
                / max(identity_total, 1.0e-15)
            )
        jitter = rng.uniform(-0.07, 0.07, size=len(seed_gains))
        axis.scatter(
            np.full(len(seed_gains), index) + jitter,
            seed_gains,
            s=34,
            alpha=0.66,
            color=colors[regime],
        )
        interval = _student_ci_seed([value / 100.0 for value in seed_gains])
        mean = 100.0 * float(interval["mean"])
        low = 100.0 * float(interval["low"])
        high = 100.0 * float(interval["high"])
        axis.errorbar(
            [index],
            [mean],
            yerr=[[mean - low], [high - mean]],
            fmt="D",
            markersize=7,
            capsize=5,
            color="#172b3a",
            zorder=5,
        )
        axis.text(index, high + 0.8, f"{mean:.1f} %", ha="center", fontsize=10)
    axis.axhline(0.0, color="#c4493d", linewidth=1.4)
    axis.set_xticks(
        range(len(REGIME_ORDER)),
        [REGIME_LABELS[regime] for regime in REGIME_ORDER],
    )
    axis.set_ylabel("Gain de MSE de K6b vs Identity-Y (%)")
    axis.set_title("Contrôle décisif : K6b perd face à la vue récente non amplifiée")
    axis.grid(axis="y", alpha=0.24)
    axis.text(
        0.01,
        0.02,
        "Valeur positive : K6b meilleur. Valeur négative : Identity-Y meilleur.\n"
        "Points : seeds ; losange et barre : moyenne et IC de Student à 95 %.",
        transform=axis.transAxes,
        fontsize=9,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#ccd3dc", "alpha": 0.92},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _regime_ci_rows(intervals: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for regime in (*REGIME_ORDER, "pooled"):
        source = (
            intervals["pooled_equal_weight_noise_regimes_within_seed"]
            if regime == "pooled"
            else intervals["by_noise_regime"][regime]
        )
        label = (
            "Pooled (régimes équipondérés par seed)"
            if regime == "pooled"
            else REGIME_LABELS[regime]
        )
        rows.append(
            "| "
            + " | ".join(
                (
                    label,
                    _ci_text(source["k6_corrected_gain_vs_radial_attacked"]),
                    _ci_text(source["k6_corrected_gain_vs_uncorrected_all_cells"]),
                    _ci_text(source["k6_corrected_gain_vs_uncorrected_attacked"]),
                    _ci_text(source["k6_corrected_benign_loss_vs_radial"], invert=True),
                )
            )
            + " |"
        )
    return "\n".join(rows)


def _cell_table(summary_rows: Sequence[Mapping[str, str]]) -> str:
    rows: list[str] = []
    for regime in REGIME_ORDER:
        for threat in THREAT_ORDER:
            row = _summary_lookup(summary_rows, regime, threat)
            rows.append(
                "| "
                + " | ".join(
                    (
                        REGIME_LABELS[regime],
                        THREAT_LABELS[threat],
                        _sci(float(row["radial_mse_mean"])),
                        _sci(float(row["k6_mse_mean"])),
                        _sci(float(row["k6_uncorrected_mse_mean"])),
                        _pct(float(row["k6_relative_mse_gain_mean"]), signed=True),
                        _pct(
                            float(
                                row[
                                    "k6_corrected_relative_mse_gain_vs_uncorrected_mean"
                                ]
                            ),
                            signed=True,
                        ),
                    )
                )
                + " |"
            )
    return "\n".join(rows)


def _all_true(mapping: Mapping[str, Any]) -> bool:
    return all(bool(value) for value in mapping.values())


def _write_reports(
    manifest: Mapping[str, Any],
    verified_hashes: Mapping[str, str],
    decision: Mapping[str, Any],
    intervals: Mapping[str, Any],
    calibration: Mapping[str, Any],
    protocol: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, str]],
    seed_rows: Sequence[Mapping[str, str]],
    magnitude_rows: Sequence[Mapping[str, Any]],
    magnitude_verification: Mapping[str, Any],
    figures: Sequence[Path],
) -> None:
    relative_figures = {
        path.name: f"../figures/gaussian_aware_g0g_k6_m0/{path.name}"
        for path in figures
    }
    attacked_ci = intervals["pooled_equal_weight_noise_regimes_within_seed"][
        "k6_corrected_gain_vs_radial_attacked"
    ]
    corrected_vs_raw_ci = intervals["pooled_equal_weight_noise_regimes_within_seed"][
        "k6_corrected_gain_vs_uncorrected_all_cells"
    ]
    validity_ok = _all_true(decision["validity_checks"])
    science_ok = _all_true(decision["scientific_checks"])
    cp95_zero_over_24 = 1.0 - 0.05 ** (1.0 / 24.0)
    cp95_zero_over_12 = 1.0 - 0.05 ** (1.0 / 12.0)
    identity_intervals = _identity_y_contrast_intervals(magnitude_rows)
    identity_hom_attacked = identity_intervals["homogeneous"]["attacked"]
    identity_hetero_attacked = identity_intervals["heteroscedastic"]["attacked"]

    report = rf"""# Écran mécanistique M0 — K6 temporel Gaussian-aware

## Verdict

L'écran M0 est **valide techniquement**, mais il **réfute le candidat K6/K6b actuel** :

- face aux deux attaques, **K6 corrigé EIV** réduit la MSE de {_ci_text(attacked_ci)} relativement au prédicteur **Radial-Y** ;
- pourtant, **K6 corrigé EIV** présente une MSE environ {_pct(-float(corrected_vs_raw_ci['mean']))} plus élevée que le même mécanisme **K6 non corrigé** ; le gain signé corrigé vs non corrigé vaut {_ci_text(corrected_vs_raw_ci)} ;
- surtout, la variante K6b reconstruite perd face au contrôle naturel **Identity-Y/K4b** sous attaque : {_ci_text(identity_hom_attacked)} en bruit homogène et {_ci_text(identity_hetero_attacked)} en bruit hétéroscédastique.

La conclusion défendable est donc précise : **le gain face à Radial-Y provenait principalement d'une mauvaise calibration de l'amplitude de ce contrôle, et non d'un bénéfice démontré de la confiance temporelle**. Le statut interne `advance_to_locked_k6_development` signifie uniquement que le runner omettait les contrôles scientifiques décisifs. Le P0 original et la promotion de K6b sont arrêtés. M0 ne valide ni K6 en entraînement fédéré end-to-end, ni sa fairness, ni sa robustesse Byzantine universelle.

![MSE par cellule]({relative_figures['mse_by_regime_and_threat.png']})

## 1. Portée et protocole

| Élément | Valeur figée |
|---|---|
| Portée | M0 synthétique, développement mécanistique uniquement |
| Cohorte | {protocol['cohort']['num_clients']} clients dont {protocol['cohort']['num_byzantine']} Byzantins, dimension {protocol['cohort']['dimension']} |
| Régimes de bruit | homogène (`σ=0,03`) ; hétéroscédastique (`σ∈{{0,012; 0,022; 0,040; 0,070}}`) |
| Menaces | aucune ; Bit-Flip ×10 ; model replacement |
| Historique | fenêtres disjointes {protocol['timeline']['older_view_rounds']} et {protocol['timeline']['newer_view_rounds']} ; déploiement au tour {protocol['timeline']['deployment_round']} |
| Gate | commun aux deux vues, calculé strictement sur le passé jusqu'au tour {protocol['timeline']['gate_source_rounds'][-1]} |
| Seeds indépendantes | 12 calibration + 12 validation nulle + 12 évaluation |
| Unité des IC | seed externe ; les cellules enfant ne sont pas traitées comme réplicats indépendants |
| Exécution | MPS, float32, fallback CPU désactivé |

Les trois candidats sont appariés sur exactement les mêmes trajectoires aléatoires :

1. **Radial-Y** : amplitude maximale `G` orientée par la vue récente, sans confiance temporelle.
2. **K6 non corrigé** : même direction radiale, modulée par une confiance issue des moments temporels bruts.
3. **K6 corrigé EIV** : même construction, mais les moments sont corrigés par une approximation delta de la covariance DP après clipping.
4. **Identity-Y/K4b**, reconstruit après ouverture : la vue récente \(Y\) sans amplification d'amplitude. Il constitue le contrôle naturel que le protocole M0 aurait dû inclure.

Les seuils nuls des deux versions K6 ont été calibrés séparément au quantile 97,5 %, sans utiliser les seeds d'évaluation. Ils valent nécessairement tous deux `{calibration['tau_a_corrected']:.9f}` dans M0 : les fenêtres sont disjointes, \(C_{{XY}}=0\), donc l'alignement corrigé est exactement le produit scalaire brut ligne par ligne. Toute différence corrigé/non corrigé provient ici des dénominateurs d'énergie, jamais du numérateur ni de \(\tau_a\).

## 2. Observations quantitatives

### 2.1 MSE par régime et menace

La MSE est mesurée contre la moyenne honnête cible au tour de déploiement. Une valeur plus faible est meilleure.

| Bruit | Menace | Radial-Y | K6 corrigé | K6 non corrigé | Corrigé vs radial | Corrigé vs non corrigé |
|---|---|---:|---:|---:|---:|---:|
{_cell_table(summary_rows)}

Deux faits coexistent dans les six cellules : K6 corrigé est meilleur que Radial-Y, mais K6 non corrigé est encore meilleur que K6 corrigé.

### 2.2 Contrastes et IC seed-level

| Régime | Corrigé vs radial, attaques | Corrigé vs non corrigé, toutes cellules | Corrigé vs non corrigé, attaques | Corrigé vs radial, sans attaque |
|---|---:|---:|---:|---:|
{_regime_ci_rows(intervals)}

Les IC sont des IC bilatéraux de Student à 95 % calculés sur 12 seeds externes. Ils n'utilisent pas les 72 cellules comme si elles étaient indépendantes.

![Contrastes seed-level]({relative_figures['seed_level_contrasts_95ci.png']})

Le nuage apparié rend le résultat négatif particulièrement lisible : chaque point compare, pour la même seed et le même régime de bruit, la MSE attaquée des versions corrigée et non corrigée.

![Comparaison corrigé/non corrigé]({relative_figures['corrected_vs_uncorrected_attacked_seed_pairs.png']})

### 2.3 Confiance temporelle et contrôle nul

La confiance moyenne corrigée est systématiquement plus élevée que la confiance non corrigée. Cette hausse ne se traduit toutefois pas par une MSE plus faible face au contrôle non corrigé.

![Confiance temporelle]({relative_figures['temporal_confidence_by_cell.png']})

| Contrôle | Corrigé | Non corrigé |
|---|---:|---:|
| Activation nulle, calibration interne (descriptif) | {_pct(float(decision['calibration_in_sample_false_activation_rate_descriptive']))} | {_pct(float(decision['calibration_in_sample_uncorrected_false_activation_rate_descriptive']))} |
| Activation nulle, seeds indépendantes (critère scientifique) | {_pct(float(decision['independent_corrected_null_false_activation_rate']))} | {_pct(float(decision['independent_uncorrected_null_false_activation_rate']))} |

Le zéro observé sur 24 contextes nuls indépendants par candidat est compatible avec le seuil choisi, mais ne certifie pas un taux inférieur à 10 %. Sa borne supérieure exacte unilatérale de Clopper--Pearson à 95 % vaut {_pct(cp95_zero_over_24)}. Par régime, `0/12` donne {_pct(cp95_zero_over_12)}. Le prochain protocole exige au moins 29 nulls indépendants par régime, ou 36 par régime avec correction de Bonferroni sur deux régimes.

### 2.4 Ablation exploratoire de magnitude (hypothèse K6b)

Le §6 du protocole pré-run annonçait une seconde amplitude candidate :

\[
p_{{K6b}}
=\kappa\min\!\left\{{G,\sqrt{{(b_Y)_+}}\right\}}h_r(Y).
\]

Dans les 72 cellules M0, \(b_Y>b_{{\min}}\), \(\sqrt{{b_Y}}<G\), la régularisation radiale est inactive et \(0<\kappa<1\). La formule se simplifie alors exactement en

\[
p_{{K6b}}
=\frac{{(a-\tau_a)_+}}{{\sqrt{{b_X}}}}\,h_r(Y).
\]

Le gain ne valide donc pas trois composantes statistiquement indépendantes. Il nomine plus précisément une **projection temporelle EIV régularisée**, dont l'amplitude est normalisée par l'énergie passée. Sa version prospective utilisera les planchers publics \(b_0>0\), \(r_{{\min}}>0\) et le cap \(G\), afin d'être globalement bornée et Lipschitz.

Elle n'a pas été sauvegardée comme bras du runner M0. Sa MSE peut néanmoins être reconstruite algébriquement, sans réexécuter l'expérience, à partir des colonnes figées. En posant \(u=h_r(Y)\), \(z=\langle u,\mathrm{{target}}\rangle\), \(d=8\) et en utilisant la MSE de Radial-Y \(Gu\),

\[
z=\frac{{G^2+\lVert\mathrm{{target}}\rVert_2^2-d\,MSE_{{radial}}}}{{2G}},
\qquad
a=\kappa\min\!\left\{{G,\sqrt{{(b_Y)_+}}\right\}},
\]

\[
MSE_{{K6b}}
=\frac{{a^2-2az+\lVert\mathrm{{target}}\rVert_2^2}}{{d}}.
\]

| Bruit | Menace | MSE Radial-Y | MSE K6b | MSE Identity-Y | Gain K6b vs radial | Gain K6b vs Identity-Y |
|---|---|---:|---:|---:|---:|---:|
{_magnitude_ablation_table(magnitude_rows)}

![Ablation exploratoire de magnitude]({relative_figures['exploratory_magnitude_ablation.png']})

Le contrôle décisif n'est pas Radial-Y, qui force artificiellement la norme à \(G=0,13\), mais Identity-Y : la vue récente conserve sa norme observée, proche de celle de la cible. K6b est inférieur à Identity-Y sur **toutes les seeds** dans les deux régimes de bruit.

![K6b contre Identity-Y]({relative_figures['k6b_vs_identity_y_seed_contrasts.png']})

| Contraste attaqué | Gain relatif de K6b et IC 95 % |
|---|---:|
| K6b vs Identity-Y, bruit homogène | {_ci_text(identity_hom_attacked)} |
| K6b vs Identity-Y, bruit hétéroscédastique | {_ci_text(identity_hetero_attacked)} |

**Statut méthodologique.** La formule K6b était mentionnée avant le run, mais ce contraste et ses gates n'étaient pas préenregistrés ; le calcul a été réalisé après ouverture des résultats. Il s'agit d'une **falsification exploratoire forte**, pas d'une confirmation. Elle interdit de promouvoir K6b sans un nouveau redesign capable de battre prospectivement Identity-Y/K4b.

Vérifications numériques : {_decimal(float(magnitude_verification['maximum_radial_identity_absolute_error']), 12)} d'erreur absolue maximale sur l'identité de reconstruction ; {_decimal(float(magnitude_verification['maximum_radial_norm_vs_cap_absolute_error']), 10)} d'écart maximal entre la norme Radial-Y et le cap \(G\) ; {magnitude_verification['row_count']} amplitudes K6b et Identity-Y reconstruites, toutes dans \([0,G]\).

## 3. Validité et traçabilité

- Vérification préalable du manifeste : **réussie**, {len(manifest['artifact_sha256'])} artefacts et {len(manifest['source_sha256'])} sources concordent bit à bit.
- Checks de validité : **{'tous réussis' if validity_ok else 'échec détecté'}**.
- Checks exploratoires enregistrés par le runner : **{'tous réussis' if science_ok else 'au moins un échec'}**, mais ils étaient incomplets et ne constituent pas la décision scientifique.
- 24 lignes de calibration, 24 lignes de validation nulle indépendante et 72 cellules d'évaluation.
- Toutes les exécutions scientifiques déclarent MPS/float32, sans fallback CPU.
- Covariances PSD dans toutes les cellules ; 96 blocs par contexte ; aucune covariance jointe dense matérialisée.
- La PSD certifie une matrice admissible, mais pas son exactitude : après clipping, \(J(Y)\Sigma J(Y)^\top\) est une approximation delta aléatoire, corrélée à l'upload. Le clipping résiduel est actif dans une part substantielle des observations ; M0 ne ferme donc ni le biais de moyenne du clipping ni les restes d'ordre supérieur.
- Les deux flux temporels sont disjoints, le gate est prévisible et aucun input du tour courant n'est utilisé.
- Marge minimale observée aux frontières de clipping : `{min(float(row['minimum_clip_margin']) for row in summary_rows):.9e}` (> 0).

Une limite de déployabilité a été identifiée après exécution : M0 met à zéro la covariance déclarée des identités byzantines remplacées à partir du masque vrai. Cette information est oracle. La prochaine étude doit conserver la covariance nominale publique ou attestée de chaque identité, ou employer un ensemble d'incertitude, sans consulter l'étiquette byzantine.

Le régime hétéroscédastique confond aussi niveau de bruit et identité attaquée : les niveaux sont affectés cycliquement, tandis que les deux dernières identités sont toujours byzantines. P0b devra contrebalancer les permutations bruit--identité et l'ensemble attaquant.

Le hash SHA-256 du manifeste scientifique vérifié est `{verified_hashes['manifest.json']}`. Le générateur du présent rapport n'écrit jamais dans le dossier scientifique figé.

## 4. Interprétation scientifique

### Observations

1. K6 corrigé et K6b améliorent Radial-Y, dont la norme est forcée à \(G=0,13\).
2. La correction EIV augmente la confiance moyenne mais détériore systématiquement la MSE relativement à la version non corrigée.
3. Identity-Y/K4b bat K6b sur les 12 seeds de chaque régime, y compris sous attaque ; les deux IC seed-level sont entièrement négatifs.
4. Les contrôles de nullité indépendants n'activent aucun des deux candidats dans les 24 contextes testés.
5. Cette taille d'échantillon ne suffit toutefois pas à certifier un taux de fausse activation inférieur à 10 % à 95 %.

### Inférences raisonnables

- M0 soutient seulement que **l'amplitude doit être calibrée** : il ne démontre aucun bénéfice propre de la modulation temporelle actuelle.
- Dans cette géométrie, soustraire la covariance DP des énergies normalisatrices rend la confiance plus agressive ; l'agressivité supplémentaire est associée à une MSE plus élevée que le contrôle non corrigé.
- Un prochain candidat doit pouvoir corriger la **direction**, pas seulement redimensionner \(Y\), et doit conserver Identity-Y/K4b comme contrôle primaire dur.

### Non identifiable avec M0

- La cause exacte de l'infériorité EIV : erreur d'approximation delta après clipping, régularisation `b_min`, modèle de covariance, ou simple inadéquation de la correction à la cible.
- L'effet sur accuracy, Worst-20, variance inter-clients, gap de fairness ou convergence d'un modèle appris.
- La robustesse Byzantine générale au-delà des deux remplacements synthétiques testés.
- Le transfert de `n=12,d=8` vers le protocole `n=25,d=64`, puis vers Fashion-MNIST/CIFAR-10.
- Une garantie exacte des moments après clipping en dehors du régime localement linéaire.
- Toute valeur confirmatoire de K6b : son résultat actuel est une falsification exploratoire face à Identity-Y.
- La déployabilité des résultats attaqués tant que la covariance des identités remplacées dépend du masque byzantin oracle.
- La séparation entre effet du niveau de bruit et effet de l'identité byzantine dans l'affectation hétéroscédastique de M0.

## 5. Décision et étape suivante

La décision correcte n'est pas « K6 Gaussian-aware validé ». Elle est :

1. **arrêter le P0 original** qui utilisait K6 fixed-\(G\) corrigé comme candidat primaire ;
2. **ne pas promouvoir K6b**, battu par Identity-Y/K4b dans chaque régime ;
3. conserver la falsification utile : une modulation scalaire de \(Y\) ne suffit pas et la correction EIV actuelle rend le mécanisme trop agressif ;
4. préenregistrer un petit écran de redesign capable de corriger la direction, avec Identity-Y/K4b comme contrôle primaire ;
5. supprimer l'oracle d'identité byzantine, contrebalancer niveau de bruit et identité attaquée, calibrer les moments post-clipping par Monte-Carlo et renforcer le contrôle nul par une borne binomiale exacte ;
6. ne passer à l'end-to-end qu'après succès de tous les gates prospectifs de ce nouveau candidat.

## 6. Limites de revendication

Le présent rapport ne revendique pas : utilité end-to-end, performance de classification, fairness, convergence, robustesse Byzantine universelle, ni nouveauté publiable. Il documente uniquement un résultat mécanistique synthétique, apparié et reproductible. Toute présentation doit conserver cette qualification.
"""

    brief = f"""# K6 Gaussian-aware — résultat M0 pour lundi

## Message à retenir

**Le candidat temporel K6/K6b actuel est réfuté par le contrôle naturel Identity-Y/K4b.**

K6 corrigé améliore Radial-Y de {_ci_text(attacked_ci)} sous attaque, mais il est moins bon que K6 non corrigé de {_pct(-float(corrected_vs_raw_ci['mean']))}. Plus important : K6b perd face à Identity-Y de {_ci_text(identity_hom_attacked)} en bruit homogène et de {_ci_text(identity_hetero_attacked)} en bruit hétéroscédastique. Le gain apparent venait donc surtout du contrôle Radial-Y, qui forçait une amplitude trop grande.

![Contrastes M0]({relative_figures['seed_level_contrasts_95ci.png']})

## Ce qui est établi

- Étude synthétique appariée : 12 seeds d'évaluation, deux régimes de bruit, deux attaques et un contrôle sans attaque.
- Exécution MPS/float32, fallback CPU interdit.
- Seuils calibrés sans les seeds d'évaluation et vérifiés sur 12 autres seeds nulles.
- IC calculés au niveau des seeds, sans pseudoréplication des cellules.
- Tous les checks techniques du manifeste passent.

## Tableau compact

| Contraste | Estimation et IC 95 % |
|---|---:|
| K6 corrigé vs Radial-Y, attaques | {_ci_text(attacked_ci)} |
| K6 corrigé vs non corrigé, toutes cellules | {_ci_text(corrected_vs_raw_ci)} |
| K6b vs Identity-Y, attaques homogènes | {_ci_text(identity_hom_attacked)} |
| K6b vs Identity-Y, attaques hétéroscédastiques | {_ci_text(identity_hetero_attacked)} |
| Activations nulles corrigé/non corrigé | 0/24 ; 0/24 (borne CP95 : {_pct(cp95_zero_over_24)}) |

## Ce qu'il faut dire au professeur

« M0 semblait d'abord favorable face à Radial-Y. L'ajout du contrôle naturel Identity-Y montre toutefois que le gain venait surtout d'une amplitude radiale forcée et mal calibrée. La projection temporelle K6b est battue sur toutes les seeds, et la correction EIV détériore encore le résultat. Nous arrêtons donc cette instanciation et testons désormais une fusion covariance-aware capable de corriger la direction. »

## Décision

**Ne pas lancer le P0 original et ne pas promouvoir K6b.** Le prochain micro-écran doit inclure Identity-Y/K4b comme contrôle primaire, supprimer toute covariance oracle, contrebalancer bruit et identité Byzantine, calibrer les moments après clipping et tester un mécanisme capable de modifier la direction. Ne pas présenter M0 comme une validation end-to-end, de fairness ou de robustesse Byzantine universelle.
"""

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    BRIEF.write_text(brief, encoding="utf-8")


def _write_report_manifest(
    scientific_manifest: Mapping[str, Any],
    verified_hashes: Mapping[str, str],
    figures: Iterable[Path],
    magnitude_rows: Sequence[Mapping[str, Any]],
    magnitude_verification: Mapping[str, Any],
) -> None:
    outputs = [REPORT, BRIEF, *figures]
    magnitude_gain_by_cell = {
        f"{regime}|{threat}": statistics.fmean(
            float(row["relative_gain_vs_radial"])
            for row in magnitude_rows
            if row["noise_regime"] == regime and row["threat"] == threat
        )
        for regime in REGIME_ORDER
        for threat in THREAT_ORDER
    }
    identity_y_intervals = _identity_y_contrast_intervals(magnitude_rows)
    payload = {
        "status": "completed",
        "scope": "post_hoc_reporting_only",
        "scientific_results_modified": False,
        "scientific_campaign_id": scientific_manifest["campaign_id"],
        "scientific_manifest_sha256": verified_hashes["manifest.json"],
        "scientific_inputs_verified_before_read": True,
        "report_generator_sha256": _sha256(Path(__file__).resolve()),
        "post_opening_exploratory_magnitude_ablation": {
            "status": "falsified_by_identity_y_control_after_opening",
            "relative_mse_gain_vs_radial_mean_by_cell": magnitude_gain_by_cell,
            "k6b_relative_mse_gain_vs_identity_y_seed_intervals": identity_y_intervals,
            "verification": dict(magnitude_verification),
        },
        "generated_artifact_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in outputs
        },
    }
    REPORT_MANIFEST.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    manifest, verified_hashes = _verify_frozen_manifest()

    # Scientific artifacts are parsed only after all hashes have passed.
    decision = _read_json(RESULTS / "decision.json")
    intervals = _read_json(RESULTS / "contrast_intervals.json")
    calibration = _read_json(RESULTS / "tau_a_calibration.json")
    protocol = _read_json(RESULTS / "resolved_protocol.json")
    summary_rows = _read_csv(RESULTS / "summary.csv")
    seed_rows = _read_csv(RESULTS / "seed_summary.csv")
    evaluation_rows = _read_csv(RESULTS / "evaluation_rows.csv")

    if not manifest.get("all_checks_pass") or not decision.get("all_checks_pass"):
        raise RuntimeError("The frozen M0 campaign did not pass its recorded checks")
    if len(summary_rows) != len(REGIME_ORDER) * len(THREAT_ORDER):
        raise RuntimeError("Unexpected number of M0 summary cells")
    if len(seed_rows) != 12 * len(REGIME_ORDER):
        raise RuntimeError("Unexpected number of M0 seed-level rows")
    if len(evaluation_rows) != 12 * len(REGIME_ORDER) * len(THREAT_ORDER):
        raise RuntimeError("Unexpected number of M0 evaluation rows")

    magnitude_rows, magnitude_verification = _magnitude_ablation_rows(
        evaluation_rows, protocol
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    figures = (
        _write_mse_figure(summary_rows),
        _write_contrast_figure(intervals),
        _write_paired_figure(seed_rows),
        _write_confidence_figure(summary_rows),
        _write_magnitude_ablation_figure(magnitude_rows),
        _write_identity_y_figure(magnitude_rows),
    )
    _write_reports(
        manifest,
        verified_hashes,
        decision,
        intervals,
        calibration,
        protocol,
        summary_rows,
        seed_rows,
        magnitude_rows,
        magnitude_verification,
        figures,
    )
    _write_report_manifest(
        manifest,
        verified_hashes,
        figures,
        magnitude_rows,
        magnitude_verification,
    )
    print(f"Wrote {REPORT.relative_to(ROOT)}")
    print(f"Wrote {BRIEF.relative_to(ROOT)}")
    print(f"Wrote {len(figures)} figures under {FIGURES.relative_to(ROOT)}")
    print(f"Wrote {REPORT_MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
