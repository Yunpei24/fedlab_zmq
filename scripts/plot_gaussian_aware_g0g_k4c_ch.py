#!/usr/bin/env python3
"""Plot a completed and independently audited G0g-K4c-CH screen.

The tool is deliberately post-run only.  It opens ``manifest.json`` first and
does not read the configuration or any raw CSV while the manifest is absent or
does not report ``completed_development``.  Once complete, the locked
independent auditor is rerun in memory against the current raw artifacts before
any figure is produced.

The outer seed is always the independent statistical unit.  Evaluation
children and preregistered contexts are averaged within seed; they are never
treated as additional replicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_gaussian_aware_reference_g0g_k4c_ch as audit  # noqa: E402

CAMPAIGN_ID = "gaussian_aware_reference_g0g_k4c_causal_headroom_mps_v1"
DEFAULT_CONFIG = ROOT / (
    "configs/ldp_gradient_far/" "gaussian_aware_reference_g0g_k4c_causal_headroom.yaml"
)
DEFAULT_RESULTS = ROOT / (
    "results/ldp_gradient_far/"
    "gaussian_aware_reference_g0g_k4c_causal_headroom_mps_v1"
)
DEFAULT_OUTPUT = ROOT / "output/figures/gaussian_aware_g0g_k4c"

K2 = "g0g_k2"
K4 = "g0g_k4_temporal_causal_gate"
K4B = "g0g_k4b_rolling_past_imputation"
K4C = "g0g_k4c_ch_current_randomness_conditional_mse_semi_oracle"
POINTWISE = "g0g_k4b_pointwise_optimal_oracle"
CANDIDATES = (K2, K4, K4B, K4C, POINTWISE)
MSE_CANDIDATES = (K4, K4B, K4C, POINTWISE)

REGIMES = ("homogeneous", "heteroscedastic")
REGIME_LABELS = {
    "homogeneous": "Bruit homogène",
    "heteroscedastic": "Bruit hétéroscédastique",
}
CANDIDATE_LABELS = {
    K4: "K4",
    K4B: "K4b",
    K4C: "K4c-CH",
    POINTWISE: "Pointwise",
}
COLORS = {
    K4: "#0072B2",
    K4B: "#E69F00",
    K4C: "#009E73",
    POINTWISE: "#4D4D4D",
}
PASS_COLOR = "#198754"
FAIL_COLOR = "#C23B22"


@dataclass(frozen=True)
class ValidatedSources:
    """Completed artifacts validated against the current independent audit."""

    manifest: dict[str, Any]
    config: dict[str, Any]
    decision: dict[str, Any]
    audit_result: dict[str, Any]
    histories: pd.DataFrame
    children: pd.DataFrame
    seed_summary: pd.DataFrame


@dataclass(frozen=True)
class PqcSpec:
    symbol: str
    title: str
    column: str
    interval_key: str
    mean_gate_key: str
    low_gate_key: str
    mean_check: str
    low_check: str
    low_is_strict: bool
    color: str


PQC_SPECS = (
    PqcSpec(
        symbol="P",
        title="Headroom pointwise relatif vs K4",
        column="pointwise_relative_mse_headroom_vs_k4",
        interval_key="pointwise_relative_mse_headroom_vs_k4_seed_ci95",
        mean_gate_key="pointwise_relative_mse_headroom_vs_k4_mean_min",
        low_gate_key=(
            "pointwise_relative_mse_headroom_vs_k4_seed_ci95_low_"
            "strictly_greater_than"
        ),
        mean_check="material_pointwise_headroom_mean",
        low_check="material_pointwise_headroom_ci",
        low_is_strict=True,
        color="#0072B2",
    ),
    PqcSpec(
        symbol="Q",
        title="Gain MSE relatif K4c-CH vs K4",
        column="semi_oracle_relative_mse_gain_vs_k4",
        interval_key="semi_oracle_relative_mse_gain_vs_k4_seed_ci95",
        mean_gate_key="semi_oracle_relative_mse_gain_vs_k4_mean_min",
        low_gate_key=(
            "semi_oracle_relative_mse_gain_vs_k4_seed_ci95_low_" "strictly_greater_than"
        ),
        mean_check="material_semi_oracle_gain_vs_k4_mean",
        low_check="material_semi_oracle_gain_vs_k4_ci",
        low_is_strict=True,
        color="#009E73",
    ),
    PqcSpec(
        symbol="C",
        title="Fraction du headroom capturée",
        column="capture_fraction",
        interval_key="semi_oracle_capture_fraction_seed_ci95",
        mean_gate_key="semi_oracle_capture_fraction_mean_min",
        low_gate_key=("semi_oracle_capture_fraction_ci95_low_strictly_greater_than"),
        mean_check="capture_mean",
        low_check="capture_ci_low",
        low_is_strict=True,
        color="#CC79A7",
    ),
)


GATE_LABELS = {
    "complete": "Artefacts complets",
    "matrix_exact": "Matrice préenregistrée exacte",
    "finite": "Toutes les métriques sont finies",
    "production_device": "Exécution MPS/float32",
    "construction_evaluation_independent": (
        "Flux construction et évaluation indépendants"
    ),
    "rng_global_unique": "Seeds enfants globalement uniques",
    "predictor_fixed_across_evaluation_children": (
        "Prédicteur fixe entre enfants d’évaluation"
    ),
    "pointwise_not_used_in_construction": (
        "Oracle pointwise absent de la construction"
    ),
    "exact_outer_seed_count": "Nombre exact de seeds externes",
    "positive_finite_mse_denominators": "Dénominateurs MSE positifs et finis",
    "construction_mc_stability": "Stabilité split-half de la construction MC",
    "contribution_cap": "Cap de contribution respecté",
    "replace_one": "Audit replace-one sans violation",
    "replace_one_complete": "Couverture replace-one complète",
    "k4_manual_formula": "Identité manuelle K4",
    "k4b_reproduction": "Reproduction du comparateur K4b",
    "no_gate_sum_normalization": "Aucune normalisation par somme des gates",
    "fixed_denominator": "Dénominateur public fixe n",
    "pointwise_oracle_dominance": "Dominance de l’oracle pointwise",
    "eligible_R": "Tous les historiques ont R > 0",
    "eligible_noise_cell_composition": "Composition exacte des cellules de bruit",
    "positive_finite_pointwise_headroom_denominators": (
        "Dénominateurs de headroom pointwise positifs"
    ),
    "exact_scientific_seed_count": "Nombre exact de seeds dans tous les IC",
    "material_pointwise_headroom_mean": "P moyen atteint son seuil",
    "material_pointwise_headroom_ci": "Borne basse IC95 de P franchit son seuil",
    "material_semi_oracle_gain_vs_k4_mean": "Q moyen atteint son seuil",
    "material_semi_oracle_gain_vs_k4_ci": ("Borne basse IC95 de Q franchit son seuil"),
    "capture_mean": "C moyen atteint son seuil",
    "capture_ci_low": "Borne basse IC95 de C franchit son seuil",
    "gain_vs_k4b": "Gain relatif moyen K4c-CH vs K4b",
    "gain_vs_k4b_ci": "Borne basse IC95 du gain vs K4b",
    "ci_vs_k4b": "Borne haute IC95 de MSE(K4c-CH − K4b)",
    "homogeneous_gain_vs_k4_ci": "Q homogène : borne basse IC95 > seuil",
    "heteroscedastic_gain_vs_k4_ci": ("Q hétéroscédastique : borne basse IC95 > seuil"),
}


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linestyle": "--",
            "lines.linewidth": 1.8,
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guard_completed(results_dir: Path) -> dict[str, Any]:
    """Read only the manifest until the production writer reports completion."""

    path = results_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing K4c-CH manifest: {path}")
    manifest = _read_json(path)
    if manifest.get("status") != "completed_development":
        raise RuntimeError(
            "K4c-CH is not complete; config, audit, and raw CSVs were not "
            f"opened: status={manifest.get('status')!r}"
        )
    if manifest.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError(
            f"Unexpected K4c-CH campaign: {manifest.get('campaign_id')!r}"
        )
    return manifest


def _as_bool(series: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    lowered = series.astype(str).str.strip().str.lower()
    if not bool(lowered.isin(("true", "false")).all()):
        raise ValueError(f"{name} is not Boolean")
    return lowered.eq("true")


def _require_columns(frame: pd.DataFrame, required: set[str], *, source: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(missing)}")


def _numeric_finite(
    frame: pd.DataFrame, columns: Sequence[str], *, source: str
) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="raise")
        if not bool(np.isfinite(values.to_numpy(dtype=float)).all()):
            raise ValueError(f"{source}.{column} contains non-finite values")
        frame[column] = values


def _require_integer_values(frame: pd.DataFrame, column: str, *, source: str) -> None:
    values = frame[column].to_numpy(dtype=float)
    if not bool(np.equal(values, np.floor(values)).all()):
        raise ValueError(f"{source}.{column} must contain exact integers")
    frame[column] = values.astype(np.int64)


def _require_json_bool(value: Any, *, name: str, allow_none: bool = False) -> None:
    if allow_none and value is None:
        return
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON Boolean")


def _validate_sources(results_dir: Path, config_path: Path) -> ValidatedSources:
    """Validate a completed result without ever accepting partial raw output."""

    manifest_path = results_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing K4c-CH manifest: {manifest_path}")
    manifest_hash_before_guard = _sha256(manifest_path)
    manifest = _guard_completed(results_dir)
    if _sha256(manifest_path) != manifest_hash_before_guard:
        raise RuntimeError("K4c-CH manifest changed while checking completion")

    required = {
        "decision": results_dir / "decision.json",
        "histories": results_dir / "frozen_history_rows.csv",
        "children": results_dir / "evaluation_child_rows.csv",
        "seeds": results_dir / "seed_summary.csv",
        "replacements": results_dir / "replace_one_audit.csv",
        "calibration_provenance": (results_dir / "frozen_calibration_provenance.json"),
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Completed K4c-CH directory is incomplete; no raw artifact was "
            "opened: " + ", ".join(missing)
        )
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing K4c-CH config: {config_path}")

    snapshot_paths = {
        "manifest": manifest_path,
        "config": config_path,
        "decision": required["decision"],
        "histories": required["histories"],
        "children": required["children"],
        "seeds": required["seeds"],
        "replacements": required["replacements"],
        "calibration_provenance": required["calibration_provenance"],
    }
    hashes_before_audit = {name: _sha256(path) for name, path in snapshot_paths.items()}

    config_value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config_value, dict):
        raise ValueError("K4c-CH configuration must be a mapping")
    config: dict[str, Any] = config_value
    if config.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError("Unexpected K4c-CH configuration")
    if _sha256(config_path) != manifest.get("config_sha256"):
        raise RuntimeError("Config differs from the completed K4c-CH manifest")

    if (
        manifest.get("device") != "mps"
        or manifest.get("dtype") != "torch.float32"
        or manifest.get("development_only") is not True
    ):
        raise RuntimeError("Figures require the completed MPS/float32 screen")
    if manifest.get("holdout_opened") is not False:
        raise RuntimeError("Reserved holdout must remain closed")
    if manifest.get("observable_past_only_predictor_constructed") is not False:
        raise RuntimeError("K4c-CH must remain a privileged development screen")

    # No output path is supplied: the independent audit is current but remains
    # read-only with respect to the completed results directory.
    audit_result = audit.audit_results(config_path, results_dir, output_path=None)
    if (
        audit_result.get("all_checks_pass") is not True
        or audit_result.get("audit_status") != "passed"
        or audit_result.get("seed_summary_differences")
        or audit_result.get("decision_differences")
        or audit_result.get("holdout_opened") is not False
    ):
        raise RuntimeError("Current independent K4c-CH audit did not pass")

    decision = _read_json(required["decision"])
    histories = pd.read_csv(required["histories"], low_memory=False)
    children = pd.read_csv(required["children"], low_memory=False)
    seed_summary = pd.read_csv(required["seeds"], low_memory=False)
    hashes_after_load = {name: _sha256(path) for name, path in snapshot_paths.items()}
    changed = sorted(
        name
        for name in snapshot_paths
        if hashes_before_audit[name] != hashes_after_load[name]
    )
    if changed:
        raise RuntimeError(
            "K4c-CH artifacts changed during audit/load: " + ", ".join(changed)
        )

    if decision.get("holdout_opened") is not False:
        raise RuntimeError("Decision artifact reports an opened holdout")
    if decision.get("loss_used_for_all_scientific_gates") != (
        "squared_l2_reference_error"
    ):
        raise RuntimeError("Scientific plots require the preregistered MSE loss")
    if audit_result["recomputed_decision"].get("decision") != decision.get("decision"):
        raise RuntimeError("Reported and independently recomputed decisions differ")

    _require_columns(
        histories,
        {"history_id", "seed", "noise_regime", "eligible_R_positive"},
        source="frozen_history_rows.csv",
    )
    _require_columns(
        children,
        {
            "history_id",
            "seed",
            "noise_regime",
            "evaluation_child",
            "candidate",
            "squared_reference_error",
        },
        source="evaluation_child_rows.csv",
    )
    _require_columns(
        seed_summary,
        {
            "seed",
            "pointwise_relative_mse_headroom_vs_k4",
            "semi_oracle_relative_mse_gain_vs_k4",
            "capture_fraction",
            "homogeneous_semi_oracle_relative_mse_gain_vs_k4",
            "heteroscedastic_semi_oracle_relative_mse_gain_vs_k4",
        },
        source="seed_summary.csv",
    )
    histories["eligible_R_positive_bool"] = _as_bool(
        histories["eligible_R_positive"],
        name="frozen_history_rows.eligible_R_positive",
    )
    _numeric_finite(histories, ["seed"], source="frozen_history_rows.csv")
    _numeric_finite(
        children,
        ["seed", "evaluation_child", "squared_reference_error"],
        source="evaluation_child_rows.csv",
    )
    _numeric_finite(
        seed_summary,
        [
            "seed",
            "pointwise_relative_mse_headroom_vs_k4",
            "semi_oracle_relative_mse_gain_vs_k4",
            "capture_fraction",
            "homogeneous_semi_oracle_relative_mse_gain_vs_k4",
            "heteroscedastic_semi_oracle_relative_mse_gain_vs_k4",
        ],
        source="seed_summary.csv",
    )
    _require_integer_values(histories, "seed", source="frozen_history_rows.csv")
    _require_integer_values(children, "seed", source="evaluation_child_rows.csv")
    _require_integer_values(
        children, "evaluation_child", source="evaluation_child_rows.csv"
    )
    _require_integer_values(seed_summary, "seed", source="seed_summary.csv")
    if bool((children["squared_reference_error"] < 0.0).any()):
        raise ValueError("Squared reference error cannot be negative")

    expected_seeds = {
        int(value) for value in config["randomness"]["development_outer_seeds"]
    }
    for name, frame in (
        ("histories", histories),
        ("children", children),
        ("seed_summary", seed_summary),
    ):
        observed_seeds = set(frame["seed"].unique())
        if observed_seeds != expected_seeds:
            raise RuntimeError(
                f"{name}: unexpected seed registry {sorted(observed_seeds)}"
            )
    if len(seed_summary) != len(expected_seeds) or bool(
        seed_summary.duplicated(["seed"]).any()
    ):
        raise RuntimeError("seed_summary.csv must contain one row per outer seed")

    expected_regimes = {
        str(regime["name"]) for regime in config["privacy_noise"]["regimes"]
    }
    if expected_regimes != set(REGIMES):
        raise RuntimeError("K4c-CH noise-regime contract changed")
    for name, frame in (("histories", histories), ("children", children)):
        if set(frame["noise_regime"].astype(str).unique()) != expected_regimes:
            raise RuntimeError(f"{name}: noise-regime registry is incomplete")
    if set(children["candidate"].astype(str).unique()) != set(CANDIDATES):
        raise RuntimeError("evaluation_child_rows.csv candidate registry changed")
    if bool(histories.duplicated(["history_id"]).any()):
        raise RuntimeError("Duplicate frozen history identifiers")
    if bool(children.duplicated(["history_id", "evaluation_child", "candidate"]).any()):
        raise RuntimeError("Duplicate evaluation-child candidate identifiers")

    if int(manifest.get("frozen_histories", -1)) != len(histories):
        raise RuntimeError("Manifest frozen-history count differs from raw CSV")
    if int(manifest.get("evaluation_child_rows", -1)) != len(children):
        raise RuntimeError("Manifest evaluation-child count differs from raw CSV")

    classification = config["statistical_analysis"]["gate_classification"]
    validity_keys = set(classification["validity"])
    scientific_keys = set(classification["scientific"])
    if set(decision.get("validity_checks", {})) != validity_keys:
        raise RuntimeError("Validity-check registry differs from the config")
    if set(decision.get("scientific_checks", {})) != scientific_keys:
        raise RuntimeError("Scientific-check registry differs from the config")

    validity_checks = decision["validity_checks"]
    scientific_checks = decision["scientific_checks"]
    combined_checks = decision.get("checks", {})
    if set(combined_checks) != validity_keys | scientific_keys:
        raise RuntimeError("Combined check registry differs from the config")
    for group_name, checks in (
        ("validity_checks", validity_checks),
        ("scientific_checks", scientific_checks),
        ("checks", combined_checks),
    ):
        for key, value in checks.items():
            _require_json_bool(value, name=f"decision.{group_name}.{key}")
    for key, value in {**validity_checks, **scientific_checks}.items():
        if combined_checks[key] is not value:
            raise RuntimeError(f"decision.checks.{key} differs from its gate group")
    for name in ("all_gates_pass", "validity_pass"):
        _require_json_bool(decision.get(name), name=f"decision.{name}")
    _require_json_bool(
        decision.get("scientific_checks_pass"),
        name="decision.scientific_checks_pass",
        allow_none=True,
    )
    if decision["validity_pass"] is True:
        if type(decision["scientific_checks_pass"]) is not bool:
            raise ValueError(
                "decision.scientific_checks_pass must be Boolean after valid screen"
            )
    elif decision["scientific_checks_pass"] is not None:
        raise ValueError(
            "decision.scientific_checks_pass must be null after invalid screen"
        )

    return ValidatedSources(
        manifest=manifest,
        config=config,
        decision=decision,
        audit_result=audit_result,
        histories=histories,
        children=children,
        seed_summary=seed_summary,
    )


def _student_ci(values: Sequence[float], t_critical: float) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) < 2:
        raise ValueError("A Student interval requires at least two finite seeds")
    mean = statistics.fmean(finite)
    sd = statistics.stdev(finite)
    half = float(t_critical) * sd / math.sqrt(len(finite))
    return {
        "n": len(finite),
        "mean": mean,
        "sd": sd,
        "low": mean - half,
        "high": mean + half,
    }


def _validate_interval(
    computed: Mapping[str, float | int],
    reported: Mapping[str, Any],
    *,
    name: str,
) -> None:
    if int(computed["n"]) != int(reported.get("n", -1)):
        raise RuntimeError(f"{name}: reported seed count differs from the CSV")
    for key in ("mean", "sd", "low", "high"):
        if not math.isclose(
            float(computed[key]),
            float(reported.get(key, float("nan"))),
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(f"{name}: reported {key} differs from seed values")


def _padded_limits(
    values: Sequence[float], *, include_zero: bool = False
) -> tuple[float, float]:
    finite = np.asarray(
        [value for value in values if math.isfinite(value)], dtype=float
    )
    if finite.size == 0:
        raise ValueError("Cannot derive plot limits without finite values")
    low = float(finite.min())
    high = float(finite.max())
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    span = high - low
    padding = 0.10 * span if span > 0.0 else max(abs(high) * 0.10, 1.0)
    return low - padding, high + padding


def _scientific_status(
    decision: Mapping[str, Any], check_names: Sequence[str]
) -> tuple[str, str]:
    """Return a gate label without interpreting science after invalidity."""

    validity_pass = decision.get("validity_pass")
    _require_json_bool(validity_pass, name="decision.validity_pass")
    if validity_pass is not True:
        return "NON INTERPRÉTABLE", "#666666"
    values = []
    for name in check_names:
        value = decision["checks"][name]
        _require_json_bool(value, name=f"decision.checks.{name}")
        values.append(value)
    passed = all(value is True for value in values)
    return ("PASS", PASS_COLOR) if passed else ("FAIL", FAIL_COLOR)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(temporary, format="png", bbox_inches="tight")
    temporary.replace(path)
    plt.close(fig)


def plot_pqc_by_seed(
    seed_summary: pd.DataFrame,
    decision: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: Path,
) -> Path:
    """Plot preregistered P, Q, and C seed ratios with both gate thresholds."""

    ordered = seed_summary.sort_values("seed").reset_index(drop=True)
    seeds = ordered["seed"].astype(int).tolist()
    t_critical = float(config["statistical_analysis"]["t_critical_df11"])
    gates = config["gates"]
    observed = decision["observed"]
    x = np.arange(len(seeds), dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.7), sharex=True)
    export = pd.DataFrame({"seed": seeds})
    for ax, spec in zip(axes, PQC_SPECS, strict=True):
        raw = ordered[spec.column].to_numpy(dtype=float)
        interval = _student_ci(raw, t_critical)
        _validate_interval(interval, observed[spec.interval_key], name=spec.symbol)
        export[spec.symbol] = raw

        values = 100.0 * raw
        mean = 100.0 * interval["mean"]
        ci_low = 100.0 * interval["low"]
        ci_high = 100.0 * interval["high"]
        mean_gate = 100.0 * float(gates[spec.mean_gate_key])
        low_gate = 100.0 * float(gates[spec.low_gate_key])
        status, status_color = _scientific_status(
            decision, (spec.mean_check, spec.low_check)
        )

        ax.axhspan(ci_low, ci_high, color=spec.color, alpha=0.11, zorder=0)
        ax.axhline(mean, color=spec.color, linewidth=2.2, zorder=2)
        ax.axhline(
            mean_gate,
            color="#111111",
            linestyle=(0, (5, 3)),
            linewidth=1.1,
            zorder=1,
        )
        ax.axhline(
            low_gate,
            color="#555555",
            linestyle=(0, (1, 2)),
            linewidth=1.3,
            zorder=1,
        )
        ax.plot(x, values, color="#A8A8A8", linewidth=0.9, zorder=2)
        ax.scatter(
            x,
            values,
            s=38,
            color=spec.color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        lower, upper = _padded_limits(
            [*values, mean, ci_low, ci_high, mean_gate, low_gate],
            include_zero=True,
        )
        ax.set_ylim(lower, upper)
        ax.set_xticks(x, [str(seed) for seed in seeds], rotation=55, ha="right")
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
        ax.set_title(
            f"{spec.symbol} — {spec.title}\n"
            f"moy. {mean:.2f} %, IC95 [{ci_low:.2f}, {ci_high:.2f}] % — "
            f"{status}",
            color=status_color,
        )
        ax.set_xlabel("Seed externe")
    axes[0].set_ylabel("Ratio (%)")

    handles = [
        Line2D(
            [0],
            [0],
            color="#A8A8A8",
            marker="o",
            markerfacecolor="#666666",
            label="Valeur par seed",
        ),
        Line2D([0], [0], color="#333333", linewidth=2.2, label="Moyenne"),
        Patch(facecolor="#777777", alpha=0.14, label="IC95 Student sur les seeds"),
        Line2D(
            [0],
            [0],
            color="#111111",
            linestyle=(0, (5, 3)),
            label="Seuil préenregistré de moyenne",
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle=(0, (1, 2)),
            label="Seuil préenregistré de borne basse",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "G0g-K4c-CH — P, Q et C par unité seed indépendante",
        fontsize=15,
        y=0.985,
    )
    fig.text(
        0.5,
        0.008,
        "Les seuils portent sur la moyenne et la borne basse de l’IC95, pas sur "
        "chaque seed prise isolément. P et Q sont relatifs à K4; C est la part "
        "du headroom pointwise capturée.",
        ha="center",
        fontsize=8.7,
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.83, bottom=0.29, wspace=0.24)
    path = output_dir / "01_p_q_c_par_seed.png"
    _save_figure(fig, path)
    _atomic_csv(export, output_dir / "01_p_q_c_seed_values.csv")
    return path


def _mse_seed_values(
    histories: pd.DataFrame,
    children: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    eligibility = histories[
        ["history_id", "seed", "noise_regime", "eligible_R_positive_bool"]
    ].rename(
        columns={
            "seed": "history_seed",
            "noise_regime": "history_noise_regime",
        }
    )
    merged = children.merge(
        eligibility,
        on="history_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not bool(merged["_merge"].eq("both").all()):
        raise RuntimeError("At least one evaluation child lacks a frozen history")
    if not bool(merged["seed"].eq(merged["history_seed"]).all()):
        raise RuntimeError("Child/history seed metadata mismatch")
    if not bool(merged["noise_regime"].eq(merged["history_noise_regime"]).all()):
        raise RuntimeError("Child/history noise-regime metadata mismatch")

    selected = merged.loc[
        merged["eligible_R_positive_bool"] & merged["candidate"].isin(MSE_CANDIDATES)
    ]
    eligible_ids = set(
        eligibility.loc[eligibility["eligible_R_positive_bool"], "history_id"].astype(
            str
        )
    )
    if not eligible_ids:
        raise RuntimeError("No eligible R-positive history is available for MSE")
    if set(selected["history_id"].astype(str)) != eligible_ids:
        raise RuntimeError("At least one eligible history lacks an MSE candidate")
    evaluation_children = int(config["nested_monte_carlo"]["evaluation_children"])
    if evaluation_children <= 0:
        raise RuntimeError("Configured evaluation-child count must be positive")
    if bool(selected.duplicated(["history_id", "evaluation_child", "candidate"]).any()):
        raise RuntimeError("Duplicate MSE child/candidate key")
    expected_rows = len(eligible_ids) * evaluation_children * len(MSE_CANDIDATES)
    if len(selected) != expected_rows:
        raise RuntimeError("Eligible history/candidate MSE matrix is incomplete")
    child_counts = selected.groupby(["history_id", "candidate"], observed=True)[
        "evaluation_child"
    ].agg(["count", "nunique", "min", "max"])
    if not bool(
        child_counts["count"].eq(evaluation_children).all()
        and child_counts["nunique"].eq(evaluation_children).all()
        and child_counts["min"].eq(0).all()
        and child_counts["max"].eq(evaluation_children - 1).all()
    ):
        raise RuntimeError("Evaluation-child registry is incomplete within a history")
    candidates_per_history = selected.groupby("history_id", observed=True)[
        "candidate"
    ].nunique()
    if not bool(candidates_per_history.eq(len(MSE_CANDIDATES)).all()):
        raise RuntimeError("At least one eligible history lacks an MSE candidate")

    # Preserve the registered hierarchy explicitly: first average the current
    # randomness children of each frozen history, then give every eligible
    # history equal weight inside its seed/regime cell.
    per_history = (
        selected.groupby(
            ["history_id", "seed", "noise_regime", "candidate"],
            observed=True,
            sort=True,
        )["squared_reference_error"]
        .mean()
        .rename("history_mse")
        .reset_index()
    )
    values = (
        per_history.groupby(
            ["seed", "noise_regime", "candidate"],
            observed=True,
            sort=True,
        )["history_mse"]
        .mean()
        .rename("mse")
        .reset_index()
    )
    expected_seeds = len(config["randomness"]["development_outer_seeds"])
    expected = {
        (regime, candidate) for regime in REGIMES for candidate in MSE_CANDIDATES
    }
    observed = set(zip(values["noise_regime"], values["candidate"], strict=True))
    if observed != expected:
        raise RuntimeError("MSE figure is missing a regime/candidate estimand")
    counts = values.groupby(["noise_regime", "candidate"], observed=True)[
        "seed"
    ].nunique()
    if not bool(counts.eq(expected_seeds).all()):
        raise RuntimeError("MSE figure is missing at least one outer seed")
    return values


def plot_mse_by_noise_regime(
    histories: pd.DataFrame,
    children: pd.DataFrame,
    config: Mapping[str, Any],
    output_dir: Path,
) -> Path:
    """Compare descriptive conditional MSEs within each noise regime."""

    values = _mse_seed_values(histories, children, config)
    summary = (
        values.groupby(["noise_regime", "candidate"], observed=True)["mse"]
        .agg(mean="mean", sd="std", n_seeds="count")
        .reset_index()
    )
    if bool(summary["sd"].isna().any()):
        raise RuntimeError("MSE dispersion requires multiple outer seeds")

    x = np.arange(len(MSE_CANDIDATES), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)
    for ax, regime in zip(axes, REGIMES, strict=True):
        regime_values = values.loc[values["noise_regime"].eq(regime)]
        pivot = regime_values.pivot(index="seed", columns="candidate", values="mse")
        pivot = pivot[list(MSE_CANDIDATES)]
        for _, row in pivot.iterrows():
            ax.plot(
                x,
                row.to_numpy(dtype=float),
                color="#B0B0B0",
                linewidth=0.8,
                alpha=0.60,
                zorder=1,
            )
            ax.scatter(
                x,
                row.to_numpy(dtype=float),
                color="#B0B0B0",
                s=12,
                alpha=0.65,
                zorder=1,
            )
        stats = summary.loc[summary["noise_regime"].eq(regime)].set_index("candidate")
        for position, candidate in enumerate(MSE_CANDIDATES):
            ax.errorbar(
                position,
                float(stats.loc[candidate, "mean"]),
                yerr=float(stats.loc[candidate, "sd"]),
                fmt="o",
                markersize=8,
                capsize=4,
                color=COLORS[candidate],
                label=CANDIDATE_LABELS[candidate],
                zorder=3,
            )
        ax.set_xticks(
            x,
            [CANDIDATE_LABELS[value] for value in MSE_CANDIDATES],
            rotation=15,
        )
        ax.set_title(REGIME_LABELS[regime])
        ax.set_xlabel("Référence candidate")
        ax.set_ylim(bottom=0.0)
    axes[0].set_ylabel(r"MSE conditionnelle $\|\widehat{F}-\mu^{\mathcal{H}}\|_2^2$")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "MSE de K4, K4b, K4c-CH et du benchmark pointwise par régime de bruit",
        fontsize=14.5,
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "Traits gris : seeds appariées. Points colorés : moyenne ± 1 SD entre "
        "seeds, après moyenne intra-seed sur les enfants et contextes éligibles. "
        "Pointwise est un benchmark non déployable.",
        ha="center",
        fontsize=8.7,
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.86, bottom=0.23, wspace=0.12)
    path = output_dir / "02_mse_k4_k4b_k4c_pointwise_par_bruit.png"
    _save_figure(fig, path)
    _atomic_csv(values, output_dir / "02_mse_by_noise_seed_values.csv")
    return path


def _q_noise_seed_values(seed_summary: pd.DataFrame) -> pd.DataFrame:
    renamed = seed_summary[
        [
            "seed",
            "homogeneous_semi_oracle_relative_mse_gain_vs_k4",
            "heteroscedastic_semi_oracle_relative_mse_gain_vs_k4",
        ]
    ].rename(
        columns={
            "homogeneous_semi_oracle_relative_mse_gain_vs_k4": "homogeneous",
            "heteroscedastic_semi_oracle_relative_mse_gain_vs_k4": ("heteroscedastic"),
        }
    )
    return renamed.melt(
        id_vars="seed",
        value_vars=list(REGIMES),
        var_name="noise_regime",
        value_name="Q",
    ).sort_values(["seed", "noise_regime"])


def plot_q_by_noise_regime(
    seed_summary: pd.DataFrame,
    decision: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: Path,
) -> Path:
    """Show the preregistered paired Q contrast by noise regime."""

    values = _q_noise_seed_values(seed_summary)
    t_critical = float(config["statistical_analysis"]["t_critical_df11"])
    interval_keys = {
        "homogeneous": "homogeneous_semi_oracle_relative_mse_gain_vs_k4_seed_ci95",
        "heteroscedastic": (
            "heteroscedastic_semi_oracle_relative_mse_gain_vs_k4_seed_ci95"
        ),
    }
    gate_keys = {
        "homogeneous": (
            "homogeneous_semi_oracle_relative_mse_gain_vs_k4_seed_ci95_low_"
            "strictly_greater_than"
        ),
        "heteroscedastic": (
            "heteroscedastic_semi_oracle_relative_mse_gain_vs_k4_seed_ci95_low_"
            "strictly_greater_than"
        ),
    }
    check_keys = {
        "homogeneous": "homogeneous_gain_vs_k4_ci",
        "heteroscedastic": "heteroscedastic_gain_vs_k4_ci",
    }
    intervals: dict[str, dict[str, float | int]] = {}
    for regime in REGIMES:
        sample = values.loc[values["noise_regime"].eq(regime), "Q"].to_numpy(
            dtype=float
        )
        intervals[regime] = _student_ci(sample, t_critical)
        _validate_interval(
            intervals[regime],
            decision["observed"][interval_keys[regime]],
            name=f"Q/{regime}",
        )

    pivot = values.pivot(index="seed", columns="noise_regime", values="Q")
    pivot = pivot[list(REGIMES)] * 100.0
    x = np.arange(len(REGIMES), dtype=float)
    fig, ax = plt.subplots(figsize=(8.7, 5.8))
    for _, row in pivot.iterrows():
        ax.plot(
            x,
            row.to_numpy(dtype=float),
            color="#AAAAAA",
            linewidth=0.9,
            alpha=0.68,
            zorder=1,
        )
        ax.scatter(
            x,
            row.to_numpy(dtype=float),
            color="#888888",
            s=20,
            alpha=0.72,
            zorder=2,
        )

    plot_values: list[float] = list(pivot.to_numpy(dtype=float).ravel())
    regime_colors = {"homogeneous": "#0072B2", "heteroscedastic": "#D55E00"}
    for position, regime in enumerate(REGIMES):
        interval = intervals[regime]
        mean = 100.0 * interval["mean"]
        low = 100.0 * interval["low"]
        high = 100.0 * interval["high"]
        threshold = 100.0 * float(config["gates"][gate_keys[regime]])
        status, status_color = _scientific_status(decision, (check_keys[regime],))
        ax.errorbar(
            position,
            mean,
            yerr=np.array([[mean - low], [high - mean]]),
            fmt="D",
            markersize=9,
            capsize=7,
            linewidth=2.1,
            color=regime_colors[regime],
            zorder=4,
        )
        ax.scatter(
            position,
            threshold,
            marker="_",
            s=700,
            linewidths=3,
            color="#111111",
            zorder=3,
        )
        ax.text(
            position,
            high,
            f"  moy. {mean:.2f} %\n  IC95 bas {low:.2f} %\n" f"  {status}",
            ha="left",
            va="bottom",
            fontsize=9,
            color=status_color,
        )
        plot_values.extend((mean, low, high, threshold))

    lower, upper = _padded_limits(plot_values, include_zero=True)
    ax.set_ylim(lower, upper + 0.08 * (upper - lower))
    ax.set_xticks(x, [REGIME_LABELS[value] for value in REGIMES])
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
    ax.set_ylabel("Q — gain MSE relatif K4c-CH vs K4 (%)")
    ax.set_title("Q sous bruit homogène et hétéroscédastique")
    handles = [
        Line2D(
            [0],
            [0],
            color="#999999",
            marker="o",
            label="Même seed reliée",
        ),
        Line2D(
            [0],
            [0],
            color="#333333",
            marker="D",
            label="Moyenne et IC95 Student",
        ),
        Line2D(
            [0],
            [0],
            color="#111111",
            marker="_",
            markersize=14,
            linestyle="None",
            label="Seuil préenregistré de borne basse",
        ),
    ]
    ax.legend(handles=handles, loc="best", frameon=False)
    fig.text(
        0.5,
        0.02,
        "Chaque segment relie les deux ratios de la même seed externe. Le gate "
        "préenregistré exige que la borne basse de chaque IC95 soit strictement "
        "supérieure à son seuil.",
        ha="center",
        fontsize=8.7,
    )
    fig.tight_layout(rect=(0.03, 0.09, 0.99, 0.98))
    path = output_dir / "03_q_homogene_vs_heteroscedastique.png"
    _save_figure(fig, path)
    _atomic_csv(values, output_dir / "03_q_noise_regime_seed_values.csv")
    return path


def _gate_rows(decision: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    classification = config["statistical_analysis"]["gate_classification"]
    validity_pass = decision.get("validity_pass")
    scientific_pass = decision.get("scientific_checks_pass")
    _require_json_bool(validity_pass, name="decision.validity_pass")
    _require_json_bool(
        scientific_pass,
        name="decision.scientific_checks_pass",
        allow_none=True,
    )
    science_evaluated = scientific_pass is not None
    groups = (
        ("Validité", classification["validity"], decision["validity_checks"]),
        (
            "Scientifique",
            classification["scientific"],
            decision["scientific_checks"],
        ),
    )
    rows: list[dict[str, Any]] = []
    for group, ordered_keys, reported in groups:
        if set(ordered_keys) != set(reported):
            raise RuntimeError(f"{group}: gate registry differs from the config")
        for key in ordered_keys:
            value = reported[key]
            _require_json_bool(value, name=f"decision.{group}.{key}")
            if group == "Scientifique" and not science_evaluated:
                status = "NON ÉVALUÉ"
            else:
                status = "PASS" if value is True else "FAIL"
            rows.append(
                {
                    "gate_class": group,
                    "gate": key,
                    "label": GATE_LABELS.get(
                        key, str(key).replace("_", " ").capitalize()
                    ),
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def plot_gate_summary(
    decision: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: Path,
) -> Path:
    """Render every preregistered validity and scientific gate."""

    rows = _gate_rows(decision, config)
    groups = ("Validité", "Scientifique")
    heights = [len(rows.loc[rows["gate_class"].eq(group)]) for group in groups]
    fig_height = max(8.0, 0.43 * max(heights) + 2.6)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, fig_height))
    for ax, group in zip(axes, groups, strict=True):
        data = rows.loc[rows["gate_class"].eq(group)].reset_index(drop=True)
        y = np.arange(len(data), dtype=float)[::-1]
        for position, row in zip(y, data.to_dict("records"), strict=True):
            status = str(row["status"])
            color = {
                "PASS": PASS_COLOR,
                "FAIL": FAIL_COLOR,
                "NON ÉVALUÉ": "#777777",
            }[status]
            label = textwrap.fill(str(row["label"]), width=49)
            ax.scatter(0.03, position, s=90, color=color, marker="o")
            ax.text(0.08, position, label, va="center", ha="left", fontsize=9.2)
            ax.text(
                0.98,
                position,
                status,
                va="center",
                ha="right",
                fontsize=9.2,
                fontweight="bold",
                color=color,
            )
        passed_count = int(data["status"].eq("PASS").sum())
        unevaluated_count = int(data["status"].eq("NON ÉVALUÉ").sum())
        heading = "Gates de validité" if group == "Validité" else "Gates scientifiques"
        suffix = f" · {unevaluated_count} non évalués" if unevaluated_count else ""
        ax.set_title(f"{heading} — {passed_count}/{len(data)} PASS{suffix}")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.8, len(data) - 0.2)
        ax.axis("off")

    validity_pass = decision["validity_pass"]
    science_value = decision.get("scientific_checks_pass")
    validity_text = "PASS" if validity_pass is True else "FAIL"
    science_text = (
        "NON ÉVALUÉE"
        if science_value is None
        else "PASS" if science_value is True else "FAIL"
    )
    fig.suptitle(
        "G0g-K4c-CH — synthèse de tous les gates préenregistrés",
        fontsize=15,
        y=0.985,
    )
    fig.text(
        0.5,
        0.045,
        f"Décision : {decision['decision']} · validité={validity_text} · "
        f"science={science_text} · holdout fermé.",
        ha="center",
        fontsize=9.4,
    )
    fig.text(
        0.5,
        0.018,
        "Vert : check satisfait. Rouge : check échoué. Gris : non évalué. "
        "Une défaillance de "
        "validité rend le screen invalide/inconclusif avant toute interprétation "
        "scientifique.",
        ha="center",
        fontsize=8.7,
    )
    fig.subplots_adjust(left=0.03, right=0.985, top=0.91, bottom=0.09, wspace=0.08)
    path = output_dir / "04_synthese_des_gates.png"
    _save_figure(fig, path)
    _atomic_csv(rows, output_dir / "04_gate_summary.csv")
    return path


def _generate_figures(sources: ValidatedSources, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_pqc_by_seed(
            sources.seed_summary, sources.decision, sources.config, output_dir
        ),
        plot_mse_by_noise_regime(
            sources.histories, sources.children, sources.config, output_dir
        ),
        plot_q_by_noise_regime(
            sources.seed_summary, sources.decision, sources.config, output_dir
        ),
        plot_gate_summary(sources.decision, sources.config, output_dir),
    ]


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _ensure_separate_output(results_dir: Path, output_dir: Path) -> None:
    if output_dir == results_dir or output_dir.is_relative_to(results_dir):
        raise RuntimeError("Figure output must stay outside the completed result tree")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = _resolve(args.results_dir)
    config_path = _resolve(args.config)
    output_dir = _resolve(args.output_dir)
    _ensure_separate_output(results_dir, output_dir)

    # Validation, including the manifest-first completion guard and the current
    # in-memory audit, finishes before the derived output directory is created.
    sources = _validate_sources(results_dir, config_path)
    _configure_style()
    paths = _generate_figures(sources, output_dir)
    print(
        "Validated completed K4c-CH development source: "
        f"status={sources.manifest['status']}, "
        f"device={sources.manifest['device']}, "
        f"seeds={len(sources.seed_summary)}, holdout_opened=false, "
        "independent_audit=passed"
    )
    for path in paths:
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
