#!/usr/bin/env python3
"""Create the fail-closed post-run analysis for G0g-K5-TP.

The script is intentionally read-only with respect to the scientific result
directory.  It refuses to read raw CSVs or create presentation artifacts until
the completed manifests, the registered decision, the independent post-run
audit, and the pre-evaluation supplemental audit all establish a valid screen.

The outer seed is the statistical unit.  Monte-Carlo children are averaged
within each frozen history, histories are then aggregated within each seed,
and only the twelve seed-level values enter confidence intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import tempfile
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

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "gaussian_aware_reference_g0g_k5_transcript_predictor_mps_v1"
DEFAULT_CONFIG = ROOT / (
    "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k5_transcript_predictor.yaml"
)
DEFAULT_RESULTS = ROOT / (
    "results/ldp_gradient_far/"
    "gaussian_aware_reference_g0g_k5_transcript_predictor_mps_v1"
)
DEFAULT_REPORT = ROOT / "output/analysis/Gaussian_Aware_G0g_K5_TP_Report.md"
DEFAULT_FIGURES = ROOT / "output/figures/gaussian_aware_g0g_k5"

K2 = "g0g_k2"
K4 = "g0g_k4_temporal_causal_gate"
K4B = "g0g_k4b_rolling_past_imputation"
K5_1D = "g0g_k5_tp_one_dimensional_control"
K5 = "g0g_k5_tp_shared_scalar_ridge"
K4C = "g0g_k4c_ch_privileged_benchmark"
POINTWISE = "g0g_k4b_pointwise_optimal_oracle"
CANDIDATES = (K2, K4, K4B, K5_1D, K5, K4C, POINTWISE)

CANDIDATE_LABELS = {
    K2: "K2 (courant)",
    K4: "K4 (gate temporelle)",
    K4B: "K4b (passé roulant)",
    K5_1D: "K5-1D (contrôle)",
    K5: "K5-TP (transcript)",
    K4C: "K4c-CH (semi-oracle)",
    POINTWISE: "Pointwise (oracle)",
}

COLORS = {
    K2: "#999999",
    K4: "#0072B2",
    K4B: "#E69F00",
    K5_1D: "#56B4E9",
    K5: "#009E73",
    K4C: "#CC79A7",
    POINTWISE: "#3B3B3B",
}

CONTRASTS = (
    {
        "key": "gain_vs_k4b",
        "label": "Gain K5 vs K4b",
        "mean_gate": "primary_gain_vs_k4b_mean_min",
        "low_gate": "primary_gain_vs_k4b_ci95_low_strictly_greater_than",
        "checks": ("gain_vs_k4b_mean", "gain_vs_k4b_ci"),
    },
    {
        "key": "gain_vs_k4",
        "label": "Gain K5 vs K4",
        "mean_gate": "primary_gain_vs_k4_mean_min",
        "low_gate": "primary_gain_vs_k4_ci95_low_strictly_greater_than",
        "checks": ("gain_vs_k4_mean", "gain_vs_k4_ci"),
    },
    {
        "key": "gain_vs_one_dimensional",
        "label": "Gain K5 vs K5-1D",
        "mean_gate": "primary_gain_vs_one_dimensional_mean_min",
        "low_gate": ("primary_gain_vs_one_dimensional_ci95_low_strictly_greater_than"),
        "checks": (
            "gain_vs_one_dimensional_mean",
            "gain_vs_one_dimensional_ci",
        ),
    },
    {
        "key": "ch_capture_fraction",
        "label": "Headroom K4c capturé",
        "mean_gate": "ch_capture_fraction_mean_min",
        "low_gate": "ch_capture_fraction_ci95_low_strictly_greater_than",
        "checks": ("capture_mean", "capture_ci"),
    },
    {
        "key": "homogeneous_gain_vs_k4b",
        "label": "Gain vs K4b — homogène",
        "mean_gate": None,
        "low_gate": "homogeneous_gain_vs_k4b_ci95_low_strictly_greater_than",
        "checks": ("homogeneous_gain_ci",),
    },
    {
        "key": "heteroscedastic_gain_vs_k4b",
        "label": "Gain vs K4b — hétéroscéd.",
        "mean_gate": None,
        "low_gate": ("heteroscedastic_gain_vs_k4b_ci95_low_strictly_greater_than"),
        "checks": ("heteroscedastic_gain_ci",),
    },
)

MSE_COLUMNS = {
    K4: "k4_integrated_mse",
    K4B: "k4b_integrated_mse",
    K5_1D: "one_dimensional_integrated_mse",
    K5: "k5_integrated_mse",
    K4C: "k4c_integrated_mse",
    POINTWISE: "pointwise_integrated_mse",
}

FIGURE_NAMES = (
    "01_contrastes_ic95_et_seuils.pdf",
    "02_mse_integree_par_methode.pdf",
    "03_mse_homogene_vs_heteroscedastique.pdf",
    "04_gains_k5_vs_k4b_par_cellule.pdf",
)


@dataclass(frozen=True)
class ValidatedSources:
    """Artifacts accepted by the post-run fail-closed barrier."""

    config: dict[str, Any]
    root_manifest: dict[str, Any]
    evaluation_manifest: dict[str, Any]
    decision: dict[str, Any]
    official_audit: dict[str, Any]
    supplemental_audit: dict[str, Any]
    predictor: dict[str, Any]
    design: dict[str, Any]
    histories: pd.DataFrame
    children: pd.DataFrame
    seed_summary: pd.DataFrame


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _json_bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON Boolean")
    return bool(value)


def _mapping_all_true(value: Any, *, name: str) -> bool:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty JSON object")
    return all(_json_bool(item, name=f"{name}.{key}") for key, item in value.items())


def _guard_metadata(results: Path, config_path: Path) -> dict[str, dict[str, Any]]:
    """Validate completion and audit metadata before opening any raw CSV."""

    root_manifest_path = results / "manifest.json"
    if not root_manifest_path.is_file():
        raise FileNotFoundError(f"Missing K5 root manifest: {root_manifest_path}")
    root_manifest = _read_json(root_manifest_path)
    if (
        root_manifest.get("campaign_id") != CAMPAIGN_ID
        or root_manifest.get("status") != "completed_development"
    ):
        raise RuntimeError(
            "K5 is not a completed development screen; raw results were not opened"
        )

    paths = {
        "evaluation_manifest": results / "evaluation/manifest.json",
        "decision": results / "evaluation/decision.json",
        "official_audit": results / "evaluation/independent_postrun_audit.json",
        "supplemental_audit": results / "pre_evaluation_supplemental_audit.json",
        "predictor": results / "frozen_predictor.json",
        "design": results / "feature_design_diagnostics.json",
        "config": config_path,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Completed K5 metadata is incomplete; raw results were not opened: "
            + ", ".join(missing)
        )

    metadata = {
        name: _read_json(path) for name, path in paths.items() if name != "config"
    }
    evaluation_manifest = metadata["evaluation_manifest"]
    decision = metadata["decision"]
    official = metadata["official_audit"]
    supplement = metadata["supplemental_audit"]
    predictor = metadata["predictor"]

    if (
        evaluation_manifest.get("status") != "completed_development"
        or evaluation_manifest.get("device") != "mps"
        or evaluation_manifest.get("holdout_opened") is not False
        or root_manifest.get("device") != "mps"
        or root_manifest.get("holdout_opened") is not False
    ):
        raise RuntimeError("K5 completion/device/holdout metadata is invalid")
    if decision.get("validity_pass") is not True:
        raise RuntimeError("K5 scientific result is invalid or inconclusive")
    if not _mapping_all_true(
        decision.get("validity_checks"), name="decision.validity_checks"
    ):
        raise RuntimeError("K5 validity checks did not all pass")
    if not isinstance(decision.get("scientific_checks"), Mapping):
        raise RuntimeError("K5 scientific check registry is missing")
    for key, value in decision["scientific_checks"].items():
        _json_bool(value, name=f"decision.scientific_checks.{key}")
    science_pass = all(decision["scientific_checks"].values())
    if decision.get("scientific_checks_pass") is not science_pass:
        raise RuntimeError("K5 scientific decision is internally inconsistent")
    expected_decision = (
        "authorize_end_to_end_development_screen"
        if science_pass
        else "stop_this_linear_transcript_predictor_instance"
    )
    if (
        decision.get("decision") != expected_decision
        or decision.get("all_gates_pass") is not science_pass
        or evaluation_manifest.get("decision") != expected_decision
        or evaluation_manifest.get("all_gates_pass") is not science_pass
        or root_manifest.get("evaluation_decision") != expected_decision
        or root_manifest.get("all_gates_pass") is not science_pass
    ):
        raise RuntimeError("K5 registered decision differs from its gate values")

    if (
        official.get("all_checks_pass") is not True
        or official.get("audit_device") != "mps"
        or official.get("holdout_opened") is not False
        or not isinstance(official.get("fit_audit"), Mapping)
        or official["fit_audit"].get("all_checks_pass") is not True
        or not _mapping_all_true(
            official["fit_audit"].get("checks"),
            name="official_audit.fit_audit.checks",
        )
        or not isinstance(official.get("evaluation_audit"), Mapping)
        or official["evaluation_audit"].get("all_checks_pass") is not True
        or not _mapping_all_true(
            official["evaluation_audit"].get("validity_checks"),
            name="official_audit.evaluation_audit.validity_checks",
        )
        or official["evaluation_audit"].get("decision_recomputed") != expected_decision
        or not isinstance(
            official["evaluation_audit"].get("decision_differences"), Mapping
        )
        or not official["evaluation_audit"]["decision_differences"]
        or any(official["evaluation_audit"]["decision_differences"].values())
    ):
        raise RuntimeError("The official independent K5 audit is not valid")
    if (
        supplement.get("pass") is not True
        or supplement.get("violations") != []
        or not _mapping_all_true(
            supplement.get("checks"), name="supplemental_audit.checks"
        )
    ):
        raise RuntimeError("The pre-evaluation supplemental K5 audit is not valid")

    predictor_hash = _sha256(paths["predictor"])
    if (
        predictor.get("campaign_id") != CAMPAIGN_ID
        or predictor.get("observable_past_only_at_inference") is not True
        or predictor.get("holdout_opened") is not False
        or root_manifest.get("frozen_predictor_sha256") != predictor_hash
        or evaluation_manifest.get("frozen_predictor_sha256") != predictor_hash
    ):
        raise RuntimeError("The frozen K5 predictor provenance is invalid")
    if root_manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("The K5 config differs from the completed manifest")
    metadata["root_manifest"] = root_manifest
    return metadata


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


def _student_ci(values: Sequence[float], t_critical: float) -> dict[str, float | int]:
    finite = [float(value) for value in values]
    if len(finite) < 2 or not all(math.isfinite(value) for value in finite):
        raise ValueError("A Student interval needs at least two finite seed values")
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


def _scientific_checks_from_intervals(
    config: Mapping[str, Any], intervals: Mapping[str, Mapping[str, float | int]]
) -> dict[str, bool]:
    """Reapply the exact preregistered scientific inequalities."""

    gates = config["gates"]
    return {
        "gain_vs_k4b_mean": float(intervals["gain_vs_k4b"]["mean"])
        >= float(gates["primary_gain_vs_k4b_mean_min"]),
        "gain_vs_k4b_ci": float(intervals["gain_vs_k4b"]["low"])
        > float(gates["primary_gain_vs_k4b_ci95_low_strictly_greater_than"]),
        "gain_vs_k4_mean": float(intervals["gain_vs_k4"]["mean"])
        >= float(gates["primary_gain_vs_k4_mean_min"]),
        "gain_vs_k4_ci": float(intervals["gain_vs_k4"]["low"])
        > float(gates["primary_gain_vs_k4_ci95_low_strictly_greater_than"]),
        "gain_vs_one_dimensional_mean": float(
            intervals["gain_vs_one_dimensional"]["mean"]
        )
        >= float(gates["primary_gain_vs_one_dimensional_mean_min"]),
        "gain_vs_one_dimensional_ci": float(intervals["gain_vs_one_dimensional"]["low"])
        > float(
            gates["primary_gain_vs_one_dimensional_ci95_low_strictly_greater_than"]
        ),
        "capture_mean": float(intervals["ch_capture_fraction"]["mean"])
        >= float(gates["ch_capture_fraction_mean_min"]),
        "capture_ci": float(intervals["ch_capture_fraction"]["low"])
        > float(gates["ch_capture_fraction_ci95_low_strictly_greater_than"]),
        "homogeneous_gain_ci": float(intervals["homogeneous_gain_vs_k4b"]["low"])
        > float(gates["homogeneous_gain_vs_k4b_ci95_low_strictly_greater_than"]),
        "heteroscedastic_gain_ci": float(
            intervals["heteroscedastic_gain_vs_k4b"]["low"]
        )
        > float(gates["heteroscedastic_gain_vs_k4b_ci95_low_strictly_greater_than"]),
    }


def _history_candidate_means(children: pd.DataFrame) -> pd.DataFrame:
    axes = [
        "history_id",
        "seed",
        "noise_regime",
        "noise_permutation",
        "outlier_geometry",
        "honest_dynamics",
        "threat",
        "assessment_round",
        "candidate",
    ]
    return (
        children.groupby(axes, observed=True, as_index=False)["squared_reference_error"]
        .mean()
        .rename(columns={"squared_reference_error": "history_mse"})
    )


def _integrated_seed_mse(history_means: pd.DataFrame) -> pd.DataFrame:
    return (
        history_means.groupby(["seed", "candidate"], observed=True, as_index=False)[
            "history_mse"
        ]
        .sum()
        .rename(columns={"history_mse": "integrated_mse"})
    )


def _recompute_seed_summary(integrated: pd.DataFrame) -> pd.DataFrame:
    pivot = integrated.pivot(index="seed", columns="candidate", values="integrated_mse")
    missing = sorted(set(CANDIDATES) - set(pivot.columns))
    if missing:
        raise RuntimeError(f"Missing candidates in integrated MSE: {missing}")
    denominator = pivot[K4] - pivot[K4C]
    if bool((denominator <= 0.0).any()):
        raise RuntimeError("Non-positive K4-to-K4c headroom denominator")
    result = pd.DataFrame(index=pivot.index)
    for candidate, column in MSE_COLUMNS.items():
        result[column] = pivot[candidate]
    result["gain_vs_k4b"] = (pivot[K4B] - pivot[K5]) / pivot[K4B]
    result["gain_vs_k4"] = (pivot[K4] - pivot[K5]) / pivot[K4]
    result["gain_vs_one_dimensional"] = (pivot[K5_1D] - pivot[K5]) / pivot[K5_1D]
    result["ch_capture_fraction"] = (pivot[K4] - pivot[K5]) / denominator
    return result.reset_index()


def _validate_sources(results: Path, config_path: Path) -> ValidatedSources:
    """Validate all sources and independently reproduce reported summaries."""

    metadata = _guard_metadata(results, config_path)
    raw_paths = {
        "histories": results / "evaluation/history_rows.csv",
        "children": results / "evaluation/evaluation_child_rows.csv",
        "seed_summary": results / "evaluation/seed_summary.csv",
        "replace_one": results / "evaluation/replace_one_audit.csv",
    }
    missing = [str(path) for path in raw_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Completed K5 raw results are incomplete: " + ", ".join(missing)
        )
    snapshot_paths = {
        **raw_paths,
        "root_manifest": results / "manifest.json",
        "evaluation_manifest": results / "evaluation/manifest.json",
        "decision": results / "evaluation/decision.json",
        "official_audit": results / "evaluation/independent_postrun_audit.json",
        "supplemental_audit": results / "pre_evaluation_supplemental_audit.json",
        "predictor": results / "frozen_predictor.json",
        "design": results / "feature_design_diagnostics.json",
        "config": config_path,
    }
    hashes_before = {name: _sha256(path) for name, path in snapshot_paths.items()}
    config_value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(config_value, dict)
        or config_value.get("campaign_id") != CAMPAIGN_ID
    ):
        raise RuntimeError("Unexpected K5 configuration")
    config: dict[str, Any] = config_value
    histories = pd.read_csv(raw_paths["histories"], low_memory=False)
    children = pd.read_csv(raw_paths["children"], low_memory=False)
    seed_summary = pd.read_csv(raw_paths["seed_summary"], low_memory=False)
    hashes_after = {name: _sha256(path) for name, path in snapshot_paths.items()}
    changed = sorted(
        name for name in snapshot_paths if hashes_before[name] != hashes_after[name]
    )
    if changed:
        raise RuntimeError(
            "K5 sources changed during validation: " + ", ".join(changed)
        )

    history_columns = {
        "history_id",
        "seed",
        "noise_regime",
        "noise_permutation",
        "outlier_geometry",
        "honest_dynamics",
        "threat",
        "assessment_round",
        "feature_max_source_round",
    }
    child_columns = {
        "history_id",
        "seed",
        "noise_regime",
        "noise_permutation",
        "outlier_geometry",
        "honest_dynamics",
        "threat",
        "assessment_round",
        "evaluation_child",
        "candidate",
        "squared_reference_error",
    }
    seed_columns = {"seed", *MSE_COLUMNS.values(), *(item["key"] for item in CONTRASTS)}
    _require_columns(histories, history_columns, source="history_rows.csv")
    _require_columns(children, child_columns, source="evaluation_child_rows.csv")
    _require_columns(seed_summary, seed_columns, source="seed_summary.csv")
    _numeric_finite(
        histories,
        ["seed", "assessment_round", "feature_max_source_round"],
        source="history_rows.csv",
    )
    _numeric_finite(
        children,
        ["seed", "assessment_round", "evaluation_child", "squared_reference_error"],
        source="evaluation_child_rows.csv",
    )
    _numeric_finite(
        seed_summary,
        ["seed", *MSE_COLUMNS.values(), *(item["key"] for item in CONTRASTS)],
        source="seed_summary.csv",
    )
    if bool((children["squared_reference_error"] < 0.0).any()):
        raise ValueError("Squared reference error cannot be negative")
    if bool(histories.duplicated(["history_id"]).any()):
        raise RuntimeError("Duplicate K5 history identifiers")
    if bool(children.duplicated(["history_id", "evaluation_child", "candidate"]).any()):
        raise RuntimeError("Duplicate K5 child/candidate rows")
    if set(children["history_id"]) != set(histories["history_id"]):
        raise RuntimeError("History identifiers differ between K5 CSVs")
    if set(children["candidate"].astype(str)) != set(CANDIDATES):
        raise RuntimeError("K5 candidate registry is incomplete or changed")

    expected_seeds = {
        int(value) for value in config["randomness"]["evaluation_outer_seeds"]
    }
    for name, frame in (
        ("histories", histories),
        ("children", children),
        ("seed_summary", seed_summary),
    ):
        observed = {int(value) for value in frame["seed"].unique()}
        if observed != expected_seeds:
            raise RuntimeError(f"{name}: unexpected outer-seed registry")
    if len(seed_summary) != len(expected_seeds) or bool(
        seed_summary.duplicated("seed").any()
    ):
        raise RuntimeError("K5 seed summary must contain one row per outer seed")
    expected_histories = int(config["gates"]["evaluation_histories_exact"])
    expected_children = int(config["nested_monte_carlo"]["evaluation_children"])
    if len(histories) != expected_histories:
        raise RuntimeError("K5 history count differs from preregistration")
    counts = children.groupby(["history_id", "candidate"], observed=True).size()
    if not bool((counts == expected_children).all()):
        raise RuntimeError("K5 child count differs across histories/candidates")

    history_means = _history_candidate_means(children)
    recomputed = _recompute_seed_summary(_integrated_seed_mse(history_means))
    stored = seed_summary.set_index("seed").sort_index()
    recomputed = recomputed.set_index("seed").sort_index()
    comparable = [
        *MSE_COLUMNS.values(),
        "gain_vs_k4b",
        "gain_vs_k4",
        "gain_vs_one_dimensional",
        "ch_capture_fraction",
    ]
    maximum_difference = max(
        float(
            np.max(
                np.abs(
                    stored[column].to_numpy(float) - recomputed[column].to_numpy(float)
                )
            )
        )
        for column in comparable
    )
    if maximum_difference > 2.0e-6:
        raise RuntimeError(
            "K5 seed summary does not reproduce from child-level results"
        )

    # Regime contrasts are recomputed from raw rows before comparison with the
    # stored seed summary and registered confidence intervals.
    regime = (
        history_means.groupby(["seed", "noise_regime", "candidate"], observed=True)[
            "history_mse"
        ]
        .sum()
        .unstack("candidate")
    )
    for regime_name, column in (
        ("homogeneous", "homogeneous_gain_vs_k4b"),
        ("heteroscedastic", "heteroscedastic_gain_vs_k4b"),
    ):
        ratios = (
            regime.loc[(slice(None), regime_name), K4B]
            - regime.loc[(slice(None), regime_name), K5]
        ) / regime.loc[(slice(None), regime_name), K4B]
        ratios.index = ratios.index.droplevel("noise_regime")
        difference = np.max(
            np.abs(
                stored[column].sort_index().to_numpy(float)
                - ratios.sort_index().to_numpy(float)
            )
        )
        if float(difference) > 2.0e-6:
            raise RuntimeError(f"K5 {regime_name} contrast does not reproduce")

    t_critical = float(config["statistical_analysis"]["t_critical_df11"])
    reported_cis = metadata["decision"].get("confidence_intervals")
    if not isinstance(reported_cis, Mapping):
        raise RuntimeError("K5 confidence-interval registry is absent")
    recomputed_cis: dict[str, Mapping[str, float | int]] = {}
    for spec in CONTRASTS:
        key = str(spec["key"])
        computed = _student_ci(stored[key].tolist(), t_critical)
        recomputed_cis[key] = computed
        reported = reported_cis.get(key)
        if not isinstance(reported, Mapping) or int(reported.get("n", -1)) != int(
            computed["n"]
        ):
            raise RuntimeError(f"K5 CI registry is malformed for {key}")
        for field in ("mean", "low", "high"):
            if not math.isclose(
                float(reported[field]),
                float(computed[field]),
                rel_tol=0.0,
                abs_tol=2.0e-6,
            ):
                raise RuntimeError(f"K5 CI does not reproduce for {key}.{field}")
    expected_science = _scientific_checks_from_intervals(config, recomputed_cis)
    if metadata["decision"].get("scientific_checks") != expected_science:
        raise RuntimeError(
            "K5 scientific checks do not reproduce from the preregistered gates"
        )

    return ValidatedSources(
        config=config,
        root_manifest=metadata["root_manifest"],
        evaluation_manifest=metadata["evaluation_manifest"],
        decision=metadata["decision"],
        official_audit=metadata["official_audit"],
        supplemental_audit=metadata["supplemental_audit"],
        predictor=metadata["predictor"],
        design=metadata["design"],
        histories=histories,
        children=children,
        seed_summary=seed_summary,
    )


def _mse_summary(sources: ValidatedSources) -> tuple[pd.DataFrame, pd.DataFrame]:
    history_means = _history_candidate_means(sources.children)
    integrated = _integrated_seed_mse(history_means)
    t_critical = float(sources.config["statistical_analysis"]["t_critical_df11"])
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        values = integrated.loc[
            integrated["candidate"] == candidate, "integrated_mse"
        ].tolist()
        interval = _student_ci(values, t_critical)
        rows.append({"candidate": candidate, **interval})
    return history_means, pd.DataFrame(rows)


def _regime_summary(
    sources: ValidatedSources, history_means: pd.DataFrame
) -> pd.DataFrame:
    per_seed = (
        history_means.groupby(["seed", "noise_regime", "candidate"], observed=True)[
            "history_mse"
        ]
        .mean()
        .reset_index()
    )
    t_critical = float(sources.config["statistical_analysis"]["t_critical_df11"])
    rows: list[dict[str, Any]] = []
    for (regime, candidate), group in per_seed.groupby(
        ["noise_regime", "candidate"], observed=True
    ):
        rows.append(
            {
                "noise_regime": str(regime),
                "candidate": str(candidate),
                **_student_ci(group["history_mse"].tolist(), t_critical),
            }
        )
    return pd.DataFrame(rows)


def _cell_gain_table(history_means: pd.DataFrame) -> pd.DataFrame:
    axes = [
        "seed",
        "noise_regime",
        "noise_permutation",
        "outlier_geometry",
        "honest_dynamics",
        "threat",
        "assessment_round",
    ]
    pivot = history_means.pivot(index=axes, columns="candidate", values="history_mse")
    gain = ((pivot[K4B] - pivot[K5]) / pivot[K4B]).rename("gain_vs_k4b")
    return gain.reset_index()


def _contrast_table(sources: ValidatedSources) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    gates = sources.config["gates"]
    intervals = sources.decision["confidence_intervals"]
    checks = sources.decision["scientific_checks"]
    for spec in CONTRASTS:
        interval = intervals[spec["key"]]
        mean_threshold = (
            float(gates[spec["mean_gate"]]) if spec["mean_gate"] is not None else None
        )
        low_threshold = float(gates[spec["low_gate"]])
        rows.append(
            {
                "key": spec["key"],
                "label": spec["label"],
                "mean": float(interval["mean"]),
                "low": float(interval["low"]),
                "high": float(interval["high"]),
                "mean_threshold": mean_threshold,
                "low_threshold": low_threshold,
                "pass": all(bool(checks[name]) for name in spec["checks"]),
            }
        )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", metadata={"Creator": "FedLab G0g-K5-TP"})
    plt.close(fig)


def _plot_contrasts(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y = np.arange(len(frame))
    means = frame["mean"].to_numpy(float) * 100.0
    lows = frame["low"].to_numpy(float) * 100.0
    highs = frame["high"].to_numpy(float) * 100.0
    colors = ["#198754" if value else "#C23B22" for value in frame["pass"]]
    for index, color in enumerate(colors):
        ax.errorbar(
            means[index],
            y[index],
            xerr=np.asarray(
                [[means[index] - lows[index]], [highs[index] - means[index]]]
            ),
            fmt="none",
            ecolor=color,
            elinewidth=2.0,
            capsize=4,
            zorder=2,
        )
    ax.scatter(means, y, c=colors, s=55, zorder=3)
    for index, row in frame.iterrows():
        if row["mean_threshold"] is not None:
            ax.scatter(
                float(row["mean_threshold"]) * 100.0,
                index,
                marker="D",
                s=36,
                facecolors="white",
                edgecolors="#6F4E37",
                zorder=4,
            )
        ax.scatter(
            float(row["low_threshold"]) * 100.0,
            index,
            marker="|",
            s=220,
            linewidths=2.1,
            color="#6F4E37",
            zorder=4,
        )
        ax.text(
            highs[index] + 1.0,
            index,
            "PASS" if bool(row["pass"]) else "FAIL",
            va="center",
            color=colors[index],
            fontweight="bold",
            fontsize=9,
        )
    ax.axvline(0.0, color="#555555", linewidth=1.0)
    ax.set_yticks(y, frame["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Gain relatif / fraction capturée (%)")
    ax.set_title("G0g-K5-TP : contrastes seed-level, IC95 Student et seuils")
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#198754",
            label="Estimation et IC95 (PASS)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#C23B22",
            label="Estimation et IC95 (FAIL)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markeredgecolor="#6F4E37",
            label="Seuil sur la moyenne",
        ),
        Line2D(
            [0],
            [0],
            marker="|",
            color="#6F4E37",
            linestyle="none",
            markersize=13,
            label="Seuil sur la borne basse",
        ),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=True)
    _save_figure(fig, path)


def _plot_mse(summary: pd.DataFrame, path: Path) -> None:
    ordered = summary.set_index("candidate").loc[list(CANDIDATES)].reset_index()
    fig, ax = plt.subplots(figsize=(9.1, 4.8))
    x = np.arange(len(ordered))
    means = ordered["mean"].to_numpy(float)
    lows = ordered["low"].to_numpy(float)
    highs = ordered["high"].to_numpy(float)
    bars = ax.bar(
        x,
        means,
        color=[COLORS[value] for value in ordered["candidate"]],
        edgecolor="white",
        linewidth=0.8,
        yerr=np.vstack((means - lows, highs - means)),
        capsize=4,
    )
    ax.set_xticks(
        x,
        [CANDIDATE_LABELS[value] for value in ordered["candidate"]],
        rotation=24,
        ha="right",
    )
    ax.set_ylabel("MSE intégrée (somme sur les 48 histoires)")
    ax.set_title("Erreur quadratique de référence par méthode — moyenne et IC95")
    ax.set_ylim(bottom=0.0)
    for bar, value in zip(bars, means, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.4g}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    _save_figure(fig, path)


def _plot_regimes(summary: pd.DataFrame, path: Path) -> None:
    candidates = (K4, K4B, K5_1D, K5, K4C, POINTWISE)
    regimes = ("homogeneous", "heteroscedastic")
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    x = np.arange(len(candidates))
    width = 0.36
    regime_style = {
        "homogeneous": ("Bruit homogène", "#56B4E9"),
        "heteroscedastic": ("Bruit hétéroscédastique", "#D55E00"),
    }
    for offset, regime in enumerate(regimes):
        current = (
            summary[summary["noise_regime"] == regime]
            .set_index("candidate")
            .loc[list(candidates)]
        )
        means = current["mean"].to_numpy(float)
        lows = current["low"].to_numpy(float)
        highs = current["high"].to_numpy(float)
        label, color = regime_style[regime]
        ax.bar(
            x + (offset - 0.5) * width,
            means,
            width,
            label=label,
            color=color,
            alpha=0.84,
            yerr=np.vstack((means - lows, highs - means)),
            capsize=3,
        )
    ax.set_xticks(
        x, [CANDIDATE_LABELS[value] for value in candidates], rotation=22, ha="right"
    )
    ax.set_ylabel("MSE moyenne par histoire")
    ax.set_title("Régimes de bruit : comparaison à nombre d'histoires normalisé")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=True)
    _save_figure(fig, path)


def _cell_label(row: Mapping[str, Any]) -> str:
    noise = "Homo" if row["noise_regime"] == "homogeneous" else "Hétéro"
    permutation = "id" if row["noise_permutation"] == "identity" else "byz-high"
    geometry = "aligné" if row["outlier_geometry"] == "aligned" else "orthog."
    dynamics = "stat." if row["honest_dynamics"] == "stationary" else "drift"
    threat = "BF×10" if row["threat"] == "bitflip_x10" else "replacement"
    return f"{noise}/{permutation} | {geometry} | {dynamics} | {threat}"


def _plot_cell_gains(cell_gains: pd.DataFrame, path: Path) -> None:
    axes = [
        "noise_regime",
        "noise_permutation",
        "outlier_geometry",
        "honest_dynamics",
        "threat",
        "assessment_round",
    ]
    mean = cell_gains.groupby(axes, observed=True, as_index=False)["gain_vs_k4b"].mean()
    mean["cell"] = mean.apply(_cell_label, axis=1)
    matrix = mean.pivot(index="cell", columns="assessment_round", values="gain_vs_k4b")
    matrix = matrix.reindex(sorted(matrix.index))
    values = matrix.to_numpy(float) * 100.0
    limit = max(5.0, float(np.nanmax(np.abs(values))))
    fig, ax = plt.subplots(figsize=(8.2, 11.5))
    image = ax.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(
        np.arange(len(matrix.columns)), [f"t={int(value)}" for value in matrix.columns]
    )
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index, fontsize=8.2)
    ax.set_title("Gain descriptif K5-TP vs K4b par cellule (moyenne des seeds)")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                f"{values[row, column]:.1f}%",
                ha="center",
                va="center",
                fontsize=7.4,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Gain relatif de MSE (%) ; positif = K5 meilleur")
    ax.grid(False)
    _save_figure(fig, path)


def _fmt_number(value: float) -> str:
    return f"{value:.6g}".replace(".", ",")


def _fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f} %".replace(".", ",")


def _fmt_ci(row: Mapping[str, Any], *, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    suffix = " %" if percent else ""
    mean = f"{scale * float(row['mean']):.4g}".replace(".", ",")
    sd = f"{scale * float(row['sd']):.3g}".replace(".", ",")
    low = f"{scale * float(row['low']):.4g}".replace(".", ",")
    high = f"{scale * float(row['high']):.4g}".replace(".", ",")
    return f"{mean} ± {sd}{suffix} ; IC95 [{low} ; {high}]{suffix}"


def _diagnostic_rows(sources: ValidatedSources) -> list[tuple[str, Any, Any]]:
    train = sources.design["train"]
    refit = sources.design["train_plus_calibration"]
    final = sources.design["final_fit"]
    control = sources.design["final_1d_fit"]
    return [
        (
            "Rang du dictionnaire",
            train["flattened_design_rank"],
            refit["flattened_design_rank"],
        ),
        (
            "RMS centrée minimale",
            train["minimum_centered_rms"],
            refit["minimum_centered_rms"],
        ),
        ("Échelles au floor", train["floor_active_count"], refit["floor_active_count"]),
        (
            "Conditionnement $\\ell_\\infty$",
            "—",
            final["condition_number_regularized_system"],
        ),
        (
            "Résidu des équations normales",
            "—",
            final["normal_equation_relative_residual"],
        ),
        ("Conditionnement K5-1D", "—", control["condition_number_regularized_system"]),
        ("Résidu K5-1D", "—", control["normal_equation_relative_residual"]),
    ]


def _report_markdown(
    sources: ValidatedSources,
    contrasts: pd.DataFrame,
    mse: pd.DataFrame,
    regimes: pd.DataFrame,
    cell_gains: pd.DataFrame,
    report_path: Path,
    figures_path: Path,
) -> str:
    passed = bool(sources.decision["all_gates_pass"])
    decision_fr = (
        "PASS : autoriser un écran end-to-end de développement"
        if passed
        else "FAIL scientifique valide : arrêter cette instanciation linéaire"
    )
    relative_figures = [
        Path("../figures/gaussian_aware_g0g_k5") / name for name in FIGURE_NAMES
    ]
    feature_names = list(sources.predictor["feature_names"])
    coefficients = [float(value) for value in sources.predictor["coefficients"]]
    scales = [float(value) for value in sources.predictor["feature_scales"]]
    one = sources.predictor["one_dimensional_control"]
    mse_by_candidate = mse.set_index("candidate")
    regime_by_axes = regimes.set_index(["noise_regime", "candidate"])

    lines = [
        "# G0g-K5-TP — prédicteur strictement fondé sur le transcript passé",
        "",
        "## Résumé exécutif",
        "",
        f"**Décision enregistrée : {decision_fr}.**",
        "",
    ]
    primary = contrasts.loc[contrasts["key"] == "gain_vs_k4b"].iloc[0]
    capture = contrasts.loc[contrasts["key"] == "ch_capture_fraction"].iloc[0]
    lines.extend(
        [
            "Le résultat est un **screen synthétique de niveau référence**, "
            "pas une validation end-to-end de LDP-Gradient-FAR. Le gain principal "
            f"K5-TP vs K4b est de **{_fmt_pct(primary['mean'])}**, avec IC95 "
            f"**[{_fmt_pct(primary['low'])} ; {_fmt_pct(primary['high'])}]**. "
            "La fraction du headroom semi-oracle capturée est de "
            f"**{_fmt_pct(capture['mean'])}**, IC95 "
            f"**[{_fmt_pct(capture['low'])} ; {_fmt_pct(capture['high'])}]**.",
            "",
            "Les deux barrières indépendantes requises sont valides : l'audit "
            "supplémentaire pré-évaluation atteste le gel du prédicteur, et "
            "l'audit officiel post-run reproduit la décision depuis les artefacts "
            "bruts. Le holdout réservé demeure fermé.",
            "",
            f"![Contrastes et seuils]({relative_figures[0]})",
            "",
            "## 1. Question, objet et protocole",
            "",
            "K5-TP cherche à reconstruire, au tour $t$, une direction "
            "compensatoire en utilisant uniquement le transcript observable jusqu'au "
            "tour $t-1$. Les six vecteurs d'entrée sont transformés par des "
            "coefficients scalaires partagés entre les 64 coordonnées, puis projetés "
            "sur la boule publique de rayon $G=0{,}13$.",
            "",
            "La supervision de fit est privilégiée : les cibles synthétiques "
            "K4c-CH sont utilisées hors ligne sur 12 seeds d'entraînement et 8 de "
            "calibration. L'évaluation gelée utilise 12 nouvelles seeds, 48 "
            "histoires par seed et 64 enfants Monte-Carlo par histoire. L'unité "
            "statistique des IC95 est toujours la seed externe.",
            "",
            "| Élément | Valeur |",
            "|---|---:|",
            f"| Seeds externes d'évaluation | {len(sources.seed_summary)} |",
            f"| Histoires gelées | {len(sources.histories)} |",
            f"| Lignes enfant-candidat | {len(sources.children)} |",
            f"| Enfants par histoire et candidat | {sources.config['nested_monte_carlo']['evaluation_children']} |",
            "| Cellules de bruit | homogène/identity ; hétéroscédastique/identity ; hétéroscédastique/byzantine-high |",
            "| Calcul scientifique | MPS, `torch.float32`, sans fallback CPU |",
            "| Perte | $\\lVert \\widehat F-\\theta_t^\\star\\rVert_2^2$ |",
            "",
            "## 2. Prédicteur gelé : $\\lambda$, échelles et coefficients",
            "",
            f"La régularisation sélectionnée est $\\lambda={_fmt_number(float(sources.predictor['selected_lambda']))}$. "
            "Chaque coefficient $\\theta_j$ multiplie $V_j/e_j$, où $e_j$ est "
            "l'échelle RMS par coordonnée estimée sur train+calibration.",
            "",
            "| $j$ | Feature strictement passée $V_j$ | Échelle $e_j$ | Coefficient $\\theta_j$ | Coefficient effectif $\\theta_j/e_j$ |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for index, (name, scale, coefficient) in enumerate(
        zip(feature_names, scales, coefficients, strict=True), start=1
    ):
        lines.append(
            f"| {index} | `{name}` | {_fmt_number(scale)} | "
            f"{_fmt_number(coefficient)} | {_fmt_number(coefficient / scale)} |"
        )
    lines.extend(
        [
            "",
            "Le contrôle unidimensionnel utilise seulement "
            f"`{one['feature_name']}`, avec $\\lambda={_fmt_number(float(one['selected_lambda']))}$, "
            f"$e={_fmt_number(float(one['feature_scale']))}$ et coefficient "
            f"$\\theta={_fmt_number(float(one['coefficient']))}$.",
            "",
            "## 3. Diagnostics numériques du fit",
            "",
            "| Diagnostic | Train | Refit train+calibration |",
            "|---|---:|---:|",
        ]
    )
    for name, train, refit in _diagnostic_rows(sources):
        train_text = str(train) if train == "—" else _fmt_number(float(train))
        refit_text = str(refit) if refit == "—" else _fmt_number(float(refit))
        lines.append(f"| {name} | {train_text} | {refit_text} |")
    lines.extend(
        [
            "",
            "Ces diagnostics contrôlent l'identifiabilité numérique du dictionnaire "
            "et la qualité de la résolution ridge ; ils ne prouvent pas à eux seuls "
            "que les features sont scientifiquement suffisantes.",
            "",
            "## 4. Contrastes préenregistrés et règle de décision",
            "",
            "| Contraste seed-level | Moyenne ± écart-type ; IC95 | Seuil moyen | Seuil borne basse | Verdict |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in contrasts.iterrows():
        values = sources.seed_summary[str(row["key"])].tolist()
        interval = _student_ci(
            values, float(sources.config["statistical_analysis"]["t_critical_df11"])
        )
        mean_gate = (
            "—"
            if row["mean_threshold"] is None
            else _fmt_pct(float(row["mean_threshold"]))
        )
        lines.append(
            f"| {row['label']} | {_fmt_ci(interval, percent=True)} | {mean_gate} | "
            f"{_fmt_pct(float(row['low_threshold']))} | "
            f"**{'PASS' if bool(row['pass']) else 'FAIL'}** |"
        )
    lines.extend(
        [
            "",
            "La décision est une intersection : tous les critères scientifiques "
            "doivent passer après tous les contrôles de validité. Cette règle "
            "n'est pas une correction de multiplicité pour des affirmations "
            "individuelles ; c'est un gate conservateur de développement.",
            "",
            "## 5. MSE de référence par méthode",
            "",
            "La MSE intégrée somme les 48 MSE moyennes par histoire au sein de "
            "chaque seed. Le tableau rapporte ensuite moyenne, écart-type et IC95 "
            "sur les 12 seeds.",
            "",
            "| Méthode | MSE intégrée moyenne ± écart-type ; IC95 |",
            "|---|---:|",
        ]
    )
    for candidate in CANDIDATES:
        row = mse_by_candidate.loc[candidate]
        lines.append(f"| {CANDIDATE_LABELS[candidate]} | {_fmt_ci(row)} |")
    lines.extend(
        [
            "",
            f"![MSE par méthode]({relative_figures[1]})",
            "",
            "## 6. Bruit homogène et hétéroscédastique",
            "",
            "Pour comparer les deux régimes malgré leur nombre différent de "
            "cellules, cette section utilise la MSE **moyenne par histoire**, et non "
            "la somme intégrée. Le global préenregistré donne un poids de $1/3$ "
            "à l'homogène et de $2/3$ à l'hétéroscédastique ; les gates "
            "séparés empêchent ce mélange de masquer un échec de régime.",
            "",
            "| Méthode | Homogène | Hétéroscédastique |",
            "|---|---:|---:|",
        ]
    )
    for candidate in (K4, K4B, K5_1D, K5, K4C, POINTWISE):
        homogeneous = regime_by_axes.loc[("homogeneous", candidate)]
        hetero = regime_by_axes.loc[("heteroscedastic", candidate)]
        lines.append(
            f"| {CANDIDATE_LABELS[candidate]} | {_fmt_ci(homogeneous)} | "
            f"{_fmt_ci(hetero)} |"
        )
    lines.extend(
        [
            "",
            f"![MSE par régime]({relative_figures[2]})",
            "",
            "## 7. Hétérogénéité descriptive des gains",
            "",
            "La figure suivante rapporte, pour chaque combinaison préenregistrée "
            "et chaque snapshot, la moyenne inter-seeds du ratio "
            "$(\\mathrm{MSE}_{K4b}-\\mathrm{MSE}_{K5})/\\mathrm{MSE}_{K4b}$. "
            "Ces cellules sont descriptives : elles localisent les échecs ou gains, "
            "mais n'ajoutent pas 48 réplicats indépendants.",
            "",
            f"![Gains par cellule]({relative_figures[3]})",
            "",
        ]
    )
    grouped = (
        cell_gains.groupby(
            [
                "noise_regime",
                "noise_permutation",
                "outlier_geometry",
                "honest_dynamics",
                "threat",
                "assessment_round",
            ],
            observed=True,
        )["gain_vs_k4b"]
        .mean()
        .sort_values()
    )
    lines.extend(
        [
            f"Le gain descriptif moyen par cellule varie de **{_fmt_pct(float(grouped.iloc[0]))}** "
            f"à **{_fmt_pct(float(grouped.iloc[-1]))}**. "
            f"{int((grouped > 0.0).sum())}/{len(grouped)} cellules ont un gain moyen positif.",
            "",
            "## 8. Observations, inférences et non-identifiable",
            "",
            "### Observations",
            "",
            f"- Les manifests sont complets, les deux audits passent et la décision "
            f"valide est `{sources.decision['decision']}`.",
            f"- K5-TP {'franchit' if passed else 'ne franchit pas'} l'intersection des "
            "gates scientifiques préenregistrés.",
            "- Les coefficients, échelles et la régularisation reportés ci-dessus "
            "sont ceux du prédicteur gelé avant toute trajectoire d'évaluation.",
            "- Les IC95 ont 12 seeds externes comme unité ; les enfants Monte-Carlo "
            "ne sont pas utilisés comme pseudo-réplicats.",
            "",
            "### Inférences autorisées",
            "",
            (
                "- Le transcript passé contient assez de signal, pour ce générateur "
                "et ce prédicteur linéaire, pour justifier un écran end-to-end de "
                "développement."
                if passed
                else "- Cette combinaison exacte de six features et d'une ridge "
                "linéaire ne satisfait pas le niveau d'effet préenregistré ; cela "
                "justifie son arrêt, pas celui de toute la famille transcript-only."
            ),
            "- Le signe séparé dans les deux régimes indique si le résultat global "
            "vient d'un seul type de bruit ou se maintient dans les deux.",
            "",
            "### Non identifiable et limites",
            "",
            "- **Ancre semi-oracle** : l'ancre est une approximation exogène du centre "
            "commun avec erreur publique fixée. Sa construction end-to-end n'est pas "
            "testée ; les résultats sont conditionnels à cette ancre favorable.",
            "- **Cible surrogate** : la cible est la moyenne propre synthétique des "
            "updates honnêtes. Une baisse de MSE de référence n'implique pas "
            "automatiquement une meilleure accuracy, convergence, robustesse ou fairness.",
            "- **Capture non bornée** : $C=(M_{K4}-M_{K5})/(M_{K4}-M_{K4c})$ "
            "est un ratio, pas une probabilité. Il n'est pas tronqué à $[0,1]$ et "
            "peut donc être négatif ou supérieur à 1 si K5 dépasse les repères.",
            "- **Mélange et gates** : l'estimand global mélange une cellule homogène "
            "et deux hétéroscédastiques. Les gates par régime protègent le signe, "
            "mais ne prouvent rien pour une autre pondération de régimes ou une autre "
            "distribution de bruit.",
            "- **Menaces et identités** : seules Bit-Flip ×10 et model replacement "
            "sont simulées, avec identités persistantes et résistance Sybil supposée.",
            "- Aucun holdout n'est ouvert et aucune promotion algorithmique ou claim "
            "de confidentialité end-to-end n'est autorisé par ce screen.",
            "",
            "## 9. Décision pratique",
            "",
            (
                "La prochaine étape autorisée est un **écran end-to-end de "
                "développement** qui branche le prédicteur gelé dans l'agrégation FAR "
                "et mesure accuracy, Worst-20, gap, robustesse et coût sous local-DP. "
                "Ce n'est toujours pas une ouverture du holdout."
                if passed
                else "La décision préenregistrée est d'arrêter cette instanciation "
                "linéaire. Toute poursuite doit annoncer une nouvelle hypothèse et un "
                "nouveau protocole avant de regarder de nouvelles données."
            ),
            "",
            "## 10. Traçabilité",
            "",
            f"- Résultats : `{sources.root_manifest.get('campaign_id')}`",
            f"- SHA-256 du prédicteur gelé : `{sources.root_manifest.get('frozen_predictor_sha256')}`",
            "- Audit officiel : `evaluation/independent_postrun_audit.json`",
            "- Audit supplémentaire : `pre_evaluation_supplemental_audit.json`",
            f"- Rapport : `{report_path}`",
            f"- Figures : `{figures_path}`",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    *,
    results: Path,
    config_path: Path,
    report_path: Path,
    figures_path: Path,
) -> dict[str, Any]:
    """Validate sources, then atomically publish one report and four figures."""

    sources = _validate_sources(results.resolve(), config_path.resolve())
    _configure_style()
    history_means, mse = _mse_summary(sources)
    regimes = _regime_summary(sources, history_means)
    cell_gains = _cell_gain_table(history_means)
    contrasts = _contrast_table(sources)
    report_text = _report_markdown(
        sources,
        contrasts,
        mse,
        regimes,
        cell_gains,
        report_path.resolve(),
        figures_path.resolve(),
    )

    report_path = report_path.resolve()
    figures_path = figures_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    figures_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="g0g_k5_analysis_", dir=figures_path.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        generated = [temporary / name for name in FIGURE_NAMES]
        _plot_contrasts(contrasts, generated[0])
        _plot_mse(mse, generated[1])
        _plot_regimes(regimes, generated[2])
        _plot_cell_gains(cell_gains, generated[3])
        temporary_report = temporary / report_path.name
        temporary_report.write_text(report_text, encoding="utf-8")
        if not all(path.is_file() and path.stat().st_size > 0 for path in generated):
            raise RuntimeError("A K5 figure was not generated correctly")
        figures_path.mkdir(parents=True, exist_ok=True)
        for source in generated:
            source.replace(figures_path / source.name)
        temporary_report.replace(report_path)
    return {
        "decision": sources.decision["decision"],
        "all_gates_pass": bool(sources.decision["all_gates_pass"]),
        "report": str(report_path),
        "figures": [str(figures_path / name) for name in FIGURE_NAMES],
        "holdout_opened": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args(argv)
    result = analyze(
        results=args.results,
        config_path=args.config,
        report_path=args.report,
        figures_path=args.figures,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
