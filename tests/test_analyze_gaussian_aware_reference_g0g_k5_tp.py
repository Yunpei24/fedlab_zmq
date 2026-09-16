"""Contract tests for the unlocked G0g-K5-TP post-run analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from scripts import analyze_gaussian_aware_reference_g0g_k5_tp as analysis

SEEDS = (101, 103)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> dict[str, Any]:
    return {
        "campaign_id": analysis.CAMPAIGN_ID,
        "randomness": {"evaluation_outer_seeds": list(SEEDS)},
        "nested_monte_carlo": {"evaluation_children": 2},
        "statistical_analysis": {"t_critical_df11": 12.706204736432095},
        "gates": {
            "evaluation_histories_exact": 8,
            "primary_gain_vs_k4b_mean_min": 0.10,
            "primary_gain_vs_k4b_ci95_low_strictly_greater_than": 0.0,
            "primary_gain_vs_k4_mean_min": 0.30,
            "primary_gain_vs_k4_ci95_low_strictly_greater_than": 0.20,
            "primary_gain_vs_one_dimensional_mean_min": 0.05,
            ("primary_gain_vs_one_dimensional_ci95_low_strictly_greater_than"): 0.0,
            "ch_capture_fraction_mean_min": 0.40,
            "ch_capture_fraction_ci95_low_strictly_greater_than": 0.30,
            "homogeneous_gain_vs_k4b_ci95_low_strictly_greater_than": 0.0,
            "heteroscedastic_gain_vs_k4b_ci95_low_strictly_greater_than": 0.0,
        },
    }


def _raw_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    histories: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    base = {
        analysis.K2: 1.20,
        analysis.K4: 1.00,
        analysis.K4B: 0.90,
        analysis.K5_1D: 0.84,
        analysis.K5: 0.50,
        analysis.K4C: 0.40,
        analysis.POINTWISE: 0.30,
    }
    for seed_index, seed in enumerate(SEEDS):
        factor = 1.0 + 0.02 * seed_index
        for regime in ("homogeneous", "heteroscedastic"):
            for round_index in (17, 24):
                history_id = f"{seed}|{regime}|{round_index}"
                common = {
                    "history_id": history_id,
                    "seed": seed,
                    "noise_regime": regime,
                    "noise_permutation": "identity",
                    "outlier_geometry": "aligned",
                    "honest_dynamics": "stationary",
                    "threat": "bitflip_x10",
                    "assessment_round": round_index,
                }
                histories.append(
                    {**common, "feature_max_source_round": round_index - 1}
                )
                for child in range(2):
                    for candidate, value in base.items():
                        children.append(
                            {
                                **common,
                                "evaluation_child": child,
                                "candidate": candidate,
                                "squared_reference_error": (
                                    factor * value + child * 1.0e-5
                                ),
                            }
                        )
    return pd.DataFrame(histories), pd.DataFrame(children)


def _seed_summary(children: pd.DataFrame) -> pd.DataFrame:
    history = analysis._history_candidate_means(children)
    integrated = analysis._integrated_seed_mse(history)
    summary = analysis._recompute_seed_summary(integrated).set_index("seed")
    regime = (
        history.groupby(["seed", "noise_regime", "candidate"], observed=True)[
            "history_mse"
        ]
        .sum()
        .unstack("candidate")
    )
    for regime_name, column in (
        ("homogeneous", "homogeneous_gain_vs_k4b"),
        ("heteroscedastic", "heteroscedastic_gain_vs_k4b"),
    ):
        values = (
            regime.loc[(slice(None), regime_name), analysis.K4B]
            - regime.loc[(slice(None), regime_name), analysis.K5]
        ) / regime.loc[(slice(None), regime_name), analysis.K4B]
        values.index = values.index.droplevel("noise_regime")
        summary[column] = values
    return summary.reset_index()


def _decision(config: dict[str, Any], seeds: pd.DataFrame) -> dict[str, Any]:
    science = {
        "gain_vs_k4b_mean": True,
        "gain_vs_k4b_ci": True,
        "gain_vs_k4_mean": True,
        "gain_vs_k4_ci": True,
        "gain_vs_one_dimensional_mean": True,
        "gain_vs_one_dimensional_ci": True,
        "capture_mean": True,
        "capture_ci": True,
        "homogeneous_gain_ci": True,
        "heteroscedastic_gain_ci": True,
    }
    intervals = {}
    for spec in analysis.CONTRASTS:
        interval = analysis._student_ci(
            seeds[str(spec["key"])].tolist(),
            config["statistical_analysis"]["t_critical_df11"],
        )
        intervals[str(spec["key"])] = {
            key: interval[key] for key in ("n", "mean", "low", "high")
        }
    return {
        "validity_pass": True,
        "validity_checks": {"complete": True, "matrix_exact": True},
        "scientific_checks_pass": True,
        "scientific_checks": science,
        "all_gates_pass": True,
        "decision": "authorize_end_to_end_development_screen",
        "confidence_intervals": intervals,
    }


def _completed_fixture(tmp_path: Path) -> tuple[Path, Path]:
    results = tmp_path / "results"
    evaluation = results / "evaluation"
    evaluation.mkdir(parents=True)
    config = _config()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    histories, children = _raw_frames()
    seeds = _seed_summary(children)
    histories.to_csv(evaluation / "history_rows.csv", index=False)
    children.to_csv(evaluation / "evaluation_child_rows.csv", index=False)
    seeds.to_csv(evaluation / "seed_summary.csv", index=False)
    pd.DataFrame([{"observed_difference": 0.001, "theoretical_bound": 0.0104}]).to_csv(
        evaluation / "replace_one_audit.csv", index=False
    )

    predictor = {
        "campaign_id": analysis.CAMPAIGN_ID,
        "observable_past_only_at_inference": True,
        "holdout_opened": False,
        "feature_names": [f"past_feature_{index}" for index in range(6)],
        "feature_scales": [0.1 + 0.01 * index for index in range(6)],
        "coefficients": [0.2 - 0.01 * index for index in range(6)],
        "selected_lambda": 0.01,
        "one_dimensional_control": {
            "feature_name": "past_feature_0",
            "feature_scale": 0.1,
            "coefficient": 0.2,
            "selected_lambda": 0.1,
        },
    }
    _json(results / "frozen_predictor.json", predictor)
    predictor_sha = _sha256(results / "frozen_predictor.json")
    design = {
        "train": {
            "flattened_design_rank": 6,
            "minimum_centered_rms": 0.01,
            "floor_active_count": 0,
        },
        "train_plus_calibration": {
            "flattened_design_rank": 6,
            "minimum_centered_rms": 0.009,
            "floor_active_count": 0,
        },
        "final_fit": {
            "condition_number_regularized_system": 15.0,
            "normal_equation_relative_residual": 1.0e-7,
        },
        "final_1d_fit": {
            "condition_number_regularized_system": 1.0,
            "normal_equation_relative_residual": 1.0e-8,
        },
    }
    _json(results / "feature_design_diagnostics.json", design)
    decision = _decision(config, seeds)
    _json(evaluation / "decision.json", decision)
    _json(
        results / "manifest.json",
        {
            "campaign_id": analysis.CAMPAIGN_ID,
            "status": "completed_development",
            "device": "mps",
            "holdout_opened": False,
            "frozen_predictor_sha256": predictor_sha,
            "config_sha256": _sha256(config_path),
            "evaluation_decision": decision["decision"],
            "all_gates_pass": True,
        },
    )
    _json(
        evaluation / "manifest.json",
        {
            "status": "completed_development",
            "device": "mps",
            "holdout_opened": False,
            "frozen_predictor_sha256": predictor_sha,
            "decision": decision["decision"],
            "all_gates_pass": True,
        },
    )
    _json(
        evaluation / "independent_postrun_audit.json",
        {
            "all_checks_pass": True,
            "audit_device": "mps",
            "holdout_opened": False,
            "fit_audit": {
                "all_checks_pass": True,
                "checks": {"fit_complete": True, "predictor_frozen": True},
            },
            "evaluation_audit": {
                "all_checks_pass": True,
                "validity_checks": {
                    "matrix_exact": True,
                    "holdout_closed": True,
                },
                "decision_recomputed": decision["decision"],
                "decision_differences": {
                    "decision": False,
                    "validity_pass": False,
                    "scientific_checks_pass": False,
                },
            },
        },
    )
    _json(
        results / "pre_evaluation_supplemental_audit.json",
        {
            "pass": True,
            "checks": {"phase_boundary": True, "fit_safe": True},
            "violations": [],
        },
    )
    return results, config_path


def test_running_screen_opens_no_raw_csv_and_creates_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "running"
    results.mkdir()
    _json(
        results / "manifest.json",
        {"campaign_id": analysis.CAMPAIGN_ID, "status": "fit_running"},
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("raw CSV was opened before completion")

    monkeypatch.setattr(analysis.pd, "read_csv", forbidden)
    report = tmp_path / "out" / "report.md"
    figures = tmp_path / "out" / "figures"
    with pytest.raises(RuntimeError, match="raw results were not opened"):
        analysis.analyze(
            results=results,
            config_path=tmp_path / "missing.yaml",
            report_path=report,
            figures_path=figures,
        )
    assert not report.exists()
    assert not figures.exists()


def test_invalid_official_audit_blocks_before_raw_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, config_path = _completed_fixture(tmp_path)
    audit_path = results / "evaluation/independent_postrun_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["all_checks_pass"] = False
    _json(audit_path, audit)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("raw CSV was opened after an invalid audit")

    monkeypatch.setattr(analysis.pd, "read_csv", forbidden)
    with pytest.raises(RuntimeError, match="official independent K5 audit"):
        analysis._validate_sources(results, config_path)


def test_completed_sources_reproduce_seed_level_metrics(tmp_path: Path) -> None:
    results, config_path = _completed_fixture(tmp_path)
    sources = analysis._validate_sources(results, config_path)
    history, mse = analysis._mse_summary(sources)
    gain = analysis._cell_gain_table(history)
    primary = mse.set_index("candidate").loc[analysis.K5, "mean"]
    baseline = mse.set_index("candidate").loc[analysis.K4B, "mean"]
    assert primary < baseline
    assert set(gain["assessment_round"]) == {17, 24}
    assert (gain["gain_vs_k4b"] > 0.0).all()


def test_analyze_publishes_one_report_and_exactly_four_figures(
    tmp_path: Path,
) -> None:
    results, config_path = _completed_fixture(tmp_path)
    report = tmp_path / "output/analysis/report.md"
    figures = tmp_path / "output/figures/k5"
    result = analysis.analyze(
        results=results,
        config_path=config_path,
        report_path=report,
        figures_path=figures,
    )
    assert result["all_gates_pass"] is True
    assert report.is_file()
    assert "Capture non bornée" in report.read_text(encoding="utf-8")
    assert sorted(path.name for path in figures.iterdir()) == sorted(
        analysis.FIGURE_NAMES
    )
    assert all(path.stat().st_size > 0 for path in figures.iterdir())


def test_student_interval_uses_outer_seeds_only() -> None:
    interval = analysis._student_ci([1.0, 3.0], 12.706204736432095)
    assert interval["n"] == 2
    assert interval["mean"] == pytest.approx(2.0)
    assert interval["sd"] == pytest.approx(2.0**0.5)
