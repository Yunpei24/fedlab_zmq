"""Contract tests for the unlocked G0g-K4c-CH post-run figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from scripts import plot_gaussian_aware_g0g_k4c_ch as figures

SEEDS = list(range(101, 113))
SCIENCE_CHECKS = [
    "material_pointwise_headroom_mean",
    "material_pointwise_headroom_ci",
    "material_semi_oracle_gain_vs_k4_mean",
    "material_semi_oracle_gain_vs_k4_ci",
    "capture_mean",
    "capture_ci_low",
    "homogeneous_gain_vs_k4_ci",
    "heteroscedastic_gain_vs_k4_ci",
]


def _config() -> dict[str, Any]:
    return {
        "campaign_id": figures.CAMPAIGN_ID,
        "randomness": {"development_outer_seeds": SEEDS},
        "privacy_noise": {
            "regimes": [
                {"name": "homogeneous"},
                {"name": "heteroscedastic"},
            ]
        },
        "nested_monte_carlo": {"evaluation_children": 2},
        "statistical_analysis": {
            "t_critical_df11": 2.2009851600916406,
            "gate_classification": {
                "validity": ["complete"],
                "scientific": SCIENCE_CHECKS,
            },
        },
        "gates": {
            "pointwise_relative_mse_headroom_vs_k4_mean_min": 0.10,
            (
                "pointwise_relative_mse_headroom_vs_k4_seed_ci95_low_"
                "strictly_greater_than"
            ): 0.05,
            "semi_oracle_relative_mse_gain_vs_k4_mean_min": 0.05,
            (
                "semi_oracle_relative_mse_gain_vs_k4_seed_ci95_low_"
                "strictly_greater_than"
            ): 0.0,
            "semi_oracle_capture_fraction_mean_min": 0.50,
            ("semi_oracle_capture_fraction_ci95_low_strictly_greater_than"): 0.30,
            (
                "homogeneous_semi_oracle_relative_mse_gain_vs_k4_seed_ci95_"
                "low_strictly_greater_than"
            ): 0.0,
            (
                "heteroscedastic_semi_oracle_relative_mse_gain_vs_k4_seed_ci95_"
                "low_strictly_greater_than"
            ): 0.0,
        },
    }


def _seed_summary(config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for index, seed in enumerate(SEEDS):
        jitter = (index - 5.5) * 0.001
        rows.append(
            {
                "seed": seed,
                "pointwise_relative_mse_headroom_vs_k4": 0.20 + jitter,
                "semi_oracle_relative_mse_gain_vs_k4": 0.12 + jitter,
                "capture_fraction": 0.60 + jitter,
                "homogeneous_semi_oracle_relative_mse_gain_vs_k4": (0.11 + jitter),
                "heteroscedastic_semi_oracle_relative_mse_gain_vs_k4": (0.125 + jitter),
            }
        )
    return pd.DataFrame(rows)


def _decision(config: dict[str, Any], seed_summary: pd.DataFrame) -> dict[str, Any]:
    t_critical = config["statistical_analysis"]["t_critical_df11"]

    def interval(column: str) -> dict[str, float | int]:
        result = figures._student_ci(
            seed_summary[column].to_numpy(dtype=float), t_critical
        )
        result["n"] = int(result["n"])
        return result

    validity = {"complete": True}
    scientific = {name: True for name in SCIENCE_CHECKS}
    return {
        "decision": "authorize_transcript_only_predictor_study",
        "all_gates_pass": True,
        "checks": {**validity, **scientific},
        "validity_checks": validity,
        "scientific_checks": scientific,
        "validity_pass": True,
        "scientific_checks_pass": True,
        "holdout_opened": False,
        "loss_used_for_all_scientific_gates": "squared_l2_reference_error",
        "observed": {
            "pointwise_relative_mse_headroom_vs_k4_seed_ci95": interval(
                "pointwise_relative_mse_headroom_vs_k4"
            ),
            "semi_oracle_relative_mse_gain_vs_k4_seed_ci95": interval(
                "semi_oracle_relative_mse_gain_vs_k4"
            ),
            "semi_oracle_capture_fraction_seed_ci95": interval("capture_fraction"),
            "homogeneous_semi_oracle_relative_mse_gain_vs_k4_seed_ci95": (
                interval("homogeneous_semi_oracle_relative_mse_gain_vs_k4")
            ),
            "heteroscedastic_semi_oracle_relative_mse_gain_vs_k4_seed_ci95": (
                interval("heteroscedastic_semi_oracle_relative_mse_gain_vs_k4")
            ),
        },
    }


def _raw_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    histories: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    candidate_base = {
        figures.K2: 1.20,
        figures.K4: 1.00,
        figures.K4B: 0.94,
        figures.K4C: 0.88,
        figures.POINTWISE: 0.80,
    }
    for seed_index, seed in enumerate(SEEDS):
        for regime_index, regime in enumerate(figures.REGIMES):
            history_id = f"{seed}|{regime}|history"
            histories.append(
                {
                    "history_id": history_id,
                    "seed": seed,
                    "noise_regime": regime,
                    "eligible_R_positive": True,
                    "eligible_R_positive_bool": True,
                }
            )
            for child in range(2):
                for candidate in figures.CANDIDATES:
                    children.append(
                        {
                            "history_id": history_id,
                            "seed": seed,
                            "noise_regime": regime,
                            "evaluation_child": child,
                            "candidate": candidate,
                            "squared_reference_error": (
                                candidate_base[candidate]
                                + 0.02 * regime_index
                                + 0.001 * seed_index
                                + 0.0001 * child
                            ),
                        }
                    )
    return pd.DataFrame(histories), pd.DataFrame(children)


def test_completion_guard_opens_no_raw_or_config_before_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "manifest.json").write_text(
        json.dumps(
            {
                "campaign_id": figures.CAMPAIGN_ID,
                "status": "running_development",
            }
        ),
        encoding="utf-8",
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("post-run guard opened a forbidden source")

    monkeypatch.setattr(figures.audit, "audit_results", forbidden)
    monkeypatch.setattr(figures.pd, "read_csv", forbidden)
    with pytest.raises(RuntimeError, match="raw CSVs were not opened"):
        figures._validate_sources(results, tmp_path / "missing.yaml")


def test_completed_synthetic_sources_validate_with_current_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    seed_summary = _seed_summary(config)
    decision = _decision(config, seed_summary)
    histories, children = _raw_frames()
    results = tmp_path / "completed"
    results.mkdir()
    seed_summary.to_csv(results / "seed_summary.csv", index=False)
    histories.drop(columns="eligible_R_positive_bool").to_csv(
        results / "frozen_history_rows.csv", index=False
    )
    children.to_csv(results / "evaluation_child_rows.csv", index=False)
    pd.DataFrame([{"trial": 0}]).to_csv(results / "replace_one_audit.csv", index=False)
    (results / "frozen_calibration_provenance.json").write_text("{}", encoding="utf-8")
    (results / "decision.json").write_text(json.dumps(decision), encoding="utf-8")
    manifest = {
        "campaign_id": figures.CAMPAIGN_ID,
        "status": "completed_development",
        "device": "mps",
        "dtype": "torch.float32",
        "development_only": True,
        "holdout_opened": False,
        "observable_past_only_predictor_constructed": False,
        "config_sha256": figures._sha256(config_path),
        "frozen_histories": len(histories),
        "evaluation_child_rows": len(children),
    }
    (results / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        figures.audit,
        "audit_results",
        lambda *args, **kwargs: {
            "all_checks_pass": True,
            "audit_status": "passed",
            "seed_summary_differences": [],
            "decision_differences": [],
            "holdout_opened": False,
            "recomputed_decision": {"decision": decision["decision"]},
        },
    )
    sources = figures._validate_sources(results, config_path)
    assert len(sources.seed_summary) == 12
    assert set(sources.children["candidate"]) == set(figures.CANDIDATES)
    assert sources.histories["eligible_R_positive_bool"].all()


def test_mse_aggregation_averages_children_before_histories() -> None:
    config = _config()
    config["randomness"]["development_outer_seeds"] = [1, 2]
    histories: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    for seed in (1, 2):
        for regime in figures.REGIMES:
            for history_index, child_values in enumerate(((0.0, 0.0), (3.0, 3.0))):
                history_id = f"{seed}|{regime}|{history_index}"
                histories.append(
                    {
                        "history_id": history_id,
                        "seed": seed,
                        "noise_regime": regime,
                        "eligible_R_positive_bool": True,
                    }
                )
                for candidate in figures.MSE_CANDIDATES:
                    for child, value in enumerate(child_values):
                        children.append(
                            {
                                "history_id": history_id,
                                "seed": seed,
                                "noise_regime": regime,
                                "candidate": candidate,
                                "evaluation_child": child,
                                "squared_reference_error": value,
                            }
                        )
    values = figures._mse_seed_values(
        pd.DataFrame(histories), pd.DataFrame(children), config
    )
    assert values["mse"].tolist() == pytest.approx([1.5] * len(values))


def test_mse_rejects_candidate_missing_from_one_history() -> None:
    config = _config()
    histories, children = _raw_frames()
    first_history = histories.iloc[0]["history_id"]
    incomplete = children.loc[
        ~(
            children["history_id"].eq(first_history)
            & children["candidate"].eq(figures.K4)
        )
    ]
    with pytest.raises(RuntimeError, match="incomplete"):
        figures._mse_seed_values(histories, incomplete, config)


def test_four_figures_render_from_synthetic_postrun_data(tmp_path: Path) -> None:
    config = _config()
    seed_summary = _seed_summary(config)
    decision = _decision(config, seed_summary)
    histories, children = _raw_frames()
    sources = figures.ValidatedSources(
        manifest={"status": "completed_development"},
        config=config,
        decision=decision,
        audit_result={"all_checks_pass": True},
        histories=histories,
        children=children,
        seed_summary=seed_summary,
    )
    figures._configure_style()
    paths = figures._generate_figures(sources, tmp_path / "figures")
    assert [path.name for path in paths] == [
        "01_p_q_c_par_seed.png",
        "02_mse_k4_k4b_k4c_pointwise_par_bruit.png",
        "03_q_homogene_vs_heteroscedastique.png",
        "04_synthese_des_gates.png",
    ]
    assert all(path.stat().st_size > 1_000 for path in paths)
    assert len(list((tmp_path / "figures").glob("*.csv"))) == 4


def test_figure_output_cannot_be_inside_result_tree(tmp_path: Path) -> None:
    results = tmp_path / "results"
    with pytest.raises(RuntimeError, match="outside"):
        figures._ensure_separate_output(results, results / "derived")


def test_real_config_gate_registry_is_fully_labelled() -> None:
    config = yaml.safe_load(figures.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    classification = config["statistical_analysis"]["gate_classification"]
    validity = {name: True for name in classification["validity"]}
    scientific = {name: True for name in classification["scientific"]}
    rows = figures._gate_rows(
        {
            "validity_checks": validity,
            "scientific_checks": scientific,
            "validity_pass": True,
            "scientific_checks_pass": True,
        },
        config,
    )
    assert len(validity) == 19
    assert len(scientific) == 15
    assert len(rows) == 34
    assert set(rows["gate"]).issubset(figures.GATE_LABELS)


def test_completed_manifest_with_missing_artifact_opens_no_raw_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "manifest.json").write_text(
        json.dumps(
            {
                "campaign_id": figures.CAMPAIGN_ID,
                "status": "completed_development",
            }
        ),
        encoding="utf-8",
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("an incomplete result artifact was opened")

    monkeypatch.setattr(figures.audit, "audit_results", forbidden)
    monkeypatch.setattr(figures.pd, "read_csv", forbidden)
    with pytest.raises(FileNotFoundError, match="no raw artifact was opened"):
        figures._validate_sources(results, tmp_path / "missing.yaml")


def test_seed_and_child_identifiers_must_be_exact_integers() -> None:
    frame = pd.DataFrame({"seed": [101.5, 102.0]})
    with pytest.raises(ValueError, match="exact integers"):
        figures._require_integer_values(frame, "seed", source="synthetic.csv")


def test_invalid_screen_greys_scientific_gates_and_rejects_string_booleans() -> None:
    config = _config()
    seed_summary = _seed_summary(config)
    decision = _decision(config, seed_summary)
    decision["validity_checks"]["complete"] = False
    decision["checks"]["complete"] = False
    decision["validity_pass"] = False
    decision["scientific_checks_pass"] = None
    rows = figures._gate_rows(decision, config)
    science = rows.loc[rows["gate_class"].eq("Scientifique")]
    assert set(science["status"]) == {"NON ÉVALUÉ"}
    assert rows.loc[rows["gate_class"].eq("Validité"), "status"].tolist() == ["FAIL"]

    decision["scientific_checks"][SCIENCE_CHECKS[0]] = "false"
    with pytest.raises(ValueError, match="JSON Boolean"):
        figures._gate_rows(decision, config)
