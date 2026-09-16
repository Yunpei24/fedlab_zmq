"""Contract tests for the preregistered G0g-K4c-CH development runner."""

from __future__ import annotations

import copy
import inspect
import json
import math
from pathlib import Path

import pytest
import torch
import yaml

from scripts import run_gaussian_aware_reference_g0g_k4_tcg as k4
from scripts import run_gaussian_aware_reference_g0g_k4c_causal_headroom as k4c
from scripts import run_gaussian_aware_reference_oracle as oracle

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k4c_causal_headroom.yaml"
)


@pytest.fixture
def production_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture
def reduced_config(production_config: dict) -> dict:
    config = copy.deepcopy(production_config)
    config["cohort"].update(
        {
            "num_clients": 6,
            "num_byzantine": 2,
            "dimension": 8,
            "block_sizes": [4, 4],
            "heterogeneity_std_by_block": [0.012, 0.025],
        }
    )
    config["cohort"]["honest_outliers"].update({"count": 1, "geometries": ["aligned"]})
    config["privacy_noise"].update(
        {
            "block_std_multipliers": [0.8, 1.4],
            "regimes": [
                {
                    "name": "heteroscedastic",
                    "client_std_multipliers": [1.0, 1.5, 2.0],
                    "permutations": ["byzantine_high"],
                }
            ],
        }
    )
    config["temporal"].update(
        {
            "standardized_clip_norm": 2.0 * math.sqrt(8.0),
            "total_rounds_needed_for_frozen_pasts": 17,
            "assessment_rounds": [17],
        }
    )
    config["honest_dynamics"]["names"] = ["stationary"]
    config["threats"]["names"] = ["bitflip_x10"]
    config["nested_monte_carlo"].update(
        {
            "construction_children": 4,
            "construction_split_a_children": 2,
            "construction_split_b_children": 2,
            "evaluation_children": 4,
        }
    )
    config["randomness"].update(
        {
            "development_outer_seeds": [7301],
            "replace_one_trials_per_seed_noise_cell": 6,
        }
    )
    oracle._configure_runtime("cpu")
    return config


def test_production_config_matrix_rng_and_scientific_scope(
    production_config: dict,
) -> None:
    k4c._validate_config(production_config)
    cells = k4c._outer_cells(production_config)
    assert len(cells) == 288
    history_ids = {
        k4c._history_id(cell, round_index)
        for cell in cells
        for round_index in production_config["temporal"]["assessment_rounds"]
    }
    assert len(history_ids) == 576
    cell = cells[0]
    construction = {
        k4c._child_seed(production_config, cell, 17, "construction", child)
        for child in range(64)
    }
    evaluation = {
        k4c._child_seed(production_config, cell, 17, "evaluation", child)
        for child in range(64)
    }
    assert len(construction) == len(evaluation) == 64
    assert not (construction & evaluation)
    contract = production_config["scientific_contract"]
    assert contract["conditions_on_latent_clean_current_state"] is True
    assert contract["observable_past_only_predictor_constructed"] is False
    assert contract["pass_authorizes_holdout_or_promotion"] is False
    assert contract["valid_scientific_fail_stops_branch"] is True
    assert (
        production_config["randomness"]["replace_one_trials_per_seed_noise_cell"]
        == production_config["cohort"]["num_clients"]
    )
    assert production_config["gates"]["replace_one_exact_trials"] == 900


def test_children_fix_clean_current_state_but_resample_fresh_noise(
    reduced_config: dict,
) -> None:
    regime = reduced_config["privacy_noise"]["regimes"][0]
    cell = {
        "seed": 7301,
        "regime": regime,
        "permutation": "byzantine_high",
        "geometry": "aligned",
        "dynamics": "stationary",
        "threat": "bitflip_x10",
    }
    components = k4._trajectory_components(
        reduced_config,
        seed=7301,
        regime=regime,
        permutation="byzantine_high",
        geometry="aligned",
        dynamics="stationary",
    )
    construction, clean_a, seed_a = k4c._current_child(
        reduced_config,
        components,
        cell,
        round_index=17,
        stream="construction",
        child=0,
    )
    evaluation, clean_b, seed_b = k4c._current_child(
        reduced_config,
        components,
        cell,
        round_index=17,
        stream="evaluation",
        child=0,
    )
    assert torch.equal(clean_a, clean_b)
    assert seed_a != seed_b
    assert not torch.equal(construction, evaluation)


def test_reduced_snapshot_enforces_independence_crn_and_certificates(
    reduced_config: dict,
) -> None:
    k2_calibration, temporal_calibration, _ = k4c._load_calibrations(reduced_config)
    regime = reduced_config["privacy_noise"]["regimes"][0]
    cell = {
        "seed": 7301,
        "regime": regime,
        "permutation": "byzantine_high",
        "geometry": "aligned",
        "dynamics": "stationary",
        "threat": "bitflip_x10",
    }
    histories, children, contexts = k4c._evaluate_outer_trajectory(
        reduced_config,
        k2_calibration,
        temporal_calibration,
        cell,
    )
    assert len(histories) == 1
    assert len(children) == 4 * len(k4c.CANDIDATES)
    assert len(contexts) == 1
    history = histories[0]
    assert history["construction_evaluation_seed_overlap"] == 0
    assert history["semi_oracle_conditions_on_latent_clean_current_state"] is True
    assert history["semi_oracle_is_observable_past_measurable"] is False
    semi_rows = [row for row in children if row["candidate"] == k4c.SEMI_ORACLE]
    assert len({row["semi_oracle_predictor_hash"] for row in semi_rows}) == 1
    assert all(
        row["pointwise_oracle_used_in_construction"] is False for row in children
    )
    by_child: dict[int, set[str]] = {}
    for row in children:
        by_child.setdefault(int(row["evaluation_child"]), set()).add(
            str(row["crn_key"])
        )
    assert all(len(keys) == 1 for keys in by_child.values())
    assert max(float(row["k4_manual_formula_error"]) for row in children) <= 1.0e-6
    assert (
        max(float(row["k4b_comparator_reproduction_error"]) for row in children)
        <= 1.0e-6
    )
    assert (
        max(float(row["fixed_denominator_formula_error"]) for row in children) <= 1.0e-6
    )
    assert not any(bool(row["normalization_by_gate_sum"]) for row in children)
    assert all(bool(row["contribution_cap_respected"]) for row in semi_rows)

    replacements = k4c._replace_one_rows(reduced_config, temporal_calibration, contexts)
    assert len(replacements) == 6
    assert {int(row["replaced_client"]) for row in replacements} == set(range(6))
    assert all(row["same_past"] is True for row in replacements)
    assert all(row["same_predictor"] is True for row in replacements)
    assert not any(row["violation"] for row in replacements)
    assert all(
        row["theoretical_bound"] == pytest.approx(2 * 0.13 / 6) for row in replacements
    )


def _synthetic_child_row(
    history_id: str,
    seed: int,
    candidate: str,
    squared_error: float,
) -> dict:
    return {
        "history_id": history_id,
        "seed": seed,
        "candidate": candidate,
        "squared_reference_error": squared_error,
        "all_finite": True,
        "contribution_cap_respected": True,
        "semi_oracle_predictor_hash": f"predictor-{history_id}",
        "pointwise_oracle_used_in_construction": False,
        "normalization_by_gate_sum": False,
        "fixed_denominator_n": 25,
        "fixed_denominator_formula_error": 0.0,
        "k4_manual_formula_error": 0.0,
        "frozen_k4_comparator_error_descriptive": 0.0,
        "k4b_comparator_reproduction_error": 0.0,
        "pointwise_excess_mse_over_candidate": 0.0,
    }


def test_summary_uses_all_R_positive_histories_without_outcome_filter(
    production_config: dict,
) -> None:
    config = copy.deepcopy(production_config)
    config["randomness"]["development_outer_seeds"] = [1, 2]
    config["statistical_analysis"].update(
        {"frozen_histories_total": 4, "num_independent_units": 2}
    )
    config["nested_monte_carlo"]["evaluation_children"] = 1
    config["gates"].update(
        {
            "exact_seed_capture_count": 2,
            "replace_one_exact_trials": 0,
            "eligible_R_positive_history_fraction_min": 1.0,
        }
    )
    history_rows = []
    child_rows = []
    for seed in (1, 2):
        for index, regime in enumerate(("homogeneous", "heteroscedastic")):
            history_id = f"{seed}|{index}"
            history_rows.append(
                {
                    "history_id": history_id,
                    "seed": seed,
                    "noise_regime": regime,
                    "noise_permutation": "identity",
                    "eligible_R_positive": True,
                    "split_a_b_aggregate_scale_distance": 0.01,
                    "semi_oracle_projection_active": False,
                    "construction_evaluation_seed_overlap": 0,
                    "construction_child_seed_registry": str(10_000 + 10 * seed + index),
                    "evaluation_child_seed_registry": str(20_000 + 10 * seed + index),
                }
            )
            # The second history has zero pointwise headroom but negative
            # semi-oracle headroom. It must still be included because R > 0.
            errors = {
                k4c.K2: 1.1,
                k4c.K4: 1.0,
                k4c.K4B: 0.9,
                k4c.SEMI_ORACLE: 0.8 if index == 0 else 1.1,
                k4c.POINTWISE: 0.5 if index == 0 else 1.0,
            }
            child_rows.extend(
                _synthetic_child_row(history_id, seed, candidate, value)
                for candidate, value in errors.items()
            )
    seed_rows, _ = k4c._summarize(config, history_rows, child_rows, [])
    assert [row["eligible_histories"] for row in seed_rows] == [2, 2]
    # (0.2 - 0.1) / (0.5 + 0.0) = 0.1 / 0.5, using both histories.
    assert all(row["capture_fraction"] == pytest.approx(0.2) for row in seed_rows)
    assert all(
        row["pointwise_relative_mse_headroom_vs_k4"] == pytest.approx(0.25)
        for row in seed_rows
    )
    assert all(
        row["semi_oracle_relative_mse_gain_vs_k4"] == pytest.approx(0.05)
        for row in seed_rows
    )


def test_production_runner_has_no_cpu_or_holdout_execution_path() -> None:
    source = inspect.getsource(k4c.run) + inspect.getsource(k4c.main)
    assert 'oracle._configure_runtime("mps")' in source
    assert 'choices=("mps",)' in source
    assert "reserved_holdout_seeds" not in source
    assert "allow_cpu_fallback" not in source


def test_decision_status_never_turns_validity_failure_into_scientific_failure() -> None:
    assert (
        k4c._decision_status(
            {"complete": False, "finite": True}, {"material_gain": False}
        )
        == "invalid_or_inconclusive_screen"
    )
    assert (
        k4c._decision_status(
            {"complete": True, "finite": True}, {"material_gain": False}
        )
        == "stop_missing_slot_imputation_branch"
    )
    assert (
        k4c._decision_status(
            {"complete": True, "finite": True}, {"material_gain": True}
        )
        == "authorize_transcript_only_predictor_study"
    )


def test_missing_one_eligible_cell_changes_valid_screen_to_scientific_stop() -> None:
    validity = {"matrix_exact": True, "finite": True}
    complete = {"eligible_R": True, "eligible_noise_cell_composition": True}
    missing_one = dict(complete, eligible_noise_cell_composition=False)
    assert (
        k4c._decision_status(validity, complete)
        == "authorize_transcript_only_predictor_study"
    )
    assert (
        k4c._decision_status(validity, missing_one)
        == "stop_missing_slot_imputation_branch"
    )


def test_manual_k4_formula_check_detects_a_corrupted_direct_sum() -> None:
    anchor = torch.tensor([0.1, -0.2], dtype=torch.float32)
    direct_sum = torch.tensor([0.4, 0.6], dtype=torch.float32)
    reference = anchor + direct_sum / 4.0
    assert k4c._manual_k4_formula_error(reference, anchor, direct_sum, 4) == 0.0
    corrupted = direct_sum + torch.tensor([0.05, 0.0], dtype=torch.float32)
    assert k4c._manual_k4_formula_error(
        reference, anchor, corrupted, 4
    ) == pytest.approx(0.0125)


def test_publication_attestation_is_exact_and_precedes_output_creation() -> None:
    lock = {"sha256": "ab" * 32}
    attestation = k4c._attest_external_lock_publication(lock, "AB" * 32)
    assert attestation["published_lock_sha256"] == "ab" * 32
    with pytest.raises(RuntimeError, match="must equal"):
        k4c._attest_external_lock_publication(lock, "cd" * 32)
    source = inspect.getsource(k4c.run)
    assert source.index("_attest_external_lock_publication") < source.index(
        "output.mkdir"
    )


def test_production_rejects_non_preregistered_output_before_lock_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="preregistered output path"):
        k4c.run(
            k4c.DEFAULT_CONFIG,
            k4c.DEFAULT_LOCK,
            tmp_path / "alternate-output",
            published_lock_sha256="ab" * 32,
        )


def test_nonfinite_json_is_null_and_report_prints_na(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = tmp_path / "payload.json"
    k4c._write_json(payload, {"capture": float("nan")})
    assert json.loads(payload.read_text(encoding="utf-8")) == {"capture": None}

    interval = {
        "n": 0,
        "mean": float("nan"),
        "sd": float("nan"),
        "low": float("nan"),
        "high": float("nan"),
    }
    observed = {
        "pointwise_relative_mse_headroom_vs_k4_seed_ci95": interval,
        "semi_oracle_relative_mse_gain_vs_k4_seed_ci95": interval,
        "semi_oracle_capture_fraction_seed_ci95": interval,
        "homogeneous_semi_oracle_relative_mse_gain_vs_k4_seed_ci95": interval,
        "heteroscedastic_semi_oracle_relative_mse_gain_vs_k4_seed_ci95": interval,
        "homogeneous_capture_fraction_seed_ci95_descriptive": interval,
        "heteroscedastic_capture_fraction_seed_ci95_descriptive": interval,
        "semi_oracle_relative_mse_gain_vs_k4b_seed_ci95": interval,
        "construction_split_disagreement_mse_ratio_max": float("nan"),
        "semi_oracle_projection_cap_saturation_rate": 0.0,
        "eligible_R_positive_history_fraction": 1.0,
        "replace_one_violations": 0,
        "replace_one_max_ratio_to_bound": 0.0,
    }
    report = tmp_path / "report.md"
    monkeypatch.setattr(k4c, "DEFAULT_REPORT", report)
    k4c._report(
        {
            "decision": "stop_missing_slot_imputation_branch",
            "checks": {"positive_headroom": False},
            "observed": observed,
        },
        k4c.ROOT / "synthetic-results",
    )
    assert "| Semi-oracle capture mean | NA |" in report.read_text(encoding="utf-8")
