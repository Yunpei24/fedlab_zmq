"""Integration tests for the frozen G0g-K1 development runner."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from scripts import run_gaussian_aware_reference_g0g_k1 as g0g
from scripts import run_gaussian_aware_reference_oracle as oracle

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k1.yaml"


def _config() -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    g0g._validate_config(config)
    return config


def _small_calibration_config() -> dict:
    config = copy.deepcopy(_config())
    config["randomness"]["calibration_folds"] = [[2026110101], [2026110203]]
    config["randomness"]["calibration_draws_per_seed"] = 1
    return config


def test_total_cap_is_not_mistaken_for_a_per_block_cap() -> None:
    config = _config()
    vectors = torch.zeros(25, 64, dtype=torch.float64)
    caps = g0g._block_caps(config, vectors)
    assert caps.tolist() == pytest.approx([0.065] * 4)
    assert torch.linalg.vector_norm(caps).item() == pytest.approx(0.13)
    assert 2.0 * torch.linalg.vector_norm(caps).item() / 25.0 == pytest.approx(0.0104)


def test_calibration_is_disjoint_and_covariance_changes_only_radii() -> None:
    config = _small_calibration_config()
    oracle._configure_runtime("cpu")
    calibration, rows = g0g._calibrate(config)
    assert rows
    assert calibration["calibration_and_development_disjoint"] is True
    assert all(value > 0.0 for value in calibration["standardized_thresholds_by_block"])

    heteroscedastic = config["privacy_noise"]["regimes"][1]
    variances, tiers = oracle._noise_variances(config, heteroscedastic, "identity")
    radii = g0g._radii(config, variances, calibration)
    low = radii[tiers.argmin()]
    high = radii[tiers.argmax()]
    assert bool((high > low).all())

    vectors = torch.zeros(25, 64, dtype=torch.float64)
    caps = g0g._block_caps(config, vectors)
    assert torch.unique(caps).numel() == 1
    assert torch.linalg.vector_norm(caps).item() == pytest.approx(0.13)


def test_one_paired_cell_emits_every_frozen_candidate() -> None:
    config = _small_calibration_config()
    oracle._configure_runtime("cpu")
    calibration, _ = g0g._calibrate(config)
    regime = config["privacy_noise"]["regimes"][1]
    rows = g0g._evaluate_cell(
        config,
        calibration,
        seed=int(config["randomness"]["development_seeds"][0]),
        regime=regime,
        permutation="byzantine_high",
        geometry="aligned",
        threat="bitflip_x10",
    )
    assert [row["candidate"] for row in rows] == list(g0g.CANDIDATES)
    assert len({row["pairing_id"] for row in rows}) == 1
    assert all(row["all_finite"] for row in rows)
    primary = next(row for row in rows if row["candidate"] == g0g.PRIMARY)
    assert primary["complete_client_influence_cap"] == pytest.approx(0.13)
    assert primary["replace_one_bound"] == pytest.approx(0.0104)
    assert primary["gate_sum_normalized"] is False


def test_frozen_matrix_cardinality_is_2000_rows() -> None:
    config = _config()
    expected = (
        len(config["randomness"]["development_seeds"])
        * len(g0g._noise_cells(config))
        * len(config["cohort"]["honest_outliers"]["geometries"])
        * len(config["threats"]["names"])
        * len(g0g.CANDIDATES)
    )
    assert expected == 2000


def test_gate_evaluator_accepts_a_complete_reduced_matrix() -> None:
    """Smoke-test every candidate/cell and the seed-level decision code."""

    config = _small_calibration_config()
    config["randomness"]["development_seeds"] = config["randomness"][
        "development_seeds"
    ][:2]
    config["randomness"]["replace_one_trials_per_seed_cell"] = 1
    oracle._configure_runtime("cpu")
    calibration, _ = g0g._calibrate(config)
    rows = []
    for seed in config["randomness"]["development_seeds"]:
        for regime, permutation in g0g._noise_cells(config):
            for geometry in config["cohort"]["honest_outliers"]["geometries"]:
                for threat in config["threats"]["names"]:
                    rows.extend(
                        g0g._evaluate_cell(
                            config,
                            calibration,
                            seed=int(seed),
                            regime=regime,
                            permutation=permutation,
                            geometry=str(geometry),
                            threat=str(threat),
                        )
                    )
    g0g._attach_paired_baselines(rows)
    replace_rows = g0g._replace_one_audit(config, calibration)
    decision = g0g._evaluate_gates(rows, replace_rows, config)
    assert decision["observed"]["development_rows"] == 800
    assert decision["observed"]["replace_one_trials"] == 10
    assert set(decision["checks"]) == {
        "complete",
        "finite",
        "contribution_cap",
        "replace_one",
        "false_tail_rate",
        "false_tail_tier_gap",
        "homogeneous_clean_noninferiority",
        "heteroscedastic_clean_noninferiority",
        "attacked_ci_vs_fcc",
        "attacked_gain_vs_fcc",
        "attacked_ci_vs_sigma_blind",
        "attacked_gain_vs_sigma_blind",
        "attacked_worst_group_ratio",
        "alie_ratio",
        "byzantine_contribution_share",
    }
