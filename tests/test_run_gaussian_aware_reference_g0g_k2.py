"""Integration tests for the frozen G0g-K2 runner."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch
import yaml

from scripts import run_gaussian_aware_reference_g0g_k1 as k1
from scripts import run_gaussian_aware_reference_g0g_k2 as k2
from scripts import run_gaussian_aware_reference_oracle as oracle

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k2.yaml"


def _config() -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    k2._validate_config(config)
    return config


def _small_config() -> dict:
    config = copy.deepcopy(_config())
    config["randomness"]["calibration_seeds"] = [2026120107, 2026120209]
    config["randomness"]["calibration_draws_per_seed"] = 1
    config["randomness"]["development_seeds"] = config["randomness"][
        "development_seeds"
    ][:2]
    config["randomness"]["replace_one_trials_per_seed_cell"] = 1
    return config


def test_scalar_statistic_matches_declared_rms_formula() -> None:
    vectors = torch.tensor([[3.0, 4.0, 0.0, 12.0]], dtype=torch.float64)
    statistic = k2._client_statistics(
        vectors,
        anchor=torch.zeros(4, dtype=torch.float64),
        scales=torch.tensor([[5.0, 6.0]], dtype=torch.float64),
        block_sizes=[2, 2],
    )
    assert statistic.item() == pytest.approx(math.sqrt((1.0 + 4.0) / 2.0))


def test_separate_calibration_makes_homogeneous_aware_and_blind_identical() -> None:
    config = _small_config()
    oracle._configure_runtime("cpu")
    calibration, rows = k2._calibrate(config)
    assert rows
    assert calibration["thresholds"]["aware"]["homogeneous"] == pytest.approx(
        calibration["thresholds"]["blind"]["homogeneous"]
    )
    regime = config["privacy_noise"]["regimes"][0]
    variances, _ = oracle._noise_variances(config, regime, "identity")
    aware = k2._radii(
        config,
        variances,
        calibration,
        regime_name="homogeneous",
        blind=False,
    )
    blind = k2._radii(
        config,
        variances,
        calibration,
        regime_name="homogeneous",
        blind=True,
    )
    assert torch.allclose(aware, blind)


def test_calibration_uses_one_probe_per_independent_stratum_cohort() -> None:
    config = _small_config()
    oracle._configure_runtime("cpu")
    calibration, rows = k2._calibrate(config)
    modes = 2
    strata_per_mode = len(k1._noise_cells(config)) * len(
        config["randomness"]["calibration_contexts"]
    )
    expected_per_stratum = (
        len(config["randomness"]["calibration_seeds"])
        * config["randomness"]["calibration_draws_per_seed"]
    )
    assert len(rows) == modes * strata_per_mode * expected_per_stratum
    assert calibration["expected_pool_size_per_stratum"] == expected_per_stratum
    assert calibration["observed_unique_cohorts"] == (
        strata_per_mode * expected_per_stratum
    )
    assert all(row["probe_regular"] for row in rows)
    assert not any(row["probe_outlier"] for row in rows)
    assert all(
        stratum["pool_size"] == expected_per_stratum
        for stratum in calibration["strata"]
    )
    assert all(
        row["deployed_threshold"] + 1.0e-12 >= row["stratum_threshold"]
        for row in rows
    )


def test_full_calibration_uses_preregistered_rank_117() -> None:
    config = _config()
    oracle._configure_runtime("cpu")
    calibration, _ = k2._calibrate(config)
    assert calibration["expected_pool_size_per_stratum"] == 128
    assert {row["conformal_rank"] for row in calibration["strata"]} == {117}


def test_no_gate_control_matches_fcc_in_a_complete_cell() -> None:
    config = _small_config()
    oracle._configure_runtime("cpu")
    calibration, _ = k2._calibrate(config)
    regime = config["privacy_noise"]["regimes"][1]
    rows = k2._evaluate_cell(
        config,
        calibration,
        seed=int(config["randomness"]["development_seeds"][0]),
        regime=regime,
        permutation="byzantine_high",
        geometry="aligned",
        threat="bitflip_x10",
    )
    k2._attach_baselines(rows)
    no_gate = next(row for row in rows if row["candidate"] == k2.NO_GATE)
    primary = next(row for row in rows if row["candidate"] == k2.PRIMARY)
    assert no_gate["difference_vs_fcc"] == pytest.approx(0.0, abs=1e-12)
    assert primary["complete_client_influence_cap"] == pytest.approx(0.13)
    assert primary["replace_one_bound"] == pytest.approx(0.0104)


def test_reduced_full_matrix_reaches_gate_evaluator() -> None:
    config = _small_config()
    oracle._configure_runtime("cpu")
    calibration, _ = k2._calibrate(config)
    rows = []
    for seed in config["randomness"]["development_seeds"]:
        for regime, permutation in k1._noise_cells(config):
            for geometry in config["cohort"]["honest_outliers"]["geometries"]:
                for threat in config["threats"]["names"]:
                    rows.extend(
                        k2._evaluate_cell(
                            config,
                            calibration,
                            seed=int(seed),
                            regime=regime,
                            permutation=permutation,
                            geometry=str(geometry),
                            threat=str(threat),
                        )
                    )
    k2._attach_baselines(rows)
    replace_rows = k2._replace_one_audit(config, calibration)
    decision = k2._evaluate_gates(rows, replace_rows, config)
    assert decision["observed"]["development_rows"] == 800
    assert decision["observed"]["fixed_global_cap_max_abs_difference_vs_fcc"] < 1e-12
    assert decision["observed"]["replace_one_trials"] == 10
    assert decision["observed"]["contribution_cap_violations"] == 0
