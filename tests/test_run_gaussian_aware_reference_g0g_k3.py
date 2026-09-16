"""Integration tests for the frozen G0g-K3 dual-gate runner."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from scripts import run_gaussian_aware_reference_g0g_k1 as k1
from scripts import run_gaussian_aware_reference_g0g_k3 as k3
from scripts import run_gaussian_aware_reference_oracle as oracle

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k3.yaml"


def _config() -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    k3._validate_config(config)
    return config


def _small_config() -> dict:
    config = copy.deepcopy(_config())
    config["randomness"]["development_seeds"] = config["randomness"][
        "development_seeds"
    ][:2]
    config["randomness"]["replace_one_trials_per_seed_cell"] = 1
    return config


def test_frozen_calibration_sha_and_thresholds_are_loaded_exactly() -> None:
    config = _config()
    calibration, provenance = k3._load_frozen_calibration(config)
    assert provenance["sha256_verified"] is True
    assert provenance["recalibrated_for_k3"] is False
    assert provenance["observed_sha256"] == k3.FROZEN_CALIBRATION_SHA256
    assert provenance["thresholds_reused_exactly"] == calibration["thresholds"]
    assert calibration["thresholds"]["aware"]["heteroscedastic"] == pytest.approx(
        1.1457048654556274
    )
    assert calibration["thresholds"]["blind"]["heteroscedastic"] == pytest.approx(
        1.2459673881530762
    )


def test_frozen_calibration_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    config = copy.deepcopy(_config())
    (tmp_path / "calibration.json").write_text("{}", encoding="utf-8")
    config["frozen_calibration"]["path"] = "calibration.json"
    with pytest.raises(RuntimeError, match="hash mismatch"):
        k3._load_frozen_calibration(config, root=tmp_path)


def test_homogeneous_dual_gate_is_exactly_k2_aware() -> None:
    config = _small_config()
    calibration, _ = k3._load_frozen_calibration(config)
    oracle._configure_runtime("cpu")
    regime = config["privacy_noise"]["regimes"][0]
    rows = k3._evaluate_cell(
        config,
        calibration,
        seed=int(config["randomness"]["development_seeds"][0]),
        regime=regime,
        permutation="identity",
        geometry="aligned",
        threat="bitflip_x10",
    )
    assert len(rows) == len(k3.CANDIDATES)
    dual = next(row for row in rows if row["candidate"] == k3.PRIMARY)
    no_gate = next(row for row in rows if row["candidate"] == k3.NO_GATE)
    assert dual["exact_difference_to_aware_reference"] <= 1.0e-7
    assert no_gate["exact_difference_to_fcc_reference"] <= 1.0e-7
    assert dual["complete_client_influence_cap"] == pytest.approx(0.13)
    assert dual["replace_one_bound"] == pytest.approx(0.0104)


def test_dual_gate_never_exceeds_aware_gate_in_heteroscedastic_cell() -> None:
    config = _small_config()
    calibration, _ = k3._load_frozen_calibration(config)
    oracle._configure_runtime("cpu")
    regime = config["privacy_noise"]["regimes"][1]
    blocks = tuple(int(value) for value in config["cohort"]["block_sizes"])
    clean, _, _, anchor = oracle._honest_clean_vectors(
        config,
        seed=2027011003,
        draw=0,
        geometry="orthogonal",
        include_outliers=True,
    )
    variances, _ = oracle._noise_variances(config, regime, "byzantine_high")
    vectors = k1._paired_private_noise(
        clean,
        variances,
        blocks,
        seed=2027011003,
        draw=0,
        geometry="orthogonal",
    )
    aware = k3._radii(
        config,
        variances,
        calibration,
        regime_name="heteroscedastic",
        blind=False,
    )
    blind = k3._radii(
        config,
        variances,
        calibration,
        regime_name="heteroscedastic",
        blind=True,
    )
    _, aware_diagnostics = k3._reference(
        k3.AWARE,
        vectors,
        anchor=anchor,
        aware_radii=aware,
        blind_radii=blind,
        config=config,
    )
    _, dual_diagnostics = k3._reference(
        k3.PRIMARY,
        vectors,
        anchor=anchor,
        aware_radii=aware,
        blind_radii=blind,
        config=config,
    )
    aware_gates = torch.tensor(aware_diagnostics["gates_by_client"])
    dual_gates = torch.tensor(dual_diagnostics["gates_by_client"])
    assert torch.all(dual_gates <= aware_gates + 1.0e-7)


def test_reduced_full_matrix_reaches_gate_evaluator_without_holdout() -> None:
    config = _small_config()
    calibration, _ = k3._load_frozen_calibration(config)
    oracle._configure_runtime("cpu")
    rows = []
    for seed in config["randomness"]["development_seeds"]:
        for regime, permutation in k1._noise_cells(config):
            for geometry in config["cohort"]["honest_outliers"]["geometries"]:
                for threat in config["threats"]["names"]:
                    rows.extend(
                        k3._evaluate_cell(
                            config,
                            calibration,
                            seed=int(seed),
                            regime=regime,
                            permutation=str(permutation),
                            geometry=str(geometry),
                            threat=str(threat),
                        )
                    )
    k3._attach_baselines(rows)
    replace_rows = k3._replace_one_audit(config, calibration)
    decision = k3._evaluate_gates(rows, replace_rows, config)
    assert decision["observed"]["development_rows"] == 500
    assert decision["observed"]["expected_development_rows"] == 500
    assert decision["observed"]["fixed_global_cap_max_abs_difference_vs_fcc"] < 1e-6
    assert decision["observed"]["homogeneous_dual_max_abs_difference_vs_aware"] < 1e-6
    assert decision["observed"]["replace_one_trials"] == 10
    assert decision["observed"]["contribution_cap_violations"] == 0
    assert set(config["randomness"]["holdout_seeds"]).isdisjoint(
        config["randomness"]["development_seeds"]
    )
