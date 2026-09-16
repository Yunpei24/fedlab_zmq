"""Reduced CPU tests for the preregistered G0g-K4-TCG runner."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch
import yaml

from scripts import run_gaussian_aware_reference_g0g_k4_tcg as k4
from scripts import run_gaussian_aware_reference_oracle as oracle

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k4_tcg.yaml"


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
    config["cohort"]["honest_outliers"].update(
        {"count": 1, "geometries": ["aligned"]}
    )
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
    config["temporal"]["standardized_clip_norm"] = 2.0 * math.sqrt(8.0)
    config["honest_dynamics"]["names"] = ["stationary"]
    config["temporal_calibration"].update(
        {
            "roots": [7101, 7117],
            "trajectories_per_root_per_context": 2,
            "expected_trajectories_per_context": 4,
            "expected_contexts": 1,
            "c0_order_statistic_one_indexed": 2,
            "c1_order_statistic_one_indexed": 4,
        }
    )
    config["threats"].update(
        {
            "names": ["alie", "ipm"],
            "separated_for_primary_gate": ["ipm"],
        }
    )
    config["randomness"].update(
        {
            "development_seeds": [7201, 7213],
            "replace_one_trials_per_seed_noise_cell": 1,
        }
    )
    oracle._configure_runtime("cpu")
    return config


def test_production_config_and_frozen_k2_hash(production_config: dict) -> None:
    k4._validate_config(production_config)
    _, provenance = k4._load_frozen_k2_calibration(production_config)
    assert provenance["sha256_verified"] is True
    assert provenance["observed_sha256"] == (
        "98a30cad8a2f92e6843346e9a9512bd321e8302b9b6e17b447b2641fffb544a5"
    )


def test_development_matrix_has_one_clean_control_per_base_context(
    reduced_config: dict,
) -> None:
    cells = k4._development_cells(reduced_config)
    clean = [cell for cell in cells if cell["schedule"] == "no_compromise"]
    attacked = [cell for cell in cells if cell["schedule"] != "no_compromise"]
    assert len(clean) == 2
    assert len(attacked) == 2 * 2 * 2
    assert all(cell["threat"] == "none" for cell in clean)
    assert len({cell["trajectory_id"] for cell in cells}) == len(cells)


def test_oracle_target_reintegrates_recovered_identities() -> None:
    clean = torch.tensor([[0.0], [2.0], [10.0]], dtype=torch.float64)
    latent = torch.tensor([False, False, True])
    assert float(k4._target_at_round(clean, latent, phase="attack")) == 1.0
    assert float(
        k4._target_at_round(
            clean,
            latent,
            phase="attack",
            compromised_during_attack_phase=False,
        )
    ) == pytest.approx(4.0)
    assert float(k4._target_at_round(clean, latent, phase="recovery")) == 4.0


def test_two_round_detection_cannot_end_before_round_15() -> None:
    gates = {
        13: torch.tensor([0.0]),
        14: torch.tensor([0.0]),
        15: torch.tensor([0.0]),
    }
    detected = k4._first_two_round_event(
        gates,
        client=0,
        first_end_round=15,
        last_end_round=17,
        threshold=0.5,
        direction="below",
    )
    assert detected == 15


def test_bounded_drift_has_zero_mean_and_respects_mahalanobis_cap(
    reduced_config: dict,
) -> None:
    reduced_config["honest_dynamics"]["names"] = ["bounded_drift"]
    regime = reduced_config["privacy_noise"]["regimes"][0]
    variances, _ = oracle._noise_variances(
        reduced_config, regime, "byzantine_high"
    )
    increments = k4._drift_increments(
        reduced_config,
        variances=variances,
        seed=19,
        geometry="aligned",
        dynamics="bounded_drift",
    )
    scales = k4._expand_block_values(
        (variances + reduced_config["references"]["variance_floor"]).sqrt(),
        reduced_config["cohort"]["block_sizes"],
    )
    norms = torch.linalg.vector_norm(increments / scales, dim=1)
    assert torch.allclose(
        increments.mean(dim=0), torch.zeros_like(increments[0]), atol=1.0e-12
    )
    assert float(norms.max()) <= 0.05 + 1.0e-12


def test_temporal_calibration_uses_trajectory_maxima_and_frozen_ranks(
    reduced_config: dict,
) -> None:
    calibration, rows = k4._calibrate_temporal(reduced_config)
    assert calibration["context_count"] == 1
    assert len(rows) == 4
    assert calibration["c0_rank_one_indexed"] == 2
    assert calibration["c1_rank_one_indexed"] == 4
    assert calibration["deployed_c1"] > calibration["deployed_c0"]
    assert calibration["maximum_domain"] == "all_clients_and_rounds_13_to_36"


def test_counterfactual_history_control_has_the_preregistered_causal_path(
    reduced_config: dict,
) -> None:
    k2_calibration, _ = k4._load_frozen_k2_calibration(reduced_config)
    temporal_calibration, _ = k4._calibrate_temporal(reduced_config)
    regime = reduced_config["privacy_noise"]["regimes"][0]
    _, alie_clients, alie_trajectories = k4._evaluate_trajectory(
        reduced_config,
        k2_calibration,
        temporal_calibration,
        seed=7201,
        regime=regime,
        permutation="byzantine_high",
        geometry="aligned",
        dynamics="stationary",
        threat="alie",
        schedule="persistent",
    )
    _, ipm_clients, ipm_trajectories = k4._evaluate_trajectory(
        reduced_config,
        k2_calibration,
        temporal_calibration,
        seed=7201,
        regime=regime,
        permutation="byzantine_high",
        geometry="aligned",
        dynamics="stationary",
        threat="ipm",
        schedule="persistent",
    )
    _, ipm_intermittent_clients, _ = k4._evaluate_trajectory(
        reduced_config,
        k2_calibration,
        temporal_calibration,
        seed=7201,
        regime=regime,
        permutation="byzantine_high",
        geometry="aligned",
        dynamics="stationary",
        threat="ipm",
        schedule="intermittent_2_on_1_off",
    )
    def history_path(rows: list[dict], field: str) -> dict[tuple[int, int], float]:
        return {
            (int(row["round"]), int(row["client"])): float(row[field])
            for row in rows
        }

    alie_control = history_path(alie_clients, "counterfactual_temporal_gate")
    ipm_control = history_path(ipm_clients, "counterfactual_temporal_gate")
    assert alie_control == ipm_control
    assert ipm_control == history_path(
        ipm_intermittent_clients, "counterfactual_temporal_gate"
    )
    assert history_path(
        alie_clients, "counterfactual_temporal_statistic"
    ) == history_path(ipm_clients, "counterfactual_temporal_statistic")

    for rows in (alie_clients, ipm_clients):
        at_13 = [row for row in rows if row["round"] == 13]
        assert all(
            float(row["temporal_statistic"])
            == float(row["counterfactual_temporal_statistic"])
            for row in at_13
        )
        assert all(
            float(row["temporal_gate"])
            == float(row["counterfactual_temporal_gate"])
            for row in at_13
        )
        at_29 = [row for row in rows if row["round"] == 29]
        assert all(
            float(row["temporal_statistic"])
            == float(row["counterfactual_temporal_statistic"])
            for row in at_29
        )
        assert all(
            float(row["temporal_gate"])
            == float(row["counterfactual_temporal_gate"])
            for row in at_29
        )

    ipm_after_delay = [row for row in ipm_clients if row["round"] == 14]
    assert any(
        not math.isclose(
            float(row["temporal_statistic"]),
            float(row["counterfactual_temporal_statistic"]),
            abs_tol=1.0e-7,
        )
        for row in ipm_after_delay
    )
    assert all(
        row["monitoring_auc"] >= 0.0 for row in alie_trajectories + ipm_trajectories
    )


def test_reduced_screen_emits_complete_raw_artifacts_and_gate_evidence(
    reduced_config: dict, tmp_path: Path
) -> None:
    k2_calibration, _ = k4._load_frozen_k2_calibration(reduced_config)
    temporal_calibration, _ = k4._calibrate_temporal(reduced_config)
    round_rows, client_rows, trajectory_rows = k4._screen_development(
        reduced_config, k2_calibration, temporal_calibration
    )
    expected_trajectories = len(k4._development_cells(reduced_config))
    assert len(round_rows) == expected_trajectories * 36 * len(k4.CANDIDATES)
    assert len(client_rows) == expected_trajectories * 24 * 6
    assert len(trajectory_rows) == expected_trajectories * len(k4.CANDIDATES)
    before_temporal = [
        row
        for row in round_rows
        if row["candidate"] == k4.PRIMARY and row["round"] <= 12
    ]
    assert max(float(row["difference_to_k2_aware"]) for row in before_temporal) == 0.0
    replace_rows = k4._replace_one_audit(
        reduced_config, k2_calibration, temporal_calibration
    )
    assert len(replace_rows) == 2
    decision = k4._evaluate_gates(
        round_rows,
        client_rows,
        trajectory_rows,
        replace_rows,
        reduced_config,
        device_name="cpu",
        temporal_c0=float(temporal_calibration["deployed_c0"]),
    )
    assert decision["checks"]["production_device"] is False
    assert decision["observed"]["complete_fraction"] == pytest.approx(1.0)
    assert decision["observed"]["finite_metric_fraction"] == pytest.approx(1.0)
    assert (
        decision["observed"]["identifier_completeness"][
            "exact_identifier_completeness"
        ]
        is True
    )
    assert decision["observed"]["replace_one_violations"] == 0
    assert decision["holdout_opened"] is False
    report = tmp_path / "report.md"
    k4._write_report(
        report,
        config=reduced_config,
        output_dir=ROOT / "results/ldp_gradient_far/test-only",
        decision=decision,
    )
    rendered = report.read_text(encoding="utf-8")
    assert "seul le facteur historique est mesurable par le passé" in rendered
    assert "Holdout ouvert : **non**" in rendered
