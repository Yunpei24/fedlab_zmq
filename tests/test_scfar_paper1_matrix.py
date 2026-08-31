"""Protocol-lock tests for the full-update SC-FAR-DP paper-1 matrices."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from scripts.run_scfar_paper1 import (
    ROOT,
    alpha_max,
    expand_tasks,
    load_matrix,
    validate_matrix,
)
from scripts.validate_scfar_gaussian_accountant import (
    DEFAULT_ORDERS,
    direct_epsilon,
)
from privacy.rdp import RDPAccountant


MATRIX_DIR = ROOT / "configs" / "scpfar" / "paper1"
MATRICES = {
    "s1_reference_tradeoff.yaml": 1908,
    "s2_full_update_ablations.yaml": 360,
    "s3_inclusion_attacks.yaml": 1224,
    "s4_central_dp.yaml": 720,
}


@pytest.mark.parametrize(("filename", "expected"), MATRICES.items())
def test_matrix_is_valid_unique_and_has_frozen_cardinality(filename: str, expected: int):
    document = load_matrix(MATRIX_DIR / filename)
    assert validate_matrix(document) == []
    tasks = expand_tasks(document, output_root=Path("/tmp/scfar-paper1-test"))
    assert len(tasks) == expected
    assert len({task.task_id for task in tasks}) == expected


@pytest.mark.parametrize("filename", MATRICES)
def test_every_paper1_task_is_full_update_full_participation(filename: str):
    document = load_matrix(MATRIX_DIR / filename)
    tasks = expand_tasks(document, output_root=Path("/tmp/scfar-paper1-test"))
    for task in tasks:
        cfg = task.config
        clients = cfg["clients"]
        algo = cfg["training"]["algo_config"]
        assert clients["sample_fraction"] == 1.0
        assert clients["min_clients"] == clients["num_clients"] == 25
        assert clients["dropout_rate"] == 0.0
        assert cfg["data"]["partition"] == "client_dirichlet_balanced"
        assert not {"num_layer_groups", "layer_selection", "rounds_per_layer"}.intersection(algo)
        if cfg["training"]["algorithm"] == "scfar_dp":
            assert algo["alpha_bound_policy"] == "error"
            assert algo["privacy_num_rounds"] == cfg["training"]["num_rounds"]
            assert set(algo["honest_outlier_client_ids"]).isdisjoint(
                set(algo["attack"].get("client_ids", []))
            )


def test_s3_tilts_are_derived_inside_the_certified_region():
    document = load_matrix(MATRIX_DIR / "s3_inclusion_attacks.yaml")
    tasks = expand_tasks(document, output_root=Path("/tmp/scfar-paper1-test"))
    for task in tasks:
        algo = task.config["training"]["algo_config"]
        kappa = float(algo["kappa_w"])
        requested = float(algo["far_alpha"])
        maximum = alpha_max(25, kappa)
        assert 0.0 <= requested <= maximum + 1e-12
        assert math.isclose(
            requested,
            float(algo["alpha_fraction_of_max"]) * maximum,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


def test_s4_infinity_lane_really_disables_noise_and_finite_lanes_enable_it():
    document = load_matrix(MATRIX_DIR / "s4_central_dp.yaml")
    tasks = expand_tasks(document, output_root=Path("/tmp/scfar-paper1-test"))
    for task in tasks:
        algo = task.config["training"]["algo_config"]
        if task.privacy_id == "no_dp":
            assert algo["enable_central_dp"] is False
            assert algo["central_noise_multiplier"] == 0.0
            assert algo["target_epsilon"] is None
        else:
            assert algo["enable_central_dp"] is True
            assert float(algo["target_epsilon"]) in {1.0, 3.0, 6.0, 10.0}


def test_honest_outliers_are_preregistered_and_exclude_byzantines():
    document = load_matrix(MATRIX_DIR / "s3_inclusion_attacks.yaml")
    tasks = expand_tasks(document, output_root=Path("/tmp/scfar-paper1-test"))
    for task in tasks:
        algo = task.config["training"]["algo_config"]
        outliers = set(algo["honest_outlier_client_ids"])
        byzantines = set(algo["attack"].get("client_ids", []))
        assert len(outliers) == 4
        assert max(outliers) < 20
        assert outliers.isdisjoint(byzantines)


def test_direct_q_one_accountant_crosscheck_matches_the_ledger():
    sigma, steps, delta = 4.25, 100, 1e-5
    direct, direct_order, direct_rdp = direct_epsilon(
        noise_multiplier=sigma, steps=steps, delta=delta
    )
    accountant = RDPAccountant(orders=DEFAULT_ORDERS)
    accountant.add_gaussian(
        channel="central_model", noise_multiplier=sigma, steps=steps
    )
    epsilon, order = accountant.epsilon(delta)
    assert epsilon == pytest.approx(direct, abs=1e-12)
    assert order == direct_order
    assert accountant.total_rdp() == pytest.approx(direct_rdp, abs=1e-12)
