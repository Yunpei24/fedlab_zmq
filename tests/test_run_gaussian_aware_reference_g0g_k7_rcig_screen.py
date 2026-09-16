from __future__ import annotations

import math

from scripts import run_gaussian_aware_reference_g0g_k7_rcig_screen as runner


def test_protocol_is_static_and_consistent() -> None:
    validation = runner._validate_protocol()
    assert all(validation["checks"].values())
    assert validation["expected_calibration_rows"] == 96
    assert validation["expected_null_validation_rows"] == 144
    assert validation["expected_evaluation_rows"] == 144
    assert runner.PROTOCOL["covariance_policy"]["retained_for_attacked_identities"]
    assert not runner.PROTOCOL["covariance_policy"][
        "byzantine_mask_used_to_modify_covariance"
    ]


def test_registered_seed_families_are_disjoint() -> None:
    calibration = set(runner.CALIBRATION_SEEDS)
    homogeneous = set(runner.NULL_VALIDATION_SEEDS_BY_REGIME["homogeneous"])
    heteroscedastic = set(runner.NULL_VALIDATION_SEEDS_BY_REGIME["heteroscedastic"])
    evaluation = set(runner.EVALUATION_SEEDS)
    assert not calibration & homogeneous
    assert not calibration & heteroscedastic
    assert not calibration & evaluation
    assert not homogeneous & heteroscedastic
    assert not homogeneous & evaluation
    assert not heteroscedastic & evaluation


def test_counterbalancing_rotates_each_identity_and_noise_tier() -> None:
    identity_counts = [0] * 12
    tier_counts = [0] * 4
    for seed in runner.EVALUATION_SEEDS:
        slot = runner._phase_slot(seed, "evaluation", "heteroscedastic")
        attackers = sorted((10 + slot + offset) % 12 for offset in range(2))
        for client in attackers:
            identity_counts[client] += 1
            tier_counts[(client + 5 * slot) % 4] += 1
    assert identity_counts == [4] * 12
    assert tier_counts == [12] * 4


def test_cp_upper_bound_matches_zero_success_closed_form() -> None:
    observed = runner._clopper_pearson_upper(0, 72, one_sided_alpha=0.025)
    expected = 1.0 - 0.025 ** (1.0 / 72.0)
    assert math.isclose(observed, expected, rel_tol=1.0e-10)
    assert observed < 0.10


def test_primary_control_and_decision_are_fail_closed() -> None:
    assert runner.CANDIDATES[0] == "identity_y"
    policy = runner.PROTOCOL["decision_policy"]
    assert policy["invalid_if_any_validity_gate_fails"]
    assert policy["stop_if_primary_science_gate_fails"]
    assert policy["advance_gaussian_aware_only_if_all_gates_pass"]
