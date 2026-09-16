from __future__ import annotations

from scripts import run_gaussian_aware_reference_g0g_k7_rcig_screen as k7
from scripts import run_gaussian_aware_reference_g0g_k7b_rcig_confirmation as runner


def test_protocol_is_valid_before_dependency_configuration() -> None:
    validation = runner._validate_protocol()
    assert all(validation["checks"].values())
    assert validation["expected_calibration_rows"] == 192
    assert validation["expected_null_validation_rows"] == 240
    assert validation["expected_evaluation_rows"] == 288


def test_all_confirmation_seeds_are_fresh_relative_to_k7() -> None:
    parent = set(
        k7.CALIBRATION_SEEDS
        + k7.NULL_VALIDATION_SEEDS_BY_REGIME["homogeneous"]
        + k7.NULL_VALIDATION_SEEDS_BY_REGIME["heteroscedastic"]
        + k7.EVALUATION_SEEDS
    )
    assert not parent & set(runner._all_seeds())


def test_locked_changes_are_precision_and_power_only() -> None:
    assert runner.PROTOCOL["innovation"]["threshold_quantile"] == 0.99
    assert runner.PROTOCOL["randomness"]["evaluation_outer_seeds"] == 48
    assert (
        runner.PROTOCOL["gaussian_specificity_gates"][
            "heteroscedastic_gain_vs_isotropic_one_sided_ci_low_min"
        ]
        == 0.0
    )
    assert (
        runner.PROTOCOL["gaussian_specificity_gates"][
            "heteroscedastic_gain_vs_euclidean_one_sided_ci_low_min"
        ]
        == -0.02
    )


def test_counterbalance_is_exact_over_48_seeds() -> None:
    identity_counts = [0] * 12
    tier_counts = [0] * 4
    for index, _seed in enumerate(runner.EVALUATION_SEEDS):
        slot = index % 12
        for offset in range(2):
            client = (10 + slot + offset) % 12
            identity_counts[client] += 1
            tier_counts[(client + 5 * slot) % 4] += 1
    assert identity_counts == [8] * 12
    assert tier_counts == [24] * 4
