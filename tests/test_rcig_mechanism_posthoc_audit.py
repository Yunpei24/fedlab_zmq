"""Arithmetic/chronology tests for the offline audit, no training or torch."""

import math

import pytest

from scripts.audit_rcig_mechanism_posthoc import (
    auc,
    covariance_parts,
    exposure,
    prefix_pairing_audit,
    quantile,
    stats,
    summarize,
)


@pytest.mark.parametrize("accuracy_shift,expected_first", [(0.0, None), (0.25, 2)])
def test_prefix_pairing_checks_metrics_separately_from_history(
    accuracy_shift, expected_first
):
    def record(scenario, shift, stat_shift):
        return {
            "phase": "R3",
            "noise": "homogeneous",
            "seed": 1,
            "scenario": scenario,
            "prefix_metrics": [
                {
                    "round": t,
                    "test_accuracy_pct": 40.0 + (shift if t >= 2 else 0.0),
                    "test_loss": 1.5,
                }
                for t in range(1, 17)
            ],
            "rows": [{"full_stat": 2.0 + stat_shift} for _ in range(5)],
        }

    audit = prefix_pairing_audit(
        [record("none", 0.0, 0.0), record("bf", accuracy_shift, 0.125)]
    )
    assert audit["n_pairs"] == 1
    assert audit["n_with_pre_attack_difference"] == int(expected_first is not None)
    assert audit["pairs"][0]["first_global_metric_difference_round"] == expected_first
    assert audit["pairs"][0]["round1_metrics_identical"]
    assert audit["max_accuracy_difference_pp"] == pytest.approx(accuracy_shift)
    assert audit["max_loss_difference"] == 0.0
    assert audit["max_stat_difference"] == 0.125


@pytest.mark.parametrize(
    "positive,negative,expected",
    [
        ([2, 3], [0, 1], 1.0),
        ([0, 1], [2, 3], 0.0),
        ([1, 1], [1, 1], 0.5),
        ([0, 1], [0, 1], 0.5),
        ([1, 2], [0, 2], 0.625),
        ([], [1], None),
    ],
)
def test_auc_direction_and_ties(positive, negative, expected):
    assert auc(positive, negative) == expected


def test_covariance_reconstruction_known_isotropic():
    # Total scalar variance = DP .000078 + process .000001 + ridge .000001.
    norm, total = 0.08, 0.00008
    parts = covariance_parts(norm, norm / math.sqrt(total), 1e-6, 1e-6)
    assert parts["total_isotropic_variance"] == pytest.approx(total)
    assert parts["dp_proxy_average_variance"] == pytest.approx(0.000078)
    assert parts["floor_fraction"] == pytest.approx(0.025)


def test_covariance_zero_and_impossible_inputs():
    assert covariance_parts(0, 0, 1e-6, 1e-6) is None
    with pytest.raises(ValueError):
        covariance_parts(0.001, 10, 1e-6, 1e-6)


@pytest.mark.parametrize(
    "public_t,old,new",
    [(17, 0.0, 0.0), (18, 0.0, 0.25), (21, 0.0, 1.0), (22, 0.25, 1.0), (25, 1.0, 1.0)],
)
def test_history_exposure_uses_zero_based_stored_bounds(public_t, old, new):
    assert exposure(public_t - 9, public_t - 6, 17, 40) == old
    assert exposure(public_t - 5, public_t - 2, 17, 40) == new
    assert exposure(0, 3, None, None) == 0.0


def test_summary_sd_is_between_values_and_nan_missing():
    result = stats([1.0, 2.0, 3.0, None, float("nan")])
    assert result["n"] == 3
    assert result["mean"] == 2.0
    assert result["sd"] == 1.0
    assert quantile([1.0, 2.0, 3.0, 4.0], 0.25) == 1.75


def test_candidate_and_deployed_error_not_conflated():
    def point(round_num, active, reference_error):
        return {
            "round": round_num,
            "attack_current": True,
            "newer_exposed_fraction": 1.0,
            "older_exposed_fraction": 1.0,
            "full_ratio": 1.1 if active else 0.9,
            "active": active,
            "frozen_after": True,
            "frozen_before": not active,
            "action": "freeze_on_activation" if active else "remain_frozen",
            "error_identity_new": 2.0,
            "error_identity_old": 2.2,
            "error_full": 1.9 if active else 2.0,
            "error_reference": reference_error,
            "error_midpoint": 1.0,
        }

    records = [
        {
            "phase": "R2",
            "noise": "homogeneous",
            "scenario": "bf",
            "timing": "after_clean",
            "seed": 1,
            "run": "fixture",
            "rows": [point(26, True, 2.1), point(27, False, 2.2)],
        }
    ]
    _, errors, events, _, _ = summarize(records)
    active = next(e for e in errors if e["selection"] == "rolling_active")
    assert active["event_rows"] == 1
    assert active["full"]["event_weighted_mean_difference"] == pytest.approx(-0.1)
    assert active["reference"]["event_weighted_mean_difference"] == pytest.approx(0.1)
    frozen = next(e for e in errors if e["selection"] == "deployed_frozen")
    assert frozen["event_rows"] == 2
    assert frozen["reference"]["event_weighted_mean_difference"] == pytest.approx(0.15)
    assert len(events) == 2
