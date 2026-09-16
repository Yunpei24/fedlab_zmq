"""Unit tests for the development-only K6 radial prescreen."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from scripts import analyze_gaussian_aware_k6_radial_prescreen as screen
from scripts import audit_gaussian_aware_k6_radial_prescreen as independent_audit


def test_registered_gates_and_radial_floor_are_fixed_constants() -> None:
    assert screen.PREREGISTERED_SCIENTIFIC_GATES == {
        "gain_vs_k4b_mean_min": 0.10,
        "gain_vs_k4b_ci95_low_strictly_greater_than": 0.0,
        "gain_vs_k4_mean_min": 0.30,
        "gain_vs_k4_ci95_low_strictly_greater_than": 0.20,
        "relative_loss_vs_k5_1d_ci95_high_strictly_less_than": 0.05,
        "homogeneous_gain_vs_k4b_ci95_low_strictly_greater_than": 0.0,
        "heteroscedastic_gain_vs_k4b_ci95_low_strictly_greater_than": 0.0,
    }
    assert screen.RADIAL_MINIMUM_DIRECTION_NORM == 1.0e-8


def test_student_interval_uses_exactly_twelve_outer_seeds() -> None:
    interval = screen._student_ci([0.25] * 12, 2.2009851600916406)
    assert interval == {
        "n": 12,
        "mean": 0.25,
        "sd": 0.0,
        "low": 0.25,
        "high": 0.25,
    }
    with pytest.raises(ValueError, match="exactly 12"):
        screen._student_ci([0.25] * 11, 2.2009851600916406)


def test_integrated_row_computes_registered_ratios_from_sums() -> None:
    histories = ("h1", "h2")
    per_history = {
        (history, candidate): value
        for history in histories
        for candidate, value in {
            screen.k5.K4: 1.0,
            screen.k5.K4B: 0.8,
            screen.k5.K5_1D: 0.5,
            screen.RADIAL: 0.48,
            screen.k5.K4C: 0.2,
            screen.k5.POINTWISE: 0.1,
        }.items()
    }
    row = screen._integrated_row(
        seed=101, history_ids=histories, history_means=per_history
    )
    assert row["histories"] == 2
    assert math.isclose(row["gain_vs_k4b"], 0.4)
    assert math.isclose(row["gain_vs_k4"], 0.52)
    assert math.isclose(row["relative_loss_vs_k5_1d"], -0.04)
    assert math.isclose(row["ch_capture_fraction"], 0.65)
    assert math.isclose(row["remaining_mse_above_k4c_fraction_of_headroom"], 0.35)


def _passing_intervals() -> tuple[dict[str, dict[str, float]], dict]:
    overall = {
        "gain_vs_k4b": {"mean": 0.20, "low": 0.10, "high": 0.30},
        "gain_vs_k4": {"mean": 0.40, "low": 0.30, "high": 0.50},
        "relative_loss_vs_k5_1d": {
            "mean": 0.01,
            "low": -0.02,
            "high": 0.04,
        },
    }
    stratified = {
        "noise_regime": {
            "homogeneous": {"mean": 0.2, "low": 0.01, "high": 0.3},
            "heteroscedastic": {"mean": 0.2, "low": 0.01, "high": 0.3},
        }
    }
    return overall, stratified


def test_scientific_decision_requires_every_registered_gate() -> None:
    overall, stratified = _passing_intervals()
    checks, passed, decision = screen._scientific_decision(overall, stratified)
    assert all(checks.values())
    assert passed is True
    assert decision == "advance_radial_confidence_k6_development"

    overall["relative_loss_vs_k5_1d"]["high"] = 0.05
    checks, passed, decision = screen._scientific_decision(overall, stratified)
    assert checks["noninferior_to_k5_1d"] is False
    assert passed is False
    assert decision == "stop_unmodulated_radial_predictor_instance"


def test_control_csv_loader_requires_one_complete_candidate_matrix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "children.csv"
    fieldnames = [
        "history_id",
        "seed",
        "noise_regime",
        "noise_permutation",
        "outlier_geometry",
        "honest_dynamics",
        "threat",
        "assessment_round",
        "evaluation_child",
        "evaluation_child_seed",
        "candidate",
        "squared_reference_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, candidate in enumerate(screen.ALL_SOURCE_CANDIDATES):
            writer.writerow(
                {
                    "history_id": "evaluation|seed|cell|17",
                    "seed": 101,
                    "noise_regime": "homogeneous",
                    "noise_permutation": "identity",
                    "outlier_geometry": "aligned",
                    "honest_dynamics": "stationary",
                    "threat": "bitflip_x10",
                    "assessment_round": 17,
                    "evaluation_child": 0,
                    "evaluation_child_seed": 12345,
                    "candidate": candidate,
                    "squared_reference_error": 0.01 + 0.001 * index,
                }
            )
    controls, metadata, counts = screen._load_control_rows(
        path, expected_histories=1, children=1
    )
    key = ("evaluation|seed|cell|17", 0)
    assert set(controls[key]) == set(screen.CONTROL_CANDIDATES)
    assert metadata[key]["evaluation_child_seed"] == "12345"
    assert counts == {candidate: 1 for candidate in screen.ALL_SOURCE_CANDIDATES}


def test_post_run_provenance_extension_is_explicit_and_complete() -> None:
    provenance = independent_audit._post_run_provenance_extension(
        independent_audit.DEFAULT_SCREEN
    )
    assert provenance["status"] == "post_run_only_not_preregistered"
    assert "not present" in provenance["interpretation"]
    assert provenance["excluded_self_referential_output"] == (
        "independent_postrun_audit.json"
    )
    assert set(provenance["scientific_output_sha256"]) == set(
        independent_audit.POST_RUN_SCIENTIFIC_OUTPUTS
    )
    assert set(provenance["implementation_sha256"]) == set(
        independent_audit.POST_RUN_IMPLEMENTATION_FILES
    )
    assert all(
        len(digest) == 64
        for group in (
            provenance["scientific_output_sha256"],
            provenance["implementation_sha256"],
        )
        for digest in group.values()
    )
