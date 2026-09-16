"""Contract tests for the development-only K6 TP-EIV microbench."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from scripts import run_gaussian_aware_reference_g0g_k6_tp_eiv_microbench as k6


def test_static_protocol_has_strictly_past_disjoint_views_and_new_seeds() -> None:
    result = k6._validate_protocol()
    assert all(result["checks"].values())
    assert result["expected_calibration_contexts"] == 24
    assert result["expected_null_validation_contexts"] == 24
    assert result["expected_evaluation_rows"] == 72
    timeline = k6.PROTOCOL["timeline"]
    assert max(timeline["gate_source_rounds"]) == 15
    assert timeline["gate_snapshot_round"] == 16
    assert timeline["older_view_rounds"] == [16, 17, 18, 19]
    assert timeline["newer_view_rounds"] == [20, 21, 22, 23]
    assert timeline["deployment_round"] == 24
    assert k6.PROTOCOL["scope"] == "M0_development_only_synthetic_mechanistic_screen"
    assert k6.PROTOCOL["cohort"] == {
        "num_clients": 12,
        "num_byzantine": 2,
        "dimension": 8,
    }
    assert "P0_n25" in k6.PROTOCOL["distinct_from"]
    assert k6.PROTOCOL["eiv"][
        "separate_equal_quantile_tau_for_corrected_and_uncorrected"
    ]
    assert not (set(k6.CALIBRATION_SEEDS) & set(k6.EVALUATION_SEEDS))
    assert not (set(k6.CALIBRATION_SEEDS) & set(k6.NULL_VALIDATION_SEEDS))
    assert not (set(k6.NULL_VALIDATION_SEEDS) & set(k6.EVALUATION_SEEDS))


def test_static_protocol_rejects_overlapping_windows() -> None:
    invalid = copy.deepcopy(k6.PROTOCOL)
    invalid["timeline"]["newer_view_rounds"] = [19, 20, 21, 22]
    with pytest.raises(ValueError, match="windows_disjoint"):
        k6._validate_protocol(invalid)


def test_piecewise_clip_jacobian_is_identity_inside_and_radial_outside() -> None:
    values = torch.tensor([[0.3, 0.4], [3.0, 4.0]], dtype=torch.float64)
    projected, jacobian, norms, margins = k6._clip_with_jacobian(values, 1.0)
    assert torch.allclose(projected[0], values[0])
    assert torch.allclose(jacobian[0], torch.eye(2, dtype=torch.float64))
    assert torch.allclose(projected[1], torch.tensor([0.6, 0.8], dtype=torch.float64))
    direction = torch.tensor([0.6, 0.8], dtype=torch.float64)
    expected = (
        torch.eye(2, dtype=torch.float64) - torch.outer(direction, direction)
    ) / 5
    assert torch.allclose(jacobian[1], expected)
    assert torch.allclose(norms, torch.tensor([0.5, 5.0], dtype=torch.float64))
    assert torch.allclose(margins, torch.tensor([0.5, 4.0], dtype=torch.float64))


def test_structured_covariance_matches_fixed_gate_unclipped_formula() -> None:
    n = int(k6.PROTOCOL["cohort"]["num_clients"])
    d = int(k6.PROTOCOL["cohort"]["dimension"])
    length = int(k6.PROTOCOL["timeline"]["window_length"])
    uploads = torch.zeros(length, n, d, dtype=torch.float32)
    coordinate_variance = torch.full_like(uploads, 0.04)
    gate = torch.ones(n, dtype=torch.float32)
    anchor = torch.zeros(d, dtype=torch.float32)
    _, covariance, diagnostics = k6._view_and_covariance(
        uploads, coordinate_variance, gate, anchor
    )
    # mean over L*n independent vectors: sigma^2/(L*n) per coordinate
    expected = torch.eye(d) * (0.04 / (length * n))
    assert torch.allclose(covariance, expected, atol=1.0e-8)
    assert diagnostics["covariance_block_count"] == length * n
    assert diagnostics["dense_joint_covariance_materialized"] is False
    assert diagnostics["covariance_psd"] is True


def test_predictable_gate_uses_only_rounds_one_to_fifteen() -> None:
    n = int(k6.PROTOCOL["cohort"]["num_clients"])
    d = int(k6.PROTOCOL["cohort"]["dimension"])
    history = torch.randn(15, n, d, generator=torch.Generator().manual_seed(7))
    gate, diagnostics = k6._predictable_gate(history)
    assert gate.shape == (n,)
    assert bool(((0.0 <= gate) & (gate <= 1.0)).all())
    assert diagnostics["source_round_max"] == 15
    assert diagnostics["snapshot_round"] == 16
    assert diagnostics["past_only"] is True
    assert diagnostics["common_across_views"] is True
    assert len(diagnostics["gate_hash"]) == 64


def test_summary_and_decision_use_paired_relative_mse() -> None:
    calibration = [
        {
            "seed": seed,
            "noise_regime": regime,
            "corrected_alignment": -1.0,
            "raw_alignment": -0.9,
            "device": "mps",
            "dtype": "torch.float32",
        }
        for seed in k6.CALIBRATION_SEEDS
        for regime in k6.NOISE_REGIMES
    ]
    rows = []
    for regime in k6.NOISE_REGIMES:
        for threat, gain in (
            ("none", -0.02),
            ("bitflip_x10", 0.2),
            ("model_replacement", 0.3),
        ):
            for seed in k6.EVALUATION_SEEDS:
                rows.append(
                    {
                        "seed": seed,
                        "noise_regime": regime,
                        "threat": threat,
                        "radial_mse": 1.0,
                        "k6_mse": 1.0 - gain,
                        "k6_uncorrected_mse": 0.95,
                        "k6_relative_mse_gain_vs_radial": gain,
                        "k6_corrected_relative_mse_gain_vs_uncorrected": 0.1,
                        "confidence": 0.5,
                        "uncorrected_confidence": 0.7,
                        "minimum_clip_margin": 0.1,
                        "covariance_psd": True,
                        "device": "mps",
                        "dtype": "torch.float32",
                        "gate_source_round_max": 15,
                        "older_rounds": "[16, 17, 18, 19]",
                        "views_disjoint": True,
                        "current_round_input_used": False,
                        "target_used_for_predictor": False,
                        "dense_joint_covariance_materialized": False,
                        "gate_hash": f"{seed}-{regime}",
                        "tau_a": 0.1,
                        "tau_a_corrected": 0.1,
                        "tau_a_uncorrected": 0.2,
                    }
                )
    summaries = k6._summary_rows(rows)
    null_validation = [
        {
            "seed": seed,
            "noise_regime": regime,
            "confidence": 0.0,
            "uncorrected_confidence": 0.0,
            "device": "mps",
            "dtype": "torch.float32",
        }
        for seed in k6.NULL_VALIDATION_SEEDS
        for regime in k6.NOISE_REGIMES
    ]
    decision = k6._decision(calibration, null_validation, rows, summaries)
    assert decision["all_checks_pass"] is True
    assert decision["decision"] == "advance_to_locked_k6_development"
    assert math.isclose(decision["attacked_relative_mse_gain"], 0.25)


def test_require_mps_rejects_enabled_or_unspecified_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    with pytest.raises(RuntimeError, match="FALLBACK=0"):
        k6._require_mps()
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(RuntimeError, match="FALLBACK=0"):
        k6._require_mps()


@pytest.mark.skipif(
    not (torch.backends.mps.is_built() and torch.backends.mps.is_available()),
    reason="MPS is unavailable in this test process",
)
def test_structured_covariance_stays_on_mps_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    device, dtype = k6._require_mps()
    n = int(k6.PROTOCOL["cohort"]["num_clients"])
    d = int(k6.PROTOCOL["cohort"]["dimension"])
    length = int(k6.PROTOCOL["timeline"]["window_length"])
    uploads = torch.zeros(length, n, d, device=device, dtype=dtype)
    variances = torch.full_like(uploads, 0.01)
    gate = torch.ones(n, device=device, dtype=dtype)
    anchor = torch.zeros(d, device=device, dtype=dtype)
    view, covariance, diagnostics = k6._view_and_covariance(
        uploads, variances, gate, anchor
    )
    torch.mps.synchronize()
    assert view.device.type == covariance.device.type == "mps"
    assert diagnostics["covariance_psd"] is True
