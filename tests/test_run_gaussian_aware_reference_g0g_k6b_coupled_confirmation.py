from __future__ import annotations

import math

import pytest
import torch

from scripts import run_gaussian_aware_reference_g0g_k6b_coupled_confirmation as k6b


def test_static_protocol_is_independent_counterbalanced_and_fail_closed() -> None:
    result = k6b._validate_protocol()
    assert all(result["checks"].values())
    assert result["expected_calibration_rows"] == 24
    assert result["expected_null_validation_rows"] == 72
    assert result["expected_evaluation_rows"] == 72
    assert result["expected_mc_calibration_rows"] == 168
    assert result["seed_collision_scan"]["unexpected_matching_paths"] == []
    assert k6b.PROTOCOL["decision_policy"]["p0_original_fixed_g"] == "stopped"
    assert (
        k6b.PROTOCOL["decision_policy"]["p0b_amplitude_only_promotion"] == "forbidden"
    )
    assert k6b.PROTOCOL["candidates"]["primary"] == "pt_eiv"
    assert "identity_y" in k6b.CANDIDATES


def test_counterbalance_exactly_balances_attacker_identity_and_noise_tier() -> None:
    schedule = k6b._counterbalance_schedule()
    identity_counts = [0] * 12
    tier_counts = [0] * 4
    for row in schedule:
        for identity in row["attacker_clients"]:
            identity_counts[identity] += 1
        for tier in row["attacker_tiers"]:
            tier_counts[tier] += 1
    assert identity_counts == [2] * 12
    assert tier_counts == [6] * 4
    assert len({row["seed"] for row in schedule}) == 12


def test_byzantine_replacement_keeps_nominal_public_covariance() -> None:
    uploads = torch.ones(24, 12, 8)
    clean = uploads.clone()
    variances = (
        torch.linspace(0.001, 0.012, 12)[None, :, None].expand_as(uploads).clone()
    )
    honest = torch.ones(12, dtype=torch.bool)
    honest[[2, 7]] = False
    _, after, active = k6b._apply_threat_without_covariance_oracle(
        uploads,
        clean,
        variances,
        honest,
        seed=k6b.EVALUATION_SEEDS[0],
        threat="model_replacement",
        device=torch.device("cpu"),
    )
    assert torch.equal(after, variances)
    assert torch.equal(active, ~honest)


def test_clopper_pearson_zero_over_36_passes_stratified_bonferroni_gate() -> None:
    upper = k6b._clopper_pearson_upper(0, 36, one_sided_alpha=0.025)
    expected = 1.0 - 0.025 ** (1.0 / 36.0)
    assert math.isclose(upper, expected, rel_tol=0.0, abs_tol=1.0e-12)
    assert upper < 0.10
    assert k6b._clopper_pearson_upper(1, 36, one_sided_alpha=0.025) > 0.10


def test_collinear_mse_reconstruction_is_exact() -> None:
    dimension = 8
    cap = 0.13
    h = torch.tensor([0.6, 0.8] + [0.0] * 6, dtype=torch.float64)
    target = torch.tensor([0.04, 0.03] + [0.0] * 6, dtype=torch.float64)
    radial = cap * h
    candidate_coefficient = 0.075
    candidate = candidate_coefficient * h
    radial_mse = float(torch.mean((radial - target).square()))
    reconstructed = k6b._mse_reconstruction(
        radial_mse=radial_mse,
        radial_norm=float(torch.linalg.vector_norm(radial)),
        target_norm=float(torch.linalg.vector_norm(target)),
        influence_cap=cap,
        coefficient_on_h=candidate_coefficient,
        h_norm=1.0,
        dimension=dimension,
    )
    assert math.isclose(
        reconstructed,
        float(torch.mean((candidate - target).square())),
        rel_tol=0.0,
        abs_tol=1.0e-14,
    )


def test_null_seed_registries_are_distinct_across_regimes() -> None:
    homogeneous = set(k6b.NULL_VALIDATION_SEEDS_BY_REGIME["homogeneous"])
    heteroscedastic = set(k6b.NULL_VALIDATION_SEEDS_BY_REGIME["heteroscedastic"])
    assert len(homogeneous) == len(heteroscedastic) == 36
    assert not homogeneous & heteroscedastic
    assert not (homogeneous | heteroscedastic) & set(k6b.EVALUATION_SEEDS)


def test_require_mps_rejects_unspecified_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    with pytest.raises(RuntimeError, match="FALLBACK=0"):
        k6b._require_mps()


@pytest.mark.skipif(
    not (torch.backends.mps.is_built() and torch.backends.mps.is_available()),
    reason="MPS is unavailable in this test process",
)
def test_primary_primitive_stays_on_mps_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    device, dtype = k6b._require_mps()
    from algorithms.gaussian_aware_reference_k6b_coupled import (
        temporal_projection_predictor,
    )

    result = temporal_projection_predictor(
        torch.tensor([0.05, 0.02], device=device, dtype=dtype),
        0.01,
        0.02,
        alignment_threshold=0.001,
        energy_floor=1.0e-5,
        influence_cap=0.13,
        minimum_direction_norm=1.0e-6,
    )
    torch.mps.synchronize()
    assert result.device.type == "mps"
