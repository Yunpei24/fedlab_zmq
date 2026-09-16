from __future__ import annotations

import math

import pytest
import torch

from algorithms.gaussian_aware_reference_k7_rcig import (
    euclidean_innovation_fusion,
    robust_covariance_innovation_fusion,
)


def _covariance(dimension: int, scale: float = 0.01) -> torch.Tensor:
    return scale * torch.eye(dimension, dtype=torch.float32)


def test_ordinary_innovation_returns_newer_view_exactly() -> None:
    older = torch.tensor([0.01, -0.01], dtype=torch.float32)
    newer = torch.tensor([0.012, -0.008], dtype=torch.float32)
    value, diagnostics = robust_covariance_innovation_fusion(
        older,
        newer,
        _covariance(2),
        _covariance(2),
        process_variance=0.001,
        ridge=1.0e-6,
        innovation_threshold=4.0,
        influence_cap=0.2,
        return_diagnostics=True,
    )
    assert torch.equal(value, newer)
    assert diagnostics["newer_view_trust"] == 1.0
    assert diagnostics["gate_active"] is False
    assert diagnostics["ordinary_innovation_returns_identity_y"] is True


def test_large_innovation_is_shrunk_towards_older_view() -> None:
    older = torch.tensor([0.02, 0.0], dtype=torch.float32)
    newer = torch.tensor([-0.18, 0.0], dtype=torch.float32)
    value, diagnostics = robust_covariance_innovation_fusion(
        older,
        newer,
        _covariance(2, 1.0e-4),
        _covariance(2, 1.0e-4),
        process_variance=1.0e-5,
        ridge=1.0e-8,
        innovation_threshold=3.0,
        influence_cap=0.2,
        return_diagnostics=True,
    )
    assert diagnostics["gate_active"] is True
    assert 0.0 < diagnostics["newer_view_trust"] < 1.0
    assert torch.linalg.vector_norm(value - older) < torch.linalg.vector_norm(
        newer - older
    )


@pytest.mark.parametrize("mode", ["full", "isotropic"])
def test_norm_cap_is_enforced(mode: str) -> None:
    older = torch.tensor([3.0, 4.0], dtype=torch.float32)
    newer = torch.tensor([-4.0, 3.0], dtype=torch.float32)
    value = robust_covariance_innovation_fusion(
        older,
        newer,
        _covariance(2),
        _covariance(2),
        process_variance=0.0,
        ridge=1.0e-6,
        innovation_threshold=2.0,
        influence_cap=0.13,
        covariance_mode=mode,  # type: ignore[arg-type]
    )
    assert float(torch.linalg.vector_norm(value)) <= 0.13 + 1.0e-6


def test_anisotropic_covariance_changes_standardised_gate() -> None:
    older = torch.zeros(2, dtype=torch.float32)
    newer = torch.tensor([0.05, 0.0], dtype=torch.float32)
    anisotropic = torch.diag(torch.tensor([0.02, 0.0001]))
    _, full = robust_covariance_innovation_fusion(
        older,
        newer,
        anisotropic,
        anisotropic,
        process_variance=0.0,
        ridge=1.0e-7,
        innovation_threshold=1.0,
        influence_cap=0.2,
        covariance_mode="full",
        return_diagnostics=True,
    )
    _, isotropic = robust_covariance_innovation_fusion(
        older,
        newer,
        anisotropic,
        anisotropic,
        process_variance=0.0,
        ridge=1.0e-7,
        innovation_threshold=1.0,
        influence_cap=0.2,
        covariance_mode="isotropic",
        return_diagnostics=True,
    )
    assert full["standardized_innovation"] < isotropic["standardized_innovation"]


def test_euclidean_control_matches_identity_below_threshold() -> None:
    older = torch.tensor([0.01, 0.0], dtype=torch.float32)
    newer = torch.tensor([0.02, 0.0], dtype=torch.float32)
    value, diagnostics = euclidean_innovation_fusion(
        older,
        newer,
        innovation_threshold=0.02,
        influence_cap=0.2,
        return_diagnostics=True,
    )
    assert torch.equal(value, newer)
    assert diagnostics["gate_active"] is False


@pytest.mark.parametrize(
    "bad_key,bad_value",
    [
        ("process_variance", -1.0),
        ("ridge", 0.0),
        ("innovation_threshold", 0.0),
        ("influence_cap", float("nan")),
    ],
)
def test_invalid_scalar_parameters_fail_closed(bad_key: str, bad_value: float) -> None:
    kwargs = {
        "process_variance": 0.0,
        "ridge": 1.0e-6,
        "innovation_threshold": 3.0,
        "influence_cap": 0.2,
    }
    kwargs[bad_key] = bad_value
    with pytest.raises(ValueError):
        robust_covariance_innovation_fusion(
            torch.zeros(2),
            torch.ones(2),
            _covariance(2),
            _covariance(2),
            **kwargs,
        )


def test_condition_number_certificate_is_finite() -> None:
    _, diagnostics = robust_covariance_innovation_fusion(
        torch.zeros(3),
        torch.ones(3) * 0.01,
        _covariance(3),
        _covariance(3),
        process_variance=1.0e-4,
        ridge=1.0e-6,
        innovation_threshold=2.0,
        influence_cap=0.2,
        return_diagnostics=True,
    )
    assert math.isfinite(diagnostics["condition_number_upper_bound"])
    assert diagnostics["condition_number_upper_bound"] >= 1.0
