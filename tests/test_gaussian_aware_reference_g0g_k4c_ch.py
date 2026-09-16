"""Unit tests for the G0g-K4c-CH semi-oracle primitives."""

from __future__ import annotations

import pytest
import torch

from algorithms.gaussian_aware_reference_k4c_ch import (
    current_randomness_conditional_mse_semi_oracle,
    fixed_denominator_imputed_reference_from_sum,
)


def test_semi_oracle_matches_unprojected_conditional_mse_formula() -> None:
    targets = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64)
    direct = torch.tensor([[0.2, 0.1], [0.4, 0.3]], dtype=torch.float64)
    predictor, diagnostics = current_randomness_conditional_mse_semi_oracle(
        targets,
        direct,
        missing_slot_mass=2.0,
        num_clients=5,
        influence_cap=1.0,
        return_diagnostics=True,
    )
    expected = (5.0 * targets.mean(dim=0) - direct.mean(dim=0)) / 2.0
    assert torch.allclose(predictor, expected, atol=1.0e-12, rtol=0.0)
    assert diagnostics["projection_applied_once_after_expectations"] is True
    assert diagnostics["pointwise_oracles_averaged"] is False
    assert diagnostics["conditions_on_latent_clean_current_state"] is True
    assert diagnostics["measurable_with_respect_to_observable_past_only"] is False


def test_projection_occurs_after_expectations_not_before() -> None:
    targets = torch.tensor([[2.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    direct = torch.zeros_like(targets)
    predictor = current_randomness_conditional_mse_semi_oracle(
        targets,
        direct,
        missing_slot_mass=1.0,
        num_clients=1,
        influence_cap=1.0,
    )
    # Proj(mean([2,0],[0,0]))=[1,0], whereas mean(Proj([2,0]),Proj([0,0]))
    # would be [0.5,0].  This distinguishes the preregistered estimand.
    assert torch.equal(predictor, torch.tensor([1.0, 0.0], dtype=torch.float64))


def test_zero_missing_mass_returns_zero_predictor() -> None:
    targets = torch.tensor([[0.2], [0.4]], dtype=torch.float32)
    direct = torch.tensor([[0.1], [0.3]], dtype=torch.float32)
    predictor = current_randomness_conditional_mse_semi_oracle(
        targets,
        direct,
        missing_slot_mass=0.0,
        num_clients=4,
        influence_cap=0.13,
    )
    assert torch.equal(predictor, torch.zeros_like(predictor))


def test_small_positive_mass_uses_same_64_eps_threshold_as_pointwise() -> None:
    targets = torch.tensor([[1.0e-6], [1.0e-6]], dtype=torch.float32)
    direct = torch.zeros_like(targets)
    mass = 1.0e-5
    assert mass > 64.0 * torch.finfo(torch.float32).eps
    # This mass is below the rejected historical 64*eps*n convention for n=25.
    assert mass < 64.0 * torch.finfo(torch.float32).eps * 25.0
    predictor = current_randomness_conditional_mse_semi_oracle(
        targets,
        direct,
        missing_slot_mass=mass,
        num_clients=25,
        influence_cap=10.0,
    )
    assert predictor.item() == pytest.approx(2.5)


def test_fixed_denominator_reference_clips_predictor_and_divides_by_n() -> None:
    anchor = torch.tensor([0.1, -0.2], dtype=torch.float64)
    direct = torch.tensor([0.3, 0.1], dtype=torch.float64)
    reference = fixed_denominator_imputed_reference_from_sum(
        anchor=anchor,
        direct_sum=direct,
        predictor=torch.tensor([3.0, 4.0], dtype=torch.float64),
        missing_slot_mass=2.0,
        num_clients=5,
        influence_cap=1.0,
    )
    bounded = torch.tensor([0.6, 0.8], dtype=torch.float64)
    expected = anchor + (direct + 2.0 * bounded) / 5.0
    assert torch.allclose(reference, expected, atol=1.0e-12, rtol=0.0)


@pytest.mark.parametrize(
    ("targets", "direct", "mass", "n", "cap"),
    [
        (torch.ones(2), torch.ones((2, 1)), 1.0, 2, 1.0),
        (torch.ones((2, 1)), torch.ones((3, 1)), 1.0, 2, 1.0),
        (torch.ones((2, 1)), torch.ones((2, 1)), -1.0, 2, 1.0),
        (torch.ones((2, 1)), torch.ones((2, 1)), 1.0, 0, 1.0),
        (torch.ones((2, 1)), torch.ones((2, 1)), 1.0, 2, 0.0),
    ],
)
def test_semi_oracle_rejects_invalid_inputs(
    targets: torch.Tensor,
    direct: torch.Tensor,
    mass: float,
    n: int,
    cap: float,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        current_randomness_conditional_mse_semi_oracle(
            targets,
            direct,
            missing_slot_mass=mass,
            num_clients=n,
            influence_cap=cap,
        )
