from __future__ import annotations

import math

import pytest
import torch

from algorithms.gaussian_aware_reference_k6_tp_eiv import temporal_eiv_confidence
from algorithms.gaussian_aware_reference_k6b_coupled import (
    bounded_energy_magnitude,
    coupled_magnitude_confidence_predictor,
    regularized_radial_direction,
    temporal_projection_predictor,
)


def test_regularized_radial_direction_is_continuous_bounded_and_zero_safe() -> None:
    zero, zero_diagnostics = regularized_radial_direction(
        torch.zeros(2), minimum_direction_norm=0.1, return_diagnostics=True
    )
    inside = regularized_radial_direction(
        torch.tensor([0.03, 0.04]), minimum_direction_norm=0.1
    )
    outside = regularized_radial_direction(
        torch.tensor([0.3, 0.4]), minimum_direction_norm=0.1
    )
    assert torch.equal(zero, torch.zeros(2))
    assert torch.allclose(inside, torch.tensor([0.3, 0.4]))
    assert torch.allclose(outside, torch.tensor([0.6, 0.8]))
    assert zero_diagnostics["continuous"] is True
    assert zero_diagnostics["global_lipschitz_bound"] == 10.0


def test_regularized_magnitude_is_bounded_and_has_declared_lipschitz_bound() -> None:
    magnitude, diagnostics = bounded_energy_magnitude(
        0.04,
        influence_cap=0.13,
        b0=0.01,
        rule="regularized",
        return_diagnostics=True,
    )
    assert math.isclose(float(magnitude), 0.13, rel_tol=1.0e-6, abs_tol=1.0e-7)
    assert diagnostics["globally_lipschitz_in_energy"] is True
    assert diagnostics["global_lipschitz_bound_in_energy"] == 10.0
    assert float(bounded_energy_magnitude(-1.0, influence_cap=0.13, b0=0.01)) == 0.0


def test_square_root_ablation_is_continuous_but_not_globally_lipschitz() -> None:
    _, diagnostics = bounded_energy_magnitude(
        1.0e-8,
        influence_cap=1.0,
        b0=0.1,
        rule="sqrt",
        return_diagnostics=True,
    )
    ratio_small = math.sqrt(1.0e-8) / 1.0e-8
    ratio_smaller = math.sqrt(1.0e-12) / 1.0e-12
    assert ratio_smaller > ratio_small
    assert diagnostics["continuous"] is True
    assert diagnostics["globally_lipschitz_in_energy"] is False
    assert diagnostics["sqrt_ablation_holder_exponent"] == 0.5


def test_temporal_projection_matches_its_formula_and_respects_cap() -> None:
    newer = torch.tensor([0.06, 0.08], dtype=torch.float64)
    predictor, diagnostics = temporal_projection_predictor(
        newer,
        alignment=0.02,
        older_energy=0.01,
        alignment_threshold=0.0,
        energy_floor=0.001,
        influence_cap=0.13,
        minimum_direction_norm=1.0e-6,
        return_diagnostics=True,
    )
    # Unprojected coefficient is .2, so the final public-ball projection binds.
    assert torch.allclose(predictor, 0.13 * newer / torch.linalg.vector_norm(newer))
    assert diagnostics["projection_active"] is True
    assert diagnostics["norm_cap_certified"] is True
    assert diagnostics["local_dp_effect"].startswith("unchanged_post_processing")


def test_sqrt_factorization_equals_temporal_projection_away_from_floors() -> None:
    newer = torch.tensor([0.048, 0.064], dtype=torch.float64)  # norm .08
    alignment = torch.tensor(0.004, dtype=torch.float64)
    energy_x = torch.tensor(0.0025, dtype=torch.float64)
    energy_y = torch.tensor(0.0064, dtype=torch.float64)
    confidence = temporal_eiv_confidence(
        alignment, energy_x, energy_y, b_min=1.0e-5, tau_a=0.0
    )
    factorized = coupled_magnitude_confidence_predictor(
        newer,
        confidence,
        energy_y,
        influence_cap=0.13,
        minimum_direction_norm=1.0e-6,
        b0=1.0e-5,
        magnitude_rule="sqrt",
    )
    projected = temporal_projection_predictor(
        newer,
        alignment,
        energy_x,
        alignment_threshold=0.0,
        energy_floor=1.0e-5,
        influence_cap=0.13,
        minimum_direction_norm=1.0e-6,
    )
    assert torch.allclose(factorized, projected, atol=1.0e-12)


def test_coupled_predictor_caps_norm_for_extreme_energy() -> None:
    predictor = coupled_magnitude_confidence_predictor(
        torch.tensor([3.0, 4.0]),
        1.0,
        1.0e12,
        influence_cap=0.13,
        minimum_direction_norm=0.1,
        b0=0.01,
    )
    assert float(torch.linalg.vector_norm(predictor)) <= 0.13 + 1.0e-7


@pytest.mark.parametrize(
    ("argument", "match"),
    [
        ({"alignment_threshold": -1.0}, "alignment_threshold"),
        ({"energy_floor": 0.0}, "energy_floor"),
        ({"influence_cap": 0.0}, "influence_cap"),
    ],
)
def test_temporal_projection_rejects_invalid_public_constants(
    argument: dict[str, float], match: str
) -> None:
    kwargs = {
        "alignment_threshold": 0.0,
        "energy_floor": 0.01,
        "influence_cap": 0.13,
        "minimum_direction_norm": 1.0e-6,
        **argument,
    }
    with pytest.raises(ValueError, match=match):
        temporal_projection_predictor(torch.tensor([0.1, 0.0]), 0.1, 0.1, **kwargs)
