"""Unit and certificate tests for the G0e bounded reference correction."""

from __future__ import annotations

import pytest
import torch

from algorithms.gaussian_aware_reference import (
    gaussian_aware_crossfit_bounded_correction,
)
from robustness.aggregators import (
    centered_clipping,
    centered_clipping_leave_one_out,
)


def _basic_inputs(dtype: torch.dtype = torch.float64):
    vectors = torch.tensor(
        [[0.20, -0.10], [-0.10, 0.15], [0.05, 0.05], [0.10, -0.05]],
        dtype=dtype,
    )
    pilot = centered_clipping(
        vectors, anchor=torch.zeros(2, dtype=dtype), tau=0.3
    )
    crossfit = centered_clipping_leave_one_out(
        vectors, anchor=torch.zeros(2, dtype=dtype), tau=0.3
    )
    return vectors, pilot, crossfit


def test_zero_correction_budget_returns_pilot_exactly():
    vectors, pilot, crossfit = _basic_inputs()
    result, diagnostics = gaussian_aware_crossfit_bounded_correction(
        vectors,
        pilot=pilot,
        crossfit_references=crossfit,
        statistical_radii=torch.full((4, 1), 0.18, dtype=torch.float64),
        deployed_radii=torch.full((4, 1), 0.2, dtype=torch.float64),
        pilot_replace_one_bound=0.15,
        influence_cap=0.1,
        regularization=0.5,
        correction_budget=0.0,
        num_steps=5,
        return_diagnostics=True,
    )
    assert torch.equal(result, pilot)
    assert diagnostics["beta"] == 0.0
    assert diagnostics["observed_correction_norm"] == 0.0
    assert diagnostics["correction_budget_respected"] is True


def test_beta_is_derived_from_public_budget_and_bounds_correction():
    vectors = torch.full((6, 2), 10.0, dtype=torch.float64)
    pilot = torch.zeros(2, dtype=torch.float64)
    crossfit = torch.zeros_like(vectors)
    budget = 0.03
    gamma = 0.5
    caps = torch.tensor([0.12, 0.16], dtype=torch.float64)
    complete_cap = 0.2
    result, diagnostics = gaussian_aware_crossfit_bounded_correction(
        vectors,
        pilot=pilot,
        crossfit_references=crossfit,
        statistical_radii=torch.full((6, 2), 4.9, dtype=torch.float64),
        deployed_radii=torch.full((6, 2), 5.0, dtype=torch.float64),
        pilot_replace_one_bound=0.0,
        block_sizes=[1, 1],
        influence_cap=caps,
        regularization=gamma,
        correction_budget=budget,
        num_steps=12,
        return_diagnostics=True,
    )
    contraction = 1.0 / (1.0 + 2.0 * gamma)
    expected_beta = gamma * budget / (
        complete_cap * (1.0 - contraction**12)
    )
    assert diagnostics["complete_client_influence_cap"] == pytest.approx(
        complete_cap
    )
    assert diagnostics["beta"] == pytest.approx(expected_beta)
    assert torch.linalg.vector_norm(result - pilot) <= budget + 1e-12
    assert diagnostics["finite_correction_norm_bound"] <= budget + 1e-12
    assert diagnostics["finite_correction_norm_bound"] == pytest.approx(budget)
    assert diagnostics["exact_correction_norm_bound"] >= budget
    assert diagnostics["beta_conservative_asymptotic"] < diagnostics["beta"]
    assert diagnostics["direct_finite_replace_one_term"] == pytest.approx(
        2.0 * budget / vectors.shape[0]
    )


def test_crossfit_references_change_diagnostics_but_not_returned_reference():
    vectors, pilot, crossfit = _basic_inputs()
    common = dict(
        pilot=pilot,
        statistical_radii=torch.full((4, 1), 0.10, dtype=torch.float64),
        deployed_radii=torch.full((4, 1), 0.12, dtype=torch.float64),
        pilot_replace_one_bound=0.15,
        influence_cap=0.20,
        regularization=0.6,
        correction_budget=0.08,
        num_steps=9,
        return_diagnostics=True,
    )
    first, first_diagnostics = gaussian_aware_crossfit_bounded_correction(
        vectors, crossfit_references=crossfit, **common
    )
    shifted_crossfit = crossfit + 100.0
    second, second_diagnostics = gaussian_aware_crossfit_bounded_correction(
        vectors, crossfit_references=shifted_crossfit, **common
    )
    assert torch.equal(first, second)
    assert first_diagnostics["crossfit_references_affect_returned_reference"] is False
    assert (
        first_diagnostics["statistical_tail_fraction_crossfit"]
        != second_diagnostics["statistical_tail_fraction_crossfit"]
    )


def test_statistical_tail_and_influence_cap_activity_are_separate():
    vectors = torch.tensor(
        [[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float64
    )
    result, diagnostics = gaussian_aware_crossfit_bounded_correction(
        vectors,
        pilot=torch.zeros(2, dtype=torch.float64),
        crossfit_references=torch.zeros_like(vectors),
        statistical_radii=torch.tensor(
            [[1.0, 10.0], [1.0, 10.0], [1.0, 10.0]], dtype=torch.float64
        ),
        deployed_radii=torch.tensor(
            [[1.2, 10.2], [1.2, 10.2], [1.2, 10.2]], dtype=torch.float64
        ),
        pilot_replace_one_bound=0.0,
        block_sizes=[1, 1],
        influence_cap=[5.0, 0.1],
        regularization=1.0,
        correction_budget=0.1,
        num_steps=6,
        return_diagnostics=True,
    )
    assert bool(torch.isfinite(result).all())
    statistical = diagnostics["statistical_tail_client_blocks"]
    online_statistical = diagnostics["online_statistical_tail_client_blocks"]
    cap_active = diagnostics["influence_cap_active_client_blocks"]
    assert statistical[0][0] is True
    assert cap_active[0][0] is False
    assert statistical[1][1] is False
    assert online_statistical[1][1] is False
    assert cap_active[1][1] is True
    assert diagnostics["influence_cap_active_clients"][1] is True


def test_deployment_margin_is_not_used_in_crossfit_tail_calibration():
    vectors = torch.tensor([[0.15], [0.0], [0.0]], dtype=torch.float64)
    _, diagnostics = gaussian_aware_crossfit_bounded_correction(
        vectors,
        pilot=torch.zeros(1, dtype=torch.float64),
        crossfit_references=torch.zeros_like(vectors),
        statistical_radii=torch.full((3, 1), 0.10, dtype=torch.float64),
        deployed_radii=torch.full((3, 1), 0.20, dtype=torch.float64),
        pilot_replace_one_bound=0.0,
        influence_cap=1.0,
        regularization=0.5,
        correction_budget=0.05,
        num_steps=4,
        return_diagnostics=True,
    )
    # The residual is a crossfit tail under the calibrated 0.10 radius, but
    # is inside the online 0.20 radius after the frozen transport margin.
    assert diagnostics["statistical_tail_client_blocks"][0][0] is True
    assert diagnostics["online_statistical_tail_client_blocks"][0][0] is False
    assert diagnostics["deployment_margin_min"] == pytest.approx(0.10)
    assert diagnostics["deployment_margin_max"] == pytest.approx(0.10)


def test_replace_one_certificate_includes_data_dependent_fcc_pilot():
    generator = torch.Generator().manual_seed(91)
    n = 8
    vectors = 0.15 * torch.randn(n, 3, generator=generator, dtype=torch.float64)
    replacement = vectors.clone()
    replacement[2] = torch.tensor([50.0, -80.0, 100.0], dtype=torch.float64)
    anchor = torch.zeros(3, dtype=torch.float64)
    pilot_radius = 0.4
    pilot_bound = 2.0 * pilot_radius / n
    calibrated = torch.full((n, 1), 0.25, dtype=torch.float64)

    def evaluate(cohort: torch.Tensor, *, diagnostics: bool):
        pilot = centered_clipping(cohort, anchor=anchor, tau=pilot_radius)
        crossfit = centered_clipping_leave_one_out(
            cohort, anchor=anchor, tau=pilot_radius
        )
        return gaussian_aware_crossfit_bounded_correction(
            cohort,
            pilot=pilot,
            crossfit_references=crossfit,
            statistical_radii=calibrated - 0.02,
            deployed_radii=calibrated,
            pilot_replace_one_bound=pilot_bound,
            influence_cap=0.2,
            regularization=0.7,
            correction_budget=0.08,
            num_steps=10,
            return_diagnostics=diagnostics,
        )

    first, diagnostics = evaluate(vectors, diagnostics=True)
    second = evaluate(replacement, diagnostics=False)
    observed = float(torch.linalg.vector_norm(first - second).item())
    assert observed <= diagnostics["finite_solver_replace_one_bound"] + 1e-12
    assert diagnostics["certificate_includes_data_dependent_pilot"] is True
    assert diagnostics["finite_solver_replace_one_bound"] >= pilot_bound
    assert diagnostics["direct_finite_replace_one_term"] > 0.0


def test_diagnostics_are_json_serializable():
    import json

    vectors, pilot, crossfit = _basic_inputs()
    _, diagnostics = gaussian_aware_crossfit_bounded_correction(
        vectors,
        pilot=pilot,
        crossfit_references=crossfit,
        statistical_radii=torch.full((4, 1), 0.18, dtype=torch.float64),
        deployed_radii=torch.full((4, 1), 0.2, dtype=torch.float64),
        pilot_replace_one_bound=0.15,
        influence_cap=0.1,
        regularization=0.5,
        correction_budget=0.05,
        num_steps=5,
        return_diagnostics=True,
    )
    json.dumps(diagnostics, allow_nan=False)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"crossfit_references": torch.zeros(3, 2)}, "shape"),
        ({"statistical_radii": -0.1}, "strictly positive"),
        ({"deployed_radii": 0.1, "statistical_radii": 0.2}, "at least"),
        ({"pilot_replace_one_bound": -0.1}, "non-negative"),
        ({"correction_budget": -0.1}, "non-negative"),
        ({"regularization": 0.0}, "strictly positive"),
        ({"num_steps": 0}, "integer >= 1"),
    ],
)
def test_invalid_g0e_configuration_is_rejected(override, match):
    vectors, pilot, crossfit = _basic_inputs()
    kwargs = dict(
        pilot=pilot,
        crossfit_references=crossfit,
        statistical_radii=torch.full((4, 1), 0.18, dtype=torch.float64),
        deployed_radii=torch.full((4, 1), 0.2, dtype=torch.float64),
        pilot_replace_one_bound=0.15,
        influence_cap=0.1,
        regularization=0.5,
        correction_budget=0.05,
        num_steps=5,
    )
    kwargs.update(override)
    with pytest.raises(ValueError, match=match):
        gaussian_aware_crossfit_bounded_correction(vectors, **kwargs)


def test_half_precision_input_keeps_float32_certificate_output():
    vectors, pilot, crossfit = _basic_inputs(dtype=torch.float16)
    result = gaussian_aware_crossfit_bounded_correction(
        vectors,
        pilot=pilot,
        crossfit_references=crossfit,
        statistical_radii=torch.full((4, 1), 0.18, dtype=torch.float32),
        deployed_radii=torch.full((4, 1), 0.2, dtype=torch.float32),
        pilot_replace_one_bound=0.15,
        influence_cap=0.1,
        regularization=0.5,
        correction_budget=0.05,
        num_steps=5,
    )
    assert result.dtype == torch.float32
    assert bool(torch.isfinite(result).all())
