"""Property and certificate tests for the G0g-K1 reference primitive."""

from __future__ import annotations

import json

import pytest
import torch

from algorithms.gaussian_aware_reference import (
    gaussian_aware_fixed_anchor_dual_gated_reference,
    gaussian_aware_fixed_anchor_gated_reference,
    gaussian_aware_fixed_anchor_scalar_gated_reference,
)
from robustness.aggregators import centered_clipping


def test_gate_has_public_ramp_and_no_sum_normalisation() -> None:
    vectors = torch.tensor([[0.5], [1.5], [2.5]], dtype=torch.float64)
    reference, diagnostics = gaussian_aware_fixed_anchor_gated_reference(
        vectors,
        anchor=torch.zeros(1, dtype=torch.float64),
        statistical_radii=torch.ones(3, 1, dtype=torch.float64),
        influence_cap=10.0,
        gate_transition_width=1.0,
        return_diagnostics=True,
    )
    assert diagnostics["gates_by_client_block"] == [[1.0], [0.5], [0.0]]
    # (1*0.5 + 0.5*1.5 + 0*2.5) / 3; no division by sum(g)=1.5.
    assert reference.item() == pytest.approx(1.25 / 3.0)
    assert diagnostics["normalization_by_gate_sum"] is False


def test_covariance_radius_changes_tolerance_but_not_authorised_cap() -> None:
    vectors = torch.tensor([[1.5], [1.5]], dtype=torch.float64)
    _, diagnostics = gaussian_aware_fixed_anchor_gated_reference(
        vectors,
        anchor=torch.zeros(1, dtype=torch.float64),
        statistical_radii=torch.tensor([[1.0], [2.0]], dtype=torch.float64),
        influence_cap=0.2,
        return_diagnostics=True,
    )
    assert diagnostics["gates_by_client_block"] == [[0.5], [1.0]]
    assert diagnostics["influence_caps_per_block"] == [0.2]
    assert max(diagnostics["client_contribution_norms"]) <= 0.2 + 1e-12
    assert diagnostics["influence_caps_are_client_independent"] is True


def test_homogeneous_clients_are_treated_identically() -> None:
    vectors = torch.tensor([[0.4, -0.3], [0.4, -0.3]], dtype=torch.float64)
    _, diagnostics = gaussian_aware_fixed_anchor_gated_reference(
        vectors,
        anchor=torch.zeros(2, dtype=torch.float64),
        statistical_radii=torch.full((2, 1), 0.5, dtype=torch.float64),
        influence_cap=0.7,
        return_diagnostics=True,
    )
    assert diagnostics["normalized_residuals_by_client_block"][0] == pytest.approx(
        diagnostics["normalized_residuals_by_client_block"][1]
    )
    assert diagnostics["gates_by_client_block"][0] == pytest.approx(
        diagnostics["gates_by_client_block"][1]
    )


def test_exact_replace_one_certificate_with_fixed_public_inputs() -> None:
    generator = torch.Generator().manual_seed(1701)
    n, d = 11, 7
    left = torch.randn(n, d, generator=generator, dtype=torch.float64)
    right = left.clone()
    right[3] = 20.0 * torch.randn(d, generator=generator, dtype=torch.float64)
    common = dict(
        anchor=torch.zeros(d, dtype=torch.float64),
        statistical_radii=torch.full((n, 1), 1.1, dtype=torch.float64),
        influence_cap=0.4,
        gate_transition_width=1.0,
    )
    first, diagnostics = gaussian_aware_fixed_anchor_gated_reference(
        left, return_diagnostics=True, **common
    )
    second = gaussian_aware_fixed_anchor_gated_reference(right, **common)
    observed = torch.linalg.vector_norm(first - second).item()
    assert observed <= diagnostics["replace_one_bound"] + 1e-12
    assert diagnostics["replace_one_bound"] == pytest.approx(0.8 / n)


def test_antipodal_replacement_attains_the_multiblock_bound() -> None:
    """A constructed pair, not random trials, reaches 2G/n exactly."""

    n = 25
    caps = torch.tensor([0.3, 0.4], dtype=torch.float64)
    left = torch.zeros(n, 4, dtype=torch.float64)
    right = left.clone()
    # Each block points in one coordinate and saturates its public cap.
    left[0] = torch.tensor([30.0, 0.0, 40.0, 0.0], dtype=torch.float64)
    right[0] = -left[0]
    common = dict(
        anchor=torch.zeros(4, dtype=torch.float64),
        statistical_radii=torch.full((n, 2), 1.0e6, dtype=torch.float64),
        block_sizes=[2, 2],
        influence_cap=caps,
        gate_transition_width=1.0,
    )
    first, diagnostics = gaussian_aware_fixed_anchor_gated_reference(
        left, return_diagnostics=True, **common
    )
    second = gaussian_aware_fixed_anchor_gated_reference(right, **common)
    observed = torch.linalg.vector_norm(first - second).item()
    assert diagnostics["complete_client_influence_cap"] == pytest.approx(0.5)
    assert observed == pytest.approx(2.0 * 0.5 / n)
    assert observed == pytest.approx(diagnostics["replace_one_bound"])
    assert diagnostics["certificate_float_tolerance"] > 0.0


def test_block_caps_give_complete_client_l2_cap() -> None:
    vectors = torch.full((4, 4), 100.0, dtype=torch.float64)
    caps = torch.tensor([0.3, 0.4], dtype=torch.float64)
    _, diagnostics = gaussian_aware_fixed_anchor_gated_reference(
        vectors,
        anchor=torch.zeros(4, dtype=torch.float64),
        statistical_radii=torch.full((4, 2), 1e6, dtype=torch.float64),
        block_sizes=[2, 2],
        influence_cap=caps,
        return_diagnostics=True,
    )
    assert diagnostics["complete_client_influence_cap"] == pytest.approx(0.5)
    assert max(diagnostics["client_contribution_norms"]) <= 0.5 + 1e-12
    assert diagnostics["reference_displacement_bound"] == pytest.approx(0.5)


def test_diagnostics_are_json_serializable() -> None:
    _, diagnostics = gaussian_aware_fixed_anchor_gated_reference(
        torch.tensor([[0.2], [-0.1]], dtype=torch.float64),
        anchor=torch.zeros(1, dtype=torch.float64),
        statistical_radii=0.5,
        influence_cap=0.3,
        return_diagnostics=True,
    )
    json.dumps(diagnostics, allow_nan=False)


def test_k2_disabled_gate_is_exactly_global_centered_clipping() -> None:
    generator = torch.Generator().manual_seed(901)
    vectors = torch.randn(9, 8, generator=generator, dtype=torch.float64)
    anchor = torch.randn(8, generator=generator, dtype=torch.float64)
    expected = centered_clipping(vectors, anchor=anchor, tau=0.4)
    observed, diagnostics = gaussian_aware_fixed_anchor_scalar_gated_reference(
        vectors,
        anchor=anchor,
        statistical_radii=torch.full((9, 2), 1.0e12, dtype=torch.float64),
        block_sizes=[3, 5],
        influence_cap=0.4,
        return_diagnostics=True,
    )
    assert torch.allclose(observed, expected, atol=1e-12, rtol=1e-12)
    assert diagnostics["gates_by_client"] == pytest.approx([1.0] * 9)
    assert diagnostics["no_gate_reduces_exactly_to_global_centered_clipping"] is True


def test_k2_covariance_changes_gate_but_not_global_cap() -> None:
    vectors = torch.tensor([[1.5, 0.0], [1.5, 0.0]], dtype=torch.float64)
    _, diagnostics = gaussian_aware_fixed_anchor_scalar_gated_reference(
        vectors,
        anchor=torch.zeros(2, dtype=torch.float64),
        statistical_radii=torch.tensor([[1.0], [2.0]], dtype=torch.float64),
        influence_cap=0.2,
        return_diagnostics=True,
    )
    assert diagnostics["gates_by_client"] == pytest.approx([0.5, 1.0])
    assert max(diagnostics["client_contribution_norms"]) <= 0.2 + 1e-12
    assert diagnostics["complete_client_influence_cap"] == pytest.approx(0.2)


def test_k2_antipodal_pair_attains_global_replace_one_bound() -> None:
    n = 25
    left = torch.zeros(n, 4, dtype=torch.float64)
    right = left.clone()
    left[0, 0] = 100.0
    right[0, 0] = -100.0
    common = dict(
        anchor=torch.zeros(4, dtype=torch.float64),
        statistical_radii=1.0e6,
        block_sizes=[2, 2],
        influence_cap=0.13,
    )
    first, diagnostics = gaussian_aware_fixed_anchor_scalar_gated_reference(
        left, return_diagnostics=True, **common
    )
    second = gaussian_aware_fixed_anchor_scalar_gated_reference(right, **common)
    observed = torch.linalg.vector_norm(first - second).item()
    assert observed == pytest.approx(2.0 * 0.13 / n)
    assert observed == pytest.approx(diagnostics["replace_one_bound"])


def test_k3_coincident_radii_reduce_exactly_to_k2() -> None:
    generator = torch.Generator().manual_seed(902)
    vectors = torch.randn(9, 8, generator=generator, dtype=torch.float64)
    anchor = torch.randn(8, generator=generator, dtype=torch.float64)
    radii = 0.5 + torch.rand(9, 2, generator=generator, dtype=torch.float64)
    common = dict(
        anchor=anchor,
        block_sizes=[3, 5],
        influence_cap=0.4,
        gate_transition_width=0.7,
    )
    expected, k2_diagnostics = gaussian_aware_fixed_anchor_scalar_gated_reference(
        vectors,
        statistical_radii=radii,
        return_diagnostics=True,
        **common,
    )
    observed, diagnostics = gaussian_aware_fixed_anchor_dual_gated_reference(
        vectors,
        statistical_radii=radii,
        common_statistical_radii=radii,
        return_diagnostics=True,
        **common,
    )
    assert torch.equal(observed, expected)
    assert diagnostics["gates_by_client"] == pytest.approx(
        k2_diagnostics["gates_by_client"]
    )
    assert diagnostics["aware_gates_by_client"] == pytest.approx(
        diagnostics["common_gates_by_client"]
    )
    assert diagnostics["equal_gate_fraction"] == pytest.approx(1.0)
    assert diagnostics["coincident_radii_reduce_exactly_to_k2"] is True


def test_k3_all_one_gates_are_exactly_global_centered_clipping() -> None:
    generator = torch.Generator().manual_seed(903)
    vectors = torch.randn(11, 7, generator=generator, dtype=torch.float64)
    anchor = torch.randn(7, generator=generator, dtype=torch.float64)
    expected = centered_clipping(vectors, anchor=anchor, tau=0.31)
    observed, diagnostics = gaussian_aware_fixed_anchor_dual_gated_reference(
        vectors,
        anchor=anchor,
        statistical_radii=torch.full((11, 2), 1.0e12, dtype=torch.float64),
        common_statistical_radii=torch.full(
            (11, 2), 1.0e12, dtype=torch.float64
        ),
        block_sizes=[3, 4],
        influence_cap=0.31,
        return_diagnostics=True,
    )
    assert torch.allclose(observed, expected, atol=1e-12, rtol=1e-12)
    assert diagnostics["gates_by_client"] == pytest.approx([1.0] * 11)
    assert (
        diagnostics["all_one_gates_reduce_exactly_to_global_centered_clipping"]
        is True
    )


def test_k3_antipodal_pair_attains_global_replace_one_bound() -> None:
    n = 25
    left = torch.zeros(n, 4, dtype=torch.float64)
    right = left.clone()
    left[0, 0] = 100.0
    right[0, 0] = -100.0
    common = dict(
        anchor=torch.zeros(4, dtype=torch.float64),
        statistical_radii=1.0e6,
        common_statistical_radii=1.0e6,
        block_sizes=[2, 2],
        influence_cap=0.13,
    )
    first, diagnostics = gaussian_aware_fixed_anchor_dual_gated_reference(
        left, return_diagnostics=True, **common
    )
    second = gaussian_aware_fixed_anchor_dual_gated_reference(right, **common)
    observed = torch.linalg.vector_norm(first - second).item()
    assert observed == pytest.approx(2.0 * 0.13 / n)
    assert observed == pytest.approx(diagnostics["replace_one_bound"])
    assert diagnostics["client_contribution_cap_respected"] is True


def test_k3_common_gate_can_restrict_covariance_aware_acceptance() -> None:
    vectors = torch.tensor([[1.5, 0.0], [0.5, 0.0]], dtype=torch.float64)
    _, diagnostics = gaussian_aware_fixed_anchor_dual_gated_reference(
        vectors,
        anchor=torch.zeros(2, dtype=torch.float64),
        statistical_radii=torch.tensor([[3.0], [1.0]], dtype=torch.float64),
        common_statistical_radii=1.0,
        influence_cap=2.0,
        return_diagnostics=True,
    )
    assert diagnostics["aware_gates_by_client"] == pytest.approx([1.0, 1.0])
    assert diagnostics["common_gates_by_client"] == pytest.approx([0.5, 1.0])
    assert diagnostics["gates_by_client"] == pytest.approx([0.5, 1.0])
    assert diagnostics["common_gate_limiting_fraction"] == pytest.approx(0.5)


def test_k3_diagnostics_are_json_serializable() -> None:
    _, diagnostics = gaussian_aware_fixed_anchor_dual_gated_reference(
        torch.tensor([[0.2], [-0.1]], dtype=torch.float64),
        anchor=torch.zeros(1, dtype=torch.float64),
        statistical_radii=0.5,
        common_statistical_radii=0.4,
        influence_cap=0.3,
        return_diagnostics=True,
    )
    json.dumps(diagnostics, allow_nan=False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gate_transition_width": 0.0}, "gate_transition_width"),
        ({"influence_cap": 0.0}, "influence_cap"),
        ({"statistical_radii": 0.0}, "statistical_radii"),
        ({"num_replacements_for_diagnostics": 0}, "num_replacements"),
    ],
)
def test_invalid_public_parameters_are_rejected(kwargs, message) -> None:
    inputs = dict(
        vectors=torch.tensor([[0.2], [-0.1]], dtype=torch.float64),
        anchor=torch.zeros(1, dtype=torch.float64),
        statistical_radii=0.5,
        influence_cap=0.3,
    )
    inputs.update(kwargs)
    with pytest.raises(ValueError, match=message):
        gaussian_aware_fixed_anchor_gated_reference(**inputs)
