"""Unit tests for the fixed-denominator G0g-K4b primitives."""

from __future__ import annotations

import math

import pytest
import torch

from algorithms.gaussian_aware_reference import (
    gaussian_aware_fixed_anchor_scalar_gated_reference,
)
from algorithms.gaussian_aware_reference_k4b import (
    FULL_TEMPORAL_MISSING_SLOT,
    INCREMENTAL_TEMPORAL_SUPPRESSION,
    fixed_denominator_past_predictor,
    gaussian_aware_fixed_anchor_past_imputed_reference,
    pointwise_optimal_full_imputation_predictor,
)


def _common(
    *, history: torch.Tensor, predictor: torch.Tensor, cap: float = 0.13
) -> dict:
    n, _, dimension = history.shape
    return {
        "anchor": torch.zeros(dimension, dtype=history.dtype),
        "statistical_radii": torch.full((n, 1), 1.0e6, dtype=history.dtype),
        "temporal_standardized_history": history,
        "enrollment_standardized_mean": torch.zeros(
            (n, dimension), dtype=history.dtype
        ),
        "enrollment_size": 8,
        "temporal_gate_inner_threshold": 1.0,
        "temporal_gate_outer_threshold": 2.0,
        "predictor": predictor,
        "block_sizes": [dimension],
        "influence_cap": cap,
    }


@pytest.mark.parametrize(
    "mode", [FULL_TEMPORAL_MISSING_SLOT, INCREMENTAL_TEMPORAL_SUPPRESSION]
)
def test_k4b_h_one_is_exactly_k2(mode: str) -> None:
    vectors = torch.tensor(
        [[0.05, 0.02], [-0.03, 0.01], [0.02, -0.04]], dtype=torch.float64
    )
    history = torch.zeros((3, 4, 2), dtype=torch.float64)
    common = _common(
        history=history,
        predictor=torch.tensor([0.12, -0.12], dtype=torch.float64),
    )
    k4b, diagnostics = gaussian_aware_fixed_anchor_past_imputed_reference(
        vectors, imputation_mode=mode, return_diagnostics=True, **common
    )
    k2 = gaussian_aware_fixed_anchor_scalar_gated_reference(
        vectors,
        anchor=common["anchor"],
        statistical_radii=common["statistical_radii"],
        block_sizes=[2],
        influence_cap=0.13,
    )
    assert torch.equal(k4b, k2)
    assert diagnostics["imputation_coefficients_by_client"] == [0.0, 0.0, 0.0]
    assert diagnostics["k4b_equals_k2_when_all_history_gates_are_one"] is True


def test_k4b_h_zero_imputes_bounded_predictor_for_every_client() -> None:
    n = 3
    predictor = torch.tensor([0.03, -0.04], dtype=torch.float64)
    history = torch.full((n, 4, 2), 100.0, dtype=torch.float64)
    reference, diagnostics = gaussian_aware_fixed_anchor_past_imputed_reference(
        torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
        return_diagnostics=True,
        **_common(history=history, predictor=predictor),
    )
    assert torch.allclose(reference, predictor, atol=1.0e-12, rtol=0.0)
    assert diagnostics["gates_by_client"] == [0.0] * n
    assert diagnostics["imputation_coefficients_by_client"] == [1.0] * n


def test_k4b_mixed_history_gate_matches_closed_form() -> None:
    history = torch.zeros((2, 4, 1), dtype=torch.float64)
    # With L=4, W0=8, c0=1 and c1=2 this is safely beyond h=0.
    history[1] = 10.0
    vectors = torch.tensor([[0.05], [0.08]], dtype=torch.float64)
    predictor = torch.tensor([0.02], dtype=torch.float64)
    reference, diagnostics = gaussian_aware_fixed_anchor_past_imputed_reference(
        vectors,
        return_diagnostics=True,
        **_common(history=history, predictor=predictor),
    )
    expected = torch.tensor([(0.05 + 0.02) / 2.0], dtype=torch.float64)
    assert torch.allclose(reference, expected, atol=1.0e-12, rtol=0.0)
    assert diagnostics["temporal_gates_by_client"] == [1.0, 0.0]
    assert diagnostics["coefficient_sum_max"] <= 1.0


def test_k4b_fractional_history_gate_is_convex_interpolation() -> None:
    temporal_scale = math.sqrt(1.0 / 4.0 + 1.0 / 8.0)
    history = torch.full((1, 4, 1), 1.5 * temporal_scale, dtype=torch.float64)
    vector = torch.tensor([[0.08]], dtype=torch.float64)
    predictor = torch.tensor([0.02], dtype=torch.float64)
    reference, diagnostics = gaussian_aware_fixed_anchor_past_imputed_reference(
        vector,
        return_diagnostics=True,
        **_common(history=history, predictor=predictor),
    )
    assert diagnostics["temporal_gates_by_client"][0] == pytest.approx(0.5)
    assert diagnostics["current_gates_by_client"][0] == pytest.approx(1.0)
    assert float(reference) == pytest.approx(0.5 * 0.08 + 0.5 * 0.02)
    assert diagnostics["coefficient_sum_max"] == pytest.approx(1.0)


def test_incremental_ablation_matches_exact_gamma_minus_g_formula() -> None:
    temporal_scale = math.sqrt(1.0 / 4.0 + 1.0 / 8.0)
    history = torch.full((1, 4, 1), 1.5 * temporal_scale, dtype=torch.float64)
    vector = torch.tensor([[0.08]], dtype=torch.float64)
    predictor = torch.tensor([0.02], dtype=torch.float64)
    reference, diagnostics = gaussian_aware_fixed_anchor_past_imputed_reference(
        vector,
        imputation_mode=INCREMENTAL_TEMPORAL_SUPPRESSION,
        return_diagnostics=True,
        **_common(history=history, predictor=predictor),
    )
    gamma = diagnostics["current_gates_by_client"][0]
    g = diagnostics["gates_by_client"][0]
    expected = g * 0.08 + (gamma - g) * 0.02
    assert float(reference) == pytest.approx(expected)
    assert diagnostics["contribution_formula"] == "g*c_plus_(gamma-g)*p"
    assert diagnostics["coefficient_sum_max"] <= 1.0 + 1.0e-12


def test_predictor_uses_public_floor_not_gate_sum_as_aggregate_denominator() -> None:
    history = torch.ones((4, 5, 1), dtype=torch.float64) * 0.1
    gates = torch.zeros((4, 5), dtype=torch.float64)
    gates[:, 0] = 1.0
    predictor, diagnostics = fixed_denominator_past_predictor(
        history,
        gates,
        minimum_accepted_mass=4.0,
        influence_cap=0.13,
        return_diagnostics=True,
    )
    assert float(predictor) == pytest.approx(0.025)
    assert diagnostics["accepted_mass_by_round"] == [1.0] * 4
    assert diagnostics["denominator_by_round"] == [4.0] * 4
    assert diagnostics["gate_sum_used_as_aggregate_denominator"] is False


@pytest.mark.parametrize(
    "mode", [FULL_TEMPORAL_MISSING_SLOT, INCREMENTAL_TEMPORAL_SUPPRESSION]
)
def test_predictor_and_k4b_respect_common_cap(mode: str) -> None:
    history = torch.full((4, 5, 2), 0.13 / math.sqrt(2.0), dtype=torch.float64)
    predictor = fixed_denominator_past_predictor(
        history,
        torch.ones((4, 5), dtype=torch.float64),
        minimum_accepted_mass=4.0,
        influence_cap=0.13,
    )
    assert float(torch.linalg.vector_norm(predictor)) <= 0.13 + 1.0e-12
    _, diagnostics = gaussian_aware_fixed_anchor_past_imputed_reference(
        torch.full((5, 2), 100.0, dtype=torch.float64),
        imputation_mode=mode,
        return_diagnostics=True,
        **_common(
            history=torch.full((5, 4, 2), 100.0, dtype=torch.float64),
            predictor=torch.tensor([100.0, 100.0], dtype=torch.float64),
        ),
    )
    assert max(diagnostics["client_contribution_norms"]) <= 0.13 + 1.0e-12
    assert diagnostics["coefficient_sum_max"] <= 1.0 + 1.0e-12


@pytest.mark.parametrize(
    "mode", [FULL_TEMPORAL_MISSING_SLOT, INCREMENTAL_TEMPORAL_SUPPRESSION]
)
def test_k4b_current_replace_one_bound_with_fixed_past_and_predictor(
    mode: str,
) -> None:
    cap = 0.13
    vectors = torch.tensor([[10.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    neighbour = vectors.clone()
    neighbour[0] = torch.tensor([-10.0, 0.0], dtype=torch.float64)
    common = _common(
        history=torch.zeros((2, 4, 2), dtype=torch.float64),
        predictor=torch.tensor([0.02, 0.01], dtype=torch.float64),
        cap=cap,
    )
    left, diagnostics = gaussian_aware_fixed_anchor_past_imputed_reference(
        vectors, imputation_mode=mode, return_diagnostics=True, **common
    )
    right = gaussian_aware_fixed_anchor_past_imputed_reference(
        neighbour, imputation_mode=mode, **common
    )
    difference = float(torch.linalg.vector_norm(left - right))
    assert difference == pytest.approx(2.0 * cap / 2.0)
    assert diagnostics["replace_one_bound"] == pytest.approx(2.0 * cap / 2.0)


@pytest.mark.parametrize(
    "mode", [FULL_TEMPORAL_MISSING_SLOT, INCREMENTAL_TEMPORAL_SUPPRESSION]
)
def test_replace_one_certificate_with_h_below_one_and_nonzero_predictor(
    mode: str,
) -> None:
    cap = 0.13
    temporal_scale = math.sqrt(1.0 / 4.0 + 1.0 / 8.0)
    history = torch.full((2, 4, 1), 1.5 * temporal_scale, dtype=torch.float64)
    predictor = torch.tensor([0.07], dtype=torch.float64)
    left_vectors = torch.tensor([[10.0], [0.02]], dtype=torch.float64)
    right_vectors = torch.tensor([[-10.0], [0.02]], dtype=torch.float64)
    common = _common(history=history, predictor=predictor, cap=cap)
    left, diagnostics = gaussian_aware_fixed_anchor_past_imputed_reference(
        left_vectors, imputation_mode=mode, return_diagnostics=True, **common
    )
    right = gaussian_aware_fixed_anchor_past_imputed_reference(
        right_vectors, imputation_mode=mode, **common
    )
    difference = float(torch.linalg.vector_norm(left - right))
    assert min(diagnostics["temporal_gates_by_client"]) < 1.0
    assert diagnostics["predictor_norm"] > 0.0
    assert difference <= 2.0 * cap / 2.0 + 1.0e-12
    assert diagnostics["client_contribution_cap_respected"] is True


def test_pointwise_oracle_is_projection_and_minimizes_primary_error() -> None:
    clipped = torch.tensor([[0.08, 0.00], [-0.03, 0.02]], dtype=torch.float64)
    gates = torch.tensor([0.5, 0.25], dtype=torch.float64)
    history = torch.tensor([0.5, 0.75], dtype=torch.float64)
    target = torch.tensor([0.06, -0.01], dtype=torch.float64)
    cap = 0.13
    predictor, diagnostics = pointwise_optimal_full_imputation_predictor(
        clipped,
        gates,
        history,
        target_direction=target,
        influence_cap=cap,
        return_diagnostics=True,
    )
    missing = float(torch.sum(1.0 - history))
    raw = (2.0 * target - torch.sum(gates[:, None] * clipped, dim=0)) / missing
    expected = clip_l2_for_test(raw, cap)
    assert torch.allclose(predictor, expected, atol=1.0e-12, rtol=0.0)
    assert diagnostics["deployable"] is False
    assert diagnostics["privacy_claimed"] is False

    def squared_error(candidate: torch.Tensor) -> float:
        reference_direction = (
            torch.sum(gates[:, None] * clipped, dim=0) + missing * candidate
        ) / 2.0
        return float(torch.sum((reference_direction - target).square()))

    alternatives = [
        torch.zeros(2, dtype=torch.float64),
        torch.tensor([cap, 0.0], dtype=torch.float64),
        torch.tensor([-cap, 0.0], dtype=torch.float64),
        torch.tensor([0.0, cap], dtype=torch.float64),
    ]
    assert all(
        squared_error(predictor) <= squared_error(value) + 1.0e-12
        for value in alternatives
    )


def test_pointwise_oracle_is_zero_when_no_temporal_slot_is_missing() -> None:
    predictor, diagnostics = pointwise_optimal_full_imputation_predictor(
        torch.tensor([[0.02], [0.03]], dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        target_direction=torch.tensor([0.5], dtype=torch.float64),
        influence_cap=0.13,
        return_diagnostics=True,
    )
    assert torch.equal(predictor, torch.zeros_like(predictor))
    assert diagnostics["missing_slot_mass"] == 0.0


def clip_l2_for_test(vector: torch.Tensor, cap: float) -> torch.Tensor:
    norm = float(torch.linalg.vector_norm(vector))
    return vector if norm <= cap else vector * (cap / norm)


def test_k4b_max_float_is_finite_or_rejected_explicitly() -> None:
    maximum = torch.finfo(torch.float32).max
    reference, diagnostics = gaussian_aware_fixed_anchor_past_imputed_reference(
        torch.tensor([[maximum, 0.0]], dtype=torch.float32),
        return_diagnostics=True,
        **{
            **_common(
                history=torch.zeros((1, 4, 2), dtype=torch.float32),
                predictor=torch.tensor([maximum, 0.0], dtype=torch.float32),
            ),
            "statistical_radii": torch.tensor([[maximum]], dtype=torch.float32),
        },
    )
    assert bool(torch.isfinite(reference).all())
    assert float(reference[0]) == pytest.approx(0.13)
    assert diagnostics["client_contribution_cap_respected"] is True


def test_predictor_rejects_nonfinite_and_unclipped_history() -> None:
    gates = torch.ones((4, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="finite"):
        fixed_denominator_past_predictor(
            torch.full((4, 2, 1), float("nan"), dtype=torch.float64),
            gates,
            minimum_accepted_mass=1.0,
            influence_cap=0.13,
        )
    with pytest.raises(ValueError, match="already be clipped"):
        fixed_denominator_past_predictor(
            torch.ones((4, 2, 1), dtype=torch.float64),
            gates,
            minimum_accepted_mass=1.0,
            influence_cap=0.13,
        )
