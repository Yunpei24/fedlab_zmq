"""Regression tests for the frozen Stage-14A synthetic score audit."""

from __future__ import annotations

import torch

from algorithms.noise_aware_scores import calibrated_null_energy_scores
from robustness.aggregators import centered_clipping, clip_l2
from scripts.run_dt_ldp_far_stage14a_effective_moments_audit import (
    _fcc_loo_energies,
    _null_energy_samples,
    _projection_order,
    _score_matrix,
)


def test_stage14a_vectorized_fcc_loo_matches_explicit_recomputation():
    vectors = torch.tensor(
        [[-2.0, 0.0, 1.0], [0.5, 1.0, -1.0], [2.0, -0.5, 0.0]],
        dtype=torch.float64,
    )
    order = torch.arange(3)
    observed = _fcc_loo_energies(
        vectors,
        coordinate_order=order,
        score_dimension=3,
        full_dimension=3,
        full_radius=0.8,
    )
    expected = []
    for client in range(len(vectors)):
        keep = torch.arange(len(vectors)) != client
        reference = centered_clipping(
            vectors[keep], anchor=torch.zeros(3, dtype=torch.float64), tau=0.8
        )
        expected.append(torch.linalg.vector_norm(vectors[client] - reference).square())
    assert torch.allclose(observed, torch.stack(expected), atol=1e-12)


def test_stage14a_null_generator_applies_server_clipping():
    scales = torch.ones(5, dtype=torch.float64)
    orders = {17: _projection_order(8, 17)}
    clipped = _null_energy_samples(
        scales=scales,
        coordinate_orders=orders,
        score_dimensions=[8],
        full_dimension=8,
        full_radius=0.5,
        noise_std=1.0,
        server_clip_norm=0.01,
        draws=20,
        seed=31,
    )[(17, 8)]
    unclipped = _null_energy_samples(
        scales=scales,
        coordinate_orders=orders,
        score_dimensions=[8],
        full_dimension=8,
        full_radius=0.5,
        noise_std=1.0,
        server_clip_norm=100.0,
        draws=20,
        seed=31,
    )[(17, 8)]
    assert float(clipped.max()) <= (2.0 * 0.01) ** 2 + 1e-12
    assert not torch.allclose(clipped, unclipped)


def test_stage14a_public_coordinate_subspaces_are_nested():
    order = _projection_order(32, 911)
    assert set(order[:8].tolist()) < set(order[:16].tolist())
    assert torch.equal(order, _projection_order(32, 911))


def test_stage14a_vectorized_score_matches_pure_score_function():
    calibration = torch.arange(1, 61, dtype=torch.float64).reshape(20, 3)
    observed = torch.tensor(
        [[10.0, 20.0, 30.0], [20.0, 30.0, 40.0]], dtype=torch.float64
    )
    for family, mode in (
        ("effective_moment", "moment"),
        ("empirical_quantile", "quantile"),
    ):
        matrix = _score_matrix(
            observed,
            calibration,
            family=family,
            individual_weight=0.5,
            z_clip=4.0,
            tail_probability=0.9,
            variance_ridge=1e-12,
        )
        rows = []
        for row in observed:
            score, _ = calibrated_null_energy_scores(
                row,
                calibration,
                mode=mode,
                z_clip=4.0,
                tail_probability=0.9,
                individual_calibration_weight=0.5,
            )
            rows.append(score)
        assert torch.allclose(matrix, torch.stack(rows), atol=1e-12)


def test_stage14a_server_clip_bound_used_in_test_is_valid():
    vectors = torch.randn(10, 4, generator=torch.Generator().manual_seed(3))
    clipped = clip_l2(vectors, 0.2)
    assert bool((torch.linalg.vector_norm(clipped, dim=1) <= 0.2 + 1e-7).all())
