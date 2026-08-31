"""Synthetic certificates for the references used by SC-Partial-FAR-DP.

These tests deliberately isolate replace-one stability from model training.
They are executable counterparts of the lemmas in the theory document: every
cohort row is already user-level clipped and the anchor is fixed with respect
to the current cohort.
"""

from __future__ import annotations

import math

import pytest
import torch

from algorithms.base import ClientState, get_algorithm
from algorithms.sc_partial_far_dp import certified_scfar_sensitivity
from robustness.aggregators import (
    centered_clipping,
    clip_l2,
    regularized_huber_reference,
)


def _replace_one_pair(
    *, n: int, d: int, clip_norm: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    first = clip_l2(torch.randn(n, d, generator=generator), clip_norm)
    second = first.clone()
    second[-1] = clip_l2(torch.randn(d, generator=generator), clip_norm)
    return first.to(torch.float64), second.to(torch.float64)


@pytest.mark.parametrize("tau", [0.35, 2.0])
def test_centered_clipping_global_replace_one_stability(tau: float):
    n, d, clip_norm = 11, 7, 1.0
    anchor = torch.linspace(-0.2, 0.2, d, dtype=torch.float64)
    bound = 2.0 * min(clip_norm, tau) / n
    for seed in range(30):
        first, second = _replace_one_pair(n=n, d=d, clip_norm=clip_norm, seed=seed)
        ref_first = centered_clipping(first, anchor=anchor, tau=tau)
        ref_second = centered_clipping(second, anchor=anchor, tau=tau)
        observed = torch.linalg.vector_norm(ref_first - ref_second)
        assert float(observed) <= bound + 1e-12


@pytest.mark.parametrize(
    ("clip_norm", "tau", "expected_radius"),
    [(1.0, 0.4, 0.4), (0.4, 1.0, 0.4)],
)
def test_centered_clipping_stability_bound_is_attained_in_one_dimension(
    clip_norm: float, tau: float, expected_radius: float
):
    n = 8
    first = torch.zeros(n, 1, dtype=torch.float64)
    second = first.clone()
    first[-1, 0] = clip_norm
    second[-1, 0] = -clip_norm
    anchor = torch.zeros(1, dtype=torch.float64)
    observed = torch.abs(
        centered_clipping(first, anchor=anchor, tau=tau)
        - centered_clipping(second, anchor=anchor, tau=tau)
    ).item()
    assert math.isclose(observed, 2.0 * expected_radius / n, rel_tol=1e-12)


def test_centered_clipping_b_replacements_telescope():
    n, d, clip_norm, tau, b = 12, 5, 1.0, 0.3, 4
    generator = torch.Generator().manual_seed(81)
    first = clip_l2(torch.randn(n, d, generator=generator), clip_norm).double()
    second = first.clone()
    second[:b] = clip_l2(torch.randn(b, d, generator=generator), clip_norm).double()
    anchor = torch.zeros(d, dtype=torch.float64)
    observed = torch.linalg.vector_norm(
        centered_clipping(first, anchor=anchor, tau=tau)
        - centered_clipping(second, anchor=anchor, tau=tau)
    )
    assert float(observed) <= 2.0 * b * min(clip_norm, tau) / n + 1e-12


@pytest.mark.parametrize("num_steps", [1, 3, 10])
def test_fixed_step_huber_reference_has_its_finite_solver_certificate(
    num_steps: int,
):
    n, d, clip_norm, tau, gamma = 13, 6, 1.0, 0.45, 0.8
    anchor = torch.linspace(-0.1, 0.1, d, dtype=torch.float64)
    contraction = 1.0 / (1.0 + 2.0 * gamma)
    bound = 2.0 * min(clip_norm, tau) / (gamma * n) * (1.0 - contraction**num_steps)
    for seed in range(20):
        first, second = _replace_one_pair(
            n=n, d=d, clip_norm=clip_norm, seed=500 + seed
        )
        ref_first = regularized_huber_reference(
            first,
            anchor=anchor,
            tau=tau,
            gamma=gamma,
            num_steps=num_steps,
        )
        ref_second = regularized_huber_reference(
            second,
            anchor=anchor,
            tau=tau,
            gamma=gamma,
            num_steps=num_steps,
        )
        observed = torch.linalg.vector_norm(ref_first - ref_second)
        assert float(observed) <= bound + 2e-12


def _client_tuples(values: tuple[float, ...]):
    tuples = []
    for client_id, value in enumerate(values):
        tuples.append(
            (
                {"weight": torch.tensor([[value]], dtype=torch.float64)},
                {
                    "client_id": client_id,
                    "dataset_size": 1,
                    "local_loss": 0.0,
                    "bytes_sent": 8,
                    "energy_j_consumed": 0.0,
                    "is_byzantine": False,
                },
                ClientState(client_id=client_id, battery_j=1.0),
            )
        )
    return tuples


def _scalar_model():
    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.zero_()
    return model


def test_scfar_automatically_uses_centered_clipping_certificate_and_public_history():
    algorithm = get_algorithm("scfar_dp")
    config = {
        **algorithm.get_default_config(),
        "user_clip_norm": 1.0,
        "reference_clip_tau": 0.5,
        "distance_clip": 2.0,
        "far_alpha": 0.1,
        "kappa_w": 2.0,
        "robust_reference": "centered_clipping",
        "enable_central_dp": False,
        "sensitivity_mode": "automatic_certified",
    }
    first = algorithm.server_aggregate(
        _scalar_model(), _client_tuples((0.1, 0.2, 0.3, 0.4)), 0, config
    )
    expected = certified_scfar_sensitivity(
        n=4,
        clip_norm=1.0,
        distance_clip=2.0,
        alpha=float(first.metrics["scfar_effective_alpha"]),
        kappa_bound=float(first.metrics["scfar_certified_kappa"]),
        reference_stability=2.0 * 0.5 / 4,
    )
    assert first.metrics["scfar_reference_certificate"] == ("centered_clipping_global")
    assert first.metrics["scfar_sensitivity_mode"] == "proved_reference_bound"
    assert first.metrics["scfar_reference_stability"] == 0.25
    assert math.isclose(first.metrics["scfar_sensitivity"], expected)

    # The next anchor is derived from round zero's released aggregate.  It is
    # not recomputed from the new cohort before F_CC is evaluated.
    second = algorithm.server_aggregate(
        _scalar_model(), _client_tuples((0.2, 0.3, 0.4, 0.5)), 1, config
    )
    assert second.metrics["scfar_reference_anchor_norm"] > 0.0


def test_scfar_huber_ablation_reports_finite_step_certificate():
    algorithm = get_algorithm("scfar_dp")
    result = algorithm.server_aggregate(
        _scalar_model(),
        _client_tuples((-0.4, -0.1, 0.2, 0.8)),
        0,
        {
            **algorithm.get_default_config(),
            "robust_reference": "regularized_huber",
            "reference_clip_tau": 0.5,
            "huber_gamma": 0.75,
            "huber_num_steps": 5,
            "enable_central_dp": False,
        },
    )
    assert result.metrics["scfar_reference_certificate"] == (
        "regularized_huber_fixed_steps"
    )
    assert result.metrics["scfar_huber_num_steps"] == 5
    assert result.metrics["scfar_reference_stability"] > 0.0
    assert result.metrics["scfar_sensitivity"] <= 2.0


def test_uncertified_reference_falls_back_or_fails_closed():
    algorithm = get_algorithm("scfar_dp")
    base = {
        **algorithm.get_default_config(),
        "robust_reference": "mean",
        "enable_central_dp": False,
    }
    fallback = algorithm.server_aggregate(
        _scalar_model(), _client_tuples((0.1, 0.2, 0.3)), 0, base
    )
    assert fallback.metrics["scfar_sensitivity"] == 2.0
    assert fallback.metrics["scfar_sensitivity_mode"] == (
        "automatic_fallback_conservative_2C"
    )

    with pytest.raises(ValueError, match="requires a certified reference"):
        algorithm.server_aggregate(
            _scalar_model(),
            _client_tuples((0.1, 0.2, 0.3)),
            0,
            {**base, "sensitivity_mode": "proved_reference_bound"},
        )


def test_scfar_refuses_release_beyond_public_privacy_horizon():
    algorithm = get_algorithm("scfar_dp")
    config = {
        **algorithm.get_default_config(),
        "enable_central_dp": True,
        "central_noise_multiplier": 2.0,
        "privacy_num_rounds": 1,
    }
    with pytest.raises(RuntimeError, match="beyond privacy_num_rounds"):
        algorithm.server_aggregate(
            _scalar_model(),
            _client_tuples((0.1, 0.2, 0.3)),
            1,
            config,
        )
