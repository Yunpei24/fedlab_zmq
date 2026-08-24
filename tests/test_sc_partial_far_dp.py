"""Unit tests for the new DP reproduction and SC-Partial-FAR paths."""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from algorithms.base import ClientState, get_algorithm
from algorithms.sc_partial_far_dp import (
    alpha_max_for_weight_factor,
    bounded_distance_scores,
)
from privacy.local_dpsgd import local_dpsgd_train
from privacy.rdp import (
    RDPAccountant,
    calibrate_composed_sampled_gaussian_noise,
    calibrate_sampled_gaussian_noise,
)


def _linear_model():
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))


def _tiny_loader():
    return DataLoader(
        TensorDataset(
            torch.tensor(
                [
                    [[[1.0, 0.0], [0.0, 0.0]]],
                    [[[0.0, 1.0], [0.0, 0.0]]],
                    [[[0.0, 0.0], [1.0, 0.0]]],
                    [[[0.0, 0.0], [0.0, 1.0]]],
                ]
            ),
            torch.tensor([0, 1, 0, 1]),
        ),
        batch_size=4,
        shuffle=False,
    )


def test_alpha_bound_controls_softmax_without_projection_or_mix():
    n, kappa_w = 10, 2.0
    alpha = alpha_max_for_weight_factor(n, kappa_w)
    scores = torch.tensor([1.0] + [0.0] * (n - 1), dtype=torch.float64)
    weights = torch.softmax(alpha * scores, dim=0)
    assert math.isclose(float(weights.max()), kappa_w / n, rel_tol=1e-12)
    assert math.isclose(float(weights.sum()), 1.0, rel_tol=1e-12)


def test_bounded_scores_are_in_public_unit_interval():
    scores = bounded_distance_scores(torch.tensor([0.0, 1.0, 4.0]), 2.0)
    assert torch.equal(scores, torch.tensor([0.0, 0.5, 1.0]))


def test_vectorized_dpsgd_matches_loop_without_noise():
    torch.manual_seed(12)
    first = _linear_model()
    second = _linear_model()
    second.load_state_dict(first.state_dict())
    vectorized, stats_v = local_dpsgd_train(
        first,
        _tiny_loader(),
        device="cpu",
        lr=0.01,
        local_epochs=1,
        clip_norm=1.0,
        noise_multiplier=0.0,
        backend="vectorized",
    )
    loop, stats_l = local_dpsgd_train(
        second,
        _tiny_loader(),
        device="cpu",
        lr=0.01,
        local_epochs=1,
        clip_norm=1.0,
        noise_multiplier=0.0,
        backend="loop",
    )
    assert stats_v.steps == stats_l.steps == 1
    for key in vectorized:
        assert torch.allclose(vectorized[key], loop[key], atol=1e-7)


def test_noise_calibration_targets_same_accountant():
    sigma = calibrate_sampled_gaussian_noise(
        target_epsilon=3.56,
        delta=1e-5,
        sampling_rate=0.05,
        steps=40,
    )
    accountant = RDPAccountant()
    accountant.add_sampled_gaussian(
        channel="model",
        sampling_rate=0.05,
        noise_multiplier=sigma,
        steps=40,
    )
    epsilon, _ = accountant.epsilon(1e-5)
    assert abs(epsilon - 3.56) < 2e-3


def test_composed_model_and_loss_calibration_targets_total_epsilon():
    ratio = 2.5
    sigma = calibrate_composed_sampled_gaussian_noise(
        target_epsilon=3.56,
        delta=1e-5,
        channels=((0.05, 80, 1.0), (0.05, 40, ratio)),
    )
    accountant = RDPAccountant()
    accountant.add_sampled_gaussian(
        channel="model", sampling_rate=0.05, noise_multiplier=sigma, steps=80
    )
    accountant.add_sampled_gaussian(
        channel="loss",
        sampling_rate=0.05,
        noise_multiplier=ratio * sigma,
        steps=40,
    )
    epsilon, _ = accountant.epsilon(1e-5)
    assert abs(epsilon - 3.56) < 2e-3


def test_auxiliary_channel_algorithms_advertise_composed_target():
    assert get_algorithm("fedfdp").get_default_config()[
        "target_epsilon_includes_auxiliary_channels"
    ]
    assert get_algorithm("dpqffl").get_default_config()[
        "target_epsilon_includes_auxiliary_channels"
    ]


def test_scfar_uses_trusted_user_clipping_and_conservative_sensitivity():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    tuples = []
    for client_id, value in enumerate((0.1, 0.2, 50.0)):
        state = ClientState(client_id=client_id, battery_j=10.0)
        tuples.append(
            (
                {"weight": torch.tensor([[value]])},
                {
                    "client_id": client_id,
                    "dataset_size": 1,
                    "local_loss": 0.0,
                    "bytes_sent": 4,
                    "energy_j_consumed": 0.0,
                },
                state,
            )
        )
    result = get_algorithm("scfar_dp").server_aggregate(
        model,
        tuples,
        round_num=0,
        config={
            "user_clip_norm": 1.0,
            "distance_clip": 2.0,
            "far_alpha": 10.0,
            "kappa_w": 2.0,
            "alpha_bound_policy": "clip",
            "robust_reference": "mean",
            "num_byzantine": 0,
            "enable_central_dp": False,
            "sensitivity_mode": "conservative_2C",
        },
    )
    assert result.metrics["scfar_user_clip_rate"] > 0
    assert result.metrics["scfar_alpha_was_clipped"] is True
    assert result.metrics["max_client_weight"] <= 2.0 / 3.0 + 1e-12
    assert result.metrics["scfar_sensitivity"] == 2.0


def test_partial_variant_trains_and_transmits_one_group():
    model = _linear_model()
    state = ClientState(client_id=0, battery_j=100.0)
    algo = get_algorithm("sc_partial_far_dp")
    update, metadata = algo.client_update(
        model,
        _tiny_loader(),
        state,
        {
            **algo.get_default_config(),
            "warmup_rounds": 0,
            "num_layer_groups": 1,
            "rounds_per_layer": 1,
            "local_epochs": 1,
            "verbose_groups": False,
        },
    )
    assert metadata["active_group_idx"] == 0
    assert update
    assert metadata["compression_ratio"] <= 1.0
