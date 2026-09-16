"""Unit tests for the new DP reproduction and SC-Partial-FAR paths."""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from algorithms.base import ClientState, get_algorithm
from algorithms.sc_partial_far_dp import (
    alpha_max_for_weight_factor,
    bounded_distance_scores,
    certified_raw_distance_scfar_sensitivity,
    raw_distance_alpha_max_for_weight_factor,
    softmax_weight_factor_bound,
)
from metrics.robustness import update_norm_diagnostics
from privacy.local_dpsgd import local_dpsgd_train
from privacy.rdp import (
    RDPAccountant,
    calibrate_composed_sampled_gaussian_noise,
    calibrate_gaussian_noise,
    calibrate_sampled_gaussian_noise,
    calibrate_sampled_without_replacement_gaussian_noise,
    gaussian_rdp,
    sampled_without_replacement_gaussian_rdp,
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


def test_raw_distance_certificate_scales_alpha_and_improves_on_2c():
    n, c, kappa = 25, 1.4, 2.0
    alpha = raw_distance_alpha_max_for_weight_factor(n, kappa, c)
    distances = torch.tensor([2.0 * c] + [0.0] * (n - 1), dtype=torch.float64)
    weights = torch.softmax(alpha * distances, dim=0)
    assert float(weights.max()) == pytest.approx(kappa / n)

    delta_f = 2.0 * c / n
    sensitivity = certified_raw_distance_scfar_sensitivity(
        n=n,
        clip_norm=c,
        alpha=alpha,
        kappa_bound=kappa,
        reference_stability=delta_f,
    )
    assert sensitivity < 2.0 * c


def test_update_norm_diagnostics_uses_whole_multitensor_updates():
    updates = [
        (
            {"a": torch.tensor([3.0, 4.0]), "b": torch.tensor([0.0])},
            {},
            None,
        ),
        (
            {"a": torch.tensor([0.0, 0.0]), "b": torch.tensor([12.0])},
            {},
            None,
        ),
    ]
    metrics = update_norm_diagnostics(updates)
    assert metrics["received_update_norm_mean"] == 8.5
    assert metrics["received_update_norm_max"] == 12.0
    assert (
        metrics["received_update_norm_p10"]
        <= metrics["received_update_norm_p50"]
        <= metrics["received_update_norm_p90"]
        <= metrics["received_update_norm_p95"]
    )


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


def test_poisson_q_one_matches_one_fixed_full_batch_step():
    """At q=1 the Bernoulli subset is the complete local dataset."""

    torch.manual_seed(21)
    fixed_model = _linear_model()
    poisson_model = _linear_model()
    poisson_model.load_state_dict(fixed_model.state_dict())
    fixed, fixed_stats = local_dpsgd_train(
        fixed_model,
        _tiny_loader(),
        device="cpu",
        lr=0.01,
        local_epochs=1,
        clip_norm=1.0,
        noise_multiplier=0.0,
        sampling_scheme="fixed_minibatch",
    )
    poisson, poisson_stats = local_dpsgd_train(
        poisson_model,
        _tiny_loader(),
        device="cpu",
        lr=0.01,
        local_epochs=99,
        clip_norm=1.0,
        noise_multiplier=0.0,
        sampling_scheme="poisson",
        poisson_sampling_rate=1.0,
        poisson_steps_per_round=1,
    )
    assert fixed_stats.steps == poisson_stats.steps == 1
    assert poisson_stats.sampling_scheme == "poisson"
    assert poisson_stats.sampling_rate == 1.0
    assert poisson_stats.expected_batch_size == 4.0
    for key in fixed:
        assert torch.allclose(fixed[key], poisson[key], atol=1e-7)


def test_fixed_without_replacement_draws_exact_public_batch_each_step():
    torch.manual_seed(27)
    model = _linear_model()
    _, stats = local_dpsgd_train(
        model,
        _tiny_loader(),
        device="cpu",
        lr=0.01,
        local_epochs=99,
        clip_norm=1.0,
        noise_multiplier=0.0,
        sampling_scheme="fixed_without_replacement",
        fixed_batch_size=2,
        fixed_steps_per_round=3,
        fixed_normalization_denominator=2.0,
    )
    assert stats.sampling_scheme == "fixed_without_replacement"
    assert stats.steps == 3
    assert stats.examples == 6
    assert stats.empty_steps == 0
    assert stats.sampling_rate == 0.5
    assert stats.expected_batch_size == 2.0
    assert stats.normalization_denominator == 2.0


def test_noise_free_counterfactual_matches_zero_noise_trajectory():
    torch.manual_seed(101)
    model = _linear_model()
    update, stats = local_dpsgd_train(
        model,
        _tiny_loader(),
        device="cpu",
        lr=0.01,
        local_epochs=1,
        clip_norm=1.0,
        noise_multiplier=0.0,
        track_noise_free_counterfactual=True,
    )
    assert stats.noise_free_delta_oracle is not None
    for key in update:
        assert torch.equal(update[key], stats.noise_free_delta_oracle[key])


def test_noise_free_counterfactual_is_distinct_when_noise_is_added():
    torch.manual_seed(102)
    model = _linear_model()
    update, stats = local_dpsgd_train(
        model,
        _tiny_loader(),
        device="cpu",
        lr=0.01,
        local_epochs=1,
        clip_norm=1.0,
        noise_multiplier=1.0,
        track_noise_free_counterfactual=True,
    )
    assert stats.noise_free_delta_oracle is not None
    assert any(
        not torch.equal(update[key], stats.noise_free_delta_oracle[key])
        for key in update
    )


def test_poisson_empty_subsets_are_valid_zero_signal_dp_steps(monkeypatch):
    """An empty Poisson subset still consumes one Gaussian-mechanism step."""

    model = _linear_model()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}

    def never_include(*shape, **kwargs):
        return torch.ones(*shape, **kwargs)

    monkeypatch.setattr(torch, "rand", never_include)
    update, stats = local_dpsgd_train(
        model,
        _tiny_loader(),
        device="cpu",
        lr=0.01,
        local_epochs=1,
        clip_norm=1.0,
        noise_multiplier=0.0,
        sampling_scheme="poisson",
        poisson_sampling_rate=0.05,
        poisson_steps_per_round=2,
    )
    assert stats.steps == 2
    assert stats.empty_steps == 2
    assert stats.examples == 0
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
        assert torch.count_nonzero(update[key]) == 0


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


def test_fixed_without_replacement_accountant_targets_replace_one_budget():
    implementation_sigma = calibrate_sampled_without_replacement_gaussian_noise(
        target_epsilon=4.0,
        delta=1e-5,
        sampling_rate=0.05,
        steps=60,
        sensitivity_multiplier=2.0,
    )
    accountant = RDPAccountant()
    accountant.add_sampled_without_replacement_gaussian(
        channel="model",
        sampling_rate=0.05,
        # The implementation adds sigma*C but replace-one sensitivity is 2C.
        noise_multiplier=implementation_sigma / 2.0,
        steps=60,
    )
    epsilon, _ = accountant.epsilon(1e-5)
    assert abs(epsilon - 4.0) < 2e-3
    assert sampled_without_replacement_gaussian_rdp(4, 0.05, 1.2) > 0.0


def test_ordinary_gaussian_rdp_and_calibration_are_explicit_q_one_lane():
    order, multiplier, steps = 8, 1.7, 25
    assert math.isclose(
        gaussian_rdp(order, multiplier),
        order / (2.0 * multiplier**2),
        rel_tol=1e-12,
    )

    sigma = calibrate_gaussian_noise(
        target_epsilon=3.0,
        delta=1e-5,
        steps=steps,
    )
    accountant = RDPAccountant()
    accountant.add_gaussian(
        channel="central_model", noise_multiplier=sigma, steps=steps
    )
    epsilon, _ = accountant.epsilon(1e-5)
    assert abs(epsilon - 3.0) < 2e-3


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
    assert 0.0 <= result.metrics["scfar_score_span"] <= 1.0
    assert 1.0 <= result.metrics["scfar_weight_quadratic_concentration"] <= 3.0
    assert (
        result.metrics["scfar_preclip_norm_p10"]
        <= result.metrics["scfar_preclip_norm_p50"]
        <= result.metrics["scfar_preclip_norm_p90"]
        <= result.metrics["scfar_preclip_norm_p95"]
        <= result.metrics["scfar_preclip_norm_max"]
    )
    assert result.metrics["central_noise_to_clean_aggregate_ratio"] is None


def test_scfar_uniform_and_reference_rules_expose_the_expected_certificates():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    tuples = []
    for client_id, value in enumerate((-1.0, 0.0, 1.0, 1.0)):
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
    algo = get_algorithm("scfar_dp")
    common = {
        "user_clip_norm": 1.0,
        "distance_clip": 2.0,
        "far_alpha": 0.0,
        "kappa_w": 1.0,
        "alpha_bound_policy": "error",
        "robust_reference": "centered_clipping",
        "reference_clip_tau": 1.0,
        "anchor_mode": "fixed_zero",
        "enable_central_dp": False,
        "sensitivity_mode": "automatic_certified",
        "honest_outlier_client_ids": [0],
    }
    uniform = algo.server_aggregate(
        model,
        tuples,
        round_num=0,
        config={**common, "scfar_aggregation_rule": "uniform"},
    )
    reference = algo.server_aggregate(
        model,
        tuples,
        round_num=0,
        config={**common, "scfar_aggregation_rule": "reference"},
    )
    assert uniform.metrics["scfar_sensitivity"] == 0.5
    assert reference.metrics["scfar_sensitivity"] == 0.5
    assert uniform.metrics["honest_outlier_count_oracle"] == 1
    assert math.isclose(
        uniform.metrics["honest_outlier_weight_mass_oracle"], 0.25, rel_tol=1e-7
    )
    assert math.isclose(
        uniform.metrics["scfar_weight_quadratic_concentration"], 1.0, rel_tol=1e-12
    )
    assert math.isclose(
        uniform.metrics["scfar_clean_aggregate_norm"], 0.25, rel_tol=1e-7
    )


def test_raw_distance_ablation_uses_untransformed_distances_and_2c_sensitivity():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    tuples = []
    for client_id, value in enumerate((-1.0, 0.0, 0.5, 1.0)):
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
                ClientState(client_id=client_id, battery_j=10.0),
            )
        )
    result = get_algorithm("scfar_dp").server_aggregate(
        model,
        tuples,
        round_num=0,
        config={
            "user_clip_norm": 1.0,
            "distance_clip": 0.1,
            "scfar_aggregation_rule": "far_raw_distance",
            "far_alpha": 0.5,
            "kappa_w": 2.0,
            "alpha_bound_policy": "error",
            "robust_reference": "centered_clipping",
            "reference_clip_tau": 1.0,
            "anchor_mode": "fixed_zero",
            "enable_central_dp": False,
            "sensitivity_mode": "conservative_2C",
        },
    )
    metrics = result.metrics
    assert metrics["scfar_score_transform"] == "raw_distance"
    assert metrics["scfar_score_is_bounded"] is False
    assert metrics["scfar_certified_kappa"] is None
    assert metrics["scfar_score_saturation_rate"] is None
    assert metrics["scfar_score_max"] > 0.1
    assert metrics["scfar_sensitivity"] == 2.0
    assert metrics["scfar_sensitivity_mode"] == "conservative_2C"
    assert metrics["scfar_weight_bound_holds"] is None
    assert (
        metrics["scfar_certified_kappa_source"]
        == "none_raw_distance_uses_conservative_2C"
    )


def test_raw_distance_fcc_can_use_the_public_2c_sensitivity_certificate():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    tuples = []
    values = torch.linspace(-1.0, 1.0, 10).tolist()
    for client_id, value in enumerate(values):
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
                ClientState(client_id=client_id, battery_j=10.0),
            )
        )
    alpha = raw_distance_alpha_max_for_weight_factor(10, 2.0, 1.0)
    result = get_algorithm("scfar_dp").server_aggregate(
        model,
        tuples,
        round_num=0,
        config={
            "user_clip_norm": 1.0,
            "scfar_aggregation_rule": "far_raw_distance",
            "far_alpha": alpha,
            "kappa_w": 2.0,
            "alpha_bound_policy": "error",
            "robust_reference": "centered_clipping",
            "reference_clip_tau": 1.0,
            "anchor_mode": "fixed_zero",
            "anchor_clip_norm": 1.0,
            "enable_central_dp": False,
            "sensitivity_mode": "proved_raw_distance_reference_bound",
        },
    )
    metrics = result.metrics
    assert metrics["scfar_public_score_range"] == pytest.approx(2.0)
    assert metrics["scfar_weight_bound_holds"] is True
    assert metrics["scfar_certified_kappa"] == pytest.approx(2.0)
    assert metrics["scfar_sensitivity"] < 2.0
    assert metrics["scfar_sensitivity_mode"] == (
        "proved_raw_distance_reference_bound"
    )


def test_scfar_dp_reports_finite_noise_to_clean_aggregate_ratio():
    torch.manual_seed(412)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    tuples = []
    for client_id, value in enumerate((0.2, 0.3, 0.4, 0.5)):
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
                ClientState(client_id=client_id, battery_j=10.0),
            )
        )
    result = get_algorithm("scfar_dp").server_aggregate(
        model,
        tuples,
        round_num=0,
        config={
            "user_clip_norm": 1.0,
            "distance_clip": 2.0,
            "scfar_aggregation_rule": "uniform",
            "far_alpha": 0.0,
            "kappa_w": 1.0,
            "alpha_bound_policy": "error",
            "robust_reference": "centered_clipping",
            "reference_clip_tau": 1.0,
            "anchor_mode": "fixed_zero",
            "enable_central_dp": True,
            "central_noise_multiplier": 1.0,
            "target_epsilon": None,
            "delta": 1e-5,
            "sensitivity_mode": "automatic_certified",
        },
    )
    expected = result.metrics["central_noise_norm"] / result.metrics[
        "scfar_clean_aggregate_norm"
    ]
    assert math.isfinite(result.metrics["central_noise_to_clean_aggregate_ratio"])
    assert math.isclose(
        result.metrics["central_noise_to_clean_aggregate_ratio"],
        expected,
        rel_tol=1e-12,
    )


def test_scfar_certified_noise_uses_public_kappa_not_observed_weights():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    tuples = []
    for client_id in range(4):
        state = ClientState(client_id=client_id, battery_j=10.0)
        tuples.append(
            (
                {"weight": torch.tensor([[0.25]])},
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
    alpha = 0.5
    result = get_algorithm("scfar_dp").server_aggregate(
        model,
        tuples,
        round_num=0,
        config={
            "user_clip_norm": 1.0,
            "distance_clip": 2.0,
            "far_alpha": alpha,
            "kappa_w": 2.0,
            "alpha_bound_policy": "error",
            "robust_reference": "centered_clipping",
            "reference_clip_tau": 1.0,
            "anchor_mode": "fixed_zero",
            "enable_central_dp": False,
            "sensitivity_mode": "automatic_certified",
        },
    )
    public_kappa = softmax_weight_factor_bound(4, alpha)
    assert result.metrics["scfar_analytical_kappa"] == 1.0
    assert math.isclose(
        result.metrics["scfar_certified_kappa"], public_kappa, rel_tol=1e-12
    )
    assert (
        result.metrics["scfar_certified_kappa"]
        > result.metrics["scfar_analytical_kappa"]
    )
    assert (
        result.metrics["scfar_certified_kappa_source"] == "public_score_range_and_alpha"
    )


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
