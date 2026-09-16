"""Contract tests for the private-gradient local-DP FAR lane."""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from algorithms.base import ClientState, get_algorithm
from privacy.local_dpsgd import private_gradient_release_fixed_without_replacement
from robustness.aggregators import (
    capped_inverse_variance_weights,
    noise_aware_centered_clipping,
)


def _classification_model():
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, -0.1], [-0.3, 0.4]]))
    return model


def _loader(size=20, batch_size=4):
    x = torch.arange(2 * size, dtype=torch.float32).reshape(size, 2) / 10.0
    y = torch.arange(size) % 2
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def test_private_gradient_uses_exact_nonempty_batch_and_does_not_mutate_model():
    torch.manual_seed(7)
    model = _classification_model()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    gradient, stats = private_gradient_release_fixed_without_replacement(
        model,
        _loader(),
        device="cpu",
        batch_size=5,
        clip_norm=0.7,
        noise_multiplier=0.0,
        backend="loop",
    )

    assert stats.steps == 1
    assert stats.examples == 5
    assert stats.empty_steps == 0
    assert stats.normalization_denominator == pytest.approx(5.0)
    assert all(
        torch.equal(value, before[name]) for name, value in model.state_dict().items()
    )
    norm = math.sqrt(
        sum(float(value.double().square().sum()) for value in gradient.values())
    )
    assert norm <= 0.7 + 1e-6


def test_private_gradient_noise_free_oracle_is_opt_in_and_batch_paired():
    torch.manual_seed(17)
    noisy, stats = private_gradient_release_fixed_without_replacement(
        _classification_model(),
        _loader(),
        device="cpu",
        batch_size=5,
        clip_norm=0.7,
        noise_multiplier=1.5,
        backend="loop",
        return_noise_free_oracle=True,
    )

    clean = stats.noise_free_delta_oracle
    assert clean is not None
    assert clean.keys() == noisy.keys()
    assert any(not torch.equal(noisy[name], clean[name]) for name in noisy)
    assert stats.mean_noise_norm > 0.0
    released_noise_norm = math.sqrt(
        sum(
            float((noisy[name] - clean[name]).double().square().sum()) for name in noisy
        )
    )
    assert released_noise_norm == pytest.approx(stats.mean_noise_norm / 5.0)


def test_capped_inverse_variance_weights_reduce_to_uniform_and_respect_cap():
    uniform = capped_inverse_variance_weights(
        torch.ones(5, dtype=torch.float64), max_weight_ratio=2.0
    )
    assert torch.allclose(uniform, torch.full_like(uniform, 0.2))

    heterogeneous = capped_inverse_variance_weights(
        torch.tensor([0.01, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64),
        max_weight_ratio=2.0,
    )
    assert heterogeneous.sum() == pytest.approx(1.0)
    assert float(heterogeneous.max()) <= 0.4 + 1e-12
    assert heterogeneous[0] > heterogeneous[-1]


def test_noise_aware_reference_has_replace_one_influence_bound():
    vectors = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [-0.1, 0.0], [0.0, 0.1]],
        dtype=torch.float64,
    )
    replacement = vectors.clone()
    replacement[0] = torch.tensor([100.0, -100.0], dtype=torch.float64)
    variances = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    kwargs = {
        "anchor": torch.zeros(2, dtype=torch.float64),
        "tau": 0.5,
        "noise_variances": variances,
        "max_weight_ratio": 2.0,
        "output_radius": 1.0,
    }
    first, diagnostics = noise_aware_centered_clipping(
        vectors, return_diagnostics=True, **kwargs
    )
    second = noise_aware_centered_clipping(replacement, **kwargs)
    observed = float(torch.linalg.vector_norm(first - second))
    assert observed <= diagnostics["noise_aware_reference_replace_one_bound"] + 1e-12
    assert float(torch.linalg.vector_norm(first)) <= 1.0 + 1e-12


def test_algorithm_client_releases_gradient_and_records_wor_accounting():
    torch.manual_seed(11)
    algorithm = get_algorithm("ldp_gradient_far")
    state = ClientState(client_id=0, battery_j=100.0)
    config = {
        **algorithm.get_default_config(),
        "device": "cpu",
        "enable_dp": True,
        "target_epsilon": None,
        "noise_multiplier": 2.0,
        "delta": 1e-5,
        "privacy_public_dataset_size": 20,
        "privacy_sampling_rate_override": 0.25,
        "fixed_batch_size": 5,
        "fixed_steps_per_round": 1,
        "clip_norm": 0.7,
        "per_sample_backend": "loop",
    }
    gradient, metadata = algorithm.client_update(
        _classification_model(), _loader(), state, config
    )
    assert gradient
    assert metadata["privacy_release_object"] == "averaged_clipped_gradient"
    assert metadata["far_update_mode"] == "single_step_gradient"
    assert metadata["privacy_batch_nonempty_by_construction"] is True
    assert metadata["privacy_query_sensitivity_l2"] == pytest.approx(2 * 0.7 / 5)
    assert metadata["privacy_gaussian_std_per_coordinate"] == pytest.approx(2 * 0.7 / 5)
    assert metadata["privacy_accounting_noise_multiplier"] == pytest.approx(1.0)
    assert metadata["privacy_epsilon"] is not None
    assert metadata["clip_rate"] is None
    assert "local_dp_noise_free_update_oracle" not in metadata
    assert "dp_noise_norm_mean" not in metadata


def test_algorithm_oracle_diagnostics_are_explicit_and_not_protocol_fields():
    torch.manual_seed(19)
    algorithm = get_algorithm("ldp_gradient_far")
    state = ClientState(client_id=0, battery_j=100.0)
    config = {
        **algorithm.get_default_config(),
        "device": "cpu",
        "enable_dp": True,
        "target_epsilon": None,
        "noise_multiplier": 2.0,
        "privacy_public_dataset_size": 20,
        "privacy_sampling_rate_override": 0.25,
        "fixed_batch_size": 5,
        "fixed_steps_per_round": 1,
        "clip_norm": 0.7,
        "per_sample_backend": "loop",
        "enable_oracle_diagnostics": True,
    }
    gradient, metadata = algorithm.client_update(
        _classification_model(), _loader(), state, config
    )
    clean = metadata["local_dp_noise_free_update_oracle"]
    assert clean.keys() == gradient.keys()
    assert metadata["dp_noise_norm_mean"] > 0.0
    assert metadata["privacy_release_object"] == "averaged_clipped_gradient"


def test_no_dp_gradient_control_exposes_clip_rate_for_calibration():
    torch.manual_seed(13)
    algorithm = get_algorithm("ldp_gradient_far")
    state = ClientState(client_id=0, battery_j=100.0)
    config = {
        **algorithm.get_default_config(),
        "device": "cpu",
        "enable_dp": False,
        "target_epsilon": None,
        "noise_multiplier": 0.0,
        "privacy_public_dataset_size": 20,
        "privacy_sampling_rate_override": 0.25,
        "fixed_batch_size": 5,
        "fixed_steps_per_round": 1,
        "clip_norm": 0.01,
        "per_sample_backend": "loop",
    }
    _, metadata = algorithm.client_update(
        _classification_model(), _loader(), state, config
    )
    assert metadata["privacy_level"] == "none"
    assert metadata["clip_rate"] == pytest.approx(1.0)


def _server_updates():
    result = []
    for client_id, (row, variance) in enumerate(
        zip(
            ([0.1, 0.0], [0.0, 0.2], [-0.1, 0.0], [0.0, -0.2]), (0.01, 0.02, 0.03, 0.04)
        )
    ):
        result.append(
            (
                {"weight": torch.tensor([row], dtype=torch.float32)},
                {
                    "client_id": client_id,
                    "privacy_gradient_release": True,
                    "privacy_upload_noise_variance_per_coordinate": variance,
                    "privacy_noise_multiplier_scale_public": 1.0,
                    "privacy_epsilon": 3.0,
                    "privacy_delta": 1e-5,
                    "privacy_sampling_scheme": "fixed_without_replacement",
                    "privacy_adjacency": "replace_one",
                    "privacy_accounting_assumption": (
                        "fixed_size_without_replacement_rdp_replace_one"
                    ),
                    "privacy_noise_multiplier": 2.0,
                    "model_steps": 1,
                    "far_update_mode": "single_step_gradient",
                    "bytes_sent": 8,
                    "energy_j_consumed": 0.0,
                    "local_loss": 0.0,
                },
                ClientState(client_id=client_id, battery_j=10.0),
            )
        )
    return result


def test_algorithm_server_uses_raw_distance_cap_and_noise_aware_reference():
    algorithm = get_algorithm("ldp_gradient_far")
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    config = {
        **algorithm.get_default_config(),
        "expected_num_clients": 4,
        "far_server_clip_norm": 1.0,
        "reference_clip_radius": 0.5,
        "far_alpha": 0.1,
        "kappa_w": 2.0,
    }
    result = algorithm.server_aggregate(model, _server_updates(), 0, config)
    metrics = result.metrics
    expected_cap = math.log(2.0 * 3.0 / 2.0) / 2.0
    assert metrics["ldp_gradient_far_alpha_cap"] == pytest.approx(expected_cap)
    assert metrics["far_score_mode"] == "raw_distance"
    assert 0.0 < metrics["far_max_weight"] <= 1.0
    assert metrics["far_public_score_range"] == pytest.approx(2.0)
    assert metrics["far_tilt_influence_certificate_claimed"] is True
    assert metrics["noise_aware_reference_weight_cap_respected"] is True


def test_scientific_device_contract_applies_to_every_reference_arm():
    algorithm = get_algorithm("ldp_gradient_far")
    model = torch.nn.Linear(2, 1, bias=False)
    updates = _server_updates()
    for _, metadata, _ in updates:
        metadata["privacy_compute_device"] = "mps:0"
    config = {
        **algorithm.get_default_config(),
        "expected_num_clients": 4,
        "far_server_clip_norm": 1.0,
        "robust_reference": "rfa",
        "rcig_required_private_device": "mps",
    }
    result = algorithm.server_aggregate(model, updates, 0, config)
    assert result.metrics["ldp_gradient_far_private_gradient_mps_fraction"] == 1.0
    assert result.metrics["ldp_gradient_far_private_compute_devices"] == ["mps:0"]
    assert result.metrics["ldp_gradient_far_private_compute_device"] == "mps:0"

    updates[0][1]["privacy_compute_device"] = "cpu"
    with pytest.raises(ValueError, match="compute-device audit failed"):
        algorithm.server_aggregate(model, updates, 0, config)


def test_oracle_error_decomposition_is_exact_and_never_drives_aggregation():
    algorithm = get_algorithm("ldp_gradient_far")
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    updates = _server_updates()
    for index, (update, metadata, state) in enumerate(updates):
        metadata["local_dp_noise_free_update_oracle"] = {
            name: 0.8 * value.clone() for name, value in update.items()
        }
        metadata["dp_noise_norm_mean"] = 0.01 * (index + 1)
        metadata["is_byzantine"] = index == 0

    result = algorithm.server_aggregate(
        model,
        updates,
        0,
        {
            **algorithm.get_default_config(),
            "expected_num_clients": 4,
            "far_server_clip_norm": 1.0,
            "reference_clip_radius": 0.5,
            "far_alpha": 0.1,
            "kappa_w": 2.0,
            "enable_oracle_diagnostics": True,
        },
    )
    metrics = result.metrics
    assert metrics["far_honest_oracle_count"] == 3
    assert metrics["far_honest_weight_mass_oracle"] == pytest.approx(
        1.0 - metrics["byzantine_weight_mass_oracle"]
    )
    assert metrics["far_byzantine_displacement_norm_oracle"] > 0.0
    assert metrics["far_honest_fixed_weight_dp_noise_norm_oracle"] > 0.0
    assert metrics["far_aggregate_error_to_clean_honest_center_norm_oracle"] > 0.0
    assert metrics["far_error_decomposition_residual_norm_oracle"] < 1e-6


def test_algorithm_rejects_alpha_above_raw_distance_certificate():
    algorithm = get_algorithm("ldp_gradient_far")
    model = torch.nn.Linear(2, 1, bias=False)
    with pytest.raises(ValueError, match="exceeds the certified"):
        algorithm.server_aggregate(
            model,
            _server_updates(),
            0,
            {
                **algorithm.get_default_config(),
                "expected_num_clients": 4,
                "far_server_clip_norm": 1.0,
                "far_alpha": 10.0,
                "kappa_w": 2.0,
                "tilt_bound_policy": "error",
            },
        )


def test_algorithm_allows_alpha_above_optional_guardrail_as_diagnostic():
    algorithm = get_algorithm("ldp_gradient_far")
    model = torch.nn.Linear(2, 1, bias=False)
    result = algorithm.server_aggregate(
        model,
        _server_updates(),
        0,
        {
            **algorithm.get_default_config(),
            "expected_num_clients": 4,
            "far_server_clip_norm": 1.0,
            "far_alpha": 10.0,
            "kappa_w": 2.0,
            "tilt_bound_policy": "diagnostic",
        },
    )
    assert result.metrics["ldp_gradient_far_requested_alpha"] == pytest.approx(10.0)
    assert result.metrics["ldp_gradient_far_effective_alpha"] == pytest.approx(10.0)
    assert result.metrics["ldp_gradient_far_alpha_cap_exceeded"] is True
    assert result.metrics["far_tilt_influence_certificate_claimed"] is False


def test_negative_alpha_reports_true_nonnegative_logit_range_and_weight_ratio():
    algorithm = get_algorithm("ldp_gradient_far")
    model = torch.nn.Linear(2, 1, bias=False)
    result = algorithm.server_aggregate(
        model,
        _server_updates(),
        0,
        {
            **algorithm.get_default_config(),
            "expected_num_clients": 4,
            "far_server_clip_norm": 1.0,
            "far_alpha": -2.0,
            "kappa_w": 2.0,
            "tilt_bound_policy": "diagnostic",
        },
    )
    metrics = result.metrics
    expected = abs(metrics["far_alpha"]) * metrics["far_score_span"]
    assert metrics["far_logit_range"] == pytest.approx(expected)
    assert metrics["far_logit_range"] >= 0.0
    assert math.log(metrics["far_weight_ratio"]) == pytest.approx(
        metrics["far_logit_range"], abs=1e-6
    )
