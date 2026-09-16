"""Integration contract for temporal RCIG inside LDP-Gradient-FAR."""

from __future__ import annotations

import copy

import pytest
import torch

from algorithms.base import ClientState, get_algorithm
from metrics.rcig_evaluation import detach_rcig_evaluation_oracles


def _model() -> torch.nn.Module:
    model = torch.nn.Linear(32, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    return model


def _updates(round_num: int, *, shift: float = 0.0):
    rows = []
    for client_id, value in enumerate((0.01, 0.02, -0.01, -0.02)):
        vector = torch.zeros(1, 32, dtype=torch.float32)
        vector[0, 0] = value + shift
        rows.append(
            (
                {"weight": vector},
                {
                    "client_id": client_id,
                    "round_num": round_num + 1,
                    "privacy_compute_device": "mps:0",
                    "privacy_gradient_release": True,
                    "privacy_upload_noise_variance_per_coordinate": 0.01,
                    "privacy_noise_multiplier_scale_public": 1.0,
                    "privacy_epsilon": 3.0,
                    "privacy_delta": 1e-5,
                    "privacy_sampling_scheme": "fixed_without_replacement",
                    "privacy_adjacency": "replace_one",
                    "privacy_accounting_assumption": (
                        "fixed_size_without_replacement_rdp_replace_one"
                    ),
                    "privacy_noise_multiplier": 1.0,
                    "privacy_sensitivity_multiplier": 2.0,
                    "privacy_fixed_batch_size": 10,
                    "dataset_size": 200,
                    "model_steps": 1,
                    "far_update_mode": "single_step_gradient",
                    "bytes_sent": 128,
                    "energy_j_consumed": 0.0,
                    "local_loss": 0.0,
                },
                ClientState(client_id=client_id, battery_j=10.0),
            )
        )
    return rows


def _config(**overrides):
    algorithm = get_algorithm("ldp_gradient_far")
    config = {
        **algorithm.get_default_config(),
        "robust_reference": "rcig_temporal",
        "expected_num_clients": 4,
        "far_server_clip_norm": 1.0,
        "far_reference_output_clip_norm": 1.0,
        "rcig_reference_output_radius": 1.0,
        "reference_clip_radius": 1.0,
        "clip_norm": 1.0,
        "enable_dp": True,
        "noise_multiplier": 1.0,
        "target_epsilon": None,
        "fixed_batch_size": 10,
        "privacy_public_dataset_size": 200,
        "privacy_sampling_rate_override": 0.05,
        "fixed_steps_per_round": 1,
        "privacy_num_rounds": 40,
        "sampling_scheme": "fixed_without_replacement",
        "privacy_adjacency": "replace_one",
        "rcig_covariance_registry": "authenticated_public_mechanism",
        "far_alpha": 0.2,
        "kappa_w": 2.0,
        "rcig_gate_window": 1,
        "rcig_old_window": 1,
        "rcig_new_window": 1,
        "rcig_public_subspace_dimension": 32,
        "rcig_public_subspace_seed": 7,
        "rcig_process_variance": 1e-5,
        "rcig_covariance_ridge": 1e-6,
        "rcig_innovation_threshold": 3.0,
        "rcig_isotropic_innovation_threshold": 3.0,
        "rcig_euclidean_innovation_threshold": 0.1,
        "rcig_min_accepted_mass": 1.0,
        "far_server_lr": 0.1,
    }
    config.update(overrides)
    return config


def _prime(algorithm, model, config):
    results = []
    for round_num in range(3):
        results.append(
            algorithm.server_aggregate(
                model,
                _updates(round_num, shift=0.001 * round_num),
                round_num,
                config,
            )
        )
    return results


def test_rcig_cold_start_is_uniform_then_uses_registered_alpha() -> None:
    algorithm = get_algorithm("ldp_gradient_far")
    model = _model()
    config = _config()
    warmup = _prime(algorithm, model, config)
    assert all(result.metrics["rcig_warmup"] for result in warmup)
    assert all(result.metrics["far_alpha"] == 0.0 for result in warmup)
    assert all(
        result.metrics["ldp_gradient_far_effective_alpha"] == 0.0 for result in warmup
    )

    deployed = algorithm.server_aggregate(model, _updates(3), 3, config)
    assert deployed.metrics["rcig_history_ready"] is True
    assert deployed.metrics["rcig_reference_strictly_past"] is True
    assert deployed.metrics["far_alpha"] == pytest.approx(0.2)
    assert deployed.metrics["ldp_gradient_far_effective_alpha"] == pytest.approx(0.2)
    assert deployed.metrics["privacy_epsilon_max"] == pytest.approx(3.0)
    assert deployed.metrics["rcig_private_gradient_mps_fraction"] == 1.0
    assert deployed.metrics["rcig_server_aggregation_device"] == "cpu"
    assert deployed.metrics["rcig_server_aggregation_dtype"] == "torch.float64"


def test_current_upload_cannot_change_deployed_rcig_reference() -> None:
    first = get_algorithm("ldp_gradient_far")
    second = get_algorithm("ldp_gradient_far")
    config = _config()
    _prime(first, _model(), config)
    _prime(second, _model(), config)

    first.server_aggregate(_model(), _updates(3, shift=0.75), 3, config)
    second.server_aggregate(_model(), _updates(3, shift=-0.75), 3, config)
    assert torch.equal(
        first._rcig_last_deployed_reference,  # type: ignore[attr-defined]
        second._rcig_last_deployed_reference,  # type: ignore[attr-defined]
    )


def test_rcig_is_invariant_to_tuple_order_and_rejects_duplicate_ids() -> None:
    first = get_algorithm("ldp_gradient_far")
    second = get_algorithm("ldp_gradient_far")
    model_a, model_b = _model(), _model()
    config = _config()
    for round_num in range(4):
        original = _updates(round_num, shift=0.001 * round_num)
        permuted = [original[index] for index in (2, 0, 3, 1)]
        result_a = first.server_aggregate(model_a, original, round_num, config)
        result_b = second.server_aggregate(model_b, permuted, round_num, config)
        assert torch.allclose(
            result_a.new_weights["weight"], result_b.new_weights["weight"]
        )
        model_a.load_state_dict(result_a.new_weights)
        model_b.load_state_dict(result_b.new_weights)

    duplicate = _updates(4)
    duplicate[1][1]["client_id"] = 0
    with pytest.raises(ValueError, match="unique"):
        first.server_aggregate(model_a, duplicate, 4, config)


def test_oracle_fields_never_drive_rcig_reference_or_model() -> None:
    first = get_algorithm("ldp_gradient_far")
    second = get_algorithm("ldp_gradient_far")
    model_a, model_b = _model(), _model()
    config = _config(enable_oracle_diagnostics=True)
    _prime(first, model_a, config)
    _prime(second, model_b, config)
    base = _updates(3)
    altered = copy.deepcopy(base)
    for index, (update, metadata, _) in enumerate(altered):
        metadata["local_dp_noise_free_update_oracle"] = {
            name: torch.full_like(value, 1000.0 * (index + 1))
            for name, value in update.items()
        }
        metadata["dp_noise_norm_mean"] = 1e6 * (index + 1)
        metadata["is_byzantine"] = index % 2 == 0
    result_a = first.server_aggregate(model_a, base, 3, config)
    result_b = second.server_aggregate(model_b, altered, 3, config)
    assert torch.equal(
        first._rcig_last_deployed_reference,  # type: ignore[attr-defined]
        second._rcig_last_deployed_reference,  # type: ignore[attr-defined]
    )
    assert torch.allclose(
        result_a.new_weights["weight"], result_b.new_weights["weight"]
    )


def test_required_oracle_boundary_rejects_unsanitized_server_metadata() -> None:
    algorithm = get_algorithm("ldp_gradient_far")
    model = _model()
    config = _config(
        enable_oracle_diagnostics=True,
        rcig_oracle_separation_required=True,
    )
    rows = _updates(0)
    rows[0][1]["local_dp_noise_free_update_oracle"] = {"weight": torch.zeros(1, 32)}
    with pytest.raises(ValueError, match="simulator-only oracle fields"):
        algorithm.server_aggregate(model, rows, 0, config)


def test_rcig_rejects_forged_public_variance_and_signature_change() -> None:
    algorithm = get_algorithm("ldp_gradient_far")
    model = _model()
    config = _config()
    forged = _updates(0)
    # Forge all mutually related client fields coherently.  The server still
    # rejects them because it reconstructs the nominal channel from its own
    # immutable registry instead of trusting this self-consistent bundle.
    forged[0][1].update(
        {
            "privacy_upload_noise_variance_per_coordinate": 0.04,
            "privacy_noise_multiplier": 2.0,
            "privacy_noise_multiplier_scale_public": 2.0,
        }
    )
    with pytest.raises(ValueError, match="disagrees"):
        algorithm.server_aggregate(model, forged, 0, config)

    algorithm.server_aggregate(model, _updates(0), 0, config)
    with pytest.raises(ValueError, match="configuration changed"):
        algorithm.server_aggregate(
            model,
            _updates(1),
            1,
            _config(rcig_public_subspace_seed=99),
        )


def test_external_attack_labels_cannot_change_server_aggregate() -> None:
    first = get_algorithm("ldp_gradient_far")
    second = get_algorithm("ldp_gradient_far")
    rows_a = _updates(0)
    rows_b = copy.deepcopy(rows_a)
    for index, (_, metadata, _) in enumerate(rows_a):
        metadata.update(
            {
                "is_byzantine": index == 0,
                "attack_name": "bf" if index == 0 else "none",
                "attack_window_active": True,
            }
        )
    for index, (_, metadata, _) in enumerate(rows_b):
        metadata.update(
            {
                "is_byzantine": index >= 2,
                "attack_name": "alie" if index >= 2 else "none",
                "attack_window_active": True,
            }
        )
    safe_a, _ = detach_rcig_evaluation_oracles(
        rows_a, enabled=False, strip_attack_oracles=True
    )
    safe_b, _ = detach_rcig_evaluation_oracles(
        rows_b, enabled=False, strip_attack_oracles=True
    )
    config = _config(external_attack_diagnostics=True)
    result_a = first.server_aggregate(_model(), safe_a, 0, config)
    result_b = second.server_aggregate(_model(), safe_b, 0, config)
    assert torch.equal(result_a.new_weights["weight"], result_b.new_weights["weight"])
    assert (
        result_a.metrics["_far_external_weight_diagnostics_payload"]
        == result_b.metrics["_far_external_weight_diagnostics_payload"]
    )
    assert result_a.metrics["far_attack_labels_visible_to_server_aggregate"] is False
    assert result_a.metrics["far_attack_config_visible_to_server_aggregate"] is False

    with pytest.raises(ValueError, match="attack-oracle fields"):
        first.server_aggregate(_model(), rows_a, 0, config)

    with pytest.raises(ValueError, match="attack configuration"):
        first.server_aggregate(
            _model(),
            safe_a,
            0,
            {**config, "attack": {"enabled": True, "client_ids": [0]}},
        )


@pytest.mark.parametrize("mode", ["full", "isotropic", "euclidean"])
def test_all_rcig_modes_expose_paired_diagnostics(mode: str) -> None:
    algorithm = get_algorithm("ldp_gradient_far")
    model = _model()
    config = _config(rcig_covariance_mode=mode)
    _prime(algorithm, model, config)
    result = algorithm.server_aggregate(model, _updates(3), 3, config)
    for key in (
        "rcig_full_innovation_stat",
        "rcig_isotropic_innovation_stat",
        "rcig_euclidean_innovation_stat",
        "rcig_full_gate_active",
        "rcig_isotropic_gate_active",
        "rcig_euclidean_gate_active",
    ):
        assert key in result.metrics
    assert result.metrics["rcig_covariance_psd_certified"] is True
