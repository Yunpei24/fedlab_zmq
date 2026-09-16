"""Tests for public active-parameter masks and sparse model deltas."""

from __future__ import annotations

import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from algorithms.base import ClientState
from algorithms.fedavg import FedAvg
from algorithms.parameter_masks import (
    configure_active_parameters,
    resolve_active_parameters,
)
from algorithms.sc_partial_far_dp import SensitivityControlledFAR
from models.registry import get_model


def _tiny_loader() -> DataLoader:
    generator = torch.Generator().manual_seed(7)
    return DataLoader(
        TensorDataset(
            torch.randn(12, 1, 28, 28, generator=generator),
            torch.arange(12) % 10,
        ),
        batch_size=4,
        shuffle=False,
    )


def test_lenet5_public_masks_have_expected_names_and_dimensions():
    model = get_model("lenet5", "fashionmnist")
    expected = {
        "full": 61706,
        "classifier_head": 59134,
        "classifier_tail": 11014,
        "last_layer": 850,
        "bias_only": 236,
    }
    selections = {
        mode: resolve_active_parameters(model, {"active_parameter_mode": mode})
        for mode in expected
    }
    assert {mode: value.active_count for mode, value in selections.items()} == expected
    assert selections["last_layer"].names == (
        "classifier.4.weight",
        "classifier.4.bias",
    )
    assert selections["classifier_tail"].names == (
        "classifier.2.weight",
        "classifier.2.bias",
        "classifier.4.weight",
        "classifier.4.bias",
    )


def test_configure_active_parameters_freezes_every_coordinate_outside_mask():
    model = get_model("lenet5", "fashionmnist")
    selection = configure_active_parameters(
        model, {"active_parameter_mode": "last_layer"}
    )
    active = set(selection.names)
    assert active
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad is (name in active)


def test_masked_fedavg_transmits_only_active_delta_and_preserves_frozen_weights():
    torch.manual_seed(11)
    model = get_model("lenet5", "fashionmnist")
    before = copy.deepcopy(model.state_dict())
    algorithm = FedAvg()
    config = {
        **algorithm.get_default_config(),
        "device": "cpu",
        "lr": 0.01,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "local_epochs": 1,
        "active_parameter_mode": "last_layer",
    }
    update, metadata = algorithm.client_update(
        model,
        _tiny_loader(),
        ClientState(client_id=0, battery_j=1000.0),
        config,
    )
    assert tuple(update) == ("classifier.4.weight", "classifier.4.bias")
    assert metadata["active_parameter_count"] == 850
    assert metadata["full_parameter_count"] == 61706
    assert metadata["bytes_sent"] == 850 * 4
    assert torch.equal(
        model.state_dict()["features.0.weight"], before["features.0.weight"]
    )


def test_server_preserves_omitted_coordinates_of_its_actual_global_model():
    torch.manual_seed(13)
    global_model = get_model("lenet5", "fashionmnist")
    frozen_before = global_model.state_dict()["features.0.weight"].clone()
    active_before = global_model.state_dict()["classifier.4.weight"].clone()
    update = {
        "classifier.4.weight": torch.ones_like(active_before) * 0.1,
        "classifier.4.bias": torch.zeros_like(
            global_model.state_dict()["classifier.4.bias"]
        ),
    }
    metadata = {
        "dataset_size": 1,
        "bytes_sent": 850 * 4,
        "energy_j_consumed": 0.0,
        "local_loss": 0.0,
    }
    result = FedAvg().server_aggregate(
        global_model,
        [(update, metadata, ClientState(client_id=0, battery_j=1.0))],
        0,
        {},
    )
    assert torch.equal(result.new_weights["features.0.weight"], frozen_before)
    assert torch.allclose(
        result.new_weights["classifier.4.weight"], active_before - 0.1
    )


def test_scfar_noise_vector_and_aggregate_use_only_the_public_active_mask():
    torch.manual_seed(17)
    global_model = get_model("lenet5", "fashionmnist")
    frozen_before = global_model.state_dict()["features.0.weight"].clone()
    state = global_model.state_dict()
    update = {
        "classifier.4.weight": torch.zeros_like(state["classifier.4.weight"]),
        "classifier.4.bias": torch.zeros_like(state["classifier.4.bias"]),
    }
    metadata = {
        "client_id": 0,
        "dataset_size": 1,
        "bytes_sent": 850 * 4,
        "energy_j_consumed": 0.0,
        "local_loss": 0.0,
        "active_parameter_mode": "last_layer",
        "active_parameter_count": 850,
        "full_parameter_count": 61706,
        "active_parameter_fraction": 850 / 61706,
        "active_parameter_names": [
            "classifier.4.weight",
            "classifier.4.bias",
        ],
    }
    algorithm = SensitivityControlledFAR()
    config = {
        **algorithm.get_default_config(),
        "scfar_aggregation_rule": "uniform",
        "far_alpha": 0.0,
        "kappa_w": 1.0,
        "enable_central_dp": True,
        "central_noise_multiplier": 1.0,
        "target_epsilon": None,
        "privacy_num_rounds": 1,
        "active_parameter_mode": "last_layer",
    }
    result = algorithm.server_aggregate(
        global_model,
        [
            (update, metadata, ClientState(client_id=0, battery_j=1.0)),
            (
                update,
                {**metadata, "client_id": 1},
                ClientState(client_id=1, battery_j=1.0),
            ),
        ],
        0,
        config,
    )
    assert result.metrics["scfar_active_parameter_dimension"] == 850
    assert result.metrics["active_parameter_count"] == 850
    assert torch.equal(result.new_weights["features.0.weight"], frozen_before)
    assert not torch.equal(
        result.new_weights["classifier.4.weight"], state["classifier.4.weight"]
    )
