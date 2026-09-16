"""Targeted tests for the explicit LeNet-5 activation ablation."""

import pytest
import torch
from torch import nn

from models.registry import LeNet5, get_model, list_models


def _activation_counts(model: nn.Module) -> tuple[int, int]:
    modules = tuple(model.modules())
    return (
        sum(isinstance(module, nn.Tanh) for module in modules),
        sum(isinstance(module, nn.ReLU) for module in modules),
    )


def test_legacy_lenet5_and_explicit_tanh_are_seed_identical():
    """Adding aliases must not silently change historical experiments."""
    torch.manual_seed(1234)
    legacy = get_model("lenet5", "fashionmnist")
    torch.manual_seed(1234)
    explicit = get_model("lenet5_tanh", "fashionmnist")

    assert legacy.activation_name == explicit.activation_name == "tanh"
    assert _activation_counts(legacy) == (4, 0)
    for name, tensor in legacy.state_dict().items():
        assert torch.equal(tensor, explicit.state_dict()[name]), name


def test_lenet5_relu_uses_relu_and_supports_training_backward():
    model = get_model("lenet5_relu", "fashionmnist")

    assert model.activation_name == "relu"
    assert _activation_counts(model) == (0, 4)
    assert all(
        module.inplace is False
        for module in model.modules()
        if isinstance(module, nn.ReLU)
    )

    inputs = torch.randn(3, 1, 28, 28)
    loss = model(inputs).square().mean()
    loss.backward()

    assert model(inputs).shape == (3, 10)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_lenet5_relu_has_relu_appropriate_zero_bias_initialization():
    model = get_model("lenet5_relu", "mnist")

    affine_layers = (
        module
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    )
    for module in affine_layers:
        assert torch.count_nonzero(module.weight) > 0
        if module.bias is not None:
            assert torch.count_nonzero(module.bias) == 0


@pytest.mark.parametrize(
    ("model_name", "dataset_name", "input_shape"),
    [
        ("lenet5_tanh", "mnist", (2, 1, 28, 28)),
        ("lenet5_relu", "fashionmnist", (2, 1, 28, 28)),
        ("lenet5_relu", "cifar10", (2, 3, 32, 32)),
    ],
)
def test_explicit_lenet5_variants_have_dataset_appropriate_output_shape(
    model_name: str,
    dataset_name: str,
    input_shape: tuple[int, ...],
):
    model = get_model(model_name, dataset_name)
    assert model(torch.randn(input_shape)).shape == (input_shape[0], 10)


def test_lenet5_aliases_are_listed_and_invalid_direct_activation_is_rejected():
    names = {entry["name"] for entry in list_models()}
    assert {"lenet5", "lenet5_tanh", "lenet5_relu"} <= names

    with pytest.raises(ValueError, match="Unsupported LeNet-5 activation"):
        LeNet5(activation="gelu")
