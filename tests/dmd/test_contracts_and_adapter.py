import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from algorithms.base import ClientState, get_algorithm
from algorithms.dmd.contracts import DMDRoundContext
from core.protocol import pack, unpack


def _loader() -> DataLoader:
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(20, 1, 2, 2, generator=generator)
    targets = torch.randint(0, 2, (20,), generator=generator)
    return DataLoader(TensorDataset(features, targets), batch_size=4, shuffle=False)


def _model() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(4, 2))


def test_round_context_is_msgpack_safe() -> None:
    context = DMDRoundContext(
        source_round=3,
        variant="cvar",
        reference=(0.1, None),
        reliability=(1.0, 0.0),
        cohort_mean_deficit=0.2,
        cvar_eta=0.4,
        cvar_tail_mass=0.2,
    )
    restored = DMDRoundContext.from_wire(unpack(pack(context.to_wire())))
    assert restored == context


def test_all_public_aliases_are_registered() -> None:
    names = [
        "dmd_mean",
        "dmd_usv",
        "dmd_tail",
        "dmd_deficit_mean_fixed_zero",
        "dmd_deficit_upper_semivariance_fixed_zero",
        "dmd_deficit_cvar_fixed_zero",
    ]
    assert all(get_algorithm(name) is not None for name in names)
    assert (
        get_algorithm("dmd_deficit_cvar_fixed_zero").get_default_config()[
            "reference_mode"
        ]
        == "fixed_zero"
    )


def test_client_falls_back_to_ce_without_context_and_meets_contract() -> None:
    algorithm = get_algorithm("dmd_mean")
    state = ClientState(client_id=2, battery_j=100.0)
    config = {**algorithm.get_default_config(), "device": "cpu", "num_classes": 2}
    update, metadata = algorithm.client_update(_model(), _loader(), state, config)
    required = {
        "client_id",
        "round_num",
        "beta_actual",
        "battery_j_remaining",
        "energy_j_consumed",
        "bytes_sent",
        "bytes_received",
        "local_loss",
        "compression_ratio",
    }
    assert required <= metadata.keys()
    assert not metadata["dmd_context_applied"]
    assert state.battery_j >= 0
    assert update


def test_client_applies_injected_context_and_server_builds_next_context() -> None:
    algorithm = get_algorithm("dmd_tail")
    context = DMDRoundContext(
        source_round=1,
        variant="cvar",
        reference=(0.5, 0.5),
        reliability=(1.0, 1.0),
        cohort_mean_deficit=0.1,
        cvar_eta=0.1,
        cvar_tail_mass=0.5,
    )
    config = {
        **algorithm.get_default_config(),
        "device": "cpu",
        "num_classes": 2,
        "cvar_tail_mass": 0.5,
        "min_reference_clients": 2,
        "dmd_round_context": context.to_wire(),
        "_server_round": 2,
    }
    model = _model()
    initial = copy.deepcopy(model.state_dict())
    updates = []
    for client_id in range(2):
        local_model = _model()
        local_model.load_state_dict(initial)
        state = ClientState(client_id=client_id, battery_j=100.0)
        update, metadata = algorithm.client_update(
            local_model, _loader(), state, config
        )
        assert metadata["dmd_context_applied"]
        updates.append((update, metadata, state))
    result = algorithm.server_aggregate(model, updates, 2, config)
    assert result.new_weights.keys() == initial.keys()
    assert "dmd_next_round_context" in result.metrics
    assert result.metrics["_server_state_updates"]["dmd_round_context"] == (
        result.metrics["dmd_next_round_context"]
    )
    assert result.metrics["dmd_reference_published_classes"] == 2
    assert len(result.metrics["dmd_tail_weights"]) == 2


def _client_update(name: str, initial: dict, config: dict) -> dict:
    algorithm = get_algorithm(name)
    model = _model()
    model.load_state_dict(initial)
    state = ClientState(client_id=0, battery_j=100.0)
    update, _ = algorithm.client_update(
        model, _loader(), state, {**algorithm.get_default_config(), **config}
    )
    return update


def test_class_balanced_ce_without_penalty_is_cb_ce_exactly() -> None:
    # Margin-on-CB-CE arms are read against a mean_mu=0 control, which is only a
    # control if it is the CB-CE objective itself, update for update.
    initial = copy.deepcopy(_model().state_dict())
    shared = {"device": "cpu", "num_classes": 2, "client_class_counts": [14, 6],
              "lr": 0.03, "momentum": 0.9, "weight_decay": 1e-4, "local_epochs": 1}
    cb_ce = _client_update("cb_ce", initial, shared)
    dmd = _client_update(
        "dmd_mean", initial, {**shared, "ce_class_weighting": "inverse_frequency"}
    )
    assert cb_ce.keys() == dmd.keys()
    assert all(torch.equal(cb_ce[key], dmd[key]) for key in cb_ce)


@pytest.mark.parametrize(
    "name, base_loss",
    [("cb_loss", "effective_number"), ("balanced_softmax", "balanced_softmax"),
     ("fedlc", "fedlc")],
)
def test_label_skew_base_loss_without_penalty_is_the_baseline_exactly(
    name: str, base_loss: str
) -> None:
    # Baselines are compared through the DMD client at mean_mu=0 so every arm
    # shares the anchor pass and random draws; that only holds if the client
    # then trains exactly like the standalone baseline.
    initial = copy.deepcopy(_model().state_dict())
    shared = {"device": "cpu", "num_classes": 2, "client_class_counts": [14, 6],
              "lr": 0.03, "momentum": 0.9, "weight_decay": 1e-4, "local_epochs": 1,
              "label_skew_beta": 0.99, "label_skew_tau": 1.5}
    baseline = _client_update(name, initial, shared)
    dmd = _client_update("dmd_mean", initial, {**shared, "base_loss": base_loss})
    assert baseline.keys() == dmd.keys()
    assert all(torch.equal(baseline[key], dmd[key]) for key in baseline)


def test_class_balanced_ce_reweights_and_requires_counts() -> None:
    initial = copy.deepcopy(_model().state_dict())
    base = {"device": "cpu", "num_classes": 2}
    plain = _client_update("dmd_mean", initial, base)
    weighted = _client_update(
        "dmd_mean",
        initial,
        {**base, "ce_class_weighting": "inverse_frequency", "client_class_counts": [14, 6]},
    )
    assert any(not torch.equal(plain[key], weighted[key]) for key in plain)
    with pytest.raises(ValueError, match="client_class_counts"):
        _client_update("dmd_mean", initial, {**base, "ce_class_weighting": "inverse_frequency"})
    with pytest.raises(ValueError, match="ce_class_weighting"):
        _client_update("dmd_mean", initial, {**base, "ce_class_weighting": "sqrt"})
