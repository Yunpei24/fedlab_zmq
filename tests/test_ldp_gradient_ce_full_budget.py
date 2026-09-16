"""Tiny MPS-only contracts for the isolated no-histogram epsilon=4 CE lane.

No scientific training campaign is run. MPS absence is an error, not a skip or
permission to compute private gradients on CPU.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from algorithms.base import ClientState, get_algorithm
from algorithms.ldp_gradient_ce_full_budget import (
    LDPGradientCEFullBudget,
    _calibrated_ce_sigma,
)
from algorithms.ldp_gradient_dmd_cb import LDPGradientDMDCB
import algorithms.ldp_gradient_ce_full_budget as ce_module
import algorithms.ldp_gradient_dmd_cb as dmd_module
import privacy.dmd_cb_private as private_module
from privacy.dmd_cb_private import (
    clip_combined_gradients,
    paired_mps_randomness,
    per_example_dmd_gradients,
    simulation_seed,
)
from privacy.rdp import (
    RDPAccountant,
    calibrate_sampled_without_replacement_gaussian_noise,
)


@pytest.fixture(scope="module", autouse=True)
def require_mps():
    assert torch.backends.mps.is_available(), "Private contracts require local MPS"


def model():
    with paired_mps_randomness(4321):
        return torch.nn.Linear(4, 10, bias=False, device="mps")


class RecordingDataset(TensorDataset):
    def __init__(self):
        with paired_mps_randomness(6789):
            x = torch.randn(20, 4, device="mps").cpu()
        super().__init__(x, torch.arange(20) % 10)
        self.accessed = []

    def __getitem__(self, index):
        self.accessed.append(index)
        return super().__getitem__(index)


def loader():
    return DataLoader(RecordingDataset(), batch_size=2)


def config(**overrides):
    return {
        **LDPGradientCEFullBudget().get_default_config(),
        "dmd_pairing_seed": 24,
        "privacy_public_dataset_size": 20,
        "fixed_batch_size": 2,
        "privacy_num_rounds": 2,
        "expected_num_clients": 5,
        **overrides,
    }


@pytest.mark.parametrize(
    "invalid",
    [
        {"enable_dp": False},
        {"dmd_mu": 0.1875},
        {"dmd_histogram_epsilon": 0.25},
        {"target_epsilon": 3.75},
        {"dmd_total_epsilon": 3.75},
        {"dmd_pairing_seed": None},
        {"private_aux_loss": True},
        {"enable_oracle_diagnostics": True},
        {"enable_noise_free_counterfactual_oracle": True},
        {"suppress_private_client_diagnostics": False},
        {"fixed_steps_per_round": 2},
        {"device": "cpu"},
        {"per_sample_backend": "serial"},
        {"sampling_scheme": "poisson"},
        {"privacy_adjacency": "add_remove"},
        {"far_alpha": 0.1},
        {"dmd_server_mode": "far_rfa", "far_alpha": 0.2},
        {"far_distance_clip": 1.0},
        {"robust_reference": "median"},
        {"clip_norm": float("nan")},
        {"delta": float("nan")},
    ],
)
def test_registration_and_fail_closed_public_configuration(invalid):
    assert isinstance(
        get_algorithm("ldp_gradient_ce_full_budget"), LDPGradientCEFullBudget
    )
    with pytest.raises(ValueError):
        LDPGradientCEFullBudget()._validated_config(config(**invalid))


def test_no_histogram_no_label_scan_no_weight_state_and_once_step_ledger(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No histogram or whole-dataset labels in CE")

    monkeypatch.setattr(dmd_module, "private_class_weights", forbidden)
    monkeypatch.setattr(private_module, "private_class_weights", forbidden)
    monkeypatch.setattr(private_module, "_dataset_labels", forbidden)
    net, data, algo = model(), loader(), LDPGradientCEFullBudget()
    state = ClientState(client_id=0, battery_j=1.0)
    before = {key: value.clone() for key, value in net.state_dict().items()}
    cpu_before, mps_before = torch.get_rng_state(), torch.mps.get_rng_state()
    for step in range(2):
        data.dataset.accessed.clear()
        update, metadata = algo.client_update(net, data, state, config())
        assert {
            "client_id",
            "round_num",
            "beta_actual",
            "battery_j_remaining",
            "energy_j_consumed",
            "bytes_sent",
            "bytes_received",
            "local_loss",
            "compression_ratio",
        } <= metadata.keys()
        assert len(data.dataset.accessed) == len(set(data.dataset.accessed)) == 2
        assert metadata["privacy_compute_device"] in {"mps", "mps:0"}
        assert metadata["privacy_dmd_histogram_epsilon"] == 0
        assert metadata["privacy_dmd_histogram_calls_total"] == 0
        assert metadata["privacy_dmd_histogram_calls_this_round"] == 0
        assert (
            metadata["privacy_gradient_target_epsilon"]
            == metadata["privacy_target_epsilon"]
            == 4
        )
        assert metadata["privacy_epsilon"] == metadata["privacy_gradient_epsilon"]
        assert (
            metadata["local_loss_available"] is False and metadata["clip_rate"] is None
        )
        assert metadata["privacy_clip_before_noise"] is True
        assert metadata["privacy_post_noise_client_clip"] is False
        assert (
            metadata["privacy_accounting_noise_multiplier"]
            == metadata["privacy_noise_multiplier"] / 2
        )
        assert metadata["privacy_query_sensitivity_l2"] == 4
        assert (
            metadata["privacy_gaussian_std_per_coordinate"]
            == metadata["privacy_noise_multiplier"] * 4 / 2
        )
        assert metadata["bytes_sent"] == metadata["bytes_received"] == 160
        assert state.round_num == state.local_steps == step + 1
        assert state.battery_j == 0
        context = state.custom["ce_full_budget_private_context"]
        assert context["gradient_steps"] == step + 1
        assert "weights" not in json.dumps(context)
        assert "counts" not in json.dumps(context) and "seed" not in json.dumps(context)
        assert set(state.custom["local_dp_accountant"]["channels"]) == {"gradient"}
        certificate = RDPAccountant()
        certificate.add_sampled_without_replacement_gaussian(
            channel="gradient",
            sampling_rate=0.1,
            noise_multiplier=metadata["privacy_noise_multiplier"] / 2,
            steps=step + 1,
        )
        assert metadata["privacy_epsilon"] == pytest.approx(
            certificate.epsilon(1e-5)[0]
        )
        assert all(value.device.type == "cpu" for value in update.values())
        assert all(
            torch.equal(value, net.state_dict()[key]) for key, value in before.items()
        )
        state.custom = json.loads(json.dumps(state.custom))
    assert metadata["privacy_epsilon"] == pytest.approx(4, abs=1e-6)
    assert torch.equal(cpu_before, torch.get_rng_state())
    assert torch.equal(mps_before, torch.mps.get_rng_state())
    with pytest.raises(ValueError, match="horizon"):
        algo.client_update(net, data, state, config())


def test_old_mu_zero_gradient_matches_bitwise_at_common_sigma_and_streams():
    sigma = calibrate_sampled_without_replacement_gaussian_noise(
        target_epsilon=3.75,
        delta=1e-5,
        sampling_rate=0.1,
        steps=2,
        sensitivity_multiplier=2,
        tolerance=1e-8,
    ) * (1 + 1e-7)
    common = dict(dmd_frozen_base_noise_multiplier=sigma, noise_multiplier=sigma)
    net, data = model(), loader()
    old = LDPGradientDMDCB()
    old_config = config(
        **common, dmd_mu=0, dmd_histogram_epsilon=0.25, target_epsilon=3.75
    )
    old_update, old_metadata = old.client_update(
        net, data, ClientState(client_id=3), old_config
    )
    sampled = data.dataset.accessed.copy()
    data.dataset.accessed.clear()
    new_update, new_metadata = LDPGradientCEFullBudget().client_update(
        net, data, ClientState(client_id=3), config(**common)
    )
    assert data.dataset.accessed == sampled
    assert all(torch.equal(new_update[key], old_update[key]) for key in new_update)
    assert (
        old_metadata["privacy_noise_multiplier"]
        == new_metadata["privacy_noise_multiplier"]
        == sigma
    )
    assert old_metadata["privacy_epsilon"] == pytest.approx(
        new_metadata["privacy_epsilon"] + 0.25
    )


def test_full_budget_sigma_is_smaller_and_same_standard_gaussian_stream():
    old_sigma = calibrate_sampled_without_replacement_gaussian_noise(
        target_epsilon=3.75,
        delta=1e-5,
        sampling_rate=0.1,
        steps=2,
        sensitivity_multiplier=2,
        tolerance=1e-8,
    ) * (1 + 1e-7)
    new_sigma = _calibrated_ce_sigma(1e-5, 0.1, 2)
    assert new_sigma < old_sigma
    net, data, algo = model(), loader(), LDPGradientCEFullBudget()
    first, metadata = algo.client_update(net, data, ClientState(client_id=2), config())
    selected = data.dataset.accessed.copy()
    data.dataset.accessed.clear()
    second, _ = algo.client_update(
        net,
        data,
        ClientState(client_id=2),
        config(noise_multiplier=old_sigma, dmd_frozen_base_noise_multiplier=old_sigma),
    )
    assert data.dataset.accessed == selected
    assert metadata["privacy_noise_multiplier"] == new_sigma
    with paired_mps_randomness(simulation_seed(24, 2, 0, "gaussian")):
        gaussian = torch.randn_like(net.weight).cpu()
    assert torch.allclose(
        second["weight"] - first["weight"],
        gaussian * ((old_sigma - new_sigma) * 4 / 2),
        atol=1e-6,
    )
    # The scientific public horizon also receives a genuinely smaller sigma.
    old_campaign_sigma = calibrate_sampled_without_replacement_gaussian_noise(
        target_epsilon=3.75,
        delta=1e-5,
        sampling_rate=0.1,
        steps=40,
        sensitivity_multiplier=2,
        tolerance=1e-8,
    ) * (1 + 1e-7)
    assert _calibrated_ce_sigma(1e-5, 0.1, 40) < old_campaign_sigma


def test_ce_per_example_joint_clipping_and_no_post_noise_clipping():
    net, data = model(), loader()
    x, y = next(iter(data))
    gradients = per_example_dmd_gradients(
        net,
        x.to("mps"),
        y.to("mps"),
        torch.ones(10, device="mps"),
        0,
    )
    clipped = clip_combined_gradients(gradients, 0.1)
    norms = torch.cat([value.flatten(1) for value in clipped.values()], dim=1).norm(
        dim=1
    )
    assert (norms <= 0.1 + 1e-6).all()
    update, metadata = LDPGradientCEFullBudget().client_update(
        net, data, ClientState(client_id=0), config(clip_norm=0.1)
    )
    assert update["weight"].norm() > 0.1
    assert metadata["privacy_post_noise_client_clip"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_context",
        "missing_accountant",
        "missing_sigma",
        "wrong_round",
        "wrong_local_steps",
        "wrong_gradient_steps",
        "histogram_context",
        "weight_state",
        "extra_channel",
        "renamed_channel",
        "altered_order",
        "changed_sigma",
        "context_weights",
    ],
)
def test_resume_state_corruption_fails_before_private_access(mutation):
    algo, net, data = LDPGradientCEFullBudget(), model(), loader()
    state = ClientState(client_id=0)
    algo.client_update(net, data, state, config())
    if mutation == "missing_context":
        state.custom.pop("ce_full_budget_private_context")
    elif mutation == "missing_accountant":
        state.custom.pop("local_dp_accountant")
    elif mutation == "missing_sigma":
        state.custom.pop("local_dp_calibrated_base_noise")
    elif mutation == "wrong_round":
        state.round_num = 0
    elif mutation == "wrong_local_steps":
        state.local_steps = 0
    elif mutation == "wrong_gradient_steps":
        state.custom["ce_full_budget_private_context"]["gradient_steps"] = 0
    elif mutation == "histogram_context":
        state.custom["dmd_cb_private_context"] = {}
    elif mutation == "weight_state":
        state.custom["dp_class_weights"] = [1] * 10
    elif mutation == "extra_channel":
        state.custom["local_dp_accountant"]["channels"]["extra"] = {}
    elif mutation == "renamed_channel":
        ledger = state.custom["local_dp_accountant"]["channels"]
        ledger["histogram"] = ledger.pop("gradient")
    elif mutation == "altered_order":
        state.custom["local_dp_accountant"]["orders"].pop()
    elif mutation == "changed_sigma":
        state.custom["local_dp_calibrated_base_noise"] *= 2
    elif mutation == "context_weights":
        state.custom["ce_full_budget_private_context"]["dp_class_weights"] = [1] * 10
    before = copy.deepcopy(state)
    data.dataset.accessed.clear()
    with pytest.raises(ValueError):
        algo.client_update(net, data, state, config())
    assert data.dataset.accessed == []
    assert state == before


@pytest.mark.parametrize(
    "changed",
    [
        {"dmd_pairing_seed": 42},
        {"clip_norm": 3.0},
        {"privacy_num_rounds": 3},
        {"privacy_noise_multiplier_scale_by_client": [2]},
        {"far_server_lr": 0.1},
    ],
)
def test_resume_public_context_changes_are_rejected(changed):
    algo, net, data, state = (
        LDPGradientCEFullBudget(),
        model(),
        loader(),
        ClientState(client_id=0),
    )
    algo.client_update(net, data, state, config())
    data.dataset.accessed.clear()
    with pytest.raises(ValueError, match="context"):
        algo.client_update(net, data, state, config(**changed))
    assert data.dataset.accessed == []


def test_frozen_sigma_exactly_verified_and_invalid_sigma_never_reads_dataset(
    monkeypatch,
):
    sigma = _calibrated_ce_sigma(1e-5, 0.1, 2)

    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen sigma must not be recalibrated")

    monkeypatch.setattr(ce_module, "_calibrated_ce_sigma", forbidden)
    algo, data = LDPGradientCEFullBudget(), loader()
    _, metadata = algo.client_update(
        model(),
        data,
        ClientState(client_id=0),
        config(
            noise_multiplier=sigma,
            dmd_frozen_base_noise_multiplier=sigma,
            privacy_noise_multiplier_scale_by_client=[2],
        ),
    )
    assert metadata["privacy_noise_multiplier"] == 2 * sigma
    assert metadata["privacy_accounting_noise_multiplier"] == sigma
    for public_sigma, frozen in ((sigma + 1, sigma), (sigma / 2, sigma / 2), (0, 0)):
        data.dataset.accessed.clear()
        state = ClientState(client_id=0)
        with pytest.raises(ValueError):
            algo.client_update(
                model(),
                data,
                state,
                config(
                    noise_multiplier=public_sigma,
                    dmd_frozen_base_noise_multiplier=frozen,
                ),
            )
        assert (
            data.dataset.accessed == [] and state.custom == {} and state.round_num == 0
        )


def fake_uploads():
    uploads = []
    for client_id, value in enumerate((0.0, 0.1, 0.2, 10.0, 15.0)):
        vector = torch.zeros(10, 4)
        vector[0, 0] = value
        metadata = {
            "client_id": client_id,
            "round_num": 1,
            "privacy_compute_device": "mps:0",
            "privacy_gradient_release": True,
            "privacy_epsilon": 2.0,
            "privacy_gradient_epsilon": 2.0,
            "privacy_dmd_histogram_epsilon": 0.0,
            "privacy_dmd_histogram_calls_total": 0,
            "privacy_dmd_histogram_calls_this_round": 0,
            "privacy_delta": 1e-5,
            "privacy_noise_multiplier": 3.0,
            "privacy_noise_multiplier_scale_public": 1.0,
            "privacy_upload_noise_variance_per_coordinate": 0.01,
            "privacy_sampling_scheme": "fixed_without_replacement",
            "privacy_adjacency": "replace_one",
            "privacy_accounting_assumption": "fixed_size_without_replacement_rdp_replace_one",
            "bytes_sent": 160,
            "energy_j_consumed": 0,
            "local_loss": 0,
        }
        uploads.append(({"weight": vector}, metadata, ClientState(client_id=client_id)))
    return uploads


@pytest.mark.parametrize("mode", ["uniform", "direct_rfa", "far_rfa"])
def test_three_servers_match_existing_postprocessing_and_zero_histogram_metrics(mode):
    net, uploads = model(), fake_uploads()
    cfg = config(dmd_server_mode=mode, far_alpha=0.1 if mode == "far_rfa" else 0)
    result = LDPGradientCEFullBudget().server_aggregate(net, uploads, 0, cfg)
    old_uploads = copy.deepcopy(uploads)
    for _, metadata, _ in old_uploads:
        metadata.update(
            privacy_gradient_epsilon=1.75,
            privacy_dmd_histogram_epsilon=0.25,
            privacy_dmd_histogram_calls_total=1,
            privacy_dmd_histogram_calls_this_round=1,
        )
    reference = LDPGradientDMDCB().server_aggregate(
        net,
        old_uploads,
        0,
        {**cfg, "dmd_histogram_epsilon": 0.25, "target_epsilon": 3.75},
    )
    assert all(
        torch.equal(value, reference.new_weights[key])
        for key, value in result.new_weights.items()
    )
    assert (
        result.metrics["privacy_target_epsilon"]
        == result.metrics["privacy_gradient_target_epsilon"]
        == 4
    )
    assert result.metrics["privacy_dmd_histogram_epsilon"] == 0
    assert result.metrics["privacy_dmd_histogram_calls_per_client_max"] == 0
    assert result.metrics["privacy_dmd_histogram_calls_this_round"] == 0
    assert (
        result.metrics["privacy_dmd_budget_composition"]
        == "gradient_rdp_only_no_histogram"
    )
    assert (
        result.metrics["privacy_epsilon_max"]
        == result.metrics["privacy_gradient_epsilon_max"]
        == 2
    )
    assert result.metrics["ldp_gradient_far_private_gradient_mps_fraction"] == 1


@pytest.mark.parametrize(
    "invalid",
    [
        {"privacy_dmd_histogram_calls_total": 1},
        {"privacy_dmd_histogram_epsilon": 0.25},
        {"privacy_compute_device": "cpu"},
        {"is_byzantine": False},
        {"raw_counts": [1]},
        {"dp_class_weights": [1]},
        {"clip_rate": 0.3},
        {"local_loss_available": True},
        {"privacy_epsilon": 4.1},
        {"privacy_gradient_epsilon": 1.0},
    ],
)
def test_server_histogram_oracle_rawstat_and_privacy_boundary(invalid):
    uploads = fake_uploads()
    uploads[0][1].update(invalid)
    with pytest.raises(ValueError):
        LDPGradientCEFullBudget().server_aggregate(model(), uploads, 0, config())


def test_mps_fallback_refused_before_dataset_access(monkeypatch):
    data = loader()
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(ValueError, match="FALLBACK"):
        LDPGradientCEFullBudget().client_update(
            model_for_refusal := torch.nn.Linear(4, 10),
            data,
            ClientState(client_id=0),
            config(),
        )
    assert data.dataset.accessed == [] and model_for_refusal.weight.device.type == "cpu"


@pytest.mark.parametrize("location", ["error_buffer", "momentum_buffer", "custom"])
def test_client_and_server_reject_private_sidechannel_state(location):
    algo, data, state = LDPGradientCEFullBudget(), loader(), ClientState(client_id=0)
    if location == "custom":
        state.custom["raw_counts"] = [1] * 10
    else:
        setattr(state, location, {"weight": torch.ones(10, 4)})
    with pytest.raises(ValueError, match="state|buffers"):
        algo.client_update(model(), data, state, config())
    assert data.dataset.accessed == []
    uploads = fake_uploads()
    uploads[0] = (uploads[0][0], uploads[0][1], state)
    with pytest.raises(ValueError, match="state|buffers"):
        algo.server_aggregate(model(), uploads, 0, config())
