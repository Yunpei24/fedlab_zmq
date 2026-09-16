"""Contract tests for the isolated DMD-CB lane; all private tensors use MPS.

Run with PYTORCH_ENABLE_MPS_FALLBACK=0 venv/bin/python -m pytest ... .
These tests fail, rather than skip or run private gradients on CPU, if MPS is
unavailable. They use tiny tensor fixtures, not scientific training runs.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader, TensorDataset

from algorithms.base import ClientState, get_algorithm
from algorithms.ldp_gradient_dmd_cb import (
    LDPGradientDMDCB,
    geometric_median_with_weights,
)
import algorithms.ldp_gradient_dmd_cb as algorithm_module
from privacy.dmd_cb_private import (
    clip_combined_gradients,
    dmd_cb_losses,
    paired_mps_randomness,
    per_example_dmd_gradients,
    private_class_weights,
    private_dmd_gradient_release,
    simulation_seed,
)
from robustness.aggregators import geometric_median
from privacy.rdp import calibrate_sampled_without_replacement_gaussian_noise


@pytest.fixture(scope="module", autouse=True)
def require_mps():
    assert torch.backends.mps.is_available(), "Private tests must run on local MPS"


def model():
    with paired_mps_randomness(4321):
        return torch.nn.Linear(4, 10, bias=False, device="mps")


def loader():
    with paired_mps_randomness(6789):
        x = torch.randn(20, 4, device="mps").cpu()
    return DataLoader(TensorDataset(x, torch.arange(20) % 10), batch_size=2)


def config(**overrides):
    return {
        **LDPGradientDMDCB().get_default_config(),
        "device": "mps",
        "dmd_pairing_seed": 24,
        "privacy_public_dataset_size": 20,
        "fixed_batch_size": 2,
        "privacy_sampling_rate_override": 0.1,
        "privacy_num_rounds": 2,
        "expected_num_clients": 5,
        **overrides,
    }


def test_registration_and_no_noise_free_lane():
    assert isinstance(get_algorithm("ldp_gradient_dmd_cb"), LDPGradientDMDCB)
    for invalid in (
        {"enable_dp": False},
        {"enable_oracle_diagnostics": True},
        {"fixed_steps_per_round": 2},
        {"dmd_total_epsilon": 3.75},
        {"dmd_pairing_seed": None},
    ):
        with pytest.raises(ValueError):
            LDPGradientDMDCB()._validated_config(config(**invalid))


def test_mu_zero_is_exact_ce_and_gradient():
    net = model()
    x, y = next(iter(loader()))
    x, y = x.to("mps"), y.to("mps")
    weights = torch.arange(1, 11, dtype=torch.float32, device="mps")
    logits = net(x)
    assert torch.equal(
        dmd_cb_losses(logits, y, weights, 0),
        torch.nn.functional.cross_entropy(logits, y, reduction="none"),
    )
    actual = per_example_dmd_gradients(net, x, y, weights, 0)

    def ce(params, one_x, one_y):
        pred = functional_call(net, params, (one_x.unsqueeze(0),))
        return torch.nn.functional.cross_entropy(pred, one_y.unsqueeze(0))

    expected = vmap(grad(ce), in_dims=(None, 0, 0))(dict(net.named_parameters()), x, y)
    for name in actual:
        assert torch.equal(actual[name], expected[name])


def test_fixed_context_gives_separable_per_example_gradients_and_joint_clip():
    net = model()
    x, y = next(iter(loader()))
    x, y = x.to("mps"), y.to("mps")
    weights = torch.arange(1, 11, dtype=torch.float32, device="mps")
    first = per_example_dmd_gradients(net, x, y, weights, 0.1875)
    changed = x.clone()
    changed[0] *= -10
    second = per_example_dmd_gradients(net, changed, y, weights, 0.1875)
    for name in first:
        assert torch.equal(first[name][1:], second[name][1:])
    radius = 0.25
    clipped = clip_combined_gradients(first, radius)
    norm = torch.cat([item.flatten(1) for item in clipped.values()], dim=1).norm(dim=1)
    assert (norm <= radius + 2e-6).all()
    raw = torch.cat([item.flatten(1) for item in first.values()], dim=1)
    expected = raw * (radius / raw.norm(dim=1).clamp_min(1e-12)).clamp(max=1)[:, None]
    assert torch.allclose(
        torch.cat([item.flatten(1) for item in clipped.values()], dim=1), expected
    )


def test_histogram_reproducible_mps_fixed_universe_and_rng_restored():
    data = loader().dataset
    cpu_before, mps_before = torch.get_rng_state(), torch.mps.get_rng_state()
    kwargs = dict(
        num_classes=10, public_dataset_size=20, epsilon=0.25, weight_cap=20, seed=88
    )
    first = private_class_weights(data, **kwargs)
    second = private_class_weights(data, **kwargs)
    assert first.device.type == "mps"
    assert first.shape == (10,) and torch.equal(first, second)
    assert torch.isfinite(first).all() and (first > 0).all() and (first <= 20).all()
    assert torch.equal(cpu_before, torch.get_rng_state())
    assert torch.equal(mps_before, torch.mps.get_rng_state())
    with pytest.raises(ValueError):
        private_class_weights(data, **{**kwargs, "public_dataset_size": 19})


def test_release_pairing_ignores_ce_classweights_and_restores_global_rng():
    net, data = model(), loader()
    initial_model = {key: value.clone() for key, value in net.state_dict().items()}
    cpu_before, mps_before = torch.get_rng_state(), torch.mps.get_rng_state()
    kwargs = dict(
        mu=0,
        batch_size=2,
        clip_norm=4,
        noise_multiplier=3,
        batch_seed=223,
        gaussian_seed=667,
    )
    a = private_dmd_gradient_release(
        net, data, class_weights=torch.ones(10, device="mps"), **kwargs
    )
    b = private_dmd_gradient_release(
        net, data, class_weights=20 * torch.ones(10, device="mps"), **kwargs
    )
    assert all(torch.equal(a[name], b[name]) for name in a)
    assert all(
        torch.equal(net.state_dict()[name], initial_model[name])
        for name in initial_model
    )
    assert all(
        value.device.type == "cpu" and torch.isfinite(value).all()
        for value in a.values()
    )
    assert torch.equal(cpu_before, torch.get_rng_state())
    assert torch.equal(mps_before, torch.mps.get_rng_state())
    assert (
        len(
            {
                simulation_seed(24, 0, 0, domain)
                for domain in ("histogram", "batch", "gaussian")
            }
        )
        == 3
    )


def test_histogram_once_composition_state_reuse_and_ce_matched_cost(monkeypatch):
    original = algorithm_module.private_class_weights
    calls = []

    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(algorithm_module, "private_class_weights", tracked)
    net, data = model(), loader()
    algo = LDPGradientDMDCB()
    state = ClientState(client_id=0, battery_j=20)
    _, first = algo.client_update(net, data, state, config())
    weights = copy.deepcopy(state.custom["dmd_cb_private_context"]["dp_class_weights"])
    # A JSON round-trip mimics transport-safe persisted state, not a redraw.
    state.custom = json.loads(json.dumps(state.custom))
    _, second = algo.client_update(net, data, state, config())
    assert len(calls) == 1
    assert state.custom["dmd_cb_private_context"]["dp_class_weights"] == weights
    assert first["privacy_dmd_histogram_calls_this_round"] == 1
    assert second["privacy_dmd_histogram_calls_this_round"] == 0
    assert second["privacy_dmd_histogram_calls_total"] == 1
    assert second["privacy_epsilon"] == pytest.approx(
        second["privacy_gradient_epsilon"] + 0.25
    )
    assert second["privacy_epsilon"] == pytest.approx(4, abs=1e-4)
    assert (
        second["privacy_accounting_noise_multiplier"]
        == second["privacy_noise_multiplier"] / 2
    )
    assert second["privacy_query_sensitivity_l2"] == 4
    assert state.battery_j >= 0
    context_text = json.dumps(state.custom["dmd_cb_private_context"])
    assert (
        "seed" not in context_text
        and "raw" not in context_text
        and "counts" not in context_text
    )
    assert not any("oracle" in name or "weights" == name for name in second)
    ce_state = ClientState(client_id=0, battery_j=20)
    _, ce = algo.client_update(net, data, ce_state, config(dmd_mu=0))
    assert len(calls) == 2
    assert ce_state.custom["dmd_cb_private_context"]["dp_class_weights"] == weights
    assert ce["privacy_epsilon"] == first["privacy_epsilon"]
    assert ce["privacy_noise_multiplier"] == first["privacy_noise_multiplier"]


def test_partial_restart_context_changes_and_missing_accountant_are_rejected():
    algo, net, data = LDPGradientDMDCB(), model(), loader()
    with pytest.raises(ValueError, match="partial restart"):
        algo.client_update(net, data, ClientState(client_id=0, round_num=1), config())
    state = ClientState(client_id=0)
    algo.client_update(net, data, state, config())
    with pytest.raises(ValueError, match="context"):
        algo.client_update(net, data, state, config(dmd_class_weight_cap=19))
    state.custom.pop("local_dp_accountant")
    with pytest.raises(ValueError, match="accountant"):
        algo.client_update(net, data, state, config())


def fake_uploads():
    result = []
    for client_id, value in enumerate((0.0, 0.1, 0.2, 10.0, 15.0)):
        vector = torch.zeros(10, 4, device="mps")
        vector[0, 0] = value
        metadata = {
            "client_id": client_id,
            "round_num": 1,
            "privacy_compute_device": "mps:0",
            "privacy_gradient_release": True,
            "privacy_epsilon": 2.0,
            "privacy_gradient_epsilon": 1.75,
            "privacy_dmd_histogram_epsilon": 0.25,
            "privacy_dmd_histogram_calls_total": 1,
            "privacy_dmd_histogram_calls_this_round": 1,
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
        result.append(
            ({"weight": vector.cpu()}, metadata, ClientState(client_id=client_id))
        )
    return result


def test_direct_rfa_is_true_aggregate_not_mean_or_far_and_has_correct_sign():
    algo, net, uploads = LDPGradientDMDCB(), model(), fake_uploads()
    before = net.weight.detach().cpu().clone()
    mean = algo.server_aggregate(net, uploads, 0, config(dmd_server_mode="uniform"))
    rfa = algo.server_aggregate(net, uploads, 0, config(dmd_server_mode="direct_rfa"))
    far = algo.server_aggregate(
        net, uploads, 0, config(dmd_server_mode="far_rfa", far_alpha=0.1)
    )
    vectors = torch.stack(
        [update["weight"].flatten() for update, _, _ in uploads]
    ).double()
    reference = geometric_median(vectors)
    direct, coeff = geometric_median_with_weights(vectors)
    assert torch.equal(direct, reference)
    assert torch.allclose(direct, (coeff[:, None] * vectors).sum(0))
    observed = (before - rfa.new_weights["weight"].cpu()) / 0.2
    assert torch.allclose(observed.flatten().double(), direct, atol=1e-6)
    assert not torch.allclose(rfa.new_weights["weight"], mean.new_weights["weight"])
    assert not torch.allclose(rfa.new_weights["weight"], far.new_weights["weight"])
    assert rfa.metrics["dmd_server_coefficient_kind"] == "direct_rfa_irls"
    assert far.metrics["dmd_server_coefficient_kind"] == "far_softmax_raw_distance"
    assert rfa.metrics["privacy_target_epsilon"] == 4
    for result in (mean, rfa, far):
        assert result.metrics["privacy_epsilon_max"] == 2
        assert result.metrics["privacy_epsilon_mean"] == 2
        assert result.metrics["privacy_model_noise_multiplier_mean"] == 3
        assert result.metrics["ldp_gradient_far_private_gradient_mps_fraction"] == 1
        assert result.metrics["ldp_gradient_far_private_compute_device"] in {
            "mps",
            "mps:0",
        }


def test_server_clipping_and_oracle_boundary():
    uploads = fake_uploads()
    algo, net = LDPGradientDMDCB(), model()
    result = algo.server_aggregate(net, uploads, 0, config(far_server_clip_norm=0.5))
    step = (net.weight.detach() - result.new_weights["weight"]) / 0.2
    assert step.norm() <= 0.5 + 1e-6
    assert result.metrics["far_server_clip_rate"] == 0.4
    uploads[0][1]["is_byzantine"] = False
    with pytest.raises(ValueError, match="oracles"):
        algo.server_aggregate(net, uploads, 0, config())


def test_cpu_refusal_before_private_access():
    with pytest.raises(ValueError, match="MPS"):
        LDPGradientDMDCB().client_update(
            model(), loader(), ClientState(client_id=0), config(device="cpu")
        )


def test_model_batch_dependence_is_forbidden():
    x, y = next(iter(loader()))
    net = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 10)).to("mps")
    with pytest.raises(ValueError, match="batch-independent"):
        per_example_dmd_gradients(
            net, x.to("mps"), y.to("mps"), torch.ones(10, device="mps"), 0.1875
        )


def test_frozen_sigma_is_used_exactly_without_recalibration(monkeypatch):
    frozen = calibrate_sampled_without_replacement_gaussian_noise(
        target_epsilon=3.75,
        delta=1e-5,
        sampling_rate=0.1,
        steps=2,
        sensitivity_multiplier=2,
        tolerance=1e-8,
    ) * (1 + 1e-7)
    algo = LDPGradientDMDCB()

    def reject_recalibration(*args, **kwargs):
        raise AssertionError("Frozen campaign sigma must not be recalibrated")

    monkeypatch.setattr(algo, "_resolved_noise_multiplier", reject_recalibration)
    kwargs = dict(
        dmd_frozen_base_noise_multiplier=frozen,
        noise_multiplier=frozen,
        privacy_noise_multiplier_scale_by_client=[2],
    )
    _, metadata = algo.client_update(
        model(), loader(), ClientState(client_id=0), config(**kwargs)
    )
    assert metadata["privacy_noise_multiplier"] == 2 * frozen
    assert metadata["privacy_accounting_noise_multiplier"] == frozen
    with pytest.raises(ValueError, match="disagrees"):
        algo.client_update(
            model(),
            loader(),
            ClientState(client_id=0),
            config(**{**kwargs, "noise_multiplier": frozen + 1}),
        )


def test_exact_batch_cardinality_and_noise_scaling():
    class RecordingDataset(TensorDataset):
        def __init__(self, *tensors):
            super().__init__(*tensors)
            self.accessed = []

        def __getitem__(self, index):
            self.accessed.append(index)
            return super().__getitem__(index)

    data = loader().dataset
    tracked = RecordingDataset(*data.tensors)
    data_loader = DataLoader(tracked, batch_size=2)
    net = model()
    kwargs = dict(
        class_weights=torch.ones(10, device="mps"),
        mu=0.1875,
        batch_size=5,
        clip_norm=4,
        batch_seed=227,
        gaussian_seed=553,
    )
    first = private_dmd_gradient_release(net, data_loader, noise_multiplier=2, **kwargs)
    assert len(tracked.accessed) == len(set(tracked.accessed)) == 5
    sampled = tracked.accessed.copy()
    tracked.accessed.clear()
    second = private_dmd_gradient_release(
        net, data_loader, noise_multiplier=3, **kwargs
    )
    assert tracked.accessed == sampled
    with paired_mps_randomness(553):
        expected_difference = torch.randn_like(net.weight) * (4 / 5)
    assert torch.allclose(
        second["weight"] - first["weight"], expected_difference.cpu(), atol=5e-7
    )
