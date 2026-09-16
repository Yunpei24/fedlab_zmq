"""Regression tests for the isolated client momentum screen."""
import copy
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from algorithms.base import ClientState
from algorithms.ldp_client_momentum import LDPClientMomentum, aggregate_rules, momentum_step
from algorithms.ldp_gradient_far import LDPGradientFAR
from scripts.run_ldp_client_momentum import matrix, tasks, run_id, config_for


def test_first_gradient_identity_and_no_cross_client_memory():
    a = {"w": torch.tensor([2., 4.])}
    b = {"w": torch.tensor([-2., 6.])}
    first = momentum_step(a, None, .9)
    assert torch.equal(first["w"], a["w"])
    assert first["w"].data_ptr() != a["w"].data_ptr()
    assert torch.allclose(momentum_step(b, first, .9)["w"], torch.tensor([1.6, 4.2]))
    assert torch.equal(momentum_step(b, first, 0)["w"], b["w"])
    assert torch.equal(momentum_step(b, None, .9)["w"], b["w"])
    assert torch.equal(first["w"], a["w"])
    with pytest.raises(ValueError):
        momentum_step(a, None, 1)


def test_three_deployed_rules_and_clip():
    x = torch.tensor([[1., 0.], [2., 0.], [30., 0.]], dtype=torch.float64)
    for arm in ("uniform", "rfa_direct", "far_rfa"):
        clipped, _, ref, d, weights, out, _, _ = aggregate_rules(x, arm, .1, 3)
        assert (clipped.norm(dim=1) <= 3).all()
        assert torch.allclose(out, (weights[:, None]*clipped).sum(0))
        if arm == "uniform":
            assert torch.allclose(out, clipped.mean(0))
        elif arm == "rfa_direct":
            assert torch.equal(out, ref)
        else:
            assert torch.allclose(weights, torch.softmax(.1*d, dim=0))
    r = aggregate_rules(x, "far_rfa", 0, 3)
    assert torch.allclose(r[5], r[0].mean(0))


def test_matrix_has_36_matched_private_channels():
    m = matrix(); grid = tasks(m)
    assert len(grid) == len({run_id(t) for t in grid}) == 36
    stamp = dict(privacy=dict(sigma=1.4655220866203307))
    configs = [config_for(m, t, stamp) for t in grid]
    for c in configs:
        a = c["training"]["algo_config"]
        assert a["fixed_steps_per_round"] == 1 and a["fixed_batch_size"] == 120
        assert a["privacy_adjacency"] == "replace_one"
        assert a["privacy_public_dataset_size"] == 6000
        assert a["num_byzantine"] == 0 and a["robust_reference"] == "rfa"
        assert a["clip_norm"] == 4 and a["far_server_clip_norm"] == 16
        assert not any(k.startswith("rcig_") for k in a)
    for noise in m["noise_regimes"]:
        subset = [c["training"]["algo_config"] for t, c in zip(grid, configs) if t["noise"] == noise]
        assert len({a["noise_multiplier"] for a in subset}) == 1


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires local MPS")
def test_real_private_channel_mps_pairing_state_and_accountant():
    # Small real private-gradient test, not a Fashion-MNIST result.
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2).to("mps")
    loader = DataLoader(TensorDataset(torch.randn(12, 3), torch.randint(0, 2, (12,))), batch_size=4)
    config = dict(device="mps", fixed_batch_size=4, batch_size=4,
        fixed_steps_per_round=1, local_epochs=1, sampling_scheme="fixed_without_replacement",
        privacy_adjacency="replace_one", privacy_public_dataset_size=12,
        privacy_sampling_rate_override=4/12, privacy_num_rounds=2,
        noise_multiplier=2., target_epsilon=None, clip_norm=1., delta=1e-5,
        enable_dp=True, enable_oracle_diagnostics=True, per_sample_backend="vectorized",
        client_momentum_beta=.9, momentum_pairing_seed=123)
    states = {b: ClientState(client_id=0, battery_j=1e10) for b in (0., .9)}
    traces = {b: [] for b in states}
    previous = None
    try:
        for t in range(2):
            uploads = {}; eps = {}
            for beta, state in states.items():
                LDPClientMomentum.audit_sink = staticmethod(traces[beta].append)
                c = dict(config, client_momentum_beta=beta, _server_round=t)
                uploads[beta], meta = LDPClientMomentum().client_update(model, loader, state, c)
                eps[beta] = meta["privacy_epsilon"]
                assert all(v.device.type == "mps" for v in state.momentum_buffer.values())
                assert meta["client_momentum_history_length"] == t+1
            assert eps[0.] == eps[.9]
            for k in uploads[0.]:
                expected = uploads[0.][k] if t == 0 else (.9*previous[k].to("mps")+.1*uploads[0.][k].to("mps")).cpu()
                assert torch.equal(uploads[.9][k], expected)
            previous = uploads[.9]
            assert traces[0.][-1]["permutations"] == traces[.9][-1]["permutations"]
            assert traces[0.][-1]["standard_gaussians"] == traces[.9][-1]["standard_gaussians"]
        with pytest.raises(RuntimeError, match="lost or repeated"):
            LDPClientMomentum().client_update(model, loader, states[.9], dict(config, _server_round=1))
    finally:
        LDPClientMomentum.audit_sink = None


def test_server_applies_declared_rule_from_first_round_and_rejects_oracle():
    model = torch.nn.Linear(2, 1, bias=False).double()
    with torch.no_grad(): model.weight.zero_()
    updates = []
    for cid in range(10):
        updates.append(({"weight": torch.tensor([[cid/10, .2]], dtype=torch.float64)},
            dict(client_id=cid, privacy_compute_device="mps", client_momentum_device="mps",
                 client_momentum_linear_noise_factor=1), ClientState(client_id=cid)))
    config = dict(momentum_arm="far_rfa", client_momentum_beta=.9, far_alpha=.1,
                  far_server_clip_norm=16, far_server_lr=.2, rfa_max_iter=100, rfa_tol=1e-6)
    for arm in ("uniform", "rfa_direct", "far_rfa"):
        c = dict(config, momentum_arm=arm)
        result = LDPClientMomentum().server_aggregate(model, updates, 0, c)
        x = torch.stack([u["weight"].flatten() for u, _, _ in updates])
        target = aggregate_rules(x, arm, .1 if arm == "far_rfa" else 0, 16)[5]
        assert torch.allclose(result.new_weights["weight"].flatten().double(), -.2*target, atol=1e-7)
        assert result.metrics["momentum_warmup_rounds"] == 0
        assert torch.equal(model.weight, torch.zeros_like(model.weight))
    leaked = copy.deepcopy(updates)
    leaked[0][1]["secret_oracle"] = 1
    with pytest.raises(RuntimeError, match="crossed"):
        LDPClientMomentum().server_aggregate(model, leaked, 0, config)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires local MPS")
def test_full_fashionmnist_harness_two_rounds(tmp_path):
    """Preflight in pytest's temporary folder, not one of the 36 research runs."""
    import json
    import yaml
    from scripts.run_ldp_client_momentum import privacy, worker
    m = matrix(); m["rounds"] = 2
    stamp = dict(privacy=privacy(m))
    t = dict(noise="heteroscedastic", seed=933001, beta=.9, arm="far_rfa")
    cfg = config_for(m, t, stamp); cfg["output_dir"] = str(tmp_path)
    cp = tmp_path/"resolved_config.yaml"; cp.write_text(yaml.safe_dump(cfg))
    worker(cp, tmp_path)
    paths = list(tmp_path.rglob("metrics.json"))
    assert len(paths) == 1
    rows = json.loads(paths[0].read_text())["rounds"]
    assert len(rows) == 2
    for k, r in enumerate(rows, 1):
        assert r["momentum_history_length"] == k
        assert r["num_evaluated_clients"] == r["num_alive_clients"] == 10
        assert r["momentum_oracle_evaluation_only"]
        assert r["momentum_arm"] == "far_rfa" and r["far_alpha"] == .1
        assert r["momentum_private_gradient_device"] == r["momentum_buffer_device"] == "mps"
    trace = [json.loads(line) for line in (tmp_path/"simulator_randomness_private_audit.jsonl").read_text().splitlines()]
    first = {r["client_id"]: r for r in trace if r["round"] == 1}
    for r in trace:
        if r["round"] == 2:
            assert r["memory_before"] == first[r["client_id"]]["memory_after"]
