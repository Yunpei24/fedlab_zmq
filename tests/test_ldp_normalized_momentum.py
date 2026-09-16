"""Numerical, privacy-channel and end-to-end checks for the 18-run ablation."""
import json
import math
import torch
import pytest
from torch.utils.data import DataLoader,TensorDataset
from algorithms.base import ClientState
from algorithms.ldp_client_momentum import LDPClientMomentum,aggregate_rules
from algorithms.ldp_normalized_momentum import LDPNormalizedMomentum,normalized_step,linear_noise_factor
from scripts import run_ldp_normalized_momentum as runner
from scripts import run_ldp_client_momentum as historical


def test_closed_form_normalization_constant_signal_and_noise_factor():
    for beta in (0.,.5,.9):
        z=None; values=[]
        for step in range(1,41):
            current={"x":torch.tensor([step/40.,(-1.)**step],dtype=torch.float64)}
            values.append(current["x"])
            z,m,den=normalized_step(current,z,beta,step)
            coeff=torch.tensor([(1-beta)*beta**(step-s)/(1-beta**step) for s in range(1,step+1)],dtype=torch.float64)
            direct=(coeff[:,None]*torch.stack(values)).sum(0)
            assert torch.allclose(m["x"],direct,atol=1e-14,rtol=1e-14)
            assert abs(float(coeff.sum())-1)<1e-14
            assert abs(float(coeff.square().sum())-linear_noise_factor(beta,step))<1e-14
            assert den==1-beta**step
        z=None
        for step in range(1,41):
            z,m,_=normalized_step({"x":torch.tensor([2.,-3.])},z,beta,step)
            assert torch.allclose(m["x"],torch.tensor([2.,-3.]),atol=2e-6)
    with pytest.raises(ValueError):normalized_step({"x":torch.ones(2)},None,.9,2)
    with pytest.raises(ValueError):normalized_step({"x":torch.ones(2)},None,1.,1)


def test_no_clip_counterfactual_recomputes_the_rule():
    x=torch.tensor([[1.,0.],[2.,0.],[50.,0.]],dtype=torch.float64)
    for arm in ("uniform","rfa_direct","far_rfa"):
        out=aggregate_rules(x,arm,.1,math.inf)
        assert torch.equal(out[0],x)
        assert torch.equal(out[5],aggregate_rules(x,arm,.1,100.)[5])
    assert not torch.allclose(aggregate_rules(x,"far_rfa",.1,3.)[5],aggregate_rules(x,"far_rfa",.1,math.inf)[5])


def test_18_run_grid_changes_only_initialization_and_provenance():
    m=runner.matrix();assert len(runner.tasks(m))==18
    stamp=dict(privacy=dict(sigma=1.4655220866203307))
    for t in runner.tasks(m):
        a=runner.config_for(m,t,stamp);b=historical.config_for(m,t,stamp)
        a.pop("output_dir");b.pop("output_dir")
        ac=a["training"]["algo_config"];bc=b["training"]["algo_config"]
        assert ac.pop("momentum_initialization")=="zero_buffer_with_bias_correction"
        for key in ("momentum_campaign","momentum_runtime_implementation"):
            ac.pop(key);bc.pop(key)
        assert a==b
        assert ac["far_server_clip_norm"]==16 and ac["clip_norm"]==4 and ac["fixed_steps_per_round"]==1


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason="local MPS required")
def test_real_private_gradients_paired_and_private_budget_unchanged():
    torch.manual_seed(7)
    model=torch.nn.Linear(3,2).to("mps")
    loader=DataLoader(TensorDataset(torch.randn(12,3),torch.randint(0,2,(12,))),batch_size=4)
    config=dict(device="mps",fixed_batch_size=4,batch_size=4,fixed_steps_per_round=1,local_epochs=1,
        sampling_scheme="fixed_without_replacement",privacy_adjacency="replace_one",privacy_public_dataset_size=12,
        privacy_sampling_rate_override=4/12,privacy_num_rounds=3,noise_multiplier=2.,target_epsilon=None,
        clip_norm=1.,delta=1e-5,enable_dp=True,enable_oracle_diagnostics=True,per_sample_backend="vectorized",
        client_momentum_beta=.9,momentum_pairing_seed=123,momentum_initialization="zero_buffer_with_bias_correction")
    states={name:ClientState(client_id=0,battery_j=1e10) for name in ("raw","normalized")}
    traces={name:[] for name in states};expected_z=None
    try:
        for t in range(3):
            LDPClientMomentum.audit_sink=staticmethod(traces["raw"].append)
            raw,meta0=LDPClientMomentum().client_update(model,loader,states["raw"],dict(config,client_momentum_beta=0.,_server_round=t))
            LDPNormalizedMomentum.audit_sink=staticmethod(traces["normalized"].append)
            out,meta=LDPNormalizedMomentum().client_update(model,loader,states["normalized"],dict(config,_server_round=t))
            expected_z,expected,_=normalized_step({k:v.to("mps") for k,v in raw.items()},expected_z,.9,t+1)
            assert meta["privacy_epsilon"]==meta0["privacy_epsilon"]
            for k in out:assert torch.equal(out[k],expected[k].cpu())
            assert all(v.device.type=="mps" for v in states["normalized"].momentum_buffer.values())
            for field in ("permutations","standard_gaussians","raw_private_gradient"):
                assert traces["raw"][-1][field]==traces["normalized"][-1][field]
        with pytest.raises(RuntimeError):
            LDPNormalizedMomentum().client_update(model,loader,states["normalized"],dict(config,_server_round=1))
    finally:
        LDPClientMomentum.audit_sink=None;LDPNormalizedMomentum.audit_sink=None


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason="local MPS required")
def test_two_round_fashionmnist_integration(tmp_path):
    import yaml
    m=runner.matrix();m["rounds"]=2
    stamp=dict(privacy=runner.privacy(m));t=dict(noise="heteroscedastic",seed=933001,beta=.9,arm="far_rfa")
    cfg=runner.config_for(m,t,stamp);cfg["output_dir"]=str(tmp_path)
    cp=tmp_path/"resolved_config.yaml";cp.write_text(yaml.safe_dump(cfg))
    runner.worker(cp,tmp_path)
    files=list(tmp_path.rglob("metrics.json"));assert len(files)==1
    rows=json.loads(files[0].read_text())["rounds"];assert len(rows)==2
    assert abs(rows[-1]["privacy_epsilon_max"]-4)<1e-4
    for k,r in enumerate(rows,1):
        assert r["momentum_initialization"]=="zero_buffer_with_bias_correction"
        assert r["momentum_buffer_device"]==r["momentum_private_gradient_device"]=="mps"
        assert r["num_evaluated_clients"]==r["num_alive_clients"]==10
        assert abs(r["momentum_linear_noise_factor"]-linear_noise_factor(.9,k))<1e-12
        assert r["momentum_oracle_evaluation_only"]
        assert not r["momentum_no_clip_counterfactual_is_end_to_end"]
        if r["far_server_clip_rate"]==0:assert r["momentum_oracle_clip_aggregate_displacement"]<1e-20
