import math
import json
from pathlib import Path
import pytest
import torch
from privacy import full_population_private_risk_v28 as m
from privacy.fair_objective import per_example,require_mps,release
from scripts import run_full_population_private_risk_calibration_v28 as run


def sample():
    require_mps();torch.manual_seed(2828);torch.mps.manual_seed(2828)
    model=torch.nn.Linear(3,2).to('mps').eval()
    x=torch.randn(17,3,device='mps');y=torch.arange(17,device='mps')%2
    return model,x,y


def test_streaming_clips_examples_not_complete_mean():
    model,x,y=sample()
    before={k:v.clone() for k,v in model.state_dict().items()}
    C=.05
    rr,rows,norms,raw=per_example(model,x,y,clip_norm=C)
    q,diag=m.clipped_population_mean(model,x,y,C=C,block_size=5)
    torch.testing.assert_close(q,rows.mean(0),rtol=3e-5,atol=3e-7)
    assert diag['accumulation_blocks']==4 and diag['population_size']==17
    assert diag['per_example_clipped_count']==int((norms>C).sum())>0
    assert math.isclose(diag['raw_population_brier_risk'],float(rr.mean()),rel_tol=2e-6)
    batch_clip=raw*(C/torch.linalg.vector_norm(raw)).clamp(max=1)
    assert float(torch.linalg.vector_norm(q-batch_clip))>1e-4
    for k,v in model.state_dict().items():assert torch.equal(v,before[k])
    assert all(p.grad is None for p in model.parameters())


def test_one_noise_release_after_all_blocks(monkeypatch):
    model,x,y=sample();calls=[]
    def recorded(query,*,noise_std,seed):
        calls.append((query.clone(),noise_std,seed))
        return release(query,noise_std=noise_std,seed=seed)
    monkeypatch.setattr(m,'release',recorded)
    private,q,diag=m.private_population_gradient(model,x,y,C=.1,block_size=4,noise_std=.2,seed=829)
    assert len(calls)==1 and torch.equal(calls[0][0],q)
    assert torch.equal(private,release(q,noise_std=.2,seed=829))
    assert diag['gaussian_releases']==1 and diag['accumulation_blocks']==5
    assert diag['replace_one_sensitivity']==.2/17
    with pytest.raises(ValueError):m.private_population_gradient(model,x,y,noise_std=0.,seed=829)


def test_replace_one_population_bound_on_real_gradients():
    model,x,y=sample();x2=x.clone();y2=y.clone();x2[3]=-50*x[3];y2[3]=1-y[3]
    C=.05
    q,_=m.clipped_population_mean(model,x,y,C=C,block_size=5)
    q2,_=m.clipped_population_mean(model,x2,y2,C=C,block_size=5)
    assert float(torch.linalg.vector_norm(q-q2))<=2*C/17+1e-7
    # This test complements the triangle-inequality proof; it is not a universal proof.


def test_finite_population_permutation_is_not_sampling_amplification():
    model,x,y=sample()
    idx=run.base.draw_indices(17,17,1818)
    assert torch.equal(idx.sort().values,torch.arange(17,device='mps'))
    q,_=m.clipped_population_mean(model,x,y,block_size=5)
    qp,_=m.clipped_population_mean(model,x[idx],y[idx],block_size=4)
    torch.testing.assert_close(q,qp,rtol=3e-5,atol=3e-7)


def test_ledger_matches_independent_public_interval_audit_and_all_prefixes():
    ev=json.loads((run.ROOT/'output/analysis/Additive_Gaussian_WOR_V27_Public_Audit.json').read_text())
    for method in m.METHODS:
        p=m.ledger(method)
        family='private_risk_channel' if method.startswith('risk_') else 'erm_all_budget'
        oracle=next(r['plan'] for r in ev['rows'] if r['family']==family and r['plan']['batch']==4800)
        assert math.isclose(p['gradient_std'],oracle['gradient_std'],rel_tol=1e-12)
        assert math.isclose(p['epsilon_realized'],oracle['epsilon_realized'],abs_tol=1e-12)
        assert p['gradient_releases']==120 and p['b']==p['N']==4800
        assert p['risk_releases']==(120 if method.startswith('risk_') else 0)
        eps=[m.prefix_epsilon(p,t)[0] for t in range(121)]
        assert eps[0]==0 and all(a<=b<=4 for a,b in zip(eps,eps[1:]))
        # Direct independent sum of the two Gaussian channels at optimal order.
        a=p['order'];base=a/(2*p['gradient_z']**2)
        risk=a/(2*p['risk_z']**2) if p['risk_z'] else 0.
        assert math.isclose(120*(base+risk)+math.log(1e5)/(a-1),eps[-1],rel_tol=1e-14)
    assert m.ledger('erm_mean')['risk_std']==0 and m.ledger('erm_mean')['gradient_std']<m.ledger('risk_mean')['gradient_std']


def test_cpu_and_invalid_population_are_rejected():
    model,x,y=sample()
    with pytest.raises(ValueError):m.clipped_population_mean(model,x.cpu(),y)
    with pytest.raises(ValueError):m.clipped_population_mean(model,x[:0],y[:0])
    with pytest.raises(ValueError):m.clipped_population_mean(model,x,y,block_size=0)
    model.train()
    with pytest.raises(ValueError):m.clipped_population_mean(model,x,y)


def test_frozen_design_and_source_unchanged():
    config,profile,stamp=run.inputs()
    assert len(run.jobs(config))==8 and profile['batch_size']==4800
    assert config['primary_method']=='risk_rfa' and config['minimum_worst20_advantage_pp']==1
    assert not config['automatic_attacks'] and not config['automatic_confirmation'] and not config['test_evaluated']
    code=Path(run.__file__).read_text()
    assert "evaluate(model,data,'test')" not in code and 'optimizer.step' not in code
    assert 'gate_evaluated=False' in code and run.SOURCE.name=='recursive_private_risk_calibration_v19'
