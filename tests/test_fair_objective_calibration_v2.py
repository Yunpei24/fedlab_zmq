"""Preflight tests for the fixed validation-only MPS calibration."""
import ast
import math
from pathlib import Path
import pytest
import torch
from scripts import run_fair_objective_calibration_v2 as run
from privacy.fair_objective import epsilon_bound,per_example,query,release,require_mps


def test_plan_is_isolated_and_calibration_only():
    m=run.config();js=run.jobs(m)
    assert len(js)==len({run.identifier(j) for j in js})==24
    assert {j['seed'] for j in js}=={170301,170302}
    assert {j['C'] for j in js}=={.5,1.,2.}
    assert {j['beta'] for j in js}=={0.,2.}
    assert run.OUT!=run.base.OUT and m['rounds']==60
    assert not m['publish_test_metrics'] and not m['automatic_confirmation']
    tree=ast.parse(Path(run.__file__).read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='evaluate']
    assert len(calls)==2 and all(n.args[-1].value=='val' for n in calls)


@pytest.mark.parametrize('beta',[0.,2.])
@pytest.mark.parametrize('C',[.5,1.,2.])
def test_full_horizon_privacy_and_public_step(beta,C):
    m=run.config();j=dict(seed=170301,beta=beta,C=C,base_lr=2.)
    p=run.parameters(m,j)
    assert p['server_lr']==2/(1+.45*beta)
    plan=run.base.privacy_plan(dict(m,methods={'arm':p}),'arm',4.)
    assert plan['steps']==60 and plan['epsilon']<=4
    assert math.isclose(plan['sensitivity'],C/240*(2+3*beta))
    assert epsilon_bound(q=.05,z=plan['z'],steps=20,delta=1e-5)[0]<plan['epsilon']


def test_mps_distortion_formula():
    require_mps()
    raw=torch.tensor([3.,4.],device='mps');clipped=raw/5
    d=run.vector_distortion(raw,clipped)
    assert d['distortion_norm']==pytest.approx(4.)
    assert d['distortion_relative']==pytest.approx(.8)
    assert d['cosine']==pytest.approx(1.)
    zero=torch.zeros(2,device='mps')
    assert run.vector_distortion(zero,zero)['distortion_relative'] is None
    with pytest.raises(ValueError):run.vector_distortion(raw.cpu(),clipped.cpu())


def test_one_mps_private_training_step_no_oracle_feedback():
    require_mps();torch.manual_seed(17)
    model=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Tanh(),torch.nn.Linear(5,2)).to('mps').eval()
    x=torch.randn(8,3,device='mps');y=torch.arange(8,device='mps')%2
    r,g,norms,raw=per_example(model,x,y,kind='brier',clip_norm=.2)
    q=query(r,g,population_size=80,beta=2,mode='naive')
    assert torch.allclose(q,(1+2*r.mean())*g.mean(0))
    before=q.clone();diag=run.vector_distortion(raw,g.mean(0));assert torch.equal(before,q)
    upload=release(q,noise_std=.01,seed=11)
    previous=torch.cat([p.detach().flatten() for p in model.parameters()])
    run.base.apply_gradient(model,upload,.4)
    after=torch.cat([p.detach().flatten() for p in model.parameters()])
    assert torch.allclose(after,previous-.4*upload,atol=1e-7)
    assert all(math.isfinite(v) for v in diag.values() if v is not None)
