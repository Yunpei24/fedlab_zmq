"""MPS tests of the new isolated lane; no CPU training or silent skips."""
import itertools
import math
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import pytest
import torch
from privacy.fair_objective import (require_mps,query,per_example,losses,sensitivity,
    wor_rdp,calibrate,epsilon_bound,release)


@pytest.fixture(scope='module',autouse=True)
def mps_required():require_mps()


def tensors():
    r=torch.tensor([.1,.4,.8,.6],device='mps')
    g=torch.tensor([[-1.,.2],[.3,.5],[.1,-.6],[.4,.7]],device='mps')
    return r,g


@pytest.mark.parametrize('b',[2,3,4])
@pytest.mark.parametrize('beta',[0.,.5,2.])
def test_unbiased_and_exact_naive_bias(b,beta):
    r,g=tensors();N=len(r)
    selected=list(itertools.combinations(range(N),b))
    target=(1+beta*r.mean())*g.mean(0)
    corrected=torch.stack([query(r[list(s)],g[list(s)],population_size=N,beta=beta,mode='unbiased') for s in selected]).mean(0)
    naive=torch.stack([query(r[list(s)],g[list(s)],population_size=N,beta=beta,mode='naive') for s in selected]).mean(0)
    bias=beta*(N-b)/(b*(N-1))*((r[:,None]*g).mean(0)-r.mean()*g.mean(0))
    torch.testing.assert_close(corrected,target,atol=2e-7,rtol=2e-6)
    torch.testing.assert_close(naive-target,bias,atol=2e-7,rtol=2e-6)


def test_two_independent_batches_are_unbiased_but_disjoint_are_not():
    r,g=tensors();comb=list(itertools.combinations(range(4),2));target=(1+2*r.mean())*g.mean(0)
    independent=torch.stack([(1+2*r[list(a)].mean())*g[list(b)].mean(0) for a,b in itertools.product(comb,repeat=2)]).mean(0)
    disjoint=torch.stack([(1+2*r[list(a)].mean())*g[list(b)].mean(0) for a,b in itertools.product(comb,repeat=2) if set(a).isdisjoint(b)]).mean(0)
    torch.testing.assert_close(independent,target,atol=2e-7,rtol=2e-6)
    assert float(torch.linalg.vector_norm(disjoint-target))>1e-3


@pytest.mark.parametrize('mode',['erm','naive','unbiased','per_example'])
def test_replace_one_sensitivity_multidimensional(mode):
    torch.mps.manual_seed(591)
    beta=0. if mode=='erm' else 2.
    for b in (2,3,8):
        bound=sensitivity(population_size=100,batch_size=b,clip_norm=1,beta=beta,mode=mode)
        for _ in range(12):
            r=torch.rand(b,device='mps');g=torch.randn(b,4,device='mps')
            g/=torch.linalg.vector_norm(g,dim=1).clamp_min(1)[:,None]
            rr=r.clone();gg=g.clone();rr[0]=torch.rand((),device='mps')
            v=torch.randn(4,device='mps');gg[0]=v/torch.linalg.vector_norm(v).clamp_min(1)
            a=query(r,g,population_size=100,beta=beta,mode=mode)
            a2=query(rr,gg,population_size=100,beta=beta,mode=mode)
            assert float(torch.linalg.vector_norm(a-a2))<=bound+1e-6


def test_noise_release_and_rng_restoration():
    q=torch.tensor([.1,-.2],device='mps');state=torch.mps.get_rng_state().clone()
    v=release(q,noise_std=.3,seed=1255)
    assert torch.equal(state,torch.mps.get_rng_state())
    v2=release(q,noise_std=.3,seed=1255);torch.testing.assert_close(v,v2,atol=0,rtol=0)
    torch.mps.manual_seed(1255);expected=q+.3*torch.randn_like(q)
    torch.testing.assert_close(v,expected,atol=0,rtol=0)
    torch.testing.assert_close(release(q,noise_std=0.,seed=5),q)


def test_brier_gradients_clip_and_target_with_real_model():
    model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Tanh(),torch.nn.Linear(4,3)).to('mps').eval()
    x=torch.randn(6,3,device='mps');y=torch.tensor([0,1,2,0,1,2],device='mps')
    r,g,norms,raw=per_example(model,x,y,clip_norm=100,chunk_size=2)
    risk=losses(model(x),y).mean();full=risk+risk.square()
    expected=torch.cat([v.flatten() for v in torch.autograd.grad(full,model.parameters())])
    assert bool(((r>=0)&(r<=1)).all())
    actual=query(r,g,population_size=6,beta=2,mode='unbiased')
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    rr,gg,nn,raw2=per_example(model,x,y,clip_norm=.001,chunk_size=3)
    assert float(torch.linalg.vector_norm(gg,dim=1).max())<=.0010001
    torch.testing.assert_close(raw,raw2,atol=2e-6,rtol=2e-5)
    clipped_target=(1+2*rr.mean())*gg.mean(0)
    assert float(torch.linalg.vector_norm(clipped_target-expected))>1e-3


def test_query_rejects_unbounded_loss_and_cpu():
    r,g=tensors()
    with pytest.raises(ValueError):query(r*10,g,population_size=4,beta=2,mode='unbiased')
    with pytest.raises(ValueError):query(r.cpu(),g.cpu(),population_size=4,beta=2,mode='unbiased')
    with pytest.raises(ValueError):query(r,g,population_size=4,beta=2,mode='erm')
    with pytest.raises(ValueError):query(r[:1],g[:1],population_size=4,beta=2,mode='unbiased')


def test_generic_wor_formula_against_direct_sum():
    # Independent implementation of Theorem 9 for modest integer orders.
    for a,q,z in itertools.product([2,3,8],[.01,.05,.5],[.7,2.,5.]):
        s=1+q*q*math.comb(a,2)*min(4*math.expm1(1/z**2),2*math.exp(1/z**2))
        s+=sum(q**j*math.comb(a,j)*2*math.exp(j*(j-1)/(2*z*z)) for j in range(3,a+1))
        expected=min(a/(2*z*z),math.log(s)/(a-1))
        assert abs(wor_rdp(a,q,z)-expected)<1e-11
    assert wor_rdp(5,0,2)==0
    assert wor_rdp(5,1,2)==5/8


def test_accounting_and_noise_cost_are_not_fake_matched():
    z=calibrate(q=.05,steps=20,epsilon=4.,delta=1e-5)
    eps,_=epsilon_bound(q=.05,z=z,steps=20,delta=1e-5)
    assert 3.999<eps<=4
    assert epsilon_bound(q=.05,z=z*.99,steps=20,delta=1e-5)[0]>4
    assert epsilon_bound(q=.05,z=z,steps=10,delta=1e-5)[0]<eps
    assert epsilon_bound(q=.05,z=z*2,steps=20,delta=1e-5)[0]<eps
    base=sensitivity(population_size=4800,batch_size=240,clip_norm=.5,beta=0,mode='erm')
    corrected=sensitivity(population_size=4800,batch_size=240,clip_norm=.5,beta=2,mode='unbiased')
    assert corrected/base>3.99
    assert epsilon_bound(q=.05,z=z*base/corrected,steps=20,delta=1e-5)[0]>4


def test_campaign_matrix_and_bounded_scope():
    from scripts.run_fair_objective_screen import config,all_jobs,privacy_plan
    m=config();cal,evaluation=all_jobs(m)
    assert len(cal)==2 and len(evaluation)==32
    assert {j['seed'] for j in cal}.isdisjoint(j['seed'] for j in evaluation)
    assert all(j['epsilon'] is None for j in cal)
    for name in m['methods']:
        p=privacy_plan(m,name,4.)
        assert p['epsilon']<=4 and p['std']==p['z']*p['sensitivity']
    assert m['server_clip'] is None and m['attacks']=='none'
    assert m['gate']['no_attack_or_recursive_promotion']
