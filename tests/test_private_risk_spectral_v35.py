import math
import pytest
import torch
from privacy.fair_objective import require_mps
from privacy.capped_private_risk import weights
from privacy.private_risk_spectral_v35 import spectral_intervals, subset_covariance_grams, risk_smea


def test_jacobi_known_spectrum_and_two_by_two():
    require_mps()
    mats=torch.tensor([[[2.,1.],[1.,2.]], [[4.,-2.],[-2.,1.]]],device='mps')
    lo,hi=spectral_intervals(mats)
    torch.testing.assert_close(lo,torch.tensor([3.,5.],device='mps'),rtol=2e-5,atol=2e-5)
    torch.testing.assert_close(lo,hi,rtol=2e-5,atol=2e-5)
    q=torch.tensor([[.6,-.8,0.],[.8,.6,0.],[0.,0.,1.]],device='mps')
    m=q@torch.diag(torch.tensor([1.,4.,2.],device='mps'))@q.T
    lo,hi=spectral_intervals(m[None])
    assert abs(float(lo)-4)<2e-5 and abs(float(hi)-4)<2e-5


def test_gram_matches_direct_weighted_covariance_factor():
    require_mps();torch.manual_seed(352)
    x=torch.randn(10,7,device='mps');a=torch.linspace(1,3,10,device='mps')
    subsets=[tuple(range(8)),tuple(range(2,10))]
    gram,ps,scale=subset_covariance_grams(x,a,subsets)
    for j,ids in enumerate(subsets):
        z=x[list(ids)];p=ps[j];mu=(p[:,None]*z).sum(0)
        b=p.sqrt()[:,None]*(z-mu)
        torch.testing.assert_close(gram[j]*scale.square(),b@b.T,rtol=3e-5,atol=3e-6)


def test_no_trimming_equals_fair_mean_and_translation():
    require_mps();torch.manual_seed(353)
    x=torch.randn(10,3,device='mps');r=torch.linspace(0,1,10,device='mps')
    lam,_=weights(r,.5);a,di=risk_smea(x,r,f_budget=0)
    torch.testing.assert_close(a,(lam[:,None]*x).sum(0),rtol=2e-5,atol=2e-6)
    a,di=risk_smea(x,r,f_budget=2)
    b,dj=risk_smea(x+2.,r,f_budget=2)
    assert di['selected_ids']==dj['selected_ids']
    torch.testing.assert_close(b,a+2,rtol=2e-5,atol=2e-6)


def test_scalar_spectral_is_minimum_weighted_variance_and_poisoning():
    require_mps()
    x=torch.tensor([[.0],[.1],[-.1],[.04],[.05],[-.08],[.08],[-.02],[40.],[41.]],device='mps')
    r=torch.zeros(10,device='mps');r[-2:]=1
    a,di=risk_smea(x,r,f_budget=2)
    assert di['selected_ids']==list(range(8))
    torch.testing.assert_close(a,x[:8].mean(0),rtol=2e-5,atol=2e-6)


def test_exact_honest_cardinality_weighted_spectral_bound():
    require_mps();torch.manual_seed(355)
    for _ in range(5):
        x=torch.randn(10,4,device='mps');x[:2]*=7
        r=torch.rand(10,device='mps');lam,_=weights(r,.5)
        a,di=risk_smea(x,r,f_budget=2)
        p=lam[2:]/lam[2:].sum();mu=(p[:,None]*x[2:]).sum(0)
        b=p.sqrt()[:,None]*(x[2:]-mu);_,u=spectral_intervals((b.T@b)[None])
        # n=10,f=2,coefficient ratio=3: (2 sqrt(v_H))² = 4 v_H.
        rhs=4*float(u)+4*di['selection_gap_upper']
        assert float((a-mu).square().sum())<=rhs+1e-4
    with pytest.raises(ValueError):risk_smea(x,r,f_budget=5)


def test_honest_trimming_bound_when_actual_attack_count_below_budget():
    require_mps();torch.manual_seed(356)
    for m in (0,1):
        x=torch.randn(10,4,device='mps');x[:m]*=7
        r=torch.rand(10,device='mps');lam,_=weights(r,.5)
        a,di=risk_smea(x,r,f_budget=2)
        p=lam[m:]/lam[m:].sum();mu=(p[:,None]*x[m:]).sum(0)
        b=p.sqrt()[:,None]*(x[m:]-mu);_,u=spectral_intervals((b.T@b)[None])
        v=float(u);cm=1+3*(2-m)/8
        bound=math.sqrt(3*m/(8-m)*(cm*v+di['selection_gap_upper']))+math.sqrt(6/(8-m)*v)
        assert float(torch.linalg.vector_norm(a-mu))<=bound+1e-4


def fake_screen():
    from scripts.screen_private_risk_spectral_v35 import SEEDS,STATES,ATTACKS,METHODS,ETAS
    import itertools
    client=dict(N=1200,class_count=[1200]+[0]*9,class_hits=[960]+[0]*9)
    return [dict(seed=s,state=st,attack=a,method=m,eta=e,evaluated=True,
                 evaluation=dict(clients=[dict(client) for _ in range(10)]))
            for s,st,a,m,e in itertools.product(SEEDS,STATES,ATTACKS,METHODS,ETAS)]


def test_screen_gate_requires_all_52_comparisons_and_no_missing():
    from scripts.screen_private_risk_spectral_v35 import decide
    rows=fake_screen();d=decide(rows)
    assert d['local_gate_passed'] and len(d['comparisons'])==52
    rows[-1]['evaluated']=False
    assert decide(rows)['local_gate_passed'] is False


def test_screen_rejects_single_seed_attack_loss_without_compensation():
    from scripts.screen_private_risk_spectral_v35 import decide
    rows=fake_screen()
    row=next(r for r in rows if r['seed']==170501 and r['state']=='parent_clean'
             and r['attack']=='persistent_alie' and r['method']=='risk_smea' and r['eta']==.125)
    row['evaluation']['clients'][2]['class_hits']=[0]*10
    assert decide(rows)['local_gate_passed'] is False
