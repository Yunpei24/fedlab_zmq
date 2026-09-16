from copy import deepcopy
import math
import pytest
import torch
from scripts.run_private_risk_continuation_v33 import attack,epsilon,decision,SEEDS,ATTACKS,METHODS
from privacy.full_population_private_risk_v28 import ledger

def ev(hits=900):
    return dict(clients=[dict(N=1200,class_count=[1200]+[0]*9,class_hits=[hits]+[0]*9) for _ in range(10)])

def results():
    return [dict(seed=s,attack=a,method=m,rows=[dict(step=k,validation=ev() if k in (4,8,12) else None,
        model_hash=str(k)) for k in range(1,13)]) for s in SEEDS for a in ATTACKS for m in METHODS]

def test_budget_is_composed_not_reset():
    p=ledger('risk_rfa');assert epsilon(p,0)[0]<=4<epsilon(p,12)[0]
    e,a=epsilon(p,12)
    assert math.isclose(e,132*a/(2*p['gradient_z']**2)+132*a/(2*p['risk_z']**2)+math.log(1e5)/(a-1))
    assert all(epsilon(p,k)[0]<epsilon(p,k+1)[0] for k in range(12))
    with pytest.raises(ValueError):epsilon(p,13)

def test_all_branches_and_endpoints_required():
    r=results();assert decision(r)['local_gate_passed']
    with pytest.raises(ValueError):decision(r[:-1])
    r[-1]=deepcopy(r[0])
    with pytest.raises(ValueError):decision(r)

def test_recovery_failure_cannot_be_hidden_by_attack_endpoint():
    r=results();r[-1]['rows'][-1]['validation']=ev(850)
    assert not decision(r)['local_gate_passed']
    r=results();r[-1]['rows'][0]['model_hash']='different'
    with pytest.raises(ValueError):decision(r)

def test_clean_margin_exact_boundary():
    r=results()
    v=next(x for x in r if x['seed']==SEEDS[0] and x['attack']=='none' and x['method']=='risk_winsor')
    v['rows'][3]['validation']=ev(899) # −1/12 pp, inside −0.1
    assert decision(r)['local_gate_passed']
    v['rows'][3]['validation']=ev(898)
    assert not decision(r)['local_gate_passed']

def test_attack_schedule_and_private_only_formula_mps():
    assert torch.backends.mps.is_available()
    x=torch.arange(40,device='mps',dtype=torch.float32).reshape(10,4)/10
    r=torch.full((10,),.2,device='mps');mu=x[2:].mean(0)
    for kind in ATTACKS:
        for k in range(1,13):
            y,s,d=attack(x,r,kind,k)
            active=kind!='none' and 5<=k<=(12 if kind=='persistent_alie' else 8)
            assert d['active']==active and torch.equal(y[2:],x[2:]) and torch.equal(s[2:],r[2:])
            if not active:assert torch.equal(y,x) and torch.equal(s,r)
            else:
                expected=(-10*mu if kind=='abrupt_bf' else mu+1.5*x[2:].std(0,correction=0)
                          if kind=='persistent_alie' else -.5*(k-4)*mu)
                torch.testing.assert_close(y[0],expected);assert float(s[0])==1.
    with pytest.raises(ValueError):attack(x,r,'none',0)
