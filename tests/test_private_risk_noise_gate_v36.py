import math
import pytest
import torch
from privacy.fair_objective import require_mps
from privacy.private_risk_noise_gate_v36 import norm_bounds,gated_risk_smea,bounded_gaussian_attack
from privacy.capped_private_risk import weights


def test_public_bounds_no_noise_scaling_and_union_allocation():
    b=norm_bounds(dimension=100,C=2,std=0,n=10)
    assert b['lower']==0 and b['upper']==2
    b=norm_bounds(dimension=61706,C=2,std=.01185581970688466,n=10)
    assert 2.8<b['lower']<3 and 3.5<b['upper']<3.8
    c=norm_bounds(dimension=61706,C=4,std=2*.01185581970688466,n=10)
    assert math.isclose(c['lower'],2*b['lower']) and math.isclose(c['upper'],2*b['upper'])
    assert math.isclose(10*132*b['per_message_two_tails_upper'],.01)
    with pytest.raises(ValueError):norm_bounds(dimension=10,C=1,std=-1,n=10)


def test_compatible_cohort_is_exact_fair_mean():
    require_mps()
    x=torch.zeros(10,1000,device='mps');x[:,0]=3.2
    r=torch.linspace(0,1,10,device='mps');lam,_=weights(r,.5)
    # Gaussian norm ~sqrt(1000)*.1; mean bound C=2.
    out,di=gated_risk_smea(x,r,C=2,stds=[.1]*10)
    assert di['branch']=='fair_mean'
    assert di['incompatible_ids']==[]
    torch.testing.assert_close(out,(lam[:,None]*x).sum(0),rtol=0,atol=0)


def test_incompatible_cohort_invokes_frozen_spectral_fallback():
    require_mps()
    x=torch.zeros(10,1000,device='mps');x[:,0]=3.2;x[:2,0]=30
    r=torch.zeros(10,device='mps')
    out,di=gated_risk_smea(x,r,C=2,stds=[.1]*10)
    assert di['branch']=='spectral' and di['incompatible_ids']==[0,1]
    assert di['spectral']['selected_ids']==list(range(2,10))
    torch.testing.assert_close(out,x[2:].mean(0),rtol=2e-5,atol=2e-6)


def test_gaussian_attack_is_reproducible_bounded_and_restores_rng():
    require_mps();torch.mps.manual_seed(3601)
    x=torch.randn(10,500,device='mps')*.1
    state=torch.mps.get_rng_state()
    a,di=bounded_gaussian_attack(x,C=2,stds=[.1]*10,seed=3612)
    after=torch.mps.get_rng_state();assert torch.equal(state,after)
    b,dj=bounded_gaussian_attack(x,C=2,stds=[.1]*10,seed=3612)
    assert torch.equal(a,b) and torch.equal(a[2:],x[2:]) and not torch.equal(a[0],a[1])
    assert di==dj and di['mean_norm']<=2+1e-5 and di['retries']==0
    shifted,meta=bounded_gaussian_attack(x,C=2,stds=[.1]*10,seed=3612,target_coordinate=499)
    assert meta['mean_norm']==2 and meta['target_coordinate']==499
    assert torch.equal(shifted[2:],x[2:])
