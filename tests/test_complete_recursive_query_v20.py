import math
import pytest
import torch
from privacy.fair_objective import require_mps
from privacy.complete_recursive_query import queries
from privacy.clipped_recursive_private_gradient import recursive_release
from privacy.fair_objective import release


def params():return dict(C=2.,theta=1-math.sqrt(.5+1e-8),D=2*(1-math.sqrt(.5+1e-8)))


def rows(seed):
    require_mps();torch.mps.manual_seed(seed)
    r=torch.randn(240,31,device='mps');return r*(2/torch.linalg.vector_norm(r,dim=1)).clamp(max=1)[:,None]


def test_radius_and_projection_pythagorean_inequality():
    e,d,p,k=queries(rows(2401),rows(2402),**params())
    for x in (d,p):assert float(torch.linalg.vector_norm(x,dim=1).max())<=k['effective_C']+2e-6
    lhs=(e-p).square().sum(1)+(p-d).square().sum(1)
    rhs=(e-d).square().sum(1)
    assert float((lhs-rhs).max())<=2e-5
    assert abs(k['query_sensitivity']-2*k['effective_C']/240)<1e-15


def test_complete_query_preserves_some_unnecessarily_clipped_differences():
    require_mps();c=torch.tensor([[.8,0.]],device='mps');z=torch.zeros_like(c)
    e,d,p,k=queries(c,z,**params())
    assert k['difference_clipped_count']==1 and k['complete_query_clipped_count']==0
    torch.testing.assert_close(p,e,rtol=0,atol=0)
    assert float(torch.linalg.vector_norm(e-d))>.1


def test_replace_one_mean_sensitivity_and_unchanged_rows():
    c,old=rows(2411),rows(2412);p=queries(c,old,**params())[2]
    c2,old2=c.clone(),old.clone();c2[0]=-c[0];old2[0]=-old[0]
    _,_,p2,k=queries(c2,old2,**params())
    torch.testing.assert_close(p[1:],p2[1:],rtol=0,atol=0)
    assert float(torch.linalg.vector_norm(p.mean(0)-p2.mean(0)))<=k['query_sensitivity']+2e-6


def test_theta_one_and_zero_difference():
    c,old=rows(2421),rows(2422)
    e,d,p,k=queries(c,old,C=2.,D=1.,theta=1.)
    torch.testing.assert_close(e,c,rtol=0,atol=0);torch.testing.assert_close(p,c,rtol=1e-5,atol=2e-6)
    e,d,p,k=queries(c,c,**params())
    torch.testing.assert_close(d,e,rtol=1e-5,atol=2e-6)
    torch.testing.assert_close(p,e,rtol=0,atol=0)


def test_invalid_or_unclipped_inputs_rejected():
    c=rows(2431)
    with pytest.raises(ValueError):queries(c*10,c,**params())
    bad=c.clone();bad[0,0]=float('nan')
    with pytest.raises(ValueError):queries(c,bad,**params())


def test_original_release_bitwise_and_same_fresh_noise():
    c,old=rows(2441),rows(2442);p=params();mem=torch.ones(c.shape[1],device='mps')*.2
    expected,d=recursive_release(c,old,mem,base_noise_std=.03,seed=811,**p)
    _,q,projected,k=queries(c,old,**p);a=1-p['theta']
    actual=a*mem+release(q.mean(0),noise_std=d['noise_std'],seed=811)
    torch.testing.assert_close(expected,actual,rtol=0,atol=0)
    alt=a*mem+release(projected.mean(0),noise_std=d['noise_std'],seed=811)
    torch.testing.assert_close(alt-actual,projected.mean(0)-q.mean(0),rtol=2e-4,atol=2e-7)
    assert math.isclose(d['query_sensitivity'],k['query_sensitivity'],rel_tol=1e-14)


def test_mechanism_gain_does_not_override_model_gate():
    from scripts.run_complete_recursive_query_diagnostic_v20 import decide,SEEDS,ROUNDS
    def records(worst):
        return [dict(seed=s,round=k,client_diagnostics=[dict(difference_batch_bias_sq=1.,complete_batch_bias_sq=.5)],
            delta=dict(worst20_pct=worst,accuracy_pct=0.,J_gain=.01)) for s in SEEDS for k in ROUNDS]
    assert decide(records(.1))['admitted_to_new_end_to_end_screen']
    assert not decide(records(-.01))['admitted_to_new_end_to_end_screen']
