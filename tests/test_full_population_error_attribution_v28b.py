import math
import pytest
import torch
from privacy.full_population_error_attribution_v28b import decompose,NAMES
from privacy.fair_objective import require_mps
from scripts.analyze_full_population_error_attribution_v28b import objective


def vectors():
    require_mps();torch.mps.manual_seed(282928)
    g=torch.randn(3,7,device='mps');c=.5*g
    x=c+.2*torch.randn_like(g)
    w=torch.tensor([.2,.3,.5],device='mps');p=torch.tensor([.3,.3,.4],device='mps');v=torch.tensor([.4,.2,.4],device='mps')
    a=(v[:,None]*x).sum(0)+.01
    return g,c,x,a,w,p,v


def test_all_five_components_and_cross_terms_close():
    args=vectors();d,h,parts=decompose(*args,objective_scale=2.,eta=.5)
    torch.testing.assert_close(args[3]-h,sum(parts.values()),rtol=3e-5,atol=3e-7)
    assert len(d['doubled_cross_products'])==10 and set(parts)==set(NAMES)
    assert abs(sum(d['first_order_gain_loss'].values())-(d['ideal_predicted_gain']-d['applied_predicted_gain']))<1e-5
    assert not d['independent_or_centered_errors_assumed'] and not d['feeds_mechanism']


def test_no_clipping_reports_or_rfa_leaves_only_noise():
    g,_,_,_,w,_,_=vectors();z=torch.ones_like(g)*.1;x=g+z;a=(w[:,None]*x).sum(0)
    d,_,parts=decompose(g,g,x,a,w,w,w,objective_scale=1.,eta=.5)
    for k in NAMES:
        if k!='effective_noise':assert d['squared_components'][k]==0
    torch.testing.assert_close(parts['effective_noise'],torch.ones(7,device='mps')*.1)


def test_mse_and_directional_cost_are_distinct():
    require_mps();g=torch.tensor([[1.,0.],[1.,0.]],device='mps');w=torch.ones(2,device='mps')/2
    x=g+torch.tensor([0.,10.],device='mps');a=x.mean(0)
    d,_,_=decompose(g,g,x,a,w,w,w,objective_scale=1.,eta=.5)
    assert d['squared_error']==100. and d['first_order_gain_loss']['effective_noise']==0.
    x=g-torch.tensor([.5,0.],device='mps');a=x.mean(0)
    d,_,_=decompose(g,g,x,a,w,w,w,objective_scale=1.,eta=.5)
    assert d['squared_error']==.25 and d['first_order_gain_loss']['effective_noise']==.25


def test_population_potential_gradient_chain_rule():
    require_mps();theta=torch.tensor(.1,device='mps',requires_grad=True)
    risks=torch.stack(((theta-.6).square(),.7+.1*theta, .1+.2*theta.square()))
    potential=risks+torch.where(risks<=.5,2*risks.square(),2*risks-.5)
    exact=torch.autograd.grad(potential.mean(),theta,retain_graph=True)[0]
    gradients=torch.stack([torch.autograd.grad(r,theta,retain_graph=True)[0] for r in risks])
    coeff=1+2*(risks/.5).clamp(max=1);w=coeff/coeff.sum()
    reconstructed=coeff.mean()*(w*gradients).sum()
    torch.testing.assert_close(exact,reconstructed,rtol=2e-6,atol=2e-7)
    assert math.isclose(objective(risks,'risk_rfa'),float(potential.mean().detach()),rel_tol=2e-6)
    assert math.isclose(objective(risks,'erm_rfa'),float(risks.mean().detach()),rel_tol=2e-6)


def test_reject_nonfinite_cpu_or_nonprobability_weights():
    args=list(vectors())
    bad=args.copy();bad[0]=bad[0].cpu()
    with pytest.raises(ValueError):decompose(*bad,objective_scale=1.,eta=.5)
    bad=args.copy();bad[-1]=bad[-1]*2
    with pytest.raises(ValueError):decompose(*bad,objective_scale=1.,eta=.5)
    with pytest.raises(ValueError):decompose(*args,objective_scale=math.nan,eta=.5)
    with pytest.raises(ValueError):decompose(*args,objective_scale=1.,eta=0.)
