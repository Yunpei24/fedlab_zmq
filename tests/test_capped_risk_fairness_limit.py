"""Constructed all-honest counterexample; NOT a DP/benchmark performance run."""
import torch
from privacy.fair_objective import require_mps
from privacy.capped_private_risk import potential,weights
from privacy.split_risk_gradient import weighted_rfa


def test_robust_median_does_not_automatically_descend_the_fair_objective():
    require_mps()
    # One shared scalar model with sample-dependent fixed features/offsets.
    # Half-Brier for binary labels equals (sigmoid(k*theta+b)-y)^2.
    y=torch.tensor([0.]*8+[1.]*2,device='mps')
    p0=torch.tensor([.05**.5]*8+[1-.3**.5]*2,device='mps')
    offset=torch.logit(p0)
    target_gradient=torch.tensor([.25]*8+[-1.]*2,device='mps')
    slope=target_gradient/(2*(p0-y)*p0*(1-p0))
    theta=torch.zeros((),device='mps',requires_grad=True)
    def risks(th):return ((slope*th+offset).sigmoid()-y).square()
    r=risks(theta)
    grads=torch.stack([torch.autograd.grad(r[i],theta,retain_graph=True)[0] for i in range(10)])
    torch.testing.assert_close(grads,target_gradient,atol=3e-7,rtol=2e-6)
    clipped=grads.detach()*torch.clamp(1/grads.detach().abs(),max=1)
    lam,_=weights(r.detach(),.25)
    median,_=weighted_rfa(clipped[:,None],lam)
    mean=(lam*clipped).sum()
    J=potential(r,.25).mean()
    Jgrad=torch.autograd.grad(J,theta)[0]
    assert float(median)>.249 and float(mean)<-.18 and float(Jgrad)<-.319
    with torch.no_grad():
        after_median=potential(risks(theta-.1*median[0]),.25).mean()
        after_mean=potential(risks(theta-.1*mean),.25).mean()
    assert float(after_median)>float(J)
    assert float(after_mean)<float(J)
    print({'initial_objective':float(J.detach()),'fair_gradient':float(Jgrad),
           'weighted_mean':float(mean),'weighted_RFA':float(median),
           'objective_after_mean':float(after_mean),'objective_after_RFA':float(after_median)})
