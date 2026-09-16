import pytest
import torch
from privacy.private_risk_winsor_v32 import winsor
from privacy.capped_private_risk import weights
from privacy.fair_objective import require_mps


def test_noiseless_honest_minorities_preserved():
    require_mps()
    x=torch.tensor([[.25]]*8+[[-1.]]*2,device='mps')
    reports=torch.tensor([.05]*8+[.3]*2,device='mps')
    before=x.clone();a,d,v=winsor(x,reports,C=1.,f_budget=2)
    lam,_=weights(reports,.5)
    torch.testing.assert_close(a,(lam[:,None]*x).sum(0),rtol=1e-5,atol=1e-6)
    assert d['clipped_count']==0 and torch.equal(x,before)


@pytest.mark.parametrize('amplitude',[1e2,1e10,1e25])
def test_arbitrary_amplitude_cannot_inflate_radius_within_budget(amplitude):
    require_mps()
    x=torch.tensor([[amplitude,0.]]*2+[[.2,-.1]]*8,device='mps')
    reports=torch.tensor([1.]*2+[.05]*8,device='mps')
    a,d,v=winsor(x,reports,C=1.,f_budget=2)
    assert d['center_norm']<=1.00001 and d['radius']<=2.00001
    assert float(torch.linalg.vector_norm(a))<=3.0001
    assert d['clipped_count']<=2


def test_honest_cost_identity_and_byzantine_bound():
    require_mps()
    x=torch.arange(30,device='mps',dtype=torch.float32).reshape(10,3)/8
    reports=torch.linspace(0,1,10,device='mps')
    a,d,v=winsor(x,reports,C=1.,f_budget=2)
    lam=v['weights'];beta=lam[:2].sum()
    h=(lam[2:,None]*x[2:]).sum(0)/(1-beta)
    eh=(lam[2:,None]*v['removed'][2:]).sum(0)
    rhs=-eh+beta*(v['center']-h)+(lam[:2,None]*v['clipped_residuals'][:2]).sum(0)
    torch.testing.assert_close(a-h,rhs,rtol=1e-5,atol=1e-6)
    bound=float(torch.linalg.vector_norm(eh)+beta*(torch.linalg.vector_norm(v['center']-h)+d['radius']))
    assert float(torch.linalg.vector_norm(a-h))<=bound+1e-5
    assert float(beta)<=6/14+1e-6


def test_invalid_byzantine_budget_rejected():
    require_mps()
    with pytest.raises(ValueError):
        winsor(torch.zeros((10,3),device='mps'),torch.zeros(10,device='mps'),C=1.,f_budget=5)
