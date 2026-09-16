import pytest
import torch
from privacy.fair_objective import require_mps
from privacy.private_risk_message_safety import sanitize
from privacy.capped_private_risk import weights
from privacy.stable_weighted_rfa import weighted_rfa


@pytest.fixture(scope='module',autouse=True)
def mps():require_mps()


def test_clean_private_messages_are_unchanged_bitwise():
    x=torch.tensor([[1.,2.],[0.,-.5]],device='mps');r=torch.tensor([.2,.9],device='mps')
    a,b,d=sanitize(x,r)
    assert torch.equal(a,x) and torch.equal(b,r) and d['invalid_message_rows']==0


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),float('-inf'),1e35])
def test_untrusted_row_does_not_crash_median_and_keeps_mass_bound(bad):
    torch.mps.manual_seed(171011)
    h=.01*torch.randn(8,16,device='mps')
    x=torch.cat([h,torch.full((2,16),bad,device='mps')])
    r=torch.tensor([.1]*8+[1.,1.],device='mps')
    a,b,d=sanitize(x,r)
    assert bool(torch.isfinite(a).all()) and bool((a[-2:]==0).all())
    assert bool((b[-2:]==0).all()) and d['invalid_message_rows']==2
    lam,_=weights(b,.5);z,_=weighted_rfa(a,lam)
    assert bool(torch.isfinite(z).all()) and float(lam[-2:].sum())<=3/7+1e-7


def test_invalid_scalar_cannot_create_unbounded_objective_weight():
    x=torch.zeros(10,2,device='mps')
    r=torch.tensor([.1]*6+[float('nan'),float('inf'),-1.,100.],device='mps')
    a,b,d=sanitize(x,r)
    assert torch.equal(a,x) and d['nonfinite_risk_reports']==2
    torch.testing.assert_close(b[-4:],torch.tensor([0.,0.,0.,1.],device='mps'))
    lam,coef=weights(b,.5)
    assert float(coef.min())>=1 and float(coef.max())<=3
