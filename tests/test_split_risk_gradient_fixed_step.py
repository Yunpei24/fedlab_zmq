import torch
import pytest
from privacy.fair_objective import require_mps
from privacy.split_risk_gradient import aggregate as previous
from privacy.split_risk_gradient_fixed_step import aggregate


@pytest.mark.parametrize('method',['erm_full','matched_mean','risk_mean','matched_rfa','risk_rfa'])
def test_fixed_step_direction_identical_to_v5(method):
    require_mps()
    x=torch.tensor([[1.,0.],[0.,2.],[-.5,1.]],device='mps')
    r=torch.tensor([.1,.7,.9],device='mps')
    a,da=aggregate(x,r,method=method)
    b,db=previous(x,r,method=method)
    assert da['eta']==2 and da['weights']==db['weights']
    torch.testing.assert_close(a,b*(2/db['eta']),atol=1e-6,rtol=2e-5)
    assert da['fixed_step']


def test_step_does_not_decay_when_risk_reports_decrease():
    require_mps()
    x=torch.tensor([[1.,0.],[0.,2.]],device='mps')
    for v in (0.,.1,.5,1.):
        a,d=aggregate(x,torch.full((2,),v,device='mps'),method='risk_mean')
        assert d['eta']==2
        torch.testing.assert_close(a,2*x.mean(0))
    with pytest.raises(ValueError):aggregate(x,None,method='erm_full',eta_fair=1.)


def test_v6_only_changes_the_step_and_preserves_budget_and_calibration_scope():
    from scripts import run_split_risk_gradient_fixed_step_v6 as r6
    from scripts import run_split_risk_gradient_v5 as r5
    m6=r6.config();m5=r5.config()
    excluded={'campaign_id','eta_fair','parent_campaign','paired_randomness_source','protocol'}
    assert {k:v for k,v in m6.items() if k not in excluded}=={k:v for k,v in m5.items() if k not in excluded}
    for method in m6['methods']:assert r6.privacy(m6,method)==r5.privacy(m5,method)
    assert r6.OUT!=r5.OUT and r6.REPORT!=r5.REPORT
