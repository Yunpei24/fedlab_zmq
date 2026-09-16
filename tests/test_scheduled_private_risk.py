import pytest
import torch
from privacy.fair_objective import require_mps
from privacy.scheduled_private_risk import learning_rate,aggregate


def test_schedule_is_public_and_fixed():
    assert [learning_rate(t) for t in (1,30,31,60)]==[2.,2.,.5,.5]
    with pytest.raises(ValueError):learning_rate(61)


def test_mean_uses_same_schedule_and_does_not_need_unnecessary_report():
    require_mps()
    x=torch.tensor([[1.,0.],[0.,1.],[-1.,0.]],device='mps')
    a,d=aggregate(x,None,kind='erm_mean',round_number=31)
    torch.testing.assert_close(a,.5*x.mean(0)); assert d['eta']==.5
    r=torch.tensor([.1,.3,.4],device='mps')
    aa,_=aggregate(x,r,kind='risk_rfa',round_number=30)
    bb,_=aggregate(x,r,kind='risk_rfa',round_number=31)
    torch.testing.assert_close(aa,4*bb)


def test_screen_does_not_hide_strong_or_matched_controls():
    from scripts import run_public_decay_private_risk_v9 as run
    m=run.config()
    assert len(run.jobs(m))==8 and len(m['historical_controls'])==4 and len(m['new_controls'])==2
    assert set(m['calibration_seeds']).isdisjoint(m['reserved_confirmation_seeds'])
    v=lambda acc,w:dict(accuracy_pct=acc,worst20_pct=w)
    rows=[dict(job=j,final=dict(validation=v(70,50))) for j in run.jobs(m)]
    old=[dict(job=dict(seed=s,arm=c),final=dict(validation=v(70,50))) for s in m['calibration_seeds'] for c in m['historical_controls']]
    for r in rows:
        if r['job']['method'].startswith('risk_'):
            r['final']['validation']=v(70,52)
    assert run.decide(m,rows,old)['selected_robust_candidate']=='risk_rfa'
    old[0]['final']['validation']=v(73,50)
    assert run.decide(m,rows,old)['selected_robust_candidate'] is None
