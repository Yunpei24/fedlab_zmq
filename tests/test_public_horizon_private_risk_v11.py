from scripts import run_public_horizon_private_risk_v11 as run
from privacy.scheduled_private_risk import learning_rate


def test_schedule_budget_recalibration_and_gate():
    m=run.config();profile,_=run.inputs(m)
    assert len(run.jobs(m))==8 and len(m['historical_controls'])==6
    assert [learning_rate(t,120) for t in (1,60,61,120)]==[2.,2.,.5,.5]
    old=run.parent.config()
    for kind in m['methods']:
        p=run.parent.privacy(profile,run.arm(kind));previous=run.parent.privacy(old,run.arm(kind))
        assert p['epsilon_realized']<=4 and p['gradient_std']>previous['gradient_std']
        if kind.startswith('risk_'):assert p['gradient_releases']==120 and p['risk_releases']==120
        else:assert p['steps']==120
    v=lambda a,w:dict(accuracy_pct=a,worst20_pct=w)
    rows=[dict(job=j,final=dict(validation=v(70,52 if j['method'].startswith('risk_') else 50))) for j in run.jobs(m)]
    historical=[dict(job=dict(seed=s,arm=c),final=dict(validation=v(70,50))) for s in m['calibration_seeds'] for c in m['historical_controls']]
    decision=run.decide(m,rows,historical)
    assert decision['selected_robust_candidate']=='risk_rfa'
    assert len(decision['gates']['risk_rfa']['comparisons'])==16
    historical[0]['final']['validation']=v(73,50)
    assert run.decide(m,rows,historical)['selected_robust_candidate'] is None
