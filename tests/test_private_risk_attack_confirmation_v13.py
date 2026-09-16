import copy
import pytest
from scripts import run_private_risk_attack_confirmation_v13 as run


def mock_rows():
    out=[]
    for j in run.jobs(run.config()):
        candidate=j['method']=='risk_rfa'
        v=dict(accuracy_pct=80.,worst20_pct=64. if candidate else 60.,
               gap_best20_worst20_pp=20. if candidate else 24.,variance_pp2=40. if candidate else 50.)
        rows=[dict(round=t,test_honest=copy.deepcopy(v) if t in (90,120) else None,
                   aggregation=dict(solver=dict(unsmoothed_objective_gap_upper=1e-5))) for t in range(1,121)]
        out.append(dict(job=j,rounds=rows))
    return out


def test_exact_grid_and_clean_replay_included():
    m=run.config()
    assert len(run.jobs(m))==64
    assert sum(j['attack']=='none' for j in run.jobs(m))==16
    assert m['require_clean_confirmation_passed']


def test_gate_uses_attack_endpoint_not_recovered_model():
    m,rows=run.config(),mock_rows()
    assert run.decide(m,rows)['attack_confirmation_passed']
    r=next(r for r in rows if r['job']==dict(seed=170601,method='risk_rfa',attack='abrupt_bf'))
    r['rounds'][89]['test_honest']['worst20_pct']=59.
    d=run.decide(m,rows)
    assert not d['attack_confirmation_passed'] and not d['attacks']['abrupt_bf']['passed']
    assert d['attacks']['persistent_alie']['passed']


def test_recovery_cost_and_solver_gate_cannot_be_hidden():
    m,rows=run.config(),mock_rows()
    r=next(r for r in rows if r['job']==dict(seed=170601,method='risk_rfa',attack='slow_ipm'))
    r['rounds'][119]['test_honest']['accuracy_pct']=74.9
    assert not run.decide(m,rows)['attack_confirmation_passed']
    rows=mock_rows()
    r=next(r for r in rows if r['job']==dict(seed=170601,method='risk_rfa',attack='persistent_alie'))
    r['rounds'][59]['aggregation']['solver']['unsmoothed_objective_gap_upper']=.00101
    d=run.decide(m,rows)
    assert not d['attack_confirmation_passed'] and not d['attacks']['persistent_alie']['solver_gate']


def test_no_selective_rows_and_fixed_honest_ids():
    with pytest.raises(ValueError):run.decide(run.config(),mock_rows()[:-1])
    clients=[dict(accuracy=a,ce_loss=1.,brier_loss=.25,balanced_accuracy=a) for a in [0.,0.]+[.8]*8]
    h=run.honest_metrics(dict(clients=clients))
    assert h['accuracy_pct']==h['worst20_pct']==80 and h['honest_ids']==list(range(2,10))
    assert h['variance_pp2']==0 and h['worst20_client_count']==2
