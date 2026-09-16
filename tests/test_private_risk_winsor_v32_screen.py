from copy import deepcopy
import pytest
from scripts.screen_private_risk_winsor_v32 import decision,evaluation_subset,SEEDS,ATTACKS,METHODS


def evaluation(hits=900):
    return dict(clients=[dict(N=1200,class_count=[1200]+[0]*9,class_hits=[hits]+[0]*9) for _ in range(10)])


def rows():
    return [dict(seed=s,attack=a,method=m,evaluation=evaluation(),numerical_checks_passed=True)
            for s in SEEDS for a in ATTACKS for m in METHODS]


def test_all_conditions_are_required():
    r=rows();assert decision(r)['local_feasibility_passed']
    r[-1]['evaluation']=evaluation(850)
    assert not decision(r)['local_feasibility_passed']


def test_incomplete_or_duplicate_probes_rejected():
    r=rows()
    with pytest.raises(ValueError):decision(r[:-1])
    r[-1]=deepcopy(r[0])
    with pytest.raises(ValueError):decision(r)


def test_honest_only_population_not_selected_after_outcome():
    ev=evaluation();ev['clients'][0]['class_hits'][0]=0;ev['clients'][1]['class_hits'][0]=0
    h=evaluation_subset(ev,range(2,10));all_=evaluation_subset(ev,range(10))
    assert h['worst20_pct']==75 and all_['worst20_pct']==0
