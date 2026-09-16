from copy import deepcopy
import pytest
from scripts.audit_private_risk_continuation_v33 import exact,independently_decide,SEEDS,CONDITIONS,METHODS

def ev(hits=900):
    return dict(clients=[dict(N=1200,class_count=[1200]+[0]*9,class_hits=[hits]+[0]*9) for _ in range(10)])

def index():
    return {(s,c,m):dict(rows=[dict(validation=ev()) for _ in range(12)]) for s in SEEDS for c in CONDITIONS for m in METHODS}

def test_eight_honest_ids_and_exact_counts():
    v=ev();v['clients'][0]['class_hits'][0]=0
    assert exact(v,range(2,10))['worst20_pct']==75
    assert exact(v,range(10))['worst20_pct']==37.5
    v['clients'][0]['class_hits'][0]=1201
    with pytest.raises(AssertionError):exact(v,range(10))

def test_independent_gate_checks_every_condition_seed_endpoint():
    ix=index();checks=independently_decide(ix)
    assert len(checks)==30 and all(c['passed'] for c in checks)
    for s in SEEDS:
        for condition in CONDITIONS[1:]:
            for step in (8,12):
                altered=deepcopy(ix);altered[s,condition,'risk_winsor']['rows'][step-1]['validation']=ev(870)
                assert not all(c['passed'] for c in independently_decide(altered))
    del ix[SEEDS[0],'none','risk_mean']
    with pytest.raises(AssertionError):independently_decide(ix)

def test_incomplete_audit_does_not_invent_four_missing_comparisons():
    ix=index()
    for method in ('risk_rfa','risk_winsor'):del ix[170502,'slow_ipm',method]
    with pytest.raises(AssertionError):independently_decide(ix)
    comparisons=independently_decide(ix,allow_incomplete=True)
    assert len(comparisons)==26
    assert not any(c['seed']==170502 and c['attack']=='slow_ipm' for c in comparisons)
    ix[170501,'abrupt_bf','risk_winsor']['rows'][7]['validation']=ev(850)
    assert any(not c['passed'] for c in independently_decide(ix,allow_incomplete=True))
