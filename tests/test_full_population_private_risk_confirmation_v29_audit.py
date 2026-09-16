import copy
from fractions import Fraction as Q
import inspect
import json
import math
import mpmath as mp
import pytest
import torch
from scripts import audit_full_population_private_risk_confirmation_v29 as a
from privacy import full_population_private_risk_confirmation_v29 as candidate


def test_independent_accounting_and_prefixes_match_frozen_plans():
    for b,m in a.ARMS:
        p=json.loads(json.dumps(candidate.plan(b,m)))
        values=a.privacy_check(p,dict(batch=b,method=m))
        assert len(values)==120 and values[-1][0]<=4
        for t,(eps,order) in enumerate(values,1):
            stored=candidate.prefix(p,t)
            assert order==stored[1] and abs(eps-stored[0])<2e-11


def test_accounting_rejects_sampling_sensitivity_or_release_changes():
    base=json.loads(json.dumps(candidate.plan(4800,'risk_rfa'))); job=dict(batch=4800,method='risk_rfa')
    for key,value in [('sampling','poisson'),('gradient_sensitivity',2/4800),('gradient_releases',240),
                      ('risk_releases',0),('risk_std',0.)]:
        p=copy.deepcopy(base); p[key]=value
        with pytest.raises(AssertionError): a.privacy_check(p,job)
    p=copy.deepcopy(base); p['rdp']['7']*=.5
    with pytest.raises(AssertionError): a.privacy_check(p,job)


def metric(primary=False,population=1000):
    raw=[650,650,800,800,800,800,800,800,900,900] if primary else [600,600,800,800,800,800,800,800,900,900]
    hits=[h*population//1000 for h in raw]; numbers=[Q(100*h,population) for h in hits]
    mean=sum(numbers)/10; ss=sorted(numbers); worst=sum(ss[:2])/2
    return dict(accuracy_pct=float(mean),worst20_pct=float(worst),
        gap_best20_worst20_pp=float(sum(ss[-2:])/2-worst),variance_pp2=float(sum((x-mean)**2 for x in numbers)/10),
        clients=[dict(N=population,class_count=[population]+[0]*9,class_hits=[h]+[0]*9) for h in hits])


def rows():
    return [dict(job=dict(seed=s,batch=b,method=m),endpoint_round=120,test_evaluated=True,
                 test=metric(b==4800 and m=='risk_rfa')) for s in a.SEEDS for b,m in a.ARMS]


def source_rows(rs):
    return [dict(job=r['job'],device='mps',test_evaluated=True,test_evaluation_rounds=[120],
                 final=dict(round=r['endpoint_round'],test=r['test'])) for r in rs]


def test_exact_count_formula_and_invalid_cells():
    v=metric(True); independent=a.exact_counts(v); original=candidate.exact_test_metrics(v)
    assert independent==original and all(isinstance(v,Q) for v in independent.values())
    assert a.exact_counts(metric(True,1200),1200)==independent
    for field,value in [('accuracy_pct',float('nan')),('variance_pp2',-2.)]:
        bad=copy.deepcopy(v); bad[field]=value
        with pytest.raises(ValueError): a.exact_counts(bad)
    bad=copy.deepcopy(v); bad['clients'][0]['class_hits'][0]=1001
    with pytest.raises(ValueError): a.exact_counts(bad)
    bad=copy.deepcopy(v); bad['clients'][0]['N']=1200
    with pytest.raises(ValueError): a.exact_counts(bad)


def test_independent_t_cdf_and_interval():
    xs=list(map(Q,(1,2,3,4))); out=a.interval(xs)
    assert out['mean']==2.5 and math.isclose(out['sd'],math.sqrt(5/3),rel_tol=1e-15)
    quantile=(2.5-out['lower_one_sided_9875'])*2/out['sd']
    with mp.workdps(60):
        t=mp.mpf(quantile)
        tail=mp.betainc(mp.mpf('1.5'),mp.mpf('.5'),0,3/(3+t*t),regularized=True)/2
        assert abs(tail-mp.mpf('.0125'))<mp.mpf('1e-15')
    assert abs(quantile-candidate.TCRIT)<1e-12
    with pytest.raises(ValueError): a.interval([Q(1),Q(2)])


def test_entire_matrix_and_runner_agree():
    rs=rows(); independent=a.independent_decision(rs); ref=candidate.decide(source_rows(rs))
    assert independent['clean_confirmation_passed']; a.compare_runner(independent,ref)
    for bad in (rs[:-1],rs[:-1]+[rs[0]]):
        with pytest.raises(ValueError): a.independent_decision(bad)
    bad=copy.deepcopy(rs); bad[0]['endpoint_round']=90
    with pytest.raises(ValueError): a.independent_decision(bad)
    bad=copy.deepcopy(rs); bad[0]['test_evaluated']=False
    with pytest.raises(ValueError): a.independent_decision(bad)


def test_no_primary_replacement_and_no_threshold_relaxation():
    rs=rows()
    for r in rs:
        if r['job']['method']=='risk_mean': r['test']=metric(True)
        if r['job']['method']=='risk_rfa': r['test']=metric(False)
    independent=a.independent_decision(rs); ref=candidate.decide(source_rows(rs))
    assert not independent['clean_confirmation_passed']; a.compare_runner(independent,ref)
    modified=copy.deepcopy(ref); modified['clean_confirmation_passed']=True
    with pytest.raises(AssertionError): a.compare_runner(independent,modified)


def test_independent_rng_does_not_advance_training_state():
    a.require_mps(); state=torch.mps.get_rng_state(); key='synthetic unit test key, not a simulation key'
    s=a.seed_for(key,192,119,4,'batch')
    assert s==a.base.seed_for(key,192,119,4,'batch')
    full=a.random_draw(s,permutation=True)
    original=a.base.draw_indices(4800,4800,s)
    assert torch.equal(full,original) and torch.equal(state,torch.mps.get_rng_state())
    z=a.random_draw(s,dimension=(17,))
    assert z.device.type=='mps' and torch.equal(z,a.random_draw(s,dimension=(17,)))
    assert torch.equal(state,torch.mps.get_rng_state())


def test_auditor_does_not_import_the_mechanism_or_runner():
    source=inspect.getsource(a)
    imports=[line for line in source.splitlines() if line.startswith(('from ','import '))]
    assert not any('privacy.full_population_private_risk_confirmation_v29' in line for line in imports)
    assert not any('run_full_population_private_risk_confirmation_v29' in line for line in imports)
    assert 'split(120)' in inspect.getsource(a.replay)
    assert 'parameter.sub_' in inspect.getsource(a.replay)
