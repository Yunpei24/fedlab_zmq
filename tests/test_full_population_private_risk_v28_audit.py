import copy
import json
import math
import pytest
from scripts import analyze_full_population_private_risk_v28 as audit


def fixtures():
    records=[];historical=[]
    for seed in audit.SEEDS:
        for method in audit.METHODS:
            primary=method=='risk_rfa'
            v=dict(zip(audit.KEYS,(79. if primary else 80.,61. if primary else 60.,
                20. if primary else 22.,70. if primary else 80.,65.,.6,.2)))
            records.append(dict(job=dict(seed=seed,method=method),validation=v,
                                endpoint_round=120,test_evaluated=False))
            if method.startswith('erm_'):historical.append(copy.deepcopy(records[-1]))
    return records,historical


def test_gate_fixed_thresholds_and_complete_matrix():
    a,b=fixtures();assert audit.decide(a,b)['calibration_passed']
    a[-1]['validation']['worst20_pct']-=1e-4
    d=audit.decide(a,b);assert not d['calibration_passed'] and not d['global_validation']
    for c in d['comparisons']:assert not c['gates']['all_seeds_worst20']
    with pytest.raises(ValueError):audit.decide(a[:-1],b)
    with pytest.raises(ValueError):audit.decide(a+[a[0]],b)
    with pytest.raises(ValueError):audit.decide(a,b[:-1])


def test_accuracy_and_variance_cannot_be_replaced_by_loss():
    a,b=fixtures();a[-1]['validation']['accuracy_pct']=78.999
    a[-1]['validation']['ce_loss']=.01
    assert not audit.decide(a,b)['calibration_passed']
    a,b=fixtures();a[-1]['validation']['variance_pp2']=91.
    assert not audit.decide(a,b)['calibration_passed']
    a,b=fixtures();a[-1]['validation']['gap_best20_worst20_pp']=25.
    assert not audit.decide(a,b)['calibration_passed']


def test_no_checkpoint_selection_or_primary_switch():
    a,b=fixtures();a[-1]['endpoint_round']=60
    with pytest.raises(ValueError):audit.decide(a,b)
    a,b=fixtures();a[-1]['test_evaluated']=True
    with pytest.raises(ValueError):audit.decide(a,b)
    a,b=fixtures();a[-1]['validation']['worst20_pct']=59.
    a[-2]['validation']['worst20_pct']=90.
    assert not audit.decide(a,b)['calibration_passed']


def test_historical_controls_are_not_silently_omitted():
    a,b=fixtures();b[-1]['validation']['accuracy_pct']=82.
    d=audit.decide(a,b)
    assert d['comparisons'][0]['passed'] and d['comparisons'][1]['passed']
    assert not d['calibration_passed'] and not d['comparisons'][-1]['passed']


def test_nonfinite_or_one_seed_not_evidence():
    with pytest.raises(ValueError):audit.moments([1.])
    with pytest.raises(ValueError):audit.moments([1.,math.nan])
    a,b=fixtures();a[-1]['validation']['accuracy_pct']=math.nan
    with pytest.raises(ValueError):audit.decide(a,b)
    assert audit.moments([1.,3.])['confidence_interval'] is None


def test_q1_gaussian_audit_rejects_spurious_amplification():
    evidence=json.loads((audit.ROOT/'output/analysis/Additive_Gaussian_WOR_V27_Public_Audit.json').read_text())
    for risk in (False,True):
        family='private_risk_channel' if risk else 'erm_all_budget'
        public=next(r['plan'] for r in evidence['rows'] if r['family']==family and r['plan']['batch']==4800)
        p=dict(N=4800,b=4800,T=120,C=2.,adjacency='replace_one',sampling='full_population',
            epsilon_cap=4.,delta=1e-5,gradient_releases=120,risk_releases=120 if risk else 0,
            accumulation_blocks_are_not_releases=True,gradient_sensitivity=4/4800,
            gradient_z=public['gradient_z'],gradient_std=public['gradient_std'],
            risk_z=public['risk_z'],risk_std=public['risk_std'] if risk else 0.,
            risk_sensitivity=1/4800 if risk else None,risk_calibration_epsilon=.25 if risk else None,
            risk_calibration_delta=5e-6 if risk else None,epsilon_realized=public['epsilon_realized'],
            order=public['order'],rdp=public['rdp'])
        method='risk_rfa' if risk else 'erm_mean'
        assert len(audit.privacy_audit(p,method))==120
        bad=copy.deepcopy(p);bad['b']=240
        with pytest.raises(AssertionError):audit.privacy_audit(bad,method)
        bad=copy.deepcopy(p);bad['rdp']['7']*=.05
        with pytest.raises(AssertionError):audit.privacy_audit(bad,method)
        bad=copy.deepcopy(p);bad['gradient_releases']=20*120
        with pytest.raises(AssertionError):audit.privacy_audit(bad,method)
