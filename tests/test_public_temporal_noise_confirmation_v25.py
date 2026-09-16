import math
import pytest
from privacy.public_temporal_noise_confirmation_v25 import plan,paired_summary,compare
from privacy.public_temporal_noise_training_v24 import prefix_ledger,round_parameters
from scripts.run_public_temporal_noise_confirmation_v25 import inputs,jobs,identifier,decide
from scripts.run_private_clipping_step_diagnostic_v16 import ledger
from scripts.audit_public_temporal_noise_v23 import independent_rdp


def test_new_matrix_and_public_admission():
    c,profile,_ = inputs(); js = jobs(c)
    assert len(js) == len({identifier(j) for j in js}) == 32
    assert profile['evaluation_rounds'][-1] == 120 and c['test_evaluation_rounds'] == [120]
    assert set(j['seed'] for j in js) == {180601,180602,180603,180604}
    assert js[0] == dict(seed=180601,grid_index=0,method='erm_mean')


@pytest.mark.parametrize('method',['erm_mean','erm_rfa','risk_mean','risk_rfa'])
@pytest.mark.parametrize('index',[0,13])
def test_ledger_independent_at_boundaries(index,method):
    p = plan(index,method); risk = method.startswith('risk_')
    for count in (0,1,59,60,61,120):
        r = prefix_ledger(p,count)
        values = {a:min(count,60)*independent_rdp(a,.05,p['z_early'])+max(0,count-60)*independent_rdp(a,.05,p['z_late'])
            +(count*a/(2*p['risk_z']**2) if risk else 0.) for a in range(2,65)}
        for a in values:
            assert math.isclose(values[a],r['rdp'][a],rel_tol=2e-12,abs_tol=2e-12)
        expected = min(v+math.log(1e5)/(a-1) for a,v in values.items()) if count else 0.
        assert math.isclose(expected,r['epsilon'],rel_tol=2e-12,abs_tol=2e-12)
        assert r['epsilon'] <= 4
    assert round_parameters(p,60)['eta'] == 2 and round_parameters(p,61)['eta'] == .5
    original = ledger(2.,method)
    assert p['risk_std'] == original['risk_std']
    if index == 0:
        assert math.isclose(p['sigma_early'],original['gradient_std'],rel_tol=2e-12)
        assert p['sigma_late'] == p['sigma_early']


def test_type_checks_still_apply_after_cache_hit():
    plan(0,'erm_mean'); plan(13,'risk_rfa')
    for index in (False,0.,True,13.,7,14):
        with pytest.raises(ValueError): plan(index,'erm_mean')


def test_interval_and_prospective_alpha_spending():
    values = [.1,.3,.5,.7]; r = paired_summary(values)
    # Exact df=3 CDF from integrating 2/(pi*sqrt(3))*(1+x*x/3)^(-2).
    critical = 3.182446305284263
    u = critical/math.sqrt(3)
    cdf = .5+(math.atan(u)+u/(1+u*u))/math.pi
    assert math.isclose(cdf,.975,abs_tol=2e-13)
    assert math.isclose(r['ci95'][0],sum(values)/4-critical*r['sd']/2,abs_tol=1e-12)
    assert sum(.025*2**(-j) for j in range(60)) <= .05
    with pytest.raises(ValueError): paired_summary(values[:3])
    with pytest.raises(ValueError): paired_summary([1.,2.,3.,float('nan')])


def test_gate_needs_every_seed_and_uncertainty_not_only_means():
    good = dict(accuracy_pct=-.2,worst20_pct=2.,gap_best20_worst20_pp=-1.,variance_pp2=-5.)
    assert compare([good]*4)['passed']
    assert not compare([good]*3+[dict(good,worst20_pct=.9)])['passed']
    varied = [dict(good,accuracy_pct=x) for x in (-.99,-.99,.1,.1)]
    result = compare(varied)
    assert result['gates']['all_seed_gates'] and not result['gates']['accuracy_ci_lower_noninferior']


def test_final_test_controls_cannot_be_selected_or_incomplete():
    c,_,_ = inputs(); rows=[]
    for j in jobs(c):
        candidate = j['grid_index'] == 13 and j['method'] == 'risk_rfa'
        test = dict(accuracy_pct=80.,worst20_pct=70.+(2 if candidate else 0),
                    gap_best20_worst20_pp=10.-int(candidate),variance_pp2=30.-int(candidate))
        rows.append(dict(job=j,final=dict(test=test)))
    assert decide(c,rows)['clean_confirmation_passed']
    with pytest.raises(ValueError): decide(c,rows[:-1])
    with pytest.raises(ValueError): decide(c,rows[:-1]+[rows[0]])
    # Even one stronger constant ERM baseline must close admission.
    rows[0]['final']['test']['worst20_pct']=71.5
    assert not decide(c,rows)['clean_confirmation_passed']
