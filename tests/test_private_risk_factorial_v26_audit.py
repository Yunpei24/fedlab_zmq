import copy
from scripts import analyze_private_risk_factorial_diagnostic_v26 as audit
from privacy.private_risk_factorial_diagnostic_v26 import treatments


def fixture():
    rows=[]
    for t in treatments():
        w={'private':0., 'oracle':2., 'uniform':1.}[t['weight']]
        n=float(t['message']=='clean_clipped_oracle')
        a=float(t['aggregator']=='rfa')
        value=w+3*n+5*a+7*w*n+11*w*n*a
        keys=('error_to_population_target_sq','error_to_clipped_batch_target_sq','predicted_J_decrease',
              'actual_J_decrease','finite_step_remainder')
        val=('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2','balanced_accuracy_pct','ce_loss','brier_loss')
        rows.append(dict(seed=1,round=2,treatment=t,effects={k:value for k in keys},validation_after={k:value for k in val}))
    return rows


def test_contrasts_keep_other_factors_fixed():
    rows=fixture(); saved=copy.deepcopy(rows)
    cs=audit.conditional_contrasts(rows)
    assert len(cs)==24 and rows==saved
    for c in cs:
        changed=[k for k in c['candidate'] if c['candidate'][k]!=c['baseline'][k]]
        assert len(changed)==1
    for c in cs:
        if c['family']=='remove_risk_noise' and c['candidate']['message']=='private':
            assert c['effects']['actual_J_decrease']==2.


def test_explicit_interactions_preserve_nonadditivity():
    values=audit.interactions(fixture())
    assert values[0]['values']['actual_J_decrease']==14.
    assert values[1]['values']['actual_J_decrease']==36.
    assert values[2]['values']['actual_J_decrease']==22.


def test_delta_direction_candidate_minus_control():
    d=audit.delta({'x':4.},{'x':7.},['x'])
    assert d=={'x':-3.}
