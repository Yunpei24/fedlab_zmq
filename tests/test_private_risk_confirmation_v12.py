import copy
import math
import pytest
import torch
from scripts import run_private_risk_confirmation_v12 as run
from privacy.scheduled_private_risk import aggregate
from privacy.private_risk_message_safety import sanitize


def mock_rows():
    rows = []
    for j in run.jobs(run.config()):
        is_risk = j['method'].startswith('risk_')
        v = dict(accuracy_pct=80., worst20_pct=63. if is_risk else 60.,
                 gap_best20_worst20_pp=20. if is_risk else 23., variance_pp2=40. if is_risk else 50.)
        rows.append(dict(job=j,final=dict(test=v)))
    return rows


def test_frozen_profile_and_joint_privacy():
    m = run.config()
    profile,stamp = run.inputs(m)
    assert len(run.jobs(m)) == 16 and len(set(j['seed'] for j in run.jobs(m))) == 4
    assert profile['model'] == 'lenet5_tanh' and profile['num_clients'] == 10
    assert profile['public_train_size'] == 4800 and profile['batch_size'] == 240
    assert profile['local_optimizer_steps'] == 0 and profile['server_clip'] is None
    assert 'privacy/private_risk_message_safety.py' in stamp
    for kind in m['methods']:
        p = run.ledger.privacy(profile,run.previous.arm(kind))
        assert p['epsilon_realized'] <= 4 and p['delta'] == 1e-5
        assert abs(run.ledger.epsilon_at(p,120,profile,kind)-p['epsilon_realized']) < 1e-10
        assert p['risk_releases'] == (120 if kind.startswith('risk_') else 0)


def test_confirmation_requires_every_seed_and_both_controls():
    m,rows = run.config(),mock_rows()
    decision = run.decide(m,rows)
    assert decision['clean_confirmation_passed']
    assert not decision['joint_privacy_fairness_robustness_validated']
    assert not decision['comparisons']['risk_mean']['primary']
    # A high mean may not rescue one failed seed.
    rows[-1]['final']['test']['worst20_pct'] = 60.99
    assert not run.decide(m,rows)['clean_confirmation_passed']
    rows = mock_rows()
    rows[1]['final']['test']['accuracy_pct'] = 81.01
    assert not run.decide(m,rows)['clean_confirmation_passed']


def test_ci_is_paired_seed_interval_and_rejects_selective_grid():
    s = run.paired_summary([1.,2.,3.,4.],3.182446305284263)
    assert s['mean'] == 2.5 and math.isclose(s['sd'],math.sqrt(5/3))
    assert math.isclose(s['ci95'][0],2.5-3.182446305284263*math.sqrt(5/3)/2)
    with pytest.raises(ValueError):
        run.decide(run.config(),mock_rows()[:-1])
    rows = mock_rows()
    rows[-1] = copy.deepcopy(rows[-2])
    with pytest.raises(ValueError):
        run.decide(run.config(),rows)
    with pytest.raises(ValueError):
        run.paired_summary([1.,2.,3.,float('nan')],3.18)


def test_ci_and_mean_fairness_gates_are_additional():
    m,rows = run.config(),mock_rows()
    for r,d in zip([r for r in rows if r['job']['method']=='risk_rfa'],[-.99,-.99,-.99,.99]):
        r['final']['test']['accuracy_pct'] += d
    result = run.decide(m,rows)
    assert result['comparisons']['erm_mean']['gates']['all_seed_gates']
    assert not result['comparisons']['erm_mean']['gates']['accuracy_ci_lower_noninferior']
    rows = mock_rows()
    for r in rows:
        if r['job']['method']=='risk_rfa':
            r['final']['test']['variance_pp2'] = 51
    assert not run.decide(m,rows)['clean_confirmation_passed']


@pytest.mark.parametrize('method',['erm_mean','erm_rfa','risk_mean','risk_rfa'])
def test_valid_message_policy_does_not_change_candidate(method):
    run.require_mps()
    torch.mps.manual_seed(839)
    x = torch.randn(10,31,device='mps')*.02
    r = torch.linspace(0,1,10,device='mps') if method.startswith('risk_') else None
    before,diag = aggregate(x,r,kind=method,round_number=75,horizon=120)
    xx,rr,safety = sanitize(x,r)
    after,newdiag = aggregate(xx,rr,kind=method,round_number=75,horizon=120)
    assert torch.equal(before,after) and diag == newdiag
    assert safety['invalid_message_rows'] == 0


def test_saved_class_count_reconstruction_rejects_bad_metric():
    c = dict(N=10,class_count=[1]*10,class_hits=[1]*5+[0]*5,
             accuracy=.5,ce_loss=1.,brier_loss=.25,balanced_accuracy=.5)
    v = dict(accuracy_pct=50.,client_accuracy_pct=50.,worst20_pct=50.,
             gap_best20_worst20_pp=0.,gap_best_worst_pp=0.,variance_pp2=0.,
             balanced_accuracy_pct=50.,ce_loss=1.,brier_loss=.25,clients=[c]*10)
    assert run.audit_evaluation(v)
    v['worst20_pct'] = 51.
    with pytest.raises(AssertionError):
        run.audit_evaluation(v)
