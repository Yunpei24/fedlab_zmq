"""Independent audit tests: real MPS attack arithmetic and scalar count fixtures.

The fixtures deliberately simulate passing/failing criteria. They are not runs,
are never saved into campaign results and cannot open the V29/V30 prerequisites.
"""
import copy
from fractions import Fraction as Q
import json
import math
from pathlib import Path
import pytest
import torch
from scripts import audit_full_population_private_risk_attacks_v30 as audit
from privacy import full_population_private_risk_attacks_v30 as mechanism
from privacy.private_risk_attacks import inject
from privacy.fair_objective import require_mps


def _evaluation(honest_hits, corrupted=(0, 1000)):
    clients = []
    numbers = []
    for hits in [*corrupted, *honest_hits]:
        numbers.append(Q(hits, 10))
        clients.append(dict(N=1000, accuracy=hits/1000, class_count=[100]*10,
            class_hits=[min(100, max(0, hits-100*k)) for k in range(10)],
            ce_loss=.6, brier_loss=.15, balanced_accuracy=hits/1000))
    average = sum(numbers)/10; ss = sorted(numbers); worst = sum(ss[:2])/2
    return dict(clients=clients, accuracy_pct=float(average), client_accuracy_pct=float(average),
        worst20_pct=float(worst), gap_best20_worst20_pp=float(sum(ss[-2:])/2-worst),
        variance_pp2=float(sum((v-average)**2 for v in numbers)/10),
        ce_loss=.6, brier_loss=.15, balanced_accuracy_pct=float(average),
        gap_best_worst_pp=float(max(numbers)-min(numbers)))


def _records():
    records = []
    for j in audit.matrix():
        hits = [650, 650, *([850]*6)] if j['method'] == 'risk_rfa' else [600, 600, *([850]*6)]
        test = _evaluation(hits)
        records.append(dict(job=j, endpoints={str(t): dict(test=copy.deepcopy(test)) for t in (90, 120)},
            max_solver_gap=.00001 if j['method'].endswith('_rfa') else None))
    return records


def _runner_records(records):
    return [dict(job=r['job'], device='mps', test_evaluation_rounds=[90, 120],
        rounds=[dict(round=t, test=copy.deepcopy(r['endpoints'][str(t)]['test']) if t in (90, 120) else None,
            aggregation=dict(solver=dict(unsmoothed_objective_gap_upper=r['max_solver_gap'])))
            for t in range(1, 121)]) for r in records]


def _find(records, method='risk_rfa', attack='abrupt_bf'):
    return next(r for r in records if r['job'] == dict(seed=180701, method=method, attack=attack))


@pytest.mark.parametrize('attack', audit.ATTACKS)
def test_independent_attack_arithmetic_all_transition_rounds_mps(attack):
    require_mps()
    x = torch.arange(70, device='mps', dtype=torch.float32).reshape(10, 7)/100
    r = torch.linspace(0, .6, 10, device='mps'); old_x, old_r = x.clone(), r.clone()
    for t in (1, 30, 31, 45, 60, 90, 91, 120):
        expected, er = audit.independent_attack(x, r, attack, t)
        generated, gr, d = inject(x, r, attack=attack, round_number=t)
        torch.testing.assert_close(expected, generated, rtol=4e-6, atol=4e-7)
        assert torch.equal(er, gr) and d['active'] == audit.active_at(attack, t)
        assert torch.equal(expected[2:], x[2:])
        assert torch.equal(x, old_x) and torch.equal(r, old_r)


def test_alie_independent_variance_is_population_not_sample_mps():
    require_mps()
    x = torch.arange(30, device='mps', dtype=torch.float32).reshape(10, 3)
    y, _ = audit.independent_attack(x, None, 'persistent_alie', 90)
    mean = x[2:].mean(0)
    population = mean+1.5*x[2:].std(0, correction=0)
    sample = mean+1.5*x[2:].std(0, correction=1)
    torch.testing.assert_close(y[0], population)
    assert not torch.allclose(y[0], sample)


def test_exact_honest_metrics_and_corrupt_exclusion():
    v = _evaluation([600, 620, 700, 750, 800, 810, 900, 920])
    x = audit.honest_counts(v)
    assert x == mechanism.exact_honest_test(v)
    assert x['worst20_pct'] == 61 and x['accuracy_pct'] == Q(305, 4)
    changed = _evaluation([600, 620, 700, 750, 800, 810, 900, 920], corrupted=(1000, 0))
    assert audit.honest_counts(changed) == x
    bad = copy.deepcopy(v); bad['variance_pp2'] += .01
    with pytest.raises(ValueError): audit.honest_counts(bad)


@pytest.mark.parametrize('case', ['pass', 'early_damage', 'recovery_loss', 'persistent_failure',
    'own_clean_gap', 'solver_candidate', 'solver_control'])
def test_independent_decision_matches_generator_for_positive_and_negative_cases(case):
    records = _records()
    if case == 'early_damage':
        _find(records)['endpoints']['90']['test'] = _evaluation([590, 590, *([850]*6)])
    elif case == 'recovery_loss':
        _find(records, attack='slow_ipm')['endpoints']['120']['test'] = _evaluation([500, 500, *([700]*6)])
    elif case == 'persistent_failure':
        _find(records, attack='persistent_alie')['endpoints']['120']['test'] = _evaluation([590, 590, *([850]*6)])
    elif case == 'own_clean_gap':
        _find(records, attack='none')['endpoints']['120']['test'] = _evaluation([750, 750, *([950]*6)])
    elif case == 'solver_candidate': _find(records)['max_solver_gap'] = .00101
    elif case == 'solver_control': _find(records, method='erm_rfa')['max_solver_gap'] = .00101
    independent = audit.independent_decision(records)
    generated = mechanism.decide(_runner_records(records))
    audit.compare_structures(independent, generated)
    assert independent['attack_confirmation_passed'] is (case == 'pass')
    assert not independent['joint_objective_validated']


def test_incomplete_or_duplicate_endpoint_grid_never_gets_a_gate():
    with pytest.raises(ValueError): audit.independent_decision(_records()[:-1])
    duplicate = _records(); duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError): audit.independent_decision(duplicate)
    missing = _records(); missing[0]['endpoints'].pop('90')
    with pytest.raises(ValueError): audit.independent_decision(missing)


def test_model_chain_detects_reset_before_recovery():
    initial = dict(weight='initial')
    rows = [dict(pre_model_sha256=initial, model_sha256=dict(weight='damaged')),
            dict(pre_model_sha256=dict(weight='damaged'), model_sha256=dict(weight='recovering'))]
    audit.verify_model_chain(rows, initial)
    rows[1]['pre_model_sha256'] = initial
    with pytest.raises(AssertionError, match='reset'): audit.verify_model_chain(rows, initial)


def test_privacy_audit_composes_both_channels_from_actual_calibration_plan():
    root = Path(__file__).resolve().parents[1]
    path = root/'results/ldp_gradient_far/full_population_private_risk_calibration_v28/seed170501__risk_rfa/metrics.json'
    p = json.loads(path.read_text())['privacy']
    j = dict(seed=180701, method='risk_rfa', batch=4800)
    prefixes = audit.clean_audit.privacy_check(p, j)
    assert len(prefixes) == 120 and 3.99999 <= prefixes[-1][0] <= 4
    wrong = copy.deepcopy(p); wrong['gradient_std'] *= .5
    with pytest.raises(AssertionError): audit.clean_audit.privacy_check(wrong, j)
    wrong = copy.deepcopy(p); wrong['rdp']['7'] -= 120*7/(2*p['risk_z']**2)
    with pytest.raises(AssertionError): audit.clean_audit.privacy_check(wrong, j)


def test_confidence_level_is_not_the_old_95_percent_interval():
    values = [Q(1), Q(2), Q(3), Q(4)]
    r = audit.clean_audit.interval(values)
    sd = math.sqrt(5/3)
    assert r['lower_one_sided_9875'] < 2.5-3.182446305*sd/2
    assert math.isclose(r['lower_one_sided_9875'], 2.5-4.176534846104499*sd/2, abs_tol=1e-12)


def test_unlaunched_audit_does_not_create_outputs(tmp_path, monkeypatch):
    require_mps()
    monkeypatch.setattr(audit, 'OUT', tmp_path/'not-launched')
    monkeypatch.setattr(audit, 'CACHE', tmp_path/'must-not-be-created')
    with pytest.raises(RuntimeError, match='not launched'): audit.main(partial=True)
    assert list(tmp_path.iterdir()) == []
