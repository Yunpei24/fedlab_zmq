"""Scalar protocol tests; mock counts are fixtures, not experimental results."""
import copy
from fractions import Fraction as Q
from pathlib import Path
import pytest
import yaml
from privacy import full_population_private_risk_attacks_v30 as v30
from privacy.full_population_private_risk_confirmation_v29 import ARMS, TCRIT


def test_grid_protocol_matches_fixed_threats_and_stricter_wave():
    root = Path(__file__).resolve().parents[1]
    c = yaml.safe_load((root/'configs/ldp_gradient_far/full_population_private_risk_attacks_v30.yaml').read_text())
    assert c['seeds'] == list(v30.SEEDS) and c['methods'] == list(v30.METHODS)
    assert c['conditions'] == list(v30.CONDITIONS)
    assert c['primary_endpoints'] == v30.ENDPOINTS
    assert c['honest_ids'] == list(v30.HONEST_IDS) and c['byzantine_ids'] == [0, 1]
    assert c['expected_runs'] == len(v30.jobs()) == 64
    assert c['clean_replays'] == sum(j['attack'] == 'none' for j in v30.jobs()) == 16
    assert c['attacked_runs'] == 48 and c['batch_size'] == c['public_train_size'] == 4800
    assert c['require_complete_independent_clean_pass'] and not c['automatic_launch']
    assert c['paired_t_critical_df3'] == TCRIT and c['one_sided_alpha'] == .0125
    assert c['confirmation_wave'] == 1 and c['test_rounds'] == [90, 120]
    assert c['minimum_worst20_gain_per_seed_pp'] == c['maximum_accuracy_loss_per_seed_pp'] == 1
    assert c['maximum_own_clean_accuracy_loss_pp'] == c['maximum_own_clean_worst20_loss_pp'] == 5
    assert c['maximum_solver_gap_diagnostic'] == .001 and c['solver_iterations'] == 40
    assert c['clip'] == 2 and c['risk_scale'] == .5 and c['server_clip'] is None
    assert c['attack_start'] == 31 and c['temporary_attack_end'] == 90
    assert c['bf_multiplier'] == 10 and c['alie_population_std_multiplier'] == 1.5
    assert c['ipm_ramp_end'] == 60 and c['ipm_final_multiplier'] == 2


def positive_clean_evidence():
    gates = {k: True for k in ('all_seed_gates', 'worst20_lower_positive', 'accuracy_lower_noninferior',
        'mean_gap_nonincreasing', 'mean_variance_nonincreasing')}
    audit = dict(audit_passed=True, gate_evaluated=True, expected_runs=24, valid_runs=24,
        manifest_sha256='fixture-hash', records=[dict(job=dict(seed=s, batch=b, method=m))
            for s in v30.SEEDS for b, m in ARMS],
        decision=dict(clean_confirmation_passed=True, primary_method='risk_rfa', primary_batch=4800,
            wave=1, one_sided_alpha=.0125, contrasts=[dict(control_batch=b, control_method=m,
                passed=True, gates=copy.deepcopy(gates)) for b in (4800, 240) for m in v30.CONTROLS]))
    status = dict(status='completed', device='mps', valid_runs=24)
    return audit, status


def test_clean_opening_rejects_partial_negative_running_or_different_manifest():
    a, s = positive_clean_evidence()
    v30.require_clean_pass(a, s, 'fixture-hash')
    for field, value in [('audit_passed', False), ('gate_evaluated', False), ('valid_runs', 23),
                         ('manifest_sha256', 'different')]:
        b = copy.deepcopy(a); b[field] = value
        with pytest.raises(RuntimeError): v30.require_clean_pass(b, s, 'fixture-hash')
    for field, value in [('clean_confirmation_passed', False), ('primary_method', 'risk_mean'),
                         ('primary_batch', 240), ('one_sided_alpha', .025), ('wave', 0)]:
        b = copy.deepcopy(a); b['decision'][field] = value
        with pytest.raises(RuntimeError): v30.require_clean_pass(b, s, 'fixture-hash')
    for field, value in [('status', 'running'), ('device', 'cpu'), ('valid_runs', 23)]:
        z = dict(s); z[field] = value
        with pytest.raises(RuntimeError): v30.require_clean_pass(a, z, 'fixture-hash')


def test_clean_opening_requires_all_arms_and_real_gate_names():
    a, s = positive_clean_evidence()
    for mutation in ('missing_arm', 'duplicate_arm', 'failed_gate', 'invented_gate', 'missing_control'):
        b = copy.deepcopy(a)
        if mutation == 'missing_arm': b['records'].pop()
        elif mutation == 'duplicate_arm': b['records'][-1] = b['records'][0]
        elif mutation == 'failed_gate': b['decision']['contrasts'][0]['gates']['all_seed_gates'] = False
        elif mutation == 'invented_gate': b['decision']['contrasts'][0]['gates'] = {'fake': True}
        else: b['decision']['contrasts'].pop()
        with pytest.raises(RuntimeError): v30.require_clean_pass(b, s, 'fixture-hash')


def test_counts(honest_hits, corrupted_hits=(0, 1000)):
    # This helper name is not a collected test (marked below).
    clients = []
    for hits in [*corrupted_hits, *honest_hits]:
        clients.append(dict(N=1000, class_count=[100]*10,
            class_hits=[min(100, max(0, hits-100*k)) for k in range(10)]))
    return dict(clients=clients)


test_counts.__test__ = False


def fixture_rows():
    records = []
    for j in v30.jobs():
        honest = [650, 650, *([850]*6)] if j['method'] == 'risk_rfa' else [600, 600, *([850]*6)]
        test = test_counts(honest)
        rounds = [dict(round=t, test=copy.deepcopy(test) if t in (90, 120) else None,
                       aggregation=dict(solver=dict(unsmoothed_objective_gap_upper=.00001)))
                  for t in range(1, 121)]
        records.append(dict(job=j, device='mps', rounds=rounds, test_evaluation_rounds=[90, 120]))
    return records


def record_for(records, method='risk_rfa', attack='abrupt_bf'):
    return next(r for r in records if r['job'] == dict(seed=v30.SEEDS[0], method=method, attack=attack))


def test_honest_metric_uses_eight_fixed_ids_exact_counts_population_variance():
    full = test_counts([600, 620, 700, 750, 800, 810, 900, 920])
    m = v30.exact_honest_test(full)
    a = [Q(x, 10) for x in [600, 620, 700, 750, 800, 810, 900, 920]]
    assert m['accuracy_pct'] == sum(a)/8 and m['worst20_pct'] == 61
    assert m['gap_best20_worst20_pp'] == 30
    assert m['variance_pp2'] == sum((x-sum(a)/8)**2 for x in a)/8
    assert all(isinstance(x, Q) for x in m.values())
    full['clients'][0]['class_hits'] = [100]*10
    full['clients'][1]['class_hits'] = [0]*10
    assert v30.exact_honest_test(full) == m
    full['clients'][3]['class_hits'][0] = 100.5
    with pytest.raises(ValueError): v30.exact_honest_test(full)


def test_complete_positive_fixture_and_no_joint_claim():
    decision = v30.decide(fixture_rows())
    assert decision['attack_confirmation_passed']
    assert not decision['joint_objective_validated'] and decision['independent_audit_required']
    assert decision['wave'] == 1 and decision['one_sided_alpha'] == .0125
    assert all(r['comparisons']['risk_mean']['primary'] is False for r in decision['attacks'].values())


@pytest.mark.parametrize('mutation', ['missing_run', 'duplicate_run', 'missing_round', 'missing_test90', 'cpu'])
def test_cannot_select_runs_or_endpoints(mutation):
    rows = fixture_rows()
    if mutation == 'missing_run': rows.pop()
    elif mutation == 'duplicate_run': rows[-1] = rows[0]
    elif mutation == 'missing_round': rows[0]['rounds'].pop(5)
    elif mutation == 'missing_test90': rows[0]['rounds'][89]['test'] = None
    else: rows[0]['device'] = 'cpu'
    with pytest.raises(ValueError): v30.decide(rows)


def test_recovery_cannot_mask_damage_at90():
    rows = fixture_rows()
    r = record_for(rows)
    r['rounds'][89]['test'] = test_counts([590, 590, *([850]*6)])
    d = v30.decide(rows)
    assert not d['attack_confirmation_passed'] and not d['attacks']['abrupt_bf']['passed']
    assert d['attacks']['persistent_alie']['passed']
    assert not all(c['passed'] for c in d['attacks']['abrupt_bf']['own_clean_comparisons'])


def test_own_clean_margins_checked_even_if_relative_controls_pass():
    rows = fixture_rows()
    r = record_for(rows, attack='none')
    r['rounds'][119]['test'] = test_counts([750, 750, *([950]*6)])
    d = v30.decide(rows)
    assert all(c['passed'] for c in d['attacks']['abrupt_bf']['comparisons'].values())
    assert not d['attack_confirmation_passed']
    assert not all(c['passed'] for c in d['attacks']['abrupt_bf']['own_clean_comparisons'])


def test_solver_gate_includes_candidate_and_rfa_control():
    rows = fixture_rows()
    record_for(rows)['rounds'][40]['aggregation']['solver']['unsmoothed_objective_gap_upper'] = .00101
    d = v30.decide(rows)
    assert not d['attacks']['abrupt_bf']['solver_gate'] and not d['attack_confirmation_passed']
    rows = fixture_rows()
    record_for(rows, method='erm_rfa')['rounds'][40]['aggregation']['solver']['unsmoothed_objective_gap_upper'] = .00101
    d = v30.decide(rows)
    assert all(r['passed'] for r in d['attacks'].values())
    assert not d['all_rfa_solver_gate'] and not d['attack_confirmation_passed']
    record_for(rows, method='erm_rfa')['rounds'][40]['aggregation']['solver']['unsmoothed_objective_gap_upper'] = float('nan')
    with pytest.raises(ValueError): v30.decide(rows)
