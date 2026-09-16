"""Independent audit decision tests; synthetic scalar records, no experimental tuning."""
import copy
import math
import pytest
from scripts.analyze_public_temporal_noise_confirmation_v25 import (
    interval, decide, compare_evidence, SEEDS, METHODS,
)
from scripts.run_public_temporal_noise_confirmation_v25 import decide as recorded_decide


def matrix():
    rows = []
    for seed in SEEDS:
        for mode in (0, 13):
            for method in METHODS:
                candidate = mode == 13 and method == 'risk_rfa'
                test = dict(accuracy_pct=80., worst20_pct=60.+2*int(candidate),
                    gap_best20_worst20_pp=20.-int(candidate), variance_pp2=50.-int(candidate))
                rows.append(dict(job=dict(seed=seed,grid_index=mode,method=method),
                                 final=dict(test=test,validation=dict(test))))
    return rows


def test_independent_student_interval():
    result = interval([1., 2., 3., 4.])
    expected_radius = 3.182446305284263*math.sqrt(5/3)/2
    assert result['mean'] == 2.5 and result['df'] == 3
    assert math.isclose(result['ci95'][0],2.5-expected_radius,abs_tol=1e-11)
    assert interval([2.]*4)['ci95'] == [2.,2.]
    for bad in ([1.,2.,3.], [1.,2.,3.,float('nan')], [1.,2.,3.,float('inf')]):
        with pytest.raises(ValueError):
            interval(bad)


def test_complete_matrix_only_and_exact_primary_candidate():
    rows = matrix()
    assert decide(rows)['clean_confirmation_passed']
    for bad in (rows[:-1], rows[:-1]+[rows[0]], rows+[rows[0]]):
        with pytest.raises(ValueError):
            decide(bad)
    # A strong constant risk arm cannot replace a failed scheduled candidate.
    for r in rows:
        if r['job']['method'] == 'risk_rfa' and r['job']['grid_index'] == 0:
            r['final']['test']['worst20_pct'] = 99.
        if r['job']['method'] == 'risk_rfa' and r['job']['grid_index'] == 13:
            r['final']['test']['worst20_pct'] = 60.
    assert not decide(rows)['clean_confirmation_passed']


def test_test_not_validation_and_all_comparators():
    rows = matrix()
    rows[0]['final']['validation']['worst20_pct'] = 100.
    assert decide(rows)['clean_confirmation_passed']
    rows[0]['final']['test']['worst20_pct'] = 61.5
    result = decide(rows)
    assert not result['clean_confirmation_passed']
    assert not result['contrasts'][0]['gates']['all_seed_gates']
    assert all(c['passed'] for c in result['contrasts'][1:])


@pytest.mark.parametrize('metric,value,gate',[
    ('worst20_pct',60.9,'all_seed_gates'),
    ('accuracy_pct',78.9,'all_seed_gates'),
    ('gap_best20_worst20_pp',25.,'mean_gap_nonincreasing'),
    ('variance_pp2',60.,'mean_variance_nonincreasing'),
])
def test_each_margin_can_close_confirmation(metric,value,gate):
    rows = matrix()
    selected = next(r for r in rows if r['job'] == dict(seed=SEEDS[0],grid_index=13,method='risk_rfa'))
    selected['final']['test'][metric] = value
    result = decide(rows)
    assert not result['clean_confirmation_passed']
    assert not result['contrasts'][0]['gates'][gate]


def test_uncertainty_can_fail_when_all_seed_margins_pass():
    rows = matrix()
    for r in rows:
        if r['job']['method'] == 'risk_rfa' and r['job']['grid_index'] == 13:
            i = SEEDS.index(r['job']['seed'])
            r['final']['test']['accuracy_pct'] += (-.99,-.99,.1,.1)[i]
    result = decide(rows)
    assert result['contrasts'][0]['gates']['all_seed_gates']
    assert not result['contrasts'][0]['gates']['accuracy_ci_lower_noninferior']
    assert not result['clean_confirmation_passed']


def test_independent_result_matches_recorded_algorithm_and_rejects_mutation():
    rows = matrix()
    config = dict(confirmation_seeds=list(SEEDS),grid_indices=[0,13],
                  primary_control_grid_indices=[0,13],primary_control_methods=['erm_mean','erm_rfa'])
    expected = recorded_decide(config, rows)
    actual = decide(rows)
    compare_evidence(actual, expected)
    altered = copy.deepcopy(expected); altered['clean_confirmation_passed'] = False
    with pytest.raises(AssertionError):
        compare_evidence(actual, altered)
    altered = copy.deepcopy(expected); altered['contrasts'][0]['summaries']['accuracy_pct']['mean'] += .01
    with pytest.raises(AssertionError):
        compare_evidence(actual, altered)
