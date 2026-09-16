"""Scalar result-analysis checks; no tensor work or new DP release."""
from copy import deepcopy
from fractions import Fraction
import math
import pytest

from scripts.diagnose_v29_completed_confirmation import exact_view, summary, decompose, SEEDS


def view():
    clients = [dict(N=1000, class_count=[1000]+[0]*9,
                    class_hits=[100*i]+[0]*9) for i in range(10)]
    return dict(clients=clients, accuracy_pct=45., worst20_pct=5.,
                gap_best20_worst20_pp=80., variance_pp2=825.,
                balanced_accuracy_pct=45., ce_loss=1., brier_loss=.2)


def test_exact_population_not_sample_variance():
    metrics, recalls, counts = exact_view(view())
    assert metrics['variance_pp2'] == Fraction(825)
    assert recalls == [Fraction(45)] + [None]*9
    assert counts == [10000] + [0]*9


@pytest.mark.parametrize('mutation', ['counts', 'hits', 'nan', 'metric', 'metric_nan'])
def test_invalid_counts_rejected(mutation):
    v = deepcopy(view())
    if mutation == 'counts':
        v['clients'][0]['class_count'][0] = 999
    elif mutation == 'hits':
        v['clients'][0]['class_hits'][0] = 1001
    elif mutation == 'nan':
        v['clients'][0]['class_hits'][0] = float('nan')
    elif mutation == 'metric_nan':
        v['worst20_pct'] = float('nan')
    else:
        v['worst20_pct'] = 6
    with pytest.raises(ValueError):
        exact_view(v)


def test_sample_sd_uses_seeds_not_clients():
    s = summary([1, 2, 3, 4])
    assert s['mean'] == 2.5
    assert s['sd'] == math.sqrt(5/3)
    with pytest.raises(ValueError):
        summary(range(10))


def test_telescoping_is_exact_not_a_new_gate():
    index = {}
    for seed in SEEDS:
        for arm, val in [((4800, 'risk_rfa'), Fraction(6045, 100)),
                         ((4800, 'erm_mean'), Fraction(574, 10)),
                         ((240, 'erm_mean'), Fraction(6015, 100))]:
            index[(seed, *arm)] = dict(values={'worst20_pct': val})
    rows = decompose(index, 'erm_mean')
    assert len(rows) == 4
    assert all(r['within_full_batch'] == 3.05 and r['erm_batch_change'] == -2.75
               and r['total'] == .3 for r in rows)
