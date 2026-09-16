"""Fixed-plan and decision tests; numerical MPS mechanism covered by v1/v2 suites."""
import ast
import copy
import math
from pathlib import Path
import pytest
from scripts import run_fair_objective_confirmation_v3 as run


def synthetic_rows(m, gains=(2., 2.1, 1.9, 2.), acc=(-.1, -.2, -.1, 0.)):
    rows = []
    for j in run.jobs(m):
        i = m['evaluation_seeds'].index(j['seed'])
        v = {k: 1. for k in run.METRICS}
        v.update(accuracy_pct=75., worst20_pct=50., variance_pp2=200., gap_best20_worst20_pp=30.)
        if j['arm'] == m['gate']['candidate']:
            v['accuracy_pct'] += acc[i]
            v['worst20_pct'] += gains[i]
        rows.append(dict(j, test=v))
    return rows


def test_frozen_plan_and_arms():
    m = run.config()
    assert len(run.jobs(m)) == 12
    assert m['arms'] == {'brier_fair_C1': dict(C=1., base_lr=2., beta=2.),
                         'brier_erm_C1': dict(C=1., base_lr=2., beta=0.),
                         'brier_erm_C2': dict(C=2., base_lr=2., beta=0.)}
    assert m['batch_size'] == 240 and m['test_checkpoint'] == m['rounds'] == 60
    assert m['evaluation_rounds'][-1] == m['rounds']
    assert m['gate']['required_seeds'] == 4
    assert m['gate']['minimum_mean_gain_pp'] == 1
    assert m['gate']['accuracy_ni_margin_pp'] == 1
    assert run.OUT not in {run.cal.OUT, run.base.OUT}


def test_isolation_restores_v2_even_on_exception(tmp_path, monkeypatch):
    previous = run.cal.OUT
    monkeypatch.setattr(run, 'OUT', tmp_path)
    with pytest.raises(RuntimeError):
        with run.isolated_trainer():
            assert run.cal.OUT == tmp_path
            raise RuntimeError('synthetic')
    assert run.cal.OUT == previous


def test_student_interval_matches_known_df3_result():
    r = run.paired_summary([1., 2., 3., 4.])
    assert r['mean'] == 2.5
    assert r['sample_sd'] == pytest.approx(math.sqrt(5 / 3))
    assert r['ci95_low'] == pytest.approx(.445739743, abs=1e-8)
    with pytest.raises(AssertionError):
        run.paired_summary([1., 2., 3.])
    with pytest.raises(AssertionError):
        run.paired_summary([1., 2., 3., float('nan')])


def test_complete_favorable_evidence_passes_both_controls():
    m = run.config()
    d = run.decision(m, synthetic_rows(m))
    assert d['passed'] and d['status'] == 'confirmed'
    assert set(d['contrasts']) == set(m['gate']['controls'])
    assert not d['automatic_followup']


@pytest.mark.parametrize('gains,acc', [((.5, .5, .5, .5), (0.,) * 4),
                                     ((-2., 6., 1., 1.), (0.,) * 4),
                                     ((2.,) * 4, (-1.,) * 4),
                                     ((2.,) * 4, (-2., -.5, .5, 1.))])
def test_failure_cannot_be_overridden_by_secondary_metrics(gains, acc):
    m = run.config()
    rows = synthetic_rows(m, gains=gains, acc=acc)
    for r in rows:
        if r['arm'] == m['gate']['candidate']:
            r['test']['variance_pp2'] = 0.
            r['test']['gap_best20_worst20_pp'] = 0.
    assert not run.decision(m, rows)['passed']


def test_winning_only_against_weaker_control_is_not_enough():
    m = run.config()
    rows = synthetic_rows(m)
    for r in rows:
        if r['arm'] == 'brier_erm_C2':
            r['test']['worst20_pct'] += 4.
    d = run.decision(m, rows)
    assert d['contrasts']['brier_erm_C1']['passed']
    assert not d['passed']


def test_incomplete_duplicate_or_foreign_results_cannot_decide():
    m = run.config()
    rows = synthetic_rows(m)
    with pytest.raises(ValueError):
        run.decision(m, rows[:-1])
    with pytest.raises(ValueError):
        run.decision(m, rows + [rows[0]])
    wrong = copy.deepcopy(rows)
    wrong[0]['seed'] = 170301
    with pytest.raises(ValueError):
        run.decision(m, wrong)


def test_holdout_guard_blocks_even_last_incomplete_training(monkeypatch):
    m = run.config()
    js = run.jobs(m)
    monkeypatch.setattr(run, 'training_complete', lambda j, stamp, conf: j != js[-1])
    with pytest.raises(RuntimeError, match='all 12'):
        run.require_all_training(m, {})
    monkeypatch.setattr(run, 'training_complete', lambda *args: True)
    run.require_all_training(m, {})


def test_only_test_evaluation_uses_final_guarded_checkpoint():
    tree = ast.parse(Path(run.__file__).read_text())
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'evaluate_final')
    assert isinstance(fn.body[0], ast.Expr) and fn.body[0].value.func.id == 'require_all_training'
    evaluations = [x for x in ast.walk(tree) if isinstance(x, ast.Call)
                   and isinstance(x.func, ast.Attribute) and x.func.attr == 'evaluate']
    assert len(evaluations) == 1 and evaluations[0].args[-1].value == 'test'
    assert 'test_checkpoint' in ast.unparse(fn)


def test_recomputed_fairness_metrics_reject_corruption():
    m = run.config()
    clients = [dict(N=1000, accuracy=.5, balanced_accuracy=.4, ce_loss=1., brier_loss=.3)
               for _ in range(10)]
    v = dict(clients=clients, accuracy_pct=50., client_accuracy_pct=50., worst20_pct=50.,
             gap_best20_worst20_pp=0., variance_pp2=0., balanced_accuracy_pct=40.,
             ce_loss=1., brier_loss=.3)
    run.validate_client_metrics(v, m)
    v['worst20_pct'] = 49.
    with pytest.raises(AssertionError):
        run.validate_client_metrics(v, m)


def test_public_steps_and_actual_noise_cost():
    m = run.config()
    plans = {}
    for j in run.jobs(m)[:3]:
        p = run.cal.parameters(m, j)
        plans[j['arm']] = (p, run.base.privacy_plan(dict(m, methods={'arm': p}), 'arm', 4.))
    fair, fp = plans['brier_fair_C1']
    classic, cp = plans['brier_erm_C1']
    assert fair['server_lr'] == pytest.approx(2 / 1.9)
    assert fp['std'] / cp['std'] == pytest.approx(4.)
    assert fp['std'] * fair['server_lr'] / (cp['std'] * classic['server_lr']) == pytest.approx(4 / 1.9)
    assert all(p['steps'] == 60 and p['epsilon'] <= 4 for _, p in plans.values())
