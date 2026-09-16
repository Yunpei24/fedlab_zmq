"""Runner safeguards; fixtures do not simulate a successful real confirmation."""
import copy
from pathlib import Path
import pytest
import torch
from scripts import run_full_population_private_risk_attacks_v30 as run
from privacy.fair_objective import require_mps


def test_exact_protocol_and_job_mapping():
    c = run.config()
    assert c['expected_runs'] == 64 and not c['automatic_launch']
    j = dict(seed=180701, method='risk_rfa', attack='slow_ipm')
    assert run.parent_job(j) == dict(seed=180701, batch=4800, method='risk_rfa')
    assert run.clean_job(j) == dict(seed=180701, method='risk_rfa', attack='none')
    assert run.identifier(j) == 'seed180701__risk_rfa__slow_ipm'
    jobs = run.jobs()
    assert [j['attack'] for j in jobs[:4]] == ['none']*4
    assert len({run.identifier(j) for j in jobs}) == 64


def test_missing_parent_audit_creates_no_campaign_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'PARENT_AUDIT', tmp_path/'missing-audit.json')
    monkeypatch.setattr(run, 'OUT', tmp_path/'must-not-be-created')
    with pytest.raises(RuntimeError, match='final V29 audit'):
        run.inputs()
    assert not run.OUT.exists() and list(tmp_path.iterdir()) == []


def test_model_hash_is_exact_and_covers_mps_model_state():
    require_mps()
    model = torch.nn.Linear(2, 1).to('mps')
    before = run.model_hash(model)
    assert set(before) == set(model.state_dict())
    assert before == run.model_hash(model)
    with torch.no_grad(): model.bias.add_(1.)
    after = run.model_hash(model)
    assert before['weight'] == after['weight'] and before['bias'] != after['bias']


def prefix_fixture():
    row = dict(epsilon_realized=.4, epsilon_order=8, aggregation={'eta': 2.},
        validation={'accuracy': .5}, private_messages_sha256='messages',
        private_reports_sha256='reports', step_sha256='step',
        model_sha256={'weight': 'new'}, pre_model_sha256={'weight': 'old'},
        attack={'attack': 'none', 'active': False})
    return row, [dict(client=i, indices_sha256=f'batch{i}', private_risk=.2) for i in range(10)]


def test_inactive_attack_label_can_differ_but_not_actual_computation():
    row, local = prefix_fixture()
    compared = copy.deepcopy(row)
    compared['attack']['attack'] = 'slow_ipm'
    run.assert_same_prefix(row, local, compared, copy.deepcopy(local))


@pytest.mark.parametrize('field', ['epsilon_realized', 'epsilon_order', 'aggregation', 'validation',
    'private_messages_sha256', 'private_reports_sha256', 'step_sha256', 'model_sha256', 'pre_model_sha256'])
def test_any_changed_prefix_computation_is_rejected(field):
    row, local = prefix_fixture()
    compared = copy.deepcopy(row); compared[field] = 'changed'
    with pytest.raises(AssertionError, match='replay differs'):
        run.assert_same_prefix(row, local, compared, local)


def test_changed_local_query_is_rejected_even_with_unchanged_aggregate():
    row, local = prefix_fixture(); changed = copy.deepcopy(local)
    changed[4]['indices_sha256'] = 'different-permutation'
    with pytest.raises(AssertionError, match='replay differs'):
        run.assert_same_prefix(row, local, row, changed)


def test_honest_metrics_exclude_corrupted_ids_and_use_count_based_test():
    clients = []
    for hits in [0, 1000, 600, 620, 700, 750, 800, 810, 900, 920]:
        clients.append(dict(N=1000, accuracy=hits/1000, class_count=[100]*10,
            class_hits=[min(100, max(0, hits-100*k)) for k in range(10)],
            ce_loss=.6, brier_loss=.15, balanced_accuracy=hits/1000))
    h = run.honest_metrics(dict(clients=clients))
    assert h['honest_ids'] == list(range(2, 10))
    assert h['worst20_client_count'] == 2 and h['worst20_pct'] == 61.
    assert h['gap_best20_worst20_pp'] == 30.
    assert h['accuracy_pct'] == h['client_accuracy_pct'] == 76.25
    assert len(h['clients']) == 8 and h['ce_loss'] == .6 and h['brier_loss'] == .15
    with pytest.raises(ValueError): run.honest_metrics(dict(clients=clients[:-1]))


def test_prepared_runner_is_not_a_dispatcher():
    # Neither import nor config validation starts a subprocess or touches outputs.
    source = Path(run.__file__).read_text()
    assert "parser.add_mutually_exclusive_group(required=True)" in source
    assert "group.add_argument('--resume'" in source
    assert "group.add_argument('--validate-only'" in source
    assert 'AUDITOR.is_file()' in source and 'AUDITOR_TEST.is_file()' in source
    assert 'require_clean_pass(audit, state' in source
