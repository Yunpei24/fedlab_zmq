#!/usr/bin/env python3
"""Frozen 12-run clean confirmation. Reuse, never modify, the v2 MPS trainer."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import secrets
import statistics as st
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
import torch
import yaml
from scripts import run_fair_objective_calibration_v2 as cal
from privacy.fair_objective import require_mps

base = cal.base
NAME = 'fair_objective_confirmation_v3'
MATRIX = ROOT / 'configs/ldp_gradient_far' / f'{NAME}.yaml'
OUT = ROOT / 'results/ldp_gradient_far' / NAME
LOG = ROOT / 'logs' / f'{NAME}.log'
REPORT = ROOT / 'output/analysis/Fair_Objective_Confirmation_V3_Status.md'
TESTS = ['tests/test_fair_objective.py', 'tests/test_fair_objective_calibration_v2.py',
         'tests/test_fair_objective_confirmation_v3.py']
T95_DF3 = 3.182446305284263
METRICS = ['accuracy_pct', 'worst20_pct', 'gap_best20_worst20_pp', 'variance_pp2',
           'balanced_accuracy_pct', 'ce_loss', 'brier_loss']


def config():
    m = yaml.safe_load(MATRIX.read_text())
    assert m['campaign_id'] == NAME and m['device'] == 'mps'
    assert m['rounds'] == m['test_checkpoint'] == 60
    assert m['expected_runs'] == 12 and m['num_clients'] == 10
    assert m['local_optimizer_steps'] == 0 and m['release_count_per_round'] == 1
    assert m['sampling'] == 'fixed_without_replacement' and m['adjacency'] == 'replace_one'
    assert m['server_aggregation'] == 'uniform' and m['server_clip'] is None
    assert m['attacks'] == 'none' and not m['automatic_followup']
    assert m['test_only_after_all_training']
    assert m['evaluation_seeds'] == [170401, 170402, 170403, 170404]
    assert set(m['evaluation_seeds']).isdisjoint({170101, 170201, 170202, 170203, 170204, 170301, 170302})
    assert m['epsilon'] == 4 and m['delta'] == 1e-5
    return m


def jobs(m):
    js = [dict(seed=seed, arm=arm, **params)
          for seed in m['evaluation_seeds'] for arm, params in m['arms'].items()]
    assert len(js) == len({cal.identifier(j) for j in js}) == m['expected_runs']
    return js


def source_stamp(m):
    paths = [Path(__file__), MATRIX, ROOT / m['protocol'], Path(cal.__file__),
             Path(base.__file__), ROOT / 'privacy/fair_objective.py',
             ROOT / 'models/registry.py', ROOT / 'datasets/registry.py',
             ROOT / 'datasets/partitioner.py', *[ROOT / x for x in TESTS]]
    return {str(p.relative_to(ROOT)): base.digest(p) for p in paths}


@contextmanager
def isolated_trainer():
    """Only the single-process trainer's output destination changes temporarily."""
    previous = cal.OUT
    try:
        cal.OUT = OUT
        yield
    finally:
        cal.OUT = previous


def training_complete(j, stamp, m):
    with isolated_trainer():
        if not cal.completed(j, stamp, m):
            return False
    d = OUT / cal.identifier(j)
    r = json.loads((d / 'metrics.json').read_text())
    assert r['parameters'] == cal.parameters(m, j)
    assert [x['round'] for x in r['rounds']] == list(range(1, 61))
    assert all(x['device'] == 'mps' for x in r['rounds'])
    assert r['final']['validation'] is not None and not r['test_evaluated']
    assert r['privacy']['steps'] == 60 and r['privacy']['delta'] == m['delta']
    assert all(math.isfinite(x['epsilon_at_round']) and x['epsilon_at_round'] <= m['epsilon']
               for x in r['rounds'])
    assert (d / 'checkpoint.pt').is_file()
    return True


def require_all_training(m, stamp):
    if not all(training_complete(j, stamp, m) for j in jobs(m)):
        raise RuntimeError('Test evaluation forbidden until all 12 trainings are complete')


def validate_client_metrics(v, m):
    """Independent scalar reconstruction; no fitting or model execution on CPU."""
    clients = v['clients']
    assert len(clients) == m['num_clients']
    assert all(c['N'] == 1000 for c in clients)
    a = [100 * c['accuracy'] for c in clients]
    assert all(0 <= x <= 100 and math.isfinite(x) for x in a)
    sorted_a = sorted(a)
    expected = dict(accuracy_pct=st.mean(a), client_accuracy_pct=st.mean(a),
                    worst20_pct=st.mean(sorted_a[:2]),
                    gap_best20_worst20_pp=st.mean(sorted_a[-2:]) - st.mean(sorted_a[:2]),
                    variance_pp2=st.pvariance(a),
                    balanced_accuracy_pct=100 * st.mean(c['balanced_accuracy'] for c in clients),
                    ce_loss=st.mean(c['ce_loss'] for c in clients),
                    brier_loss=st.mean(c['brier_loss'] for c in clients))
    for name, value in expected.items():
        assert math.isfinite(v[name]) and math.isclose(v[name], value, abs_tol=1e-8, rel_tol=1e-8), name


def test_complete(j, stamp, m):
    d = OUT / cal.identifier(j)
    status = d / 'evaluation_status.json'
    if not status.exists():
        return False
    s = json.loads(status.read_text())
    if s['status'] != 'completed':
        return False
    r = json.loads((d / 'test_metrics.json').read_text())
    assert s['metrics_sha256'] == base.digest(d / 'test_metrics.json')
    assert r['source_stamp'] == stamp and r['job'] == j and r['device'] == 'mps'
    assert r['checkpoint_round'] == m['test_checkpoint']
    assert r['training_metrics_sha256'] == base.digest(d / 'metrics.json')
    assert r['checkpoint_sha256'] == base.digest(d / 'checkpoint.pt')
    assert r['all_training_completed_before_evaluation']
    validate_client_metrics(r['test'], m)
    return True


def evaluate_final(m, j, data, stamp):
    require_all_training(m, stamp)
    base.verify_stamp(stamp)
    require_mps()
    if test_complete(j, stamp, m):
        return
    d = OUT / cal.identifier(j)
    r = json.loads((d / 'metrics.json').read_text())
    assert r['splits'] == data['splits']
    cp = torch.load(d / 'checkpoint.pt', map_location='cpu', weights_only=True)
    assert cp['job'] == j and cp['source_stamp'] == stamp and cp['round'] == m['test_checkpoint']
    model = base.new_model(m, j['seed'])
    model.load_state_dict(cp['model'])
    assert all(p.device.type == 'mps' for p in model.parameters())
    v = cal.add_objectives(base.evaluate(model, data, 'test'), j['beta'])
    validate_client_metrics(v, m)
    base.save(d / 'test_metrics.json', dict(job=j, device='mps', test=v,
        checkpoint_round=cp['round'], source_stamp=stamp,
        training_metrics_sha256=base.digest(d / 'metrics.json'),
        checkpoint_sha256=base.digest(d / 'checkpoint.pt'),
        all_training_completed_before_evaluation=True, evaluated_at=time.time()))
    base.save(d / 'evaluation_status.json', dict(status='completed', device='mps',
        metrics_sha256=base.digest(d / 'test_metrics.json')))


def paired_summary(values):
    assert len(values) == 4 and all(math.isfinite(v) for v in values)
    mean = st.mean(values)
    sd = st.stdev(values)
    half = T95_DF3 * sd / 2
    return dict(n=4, values=values, mean=mean, sample_sd=sd,
                ci95_low=mean - half, ci95_high=mean + half)


def decision(m, rows):
    """Fail closed on incomplete, repeated or unexpected arm/seed observations."""
    expected = {(j['seed'], j['arm']) for j in jobs(m)}
    keys = [(r['seed'], r['arm']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Complete unique 12-run evidence required before decision')
    by_key = dict(zip(keys, rows))
    gate = m['gate']
    contrasts = {}
    for ctrl in gate['controls']:
        diff = {}
        for metric in METRICS:
            diff[metric] = paired_summary([
                by_key[(s, gate['candidate'])]['test'][metric] - by_key[(s, ctrl)]['test'][metric]
                for s in m['evaluation_seeds']])
        primary = diff[gate['primary']]
        checks = dict(
            worst20_mean_gain=primary['mean'] >= gate['minimum_mean_gain_pp'],
            worst20_positive_lower_ci=primary['ci95_low'] > gate['primary_ci_lower_strictly_above'],
            accuracy_noninferior=diff['accuracy_pct']['ci95_low'] > -gate['accuracy_ni_margin_pp'])
        contrasts[ctrl] = dict(differences=diff, checks=checks, passed=all(checks.values()))
    passed = all(x['passed'] for x in contrasts.values())
    return dict(status='confirmed' if passed else 'not_confirmed', passed=passed,
                criteria=gate, seed_order=m['evaluation_seeds'], contrasts=contrasts,
                intervals='marginal paired Student two-sided 95%; four seeds; not simultaneous intervals',
                automatic_followup=False)


def audit_pairing(m, stamp):
    require_all_training(m, stamp)
    findings = []
    for seed in m['evaluation_seeds']:
        records = []
        for j in [x for x in jobs(m) if x['seed'] == seed]:
            d = OUT / cal.identifier(j)
            r = json.loads((d / 'metrics.json').read_text())
            o = json.loads((d / 'simulator_oracle.json').read_text())
            assert not o['privacy_protected'] and not o['feeds_mechanism']
            assert len(o['rounds']) == m['rounds']
            batch = [[c['batch_hash'] for c in t['clients']] for t in o['rounds']]
            assert all(len(t) == m['num_clients'] for t in batch)
            records.append((r['splits'], r['initial']['clients'], batch))
        assert all(x == records[0] for x in records[1:]), f'Pairing mismatch seed {seed}'
        findings.append(dict(seed=seed, splits_equal=True, initial_metrics_equal=True,
            all_600_batch_hashes_equal=True,
            gaussian_pairing='same campaign key and seed/round/client label; code provenance, not exported noise'))
    return findings


def report(m, stamp, evidence=None):
    trained = sum(training_complete(j, stamp, m) for j in jobs(m))
    tested = sum(test_complete(j, stamp, m) for j in jobs(m))
    lines = ['# Confirmation Brier équitable v3', '',
             f'Entraînements valides : **{trained}/12**. Tests finaux validés : **{tested}/12**.', '',
             'MPS uniquement ; Fashion-MNIST ; 10 clients ; 60 tours ; batch 240 ; epsilon ≤ 4.', '',
             'Trois configurations sur les seeds 170401, 170402, 170403 et 170404. '
             'Aucune attaque, aucune suite automatique. Le test est évalué seulement après les douze entraînements.', '',
             '[Protocole et critères fixés avant lancement](Fair_Objective_Confirmation_V3_Protocol.md)', '']
    if evidence is None:
        lines += ['**Aucun verdict avant la fin des douze évaluations.** Worst-20 est le critère principal ; '
                  'l’accuracy est le garde-fou. Les statistiques secondaires ne changent pas cette règle.']
    else:
        gate = evidence['decision']
        verdict = 'confirmé selon le critère préenregistré' if gate['passed'] else 'non confirmé selon le critère préenregistré'
        lines += [f'## Verdict : {verdict}', '',
                  'Moyenne ± écart-type échantillonnal sur les quatre seeds ; évaluation test au tour 60.', '',
                  '| Configuration | Accuracy (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) | Balanced acc. (%) | Loss CE | Loss Brier |',
                  '|:--|--:|--:|--:|--:|--:|--:|--:|']
        def fmt(seq):
            return f'{st.mean(seq):.3f} ± {st.stdev(seq):.3f}'
        for arm in m['arms']:
            vs = [r['test'] for r in evidence['rows'] if r['arm'] == arm]
            lines.append('| ' + arm + ' | ' + ' | '.join(fmt([r[k] for r in vs]) for k in METRICS) + ' |')
        lines += ['', '## Différences appariées : candidat moins contrôle', '',
                  '| Contrôle | Delta Worst-20, IC95 (pp) | Delta accuracy, IC95 (pp) | Gain ≥ 1 pp | IC Worst-20 > 0 | Non-infériorité accuracy |',
                  '|:--|:--|:--|:--|:--|:--|']
        def ci(x):
            return f"{x['mean']:+.3f} [{x['ci95_low']:+.3f} ; {x['ci95_high']:+.3f}]"
        for ctrl, c in gate['contrasts'].items():
            checks = ['oui' if v else 'non' for v in c['checks'].values()]
            lines.append('| ' + ctrl + ' | ' + ci(c['differences']['worst20_pct']) + ' | ' +
                         ci(c['differences']['accuracy_pct']) + ' | ' + ' | '.join(checks) + ' |')
        lines += ['', '## Résultats par seed', '',
                  '| Seed | Configuration | Accuracy (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) | Balanced acc. (%) | Loss CE | Loss Brier |',
                  '|--:|:--|--:|--:|--:|--:|--:|--:|--:|']
        for row in evidence['rows']:
            lines.append(f"| {row['seed']} | {row['arm']} | " + ' | '.join(f"{row['test'][k]:.4f}" for k in METRICS) + ' |')
        lines += ['', '## Portée', '',
                  'Les résultats portent sur ce benchmark, cette configuration et quatre seeds. '
                  'Les IC Student sont marginaux et dépendent d’une approximation inter-seeds ; '
                  'les clients et les tours ne sont pas comptés comme répétitions indépendantes. '
                  'Un résultat non confirmé n’est pas une impossibilité générale de la local-DP. '
                  'Aucune conclusion de robustesse byzantine ne découle d’une expérience sans attaque.', '',
                  '[Evidence complète : différences, critères et audits](../../results/ldp_gradient_far/fair_objective_confirmation_v3/evidence.json)']
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(lines) + '\n')


@contextmanager
def lock():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'campaign.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Campaign already active; no duplicate')
        yield


def worker(m):
    require_mps()
    with lock():
        try:
            stamp = source_stamp(m)
            manifest = OUT / 'manifest.json'
            if manifest.exists():
                saved = json.loads(manifest.read_text())
                assert saved['source_stamp'] == stamp and saved['config'] == m
            else:
                base.save(manifest, dict(config=m, source_stamp=stamp, device='mps',
                    torch_version=torch.__version__, created_at=time.time()))
            base.save(OUT / 'status.json', dict(status='preflight', pid=os.getpid(), device='mps', planned_runs=12))
            report(m, stamp)
            tests = OUT / 'tests.json'
            if not tests.exists():
                proc = subprocess.run([sys.executable, '-m', 'pytest', *TESTS, '-q'],
                    cwd=ROOT, capture_output=True, text=True)
                base.save(tests, dict(passed=proc.returncode == 0, output=proc.stdout + proc.stderr,
                    source_stamp=stamp, device='mps'))
            saved = json.loads(tests.read_text())
            assert saved['passed'] and saved['source_stamp'] == stamp, 'Preflight tests failed or changed'
            base.verify_stamp(stamp)
            print('Preflight passed; 12 runs MPS, fixed final-checkpoint test after all training.', flush=True)
            secret = OUT / 'simulator_secret.json'
            if not secret.exists():
                base.save(secret, dict(key=secrets.token_hex(32), not_public=True))
            key = json.loads(secret.read_text())['key']
            for seed in m['evaluation_seeds']:
                js = [j for j in jobs(m) if j['seed'] == seed]
                if all(training_complete(j, stamp, m) for j in js):
                    continue
                data = base.prepare(m, seed)
                for j in js:
                    with isolated_trainer():
                        cal.train(m, j, data, key, stamp)
                    assert training_complete(j, stamp, m)
                    report(m, stamp)
                del data
                torch.mps.empty_cache()
            pair_audit = audit_pairing(m, stamp)
            base.save(OUT / 'training_pairing_audit.json', dict(source_stamp=stamp, seeds=pair_audit))
            for seed in m['evaluation_seeds']:
                js = [j for j in jobs(m) if j['seed'] == seed]
                if all(test_complete(j, stamp, m) for j in js):
                    continue
                data = base.prepare(m, seed)
                for j in js:
                    base.save(OUT / 'status.json', dict(status='evaluating_final_test',
                        pid=os.getpid(), device='mps', active=cal.identifier(j), valid_training_runs=12))
                    evaluate_final(m, j, data, stamp)
                    report(m, stamp)
                del data
                torch.mps.empty_cache()
            rows = []
            for j in jobs(m):
                assert test_complete(j, stamp, m)
                d = OUT / cal.identifier(j)
                r = json.loads((d / 'test_metrics.json').read_text())
                rows.append(dict(j, test=r['test'], privacy=json.loads((d / 'metrics.json').read_text())['privacy'],
                    training_sha256=base.digest(d / 'metrics.json'), test_sha256=base.digest(d / 'test_metrics.json')))
            base.verify_stamp(stamp)
            evidence = dict(source_stamp=stamp, pairing_audit=pair_audit, rows=rows, decision=decision(m, rows))
            base.save(OUT / 'evidence.json', evidence)
            report(m, stamp, evidence)
            base.save(OUT / 'status.json', dict(status='completed', valid_training_runs=12,
                valid_test_runs=12, device='mps', gate=evidence['decision']['status'], no_followup_launched=True))
            print('Completed: ' + evidence['decision']['status'] + '; no follow-up launched.', flush=True)
        except Exception as exc:
            failure = dict(status='failed', error=repr(exc), pid=os.getpid(), time=time.time())
            base.save(OUT / 'failure.json', failure)
            base.save(OUT / 'status.json', failure)
            raise


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    for flag in ['launch', 'worker', 'plan', 'status']:
        group.add_argument('--' + flag, action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    m = config()
    if a.plan:
        print(json.dumps(dict(config=m, jobs=jobs(m)), indent=2))
        return
    if a.status:
        status = OUT / 'status.json'
        print(status.read_text() if status.exists() else 'not started')
        if (OUT / 'manifest.json').exists():
            stamp = json.loads((OUT / 'manifest.json').read_text())['source_stamp']
            base.verify_stamp(stamp)
            print(json.dumps(dict(valid_training=sum(training_complete(j, stamp, m) for j in jobs(m)),
                valid_test=sum(test_complete(j, stamp, m) for j in jobs(m)), planned=12)))
        return
    if not a.resume:
        p.error('--resume required')
    if a.worker:
        worker(m)
        return
    require_mps()
    with lock():
        pass
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a') as f:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', '--resume'],
            cwd=ROOT, env=dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='0'),
            stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(pid=proc.pid, log=str(LOG), device='mps', planned_runs=12)))


if __name__ == '__main__':
    main()
