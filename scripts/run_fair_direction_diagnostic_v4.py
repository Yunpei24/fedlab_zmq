#!/usr/bin/env python3
"""Matched one-step diagnostic on frozen calibration states; never a DP export."""
import argparse
from contextlib import contextmanager
import fcntl
import json
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
from privacy.fair_objective import per_example, release, require_mps
from metrics.fair_direction_audit import (flat_parameters, set_parameters, mean_brier_gradient,
    clip_rows, directional_metrics, actual_metrics)

base = cal.base
NAME = 'fair_direction_diagnostic_v4'
MATRIX = ROOT/'configs/ldp_gradient_far'/f'{NAME}.yaml'
OUT = ROOT/'results/ldp_gradient_far'/NAME
LOG = ROOT/'logs'/f'{NAME}.log'
REPORT = ROOT/'output/analysis/Fair_Direction_Diagnostic_V4_Status.md'
TESTS = ['tests/test_fair_objective.py', 'tests/test_fair_direction_diagnostic_v4.py']


def config():
    m = yaml.safe_load(MATRIX.read_text())
    assert m['device'] == 'mps' and m['campaign_id'] == NAME
    assert m['oracle_only'] and not m['use_test'] and not m['automatic_training']
    assert m['calibration_seeds'] == [170301, 170302]
    assert m['expected_states'] == 4 and m['expected_replay_blocks'] == 32
    assert m['expected_one_step_evaluations'] == 320 and len(m['conditions']) == 10
    return m


def states(m):
    return [dict(seed=s, C=c, beta=m['source_beta'], base_lr=m['source_base_lr'])
            for s in m['calibration_seeds'] for c in m['source_clips']]


def source_stamp(m):
    paths = [Path(__file__), MATRIX, ROOT/m['protocol'], ROOT/'metrics/fair_direction_audit.py',
             Path(cal.__file__), Path(base.__file__), ROOT/'privacy/fair_objective.py',
             ROOT/'models/registry.py', ROOT/'datasets/registry.py', ROOT/'datasets/partitioner.py',
             *[ROOT/p for p in TESTS]]
    return {str(p.relative_to(ROOT)): base.digest(p) for p in paths}


def verify_sources(m):
    old = json.loads((cal.OUT/'manifest.json').read_text())
    base.verify_stamp(old['source_stamp'])
    source = {}
    for j in states(m):
        assert cal.completed(j, old['source_stamp'], old['config'])
        d = cal.OUT/cal.identifier(j)
        source[cal.identifier(j)] = {name: base.digest(d/name) for name in ['checkpoint.pt', 'metrics.json']}
    return old, source


def block_path(j, rep):
    return OUT/cal.identifier(j)/f'replay_{rep:02d}.json'


def completed(j, rep, stamp, source):
    p = block_path(j, rep)
    if not p.exists():
        return False
    r = json.loads(p.read_text())
    assert r['status'] == 'completed' and r['device'] == 'mps'
    assert r['source_stamp'] == stamp and r['source'] == source[cal.identifier(j)]
    assert r['job'] == j and r['replay'] == rep
    assert len(r['conditions']) == 10 and r['all_conditions_start_at_identical_parameters']
    return True


def matched_aggregates(risks, raws, gaussian, plans):
    """All argument tensors and aggregates stay on MPS; same Gaussian per arm."""
    beta = 2.
    aggregates = {}
    coefficients = [1+beta*r.mean() for r in risks]
    means = torch.stack([g.mean(0) for g in raws])
    aggregates['raw_erm'] = means.mean(0)
    aggregates['raw_fair'] = torch.stack([a*g for a, g in zip(coefficients, means)]).mean(0)
    for C in (1, 2):
        clipped = torch.stack([clip_rows(g, C).mean(0) for g in raws])
        fair = torch.stack([a*g for a, g in zip(coefficients, clipped)])
        aggregates[f'clip{C}_erm'] = clipped.mean(0)
        aggregates[f'clip{C}_fair'] = fair.mean(0)
        for name, qs in [('erm', clipped), ('fair', fair)]:
            std = plans[f'dp{C}_{name}']['std']
            aggregates[f'dp{C}_{name}'] = (qs+std*gaussian).mean(0)
    return aggregates


def privacy_plans(old, m):
    plans = {}
    for C in (1, 2):
        for name, beta in [('erm', 0.), ('fair', 2.)]:
            method = dict(clip_norm=C, beta=beta, mode='erm' if beta == 0 else 'naive')
            plan = base.privacy_plan(dict(old['config'], rounds=m['reference_horizon'], methods={'arm': method}), 'arm', m['epsilon'])
            plans[f'dp{C}_{name}'] = plan
    return plans


def run_block(m, j, rep, data, model, original, before, grads, hard, key, stamp, source, plans):
    require_mps()
    base.verify_stamp(stamp)
    assert torch.equal(flat_parameters(model), original)
    risks, raws, noise, batch_hashes = [], [], [], []
    for cid, ids in enumerate(data['train']):
        # Source C deliberately omitted: batches/noise also paired across the two source states.
        idx = base.draw_indices(len(ids), 240, base.seed_for(key, j['seed'], rep, cid, 'batch'))
        r, g, norms, mean = per_example(model, data['x'][ids[idx]], data['y'][ids[idx]],
            kind='brier', clip_norm=1e10, chunk_size=m['gradient_chunk'])
        assert float(norms.max()) < 1e10, 'Raw oracle unexpectedly clipped'
        risks.append(r)
        raws.append(g)
        noise.append(release(torch.zeros_like(mean), noise_std=1.,
            seed=base.seed_for(key, j['seed'], rep, cid, 'gaussian')))
        batch_hashes.append(base.ids_hash(idx))
    gaussian = torch.stack(noise)
    aggregates = matched_aggregates(risks, raws, gaussian, plans)
    rows = {}
    for name in m['conditions']:
        base.save(OUT/'status.json', dict(status='running', device='mps', pid=os.getpid(),
            active=cal.identifier(j), replay=rep+1, condition=name, time=time.time()))
        assert torch.equal(flat_parameters(model), original)
        eta = m['eta_fair'] if name.endswith('fair') else m['eta_erm']
        u = eta*aggregates[name]
        direction = directional_metrics(u, grads, [c['brier_loss'] for c in before['clients']], hard)
        try:
            set_parameters(model, original-u)
            after = base.evaluate(model, data, 'val')
        finally:
            set_parameters(model, original)
        actual = actual_metrics(before, after, hard, direction)
        rows[name] = dict(eta=eta, direction=direction, after=after, actual=actual,
            privacy=plans.get(name), raw_oracle_not_private=name not in plans)
    assert torch.equal(flat_parameters(model), original)
    record = dict(status='completed', device='mps', job=j, replay=rep, conditions=rows,
        before=before, hard_clients_fixed=hard, batch_hashes=batch_hashes,
        source_stamp=stamp, source=source[cal.identifier(j)],
        all_conditions_start_at_identical_parameters=True, oracle_only=True,
        feeds_training=False, test_evaluated=False,
        client_gradient_conflicts=(grads@grads.T).cpu().tolist())
    base.save(block_path(j, rep), record)
    print(f'{cal.identifier(j)} replay {rep+1}/{m["batch_replays_per_state"]} completed: 10 one-step evaluations on MPS', flush=True)


def summarize(m, stamp, source):
    blocks = []
    for j in states(m):
        for rep in range(m['batch_replays_per_state']):
            if completed(j, rep, stamp, source):
                blocks.append(json.loads(block_path(j, rep).read_text()))
    lines = ['# Diagnostic directionnel v4 — états de calibration', '',
        f'**{len(blocks)}/32 blocs complets**, {10*len(blocks)}/320 pas évalués. MPS uniquement.', '',
        'Même modèle et mêmes batches entre conditions ; validation seulement ; oracles non privés. '
        'Deux seeds et deux checkpoints sources par seed : les replays ne sont pas des seeds indépendantes.', '',
        '[Protocole](Fair_Direction_Diagnostic_V4_Protocol.md)', '']
    evidence = None
    if len(blocks) == 32:
        state_rows = []
        for j in states(m):
            rs = [b for b in blocks if b['job'] == j]
            for name in m['conditions']:
                actual = {k: st.mean(b['conditions'][name]['actual'][k] for b in rs)
                          for k in ['fixed_hard_loss_gain', 'J2_gain', 'accuracy_gain_pp', 'worst20_gain_pp',
                                    'fixed_hard_accuracy_gain_pp', 'gap_change_pp', 'variance_change_pp2']}
                directional = {k: st.mean(b['conditions'][name]['direction']['fixed_hard_clients'][k] for b in rs)
                               for k in ['predicted_loss_gain', 'gain_per_unit_step', 'cosine']}
                state_rows.append(dict(job=j, condition=name, actual=actual, direction_hard=directional))
        contrasts, gates = [], {}
        for prefix in ['raw', 'clip1', 'clip2', 'dp1', 'dp2']:
            paired = []
            for j in states(m):
                a = next(x for x in state_rows if x['job'] == j and x['condition'] == prefix+'_fair')
                b = next(x for x in state_rows if x['job'] == j and x['condition'] == prefix+'_erm')
                paired.append(dict(job=j, prefix=prefix,
                    hard_loss_gain_advantage=a['actual']['fixed_hard_loss_gain']-b['actual']['fixed_hard_loss_gain'],
                    accuracy_advantage_pp=a['actual']['accuracy_gain_pp']-b['actual']['accuracy_gain_pp'],
                    worst20_advantage_pp=a['actual']['worst20_gain_pp']-b['actual']['worst20_gain_pp'],
                    J2_gain_advantage=a['actual']['J2_gain']-b['actual']['J2_gain'],
                    directional_advantage=a['direction_hard']['gain_per_unit_step']-b['direction_hard']['gain_per_unit_step']))
            contrasts.extend(paired)
            per_seed = [dict(seed=s,
                hard_loss_gain_advantage=st.mean(x['hard_loss_gain_advantage'] for x in paired if x['job']['seed'] == s),
                accuracy_advantage_pp=st.mean(x['accuracy_advantage_pp'] for x in paired if x['job']['seed'] == s))
                for s in m['calibration_seeds']]
            criteria = m['screen']
            checks = dict(states_with_hard_loss_benefit=sum(x['hard_loss_gain_advantage'] > criteria['minimum_hard_loss_gain'] for x in paired)
                          >= criteria['minimum_states_with_hard_loss_benefit'],
                          seeds_with_hard_loss_benefit=sum(x['hard_loss_gain_advantage'] > criteria['minimum_hard_loss_gain'] for x in per_seed)
                          >= criteria['minimum_seeds_with_hard_loss_benefit'],
                          accuracy_guard=all(x['accuracy_advantage_pp'] >= -criteria['maximum_mean_accuracy_loss_pp'] for x in per_seed))
            gates[prefix] = dict(passed=all(checks.values()), checks=checks, per_seed=per_seed)
        lines += ['## Écart équitable moins classique, état commun', '',
            'Les valeurs ci-dessous sont les moyennes des huit replays à chaque état. '
            'Un gain de loss positif favorise l’équitable ; accuracy/Worst-20 en points.', '',
            '| Seed | C du checkpoint source | Condition | Avantage gain loss difficiles | Delta accuracy | Delta Worst-20 | Avantage gain J2 |',
            '|--:|--:|:--|--:|--:|--:|--:|']
        for x in contrasts:
            lines.append(f"| {x['job']['seed']} | {x['job']['C']} | {x['prefix']} | {x['hard_loss_gain_advantage']:+.6f} | {x['accuracy_advantage_pp']:+.3f} | {x['worst20_advantage_pp']:+.3f} | {x['J2_gain_advantage']:+.6f} |")
        lines += ['', '## Écran diagnostique préenregistré', '',
                  '| Condition | Marge locale selon les critères |', '|:--|:--|']
        for prefix, g in gates.items():
            lines.append(f'| {prefix} | '+('oui' if g['passed'] else 'non')+' |')
        lines += ['', '**Aucun de ces verdicts ne valide seul une méthode privée, robuste et équitable end-to-end.** '
                  'Un avantage de loss n’est pas automatiquement un avantage d’accuracy. '
                  'Les contrôles sans bruit/clipping sont des oracles, jamais des méthodes DP.', '',
                  '[Evidence détaillée](../../results/ldp_gradient_far/fair_direction_diagnostic_v4/evidence.json)']
        evidence = dict(source_stamp=stamp, source=source, states=state_rows,
            contrasts=contrasts, gates=gates, replay_blocks=32, evaluations=320,
            privacy_protected=False, confirmatory_evidence=False, no_training_launched=True)
        base.save(OUT/'evidence.json', evidence)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(lines)+'\n')
    return len(blocks), evidence


@contextmanager
def lock():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'campaign.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Diagnostic already active; no duplicate')
        yield


def worker(m):
    require_mps()
    with lock():
        try:
            stamp = source_stamp(m)
            old, source = verify_sources(m)
            manifest = OUT/'manifest.json'
            if manifest.exists():
                prev = json.loads(manifest.read_text())
                assert prev['source_stamp'] == stamp and prev['source'] == source and prev['config'] == m
            else:
                base.save(manifest, dict(config=m, source_stamp=stamp, source=source, device='mps', created_at=time.time()))
            base.save(OUT/'status.json', dict(status='preflight', pid=os.getpid(), device='mps'))
            tests = OUT/'tests.json'
            if not tests.exists():
                p = subprocess.run([sys.executable, '-m', 'pytest', *TESTS, '-q'], cwd=ROOT, capture_output=True, text=True)
                base.save(tests, dict(passed=p.returncode == 0, output=p.stdout+p.stderr, source_stamp=stamp))
            proof = json.loads(tests.read_text())
            assert proof['passed'] and proof['source_stamp'] == stamp
            secret = OUT/'simulator_secret.json'
            if not secret.exists():
                base.save(secret, dict(key=secrets.token_hex(32), not_public=True))
            key = json.loads(secret.read_text())['key']
            plans = privacy_plans(old, m)
            summarize(m, stamp, source)
            for seed in m['calibration_seeds']:
                data = base.prepare(old['config'], seed)
                for j in [x for x in states(m) if x['seed'] == seed]:
                    if all(completed(j, k, stamp, source) for k in range(m['batch_replays_per_state'])):
                        continue
                    cp = torch.load(cal.OUT/cal.identifier(j)/'checkpoint.pt', map_location='cpu', weights_only=True)
                    assert cp['job'] == j and cp['round'] == m['source_round']
                    model = base.new_model(old['config'], seed)
                    model.load_state_dict(cp['model'])
                    original = flat_parameters(model)
                    before = base.evaluate(model, data, 'val')
                    hard = sorted(range(10), key=lambda i: (before['clients'][i]['accuracy'], i))[:2]
                    gs = []
                    for cid, ids in enumerate(data['val']):
                        risk, gradient = mean_brier_gradient(model, data['x'][ids], data['y'][ids], m['evaluation_batch'])
                        assert abs(risk-before['clients'][cid]['brier_loss']) < 1e-6
                        gs.append(gradient)
                    grads = torch.stack(gs)
                    for rep in range(m['batch_replays_per_state']):
                        if not completed(j, rep, stamp, source):
                            run_block(m, j, rep, data, model, original, before, grads, hard, key, stamp, source, plans)
                            summarize(m, stamp, source)
                    del model, original, grads, gs, cp
                    torch.mps.empty_cache()
                del data
                torch.mps.empty_cache()
            base.verify_stamp(stamp)
            assert verify_sources(m)[1] == source
            count, evidence = summarize(m, stamp, source)
            assert count == 32
            base.save(OUT/'status.json', dict(status='completed', device='mps', replay_blocks=32,
                one_step_evaluations=320, gates={k: v['passed'] for k, v in evidence['gates'].items()},
                no_training_launched=True))
            print('Diagnostic complete; examine evidence before further training.', flush=True)
        except Exception as exc:
            record = dict(status='failed', error=repr(exc), pid=os.getpid(), time=time.time())
            base.save(OUT/'failure.json', record)
            base.save(OUT/'status.json', record)
            raise


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    for name in ['launch', 'worker', 'status', 'plan']:
        group.add_argument('--'+name, action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    m = config()
    if a.status:
        status = OUT/'status.json'
        print(status.read_text() if status.exists() else 'not started')
        if (OUT/'manifest.json').exists():
            manifest = json.loads((OUT/'manifest.json').read_text())
            base.verify_stamp(manifest['source_stamp'])
            print(json.dumps(dict(completed_blocks=sum(completed(j, k, manifest['source_stamp'], manifest['source'])
                for j in states(m) for k in range(m['batch_replays_per_state'])), expected_blocks=32)))
        return
    if a.plan:
        print(json.dumps(dict(config=m, states=states(m)), indent=2))
        return
    if not a.resume:
        p.error('--resume is required')
    if a.worker:
        worker(m)
        return
    require_mps()
    with lock():
        pass
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a') as stream:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', '--resume'],
            cwd=ROOT, env=dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='0'), stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(pid=process.pid, log=str(LOG), planned_blocks=32, device='mps')))


if __name__ == '__main__':
    main()
