#!/usr/bin/env python3
"""Eight paired one-step diagnostic blocks, MPS only, no test or training."""
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
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
import torch
import yaml
from scripts import run_capped_private_risk_calibration_v7 as parent
from scripts import run_fair_objective_screen as base
from privacy.fair_objective import require_mps, per_example, release
from privacy.split_risk_gradient import private_risk, weighted_rfa as legacy_rfa
from privacy.capped_private_risk import weights
from privacy.stable_weighted_rfa import weighted_rfa, stable_norm
from metrics.fair_direction_audit import (flat_parameters, set_parameters, mean_brier_gradient,
    directional_metrics, actual_metrics)

NAME = 'fair_direction_curvature_v8'
MATRIX = ROOT/'configs/ldp_gradient_far'/f'{NAME}.yaml'
OUT = ROOT/'results/ldp_gradient_far'/NAME
LOG = ROOT/'logs'/f'{NAME}.log'
REPORT = ROOT/'output/analysis/Fair_Direction_Curvature_V8_Analyse.md'


def config():
    m = yaml.safe_load(MATRIX.read_text())
    assert m['campaign_id'] == NAME and m['device'] == 'mps'
    assert m['calibration_seeds'] == [170501, 170502] and m['source_arm'] == 'erm_mean_C2'
    assert m['clip'] == 2 and m['risk_scale'] == .5 and m['etas'] == [.5, 1., 2.]
    assert m['replays_per_seed'] == 4 and len(m['conditions']) == 8
    assert m['expected_blocks'] == 8 and m['expected_evaluations'] == 192
    assert m['oracle_only'] and not any(m[k] for k in ('training', 'test_evaluated', 'automatic_promotion'))
    return m


def objective(risks, c):
    return st.mean(r + (r*r/c if r <= c else 2*r-c) for r in risks)


def inputs(m):
    old = json.loads((parent.OUT/'manifest.json').read_text())
    base.verify_stamp(old['source_stamp'])
    assert json.loads((parent.OUT/'status.json').read_text())['status'] == 'completed'
    files = [Path(__file__), MATRIX, ROOT/m['protocol'], ROOT/'privacy/stable_weighted_rfa.py',
             ROOT/'privacy/capped_private_risk.py', ROOT/'metrics/fair_direction_audit.py',
             ROOT/'tests/test_stable_weighted_rfa.py', ROOT/'tests/test_fair_direction_curvature_v8.py']
    stamp = dict(old['source_stamp'])
    stamp.update({str(p.relative_to(ROOT)): base.digest(p) for p in files})
    sources = {}
    for seed in m['calibration_seeds']:
        j = dict(seed=seed, arm=m['source_arm'])
        assert parent.completed(j, old['config'], old['source_stamp'])
        d = parent.OUT/parent.identifier(j)
        sources[str(seed)] = {f: base.digest(d/f) for f in ('checkpoint.pt', 'metrics.json')}
    return old['config'], stamp, sources


def path(seed, rep):
    return OUT/f'seed{seed}_replay{rep:02d}.json'


def completed(seed, rep, stamp, sources):
    p = path(seed, rep)
    if not p.exists():
        return False
    r = json.loads(p.read_text())
    assert r['status'] == 'completed' and r['device'] == 'mps'
    assert r['source_stamp'] == stamp and r['sources'] == sources[str(seed)]
    assert r['seed'] == seed and r['replay'] == rep and len(r['evaluations']) == 24
    assert r['identical_start_parameters'] and not r['test_evaluated'] and r['oracle_only']
    return True


def block(m, old, seed, rep, model, data, original, before, grads, key, stamp, sources):
    assert torch.equal(flat_parameters(model), original)
    p = parent.privacy(old, 'risk_mean_C2_scale0.5')
    full = parent.privacy(old, 'erm_mean_C2')
    raw, clipped, gauss, reports, risks, clients = [], [], [], [], [], []
    for cid, ids in enumerate(data['train']):
        report, risk = private_risk(model, data['x'][ids], data['y'][ids],
            noise_std=p['risk_std'], seed=base.seed_for(key, seed, rep, cid, 'risk'), N=p['N'])
        idx = base.draw_indices(len(ids), p['b'], base.seed_for(key, seed, rep, cid, 'batch'))
        _, gc, norms, g = per_example(model, data['x'][ids[idx]], data['y'][ids[idx]], clip_norm=m['clip'])
        raw.append(g); clipped.append(gc.mean(0)); reports.append(report); risks.append(risk)
        gauss.append(release(torch.zeros_like(g), noise_std=1., seed=base.seed_for(key, seed, rep, cid, 'gaussian')))
        clients.append(dict(client=cid, batch_hash=base.ids_hash(idx), raw_risk=float(risk),
            private_risk=float(report), clip_fraction=float((norms>m['clip']).float().mean()),
            raw_gradient_norm=float(stable_norm(g)), clipped_gradient_norm=float(stable_norm(gc.mean(0)))))
    raw, clipped, gauss = map(torch.stack, (raw, clipped, gauss))
    lam, _ = weights(torch.stack(reports), m['risk_scale'])
    exact_lam, _ = weights(torch.stack(risks), m['risk_scale'])
    messages = clipped + p['gradient_std']*gauss
    rfa, solver = weighted_rfa(messages, lam)
    legacy, legacy_diag = legacy_rfa(messages, lam)
    directions = dict(raw_mean=raw.mean(0), raw_risk=(exact_lam[:, None]*raw).sum(0),
        clipped_mean=clipped.mean(0), clipped_risk=(exact_lam[:, None]*clipped).sum(0),
        dp_mean_full=(clipped+full['gradient_std']*gauss).mean(0),
        dp_mean_matched=messages.mean(0), dp_risk_mean=(lam[:, None]*messages).sum(0), dp_risk_rfa=rfa)
    assert set(directions) == set(m['conditions'])
    hard = sorted(range(len(before['clients'])), key=lambda i: before['clients'][i]['accuracy'])[:2]
    valrisks = [c['brier_loss'] for c in before['clients']]
    aa = torch.tensor([1+2*min(r/m['risk_scale'], 1) for r in valrisks], device='mps')
    gj = (aa[:, None]*grads).mean(0)
    out = []
    for eta in m['etas']:
        for name in m['conditions']:
            require_mps(); base.verify_stamp(stamp)
            base.save(OUT/'status.json', dict(status='running', device='mps', seed=seed,
                replay=rep+1, condition=name, eta=eta, pid=os.getpid()))
            u = eta*directions[name]
            assert torch.equal(flat_parameters(model), original)
            predicted = directional_metrics(u, grads, valrisks, hard)
            predicted_j = float(torch.dot(gj, u))
            try:
                set_parameters(model, original-u)
                after = base.evaluate(model, data, 'val')
            finally:
                set_parameters(model, original)
            actual = actual_metrics(before, after, hard, predicted)
            actual['Jc_gain'] = objective(valrisks, m['risk_scale'])-objective([c['brier_loss'] for c in after['clients']], m['risk_scale'])
            actual['Jc_taylor_remainder'] = predicted_j-actual['Jc_gain']
            out.append(dict(condition=name, eta=eta, predicted=predicted, predicted_Jc_gain=predicted_j,
                            actual=actual, after=after))
    assert torch.equal(flat_parameters(model), original)
    record = dict(status='completed', device='mps', seed=seed, replay=rep, source_stamp=stamp,
        sources=sources[str(seed)], before=before, hard_clients_fixed=hard, clients=clients,
        evaluations=out, solver=solver, legacy_solver=legacy_diag,
        legacy_solver_direction_distance=float(stable_norm(rfa-legacy)),
        risk_weights=lam.cpu().tolist(), exact_risk_weights=exact_lam.cpu().tolist(),
        privacy_plans=dict(split=p, gradient_only=full),
        directions_pairwise_distance={f'{a}__{b}':float(stable_norm(directions[a]-directions[b])) for a,b in
            [('raw_mean','clipped_mean'),('clipped_mean','dp_mean_matched'),('dp_mean_matched','dp_risk_mean'),('dp_risk_mean','dp_risk_rfa')]},
        identical_start_parameters=True, oracle_only=True, test_evaluated=False, training=False)
    base.save(path(seed, rep), record)
    print(f'seed{seed} replay {rep+1}/4: 24 one-step evaluations completed on MPS', flush=True)


def report(m, stamp, sources):
    blocks = [json.loads(path(s, k).read_text()) for s in m['calibration_seeds'] for k in range(4)
              if completed(s, k, stamp, sources)]
    rows = []
    for seed in m['calibration_seeds']:
        bb = [b for b in blocks if b['seed'] == seed]
        if not bb:
            continue
        for eta in m['etas']:
            for name in m['conditions']:
                ee = [e for b in bb for e in b['evaluations'] if e['eta']==eta and e['condition']==name]
                keys = ('accuracy_gain_pp','worst20_gain_pp','fixed_hard_loss_gain','mean_loss_gain','Jc_gain','Jc_taylor_remainder','gap_change_pp','variance_change_pp2')
                rows.append(dict(seed=seed, eta=eta, condition=name, replays=len(ee),
                    actual={k:st.mean(e['actual'][k] for e in ee) for k in keys},
                    predicted_Jc_gain=st.mean(e['predicted_Jc_gain'] for e in ee),
                    predicted_hard_gain=st.mean(e['predicted']['fixed_hard_clients']['predicted_loss_gain'] for e in ee)))
    evidence = dict(completed_blocks=len(blocks), expected_blocks=8, evaluations=24*len(blocks), rows=rows,
                    confirmatory=False, training=False, test_evaluated=False, source_stamp=stamp, sources=sources)
    base.save(OUT/'evidence.json', evidence)
    lines = ['# V8 — diagnostic directionnel et courbure', '',
             f'**{len(blocks)}/8 blocs, {24*len(blocks)}/192 évaluations à un pas**, MPS, validation uniquement.', '',
             'Deux seeds de calibration, quatre replays chacune. Pas de confirmation indépendante ; aucun candidat promu.', '',
             '[Protocole préenregistré](Fair_Direction_Curvature_V8_Protocol.md)', '',
             'Un gain de loss positif indique une amélioration. Les deltas accuracy/Worst-20 sont en points. '
             'Les directions brutes/clippées sans bruit sont des oracles, pas des méthodes DP.', '',
             '| Seed | Pas | Direction | Δ acc. | Δ Worst-20 | Gain loss difficiles | Gain Jc prédit | Gain Jc réel | Reste Taylor Jc |',
             '|--:|--:|:--|--:|--:|--:|--:|--:|--:|']
    for r in rows:
        a=r['actual']
        lines.append(f"| {r['seed']} | {r['eta']:g} | {r['condition']} | {a['accuracy_gain_pp']:+.3f} | {a['worst20_gain_pp']:+.3f} | {a['fixed_hard_loss_gain']:+.6f} | {r['predicted_Jc_gain']:+.6f} | {a['Jc_gain']:+.6f} | {a['Jc_taylor_remainder']:+.6f} |")
    if len(blocks)==8:
        lines += ['', '## Solveur sur les messages réels', '',
                  f"Écart maximal entre anciennes/nouvelles directions RFA : {max(b['legacy_solver_direction_distance'] for b in blocks):.8g}.", '',
                  'Cette comparaison est distincte de l’audit de messages extrêmes. Une correction numérique peut être essentielle sous attaque tout en changeant très peu les directions propres.', '',
                  '## Limites', '',
                  'Une loss de validation à un pas ne prédit pas automatiquement une trajectoire complète. '
                  'Les replays ne sont pas des seeds indépendantes et les états sont terminaux, non des débuts d’entraînement. '
                  'Aucun seuil de validation globale n’est modifié et aucune attaque n’est lancée automatiquement.', '',
                  '[Evidence complète](../../results/ldp_gradient_far/fair_direction_curvature_v8/evidence.json)']
    REPORT.write_text('\n'.join(lines)+'\n')
    return len(blocks)


@contextmanager
def lock():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'campaign.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Already active; no duplicate')
        yield


def worker(m):
    require_mps()
    with lock():
        try:
            old, stamp, sources = inputs(m)
            manifest = OUT/'manifest.json'
            content = dict(config=m, source_stamp=stamp, sources=sources)
            if manifest.exists():
                assert json.loads(manifest.read_text()) == content
            else:
                base.save(manifest, content)
            tests = OUT/'tests.json'
            if not tests.exists():
                p = subprocess.run([sys.executable,'-m','pytest','tests/test_stable_weighted_rfa.py',
                    'tests/test_fair_direction_curvature_v8.py','-q'], cwd=ROOT, capture_output=True, text=True)
                base.save(tests, dict(passed=p.returncode==0, output=p.stdout+p.stderr, source_stamp=stamp))
            assert json.loads(tests.read_text())['passed']
            secret = OUT/'simulator_secret.json'
            if not secret.exists():
                base.save(secret, dict(key=secrets.token_hex(32), not_public=True))
            key = json.loads(secret.read_text())['key']
            for seed in m['calibration_seeds']:
                if all(completed(seed, k, stamp, sources) for k in range(4)):
                    continue
                data = base.prepare(old, seed)
                d = parent.OUT/parent.identifier(dict(seed=seed,arm=m['source_arm']))
                checkpoint = torch.load(d/'checkpoint.pt', map_location='cpu', weights_only=True)
                assert checkpoint['round'] == 60
                model = base.new_model(old, seed); model.load_state_dict(checkpoint['model'])
                original = flat_parameters(model)
                before = base.evaluate(model, data, 'val')
                previous = json.loads((d/'metrics.json').read_text())
                assert data['splits'] == previous['splits']
                for metric in ('accuracy_pct','worst20_pct','brier_loss'):
                    assert abs(before[metric]-previous['final']['validation'][metric]) < 1e-6
                gg=[]
                for cid, ids in enumerate(data['val']):
                    risk, grad = mean_brier_gradient(model,data['x'][ids],data['y'][ids])
                    assert abs(risk-before['clients'][cid]['brier_loss']) < 1e-5
                    gg.append(grad)
                grads=torch.stack(gg)
                for k in range(4):
                    if not completed(seed,k,stamp,sources):
                        block(m,old,seed,k,model,data,original,before,grads,key,stamp,sources)
                    report(m,stamp,sources)
                del data, model, grads, original, checkpoint
                torch.mps.empty_cache()
            count=report(m,stamp,sources); assert count==8
            base.save(OUT/'status.json',dict(status='completed',device='mps',blocks=8,evaluations=192,
                no_training_or_confirmation_launched=True))
        except Exception as exc:
            base.save(OUT/'status.json',dict(status='failed',error=repr(exc),device='mps',pid=os.getpid()))
            raise


def main():
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True)
    for flag in ('launch','worker','status'):g.add_argument('--'+flag,action='store_true')
    p.add_argument('--resume',action='store_true');args=p.parse_args();m=config()
    if args.status:
        print((OUT/'status.json').read_text() if (OUT/'status.json').exists() else 'not started'); return
    if not args.resume:p.error('--resume required')
    if args.worker:worker(m);return
    require_mps()
    with lock():pass
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open('a') as f:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker','--resume'],cwd=ROOT,
            env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=proc.pid,device='mps',blocks=8,evaluations=192,log=str(LOG))))


if __name__=='__main__':main()
