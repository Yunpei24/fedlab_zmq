#!/usr/bin/env python3
"""Frozen, paired one-step clipping/learning-rate diagnostic; MPS only."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import secrets
import statistics as st
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
import torch
from scripts import run_fair_objective_screen as base
from scripts import run_private_fairness_population_diagnostic_v15 as population
from scripts.run_private_risk_channel_replay_v14 import risks
from privacy.fair_objective import require_mps, per_example, release, calibrate, epsilon_bound
from privacy.split_risk_gradient import plan
from privacy.capped_private_risk import weights, potential
from privacy.stable_weighted_rfa import weighted_rfa, stable_norm
from privacy.private_risk_message_safety import sanitize

NAME = 'private_clipping_step_diagnostic_v16'
OUT = ROOT / 'results/ldp_gradient_far' / NAME
PROTOCOL = ROOT / 'output/analysis/Private_Clipping_Step_Diagnostic_V16_Protocol.md'
REPORT = ROOT / 'output/analysis/Private_Clipping_Step_Diagnostic_V16_Analyse.md'
TEST = ROOT / 'tests/test_private_clipping_step_diagnostic_v16.py'
SEEDS = [170501, 170502]
COUPLES = [(2., .5), (4., .25), (8., .125), (2., .25), (2., .125)]
RULES = ['erm_mean', 'erm_rfa', 'risk_mean', 'risk_rfa']
REPLAYS = 4
TOTAL = len(SEEDS) * len(COUPLES) * len(RULES) * REPLAYS


def inputs():
    old = json.loads((population.OUT / 'manifest.json').read_text())
    base.verify_stamp(old['source_stamp'])
    audit = ROOT / 'output/analysis/Private_Fairness_Population_V15_Independent_Audit.json'
    assert json.loads(audit.read_text())['audit_passed']
    stamp = dict(old['source_stamp'])
    paths = [Path(__file__), PROTOCOL, TEST, audit, ROOT/'privacy/private_risk_message_safety.py']
    paths += [population.OUT / f'population_seed{s}.pt' for s in SEEDS]
    stamp.update({str(p.relative_to(ROOT)): base.digest(p) for p in paths})
    return old['profile'], stamp


def reclipped(rows, C):
    if rows.device.type != 'mps' or rows.ndim != 2 or not 0 < C <= 8:
        raise ValueError('MPS rows and 0<C<=8 required')
    n = torch.linalg.vector_norm(rows, dim=1)
    return rows * (C / n.clamp_min(1e-20)).clamp(max=1)[:, None]


def ledger(C, rule):
    if rule not in RULES:
        raise ValueError(rule)
    if rule.startswith('risk_'):
        return plan(N=4800, b=240, T=120, C=C, epsilon=4., delta=1e-5, epsilon_risk=.25)
    z = calibrate(q=.05, steps=120, epsilon=4., delta=1e-5)
    ep, order = epsilon_bound(q=.05, z=z, steps=120, delta=1e-5)
    return dict(N=4800, b=240, T=120, C=C, gradient_z=z, gradient_std=2*C*z/240,
                risk_std=0., risk_releases=0, gradient_releases=120,
                epsilon_realized=ep, epsilon_cap=4., delta=1e-5, order=order,
                gradient_sensitivity=2*C/240, adjacency='replace_one',
                sampling='fixed_without_replacement', scope='sample-level per-client per-run')


def aggregate(messages, reports, rule):
    chosen = weights(reports, .5)[0] if rule.startswith('risk_') else torch.ones(len(messages), device='mps')/len(messages)
    if rule.endswith('rfa'):
        A, diag = weighted_rfa(messages, chosen)
        effective = torch.tensor(diag['stationary_weights'], device='mps')
        mismatch = float((effective-chosen).abs().sum())
    else:
        A, diag, mismatch = (chosen[:, None]*messages).sum(0), None, 0.
    return A, dict(objective_weights=chosen.cpu().tolist(), solver=diag,
                   effective_weight_l1=mismatch)


def grouped(rows):
    result = []
    for seed in SEEDS:
        for C, eta in COUPLES:
            for rule in RULES:
                rr = [r for r in rows if (r['seed'], r['C'], r['eta'], r['rule']) == (seed, C, eta, rule)]
                if not rr:
                    continue
                g = dict(seed=seed, C=C, eta=eta, rule=rule, n=len(rr))
                for k in ('J_gain', 'accuracy_delta_pp', 'worst20_delta_pp', 'mse_of_step', 'predicted_J_gain'):
                    g[k] = st.mean(r[k] for r in rr)
                for k in ('variance_pp2', 'gap_best20_worst20_pp', 'ce_loss', 'brier_loss', 'balanced_accuracy_pct'):
                    g[k] = st.mean(r['validation'][k] for r in rr)
                g['effective_weight_l1'] = st.mean(r['aggregation']['effective_weight_l1'] for r in rr)
                result.append(g)
    return result


def decision(gs):
    def get(seed, C, eta, rule):
        found = [g for g in gs if (g['seed'], g['C'], g['eta'], g['rule']) == (seed, C, eta, rule)]
        assert len(found) == 1 and found[0]['n'] == REPLAYS
        return found[0]
    checks = []
    for C, eta in COUPLES[1:3]:
        comparisons = []
        for seed in SEEDS:
            candidate = get(seed, C, eta, 'risk_rfa')
            controls = [('original', 2., .5, 'risk_rfa', .25, -.10, True),
                        ('step_only', 2., eta, 'risk_rfa', .10, None, True),
                        ('same_pair_erm_mean', C, eta, 'erm_mean', .50, -.25, False),
                        ('same_pair_erm_rfa', C, eta, 'erm_rfa', .50, -.25, False)]
            for label, c, e, rule, minw, mina, needj in controls:
                control = get(seed, c, e, rule)
                delta = {k: candidate[k]-control[k] for k in ('J_gain', 'accuracy_delta_pp', 'worst20_delta_pp')}
                criteria = dict(worst20=delta['worst20_delta_pp'] >= minw,
                                accuracy=mina is None or delta['accuracy_delta_pp'] >= mina,
                                J=not needj or delta['J_gain'] >= -1e-7)
                comparisons.append(dict(seed=seed, control=label, delta=delta, criteria=criteria, passed=all(criteria.values())))
        checks.append(dict(C=C, eta=eta, comparisons=comparisons, passed=all(c['passed'] for c in comparisons)))
    selected = next((dict(C=c['C'], eta=c['eta']) for c in checks if c['passed']), None)
    return dict(checks=checks, selected_for_end_to_end_calibration=selected, global_validation=False,
                independent_confirmation=False, attacks_launched=False)


def report(stamp):
    files = sorted(OUT.glob('seed*__r*__C*__*.json'))
    rows = [json.loads(p.read_text()) for p in files]
    for r, p in zip(rows, files):
        assert r['source_stamp'] == stamp and r['device'] == 'mps' and not r['test_evaluated']
        assert base.digest(p.with_suffix('.pt')) == r['vector_sha256']
    gs = grouped(rows)
    d = decision(gs) if len(rows) == TOTAL else None
    lines = ['# V16 — seuil de clipping et pas à bruit appliqué apparié', '',
             f'**{len(rows)}/{TOTAL} interventions MPS** ; deux états, quatre replays par état, validation seule.', '',
             'Les valeurs sont des changements après un seul pas, pas des résultats d’entraînement final.', '',
             '| Seed | C | η | Règle | Gain J réel | Δ accuracy (pp) | Δ Worst-20 (pp) | MSE du pas | Écart L1 des poids effectifs |',
             '|--:|--:|--:|:--|--:|--:|--:|--:|--:|']
    for g in gs:
        lines.append(f"| {g['seed']} | {g['C']:g} | {g['eta']:g} | {g['rule']} | {g['J_gain']:+.6f} | {g['accuracy_delta_pp']:+.4f} | {g['worst20_delta_pp']:+.4f} | {g['mse_of_step']:.6f} | {g['effective_weight_l1']:.4f} |")
    if d is not None:
        assert len(rows) == TOTAL and all(g['n'] == REPLAYS for g in gs)
        base.save(OUT/'evidence.json', dict(source_stamp=stamp, grouped=gs, decision=d))
        lines += ['', '## Décision de diagnostic', '',
                  f"Couple admis pour une éventuelle calibration end-to-end : **{d['selected_for_end_to_end_calibration'] or 'aucun'}**."]
        for c in d['checks']:
            lines += ['', f"### C={c['C']:g}, η={c['eta']:g} : {'passe' if c['passed'] else 'échoue'}", '',
                      '| Seed | Contrôle | Δ gain J | Δ accuracy (pp) | Δ Worst-20 (pp) | Passe |',
                      '|--:|:--|--:|--:|--:|:--|']
            for x in c['comparisons']:
                y=x['delta'];lines.append(f"| {x['seed']} | {x['control']} | {y['J_gain']:+.6f} | {y['accuracy_delta_pp']:+.4f} | {y['worst20_delta_pp']:+.4f} | {x['passed']} |")
    lines += ['', 'Bruit de chaque message appliqué identique pour ηC=1, à famille de privacy fixée ; cela ne garantit pas la même covariance après RFA. Les replays ne sont pas des seeds d’entraînement indépendantes. Les oracles et sorties de validation ne sont pas des releases DP.', '',
              '[Protocole préenregistré](Private_Clipping_Step_Diagnostic_V16_Protocol.md).']
    REPORT.write_text('\n'.join(lines)+'\n')
    return len(rows), d


def one(seed, replay, C, eta, rule, model, state, data, initial, pop, raw, noise, report_noise, batch_hashes, clips, stamp):
    identifier = f'seed{seed}__r{replay}__C{C:g}_eta{eta:g}__{rule}'
    path=OUT/(identifier+'.json')
    if path.exists():
        old=json.loads(path.read_text());assert old['source_stamp']==stamp and old['vector_sha256']==base.digest(path.with_suffix('.pt'))
        return
    model.load_state_dict(state)
    p = ledger(C, rule)
    sent = raw[C]+p['gradient_std']*noise
    rr = (pop['risks']+p['risk_std']*report_noise).clamp(0,1) if rule.startswith('risk_') else None
    sent, rr, safety = sanitize(sent, rr)
    assert safety['invalid_message_rows']==safety['nonfinite_risk_reports']==0
    A, diag = aggregate(sent, rr, rule); step=eta*A
    _, GJ, target = population.fair_directions(pop['risks'],pop['raw'])
    mse=float(stable_norm(step-.5*target).square());predicted=float(torch.dot(GJ,step))
    base.apply_gradient(model,step,1.)
    v=base.evaluate(model,data,'val')
    before=float(potential(pop['risks'],.5).mean());after=float(potential(risks(model,data),.5).mean())
    vectorpath=path.with_suffix('.pt')
    base.checkpoint(vectorpath,dict(step=step.detach().cpu(),messages=sent.detach().cpu(),
        reports=None if rr is None else rr.detach().cpu(),source_stamp=stamp,privacy_protected=False))
    row=dict(seed=seed,replay=replay,C=C,eta=eta,rule=rule,device='mps',source_stamp=stamp,
        privacy=p,applied_message_noise_std=eta*p['gradient_std'],batch_hashes=batch_hashes,
        clip_fraction_by_client=clips[C],aggregation=diag,initial_validation=initial,validation=v,
        J_before=before,J_after=after,J_gain=before-after,predicted_J_gain=predicted,
        mse_of_step=mse,step_norm=float(stable_norm(step)),accuracy_delta_pp=v['accuracy_pct']-initial['accuracy_pct'],
        worst20_delta_pp=v['worst20_pct']-initial['worst20_pct'],test_evaluated=False,
        cumulative_training_steps=0,oracles_not_private=True,vector_sha256=base.digest(vectorpath))
    base.save(path,row)


def worker():
    require_mps();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            profile, stamp=inputs()
            manifest=dict(profile=profile,source_stamp=stamp,seeds=SEEDS,replays=REPLAYS,
                          couples=COUPLES,rules=RULES,total=TOTAL)
            # JSON normalizes tuples into lists.
            manifest=json.loads(json.dumps(manifest))
            if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
            else:base.save(OUT/'manifest.json',manifest)
            if not (OUT/'tests.json').exists():
                p=subprocess.run([sys.executable,'-m','pytest',str(TEST),'-q'],cwd=ROOT,capture_output=True,text=True)
                base.save(OUT/'tests.json',dict(passed=p.returncode==0,output=p.stdout+p.stderr,source_stamp=stamp))
            tests=json.loads((OUT/'tests.json').read_text());assert tests['passed'] and tests['source_stamp']==stamp
            keypath=OUT/'simulator_secret.json'
            if not keypath.exists():
                assert not list(OUT.glob('seed*__r*__C*__*.json'))
                base.save(keypath,dict(key=secrets.token_hex(32),private_release=False))
            key=json.loads(keypath.read_text())['key']
            for seed in SEEDS:
                data=base.prepare(profile,seed);model=base.new_model(profile,seed)
                state=torch.load(population.SOURCE/f'seed{seed}__erm_mean'/'checkpoint.pt',map_location='cpu',weights_only=True)['model']
                model.load_state_dict(state);initial=base.evaluate(model,data,'val')
                pop=torch.load(population.OUT/f'population_seed{seed}.pt',map_location='mps',weights_only=True)['population']
                assert initial==json.loads((population.SOURCE/f'seed{seed}__erm_mean'/'metrics.json').read_text())['final']['validation']
                for replay in range(REPLAYS):
                    model.load_state_dict(state);raw={C:[] for C in (2.,4.,8.)};clips={C:[] for C in raw}
                    noises=[];report_noises=[];batch_hashes=[]
                    base.save(OUT/'status.json',dict(status='running',device='mps',seed=seed,replay=replay,phase='batch_gradients',pid=os.getpid()))
                    for cid, ids in enumerate(data['train']):
                        idx=base.draw_indices(4800,240,base.seed_for(key,seed,replay,cid,'batch'))
                        _,g8,norms,_=per_example(model,data['x'][ids[idx]],data['y'][ids[idx]],clip_norm=8.)
                        for C in raw:
                            raw[C].append(reclipped(g8,C).mean(0));clips[C].append(float((norms>C).float().mean()))
                        noises.append(release(torch.zeros_like(raw[2.][-1]),noise_std=1.,seed=base.seed_for(key,seed,replay,cid,'gradient')))
                        report_noises.append(release(torch.zeros(1,device='mps'),noise_std=1.,seed=base.seed_for(key,seed,replay,cid,'risk')).squeeze())
                        batch_hashes.append(base.ids_hash(idx))
                    raw={C:torch.stack(g) for C,g in raw.items()};noise=torch.stack(noises);rn=torch.stack(report_noises)
                    cache=OUT/f'block_seed{seed}_r{replay}.pt'
                    if not cache.exists():base.checkpoint(cache,dict(raw={str(C):g.detach().cpu() for C,g in raw.items()},
                        noise=noise.detach().cpu(),report_noise=rn.detach().cpu(),batch_hashes=batch_hashes,
                        source_stamp=stamp,privacy_protected=False))
                    for C,eta in COUPLES:
                        for rule in RULES:
                            base.verify_stamp(stamp)
                            one(seed,replay,C,eta,rule,model,state,data,initial,pop,raw,noise,rn,batch_hashes,clips,stamp)
                            count,d=report(stamp)
                            base.save(OUT/'status.json',dict(status='running',device='mps',valid_interventions=count,total=TOTAL,pid=os.getpid()))
                            print(f'{count}/{TOTAL} interventions complete',flush=True)
                del data,model,pop;torch.mps.empty_cache()
            count,d=report(stamp);assert count==TOTAL
            base.save(OUT/'status.json',dict(status='completed',device='mps',valid_interventions=count,
                selected_for_end_to_end_calibration=d['selected_for_end_to_end_calibration'],next_campaign_launched=False))
        except Exception as exc:
            base.save(OUT/'status.json',dict(status='failed',device='mps',error=repr(exc),pid=os.getpid()));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true',required=True)
    parser.add_argument('--resume',action='store_true',required=True);parser.parse_args();worker()


if __name__=='__main__':main()
