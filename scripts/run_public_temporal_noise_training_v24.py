#!/usr/bin/env python3
"""Prospective real-gradient transfer of two public noise allocations, MPS only."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
import torch
import yaml
from scripts import run_fair_objective_screen as base
from scripts import run_fresh_private_risk_step_v22_r1 as controls
from scripts.analyze_private_risk_confirmation_v12 import verify
from privacy.fair_objective import require_mps, per_example, release
from privacy.split_risk_gradient import private_risk
from privacy.scheduled_private_risk import aggregate
from privacy.private_risk_message_safety import sanitize
from privacy.public_temporal_noise_training_v24 import (
    METHODS, GRID_INDICES, plan, round_parameters, prefix_ledger, contrast_passes)

NAME = 'public_temporal_noise_training_v24'
OUT = ROOT/'results/ldp_gradient_far'/NAME
MATRIX = ROOT/'configs/ldp_gradient_far'/f'{NAME}.yaml'
PROTOCOL = ROOT/'output/analysis/Public_Temporal_Noise_Training_V24_Protocol.md'
REPORT = ROOT/'output/analysis/Public_Temporal_Noise_Training_V24_Status.md'
TEST = ROOT/'tests/test_public_temporal_noise_training_v24.py'
PUBLIC = ROOT/'results/ldp_gradient_far/public_temporal_noise_audit_v23/evidence.json'
PRIOR = controls.prior.OUT


def canonical(value):
    return json.loads(json.dumps(value))


def inputs():
    c = yaml.safe_load(MATRIX.read_text())
    assert c['campaign_id'] == NAME and c['device'] == 'mps'
    assert c['seeds'] == [170501,170502] and c['public_grid_indices'] == list(GRID_INDICES)
    assert c['candidate_order'] == [7,13] and c['methods'] == list(METHODS) and c['expected_runs'] == 16
    expected = dict(dataset='fashionmnist', model='lenet5_tanh', num_clients=10,
        partition='client_dirichlet_balanced', dirichlet_beta=.1, public_train_size=4800,
        validation_size=1200, batch_size=240, rounds=120, boundary_round=60,
        local_optimizer_steps=0, clip=2., server_clip=None, recursive_memory=False,
        epsilon=4., delta=1e-5, risk_epsilon=.25, risk_scale=.5,
        adjacency='replace_one', sampling='fixed_without_replacement', primary_checkpoint=120,
        attacks='none', test_evaluated=False, automatic_confirmation=False, automatic_attacks=False,
        minimum_worst20_advantage_pp=1., maximum_accuracy_loss_pp=1.)
    assert all(c[k] == v for k,v in expected.items()) and c['learning_rates'] == [2.,.5]
    assert c['gate_controls'] == ['both_new_allocations','unchanged_v19','historical_v11']
    assert c['gate_control_methods'] == ['erm_mean','erm_rfa'] and c['mean_gap_and_variance_must_not_increase']
    a = ROOT/'output/analysis/Fresh_Private_Risk_Step_V22_Analyse.json'
    audit = json.loads(a.read_text())
    assert audit['audit_passed'] and not audit['eligible_for_independent_confirmation']
    public = json.loads(PUBLIC.read_text())
    assert public['audit_passed'] and public['admitted_to_real_diagnostic'] and not public['model_data_read']
    old = json.loads((controls.OUT/'manifest.json').read_text())
    stamp = dict(old['source_stamp']); stamp.update(public['source_stamp']); base.verify_stamp(stamp)
    for risk in (False,True):
        family = [r for r in public['rows'] if r['with_risk'] == risk]
        assert min(family, key=lambda r:r['energy_proxy'])['grid_index'] == 13
        assert min(family, key=lambda r:60*2*r['sigma_early']**2+60*.5*r['sigma_late']**2)['grid_index'] == 7
        for k in GRID_INDICES:
            row = next(r for r in family if r['grid_index'] == k)
            p = canonical(plan(k, 'risk_rfa' if risk else 'erm_mean'))
            assert all(p[name] == row[name] for name in ('z_early','z_late','sigma_early','sigma_late','epsilon_realized','rdp'))
    files = [Path(__file__), MATRIX, PROTOCOL, TEST, a, PUBLIC,
             ROOT/'privacy/public_temporal_noise_training_v24.py',
             ROOT/'output/analysis/Public_Temporal_Noise_V23_Convergence_Limit.md']
    files += [PRIOR/f'seed{s}__fresh__{m}/{name}' for s in c['seeds'] for m in METHODS
              for name in ('metrics.json','simulator_oracle.json','orchestration_status.json')]
    files += [controls.historical(s,m) for s in c['seeds'] for m in ('erm_mean','erm_rfa')]
    stamp.update({str(p.relative_to(ROOT)):base.digest(p) for p in files})
    profile = {k: c[k] for k in ('dataset','model','num_clients','partition','dirichlet_beta',
                               'public_train_size','validation_size','rounds')}
    profile['evaluation_rounds'] = [1]+list(range(10,121,10))
    return c, profile, stamp


def jobs(c):
    return [dict(seed=s,grid_index=k,method=m) for s in c['seeds'] for k in c['public_grid_indices'] for m in METHODS]


def identifier(j):
    return f'seed{j["seed"]}__k{j["grid_index"]}__{j["method"]}'


def completed(j, stamp):
    path = OUT/identifier(j); status = path/'orchestration_status.json'
    if not status.exists() or json.loads(status.read_text())['status'] != 'completed':
        return False
    s = json.loads(status.read_text()); r = json.loads((path/'metrics.json').read_text())
    assert s['metrics_sha256'] == base.digest(path/'metrics.json') and s['oracle_sha256'] == base.digest(path/'simulator_oracle.json')
    assert r['source_stamp'] == stamp and r['job'] == j and r['device'] == 'mps' and not r['test_evaluated']
    assert r['privacy'] == canonical(plan(j['grid_index'],j['method']))
    assert r['privacy']['epsilon_realized'] <= 4 and [x['round'] for x in r['rounds']] == list(range(1,121))
    verify(r['final']['validation'])
    return True


def train(j, config, profile, stamp, data, key):
    if completed(j,stamp):
        return
    path = OUT/identifier(j); path.mkdir(parents=True,exist_ok=True)
    p = plan(j['grid_index'],j['method'])
    prior_path = PRIOR/f'seed{j["seed"]}__fresh__{j["method"]}'
    reference = json.loads((prior_path/'metrics.json').read_text())
    expected = json.loads((prior_path/'simulator_oracle.json').read_text())['rounds']
    assert p['risk_std'] == reference['privacy']['risk_std']
    model = base.new_model(profile,j['seed'])
    rows,oracles,start,elapsed = [],[],0,0.
    cp = path/'checkpoint.pt'; key_sha = base.digest(PRIOR/'simulator_secret.json')
    if cp.exists():
        state = torch.load(cp,map_location='cpu',weights_only=True)
        assert state['source_stamp'] == stamp and state['job'] == j and state['key_sha'] == key_sha
        model.load_state_dict(state['model'])
        rows,oracles,start,elapsed,initial = (state[k] for k in ('rows','oracles','round','elapsed_seconds','initial'))
    else:
        initial = base.evaluate(model,data,'val')
    assert initial == reference['initial'] and data['splits'] == reference['splits']
    base.save(path/'public_protocol.json',dict(config=config,profile=profile,job=j,privacy=p,source_stamp=stamp))
    code = {k:v for k,v in stamp.items() if Path(k).suffix in ('.py','.yaml','.md')}
    for t in range(start,120):
        tick = time.monotonic(); require_mps(); base.verify_stamp(code)
        rp = round_parameters(p,t+1)
        status = dict(status='running',device='mps',active=identifier(j),round=t+1,total_rounds=120,
                      phase=rp['phase'],pid=os.getpid(),updated_unix=time.time())
        base.save(OUT/'status.json',status); base.save(path/'orchestration_status.json',status)
        before = {name:v.detach().cpu().clone() for name,v in model.state_dict().items()}
        sent,reports,local = [],[],[]
        for cid,ids in enumerate(data['train']):
            rr,raw = None,None
            if j['method'].startswith('risk_'):
                rr,raw = private_risk(model,data['x'][ids],data['y'][ids],noise_std=p['risk_std'],
                    seed=base.seed_for(key,j['seed'],t,cid,'risk'),N=4800)
                reports.append(rr)
            ix = base.draw_indices(4800,240,base.seed_for(key,j['seed'],t,cid,'batch'))
            ih = base.ids_hash(ix); assert ih == expected[t]['clients'][cid]['batch_hash']
            _,g,norms,_ = per_example(model,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)
            sent.append(release(g.mean(0),noise_std=rp['sigma'],seed=base.seed_for(key,j['seed'],t,cid,'gaussian')))
            local.append(dict(client=cid,batch_hash=ih,gradient_clipped_count=int((norms>2).sum()),batch_size=240,
                              private_risk=None if rr is None else float(rr),raw_risk=None if raw is None else float(raw)))
        messages,r,safety = sanitize(torch.stack(sent),None if not reports else torch.stack(reports))
        assert safety['invalid_message_rows'] == safety['nonfinite_risk_reports'] == 0
        step,agg = aggregate(messages,r,kind=j['method'],round_number=t+1,horizon=120)
        assert agg['eta'] == rp['eta']
        agg['message_safety'] = safety; base.apply_gradient(model,step,1.)
        val = base.evaluate(model,data,'val') if t+1 in profile['evaluation_rounds'] else None
        if val is not None:
            verify(val)
        prefix = prefix_ledger(p,t+1); assert prefix['epsilon'] <= 4
        rows.append(dict(round=t+1,device='mps',epsilon_realized=prefix['epsilon'],
                         privacy_order=prefix['order'],noise_schedule=rp,aggregation=agg,validation=val))
        oracles.append(dict(round=t+1,clients=local)); elapsed += time.monotonic()-tick
        base.checkpoint(cp,dict(model={name:v.detach().cpu() for name,v in model.state_dict().items()},previous_model=before,
            rows=rows,oracles=oracles,round=t+1,elapsed_seconds=elapsed,initial=initial,source_stamp=stamp,job=j,key_sha=key_sha))
        if val is not None:
            print(f'{identifier(j)} {t+1}/120 acc={val["accuracy_pct"]:.3f} W20={val["worst20_pct"]:.3f}',flush=True)
    base.verify_stamp(stamp)
    base.save(path/'metrics.json',dict(job=j,source_stamp=stamp,device='mps',privacy=p,initial=initial,rounds=rows,final=rows[-1],
        splits=data['splits'],test_evaluated=False,per_client_batch_gradient_evaluations=120,
        validation_and_oracles_not_private=True,elapsed_seconds=elapsed))
    base.save(path/'simulator_oracle.json',dict(privacy_protected=False,feeds_mechanism=False,rounds=oracles))
    base.save(path/'orchestration_status.json',dict(status='completed',device='mps',job=j,round=120,
        metrics_sha256=base.digest(path/'metrics.json'),oracle_sha256=base.digest(path/'simulator_oracle.json')))
    del model; torch.mps.empty_cache()


def decide(config, rows):
    assert len(rows) == 16
    index = {(r['job']['seed'],r['job']['grid_index'],r['job']['method']):r['final']['validation'] for r in rows}
    candidates = []
    for candidate in config['candidate_order']:
        contrasts = []
        for mode in (7,13,'unchanged_v19','historical_v11'):
            for method in ('erm_mean','erm_rfa'):
                pairs = []
                for seed in config['seeds']:
                    a = index[seed,candidate,'risk_rfa']
                    if mode == 'unchanged_v19':
                        b = json.loads((PRIOR/f'seed{seed}__fresh__{method}/metrics.json').read_text())['final']['validation']
                    elif mode == 'historical_v11':
                        b = json.loads(controls.historical(seed,method).read_text())['final']['validation']
                    else:
                        b = index[seed,mode,method]
                    delta = {k:a[k]-b[k] for k in ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2')}
                    pairs.append(dict(seed=seed,delta=delta,passed=delta['accuracy_pct']>=-1. and delta['worst20_pct']>=1.))
                contrasts.append(dict(control_mode=mode,control_method=method,pairs=pairs,
                    passed=contrast_passes([p['delta'] for p in pairs]),standard_gaussians_paired=mode!='historical_v11'))
        candidates.append(dict(grid_index=candidate,contrasts=contrasts,passed=all(c['passed'] for c in contrasts)))
    eligible = [c['grid_index'] for c in candidates if c['passed']]
    return dict(candidates=candidates,selected=None if not eligible else eligible[0],
                eligible_for_independent_confirmation=bool(eligible),global_validation=False)


def report(config, stamp):
    rows = [json.loads((OUT/identifier(j)/'metrics.json').read_text()) for j in jobs(config) if completed(j,stamp)]
    lines = ['# V24 — allocation publique du bruit, gradients frais','',
        f'**{len(rows)}/16 runs MPS valides** ; deux seeds de calibration connues, ε≤4, aucun test final.', '',
        '| Seed | Ratio σ_fin/σ_début | Règle | Accuracy % | Worst-20 % | Gap pp | Variance pp² | Brier |',
        '|--:|--:|:--|--:|--:|--:|--:|--:|']
    for r in rows:
        j,v = r['job'],r['final']['validation']
        lines.append(f'| {j["seed"]} | {r["privacy"]["ratio"]:.6f} | {j["method"]} | {v["accuracy_pct"]:.4f} | {v["worst20_pct"]:.4f} | {v["gap_best20_worst20_pp"]:.4f} | {v["variance_pp2"]:.4f} | {v["brier_loss"]:.5f} |')
    decision = None
    if len(rows) == 16:
        decision = decide(config,rows)
        base.save(OUT/'evidence.json',dict(source_stamp=stamp,decision=decision))
        lines += ['',f'Admission à confirmation : **{"PASS" if decision["eligible_for_independent_confirmation"] else "FAIL"}**. Audit indépendant nécessaire ; aucune validation conjointe.', '']
    lines += ['','[Protocole figé](Public_Temporal_Noise_Training_V24_Protocol.md).']
    REPORT.write_text('\n'.join(lines)+'\n')
    return len(rows),decision


def worker():
    require_mps(); OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            config,profile,stamp = inputs(); manifest = dict(config=config,profile=profile,source_stamp=stamp)
            if (OUT/'manifest.json').exists():
                assert json.loads((OUT/'manifest.json').read_text()) == manifest
            else:
                base.save(OUT/'manifest.json',manifest)
            if not (OUT/'tests.json').exists():
                p = subprocess.run([sys.executable,'-m','pytest',str(TEST),'-q'],cwd=ROOT,capture_output=True,text=True)
                base.save(OUT/'tests.json',dict(passed=p.returncode==0,source_stamp=stamp,output=p.stdout+p.stderr))
            tests = json.loads((OUT/'tests.json').read_text()); assert tests['passed'] and tests['source_stamp'] == stamp
            key = json.loads((PRIOR/'simulator_secret.json').read_text())['key']
            assert ROOT/config['paired_randomness_source'] == PRIOR/'simulator_secret.json'
            for seed in config['seeds']:
                data = base.prepare(profile,seed)
                for j in [x for x in jobs(config) if x['seed'] == seed]:
                    train(j,config,profile,stamp,data,key)
                    n,_ = report(config,stamp); print(f'{n}/16 completed',flush=True)
                del data; torch.mps.empty_cache()
            n,d = report(config,stamp); assert n == 16
            base.save(OUT/'status.json',dict(status='completed',device='mps',valid_runs=16,
                eligible_for_independent_confirmation=d['eligible_for_independent_confirmation'],selected=d['selected'],next_campaign_launched=False))
        except Exception as exc:
            base.save(OUT/'status.json',dict(status='failed',device='mps',error=repr(exc),pid=os.getpid()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--resume',action='store_true',required=True)
    parser.parse_args(); worker()
