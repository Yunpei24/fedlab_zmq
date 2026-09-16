#!/usr/bin/env python3
"""Non-cumulative, calibration-only MPS replay of a public noise allocation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import secrets
import statistics as st
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_fair_objective_screen as base
from privacy.fair_objective import require_mps,per_example,release,losses
from privacy.split_risk_gradient import plan
from privacy.capped_private_risk import weights,potential
from privacy.scheduled_private_risk import aggregate
from privacy.private_risk_message_safety import sanitize
from privacy.stable_weighted_rfa import stable_norm

NAME='private_risk_channel_replay_v14'
OUT=ROOT/'results/ldp_gradient_far'/NAME
SOURCE=ROOT/'results/ldp_gradient_far/public_horizon_private_risk_v11'
REPORT=ROOT/'output/analysis/Private_Risk_Channel_Replay_V14_Analyse.md'
PROTOCOL=ROOT/'output/analysis/Private_Risk_Channel_Replay_V14_Protocol.md'
SEEDS=[170501,170502]
METHODS=['original','reallocated','oracle_risk']


def inputs():
    old=json.loads((SOURCE/'manifest.json').read_text());base.verify_stamp(old['source_stamp'])
    a=json.loads((ROOT/'output/analysis/Private_Risk_Channel_Allocation_V14_Audit.json').read_text())
    assert a['minimum_public_bound_target']==.5 and not a['promotion']
    stamp=dict(old['source_stamp'])
    paths=[Path(__file__),PROTOCOL,ROOT/'output/analysis/Private_Risk_Channel_Allocation_V14_Audit.json',
           ROOT/'privacy/private_risk_message_safety.py',ROOT/'tests/test_private_risk_channel_replay_v14.py']
    for seed in SEEDS:
        d=SOURCE/f'seed{seed}__erm_mean'
        paths.extend([d/'checkpoint.pt',d/'metrics.json'])
    stamp.update({str(p.relative_to(ROOT)):base.digest(p) for p in paths})
    return old['profile'],stamp


@torch.no_grad()
def risks(model,data):
    values=[]
    for ids in data['train']:
        total=torch.zeros((),device='mps')
        for b in ids.split(512):total+=losses(model(data['x'][b]),data['y'][b],'brier').sum()
        values.append(total/len(ids))
    return torch.stack(values)


def objective(v):
    # Host scalar reduction of already-evaluated losses, not model computation.
    return st.mean(r+r*r/.5 if r<=.5 else 3*r-.5 for r in [c['brier_loss'] for c in v['clients']])


def one(seed,replay,profile,stamp,key,data,checkpoint,rawrisk,initial):
    path=OUT/f'seed{seed}__replay{replay}.json'
    if path.exists():
        r=json.loads(path.read_text());assert r['source_stamp']==stamp and r['device']=='mps'
        return
    model=base.new_model(profile,seed);model.load_state_dict(checkpoint)
    original=plan(N=4800,b=240,T=120,C=2.,epsilon=4.,delta=1e-5,epsilon_risk=.25)
    changed=plan(N=4800,b=240,T=120,C=2.,epsilon=4.,delta=1e-5,epsilon_risk=.5)
    clean=[];noise=[];report_noise=[];batch_hashes=[]
    for cid,ids in enumerate(data['train']):
        idx=base.draw_indices(4800,240,base.seed_for(key,seed,replay,cid,'batch'))
        _,g,_,_=per_example(model,data['x'][ids[idx]],data['y'][ids[idx]],clip_norm=2.)
        mean=g.mean(0);clean.append(mean);batch_hashes.append(base.ids_hash(idx))
        noise.append(release(torch.zeros_like(mean),noise_std=1.,seed=base.seed_for(key,seed,replay,cid,'gradient')))
        report_noise.append(release(torch.zeros_like(rawrisk[cid:cid+1]),noise_std=1.,seed=base.seed_for(key,seed,replay,cid,'risk')).squeeze())
    clean=torch.stack(clean);noise=torch.stack(noise);report_noise=torch.stack(report_noise)
    wtrue,_=weights(rawrisk,.5);target=(wtrue[:,None]*clean).sum(0)
    rows=[];baseline=None
    for method in METHODS:
        model.load_state_dict(checkpoint)
        p=original if method=='original' else changed
        rr=rawrisk if method=='oracle_risk' else (rawrisk+p['risk_std']*report_noise).clamp(0,1)
        sent=clean+p['gradient_std']*noise
        sent,rr,safety=sanitize(sent,rr)
        assert safety['invalid_message_rows']==safety['nonfinite_risk_reports']==0
        step,diag=aggregate(sent,rr,kind='risk_rfa',round_number=120,horizon=120)
        a=step/.5
        if baseline is None:baseline=a.clone()
        w,_=weights(rr,.5)
        base.apply_gradient(model,step,1.)
        v=base.evaluate(model,data,'val')
        rows.append(dict(method=method,eligible_private=method!='oracle_risk',privacy_plan=p,
            mse_to_fair_clipped_batch_target=float(stable_norm(a-target).square()),
            aggregate_shift_vs_original=float(stable_norm(a-baseline)),
            risk_weight_l1_error=float((w-wtrue).abs().sum()),aggregation=diag,validation=v,
            J_gain=objective(initial)-objective(v)))
    base.save(path,dict(seed=seed,replay=replay,device='mps',source_stamp=stamp,initial=initial,rows=rows,
        batch_hashes=batch_hashes,oracle_only=True,test_evaluated=False,training_steps=0,
        target='True-risk weighted mean of clipped batch gradients, not a population gradient'))


def report(stamp):
    blocks=[json.loads(p.read_text()) for p in sorted(OUT.glob('seed*__replay*.json'))]
    assert all(b['source_stamp']==stamp and b['device']=='mps' and not b['test_evaluated'] for b in blocks)
    lines=['# V14 — réallocation du bruit : transfert réel à vérifier','',
        f'**{len(blocks)}/16 blocs MPS**, trois évaluations par bloc, sans entraînement cumulatif ni test.', '',
        '| Seed | Condition | MSE agrégat | Erreur L1 des poids | Gain J après pas | Accuracy (%) | Worst-20 (%) |',
        '|--:|:--|--:|--:|--:|--:|--:|']
    grouped=[]
    for seed in SEEDS:
        bb=[b for b in blocks if b['seed']==seed]
        if not bb:continue
        for method in METHODS:
            q=[next(r for r in b['rows'] if r['method']==method) for b in bb]
            d=dict(seed=seed,method=method,n_replays=len(q),mse=st.mean(r['mse_to_fair_clipped_batch_target'] for r in q),
                weight_l1=st.mean(r['risk_weight_l1_error'] for r in q),J_gain=st.mean(r['J_gain'] for r in q),
                accuracy=st.mean(r['validation']['accuracy_pct'] for r in q),worst20=st.mean(r['validation']['worst20_pct'] for r in q))
            grouped.append(d)
            lines.append(f"| {seed} | {method} | {d['mse']:.6f} | {d['weight_l1']:.6f} | {d['J_gain']:+.6f} | {d['accuracy']:.4f} | {d['worst20']:.4f} |")
    decision=None
    if len(blocks)==16:
        checks=[]
        for seed in SEEDS:
            a=next(r for r in grouped if r['seed']==seed and r['method']=='reallocated')
            b=next(r for r in grouped if r['seed']==seed and r['method']=='original')
            changes=dict(mse_relative_reduction=1-a['mse']/b['mse'],J_gain_delta=a['J_gain']-b['J_gain'],
                accuracy_delta_pp=a['accuracy']-b['accuracy'],worst20_delta_pp=a['worst20']-b['worst20'])
            gates=dict(mse=changes['mse_relative_reduction']>=.01,J=changes['J_gain_delta']>=0,
                       accuracy=changes['accuracy_delta_pp']>=-.1,worst20=changes['worst20_delta_pp']>=0)
            checks.append(dict(seed=seed,changes=changes,gates=gates,passed=all(gates.values())))
        decision=dict(calibration_admissible=all(c['passed'] for c in checks),checks=checks,global_validation=False)
        base.save(OUT/'evidence.json',dict(grouped=grouped,decision=decision,source_stamp=stamp))
        lines+=['',f"Justification d'une nouvelle calibration end-to-end : **{'PASS' if decision['calibration_admissible'] else 'FAIL'}**."]
        for c in checks:lines+=['',f"Seed {c['seed']} : {c['changes']} ; critères {c['gates']}."]
    lines+=['','Le risque oracle est non privé, exclusivement diagnostique. La cible est une moyenne équitable de batch, '
        'pas une preuve de fairness ni de convergence. Aucune nouvelle campagne ne suit automatiquement.', '',
        '[Protocole](Private_Risk_Channel_Replay_V14_Protocol.md).']
    REPORT.write_text('\n'.join(lines)+'\n');return len(blocks),decision


def worker():
    require_mps();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            profile,stamp=inputs();manifest=dict(source_stamp=stamp,profile=profile,seeds=SEEDS,replays=8,methods=METHODS)
            if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
            else:base.save(OUT/'manifest.json',manifest)
            tests=OUT/'tests.json'
            if not tests.exists():
                p=subprocess.run([sys.executable,'-m','pytest','tests/test_private_risk_channel_replay_v14.py','-q'],
                    cwd=ROOT,capture_output=True,text=True)
                base.save(tests,dict(passed=p.returncode==0,output=p.stdout+p.stderr,source_stamp=stamp))
            assert json.loads(tests.read_text())['passed']
            secret=OUT/'simulator_secret.json'
            if not secret.exists():
                assert not list(OUT.glob('seed*__replay*.json'))
                base.save(secret,dict(key=secrets.token_hex(32),private_release=False))
            key=json.loads(secret.read_text())['key']
            for seed in SEEDS:
                data=base.prepare(profile,seed)
                model=base.new_model(profile,seed)
                saved=torch.load(SOURCE/f'seed{seed}__erm_mean'/'checkpoint.pt',map_location='cpu',weights_only=True)
                model.load_state_dict(saved['model']);initial=base.evaluate(model,data,'val')
                assert initial==json.loads((SOURCE/f'seed{seed}__erm_mean'/'metrics.json').read_text())['final']['validation']
                raw=risks(model,data)
                for replay in range(8):
                    base.verify_stamp(stamp);require_mps()
                    base.save(OUT/'status.json',dict(status='running',device='mps',seed=seed,replay=replay,pid=os.getpid()))
                    one(seed,replay,profile,stamp,key,data,saved['model'],raw,initial)
                    count,_=report(stamp);print(f'{count}/16 blocks complete',flush=True)
                del data,model;torch.mps.empty_cache()
            count,decision=report(stamp);assert count==16
            base.save(OUT/'status.json',dict(status='completed',device='mps',valid_blocks=count,
                calibration_admissible=decision['calibration_admissible'],next_campaign_launched=False))
        except Exception as exc:
            base.save(OUT/'status.json',dict(status='failed',error=repr(exc),device='mps',pid=os.getpid()));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--launch',action='store_true')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if not args.resume or args.worker==args.launch:parser.error('Choose --worker or --launch, plus --resume')
    if args.worker:worker();return
    require_mps();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'campaign.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    log=ROOT/'logs'/f'{NAME}.log'
    with log.open('a') as f:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker','--resume'],cwd=ROOT,
            env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=proc.pid,device='mps',blocks=16,log=str(log))))


if __name__=='__main__':main()
