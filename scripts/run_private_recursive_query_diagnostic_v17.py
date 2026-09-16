#!/usr/bin/env python3
"""MPS feasibility of pre-noise, per-example-clipped temporal increments."""
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_fair_objective_screen as base
from scripts import run_private_clipping_step_diagnostic_v16 as prior
from privacy.fair_objective import require_mps,per_example
from privacy.stable_weighted_rfa import stable_norm

NAME='private_recursive_query_diagnostic_v17'
OUT=ROOT/'results/ldp_gradient_far'/NAME
PROTOCOL=ROOT/'output/analysis/Private_Recursive_Query_V17_Protocol_and_Derivation.md'
REPORT=ROOT/'output/analysis/Private_Recursive_Query_V17_Analyse.md'
TEST=ROOT/'tests/test_private_recursive_query_diagnostic_v17.py'
SEEDS=[170501,170502]
THETAS=[.1,.2,.5]
RATIOS=[.025,.05,.1,.2,.4]


def inputs():
    previous=json.loads((prior.OUT/'manifest.json').read_text());base.verify_stamp(previous['source_stamp'])
    audit=ROOT/'output/analysis/Private_Clipping_Step_V16_Independent_Audit.json'
    assert json.loads(audit.read_text())['audit_passed']
    stamp=dict(previous['source_stamp'])
    files=[Path(__file__),PROTOCOL,TEST,audit]
    files += [prior.OUT/f'seed{s}__r{r}__C2_eta0.5__risk_rfa.pt' for s in SEEDS for r in range(4)]
    stamp.update({str(p.relative_to(ROOT)):base.digest(p) for p in files})
    return previous['profile'],stamp


def clipped_increment(now,before,D):
    if now.device.type!='mps' or before.device.type!='mps' or now.shape!=before.shape or now.ndim!=2:
        raise ValueError('Matching per-example MPS gradients required')
    if not math.isfinite(D) or D<=0:raise ValueError('Positive public D')
    change=now-before;n=torch.linalg.vector_norm(change,dim=1)
    return change*(D/n.clamp_min(1e-20)).clamp(max=1)[:,None],n


def public_ratios(theta,ratio,t=20):
    if not 0<theta<=1 or not 0<ratio<=2 or t<0:raise ValueError('Invalid public parameters')
    a=1-theta;r=theta+(1-theta)*ratio
    stationary=r*r/(theta*(2-theta))
    return dict(fresh_noise_std_ratio=r,stationary_noise_variance_ratio=stationary,
                finite_noise_variance_ratio=a**(2*t)+stationary*(1-a**(2*t)))


def quantile(values,p):
    v=sorted(values);x=(len(v)-1)*p;k=int(x)
    return v[k] if k==len(v)-1 else v[k]+(x-k)*(v[k+1]-v[k])


def summarize(rows):
    groups=[];checks=[]
    for theta in THETAS:
        for ratio in RATIOS:
            pub=public_ratios(theta,ratio);local=[]
            for seed in SEEDS:
                blocks=[r for r in rows if r['seed']==seed]
                if not blocks:continue
                values=[next(v for v in b['variants'] if v['theta']==theta and v['D_over_C']==ratio) for b in blocks]
                bs=[v['coherent_bias_proxy_relative'] for v in values]
                cs=[v['fraction_clipped'] for v in values]
                g=dict(seed=seed,theta=theta,D_over_C=ratio,n=len(blocks),**pub,
                    median_bias=st.median(bs),p90_bias=quantile(bs,.9),median_clip=st.median(cs),
                    fraction_blocks_below_10pct_clip=st.mean(c<=.1 for c in cs))
                g['passed']=g['n']==40 and g['fraction_blocks_below_10pct_clip']>=.9 and g['median_bias']<=.1 and g['p90_bias']<=.25
                groups.append(g);local.append(g)
            checks.append(dict(theta=theta,D_over_C=ratio,**pub,
                passed=len(local)==2 and all(g['passed'] for g in local) and pub['stationary_noise_variance_ratio']<=.5 and pub['finite_noise_variance_ratio']<=.55))
    accepted=sorted((c for c in checks if c['passed']),key=lambda c:(-c['theta'],-c['D_over_C']))
    selected={k:accepted[0][k] for k in ('theta','D_over_C')} if accepted else None
    return groups,dict(checks=checks,selected_for_calibration=selected,global_validation=False,next_training_launched=False)


def report(stamp):
    files=sorted(OUT.glob('seed*__r*__client*.json'));rows=[json.loads(p.read_text()) for p in files]
    for p,r in zip(files,rows):
        assert r['source_stamp']==stamp and r['device']=='mps' and not r['test_evaluated']
        assert base.digest(p.with_suffix('.pt'))==r['vectors_sha256']
    gs,decision=summarize(rows)
    lines=['# V17 — faisabilité d’une requête récursive avant bruit','',f'**{len(rows)}/80 blocs de gradients appariés**, MPS, calibration seule.', '',
        '| Seed | θ | D/C | V∞/Vbase | V20/Vbase | Clip médian (%) | Blocs clip≤10 % (%) | Biais relatif médian | Biais relatif p90 | Critères locaux |',
        '|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|']
    for g in gs:
        lines.append(f"| {g['seed']} | {g['theta']:g} | {g['D_over_C']:g} | {g['stationary_noise_variance_ratio']:.4f} | {g['finite_noise_variance_ratio']:.4f} | {100*g['median_clip']:.2f} | {100*g['fraction_blocks_below_10pct_clip']:.1f} | {g['median_bias']:.4f} | {g['p90_bias']:.4f} | {g['passed']} |")
    if len(rows)==80:
        assert {(r['seed'],r['replay'],r['client']) for r in rows}=={(s,p,c) for s in SEEDS for p in range(4) for c in range(10)}
        base.save(OUT/'evidence.json',dict(source_stamp=stamp,groups=gs,decision=decision))
        lines+=['',f"Couple admis à une éventuelle calibration : **{decision['selected_for_calibration'] or 'aucun'}**."]
    lines+=['','La variance concerne uniquement la composante linéaire des Gaussiennes. Le proxy de biais est estimé sur un batch, et ne certifie ni le biais de population ni la covariance de RFA. Aucune accuracy finale, fairness ou robustesse n’est validée par ce screen. Les tests de modèle et les attaques restent nécessaires.', '',
        '[Dérivation, hypothèses et critères préenregistrés](Private_Recursive_Query_V17_Protocol_and_Derivation.md).']
    REPORT.write_text('\n'.join(lines)+'\n');return len(rows),decision


def worker():
    require_mps();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            profile,stamp=inputs();manifest=dict(profile=profile,source_stamp=stamp,seeds=SEEDS,theta=THETAS,D_over_C=RATIOS,blocks=80)
            if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
            else:base.save(OUT/'manifest.json',manifest)
            if not (OUT/'tests.json').exists():
                p=subprocess.run([sys.executable,'-m','pytest',str(TEST),'-q'],cwd=ROOT,capture_output=True,text=True)
                base.save(OUT/'tests.json',dict(passed=p.returncode==0,source_stamp=stamp,output=p.stdout+p.stderr))
            tests=json.loads((OUT/'tests.json').read_text());assert tests['passed'] and tests['source_stamp']==stamp
            kp=OUT/'simulator_secret.json'
            if not kp.exists():
                assert not list(OUT.glob('seed*__r*__client*.json'))
                base.save(kp,dict(key=secrets.token_hex(32),privacy_protected=False))
            key=json.loads(kp.read_text())['key']
            for seed in SEEDS:
                data=base.prepare(profile,seed);oldmodel=base.new_model(profile,seed);model=base.new_model(profile,seed)
                state=torch.load(prior.population.SOURCE/f'seed{seed}__erm_mean'/'checkpoint.pt',map_location='cpu',weights_only=True)['model']
                oldmodel.load_state_dict(state)
                for replay in range(4):
                    step=torch.load(prior.OUT/f'seed{seed}__r{replay}__C2_eta0.5__risk_rfa.pt',map_location='mps',weights_only=True)['step']
                    model.load_state_dict(state);base.apply_gradient(model,step,1.)
                    for cid,ids in enumerate(data['train']):
                        base.verify_stamp(stamp)
                        file=OUT/f'seed{seed}__r{replay}__client{cid}.json'
                        if file.exists():
                            r=json.loads(file.read_text());assert r['source_stamp']==stamp and base.digest(file.with_suffix('.pt'))==r['vectors_sha256'];continue
                        base.save(OUT/'status.json',dict(status='running',device='mps',seed=seed,replay=replay,client=cid,pid=os.getpid()))
                        ix=base.draw_indices(4800,240,base.seed_for(key,seed,replay,cid,'new_batch'))
                        _,before,_,_=per_example(oldmodel,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)
                        _,now,_,_=per_example(model,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)
                        targetnorm=max(float(stable_norm(now.mean(0))),1e-6);variants=[];increment_means={}
                        for ratio in RATIOS:
                            delta,norms=clipped_increment(now,before,2*ratio)
                            increment_means[str(ratio)]=delta.mean(0).detach().cpu()
                            bnorm=float(stable_norm((delta-(now-before)).mean(0)))
                            frac=float((norms>2*ratio).float().mean())
                            for theta in THETAS:
                                variants.append(dict(theta=theta,D_over_C=ratio,**public_ratios(theta,ratio),
                                    increment_bias_norm=bnorm,fraction_clipped=frac,
                                    coherent_bias_proxy_relative=(1-theta)/theta*bnorm/targetnorm))
                        vp=file.with_suffix('.pt')
                        base.checkpoint(vp,dict(before_mean=before.mean(0).detach().cpu(),now_mean=now.mean(0).detach().cpu(),
                            increment_means=increment_means,increment_norms=norms.detach().cpu(),
                            indices=ix.detach().cpu(),source_stamp=stamp,privacy_protected=False))
                        base.save(file,dict(seed=seed,replay=replay,client=cid,device='mps',source_stamp=stamp,
                            batch_hash=base.ids_hash(ix),variants=variants,current_batch_mean_gradient_norm=targetnorm,
                            test_evaluated=False,training_launched=False,oracles_not_private=True,vectors_sha256=base.digest(vp)))
                        count,_=report(stamp);print(f'{count}/80 paired client blocks complete',flush=True)
                del data,model,oldmodel;torch.mps.empty_cache()
            count,d=report(stamp);assert count==80
            base.save(OUT/'status.json',dict(status='completed',device='mps',valid_blocks=count,
                selected_for_calibration=d['selected_for_calibration'],next_training_launched=False))
        except Exception as exc:
            base.save(OUT/'status.json',dict(status='failed',device='mps',error=repr(exc),pid=os.getpid()));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true',required=True)
    p.add_argument('--resume',action='store_true',required=True);p.parse_args();worker()


if __name__=='__main__':main()
