#!/usr/bin/env python3
"""MPS full-population causal decomposition at two frozen calibration states."""
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
from privacy.fair_objective import require_mps,per_example,release
from privacy.split_risk_gradient import plan
from privacy.capped_private_risk import weights,potential
from privacy.scheduled_private_risk import aggregate
from privacy.stable_weighted_rfa import stable_norm
from scripts.run_private_risk_channel_replay_v14 import risks

NAME='private_fairness_population_diagnostic_v15'
OUT=ROOT/'results/ldp_gradient_far'/NAME
SOURCE=ROOT/'results/ldp_gradient_far/public_horizon_private_risk_v11'
REPORT=ROOT/'output/analysis/Private_Fairness_Population_Diagnostic_V15_Analyse.md'
PROTOCOL=ROOT/'output/analysis/Private_Fairness_Population_Diagnostic_V15_Protocol.md'
SEEDS=[170501,170502]
POPULATION=['population_erm','population_fair','population_clipped_fair','population_clipped_rfa']
SAMPLED=['batch_fair','batch_rfa','private_gradient_rfa','private_both_rfa']


def inputs():
    old=json.loads((SOURCE/'manifest.json').read_text());base.verify_stamp(old['source_stamp'])
    stamp=dict(old['source_stamp'])
    paths=[Path(__file__),PROTOCOL,ROOT/'scripts/run_private_risk_channel_replay_v14.py',
           ROOT/'tests/test_private_fairness_population_diagnostic_v15.py']
    for seed in SEEDS:paths+=[SOURCE/f'seed{seed}__erm_mean'/'checkpoint.pt',SOURCE/f'seed{seed}__erm_mean'/'metrics.json']
    stamp.update({str(p.relative_to(ROOT)):base.digest(p) for p in paths})
    return old['profile'],stamp


def full_gradients(model,data):
    """Streaming per-example gradients. Never holds the entire N x d matrix."""
    raw,clipped,rr,clipfractions=[],[],[],[]
    d=sum(p.numel() for p in model.parameters() if p.requires_grad)
    for ids in data['train']:
        gs=torch.zeros(d,device='mps');cs=torch.zeros_like(gs);ls=torch.zeros((),device='mps');nf=torch.zeros((),device='mps')
        for ix in ids.split(240):
            losses,g,norms,unclipped=per_example(model,data['x'][ix],data['y'][ix],clip_norm=2.)
            gs+=len(ix)*unclipped;cs+=g.sum(0);ls+=losses.sum();nf+=(norms>2).float().sum()
        raw.append(gs/len(ids));clipped.append(cs/len(ids));rr.append(ls/len(ids));clipfractions.append(nf/len(ids))
    return dict(raw=torch.stack(raw),clipped=torch.stack(clipped),risks=torch.stack(rr),clip_fraction=torch.stack(clipfractions))


def fair_directions(risks_,raw):
    lam,a=weights(risks_,.5)
    return lam,(a[:,None]*raw).mean(0),(lam[:,None]*raw).sum(0)


def intervene(*,model,state,data,seed,condition,replay,A,target,GJ,Jbefore,initial,stamp,diag=None):
    identifier=f'seed{seed}__{condition}__r{replay}'
    path=OUT/(identifier+'.json')
    if path.exists():
        saved=json.loads(path.read_text());assert saved['source_stamp']==stamp and saved['device']=='mps'
        assert saved['vector_sha256']==base.digest(OUT/(identifier+'.pt'))
        return
    model.load_state_dict(state)
    eta=.5;predicted=float(eta*torch.dot(GJ,A))
    err=float(stable_norm(A-target).square())
    denom=stable_norm(A)*stable_norm(target)
    cosine=float(torch.dot(A,target)/denom) if float(denom)>0 else None
    base.apply_gradient(model,A,eta)
    after=base.evaluate(model,data,'val')
    train_risks_after=risks(model,data)
    Jafter=float(potential(train_risks_after,.5).mean())
    actual=Jbefore-Jafter
    vectorpath=OUT/(identifier+'.pt')
    base.checkpoint(vectorpath,dict(aggregate=A.detach().cpu(),target=target.detach().cpu(),true_J_gradient=GJ.detach().cpu(),
        source_stamp=stamp,privacy_protected=False))
    row=dict(seed=seed,condition=condition,replay=replay,device='mps',source_stamp=stamp,test_evaluated=False,
        cumulative_training_steps=0,eta=eta,initial_validation=initial,validation=after,
        J_population_before=Jbefore,J_population_after=Jafter,J_population_gain=actual,
        predicted_J_gain=predicted,taylor_remainder=predicted-actual,mse_to_unclipped_fair_population=err,
        cosine_to_unclipped_fair_population=cosine,aggregate_norm=float(stable_norm(A)),
        accuracy_delta_pp=after['accuracy_pct']-initial['accuracy_pct'],
        worst20_delta_pp=after['worst20_pct']-initial['worst20_pct'],
        algorithm_diagnostics=diag,vector_sha256=base.digest(vectorpath),oracles_not_private=True,
        mechanism_private_only_if_full_training_protocol_accounted=condition=='private_both_rfa')
    base.save(path,row)


def report(stamp):
    rows=[json.loads(p.read_text()) for p in sorted(OUT.glob('seed*__*__r*.json'))]
    assert all(r['source_stamp']==stamp and r['device']=='mps' for r in rows)
    grouped=[]
    lines=['# V15 — décomposition locale avec gradients de population','',
        f'**{len(rows)}/40 interventions MPS terminées**, validation uniquement ; deux états de calibration.', '',
        '| Seed | Direction | Répétitions | MSE vs gradient équitable exact | Gain J prédit | Gain J réel | Δ acc. (pp) | Δ Worst-20 (pp) |',
        '|--:|:--|--:|--:|--:|--:|--:|--:|']
    for seed in SEEDS:
        for condition in POPULATION+SAMPLED:
            rr=[r for r in rows if r['seed']==seed and r['condition']==condition]
            if not rr:continue
            keys=['mse_to_unclipped_fair_population','predicted_J_gain','J_population_gain','accuracy_delta_pp','worst20_delta_pp']
            g=dict(seed=seed,condition=condition,n=len(rr),**{k:st.mean(r[k] for r in rr) for k in keys})
            grouped.append(g)
            lines.append(f"| {seed} | {condition} | {len(rr)} | {g[keys[0]]:.6f} | {g[keys[1]]:+.6f} | {g[keys[2]]:+.6f} | {g[keys[3]]:+.4f} | {g[keys[4]]:+.4f} |")
    if len(rows)==40:
        for seed in SEEDS:
            assert all(sum(r['seed']==seed and r['condition']==c for r in rows)==1 for c in POPULATION)
            assert all(sum(r['seed']==seed and r['condition']==c for r in rows)==4 for c in SAMPLED)
        contrasts=[]
        for seed in SEEDS:
            for left,right,label in [('population_fair','population_erm','objectif équitable'),
                ('population_clipped_fair','population_fair','clipping'),
                ('population_clipped_rfa','population_clipped_fair','RFA sur population'),
                ('batch_fair','population_clipped_fair','sampling moyenne'),
                ('batch_rfa','batch_fair','RFA sur batch'),
                ('private_gradient_rfa','batch_rfa','bruit gradient'),
                ('private_both_rfa','private_gradient_rfa','bruit risque')]:
                a=next(g for g in grouped if g['seed']==seed and g['condition']==left)
                b=next(g for g in grouped if g['seed']==seed and g['condition']==right)
                contrasts.append(dict(seed=seed,intervention=label,treated=left,control=right,
                    delta={k:a[k]-b[k] for k in keys}))
        base.save(OUT/'evidence.json',dict(grouped=grouped,contrasts=contrasts,source_stamp=stamp,
            candidate_promoted=False,global_validation=False))
        lines+=['','## Contrastes au même état','',
                '| Seed | Intervention | Δ gain réel de J | Δ gain accuracy (pp) | Δ gain Worst-20 (pp) |',
                '|--:|:--|--:|--:|--:|']
        for c in contrasts:
            d=c['delta'];lines.append(f"| {c['seed']} | {c['intervention']} | {d['J_population_gain']:+.6f} | {d['accuracy_delta_pp']:+.4f} | {d['worst20_delta_pp']:+.4f} |")
    lines+=['','Ces contrastes ne s’additionnent pas pour expliquer une trajectoire d’entraînement. '
        'Les gradients exacts, les risques train et les interventions sans DP sont des oracles de diagnostic. '
        'Une MSE faible ne suffit pas à établir la fairness. Aucune promotion ni campagne suivante automatique.', '',
        '[Protocole et définitions](Private_Fairness_Population_Diagnostic_V15_Protocol.md).']
    REPORT.write_text('\n'.join(lines)+'\n');return len(rows)


def worker():
    require_mps();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            profile,stamp=inputs();content=dict(profile=profile,source_stamp=stamp,seeds=SEEDS,eta=.5,replays=4)
            if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==content
            else:base.save(OUT/'manifest.json',content)
            testpath=OUT/'tests.json'
            if not testpath.exists():
                p=subprocess.run([sys.executable,'-m','pytest','tests/test_private_fairness_population_diagnostic_v15.py','-q'],
                                 cwd=ROOT,capture_output=True,text=True)
                base.save(testpath,dict(passed=p.returncode==0,output=p.stdout+p.stderr,source_stamp=stamp))
            assert json.loads(testpath.read_text())['passed']
            secret=OUT/'simulator_secret.json'
            if not secret.exists():
                assert not list(OUT.glob('seed*__*__r*.json'))
                base.save(secret,dict(key=secrets.token_hex(32),private_release=False))
            key=json.loads(secret.read_text())['key']
            p=plan(N=4800,b=240,T=120,C=2.,epsilon=4.,delta=1e-5,epsilon_risk=.25)
            base.save(OUT/'diagnostic_privacy_plan.json',p)
            for seed in SEEDS:
                data=base.prepare(profile,seed);model=base.new_model(profile,seed)
                checkpoint=torch.load(SOURCE/f'seed{seed}__erm_mean'/'checkpoint.pt',map_location='cpu',weights_only=True)['model']
                model.load_state_dict(checkpoint);initial=base.evaluate(model,data,'val')
                assert initial==json.loads((SOURCE/f'seed{seed}__erm_mean'/'metrics.json').read_text())['final']['validation']
                base.save(OUT/'status.json',dict(status='running',device='mps',seed=seed,phase='full_population_gradients',pid=os.getpid()))
                popfile=OUT/f'population_seed{seed}.pt'
                if popfile.exists():
                    cache=torch.load(popfile,map_location='cpu',weights_only=True);assert cache['source_stamp']==stamp
                    pop={k:v.to('mps') for k,v in cache['population'].items()}
                else:
                    pop=full_gradients(model,data)
                    base.checkpoint(popfile,dict(population={k:v.detach().cpu() for k,v in pop.items()},source_stamp=stamp,privacy_protected=False))
                lam,GJ,target=fair_directions(pop['risks'],pop['raw'])
                Jbefore=float(potential(pop['risks'],.5).mean())
                rfa,diag=aggregate(pop['clipped'],pop['risks'],kind='risk_rfa',round_number=120,horizon=120)
                directions=[pop['raw'].mean(0),target,(lam[:,None]*pop['clipped']).sum(0),rfa/.5]
                for condition,A in zip(POPULATION,directions):
                    base.verify_stamp(stamp)
                    intervene(model=model,state=checkpoint,data=data,seed=seed,condition=condition,replay=-1,A=A,target=target,GJ=GJ,
                        Jbefore=Jbefore,initial=initial,stamp=stamp,diag=diag if condition.endswith('rfa') else None)
                    count=report(stamp);print(f'{count}/40 interventions',flush=True)
                for replay in range(4):
                    base.save(OUT/'status.json',dict(status='running',device='mps',seed=seed,phase='paired_batches',replay=replay,pid=os.getpid()))
                    model.load_state_dict(checkpoint);clean=[];noise=[];rn=[]
                    for cid,ids in enumerate(data['train']):
                        idx=base.draw_indices(4800,240,base.seed_for(key,seed,replay,cid,'batch'))
                        _,g,_,_=per_example(model,data['x'][ids[idx]],data['y'][ids[idx]],clip_norm=2.)
                        gg=g.mean(0);clean.append(gg)
                        noise.append(release(torch.zeros_like(gg),noise_std=p['gradient_std'],seed=base.seed_for(key,seed,replay,cid,'gradient')))
                        rn.append(release(torch.zeros(1,device='mps'),noise_std=p['risk_std'],seed=base.seed_for(key,seed,replay,cid,'risk')).squeeze())
                    clean=torch.stack(clean);private=clean+torch.stack(noise);rp=(pop['risks']+torch.stack(rn)).clamp(0,1)
                    for condition in SAMPLED:
                        base.verify_stamp(stamp)
                        if condition=='batch_fair':A=(lam[:,None]*clean).sum(0);diag=None
                        else:
                            A,diag=aggregate(clean if condition=='batch_rfa' else private,
                                rp if condition=='private_both_rfa' else pop['risks'],kind='risk_rfa',round_number=120,horizon=120)
                            A=A/.5
                        intervene(model=model,state=checkpoint,data=data,seed=seed,condition=condition,replay=replay,A=A,target=target,GJ=GJ,
                            Jbefore=Jbefore,initial=initial,stamp=stamp,diag=diag)
                        count=report(stamp);print(f'{count}/40 interventions',flush=True)
                del data,model,pop;torch.mps.empty_cache()
            count=report(stamp);assert count==40
            base.save(OUT/'status.json',dict(status='completed',device='mps',valid_interventions=count,next_campaign_launched=False))
        except Exception as exc:
            base.save(OUT/'status.json',dict(status='failed',device='mps',error=repr(exc),pid=os.getpid()));raise


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--worker',action='store_true');g.add_argument('--launch',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    if not a.resume:p.error('--resume required')
    if a.worker:worker();return
    require_mps();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'campaign.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (ROOT/'logs'/f'{NAME}.log').open('a') as stream:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker','--resume'],cwd=ROOT,
            env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0'),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=proc.pid,device='mps',interventions=40)))


if __name__=='__main__':main()
