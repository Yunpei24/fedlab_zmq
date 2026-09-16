#!/usr/bin/env python3
"""Small, gated, MPS-only screen of a debiased private client-fairness gradient.

Launch: venv/bin/python scripts/run_fair_objective_screen.py --launch --resume
No Byzantine/recursive phase is started. All historic runs are untouched.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import secrets
import statistics as st
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
import yaml
from torchvision.datasets import FashionMNIST
from datasets.partitioner import partition_dataset
from models.registry import get_model
from privacy.fair_objective import (require_mps,per_example,losses,query,sensitivity,
                                   calibrate,epsilon_bound,release)

CAMPAIGN='fair_objective_screen_v1'
MATRIX=ROOT/'configs/ldp_gradient_far'/f'{CAMPAIGN}.yaml'
OUT=ROOT/'results/ldp_gradient_far'/CAMPAIGN
LOG=ROOT/'logs'/f'{CAMPAIGN}.log'
REPORT=ROOT/'output/analysis/Fair_Objective_Screen_V1_Status.md'


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        os.fchmod(stream.fileno(),0o600)
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def checkpoint(path,value):
    temporary=path.with_name(path.name+'.tmp')
    torch.save(value,temporary);os.chmod(temporary,0o600);os.replace(temporary,path)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config():
    m=yaml.safe_load(MATRIX.read_text())
    assert m['campaign_id']==CAMPAIGN and m['device']=='mps'
    assert m['num_clients']==10 and m['public_train_size']+m['validation_size']==6000
    assert m['rounds']==20 and m['local_gradient_releases_per_round']==1 and m['local_optimizer_steps']==0
    assert m['privacy_adjacency']=='replace_one' and m['server_clip'] is None
    assert m['server_aggregation']=='uniform' and m['attacks']=='none'
    assert m['calibration_seed'] not in m['evaluation_seeds'] and len(set(m['evaluation_seeds']))==4
    assert list(m['methods'])==['ce','brier_erm','brier_naive','brier_unbiased']
    return m


def source_stamp(m):
    paths=[Path(__file__),MATRIX,ROOT/'privacy/fair_objective.py',ROOT/'models/registry.py',
           ROOT/'datasets/partitioner.py',ROOT/m['protocol'],ROOT/'tests/test_fair_objective.py']
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def verify_stamp(stamp):
    for name,value in stamp.items():
        if digest(ROOT/name)!=value:raise RuntimeError('Frozen source/configuration changed: '+name)


def seed_for(key,*parts):
    return int.from_bytes(hashlib.sha256((key+'/'+'/'.join(map(str,parts))).encode()).digest()[:8],'big')%(2**63-1)


def draw_indices(n,b,seed):
    state=torch.mps.get_rng_state()
    try:
        torch.mps.manual_seed(seed)
        return torch.randperm(n,device='mps')[:b]
    finally:torch.mps.set_rng_state(state)


def ids_hash(ids):
    return hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest()


def prepare(m,seed):
    # Dataset construction/partition bookkeeping is host I/O; all tensor
    # inference, per-example gradients, clipping and noise run on MPS.
    train=FashionMNIST(ROOT/'data',train=True,download=False)
    test=FashionMNIST(ROOT/'data',train=False,download=False)
    kwargs=dict(num_clients=m['num_clients'],partition=m['partition'],alpha=m['dirichlet_beta'],seed=seed)
    train_parts=partition_dataset(train,**kwargs)
    test_parts=partition_dataset(test,**kwargs)
    x=(train.data.to('mps').float().unsqueeze(1)/255-.2860)/.3530
    y=train.targets.to('mps')
    xt=(test.data.to('mps').float().unsqueeze(1)/255-.2860)/.3530
    yt=test.targets.to('mps')
    train_ids=[];val_ids=[];test_ids=[];audits=[]
    for cid,(tr,te) in enumerate(zip(train_parts,test_parts)):
        assert len(tr)==6000 and len(te)==1000
        parent=torch.tensor(tr.indices,device='mps',dtype=torch.long)
        perm=draw_indices(6000,6000,seed_for(str(seed),'split',cid))
        ti=parent[perm[:m['public_train_size']]]
        vi=parent[perm[m['public_train_size']:]]
        ei=torch.tensor(te.indices,device='mps',dtype=torch.long)
        train_ids.append(ti);val_ids.append(vi);test_ids.append(ei)
        audits.append(dict(client=cid,N=len(ti),validation=len(vi),test=len(ei),
                           train_sha256=ids_hash(ti),val_sha256=ids_hash(vi),test_sha256=ids_hash(ei)))
    return dict(x=x,y=y,xt=xt,yt=yt,train=train_ids,val=val_ids,test=test_ids,splits=audits)


def new_model(m,seed):
    torch.manual_seed(seed);torch.mps.manual_seed(seed)
    model=get_model(m['model'],m['dataset']).to('mps').eval()
    return model


@torch.no_grad()
def apply_gradient(model,gradient,eta):
    offset=0
    for p in model.parameters():
        if p.requires_grad:
            n=p.numel();p.sub_(eta*gradient[offset:offset+n].view_as(p));offset+=n
    assert offset==gradient.numel()


@torch.no_grad()
def evaluate(model,data,which):
    xx,yy=(data['xt'],data['yt']) if which=='test' else (data['x'],data['y'])
    rows=[]
    for ids in data[which]:
        hits=0;ce=0.;br=0.;class_hits=torch.zeros(10,device='mps');class_count=torch.zeros(10,device='mps')
        for batch in ids.split(256):
            out=model(xx[batch]);lab=yy[batch];pred=out.argmax(1)
            hits+=int((pred==lab).sum());ce+=float(losses(out,lab,'ce').sum());br+=float(losses(out,lab,'brier').sum())
            for k in range(10):
                class_count[k]+=(lab==k).sum();class_hits[k]+=((lab==k)&(pred==lab)).sum()
        present=class_count>0
        rows.append(dict(accuracy=hits/len(ids),ce_loss=ce/len(ids),brier_loss=br/len(ids),
                         balanced_accuracy=float((class_hits[present]/class_count[present]).mean()),
                         class_count=class_count.cpu().tolist(),class_hits=class_hits.cpu().tolist(),N=len(ids)))
    acc=[r['accuracy'] for r in rows];tail=max(1,math.ceil(.2*len(rows)));sort=sorted(acc)
    return dict(accuracy_pct=100*st.mean(acc),client_accuracy_pct=100*st.mean(acc),
                worst20_pct=100*st.mean(sort[:tail]),gap_best20_worst20_pp=100*(st.mean(sort[-tail:])-st.mean(sort[:tail])),
                gap_best_worst_pp=100*(max(acc)-min(acc)),variance_pp2=10000*st.pvariance(acc),
                ce_loss=st.mean(r['ce_loss'] for r in rows),brier_loss=st.mean(r['brier_loss'] for r in rows),
                balanced_accuracy_pct=100*st.mean(r['balanced_accuracy'] for r in rows),clients=rows)


def privacy_plan(m,method,epsilon,N=None,b=None):
    N=N or m['public_train_size'];b=b or m['batch_size']
    p=m['methods'][method]
    delta_query=sensitivity(population_size=N,batch_size=b,clip_norm=p['clip_norm'],beta=p['beta'],mode=p['mode'])
    if epsilon is None:return dict(enabled=False,sensitivity=delta_query,std=0.,z=None,epsilon=None,delta=None)
    z=calibrate(q=b/N,steps=m['rounds'],epsilon=epsilon,delta=m['delta'])
    realized,order=epsilon_bound(q=b/N,z=z,steps=m['rounds'],delta=m['delta'])
    assert realized<=epsilon
    return dict(enabled=True,sensitivity=delta_query,std=z*delta_query,z=z,epsilon=realized,
                delta=m['delta'],order=order,q=b/N,N=N,b=b,steps=m['rounds'],
                accountant='WBK2019 generic Theorem 9 + basic RDP conversion',
                scope='ideal per-client per-run message mechanism; not oracle exports or cross-run releases')


def identifier(job):
    budget='nodp' if job['epsilon'] is None else 'eps4'
    return f"{job['phase']}__seed{job['seed']}__{job['method']}__{budget}"


def all_jobs(m):
    calibration=[dict(phase='calibration',seed=m['calibration_seed'],method=a,epsilon=None) for a in ('ce','brier_erm')]
    evaluation=[dict(phase='evaluation',seed=s,method=a,epsilon=e)
                for s,e,a in itertools.product(m['evaluation_seeds'],m['budgets'],m['methods'])]
    assert len(evaluation)==32
    return calibration,evaluation


def completed(job,stamp):
    d=OUT/identifier(job);p=d/'orchestration_status.json'
    if not p.exists() or json.loads(p.read_text()).get('status')!='completed':return False
    meta=json.loads(p.read_text());metrics=json.loads((d/'metrics.json').read_text())
    if metrics['source_stamp']!=stamp or meta['metrics_sha256']!=digest(d/'metrics.json'):
        raise RuntimeError('Existing completed run integrity mismatch')
    if len(metrics['rounds'])!=20 or metrics['device']!='mps':raise RuntimeError('Invalid completed run')
    return True


def train(m,job,data,key,stamp):
    dest=OUT/identifier(job);dest.mkdir(parents=True,exist_ok=True)
    if completed(job,stamp):return
    p=m['methods'][job['method']];plan=privacy_plan(m,job['method'],job['epsilon'])
    model=new_model(m,job['seed']);rows=[];oracles=[];start=0
    cp=dest/'checkpoint.pt'
    if cp.exists():
        state=torch.load(cp,map_location='cpu',weights_only=True)
        assert state['source_stamp']==stamp and state['job']==job
        model.load_state_dict(state['model']);rows=state['rows'];oracles=state['oracle_rows'];start=state['round']
        initial=state['initial']
    else:initial=dict(validation=evaluate(model,data,'val'),test=evaluate(model,data,'test'))
    save(dest/'public_protocol.json',dict(job=job,parameters=p,privacy=plan,source_stamp=stamp,
         local_steps=0,release_count_per_round=1,augmentation='none',server_clip=None,aggregation='uniform'))
    t0=time.time()
    for t in range(start,m['rounds']):
        verify_stamp(stamp);require_mps()
        save(dest/'orchestration_status.json',dict(status='running',job=job,round=t+1,pid=os.getpid(),device='mps'))
        messages=[];private_diagnostics=[]
        for cid,ids in enumerate(data['train']):
            sel=draw_indices(len(ids),m['batch_size'],seed_for(key,job['seed'],t,cid,'batch'))
            r,g,norms,_=per_example(model,data['x'][ids[sel]],data['y'][ids[sel]],kind=p['loss'],clip_norm=p['clip_norm'])
            clean=query(r,g,population_size=len(ids),beta=p['beta'],mode=p['mode'])
            message=release(clean,noise_std=plan['std'],seed=seed_for(key,job['seed'],t,cid,'gaussian'))
            if not bool(torch.isfinite(message).all()):raise FloatingPointError('Non-finite message')
            messages.append(message)
            private_diagnostics.append(dict(client=cid,batch_hash=ids_hash(sel),
                clip_fraction=float((norms>p['clip_norm']).float().mean()),
                raw_gradient_norm_mean=float(norms.mean()),query_norm=float(torch.linalg.vector_norm(clean)),
                loss_mean=float(r.mean()),private_diagnostic_not_sent=True))
        applied=torch.stack(messages).mean(0)
        apply_gradient(model,applied,p['server_lr'])
        row=dict(round=t+1,device='mps',noise_std=plan['std'],sensitivity=plan['sensitivity'],
                 aggregate_norm=float(torch.linalg.vector_norm(applied)),validation=None,test=None)
        if t+1 in (1,5,10,15,20):
            row['validation']=evaluate(model,data,'val');row['test']=evaluate(model,data,'test')
        rows.append(row);oracles.append(dict(round=t+1,clients=private_diagnostics))
        checkpoint(cp,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},rows=rows,
                          oracle_rows=oracles,round=t+1,initial=initial,source_stamp=stamp,job=job))
        save(OUT/'status.json',dict(status='running',active=identifier(job),round=t+1,total_rounds=m['rounds'],
                                  device='mps',pid=os.getpid()))
        detail='' if row['test'] is None else f" test={row['test']['accuracy_pct']:.2f}% W20={row['test']['worst20_pct']:.2f}%"
        print(f"{identifier(job)} {t+1}/{m['rounds']}{detail}",flush=True)
    value=dict(job=job,device='mps',source_stamp=stamp,initial=initial,rounds=rows,
               final=rows[-1],privacy=plan,splits=data['splits'],elapsed_seconds=time.time()-t0,
               research_evaluation_not_a_private_release_bundle=True)
    save(dest/'metrics.json',value)
    save(dest/'simulator_oracle.json',dict(privacy_protected=False,feeds_mechanism=False,rounds=oracles))
    save(dest/'orchestration_status.json',dict(status='completed',job=job,round=m['rounds'],device='mps',
          metrics_sha256=digest(dest/'metrics.json'),pid=os.getpid()))


def mechanism_audit(m,data,key,stamp):
    path=OUT/'mechanism_audit.json'
    if path.exists():
        old=json.loads(path.read_text());assert old['source_stamp']==stamp
        return old
    host=dict(phase='calibration',seed=m['calibration_seed'],method='brier_erm',epsilon=None)
    state=torch.load(OUT/identifier(host)/'checkpoint.pt',map_location='cpu',weights_only=True)
    model=new_model(m,m['calibration_seed']);model.load_state_dict(state['model'])
    a=m['audit'];N=a['population_per_client'];beta=a['beta'];records=[];pools=[]
    candidates=['erm','naive','unbiased','per_example','independent_batches']
    cohort={(b,k):[] for b,k in itertools.product(a['batches'],candidates)}
    for cid,ids in enumerate(data['train']):
        selected=draw_indices(len(ids),N,seed_for(key,'audit_pool',cid))
        pools.append(ids[selected])
        r,g,norms,raw=per_example(model,data['x'][ids[selected]],data['y'][ids[selected]],clip_norm=a['clip_norm'])
        R=r.mean();mean=g.mean(0);rg=(r[:,None]*g).mean(0)
        target=(1+beta*R)*mean;true_target=(1+beta*R)*raw
        norm2=float(target.square().sum())
        for b in a['batches']:
            exact_bias=beta*(N-b)/(b*(N-1))*(rg-R*mean)
            costs={k:[] for k in candidates}
            for repeat in range(a['monte_carlo_repeats']):
                s=draw_indices(N,b,seed_for(key,'audit_batch',cid,b,repeat))
                s2=draw_indices(N,b,seed_for(key,'audit_loss_batch',cid,b,repeat))
                qs={k:query(r[s],g[s],population_size=N,beta=0. if k=='erm' else beta,mode=k) for k in candidates[:-1]}
                qs['independent_batches']=(1+beta*r[s2].mean())*g[s].mean(0)
                if repeat==0:
                    for k,v in qs.items():cohort[b,k].append(v.clone())
                for k,v in qs.items():costs[k].append(float((v-target).square().sum()))
            dp={}
            for mode in candidates[:-1]:
                bb=0. if mode=='erm' else beta
                sensitivity_value=sensitivity(population_size=N,batch_size=b,clip_norm=a['clip_norm'],beta=bb,mode=mode)
                z=calibrate(q=b/N,steps=m['rounds'],epsilon=4.,delta=m['delta'])
                std=z*sensitivity_value
                dp[mode]=dict(sensitivity=sensitivity_value,std=std,expected_gaussian_mse=g.shape[1]*std**2,
                              expected_total_mse=st.mean(costs[mode])+g.shape[1]*std**2)
            records.append(dict(client=cid,N_diagnostic=N,b=b,dimension=g.shape[1],risk=float(R),
                target_norm=math.sqrt(norm2),clip_fraction=float((norms>a['clip_norm']).float().mean()),
                clipping_bias_norm=float(torch.linalg.vector_norm(target-true_target)),
                exact_naive_bias_norm=float(torch.linalg.vector_norm(exact_bias)),
                exact_naive_bias_relative=float(torch.linalg.vector_norm(exact_bias))/max(math.sqrt(norm2),1e-12),
                exact_unbiased_bias=0.,monte_carlo_sampling_mse={k:st.mean(v) for k,v in costs.items()},
                corrected_minus_naive_mse=st.mean(x-y for x,y in zip(costs['unbiased'],costs['naive'])),
                dp_at_diagnostic_N_not_training_N=dp,
                independent_batches_privacy_claim=False))
        print(f'mechanism audit {cid+1}/{m["num_clients"]} clients',flush=True)
    @torch.no_grad()
    def objective():
        risks=[]
        for pool in pools:
            risks.append(sum(float(losses(model(data['x'][s]),data['y'][s]).sum()) for s in pool.split(256))/N)
        return dict(J=st.mean(r+beta*r*r/2 for r in risks),mean_risk=st.mean(risks),
                    variance_risk=st.pvariance(risks),worst20_risk=st.mean(sorted(risks,reverse=True)[:2]))
    initial=objective();steps=[];eta=m['methods']['brier_unbiased']['server_lr']
    for (b,k),vectors in cohort.items():
        for dp in (False,True):
            if dp and k=='independent_batches':continue
            model.load_state_dict(state['model'])
            if dp:
                delta_query=sensitivity(population_size=N,batch_size=b,clip_norm=a['clip_norm'],
                                        beta=0 if k=='erm' else beta,mode=k)
                z=calibrate(q=b/N,steps=m['rounds'],epsilon=4.,delta=m['delta'])
                vectors_= [release(v,noise_std=z*delta_query,seed=seed_for(key,'audit_step_noise',b,i)) for i,v in enumerate(vectors)]
            else:vectors_=vectors
            applied=torch.stack(vectors_).mean(0);apply_gradient(model,applied,eta)
            after=objective()
            steps.append(dict(b=b,method=k,dp=dp,common_step=eta,initial=initial,after=after,
                              delta_J=after['J']-initial['J'],aggregate_norm=float(torch.linalg.vector_norm(applied))))
    result=dict(status='completed',source_stamp=stamp,device='mps',records=records,one_step_risk=steps,
        target='gradient of client quadratic risk on fixed 512-example training-only diagnostic pools',
        privacy_protected=False,feeds_training=False,independent_batches='non-private diagnostic control only')
    save(path,result);return result


def health_gate(m,stamp):
    c,e=all_jobs(m)[0]
    cm=json.loads((OUT/identifier(c)/'metrics.json').read_text())
    em=json.loads((OUT/identifier(e)/'metrics.json').read_text())
    ce=cm['final']['validation']['accuracy_pct'];br=em['final']['validation']['accuracy_pct']
    gain=br-em['initial']['validation']['accuracy_pct']
    tests=json.loads((OUT/'math_audit.json').read_text())
    checks=dict(math_tests=tests['passed'],sources_match=tests['source_stamp']==stamp,
                brier_learns=gain>=m['gate']['brier_validation_accuracy_gain_min_pp'],
                brier_not_strongly_inferior=br>=ce-m['gate']['brier_vs_ce_validation_max_drop_pp'])
    result=dict(promote=all(checks.values()),checks=checks,brier_validation_accuracy=br,
                ce_validation_accuracy=ce,brier_gain_pp=gain,selection_uses_test=False,
                warning='single calibration seed learnability sanity gate, not a novelty/fairness result')
    save(OUT/'health_gate.json',result);return result


def make_report(m):
    _,eval_jobs=all_jobs(m);rows=[]
    for p in OUT.glob('*/metrics.json'):
        x=json.loads(p.read_text());j=x['job'];f=x['final']['test'];v=x['final']['validation']
        rows.append((j,x,f,v))
    lines=['# Gradient privé équitable : écran court','',
           'Les trajectoires de calibration et d’évaluation sont séparées. Le mécanisme est local au niveau exemple, replace-one, avec un batch fixe sans remise. Les diagnostics bruts sont des oracles de simulation, pas des sorties DP.','',
           '| Phase | Seed | Méthode | Budget | Test Acc. (%) | Worst-20 (%) | Variance (pp²) | Gap B20-W20 (pp) | Loss CE | Loss Brier |',
           '|:--|--:|:--|:--|--:|--:|--:|--:|--:|--:|']
    for j,x,f,v in sorted(rows,key=lambda z:(z[0]['phase'],z[0]['seed'],str(z[0]['epsilon']),z[0]['method'])):
        eps='sans DP' if j['epsilon'] is None else f"{x['privacy']['epsilon']:.6f}"
        lines.append(f"| {j['phase']} | {j['seed']} | {j['method']} | {eps} | {f['accuracy_pct']:.2f} | {f['worst20_pct']:.2f} | {f['variance_pp2']:.2f} | {f['gap_best20_worst20_pp']:.2f} | {f['ce_loss']:.4f} | {f['brier_loss']:.4f} |")
    eval_rows=[r for r in rows if r[0]['phase']=='evaluation']
    if eval_rows:
        lines+=['','## Moyenne ± écart-type entre seeds indépendantes','',
                '| Méthode | Budget | Seeds | Accuracy (%) | Worst-20 (%) | Gap B20-W20 (pp) | Variance (pp²) |',
                '|:--|:--|--:|:--|:--|:--|:--|']
        for name,eps in itertools.product(m['methods'],m['budgets']):
            rr=[r[2] for r in eval_rows if r[0]['method']==name and r[0]['epsilon']==eps]
            if not rr:continue
            cells=[]
            for metric in ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2'):
                values=[r[metric] for r in rr]
                cells.append(f'{st.mean(values):.2f} ± {st.stdev(values):.2f}' if len(values)>1 else f'{values[0]:.2f} (SD indisponible)')
            lines.append(f"| {name} | {'sans DP' if eps is None else 'epsilon=4'} | {len(rr)}/4 | "+' | '.join(cells)+' |')
        lines+=['','## Différences appariées : corrigé moins contrôle','',
                '| Contrôle | Budget | Paires | Delta accuracy (pp) | Delta Worst-20 (pp) |',
                '|:--|:--|--:|:--|:--|']
        for control,eps in itertools.product(('brier_erm','brier_naive','ce'),m['budgets']):
            mapping={(j['seed'],j['method']):f for j,_,f,_ in eval_rows if j['epsilon']==eps}
            pairs=[(mapping[s,'brier_unbiased'],mapping[s,control]) for s in m['evaluation_seeds']
                   if (s,'brier_unbiased') in mapping and (s,control) in mapping]
            if not pairs:continue
            cells=[]
            for metric in ('accuracy_pct','worst20_pct'):
                values=[a[metric]-b[metric] for a,b in pairs]
                value=f'{st.mean(values):+.2f}'
                if len(values)>1:value+=f' ± {st.stdev(values):.2f}'
                if len(values)==4:
                    half=3.182446305284263*st.stdev(values)/2
                    value+=f' ; IC95 [{st.mean(values)-half:+.2f}, {st.mean(values)+half:+.2f}]'
                cells.append(value)
            lines.append(f"| {control} | {'sans DP' if eps is None else 'epsilon=4'} | {len(pairs)}/4 | "+' | '.join(cells)+' |')
        lines+=['','IC95 de Student sur quatre différences par seed, descriptif et sans correction de comparaisons multiples. Les tours ne servent pas de réplications.']
    gate=OUT/'health_gate.json'
    if gate.exists():
        g=json.loads(gate.read_text())
        lines+=['','## Gate de faisabilité propre','',f"Promotion de l’écran propre : **{'oui' if g['promote'] else 'non'}**.",
                f"Validation Brier : {g['brier_validation_accuracy']:.2f} % ; CE : {g['ce_validation_accuracy']:.2f} %. Gain Brier depuis l’initialisation : {g['brier_gain_pp']:.2f} pp.",
                'Le gate ne sélectionne ni learning rate ni clip sur les résultats de test. Un échec arrête la grille de 32 runs ; il ne réfute pas toutes les losses équitables.']
    audit=OUT/'mechanism_audit.json'
    if audit.exists():
        a=json.loads(audit.read_text())
        lines+=['','## Biais d’objectif : vrais gradients Fashion-MNIST','',
                'Population diagnostique fixe : 512 exemples d’entraînement par client. La cible est celle de ces pools, pas une estimation prétendument exacte sur les 4 800 exemples locaux. Le biais de clipping est séparé du biais de mini-batch.', '',
                '| Batch | Biais naïf / norme cible, médiane (%) | Différence MSE corrigé - naïf, moyenne |',
                '|--:|--:|--:|']
        for b in m['audit']['batches']:
            rr=[r for r in a['records'] if r['b']==b]
            lines.append(f"| {b} | {100*st.median(r['exact_naive_bias_relative'] for r in rr):.6f} | {st.mean(r['corrected_minus_naive_mse'] for r in rr):.8g} |")
        lines+=['','Une valeur négative de la dernière colonne favorise le corrigé sur la MSE d’échantillonnage. Cela ne constitue pas encore un gain d’accuracy ou de fairness. Les énergies de bruit du JSON sont recalibrées pour N=512 ; ne pas les confondre avec celles du training N=4800.']
        lines+=['','### Variation réelle du risque après un pas commun','',
                'Un seul tirage apparié par condition : diagnostic local, pas une trajectoire indépendante. Delta J négatif signifie une diminution de la cible sur les pools d’entraînement.', '',
                '| Batch | Méthode | Bruit | Delta J |', '|--:|:--|:--|--:|']
        for row in a['one_step_risk']:
            lines.append(f"| {row['b']} | {row['method']} | {'epsilon=4 (N=512)' if row['dp'] else 'sans DP'} | {row['delta_J']:+.6f} |")
    finished=sum(1 for j,_,_,_ in rows if j['phase']=='evaluation')
    lines+=['','## État et limites','',f'Évaluation indépendante : {finished}/{len(eval_jobs)} runs terminés.',
            'Ni récursion PriSMA, ni attaque, ni nouvelle référence robuste n’est lancée dans cet écran. Les prochaines phases requièrent une lecture des gains et coûts sur le modèle.',
            '',f'Résultats : `{OUT}`.']
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    # Generated report, not hand-authored content.
    REPORT.write_text('\n'.join(lines)+'\n')


@contextmanager
def campaign_lock():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'campaign.lock').open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Campaign already active; do not duplicate')
        yield


def worker(m):
    require_mps()
    with campaign_lock():
        stamp=source_stamp(m)
        manifest=OUT/'manifest.json'
        if manifest.exists():assert json.loads(manifest.read_text())['source_stamp']==stamp
        else:save(manifest,dict(campaign=CAMPAIGN,source_stamp=stamp,config=m,created=time.time(),
                               torch_version=torch.__version__,python_version=sys.version,device='mps',fallback=0))
        keyfile=OUT/'simulator_secret.json'
        if not keyfile.exists():save(keyfile,dict(key=secrets.token_hex(32),not_a_public_seed=True))
        key=json.loads(keyfile.read_text())['key']
        audit_path=OUT/'math_audit.json'
        if not audit_path.exists():
            proc=subprocess.run([sys.executable,'-m','pytest','tests/test_fair_objective.py','-q'],cwd=ROOT,
                                capture_output=True,text=True)
            save(audit_path,dict(passed=proc.returncode==0,returncode=proc.returncode,
                                output=proc.stdout+proc.stderr,source_stamp=stamp,device='mps'))
            if proc.returncode:raise RuntimeError('Math/privacy tests failed; no training launched')
        else:
            a=json.loads(audit_path.read_text());assert a['passed'] and a['source_stamp']==stamp
        cal,evaluation=all_jobs(m)
        data=prepare(m,m['calibration_seed'])
        for j in cal:train(m,j,data,key,stamp);make_report(m)
        mechanism_audit(m,data,key,stamp)
        gate=health_gate(m,stamp);make_report(m)
        if not gate['promote']:
            save(OUT/'status.json',dict(status='stopped_health_gate',gate=gate,
                                      calibration_completed=2,evaluation_completed=0,device='mps'))
            print('Health gate failed; 32-run evaluation NOT launched.',flush=True);return
        del data;torch.mps.empty_cache()
        for seed in m['evaluation_seeds']:
            if all(completed(j,stamp) for j in evaluation if j['seed']==seed):continue
            data=prepare(m,seed)
            for j in evaluation:
                if j['seed']==seed:train(m,j,data,key,stamp);make_report(m)
            del data;torch.mps.empty_cache()
        save(OUT/'status.json',dict(status='completed',calibration_completed=2,evaluation_completed=32,
                                  device='mps',no_followup_phase_launched=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--worker',action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--status',action='store_true')
    args=parser.parse_args();m=config()
    if args.status:
        print((OUT/'status.json').read_text() if (OUT/'status.json').exists() else 'not launched');return
    if not args.resume:parser.error('--resume is mandatory')
    if args.launch:
        require_mps();OUT.mkdir(parents=True,exist_ok=True);LOG.parent.mkdir(parents=True,exist_ok=True)
        with campaign_lock():pass
        with LOG.open('a') as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker','--resume'],
                  cwd=ROOT,env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0'),stdout=log,stderr=subprocess.STDOUT,
                  start_new_session=True)
        print(json.dumps(dict(pid=proc.pid,log=str(LOG),output=str(OUT),device='mps')));return
    if args.worker:
        try:worker(m)
        except Exception as exc:
            save(OUT/'failure.json',dict(status='failed',error=repr(exc),pid=os.getpid(),time=time.time()))
            save(OUT/'status.json',dict(status='failed',error=repr(exc),pid=os.getpid(),time=time.time()))
            raise
        return
    parser.error('Specify --launch, --worker or --status')


if __name__=='__main__':main()
