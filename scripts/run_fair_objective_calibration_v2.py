#!/usr/bin/env python3
"""Isolated validation-only calibration; MPS required, no follow-up launch."""
import argparse
from contextlib import contextmanager
import fcntl
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
from scripts import run_fair_objective_screen as base
from privacy.fair_objective import require_mps,per_example,query,release,epsilon_bound

NAME='fair_objective_calibration_v2'
MATRIX=ROOT/'configs/ldp_gradient_far'/f'{NAME}.yaml'
OUT=ROOT/'results/ldp_gradient_far'/NAME
LOG=ROOT/'logs'/f'{NAME}.log'
REPORT=ROOT/'output/analysis/Fair_Objective_Calibration_V2_Status.md'


def config():
    m=yaml.safe_load(MATRIX.read_text())
    assert m['campaign_id']==NAME and m['device']=='mps'
    assert m['rounds']==60 and m['expected_runs']==24 and m['num_clients']==10
    assert m['sampling']=='fixed_without_replacement' and m['adjacency']=='replace_one'
    assert m['local_optimizer_steps']==0 and m['release_count_per_round']==1
    assert m['server_aggregation']=='uniform' and m['server_clip'] is None and m['attacks']=='none'
    assert not m['publish_test_metrics'] and not m['automatic_confirmation']
    assert set(m['calibration_seeds']).isdisjoint({170101,170201,170202,170203,170204})
    return m


def jobs(m):
    return [dict(seed=s,C=c,base_lr=lr,beta=b) for s,c,lr,b in itertools.product(
        m['calibration_seeds'],m['clip_grid'],m['base_lr_grid'],m['beta_grid'])]


def identifier(j):
    value=f"seed{j['seed']}__C{j['C']:g}__lr{j['base_lr']:g}__beta{j['beta']:g}"
    return value.replace('.','p')


def parameters(m,j):
    return dict(loss='brier',mode='erm' if j['beta']==0 else 'naive',beta=j['beta'],
        clip_norm=j['C'],server_lr=j['base_lr']/(1+m['initial_uniform_brier']*j['beta']))


def source_stamp(m):
    files=[Path(__file__),MATRIX,ROOT/m['protocol'],ROOT/'privacy/fair_objective.py',
        ROOT/'scripts/run_fair_objective_screen.py',ROOT/'models/registry.py',
        ROOT/'datasets/partitioner.py',ROOT/'tests/test_fair_objective_calibration_v2.py',
        ROOT/'tests/test_fair_objective.py']
    return {str(p.relative_to(ROOT)):base.digest(p) for p in files}


def completed(j,stamp,m):
    d=OUT/identifier(j);p=d/'orchestration_status.json'
    if not p.exists():return False
    s=json.loads(p.read_text())
    if s['status']!='completed':return False
    r=json.loads((d/'metrics.json').read_text())
    assert s['metrics_sha256']==base.digest(d/'metrics.json')
    assert r['source_stamp']==stamp and r['job']==j and r['device']=='mps'
    assert len(r['rounds'])==s['round']==m['rounds']
    assert r['privacy']['epsilon']<=m['epsilon'] and 'test' not in r['final']
    return True


def vector_distortion(raw,clipped):
    if raw.device.type!='mps' or clipped.device.type!='mps':raise ValueError('MPS vectors required')
    rn=float(torch.linalg.vector_norm(raw));cn=float(torch.linalg.vector_norm(clipped))
    err=float(torch.linalg.vector_norm(clipped-raw))
    return dict(raw_norm=rn,clipped_norm=cn,distortion_norm=err,
        distortion_relative=err/rn if rn>1e-12 else None,
        cosine=max(-1.,min(1.,float(torch.dot(raw,clipped))/(rn*cn))) if rn*cn>1e-20 else None)


def add_objectives(val,beta):
    risks=[r['brier_loss'] for r in val['clients']]
    val.update(J_beta2=st.mean(r+r*r for r in risks),
        own_objective=st.mean(r+beta/2*r*r for r in risks),
        mean_risk=st.mean(risks),risk_variance=st.pvariance(risks))
    return val


def train(m,j,data,key,stamp):
    if completed(j,stamp,m):return
    dest=OUT/identifier(j);dest.mkdir(parents=True,exist_ok=True)
    p=parameters(m,j)
    plan=base.privacy_plan(dict(m,methods={'arm':p}),'arm',m['epsilon'])
    model=base.new_model(m,j['seed']);rows=[];oracles=[];start=0
    cp=dest/'checkpoint.pt'
    if cp.exists():
        state=torch.load(cp,map_location='cpu',weights_only=True)
        assert state['source_stamp']==stamp and state['job']==j
        model.load_state_dict(state['model']);rows=state['rows'];oracles=state['oracles'];start=state['round'];initial=state['initial']
    else:initial=add_objectives(base.evaluate(model,data,'val'),j['beta'])
    base.save(dest/'public_protocol.json',dict(job=j,parameters=p,privacy=plan,
        source_stamp=stamp,split='validation_only',test_evaluated=False,
        sampling='fixed_without_replacement',local_optimizer_steps=0))
    for t in range(start,m['rounds']):
        require_mps();base.verify_stamp(stamp)
        status=dict(status='running',active=identifier(j),round=t+1,total_rounds=m['rounds'],pid=os.getpid(),device='mps')
        base.save(OUT/'status.json',status);base.save(dest/'orchestration_status.json',dict(status,job=j))
        messages=[];clean_messages=[];unclipped_queries=[];diagnostics=[]
        for cid,ids in enumerate(data['train']):
            sel=base.draw_indices(len(ids),m['batch_size'],base.seed_for(key,j['seed'],t,cid,'batch'))
            r,g,norms,raw=per_example(model,data['x'][ids[sel]],data['y'][ids[sel]],kind='brier',clip_norm=p['clip_norm'])
            q=query(r,g,population_size=len(ids),beta=j['beta'],mode=p['mode'])
            y=release(q,noise_std=plan['std'],seed=base.seed_for(key,j['seed'],t,cid,'gaussian'))
            if not bool(torch.isfinite(y).all()):raise FloatingPointError('Non-finite private gradient')
            coefficient=1+j['beta']*float(r.mean())
            rawq=coefficient*raw
            messages.append(y);clean_messages.append(q);unclipped_queries.append(rawq)
            diagnostics.append(dict(client=cid,batch_hash=base.ids_hash(sel),
                loss_mean=float(r.mean()),coefficient=coefficient,
                clip_fraction=float((norms>p['clip_norm']).float().mean()),
                gradient_norm_mean=float(norms.mean()),query_norm=float(torch.linalg.vector_norm(q)),
                **vector_distortion(raw,g.mean(0))))
        applied=torch.stack(messages).mean(0)
        clean=torch.stack(clean_messages).mean(0);raw=torch.stack(unclipped_queries).mean(0)
        aggdiag=vector_distortion(raw,clean)
        noise_norm=float(torch.linalg.vector_norm(applied-clean))
        oracles.append(dict(round=t+1,clients=diagnostics,aggregate_clip=aggdiag,
            aggregate_dp_noise_norm=noise_norm,
            noise_to_clean_signal=noise_norm/aggdiag['clipped_norm'] if aggdiag['clipped_norm']>1e-12 else None))
        # Diagnostics above are never used to alter the update or its parameters.
        base.apply_gradient(model,applied,p['server_lr'])
        row=dict(round=t+1,device='mps',validation=None,noise_std=plan['std'],
            epsilon_at_round=epsilon_bound(q=plan['q'],z=plan['z'],steps=t+1,delta=m['delta'])[0])
        if t+1 in m['evaluation_rounds']:
            row['validation']=add_objectives(base.evaluate(model,data,'val'),j['beta'])
        rows.append(row)
        base.checkpoint(cp,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            rows=rows,oracles=oracles,round=t+1,initial=initial,source_stamp=stamp,job=j))
        detail='' if row['validation'] is None else f" val_acc={row['validation']['accuracy_pct']:.2f}% val_W20={row['validation']['worst20_pct']:.2f}%"
        print(f'{identifier(j)} {t+1}/{m["rounds"]}'+detail,flush=True)
    base.save(dest/'metrics.json',dict(job=j,parameters=p,privacy=plan,device='mps',
        initial=initial,rounds=rows,final=rows[-1],splits=data['splits'],source_stamp=stamp,
        test_evaluated=False,research_diagnostics_not_private_release_bundle=True))
    base.save(dest/'simulator_oracle.json',dict(privacy_protected=False,feeds_mechanism=False,rounds=oracles))
    base.save(dest/'orchestration_status.json',dict(status='completed',job=j,device='mps',
        round=m['rounds'],metrics_sha256=base.digest(dest/'metrics.json')))


def report(m,stamp):
    rr=[]
    for j in jobs(m):
        if completed(j,stamp,m):
            r=json.loads((OUT/identifier(j)/'metrics.json').read_text());v=r['final']['validation']
            o=json.loads((OUT/identifier(j)/'simulator_oracle.json').read_text())
            ds=[x for t in o['rounds'] for x in t['clients']]
            rr.append(dict(j,r=r,v=v,clip=100*st.mean(x['clip_fraction'] for x in ds),
                distortion=st.mean(x['distortion_relative'] for x in ds if x['distortion_relative'] is not None)))
    lines=['# Calibration objectif équitable v2 — validation uniquement','',
        f'**{len(rr)}/24 runs terminés.** MPS ; 60 tours ; epsilon final ≤ 4. Aucun score de test, aucune phase suivante automatique.', '',
        'Grille : C = 0,5 / 1 / 2 ; pas de base = 1 / 2 ; beta = 0 / 2 ; seeds 170301 et 170302. Le pas serveur vaut pas de base / (1 + 0,45 beta).', '',
        '| Seed | C | Pas de base | Beta | Val. acc (%) | Val. Worst-20 (%) | Val. gap (pp) | Val. J_beta2 | Clip (%) | Distorsion relative moyenne du batch |',
        '|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|']
    for x in rr:
        v=x['v'];lines.append(f"| {x['seed']} | {x['C']} | {x['base_lr']} | {x['beta']} | {v['accuracy_pct']:.2f} | {v['worst20_pct']:.2f} | {v['gap_best20_worst20_pp']:.2f} | {v['J_beta2']:.5f} | {x['clip']:.2f} | {x['distortion']:.4f} |")
    lines+=['','Les trajectoires et oracles vectoriels sont sauvegardés par run. Les valeurs à T=20/40 proviennent du bruit calibré pour T=60 et ne sont pas des budgets epsilon=4 indépendants.', '',
        'Une amélioration sur ces validations ne constitue pas une confirmation indépendante. Le protocole ne sélectionne aucun gagnant automatiquement.', '',
        '[Protocole](Fair_Objective_Calibration_V2_Protocol.md)']
    REPORT.write_text('\n'.join(lines)+'\n')
    return len(rr)


@contextmanager
def lock():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'campaign.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Campaign already active; no duplicate')
        yield


def worker(m):
    require_mps()
    with lock():
        try:
            stamp=source_stamp(m);manifest=OUT/'manifest.json'
            if manifest.exists():assert json.loads(manifest.read_text())['source_stamp']==stamp
            else:base.save(manifest,dict(config=m,source_stamp=stamp,device='mps',torch_version=torch.__version__,created=time.time()))
            base.save(OUT/'status.json',dict(status='preflight',device='mps',pid=os.getpid(),planned_runs=24))
            report(m,stamp)
            a=OUT/'tests.json'
            if not a.exists():
                proc=subprocess.run([sys.executable,'-m','pytest','tests/test_fair_objective.py',
                    'tests/test_fair_objective_calibration_v2.py','-q'],cwd=ROOT,capture_output=True,text=True)
                base.save(a,dict(passed=proc.returncode==0,output=proc.stdout+proc.stderr,source_stamp=stamp,device='mps'))
                if proc.returncode:raise RuntimeError('Preflight tests failed')
            else:
                test=json.loads(a.read_text());assert test['passed'] and test['source_stamp']==stamp
            print('Preflight passed; validation-only calibration on MPS, 24 runs.',flush=True)
            k=OUT/'simulator_secret.json'
            if not k.exists():base.save(k,dict(key=secrets.token_hex(32),not_public=True))
            key=json.loads(k.read_text())['key']
            for seed in m['calibration_seeds']:
                js=[j for j in jobs(m) if j['seed']==seed]
                if all(completed(j,stamp,m) for j in js):continue
                data=base.prepare(m,seed)
                for j in js:train(m,j,data,key,stamp);report(m,stamp)
                del data;torch.mps.empty_cache()
            count=report(m,stamp)
            assert count==24
            base.save(OUT/'status.json',dict(status='completed',valid_runs=count,device='mps',test_evaluated=False,no_followup_launched=True))
        except Exception as exc:
            fail=dict(status='failed',error=repr(exc),pid=os.getpid(),time=time.time())
            base.save(OUT/'failure.json',fail);base.save(OUT/'status.json',fail);raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--worker',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--status',action='store_true');p.add_argument('--plan',action='store_true')
    a=p.parse_args();m=config()
    if a.status:
        print((OUT/'status.json').read_text() if (OUT/'status.json').exists() else 'not started');return
    if a.plan:print(json.dumps(dict(runs=len(jobs(m)),config=m),indent=2));return
    if not a.resume:p.error('--resume required')
    if a.launch:
        require_mps()
        with lock():pass
        LOG.parent.mkdir(parents=True,exist_ok=True)
        with LOG.open('a') as f:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker','--resume'],
                cwd=ROOT,env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=proc.pid,log=str(LOG),device='mps',planned_runs=24)));return
    if a.worker:worker(m);return
    p.error('Choose --launch, --worker, --plan or --status')


if __name__=='__main__':main()
