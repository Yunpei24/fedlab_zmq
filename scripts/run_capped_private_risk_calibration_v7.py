#!/usr/bin/env python3
"""Finite clean calibration: bounded risk emphasis x clipping, MPS-only."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
import yaml
from scripts import run_fair_objective_screen as base
from scripts import run_split_risk_gradient_fixed_step_v6 as parent
from scripts.run_fair_objective_calibration_v2 import add_objectives
from privacy.fair_objective import require_mps,per_example,release,epsilon_bound,wor_rdp
from privacy.split_risk_gradient import plan,private_risk
from privacy.capped_private_risk import aggregate

NAME='capped_private_risk_calibration_v7'
MATRIX=ROOT/'configs/ldp_gradient_far'/f'{NAME}.yaml'
OUT=ROOT/'results/ldp_gradient_far'/NAME
LOG=ROOT/'logs'/f'{NAME}.log'
REPORT=ROOT/'output/analysis/Capped_Private_Risk_Calibration_V7_Status.md'
TESTS=['tests/test_fair_objective.py','tests/test_split_risk_gradient.py','tests/test_capped_private_risk.py']


def config():
    m=yaml.safe_load(MATRIX.read_text())
    assert m['campaign_id']==NAME and m['device']=='mps' and m['expected_runs']==24
    assert m['clips']==[1.,2.] and m['risk_scales']==[.5,.25] and m['eta']==2
    assert m['calibration_seeds']==[170501,170502] and len(m['reserved_confirmation_seeds'])==4
    assert set(m['calibration_seeds']).isdisjoint(m['reserved_confirmation_seeds'])
    assert not m['test_evaluated'] and not m['automatic_attacks'] and not m['automatic_confirmation']
    assert m['attacks']=='none' and m['server_clip'] is None and m['local_optimizer_steps']==0
    return m


def arms(m):
    out={f'{kind}_C{C:g}':dict(kind=kind,C=C,scale=None) for C in m['clips'] for kind in ('erm_mean','erm_rfa')}
    out.update({f'{kind}_C{C:g}_scale{s:g}':dict(kind=kind,C=C,scale=s) for C in m['clips'] for s in m['risk_scales'] for kind in ('risk_mean','risk_rfa')})
    return out


def jobs(m):return [dict(seed=s,arm=a) for s in m['calibration_seeds'] for a in arms(m)]
def identifier(j):return f"seed{j['seed']}__{j['arm']}"


def source_stamp(m):
    files=[Path(__file__),MATRIX,ROOT/m['protocol'],ROOT/'privacy/capped_private_risk.py',ROOT/'privacy/split_risk_gradient.py',
        ROOT/'privacy/fair_objective.py',Path(base.__file__),Path(parent.__file__),
        ROOT/'privacy/split_risk_gradient_fixed_step.py',ROOT/'scripts/run_split_risk_gradient_v5.py',
        ROOT/'scripts/run_fair_objective_calibration_v2.py',ROOT/'datasets/registry.py',ROOT/'models/registry.py',
        ROOT/'datasets/partitioner.py',*[ROOT/p for p in TESTS]]
    return {str(p.relative_to(ROOT)):base.digest(p) for p in files}


def privacy(m,arm):
    a=arms(m)[arm]
    if a['kind'].startswith('erm_'):
        pp=dict(clip_norm=a['C'],beta=0.,mode='erm')
        p=base.privacy_plan(dict(m,methods={'control':pp}),'control',m['epsilon'])
        p.update(epsilon_realized=p['epsilon'],gradient_std=p['std'],risk_std=0.,risk_releases=0)
        return p
    return plan(N=m['public_train_size'],b=m['batch_size'],T=m['rounds'],C=a['C'],
        epsilon=m['epsilon'],delta=m['delta'],epsilon_risk=m['epsilon_risk'])


def epsilon_at(p,t,m,kind):
    if kind.startswith('erm_'):return epsilon_bound(q=p['q'],z=p['z'],steps=t,delta=m['delta'])[0]
    import math
    return min(t*wor_rdp(a,p['b']/p['N'],p['gradient_z'])+t*a/(2*p['risk_z']**2)+math.log(1/m['delta'])/(a-1) for a in range(2,65))


def completed(j,m,stamp):
    d=OUT/identifier(j)
    if not (d/'orchestration_status.json').exists():return False
    s=json.loads((d/'orchestration_status.json').read_text())
    if s['status']!='completed':return False
    r=json.loads((d/'metrics.json').read_text())
    assert r['source_stamp']==stamp and r['job']==j and r['device']=='mps' and not r['test_evaluated']
    assert s['metrics_sha256']==base.digest(d/'metrics.json') and r['privacy']['epsilon_realized']<=4
    assert [t['round'] for t in r['rounds']]==list(range(1,61))
    assert all(t['device']=='mps' and t['epsilon_realized']<=4 and t['aggregation']['eta']==2 for t in r['rounds'])
    assert [t['round'] for t in r['rounds'] if t['validation'] is not None]==m['evaluation_rounds']
    return True


def train(m,j,data,key,stamp):
    if completed(j,m,stamp):return
    d=OUT/identifier(j);d.mkdir(parents=True,exist_ok=True)
    a=arms(m)[j['arm']];p=privacy(m,j['arm']);model=base.new_model(m,j['seed'])
    rows,oracle,start=[],[],0;cp=d/'checkpoint.pt'
    if cp.exists():
        state=torch.load(cp,map_location='cpu',weights_only=True)
        assert state['source_stamp']==stamp and state['job']==j
        model.load_state_dict(state['model'])
        rows,oracle,start,initial=state['rows'],state['oracle'],state['round'],state['initial']
    else:initial=add_objectives(base.evaluate(model,data,'val'),2.)
    base.save(d/'public_protocol.json',dict(job=j,arm=a,privacy=p,config=m,source_stamp=stamp))
    for t in range(start,m['rounds']):
        require_mps();base.verify_stamp(stamp)
        status=dict(status='running',active=identifier(j),round=t+1,total_rounds=m['rounds'],device='mps',pid=os.getpid())
        base.save(OUT/'status.json',status);base.save(d/'orchestration_status.json',status)
        messages,reports,local=[],[],[]
        for cid,ids in enumerate(data['train']):
            rr,raw=None,None
            if a['kind'].startswith('risk_'):
                rr,raw=private_risk(model,data['x'][ids],data['y'][ids],noise_std=p['risk_std'],
                    seed=base.seed_for(key,j['seed'],t,cid,'risk'),N=m['public_train_size'])
                reports.append(rr)
            idx=base.draw_indices(len(ids),m['batch_size'],base.seed_for(key,j['seed'],t,cid,'batch'))
            _,g,norms,_=per_example(model,data['x'][ids[idx]],data['y'][ids[idx]],clip_norm=a['C'])
            messages.append(release(g.mean(0),noise_std=p['gradient_std'],seed=base.seed_for(key,j['seed'],t,cid,'gaussian')))
            local.append(dict(client=cid,batch_hash=base.ids_hash(idx),clip_fraction=float((norms>a['C']).float().mean()),
                private_risk=None if rr is None else float(rr),raw_risk_research_only=None if raw is None else float(raw)))
        u,diag=aggregate(torch.stack(messages),None if not reports else torch.stack(reports),kind=a['kind'],scale=a['scale'],eta=m['eta'])
        if not bool(torch.isfinite(u).all()):raise FloatingPointError('Nonfinite update')
        base.apply_gradient(model,u,1.)
        v=add_objectives(base.evaluate(model,data,'val'),2.) if t+1 in m['evaluation_rounds'] else None
        if v is not None and a['scale'] is not None:
            c=a['scale'];risks=[x['brier_loss'] for x in v['clients']]
            v['capped_risk_objective']=sum(r+(r*r/c if r<=c else 2*r-c) for r in risks)/len(risks)
        rows.append(dict(round=t+1,device='mps',epsilon_realized=epsilon_at(p,t+1,m,a['kind']),aggregation=diag,validation=v))
        oracle.append(dict(round=t+1,clients=local))
        base.checkpoint(cp,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            rows=rows,oracle=oracle,round=t+1,initial=initial,source_stamp=stamp,job=j))
        detail='' if v is None else f" val_acc={v['accuracy_pct']:.2f}% W20={v['worst20_pct']:.2f}%"
        print(f'{identifier(j)} {t+1}/{m["rounds"]}'+detail,flush=True)
    base.save(d/'metrics.json',dict(job=j,arm=a,source_stamp=stamp,device='mps',privacy=p,
        initial=initial,rounds=rows,final=rows[-1],splits=data['splits'],test_evaluated=False,
        validation_and_oracles_not_private=True))
    base.save(d/'simulator_oracle.json',dict(privacy_protected=False,rounds=oracle))
    base.save(d/'orchestration_status.json',dict(status='completed',device='mps',job=j,round=60,metrics_sha256=base.digest(d/'metrics.json')))


def decide(m,rows):
    gates={}
    for candidate in [name for name,a in arms(m).items() if a['kind'].startswith('risk_')]:
        comparisons=[]
        for control in m['screen']['controls']:
            for seed in m['calibration_seeds']:
                a=next(r['final']['validation'] for r in rows if r['job']==dict(seed=seed,arm=candidate))
                b=next(r['final']['validation'] for r in rows if r['job']==dict(seed=seed,arm=control))
                da,dw=a['accuracy_pct']-b['accuracy_pct'],a['worst20_pct']-b['worst20_pct']
                comparisons.append(dict(seed=seed,control=control,accuracy_delta_pp=da,worst20_delta_pp=dw,
                    passed=da>=-m['screen']['maximum_accuracy_loss_pp'] and dw>=m['screen']['minimum_worst20_advantage_pp']))
        gates[candidate]=dict(passed=all(c['passed'] for c in comparisons),comparisons=comparisons)
    selected=next((a for a in m['screen']['robust_selection_order'] if gates[a]['passed']),None)
    return dict(gates=gates,selected_robust_candidate=selected,confirmatory=False)


def report(m,stamp):
    rows=[json.loads((OUT/identifier(j)/'metrics.json').read_text()) for j in jobs(m) if completed(j,m,stamp)]
    lines=['# Risque privé plafonné — calibration v7','',f'**{len(rows)}/24 runs valides**, MPS, validation seulement.', '',
        '[Protocole préenregistré](Capped_Private_Risk_Calibration_V7_Protocol.md)','',
        '| Seed | Règle | Accuracy (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) | Brier |',
        '|--:|:--|--:|--:|--:|--:|--:|']
    for r in rows:
        v=r['final']['validation'];j=r['job']
        lines.append(f"| {j['seed']} | {j['arm']} | {v['accuracy_pct']:.3f} | {v['worst20_pct']:.3f} | {v['gap_best20_worst20_pp']:.3f} | {v['variance_pp2']:.3f} | {v['brier_loss']:.5f} |")
    decision=None
    if len(rows)==24:
        decision=decide(m,rows)
        base.save(OUT/'evidence.json',dict(rows=rows,decision=decision,source_stamp=stamp))
        lines+=['','## Critères fixés','']
        for name,g in decision['gates'].items():lines.append(f"- {name} : "+('passe' if g['passed'] else 'échoue')+'.')
        lines+=['',f"Candidate robuste sélectionnée pour une éventuelle confirmation : {decision['selected_robust_candidate'] or 'aucune'}.",
                '', 'Ce résultat de calibration n’est ni une confirmation indépendante ni un test d’attaque.']
    REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text('\n'.join(lines)+'\n')
    return len(rows),decision


@contextmanager
def lock():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'campaign.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Already active; no duplicate')
        yield


def worker(m):
    require_mps()
    with lock():
        try:
            old=json.loads((parent.OUT/'manifest.json').read_text());base.verify_stamp(old['source_stamp'])
            assert json.loads((parent.OUT/'status.json').read_text())['status']=='completed'
            assert all(parent.completed(j,old['config'],old['source_stamp']) for j in parent.jobs(old['config']))
            stamp=source_stamp(m);manifest=OUT/'manifest.json'
            if manifest.exists():assert json.loads(manifest.read_text())['source_stamp']==stamp
            else:base.save(manifest,dict(config=m,source_stamp=stamp,device='mps',created_at=time.time(),parent_manifest_sha256=base.digest(parent.OUT/'manifest.json')))
            base.save(OUT/'status.json',dict(status='preflight',device='mps',pid=os.getpid()))
            tests=OUT/'tests.json'
            if not tests.exists():
                p=subprocess.run([sys.executable,'-m','pytest',*TESTS,'-q'],cwd=ROOT,capture_output=True,text=True)
                base.save(tests,dict(passed=p.returncode==0,output=p.stdout+p.stderr,source_stamp=stamp))
            test=json.loads(tests.read_text());assert test['passed'] and test['source_stamp']==stamp
            secret=OUT/'simulator_secret.json';parentkey=json.loads((ROOT/m['paired_randomness_source']).read_text())['key']
            if not secret.exists():base.save(secret,dict(key=parentkey,not_public=True,paired_with=m['parent_campaign']))
            key=json.loads(secret.read_text())['key'];assert key==parentkey
            report(m,stamp)
            for seed in m['calibration_seeds']:
                js=[j for j in jobs(m) if j['seed']==seed]
                if all(completed(j,m,stamp) for j in js):continue
                data=base.prepare(m,seed)
                for j in js:train(m,j,data,key,stamp);report(m,stamp)
                del data;torch.mps.empty_cache()
            count,decision=report(m,stamp);assert count==24
            for seed in m['calibration_seeds']:
                baseline=json.loads((OUT/identifier(dict(seed=seed,arm='erm_mean_C1'))/'metrics.json').read_text())
                previous=json.loads((parent.OUT/f'seed{seed}__erm_full'/'metrics.json').read_text())
                for metric in ('accuracy_pct','worst20_pct','brier_loss'):
                    assert abs(baseline['final']['validation'][metric]-previous['final']['validation'][metric])<1e-6
                patterns=[]
                for j in jobs(m):
                    if j['seed']!=seed:continue
                    r=json.loads((OUT/identifier(j)/'metrics.json').read_text())
                    assert r['initial']==baseline['initial'] and r['splits']==baseline['splits']
                    o=json.loads((OUT/identifier(j)/'simulator_oracle.json').read_text())
                    patterns.append([[c['batch_hash'] for c in t['clients']] for t in o['rounds']])
                assert all(p==patterns[0] for p in patterns)
            base.save(OUT/'pairing_audit.json',dict(passed=True,erm_parent_reproduced=True,all_batches_and_splits_paired=True))
            base.save(OUT/'status.json',dict(status='completed',device='mps',valid_runs=24,
                selected_robust_candidate=decision['selected_robust_candidate'],no_confirmation_or_attacks_launched=True))
        except Exception as exc:
            failure=dict(status='failed',error=repr(exc),pid=os.getpid(),time=time.time())
            base.save(OUT/'failure.json',failure);base.save(OUT/'status.json',failure)
            raise


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for flag in ('launch','worker','status','plan'):g.add_argument('--'+flag,action='store_true')
    p.add_argument('--resume',action='store_true');a=p.parse_args();m=config()
    if a.plan:print(json.dumps(dict(config=m,arms=arms(m),jobs=jobs(m)),indent=2));return
    if a.status:
        print((OUT/'status.json').read_text() if (OUT/'status.json').exists() else 'not started')
        if (OUT/'manifest.json').exists():
            stamp=json.loads((OUT/'manifest.json').read_text())['source_stamp'];base.verify_stamp(stamp)
            print(json.dumps(dict(valid_runs=sum(completed(j,m,stamp) for j in jobs(m)),expected=24)))
        return
    if not a.resume:p.error('--resume required')
    if a.worker:worker(m);return
    require_mps()
    with lock():pass
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open('a') as f:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker','--resume'],cwd=ROOT,
            env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=proc.pid,device='mps',runs=24,log=str(LOG))))


if __name__=='__main__':main()
