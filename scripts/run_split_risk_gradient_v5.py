#!/usr/bin/env python3
"""Fresh private risk + standard private gradient, clean MPS calibration only."""
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
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
import yaml
from scripts import run_fair_objective_screen as base
from scripts.run_fair_objective_calibration_v2 import add_objectives
from privacy.fair_objective import require_mps,per_example,release,epsilon_bound,wor_rdp
from privacy.split_risk_gradient import plan,private_risk,aggregate

NAME='split_risk_gradient_v5'
MATRIX=ROOT/'configs/ldp_gradient_far'/f'{NAME}.yaml'
OUT=ROOT/'results/ldp_gradient_far'/NAME
LOG=ROOT/'logs'/f'{NAME}.log'
REPORT=ROOT/'output/analysis/Split_Risk_Gradient_V5_Status.md'
TESTS=['tests/test_fair_objective.py','tests/test_split_risk_gradient.py']


def config():
    m=yaml.safe_load(MATRIX.read_text())
    assert m['campaign_id']==NAME and m['device']=='mps'
    assert m['calibration_seeds']==[170501,170502] and m['expected_runs']==10
    assert m['risk_refresh_every']==1 and not m['test_evaluated']
    assert m['attacks']=='none' and m['server_clip'] is None
    assert not m['automatic_attacks'] and m['local_optimizer_steps']==0
    return m


def jobs(m):
    return [dict(seed=s,method=a) for s in m['calibration_seeds'] for a in m['methods']]


def identifier(j):return f"seed{j['seed']}__{j['method']}"


def source_stamp(m):
    paths=[Path(__file__),MATRIX,ROOT/m['protocol'],ROOT/'privacy/split_risk_gradient.py',
        ROOT/'privacy/fair_objective.py',Path(base.__file__),
        ROOT/'scripts/run_fair_objective_calibration_v2.py',ROOT/'models/registry.py',
        ROOT/'datasets/registry.py',ROOT/'datasets/partitioner.py',*[ROOT/p for p in TESTS]]
    return {str(p.relative_to(ROOT)):base.digest(p) for p in paths}


def privacy(m,method):
    if method=='erm_full':
        params=dict(clip_norm=m['clip_norm'],beta=0.,mode='erm')
        p=base.privacy_plan(dict(m,methods={'arm':params}),'arm',m['epsilon'])
        p.update(epsilon_realized=p['epsilon'],gradient_std=p['std'],risk_std=0.,risk_releases=0)
        return p
    return plan(N=m['public_train_size'],b=m['batch_size'],T=m['rounds'],C=m['clip_norm'],
        epsilon=m['epsilon'],delta=m['delta'],epsilon_risk=m['epsilon_risk'])


def completed(j,m,stamp):
    d=OUT/identifier(j)
    if not (d/'orchestration_status.json').exists():return False
    s=json.loads((d/'orchestration_status.json').read_text())
    if s['status']!='completed':return False
    r=json.loads((d/'metrics.json').read_text())
    assert s['metrics_sha256']==base.digest(d/'metrics.json')
    assert r['source_stamp']==stamp and r['job']==j and r['device']=='mps'
    assert len(r['rounds'])==m['rounds'] and [x['round'] for x in r['rounds']]==list(range(1,m['rounds']+1))
    assert r['privacy']['epsilon_realized']<=m['epsilon'] and not r['test_evaluated']
    assert r['privacy']['delta']==m['delta']
    assert all(x['device']=='mps' and x['epsilon_realized']<=m['epsilon'] for x in r['rounds'])
    assert [x['round'] for x in r['rounds'] if x['validation'] is not None]==m['evaluation_rounds']
    assert r['final']['validation'] is not None
    return True


def epsilon_at(p,t,m,method):
    if method=='erm_full':return epsilon_bound(q=p['q'],z=p['z'],steps=t,delta=m['delta'])[0]
    import math
    return min(t*wor_rdp(a,p['b']/p['N'],p['gradient_z'])+t*a/(2*p['risk_z']**2)+math.log(1/m['delta'])/(a-1)
               for a in range(2,65))


def train(m,j,data,key,stamp):
    if completed(j,m,stamp):return
    d=OUT/identifier(j)
    d.mkdir(parents=True,exist_ok=True)
    model=base.new_model(m,j['seed'])
    p=privacy(m,j['method'])
    rows,oracle,start=[],[],0
    cp=d/'checkpoint.pt'
    if cp.exists():
        state=torch.load(cp,map_location='cpu',weights_only=True)
        assert state['source_stamp']==stamp and state['job']==j
        model.load_state_dict(state['model'])
        rows,oracle,start,initial=state['rows'],state['oracle'],state['round'],state['initial']
    else:initial=add_objectives(base.evaluate(model,data,'val'),m['beta'])
    base.save(d/'public_protocol.json',dict(job=j,privacy=p,config=m,source_stamp=stamp))
    for t in range(start,m['rounds']):
        require_mps()
        base.verify_stamp(stamp)
        status=dict(status='running',active=identifier(j),round=t+1,total_rounds=m['rounds'],pid=os.getpid(),device='mps')
        base.save(OUT/'status.json',status)
        base.save(d/'orchestration_status.json',status)
        messages,reports,local=[],[],[]
        for cid,ids in enumerate(data['train']):
            noisy_risk,raw_risk=None,None
            if j['method']!='erm_full':
                noisy_risk,raw_risk=private_risk(model,data['x'][ids],data['y'][ids],
                    noise_std=p['risk_std'],seed=base.seed_for(key,j['seed'],t,cid,'risk'),N=m['public_train_size'])
                reports.append(noisy_risk)  # only this private value goes to aggregation
            idx=base.draw_indices(len(ids),m['batch_size'],base.seed_for(key,j['seed'],t,cid,'batch'))
            _,g,norms,_=per_example(model,data['x'][ids[idx]],data['y'][ids[idx]],kind='brier',clip_norm=m['clip_norm'])
            msg=release(g.mean(0),noise_std=p['gradient_std'],seed=base.seed_for(key,j['seed'],t,cid,'gaussian'))
            messages.append(msg)
            local.append(dict(client=cid,batch_hash=base.ids_hash(idx),
                clip_fraction=float((norms>m['clip_norm']).float().mean()),
                risk_report=None if noisy_risk is None else float(noisy_risk),
                raw_risk_research_only=None if raw_risk is None else float(raw_risk)))
        u,diag=aggregate(torch.stack(messages),None if not reports else torch.stack(reports),
            method=j['method'],beta=m['beta'],eta_erm=m['eta_erm'],eta_fair=m['eta_fair'])
        if not bool(torch.isfinite(u).all()):raise FloatingPointError('Non-finite parameter update')
        base.apply_gradient(model,u,1.)
        row=dict(round=t+1,device='mps',epsilon_realized=epsilon_at(p,t+1,m,j['method']),
            aggregation=diag,validation=None)
        if t+1 in m['evaluation_rounds']:
            row['validation']=add_objectives(base.evaluate(model,data,'val'),m['beta'])
        rows.append(row)
        oracle.append(dict(round=t+1,clients=local))
        base.checkpoint(cp,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            rows=rows,oracle=oracle,round=t+1,initial=initial,source_stamp=stamp,job=j))
        detail='' if row['validation'] is None else f" val_acc={row['validation']['accuracy_pct']:.2f}% W20={row['validation']['worst20_pct']:.2f}%"
        print(f'{identifier(j)} {t+1}/{m["rounds"]}'+detail,flush=True)
    base.save(d/'metrics.json',dict(job=j,device='mps',source_stamp=stamp,privacy=p,
        initial=initial,rounds=rows,final=rows[-1],splits=data['splits'],test_evaluated=False,
        research_validation_not_private_export=True))
    base.save(d/'simulator_oracle.json',dict(privacy_protected=False,
        server_uses_only_private_risk_reports=True,rounds=oracle))
    base.save(d/'orchestration_status.json',dict(status='completed',device='mps',job=j,
        metrics_sha256=base.digest(d/'metrics.json'),round=m['rounds']))


def report(m,stamp):
    rows=[]
    for j in jobs(m):
        if completed(j,m,stamp):rows.append(json.loads((OUT/identifier(j)/'metrics.json').read_text()))
    lines=['# Risque privé + gradient privé — calibration v5','',f'**{len(rows)}/10 runs valides**, MPS, validation uniquement.',
        '', '60 tours, 10 clients, batch 240, C=1, epsilon réalisé ≈ 4 ; risques frais à chaque tour. '
        'Aucune attaque et aucune évaluation test. Pas de conclusion avant les dix runs.', '',
        '[Protocole](Split_Risk_Gradient_V5_Protocol.md)', '',
        '| Seed | Règle | Val. accuracy (%) | Val. Worst-20 (%) | Gap (pp) | Variance (pp²) | Loss Brier |',
        '|--:|:--|--:|--:|--:|--:|--:|']
    for r in rows:
        v=r['final']['validation']
        lines.append(f"| {r['job']['seed']} | {r['job']['method']} | {v['accuracy_pct']:.3f} | {v['worst20_pct']:.3f} | {v['gap_best20_worst20_pp']:.3f} | {v['variance_pp2']:.3f} | {v['brier_loss']:.5f} |")
    result=None
    if len(rows)==10:
        gates={}
        for candidate,controls in [('risk_mean',m['screen']['primary_controls']),('risk_rfa',m['screen']['robust_controls'])]:
            comparisons=[]
            for control in controls:
                for seed in m['calibration_seeds']:
                    a=next(r['final']['validation'] for r in rows if r['job']==dict(seed=seed,method=candidate))
                    b=next(r['final']['validation'] for r in rows if r['job']==dict(seed=seed,method=control))
                    da,dw=a['accuracy_pct']-b['accuracy_pct'],a['worst20_pct']-b['worst20_pct']
                    passed=da>=-m['screen']['maximum_accuracy_loss_pp'] and dw>=m['screen']['minimum_worst20_advantage_pp']
                    comparisons.append(dict(seed=seed,control=control,accuracy_delta_pp=da,worst20_delta_pp=dw,passed=passed))
            gates[candidate]=dict(passed=all(c['passed'] for c in comparisons),comparisons=comparisons)
        # Validate randomization and split pairing, without treating divergent risks as identical.
        for seed in m['calibration_seeds']:
            group=[r for r in rows if r['job']['seed']==seed]
            assert all(r['splits']==group[0]['splits'] and r['initial']==group[0]['initial'] for r in group)
            patterns=[]
            for r in group:
                o=json.loads((OUT/identifier(r['job'])/'simulator_oracle.json').read_text())
                patterns.append([[c['batch_hash'] for c in t['clients']] for t in o['rounds']])
            assert all(p==patterns[0] for p in patterns)
        robustness_cost=[]
        for seed in m['calibration_seeds']:
            a=next(r['final']['validation'] for r in rows if r['job']==dict(seed=seed,method='matched_rfa'))
            b=next(r['final']['validation'] for r in rows if r['job']==dict(seed=seed,method='matched_mean'))
            robustness_cost.append(dict(seed=seed,accuracy_delta_pp=a['accuracy_pct']-b['accuracy_pct'],
                worst20_delta_pp=a['worst20_pct']-b['worst20_pct']))
        result=dict(source_stamp=stamp,gates=gates,rows=rows,matched_rfa_minus_matched_mean=robustness_cost,
            paired_splits_initialization_batches=True,
            validation_only=True,not_a_robustness_confirmation=True)
        base.save(OUT/'evidence.json',result)
        lines+=['','## Écran préenregistré','']
        for a,g in gates.items():lines.append(f"- {a} : "+('passe' if g['passed'] else 'ne passe pas')+'.')
        lines+=['','Deux seeds de calibration : même en cas de réussite, aucune validation finale ni robustesse démontrée.', '',
                 '[Evidence](../../results/ldp_gradient_far/split_risk_gradient_v5/evidence.json)']
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text('\n'.join(lines)+'\n')
    return len(rows),result


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
            stamp=source_stamp(m)
            manifest=OUT/'manifest.json'
            if manifest.exists():assert json.loads(manifest.read_text())['source_stamp']==stamp
            else:base.save(manifest,dict(config=m,source_stamp=stamp,device='mps',created_at=time.time()))
            base.save(OUT/'status.json',dict(status='preflight',pid=os.getpid(),device='mps'))
            tests=OUT/'tests.json'
            if not tests.exists():
                p=subprocess.run([sys.executable,'-m','pytest',*TESTS,'-q'],cwd=ROOT,capture_output=True,text=True)
                base.save(tests,dict(passed=p.returncode==0,output=p.stdout+p.stderr,source_stamp=stamp))
            t=json.loads(tests.read_text())
            assert t['passed'] and t['source_stamp']==stamp
            secret=OUT/'simulator_secret.json'
            if not secret.exists():base.save(secret,dict(key=secrets.token_hex(32),not_public=True))
            key=json.loads(secret.read_text())['key']
            report(m,stamp)
            for seed in m['calibration_seeds']:
                js=[j for j in jobs(m) if j['seed']==seed]
                if all(completed(j,m,stamp) for j in js):continue
                data=base.prepare(m,seed)
                for j in js:
                    train(m,j,data,key,stamp)
                    report(m,stamp)
                del data
                torch.mps.empty_cache()
            count,result=report(m,stamp)
            assert count==10
            base.save(OUT/'status.json',dict(status='completed',device='mps',valid_runs=10,
                gates={a:g['passed'] for a,g in result['gates'].items()},test_evaluated=False,no_attacks_launched=True))
        except Exception as exc:
            fail=dict(status='failed',error=repr(exc),pid=os.getpid(),time=time.time())
            base.save(OUT/'failure.json',fail)
            base.save(OUT/'status.json',fail)
            raise


def main():
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group(required=True)
    for flag in ['launch','worker','status','plan']:g.add_argument('--'+flag,action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();m=config()
    if a.plan:print(json.dumps(dict(config=m,jobs=jobs(m)),indent=2));return
    if a.status:
        print((OUT/'status.json').read_text() if (OUT/'status.json').exists() else 'not started')
        if (OUT/'manifest.json').exists():
            stamp=json.loads((OUT/'manifest.json').read_text())['source_stamp'];base.verify_stamp(stamp)
            print(json.dumps(dict(valid_runs=sum(completed(j,m,stamp) for j in jobs(m)),expected=10)))
        return
    if not a.resume:p.error('--resume required')
    if a.worker:worker(m);return
    require_mps()
    with lock():pass
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open('a') as f:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker','--resume'],cwd=ROOT,
            env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=proc.pid,log=str(LOG),device='mps',runs=10)))


if __name__=='__main__':main()
