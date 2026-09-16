#!/usr/bin/env python3
"""Single longer-horizon calibration at fixed total epsilon, MPS only."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
import yaml
from scripts import run_capped_private_risk_calibration_v7 as parent
from scripts import run_public_decay_private_risk_v9 as previous
from scripts import run_fair_objective_screen as base
from privacy.fair_objective import require_mps,per_example,release
from privacy.split_risk_gradient import private_risk
from privacy.scheduled_private_risk import aggregate,learning_rate

NAME='public_horizon_private_risk_v11'
MATRIX=ROOT/'configs/ldp_gradient_far'/f'{NAME}.yaml'
OUT=ROOT/'results/ldp_gradient_far'/NAME
LOG=ROOT/'logs'/f'{NAME}.log'
REPORT=ROOT/'output/analysis/Public_Horizon_Private_Risk_V11_Status.md'
jobs=previous.jobs
identifier=previous.identifier
arm=previous.arm
decide=previous.decide


def config():
    m=yaml.safe_load(MATRIX.read_text())
    assert m['campaign_id']==NAME and m['device']=='mps' and m['rounds']==120 and m['expected_runs']==8
    assert m['clip']==2 and m['risk_scale']==.5 and m['calibration_seeds']==[170501,170502]
    assert m['attacks']=='none' and not m['test_evaluated'] and not m['automatic_attacks'] and not m['automatic_confirmation']
    assert m['epsilon']==4 and m['delta']==1e-5 and m['minimum_worst20_advantage_pp']==1 and m['maximum_accuracy_loss_pp']==1
    return m


def inputs(m):
    old=json.loads((previous.OUT/'manifest.json').read_text());base.verify_stamp(old['source_stamp'])
    assert json.loads((previous.OUT/'status.json').read_text())['status']=='completed'
    profile=dict(old['inherited_config'],rounds=120,evaluation_rounds=m['evaluation_rounds'])
    stamp=dict(old['source_stamp'])
    files=[Path(__file__),MATRIX,ROOT/m['protocol'],ROOT/'tests/test_public_horizon_private_risk_v11.py']
    stamp.update({str(p.relative_to(ROOT)):base.digest(p) for p in files})
    return profile,stamp


def completed(j,m,stamp):
    d=OUT/identifier(j)
    if not (d/'orchestration_status.json').exists():return False
    s=json.loads((d/'orchestration_status.json').read_text())
    if s['status']!='completed':return False
    r=json.loads((d/'metrics.json').read_text())
    assert r['job']==j and r['source_stamp']==stamp and r['device']=='mps' and not r['test_evaluated']
    assert s['metrics_sha256']==base.digest(d/'metrics.json') and r['privacy']['epsilon_realized']<=4
    assert [a['round'] for a in r['rounds']]==list(range(1,121))
    assert all(a['device']=='mps' and a['aggregation']['eta']==learning_rate(a['round'],120) and a['epsilon_realized']<=4 for a in r['rounds'])
    assert [a['round'] for a in r['rounds'] if a['validation'] is not None]==m['evaluation_rounds']
    return True


def train(m,profile,j,data,key,stamp):
    if completed(j,m,stamp):return
    d=OUT/identifier(j);d.mkdir(parents=True,exist_ok=True)
    p=parent.privacy(profile,arm(j['method']));model=base.new_model(profile,j['seed'])
    rows,oracle,start=[],[],0;cp=d/'checkpoint.pt'
    if cp.exists():
        state=torch.load(cp,map_location='cpu',weights_only=True)
        assert state['source_stamp']==stamp and state['job']==j
        model.load_state_dict(state['model']);rows,oracle,start,initial=state['rows'],state['oracle'],state['round'],state['initial']
    else:initial=base.evaluate(model,data,'val')
    base.save(d/'public_protocol.json',dict(config=m,profile=profile,job=j,privacy=p,source_stamp=stamp))
    for t in range(start,120):
        require_mps();base.verify_stamp(stamp)
        status=dict(status='running',device='mps',active=identifier(j),round=t+1,total_rounds=120,pid=os.getpid())
        base.save(OUT/'status.json',status);base.save(d/'orchestration_status.json',status)
        messages,reports,local=[],[],[]
        for cid,ids in enumerate(data['train']):
            rr,raw=None,None
            if j['method'].startswith('risk_'):
                rr,raw=private_risk(model,data['x'][ids],data['y'][ids],noise_std=p['risk_std'],
                    seed=base.seed_for(key,j['seed'],t,cid,'risk'),N=profile['public_train_size'])
                reports.append(rr)
            idx=base.draw_indices(len(ids),profile['batch_size'],base.seed_for(key,j['seed'],t,cid,'batch'))
            _,g,norms,_=per_example(model,data['x'][ids[idx]],data['y'][ids[idx]],clip_norm=2.)
            messages.append(release(g.mean(0),noise_std=p['gradient_std'],seed=base.seed_for(key,j['seed'],t,cid,'gaussian')))
            local.append(dict(client=cid,batch_hash=base.ids_hash(idx),clip_fraction=float((norms>2).float().mean()),
                private_risk=None if rr is None else float(rr),raw_risk_research_only=None if raw is None else float(raw)))
        u,diag=aggregate(torch.stack(messages),None if not reports else torch.stack(reports),kind=j['method'],round_number=t+1,horizon=120)
        if not bool(torch.isfinite(u).all()):raise FloatingPointError('Nonfinite update')
        base.apply_gradient(model,u,1.)
        v=base.evaluate(model,data,'val') if t+1 in m['evaluation_rounds'] else None
        rows.append(dict(round=t+1,device='mps',epsilon_realized=parent.epsilon_at(p,t+1,profile,j['method']),aggregation=diag,validation=v))
        oracle.append(dict(round=t+1,clients=local))
        base.checkpoint(cp,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},rows=rows,oracle=oracle,
            round=t+1,initial=initial,source_stamp=stamp,job=j))
        if v is not None:print(f"{identifier(j)} {t+1}/120 eta={diag['eta']:g} val_acc={v['accuracy_pct']:.3f} W20={v['worst20_pct']:.3f}",flush=True)
    base.save(d/'metrics.json',dict(job=j,source_stamp=stamp,device='mps',privacy=p,initial=initial,
        rounds=rows,final=rows[-1],splits=data['splits'],test_evaluated=False,validation_and_oracles_not_private=True))
    base.save(d/'simulator_oracle.json',dict(privacy_protected=False,rounds=oracle))
    base.save(d/'orchestration_status.json',dict(status='completed',device='mps',job=j,round=120,metrics_sha256=base.digest(d/'metrics.json')))


def historical(m):
    rows=[]
    for seed in m['calibration_seeds']:
        for c in m['historical_controls']:
            campaign,kind=c.split('_',1)
            source=parent.OUT if campaign=='v7' else previous.OUT
            r=json.loads((source/f'seed{seed}__{kind}'/'metrics.json').read_text())
            rows.append(dict(job=dict(seed=seed,arm=c),final=r['final'],source=str(source/f'seed{seed}__{kind}'/'metrics.json')))
    return rows


def report(m,stamp):
    rows=[json.loads((OUT/identifier(j)/'metrics.json').read_text()) for j in jobs(m) if completed(j,m,stamp)]
    lines=['# V11 — horizon 120 à budget fixé','',f'**{len(rows)}/8 runs valides**, MPS, validation uniquement.', '',
        '| Seed | Méthode | Accuracy (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) | Brier |',
        '|--:|:--|--:|--:|--:|--:|--:|']
    for r in rows:
        j,v=r['job'],r['final']['validation']
        lines.append(f"| {j['seed']} | {j['method']} | {v['accuracy_pct']:.3f} | {v['worst20_pct']:.3f} | {v['gap_best20_worst20_pp']:.3f} | {v['variance_pp2']:.3f} | {v['brier_loss']:.5f} |")
    decision=None
    if len(rows)==8:
        old=historical(m);decision=decide(m,rows,old)
        base.save(OUT/'evidence.json',dict(rows=rows,historical_controls=old,decision=decision,source_stamp=stamp))
        lines+=['',f"Candidate robuste admissible : {decision['selected_robust_candidate'] or 'aucune'}.", '',
                'Aucune confirmation ni attaque n’est lancée automatiquement. Un passage propre ne serait pas une validation finale.']
    lines+=['','[Protocole](Public_Horizon_Private_Risk_V11_Protocol.md).']
    REPORT.write_text('\n'.join(lines)+'\n');return len(rows),decision


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
            profile,stamp=inputs(m);manifest=OUT/'manifest.json';content=dict(config=m,profile=profile,source_stamp=stamp)
            if manifest.exists():assert json.loads(manifest.read_text())==content
            else:base.save(manifest,content)
            tests=OUT/'tests.json'
            if not tests.exists():
                p=subprocess.run([sys.executable,'-m','pytest','tests/test_public_horizon_private_risk_v11.py','tests/test_scheduled_private_risk.py',
                    'tests/test_stable_weighted_rfa.py','tests/test_split_risk_gradient.py','-q'],cwd=ROOT,capture_output=True,text=True)
                base.save(tests,dict(passed=p.returncode==0,output=p.stdout+p.stderr,source_stamp=stamp))
            assert json.loads(tests.read_text())['passed']
            plans={kind:parent.privacy(profile,arm(kind)) for kind in m['methods']}
            base.save(OUT/'privacy_plans.json',plans)
            key=json.loads((ROOT/profile['paired_randomness_source']).read_text())['key']
            for seed in m['calibration_seeds']:
                js=[j for j in jobs(m) if j['seed']==seed]
                if all(completed(j,m,stamp) for j in js):continue
                data=base.prepare(profile,seed)
                for j in js:train(m,profile,j,data,key,stamp);report(m,stamp)
                del data;torch.mps.empty_cache()
            count,decision=report(m,stamp);assert count==8
            for seed in m['calibration_seeds']:
                baseline=None;pattern=None
                for j in [j for j in jobs(m) if j['seed']==seed]:
                    d=OUT/identifier(j);r=json.loads((d/'metrics.json').read_text())
                    o=json.loads((d/'simulator_oracle.json').read_text())
                    now=[[c['batch_hash'] for c in t['clients']] for t in o['rounds']]
                    if baseline is None:baseline=r;pattern=now
                    assert r['initial']==baseline['initial'] and r['splits']==baseline['splits'] and now==pattern
            base.save(OUT/'pairing_audit.json',dict(passed=True,all_splits_initial_models_batches_paired=True))
            base.save(OUT/'status.json',dict(status='completed',device='mps',valid_runs=8,selected_robust_candidate=decision['selected_robust_candidate'],
                no_confirmation_or_attacks_launched=True))
        except Exception as exc:
            base.save(OUT/'status.json',dict(status='failed',device='mps',error=repr(exc),pid=os.getpid()));raise


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for flag in ('launch','worker','status'):g.add_argument('--'+flag,action='store_true')
    p.add_argument('--resume',action='store_true');a=p.parse_args();m=config()
    if a.status:print((OUT/'status.json').read_text() if (OUT/'status.json').exists() else 'not started');return
    if not a.resume:p.error('--resume required')
    if a.worker:worker(m);return
    require_mps()
    with lock():pass
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open('a') as f:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker','--resume'],cwd=ROOT,
            env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=proc.pid,device='mps',runs=8,log=str(LOG))))


if __name__=='__main__':main()
