#!/usr/bin/env python3
"""Public late-step decay ablation; 8 MPS calibration runs, no automatic promotion."""
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
from scripts import run_fair_objective_screen as base
from privacy.fair_objective import require_mps,per_example,release
from privacy.split_risk_gradient import private_risk
from privacy.scheduled_private_risk import aggregate,learning_rate

NAME='public_decay_private_risk_v9'
MATRIX=ROOT/'configs/ldp_gradient_far'/f'{NAME}.yaml'
OUT=ROOT/'results/ldp_gradient_far'/NAME
LOG=ROOT/'logs'/f'{NAME}.log'
REPORT=ROOT/'output/analysis/Public_Decay_Private_Risk_V9_Status.md'


def config():
    m=yaml.safe_load(MATRIX.read_text())
    assert m['campaign_id']==NAME and m['device']=='mps' and m['expected_runs']==8
    assert m['clip']==2 and m['risk_scale']==.5 and m['rounds']==60
    assert m['calibration_seeds']==[170501,170502] and m['attacks']=='none'
    assert not m['test_evaluated'] and not m['automatic_attacks'] and not m['automatic_confirmation']
    assert m['learning_rate_first_half']==2 and m['learning_rate_second_half']==.5
    assert m['minimum_worst20_advantage_pp']==1 and m['maximum_accuracy_loss_pp']==1
    return m


def jobs(m):return [dict(seed=s,method=k) for s in m['calibration_seeds'] for k in m['methods']]
def identifier(j):return f"seed{j['seed']}__{j['method']}"
def arm(kind):return f'{kind}_C2_scale0.5' if kind.startswith('risk_') else f'{kind}_C2'


def inputs(m):
    old=json.loads((parent.OUT/'manifest.json').read_text());base.verify_stamp(old['source_stamp'])
    assert json.loads((parent.OUT/'status.json').read_text())['status']=='completed'
    stamp=dict(old['source_stamp'])
    files=[Path(__file__),MATRIX,ROOT/m['protocol'],ROOT/'privacy/scheduled_private_risk.py',
           ROOT/'privacy/stable_weighted_rfa.py',ROOT/'tests/test_stable_weighted_rfa.py',ROOT/'tests/test_scheduled_private_risk.py']
    stamp.update({str(p.relative_to(ROOT)):base.digest(p) for p in files})
    return old['config'],stamp


def completed(j,m,stamp):
    d=OUT/identifier(j)
    if not (d/'orchestration_status.json').exists():return False
    s=json.loads((d/'orchestration_status.json').read_text())
    if s['status']!='completed':return False
    r=json.loads((d/'metrics.json').read_text())
    assert r['job']==j and r['source_stamp']==stamp and r['device']=='mps' and not r['test_evaluated']
    assert s['metrics_sha256']==base.digest(d/'metrics.json') and r['privacy']['epsilon_realized']<=4
    assert [a['round'] for a in r['rounds']]==list(range(1,61))
    assert all(a['device']=='mps' and a['aggregation']['eta']==learning_rate(a['round']) and a['epsilon_realized']<=4 for a in r['rounds'])
    return True


def train(m,old,j,data,key,stamp):
    if completed(j,m,stamp):return
    d=OUT/identifier(j);d.mkdir(parents=True,exist_ok=True)
    p=parent.privacy(old,arm(j['method']));model=base.new_model(old,j['seed'])
    rows,oracle,start=[],[],0;cp=d/'checkpoint.pt'
    if cp.exists():
        state=torch.load(cp,map_location='cpu',weights_only=True)
        assert state['source_stamp']==stamp and state['job']==j
        model.load_state_dict(state['model']);rows,oracle,start,initial=state['rows'],state['oracle'],state['round'],state['initial']
    else:initial=base.evaluate(model,data,'val')
    base.save(d/'public_protocol.json',dict(config=m,inherited_config=old,job=j,privacy=p,source_stamp=stamp))
    for t in range(start,60):
        require_mps();base.verify_stamp(stamp)
        status=dict(status='running',device='mps',active=identifier(j),round=t+1,total_rounds=60,pid=os.getpid())
        base.save(OUT/'status.json',status);base.save(d/'orchestration_status.json',status)
        messages,reports,local=[],[],[]
        for cid,ids in enumerate(data['train']):
            rr,raw=None,None
            if j['method'].startswith('risk_'):
                rr,raw=private_risk(model,data['x'][ids],data['y'][ids],noise_std=p['risk_std'],
                    seed=base.seed_for(key,j['seed'],t,cid,'risk'),N=old['public_train_size'])
                reports.append(rr)
            idx=base.draw_indices(len(ids),old['batch_size'],base.seed_for(key,j['seed'],t,cid,'batch'))
            _,g,norms,_=per_example(model,data['x'][ids[idx]],data['y'][ids[idx]],clip_norm=2.)
            messages.append(release(g.mean(0),noise_std=p['gradient_std'],seed=base.seed_for(key,j['seed'],t,cid,'gaussian')))
            local.append(dict(client=cid,batch_hash=base.ids_hash(idx),clip_fraction=float((norms>2).float().mean()),
                private_risk=None if rr is None else float(rr),raw_risk_research_only=None if raw is None else float(raw)))
        u,diag=aggregate(torch.stack(messages),None if not reports else torch.stack(reports),kind=j['method'],round_number=t+1)
        if not bool(torch.isfinite(u).all()):raise FloatingPointError('Nonfinite update')
        base.apply_gradient(model,u,1.)
        v=base.evaluate(model,data,'val') if t+1 in old['evaluation_rounds'] else None
        rows.append(dict(round=t+1,device='mps',epsilon_realized=parent.epsilon_at(p,t+1,old,j['method']),aggregation=diag,validation=v))
        oracle.append(dict(round=t+1,clients=local))
        base.checkpoint(cp,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            rows=rows,oracle=oracle,round=t+1,initial=initial,source_stamp=stamp,job=j))
        if v is not None:print(f"{identifier(j)} {t+1}/60 eta={diag['eta']:g} val_acc={v['accuracy_pct']:.3f} W20={v['worst20_pct']:.3f}",flush=True)
    base.save(d/'metrics.json',dict(job=j,source_stamp=stamp,device='mps',privacy=p,initial=initial,
        rounds=rows,final=rows[-1],splits=data['splits'],test_evaluated=False,validation_and_oracles_not_private=True))
    base.save(d/'simulator_oracle.json',dict(privacy_protected=False,rounds=oracle))
    base.save(d/'orchestration_status.json',dict(status='completed',device='mps',job=j,round=60,metrics_sha256=base.digest(d/'metrics.json')))


def decide(m,rows,historical):
    gates={}
    for candidate in ('risk_mean','risk_rfa'):
        comparisons=[]
        for seed in m['calibration_seeds']:
            a=next(r['final']['validation'] for r in rows if r['job']==dict(seed=seed,method=candidate))
            controls=[(c,next(r['final']['validation'] for r in rows if r['job']==dict(seed=seed,method=c))) for c in m['new_controls']]
            controls += [('v7/'+c,next(r['final']['validation'] for r in historical if r['job']==dict(seed=seed,arm=c))) for c in m['historical_controls']]
            for c,b in controls:
                da,dw=a['accuracy_pct']-b['accuracy_pct'],a['worst20_pct']-b['worst20_pct']
                comparisons.append(dict(seed=seed,control=c,accuracy_delta_pp=da,worst20_delta_pp=dw,
                    passed=da>=-m['maximum_accuracy_loss_pp'] and dw>=m['minimum_worst20_advantage_pp']))
        gates[candidate]=dict(passed=all(c['passed'] for c in comparisons),comparisons=comparisons)
    return dict(gates=gates,selected_robust_candidate='risk_rfa' if gates['risk_rfa']['passed'] else None,confirmatory=False)


def report(m,old,stamp):
    rows=[json.loads((OUT/identifier(j)/'metrics.json').read_text()) for j in jobs(m) if completed(j,m,stamp)]
    lines=['# V9 — décroissance publique du pas','',f'**{len(rows)}/8 runs valides**, MPS, validation uniquement.', '',
        '| Seed | Méthode | Accuracy (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) | Brier |',
        '|--:|:--|--:|--:|--:|--:|--:|']
    for r in rows:
        j,v=r['job'],r['final']['validation']
        lines.append(f"| {j['seed']} | {j['method']} | {v['accuracy_pct']:.3f} | {v['worst20_pct']:.3f} | {v['gap_best20_worst20_pp']:.3f} | {v['variance_pp2']:.3f} | {v['brier_loss']:.5f} |")
    decision=None
    if len(rows)==8:
        historical=[json.loads((parent.OUT/parent.identifier(dict(seed=s,arm=c))/'metrics.json').read_text())
                    for s in m['calibration_seeds'] for c in m['historical_controls']]
        decision=decide(m,rows,historical)
        base.save(OUT/'evidence.json',dict(rows=rows,historical_controls=historical,decision=decision,source_stamp=stamp))
        lines+=['',f"Candidate robuste admissible : {decision['selected_robust_candidate'] or 'aucune'}.", '',
                'Admissibilité propre seulement ; aucune attaque ni confirmation indépendante dans ces résultats.', '',
                '[Protocole](Public_Decay_Private_Risk_V9_Protocol.md)',
                '[Toutes les différences](../../results/ldp_gradient_far/public_decay_private_risk_v9/evidence.json)']
    REPORT.write_text('\n'.join(lines)+'\n')
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
            old,stamp=inputs(m);manifest=OUT/'manifest.json';content=dict(config=m,inherited_config=old,source_stamp=stamp)
            if manifest.exists():assert json.loads(manifest.read_text())==content
            else:base.save(manifest,content)
            test=OUT/'tests.json'
            if not test.exists():
                p=subprocess.run([sys.executable,'-m','pytest','tests/test_stable_weighted_rfa.py','tests/test_scheduled_private_risk.py','tests/test_split_risk_gradient.py','-q'],cwd=ROOT,capture_output=True,text=True)
                base.save(test,dict(passed=p.returncode==0,output=p.stdout+p.stderr,source_stamp=stamp))
            assert json.loads(test.read_text())['passed']
            key=json.loads((ROOT/old['paired_randomness_source']).read_text())['key']
            base.save(OUT/'pairing_policy.json',dict(source=old['paired_randomness_source'],same_standard_gaussians=True,same_batches=True,not_jointly_DP_releasable=True))
            for seed in m['calibration_seeds']:
                js=[j for j in jobs(m) if j['seed']==seed]
                if all(completed(j,m,stamp) for j in js):continue
                data=base.prepare(old,seed)
                for j in js:train(m,old,j,data,key,stamp);report(m,old,stamp)
                del data;torch.mps.empty_cache()
            count,decision=report(m,old,stamp);assert count==8
            for j in jobs(m):
                r=json.loads((OUT/identifier(j)/'metrics.json').read_text())
                prior=json.loads((parent.OUT/parent.identifier(dict(seed=j['seed'],arm=arm(j['method'])))/'metrics.json').read_text())
                assert r['splits']==prior['splits'] and r['initial']['accuracy_pct']==prior['initial']['accuracy_pct']
                if j['method'].endswith('mean'):
                    # Identical rule/noise until the public schedule changes.
                    for a,b in zip(r['rounds'][:30],prior['rounds'][:30]):
                        if a['validation']:
                            assert abs(a['validation']['accuracy_pct']-b['validation']['accuracy_pct'])<1e-6
            base.save(OUT/'status.json',dict(status='completed',device='mps',valid_runs=8,
                selected_robust_candidate=decision['selected_robust_candidate'],no_confirmation_or_attacks_launched=True))
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
