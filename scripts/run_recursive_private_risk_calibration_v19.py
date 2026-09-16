#!/usr/bin/env python3
"""Frozen 16-run clean calibration of the analytically selected recursive query."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import secrets
import statistics as st
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
import yaml
from scripts import run_fair_objective_screen as base
from scripts import run_private_recursive_frontier_diagnostic_v18 as screen
from scripts import run_capped_private_risk_calibration_v7 as accounting
from scripts.run_private_risk_confirmation_v12 import audit_evaluation
from scripts.run_private_clipping_step_diagnostic_v16 import ledger
from privacy.fair_objective import require_mps,per_example,release
from privacy.split_risk_gradient import private_risk
from privacy.scheduled_private_risk import aggregate
from privacy.private_risk_message_safety import sanitize
from privacy.clipped_recursive_private_gradient import recursive_release
from privacy.recursive_public_calibration import largest_increment_ratio

NAME='recursive_private_risk_calibration_v19'
OUT=ROOT/'results/ldp_gradient_far'/NAME
MATRIX=ROOT/'configs/ldp_gradient_far'/f'{NAME}.yaml'
REPORT=ROOT/'output/analysis/Recursive_Private_Risk_Calibration_V19_Status.md'
TESTS=['tests/test_recursive_private_risk_calibration_v19.py','tests/test_private_risk_message_safety.py',
       'tests/test_scheduled_private_risk.py','tests/test_stable_weighted_rfa.py','tests/test_recursive_public_calibration_v18.py']


def inputs():
    m=yaml.safe_load(MATRIX.read_text())
    assert m['campaign_id']==NAME and m['seeds']==[170501,170502] and m['expected_runs']==16
    assert m['modes']==['fresh','recursive'] and m['methods']==['erm_mean','erm_rfa','risk_mean','risk_rfa']
    assert m['device']=='mps' and m['rounds']==120 and m['clip']==2 and m['epsilon']==4 and m['delta']==1e-5
    assert not m['test_evaluated'] and not m['automatic_attacks'] and not m['automatic_confirmation']
    previous=json.loads((screen.OUT/'manifest.json').read_text());base.verify_stamp(previous['source_stamp'])
    audit=ROOT/'output/analysis/Private_Recursive_Query_V18_Independent_Audit.json'
    ev=json.loads(audit.read_text());assert ev['audit_passed'] and ev['admitted_to_calibration']
    profile=dict(previous['profile'],evaluation_rounds=m['validation_rounds'])
    stamp=dict(previous['source_stamp'])
    files=[Path(__file__),MATRIX,ROOT/m['protocol'],audit,ROOT/'privacy/clipped_recursive_private_gradient.py',
           ROOT/'scripts/run_private_risk_confirmation_v12.py',*[ROOT/f for f in TESTS]]
    files += [screen.prior.prior.population.SOURCE/f'seed{s}__{k}'/'metrics.json'
              for s in m['seeds'] for k in ('erm_mean','erm_rfa')]
    stamp.update({str(p.relative_to(ROOT)):base.digest(p) for p in files})
    return m,profile,stamp


def jobs(m):return [dict(seed=s,mode=mode,method=k) for s in m['seeds'] for mode in m['modes'] for k in m['methods']]
def identifier(j):return f"seed{j['seed']}__{j['mode']}__{j['method']}"


def completed(j,stamp):
    d=OUT/identifier(j);f=d/'orchestration_status.json'
    if not f.exists() or json.loads(f.read_text())['status']!='completed':return False
    s=json.loads(f.read_text());r=json.loads((d/'metrics.json').read_text())
    assert r['source_stamp']==stamp and r['job']==j and r['device']=='mps' and not r['test_evaluated']
    assert s['metrics_sha256']==base.digest(d/'metrics.json') and s['oracle_sha256']==base.digest(d/'simulator_oracle.json')
    assert [x['round'] for x in r['rounds']]==list(range(1,121)) and r['privacy']['epsilon_realized']<=4
    audit_evaluation(r['final']['validation']);return True


def train(j,m,profile,stamp,data,key):
    if completed(j,stamp):return
    d=OUT/identifier(j);d.mkdir(parents=True,exist_ok=True)
    p=ledger(2.,j['method']);pub=largest_increment_ratio(C=2.)
    model=base.new_model(profile,j['seed']);previous=base.new_model(profile,j['seed'])
    rows=[];oracles=[];start=0;elapsed=0.;memory=None;initial=None;cp=d/'checkpoint.pt'
    if cp.exists():
        state=torch.load(cp,map_location='cpu',weights_only=True)
        assert state['source_stamp']==stamp and state['job']==j and state['key_sha']==base.digest(OUT/'simulator_secret.json')
        model.load_state_dict(state['model']);previous.load_state_dict(state['previous_model'])
        memory=None if state['memory'] is None else state['memory'].to('mps')
        rows,oracles,start,elapsed,initial=state['rows'],state['oracles'],state['round'],state['elapsed_seconds'],state['initial']
    else:initial=base.evaluate(model,data,'val')
    code_stamp={k:v for k,v in stamp.items() if Path(k).suffix in ('.py','.yaml','.md')}
    base.save(d/'public_protocol.json',dict(config=m,profile=profile,job=j,privacy=p,recursive_parameters=pub,source_stamp=stamp))
    for t in range(start,120):
        tick=time.monotonic();require_mps();base.verify_stamp(code_stamp)
        status=dict(status='running',device='mps',active=identifier(j),round=t+1,total_rounds=120,pid=os.getpid(),updated_unix=time.time())
        base.save(OUT/'status.json',status);base.save(d/'orchestration_status.json',status)
        sent=[];reports=[];local=[]
        for cid,ids in enumerate(data['train']):
            rr=raw=None
            if j['method'].startswith('risk_'):
                rr,raw=private_risk(model,data['x'][ids],data['y'][ids],noise_std=p['risk_std'],
                    seed=base.seed_for(key,j['seed'],t,cid,'risk'),N=4800)
                reports.append(rr)
            ix=base.draw_indices(4800,240,base.seed_for(key,j['seed'],t,cid,'batch'))
            _,g,norms,_=per_example(model,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)
            noise_seed=base.seed_for(key,j['seed'],t,cid,'gaussian')
            if j['mode']=='fresh':
                message=release(g.mean(0),noise_std=p['gradient_std'],seed=noise_seed)
                diag=dict(initial=True,effective_C=2.,query_sensitivity=4/240,noise_std=p['gradient_std'],
                          clipping_increment_count=0,batch_size=240,increment_bias_proxy_norm=0.)
            else:
                oldg=None if t==0 else per_example(previous,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)[1]
                message,diag=recursive_release(g,oldg,None if t==0 else memory[cid],C=2.,D=pub['D'],theta=pub['theta'],
                    base_noise_std=p['gradient_std'],seed=noise_seed)
            assert abs(diag['noise_std']/diag['query_sensitivity']-p['gradient_z'])<1e-10
            sent.append(message)
            local.append(dict(client=cid,batch_hash=base.ids_hash(ix),gradient_clipped_count=int((norms>2).sum()),batch_size=240,
                recursion=diag,private_risk=None if rr is None else float(rr),raw_risk=None if raw is None else float(raw)))
        messages,r,safety=sanitize(torch.stack(sent),None if not reports else torch.stack(reports))
        assert safety['invalid_message_rows']==safety['nonfinite_risk_reports']==0
        step,agg=aggregate(messages,r,kind=j['method'],round_number=t+1,horizon=120);agg['message_safety']=safety
        if j['mode']=='recursive':
            previous.load_state_dict(model.state_dict());memory=messages.detach().clone()
        base.apply_gradient(model,step,1.)
        val=base.evaluate(model,data,'val') if t+1 in profile['evaluation_rounds'] else None
        if val is not None:audit_evaluation(val)
        # Every normalized Gaussian query has the same z, even as sensitivity/std shrink together.
        from privacy.fair_objective import wor_rdp
        import math
        ep=min((t+1)*wor_rdp(a,.05,p['gradient_z'])+((t+1)*a/(2*p['risk_z']**2) if j['method'].startswith('risk_') else 0)
               +math.log(1e5)/(a-1) for a in range(2,65))
        assert ep<=4
        rows.append(dict(round=t+1,device='mps',epsilon_realized=ep,aggregation=agg,validation=val))
        oracles.append(dict(round=t+1,clients=local));elapsed+=time.monotonic()-tick
        base.checkpoint(cp,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            previous_model={k:v.detach().cpu() for k,v in previous.state_dict().items()},
            memory=None if memory is None else memory.detach().cpu(),rows=rows,oracles=oracles,round=t+1,
            elapsed_seconds=elapsed,initial=initial,source_stamp=stamp,job=j,key_sha=base.digest(OUT/'simulator_secret.json')))
        if val is not None:print(f"{identifier(j)} {t+1}/120 acc={val['accuracy_pct']:.3f} W20={val['worst20_pct']:.3f}",flush=True)
    base.verify_stamp(stamp)
    base.save(d/'metrics.json',dict(job=j,source_stamp=stamp,device='mps',privacy=p,recursive_parameters=pub,
        initial=initial,rounds=rows,final=rows[-1],splits=data['splits'],test_evaluated=False,
        per_client_batch_gradient_evaluations=120 if j['mode']=='fresh' else 239,
        validation_and_oracles_not_private=True,elapsed_seconds=elapsed))
    base.save(d/'simulator_oracle.json',dict(privacy_protected=False,feeds_mechanism=False,rounds=oracles))
    base.save(d/'orchestration_status.json',dict(status='completed',device='mps',job=j,round=120,
        metrics_sha256=base.digest(d/'metrics.json'),oracle_sha256=base.digest(d/'simulator_oracle.json')))
    del model,previous,memory;torch.mps.empty_cache()


def decide(m,rows):
    assert len(rows)==16
    index={(r['job']['seed'],r['job']['mode'],r['job']['method']):r['final']['validation'] for r in rows}
    contrasts=[]
    for mode in m['modes']+['historical_v11']:
        for method in ('erm_mean','erm_rfa'):
            pairs=[]
            for seed in m['seeds']:
                a=index[seed,'recursive','risk_rfa']
                if mode=='historical_v11':
                    b=json.loads((screen.prior.prior.population.SOURCE/f'seed{seed}__{method}'/'metrics.json').read_text())['final']['validation']
                else:b=index[seed,mode,method]
                delta={k:a[k]-b[k] for k in ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2')}
                pairs.append(dict(seed=seed,delta=delta,passed=delta['accuracy_pct']>=-1 and delta['worst20_pct']>=1))
            ok=all(p['passed'] for p in pairs) and all(st.mean(p['delta'][k] for p in pairs)<=0 for k in ('gap_best20_worst20_pp','variance_pp2'))
            contrasts.append(dict(control_mode=mode,control_method=method,pairs=pairs,passed=ok,noise_paired=mode!='historical_v11'))
    return dict(eligible_for_independent_confirmation=all(c['passed'] for c in contrasts),contrasts=contrasts,
                global_validation=False,attacks_launched=False)


def report(m,stamp):
    rows=[json.loads((OUT/identifier(j)/'metrics.json').read_text()) for j in jobs(m) if completed(j,stamp)]
    lines=['# V19 — calibration end-to-end de la requête récursive','',f'**{len(rows)}/16 runs MPS valides**, validation uniquement.', '',
        '| Seed | Requête | Agrégation | Accuracy (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) | Brier |',
        '|--:|:--|:--|--:|--:|--:|--:|--:|']
    for r in rows:
        j,v=r['job'],r['final']['validation'];lines.append(f"| {j['seed']} | {j['mode']} | {j['method']} | {v['accuracy_pct']:.3f} | {v['worst20_pct']:.3f} | {v['gap_best20_worst20_pp']:.3f} | {v['variance_pp2']:.3f} | {v['brier_loss']:.5f} |")
    d=decide(m,rows) if len(rows)==16 else None
    if d is not None:
        base.save(OUT/'evidence.json',dict(source_stamp=stamp,decision=d))
        lines+=['',f"Admission à confirmation indépendante : **{'PASS' if d['eligible_for_independent_confirmation'] else 'FAIL'}**.", '',
            'Les deux seeds sont de calibration, non de confirmation. Aucun test final ni attaque lancé automatiquement.']
    lines+=['','[Protocole](Recursive_Private_Risk_Calibration_V19_Protocol.md).'];REPORT.write_text('\n'.join(lines)+'\n')
    return len(rows),d


def worker():
    require_mps();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            m,profile,stamp=inputs();manifest=dict(config=m,profile=profile,source_stamp=stamp)
            if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
            else:base.save(OUT/'manifest.json',manifest)
            if not (OUT/'tests.json').exists():
                p=subprocess.run([sys.executable,'-m','pytest',*TESTS,'-q'],cwd=ROOT,capture_output=True,text=True)
                base.save(OUT/'tests.json',dict(passed=p.returncode==0,output=p.stdout+p.stderr,source_stamp=stamp))
            tests=json.loads((OUT/'tests.json').read_text());assert tests['passed'] and tests['source_stamp']==stamp
            secret=OUT/'simulator_secret.json'
            if not secret.exists():
                assert not list(OUT.glob('seed*/checkpoint.pt'));base.save(secret,dict(key=secrets.token_hex(32),privacy_protected=False))
            key=json.loads(secret.read_text())['key']
            for seed in m['seeds']:
                data=base.prepare(profile,seed)
                for j in [j for j in jobs(m) if j['seed']==seed]:
                    train(j,m,profile,stamp,data,key);count,d=report(m,stamp);print(f'{count}/16 runs completed',flush=True)
                del data;torch.mps.empty_cache()
            count,d=report(m,stamp);assert count==16
            patterns={};initials={};splits={}
            for j in jobs(m):
                folder=OUT/identifier(j);r=json.loads((folder/'metrics.json').read_text())
                oracle=json.loads((folder/'simulator_oracle.json').read_text())
                pattern=[[x['batch_hash'] for x in t['clients']] for t in oracle['rounds']]
                s=j['seed']
                if s in patterns:assert patterns[s]==pattern and initials[s]==r['initial'] and splits[s]==r['splits']
                else:patterns[s],initials[s],splits[s]=pattern,r['initial'],r['splits']
            base.save(OUT/'pairing_audit.json',dict(passed=True,source_stamp=stamp))
            base.save(OUT/'status.json',dict(status='completed',device='mps',valid_runs=16,eligible_for_independent_confirmation=d['eligible_for_independent_confirmation'],next_campaign_launched=False))
        except Exception as exc:
            base.save(OUT/'status.json',dict(status='failed',device='mps',error=repr(exc),pid=os.getpid()));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true',required=True);p.add_argument('--resume',action='store_true',required=True)
    p.parse_args();worker()


if __name__=='__main__':main()
