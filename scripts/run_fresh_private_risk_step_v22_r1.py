#!/usr/bin/env python3
"""Fresh private gradients, fixed risk objective, two public step controls."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
import yaml
from scripts import diagnose_prisma_global_step_v21 as diagnostic
from scripts import run_fair_objective_screen as base
from scripts import run_recursive_private_risk_calibration_v19 as prior
from scripts.analyze_private_risk_confirmation_v12 import verify
from scripts.run_private_clipping_step_diagnostic_v16 import ledger
from privacy.fair_objective import require_mps,per_example,release,wor_rdp
from privacy.split_risk_gradient import private_risk
from privacy.scheduled_private_risk import aggregate
from privacy.aggregate_radial_step_v21 import controlled_step
from privacy.private_risk_message_safety import sanitize

OUT=ROOT/'results/ldp_gradient_far/fresh_private_risk_step_v22_r1'
MATRIX=ROOT/'configs/ldp_gradient_far/fresh_private_risk_step_v22.yaml'
PROTOCOL=ROOT/'output/analysis/Fresh_Private_Risk_Step_V22_Protocol.md'
REPORT=ROOT/'output/analysis/Fresh_Private_Risk_Step_V22_R1_Status.md'
TEST=ROOT/'tests/test_fresh_private_risk_step_v22_serialization.py'
METHODS=('erm_mean','erm_rfa','risk_mean','risk_rfa')


def historical(seed,method):
    return prior.screen.prior.prior.population.SOURCE/f'seed{seed}__{method}'/'metrics.json'


def inputs():
    config=yaml.safe_load(MATRIX.read_text());assert config['campaign_id']=='fresh_private_risk_step_v22'
    assert config['seeds']==[170501,170502] and config['expected_runs']==16 and config['step_controls']==['half','global_clip']
    assert config['rounds']==120 and config['clip']==2 and config['aggregate_radius']==1 and config['epsilon']==4
    assert config['batch_size']==240 and config['local_optimizer_steps']==0 and config['delta']==1e-5 and config['device']=='mps'
    assert config['methods']==list(METHODS) and config['candidate_order']==['global_clip','half'] and config['attacks']=='none'
    assert config['minimum_worst20_advantage_pp']==config['maximum_accuracy_loss_pp']==1.
    assert not config['test_evaluated'] and not config['automatic_attacks'] and not config['automatic_confirmation']
    a=ROOT/'output/analysis/PriSMA_Global_Step_V21_Independent_Audit.json'
    audit=json.loads(a.read_text());assert audit['audit_passed'] and not audit['admitted_to_end_to_end_screen']
    m=json.loads((diagnostic.OUT/'manifest.json').read_text());stamp=dict(m['source_stamp']);base.verify_stamp(stamp)
    files=[Path(__file__),MATRIX,PROTOCOL,TEST,a,ROOT/'output/analysis/Fresh_Private_Risk_Step_V22_Serialization_Amendment.md',ROOT/'scripts/run_fresh_private_risk_step_v22.py',ROOT/'tests/test_fresh_private_risk_step_v22.py']
    failed=ROOT/'results/ldp_gradient_far/fresh_private_risk_step_v22'
    assert json.loads((failed/'status.json').read_text())['status']=='failed'
    files += [failed/'status.json',failed/'manifest.json']
    files += [failed/f'seed170501__half__{method}/{name}' for method in ('erm_mean','erm_rfa') for name in ('metrics.json','checkpoint.pt')]
    files += [prior.OUT/f'seed{s}__fresh__{method}/{name}' for s in config['seeds'] for method in METHODS
              for name in ('metrics.json','simulator_oracle.json','orchestration_status.json')]
    files += [historical(s,method) for s in config['seeds'] for method in ('erm_mean','erm_rfa')]
    stamp.update({str(p.relative_to(ROOT)):base.digest(p) for p in files})
    return config,m['profile'],stamp


def jobs(config):
    return [dict(seed=s,step_control=c,method=m) for s in config['seeds'] for c in config['step_controls'] for m in METHODS]
def identifier(j):return f'seed{j["seed"]}__{j["step_control"]}__{j["method"]}'


def completed(j,stamp):
    path=OUT/identifier(j);s=path/'orchestration_status.json'
    if not s.exists() or json.loads(s.read_text())['status']!='completed':return False
    status=json.loads(s.read_text());r=json.loads((path/'metrics.json').read_text())
    assert status['metrics_sha256']==base.digest(path/'metrics.json') and status['oracle_sha256']==base.digest(path/'simulator_oracle.json')
    assert r['source_stamp']==stamp and r['job']==j and r['device']=='mps' and not r['test_evaluated']
    assert r['privacy']['epsilon_realized']<=4 and [x['round'] for x in r['rounds']]==list(range(1,121))
    verify(r['final']['validation']);return True


def train(j,config,profile,stamp,data,key):
    if completed(j,stamp):return
    path=OUT/identifier(j);path.mkdir(parents=True,exist_ok=True);p=ledger(2.,j['method'])
    reference=json.loads((prior.OUT/f'seed{j["seed"]}__fresh__{j["method"]}/metrics.json').read_text())
    assert json.loads(json.dumps(p))==reference['privacy'];expected=json.loads((prior.OUT/f'seed{j["seed"]}__fresh__{j["method"]}/simulator_oracle.json').read_text())['rounds']
    model=base.new_model(profile,j['seed']);rows=[];oracles=[];start=0;elapsed=0.;cp=path/'checkpoint.pt'
    key_sha=base.digest(prior.OUT/'simulator_secret.json')
    if cp.exists():
        state=torch.load(cp,map_location='cpu',weights_only=True)
        assert state['source_stamp']==stamp and state['job']==j and state['key_sha']==key_sha
        model.load_state_dict(state['model']);rows,oracles,start,elapsed,initial=(state[k] for k in ('rows','oracles','round','elapsed_seconds','initial'))
    else:initial=base.evaluate(model,data,'val')
    assert initial==reference['initial'] and data['splits']==reference['splits']
    base.save(path/'public_protocol.json',dict(config=config,profile=profile,job=j,privacy=p,source_stamp=stamp))
    code={k:v for k,v in stamp.items() if Path(k).suffix in ('.py','.yaml','.md')}
    for t in range(start,120):
        tick=time.monotonic();require_mps();base.verify_stamp(code)
        status=dict(status='running',device='mps',active=identifier(j),round=t+1,total_rounds=120,pid=os.getpid(),updated_unix=time.time())
        base.save(OUT/'status.json',status);base.save(path/'orchestration_status.json',status)
        sent=[];reports=[];local=[]
        before={name:v.detach().cpu().clone() for name,v in model.state_dict().items()}
        for cid,ids in enumerate(data['train']):
            rr=raw=None
            if j['method'].startswith('risk_'):
                rr,raw=private_risk(model,data['x'][ids],data['y'][ids],noise_std=p['risk_std'],
                    seed=base.seed_for(key,j['seed'],t,cid,'risk'),N=4800);reports.append(rr)
            ix=base.draw_indices(4800,240,base.seed_for(key,j['seed'],t,cid,'batch'));ih=base.ids_hash(ix)
            assert ih==expected[t]['clients'][cid]['batch_hash']
            _,g,norms,_=per_example(model,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)
            sent.append(release(g.mean(0),noise_std=p['gradient_std'],seed=base.seed_for(key,j['seed'],t,cid,'gaussian')))
            local.append(dict(client=cid,batch_hash=ih,gradient_clipped_count=int((norms>2).sum()),batch_size=240,
                private_risk=None if rr is None else float(rr),raw_risk=None if raw is None else float(raw)))
        messages,r,safety=sanitize(torch.stack(sent),None if not reports else torch.stack(reports))
        assert safety['invalid_message_rows']==safety['nonfinite_risk_reports']==0
        original,agg=aggregate(messages,r,kind=j['method'],round_number=t+1,horizon=120)
        step,ctrl=controlled_step(original/agg['eta'],agg['eta'],j['step_control'],radius=1.)
        agg.update(step_control=ctrl,message_safety=safety);base.apply_gradient(model,step,1.)
        val=base.evaluate(model,data,'val') if t+1 in profile['evaluation_rounds'] else None
        if val is not None:verify(val)
        ep=min((t+1)*wor_rdp(a,.05,p['gradient_z'])+((t+1)*a/(2*p['risk_z']**2) if j['method'].startswith('risk_') else 0.)
               +math.log(1e5)/(a-1) for a in range(2,65));assert ep<=4
        rows.append(dict(round=t+1,device='mps',epsilon_realized=ep,aggregation=agg,validation=val))
        oracles.append(dict(round=t+1,clients=local));elapsed+=time.monotonic()-tick
        base.checkpoint(cp,dict(model={name:v.detach().cpu() for name,v in model.state_dict().items()},previous_model=before,
            rows=rows,oracles=oracles,round=t+1,elapsed_seconds=elapsed,initial=initial,source_stamp=stamp,job=j,key_sha=key_sha))
        if val is not None:print(f'{identifier(j)} {t+1}/120 acc={val["accuracy_pct"]:.3f} W20={val["worst20_pct"]:.3f}',flush=True)
    base.verify_stamp(stamp)
    base.save(path/'metrics.json',dict(job=j,source_stamp=stamp,device='mps',privacy=p,initial=initial,rounds=rows,final=rows[-1],
        splits=data['splits'],test_evaluated=False,per_client_batch_gradient_evaluations=120,
        validation_and_oracles_not_private=True,elapsed_seconds=elapsed))
    base.save(path/'simulator_oracle.json',dict(privacy_protected=False,feeds_mechanism=False,rounds=oracles))
    base.save(path/'orchestration_status.json',dict(status='completed',device='mps',job=j,round=120,
        metrics_sha256=base.digest(path/'metrics.json'),oracle_sha256=base.digest(path/'simulator_oracle.json')))
    del model;torch.mps.empty_cache()


def contrast_passes(pairs):
    return (all(d['accuracy_pct']>=-1. and d['worst20_pct']>=1. for d in pairs)
            and all(st.mean(d[k] for d in pairs)<=0 for k in ('gap_best20_worst20_pp','variance_pp2')))


def decide(config,rows):
    assert len(rows)==16;index={(r['job']['seed'],r['job']['step_control'],r['job']['method']):r['final']['validation'] for r in rows}
    candidates=[]
    for candidate in config['candidate_order']:
        contrasts=[]
        for control_mode in ('half','global_clip','unchanged_v19','historical_v11'):
            for method in ('erm_mean','erm_rfa'):
                pairs=[]
                for seed in config['seeds']:
                    a=index[seed,candidate,'risk_rfa']
                    if control_mode=='unchanged_v19':b=json.loads((prior.OUT/f'seed{seed}__fresh__{method}/metrics.json').read_text())['final']['validation']
                    elif control_mode=='historical_v11':b=json.loads(historical(seed,method).read_text())['final']['validation']
                    else:b=index[seed,control_mode,method]
                    delta={k:a[k]-b[k] for k in ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2')}
                    pairs.append(dict(seed=seed,delta=delta,passed=delta['accuracy_pct']>=-1. and delta['worst20_pct']>=1.))
                contrasts.append(dict(control_mode=control_mode,control_method=method,pairs=pairs,
                    passed=contrast_passes([p['delta'] for p in pairs]),noise_paired=control_mode!='historical_v11'))
        candidates.append(dict(candidate=candidate,contrasts=contrasts,passed=all(c['passed'] for c in contrasts)))
    eligible=[c['candidate'] for c in candidates if c['passed']]
    return dict(candidates=candidates,selected=None if not eligible else eligible[0],eligible_for_independent_confirmation=bool(eligible),global_validation=False)


def report(config,stamp):
    rows=[json.loads((OUT/identifier(j)/'metrics.json').read_text()) for j in jobs(config) if completed(j,stamp)]
    lines=['# V22 — contrôle du pas, gradients privés frais','',f'**{len(rows)}/16 runs MPS valides** ; deux seeds de calibration, aucun test final.', '',
        '| Seed | Pas | Agrégation | Accuracy % | Worst-20 % | Gap pp | Variance pp² | Brier |',
        '|--:|:--|:--|--:|--:|--:|--:|--:|']
    for r in rows:
        j,v=r['job'],r['final']['validation'];lines.append(f'| {j["seed"]} | {j["step_control"]} | {j["method"]} | {v["accuracy_pct"]:.4f} | {v["worst20_pct"]:.4f} | {v["gap_best20_worst20_pp"]:.4f} | {v["variance_pp2"]:.4f} | {v["brier_loss"]:.5f} |')
    decision=None
    if len(rows)==16:
        decision=decide(config,rows);base.save(OUT/'evidence.json',dict(source_stamp=stamp,decision=decision))
        lines+=['',f'Admission à confirmation : **{"PASS" if decision["eligible_for_independent_confirmation"] else "FAIL"}**. Audit indépendant nécessaire. Aucune attaque lancée.', '']
    lines+=['','[Protocole prospectif](Fresh_Private_Risk_Step_V22_Protocol.md).'];REPORT.write_text('\n'.join(lines)+'\n')
    return len(rows),decision


def worker():
    require_mps();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            config,profile,stamp=inputs();manifest=dict(config=config,profile=profile,source_stamp=stamp)
            if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
            else:base.save(OUT/'manifest.json',manifest)
            if not (OUT/'tests.json').exists():
                p=subprocess.run([sys.executable,'-m','pytest',str(TEST),str(diagnostic.TEST),'-q'],cwd=ROOT,capture_output=True,text=True)
                base.save(OUT/'tests.json',dict(passed=p.returncode==0,source_stamp=stamp,output=p.stdout+p.stderr))
            tests=json.loads((OUT/'tests.json').read_text());assert tests['passed'] and tests['source_stamp']==stamp
            key=json.loads((prior.OUT/'simulator_secret.json').read_text())['key']
            for seed in config['seeds']:
                data=base.prepare(profile,seed)
                for j in [x for x in jobs(config) if x['seed']==seed]:
                    train(j,config,profile,stamp,data,key);n,_=report(config,stamp);print(f'{n}/16 completed',flush=True)
                del data;torch.mps.empty_cache()
            n,d=report(config,stamp);assert n==16
            base.save(OUT/'status.json',dict(status='completed',device='mps',valid_runs=16,
                eligible_for_independent_confirmation=d['eligible_for_independent_confirmation'],selected=d['selected'],next_campaign_launched=False))
        except Exception as exc:
            base.save(OUT/'status.json',dict(status='failed',device='mps',error=repr(exc),pid=os.getpid()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true',required=True);p.parse_args();worker()
