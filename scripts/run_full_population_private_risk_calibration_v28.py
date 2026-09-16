#!/usr/bin/env python3
"""Eight frozen clean calibration runs, MPS only, no automatic promotion."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
import yaml
from scripts import run_fair_objective_screen as base
from scripts.run_private_risk_confirmation_v12 import audit_evaluation
from privacy.fair_objective import require_mps
from privacy.full_population_private_risk_v28 import METHODS, ledger, prefix_epsilon, private_population_gradient
from privacy.split_risk_gradient import private_risk
from privacy.scheduled_private_risk import aggregate
from privacy.private_risk_message_safety import sanitize

NAME='full_population_private_risk_calibration_v28'
OUT=ROOT/'results/ldp_gradient_far'/NAME
SOURCE=ROOT/'results/ldp_gradient_far/recursive_private_risk_calibration_v19'
MATRIX=ROOT/'configs/ldp_gradient_far'/f'{NAME}.yaml'
TESTS=['tests/test_full_population_private_risk_v28.py','tests/test_private_risk_message_safety.py',
       'tests/test_stable_weighted_rfa.py','tests/test_scheduled_private_risk.py']
REPORT=ROOT/'output/analysis/Full_Population_Private_Risk_Calibration_V28_Status.md'


def inputs():
    m=yaml.safe_load(MATRIX.read_text())
    assert m['campaign_id']==NAME and m['device']=='mps' and m['seeds']==[170501,170502]
    assert m['methods']==list(METHODS) and m['expected_runs']==8 and m['primary_method']=='risk_rfa'
    expected=dict(dataset='fashionmnist',model='lenet5_tanh',num_clients=10,partition='client_dirichlet_balanced',
        dirichlet_beta=.1,public_train_size=4800,validation_size=1200,batch_size=4800,gradient_accumulation_block=240,
        rounds=120,local_optimizer_steps=0,clip=2.,risk_scale=.5,epsilon=4.,delta=1e-5,epsilon_risk=.25,
        adjacency='replace_one',sampling='full_population',server_clip=None,eta_early=2.,eta_late=.5,
        eta_switch_after_round=60,minimum_worst20_advantage_pp=1.,maximum_accuracy_loss_pp=1.,
        all_seeds_required=True,mean_gap_and_variance_must_not_increase=True,test_evaluated=False,
        automatic_confirmation=False,automatic_attacks=False)
    assert all(m[k]==v for k,v in expected.items())
    assert m['validation_rounds']==[1,*range(10,121,10)]
    assert m['controls']==['full_erm_mean','full_erm_rfa','v19_fresh_erm_mean','v19_fresh_erm_rfa']
    assert ROOT/m['paired_randomness_source']==SOURCE/'simulator_secret.json'
    old=json.loads((SOURCE/'manifest.json').read_text())
    stamp=dict(old['source_stamp']);base.verify_stamp(stamp)
    profile=dict(old['profile'],**{k:m[k] for k in ('dataset','model','num_clients','partition','dirichlet_beta',
                 'public_train_size','validation_size','batch_size','rounds','local_optimizer_steps')},
                 evaluation_rounds=m['validation_rounds'])
    # Clean old selection fields out of the execution profile: only m defines this gate.
    for key in ('screen','expected_runs','clips','risk_scales','reserved_confirmation_seeds'):
        profile.pop(key,None)
    profile.update(campaign_id=NAME,protocol=m['protocol'],parent_campaign='recursive_private_risk_calibration_v19')
    paths=[Path(__file__),MATRIX,ROOT/m['protocol'],ROOT/'privacy/full_population_private_risk_v28.py',
           *[ROOT/f for f in TESTS],ROOT/'output/analysis/Additive_Gaussian_WOR_V27_Public_Audit.json',
           ROOT/'output/analysis/Private_Risk_Factorial_Diagnostic_V26_Analyse.json',
           ROOT/'output/analysis/Private_Fairness_Robustness_Prospective_Confirmation_Ledger.json']
    assert json.loads(paths[-3].read_text())['audit_passed'] and json.loads(paths[-2].read_text())['audit_passed']
    for seed in m['seeds']:
        for method in METHODS:
            d=SOURCE/f'seed{seed}__fresh__{method}'
            s=json.loads((d/'orchestration_status.json').read_text())
            r=json.loads((d/'metrics.json').read_text())
            assert s['status']=='completed' and r['device']=='mps' and r['source_stamp']==old['source_stamp']
            assert s['metrics_sha256']==base.digest(d/'metrics.json') and s['oracle_sha256']==base.digest(d/'simulator_oracle.json')
            paths += [d/'metrics.json',d/'simulator_oracle.json',d/'orchestration_status.json']
    stamp.update({str(p.relative_to(ROOT)):base.digest(p) for p in paths})
    return m,profile,stamp


def jobs(m):return [dict(seed=s,method=k) for s in m['seeds'] for k in m['methods']]
def identifier(j):return f"seed{j['seed']}__{j['method']}"
def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def completed(j,stamp):
    folder=OUT/identifier(j)
    path=folder/'orchestration_status.json'
    if not path.exists() or json.loads(path.read_text())['status']!='completed':return False
    s=json.loads(path.read_text());r=json.loads((folder/'metrics.json').read_text())
    assert r['source_stamp']==stamp and r['job']==j and r['device']=='mps' and not r['test_evaluated']
    for name,key in [('metrics.json','metrics_sha256'),('simulator_oracle.json','oracle_sha256'),('checkpoint.pt','checkpoint_sha256')]:
        assert s[key]==base.digest(folder/name)
    assert [t['round'] for t in r['rounds']]==list(range(1,121)) and r['privacy']['epsilon_realized']<=4
    assert r['gradient_examples_per_client']==576000 and r['private_gradient_releases_per_client']==120
    audit_evaluation(r['final']['validation'])
    return True


def progress(job,round_number,stage,client=None):
    s=dict(status='running',device='mps',active=identifier(job),round=round_number,total_rounds=120,
           stage=stage,client=client,pid=os.getpid(),updated_unix=time.time())
    base.save(OUT/'status.json',s);base.save(OUT/identifier(job)/'orchestration_status.json',s)


def train(j,m,profile,stamp,data,key):
    if completed(j,stamp):return
    d=OUT/identifier(j);d.mkdir(parents=True,exist_ok=True)
    p=ledger(j['method']);model=base.new_model(profile,j['seed'])
    prior=json.loads((SOURCE/f"seed{j['seed']}__fresh__{j['method']}"/'metrics.json').read_text())
    prior_oracle=json.loads((SOURCE/f"seed{j['seed']}__fresh__{j['method']}"/'simulator_oracle.json').read_text())
    assert data['splits']==prior['splits']
    key_sha=base.digest(SOURCE/'simulator_secret.json')
    cp=d/'checkpoint.pt';rows=[];oracles=[];start=0;elapsed=0.
    if cp.exists():
        saved=torch.load(cp,map_location='cpu',weights_only=True)
        assert saved['source_stamp']==stamp and saved['job']==j and saved['key_sha']==key_sha
        assert saved['privacy']==p
        model.load_state_dict(saved['model'])
        rows,oracles,start,elapsed,initial=saved['rows'],saved['oracles'],saved['round'],saved['elapsed_seconds'],saved['initial']
        assert [r['round'] for r in rows]==list(range(1,start+1))
    else:
        initial=base.evaluate(model,data,'val');assert initial==prior['initial']
    code_stamp={k:v for k,v in stamp.items() if Path(k).suffix in ('.py','.yaml','.md')}
    base.save(d/'public_protocol.json',dict(config=m,profile=profile,job=j,privacy=p,source_stamp=stamp))
    for t in range(start,120):
        tick=time.monotonic();require_mps();base.verify_stamp(code_stamp)
        progress(j,t+1,'population_gradients')
        before=cpu_state(model);sent=[];reports=[];clean=[];local=[]
        for cid,ids in enumerate(data['train']):
            progress(j,t+1,'population_gradients',cid)
            rr=raw=None
            if j['method'].startswith('risk_'):
                rr,raw=private_risk(model,data['x'][ids],data['y'][ids],noise_std=p['risk_std'],
                                  seed=base.seed_for(key,j['seed'],t,cid,'risk'),N=4800)
                reports.append(rr)
            ix=base.draw_indices(4800,4800,base.seed_for(key,j['seed'],t,cid,'batch'))
            assert torch.equal(ix.sort().values,torch.arange(4800,device='mps'))
            legacy_prefix=base.ids_hash(ix[:240])
            assert legacy_prefix==prior_oracle['rounds'][t]['clients'][cid]['batch_hash']
            message,query,diag=private_population_gradient(model,data['x'][ids[ix]],data['y'][ids[ix]],
                C=2.,block_size=240,noise_std=p['gradient_std'],seed=base.seed_for(key,j['seed'],t,cid,'gaussian'))
            assert diag['population_size']==4800 and diag['accumulation_blocks']==20 and diag['gaussian_releases']==1
            local.append(dict(client=cid,permutation_sha256=base.ids_hash(ix),legacy_prefix_sha256=legacy_prefix,
                  private_message_sha256=base.ids_hash(message),clean_mean_sha256=base.ids_hash(query),
                  private_risk=None if rr is None else float(rr),raw_risk=None if raw is None else float(raw),
                  gradient=diag))
            sent.append(message);clean.append(query)
        for name,value in model.state_dict().items():
            assert torch.equal(value,before[name].to('mps')), 'Local gradient computation modified the global model'
        messages,r,safety=sanitize(torch.stack(sent),None if not reports else torch.stack(reports))
        assert safety['invalid_message_rows']==safety['nonfinite_risk_reports']==0
        step,agg=aggregate(messages,r,kind=j['method'],round_number=t+1,horizon=120)
        assert agg['eta']==(2. if t+1<=60 else .5)
        agg['message_safety']=safety
        base.apply_gradient(model,step,1.)
        progress(j,t+1,'validation_and_checkpoint')
        val=base.evaluate(model,data,'val') if t+1 in m['validation_rounds'] else None
        if val is not None:audit_evaluation(val)
        ep,order=prefix_epsilon(p,t+1);assert ep<=4
        rows.append(dict(round=t+1,device='mps',epsilon_realized=ep,epsilon_order=order,aggregation=agg,validation=val))
        oracles.append(dict(round=t+1,clients=local));elapsed+=time.monotonic()-tick
        base.checkpoint(cp,dict(model=cpu_state(model),pre_round_model=before,
            last_private_messages=messages.detach().cpu(),last_reports=None if r is None else r.detach().cpu(),
            last_clean_means=torch.stack(clean).detach().cpu(),last_step=step.detach().cpu(),
            rows=rows,oracles=oracles,round=t+1,elapsed_seconds=elapsed,initial=initial,
            privacy=p,job=j,source_stamp=stamp,key_sha=key_sha,privacy_protected=False))
        if val is not None:
            print(f"{identifier(j)} {t+1}/120 acc={val['accuracy_pct']:.4f} W20={val['worst20_pct']:.4f} elapsed={elapsed:.1f}s",flush=True)
    base.verify_stamp(stamp)
    base.save(d/'metrics.json',dict(job=j,source_stamp=stamp,device='mps',privacy=p,initial=initial,
        rounds=rows,final=rows[-1],splits=data['splits'],test_evaluated=False,validation_and_oracles_not_private=True,
        gradient_examples_per_client=120*4800,private_gradient_releases_per_client=120,local_optimizer_steps=0,
        elapsed_seconds=elapsed))
    base.save(d/'simulator_oracle.json',dict(privacy_protected=False,feeds_mechanism=False,rounds=oracles))
    base.save(d/'orchestration_status.json',dict(status='completed',device='mps',job=j,round=120,
        metrics_sha256=base.digest(d/'metrics.json'),oracle_sha256=base.digest(d/'simulator_oracle.json'),
        checkpoint_sha256=base.digest(cp),gate_evaluated=False))
    del model;torch.mps.empty_cache()


def report(m,stamp):
    rows=[json.loads((OUT/identifier(j)/'metrics.json').read_text()) for j in jobs(m) if completed(j,stamp)]
    lines=['# V28 — gradient privé de population complète','',f'**{len(rows)}/8 runs MPS terminés.** '
        'Calibration uniquement, test non évalué. Le gate attend l’audit indépendant.', '',
        '| Seed | Méthode | Accuracy (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) | CE | Demi-Brier |',
        '|--:|:--|--:|--:|--:|--:|--:|--:|']
    for r in rows:
        j,v=r['job'],r['final']['validation']
        lines.append(f"| {j['seed']} | {j['method']} | {v['accuracy_pct']:.4f} | {v['worst20_pct']:.4f} | {v['gap_best20_worst20_pp']:.4f} | {v['variance_pp2']:.4f} | {v['ce_loss']:.6f} | {v['brier_loss']:.6f} |")
    lines+=['','B=N : une moyenne de4800 gradients individuellement clippés, un seul ajout gaussien, puis une mise à jour serveur. '
            'Aucune époque d’optimisation locale, aucun contrôle de risque sans bruit utilisé par le serveur.', '',
            '[Protocole](Full_Population_Private_Risk_Calibration_V28_Protocol.md).']
    REPORT.write_text('\n'.join(lines)+'\n')
    return len(rows)


def worker():
    require_mps();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            m,profile,stamp=inputs()
            manifest=dict(config=m,profile=profile,source_stamp=stamp,device='mps',torch_version=str(torch.__version__),fallback=False)
            if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
            else:base.save(OUT/'manifest.json',manifest)
            if not (OUT/'tests.json').exists():
                proc=subprocess.run([sys.executable,'-m','pytest',*TESTS,'-q'],cwd=ROOT,capture_output=True,text=True)
                base.save(OUT/'tests.json',dict(passed=proc.returncode==0,output=proc.stdout+proc.stderr,source_stamp=stamp))
            tests=json.loads((OUT/'tests.json').read_text());assert tests['passed'] and tests['source_stamp']==stamp
            key=json.loads((SOURCE/'simulator_secret.json').read_text())['key']
            for seed in m['seeds']:
                data=base.prepare(profile,seed)
                for j in [j for j in jobs(m) if j['seed']==seed]:
                    train(j,m,profile,stamp,data,key)
                    count=report(m,stamp);print(f'V28 {count}/8 runs completed; independent audit still required',flush=True)
                del data;torch.mps.empty_cache()
            assert report(m,stamp)==8;base.verify_stamp(stamp)
            base.save(OUT/'status.json',dict(status='completed',device='mps',valid_runs=8,gate_evaluated=False,
                global_validation=False,next_campaign_launched=False,source_stamp=stamp))
        except Exception as exc:
            base.save(OUT/f'failure_{time.time_ns()}.json',dict(error=repr(exc),device='mps',pid=os.getpid()))
            base.save(OUT/'status.json',dict(status='failed',device='mps',error=repr(exc),pid=os.getpid()))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true',required=True)
    parser.parse_args();worker()
