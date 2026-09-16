#!/usr/bin/env python3
"""Independent, fixed 24-run V29 confirmation; MPS only and no automatic attacks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
import yaml
from scripts import run_fair_objective_screen as base
from scripts.analyze_private_risk_confirmation_v12 import verify
from privacy.fair_objective import require_mps
from privacy.split_risk_gradient import private_risk
from privacy.scheduled_private_risk import aggregate
from privacy.private_risk_message_safety import sanitize
from privacy.full_population_private_risk_confirmation_v29 import SEEDS,METHODS,ARMS,TCRIT,plan,prefix,private_message,decide

NAME='full_population_private_risk_confirmation_v29'
OUT=ROOT/'results/ldp_gradient_far'/NAME
PARENT=ROOT/'results/ldp_gradient_far/full_population_private_risk_calibration_v28'
MATRIX=ROOT/'configs/ldp_gradient_far'/f'{NAME}.yaml'
TESTS=['tests/test_full_population_private_risk_confirmation_v29.py',
       'tests/test_full_population_private_risk_v28.py','tests/test_private_risk_message_safety.py',
       'tests/test_stable_weighted_rfa.py','tests/test_scheduled_private_risk.py']
REPORT=ROOT/'output/analysis/Full_Population_Private_Risk_Confirmation_V29_Status.md'


def inputs():
    c=yaml.safe_load(MATRIX.read_text())
    assert c['campaign_id']==NAME and c['device']=='mps' and c['seeds']==list(SEEDS)
    assert c['full_methods']==list(METHODS) and c['small_methods']==['erm_mean','erm_rfa']
    expected=dict(expected_runs=24,dataset='fashionmnist',model='lenet5_tanh',num_clients=10,
        partition='client_dirichlet_balanced',dirichlet_beta=.1,public_train_size=4800,validation_size=1200,
        gradient_accumulation_block=240,rounds=120,local_optimizer_steps=0,clip=2.,risk_scale=.5,
        epsilon=4.,delta=1e-5,epsilon_risk=.25,adjacency='replace_one',server_clip=None,
        eta_early=2.,eta_late=.5,eta_switch_after_round=60,primary_method='risk_rfa',primary_batch=4800,
        minimum_worst20_advantage_pp=1.,maximum_accuracy_loss_pp=1.,all_seeds_required=True,
        mean_gap_and_variance_must_not_increase=True,prospective_confirmation_wave=1,
        one_sided_alpha=.0125,paired_t_critical_df3=TCRIT,automatic_attacks=False,automatic_retry=False)
    assert all(c[k]==v for k,v in expected.items())
    assert c['batch_sizes']==[4800,240] and c['test_rounds']==[120]
    assert c['validation_rounds']==[1,*range(10,121,10)]
    old=json.loads((PARENT/'manifest.json').read_text()); stamp=dict(old['source_stamp']); base.verify_stamp(stamp)
    folder=ROOT/'output/analysis'
    needed=['Full_Population_Private_Risk_Calibration_V28_Analyse.json',
        'Full_Population_V28_Exact_Count_Gate_Audit.json',
        'Full_Population_Error_Attribution_V28b_Independent_Audit.json',
        'Full_Population_V28_Terminal_Aggregation_Counterfactual.json',
        'Full_Population_V28_Terminal_Counterfactual_Numerical_Audit.json']
    reports=[json.loads((folder/n).read_text()) for n in needed]
    a,e,b,d,n=reports
    assert a['audit_passed'] and a['valid_runs']==8 and a['decision']['calibration_passed']
    assert e['exact_calibration_passed'] and e['agrees_with_main_audit'] and not e['primary_gate_overridden']
    assert b['audit_passed'] and b['valid_states']==d['audited_states']==n['verified_states']==8
    ls=json.loads((ROOT/c['ledger_snapshot']).read_text())
    assert ls['waves'][0]['clean_confirmation_passed'] is False
    assert ls['waves'][1]['seeds']==list(SEEDS) and ls['waves'][1]['one_sided_alpha']==.0125
    assert ls['immutable_snapshot'] and ls['waves'][1]['clean_confirmation_passed'] is None
    assert set(SEEDS).isdisjoint(old['config']['seeds'])
    profile={k:c[k] for k in ('dataset','model','num_clients','partition','dirichlet_beta',
        'public_train_size','validation_size','rounds','local_optimizer_steps')}
    profile['evaluation_rounds']=c['validation_rounds']
    files=[Path(__file__),MATRIX,ROOT/c['protocol'],ROOT/c['ledger_snapshot'],
        ROOT/'privacy/full_population_private_risk_confirmation_v29.py',
        ROOT/'scripts/analyze_private_risk_confirmation_v12.py',*[ROOT/t for t in TESTS],
        *[folder/f for f in needed],PARENT/'manifest.json',PARENT/'status.json']
    files.extend(PARENT/f"seed{s}__{m}"/f for s in old['config']['seeds'] for m in METHODS
        for f in ('metrics.json','simulator_oracle.json','orchestration_status.json'))
    stamp.update({str(p.relative_to(ROOT)):base.digest(p) for p in files})
    # B=240 remains the exact historical public calibration, not a new tuning.
    historical=ROOT/'results/ldp_gradient_far/recursive_private_risk_calibration_v19'
    for method in ('erm_mean','erm_rfa'):
        old_p=json.loads((historical/f'seed170501__fresh__{method}'/'metrics.json').read_text())['privacy']
        p=plan(240,method)
        assert all(p[k]==old_p[k] for k in ('gradient_z','gradient_std','epsilon_realized','order'))
    return c,profile,stamp


def jobs(): return [dict(seed=s,batch=b,method=m) for s in SEEDS for b,m in ARMS]
def identifier(j): return f"seed{j['seed']}__b{j['batch']}__{j['method']}"
def cpu_state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def completed(j,stamp):
    d=OUT/identifier(j); sp=d/'orchestration_status.json'
    if not sp.exists() or json.loads(sp.read_text())['status']!='completed': return False
    s=json.loads(sp.read_text()); r=json.loads((d/'metrics.json').read_text())
    assert s['round']==120 and s['device']==r['device']=='mps' and r['source_stamp']==stamp and r['job']==j
    for name,key in [('metrics.json','metrics_sha256'),('simulator_oracle.json','oracle_sha256'),('checkpoint.pt','checkpoint_sha256')]:
        assert s[key]==base.digest(d/name)
    assert r['privacy']==json.loads(json.dumps(plan(j['batch'],j['method'])))
    assert r['privacy']['epsilon_realized']<=4 and r['test_evaluated'] and r['test_evaluation_rounds']==[120]
    assert [v['round'] for v in r['rounds']]==list(range(1,121))
    assert [v['round'] for v in r['rounds'] if v['test'] is not None]==[120]
    assert r['final']==r['rounds'][-1] and r['gradient_examples_per_client']==120*j['batch']
    assert r['private_gradient_releases_per_client']==120 and r['local_optimizer_steps']==0
    verify(r['final']['validation']); verify(r['final']['test'])
    return True


def progress(j,t,stage,client=None):
    state=dict(status='running',device='mps',active=identifier(j),round=t,total_rounds=120,
               stage=stage,client=client,pid=os.getpid(),updated_unix=time.time())
    base.save(OUT/'status.json',state); base.save(OUT/identifier(j)/'orchestration_status.json',state)


def train(j,c,profile,stamp,data,key):
    if completed(j,stamp): return
    d=OUT/identifier(j); d.mkdir(parents=True,exist_ok=True)
    p=plan(j['batch'],j['method']); model=base.new_model(profile,j['seed'])
    control=dict(seed=j['seed'],batch=4800,method='erm_mean'); ref=None; ref_oracle=None
    if j!=control:
        assert completed(control,stamp)
        ref=json.loads((OUT/identifier(control)/'metrics.json').read_text())
        ref_oracle=json.loads((OUT/identifier(control)/'simulator_oracle.json').read_text())['rounds']
    cp=d/'checkpoint.pt'; key_sha=base.digest(OUT/'simulator_secret.json')
    rows=[]; oracles=[]; start=0; elapsed=0.
    if cp.exists():
        saved=torch.load(cp,map_location='cpu',weights_only=True)
        assert saved['job']==j and saved['source_stamp']==stamp and saved['key_sha']==key_sha and saved['privacy']==p
        model.load_state_dict(saved['model'])
        rows,oracles,start,elapsed,initial=(saved[k] for k in ('rows','oracles','round','elapsed_seconds','initial'))
        assert [r['round'] for r in rows]==list(range(1,start+1))
    else: initial=base.evaluate(model,data,'val')
    if ref is not None: assert initial==ref['initial'] and data['splits']==ref['splits']
    base.save(d/'public_protocol.json',dict(config=c,profile=profile,job=j,privacy=p,source_stamp=stamp))
    code={k:v for k,v in stamp.items() if Path(k).suffix in ('.py','.yaml','.md')}
    for t in range(start,120):
        tick=time.monotonic(); require_mps(); base.verify_stamp(code); before=cpu_state(model)
        sent=[]; clean=[]; reports=[]; local=[]
        for cid,ids in enumerate(data['train']):
            progress(j,t+1,'private_gradients',cid); rr=raw=None
            if j['method'].startswith('risk_'):
                rr,raw=private_risk(model,data['x'][ids],data['y'][ids],noise_std=p['risk_std'],
                    seed=base.seed_for(key,j['seed'],t,cid,'risk'),N=4800); reports.append(rr)
            ix=base.draw_indices(4800,j['batch'],base.seed_for(key,j['seed'],t,cid,'batch'))
            assert len(torch.unique(ix))==j['batch']
            ih=base.ids_hash(ix); ph=base.ids_hash(ix[:240])
            if ref_oracle is not None:
                old=ref_oracle[t]['clients'][cid]
                assert ph==old['batch_prefix240_sha256']
                if j['batch']==4800: assert ih==old['indices_sha256']
            message,query,diag=private_message(model,data['x'][ids[ix]],data['y'][ids[ix]],batch=j['batch'],
                noise_std=p['gradient_std'],seed=base.seed_for(key,j['seed'],t,cid,'gaussian'))
            assert diag['gaussian_releases']==1 and diag['local_optimizer_steps']==0
            sent.append(message); clean.append(query)
            local.append(dict(client=cid,batch_size=j['batch'],indices_sha256=ih,batch_prefix240_sha256=ph,
                private_message_sha256=base.ids_hash(message),clean_mean_sha256=base.ids_hash(query),
                private_risk=None if rr is None else float(rr),raw_risk=None if raw is None else float(raw),gradient=diag))
        for name,value in model.state_dict().items(): assert torch.equal(value,before[name].to('mps'))
        messages,r,safety=sanitize(torch.stack(sent),None if not reports else torch.stack(reports))
        assert safety['invalid_message_rows']==safety['nonfinite_risk_reports']==0
        step,agg=aggregate(messages,r,kind=j['method'],round_number=t+1,horizon=120)
        assert agg['eta']==(2. if t<60 else .5); agg['message_safety']=safety
        base.apply_gradient(model,step,1.); progress(j,t+1,'evaluation_and_checkpoint')
        val=base.evaluate(model,data,'val') if t+1 in c['validation_rounds'] else None
        test=base.evaluate(model,data,'test') if t+1==120 else None
        if val is not None: verify(val)
        if test is not None: verify(test)
        eps,order=prefix(p,t+1); assert eps<=4
        rows.append(dict(round=t+1,device='mps',epsilon_realized=eps,epsilon_order=order,
            aggregation=agg,validation=val,test=test))
        oracles.append(dict(round=t+1,clients=local)); elapsed+=time.monotonic()-tick
        base.checkpoint(cp,dict(model=cpu_state(model),pre_round_model=before,
            last_private_messages=messages.detach().cpu(),last_clean_means=torch.stack(clean).detach().cpu(),
            last_reports=None if r is None else r.detach().cpu(),last_step=step.detach().cpu(),rows=rows,oracles=oracles,
            round=t+1,elapsed_seconds=elapsed,initial=initial,privacy=p,job=j,source_stamp=stamp,
            key_sha=key_sha,privacy_protected=False))
        if val is not None:
            print(f"{identifier(j)} {t+1}/120 val_acc={val['accuracy_pct']:.4f} W20={val['worst20_pct']:.4f} elapsed={elapsed:.1f}s",flush=True)
    base.verify_stamp(stamp)
    base.save(d/'metrics.json',dict(job=j,source_stamp=stamp,device='mps',privacy=p,initial=initial,
        rounds=rows,final=rows[-1],splits=data['splits'],test_evaluated=True,test_evaluation_rounds=[120],
        validation_test_and_oracles_not_private=True,gradient_examples_per_client=120*j['batch'],
        private_gradient_releases_per_client=120,local_optimizer_steps=0,elapsed_seconds=elapsed))
    base.save(d/'simulator_oracle.json',dict(privacy_protected=False,feeds_mechanism=False,rounds=oracles))
    base.save(d/'orchestration_status.json',dict(status='completed',device='mps',job=j,round=120,
        metrics_sha256=base.digest(d/'metrics.json'),oracle_sha256=base.digest(d/'simulator_oracle.json'),
        checkpoint_sha256=base.digest(cp),gate_evaluated=False))
    del model; torch.mps.empty_cache()


def report(c,stamp):
    rows=[json.loads((OUT/identifier(j)/'metrics.json').read_text()) for j in jobs() if completed(j,stamp)]
    lines=['# V29 — confirmation indépendante du gradient de population privé','',
        f'**{len(rows)}/24 runs MPS terminés.** Test final120, candidate risque-RFA B=4800 fixée.', '',
        '| Seed | Batch | Méthode | Test accuracy (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) |',
        '|--:|--:|:--|--:|--:|--:|--:|']
    for r in rows:
        j,v=r['job'],r['final']['test']
        lines.append(f"| {j['seed']} | {j['batch']} | {j['method']} | {v['accuracy_pct']:.4f} | {v['worst20_pct']:.4f} | {v['gap_best20_worst20_pp']:.4f} | {v['variance_pp2']:.4f} |")
    decision=decide(rows) if len(rows)==24 else None
    if decision is not None:
        base.save(OUT/'evidence_unverified.json',dict(source_stamp=stamp,decision=decision,independent_audit_passed=False))
        lines+=['',f"Critères calculés par le runner : {'PASS' if decision['clean_confirmation_passed'] else 'FAIL'}. Audit indépendant encore requis."]
    lines+=['','Aucune validation conjointe ni attaque automatique. Tous les résultats sont conservés.', '',
        '[Protocole](Full_Population_Private_Risk_Confirmation_V29_Protocol.md).']
    REPORT.write_text('\n'.join(lines)+'\n'); return len(rows)


def worker():
    require_mps(); OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            c,profile,stamp=inputs()
            manifest=dict(config=c,profile=profile,source_stamp=stamp,device='mps',torch_version=str(torch.__version__),fallback=False)
            if (OUT/'manifest.json').exists(): assert json.loads((OUT/'manifest.json').read_text())==manifest
            else:
                found=subprocess.run(['rg','--files','--hidden','--no-ignore','results'],cwd=ROOT,capture_output=True,text=True,check=True)
                pattern=re.compile(r'(?<![0-9])18070[1-4](?![0-9])'); own=str(OUT.relative_to(ROOT))+'/'
                conflicts=[p for p in found.stdout.splitlines() if pattern.search(p) and not p.startswith(own)]
                assert not conflicts, 'Confirmation seeds already used in prior result paths'
                base.save(OUT/'seed_reservation.json',dict(seeds=list(SEEDS),prior_output_path_conflicts=conflicts,
                    search_scope='All hidden/ignored result paths; declarations searched before protocol creation',reserved_before_training=True))
                base.save(OUT/'manifest.json',manifest)
            if not (OUT/'tests.json').exists():
                tests=subprocess.run([sys.executable,'-m','pytest',*TESTS,'-q'],cwd=ROOT,capture_output=True,text=True)
                base.save(OUT/'tests.json',dict(passed=tests.returncode==0,source_stamp=stamp,output=tests.stdout+tests.stderr))
            evidence=json.loads((OUT/'tests.json').read_text()); assert evidence['passed'] and evidence['source_stamp']==stamp
            if not (OUT/'simulator_secret.json').exists():
                assert not list(OUT.glob('seed*/checkpoint.pt'))
                base.save(OUT/'simulator_secret.json',dict(key=secrets.token_hex(32),private_release=False))
            key=json.loads((OUT/'simulator_secret.json').read_text())['key']
            for seed in SEEDS:
                data=base.prepare(profile,seed)
                for j in [j for j in jobs() if j['seed']==seed]:
                    train(j,c,profile,stamp,data,key); count=report(c,stamp)
                    print(f'V29 {count}/24 completed; independent audit required',flush=True)
                del data; torch.mps.empty_cache()
            assert report(c,stamp)==24; base.verify_stamp(stamp)
            base.save(OUT/'status.json',dict(status='completed',device='mps',valid_runs=24,
                independent_audit_passed=False,global_validation=False,next_campaign_launched=False,source_stamp=stamp))
        except Exception as exc:
            state=dict(status='failed',device='mps',error=repr(exc),pid=os.getpid())
            base.save(OUT/f'failure_{time.time_ns()}.json',state); base.save(OUT/'status.json',state); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--resume',required=True,action='store_true')
    parser.parse_args(); worker()
