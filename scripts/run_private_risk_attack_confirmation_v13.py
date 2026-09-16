#!/usr/bin/env python3
"""Matched clean/attack/recovery validation, conditional on independent V12 PASS."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
import yaml
from scripts import run_private_risk_confirmation_v12 as parent
from scripts import run_fair_objective_screen as base
from privacy.fair_objective import require_mps,per_example,release
from privacy.split_risk_gradient import private_risk
from privacy.scheduled_private_risk import aggregate
from privacy.private_risk_message_safety import sanitize
from privacy.private_risk_attacks import inject
from privacy.stable_weighted_rfa import stable_norm

NAME='private_risk_attack_confirmation_v13'
MATRIX=ROOT/'configs/ldp_gradient_far'/f'{NAME}.yaml'
OUT=ROOT/'results/ldp_gradient_far'/NAME
LOG=ROOT/'logs'/f'{NAME}.log'
REPORT=ROOT/'output/analysis/Private_Risk_Attack_Confirmation_V13_Status.md'
TESTS=['tests/test_private_risk_attack_confirmation_v13.py','tests/test_private_risk_attacks.py',
       'tests/test_private_risk_message_safety.py','tests/test_stable_weighted_rfa.py']


def config():
    m=yaml.safe_load(MATRIX.read_text())
    assert m['campaign_id']==NAME and m['device']=='mps' and m['rounds']==120
    assert m['confirmation_seeds']==[170601,170602,170603,170604]
    assert m['methods']==['erm_mean','erm_rfa','risk_mean','risk_rfa']
    assert m['attacks']==['none','abrupt_bf','persistent_alie','slow_ipm'] and m['expected_runs']==64
    assert m['byzantine_ids']==[0,1] and m['forged_risk']==1
    assert m['attack_start']==31 and m['temporary_attack_end']==90 and m['recovery_rounds']==[91,120]
    assert m['bf_multiplier']==10 and m['alie_std_multiplier']==1.5
    assert m['ipm_ramp_end']==60 and m['ipm_final_multiplier']==2
    assert m['clip']==2 and m['risk_scale']==.5 and m['epsilon']==4 and m['delta']==1e-5
    assert m['require_clean_confirmation_passed'] and not m['automatic_next_campaign']
    assert m['minimum_worst20_advantage_pp_per_seed']==m['maximum_accuracy_loss_pp_per_seed']==1
    assert m['maximum_loss_vs_own_clean_accuracy_pp']==m['maximum_loss_vs_own_clean_worst20_pp']==5
    assert m['primary_endpoints']==dict(abrupt_bf=90,persistent_alie=120,slow_ipm=90)
    assert m['test_evaluation_rounds']==[90,120] and m['maximum_solver_gap_diagnostic']==.001
    return m


def jobs(m):
    return [dict(seed=s,method=k,attack=a) for s in m['confirmation_seeds'] for a in m['attacks'] for k in m['methods']]


def identifier(j):return f"seed{j['seed']}__{j['method']}__{j['attack']}"


def inputs(m):
    previous=json.loads((parent.OUT/'manifest.json').read_text())
    base.verify_stamp(previous['source_stamp'])
    status=json.loads((parent.OUT/'status.json').read_text())
    if status['status']!='completed' or not status['clean_confirmation_passed']:
        raise RuntimeError('Independent clean confirmation not positive; attacks are not authorized by this protocol')
    auditpath=ROOT/'output/analysis/Private_Risk_Confirmation_V12_Analyse.json'
    audit=json.loads(auditpath.read_text())
    assert audit['audit_passed'] and audit['clean_confirmation_passed'] and audit['valid_runs']==16
    assert audit['manifest_sha256']==base.digest(parent.OUT/'manifest.json')
    for j in parent.jobs(previous['config']):
        assert parent.completed(j,previous['config'],previous['profile'],previous['source_stamp'])
    profile=dict(previous['profile'],evaluation_rounds=m['validation_evaluation_rounds'])
    stamp=dict(previous['source_stamp'])
    paths=[Path(__file__),MATRIX,ROOT/m['protocol'],ROOT/'privacy/private_risk_attacks.py',auditpath,
           ROOT/'scripts/analyze_private_risk_confirmation_v12.py',*[ROOT/p for p in TESTS]]
    stamp.update({str(p.relative_to(ROOT)):base.digest(p) for p in paths})
    return profile,stamp


def honest_metrics(full):
    """Keep identities 2..9 fixed, including in clean and recovery phases."""
    clients=full['clients'][2:]
    assert len(clients)==8
    acc=[c['accuracy'] for c in clients];a=sorted(acc);tail=2
    return dict(accuracy_pct=100*st.mean(acc),client_accuracy_pct=100*st.mean(acc),
        worst20_pct=100*st.mean(a[:tail]),gap_best20_worst20_pp=100*(st.mean(a[-tail:])-st.mean(a[:tail])),
        gap_best_worst_pp=100*(max(a)-min(a)),variance_pp2=10000*st.pvariance(acc),
        ce_loss=st.mean(c['ce_loss'] for c in clients),brier_loss=st.mean(c['brier_loss'] for c in clients),
        balanced_accuracy_pct=100*st.mean(c['balanced_accuracy'] for c in clients),
        clients=clients,honest_ids=list(range(2,10)),worst20_client_count=2)


def at(r,t):return next(x['test_honest'] for x in r['rounds'] if x['round']==t)


def decide(m,rows):
    expected={(j['seed'],j['method'],j['attack']) for j in jobs(m)}
    index={(r['job']['seed'],r['job']['method'],r['job']['attack']):r for r in rows}
    if len(rows)!=64 or set(index)!=expected or len(index)!=len(rows):
        raise ValueError('All 64 unique clean/attack runs required')
    results={}
    keys=['accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2']
    for attack,t in m['primary_endpoints'].items():
        comparisons={}
        for control in [*m['primary_controls'],'risk_mean']:
            pairs=[]
            for seed in m['confirmation_seeds']:
                a,b=at(index[seed,'risk_rfa',attack],t),at(index[seed,control,attack],t)
                delta={k:a[k]-b[k] for k in keys}
                pairs.append(dict(seed=seed,delta=delta,passed=delta['accuracy_pct']>=-1 and delta['worst20_pct']>=1))
            s={k:parent.paired_summary([p['delta'][k] for p in pairs],m['paired_t_critical_df3']) for k in keys}
            gates=dict(all_seeds=all(p['passed'] for p in pairs),worst20_ci=s['worst20_pct']['ci95'][0]>0,
                accuracy_ci=s['accuracy_pct']['ci95'][0]>=-1,gap=s['gap_best20_worst20_pp']['mean']<=0,
                variance=s['variance_pp2']['mean']<=0)
            comparisons[control]=dict(primary=control in m['primary_controls'],pairs=pairs,summaries=s,
                                     gates=gates,passed=all(gates.values()))
        clean=[]
        for seed in m['confirmation_seeds']:
            for endpoint in sorted({t,120}):
                a,b=at(index[seed,'risk_rfa',attack],endpoint),at(index[seed,'risk_rfa','none'],endpoint)
                da,dw=a['accuracy_pct']-b['accuracy_pct'],a['worst20_pct']-b['worst20_pct']
                clean.append(dict(seed=seed,round=endpoint,accuracy_delta_pp=da,worst20_delta_pp=dw,passed=da>=-5 and dw>=-5))
        solver_max=max(x['aggregation']['solver']['unsmoothed_objective_gap_upper']
            for r in rows if r['job']['attack']==attack and r['job']['method']=='risk_rfa' for x in r['rounds'])
        passed=all(comparisons[c]['passed'] for c in m['primary_controls']) and all(c['passed'] for c in clean) and solver_max<=.001
        results[attack]=dict(primary_round=t,comparisons=comparisons,own_clean_comparisons=clean,
                            solver_max_gap=solver_max,solver_gate=solver_max<=.001,passed=passed)
    return dict(attacks=results,attack_confirmation_passed=all(a['passed'] for a in results.values()),
                scope='Only this fixed benchmark, seeds and three attacks; no universal model guarantee')


def completed(j,m,profile,stamp):
    d=OUT/identifier(j);path=d/'orchestration_status.json'
    if not path.exists():return False
    status=json.loads(path.read_text())
    if status['status']!='completed':return False
    r=json.loads((d/'metrics.json').read_text())
    assert status['metrics_sha256']==base.digest(d/'metrics.json') and status['oracle_sha256']==base.digest(d/'simulator_oracle.json')
    assert r['job']==j and r['source_stamp']==stamp and r['device']=='mps'
    assert [x['round'] for x in r['rounds']]==list(range(1,121))
    assert [x['round'] for x in r['rounds'] if x['test'] is not None]==[90,120]
    assert r['privacy']['epsilon_realized']<=4 and r['privacy']['delta']==1e-5
    for row in r['rounds']:
        assert row['device']=='mps' and row['epsilon_realized']<=4
        assert row['aggregation']['eta']==(2. if row['round']<=60 else .5)
        assert row['attack']['active']==(j['attack']!='none' and 31<=row['round']<=(120 if j['attack']=='persistent_alie' else 90))
        if row['test'] is not None:
            parent.audit_evaluation(row['test']);parent.audit_evaluation(row['test_honest'])
            assert row['test_honest']==honest_metrics(row['test'])
    prior=json.loads((parent.OUT/parent.identifier(dict(seed=j['seed'],method=j['method']))/'metrics.json').read_text())
    assert r['initial']==prior['initial'] and r['splits']==prior['splits']
    for row,old in zip(r['rounds'],prior['rounds']):
        if (j['attack']=='none' or row['round']<=30) and old['validation'] is not None:
            assert row['validation']==old['validation'],'Clean/prefix replay differs'
    if j['attack']=='none':assert r['final']['test']==prior['final']['test']
    return True


def train(m,profile,j,data,key,stamp):
    if completed(j,m,profile,stamp):return
    d=OUT/identifier(j);d.mkdir(parents=True,exist_ok=True)
    p=parent.ledger.privacy(profile,parent.previous.arm(j['method']))
    model=base.new_model(profile,j['seed']);rows,oracle,start=[],[],0
    cp=d/'checkpoint.pt';key_hash=base.digest(parent.OUT/'simulator_secret.json')
    if cp.exists():
        saved=torch.load(cp,map_location='cpu',weights_only=True)
        assert saved['source_stamp']==stamp and saved['job']==j and saved['key_hash']==key_hash
        model.load_state_dict(saved['model']);rows,oracle,start,initial=saved['rows'],saved['oracle'],saved['round'],saved['initial']
    else:initial=base.evaluate(model,data,'val')
    base.save(d/'public_protocol.json',dict(config=m,profile=profile,job=j,privacy=p,source_stamp=stamp))
    for t in range(start,120):
        require_mps();base.verify_stamp(stamp)
        status=dict(status='running',active=identifier(j),device='mps',round=t+1,total_rounds=120,pid=os.getpid(),updated_unix=time.time())
        base.save(OUT/'status.json',status);base.save(d/'orchestration_status.json',status)
        messages,reports,clean,local=[],[],[],[]
        for cid,ids in enumerate(data['train']):
            rr,raw=None,None
            if j['method'].startswith('risk_'):
                rr,raw=private_risk(model,data['x'][ids],data['y'][ids],noise_std=p['risk_std'],
                    seed=base.seed_for(key,j['seed'],t,cid,'risk'),N=profile['public_train_size'])
                reports.append(rr)
            idx=base.draw_indices(len(ids),profile['batch_size'],base.seed_for(key,j['seed'],t,cid,'batch'))
            _,g,norms,_=per_example(model,data['x'][ids[idx]],data['y'][ids[idx]],clip_norm=2.)
            query=g.mean(0);clean.append(query)
            messages.append(release(query,noise_std=p['gradient_std'],seed=base.seed_for(key,j['seed'],t,cid,'gaussian')))
            local.append(dict(client=cid,batch_hash=base.ids_hash(idx),clip_fraction=float((norms>2).float().mean()),
                raw_risk_research_only=None if raw is None else float(raw),honest_generated_risk=None if rr is None else float(rr)))
        sent,r,ad=inject(torch.stack(messages),None if not reports else torch.stack(reports),attack=j['attack'],round_number=t+1)
        sent,r,safety=sanitize(sent,r)
        if safety['invalid_message_rows'] or safety['nonfinite_risk_reports']:
            raise FloatingPointError('Nonfinite message/attack in fixed finite benchmark')
        u,diag=aggregate(sent,r,kind=j['method'],round_number=t+1,horizon=120)
        diag['message_safety']=safety
        if not bool(torch.isfinite(u).all()):raise FloatingPointError('Nonfinite aggregate')
        # Oracle-only: computed after aggregation and never used in the defense.
        target=torch.stack(clean)[2:].mean(0);a=u/diag['eta']
        weights=diag['objective_weights'] if diag['solver'] is None else diag['solver']['stationary_weights']
        effective=torch.tensor(weights,device='mps',dtype=torch.float32)
        byzpart=(effective[:2,None]*sent[:2]).sum(0)
        oa=dict(aggregate_squared_error_to_unnoised_clipped_honest_mean=float(stable_norm(a-target).square()),
            honest_target_norm=float(stable_norm(target)),aggregate_norm=float(stable_norm(a)),
            designated_objective_mass=sum(diag['objective_weights'][:2]),designated_effective_mass=sum(weights[:2]),
            active_byzantine_contribution_norm=float(stable_norm(byzpart)) if ad['active'] else 0.,
            reconstruction_error=float(stable_norm(a-(effective[:,None]*sent).sum(0))),
            target='Unweighted mean of eight clipped, non-noised honest batch gradients; not an exact population gradient')
        base.apply_gradient(model,u,1.)
        v=base.evaluate(model,data,'val') if t+1 in m['validation_evaluation_rounds'] else None
        test=base.evaluate(model,data,'test') if t+1 in m['test_evaluation_rounds'] else None
        rows.append(dict(round=t+1,device='mps',epsilon_realized=parent.ledger.epsilon_at(p,t+1,profile,j['method']),
            aggregation=diag,attack=ad,validation=v,validation_honest=None if v is None else honest_metrics(v),
            test=test,test_honest=None if test is None else honest_metrics(test)))
        oracle.append(dict(round=t+1,clients=local,aggregate=oa))
        base.checkpoint(cp,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},rows=rows,oracle=oracle,
            round=t+1,initial=initial,job=j,source_stamp=stamp,key_hash=key_hash))
        if v is not None:
            h=rows[-1]['validation_honest']
            print(f"{identifier(j)} {t+1}/120 honest_acc={h['accuracy_pct']:.3f} W20={h['worst20_pct']:.3f}",flush=True)
    base.save(d/'metrics.json',dict(job=j,device='mps',source_stamp=stamp,privacy=p,initial=initial,rounds=rows,final=rows[-1],
        splits=data['splits'],test_evaluation_rounds=[90,120],evaluation_and_oracles_not_private=True))
    base.save(d/'simulator_oracle.json',dict(privacy_protected=False,feeds_mechanism=False,rounds=oracle))
    base.save(d/'orchestration_status.json',dict(status='completed',device='mps',job=j,round=120,
        metrics_sha256=base.digest(d/'metrics.json'),oracle_sha256=base.digest(d/'simulator_oracle.json')))


def report(m,profile,stamp):
    rows=[json.loads((OUT/identifier(j)/'metrics.json').read_text()) for j in jobs(m) if completed(j,m,profile,stamp)]
    lines=['# V13 — contrôle propre, attaques et récupération','',f'**{len(rows)}/64 runs MPS vérifiés.**', '',
        'Métriques ci-dessous : huit identités honnêtes fixées 2–9. Pas de sélection de checkpoint.', '',
        '| Seed | Attaque | Règle | Tour | Test acc. (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) |',
        '|--:|:--|:--|--:|--:|--:|--:|--:|']
    for r in rows:
        j=r['job'];t=m['primary_endpoints'].get(j['attack'],120);v=at(r,t)
        lines.append(f"| {j['seed']} | {j['attack']} | {j['method']} | {t} | {v['accuracy_pct']:.3f} | {v['worst20_pct']:.3f} | {v['gap_best20_worst20_pp']:.3f} | {v['variance_pp2']:.3f} |")
    decision=None
    if len(rows)==64:
        decision=decide(m,rows)
        base.save(OUT/'evidence.json',dict(decision=decision,source_stamp=stamp,
            metric_hashes={identifier(r['job']):base.digest(OUT/identifier(r['job'])/'metrics.json') for r in rows}))
        lines+=['',f"Gate attaques : **{'PASS' if decision['attack_confirmation_passed'] else 'FAIL'}**."]
        for attack,d in decision['attacks'].items():
            lines+=['',f"- {attack} : {'PASS' if d['passed'] else 'FAIL'}, borne de solveur maximale {d['solver_max_gap']:.8g}."]
    lines+=['','[Protocole figé](Private_Risk_Attack_Confirmation_V13_Protocol.md).']
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
            testpath=OUT/'tests.json'
            if not testpath.exists():
                p=subprocess.run([sys.executable,'-m','pytest',*TESTS,'-q'],cwd=ROOT,capture_output=True,text=True)
                base.save(testpath,dict(passed=p.returncode==0,output=p.stdout+p.stderr,source_stamp=stamp))
            assert json.loads(testpath.read_text())['passed']
            key=json.loads((parent.OUT/'simulator_secret.json').read_text())['key']
            for seed in m['confirmation_seeds']:
                js=[j for j in jobs(m) if j['seed']==seed]
                if all(completed(j,m,profile,stamp) for j in js):continue
                data=base.prepare(profile,seed)
                for j in js:train(m,profile,j,data,key,stamp);report(m,profile,stamp)
                del data;torch.mps.empty_cache()
            count,decision=report(m,profile,stamp);assert count==64
            for j in jobs(m):
                o=json.loads((OUT/identifier(j)/'simulator_oracle.json').read_text())
                p=json.loads((parent.OUT/parent.identifier(dict(seed=j['seed'],method=j['method']))/'simulator_oracle.json').read_text())
                assert [[c['batch_hash'] for c in t['clients']] for t in o['rounds']]==[[c['batch_hash'] for c in t['clients']] for t in p['rounds']]
            base.save(OUT/'pairing_audit.json',dict(passed=True,clean_replay_matches_v12=True,preattack_prefix_matches_v12=True,
                all_batches_paired=True,private_gaussians_paired_by_frozen_seed_domains=True))
            base.save(OUT/'status.json',dict(status='completed',device='mps',valid_runs=count,
                attack_confirmation_passed=decision['attack_confirmation_passed'],next_campaign_launched=False))
        except Exception as exc:
            base.save(OUT/'status.json',dict(status='failed',device='mps',error=repr(exc),pid=os.getpid()));raise


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for flag in ('launch','worker','status'):g.add_argument('--'+flag,action='store_true')
    p.add_argument('--resume',action='store_true');a=p.parse_args();m=config()
    if a.status:print((OUT/'status.json').read_text() if (OUT/'status.json').exists() else 'not started');return
    if not a.resume:p.error('--resume required')
    if a.worker:worker(m);return
    require_mps();inputs(m)
    with lock():pass
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open('a') as f:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker','--resume'],cwd=ROOT,
            env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=proc.pid,device='mps',runs=64,log=str(LOG))))


if __name__=='__main__':main()
