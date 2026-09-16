#!/usr/bin/env python3
"""Replay V19 hosts; alter only per-example query clipping at ten fixed states."""
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
from scripts import run_recursive_private_risk_calibration_v19 as prior
from scripts import run_fair_objective_screen as base
from scripts.run_private_risk_channel_replay_v14 import risks
from scripts.analyze_private_risk_confirmation_v12 import verify
from privacy.fair_objective import require_mps,per_example,release
from privacy.split_risk_gradient import private_risk
from privacy.clipped_recursive_private_gradient import recursive_release
from privacy.complete_recursive_query import queries
from privacy.capped_private_risk import potential
from privacy.scheduled_private_risk import aggregate
from privacy.private_risk_message_safety import sanitize

NAME='complete_recursive_query_diagnostic_v20'
OUT=ROOT/'results/ldp_gradient_far'/NAME
PROTOCOL=ROOT/'output/analysis/Recursive_Complete_Query_V20_Diagnostic_Protocol.md'
REPORT=ROOT/'output/analysis/Recursive_Complete_Query_V20_Analyse.md'
SEEDS=[170501,170502];ROUNDS=[2,10,30,60,120]
TEST=ROOT/'tests/test_complete_recursive_query_v20.py'


def inputs():
    audit=ROOT/'output/analysis/Recursive_Private_Risk_V19_Analyse.json'
    a=json.loads(audit.read_text());assert a['audit_passed'] and not a['eligible_for_independent_confirmation']
    assert json.loads((prior.OUT/'status.json').read_text())['status']=='completed'
    m=json.loads((prior.OUT/'manifest.json').read_text());base.verify_stamp(m['source_stamp'])
    stamp=dict(m['source_stamp'])
    paths=[Path(__file__),PROTOCOL,TEST,ROOT/'privacy/complete_recursive_query.py',audit,
           ROOT/'scripts/run_private_risk_channel_replay_v14.py',ROOT/'scripts/analyze_private_risk_confirmation_v12.py']
    for seed in SEEDS:
        folder=prior.OUT/f'seed{seed}__recursive__risk_rfa'
        paths += [folder/'metrics.json',folder/'simulator_oracle.json',folder/'checkpoint.pt']
    stamp.update({str(p.relative_to(ROOT)):base.digest(p) for p in paths})
    return m['profile'],stamp


def close(a,b):assert math.isclose(a,b,abs_tol=2e-6,rel_tol=2e-5),(a,b)


def evaluate_probe(seed,k,model,data,before,previous_model,private_memory,sent,alternative,reports,diags,stamp):
    initial=base.evaluate(model,data,'val');J0=float(potential(risks(model,data),.5).mean())
    rows=[]
    for name,messages in [('difference',sent),('complete',alternative)]:
        model.load_state_dict(before)
        step,aggregation=aggregate(messages,reports,kind='risk_rfa',round_number=k,horizon=120)
        base.apply_gradient(model,step,1.)
        v=base.evaluate(model,data,'val');verify(v)
        J=float(potential(risks(model,data),.5).mean())
        rows.append(dict(method=name,validation=v,J=J,J_gain=J0-J,aggregation=aggregation))
    model.load_state_dict(before)
    d={metric:rows[1]['validation'][metric]-rows[0]['validation'][metric] for metric in
       ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2','ce_loss','brier_loss','balanced_accuracy_pct')}
    d['J_gain']=rows[1]['J_gain']-rows[0]['J_gain']
    record=dict(seed=seed,round=k,source_stamp=stamp,device='mps',initial=initial,J0=J0,rows=rows,delta=d,
        client_diagnostics=diags,test_evaluated=False,oracle_only=True,noncumulative=True)
    path=OUT/f'seed{seed}__round{k}.json';base.save(path,record)
    base.checkpoint(OUT/f'seed{seed}__round{k}_vectors.pt',dict(before=before,sent=sent.detach().cpu(),
        complete=alternative.detach().cpu(),reports=reports.detach().cpu(),source_stamp=stamp,
        previous_model={name:v.detach().cpu().clone() for name,v in previous_model.state_dict().items()},
        private_memory=private_memory.detach().cpu()))
    print(f'probe {seed} t{k}: delta acc={d["accuracy_pct"]:+.4f}, W20={d["worst20_pct"]:+.4f}',flush=True)


def replay(seed,profile,stamp,key):
    done=OUT/f'seed{seed}__replay_audit.json'
    if done.exists():
        a=json.loads(done.read_text());assert a['passed'] and a['source_stamp']==stamp;return
    source=prior.OUT/f'seed{seed}__recursive__risk_rfa'
    expected=json.loads((source/'metrics.json').read_text());raw=json.loads((source/'simulator_oracle.json').read_text())['rounds']
    data=base.prepare(profile,seed);model=base.new_model(profile,seed);previous=base.new_model(profile,seed)
    p=expected['privacy'];pub=expected['recursive_parameters'];memory=None;start=0
    cp=OUT/f'seed{seed}__replay_checkpoint.pt'
    if cp.exists():
        saved=torch.load(cp,map_location='cpu',weights_only=True);assert saved['source_stamp']==stamp
        assert saved['key_sha']==base.digest(prior.OUT/'simulator_secret.json')
        model.load_state_dict(saved['model']);previous.load_state_dict(saved['previous_model'])
        memory=saved['memory'].to('mps');start=saved['round']
    else:
        v=base.evaluate(model,data,'val')
        assert v==expected['initial'] and data['splits']==expected['splits']
    for t in range(start,120):
        require_mps();k=t+1
        base.save(OUT/'status.json',dict(status='running',device='mps',seed=seed,round=k,pid=os.getpid(),updated_unix=time.time()))
        sent=[];alternate=[];reports=[];diagnostics=[]
        for cid,ids in enumerate(data['train']):
            rr,rraw=private_risk(model,data['x'][ids],data['y'][ids],noise_std=p['risk_std'],
                seed=base.seed_for(key,seed,t,cid,'risk'),N=4800)
            orig=raw[t]['clients'][cid];close(float(rr),orig['private_risk']);close(float(rraw),orig['raw_risk'])
            reports.append(rr)
            ix=base.draw_indices(4800,240,base.seed_for(key,seed,t,cid,'batch'));assert base.ids_hash(ix)==orig['batch_hash']
            _,g,norms,_=per_example(model,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)
            assert int((norms>2).sum())==orig['gradient_clipped_count']
            oldg=None if t==0 else per_example(previous,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)[1]
            noise_seed=base.seed_for(key,seed,t,cid,'gaussian')
            message,diag=recursive_release(g,oldg,None if t==0 else memory[cid],C=2.,D=pub['D'],theta=pub['theta'],
                base_noise_std=p['gradient_std'],seed=noise_seed)
            assert diag['clipping_increment_count']==orig['recursion']['clipping_increment_count']
            sent.append(message)
            if k in ROUNDS:
                exact,diff,projected,d=queries(g,oldg,C=2.,D=pub['D'],theta=pub['theta'])
                close(d['query_sensitivity'],diag['query_sensitivity']);close(diag['noise_std']/d['query_sensitivity'],p['gradient_z'])
                a=1-pub['theta'];oldprivate=a*memory[cid]+release(diff.mean(0),noise_std=diag['noise_std'],seed=noise_seed)
                torch.testing.assert_close(oldprivate,message,rtol=0,atol=0)
                alt=a*memory[cid]+release(projected.mean(0),noise_std=diag['noise_std'],seed=noise_seed)
                torch.testing.assert_close(alt-message,projected.mean(0)-diff.mean(0),rtol=2e-4,atol=2e-7)
                alternate.append(alt)
                lhs=(exact-projected).square().sum(1)+(projected-diff).square().sum(1)
                rhs=(exact-diff).square().sum(1);violation=float((lhs-rhs).max());assert violation<=2e-5
                bound=max(float(torch.linalg.vector_norm(z,dim=1).max()) for z in (diff,projected))
                assert bound<=d['effective_C']+2e-6
                d.update(client=cid,batch_hash=base.ids_hash(ix),max_row_norm=bound,
                    pythagorean_max_residual=violation,noise_std=diag['noise_std'],
                    difference_batch_bias_sq=float((diff-exact).mean(0).square().sum()),
                    complete_batch_bias_sq=float((projected-exact).mean(0).square().sum()),
                    difference_individual_distortion= float((diff-exact).square().sum(1).mean()),
                    complete_individual_distortion=float((projected-exact).square().sum(1).mean()))
                diagnostics.append(d)
        messages,r,safety=sanitize(torch.stack(sent),torch.stack(reports));assert safety['invalid_message_rows']==safety['nonfinite_risk_reports']==0
        before={name:v.detach().cpu().clone() for name,v in model.state_dict().items()}
        if k in ROUNDS:evaluate_probe(seed,k,model,data,before,previous,memory,messages,torch.stack(alternate),r,diagnostics,stamp)
        step,agg=aggregate(messages,r,kind='risk_rfa',round_number=k,horizon=120)
        previous.load_state_dict(before);memory=messages.detach().clone();base.apply_gradient(model,step,1.)
        if expected['rounds'][t]['validation'] is not None:
            val=base.evaluate(model,data,'val')
            for metric in ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2','ce_loss','brier_loss'):
                close(val[metric],expected['rounds'][t]['validation'][metric])
            print(f'host {seed} {k}/120 reconstructed',flush=True)
        base.checkpoint(cp,dict(model={name:v.detach().cpu() for name,v in model.state_dict().items()},
            previous_model=before,memory=memory.detach().cpu(),round=k,source_stamp=stamp,
            key_sha=base.digest(prior.OUT/'simulator_secret.json')))
    target=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=True)
    for name,value in model.state_dict().items():torch.testing.assert_close(value,target['model'][name].to('mps'),rtol=0,atol=0)
    torch.testing.assert_close(memory,target['memory'].to('mps'),rtol=0,atol=0)
    base.save(done,dict(passed=True,seed=seed,source_stamp=stamp,all_round_batches_risks_clipping_checked=True,
        final_model_and_memory_bitwise_equal=True,all_logged_validation_recomputed=True,device='mps'))
    del data,model,previous,memory;torch.mps.empty_cache()


def decide(records):
    evidence=[]
    for seed in SEEDS:
        a=[r for r in records if r['seed']==seed];assert sorted(r['round'] for r in a)==ROUNDS
        early=[d for r in a if r['round']!=120 for d in r['client_diagnostics']]
        old=st.mean(d['difference_batch_bias_sq'] for d in early);new=st.mean(d['complete_batch_bias_sq'] for d in early)
        tests=dict(batch_bias_reduced_10pct=new<=.9*old,worst20_mean_nonnegative=st.mean(r['delta']['worst20_pct'] for r in a)>=0,
            worst20_no_loss_over_half_pp=min(r['delta']['worst20_pct'] for r in a)>=-.5,
            mean_accuracy_loss_at_most_point_one=st.mean(r['delta']['accuracy_pct'] for r in a)>=-.1,
            mean_J_gain_not_worse=st.mean(r['delta']['J_gain'] for r in a)>=0)
        evidence.append(dict(seed=seed,early_original_batch_bias_sq=old,early_complete_batch_bias_sq=new,
            relative_bias_reduction=None if old==0 else 1-new/old,tests=tests,passed=all(tests.values())))
    return dict(seeds=evidence,admitted_to_new_end_to_end_screen=all(e['passed'] for e in evidence),global_validation=False)


def finish(stamp):
    records=[json.loads((OUT/f'seed{s}__round{k}.json').read_text()) for s in SEEDS for k in ROUNDS]
    assert all(r['source_stamp']==stamp for r in records)
    decision=decide(records);base.save(OUT/'evidence.json',dict(source_stamp=stamp,decision=decision))
    lines=['# V20 — intervention sur le clipping de la requête complète','',
        '**10 états appariés**, deux trajectoires hôtes reproduites exactement ; aucune nouvelle trajectoire candidate entraînée.', '',
        '| Seed | Tour | Δ accuracy (pp) | Δ Worst-20 (pp) | Δ gap (pp) | Δ variance (pp²) | Δ gain J | Biais batch original | Biais batch projection |',
        '|--:|--:|--:|--:|--:|--:|--:|--:|--:|']
    for r in records:
        d=r['delta'];c=r['client_diagnostics']
        lines.append(f"| {r['seed']} | {r['round']} | {d['accuracy_pct']:+.4f} | {d['worst20_pct']:+.4f} | {d['gap_best20_worst20_pp']:+.4f} | {d['variance_pp2']:+.4f} | {d['J_gain']:+.7f} | {st.mean(x['difference_batch_bias_sq'] for x in c):.6g} | {st.mean(x['complete_batch_bias_sq'] for x in c):.6g} |")
    lines+=['',f"Admission à un nouvel écran : **{'PASS' if decision['admitted_to_new_end_to_end_screen'] else 'FAIL'}**.", '',
        'La distorsion individuelle est analytiquement moindre ; le biais de batch et les métriques du modèle ne sont pas garantis par cette propriété. Les interventions sont non cumulatives. Les seeds sont déjà connues. Aucune preuve de robustesse empirique ni confirmation indépendante ne résulte de ce test.', '',
        'Le biais de batch est la moyenne sur les dix clients de la norme au carré du résidu moyen de requête, pas une erreur vers le vrai gradient de population.', '',
        '[Protocole et critères préenregistrés](Recursive_Complete_Query_V20_Diagnostic_Protocol.md).']
    REPORT.write_text('\n'.join(lines)+'\n')
    base.save(OUT/'status.json',dict(status='completed',device='mps',interventions=10,decision=decision))
    print(json.dumps(decision),flush=True)


def worker():
    require_mps();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            profile,stamp=inputs();manifest=dict(profile=profile,source_stamp=stamp,seeds=SEEDS,rounds=ROUNDS)
            if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
            else:base.save(OUT/'manifest.json',manifest)
            if not (OUT/'tests.json').exists():
                p=subprocess.run([sys.executable,'-m','pytest',str(TEST),'-q'],cwd=ROOT,capture_output=True,text=True)
                base.save(OUT/'tests.json',dict(passed=p.returncode==0,output=p.stdout+p.stderr,source_stamp=stamp))
            assert json.loads((OUT/'tests.json').read_text())['passed']
            key=json.loads((prior.OUT/'simulator_secret.json').read_text())['key']
            for seed in SEEDS:replay(seed,profile,stamp,key)
            base.verify_stamp(stamp);finish(stamp)
        except Exception as exc:
            base.save(OUT/'status.json',dict(status='failed',device='mps',error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true',required=True);p.add_argument('--resume',action='store_true',required=True)
    p.parse_args();worker()
