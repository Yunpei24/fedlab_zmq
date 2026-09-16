#!/usr/bin/env python3
"""Six prespecified, noncumulative steps at each of ten V20 host states."""
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
from scripts import run_complete_recursive_query_diagnostic_v20 as parent
from scripts import diagnose_recursive_population_v20 as population
from scripts import run_fair_objective_screen as base
from scripts.run_private_risk_channel_replay_v14 import risks
from scripts.analyze_private_risk_confirmation_v12 import verify
from privacy.fair_objective import require_mps,per_example,release
from privacy.capped_private_risk import potential
from privacy.scheduled_private_risk import aggregate
from privacy.aggregate_radial_step_v21 import controlled_step

OUT=ROOT/'results/ldp_gradient_far/prisma_global_step_diagnostic_v21'
PROTOCOL=ROOT/'output/analysis/PriSMA_Recursive_V21_Theory_and_Protocol.md'
REPORT=ROOT/'output/analysis/PriSMA_Global_Step_V21_Analyse.md'
TEST=ROOT/'tests/test_aggregate_radial_step_v21.py'
SEEDS=[170501,170502];ROUNDS=[2,10,30,60,120]
BRANCHES=[f'{q}__{c}' for q in ('recursive','fresh') for c in ('unchanged','half','global_clip')]


def inputs():
    audit=ROOT/'output/analysis/Recursive_Complete_Query_V20_Population_Independent_Audit.json'
    assert json.loads(audit.read_text())['audit_passed']
    m=json.loads((population.OUT/'manifest.json').read_text());stamp=dict(m['source_stamp']);base.verify_stamp(stamp)
    files=[Path(__file__),PROTOCOL,TEST,ROOT/'privacy/aggregate_radial_step_v21.py',audit]
    files += [parent.prior.OUT/f'seed{s}__fresh__risk_rfa/metrics.json' for s in SEEDS]
    files += [population.OUT/f'seed{s}__round{k}{ext}' for s in SEEDS for k in ROUNDS for ext in ('.json','.pt')]
    stamp.update({str(p.relative_to(ROOT)):base.digest(p) for p in files})
    return m['profile'],stamp


def decide(records):
    evidence=[]
    for seed in SEEDS:
        rr=[r for r in records if r['seed']==seed];assert sorted(r['round'] for r in rr)==ROUNDS
        contrasts=[]
        for control in BRANCHES:
            if control=='recursive__global_clip':continue
            ds=[]
            for r in rr:
                ix={x['branch']:x for x in r['rows']};a=ix['recursive__global_clip'];b=ix[control]
                ds.append(dict(round=r['round'],accuracy=a['validation']['accuracy_pct']-b['validation']['accuracy_pct'],
                    worst20=a['validation']['worst20_pct']-b['validation']['worst20_pct'],J=a['J_gain']-b['J_gain']))
            avg={key:st.mean(d[key] for d in ds) for key in ('accuracy','worst20','J')}
            tests=dict(mean_accuracy_noninferior=avg['accuracy']>=-.1,mean_worst20_noninferior=avg['worst20']>=0.,mean_J_noninferior=avg['J']>=0.)
            if control=='recursive__unchanged':
                tests.update(worst20_improved_quarter_pp=avg['worst20']>=.25,J_strictly_improved=avg['J']>0.,
                    no_state_worst20_loss_over_half_pp=min(d['worst20'] for d in ds)>=-.5)
            contrasts.append(dict(control=control,mean_differences=avg,per_state=ds,tests=tests,passed=all(tests.values())))
        evidence.append(dict(seed=seed,contrasts=contrasts,passed=all(c['passed'] for c in contrasts)))
    return dict(seeds=evidence,admitted_to_end_to_end_screen=all(e['passed'] for e in evidence),global_validation=False)


def probe(seed,k,profile,stamp,key,data):
    path=OUT/f'seed{seed}__round{k}.json'
    if path.exists():
        r=json.loads(path.read_text());assert r['source_stamp']==stamp and r['vector_sha256']==base.digest(path.with_suffix('.pt'));return
    old=json.loads((parent.OUT/f'seed{seed}__round{k}.json').read_text())
    saved=torch.load(parent.OUT/f'seed{seed}__round{k}_vectors.pt',map_location='cpu',weights_only=True)
    pv=torch.load(population.OUT/f'seed{seed}__round{k}.pt',map_location='cpu',weights_only=True)
    source=parent.prior.OUT/f'seed{seed}__fresh__risk_rfa/metrics.json'
    ledger=json.loads(source.read_text())['privacy'];sigma=ledger['gradient_std']
    assert ledger['epsilon_realized']<=4 and ledger['gradient_sensitivity']==4/240
    model=base.new_model(profile,seed);model.load_state_dict(saved['before'])
    v0=base.evaluate(model,data,'val');verify(v0);assert v0==old['initial']
    J0=float(potential(risks(model,data),.5).mean());assert math.isclose(J0,old['J0'],abs_tol=2e-6)
    fresh=[];batches=[]
    for cid,ids in enumerate(data['train']):
        ix=base.draw_indices(4800,240,base.seed_for(key,seed,k-1,cid,'batch'))
        digest=base.ids_hash(ix);assert digest==old['client_diagnostics'][cid]['batch_hash']
        g=per_example(model,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)[1]
        message=release(g.mean(0),noise_std=sigma,seed=base.seed_for(key,seed,k-1,cid,'gaussian'))
        fresh.append(message);batches.append(digest)
        noise_recursive=pv['clients'][cid]['methods']['difference']['noise'].to('mps')
        ratio=old['client_diagnostics'][cid]['noise_std']/sigma
        torch.testing.assert_close(message-g.mean(0),noise_recursive/ratio,rtol=2e-4,atol=4e-7)
    sent={'recursive':saved['sent'].to('mps'),'fresh':torch.stack(fresh)}
    reports=saved['reports'].to('mps');G=pv['true_J_gradient'].to('mps');target=pv['unclipped_target'].to('mps')
    rows=[];vectors={}
    for mode in ('recursive','fresh'):
        original,agg=aggregate(sent[mode],reports,kind='risk_rfa',round_number=k,horizon=120)
        A=original/agg['eta']
        for control in ('unchanged','half','global_clip'):
            branch=f'{mode}__{control}'
            base.save(OUT/'status.json',dict(status='running',device='mps',seed=seed,round=k,branch=branch,pid=os.getpid(),updated_unix=time.time()))
            step,diag=controlled_step(A,agg['eta'],control,radius=1.)
            model.load_state_dict(saved['before']);base.apply_gradient(model,step,1.)
            v=base.evaluate(model,data,'val');verify(v)
            J=float(potential(risks(model,data),.5).mean());gain=J0-J;first=float(torch.dot(G,step))
            if branch=='recursive__unchanged':
                for metric in ('accuracy_pct','worst20_pct','ce_loss','variance_pp2'):
                    assert math.isclose(v[metric],old['rows'][0]['validation'][metric],abs_tol=2e-5)
                assert math.isclose(gain,old['rows'][0]['J_gain'],abs_tol=2e-6)
            effective=step/agg['eta']
            rows.append(dict(branch=branch,control=diag,validation=v,J=J,J_gain=gain,first_order_decrease=first,
                first_order_residual=first-gain,effective_aggregate_error_sq=float((effective-target).square().sum()),aggregation=agg))
            vectors[branch]=step.detach().cpu()
            print(f'{seed} t{k} {branch}: acc={v["accuracy_pct"]:.4f} W20={v["worst20_pct"]:.4f} Jgain={gain:+.7f}',flush=True)
    model.load_state_dict(saved['before'])
    base.checkpoint(path.with_suffix('.pt'),dict(steps=vectors,fresh_messages=sent['fresh'].detach().cpu(),source_stamp=stamp,privacy_protected=False))
    base.save(path,dict(seed=seed,round=k,source_stamp=stamp,device='mps',initial=v0,J0=J0,rows=rows,
        batch_hashes=batches,ledger=ledger,vector_sha256=base.digest(path.with_suffix('.pt')),oracle_only=True,
        test_evaluated=False,noncumulative=True,host='V19 recursive risk-RFA'))
    del model;torch.mps.empty_cache()


def finish(stamp):
    records=[json.loads((OUT/f'seed{s}__round{k}.json').read_text()) for s in SEEDS for k in ROUNDS]
    assert all(r['source_stamp']==stamp and len(r['rows'])==6 for r in records)
    decision=decide(records)
    sigma=records[0]['ledger']['gradient_std']
    sample=torch.load(OUT/f'seed{SEEDS[0]}__round{ROUNDS[0]}.pt',map_location='cpu',weights_only=True)
    d=sample['fresh_messages'].shape[1];assert d==61706
    theta=1-math.sqrt(.5+1e-8);ratio=theta+(1-theta)*theta
    lower=16*d*(sigma*ratio)**2/(10*2*1)
    theory=dict(source='PriSMA lemma D.4, Eq67',dimension=d,C1=2.,C2=1.,M=10,base_noise_std=sigma,
        recursive_noise_std=sigma*ratio,theta=theta,required_theta_at_least=lower,condition_passes=theta>=lower,
        sufficient_condition_only=True,not_a_DP_failure=True,not_a_nonconvergence_proof=True)
    base.save(OUT/'evidence.json',dict(source_stamp=stamp,decision=decision,theory=theory))
    lines=['# V21 — clipping de l’agrégat et coût du pas fini','',
        'Diagnostic non cumulatif sur deux seeds déjà connues, dix états et 60 branches ; MPS, aucun test final évalué.', '',
        f'Condition suffisante PriSMA (67) : γ devrait être au moins {lower:.6f}, contre {theta:.6f} utilisé. Cela interdit de transférer cette preuve, pas de déclarer la privacy invalide.', '',
        '| Seed | Tour | Message / pas | Accuracy | Worst-20 | Gap Best20–Worst20 | Variance pp² | Gain J | Facteur du pas |',
        '|--:|--:|:--|--:|--:|--:|--:|--:|--:|']
    for r in records:
        for x in r['rows']:
            v=x['validation'];lines.append(f'| {r["seed"]} | {r["round"]} | {x["branch"]} | {v["accuracy_pct"]:.4f} | {v["worst20_pct"]:.4f} | {v["gap_best20_worst20_pp"]:.4f} | {v["variance_pp2"]:.4f} | {x["J_gain"]:+.7f} | {x["control"]["factor"]:.6f} |')
    lines+=['','## Contrastes moyens par seed : récursif + clipping global moins témoin','',
        '| Seed | Témoin | Δ accuracy pp | Δ Worst-20 pp | Δ gain J | Critères |','|--:|:--|--:|--:|--:|:--|']
    for s in decision['seeds']:
        for c in s['contrasts']:
            a=c['mean_differences'];lines.append(f'| {s["seed"]} | {c["control"]} | {a["accuracy"]:+.4f} | {a["worst20"]:+.4f} | {a["J"]:+.7f} | {"PASS" if c["passed"] else "FAIL"} |')
    lines+=['',f'Admission au nouvel écran : **{"PASS" if decision["admitted_to_end_to_end_screen"] else "FAIL"}**. Audit indépendant requis avant tout usage de cette admission.', '',
        'Les gradients frais sont évalués aux états hôtes récursifs, pas à leur propre trajectoire. Les erreurs sont des réalisations, pas des MSE moyennées sur de nouveaux bruits. Un gain d’un pas ne démontre pas un gain cumulé, une robustesse aux attaques ni une validation indépendante. Les oracles et les branches appariées ne forment pas un transcript conjoint privé.', '',
        '[Protocole et lecture théorique](PriSMA_Recursive_V21_Theory_and_Protocol.md).']
    REPORT.write_text('\n'.join(lines)+'\n')
    base.save(OUT/'status.json',dict(status='completed',device='mps',states=10,branches=60,decision=decision))
    print(json.dumps(decision),flush=True)


def main():
    require_mps();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            profile,stamp=inputs();manifest=dict(profile=profile,source_stamp=stamp)
            if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
            else:base.save(OUT/'manifest.json',manifest)
            if not (OUT/'tests.json').exists():
                p=subprocess.run([sys.executable,'-m','pytest',str(TEST),'-q'],cwd=ROOT,capture_output=True,text=True)
                base.save(OUT/'tests.json',dict(passed=p.returncode==0,source_stamp=stamp,output=p.stdout+p.stderr))
            t=json.loads((OUT/'tests.json').read_text());assert t['passed'] and t['source_stamp']==stamp
            key=json.loads((parent.prior.OUT/'simulator_secret.json').read_text())['key']
            for seed in SEEDS:
                data=base.prepare(profile,seed)
                for k in ROUNDS:probe(seed,k,profile,stamp,key,data)
                del data;torch.mps.empty_cache()
            base.verify_stamp(stamp);finish(stamp)
        except Exception as exc:
            base.save(OUT/'status.json',dict(status='failed',device='mps',error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true',required=True);p.parse_args();main()
