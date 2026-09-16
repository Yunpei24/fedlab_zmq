#!/usr/bin/env python3
"""Post-hoc population decomposition at every preselected V20 state; MPS only."""
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
from scripts import run_fair_objective_screen as base
from privacy.fair_objective import require_mps,per_example
from privacy.complete_recursive_query import queries
from privacy.capped_private_risk import weights,potential
from privacy.scheduled_private_risk import aggregate

OUT=ROOT/'results/ldp_gradient_far/recursive_population_diagnostic_v20'
PROTOCOL=ROOT/'output/analysis/Recursive_Complete_Query_V20_Population_Diagnostic_Protocol.md'
REPORT=ROOT/'output/analysis/Recursive_Complete_Query_V20_Population_Analyse.md'
TEST=ROOT/'tests/test_recursive_population_diagnostic_v20.py'


def sq(x):return float(x.square().sum())


def inputs():
    a=ROOT/'output/analysis/Recursive_Complete_Query_V20_Independent_Audit.json'
    assert json.loads(a.read_text())['audit_passed']
    m=json.loads((parent.OUT/'manifest.json').read_text());stamp=dict(m['source_stamp']);base.verify_stamp(stamp)
    paths=[Path(__file__),PROTOCOL,TEST,a]
    paths += [parent.OUT/f'seed{s}__round{k}{suffix}' for s in parent.SEEDS for k in parent.ROUNDS for suffix in ('.json','_vectors.pt')]
    stamp.update({str(p.relative_to(ROOT)):base.digest(p) for p in paths})
    return m['profile'],stamp


def state(seed,k,profile,stamp,key,data):
    path=OUT/f'seed{seed}__round{k}.json'
    if path.exists():
        r=json.loads(path.read_text());assert r['source_stamp']==stamp and r['vector_sha256']==base.digest(path.with_suffix('.pt'));return
    record=json.loads((parent.OUT/f'seed{seed}__round{k}.json').read_text())
    saved=torch.load(parent.OUT/f'seed{seed}__round{k}_vectors.pt',map_location='cpu',weights_only=True)
    model=base.new_model(profile,seed);oldmodel=base.new_model(profile,seed)
    model.load_state_dict(saved['before']);oldmodel.load_state_dict(saved['previous_model'])
    th=1-math.sqrt(.5+1e-8);a=1-th;D=2*th
    memory=saved['private_memory'].to('mps');sent={'difference':saved['sent'].to('mps'),'complete':saved['complete'].to('mps')}
    pop=[];client_records=[];vector_clients=[]
    for cid,ids in enumerate(data['train']):
        base.save(OUT/'status.json',dict(status='running',device='mps',seed=seed,round=k,client=cid,pid=os.getpid(),updated_unix=time.time()))
        totals={name:torch.zeros_like(memory[cid]) for name in ('raw','clipped','previous','difference','complete')}
        risk=torch.zeros((),device='mps')
        for block in ids.split(240):
            rr,g,_,raw=per_example(model,data['x'][block],data['y'][block],clip_norm=2.)
            gp=per_example(oldmodel,data['x'][block],data['y'][block],clip_norm=2.)[1]
            _,qd,qp,_=queries(g,gp,C=2.,D=D,theta=th)
            fraction=len(block)/len(ids);risk+=rr.sum()/len(ids)
            for name,value in [('raw',raw),('clipped',g.mean(0)),('previous',gp.mean(0)),('difference',qd.mean(0)),('complete',qp.mean(0))]:
                totals[name]+=fraction*value
        ix=base.draw_indices(4800,240,base.seed_for(key,seed,k-1,cid,'batch'))
        assert base.ids_hash(ix)==record['client_diagnostics'][cid]['batch_hash']
        g=per_example(model,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)[1]
        gp=per_example(oldmodel,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)[1]
        _,qd,qp,_=queries(g,gp,C=2.,D=D,theta=th)
        old_error=a*(memory[cid]-totals['previous']);diag={};vectors={}
        for name,q in [('difference',qd),('complete',qp)]:
            bias=totals[name]-(totals['clipped']-a*totals['previous'])
            sampling=q.mean(0)-totals[name]
            noise=sent[name][cid]-a*memory[cid]-q.mean(0)
            error=sent[name][cid]-totals['clipped']
            torch.testing.assert_close(error,old_error+bias+sampling+noise,rtol=2e-5,atol=2e-7)
            diag[name]=dict(past_error_sq=sq(old_error),population_query_bias_sq=sq(bias),sampling_error_sq=sq(sampling),
                gaussian_realization_sq=sq(noise),message_error_sq=sq(error),
                past_bias_cross=float(2*torch.dot(old_error,bias)),past_sampling_cross=float(2*torch.dot(old_error,sampling)),
                bias_sampling_cross=float(2*torch.dot(bias,sampling)),message_error_without_fresh_noise_sq=sq(error-noise))
            vectors[name]=dict(bias=bias.detach().cpu(),sampling=sampling.detach().cpu(),noise=noise.detach().cpu(),error=error.detach().cpu())
        torch.testing.assert_close(vectors['difference']['noise'].to('mps'),vectors['complete']['noise'].to('mps'),rtol=2e-4,atol=2e-7)
        pop.append(dict(**totals,risk=risk));client_records.append(dict(client=cid,methods=diag))
        vector_clients.append(dict(past_error=old_error.detach().cpu(),population={name:v.detach().cpu() for name,v in totals.items()},methods=vectors))
        print(f'population {seed} t{k}: {cid+1}/10 clients',flush=True)
    risks_=torch.stack([p['risk'] for p in pop]);raw=torch.stack([p['raw'] for p in pop]);clipped=torch.stack([p['clipped'] for p in pop])
    lam,coeff=weights(risks_,.5);GJ=(coeff[:,None]*raw).mean(0)
    target=(lam[:,None]*raw).sum(0);clipped_target=(lam[:,None]*clipped).sum(0)
    torch.testing.assert_close(GJ/coeff.mean(),target,rtol=2e-5,atol=2e-7)
    J0=float(potential(risks_,.5).mean());assert math.isclose(J0,record['J0'],rel_tol=2e-5,abs_tol=2e-6)
    measures=[];aggregates={}
    for name,row in zip(('difference','complete'),record['rows']):
        step,diag=aggregate(sent[name],saved['reports'].to('mps'),kind='risk_rfa',round_number=k,horizon=120)
        eta=diag['eta'];A=step/eta;align=float(torch.dot(GJ,A));decrease=row['J_gain']
        measures.append(dict(method=name,error_to_unclipped_fair_target_sq=sq(A-target),error_to_clipped_fair_target_sq=sq(A-clipped_target),
            true_J_gradient_dot_aggregate=align,step_norm=math.sqrt(sq(step)),eta=eta,
            predicted_first_order_decrease=eta*align,actual_J_decrease=decrease,
            exact_first_order_residual=eta*align-decrease,validation=row['validation']))
        aggregates[name]=A.detach().cpu()
    vec=path.with_suffix('.pt');base.checkpoint(vec,dict(clients=vector_clients,aggregates=aggregates,
        true_J_gradient=GJ.detach().cpu(),unclipped_target=target.detach().cpu(),clipped_target=clipped_target.detach().cpu(),
        raw_risks=risks_.detach().cpu(),source_stamp=stamp,privacy_protected=False))
    base.save(path,dict(seed=seed,round=k,source_stamp=stamp,device='mps',clients=client_records,aggregates=measures,
        actual_objective='mean_train_client Phi(half_Brier_risk)',population_examples_per_client=4800,
        oracle_only=True,feeds_mechanism=False,test_evaluated=False,vector_sha256=base.digest(vec)))
    del model,oldmodel;torch.mps.empty_cache()


def report(stamp):
    rows=[json.loads((OUT/f'seed{s}__round{k}.json').read_text()) for s in parent.SEEDS for k in parent.ROUNDS]
    lines=['# V20 — diagnostic sur les vrais gradients de population','',
        'Dix états, chacun recalculé sur4800 exemples par client. Diagnostic post-hoc sur MPS ; aucune modification des critères ni de l’algorithme.', '',
        '| Seed | Tour | Placement | Erreur² vers cible équitable non clippée | Alignement ∇J·A | Longueur du pas | Baisse J premier ordre | Baisse J réalisée | Résidu premier ordre |',
        '|--:|--:|:--|--:|--:|--:|--:|--:|--:|']
    for r in rows:
        for d in r['aggregates']:
            lines.append(f"| {r['seed']} | {r['round']} | {d['method']} | {d['error_to_unclipped_fair_target_sq']:.6g} | {d['true_J_gradient_dot_aggregate']:+.6g} | {d['step_norm']:.6g} | {d['predicted_first_order_decrease']:+.6g} | {d['actual_J_decrease']:+.6g} | {d['exact_first_order_residual']:+.6g} |")
    lines+=['','## Décomposition des erreurs des messages, moyennes sur les clients','',
        '| Seed | Tour | Placement | Erreur passée² | Biais population² | Sampling² | 2⟨passé,biais⟩ | Erreur sans bruit frais² |',
        '|--:|--:|:--|--:|--:|--:|--:|--:|']
    for r in rows:
        for name in ('difference','complete'):
            ds=[c['methods'][name] for c in r['clients']]
            cells=[f'{st.mean(d[key] for d in ds):+.6g}' for key in ('past_error_sq','population_query_bias_sq','sampling_error_sq','past_bias_cross','message_error_without_fresh_noise_sq')]
            lines.append('| '+' | '.join([str(r['seed']),str(r['round']),name,*cells])+' |')
    lines+=['','Une erreur de requête plus faible ne supprime pas les autres termes, notamment la mémoire. Les erreurs quadratiques sont des réalisations à un état, pas des espérances estimées sur de multiples bruits. Le résidu premier ordre ne constitue pas une borne de lissité. Les oracles ne sont pas privés et n’entrent jamais dans le mécanisme.', '',
        '[Définitions et protocole prospectif du calcul](Recursive_Complete_Query_V20_Population_Diagnostic_Protocol.md).']
    REPORT.write_text('\n'.join(lines)+'\n')
    base.save(OUT/'status.json',dict(status='completed',device='mps',states=10,global_validation=False,source_stamp=stamp))


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
                base.save(OUT/'tests.json',dict(passed=p.returncode==0,output=p.stdout+p.stderr,source_stamp=stamp))
            tests=json.loads((OUT/'tests.json').read_text());assert tests['passed'] and tests['source_stamp']==stamp
            key=json.loads((parent.prior.OUT/'simulator_secret.json').read_text())['key']
            for seed in parent.SEEDS:
                data=base.prepare(profile,seed)
                for k in parent.ROUNDS:state(seed,k,profile,stamp,key,data)
                del data;torch.mps.empty_cache()
            base.verify_stamp(stamp);report(stamp)
        except Exception as exc:
            base.save(OUT/'status.json',dict(status='failed',error=repr(exc),device='mps'));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true',required=True);p.parse_args();main()
