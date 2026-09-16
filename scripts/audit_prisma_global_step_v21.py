#!/usr/bin/env python3
"""Independently reconstruct controls, endpoint evaluations and V21 decisions."""
import json
import math
import os
from pathlib import Path
import statistics as st
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import diagnose_prisma_global_step_v21 as run
from scripts import run_fair_objective_screen as base
from scripts.run_private_risk_channel_replay_v14 import risks
from scripts.analyze_private_risk_confirmation_v12 import verify
from privacy.fair_objective import require_mps,per_example,release
from privacy.stable_weighted_rfa import weighted_rfa


def close(a,b):assert math.isclose(a,b,rel_tol=3e-5,abs_tol=2e-6),(a,b)
def J(model,data):
    r=risks(model,data)
    return float(torch.where(r<=.5,r+2*r*r,3*r-.5).mean())


def main():
    require_mps();out=run.OUT;assert json.loads((out/'status.json').read_text())['status']=='completed'
    manifest=json.loads((out/'manifest.json').read_text());stamp=manifest['source_stamp'];base.verify_stamp(stamp)
    assert json.loads((out/'tests.json').read_text())['passed']
    key=json.loads((run.parent.prior.OUT/'simulator_secret.json').read_text())['key']
    records=[];count=0
    for seed in (170501,170502):
        data=base.prepare(manifest['profile'],seed);model=base.new_model(manifest['profile'],seed)
        for k in (2,10,30,60,120):
            path=out/f'seed{seed}__round{k}.json';r=json.loads(path.read_text());assert r['source_stamp']==stamp
            assert r['device']=='mps' and not r['test_evaluated'] and r['noncumulative'] and r['oracle_only']
            assert r['vector_sha256']==base.digest(path.with_suffix('.pt'))
            saved=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=True)
            host=torch.load(run.parent.OUT/f'seed{seed}__round{k}_vectors.pt',map_location='cpu',weights_only=True)
            pop=torch.load(run.population.OUT/f'seed{seed}__round{k}.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(host['before']);close(J(model,data),r['J0'])
            sigma=r['ledger']['gradient_std'];fresh=[]
            for cid,ids in enumerate(data['train']):
                ix=base.draw_indices(4800,240,base.seed_for(key,seed,k-1,cid,'batch'))
                assert base.ids_hash(ix)==r['batch_hashes'][cid]
                g=per_example(model,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)[1]
                fresh.append(release(g.mean(0),noise_std=sigma,seed=base.seed_for(key,seed,k-1,cid,'gaussian')))
            fresh=torch.stack(fresh)
            torch.testing.assert_close(fresh,saved['fresh_messages'].to('mps'),rtol=0,atol=0)
            rr=host['reports'].to('mps');coeff=1+2*(rr/.5).clamp(0,1);lam=coeff/coeff.sum()
            rows={d['branch']:d for d in r['rows']};assert set(rows)==set(run.BRANCHES)
            eta=2. if k<=60 else .5
            for mode,messages in [('recursive',host['sent'].to('mps')),('fresh',fresh)]:
                A,_=weighted_rfa(messages,lam);norm=float(torch.linalg.vector_norm(A))
                for control in ('unchanged','half','global_clip'):
                    branch=f'{mode}__{control}';row=rows[branch]
                    # Reconstruct independently, without controlled_step or scheduled aggregate.
                    factor={'unchanged':1.,'half':.5,'global_clip':min(1.,1./norm) if norm else 1.}[control]
                    step=eta*factor*A
                    torch.testing.assert_close(step,saved['steps'][branch].to('mps'),rtol=2e-5,atol=2e-7)
                    close(row['control']['factor'],factor);close(row['control']['eta'],eta)
                    close(row['control']['step_norm'],float(torch.linalg.vector_norm(step)))
                    if control=='global_clip':assert float(torch.linalg.vector_norm(step))<=eta+2e-6
                    model.load_state_dict(host['before']);base.apply_gradient(model,saved['steps'][branch].to('mps'),1.)
                    val=base.evaluate(model,data,'val');verify(val);verify(row['validation'])
                    for metric in ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2','ce_loss','brier_loss','balanced_accuracy_pct'):
                        close(val[metric],row['validation'][metric])
                    gain=r['J0']-J(model,data);close(gain,row['J_gain'])
                    first=float(torch.dot(pop['true_J_gradient'].to('mps'),step))
                    close(first,row['first_order_decrease']);close(first-gain,row['first_order_residual'])
                    close(float((step/eta-pop['unclipped_target'].to('mps')).square().sum()),row['effective_aggregate_error_sq'])
                    count+=1
            records.append(r);print(f'{count}/60 endpoint evaluations and steps verified',flush=True)
        del data,model;torch.mps.empty_cache()
    evidence=json.loads((out/'evidence.json').read_text());dec=evidence['decision'];seed_passes=[]
    for s in dec['seeds']:
        host=[r for r in records if r['seed']==s['seed']];cp=[]
        for c in s['contrasts']:
            ds=[]
            for r in host:
                ix={x['branch']:x for x in r['rows']};a=ix['recursive__global_clip'];b=ix[c['control']]
                ds.append((a['validation']['accuracy_pct']-b['validation']['accuracy_pct'],
                           a['validation']['worst20_pct']-b['validation']['worst20_pct'],a['J_gain']-b['J_gain']))
            avg=[st.mean(d[i] for d in ds) for i in range(3)]
            for i,name in enumerate(('accuracy','worst20','J')):close(avg[i],c['mean_differences'][name])
            ok=avg[0]>=-.1 and avg[1]>=0 and avg[2]>=0
            if c['control']=='recursive__unchanged':ok=ok and avg[1]>=.25 and avg[2]>0 and min(d[1] for d in ds)>=-.5
            assert ok==c['passed'];cp.append(ok)
        assert all(cp)==s['passed'];seed_passes.append(all(cp))
    assert all(seed_passes)==dec['admitted_to_end_to_end_screen']
    th=evidence['theory'];sigma=records[0]['ledger']['gradient_std'];gamma=th['theta'];rho=gamma+(1-gamma)*gamma
    expected=16*61706*(sigma*rho)**2/(10*2)
    close(expected,th['required_theta_at_least']);assert (gamma>=expected)==th['condition_passes']
    base.verify_stamp(stamp)
    base.save(ROOT/'output/analysis/PriSMA_Global_Step_V21_Independent_Audit.json',dict(audit_passed=True,
        device='mps',states=10,endpoint_evaluations=count,all_fresh_messages_bitwise_recomputed=True,
        all_controls_reconstructed_without_control_helper=True,all_gates_independently_recomputed=True,
        scope='Source hashes, fresh per-example queries, all controlled steps, 60 model/true-J evaluations, statistics and gate. Inherited V19/V20 audit covers original recursive host and accountant; no new universal convergence or Byzantine certificate.',
        admitted_to_end_to_end_screen=all(seed_passes),global_validation=False,source_stamp=stamp,auditor_sha256=base.digest(Path(__file__))))
    print(json.dumps(dict(audit_passed=True,admitted_to_end_to_end_screen=all(seed_passes))),flush=True)


if __name__=='__main__':main()
