#!/usr/bin/env python3
"""Independent recomputation of V20 queries, paired releases and endpoints."""
import json
import math
import os
from pathlib import Path
import statistics as st
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_complete_recursive_query_diagnostic_v20 as run
from scripts import run_fair_objective_screen as base
from scripts.run_private_risk_channel_replay_v14 import risks
from scripts.analyze_private_risk_confirmation_v12 import verify
from privacy.fair_objective import require_mps,per_example,release
from privacy.split_risk_gradient import private_risk
from privacy.capped_private_risk import potential
from privacy.scheduled_private_risk import aggregate


def close(a,b):assert math.isclose(a,b,rel_tol=2e-5,abs_tol=2e-6),(a,b)


def main():
    require_mps();out=run.OUT;m=json.loads((out/'manifest.json').read_text());stamp=m['source_stamp'];base.verify_stamp(stamp)
    assert json.loads((out/'status.json').read_text())['status']=='completed'
    assert json.loads((out/'tests.json').read_text())['passed']
    key=json.loads((run.prior.OUT/'simulator_secret.json').read_text())['key'];records=[]
    th=1-math.sqrt(.5+1e-8);a=1-th;D=2*th;S=2*th+a*D
    for seed in run.SEEDS:
        data=base.prepare(m['profile'],seed);model=base.new_model(m['profile'],seed);oldmodel=base.new_model(m['profile'],seed)
        source=json.loads((run.prior.OUT/f'seed{seed}__recursive__risk_rfa'/'metrics.json').read_text());p=source['privacy']
        assert json.loads((out/f'seed{seed}__replay_audit.json').read_text())['final_model_and_memory_bitwise_equal']
        for k in run.ROUNDS:
            path=out/f'seed{seed}__round{k}.json';r=json.loads(path.read_text());assert r['source_stamp']==stamp and r['oracle_only'] and not r['test_evaluated']
            v=torch.load(out/f'seed{seed}__round{k}_vectors.pt',map_location='cpu',weights_only=True);assert v['source_stamp']==stamp
            model.load_state_dict(v['before']);oldmodel.load_state_dict(v['previous_model']);mem=v['private_memory'].to('mps')
            verify(r['initial']);J0=float(potential(risks(model,data),.5).mean());close(J0,r['J0'])
            original=[];projected=[];reports=[]
            for cid,ids in enumerate(data['train']):
                d=r['client_diagnostics'][cid];assert d['client']==cid
                rr,_=private_risk(model,data['x'][ids],data['y'][ids],noise_std=p['risk_std'],seed=base.seed_for(key,seed,k-1,cid,'risk'),N=4800)
                reports.append(rr);torch.testing.assert_close(rr,v['reports'][cid].to('mps'),rtol=0,atol=0)
                ix=base.draw_indices(4800,240,base.seed_for(key,seed,k-1,cid,'batch'));assert base.ids_hash(ix)==d['batch_hash']
                g=per_example(model,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)[1]
                gprev=per_example(oldmodel,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)[1]
                diff=g-gprev;dn=diff.square().sum(1).sqrt();exact=g-a*gprev;en=exact.square().sum(1).sqrt()
                # Independent row formulas, not the new query module.
                q1=th*g+a*(diff*(D/dn.clamp_min(1e-20)).clamp(max=1)[:,None])
                q2=exact*(S/en.clamp_min(1e-20)).clamp(max=1)[:,None]
                assert d['difference_clipped_count']==int((dn>D).sum()) and d['complete_query_clipped_count']==int((en>S).sum())
                close(d['difference_batch_bias_sq'],float((q1-exact).mean(0).square().sum()))
                close(d['complete_batch_bias_sq'],float((q2-exact).mean(0).square().sum()))
                close(d['difference_individual_distortion'],float((q1-exact).square().sum(1).mean()))
                close(d['complete_individual_distortion'],float((q2-exact).square().sum(1).mean()))
                assert float(((exact-q2).square().sum(1)+(q2-q1).square().sum(1)-(exact-q1).square().sum(1)).max())<=2e-5
                close(d['effective_C'],S);close(d['query_sensitivity'],2*S/240);close(d['noise_std'],p['gradient_z']*2*S/240)
                for q,arr in [(q1,original),(q2,projected)]:
                    assert float(q.square().sum(1).sqrt().max())<=S+2e-6
                    arr.append(a*mem[cid]+release(q.mean(0),noise_std=p['gradient_std']*S/2,seed=base.seed_for(key,seed,k-1,cid,'gaussian')))
            original=torch.stack(original);projected=torch.stack(projected);reports=torch.stack(reports)
            torch.testing.assert_close(original,v['sent'].to('mps'),rtol=2e-5,atol=2e-7)
            torch.testing.assert_close(projected,v['complete'].to('mps'),rtol=2e-5,atol=2e-7)
            for messages,row in [(v['sent'].to('mps'),r['rows'][0]),(v['complete'].to('mps'),r['rows'][1])]:
                model.load_state_dict(v['before']);step,_=aggregate(messages,reports,kind='risk_rfa',round_number=k,horizon=120)
                base.apply_gradient(model,step,1.);val=base.evaluate(model,data,'val');verify(val);verify(row['validation'])
                for metric in ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2','balanced_accuracy_pct','ce_loss','brier_loss'):
                    close(val[metric],row['validation'][metric])
                J=float(potential(risks(model,data),.5).mean());close(J,row['J']);close(J0-J,row['J_gain'])
            for metric,delta in r['delta'].items():
                calc=(r['rows'][1]['J_gain']-r['rows'][0]['J_gain'] if metric=='J_gain' else
                      r['rows'][1]['validation'][metric]-r['rows'][0]['validation'][metric])
                close(calc,delta)
            records.append(r);print(f'{len(records)}/10 states independently reconstructed',flush=True)
        del data,model,oldmodel;torch.mps.empty_cache()
    evidence=[]
    for seed in run.SEEDS:
        rows=[r for r in records if r['seed']==seed];assert sorted(r['round'] for r in rows)==[2,10,30,60,120]
        old=st.mean(d['difference_batch_bias_sq'] for r in rows if r['round']<120 for d in r['client_diagnostics'])
        new=st.mean(d['complete_batch_bias_sq'] for r in rows if r['round']<120 for d in r['client_diagnostics'])
        ok=(new<=.9*old and st.mean(r['delta']['worst20_pct'] for r in rows)>=0
            and min(r['delta']['worst20_pct'] for r in rows)>=-.5 and st.mean(r['delta']['accuracy_pct'] for r in rows)>=-.1
            and st.mean(r['delta']['J_gain'] for r in rows)>=0)
        evidence.append(dict(seed=seed,passed=ok,relative_bias_reduction=None if old==0 else 1-new/old))
    admitted=all(e['passed'] for e in evidence)
    assert admitted==json.loads((out/'evidence.json').read_text())['decision']['admitted_to_new_end_to_end_screen']
    result=dict(audit_passed=True,states=10,device='mps',per_example_queries_and_noise_reconstructed=True,
        all_endpoints_recomputed=True,source_stamp=stamp,auditor_sha256=base.digest(Path(__file__)),
        seeds=evidence,admitted_to_new_end_to_end_screen=admitted,global_validation=False)
    base.save(ROOT/'output/analysis/Recursive_Complete_Query_V20_Independent_Audit.json',result)
    print(json.dumps(dict(audit_passed=True,admitted_to_new_end_to_end_screen=admitted)),flush=True)


if __name__=='__main__':main()
