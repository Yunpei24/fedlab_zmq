#!/usr/bin/env python3
"""MPS replay of all stored intervention vectors; re-evaluate J and validation."""
import hashlib
import json
import math
import os
from pathlib import Path
import statistics as st
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_fair_objective_screen as base
from scripts import run_private_fairness_population_diagnostic_v15 as run
from scripts.analyze_private_risk_confirmation_v12 import verify
from privacy.fair_objective import require_mps
from privacy.stable_weighted_rfa import stable_norm


def close(a,b):
    assert math.isclose(a,b,rel_tol=2e-5,abs_tol=2e-6),(a,b)


def main():
    require_mps()
    out=run.OUT
    manifest=json.loads((out/'manifest.json').read_text());base.verify_stamp(manifest['source_stamp'])
    assert json.loads((out/'status.json').read_text())['status']=='completed'
    paths=sorted(out.glob('seed*__*__r*.json'));assert len(paths)==40
    rows=[]
    for seed in run.SEEDS:
        data=base.prepare(manifest['profile'],seed);model=base.new_model(manifest['profile'],seed)
        checkpoint=torch.load(run.SOURCE/f'seed{seed}__erm_mean'/'checkpoint.pt',map_location='cpu',weights_only=True)['model']
        pop=torch.load(out/f'population_seed{seed}.pt',map_location='mps',weights_only=True)['population']
        a=1+2*(pop['risks']/.5).clamp(0,1)
        true_target=((a/a.sum())[:,None]*pop['raw']).sum(0)
        true_J=(a[:,None]*pop['raw']).mean(0)
        for file in [p for p in paths if p.name.startswith(f'seed{seed}__')]:
            r=json.loads(file.read_text());assert r['source_stamp']==manifest['source_stamp'] and r['device']=='mps'
            vecfile=file.with_suffix('.pt');assert base.digest(vecfile)==r['vector_sha256']
            vec=torch.load(vecfile,map_location='mps',weights_only=True)
            assert vec['source_stamp']==manifest['source_stamp'] and not vec['privacy_protected']
            A=vec['aggregate'];target=vec['target'];GJ=vec['true_J_gradient']
            torch.testing.assert_close(target,true_target,rtol=2e-6,atol=2e-7)
            torch.testing.assert_close(GJ,true_J,rtol=2e-6,atol=2e-7)
            close(float(stable_norm(A-target).square()),r['mse_to_unclipped_fair_population'])
            close(float(.5*torch.dot(GJ,A)),r['predicted_J_gain'])
            if r['condition']=='population_fair':assert r['predicted_J_gain']>=0
            model.load_state_dict(checkpoint)
            base.apply_gradient(model,A,.5)
            v=base.evaluate(model,data,'val');verify(v);verify(r['validation'])
            for k in ('accuracy_pct','worst20_pct','variance_pp2','gap_best20_worst20_pp','ce_loss','brier_loss','balanced_accuracy_pct'):
                close(v[k],r['validation'][k])
            risk_after=run.risks(model,data)
            # Independent scalar expression for the public fair-risk potential.
            values=risk_after.cpu().tolist()
            actual_after=st.mean(x+2*x*x if x<=.5 else 3*x-.5 for x in values)
            close(actual_after,r['J_population_after'])
            close(r['J_population_before']-actual_after,r['J_population_gain'])
            close(r['predicted_J_gain']-r['J_population_gain'],r['taylor_remainder'])
            close(v['accuracy_pct']-r['initial_validation']['accuracy_pct'],r['accuracy_delta_pp'])
            close(v['worst20_pct']-r['initial_validation']['worst20_pct'],r['worst20_delta_pp'])
            rows.append(r)
            print(f'{len(rows)}/40 interventions replayed and verified',flush=True)
        del data,model,pop;torch.mps.empty_cache()
    evidence=json.loads((out/'evidence.json').read_text())
    for c in evidence['contrasts']:
        aa=[r for r in rows if r['seed']==c['seed'] and r['condition']==c['treated']]
        bb=[r for r in rows if r['seed']==c['seed'] and r['condition']==c['control']]
        for k,val in c['delta'].items():close(st.mean(r[k] for r in aa)-st.mean(r[k] for r in bb),val)
    payload=dict(audit_passed=True,device='mps',interventions_replayed=40,validation_and_J_recomputed=True,
        contrasts=evidence['contrasts'],grouped=evidence['grouped'],candidate_promoted=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dest=ROOT/'output/analysis/Private_Fairness_Population_V15_Independent_Audit.json'
    base.save(dest,payload)
    print(json.dumps(dict(audit_passed=True,interventions_replayed=40,report=str(dest))))


if __name__=='__main__':main()
