#!/usr/bin/env python3
"""Independent scalar audit of all 48 stored V14 evaluations and fixed gates."""
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts.analyze_private_risk_confirmation_v12 import verify
from privacy.fair_objective import wor_rdp


def main():
    out=ROOT/'results/ldp_gradient_far/private_risk_channel_replay_v14'
    manifest=json.loads((out/'manifest.json').read_text())
    for path,sha in manifest['source_stamp'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    status=json.loads((out/'status.json').read_text());assert status['status']=='completed'
    blocks=[json.loads(p.read_text()) for p in sorted(out.glob('seed*__replay*.json'))]
    assert len(blocks)==16 and {(b['seed'],b['replay']) for b in blocks}=={(s,r) for s in (170501,170502) for r in range(8)}
    def J(v):return st.mean(r+r*r/.5 if r<=.5 else 3*r-.5 for r in [c['brier_loss'] for c in v['clients']])
    for b in blocks:
        assert b['device']=='mps' and b['oracle_only'] and not b['test_evaluated'] and b['training_steps']==0
        assert b['source_stamp']==manifest['source_stamp']
        verify(b['initial']);assert [r['method'] for r in b['rows']]==['original','reallocated','oracle_risk']
        for r in b['rows']:
            verify(r['validation'])
            assert math.isclose(J(b['initial'])-J(r['validation']),r['J_gain'],abs_tol=1e-12)
            assert r['eligible_private']==(r['method']!='oracle_risk')
            if r['eligible_private']:
                p=r['privacy_plan']
                eps=min(120*wor_rdp(a,.05,p['gradient_z'])+120*a/(2*p['risk_z']**2)+math.log(1e5)/(a-1) for a in range(2,65))
                assert math.isclose(eps,p['epsilon_realized'],abs_tol=1e-12) and eps<=4
            assert r['mse_to_fair_clipped_batch_target']>=0 and math.isfinite(r['mse_to_fair_clipped_batch_target'])
    grouped=[]
    checks=[]
    for seed in (170501,170502):
        means={}
        for method in ('original','reallocated','oracle_risk'):
            q=[r for b in blocks if b['seed']==seed for r in b['rows'] if r['method']==method]
            assert len(q)==8
            means[method]=dict(mse=st.mean(r['mse_to_fair_clipped_batch_target'] for r in q),
                J=st.mean(r['J_gain'] for r in q),accuracy=st.mean(r['validation']['accuracy_pct'] for r in q),
                worst20=st.mean(r['validation']['worst20_pct'] for r in q))
        a,b=means['reallocated'],means['original']
        changes=dict(mse_relative_reduction=1-a['mse']/b['mse'],J_gain_delta=a['J']-b['J'],
                     accuracy_delta_pp=a['accuracy']-b['accuracy'],worst20_delta_pp=a['worst20']-b['worst20'])
        gates=dict(mse=changes['mse_relative_reduction']>=.01,J=changes['J_gain_delta']>=0,
                   accuracy=changes['accuracy_delta_pp']>=-.1,worst20=changes['worst20_delta_pp']>=0)
        checks.append(dict(seed=seed,changes=changes,gates=gates,passed=all(gates.values())))
        grouped.append(dict(seed=seed,means=means))
    evidence=json.loads((out/'evidence.json').read_text())
    assert checks==evidence['decision']['checks']
    admissible=all(c['passed'] for c in checks)
    assert admissible==status['calibration_admissible']==evidence['decision']['calibration_admissible']
    result=dict(audit_passed=True,valid_blocks=16,validation_evaluations=48,test_evaluated=False,
                calibration_admissible=admissible,checks=checks,grouped=grouped,
                missing_vector_replay_audit='Saved scalar squared errors cannot independently reconstruct missing full vectors; computed by frozen MPS code.',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dest=ROOT/'output/analysis/Private_Risk_Channel_Replay_V14_Independent_Audit.json'
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
