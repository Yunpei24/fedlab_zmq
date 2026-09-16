#!/usr/bin/env python3
"""Audit V20 population directions through independent ordinary batch backprop."""
import json
import math
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import diagnose_recursive_population_v20 as run
from scripts import run_fair_objective_screen as base
from scripts.run_private_risk_channel_replay_v14 import risks
from privacy.fair_objective import require_mps,losses
from privacy.capped_private_risk import weights


def close(a,b):assert math.isclose(a,b,rel_tol=3e-5,abs_tol=2e-6),(a,b)


def main():
    require_mps();out=run.OUT;assert json.loads((out/'status.json').read_text())['status']=='completed'
    m=json.loads((out/'manifest.json').read_text());stamp=m['source_stamp'];base.verify_stamp(stamp)
    assert json.loads((out/'tests.json').read_text())['passed'];count=0;maximum_relative_gradient_error=0.
    for seed in run.parent.SEEDS:
        data=base.prepare(m['profile'],seed);model=base.new_model(m['profile'],seed)
        for k in run.parent.ROUNDS:
            path=out/f'seed{seed}__round{k}.json';r=json.loads(path.read_text())
            assert r['vector_sha256']==base.digest(path.with_suffix('.pt')) and r['source_stamp']==stamp
            v=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=True)
            original=torch.load(run.parent.OUT/f'seed{seed}__round{k}_vectors.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(original['before']);rr=risks(model,data);lam,coeff=weights(rr,.5)
            torch.testing.assert_close(rr,v['raw_risks'].to('mps'),rtol=2e-5,atol=2e-7)
            model.zero_grad(set_to_none=True)
            # This differentiates ordinary batch means, not per-example vmap.
            for cid,ids in enumerate(data['train']):
                for block in ids.split(256):
                    val=losses(model(data['x'][block]),data['y'][block],'brier').sum()
                    (coeff[cid].detach()*val/(10*len(ids))).backward()
            gradient=torch.cat([p.grad.detach().flatten() for p in model.parameters()])
            saved=v['true_J_gradient'].to('mps')
            rel=float(torch.linalg.vector_norm(gradient-saved)/torch.linalg.vector_norm(saved).clamp_min(1e-8))
            assert rel<=2e-4;maximum_relative_gradient_error=max(maximum_relative_gradient_error,rel)
            torch.testing.assert_close(gradient/coeff.mean(),v['unclipped_target'].to('mps'),rtol=2e-4,atol=2e-7)
            for name,agg in zip(('difference','complete'),r['aggregates']):
                A=v['aggregates'][name].to('mps');target=v['unclipped_target'].to('mps')
                close(float((A-target).square().sum()),agg['error_to_unclipped_fair_target_sq'])
                close(float(torch.dot(saved,A)),agg['true_J_gradient_dot_aggregate'])
                close(agg['predicted_first_order_decrease']-agg['actual_J_decrease'],agg['exact_first_order_residual'])
                for c,ds in zip(v['clients'],r['clients']):
                    p=c['past_error'].to('mps');b=c['methods'][name]['bias'].to('mps');xi=c['methods'][name]['sampling'].to('mps')
                    z=c['methods'][name]['noise'].to('mps');e=c['methods'][name]['error'].to('mps');d=ds['methods'][name]
                    torch.testing.assert_close(e,p+b+xi+z,rtol=2e-5,atol=2e-7)
                    for key,x in [('past_error_sq',p),('population_query_bias_sq',b),('sampling_error_sq',xi),('gaussian_realization_sq',z),('message_error_sq',e),('message_error_without_fresh_noise_sq',e-z)]:
                        close(d[key],float(x.square().sum()))
                    close(d['past_bias_cross'],float(2*torch.dot(p,b)))
                    close(d['bias_sampling_cross'],float(2*torch.dot(b,xi)))
            count+=1;print(f'{count}/10 population gradients verified by independent batch backprop',flush=True)
        del data,model;torch.mps.empty_cache()
    base.save(ROOT/'output/analysis/Recursive_Complete_Query_V20_Population_Independent_Audit.json',
        dict(audit_passed=True,device='mps',states=count,max_relative_J_gradient_error=maximum_relative_gradient_error,
            source_stamp=stamp,auditor_sha256=base.digest(Path(__file__)),global_validation=False,
            scope='Independent ordinary-batch backprop of true J gradient; stored-vector identities/errors and hashes. Per-example population clipping means are not independently recomputed a second time.'))
    print(json.dumps(dict(audit_passed=True,max_relative_J_gradient_error=maximum_relative_gradient_error)))


if __name__=='__main__':main()
