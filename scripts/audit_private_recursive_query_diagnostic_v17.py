#!/usr/bin/env python3
"""Recompute all 80 two-model gradient blocks and every registered V17 criterion."""
import json
import math
import os
from pathlib import Path
import statistics as st
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_private_recursive_query_diagnostic_v17 as run
from scripts import run_fair_objective_screen as base
from privacy.fair_objective import require_mps,per_example


def close(a,b):
    assert math.isclose(a,b,rel_tol=3e-5,abs_tol=3e-6),(a,b)


def main():
    require_mps();out=run.OUT;manifest=json.loads((out/'manifest.json').read_text())
    stamp=manifest['source_stamp'];base.verify_stamp(stamp)
    assert json.loads((out/'status.json').read_text())['status']=='completed'
    key=json.loads((out/'simulator_secret.json').read_text())['key']
    files=sorted(out.glob('seed*__r*__client*.json'));assert len(files)==80
    recomputed=[]
    for seed in run.SEEDS:
        data=base.prepare(manifest['profile'],seed)
        old=base.new_model(manifest['profile'],seed);new=base.new_model(manifest['profile'],seed)
        state=torch.load(run.prior.population.SOURCE/f'seed{seed}__erm_mean'/'checkpoint.pt',map_location='cpu',weights_only=True)['model']
        old.load_state_dict(state)
        for replay in range(4):
            step=torch.load(run.prior.OUT/f'seed{seed}__r{replay}__C2_eta0.5__risk_rfa.pt',map_location='mps',weights_only=True)['step']
            new.load_state_dict(state);base.apply_gradient(new,step,1.)
            for cid,ids in enumerate(data['train']):
                path=out/f'seed{seed}__r{replay}__client{cid}.json';r=json.loads(path.read_text())
                assert r['source_stamp']==stamp and r['device']=='mps' and not r['test_evaluated']
                assert base.digest(path.with_suffix('.pt'))==r['vectors_sha256']
                cache=torch.load(path.with_suffix('.pt'),map_location='mps',weights_only=True)
                assert cache['source_stamp']==stamp and not cache['privacy_protected']
                ix=base.draw_indices(4800,240,base.seed_for(key,seed,replay,cid,'new_batch'))
                assert base.ids_hash(ix)==r['batch_hash'];assert torch.equal(ix,cache['indices'])
                _,g0,_,_=per_example(old,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)
                _,g1,_,_=per_example(new,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)
                torch.testing.assert_close(g0.mean(0),cache['before_mean'],rtol=2e-5,atol=2e-6)
                torch.testing.assert_close(g1.mean(0),cache['now_mean'],rtol=2e-5,atol=2e-6)
                diff=g1-g0;norms=torch.linalg.vector_norm(diff,dim=1)
                torch.testing.assert_close(norms,cache['increment_norms'],rtol=2e-5,atol=2e-6)
                denom=max(float(torch.linalg.vector_norm(g1.mean(0))),1e-6)
                close(denom,r['current_batch_mean_gradient_norm'])
                assert len(r['variants'])==15
                for ratio in run.RATIOS:
                    clipped=diff*torch.minimum(torch.ones_like(norms),2*ratio/norms.clamp_min(1e-20))[:,None]
                    torch.testing.assert_close(clipped.mean(0),cache['increment_means'][str(ratio)],rtol=2e-5,atol=2e-6)
                    norm=float(torch.linalg.vector_norm((clipped-diff).mean(0)))
                    fraction=float((norms>2*ratio).float().mean())
                    for theta in run.THETAS:
                        v=next(v for v in r['variants'] if v['theta']==theta and v['D_over_C']==ratio)
                        close(norm,v['increment_bias_norm']);close(fraction,v['fraction_clipped'])
                        close((1-theta)*norm/(theta*denom),v['coherent_bias_proxy_relative'])
                        sigma_ratio=theta+(1-theta)*ratio
                        variance=1.
                        for _ in range(20):variance=(1-theta)**2*variance+sigma_ratio**2
                        close(variance,v['finite_noise_variance_ratio'])
                        close(sigma_ratio**2/(1-(1-theta)**2),v['stationary_noise_variance_ratio'])
                recomputed.append(r);print(f'{len(recomputed)}/80 paired blocks independently reconstructed',flush=True)
        del data,old,new;torch.mps.empty_cache()
    evidence=json.loads((out/'evidence.json').read_text());expected=[]
    for theta in run.THETAS:
        for ratio in run.RATIOS:
            r=theta+(1-theta)*ratio;vinf=r*r/(1-(1-theta)**2)
            finite=(1-theta)**40+vinf*(1-(1-theta)**40)
            passed=vinf<=.5 and finite<=.55
            for seed in run.SEEDS:
                vv=[next(v for v in q['variants'] if v['theta']==theta and v['D_over_C']==ratio) for q in recomputed if q['seed']==seed]
                assert len(vv)==40
                bb=sorted(v['coherent_bias_proxy_relative'] for v in vv)
                p90=.9*bb[35]+.1*bb[36]  # position (40−1)*.9 = 35.1
                fraction=sum(v['fraction_clipped']<=.1 for v in vv)/40
                ok=st.median(bb)<=.1 and p90<=.25 and fraction>=.9
                g=next(g for g in evidence['groups'] if g['seed']==seed and g['theta']==theta and g['D_over_C']==ratio)
                close(g['p90_bias'],p90);close(g['median_bias'],st.median(bb))
                close(g['fraction_blocks_below_10pct_clip'],fraction);assert g['passed']==ok
                passed=passed and ok
            stored=next(c for c in evidence['decision']['checks'] if c['theta']==theta and c['D_over_C']==ratio)
            assert stored['passed']==passed
            if passed:expected.append(dict(theta=theta,D_over_C=ratio))
    expected.sort(key=lambda x:(-x['theta'],-x['D_over_C']))
    selected=expected[0] if expected else None
    assert selected==evidence['decision']['selected_for_calibration']
    dest=ROOT/'output/analysis/Private_Recursive_Query_V17_Independent_Audit.json'
    base.save(dest,dict(audit_passed=True,device='mps',blocks_reconstructed=80,variants_recomputed=1200,
        source_sha256=base.digest(Path(__file__)),selected_for_calibration=selected,global_validation=False))
    print(json.dumps(dict(audit_passed=True,blocks_reconstructed=80,selected=selected)))


if __name__=='__main__':main()
