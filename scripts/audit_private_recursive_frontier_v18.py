#!/usr/bin/env python3
"""MPS reconstruction of V18 plus exact-count boundary correction for V17."""
import json
import math
import os
from pathlib import Path
import statistics as st
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_private_recursive_frontier_diagnostic_v18 as run
from scripts import run_fair_objective_screen as base
from privacy.fair_objective import require_mps,per_example


def close(a,b):
    assert math.isclose(a,b,rel_tol=3e-5,abs_tol=3e-6),(a,b)


def main():
    require_mps();out=run.OUT;manifest=json.loads((out/'manifest.json').read_text());stamp=manifest['source_stamp']
    base.verify_stamp(stamp);assert json.loads((out/'status.json').read_text())['status']=='completed'
    theta=1-math.sqrt(.5+1e-8);D=2*theta
    rows=[];v17blocks=[];boundary=[]
    for seed in run.prior.SEEDS:
        data=base.prepare(manifest['profile'],seed);old=base.new_model(manifest['profile'],seed);new=base.new_model(manifest['profile'],seed)
        state=torch.load(run.prior.prior.population.SOURCE/f'seed{seed}__erm_mean'/'checkpoint.pt',map_location='cpu',weights_only=True)['model']
        old.load_state_dict(state)
        for replay in range(4):
            step=torch.load(run.prior.prior.OUT/f'seed{seed}__r{replay}__C2_eta0.5__risk_rfa.pt',map_location='mps',weights_only=True)['step']
            new.load_state_dict(state);base.apply_gradient(new,step,1.)
            for cid,ids in enumerate(data['train']):
                file=out/f'seed{seed}__r{replay}__client{cid}.json';r=json.loads(file.read_text())
                assert r['source_stamp']==stamp and r['device']=='mps' and not r['test_evaluated']
                assert base.digest(file.with_suffix('.pt'))==r['vector_sha256']
                vec=torch.load(file.with_suffix('.pt'),map_location='mps',weights_only=True)
                previous=run.prior.OUT/file.name;pv=json.loads(previous.read_text())
                pc=torch.load(previous.with_suffix('.pt'),map_location='mps',weights_only=True);ix=pc['indices']
                assert base.ids_hash(ix)==r['batch_hash']==pv['batch_hash']
                _,g0,_,_=per_example(old,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)
                _,g1,_,_=per_example(new,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)
                delta=g1-g0;norm=torch.linalg.vector_norm(delta,dim=1)
                cl=delta*(D/norm.clamp_min(1e-20)).clamp(max=1)[:,None]
                bias=(cl-delta).mean(0);count=int((norm>D).sum())
                proxy=(1-theta)/theta*float(torch.linalg.vector_norm(bias))/max(float(torch.linalg.vector_norm(g1.mean(0))),1e-6)
                torch.testing.assert_close(bias,vec['error_mean'],rtol=2e-5,atol=2e-6)
                torch.testing.assert_close(norm,vec['increment_norms'],rtol=2e-5,atol=2e-6)
                close(proxy,r['coherent_bias_proxy_relative']);close(count/240,r['fraction_clipped'])
                r['exact_clipped_count']=count;rows.append(r)
                for ratio in run.prior.RATIOS:
                    k=int((norm>2*ratio).sum())
                    for v in pv['variants']:
                        if v['D_over_C']!=ratio:continue
                        close(k/240,v['fraction_clipped'])
                        v17blocks.append(dict(seed=seed,theta=v['theta'],ratio=ratio,count=k,bias=v['coherent_bias_proxy_relative']))
                        if (v['fraction_clipped']<=.1)!=(k<=24):
                            boundary.append(dict(seed=seed,replay=replay,client=cid,theta=v['theta'],D_over_C=ratio,
                                count=k,denominator=240,stored_fraction=v['fraction_clipped'],exact_fraction=k/240))
                print(f'{len(rows)}/80 frontier blocks independently audited',flush=True)
        del data,old,new;torch.mps.empty_cache()
    evidence=json.loads((out/'evidence.json').read_text());allpass=True
    for seed in run.prior.SEEDS:
        rr=[r for r in rows if r['seed']==seed];assert len(rr)==40
        b=sorted(r['coherent_bias_proxy_relative'] for r in rr)
        median=st.median(b);p90=.9*b[35]+.1*b[36];fraction=sum(r['exact_clipped_count']<=24 for r in rr)/40
        g=next(x for x in evidence['checks'] if x['seed']==seed)
        close(g['bias_median'],median);close(g['bias_p90'],p90);close(g['fraction_blocks_below_10pct_clip'],fraction)
        ok=median<=.1 and p90<=.25 and fraction>=.9
        assert ok==g['passed'];allpass=allpass and ok
    vinf=theta*(2-theta);v20=(1-theta)**40+vinf*(1-(1-theta)**40)
    close(vinf,evidence['public']['stationary_ratio']);close(v20,evidence['public']['finite_ratio_t20'])
    admitted=allpass and vinf<=.5 and v20<=.55
    assert admitted==evidence['admitted_to_calibration']
    base.save(ROOT/'output/analysis/Private_Recursive_Query_V18_Independent_Audit.json',
        dict(audit_passed=True,device='mps',blocks_reconstructed=80,exact_counts_used=True,admitted_to_calibration=admitted,
             global_validation=False,source_sha256=base.digest(Path(__file__))))
    # Preserve frozen evidence, record the numerical threshold correction separately.
    corrected=[];eligible=[]
    for t in run.prior.THETAS:
        for d in run.prior.RATIOS:
            public=(t+(1-t)*d)**2/(t*(2-t));vp=(1-t)**40+public*(1-(1-t)**40)
            total=public<=.5 and vp<=.55
            for seed in run.prior.SEEDS:
                blocks=[b for b in v17blocks if (b['seed'],b['theta'],b['ratio'])==(seed,t,d)];assert len(blocks)==40
                biases=sorted(b['bias'] for b in blocks);frac=sum(b['count']<=24 for b in blocks)/40
                med=st.median(biases);p90=.9*biases[35]+.1*biases[36]
                ok=frac>=.9 and med<=.1 and p90<=.25;total=total and ok
                corrected.append(dict(seed=seed,theta=t,D_over_C=d,exact_fraction_blocks_below_10pct_clip=frac,
                                      bias_median=med,bias_p90=p90,local_passed=ok))
            if total:eligible.append(dict(theta=t,D_over_C=d))
    assert not eligible  # The global V17 decision must remain negative.
    base.save(ROOT/'output/analysis/Private_Recursive_Query_V17_Exact_Counts_Amendment.json',
        dict(audit_passed=True,device='mps',boundary_cases=boundary,corrected_groups=corrected,
             eligible=eligible,global_decision_unchanged=True,
             reason='24/240 equals 10%; a float32 value 0.10000000149 must not make this block fail.',
             historical_outputs_preserved=True))
    print(json.dumps(dict(audit_passed=True,admitted_to_calibration=admitted,V17_boundary_rows=len(boundary))))


if __name__=='__main__':main()
