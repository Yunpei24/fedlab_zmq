#!/usr/bin/env python3
"""Independently reconstruct messages, privacy, steps and endpoints for V16."""
import json
import math
import os
from pathlib import Path
import statistics as st
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_private_clipping_step_diagnostic_v16 as run
from scripts import run_fair_objective_screen as base
from scripts.analyze_private_risk_confirmation_v12 import verify
from privacy.fair_objective import require_mps, wor_rdp, per_example, release
from privacy.stable_weighted_rfa import weighted_rfa, stable_norm


def close(a,b):
    assert math.isclose(a,b,rel_tol=2e-5,abs_tol=2e-6),(a,b)


def scalar_J(values):
    return st.mean(x+2*x*x if x<=.5 else 3*x-.5 for x in values)


def main():
    require_mps()
    out=run.OUT;manifest=json.loads((out/'manifest.json').read_text());stamp=manifest['source_stamp']
    base.verify_stamp(stamp)
    assert json.loads((out/'status.json').read_text())['status']=='completed'
    rows=[json.loads(p.read_text()) for p in sorted(out.glob('seed*__r*__C*__*.json'))]
    expected={(s,r,C,e,k) for s in run.SEEDS for r in range(4) for C,e in run.COUPLES for k in run.RULES}
    assert len(rows)==160 and {(r['seed'],r['replay'],r['C'],r['eta'],r['rule']) for r in rows}==expected
    key=json.loads((out/'simulator_secret.json').read_text())['key']
    done=0;batch_reconstructions=0;hashes={}
    for seed in run.SEEDS:
        data=base.prepare(manifest['profile'],seed);model=base.new_model(manifest['profile'],seed)
        state=torch.load(run.population.SOURCE/f'seed{seed}__erm_mean'/'checkpoint.pt',map_location='cpu',weights_only=True)['model']
        pop=torch.load(run.population.OUT/f'population_seed{seed}.pt',map_location='mps',weights_only=True)['population']
        coeff=1+2*(pop['risks']/.5).clamp(0,1)
        GJ=(coeff[:,None]*pop['raw']).mean(0)
        target=((coeff/coeff.sum())[:,None]*pop['raw']).sum(0)
        model.load_state_dict(state);initial=base.evaluate(model,data,'val');verify(initial)
        for replay in range(4):
            cachepath=out/f'block_seed{seed}_r{replay}.pt'
            cache=torch.load(cachepath,map_location='mps',weights_only=True)
            assert cache['source_stamp']==stamp and not cache['privacy_protected']
            hashes[cachepath.name]=base.digest(cachepath)
            model.load_state_dict(state)
            for cid,ids in enumerate(data['train']):
                idx=base.draw_indices(4800,240,base.seed_for(key,seed,replay,cid,'batch'))
                assert base.ids_hash(idx)==cache['batch_hashes'][cid]
                _,g8,_,_=per_example(model,data['x'][ids[idx]],data['y'][ids[idx]],clip_norm=8.)
                for C in (2.,4.,8.):
                    # Separately spelled radial clipping, no runner helper.
                    n=torch.linalg.vector_norm(g8,dim=1)
                    direct=(g8*torch.minimum(torch.ones_like(n),C/n.clamp_min(1e-20))[:,None]).mean(0)
                    torch.testing.assert_close(direct,cache['raw'][str(C)][cid],rtol=2e-5,atol=2e-6)
                zn=release(torch.zeros_like(cache['noise'][cid]),noise_std=1.,seed=base.seed_for(key,seed,replay,cid,'gradient'))
                zr=release(torch.zeros(1,device='mps'),noise_std=1.,seed=base.seed_for(key,seed,replay,cid,'risk')).squeeze()
                torch.testing.assert_close(zn,cache['noise'][cid],rtol=0,atol=0)
                torch.testing.assert_close(zr,cache['report_noise'][cid],rtol=0,atol=0)
                batch_reconstructions+=1
            for r in [r for r in rows if r['seed']==seed and r['replay']==replay]:
                C,eta,kind=r['C'],r['eta'],r['rule'];p=r['privacy']
                assert r['source_stamp']==stamp and r['device']=='mps' and not r['test_evaluated']
                assert r['cumulative_training_steps']==0 and p['epsilon_realized']<=4
                eps=min(120*wor_rdp(a,.05,p['gradient_z'])+
                        (120*a/(2*p['risk_z']**2) if kind.startswith('risk_') else 0)+math.log(1e5)/(a-1)
                        for a in range(2,65))
                close(eps,p['epsilon_realized']);close(p['gradient_std'],2*C*p['gradient_z']/240)
                close(r['applied_message_noise_std'],eta*p['gradient_std'])
                msg=cache['raw'][str(C)]+p['gradient_std']*cache['noise']
                if kind.startswith('risk_'):
                    close(p['risk_std'],p['risk_z']/4800)
                    rr=(pop['risks']+p['risk_std']*cache['report_noise']).clamp(0,1)
                    a=1+2*(rr/.5).clamp(0,1);lam=a/a.sum()
                else:rr=None;lam=torch.ones(10,device='mps')/10
                A=weighted_rfa(msg,lam)[0] if kind.endswith('rfa') else (lam[:,None]*msg).sum(0)
                name=f'seed{seed}__r{replay}__C{C:g}_eta{eta:g}__{kind}'
                f=out/(name+'.pt');assert base.digest(f)==r['vector_sha256']
                vec=torch.load(f,map_location='mps',weights_only=True)
                assert vec['source_stamp']==stamp and not vec['privacy_protected']
                torch.testing.assert_close(msg,vec['messages'],rtol=0,atol=0)
                torch.testing.assert_close(eta*A,vec['step'],rtol=2e-5,atol=2e-6)
                if rr is None:assert vec['reports'] is None
                else:torch.testing.assert_close(rr,vec['reports'],rtol=0,atol=0)
                close(float(stable_norm(vec['step']-.5*target).square()),r['mse_of_step'])
                close(float(torch.dot(GJ,vec['step'])),r['predicted_J_gain'])
                model.load_state_dict(state);base.apply_gradient(model,vec['step'],1.)
                v=base.evaluate(model,data,'val');verify(v);verify(r['validation'])
                for k in ('accuracy_pct','worst20_pct','variance_pp2','gap_best20_worst20_pp','ce_loss','brier_loss','balanced_accuracy_pct'):
                    close(v[k],r['validation'][k])
                close(scalar_J(run.risks(model,data).cpu().tolist()),r['J_after'])
                close(scalar_J(pop['risks'].cpu().tolist()),r['J_before'])
                close(r['J_before']-r['J_after'],r['J_gain'])
                close(v['accuracy_pct']-initial['accuracy_pct'],r['accuracy_delta_pp'])
                close(v['worst20_pct']-initial['worst20_pct'],r['worst20_delta_pp'])
                assert cache['batch_hashes']==r['batch_hashes']
                done+=1;print(f'{done}/160 endpoint replays audited',flush=True)
        del data,model,pop;torch.mps.empty_cache()
    for seed in run.SEEDS:
        for replay in range(4):
            for kind in run.RULES:
                vals=[r['applied_message_noise_std'] for r in rows if r['seed']==seed and r['replay']==replay and r['rule']==kind and r['eta']*r['C']==1]
                assert len(vals)==3 and max(vals)-min(vals)<1e-15
    ev=json.loads((out/'evidence.json').read_text())
    # Independently enumerate the registered contrasts from raw endpoint rows.
    def avg(seed,C,eta,rule,k):
        rr=[r for r in rows if (r['seed'],r['C'],r['eta'],r['rule'])==(seed,C,eta,rule)]
        assert len(rr)==4
        return st.mean(r[k] for r in rr)
    passed=[]
    for c in ev['decision']['checks']:
        truth=[]
        C,eta=c['C'],c['eta']
        for comparison in c['comparisons']:
            s= comparison['seed'];label=comparison['control']
            if label=='original':cc,ee,k,wmin,amin,j=2.,.5,'risk_rfa',.25,-.1,True
            elif label=='step_only':cc,ee,k,wmin,amin,j=2.,eta,'risk_rfa',.1,None,True
            else:cc,ee,k,wmin,amin,j=C,eta,label.removeprefix('same_pair_'),.5,-.25,False
            dd={key:avg(s,C,eta,'risk_rfa',key)-avg(s,cc,ee,k,key) for key in ('J_gain','accuracy_delta_pp','worst20_delta_pp')}
            for key,val in dd.items():close(val,comparison['delta'][key])
            ok=dd['worst20_delta_pp']>=wmin and (amin is None or dd['accuracy_delta_pp']>=amin) and (not j or dd['J_gain']>=-1e-7)
            assert ok==comparison['passed'];truth.append(ok)
        assert len(truth)==8 and all(truth)==c['passed']
        if all(truth):passed.append(dict(C=C,eta=eta))
    selected=passed[0] if passed else None
    assert selected==ev['decision']['selected_for_end_to_end_calibration']
    dest=ROOT/'output/analysis/Private_Clipping_Step_V16_Independent_Audit.json'
    base.save(dest,dict(audit_passed=True,device='mps',interventions_replayed=done,batches_reconstructed=batch_reconstructions,
        pairing_verified=True,privacy_recomputed=True,validation_and_J_recomputed=True,block_hashes=hashes,
        selected_for_end_to_end_calibration=selected,global_validation=False,source_sha256=base.digest(Path(__file__))))
    print(json.dumps(dict(audit_passed=True,interventions_replayed=done,selected=selected)))


if __name__=='__main__':main()
