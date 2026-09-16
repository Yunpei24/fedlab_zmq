#!/usr/bin/env python3
"""Single analytically determined boundary point; no test-based threshold search."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_fair_objective_screen as base
from scripts import run_private_recursive_query_diagnostic_v17 as prior
from privacy.fair_objective import require_mps,per_example
from privacy.recursive_public_calibration import largest_increment_ratio

NAME='private_recursive_frontier_diagnostic_v18'
OUT=ROOT/'results/ldp_gradient_far'/NAME
PROTOCOL=ROOT/'output/analysis/Private_Recursive_Query_V18_Public_Frontier_Protocol.md'
REPORT=ROOT/'output/analysis/Private_Recursive_Query_V18_Public_Frontier_Analyse.md'
TEST=ROOT/'tests/test_recursive_public_calibration_v18.py'


def inputs():
    old=json.loads((prior.OUT/'manifest.json').read_text());base.verify_stamp(old['source_stamp'])
    audit=ROOT/'output/analysis/Private_Recursive_Query_V17_Independent_Audit.json'
    assert json.loads(audit.read_text())['audit_passed']
    files=[Path(__file__),PROTOCOL,TEST,ROOT/'privacy/recursive_public_calibration.py',audit]
    files+=sorted(prior.OUT.glob('seed*__r*__client*.pt'))
    stamp=dict(old['source_stamp']);stamp.update({str(p.relative_to(ROOT)):base.digest(p) for p in files})
    return old['profile'],stamp


def report(stamp):
    rows=[json.loads(p.read_text()) for p in sorted(OUT.glob('seed*__r*__client*.json'))]
    assert all(r['source_stamp']==stamp and r['device']=='mps' for r in rows)
    public=largest_increment_ratio(C=2.);checks=[]
    lines=['# V18 — un seuil déterminé analytiquement','',f'**{len(rows)}/80 blocs MPS** ; mêmes critères que V17.', '',
        f"θ={public['theta']:.9f}, D={public['D']:.9f}, D/C={public['D_over_C']:.9f}.",
        f"Ratio de variance linéaire : stationnaire {public['stationary_ratio']:.9f}, au temps20 {public['finite_ratio_t20']:.9f}.", '',
        '| Seed | Blocs | Clip médian (%) | Blocs clip≤10 % (%) | Biais relatif médian | Biais relatif p90 | Critères locaux |',
        '|--:|--:|--:|--:|--:|--:|:--|']
    for seed in prior.SEEDS:
        rr=[r for r in rows if r['seed']==seed]
        if not rr:continue
        clip=[r['fraction_clipped'] for r in rr];bias=[r['coherent_bias_proxy_relative'] for r in rr]
        d=dict(seed=seed,n=len(rr),clip_median=st.median(clip),fraction_blocks_below_10pct_clip=st.mean(c<=.1 for c in clip),
               bias_median=st.median(bias),bias_p90=prior.quantile(bias,.9))
        d['passed']=d['n']==40 and d['fraction_blocks_below_10pct_clip']>=.9 and d['bias_median']<=.1 and d['bias_p90']<=.25
        checks.append(d)
        lines.append(f"| {seed} | {len(rr)} | {100*d['clip_median']:.2f} | {100*d['fraction_blocks_below_10pct_clip']:.1f} | {d['bias_median']:.4f} | {d['bias_p90']:.4f} | {d['passed']} |")
    admitted=len(rows)==80 and len(checks)==2 and all(c['passed'] for c in checks) and public['stationary_ratio']<=.5 and public['finite_ratio_t20']<=.55
    if len(rows)==80:
        assert {(r['seed'],r['replay'],r['client']) for r in rows}=={(s,p,c) for s in prior.SEEDS for p in range(4) for c in range(10)}
        base.save(OUT/'evidence.json',dict(public=public,checks=checks,admitted_to_calibration=admitted,source_stamp=stamp,global_validation=False))
        lines+=['',f"Admission à une calibration end-to-end : **{'PASS' if admitted else 'FAIL'}**."]
    lines+=['','Aucun seuil sélectionné sur l’accuracy de test ; aucun critère V17 relâché. '
        'Ceci reste un diagnostic sur deux états tardifs et quatre paires par état, non une validation sur toutes les phases d’entraînement. '
        'La robustesse et les métriques finales du modèle restent à établir.', '',
        '[Calcul public et protocole](Private_Recursive_Query_V18_Public_Frontier_Protocol.md).']
    REPORT.write_text('\n'.join(lines)+'\n');return len(rows),admitted


def worker():
    require_mps();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            profile,stamp=inputs();public=largest_increment_ratio(C=2.)
            manifest=dict(profile=profile,source_stamp=stamp,public=public,blocks=80)
            if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
            else:base.save(OUT/'manifest.json',manifest)
            if not (OUT/'tests.json').exists():
                p=subprocess.run([sys.executable,'-m','pytest',str(TEST),'-q'],cwd=ROOT,capture_output=True,text=True)
                base.save(OUT/'tests.json',dict(passed=p.returncode==0,output=p.stdout+p.stderr,source_stamp=stamp))
            test=json.loads((OUT/'tests.json').read_text());assert test['passed'] and test['source_stamp']==stamp
            for seed in prior.SEEDS:
                data=base.prepare(profile,seed);old=base.new_model(profile,seed);new=base.new_model(profile,seed)
                state=torch.load(prior.prior.population.SOURCE/f'seed{seed}__erm_mean'/'checkpoint.pt',map_location='cpu',weights_only=True)['model']
                old.load_state_dict(state)
                for replay in range(4):
                    step=torch.load(prior.prior.OUT/f'seed{seed}__r{replay}__C2_eta0.5__risk_rfa.pt',map_location='mps',weights_only=True)['step']
                    new.load_state_dict(state);base.apply_gradient(new,step,1.)
                    for cid,ids in enumerate(data['train']):
                        path=OUT/f'seed{seed}__r{replay}__client{cid}.json'
                        if path.exists():
                            assert json.loads(path.read_text())['source_stamp']==stamp;continue
                        base.verify_stamp(stamp)
                        source=prior.OUT/path.with_suffix('.pt').name
                        cache=torch.load(source,map_location='mps',weights_only=True);ix=cache['indices']
                        _,g0,_,_=per_example(old,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)
                        _,g1,_,_=per_example(new,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)
                        torch.testing.assert_close(g0.mean(0),cache['before_mean'],rtol=2e-5,atol=2e-6)
                        torch.testing.assert_close(g1.mean(0),cache['now_mean'],rtol=2e-5,atol=2e-6)
                        diff=g1-g0;norms=torch.linalg.vector_norm(diff,dim=1)
                        clipped=diff*torch.minimum(torch.ones_like(norms),public['D']/norms.clamp_min(1e-20))[:,None]
                        err=(clipped-diff).mean(0);bn=float(torch.linalg.vector_norm(err))
                        denom=max(float(torch.linalg.vector_norm(g1.mean(0))),1e-6)
                        vp=path.with_suffix('.pt')
                        base.checkpoint(vp,dict(error_mean=err.detach().cpu(),current_mean=g1.mean(0).detach().cpu(),
                            increment_norms=norms.detach().cpu(),source_stamp=stamp,privacy_protected=False))
                        base.save(path,dict(seed=seed,replay=replay,client=cid,source_stamp=stamp,device='mps',
                            source_vectors=str(source.relative_to(ROOT)),batch_hash=base.ids_hash(ix),public=public,
                            increment_bias_norm=bn,current_mean_norm=denom,fraction_clipped=float((norms>public['D']).float().mean()),
                            coherent_bias_proxy_relative=(1-public['theta'])/public['theta']*bn/denom,
                            test_evaluated=False,oracles_not_private=True,vector_sha256=base.digest(vp)))
                        count,admitted=report(stamp)
                        base.save(OUT/'status.json',dict(status='running',device='mps',valid_blocks=count,pid=os.getpid()))
                        print(f'{count}/80 public-frontier blocks complete',flush=True)
                del data,old,new;torch.mps.empty_cache()
            count,admitted=report(stamp);assert count==80
            base.save(OUT/'status.json',dict(status='completed',device='mps',valid_blocks=count,admitted_to_calibration=admitted,next_training_launched=False))
        except Exception as exc:
            base.save(OUT/'status.json',dict(status='failed',device='mps',error=repr(exc),pid=os.getpid()));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true',required=True)
    p.add_argument('--resume',action='store_true',required=True);p.parse_args();worker()


if __name__=='__main__':main()
