#!/usr/bin/env python3
"""Independent V28b gradient/attribution audit, MPS only, no training."""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import statistics as st
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_fair_objective_screen as base
from privacy.fair_objective import require_mps

SOURCE=ROOT/'results/ldp_gradient_far/full_population_private_risk_calibration_v28'
DIAG=ROOT/'results/ldp_gradient_far/full_population_error_attribution_v28b'
DEST=ROOT/'output/analysis/Full_Population_Error_Attribution_V28b_Independent_Audit'
CACHE=ROOT/'output/analysis/audit_full_population_error_attribution_v28b'
TEST=ROOT/'tests/test_full_population_error_attribution_v28b_independent_audit.py'
METHODS=('erm_mean','erm_rfa','risk_mean','risk_rfa');SEEDS=(170501,170502)
NAMES=('clipping','private_risk_weights','rfa_geometry','effective_noise','numerical_solver')


def polynomial_brier(logits,y):
    # Equivalent to half the squared distance to one-hot y; independent expression.
    p=logits.softmax(1)
    return (p.square().sum(1)-2*p.gather(1,y[:,None]).squeeze(1)+1)/2


def population(model,x,y,block_size=256):
    require_mps()
    assert x.device.type==y.device.type=='mps' and not model.training
    params=[p for p in model.parameters() if p.requires_grad]
    assert all(p.device.type=='mps' for p in params)
    gradient=torch.zeros(sum(p.numel() for p in params),device='mps')
    risk=torch.zeros((),device='mps')
    for start in range(0,len(x),block_size):
        batch_risk=polynomial_brier(model(x[start:start+block_size]),y[start:start+block_size]).sum()/len(x)
        grads=torch.autograd.grad(batch_risk,params)
        gradient+=torch.cat([g.flatten() for g in grads]).detach()
        risk+=batch_risk.detach()
    return risk,gradient


def close(actual,expected):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert math.isclose(actual,expected,rel_tol=5e-5,abs_tol=4e-7),(actual,expected)


def one(job,profile,data,audit_stamp):
    name=f"seed{job['seed']}__{job['method']}";folder=DIAG/name
    r=json.loads((folder/'diagnostic.json').read_text());base.verify_stamp(r['source_stamp']);base.verify_stamp(r['input_stamp'])
    assert r['job']==job and r['round']==120 and not r['test_evaluated'] and r['diagnostic_only']
    assert r['vectors_sha256']==base.digest(folder/'oracle_vectors.pt')
    inputs={str((folder/f).relative_to(ROOT)):base.digest(folder/f) for f in ('diagnostic.json','oracle_vectors.pt')}
    signature=dict(audit_stamp=audit_stamp,inputs=inputs)
    path=CACHE/(name+'.json')
    if path.exists():
        old=json.loads(path.read_text())
        if old['signature']==signature:return old['result']
    saved=torch.load(folder/'oracle_vectors.pt',map_location='cpu',weights_only=True)
    cp=torch.load(SOURCE/name/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert saved['source_stamp']==r['source_stamp'] and saved['input_stamp']==r['input_stamp']
    model=base.new_model(profile,job['seed']);model.load_state_dict(cp['pre_round_model'])
    state={k:v.clone() for k,v in model.state_dict().items()};raw=[];risks=[]
    for ids in data['train']:
        risk,g=population(model,data['x'][ids],data['y'][ids]);risks.append(risk);raw.append(g)
    raw=torch.stack(raw);risks=torch.stack(risks)
    expected_raw=saved['raw_gradients'].to('mps')
    torch.testing.assert_close(raw,expected_raw,rtol=5e-4,atol=3e-7)
    torch.testing.assert_close(risks,saved['risks_before'].to('mps'),rtol=3e-5,atol=3e-7)
    for k,v in model.state_dict().items():assert torch.equal(v,state[k])
    kind=job['method'];fair=kind.startswith('risk_')
    if fair:
        a=1+4*risks.clamp(max=.5);oracle=a/a.sum();scale=float(a.mean())
    else:oracle=torch.ones(10,device='mps')/10;scale=1.
    clipped=saved['clipped_means'].to('mps');X=saved['messages'].to('mps');A=saved['applied'].to('mps')
    lam=saved['received_weights'].to('mps');nu=saved['effective_weights'].to('mps');eta=.5
    torch.testing.assert_close(oracle,saved['oracle_weights'].to('mps'),rtol=3e-5,atol=3e-7)
    h=torch.einsum('i,ij->j',oracle,raw);G=scale*h
    comp=dict(zip(NAMES,(
        torch.einsum('i,ij->j',oracle,clipped-raw),
        torch.einsum('i,ij->j',lam-oracle,clipped),
        torch.einsum('i,ij->j',nu-lam,clipped),
        torch.einsum('i,ij->j',nu,X-clipped),
        A-torch.einsum('i,ij->j',nu,X))))
    torch.testing.assert_close(A-h,sum(comp.values()),rtol=6e-5,atol=4e-7)
    dec=r['decomposition'];squares={k:float(v.square().sum()) for k,v in comp.items()}
    cross={a+'__'+b:float(2*torch.dot(comp[a],comp[b])) for a,b in itertools.combinations(NAMES,2)}
    costs={k:float(-eta*torch.dot(G,v)) for k,v in comp.items()}
    for k in NAMES:
        close(squares[k],dec['squared_components'][k]);close(costs[k],dec['first_order_gain_loss'][k])
        torch.testing.assert_close(comp[k],saved['components'][k].to('mps'),rtol=6e-4,atol=6e-7)
    for k,v in cross.items():close(v,dec['doubled_cross_products'][k])
    square=float((A-h).square().sum());ideal=float(eta*torch.dot(G,h));actual_pred=float(eta*torch.dot(G,A))
    close(square,sum(squares.values())+sum(cross.values()));close(square,dec['squared_error'])
    close(ideal,dec['ideal_predicted_gain']);close(actual_pred,dec['applied_predicted_gain'])
    close(ideal-actual_pred,sum(costs.values()))
    model.load_state_dict(cp['model']);after=[]
    with torch.no_grad():
        for ids in data['train']:
            risk=torch.zeros((),device='mps')
            for batch in ids.split(256):risk+=polynomial_brier(model(data['x'][batch]),data['y'][batch]).sum()/4800
            after.append(risk)
    after=torch.stack(after)
    torch.testing.assert_close(after,saved['risks_after'].to('mps'),rtol=3e-5,atol=3e-7)
    def J(values):
        scalars=values.cpu().tolist()  # Reporting arithmetic only; all model/gradient work was MPS.
        return st.mean(x+(2*x*x if x<=.5 else 2*x-.5) if fair else x for x in scalars)
    before_J=J(risks);after_J=J(after);gain=before_J-after_J
    close(before_J,r['J_before']);close(after_J,r['J_after']);close(gain,r['actual_objective_gain'])
    close(actual_pred-gain,r['finite_step_remainder'])
    result=dict(job=job,audit_passed=True,device='mps',state='pre120 to post120',
        raw_gradient_max_absolute_error=float((raw-expected_raw).abs().max()),
        raw_gradient_relative_l2_error=float(torch.linalg.vector_norm(raw-expected_raw)/torch.linalg.vector_norm(expected_raw)),
        squared_error=square,squared_components=squares,first_order_gain_loss=costs,
        ideal_predicted_gain=ideal,applied_predicted_gain=actual_pred,actual_objective_gain=gain,
        finite_step_remainder=actual_pred-gain,
        effective_noise_square_ratio=None if square==0 else squares['effective_noise']/square,
        clipping_fraction_of_ideal_predicted_gain=None if ideal<=1e-12 else costs['clipping']/ideal,
        validation_delta=r['actual_validation_delta'],global_validation=False,private_oracle=True)
    base.save(path,dict(signature=signature,result=result));del model;torch.mps.empty_cache()
    print(f"V28b independent audit {name}: batched polynomial-Brier gradient and all identities PASS",flush=True)
    return result


def main(partial):
    require_mps();manifest=json.loads((DIAG/'manifest.json').read_text());base.verify_stamp(manifest['source_stamp'])
    tests=json.loads((DIAG/'tests.json').read_text());assert tests['passed'] and tests['source_stamp']==manifest['source_stamp']
    stamp={str(p.relative_to(ROOT)):base.digest(p) for p in (Path(__file__),TEST)}
    test_path=CACHE/'tests.json'
    previous=json.loads(test_path.read_text()) if test_path.exists() else None
    if previous is None or previous['stamp']!=stamp:
        proc=subprocess.run([sys.executable,'-m','pytest',str(TEST),'-q'],cwd=ROOT,capture_output=True,text=True)
        previous=dict(passed=proc.returncode==0,output=proc.stdout+proc.stderr,stamp=stamp);base.save(test_path,previous)
    assert previous['passed'],previous['output']
    profile=json.loads((SOURCE/'manifest.json').read_text())['profile'];rows=[]
    for seed in SEEDS:
        jobs=[dict(seed=seed,method=m) for m in METHODS if (DIAG/f'seed{seed}__{m}'/'diagnostic.json').exists()]
        if not jobs:continue
        data=base.prepare(profile,seed)
        for j in jobs:rows.append(one(j,profile,data,stamp))
        del data;torch.mps.empty_cache()
    if not partial:assert len(rows)==8
    dest=DEST.with_name(DEST.name+('' if len(rows)==8 else '_Partial'))
    base.save(dest.with_suffix('.json'),dict(audit_passed=True,valid_states=len(rows),expected_states=8,
        rows=rows,source_stamp=stamp,parent_manifest_sha256=base.digest(DIAG/'manifest.json'),global_validation=False))
    lines=['# V28b — vérification indépendante des gradients et de l’attribution','',
        f'**{len(rows)}/8 états vérifiés sur MPS.** Pas de validation globale et aucun changement du gate V28.', '',
        'Le gradient brut est recalculé par différentiation d’une loss de batch complet accumulée en blocs de 256, '
        'sans vmap de gradients individuels et avec une autre expression algébrique du demi-Brier. '
        'Les cinq composantes, les dix termes croisés et leurs projections sont ensuite reconstruits.', '',
        '| Seed | Méthode | Écart maximal gradient | Bruit² / erreur² | Clipping / gain idéal prédit | Gain J réel | Δ accuracy (pp) | Δ Worst-20 (pp) |',
        '|--:|:--|--:|--:|--:|--:|--:|--:|']
    for r in rows:
        ratios=['non défini' if r[k] is None else f'{100*r[k]:.3f} %' for k in ('effective_noise_square_ratio','clipping_fraction_of_ideal_predicted_gain')]
        lines.append(f"| {r['job']['seed']} | {r['job']['method']} | {r['raw_gradient_max_absolute_error']:.3g} | "+' | '.join(ratios)+f" | {r['actual_objective_gain']:+.6g} | {r['validation_delta']['accuracy_pct']:+.4f} | {r['validation_delta']['worst20_pct']:+.4f} |")
    lines+=['','Les deux ratios ont des dénominateurs et des unités conceptuelles différents. '
        'Le premier décrit la géométrie euclidienne de l’erreur réalisée ; le second décrit une réduction '
        'du gain de loss **au premier ordre**. À cause des termes croisés, le premier n’est pas une décomposition '
        'en parts positives devant sommer à 100 %. Ni l’un ni l’autre ne mesure directement un coût en accuracy.', '',
        '[Protocole et identités](Full_Population_Error_Attribution_V28b_Protocol.md) · '
        f'[Audit numérique]({dest.name}.json).']
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');main(p.parse_args().partial)
