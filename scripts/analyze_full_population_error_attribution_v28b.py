#!/usr/bin/env python3
"""Read-only MPS model diagnostic on independently audited V28 checkpoints."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_fair_objective_screen as base
from scripts.analyze_private_risk_confirmation_v12 import verify
from privacy.fair_objective import require_mps,per_example,losses
from privacy.full_population_error_attribution_v28b import decompose,NAMES

SOURCE=ROOT/'results/ldp_gradient_far/full_population_private_risk_calibration_v28'
AUDIT=ROOT/'output/analysis/audit_full_population_private_risk_v28'
OUT=ROOT/'results/ldp_gradient_far/full_population_error_attribution_v28b'
DEST=ROOT/'output/analysis/Full_Population_Error_Attribution_V28b_Analyse'
PROTOCOL=ROOT/'output/analysis/Full_Population_Error_Attribution_V28b_Protocol.md'
TEST=ROOT/'tests/test_full_population_error_attribution_v28b.py'
METHODS=('erm_mean','erm_rfa','risk_mean','risk_rfa');SEEDS=(170501,170502)


def identifier(job):return f"seed{job['seed']}__{job['method']}"


def objective(risks,kind):
    risks=risks.detach()  # This helper reports values; the chain-rule test differentiates separately.
    if kind.startswith('erm_'):return float(risks.mean())
    return float((risks+torch.where(risks<=.5,2*risks.square(),2*risks-.5)).mean())


def source_evidence(job):
    d=SOURCE/identifier(job);cache=AUDIT/(identifier(job)+'.json')
    if not cache.exists():return None
    audit=json.loads(cache.read_text());s=json.loads((d/'orchestration_status.json').read_text())
    assert s['status']=='completed' and s['device']=='mps' and s['round']==120
    assert audit['record']['job']==job and audit['record']['replay']['final_model_bitwise']
    assert audit['record']['replay']['final_update_bitwise'] and audit['record']['replay']['validation_exact']
    signature=audit['cache_signature']
    assert signature['manifest']==base.digest(SOURCE/'manifest.json')
    assert signature['analysis']==base.digest(ROOT/'scripts/analyze_full_population_private_risk_v28.py')
    assert signature['test']==base.digest(ROOT/'tests/test_full_population_private_risk_v28_audit.py')
    paths=[d/name for name in signature['files']]+[cache]
    for name,sha in signature['files'].items():assert base.digest(d/name)==sha
    return {str(p.relative_to(ROOT)):base.digest(p) for p in paths}


def analyze(job,profile,data,stamp,inputs):
    out=OUT/identifier(job);result_file=out/'diagnostic.json'
    if result_file.exists():
        r=json.loads(result_file.read_text())
        assert r['source_stamp']==stamp and r['input_stamp']==inputs
        assert r['vectors_sha256']==base.digest(out/'oracle_vectors.pt')
        return r
    cp=torch.load(SOURCE/identifier(job)/'checkpoint.pt',map_location='cpu',weights_only=True)
    model=base.new_model(profile,job['seed']);model.load_state_dict(cp['pre_round_model'])
    before_state={k:v.clone() for k,v in model.state_dict().items()}
    before=base.evaluate(model,data,'val');verify(before)
    raw=[];clipped=[];risks=[];client_stats=[]
    # Dataset order, not the runner's random accumulation order: both include all records.
    for cid,ids in enumerate(data['train']):
        d=sum(p.numel() for p in model.parameters() if p.requires_grad)
        g=torch.zeros(d,device='mps');c=torch.zeros_like(g);risk=torch.zeros((),device='mps');nclip=0
        for batch in ids.split(120):
            rr,rows,norms,raw_mean=per_example(model,data['x'][batch],data['y'][batch],clip_norm=2.)
            g+=raw_mean*(len(batch)/4800);c+=rows.sum(0)/4800;risk+=rr.sum()/4800
            nclip+=int((norms>2.).sum())
        torch.testing.assert_close(c,cp['last_clean_means'][cid].to('mps'),rtol=6e-5,atol=4e-7)
        original=cp['oracles'][-1]['clients'][cid]['gradient']
        assert original['population_size']==4800 and nclip==original['per_example_clipped_count']
        assert math.isclose(float(risk),original['raw_population_brier_risk'],rel_tol=3e-5,abs_tol=3e-7)
        norm=float(torch.linalg.vector_norm(g));bias=float(torch.linalg.vector_norm(c-g))
        raw.append(g);clipped.append(c);risks.append(risk)
        client_stats.append(dict(client=cid,raw_gradient_norm=norm,clipped_mean_norm=float(torch.linalg.vector_norm(c)),
            clipping_bias_norm=bias,relative_clipping_bias=None if norm<=1e-12 else bias/norm,
            clipping_fraction=nclip/4800,risk_before=float(risk),validation_accuracy_before=100*before['clients'][cid]['accuracy']))
    for name,value in model.state_dict().items():assert torch.equal(value,before_state[name])
    raw=torch.stack(raw);clipped=torch.stack(clipped);risks=torch.stack(risks)
    agg=cp['rows'][-1]['aggregation'];eta=agg['eta'];assert eta==.5
    received=torch.tensor(agg['objective_weights'],device='mps')
    if job['method'].startswith('risk_'):
        coeff=1+2*(risks/.5).clamp(max=1);oracle=coeff/coeff.sum();scale=float(coeff.mean())
    else:oracle=torch.ones(10,device='mps')/10;scale=1.
    effective=received if agg['solver'] is None else torch.tensor(agg['solver']['stationary_weights'],device='mps')
    messages=cp['last_private_messages'].to('mps');applied=cp['last_step'].to('mps')/eta
    decomposition,target,components=decompose(raw,clipped,messages,applied,oracle,received,effective,objective_scale=scale,eta=eta)
    model.load_state_dict(cp['model']);after=base.evaluate(model,data,'val');verify(after)
    assert after==cp['rows'][-1]['validation']
    risks_after=[]
    with torch.no_grad():
        for ids in data['train']:
            r=torch.zeros((),device='mps')
            for batch in ids.split(256):r+=losses(model(data['x'][batch]),data['y'][batch],'brier').sum()/4800
            risks_after.append(r)
    risks_after=torch.stack(risks_after)
    j_before=objective(risks,job['method']);j_after=objective(risks_after,job['method']);actual_gain=j_before-j_after
    for cid,c in enumerate(client_stats):
        c.update(oracle_weight=float(oracle[cid]),received_weight=float(received[cid]),effective_weight=float(effective[cid]),
                 risk_after=float(risks_after[cid]),validation_accuracy_after=100*after['clients'][cid]['accuracy'])
    state=dict(raw_gradients=raw.cpu(),clipped_means=clipped.cpu(),messages=messages.cpu(),applied=applied.cpu(),
               oracle_weights=oracle.cpu(),received_weights=received.cpu(),effective_weights=effective.cpu(),
               target=target.cpu(),components={k:v.cpu() for k,v in components.items()},
               risks_before=risks.cpu(),risks_after=risks_after.cpu(),objective_scale=scale,eta=eta,
               privacy_protected=False,feeds_mechanism=False,source_stamp=stamp,input_stamp=inputs)
    base.checkpoint(out/'oracle_vectors.pt',state)
    result=dict(job=job,round=120,device='mps',diagnostic_only=True,privacy_protected=False,feeds_mechanism=False,
        test_evaluated=False,global_validation=False,source_stamp=stamp,input_stamp=inputs,
        vectors_sha256=base.digest(out/'oracle_vectors.pt'),decomposition=decomposition,clients=client_stats,
        objective_name='mean_half_brier' if job['method'].startswith('erm_') else 'mean_capped_fair_potential_of_population_half_brier',
        J_before=j_before,J_after=j_after,actual_objective_gain=actual_gain,
        finite_step_remainder=decomposition['applied_predicted_gain']-actual_gain,
        validation_before=before,validation_after=after,
        actual_validation_delta={k:after[k]-before[k] for k in ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2','ce_loss','brier_loss')})
    base.verify_stamp(stamp);base.verify_stamp(inputs);base.save(result_file,result)
    print(f"V28b {identifier(job)}: vector/squared/directional identities verified; actual J gain={actual_gain:+.7g}",flush=True)
    del model;torch.mps.empty_cache()
    return result


def write(records,stamp):
    suffix='' if len(records)==8 else '_Partial';dest=DEST.with_name(DEST.name+suffix)
    base.save(dest.with_suffix('.json'),dict(valid_states=len(records),expected_states=8,independent_seeds=2,
        source_stamp=stamp,records=records,global_validation=False,gate_changed=False,attacks_launched=False))
    lines=['# V28b — de l’erreur du gradient au pas réellement appliqué','',
        f'**{len(records)}/8 états audités analysés. Diagnostic descriptif, pas de validation supplémentaire.**','',
        'Chaque ligne concerne le dernier pas d’une trajectoire, au tour 120. Le test reste fermé. '
        'Les méthodes partent d’états différents : les écarts entre lignes ne sont pas des effets causaux appariés à modèle identique.','',
        '## 1. Erreur au carré par rapport à une cible non clippée','',
        'La cible h* utilise les gradients bruts de population et les risques exacts pour les poids ; '
        'elle est un oracle, jamais un message privé admissible. Toutes les composantes sont dans l’espace des paramètres.', '',
        '| Seed | Méthode | Erreur² totale | Clipping² | Poids privés² | RFA² | Bruit² | Numérique² | Termes croisés |',
        '|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|']
    for r in records:
        d=r['decomposition'];vals=[d['squared_error'],*[d['squared_components'][k] for k in NAMES],sum(d['doubled_cross_products'].values())]
        lines.append(f"| {r['job']['seed']} | {r['job']['method']} | "+' | '.join(f'{v:.6g}' for v in vals)+' |')
    lines+=['','## 2. Contributions à la perte du gain prédit au premier ordre','',
        'On compare η〈∇J,h*〉 à η〈∇J,A〉. Pour chaque composante e, le nombre présenté est −η〈∇J,e〉. '
        '**Positif : le gain prédit diminue. Négatif : il augmente au premier ordre.** '
        'Ce signe ne suffit pas à prédire le gain réel à pas fini.', '',
        '| Seed | Méthode | Gain idéal prédit | Gain appliqué prédit | Clipping | Poids privés | RFA | Bruit | Numérique |',
        '|--:|:--|--:|--:|--:|--:|--:|--:|--:|']
    for r in records:
        d=r['decomposition'];vals=[d['ideal_predicted_gain'],d['applied_predicted_gain'],*[d['first_order_gain_loss'][k] for k in NAMES]]
        lines.append(f"| {r['job']['seed']} | {r['job']['method']} | "+' | '.join(f'{v:+.6g}' for v in vals)+' |')
    lines+=['','## 3. Ce qui s’est réellement passé sur le dernier pas','',
        '| Seed | Méthode | Gain J prédit | Gain J réel | Reste exact | Δ accuracy validation (pp) | Δ Worst-20 (pp) |',
        '|--:|:--|--:|--:|--:|--:|--:|']
    for r in records:
        vals=[r['decomposition']['applied_predicted_gain'],r['actual_objective_gain'],r['finite_step_remainder'],
              r['actual_validation_delta']['accuracy_pct'],r['actual_validation_delta']['worst20_pct']]
        lines.append(f"| {r['job']['seed']} | {r['job']['method']} | "+' | '.join(f'{v:+.6g}' for v in vals)+' |')
    lines+=['','J est la moyenne demi-Brier pour ERM et le potentiel équitable des risques demi-Brier pour « risk ». '
        'Leurs valeurs ne sont donc pas interchangeables. Un gain J positif est une baisse de cet objectif, '
        'pas une preuve de gain d’accuracy ou de Worst-20. Le reste peut être négatif ; ce n’est pas une mesure de lissité globale.']
    for r in records:
        lines+=['',f"## Clients — {identifier(r['job'])}",'',
            '| Client | Risque avant | Acc. avant (%) | Acc. après (%) | Clip (%) | λ oracle | λ reçu | ν effectif | Biais de clip / norme brute |',
            '|--:|--:|--:|--:|--:|--:|--:|--:|--:|']
        for c in r['clients']:
            ratio='non défini' if c['relative_clipping_bias'] is None else f"{c['relative_clipping_bias']:.4f}"
            lines.append(f"| {c['client']} | {c['risk_before']:.4f} | {c['validation_accuracy_before']:.3f} | {c['validation_accuracy_after']:.3f} | {100*c['clipping_fraction']:.2f} | {c['oracle_weight']:.4f} | {c['received_weight']:.4f} | {c['effective_weight']:.4f} | {ratio} |")
    lines+=['','Un ratio de biais supérieur à un est possible : le clipping individuel modifie les annulations '
        'avant la moyenne. Le taux de clipping seul ne mesure donc ni l’amplitude ni la direction du biais. '
        'Aucune borne ou hypothèse d’indépendance des erreurs n’est ajoutée. Ces résultats ne changent pas '
        'les critères finaux V28 et ne déclenchent pas de nouvelle grille.', '',
        '[Définitions, identités et portée](Full_Population_Error_Attribution_V28b_Protocol.md) · '
        f'[Evidence numérique]({dest.name}.json).']
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def main(partial):
    require_mps();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'diagnostic.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original=json.loads((SOURCE/'manifest.json').read_text());stamp=dict(original['source_stamp']);base.verify_stamp(stamp)
        for path in (Path(__file__),PROTOCOL,TEST,ROOT/'privacy/full_population_error_attribution_v28b.py',
                     ROOT/'scripts/analyze_full_population_private_risk_v28.py',ROOT/'tests/test_full_population_private_risk_v28_audit.py',SOURCE/'manifest.json'):
            stamp[str(path.relative_to(ROOT))]=base.digest(path)
        manifest=dict(source_stamp=stamp,methods=list(METHODS),seeds=list(SEEDS),state='pre120 and post120',
                      expected_states=8,diagnostic_only=True,privacy_protected=False,automatic_next=False)
        if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
        else:base.save(OUT/'manifest.json',manifest)
        if not (OUT/'tests.json').exists():
            proc=subprocess.run([sys.executable,'-m','pytest',str(TEST),'-q'],cwd=ROOT,capture_output=True,text=True)
            base.save(OUT/'tests.json',dict(passed=proc.returncode==0,output=proc.stdout+proc.stderr,source_stamp=stamp))
        tests=json.loads((OUT/'tests.json').read_text());assert tests['passed'] and tests['source_stamp']==stamp
        jobs=[]
        for seed in SEEDS:
            for method in METHODS:
                job=dict(seed=seed,method=method);inputs=source_evidence(job)
                if inputs is not None:jobs.append((job,inputs))
        if not partial:assert len(jobs)==8,'Eight previously audited completed runs required'
        records=[]
        for seed in SEEDS:
            selected=[item for item in jobs if item[0]['seed']==seed]
            if not selected:continue
            data=base.prepare(original['profile'],seed)
            for job,inputs in selected:records.append(analyze(job,original['profile'],data,stamp,inputs))
            del data;torch.mps.empty_cache()
        write(records,stamp);base.verify_stamp(stamp)
        base.save(OUT/'status.json',dict(status='completed' if len(records)==8 else 'partial',valid_states=len(records),
                  expected_states=8,device='mps',global_validation=False,gate_changed=False,source_stamp=stamp))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',required=True,action='store_true');p.add_argument('--partial',action='store_true')
    main(p.parse_args().partial)
