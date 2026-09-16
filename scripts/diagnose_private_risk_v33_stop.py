"""Reproduce the stopped V33 step without applying it or resuming training."""
import json
import os
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_fair_objective_screen as base
from scripts.run_private_risk_continuation_v33 import attack,aggregate,model_hash
from privacy.fair_objective import require_mps
from privacy.full_population_private_risk_v28 import private_population_gradient
from privacy.split_risk_gradient import private_risk
from privacy.capped_private_risk import weights
from privacy.stable_weighted_rfa import weighted_rfa,stable_norm

OUT=ROOT/'results/ldp_gradient_far/private_risk_continuation_v33'
DEST=ROOT/'output/analysis/Private_Risk_V33_Stop_Diagnosis'
VECTORS=ROOT/'results/ldp_gradient_far/private_risk_v33_stop_diagnosis/vectors.pt'

def main():
    require_mps();manifest=json.loads((OUT/'manifest.json').read_text());base.verify_stamp(manifest['stamp'])
    assert json.loads((OUT/'status.json').read_text())['status']=='failed'
    folder=OUT/'seed170502__slow_ipm__risk_rfa'
    status=json.loads((folder/'orchestration_status.json').read_text());assert status['step']==5
    cp=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert [r['step'] for r in cp['rows']]==[1,2,3,4]
    assert cp['signature']['source_stamp']==manifest['stamp']
    parent=ROOT/'results/ldp_gradient_far/full_population_private_risk_calibration_v28/manifest.json'
    profile=json.loads(parent.read_text())['profile'];plan=manifest['privacy_parent']
    protected_paths=[OUT/'status.json',OUT/'failure.json',folder/'checkpoint.pt',folder/'orchestration_status.json',
        OUT/'manifest.json',Path(__file__)]
    hashes={str(p.relative_to(ROOT)):base.digest(p) for p in protected_paths}
    seed,k=170502,5
    data=base.prepare(profile,seed);model=base.new_model(profile,seed);model.load_state_dict(cp['model'])
    before=model_hash(model);assert before==cp['rows'][-1]['model_hash']
    key=json.loads((ROOT/'results/ldp_gradient_far/recursive_private_risk_calibration_v19/simulator_secret.json').read_text())['key']
    sent=[];reports=[];clean=[]
    for cid,ids in enumerate(data['train']):
        rr,_=private_risk(model,data['x'][ids],data['y'][ids],noise_std=plan['risk_std'],
            seed=base.seed_for(key,'v33',seed,k,cid,'risk'),N=4800)
        ix=base.draw_indices(4800,4800,base.seed_for(key,'v33',seed,k,cid,'batch'))
        msg,query,diag=private_population_gradient(model,data['x'][ids[ix]],data['y'][ids[ix]],C=2.,block_size=240,
            noise_std=plan['gradient_std'],seed=base.seed_for(key,'v33',seed,k,cid,'gaussian'))
        sent.append(msg);reports.append(rr);clean.append(query)
    assert model_hash(model)==before
    raw,r0=torch.stack(sent),torch.stack(reports);x,r,info=attack(raw,r0,'slow_ipm',5)
    captured=None
    try:aggregate(x,r,'risk_rfa')
    except AssertionError:captured=traceback.format_exc()
    assert captured is not None,'Original failing assertion was not reproduced; do not assume a solver failure'
    w,_=weights(r,.5);points={};rows=[]
    # Fixed diagnostic iteration counts, not applied updates or changed V33 settings.
    for n in (40,80,160,320):
        A,di=weighted_rfa(x,w,iterations=n)
        distances=stable_norm(x-A,dim=1)
        objective=float((w*distances).sum())
        points[n]=A
        rows.append(dict(iterations=n,objective=objective,solver=di,applied_to_model=False))
    ref=rows[-1]['objective']
    for row in rows:
        row['objective_above_320_iter']=row['objective']-ref
        row['distance_to_320_iter']=float(stable_norm(points[row['iterations']]-points[320]))
    _,clean_diag,_=aggregate(raw,r0,'risk_rfa')
    assert rows[0]['solver']['unsmoothed_objective_gap_upper']>.001
    # Current-step honest messages must match the clean RFA branch: identical pre-attack state/noise.
    clean_rec=json.loads((OUT/'seed170502__none__risk_rfa/metrics.json').read_text())
    assert clean_rec['rows'][3]['model_hash']==before
    for cid,msg in enumerate(sent):
        assert base.ids_hash(msg)==clean_rec['rows'][4]['clients'][cid]['private_message_hash']
    assert model_hash(model)==before
    VECTORS.parent.mkdir(parents=True,exist_ok=True)
    base.checkpoint(VECTORS,dict(raw_messages=raw.cpu(),raw_reports=r0.cpu(),attacked_messages=x.cpu(),
        attacked_reports=r.cpu(),clean_means=torch.stack(clean).cpu(),points={n:a.cpu() for n,a in points.items()},
        source_stamp=manifest['stamp'],seed=seed,step=5,privacy_protected=False))
    base.verify_stamp(manifest['stamp'])
    for p,h in hashes.items():assert base.digest(ROOT/p)==h
    result=dict(reproduced=True,device='mps',seed=seed,step=5,method='risk_rfa',attack=info,
        failing_line='scripts/run_private_risk_continuation_v33.py:56',traceback=captured,
        numerical_threshold=.001,iteration_diagnostics=rows,clean_solver=clean_diag,
        original_honest_messages_match_clean_branch=True,no_update_applied=True,no_training_resumed=True,
        V33_modified=False,global_validation=False,inputs=hashes,source_stamp=manifest['stamp'],
        vectors_sha256=base.digest(VECTORS),note='Higher iterations are diagnostic only; original40-iteration protocol remains failed.')
    base.save(DEST.with_suffix('.json'),result)
    lines=['# V33 — arrêt numérique reproduit sur MPS','',
        'La campagne est arrêtée :22 trajectoires terminées, une interrompue et une non lancée. '
        'Le pas5 de `seed170502__slow_ipm__risk_rfa` est reproduit à partir de son checkpoint '
        'après le pas4. Aucun nouveau pas n’est appliqué au modèle.', '',
        'La borne enregistrée sur la sous-optimalité du solveur après40 itérations dépasse '
        'le seuil fixé de0,001. L’assertion de la ligne56 arrête donc correctement le lanceur. '
        'Les dix messages privés reconstruits concordent exactement avec ceux de la branche '
        'RFA propre au même pas. Le défaut n’est pas un changement de données ou de tirage.', '',
        '| Itérations diagnostiques | Borne de sous-optimalité | Objectif non lissé | Objectif moins celui à320 | Distance à la sortie à320 |',
        '|--:|--:|--:|--:|--:|']
    for row in rows:
        lines.append(f"| {row['iterations']} | {row['solver']['unsmoothed_objective_gap_upper']:.9f} | {row['objective']:.9f} | {row['objective_above_320_iter']:.9f} | {row['distance_to_320_iter']:.9f} |")
    lines+=['','Une borne supérieure trop grande ne démontre pas que l’erreur réelle dépasse '
        'le seuil ; elle signifie que ce certificat ne justifie pas la précision requise. '
        'La sortie à320 est un témoin numérique, pas un oracle exact. Les itérations supplémentaires '
        'servent au diagnostic ; elles ne remplacent pas rétroactivement le solveur40 de V33.', '',
        'La continuation de V33 n’est pas autorisée par ce diagnostic. Le premier échec '
        'd’utilité sous Bit-Flip reste conservé indépendamment de cet arrêt numérique. '
        'Il faut auditer les22 trajectoires comme une campagne partielle, avec les deux '
        'cas manquants explicitement indiqués.', '',
        '[Trace, valeurs et empreintes](Private_Risk_V33_Stop_Diagnosis.json)']
    DEST.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(reproduced=True,rows=rows,global_validation=False)))

if __name__=='__main__':main()
