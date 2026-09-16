"""Independent V34 counts/contrasts and actual-model directional differences."""
import fcntl
import json
import math
import os
from pathlib import Path
import statistics as st
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_fair_objective_screen as base
from scripts.audit_private_risk_continuation_v33 import exact
from privacy.fair_objective import require_mps
from privacy.stable_weighted_rfa import weighted_rfa

SOURCE=ROOT/'results/ldp_gradient_far/private_risk_alie_model_v34'
V33=ROOT/'results/ldp_gradient_far/private_risk_continuation_v33'
OUT=ROOT/'results/ldp_gradient_far/private_risk_v34_direction_audit'
DEST=ROOT/'output/analysis/Private_Risk_V34_Direction_Step_Audit'
DOC=ROOT/'output/analysis/Private_Risk_V34_Direction_Step_Protocol.md'
ETAS=(0.,1/32,1/16,1/8,1/4,1/2)


def vals(ev):
    values={k:float(v) for k,v in exact(ev,range(2,10)).items()}
    for k in ('ce_loss','brier_loss'):
        values[k]=st.mean(c[k] for c in ev['clients'][2:])
    risks=[c['brier_loss'] for c in ev['clients'][2:]]
    values['J_val']=st.mean(r+2*r*r if r<=.5 else 3*r-.5 for r in risks)
    return values


def run():
    require_mps()
    src=json.loads((SOURCE/'results.json').read_text())
    status=json.loads((SOURCE/'status.json').read_text())
    assert status['status']=='completed' and status['results_sha256']==base.digest(SOURCE/'results.json')
    assert src['evaluated']==src['attempted']==24 and src['device']=='mps'
    stamp=dict(src['signature']['source_stamp']);base.verify_stamp(stamp)
    for p in (Path(__file__),DOC,SOURCE/'results.json',SOURCE/'status.json'):
        stamp[str(p.relative_to(ROOT))]=base.digest(p)
    if (OUT/'results.json').exists():
        old=json.loads((OUT/'results.json').read_text());assert old['source_stamp']==stamp
        return old
    index={}
    for r in src['rows']:
        assert r['evaluated'] and r['diagnostics']['solver_gap']<=.001
        k=(r['seed'],r['method'],r['cell']);assert k not in index;index[k]=r
        for key,value in vals(r['evaluation']).items():
            assert math.isclose(value,r['honest'][key],rel_tol=1e-10,abs_tol=1e-10)
        hs=r['honest'];assert math.isclose(hs['actual_delta_J']-hs['first_order_delta_J'],hs['remainder_J'],abs_tol=1e-12)
    assert len(index)==24
    for c in src['contrasts']:
        for metric,record in c['metrics'].items():
            v={cell:index[c['seed'],c['method'],cell]['honest'][metric] for cell in ('00','10','01','11')}
            expected=dict(gradient_original_report=v['10']-v['00'],report_original_gradient=v['01']-v['00'],
                report_forged_gradient=v['11']-v['10'],gradient_forged_report=v['11']-v['01'],
                interaction=(v['11']-v['00'])-(v['10']-v['00'])-(v['01']-v['00']))
            for key,value in expected.items():assert math.isclose(value,record[key],abs_tol=1e-10)
    profile=json.loads((ROOT/'results/ldp_gradient_far/full_population_private_risk_calibration_v28/manifest.json').read_text())['profile']
    points,derivatives=[],[]
    for seed in (170501,170502):
        cp=torch.load(V33/f'seed{seed}__persistent_alie__risk_winsor/checkpoint.pt',map_location='cpu',weights_only=True)
        model=base.new_model(profile,seed);data=base.prepare(profile,seed)
        for cell in ('00','11'):
            x=cp['last_private_messages'].to('mps');r=cp['last_reports'].to('mps')
            if cell=='11':
                mu=x[2:].mean(0);sd=((x[2:]-mu).square().mean(0)).sqrt()
                x=x.clone();r=r.clone();x[:2]=mu+1.5*sd;r[:2]=1.
            coeff=1+4*r.clamp(0,.5);lam=coeff/coeff.sum()
            pilot,di=weighted_rfa(x,torch.full((10,),.1,device='mps'))
            assert di['unsmoothed_objective_gap_upper']<=.001
            c=pilot*(2/torch.linalg.vector_norm(pilot).clamp_min(1e-30)).clamp(max=1)
            ds=torch.linalg.vector_norm(x-c,dim=1)
            radius=max(4.,float(ds.sort().values[7]))
            a=(lam[:,None]*(c+(x-c)*(radius/ds.clamp_min(1e-30)).clamp(max=1)[:,None])).sum(0)
            local={}
            for eta in (*ETAS,-.001,.001):
                model.load_state_dict(cp['pre_round_model']);base.apply_gradient(model,a,eta)
                ev=base.evaluate(model,data,'val');m=vals(ev);local[eta]=m
                if eta in ETAS:
                    points.append(dict(seed=seed,cell=cell,eta=eta,honest=m,device='mps',evaluation=ev))
                    print(f"V34 step audit seed={seed} cell={cell} eta={eta:.5f} J={m['J_val']:.7f} W={m['worst20_pct']:.4f}",flush=True)
                if eta==.5:
                    ref=index[seed,'risk_winsor',cell]['evaluation']
                    for new,old in zip(ev['clients'],ref['clients']):
                        assert new['class_count']==old['class_count'] and new['class_hits']==old['class_hits']
                    assert math.isclose(m['J_val'],vals(ref)['J_val'],rel_tol=2e-6,abs_tol=2e-6)
            derivative=(local[.001]['J_val']-local[-.001]['J_val'])/.002
            predicted=2*index[seed,'risk_winsor',cell]['honest']['first_order_delta_J']
            error=abs(derivative-predicted);tolerance=.02*abs(predicted)+.00005
            derivatives.append(dict(seed=seed,cell=cell,central_difference=derivative,
                oracle_directional_derivative=predicted,absolute_error=error,tolerance=tolerance,
                passed=error<=tolerance))
            base.save(OUT/'partial.json',dict(points=points,derivatives=derivatives,source_stamp=stamp))
        del model,data,cp;torch.mps.empty_cache()
    base.verify_stamp(stamp)
    assert len(points)==24 and len(derivatives)==4
    result=dict(source_stamp=stamp,points=points,derivatives=derivatives,device='mps',
        passed=all(d['passed'] for d in derivatives),counts_and_contrasts_passed=True,
        reference_counts_replayed=4,model_evaluations=32,global_validation=False,
        chooses_learning_rate=False,changes_V33=False,posthoc_diagnostic=True)
    base.save(OUT/'results.json',result);base.save(DEST.with_suffix('.json'),result)
    return result


def report(r):
    lines=['# V34 — direction oracle vérifiée, effet de la taille du pas','',
        f"**32 évaluations sur MPS ; contrôle des quatre dérivées centrales : {'PASS' if r['passed'] else 'FAIL'}.**",'',
        'Diagnostic exploratoire fixé après V34, pas une confirmation. Aucun learning rate sélectionné. '
        'Chaque point recharge le même modèle pré-pas 12, déjà affecté par ALIE aux tours précédents. '
        '00 = aucune falsification au pas courant ; 11 = gradients et rapports falsifiés.', '',
        '| Seed | Cellule | Dérivée centrale | Dérivée oracle | Erreur absolue | Tolérance |',
        '|--:|:--|--:|--:|--:|--:|']
    for d in r['derivatives']:
        lines.append(f"| {d['seed']} | {d['cell']} | "+' | '.join(f'{d[k]:+.8f}' for k in
            ('central_difference','oracle_directional_derivative','absolute_error','tolerance'))+' |')
    lines+=['','Une dérivée positive signifie que, pour les pas positifs suffisamment petits, cette '
        'direction augmente le risque équitable **de validation**. Ce n’est pas une affirmation pour tout eta ni '
        'une équivalence avec les accuracies. Une dérivée négative ne garantit pas la baisse au pas fini 0,5.', '',
        '| Seed | Cellule | Eta | Acc honnête (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) | CE | Brier | J |',
        '|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|']
    for p in r['points']:
        lines.append(f"| {p['seed']} | {p['cell']} | {p['eta']:.5f} | "+' | '.join(f"{p['honest'][k]:.7f}"
            for k in ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2','ce_loss','brier_loss','J_val'))+' |')
    lines+=['','Comptages et contrastes des 24 cellules V34 recalculés ; quatre points eta=0,5 rejoués '
        'avec une écriture indépendante du clipping (RFA, modèle et évaluation partagés). Sources vérifiées '
        'inchangées. Les résultats complets contiennent aussi les métriques par client. Aucune nouvelle donnée '
        'de test ni seed de confirmation. Exports de recherche hors transcript DP.', '',
        '[Protocole](Private_Risk_V34_Direction_Step_Protocol.md) · [Evidence](Private_Risk_V34_Direction_Step_Audit.json)']
    DEST.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'audit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=run();report(result)
        print(json.dumps(dict(passed=result['passed'],model_evaluations=32,derivatives=result['derivatives'])))
