"""Separate gradient/report poisoning on frozen real private messages, MPS."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_fair_objective_screen as base
from scripts.analyze_full_population_error_attribution_v28b import source_evidence
from privacy.fair_objective import require_mps
from privacy.capped_private_risk import weights
from privacy.private_risk_attacks import inject
from privacy.private_risk_winsor_v32 import winsor
from privacy.stable_weighted_rfa import weighted_rfa,stable_norm

SOURCE=ROOT/'results/ldp_gradient_far/full_population_private_risk_calibration_v28'
DEST=ROOT/'output/analysis/Private_Risk_Report_Gradient_Factorial'
DOC=ROOT/'output/analysis/Private_Risk_Report_Gradient_Factorial_Protocol.md'

def main():
    require_mps();stamp=dict(json.loads((SOURCE/'manifest.json').read_text())['source_stamp'])
    for s in (170501,170502):stamp.update(source_evidence(dict(seed=s,method='risk_rfa')))
    for p in (Path(__file__),DOC,ROOT/'privacy/private_risk_winsor_v32.py',ROOT/'privacy/private_risk_attacks.py'):
        stamp[str(p.relative_to(ROOT))]=base.digest(p)
    base.verify_stamp(stamp);rows=[]
    for seed in (170501,170502):
        cp=torch.load(SOURCE/f'seed{seed}__risk_rfa'/'checkpoint.pt',map_location='cpu',weights_only=True)
        x0=cp['last_private_messages'].to('mps');r0=cp['last_reports'].to('mps');mu=cp['last_clean_means'].to('mps')
        w0,_=weights(r0,.5);p=w0[2:]/w0[2:].sum();target=(p[:,None]*mu[2:]).sum(0)
        for attack in ('abrupt_bf','persistent_alie','slow_ipm'):
            x1,r1,info=inject(x0,r0,attack=attack,round_number=90)
            assert info['active'] and torch.equal(r0[2:],r1[2:]) and torch.equal(x0[2:],x1[2:])
            w1,_=weights(r1,.5)
            for method in ('risk_mean','risk_rfa','risk_winsor'):
                outputs={};vectors={};solver_checks=[]
                for g,x in enumerate((x0,x1)):
                    for r,reports in enumerate((r0,r1)):
                        w,_=weights(reports,.5)
                        if method=='risk_mean':a=(w[:,None]*x).sum(0);vectors[g,r]=x
                        elif method=='risk_rfa':
                            a,di=weighted_rfa(x,w);assert di['unsmoothed_objective_gap_upper']<=.001
                            solver_checks.append(di['unsmoothed_objective_gap_upper'])
                        else:
                            a,di,v=winsor(x,reports,C=2.,f_budget=2)
                            assert di['pilot_solver']['unsmoothed_objective_gap_upper']<=.001
                            vectors[g,r]=v['transformed'];solver_checks.append(di['pilot_solver']['unsmoothed_objective_gap_upper'])
                        outputs[g,r]=a
                A00,A10,A01,A11=(outputs[i] for i in ((0,0),(1,0),(0,1),(1,1)))
                G=A10-A00;R=A01-A00;I=A11-A10-A01+A00;total=A11-A00
                torch.testing.assert_close(total,G+R+I,rtol=2e-5,atol=1e-6)
                if method!='risk_rfa':
                    assert torch.equal(vectors[0,0],vectors[0,1]) and torch.equal(vectors[1,0],vectors[1,1])
                    dx=vectors[1,0]-vectors[0,0];dw=w1-w0
                    torch.testing.assert_close(G,(w0[:,None]*dx).sum(0),rtol=3e-5,atol=2e-6)
                    torch.testing.assert_close(R,(dw[:,None]*vectors[0,0]).sum(0),rtol=3e-5,atol=2e-6)
                    torch.testing.assert_close(I,(dw[:,None]*dx).sum(0),rtol=3e-5,atol=2e-6)
                squared=[v.square().sum() for v in (G,R,I)]
                cross=[2*(v*w).sum() for v,w in ((G,R),(G,I),(R,I))]
                torch.testing.assert_close(total.square().sum(),sum(squared+cross),rtol=3e-5,atol=3e-6)
                errors={f'E{g}{r}':float((a-target).square().sum()) for (g,r),a in outputs.items()}
                rows.append(dict(seed=seed,attack=attack,method=method,errors=errors,
                    report_effect_no_gradient_forgery=errors['E01']-errors['E00'],
                    report_effect_with_gradient_forgery=errors['E11']-errors['E10'],
                    gradient_effect_original_reports=errors['E10']-errors['E00'],
                    gradient_effect_forged_reports=errors['E11']-errors['E01'],
                    original_reserved_pair_mass=float(w0[:2].sum()),forged_reserved_pair_mass=float(w1[:2].sum()),
                    G_norm=float(stable_norm(G)),R_norm=float(stable_norm(R)),I_norm=float(stable_norm(I)),
                    total_norm=float(stable_norm(total)),twice_cross_products=[float(v) for v in cross],
                    solver_gap_bounds=solver_checks,identities_passed=True,device='mps'))
        del cp;torch.mps.empty_cache()
    assert len(rows)==18;base.verify_stamp(stamp)
    result=dict(rows=rows,aggregate_evaluations=72,identities_passed=True,device='mps',source_stamp=stamp,
        global_validation=False,training=False,model_evaluated=False,changes_to_V33=False,
        target='fixed honest clipped-gradient mean weighted by original private honest risk reports',
        state='V28 pre-round120, not a replay of the V33 failure trajectory')
    base.save(DEST.with_suffix('.json'),result)
    lines=['# Diagnostic factoriel : gradient falsifié et rapport falsifié','',
        '**72 agrégats sur les mêmes messages privés réels de deux anciens états V28, sur MPS.** '
        'Aucun entraînement, aucun seuil sélectionné, aucune confirmation. Les identités vectorielles '
        'et les produits croisés ont été vérifiés.', '',
        'Cible : moyenne des gradients honnêtes déjà clippés, avec poids de risque privé honnêtes '
        'fixés. Ce diagnostic ne décompose pas la perte d’accuracy de V33, dont les états divergent.', '',
        'E00 : aucune falsification ; E10 : gradients seuls ; E01 : rapports seuls ; E11 : les deux. '
        'E est une erreur quadratique d’agrégat, pas une erreur de classification. Un effet marginal '
        'négatif signifie que cette erreur baisse dans ce contrefactuel.', '',
        '| Seed | Attaque | Règle | E00 | E10 | E01 | E11 | Effet rapport sur gradient falsifié (E11−E10) |',
        '|--:|:--|:--|--:|--:|--:|--:|--:|']
    for r in rows:
        e=r['errors'];lines.append(f"| {r['seed']} | {r['attack']} | {r['method']} | "+
            ' | '.join(f'{e[k]:.6f}' for k in ('E00','E10','E01','E11'))+f" | {r['report_effect_with_gradient_forgery']:+.6f} |")
    lines+=['','## Décomposition vectorielle, normes non additives','',
        '| Seed | Attaque | Règle | Masse paire originale | Masse falsifiée | Norme G | Norme R | Norme interaction | Norme totale |',
        '|--:|:--|:--|--:|--:|--:|--:|--:|--:|']
    for r in rows:
        lines.append(f"| {r['seed']} | {r['attack']} | {r['method']} | "+' | '.join(f'{r[k]:.6f}' for k in
            ('original_reserved_pair_mass','forged_reserved_pair_mass','G_norm','R_norm','I_norm','total_norm'))+' |')
    lines+=['','Les rapports honnêtes ne sont jamais falsifiés. Le fait d’isoler le canal des '
        'rapports est un outil de diagnostic : il ne rend pas ce canal authentifiable en présence '
        'de clients arbitrairement malveillants. Une modification de mécanisme nécessiterait un '
        'nouveau protocole, sans changer le verdict V33.', '',
        '[Protocole et équations](Private_Risk_Report_Gradient_Factorial_Protocol.md) · '
        '[Résultats et empreintes](Private_Risk_Report_Gradient_Factorial.json)']
    DEST.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(aggregate_evaluations=72,passed=True,rows=rows)))

if __name__=='__main__':main()
