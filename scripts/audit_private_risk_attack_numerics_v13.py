#!/usr/bin/env python3
"""Constructed MPS audit; records both large residual bounds and actual shifts."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from privacy.fair_objective import require_mps
from privacy.private_risk_attacks import inject
from privacy.capped_private_risk import weights
from privacy.stable_weighted_rfa import weighted_rfa,stable_norm
from scripts.run_fair_objective_screen import save,digest


def main():
    require_mps()
    x=torch.arange(50,device='mps',dtype=torch.float32).reshape(10,5)/100
    r=torch.linspace(0,.5,10,device='mps')
    rows=[]
    for attack in ('abrupt_bf','persistent_alie','slow_ipm'):
        y,s,_=inject(x,r,attack=attack,round_number=60)
        w,_=weights(s,.5)
        comparison,_=weighted_rfa(y,w,iterations=640)
        cmp_obj=float((w*stable_norm(y-comparison,dim=1)).sum())
        for n in (40,80,160,320,640):
            a,d=weighted_rfa(y,w,iterations=n)
            obj=float((w*stable_norm(y-a,dim=1)).sum())
            rows.append(dict(attack=attack,iterations=n,
                reported_objective_gap_upper=d['unsmoothed_objective_gap_upper'],
                objective_difference_vs_640=obj-cmp_obj,
                point_distance_vs_640=float(stable_norm(a-comparison)),
                below_1e3=d['unsmoothed_objective_gap_upper']<=1e-3,
                residual_norm=d['residual_norm']))
    dest=ROOT/'output/analysis/Private_Risk_Attack_Numerics_V13_Audit'
    payload=dict(device='mps',rows=rows,candidate_unchanged=True,
        interpretation='A reported upper bound above 1e-3 is not a lower bound on actual error. 640 is a comparison, not an exact oracle.',
        source_stamp={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'privacy/private_risk_attacks.py',ROOT/'privacy/stable_weighted_rfa.py']})
    save(dest.with_suffix('.json'),payload)
    lines=['# V13 — précision numérique : limite du diagnostic à 40 itérations','',
        'Audit construit sur MPS, hors données et hors entraînement. La première assertion « borne <10⁻³ pour tout exemple construit » a échoué. '
        'Elle supposait à tort une précision universelle pour un nombre fixe d’itérations. Le test de non-régression conserve maintenant cet échec construit explicitement.', '',
        '| Attaque | Itérations | Borne supérieure d’écart d’objectif | Différence d’objectif vs 640 itérations | Distance au point à 640 |',
        '|:--|--:|--:|--:|--:|']
    for r in rows:
        lines.append(f"| {r['attack']} | {r['iterations']} | {r['reported_objective_gap_upper']:.8g} | {r['objective_difference_vs_640']:.8g} | {r['point_distance_vs_640']:.8g} |")
    lines+=['','Le dépassement d’une **borne supérieure conservatrice** ne démontre pas que l’erreur réelle dépasse ce seuil. '
        'Le résultat à 640 itérations est un comparateur numérique, pas un optimum certifié. '
        'Les arrondis float32 et la très petite valeur de lissage peuvent limiter la réduction du résidu.', '',
        'La règle des campagnes V12/V13 reste à 40 itérations. Le seuil de diagnostic 10⁻³ de V13 reste inchangé : '
        's’il échoue sur les uploads réels, le gate de précision sera signalé négatif, sans changer silencieusement le solveur ou le seuil. '
        'Ce test construit n’utilise aucune accuracy et ne sélectionne aucun hyperparamètre du modèle.', '',
        '[Calculs complets](Private_Risk_Attack_Numerics_V13_Audit.json).']
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(device='mps',rows=len(rows),report=str(dest.with_suffix('.md')),candidate_unchanged=True)))


if __name__=='__main__':main()
