#!/usr/bin/env python3
"""Public accounting-only grid with model dimension read on MPS; no training."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
from privacy.fair_objective import require_mps
from privacy.split_risk_gradient import plan
from models.registry import get_model
from scripts.run_fair_objective_screen import save,digest


def main():
    require_mps()
    model=get_model('lenet5_tanh','fashionmnist').to('mps')
    d=sum(p.numel() for p in model.parameters() if p.requires_grad)
    n=10;C=2.;cn=max((n+8*k)/(n+2*k)**2 for k in range(n+1))
    assert cn==34/256
    rows=[]
    for er in (.25,.35,.5,.75,1.,1.5,2.):
        p=plan(N=4800,b=240,T=120,C=C,epsilon=4.,delta=1e-5,epsilon_risk=er)
        risk=64*C*C*p['risk_std']**2
        gradient=d*cn*p['gradient_std']**2
        rows.append(dict(risk_calibration_target=er,ledger=p,public_risk_component=risk,
                         public_gradient_component=gradient,public_mean_mse_bound=risk+gradient))
    best=min(rows,key=lambda r:r['public_mean_mse_bound'])
    decrease=1-best['public_mean_mse_bound']/rows[0]['public_mean_mse_bound']
    protocol=ROOT/'output/analysis/Private_Risk_Channel_Allocation_Diagnostic_V14_Protocol.md'
    dest=ROOT/'output/analysis/Private_Risk_Channel_Allocation_V14_Audit'
    payload=dict(device='mps',dimension=d,num_clients=n,weight_square_sum_bound=cn,rows=rows,
        minimum_public_bound_target=best['risk_calibration_target'],public_bound_relative_reduction=decrease,
        rfa_mse_established=False,model_improvement_established=False,promotion=False,training_runs=0,
        source_stamp={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),protocol,ROOT/'privacy/split_risk_gradient.py',ROOT/'privacy/fair_objective.py',ROOT/'models/registry.py']})
    save(dest.with_suffix('.json'),payload)
    lines=['# V14 — allocation des deux canaux : calcul public','',
        f'Dimension réelle du modèle : **{d}** ; n=10, T=120, C=2, B=240, N=4800, ε≤4 et δ=10⁻⁵.', '',
        'Aucun entraînement, aucune accuracy consultée, aucune promotion. Cette borne concerne la moyenne pondérée, **pas RFA**.', '',
        '| Cible risque | Écart-type risque | Écart-type gradient | Terme risque | Terme gradient | Borne MSE moyenne | ε conjoint |',
        '|--:|--:|--:|--:|--:|--:|--:|']
    for r in rows:
        p=r['ledger']
        lines.append(f"| {r['risk_calibration_target']:g} | {p['risk_std']:.6f} | {p['gradient_std']:.6f} | {r['public_risk_component']:.6f} | {r['public_gradient_component']:.6f} | {r['public_mean_mse_bound']:.6f} | {p['epsilon_realized']:.8f} |")
    lines+=['',f"Minimum de cette grille publique : cible risque **{best['risk_calibration_target']:g}**, diminution de borne **{100*decrease:.2f} %** face à 0,25.", '',
        'Une diminution de borne n’est pas une diminution mesurée de la MSE réelle. La borne ne démontre ni un gain de modèle, '
        'ni la qualité des coefficients de reconstruction de RFA. Les coefficients de risque sont indépendants du bruit de gradient, '
        'mais les coefficients effectifs de la médiane ne le sont pas.', '',
        '[Dérivation et protocole](Private_Risk_Channel_Allocation_Diagnostic_V14_Protocol.md). '
        '[Ledger et tous les calculs](Private_Risk_Channel_Allocation_V14_Audit.json).']
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(dimension=d,best_public_target=best['risk_calibration_target'],bound_reduction=decrease,
                         promotion=False,training_runs=0,report=str(dest.with_suffix('.md')))))


if __name__=='__main__':main()
