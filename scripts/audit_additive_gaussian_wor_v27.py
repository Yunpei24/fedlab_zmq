#!/usr/bin/env python3
"""Public, model-free calibration frontier; no training or data evaluation."""
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import mpmath
from scripts import run_fair_objective_screen as base
from privacy.fair_objective import calibrate, wor_rdp
from privacy.additive_gaussian_wor_v27 import plan, profile

REPORT=ROOT/'output/analysis/Additive_Gaussian_WOR_V27_Public_Audit'


def generic(batch,zr):
    def epsilon(z):
        return min(120*wor_rdp(a,batch/4800,z)+(120*a/(2*zr**2) if zr else 0.)
                   +math.log(1e5)/(a-1) for a in range(2,65))
    low,high=.25,32.
    for _ in range(60):
        mid=(low+high)/2
        if epsilon(mid)>4:low=mid
        else:high=mid
    z=high*(1+1e-8)
    return dict(gradient_z=z,gradient_std=4*z/batch,epsilon_realized=epsilon(z))


def main():
    paths=[Path(__file__),ROOT/'privacy/additive_gaussian_wor_v27.py',ROOT/'privacy/fair_objective.py',
           ROOT/'tests/test_additive_gaussian_wor_v27.py',
           ROOT/'output/analysis/Private_Risk_V26_Decision_and_V27_Accounting_Audit.md',
           ROOT/'tmp/pdfs/WBK2019_Supplement_Source.pdf']
    stamp={str(p.relative_to(ROOT)):base.digest(p) for p in paths}
    rows=[]
    zr=calibrate(q=1.,steps=120,epsilon=.25,delta=5e-6)
    for batch in (240,480,960,2400,4800):
        for family,z in [('erm_all_budget',None),('private_risk_channel',zr)]:
            base.verify_stamp(stamp)
            p=plan(batch=batch,risk_z=z)
            old=generic(batch,z)
            check=profile(q=batch/4800,z=p['gradient_z'],precision=320)
            eps320=min(120*v+(120*a/(2*z**2) if z else 0.)+math.log(1e5)/(a-1) for a,v in check.items())
            assert eps320<=4 and math.isclose(eps320,p['epsilon_realized'],rel_tol=1e-13)
            r=dict(family=family,plan=p,generic=old,epsilon_check_320_digits=eps320,
                   gradient_std_ratio_to_generic=p['gradient_std']/old['gradient_std'],
                   uniform_linear_noise_rms=math.sqrt(61706/10)*p['gradient_std'],
                   examples_per_client=120*batch,compute_factor_to_B240=batch/240)
            rows.append(r)
            print(f'V27 {family} B={batch}: std {old["gradient_std"]:.8f} -> {p["gradient_std"]:.8f}, epsilon={eps320:.9f}',flush=True)
            base.save(REPORT.with_name(REPORT.name+'_Partial').with_suffix('.json'),dict(status='partial',rows=rows,source_stamp=stamp))
    base.verify_stamp(stamp)
    ev=dict(audit_passed=True,rows=rows,source_stamp=stamp,mpmath_version=mpmath.__version__,
            calculation='public scalar accounting only; no model, gradient or dataset evaluated',
            private_operations_executed=0,training_launched=False,global_validation=False,completed_unix=time.time())
    base.save(REPORT.with_suffix('.json'),ev)
    lines=['# V27 — marge publique de bruit gaussien','',
        'Audit numérique par intervalles160 chiffres et vérification320 chiffres. Aucun entraînement, donnée ou métrique de modèle consulté.', '',
        'N=4800, T=120, C=2, epsilon total4, delta10⁻⁵ ; sans remise, replace-one. '
        'Le canal risque conserve sa calibration originale ; ERM dépense tout le budget dans les gradients.', '',
        '| Famille | B | Écart-type générique | Écart-type gaussien spécifique | Ratio | ε recalculé | Coût exemples vs B240 |',
        '|:--|--:|--:|--:|--:|--:|--:|']
    for r in rows:
        p=r['plan'];lines.append(f"| {r['family']} | {p['batch']} | {r['generic']['gradient_std']:.8f} | {p['gradient_std']:.8f} | {r['gradient_std_ratio_to_generic']:.5f} | {r['epsilon_check_320_digits']:.8f} | ×{r['compute_factor_to_B240']:g} |")
    lines+=['','## Ce que ce tableau ne prouve pas','',
        'Il ne mesure ni accuracy, ni fairness, ni robustesse. Le multiplicateur z augmente avec B, alors que le bruit injecté vaut z×2C/B. '
        'Une meilleure borne mathématique ne modifie pas rétroactivement les trajectoires historiques. '
        'Le gain de comptabilité doit aussi être accordé aux baselines. Le RMS linéaire du JSON ne décrit pas automatiquement la covariance de RFA.', '',
        'B=N élimine le sous-échantillonnage : on utilise directement la borne gaussienne exacte de sensibilité2C/N. '
        'La rupture possible près de q=1 dans la borne ne prouve pas une rupture intrinsèque de la privacy ou de l’accuracy.', '',
        '[Définitions, applicabilité et limites](Private_Risk_V26_Decision_and_V27_Accounting_Audit.md) · '
        '[Evidence complète](Additive_Gaussian_WOR_V27_Public_Audit.json).']
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
