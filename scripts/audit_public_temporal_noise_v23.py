#!/usr/bin/env python3
"""Compute a public grid and independently check RDP and noise-energy formulas."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
from scripts import run_fair_objective_screen as base
from privacy.fair_objective import require_mps
from privacy.public_temporal_noise_v23 import plan,simplified_gaussian
from privacy.split_risk_gradient import plan as risk_plan
from scripts.run_private_clipping_step_diagnostic_v16 import ledger

OUT=ROOT/'results/ldp_gradient_far/public_temporal_noise_audit_v23'
PROTOCOL=ROOT/'output/analysis/Public_Temporal_Noise_V23_Protocol.md'
REPORT=ROOT/'output/analysis/Public_Temporal_Noise_V23_Analyse.md'
TEST=ROOT/'tests/test_public_temporal_noise_v23.py'


def independent_rdp(a,q,z):
    # Direct log-binomial expansion, not the source's running log-add loop.
    inv=1/(z*z);second=min(math.log(4)+math.log(math.expm1(inv)),math.log(2)+inv)
    terms=[0.]+[math.log(math.comb(a,j))+j*math.log(q)+(second if j==2 else math.log(2)+j*(j-1)*inv/2) for j in range(2,a+1)]
    top=max(terms);sampled=(top+math.log(sum(math.exp(v-top) for v in terms)))/(a-1)
    return min(a*inv/2,sampled)


def main():
    require_mps();OUT.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),PROTOCOL,TEST,ROOT/'privacy/public_temporal_noise_v23.py',ROOT/'privacy/fair_objective.py',ROOT/'privacy/split_risk_gradient.py',ROOT/'scripts/run_private_clipping_step_diagnostic_v16.py']
    stamp={str(p.relative_to(ROOT)):base.digest(p) for p in files}
    assert not (OUT/'evidence.json').exists(),'Do not replace a completed public audit'
    p=subprocess.run([sys.executable,'-m','pytest',str(TEST),'-q'],cwd=ROOT,capture_output=True,text=True)
    base.save(OUT/'tests.json',dict(passed=p.returncode==0,output=p.stdout+p.stderr,source_stamp=stamp));assert p.returncode==0
    all_rows=[];decisions=[]
    for risk in (False,True):
        family=[]
        for k in range(33):
            ratio=2**(k/16);r=plan(ratio,risk)
            independently={a:60*independent_rdp(a,.05,r['z_early'])+60*independent_rdp(a,.05,r['z_late'])
                +(120*a/(2*r['risk_z']**2) if risk else 0.) for a in range(2,65)}
            for a,value in independently.items():assert math.isclose(value,r['rdp'][a],rel_tol=2e-12,abs_tol=2e-12)
            epsilon=min(value+math.log(1e5)/(a-1) for a,value in independently.items())
            assert epsilon<=4 and math.isclose(epsilon,r['epsilon_realized'],rel_tol=2e-12,abs_tol=2e-12)
            var_list=[r['sigma_early']**2]*60+[r['sigma_late']**2]*60
            actual=sum(eta**2*v for eta,v in zip([2.]*60+[.5]*60,var_list))
            assert math.isclose(actual,r['energy_proxy'],rel_tol=1e-12)
            r.update(grid_index=k,independently_verified=True);family.append(r);all_rows.append(r)
            print(f'public accounting {"risk" if risk else "ERM"} {k+1}/33 r={ratio:.6f} energy={actual:.8f}',flush=True)
        legacy=ledger(2.,'risk_rfa' if risk else 'erm_mean')
        assert math.isclose(family[0]['z_early'],legacy['gradient_z'],rel_tol=2e-12)
        best=min(family,key=lambda r:(r['energy_proxy'],r['grid_index']))
        decision=dict(with_risk=risk,best_grid_ratio=best['ratio'],best_grid_index=best['grid_index'],
            baseline_energy=family[0]['energy_proxy'],best_energy=best['energy_proxy'],
            relative_reduction=1-best['energy_proxy']/family[0]['energy_proxy'],
            ratio_two_relative_energy=family[16]['energy_proxy']/family[0]['energy_proxy'])
        decision['public_proxy_gate_passed']=decision['relative_reduction']>=.1;decisions.append(decision)
    base.verify_stamp(stamp);admitted=all(d['public_proxy_gate_passed'] for d in decisions)
    base.save(OUT/'evidence.json',dict(source_stamp=stamp,rows=all_rows,decisions=decisions,
        simplified=simplified_gaussian(),audit_passed=True,device_context='MPS available; no tensor training, host scalar accountant only',
        model_data_read=False,global_validation=False,admitted_to_real_diagnostic=admitted,automatic_training=False))
    lines=['# V23 — allocation temporelle : contrôler le vrai ledger','',
        '**66 calculs publics vérifiés**, sans gradients ni résultats de modèle. Accountant fixed-WOR avec composition jointe du risque lorsqu’il est utilisé.', '',
        'La formule gaussienne non échantillonnée suggère un rapport d’écarts-types2 et une réduction de26,47% de l’énergie. Ce n’est pas le coût de notre mécanisme échantillonné.', '',
        '| Canal | Meilleur rapport sur grille | Réduction du proxy | Proxy avec rapport2 / bruit constant | Gate ≥10% |',
        '|:--|--:|--:|--:|:--|']
    for d in decisions:
        lines.append(f'| {"risque + gradient" if d["with_risk"] else "gradient ERM"} | {d["best_grid_ratio"]:.6f} | {100*d["relative_reduction"]:.4f}% | {d["ratio_two_relative_energy"]:.6f} | {d["public_proxy_gate_passed"]} |')
    lines+=['',f'Admission à un diagnostic réel : **{"PASS" if admitted else "FAIL"}**. Aucun entraînement lancé par ce calcul.', '',
        '## Grille complète','',
        '| Canal | Ratio | σ début | σ fin | ε réalisé | Ordre | Proxy |','|:--|--:|--:|--:|--:|--:|--:|']
    for r in all_rows:
        lines.append(f'| {"risque" if r["with_risk"] else "ERM"} | {r["ratio"]:.6f} | {r["sigma_early"]:.8f} | {r["sigma_late"]:.8f} | {r["epsilon_realized"]:.8f} | {r["order"]} | {r["energy_proxy"]:.8f} |')
    lines+=['','Le minimum est celui d’une grille publique finie, pas un optimum global continu. Le proxy contrôle une composante gaussienne linéaire pour la moyenne ; il n’est ni une MSE totale du modèle ni une covariance de RFA. Même une forte diminution aurait exigé des tests de modèle/fairness puis des attaques. V22 n’est pas modifié.', '',
        '[Protocole et démonstrations](Public_Temporal_Noise_V23_Protocol.md).']
    REPORT.write_text('\n'.join(lines)+'\n');print(json.dumps(dict(decisions=decisions,admitted=admitted)),flush=True)


if __name__=='__main__':main()
