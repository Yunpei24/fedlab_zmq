#!/usr/bin/env python3
"""Audit V6 and paired step-only contrasts with the frozen V5 campaign."""
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_split_risk_gradient_v5 import independent_wor, LABELS, METRICS
SOURCE=ROOT/'results/ldp_gradient_far/split_risk_gradient_fixed_step_v6'
PREVIOUS=ROOT/'results/ldp_gradient_far/split_risk_gradient_v5'
OUT=ROOT/'output/analysis/Split_Risk_Gradient_Fixed_Step_V6_Analyse.md'


def main():
    status=json.loads((SOURCE/'status.json').read_text());assert status['status']=='completed'
    manifest=json.loads((SOURCE/'manifest.json').read_text());m=manifest['config']
    assert json.loads((SOURCE/'parent_pairing_audit.json').read_text())['passed']
    for p,h in manifest['source_stamp'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    evidence=json.loads((SOURCE/'evidence.json').read_text())
    rows=[];steps=[];vsold=[]
    for seed in m['calibration_seeds']:
        for method in m['methods']:
            name=f'seed{seed}__{method}';folder=SOURCE/name
            r=json.loads((folder/'metrics.json').read_text());old=json.loads((PREVIOUS/name/'metrics.json').read_text())
            s=json.loads((folder/'orchestration_status.json').read_text())
            assert s['status']=='completed' and s['metrics_sha256']==hashlib.sha256((folder/'metrics.json').read_bytes()).hexdigest()
            assert r['source_stamp']==manifest['source_stamp'] and not r['test_evaluated'] and r['device']=='mps'
            assert [t['round'] for t in r['rounds']]==list(range(1,61))
            assert r['privacy']==old['privacy'] and r['splits']==old['splits'] and r['initial']==old['initial']
            p=r['privacy'];z=p.get('gradient_z',p.get('z'))
            eps=min(60*independent_wor(a,.05,z)+(0 if method=='erm_full' else 60*a/(2*p['risk_z']**2))
                    +math.log(1/p['delta'])/(a-1) for a in range(2,65))
            assert abs(eps-p['epsilon_realized'])<1e-10 and eps<=4
            for t in r['rounds']:
                assert t['device']=='mps' and t['aggregation']['eta']==2 and t['epsilon_realized']<=4
                if t['validation'] is None:continue
                v=t['validation'];aa=[100*c['accuracy'] for c in v['clients']]
                assert abs(st.mean(aa)-v['accuracy_pct'])<1e-10
                assert abs(st.pvariance(aa)-v['variance_pp2'])<1e-9
                assert abs(st.mean(sorted(aa)[:2])-v['worst20_pct'])<1e-10
                assert abs(st.mean(sorted(aa)[-2:])-st.mean(sorted(aa)[:2])-v['gap_best20_worst20_pp'])<1e-10
            v=r['final']['validation'];b=old['final']['validation']
            rows.append(dict(seed=seed,method=method,metrics=v))
            vsold.append(dict(seed=seed,method=method,delta={k:v[k]-b[k] for k in METRICS}))
            o=json.loads((folder/'simulator_oracle.json').read_text());oo=json.loads((PREVIOUS/name/'simulator_oracle.json').read_text())
            assert [[c['batch_hash'] for c in t['clients']] for t in o['rounds']]==[[c['batch_hash'] for c in t['clients']] for t in oo['rounds']]
    comparisons=[]
    for candidate,controls in [('risk_mean',m['screen']['primary_controls']),('risk_rfa',m['screen']['robust_controls'])]:
        for control in controls:
            for seed in m['calibration_seeds']:
                a=next(r['metrics'] for r in rows if r['seed']==seed and r['method']==candidate)
                b=next(r['metrics'] for r in rows if r['seed']==seed and r['method']==control)
                d={k:a[k]-b[k] for k in METRICS}
                comparisons.append(dict(seed=seed,candidate=candidate,control=control,delta=d,
                    passed=d['accuracy_pct']>=-1 and d['worst20_pct']>=1))
    gates={a:all(r['passed'] for r in comparisons if r['candidate']==a) for a in ('risk_mean','risk_rfa')}
    assert gates==status['gates']=={a:g['passed'] for a,g in evidence['gates'].items()}
    lines=['# V6 : risque privé, agrégation pondérée et pas constant','',
      '**10/10 runs valides sur MPS. Pas fixe 2 vérifié à chaque tour.** '
      'Contrôle ERM v5 reproduit ; batches, modèles initiaux, partitions et budgets appariés avec v5.', '',
      '## 1. Verdict selon les critères fixés','']
    for a,g in gates.items():lines.append(f"- {LABELS[a]} : **{'passe' if g else 'échoue'}** l’écran de calibration.")
    lines+=['','Ces résultats portent sur deux seeds de calibration reprises de v5, sans attaque et sans test final. '
      'Le passage d’un écran n’est pas une validation du triplet local-DP + fairness + robustness.', '',
      '## 2. Paramètre isolé','',
      'V5 appliquait un pas (2/1,9)×moyenne(1+2R̃_i), qui diminue pendant l’apprentissage. V6 applique 2 à toutes les règles. '
      'Les rapports de risque, le bruit, le clipping C=1, les coefficients et le solveur restent inchangés. '
      'Fashion-MNIST / LeNet-5 tanh, dix clients, Dirichlet équilibré 0,1, 60 tours, batch 240 sans remise, '
      'aucune epoch locale, epsilon≈4, delta=10⁻⁵, 1200 exemples de validation par client.', '',
      'Les contrôles uniformes avec budget partagé conservent le léger coût du rapport privé. ERM full donne '
      'tout epsilon=4 à ses gradients. À pas constant, les contrôles permettent de distinguer le coût du rapport et l’effet des poids.', '',
      '## 3. Résultats complets par seed','',
      '| Seed | Règle | Accuracy (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) | Loss Brier | J2 |',
      '|--:|:--|--:|--:|--:|--:|--:|--:|']
    for r in rows:
        a=r['metrics']
        lines.append(f"| {r['seed']} | {LABELS[r['method']]} | {a['accuracy_pct']:.3f} | {a['worst20_pct']:.3f} | {a['gap_best20_worst20_pp']:.3f} | {a['variance_pp2']:.3f} | {a['brier_loss']:.5f} | {a['J_beta2']:.5f} |")
    lines+=['','## 4. Différences candidate − contrôle','',
      'Seuils : gain Worst-20 ≥ 1 pp et perte d’accuracy ≤ 1 pp, sur chaque seed et contre les deux contrôles. '
      'Aucune moyenne favorable ne remplace une comparaison échouée.', '',
      '| Candidate − contrôle | Seed | Δ accuracy (pp) | Δ Worst-20 (pp) | Δ gap (pp) | Écran |',
      '|:--|--:|--:|--:|--:|:--|']
    for r in comparisons:
        d=r['delta']
        lines.append(f"| {LABELS[r['candidate']]} − {LABELS[r['control']]} | {r['seed']} | {d['accuracy_pct']:+.3f} | {d['worst20_pct']:+.3f} | {d['gap_best20_worst20_pp']:+.3f} | {'passe' if r['passed'] else 'échoue'} |")
    lines+=['','## 5. Effet de la seule modification du pas : v6 − v5','',
      '| Règle | Seed | Δ accuracy (pp) | Δ Worst-20 (pp) | Δ J2 |',
      '|:--|--:|--:|--:|--:|']
    for r in vsold:
        d=r['delta'];lines.append(f"| {LABELS[r['method']]} | {r['seed']} | {d['accuracy_pct']:+.3f} | {d['worst20_pct']:+.3f} | {d['J_beta2']:+.6f} |")
    lines+=['','## 6. Limites et décision','',
      'Le changement de pas peut expliquer une partie du déficit de v5, mais il ne prouve pas à lui seul '
      'une supériorité équitable. Les clients difficiles peuvent changer, et une meilleure loss J2 ne suffit pas : '
      'les différences d’accuracy et de Worst-20 restent les critères principaux.', '',
      'RFA pondérée possède un contrôle conditionnel d’influence, non une performance sous attaque déjà mesurée. '
      'Toute poursuite doit conserver des contrôles solides (y compris ERM C=2 issu des calibrations précédentes), '
      'des seeds indépendantes, le même budget total et des attaques effectivement exécutées. '
      'Une comparaison nouvelle reste explicitement distincte de la présente calibration.', '',
      'Les métriques et budgets ont été recomputés indépendamment, les empreintes vérifiées, et tous les pas vérifiés égaux à 2. '
      'Privacy sample-level par exécution ; pas de garantie pour les oracles et publications conjointes de simulations à bruit partagé.', '',
      '## Sources','',
      '- [Protocole v6 préenregistré](Split_Risk_Gradient_Fixed_Step_V6_Protocol.md).',
      '- [Analyse v5, conservée](Split_Risk_Gradient_V5_Analyse.md).',
      '- [Evidence indépendante](Split_Risk_Gradient_Fixed_Step_V6_Analyse.json).',
      '- [Evidence et critères de la campagne](../../results/ldp_gradient_far/split_risk_gradient_fixed_step_v6/evidence.json).','']
    OUT.with_suffix('.json').write_text(json.dumps(dict(rows=rows,comparisons=comparisons,v6_minus_v5=vsold,
        gates=gates,metrics_and_privacy_recomputed=True,source_stamp=manifest['source_stamp']),indent=2,allow_nan=False)+'\n')
    OUT.write_text('\n'.join(lines))
    import re
    assert all((OUT.parent/p).resolve().exists() for p in re.findall(r'\]\(([^)]+)\)',OUT.read_text()))
    print(json.dumps(dict(report=str(OUT),gates=gates,comparisons=comparisons),indent=2))


if __name__=='__main__':main()
