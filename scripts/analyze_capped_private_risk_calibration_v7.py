#!/usr/bin/env python3
"""Independent reconstruction of V7 privacy, metrics, pairing and fixed gates."""
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_split_risk_gradient_v5 import independent_wor, METRICS
SOURCE=ROOT/'results/ldp_gradient_far/capped_private_risk_calibration_v7'
OUT=ROOT/'output/analysis/Capped_Private_Risk_Calibration_V7_Analyse.md'


def main():
    s=json.loads((SOURCE/'status.json').read_text());assert s['status']=='completed' and s['valid_runs']==24
    manifest=json.loads((SOURCE/'manifest.json').read_text());m=manifest['config']
    assert json.loads((SOURCE/'pairing_audit.json').read_text())['passed']
    tests=json.loads((SOURCE/'tests.json').read_text());assert tests['passed'] and tests['source_stamp']==manifest['source_stamp']
    for p,h in manifest['source_stamp'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    recorded=json.loads((SOURCE/'evidence.json').read_text());rows=[];diagnostics=[]
    paths=sorted(SOURCE.glob('seed*/metrics.json'));assert len(paths)==24
    for path in paths:
        r=json.loads(path.read_text());status=json.loads(path.with_name('orchestration_status.json').read_text())
        assert status['status']=='completed' and status['metrics_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        assert r['source_stamp']==manifest['source_stamp'] and r['device']=='mps' and not r['test_evaluated']
        assert [t['round'] for t in r['rounds']]==list(range(1,61))
        assert all(t['device']=='mps' and t['aggregation']['eta']==2 for t in r['rounds'])
        p=r['privacy'];z=p.get('gradient_z',p.get('z'));risk=r['arm']['kind'].startswith('risk_')
        eps=min(60*independent_wor(a,.05,z)+(60*a/(2*p['risk_z']**2) if risk else 0)
                +math.log(1/p['delta'])/(a-1) for a in range(2,65))
        assert abs(eps-p['epsilon_realized'])<1e-10 and eps<=4
        assert math.isclose(p['gradient_std'],z*2*r['arm']['C']/m['batch_size'],rel_tol=1e-14,abs_tol=1e-16)
        for t in r['rounds']:
            if t['validation'] is None:continue
            v=t['validation'];aa=[100*c['accuracy'] for c in v['clients']];rr=[c['brier_loss'] for c in v['clients']]
            assert abs(st.mean(aa)-v['accuracy_pct'])<1e-9
            assert abs(st.pvariance(aa)-v['variance_pp2'])<1e-8
            assert abs(st.mean(sorted(aa)[:2])-v['worst20_pct'])<1e-9
            assert abs(st.mean(sorted(aa)[-2:])-st.mean(sorted(aa)[:2])-v['gap_best20_worst20_pp'])<1e-9
            assert abs(st.mean(x+x*x for x in rr)-v['J_beta2'])<1e-9
            for c in v['clients']:
                assert sum(c['class_count'])==c['N']
                assert abs(sum(c['class_hits'])/c['N']-c['accuracy'])<1e-9
        o=json.loads(path.with_name('simulator_oracle.json').read_text())
        assert [t['round'] for t in o['rounds']]==list(range(1,61))
        agg=[t['aggregation'] for t in r['rounds']]
        d=dict(r['job'],clip_pct=100*st.mean(c['clip_fraction'] for t in o['rounds'] for c in t['clients']),
            objective_weight_max_median=st.median(max(a['objective_weights']) for a in agg),
            stationary_weight_max_median=st.median(max(a['stationary_weights']) for a in agg),
            objective_weight_concentration_median=st.median(10*sum(w*w for w in a['objective_weights']) for a in agg),
            stationary_reconstruction_error_max=max(a['stationary_reconstruction_error'] for a in agg),
            rfa_minus_weighted_mean_norm_median=st.median(a['reference_minus_weighted_mean_norm'] for a in agg))
        if risk:
            d['coefficient_saturation_pct_median']=100*st.median(a['saturated_coefficient_fraction'] for a in agg)
            d['risk_scale']=r['arm']['scale']
        rows.append(r);diagnostics.append(d)
    # Check pairing ourselves across all 12 rules for each calibration seed.
    for seed in m['calibration_seeds']:
        selected=[r for r in rows if r['job']['seed']==seed];reference=selected[0]
        assert len(selected)==12
        patterns=[]
        for r in selected:
            assert r['splits']==reference['splits'] and r['initial']==reference['initial']
            p=SOURCE/f"seed{seed}__{r['job']['arm']}"/'simulator_oracle.json'
            o=json.loads(p.read_text());patterns.append([[c['batch_hash'] for c in t['clients']] for t in o['rounds']])
        assert all(p==patterns[0] for p in patterns)
    candidates=sorted({r['job']['arm'] for r in rows if r['arm']['kind'].startswith('risk_')})
    comparisons=[];gates={}
    for cand in candidates:
        for ctrl in m['screen']['controls']:
            for seed in m['calibration_seeds']:
                a=next(r['final']['validation'] for r in rows if r['job']==dict(seed=seed,arm=cand))
                b=next(r['final']['validation'] for r in rows if r['job']==dict(seed=seed,arm=ctrl))
                delta={k:a[k]-b[k] for k in METRICS}
                comparisons.append(dict(candidate=cand,control=ctrl,seed=seed,delta=delta,
                    passed=delta['accuracy_pct']>=-m['screen']['maximum_accuracy_loss_pp'] and delta['worst20_pct']>=m['screen']['minimum_worst20_advantage_pp']))
        gates[cand]=all(c['passed'] for c in comparisons if c['candidate']==cand)
    selected=next((a for a in m['screen']['robust_selection_order'] if gates[a]),None)
    assert selected==recorded['decision']['selected_robust_candidate']==s['selected_robust_candidate']
    assert gates=={a:g['passed'] for a,g in recorded['decision']['gates'].items()}
    lines=['# V7 — risque privé plafonné : analyse de la grille finie','',
      '**24/24 runs valides, 43 tests MPS préalables, budgets et métriques recomputés.** '
      'Calibration seulement : deux seeds reprises, aucune attaque ni évaluation test.', '',
      '## 1. Décision préenregistrée','',
      f"Candidate robuste admissible pour une confirmation indépendante : **{selected or 'aucune'}**.", '',
      'Passer exige, sur chacune des deux seeds et face à chacun des quatre contrôles, au moins +1 pp de Worst-20 '
      'et une perte d’accuracy au plus 1 pp. Aucune moyenne favorable ne remplace ces huit comparaisons par candidat. '
      'La sélection parmi les candidats RFA admissibles suit l’ordre fixé avant les runs, pas la meilleure accuracy observée.', '',
      '## 2. Mécanisme testé','',
      'Les clients publient séparément un risque Brier moyen privé et une moyenne privée de gradients clippés. '
      'Le serveur pose a_i=1+2 min(R̃_i/c,1), puis λ_i=a_i/somme(a_j). '
      'Le coefficient est la dérivée de la fonctionnelle Φ_c du protocole, pas une distance à une référence FAR. '
      'La moyenne ou RFA est pondérée directement. Le pas global reste 2.', '',
      'Fashion-MNIST / LeNet-5 tanh, dix clients, Dirichlet équilibré 0,1, 60 tours, batch 240 sans remise, '
      'C∈{1,2}, c∈{0,5 ; 0,25}, N=4800, validation=1200 exemples/client. '
      'Aucune epoch locale et aucun clipping serveur. Seeds de calibration 170501,170502.', '',
      'Les quatre contrôles sont ERM uniforme et RFA uniforme, chacun à C=1 et C=2, avec tout ε=4 pour les gradients. '
      'Les candidats partagent le même budget total entre risque et gradient, à δ=10⁻⁵. '
      'La garantie est sample-level côté client par exécution, pas client-level, ni une garantie des oracles et simulations conjointes à bruit partagé.', '',
      '## 3. Tableau complet par seed','',
      '| Seed | Règle | Accuracy (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) | Brier |',
      '|--:|:--|--:|--:|--:|--:|--:|']
    for r in rows:
        v=r['final']['validation'];j=r['job']
        lines.append(f"| {j['seed']} | {j['arm']} | {v['accuracy_pct']:.3f} | {v['worst20_pct']:.3f} | {v['gap_best20_worst20_pp']:.3f} | {v['variance_pp2']:.3f} | {v['brier_loss']:.5f} |")
    lines+=['','## 4. Moyenne ± écart-type entre les deux seeds','',
      'Descriptif uniquement ; pas d’IC de confirmation sur ces seeds réutilisées pour la calibration.', '',
      '| Règle | Accuracy (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) | Balanced accuracy (%) |',
      '|:--|--:|--:|--:|--:|--:|']
    for arm in sorted({r['job']['arm'] for r in rows}):
        vv=[r['final']['validation'] for r in rows if r['job']['arm']==arm]
        cells=[f'{st.mean(v[k] for v in vv):.3f} ± {st.stdev(v[k] for v in vv):.3f}' for k in ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2','balanced_accuracy_pct')]
        lines.append('| '+arm+' | '+' | '.join(cells)+' |')
    lines+=['','## 5. Critères, sans masquer les contrôles forts','',
      '| Candidat | Comparaisons passées / 8 | Plus faible Δ Worst-20 (pp) | Plus faible Δ accuracy (pp) | Admissible |',
      '|:--|--:|--:|--:|:--|']
    for a in candidates:
        cs=[c for c in comparisons if c['candidate']==a]
        lines.append(f"| {a} | {sum(c['passed'] for c in cs)}/8 | {min(c['delta']['worst20_pct'] for c in cs):+.3f} | {min(c['delta']['accuracy_pct'] for c in cs):+.3f} | {'oui' if gates[a] else 'non'} |")
    lines+=['','Tous les deltas par seed, y compris ceux qui font échouer la décision, sont conservés dans l’evidence JSON.', '',
      '## 6. Clipping et influence : ce que les poids représentent','',
      'λ pondère l’objectif de la médiane géométrique ; ce n’est pas nécessairement le coefficient effectif des messages. '
      'Les poids de Weiszfeld au point calculé sont ν_i proportionnels à λ_i/sqrt(||Y_i−z||²+lissage²). '
      'Leur reconstruction du point comporte un résidu dû aux itérations finies ; ce résidu est enregistré, pas supposé nul.', '',
      '| Seed | Règle | Clipping exemples (%) | Médiane max λ | Médiane max ν | Saturation coefficients (%) |',
      '|--:|:--|--:|--:|--:|--:|']
    for d in diagnostics:
        saturation='—' if 'coefficient_saturation_pct_median' not in d else f"{d['coefficient_saturation_pct_median']:.1f}"
        lines.append(f"| {d['seed']} | {d['arm']} | {d['clip_pct']:.2f} | {d['objective_weight_max_median']:.4f} | {d['stationary_weight_max_median']:.4f} | {saturation} |")
    lines+=['','## 7. Interprétation et suite autorisée','',
      'La borne W_B≤3m/(n+2m) est identique pour toutes les échelles c. Elle concerne la masse des poids d’objectif '
      'et donne un contrôle conditionnel d’influence, pas des accuracies sous attaque déjà observées. '
      'Réduire c renforce la pondération des risques modérés mais peut aussi amplifier le bruit des rapports et saturer des clients. '
      'C=2 réduit généralement le clipping tout en doublant l’écart-type du gradient ; il ne s’agit pas d’un contrôle sans coût.', '',
      'Aucune robustesse empirique ne peut être conclue de runs propres. Si une candidate est admissible, '
      'elle doit être figée avant les quatre seeds réservées et les attaques. Sinon la grille reste négative ; '
      'aucun seuil intermédiaire n’est choisi automatiquement après lecture des résultats.', '',
      '## Sources locales vérifiées','',
      '- [Protocole figé](Capped_Private_Risk_Calibration_V7_Protocol.md).',
      '- [Evidence indépendante et toutes les différences](Capped_Private_Risk_Calibration_V7_Analyse.json).',
      '- [Evidence originale](../../results/ldp_gradient_far/capped_private_risk_calibration_v7/evidence.json).',
      '- [Ablation précédente du pas](Split_Risk_Gradient_Fixed_Step_V6_Analyse.md).','']
    OUT.with_suffix('.json').write_text(json.dumps(dict(gates=gates,selected=selected,comparisons=comparisons,diagnostics=diagnostics,
        source_stamp=manifest['source_stamp'],privacy_and_metrics_recomputed=True),indent=2,allow_nan=False)+'\n')
    OUT.write_text('\n'.join(lines))
    import re
    assert all((OUT.parent/p).resolve().exists() for p in re.findall(r'\]\(([^)]+)\)',OUT.read_text()))
    print(json.dumps(dict(report=str(OUT),gates=gates,selected=selected),indent=2))


if __name__=='__main__':main()
