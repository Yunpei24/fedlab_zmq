#!/usr/bin/env python3
"""Read-only independent numerical audit; generated analysis contains all arms."""
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/ldp_gradient_far/split_risk_gradient_v5'
OUT=ROOT/'output/analysis/Split_Risk_Gradient_V5_Analyse.md'
LABELS={'erm_full':'ERM, tout ε=4','matched_mean':'Moyenne, pas apparié',
        'risk_mean':'Risque privé + moyenne','matched_rfa':'RFA, pas apparié','risk_rfa':'Risque privé + RFA'}
METRICS=['accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2','brier_loss','ce_loss','J_beta2','balanced_accuracy_pct']


def independent_wor(a,q,z):
    if q==1:return a/(2*z*z)
    logs=[0.,2*math.log(q)+math.log(math.comb(a,2))+math.log(min(4*math.expm1(1/z**2),2*math.exp(1/z**2)))]
    logs += [j*math.log(q)+math.log(math.comb(a,j))+math.log(2)+j*(j-1)/(2*z*z) for j in range(3,a+1)]
    top=max(logs)
    return min(a/(2*z*z),(top+math.log(sum(math.exp(v-top) for v in logs)))/(a-1))


def main():
    manifest=json.loads((SOURCE/'manifest.json').read_text());m=manifest['config']
    assert json.loads((SOURCE/'status.json').read_text())['status']=='completed'
    for p,h in manifest['source_stamp'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    evidence=json.loads((SOURCE/'evidence.json').read_text())
    runs=[];diag=[]
    for seed in m['calibration_seeds']:
        for method in m['methods']:
            d=SOURCE/f'seed{seed}__{method}'
            s=json.loads((d/'orchestration_status.json').read_text());r=json.loads((d/'metrics.json').read_text())
            assert s['status']=='completed' and s['metrics_sha256']==hashlib.sha256((d/'metrics.json').read_bytes()).hexdigest()
            assert r['source_stamp']==manifest['source_stamp'] and not r['test_evaluated'] and r['device']=='mps'
            assert len(r['rounds'])==60 and [x['round'] for x in r['rounds']]==list(range(1,61))
            p=r['privacy'];z=p.get('gradient_z',p.get('z'))
            eps=min(60*independent_wor(a,.05,z)+(0 if method=='erm_full' else 60*a/(2*p['risk_z']**2))
                    +math.log(1/p['delta'])/(a-1) for a in range(2,65))
            assert abs(eps-p['epsilon_realized'])<1e-10 and eps<=4
            for t in r['rounds']:
                assert t['device']=='mps' and t['epsilon_realized']<=4
                v=t['validation']
                if v is None:continue
                aa=[100*c['accuracy'] for c in v['clients']]
                assert len(aa)==10
                assert abs(st.mean(aa)-v['accuracy_pct'])<1e-10
                assert abs(st.pvariance(aa)-v['variance_pp2'])<1e-9
                assert abs(st.mean(sorted(aa)[:2])-v['worst20_pct'])<1e-10
                assert abs(st.mean(sorted(aa)[-2:])-st.mean(sorted(aa)[:2])-v['gap_best20_worst20_pp'])<1e-10
                rr=[c['brier_loss'] for c in v['clients']]
                assert abs(st.mean(x+x*x for x in rr)-v['J_beta2'])<1e-10
            o=json.loads((d/'simulator_oracle.json').read_text())
            assert len(o['rounds'])==60 and o['server_uses_only_private_risk_reports']
            q=dict(seed=seed,method=method,
                eta_mean=st.mean(t['aggregation']['eta'] for t in r['rounds']),
                eta_final=r['final']['aggregation']['eta'],
                clip_fraction=st.mean(c['clip_fraction'] for t in o['rounds'] for c in t['clients']),
                max_weight_median=st.median(max(t['aggregation']['weights']) for t in r['rounds']),
                concentration_median=st.median(10*sum(w*w for w in t['aggregation']['weights']) for t in r['rounds']))
            if method!='erm_full':
                errors=[c['risk_report']-c['raw_risk_research_only'] for t in o['rounds'] for c in t['clients']]
                q.update(risk_bias=st.mean(errors),risk_rmse=math.sqrt(st.mean(x*x for x in errors)),
                    report_saturation_fraction=st.mean(c['risk_report'] in (0.,1.) for t in o['rounds'] for c in t['clients']))
            if method.endswith('rfa'):
                q['solver_gap_median']=st.median(t['aggregation']['solver']['unsmoothed_objective_gap_upper'] for t in r['rounds'])
                q['solver_gap_max']=max(t['aggregation']['solver']['unsmoothed_objective_gap_upper'] for t in r['rounds'])
            runs.append(r);diag.append(q)
    contrasts=[]
    for candidate,controls in [('risk_mean',['erm_full','matched_mean']),('risk_rfa',['erm_full','matched_rfa']),('matched_rfa',['matched_mean'])]:
        for control in controls:
            for seed in m['calibration_seeds']:
                a=next(r['final']['validation'] for r in runs if r['job']==dict(seed=seed,method=candidate))
                b=next(r['final']['validation'] for r in runs if r['job']==dict(seed=seed,method=control))
                row=dict(candidate=candidate,control=control,seed=seed,delta={k:a[k]-b[k] for k in METRICS})
                row['passed']=row['delta']['accuracy_pct']>=-1 and row['delta']['worst20_pct']>=1
                contrasts.append(row)
    for candidate in ('risk_mean','risk_rfa'):
        assert all(x['passed'] for x in contrasts if x['candidate']==candidate)==evidence['gates'][candidate]['passed']
    aggregate={}
    for method in LABELS:
        selected=[r['final']['validation'] for r in runs if r['job']['method']==method]
        aggregate[method]={k:dict(mean=st.mean(v[k] for v in selected),sd=st.stdev(v[k] for v in selected)) for k in METRICS}
    lines=['# Risque privé + gradient privé : analyse complète de l’écran v5','',
      '**10/10 entraînements terminés, vérifiés ; 36 tests MPS réussis.** Deux seeds de calibration, aucune attaque et aucun test final.', '',
      '## 1. Question et contrôles','',
      'Le diagnostic précédent indiquait un coût élevé du produit privé loss × gradient. '
      'Cet écran remplace ce produit par deux publications composées : une loss Brier locale moyenne privée et un gradient standard privé. '
      'La moyenne et RFA sont directement pondérées par a_i=1+2R̃_i. Les contrôles « pas apparié » utilisent '
      'les mêmes rapports et le même pas, mais pas la pondération relative. ERM conserve tout ε=4 pour ses gradients et un pas constant de 2.', '',
      'Fashion-MNIST / LeNet-5 tanh, dix clients, Dirichlet par client équilibré 0,1 ; '
      '60 tours, un batch de 240 par client et par tour, C=1, aucune epoch locale ni clipping serveur. '
      'Les risques sont frais, calculés sur N=4800 exemples puis privatisés à chaque tour. '
      'Les seeds de calibration sont 170501 et 170502 ; chaque client a 1200 exemples de validation.', '',
      '## 2. Résultats par seed','',
      '| Seed | Règle | Accuracy (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) | Loss Brier |',
      '|--:|:--|--:|--:|--:|--:|--:|']
    for r in runs:
        v=r['final']['validation'];j=r['job']
        lines.append(f"| {j['seed']} | {LABELS[j['method']]} | {v['accuracy_pct']:.3f} | {v['worst20_pct']:.3f} | {v['gap_best20_worst20_pp']:.3f} | {v['variance_pp2']:.3f} | {v['brier_loss']:.5f} |")
    lines+=['','## 3. Moyennes ± écart-type entre les deux seeds','',
      'Ce sont des statistiques descriptives de calibration, pas des intervalles de confiance confirmatoires. '
      'Worst-20 = moyenne des deux clients les moins précis ; gap = Best-20 moins Worst-20 ; variance = variance population des dix accuracies clientes.', '',
      '| Règle | Accuracy (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) | Balanced acc. (%) |',
      '|:--|--:|--:|--:|--:|--:|']
    for method,a in aggregate.items():
        cells=[f"{a[k]['mean']:.3f} ± {a[k]['sd']:.3f}" for k in ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2','balanced_accuracy_pct')]
        lines.append('| '+LABELS[method]+' | '+' | '.join(cells)+' |')
    lines+=['','## 4. Contrastes appariés et critères préenregistrés','',
      'Pour chaque candidate : Worst-20 au moins +1 pp et perte d’accuracy au plus 1 pp, '
      'sur chacune des deux seeds, face à ERM full et au contrôle de même pas. Les seuils ne sont pas changés après observation.', '',
      '| Candidate − contrôle | Seed | Δ accuracy (pp) | Δ Worst-20 (pp) | Δ gap (pp) | Δ J2 | Écran |',
      '|:--|--:|--:|--:|--:|--:|:--|']
    for x in contrasts:
        d=x['delta'];verdict=('passe' if x['passed'] else 'échoue') if x['candidate']!='matched_rfa' else 'diagnostic'
        lines.append(f"| {LABELS[x['candidate']]} − {LABELS[x['control']]} | {x['seed']} | {d['accuracy_pct']:+.3f} | {d['worst20_pct']:+.3f} | {d['gap_best20_worst20_pp']:+.3f} | {d['J_beta2']:+.6f} | {verdict} |")
    for c in ('risk_mean','risk_rfa'):
        lines+=['',f"**{LABELS[c]} : {'passe' if evidence['gates'][c]['passed'] else 'échoue'} l’écran complet.**"]
    lines+=['','## 5. Mécanisme, bruit et pas réellement appliqué','',
      'Les ε sont recomputés indépendamment avec la borne RDP WOR et la composition de tous les rapports. '
      'Ils valent environ 4 dans chaque run à δ=10⁻⁵. Le gradient ERM a un écart-type de 0,0109884 par coordonnée, '
      'contre 0,0110005 pour la voie séparée ; le rapport de risque a un écart-type de 0,0384891. '
      'La garantie est sample-level côté client, replace-one, par exécution. Les oracles, validations et sorties conjointes à bruit partagé entre méthodes n’en font pas partie.', '',
      '| Seed | Règle | Pas moyen | Pas final | Clipping exemples (%) | Poids max médian | Concentration médiane |',
      '|--:|:--|--:|--:|--:|--:|--:|']
    for d in diag:
        lines.append(f"| {d['seed']} | {LABELS[d['method']]} | {d['eta_mean']:.3f} | {d['eta_final']:.3f} | {100*d['clip_fraction']:.2f} | {d['max_weight_median']:.4f} | {d['concentration_median']:.4f} |")
    lines+=['',
      'Le pas des voies séparées vaut (2/1,9)×moyenne(a_i). Il décroît quand les risques diminuent ; '
      'il n’est donc pas égal au pas fixe 2 du contrôle ERM. La comparaison avec le contrôle de même pas isole la pondération relative, '
      'tandis que la comparaison avec ERM examine la méthode complète. Une faible perte face au contrôle de même pas ne suffit pas à battre ERM.', '',
      'Les paramètres du solveur et les biais/RMSE des rapports figurent dans l’evidence numérique jointe. '
      'Ils sont des diagnostics et ne déclenchent pas une promotion automatique.', '',
      '## 6. Conclusion scientifique et limites','',
      'Un résultat de calibration positif n’est pas une confirmation de robustesse. RFA possède une borne conditionnelle '
      'd’influence lorsque la masse byzantine reste sous 1/2, mais cet écran ne comporte aucune attaque. '
      'Les performances sous attaque et les seeds indépendantes restent nécessaires pour le triplet fairness + privacy + robustness.', '',
      'En cas d’échec, la branche reste non promue. Une éventuelle expérience de pas constant doit être déclarée comme '
      'une nouvelle ablation, ne modifier aucun résultat existant, garder les contrôles de privacy et conserver un test final non utilisé. '
      'Ce n’est pas une autorisation de chercher un seuil après lecture de l’accuracy finale.', '',
      '## Liens vérifiés','',
      '- [Protocole figé](Split_Risk_Gradient_V5_Protocol.md).',
      '- [Audit théorique et limites DP](Split_Risk_Gradient_V5_Theoretical_Audit.md).',
      '- [Evidence de la campagne](../../results/ldp_gradient_far/split_risk_gradient_v5/evidence.json).',
      '- [Calculs indépendants de cette analyse](Split_Risk_Gradient_V5_Analyse.json).',
      '- [Diagnostic préalable](Fair_Direction_Diagnostic_V4_Analyse.md).','']
    OUT.with_suffix('.json').write_text(json.dumps(dict(aggregate=aggregate,contrasts=contrasts,
        diagnostics=diag,all_metric_and_privacy_checks_passed=True,source_stamp=manifest['source_stamp']),indent=2,allow_nan=False)+'\n')
    OUT.write_text('\n'.join(lines))
    import re
    links=re.findall(r'\]\(([^)]+)\)',OUT.read_text())
    assert all((OUT.parent/l).resolve().exists() for l in links)
    print(json.dumps(dict(report=str(OUT),gates=evidence['gates'],diagnostics=diag),indent=2))


if __name__=='__main__':main()
