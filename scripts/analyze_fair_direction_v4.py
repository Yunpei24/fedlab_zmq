#!/usr/bin/env python3
"""Recompute scalar summaries from all frozen v4 replay files, no model execution."""
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/ldp_gradient_far/fair_direction_diagnostic_v4'
OUT=ROOT/'output/analysis/Fair_Direction_Diagnostic_V4_Analyse.md'


def main():
    paths=sorted(SOURCE.glob('seed*/replay_*.json'))
    assert len(paths)==32
    blocks=[json.loads(p.read_text()) for p in paths]
    assert all(b['status']=='completed' and b['device']=='mps' and not b['test_evaluated']
               and b['all_conditions_start_at_identical_parameters'] for b in blocks)
    evidence=json.loads((SOURCE/'evidence.json').read_text())
    assert evidence['evaluations']==320
    labels={
      'raw_erm':'Classique, brut sans bruit', 'raw_fair':'Équitable, brut sans bruit',
      'clip1_erm':'Classique, C=1 sans bruit','clip1_fair':'Équitable, C=1 sans bruit',
      'clip2_erm':'Classique, C=2 sans bruit','clip2_fair':'Équitable, C=2 sans bruit',
      'dp1_erm':'Classique, C=1 privé','dp1_fair':'Équitable, C=1 privé',
      'dp2_erm':'Classique, C=2 privé','dp2_fair':'Équitable, C=2 privé'}
    metrics=['accuracy_gain_pp','worst20_gain_pp','fixed_hard_loss_gain','J2_gain']
    overall={c:{k:st.mean(b['conditions'][c]['actual'][k] for b in blocks) for k in metrics} for c in labels}
    conflicts=[]
    for b in [b for b in blocks if b['replay']==0]:
        matrix=b['client_gradient_conflicts'];hard=b['hard_clients_fixed'];others=[i for i in range(10) if i not in hard]
        cross=sum(matrix[i][j] for i in hard for j in others)
        hh=sum(matrix[i][j] for i in hard for j in hard);oo=sum(matrix[i][j] for i in others for j in others)
        cosines=[matrix[i][j]/math.sqrt(matrix[i][i]*matrix[j][j]) for i in range(10) for j in range(i+1,10)]
        conflicts.append(dict(job=b['job'],hard_clients=hard,cosine_hard_vs_other=cross/math.sqrt(hh*oo),
            negative_pair_fraction=sum(x<0 for x in cosines)/len(cosines),min_cosine=min(cosines)))
    # Independent replay-to-state recomputation, rather than trusting the summary table.
    for row in evidence['states']:
        selected=[b for b in blocks if b['job']==row['job']]
        assert len(selected)==8
        for k in metrics:
            assert abs(st.mean(b['conditions'][row['condition']]['actual'][k] for b in selected)-row['actual'][k])<1e-12
    for p in ['raw','clip1','clip2','dp1','dp2']:
        for k,kk in [('accuracy_gain_pp','accuracy_advantage_pp'),('worst20_gain_pp','worst20_advantage_pp'),
                     ('J2_gain','J2_gain_advantage'),('fixed_hard_loss_gain','hard_loss_gain_advantage')]:
            assert abs(overall[p+'_fair'][k]-overall[p+'_erm'][k]-st.mean(c[kk] for c in evidence['contrasts'] if c['prefix']==p))<1e-12
    lines=['# Diagnostic v4 : le gain directionnel résiste-t-il au clipping et au bruit ?', '',
      '**Diagnostic terminé : 32 blocs appariés, 320 évaluations d’un seul pas, exclusivement sur MPS.**', '',
      '## 1. Conclusion exacte', '',
      'La pondération du risque conserve un avantage local sur les clients difficiles à C=1 sans bruit. '
      'Sous le bruit de la requête couplée, cet avantage devient insuffisamment régulier selon les critères préenregistrés. '
      'Cela ne signifie pas que tous les gains de fairness disparaissent : le Worst-20 reste meilleur à état commun dans les quatre états C=1 privés. '
      'En revanche, le risque équitable J2 est moins bien amélioré dans les quatre états, et une des deux seeds échoue sur la loss des clients difficiles.', '',
      'Les contrôles bruts montrent un autre écueil : une méthode peut être moins mauvaise que son contrôle tout en dégradant le modèle. '
      'Nous ne qualifions donc pas un avantage relatif de validation end-to-end.', '',
      '## 2. Ce qui est comparé', '',
      'Quatre modèles figés issus de la calibration v2 : deux seeds (170301, 170302), chacune avec un entraînement source ERM C=1 ou C=2, à T=60. '
      'Sur chaque état, huit nouveaux batches par client sont rejoués sous dix conditions. Chaque condition repart exactement du même modèle ; '
      'les batches et les réalisations gaussiennes standard sont partagés. Le modèle est rétabli après chaque pas.', '',
      'Les 32 blocs ne sont pas 32 réplications indépendantes : il n’y a que deux seeds et quatre états corrélés. '
      'Les moyennes ci-dessous sont descriptives, sans IC95 de confirmation. Aucun jeu test ni attaque n’est utilisé.', '',
      'Le gradient classique utilise la moyenne des gradients Brier. La variante équitable multiplie chaque contribution cliente par '
      '1 + 2 × sa loss Brier moyenne de batch. Les pas sont respectivement 2 et 2/1,9. '
      'Le diagnostic normalise aussi les directions pour ne pas confondre orientation et amplitude, mais les différences d’un pas restent dépendantes des pas choisis.', '',
      '**Deux sens de « clients difficiles ».** Les deux clients les moins précis au modèle de départ sont fixés avant les essais pour mesurer leur loss. '
      'Le Worst-20 de performance est, lui, recalculé après le pas et peut désigner d’autres clients. Les deux métriques ne sont pas interchangeables.', '',
      'J2 est la moyenne des risques clients R_i + R_i², avec R_i la loss Brier moyenne sur validation. '
      'Un gain de loss signifie valeur avant moins valeur après : positif = amélioration. '
      'Un gain d’accuracy signifie valeur après moins valeur avant, en points de pourcentage.', '',
      '## 3. Gains absolus : le modèle s’améliore-t-il réellement ?', '',
      '| Condition | Gain accuracy (pp) | Gain Worst-20 (pp) | Gain loss des clients difficiles | Gain J2 |',
      '|:--|--:|--:|--:|--:|']
    for c,label in labels.items():
        a=overall[c]
        lines.append(f"| {label} | {a['accuracy_gain_pp']:+.3f} | {a['worst20_gain_pp']:+.3f} | {a['fixed_hard_loss_gain']:+.6f} | {a['J2_gain']:+.6f} |")
    lines+=['',
      '**Observation.** C=1 sans bruit est le régime local le plus propre de ce diagnostic : les deux méthodes progressent en moyenne, '
      'et l’équitable progresse davantage. En brut sans clipping, elles perdent toutes les deux plusieurs points d’accuracy. '
      'Les directions non clippées sont donc testées avec un pas trop agressif pour ces états ; ce n’est pas une preuve qu’un entraînement non clippé serait intrinsèquement mauvais.', '',
      '## 4. Avantages relatifs et critères fixés avant exécution', '',
      '| Condition | Équitable − classique : accuracy (pp) | Worst-20 (pp) | Gain J2 | Écran local |',
      '|:--|--:|--:|--:|:--|']
    for p in ['raw','clip1','clip2','dp1','dp2']:
        a,b=overall[p+'_fair'],overall[p+'_erm']
        lines.append(f"| {p} | {a['accuracy_gain_pp']-b['accuracy_gain_pp']:+.3f} | {a['worst20_gain_pp']-b['worst20_gain_pp']:+.3f} | {a['J2_gain']-b['J2_gain']:+.6f} | {'passe' if evidence['gates'][p]['passed'] else 'échoue'} |")
    lines+=['',
      'L’écran exige un avantage de loss des clients difficiles sur au moins trois états sur quatre, '
      'un avantage moyen sur chacune des deux seeds, et une perte d’accuracy ne dépassant pas 0,25 pp par seed.', '',
      '- **dp1 échoue** : sur la seed 170301, l’avantage moyen de loss difficile vaut −0,000513, malgré un Worst-20 local favorable.',
      '- **dp2 échoue** : la seed 170301 perd 0,424 pp d’accuracy, au-delà de la marge de 0,25 pp.',
      '- **raw et clip2 passent l’écran relatif**, mais leurs gains absolus peuvent être négatifs : ces passages ne sont pas une autorisation d’annoncer une méthode utile.', '',
      '## 5. Ce que le diagnostic isole, et ses limites', '',
      '**Observé.** À C=1, la direction équitable sans bruit s’aligne mieux avec le gradient de loss des clients difficiles '
      'dans les quatre états, même par unité de norme. Après ajout du bruit, cet avantage directionnel normalisé devient négatif dans les quatre états. '
      'Le resserrement générique de sensibilité disponible ne réduit que 0,208 % de l’écart-type, insuffisant pour changer ce rapport de presque quatre entre la requête couplée et ERM.', '',
      '**Inférence.** Le coût du couplage loss × gradient est une cause plausible et directement testable de la fragilité. '
      'La pondération n’est pas sans signal utile avant le bruit. Cela motive la publication séparée d’un risque scalaire privé et d’un gradient privé standard.', '',
      '**Non identifié.** Ce diagnostic ne prouve pas que cette séparation améliore l’apprentissage complet. '
      'Il ne distingue pas à lui seul toutes les interactions entre pas, risque de batch, biais de clipping, dynamique d’entraînement et non-IID. '
      'Les oracles privés de recherche ne sont jamais des entrées serveur autorisées. Il n’évalue aucune robustesse byzantine.', '',
      '## 6. Conflits entre gradients honnêtes', '',
      'Les matrices de produits scalaires des gradients Brier de validation permettent un diagnostic sans nouvel entraînement. '
      'Le cosinus ci-dessous compare la moyenne des gradients des deux clients difficiles à celle des huit autres ; '
      'la proportion négative porte sur les 45 paires de clients, pas sur des seeds indépendantes. Ce sont des oracles propres, non des uploads privés.', '',
      '| Seed | C du checkpoint | Cosinus difficiles / autres | Paires à cosinus négatif (%) |',
      '|--:|--:|--:|--:|']
    for c in conflicts:
        lines.append(f"| {c['job']['seed']} | {c['job']['C']:g} | {c['cosine_hard_vs_other']:.4f} | {100*c['negative_pair_fraction']:.2f} |")
    lines+=['',
      'Les gradients moyens des deux groupes ne sont pas opposés ici, mais sont presque orthogonaux dans trois états sur quatre. '
      'Entre 55,56 % et 68,89 % des paires de clients ont un produit scalaire négatif. '
      'Il existe donc un conflit entre objectifs clients avant même le bruit DP ; améliorer l’estimation d’une moyenne ne supprime pas ce compromis. '
      'Cela ne prouve pas que RFA est la cause des échecs observés : son coût doit être comparé directement à celui de la moyenne.', '',
      '## 7. Décision et suite', '',
      'La confirmation v3 demeure négative : la méthode couplée n’est pas promue. '
      'L’écran v5 garde C=1 et compare une séparation risque/gradient à ERM disposant de tout epsilon=4, '
      'puis à des contrôles de même budget et de même pas. Une RFA pondérée opère directement sur l’agrégat. '
      'Les deux nouvelles seeds sont réservées à la calibration ; une réussite nécessiterait encore une confirmation indépendante et des attaques.', '',
      'La cible ne change pas : performances et Worst-20 sur données réservées, confidentialité effectivement comptabilisée, '
      'puis robustesse sous attaques, sans sélectionner seulement les runs favorables.', '',
      '## Sources locales vérifiées', '',
      '- [Détail par état et protocole de lecture](Fair_Direction_Diagnostic_V4_Status.md).',
      '- [Protocole v4 figé](Fair_Direction_Diagnostic_V4_Protocol.md).',
      '- [Résultats et critères complets](../../results/ldp_gradient_far/fair_direction_diagnostic_v4/evidence.json).',
      '- [Audit de sensibilité](Fair_Objective_Sensitivity_Tightness_V4.md).',
      '- [Confirmation v3 négative](Fair_Objective_Confirmation_V3_Analyse.md).',
      '- [Protocole v5 : séparation des publications](Split_Risk_Gradient_V5_Protocol.md).','']
    # Artifact generation only. No source or experiment record is modified.
    OUT.write_text('\n'.join(lines))
    scalar_evidence=dict(source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        overall=overall,gradient_conflicts=conflicts,independent_seeds=2,states=4,replay_blocks=32,evaluations=320,
        replicated_summary_agreement=True,not_end_to_end_validation=True)
    OUT.with_suffix('.json').write_text(json.dumps(scalar_evidence,indent=2,allow_nan=False)+'\n')
    import re
    links=re.findall(r'\]\(([^)]+)\)',OUT.read_text())
    assert all((OUT.parent/l).resolve().exists() for l in links)
    print(json.dumps(dict(report=str(OUT),verified_local_links=len(links),overall=overall),indent=2))


if __name__=='__main__':main()
