#!/usr/bin/env python3
"""Recompute the completed v2 calibration from immutable JSON metrics.

No model inference, test evaluation, training or automatic selection is performed.
Scalar statistics use the host; the original 24 trainings and vector oracles used MPS.
"""
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/ldp_gradient_far/fair_objective_calibration_v2'
REPORT=ROOT/'output/analysis/Fair_Objective_Calibration_V2_Analyse.md'
EVIDENCE=ROOT/'output/analysis/fair_objective_calibration_v2_analysis/evidence.json'
FIELDS=['accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2',
        'balanced_accuracy_pct','ce_loss','brier_loss','J_beta2','risk_variance']
SEEDS=[170301,170302]


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ms(x,d=2):return f'{st.mean(x):.{d}f} ± {st.stdev(x):.{d}f}'
def near(a,b):assert math.isclose(a,b,rel_tol=1e-7,abs_tol=1e-8),(a,b)
def rms(xs):return math.sqrt(st.mean(x*x for x in xs))


def verify_validation(v):
    cs=v['clients'];aa=[100*c['accuracy'] for c in cs];rr=[c['brier_loss'] for c in cs]
    assert len(cs)==10 and all(c['N']==1200 for c in cs)
    near(v['accuracy_pct'],st.mean(aa));near(v['variance_pp2'],st.pvariance(aa))
    near(v['worst20_pct'],st.mean(sorted(aa)[:2]))
    near(v['gap_best20_worst20_pp'],st.mean(sorted(aa)[-2:])-st.mean(sorted(aa)[:2]))
    near(v['gap_best_worst_pp'],max(aa)-min(aa))
    near(v['ce_loss'],st.mean(c['ce_loss'] for c in cs))
    near(v['brier_loss'],st.mean(rr));near(v['risk_variance'],st.pvariance(rr))
    near(v['J_beta2'],st.mean(r+r*r for r in rr))
    near(v['J_beta2'],st.mean(rr)+st.mean(rr)**2+st.pvariance(rr))
    near(v['balanced_accuracy_pct'],100*st.mean(c['balanced_accuracy'] for c in cs))


def main():
    manifest=read(SOURCE/'manifest.json');cfg=manifest['config']
    frozen=manifest['source_stamp']
    for path,digest in frozen.items():assert sha(ROOT/path)==digest, path
    assert read(SOURCE/'status.json')['status']=='completed'
    assert not (SOURCE/'failure.json').exists()
    expected=set(itertools.product(SEEDS,[.5,1.,2.],[1.,2.],[0.,2.]))
    runs={};sources=[];batch_hashes={};initials={};splits={}
    for p in sorted(SOURCE.glob('*/metrics.json')):
        m=read(p);s=read(p.parent/'orchestration_status.json');j=m['job']
        key=(j['seed'],j['C'],j['base_lr'],j['beta'])
        assert key in expected and key not in runs
        assert s['status']=='completed' and s['metrics_sha256']==sha(p)
        assert m['source_stamp']==frozen and s['device']==m['device']=='mps'
        assert s['round']==len(m['rounds'])==60 and not m['test_evaluated']
        assert 'test' not in m['final']
        assert m['privacy']['epsilon']<=4 and m['privacy']['steps']==60
        near(m['privacy']['std'],m['privacy']['sensitivity']*m['privacy']['z'])
        near(m['privacy']['sensitivity'],j['C']/240*(2+3*j['beta']))
        near(m['parameters']['server_lr'],j['base_lr']/(1+.45*j['beta']))
        prev=0
        verify_validation(m['initial'])
        # own_objective depends on beta but underlying predictions must be identical.
        signature=[(c['accuracy'],c['brier_loss'],c['ce_loss']) for c in m['initial']['clients']]
        if j['seed'] in initials:
            assert signature==initials[j['seed']]
            assert m['splits']==splits[j['seed']]
        else:initials[j['seed']]=signature;splits[j['seed']]=m['splits']
        for t,row in enumerate(m['rounds'],1):
            assert row['round']==t and row['device']=='mps'
            assert prev<=row['epsilon_at_round']<=4
            prev=row['epsilon_at_round']
            if row['validation'] is not None:verify_validation(row['validation'])
        op=p.parent/'simulator_oracle.json';o=read(op)
        assert not o['privacy_protected'] and not o['feeds_mechanism'] and len(o['rounds'])==60
        for row in o['rounds']:
            assert len(row['clients'])==10
            for c in row['clients']:
                pairkey=(j['seed'],row['round'],c['client'])
                if pairkey in batch_hashes:assert batch_hashes[pairkey]==c['batch_hash']
                else:batch_hashes[pairkey]=c['batch_hash']
        cs=[c for t in o['rounds'] for c in t['clients']]
        val=m['final']['validation'];eta=m['parameters']['server_lr']
        diag=dict(clip_pct=100*st.mean(c['clip_fraction'] for c in cs),
            batch_distortion_relative=st.mean(c['distortion_relative'] for c in cs if c['distortion_relative'] is not None),
            batch_cosine=st.mean(c['cosine'] for c in cs if c['cosine'] is not None),
            aggregate_distortion_relative=st.mean(t['aggregate_clip']['distortion_relative'] for t in o['rounds'] if t['aggregate_clip']['distortion_relative'] is not None),
            aggregate_clip_step_rms=eta*rms([t['aggregate_clip']['distortion_norm'] for t in o['rounds']]),
            aggregate_noise_step_rms=eta*rms([t['aggregate_dp_noise_norm'] for t in o['rounds']]),
            noise_to_clean_signal_median=st.median(t['noise_to_clean_signal'] for t in o['rounds'] if t['noise_to_clean_signal'] is not None),
            coefficient_ratio_median=st.median(max(c['coefficient'] for c in t['clients'])/min(c['coefficient'] for c in t['clients']) for t in o['rounds']),
            coefficient_mean=st.mean(c['coefficient'] for c in cs))
        run=dict(job=j,privacy=m['privacy'],eta=eta,metrics={f:val[f] for f in FIELDS},diagnostics=diag,
            checkpoints={str(t):{**{f:m['rounds'][t-1]['validation'][f] for f in FIELDS},
                'epsilon':m['rounds'][t-1]['epsilon_at_round']} for t in (20,40,50,60)})
        runs[key]=run
        sources += [dict(path=str(p.relative_to(ROOT)),sha256=sha(p)),dict(path=str(op.relative_to(ROOT)),sha256=sha(op))]
    assert set(runs)==expected
    groups={}
    for C,lr,beta in itertools.product([.5,1.,2.],[1.,2.],[0.,2.]):
        rr=[runs[(s,C,lr,beta)] for s in SEEDS]
        groups[C,lr,beta]=dict(C=C,base_lr=lr,beta=beta,
            mean={f:st.mean(r['metrics'][f] for r in rr) for f in FIELDS},
            sd={f:st.stdev(r['metrics'][f] for r in rr) for f in FIELDS},
            diagnostics={f:st.mean(r['diagnostics'][f] for r in rr) for f in rr[0]['diagnostics']})
    pairs=[]
    for C,lr in itertools.product([.5,1.,2.],[1.,2.]):
        for seed in SEEDS:
            a,b=runs[seed,C,lr,2.],runs[seed,C,lr,0.]
            pairs.append(dict(seed=seed,C=C,base_lr=lr,
                deltas={f:a['metrics'][f]-b['metrics'][f] for f in FIELDS},
                step_noise_ratio=a['diagnostics']['aggregate_noise_step_rms']/b['diagnostics']['aggregate_noise_step_rms']))
    # This is a descriptive comparison of best mean validation accuracy, not a promotion.
    best={beta:max((g for k,g in groups.items() if k[2]==beta),key=lambda g:g['mean']['accuracy_pct']) for beta in (0.,2.)}
    best_pairs=[]
    for s in SEEDS:
        aa=runs[s,best[2.]['C'],best[2.]['base_lr'],2.]
        bb=runs[s,best[0.]['C'],best[0.]['base_lr'],0.]
        best_pairs.append(dict(seed=s,deltas={f:aa['metrics'][f]-bb['metrics'][f] for f in FIELDS}))
    evid=dict(validated_runs=24,device='mps',test_evaluated=False,
        batches_paired_verified=True,initializations_splits_verified=True,
        gaussian_pairing='shared seeds in frozen source; full noise vectors not exported',
        source_hashes=sources,runs=list(runs.values()),groups=list(groups.values()),
        paired_contrasts=pairs,best_mean_validation_accuracy=best,best_mean_validation_accuracy_pairs=best_pairs,
        uncertainty='two calibration seeds only; sample SD, no significance claim; no automatic promotion')
    EVIDENCE.parent.mkdir(parents=True,exist_ok=True)
    EVIDENCE.write_text(json.dumps(evid,indent=2,allow_nan=False))
    lines=['# Objectif équitable privé — analyse de la calibration v2', '',
      '## Conclusion', '',
      '**Les 24 runs sont terminés et valides sur MPS. La calibration améliore fortement le contrôle Brier classique, mais ne démontre pas une supériorité générale de l’objectif équitable.** Les résultats ci-dessous sont exclusivement de validation, sur deux seeds de calibration : ils ne sont pas des résultats test ni une confirmation publiable.', '',
      'Le candidat produit parfois un compromis variance/gap contre accuracy ; les gains de Worst-20 ne sont pas réguliers. Il échoue nettement à C = 2 et pas de base = 2. Le plus grand C réduit la distorsion de clipping dans plusieurs comparaisons, mais augmente aussi le bruit. La comparaison ne contient pas de témoin sans DP permettant d’attribuer causalement la dégradation au seul bruit.', '',
      '## 1. Intégrité et protocole exact', '',
      '- 24/24 statuts completed ; 60 lignes de tours par run ; appareil MPS pour chaque tour.',
      '- Hashes des métriques et sources figées vérifiés ; métriques de validation recalculées à partir des dix résultats clients.',
      '- Splits et initialisations identiques entre configurations d’une seed. Hashes des batches identiques pour chacun des 600 couples client-tour, entre les douze configurations.',
      '- Gaussiens standards appariés par la construction du seed dans le code figé ; leurs vecteurs complets ne sont pas exportés.',
      '- Fashion-MNIST, LeNet-5 tanh, dix clients, Dirichlet par client à tailles contrôlées (0,1). N local = 4 800, validation = 1 200 ; batch 240 sans remise dans un tour, avec rééchantillonnage entre tours.',
      '- Une requête privée par client/tour, aucune epoch locale, moyenne serveur sans clipping, aucune attaque.',
      '- C = 0,5 / 1 / 2 ; pas de base = 1 / 2 ; beta = 0 (classique) ou 2 (équitable naïf). Seeds 170301 et 170302.',
      '- Pas serveur effectif eta = pas de base / (1 + 0,45 beta). À pas de base fixé, eta change entre les objectifs : ce facteur public fait partie de la méthode testée, il doit être conservé dans l’interprétation.', '',
      '**Toutes les valeurs ± sont moyenne ± écart-type d’échantillon sur deux seeds.** Les tours et les dix clients ne sont pas traités comme des réplications indépendantes. Aucune significativité statistique n’est revendiquée.', '',
      '## 2. Résultats finaux par réglage', '',
      'Variance : variance population des accuracies des dix clients, en pp². Gap : moyenne Best-20 moins moyenne Worst-20, en points de pourcentage. Un petit gap peut provenir d’une dégradation des meilleurs clients ; il doit être lu avec accuracy et Worst-20.', '',
      '| C | Pas de base | Objectif | Accuracy (%) ↑ | Worst-20 (%) ↑ | Variance (pp²) ↓ | Gap (pp) ↓ |',
      '|--:|--:|:--|--:|--:|--:|--:|']
    for C,lr,b in sorted(groups):
        rr=[runs[s,C,lr,b] for s in SEEDS]
        lines.append(f'| {C:g} | {lr:g} | {"Classique" if b==0 else "Équitable"} | '+' | '.join(ms([r['metrics'][f] for r in rr]) for f in FIELDS[:4])+' |')
    # Swap variance/gap values to match the table header (FIELDS orders gap first).
    for i,line in enumerate(lines):
        if line.startswith('| ') and ('| Classique |' in line or '| Équitable |' in line):
            parts=line.split('|');parts[-3],parts[-2]=parts[-2],parts[-3];lines[i]='|'.join(parts)
    lines += ['', '## 3. Comparaisons strictement appariées', '',
      'Équitable moins classique, à C et pas de base identiques. Les différences par seed sont affichées séparément pour ne pas masquer l’hétérogénéité.', '',
      '| C | Pas de base | Delta accuracy, seed 170301 / 170302 (pp) | Delta Worst-20, seed 170301 / 170302 (pp) | Delta gap moyen (pp) |',
      '|--:|--:|--:|--:|--:|']
    for C,lr in itertools.product([.5,1.,2.],[1.,2.]):
        pp=sorted([p for p in pairs if p['C']==C and p['base_lr']==lr],key=lambda p:p['seed'])
        fmt=lambda f:' / '.join(f"{p['deltas'][f]:+.2f}" for p in pp)
        lines.append(f"| {C:g} | {lr:g} | {fmt('accuracy_pct')} | {fmt('worst20_pct')} | {st.mean(p['deltas']['gap_best20_worst20_pp'] for p in pp):+.2f} |")
    lines += ['',
      'Observations : à C=0,5 / pas=1 et C=1 / pas=1, les deux seeds gagnent en accuracy et Worst-20. Cependant, ces réglages restent inférieurs au contrôle classique mieux réglé. À C=1 / pas=2, le gain moyen de Worst-20 cache une baisse sur la seed 170301 et une hausse sur 170302. À C=2 / pas=2, accuracy et Worst-20 baissent sur les deux seeds.', '',
      '## 4. Comparer aussi à un contrôle correctement réglé', '',
      'Sélection descriptive **après observation de la validation** : meilleur réglage de chaque famille selon son accuracy moyenne finale. Ce choix n’est ni une preuve d’optimalité ni une promotion pour publication.', '',
      '| Famille | C | Pas de base | Accuracy (%) | Worst-20 (%) | Variance (pp²) | Gap (pp) |',
      '|:--|--:|--:|--:|--:|--:|--:|']
    for b in (0.,2.):
        g=best[b];rr=[runs[s,g['C'],g['base_lr'],b] for s in SEEDS]
        lines.append(f"| {'Classique' if b==0 else 'Équitable'} | {g['C']:g} | {g['base_lr']:g} | "+' | '.join(ms([r['metrics'][f] for r in rr]) for f in ('accuracy_pct','worst20_pct','variance_pp2','gap_best20_worst20_pp'))+' |')
    delta={f:st.mean(p['deltas'][f] for p in best_pairs) for f in FIELDS}
    lines += ['',f"Par rapport au meilleur contrôle classique de cet écran, le meilleur candidat en accuracy a : **{delta['accuracy_pct']:+.2f} pp d’accuracy**, **{delta['worst20_pct']:+.2f} pp de Worst-20**, {delta['gap_best20_worst20_pp']:+.2f} pp de gap et {delta['variance_pp2']:+.2f} pp² de variance.", '',
      'La variance et le gap plus faibles sont un compromis possible, pas une amélioration de tous les critères. Les deux seeds sont insuffisantes pour établir ce compromis comme régulier. À C=1 / pas=2, le candidat garde une variance moyenne inférieure au contrôle du même réglage, mais perd en accuracy et n’améliore pas toujours le bas de distribution.', '',
      '## 5. Losses, balanced accuracy et objectif explicite', '',
      'J_beta2 = moyenne(R_i + R_i²), avec R_i la loss Brier du client sur validation. J_beta2 est évalué pour toutes les méthodes. La balanced accuracy est la moyenne des recalls de classes présentes chez chaque client, puis la moyenne entre clients ; ce n’est pas une mesure de parité entre groupes sensibles.', '',
      '| C | Pas | Objectif | Balanced accuracy (%) ↑ | Loss CE ↓ | Loss Brier ↓ | J_beta2 ↓ |',
      '|--:|--:|:--|--:|--:|--:|--:|']
    for C,lr,b in sorted(groups):
        rr=[runs[s,C,lr,b] for s in SEEDS]
        cells=[ms([r['metrics']['balanced_accuracy_pct'] for r in rr])]+[ms([r['metrics'][f] for r in rr],4) for f in ('ce_loss','brier_loss','J_beta2')]
        lines.append(f"| {C:g} | {lr:g} | {'Classique' if b==0 else 'Équitable'} | "+' | '.join(cells)+' |')
    worse=sum(p['deltas']['J_beta2']>0 for p in pairs)
    lines += ['',f"Le candidat a un J_beta2 final **plus élevé dans {worse}/12 comparaisons appariées**. Ce résultat ne réfute pas l’objectif mathématique : bruit, clipping, requête naïve et pas effectif empêchent de supposer qu’il a atteint son optimum. Il montre toutefois que cette instanciation ne minimise pas mieux sa propre cible dans la plupart des comparaisons.", '',
      '## 6. Ce que le clipping déforme réellement', '',
      'Fraction clippée : part des gradients individuels dont la norme dépasse C. Distorsion relative du batch : norme(G_clippé − G_brut) / norme(G_brut). Cosinus : alignement de ces deux moyennes de batch. Moyenne sur les 600 couples client-tour de chaque run, puis sur les deux seeds. Ce sont des oracles de simulation, pas des sorties privées. Une distorsion de 0,40 n’est ni « 40 % de données supprimées » ni une borne sur le biais du gradient population.', '',
      '| C | Pas | Objectif | Gradients clippés (%) | Distorsion relative du batch | Cosinus du batch |',
      '|--:|--:|:--|--:|--:|--:|']
    for C,lr,b in sorted(groups):
        rr=[runs[s,C,lr,b]['diagnostics'] for s in SEEDS]
        lines.append(f"| {C:g} | {lr:g} | {'Classique' if b==0 else 'Équitable'} | {ms([r['clip_pct'] for r in rr])} | {ms([r['batch_distortion_relative'] for r in rr],4)} | {ms([r['batch_cosine'] for r in rr],4)} |")
    lines += ['',
      'À pas de base 1, le classique passe d’environ 72 % de gradients clippés à C=0,5 à environ 16 % à C=2 ; la distorsion relative baisse d’environ 0,44 à 0,08. Cela confirme que C=0,5 tronquait substantiellement les mises à jour de batch. Le candidat à C=2 subit encore environ 31 % de clipping et une distorsion proche de 0,17 : les trajectoires ne sont pas identiques.', '',
      'Augmenter C ne suffit donc pas à rendre le candidat supérieur. Les calculs de distorsion portent sur les batches réalisés ; ce ne sont pas des mesures exactes du gradient de toute la population locale.', '',
      '### 6.1 La pondération est-elle toujours presque uniforme ?', '',
      'Il s’agit des coefficients locaux a_i = 1 + 2 × loss moyenne du batch, pas de poids FAR normalisés. Pour chaque run, on prend la médiane sur les 60 tours de max(a_i)/min(a_i), puis la moyenne ± écart-type entre les deux seeds.', '',
      '| C | Pas de base | Ratio max/min des coefficients équitables |',
      '|--:|--:|--:|']
    for C,lr in itertools.product([.5,1.,2.],[1.,2.]):
        rr=[runs[s,C,lr,2.]['diagnostics']['coefficient_ratio_median'] for s in SEEDS]
        lines.append(f'| {C:g} | {lr:g} | {ms(rr,3)} |')
    lines += ['',
      'Au réglage candidat C=1/pas=2, ce ratio vaut environ 1,40. On ne peut donc plus expliquer cette v2 uniquement par des coefficients presque identiques. Une pondération plus différenciée reste insuffisante pour garantir le bénéfice sur le modèle, notamment lorsque clipping et bruit modifient les gradients. La comparaison au ratio de v1 est descriptive, car seeds et horizon ont changé.', '',
      '## 7. Privacy et bruit dans le pas réellement appliqué', '',
      'Même N=4 800, b=240 et T=60 pour tous. Epsilon final = 3,9999999519851683, delta = 10^-5. Multiplicateur z = 1,3186076347774507, défini par rapport à la sensibilité complète. À beta=2, la borne de sensibilité du naïf est quatre fois celle du classique à C égal.', '',
      '| C | Objectif | Sensibilité Delta | Écart-type par coordonnée s |',
      '|--:|:--|--:|--:|']
    for C,b in itertools.product([.5,1.,2.],[0.,2.]):
        p=runs[SEEDS[0],C,1.,b]['privacy']
        lines.append(f"| {C:g} | {'Classique' if b==0 else 'Équitable'} | {p['sensitivity']:.8f} | {p['std']:.8f} |")
    lines += ['',
      'À C et pas de base identiques, le candidat a un écart-type de bruit quatre fois plus grand dans le message, et un pas serveur divisé par 1,9. L’écart-type du bruit dans la mise à jour de paramètres est donc multiplié par **4/1,9 = 2,1053**, et son énergie quadratique attendue par **(4/1,9)² = 4,4321**. Le nombre de clients est identique, donc le facteur 1/sqrt(n) de la moyenne se simplifie dans ce rapport.', '',
      'Ce rapport ne signifie pas epsilon différent : tous les mécanismes respectent le même budget avec des sensibilités différentes. Il décrit un coût de la requête équitable **avec les bornes utilisées**, pas une impossibilité fondamentale ni une borne inférieure optimale.', '',
      '| C | Pas | Objectif | RMS du bruit dans le pas modèle | RMS de la déformation de clipping dans le pas modèle | Ratio médian bruit / signal clippé |',
      '|--:|--:|:--|--:|--:|--:|']
    for C,lr,b in sorted(groups):
        rr=[runs[s,C,lr,b]['diagnostics'] for s in SEEDS]
        lines.append(f"| {C:g} | {lr:g} | {'Classique' if b==0 else 'Équitable'} | {ms([r['aggregate_noise_step_rms'] for r in rr],4)} | {ms([r['aggregate_clip_step_rms'] for r in rr],4)} | {ms([r['noise_to_clean_signal_median'] for r in rr],3)} |")
    lines += ['',
      'RMS signifie racine de la moyenne des carrés sur les 60 tours d’un run. Pour le bruit, on utilise eta × norme(A_privé − A_propre). Pour le clipping, eta × norme(A_propre_clippé − A_propre_non_clippé), à coefficients de loss du même batch. La direction appliquée contient aussi l’erreur de sampling ; ces deux normes ne constituent pas sa MSE totale par rapport au vrai gradient population.', '',
      'Un ratio bruit/signal supérieur à 1 en norme globale ne prouve pas à lui seul que l’apprentissage échoue : le contrôle classique apprend ici malgré de grands ratios. La direction, la courbure de la loss et la répartition du bruit dans les paramètres ne sont pas résumées par cette seule norme.', '',
      '**Inférence, pas attribution causale :** le surcoût de bruit et l’interaction avec le pas sont des explications plausibles de la forte baisse à C=2/pas=2. Aucun témoin sans DP strictement apparié n’a été entraîné dans cette v2. Les données ne permettent donc pas d’affirmer que le bruit explique à lui seul les 8,69 points perdus.', '',
      '## 8. Évolution à 20, 40 et 60 tours', '',
      'Les checkpoints sont ceux d’une trajectoire calibrée pour T=60. Leur epsilon réalisé est environ **2,465 à T=20**, **3,309 à T=40**, puis **4 à T=60**. Les comparer renseigne sur la trajectoire, pas sur un effet d’horizon à epsilon fixé à 4 à chaque arrêt.', '',
      '| C | Pas | Objectif | Accuracy 20 → 40 → 60 (%) | Worst-20 20 → 40 → 60 (%) | Delta J, 50 → 60 |',
      '|--:|--:|:--|:--|:--|--:|']
    for C,lr,b in sorted(groups):
        rr=[runs[s,C,lr,b] for s in SEEDS]
        a=' → '.join(f"{st.mean(r['checkpoints'][str(t)]['accuracy_pct'] for r in rr):.2f}" for t in (20,40,60))
        w=' → '.join(f"{st.mean(r['checkpoints'][str(t)]['worst20_pct'] for r in rr):.2f}" for t in (20,40,60))
        d=[r['checkpoints']['60']['J_beta2']-r['checkpoints']['50']['J_beta2'] for r in rr]
        lines.append(f"| {C:g} | {lr:g} | {'Classique' if b==0 else 'Équitable'} | {a} | {w} | {ms(d,5)} |")
    lines += ['',
      'Le contrôle classique continue de progresser entre 40 et 60 tours dans toutes les configurations. Prolonger automatiquement tous les runs n’est pas justifié : il faudrait recalibrer le bruit à horizon accru et ne pas sélectionner un horizon à partir du test. La comparaison v1/v2 ne doit pas être traitée comme appariée : seeds, réglages et calendrier de bruit diffèrent.', '',
      '## 9. Décision scientifique', '',
      '1. **Résultat acquis dans cet écran :** le contrôle Brier privé était sensible au réglage de clipping/pas/horizon ; il atteint ici environ 75,71 % d’accuracy validation et 51,73 % de Worst-20 au meilleur réglage en accuracy moyenne.',
      '2. **Résultat candidat :** l’objectif équitable peut réduire variance et gap dans certains régimes. Ce compromis doit être distingué d’un bénéfice robuste sur les clients les moins performants ; celui-ci n’est pas démontré.',
      '3. **Résultat négatif :** la configuration C=2/pas=2 est à écarter comme candidat prioritaire ; elle dégrade accuracy et Worst-20 sur les deux seeds.',
      '4. **Ne pas promouvoir aux attaques à ce stade.** La moyenne serveur n’a aucune garantie byzantine et nous n’avons pas encore un avantage propre régulier à préserver.',
      '5. Si l’on veut vérifier le seul compromis restant, figer avant toute nouvelle évaluation le candidat C=1/pas=2/beta=2 et les deux contrôles classiques C=1/pas=2 et C=2/pas=2 ; utiliser de nouvelles seeds, une marge acceptable d’accuracy et un critère principal de fairness définis avec l’objectif scientifique. Cela serait une confirmation du compromis, pas une recherche de configurations jusqu’à obtenir un gagnant. Aucun de ces runs n’est lancé par cette analyse.', '',
      'La calibration actuelle ne suffit pas à annoncer une méthode supérieure combinant fairness, robustesse et DP, ni à affirmer que tous les objectifs de fairness privés sont sans intérêt.', '',
      '## 10. Résultats par seed', '',
      '| Seed | C | Pas | Beta | Accuracy (%) | Worst-20 (%) | Variance (pp²) | Gap (pp) | Loss CE | J_beta2 |',
      '|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|']
    for key,r in sorted(runs.items()):
        seed,C,lr,b=key;v=r['metrics']
        lines.append(f"| {seed} | {C:g} | {lr:g} | {b:g} | {v['accuracy_pct']:.2f} | {v['worst20_pct']:.2f} | {v['variance_pp2']:.2f} | {v['gap_best20_worst20_pp']:.2f} | {v['ce_loss']:.4f} | {v['J_beta2']:.5f} |")
    lines += ['', '[Protocole figé](Fair_Objective_Calibration_V2_Protocol.md) · [Suivi des runs](Fair_Objective_Calibration_V2_Status.md) · [Évidence numérique](fair_objective_calibration_v2_analysis/evidence.json)', '']
    REPORT.write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(REPORT),evidence=str(EVIDENCE),validated_runs=24,
        best_classic=best[0.],best_fair=best[2.],paired_best_mean_delta=delta,
        J_worse_pairs=worse,new_runs_launched=0),indent=2))


if __name__=='__main__':main()
