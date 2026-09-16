#!/usr/bin/env python3
"""Read-only audit of frozen training outputs; no model evaluation or new runs.

Standard-library scalar summaries of existing MPS metrics. Raw simulator losses
remain research oracles, not a new DP release. No test-based tuning is performed.
"""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/ldp_gradient_far/fair_objective_screen_v1'
DEST = ROOT / 'output/analysis/fair_objective_screen_v1_audit'
METHODS = ['ce', 'brier_erm', 'brier_naive', 'brier_unbiased']
LABELS = dict(ce='CE', brier_erm='Brier classique', brier_naive='Équitable naïf',
              brier_unbiased='Équitable corrigé')
SEEDS = [170201, 170202, 170203, 170204]


def load(path):
    return json.loads(path.read_text())


def avg_ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.] * len(xs)
    k = 0
    while k < len(order):
        j = k + 1
        while j < len(order) and xs[order[j]] == xs[order[k]]:
            j += 1
        for i in order[k:j]:
            out[i] = (k + j - 1) / 2 + 1
        k = j
    return out


def correlation(x, y):
    a, b = st.mean(x), st.mean(y)
    den = math.sqrt(sum((v-a)**2 for v in x) * sum((v-b)**2 for v in y))
    return sum((v-a)*(w-b) for v, w in zip(x, y))/den if den > 1e-18 else None


def spearman(x, y):
    return correlation(avg_ranks(x), avg_ranks(y))


def fmt(xs, digits=2):
    xs = [x for x in xs if x is not None]
    if not xs:
        return 'non défini'
    return f'{st.mean(xs):.{digits}f} ± {st.stdev(xs):.{digits}f}' if len(xs)>1 else f'{xs[0]:.{digits}f}'


def target(v, beta=2.):
    rr = [c['brier_loss'] for c in v['clients']]
    mu, var = st.mean(rr), st.pvariance(rr)
    direct = st.mean(r + beta/2*r*r for r in rr)
    assert math.isclose(direct, mu + beta/2*(mu*mu + var), abs_tol=1e-12)
    return dict(risk_mean=mu, risk_variance=var, objective_beta2=direct,
                worst20_risk=st.mean(sorted(rr)[-2:]))


def validation_at(m, t):
    return m['initial']['validation'] if t == 0 else m['rounds'][t-1]['validation']


def audit():
    runs, sources = {}, []
    for path in sorted(SOURCE.glob('*/metrics.json')):
        m = load(path)
        meta = load(path.parent/'orchestration_status.json')
        assert meta['status'] == 'completed' and meta['device'] == m['device'] == 'mps'
        assert len(m['rounds']) == meta['round'] == 20
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert meta['metrics_sha256'] == digest
        sources.append(dict(path=str(path.relative_to(ROOT)), sha256=digest))
        if m['job']['phase'] != 'evaluation':
            continue
        job = m['job']; budget = 'noDP' if job['epsilon'] is None else 'eps4'
        key = (job['seed'], job['method'], budget)
        assert key not in runs
        oracle_path = path.parent/'simulator_oracle.json'
        oracle = load(oracle_path)
        assert not oracle['feeds_mechanism'] and not oracle['privacy_protected']
        assert len(oracle['rounds']) == 20
        sources.append(dict(path=str(oracle_path.relative_to(ROOT)), sha256=hashlib.sha256(oracle_path.read_bytes()).hexdigest()))
        runs[key] = (m, oracle)
    assert len(runs) == 32 and len(sources) == 66

    trajectories, aligned, summaries, clients, paired = [], [], [], [], []
    for (seed, method, budget), (m, oracle) in runs.items():
        for t in (0, 1, 5, 10, 15, 20):
            v = validation_at(m, t)
            trajectories.append(dict(seed=seed, method=method, budget=budget, round=t,
                accuracy=v['accuracy_pct'], worst20=v['worst20_pct'],
                gap=v['gap_best20_worst20_pp'], variance=v['variance_pp2'],
                ce_loss=v['ce_loss'], balanced_accuracy=v['balanced_accuracy_pct'], **target(v)))
        v15, v20 = validation_at(m,15), validation_at(m,20)
        summary = dict(seed=seed, method=method, budget=budget,
            delta_val_accuracy_15_20=v20['accuracy_pct']-v15['accuracy_pct'],
            delta_val_worst20_15_20=v20['worst20_pct']-v15['worst20_pct'],
            delta_val_ce_loss_15_20=v20['ce_loss']-v15['ce_loss'],
            delta_val_J_15_20=target(v20)['objective_beta2']-target(v15)['objective_beta2'],
            test_val_accuracy_gap=m['final']['test']['accuracy_pct']-v20['accuracy_pct'],
            clip_mean=st.mean(c['clip_fraction'] for r in oracle['rounds'] for c in r['clients']),
            clip_max=max(c['clip_fraction'] for r in oracle['rounds'] for c in r['clients']),
            **target(v20))
        # These statistics concern a local multiplier, NOT normalized FAR weights.
        if method.startswith('brier_'):
            beta = 0. if method == 'brier_erm' else 2.
            spreads, ratios, coefficients = [], [], []
            for row in oracle['rounds']:
                aa = [1+beta*c['loss_mean'] for c in row['clients']]
                spreads.append(max(aa)-min(aa)); ratios.append(max(aa)/min(aa))
                coefficients.extend(aa)
            summary.update(coefficient_min=min(coefficients), coefficient_max=max(coefficients),
                           coefficient_span_median=st.median(spreads),
                           coefficient_maxmin_median=st.median(ratios))
        summaries.append(summary)
        if method in ('brier_naive', 'brier_unbiased'):
            # The batch of t is drawn at w_(t-1) in the 1-based run logs.
            # Compare with validation AFTER t-1, never after the same round t.
            for t in (1, 2, 6, 11, 16):
                vc = validation_at(m,t-1)['clients']
                oc = oracle['rounds'][t-1]['clients']
                rr = [c['loss_mean'] for c in oc]
                low = sorted(range(10),key=lambda i: (vc[i]['accuracy'],i))[:2]
                high = sorted(range(10),key=lambda i: (-rr[i],i))[:2]
                aligned.append(dict(seed=seed, method=method,budget=budget,round=t,
                    validation_round=t-1,
                    rank_corr_batchloss_validation_brier=spearman(rr,[c['brier_loss'] for c in vc]),
                    rank_corr_batchloss_validation_error=spearman(rr,[1-c['accuracy'] for c in vc]),
                    recall_lowest_accuracy_two=len(set(low)&set(high))/2,
                    smallest_accuracy_tie_at_cut=vc[sorted(range(10),key=lambda i:(vc[i]['accuracy'],i))[1]]['accuracy']==vc[sorted(range(10),key=lambda i:(vc[i]['accuracy'],i))[2]]['accuracy']))
        for cid,c in enumerate(v20['clients']):
            ce = runs[(seed,'ce',budget)][0]
            baseline = validation_at(ce,20)['clients'][cid]
            first = validation_at(ce,5)['clients']
            hard = sorted(range(10), key=lambda i:(first[i]['accuracy'],i))[:2]
            clients.append(dict(seed=seed,method=method,budget=budget,client=cid,
                val_accuracy=100*c['accuracy'], val_brier=c['brier_loss'], val_ce=c['ce_loss'],
                val_balanced_accuracy=100*c['balanced_accuracy'],
                delta_accuracy_vs_ce=100*(c['accuracy']-baseline['accuracy']),
                ce_hard_at_round5=cid in hard))
    # Paired validation differences: four independent seeds, no round pseudoreplication.
    for budget in ('noDP','eps4'):
        for control in ('ce','brier_erm','brier_naive'):
            for seed in SEEDS:
                x = validation_at(runs[(seed,'brier_unbiased',budget)][0],20)
                y = validation_at(runs[(seed,control,budget)][0],20)
                a,b = target(x),target(y)
                paired.append(dict(seed=seed,budget=budget,control=control,
                    delta_accuracy=x['accuracy_pct']-y['accuracy_pct'],
                    delta_worst20=x['worst20_pct']-y['worst20_pct'],
                    delta_J=a['objective_beta2']-b['objective_beta2'],
                    delta_mean_risk=a['risk_mean']-b['risk_mean'],
                    delta_risk_variance=a['risk_variance']-b['risk_variance']))
    result = dict(source='frozen fair_objective_screen_v1',validated_runs=34,
        evaluation_runs=32,sources=sources,trajectories=trajectories,
        aligned_pre_update_diagnostics=aligned,summaries=summaries,
        final_validation_clients=clients,paired_validation=paired,
        limitations=['post-hoc descriptive audit, not new independent confirmation',
                     'no full-training client risk or unclipped full gradient trajectory saved',
                     'oracle batch losses are not DP releases and never enter the server',
                     'client identities/splits are paired; trajectories diverge between methods'])
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/'evidence.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    lines = ['# Audit de l’objectif équitable privé : optimisation et clients difficiles', '',
      'Audit post-hoc des **32 runs d’évaluation terminés**, sans nouvel entraînement. Les 34 runs, calibration comprise, ont un statut terminé, un appareil MPS et un hash de métriques conforme. Les résultats historiques sont inchangés.', '',
      'Les tableaux ci-dessous utilisent **la validation**, jamais le test pour choisir un paramètre. La source est néanmoins une campagne déjà examinée : ce diagnostic reste exploratoire, et ne devient pas une confirmation indépendante par changement de nom du split.', '',
      '## 1. L’entraînement a-t-il atteint un plateau ?', '',
      'Différence validation au tour 20 moins tour 15, moyenne ± écart-type sur quatre seeds. Une baisse de J ou de loss est favorable. J est toujours évalué avec beta = 2, même pour les contrôles qui ne l’optimisent pas directement.', '',
      '| Budget | Méthode | Delta accuracy (pp) | Delta Worst-20 (pp) | Delta CE loss | Delta J | J baisse (seeds) |',
      '|:--|:--|--:|--:|--:|--:|--:|']
    for budget in ('noDP','eps4'):
        for method in METHODS:
            rows=[x for x in summaries if x['budget']==budget and x['method']==method]
            lines.append(f"| {budget} | {LABELS[method]} | {fmt([x['delta_val_accuracy_15_20'] for x in rows])} | {fmt([x['delta_val_worst20_15_20'] for x in rows])} | {fmt([x['delta_val_ce_loss_15_20'] for x in rows],4)} | {fmt([x['delta_val_J_15_20'] for x in rows],4)} | {sum(x['delta_val_J_15_20']<0 for x in rows)}/4 |")
    lines += ['', 'Le budget de calcul est de 20 × 240 = 4 800 exemples tirés par client : autant que la taille locale, **en comptant les répétitions**. Ce n’est pas une epoch couvrant chaque exemple. Un exemple a une probabilité (1−240/4800)^20 ≈ 35,85 % de n’avoir jamais été tiré ; la couverture attendue est 64,15 %. Les modèles changent entre tirages. Une pente encore favorable n’assure pas un gain à horizon long ; prolonger à epsilon fixe impose de recalibrer le bruit.', '',
      '## 2. Les grandes losses désignent-elles les faibles accuracies ?', '',
      'Le batch de chaque tour t est mesuré avant la mise à jour. Nous l’alignons uniquement avec la validation du même modèle pré-tour (t−1), disponible pour t = 1, 2, 6, 11, 16. Le tableau exclut les deux premiers instants proches de l’initialisation ; il moyenne t = 6, 11, 16 **au sein de chaque seed**, puis résume les quatre seeds.', '',
      'Corrélation de rang de Spearman ; +1 signifie le même classement. Le rappel indique combien des deux clients de plus faible accuracy validation appartiennent aux deux clients de plus forte loss de batch (0, 1/2 ou 1). Les égalités sont départagées par identifiant ; les détails sont conservés dans evidence.json.', '',
      '| Budget | Méthode | Corr. loss batch / loss validation | Corr. loss batch / erreur classification | Rappel des deux clients difficiles (%) |',
      '|:--|:--|--:|--:|--:|']
    for budget in ('noDP','eps4'):
        for method in ('brier_naive','brier_unbiased'):
            values=collections.defaultdict(list)
            for seed in SEEDS:
                rows=[r for r in aligned if r['budget']==budget and r['method']==method and r['seed']==seed and r['round'] in (6,11,16)]
                for field in ('rank_corr_batchloss_validation_brier','rank_corr_batchloss_validation_error','recall_lowest_accuracy_two'):
                    values[field].append(st.mean(r[field] for r in rows if r[field] is not None))
            lines.append(f"| {budget} | {LABELS[method]} | {fmt(values['rank_corr_batchloss_validation_brier'],3)} | {fmt(values['rank_corr_batchloss_validation_error'],3)} | {fmt([100*v for v in values['recall_lowest_accuracy_two']])} |")
    lines += ['', 'Ce classement ne mesure **pas** l’alignement des directions de gradients. Reconnaître un client difficile ne prouve pas que sa mise à jour améliore son accuracy ou celle des autres. Le coefficient moyen du corrigé vaut 1 + beta × loss moyenne du batch, mais sa requête n’est pas exactement ce coefficient multiplié par le gradient moyen.', '',
      '## 3. Amplitude de la pondération et clipping', '',
      'Pas de poids FAR lambda, ni de softmax dans cet écran. Le ratio décrit les coefficients locaux moyens. Il n’est pas un facteur de concentration de poids normalisés.', '',
      '| Budget | Méthode | Médiane temporelle max(coefficient)/min(coefficient) | Fraction moyenne de gradients clippés (%) |',
      '|:--|:--|--:|--:|']
    for budget in ('noDP','eps4'):
        for method in METHODS:
            rows=[x for x in summaries if x['budget']==budget and x['method']==method]
            ratio=fmt([x['coefficient_maxmin_median'] for x in rows],3) if method.startswith('brier') else '1 (ERM)'
            lines.append(f"| {budget} | {LABELS[method]} | {ratio} | {fmt([100*x['clip_mean'] for x in rows])} |")
    lines += ['', 'Moyennes sur les 200 couples client-tour de chaque run, puis moyenne ± écart-type entre quatre seeds. Une fraction faible ne borne pas à elle seule le biais vectoriel du clipping.', '',
      '## 4. Est-ce que le candidat optimise mieux sa propre cible ?', '',
      'Au tour 20, corrigé moins contrôle sur validation. J = moyenne(R + R²), R = demi-Brier moyenne du client. La variance ici est celle des **losses**, pas la variance des accuracies en pp².', '',
      '| Budget | Contrôle | Delta J | Delta risque moyen | Delta variance des risques | Delta Worst-20 (pp) |',
      '|:--|:--|--:|--:|--:|--:|']
    for budget in ('noDP','eps4'):
        for control in ('ce','brier_erm','brier_naive'):
            rows=[x for x in paired if x['budget']==budget and x['control']==control]
            lines.append(f"| {budget} | {LABELS[control]} | {fmt([x['delta_J'] for x in rows],5)} | {fmt([x['delta_mean_risk'] for x in rows],5)} | {fmt([x['delta_risk_variance'] for x in rows],5)} | {fmt([x['delta_worst20'] for x in rows])} |")
    lines += ['', 'J = moyenne(R) + moyenne(R)² + variance(R) pour beta = 2. Il combine niveau moyen et dispersion des losses ; il ne minimise pas directement le Worst-20 ou le gap d’accuracy. Les différences de clipping et de pas entre CE et Brier empêchent d’en déduire un effet causal de la loss seule. Entre ERM Brier et équitable, le pas serveur est également différent selon la règle publique de compensation initiale.', '',
      '## 5. Suivi de clients difficiles fixes', '',
      'Pour chaque seed et budget, les deux clients difficiles sont identifiés par la **validation CE au tour 5**, puis conservés identiques entre méthodes. Le tableau donne leur différence moyenne d’accuracy validation finale par rapport à CE. Sélection post-hoc et dépendante du contrôle : diagnostic descriptif, ni nouveau Worst-20 (les identités peuvent changer), ni preuve causale.', '',
      '| Budget | Méthode | Delta accuracy des clients difficiles fixes (pp) | Seeds avec gain |',
      '|:--|:--|--:|--:|']
    for budget in ('noDP','eps4'):
        for method in METHODS[1:]:
            per_seed=[st.mean(r['delta_accuracy_vs_ce'] for r in clients if r['seed']==s and r['budget']==budget and r['method']==method and r['ce_hard_at_round5']) for s in SEEDS]
            lines.append(f"| {budget} | {LABELS[method]} | {fmt(per_seed)} | {sum(v>0 for v in per_seed)}/4 |")
    lines += ['', '## 6. Paramètres DP réels', '',
      '| Méthode | Sensibilité de la requête | Écart-type du bruit par coordonnée | Epsilon |',
      '|:--|--:|--:|--:|']
    for method in METHODS:
        p=runs[(SEEDS[0],method,'eps4')][0]['privacy']
        lines.append(f"| {LABELS[method]} | {p['sensitivity']:.8f} | {p['std']:.8f} | {p['epsilon']:.8f} |")
    lines += ['', 'La sensibilité de la requête équitable est environ quatre fois celle de Brier ERM avec le même C. Naïf et corrigé ont presque la même sensibilité ; ce surcoût n’est donc pas propre à la correction du biais. Le pas serveur réduit l’effet du bruit dans l’espace des paramètres, mais agit aussi sur le signal. CE utilise C = 4 contre 0,5 pour Brier : on ne peut pas comparer les écarts-types comme si les signaux avaient la même échelle.', '',
      '## 7. Ce que les fichiers ne permettent pas de conclure', '',
      '- La loss moyenne du batch n’est pas le risque exact sur les 4 800 exemples locaux. La validation en est un indicateur séparé, pas une identité.',
      '- Les oracles sauvegardés ne contiennent pas les directions individuelles des gradients ou leurs produits scalaires avec les gradients de validation. On ne peut pas attribuer une stagnation à un conflit directionnel précis.',
      '- Un taux de clipping ne donne pas une borne suffisante sur son biais.',
      '- Les 20 tours ne démontrent ni convergence ni plateau général. Ils ne permettent pas non plus de prédire les performances après recalibration DP pour un horizon plus long.',
      '- Quatre seeds et des comparaisons exploratoires ne suffisent pas à transformer de petites différences en supériorité générale.', '',
      '## 8. Suites non automatiques', '',
      'Lire cet audit avant toute promotion. Une étude de calibration éventuelle doit conserver les mêmes données, tirages, objective et budget, régler les pas uniquement sur de nouvelles seeds de validation, recalibrer le bruit si T change, et exclure les seeds finales déjà inspectées de la confirmation. Pas d’attaque, pas de récursion, pas de nouvelle référence ajoutée automatiquement.', '',
      '[Rapport initial](../Fair_Objective_Screen_V1_Status.md) · [Protocole figé](../Fair_Objective_Screen_V1_Protocol.md) · [Évidence numérique](evidence.json)', '']
    (DEST/'Audit_Optimisation_et_Clients.md').write_text('\n'.join(lines))
    print(json.dumps(dict(validated_runs=34,report=str(DEST/'Audit_Optimisation_et_Clients.md'),
                         evidence=str(DEST/'evidence.json'),new_training_runs=0),indent=2))


if __name__ == '__main__':
    assert avg_ranks([3,1,1,2]) == [4,1.5,1.5,3]
    assert abs(spearman([1,2,3],[3,2,1])+1)<1e-12
    assert correlation([1,1],[2,3]) is None
    audit()
