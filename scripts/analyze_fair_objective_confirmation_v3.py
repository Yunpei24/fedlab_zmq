#!/usr/bin/env python3
"""Read-only audit of frozen runs; writes only a separate analysis and evidence."""
import json
import math
from pathlib import Path
import re
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from scripts import run_fair_objective_confirmation_v3 as run

DEST = ROOT / 'output/analysis/fair_objective_confirmation_v3_analysis'
REPORT = ROOT / 'output/analysis/Fair_Objective_Confirmation_V3_Analyse.md'
LABELS = {'brier_fair_C1': 'Brier équitable, C=1, beta=2',
          'brier_erm_C1': 'Brier classique, C=1',
          'brier_erm_C2': 'Brier classique, C=2'}


def finite_tree(obj):
    if isinstance(obj, dict):
        for x in obj.values():
            finite_tree(x)
    elif isinstance(obj, list):
        for x in obj:
            finite_tree(x)
    elif isinstance(obj, float):
        assert math.isfinite(obj)


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-8, rel_tol=1e-8), (a, b)


def pair_stats(values):
    # Independently reproduce the frozen CI, without calling run.paired_summary.
    n = len(values)
    assert n == 4
    mean = sum(values) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
    half = 3.182446305284263 * sd / math.sqrt(n)
    return dict(mean=mean, sample_sd=sd, ci95_low=mean-half, ci95_high=mean+half,
                values=values, n=n)


def main():
    m = run.config()
    manifest = json.loads((run.OUT / 'manifest.json').read_text())
    stamp = manifest['source_stamp']
    run.base.verify_stamp(stamp)
    assert manifest['config'] == m
    status = json.loads((run.OUT / 'status.json').read_text())
    assert status['status'] == 'completed' and status['device'] == 'mps'
    assert status['valid_training_runs'] == status['valid_test_runs'] == 12
    tests = json.loads((run.OUT / 'tests.json').read_text())
    assert tests['passed'] and tests['source_stamp'] == stamp
    evidence = json.loads((run.OUT / 'evidence.json').read_text())
    finite_tree(evidence)
    assert evidence['source_stamp'] == stamp
    assert evidence['pairing_audit'] == run.audit_pairing(m, stamp)
    run.require_all_training(m, stamp)
    rows = []
    latest_training_time = max((run.OUT / run.cal.identifier(j) / 'metrics.json').stat().st_mtime
                               for j in run.jobs(m))
    for j in run.jobs(m):
        assert run.test_complete(j, stamp, m)
        d = run.OUT / run.cal.identifier(j)
        train = json.loads((d / 'metrics.json').read_text())
        test = json.loads((d / 'test_metrics.json').read_text())
        oracle = json.loads((d / 'simulator_oracle.json').read_text())
        finite_tree(train)
        finite_tree(test)
        finite_tree(oracle)
        assert test['evaluated_at'] >= latest_training_time
        exported = next(x for x in evidence['rows'] if (x['seed'], x['arm']) == (j['seed'], j['arm']))
        assert exported['test'] == test['test']
        assert exported['privacy'] == train['privacy']
        assert exported['test_sha256'] == run.base.digest(d / 'test_metrics.json')
        assert exported['training_sha256'] == run.base.digest(d / 'metrics.json')
        v = test['test']
        for c in v['clients']:
            assert sum(c['class_count']) == c['N']
            close(c['accuracy'], sum(c['class_hits']) / c['N'])
            recalls = [a/b for a, b in zip(c['class_hits'], c['class_count']) if b > 0]
            assert abs(c['balanced_accuracy'] - st.mean(recalls)) < 2e-7
        risks = [c['brier_loss'] for c in v['clients']]
        close(v['J_beta2'], st.mean(x+x*x for x in risks))
        close(v['J_beta2'], st.mean(risks) + st.mean(risks)**2 + st.pvariance(risks))
        assert [t['round'] for t in oracle['rounds']] == list(range(1, 61))
        client_diags = [c for t in oracle['rounds'] for c in t['clients']]
        p = train['privacy']
        close(p['sensitivity'], j['C'] * (2 + 3*j['beta']) / m['batch_size'])
        close(p['std'], p['z'] * p['sensitivity'])
        eta = train['parameters']['server_lr']
        val = {r['round']: r['validation'] for r in train['rounds'] if r['validation'] is not None}
        rows.append(dict(j, test=v, privacy=p, eta=eta,
            step_noise_per_coordinate=p['std'] * eta / math.sqrt(m['num_clients']),
            clip_pct=100*st.mean(c['clip_fraction'] for c in client_diags),
            batch_distortion=st.mean(c['distortion_relative'] for c in client_diags
                                    if c['distortion_relative'] is not None),
            aggregate_distortion_median=st.median(t['aggregate_clip']['distortion_relative'] for t in oracle['rounds']),
            noise_signal_median=st.median(t['noise_to_clean_signal'] for t in oracle['rounds']),
            coefficient_ratio_median=st.median(max(c['coefficient'] for c in t['clients']) /
                                              min(c['coefficient'] for c in t['clients']) for t in oracle['rounds']),
            val_acc_gain_40_60=val[60]['accuracy_pct'] - val[40]['accuracy_pct'],
            val_worst20_gain_40_60=val[60]['worst20_pct'] - val[40]['worst20_pct']))
    decision = run.decision(m, rows)
    assert decision == evidence['decision'] and decision['status'] == status['gate']
    comparisons = {}
    for ctrl in m['gate']['controls']:
        contrasts = {}
        for metric in run.METRICS + ['J_beta2']:
            diffs = []
            for seed in m['evaluation_seeds']:
                a = next(r for r in rows if r['seed'] == seed and r['arm'] == m['gate']['candidate'])
                b = next(r for r in rows if r['seed'] == seed and r['arm'] == ctrl)
                diffs.append(a['test'][metric] - b['test'][metric])
            contrasts[metric] = pair_stats(diffs)
            if metric in run.METRICS:
                for key in ['mean', 'sample_sd', 'ci95_low', 'ci95_high']:
                    close(contrasts[metric][key], decision['contrasts'][ctrl]['differences'][metric][key])
        comparisons[ctrl] = contrasts

    def group(arm):
        return [r for r in rows if r['arm'] == arm]

    def ms(xs, digits=2):
        return f'{st.mean(xs):.{digits}f} ± {st.stdev(xs):.{digits}f}'

    def ci(x):
        return f"{x['mean']:+.2f} [{x['ci95_low']:+.2f} ; {x['ci95_high']:+.2f}]"

    lines = ['# Confirmation Brier équitable v3 — analyse des 12 runs', '',
        '## 1. Verdict', '',
        '**Les 12 entraînements et les 12 tests finaux sont complets et valides, sur MPS. '
        'La configuration Brier équitable naïve (C=1, beta=2) ne confirme pas un avantage accuracy/fairness.** '
        'Les trois critères préenregistrés échouent face à chacun des deux contrôles. '
        'Aucune nouvelle phase n’a été lancée.', '',
        'Il ne s’agit pas seulement d’un intervalle trop large : l’accuracy baisse sur les quatre seeds '
        'face aux deux contrôles. Face au classique C=1, même la borne supérieure de l’IC du delta '
        'accuracy est inférieure à −1 point. Le Worst-20 baisse en moyenne, mais ses IC contiennent zéro : '
        'on ne conclut pas à une dégradation universelle de cette métrique.', '',
        '## 2. Protocole et vérifications', '',
        'Fashion-MNIST / LeNet-5 tanh ; 10 clients ; Dirichlet par client à tailles contrôlées, paramètre 0,1 ; '
        '4 800 exemples train, 1 200 validation et 1 000 test par client. '
        '60 tours, batch 240 sans remise au sein du tour, un gradient privé par tour ; aucune epoch locale. '
        'Moyenne uniforme au serveur, sans clipping serveur, sans attaque. Seeds : 170401, 170402, 170403, 170404.', '',
        'Les paramètres sont issus de la calibration v2 ; aucun nouveau réglage ni choix de checkpoint sur le test. '
        'Les 12 évaluations test ont eu lieu après les 12 entraînements, au tour 60 uniquement.', '',
        'Audit effectué : empreintes des sources et métriques, checkpoints de test, 60 tours consécutifs par run, '
        'valeurs finies, device MPS, budget, recomposition des métriques depuis les résultats clients et classes, '
        'splits et 600 indices de batch client-tour appariés par seed, initialisations évaluées identiques, '
        'et recalcul indépendant des différences et IC Student. Les 43 tests préalables étaient passés. '
        'L’appariement gaussien est garanti par le chemin de génération et ses clés communes, sans publier les tirages. '
        'Ce n’est pas une comparaison à gradients identiques après le premier tour.', '',
        '## 3. Résultats finaux test', '',
        'Moyenne ± écart-type échantillonnal sur **quatre seeds**, pas sur les clients. '
        'Worst-20 : moyenne des deux accuracies clientes les plus faibles. Gap : Best-20 moins Worst-20. '
        'Variance : dispersion population des dix accuracies, en pp².', '',
        '| Configuration | Accuracy (%) ↑ | Worst-20 (%) ↑ | Gap (pp) ↓ | Variance (pp²) ↓ | Balanced accuracy (%) ↑ |',
        '|:--|--:|--:|--:|--:|--:|']
    for arm in m['arms']:
        lines.append('| ' + LABELS[arm] + ' | ' + ' | '.join(ms([r['test'][k] for r in group(arm)])
            for k in run.METRICS[:5]) + ' |')
    lines += ['', '| Configuration | Loss CE ↓ | Demi-loss Brier ↓ | Risque équitable J2 ↓ |',
              '|:--|--:|--:|--:|']
    for arm in m['arms']:
        lines.append('| ' + LABELS[arm] + ' | ' + ' | '.join(ms([r['test'][k] for r in group(arm)], 4)
            for k in ['ce_loss', 'brier_loss', 'J_beta2']) + ' |')
    lines += ['', 'J2 = moyenne des (R_i + R_i²), où R_i est la demi-loss Brier test du client i. '
              'C’est la même cible descriptive pour toutes les méthodes, et non leur loss d’entraînement observée. '
              'Le candidat a un J2 **plus élevé sur chacune des quatre seeds face à chacun des deux contrôles**. '
              'Son échec ne vient donc pas seulement d’un choix de métrique accuracy différent de son objectif équitable.', '',
              'La balanced accuracy présentée est la moyenne, sur les clients, du rappel moyen de leurs classes présentes. '
              'Ce n’est pas une balanced accuracy globale recalculée après regroupement des classes.', '',
              '## 4. Comparaisons appariées et critère fixé', '',
              'Delta = candidat moins contrôle. IC95 bilatéral Student, trois degrés de liberté ; '
              'ces IC sont marginaux, pas simultanés. Avec quatre seeds, leur fiabilité dépend fortement '
              'de l’hypothèse de différences indépendantes et approximativement normales.', '',
              '| Contrôle | Delta accuracy, IC95 (pp) | Delta Worst-20, IC95 (pp) | Delta gap, IC95 (pp) |',
              '|:--|:--|:--|:--|']
    for ctrl, c in comparisons.items():
        lines.append('| '+LABELS[ctrl]+' | '+' | '.join(ci(c[k]) for k in
            ['accuracy_pct', 'worst20_pct', 'gap_best20_worst20_pp'])+' |')
    lines += ['', 'Pour passer : gain moyen Worst-20 ≥ 1 pp, borne inférieure IC Worst-20 > 0, '
              'et borne inférieure IC accuracy > −1 pp, **contre les deux contrôles**. '
              'Les six vérifications sont négatives ; aucune métrique secondaire ne remplace ce critère.', '',
              '| Seed | Delta acc. vs C=1 | Delta Worst-20 vs C=1 | Delta acc. vs C=2 | Delta Worst-20 vs C=2 |',
              '|--:|--:|--:|--:|--:|']
    for i, seed in enumerate(m['evaluation_seeds']):
        values = [comparisons[c][k]['values'][i] for c in m['gate']['controls'] for k in ['accuracy_pct', 'worst20_pct']]
        lines.append(f'| {seed} | '+' | '.join(f'{v:+.2f}' for v in values)+' |')
    lines += ['', 'La variance moyenne légèrement plus basse que C=1 (−2,22 pp²) ne constitue pas '
              'un gain de fairness robuste : elle augmente sur trois seeds sur quatre et baisse surtout sur la seed 170403. '
              'Le gap est légèrement plus élevé en moyenne, tandis que le Worst-20 et l’accuracy baissent. '
              'Le contrôle C=2 est lui aussi variable en Worst-20 : son mauvais résultat sur la seed 170401 '
              'ne doit pas être masqué par sa meilleure accuracy moyenne.', '',
              '## 5. Ce que montrent clipping, bruit et objectif', '',
              '### 5.1 Même budget, bruit effectif différent', '',
              'Tous les runs atteignent epsilon = 3,999999952 et delta = 10^-5, avec '
              'le même multiplicateur z = 1,318607635 et l’accountant générique sans remise WBK2019. '
              'Ce budget concerne les messages par client et par run, pas les oracles de recherche ni la publication cumulée des campagnes.', '',
              '| Configuration | Borne de sensibilité | Bruit upload, écart-type par coordonnée | Pas serveur | Bruit du pas agrégé, écart-type par coordonnée |',
              '|:--|--:|--:|--:|--:|']
    for arm in m['arms']:
        r = group(arm)[0]
        lines.append(f"| {LABELS[arm]} | {r['privacy']['sensitivity']:.6f} | {r['privacy']['std']:.6f} | {r['eta']:.6f} | {r['step_noise_per_coordinate']:.6f} |")
    lines += ['', 'La borne utilisée est 2C/b pour le classique et C(2+3 beta)/b pour le naïf. '
              'À C=1, le candidat ajoute donc **4 fois** l’écart-type de bruit à l’upload. '
              'Avec son pas 2/1,9 au lieu de 2, le bruit ajouté aux paramètres après moyenne uniforme '
              'a encore **2,105 fois** l’écart-type, soit **4,432 fois** la variance par coordonnée. '
              'L’écart-type de ce pas est eta × std(upload) / racine(10). '
              'Face à C=2, le rapport d’écart-type du pas n’est plus que 1,053 : le seul niveau absolu de bruit '
              'ne suffit donc pas à expliquer toutes les comparaisons.', '',
              'Il s’agit du coût de la borne de sensibilité **employée**, pas d’une preuve qu’aucune analyse DP '
              'plus fine ne puisse réduire ce coût. Le même epsilon n’impose pas le même bruit physique.', '',
              '### 5.2 Le clipping déforme encore le signal', '',
              'Pour chaque run : moyenne sur les 60 tours et les 10 clients du taux de clipping et de la '
              'distorsion relative de la moyenne de batch ; puis moyenne ± écart-type entre seeds. '
              'La distorsion relative est ||G_clippé − G_brut|| / ||G_brut||, **avant bruit**. '
              'Ce n’est pas le pourcentage de gradients clippés, ni la fraction de signal utile forcément perdue.', '',
              '| Configuration | Gradients individuels clippés (%) | Distorsion relative moyenne du batch | Rapport max/min des coefficients, médiane temporelle |',
              '|:--|--:|--:|--:|']
    for arm in m['arms']:
        lines.append('| '+LABELS[arm]+' | '+' | '.join(ms([r[k] for r in group(arm)], 3)
            for k in ['clip_pct', 'batch_distortion', 'coefficient_ratio_median'])+' |')
    lines += ['', 'Le candidat ne se comporte pas exactement comme le classique : les coefficients '
              '1+2 × loss moyenne du batch sont différenciés (rapport max/min médian ≈ 1,37). '
              'Mais cette différenciation n’apporte pas un meilleur modèle. Sa distorsion moyenne de batch '
              'est d’environ 0,46, contre 0,39 pour C=1 classique et 0,21 pour C=2 classique. '
              'Ces statistiques sont mesurées sur des trajectoires différentes : elles ne constituent pas '
              'une intervention causale isolant le clipping à modèle commun.', '',
              '### 5.3 Horizon : aucune garantie de convergence à 60 tours', '',
              '| Configuration | Gain validation accuracy entre 40 et 60 (pp) | Gain validation Worst-20 entre 40 et 60 (pp) |',
              '|:--|--:|--:|']
    for arm in m['arms']:
        lines.append('| '+LABELS[arm]+' | '+' | '.join(ms([r[k] for r in group(arm)])
            for k in ['val_acc_gain_40_60', 'val_worst20_gain_40_60'])+' |')
    lines += ['', 'Le candidat continue à améliorer un peu son accuracy validation, mais son Worst-20 '
              'validation baisse en moyenne entre 40 et 60. Les contrôles progressent encore. '
              'Cela ne prouve pas que prolonger résoudrait le problème. Une extension à epsilon fixe '
              'exigerait de recalibrer le bruit pour le nouvel horizon ; elle ne serait pas une simple '
              'prolongation gratuite de ces résultats.', '',
              '## 6. Observations, inférences et non-identifiables', '',
              '**Observations.** Aucun critère de confirmation ne passe. Accuracy inférieure sur 4/4 seeds '
              'face à chaque contrôle ; Worst-20 inférieur sur 3/4 ; losses CE/Brier et J2 supérieurs sur 4/4. '
              'Le coût DP appliqué est plus grand et la distorsion de clipping demeure forte.', '',
              '**Inférence raisonnable.** Dans cette instanciation, la pondération locale par le risque '
              'n’apporte pas un bénéfice suffisant pour compenser les coûts conjoints d’optimisation, '
              'de clipping et du bruit calibré. Le signal favorable aperçu sur deux seeds de calibration '
              'ne s’est pas confirmé sur quatre nouvelles seeds.', '',
              '**Non identifiable.** Cette campagne ne permet pas d’attribuer une fraction précise '
              'de l’échec au bruit, au pas serveur, au clipping ou au biais de l’estimateur naïf. '
              'Elle ne teste ni la version débiaisée dans ce réglage, ni un mécanisme sans DP apparié, '
              'ni une convergence asymptotique, ni les attaques byzantines. Les nouvelles seeds partagent '
              'le benchmark historique : elles ne démontrent pas un transfert à un autre dataset.', '',
              '## 7. Décision', '',
              '**Ne pas promouvoir cette configuration Brier équitable vers une campagne byzantine '
              'ou une revendication de méthode supérieure.** Conserver les deux Brier classiques comme '
              'contrôles et archiver ce résultat négatif documenté. Cela arrête cette branche expérimentale '
              'à ce réglage, pas la recherche fairness + robustesse + DP.', '',
              'La suite utile, si l’on revient sur cet objectif, doit commencer par une hypothèse précise '
              'sur un coût à réduire et un audit théorique du mécanisme, pas par une nouvelle grille '
              'choisie après consultation de ces tests. Les seeds de cette confirmation ne doivent plus '
              'être considérées comme un jeu de sélection intact.', '',
              '## Sources locales', '',
              '- [Protocole préenregistré](Fair_Objective_Confirmation_V3_Protocol.md).',
              '- [État et résultats par seed](Fair_Objective_Confirmation_V3_Status.md).',
              '- [Analyse de calibration v2](Fair_Objective_Calibration_V2_Analyse.md).',
              '- [Evidence de campagne](../../results/ldp_gradient_far/fair_objective_confirmation_v3/evidence.json).',
              '- [Audit complémentaire et diagnostics](fair_objective_confirmation_v3_analysis/evidence.json).']
    DEST.mkdir(parents=True, exist_ok=True)
    audit = dict(source_stamp=stamp, analysis_source_sha256=run.base.digest(Path(__file__)),
        valid_training=12, valid_test=12, tests_passed=True, source_integrity=True,
        test_after_all_training=True, client_and_class_metrics_recomputed=True,
        paired_intervals_independently_recomputed=True, pairing=evidence['pairing_audit'],
        rows=rows, contrasts=comparisons, decision=decision)
    run.base.save(DEST / 'evidence.json', audit)
    REPORT.write_text('\n'.join(lines)+'\n')
    for target in re.findall(r'\]\(([^)]+)\)', REPORT.read_text()):
        assert (REPORT.parent / target).resolve().exists(), target
    print(json.dumps(dict(report=str(REPORT), valid_runs=12, valid_tests=12,
                          decision=decision['status'], local_links_valid=True)))


if __name__ == '__main__':
    main()
