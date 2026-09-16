"""Post-confirmation descriptive analysis; no training, selection or new gate.

Reads the completed independent MPS audit and integer evaluation counts.
CPU scalar arithmetic only. Never reads the simulation key or changes V29/V30.
"""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/analysis'
AUDIT = OUT / 'Full_Population_Private_Risk_Confirmation_V29_Analyse.json'
LEDGER = OUT / 'Private_Fairness_Robustness_Confirmation_Ledger_V29.json'
DEST = OUT / 'Full_Population_V29_Completion_Diagnostic'
SEEDS = (180701, 180702, 180703, 180704)
ARMS = ((4800, 'erm_mean'), (4800, 'erm_rfa'), (4800, 'risk_mean'),
        (4800, 'risk_rfa'), (240, 'erm_mean'), (240, 'erm_rfa'))
METRICS = ('accuracy_pct', 'worst20_pct', 'gap_best20_worst20_pp', 'variance_pp2',
           'balanced_accuracy_pct', 'ce_loss', 'brier_loss')
CLASSES = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
           'Shirt', 'Sneaker', 'Bag', 'Ankle boot')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    values = [float(v) for v in values]
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise ValueError('Four finite seed-level values required')
    return dict(mean=st.mean(values), sd=st.stdev(values), values=values)


def exact_view(view):
    """Class-pooled recalls differ from mean client-macro accuracy."""
    clients = view['clients']
    if len(clients) != 10:
        raise ValueError('Expected ten clients')
    acc, macro = [], []
    counts, hits = [0] * 10, [0] * 10
    for client in clients:
        c, h = client['class_count'], client['class_hits']
        if len(c) != 10 or len(h) != 10 or client['N'] != 1000:
            raise ValueError('Invalid test population')
        if any(not math.isfinite(x) or x != int(x) or x < 0 for x in c + h):
            raise ValueError('Invalid integer counts')
        if sum(c) != client['N'] or any(y > x for x, y in zip(c, h)):
            raise ValueError('Counts inconsistent')
        acc.append(Fraction(100 * int(sum(h)), client['N']))
        present = [Fraction(100 * int(y), int(x)) for x, y in zip(c, h) if x]
        macro.append(sum(present) / len(present))
        counts = [a + int(b) for a, b in zip(counts, c)]
        hits = [a + int(b) for a, b in zip(hits, h)]
    ordered = sorted(acc)
    mean = sum(acc) / 10
    result = dict(accuracy_pct=mean, worst20_pct=sum(ordered[:2]) / 2,
                  gap_best20_worst20_pp=(sum(ordered[-2:]) - sum(ordered[:2])) / 2,
                  variance_pp2=sum((a - mean)**2 for a in acc) / 10,
                  balanced_accuracy_pct=sum(macro) / 10,
                  ce_loss=view['ce_loss'], brier_loss=view['brier_loss'])
    for key, value in result.items():
        if not math.isfinite(float(value)) or not math.isfinite(view[key]) or abs(float(value) - view[key]) > (2e-5 if key == 'balanced_accuracy_pct' else 1e-9):
            raise ValueError(f'Counts/metric mismatch: {key}')
    recalls = [Fraction(100*h, n) if n else None for h, n in zip(hits, counts)]
    return result, recalls, counts


def contrast(index, left, right):
    return {k: summary(index[(s, *left)]['values'][k] - index[(s, *right)]['values'][k]
                       for s in SEEDS) for k in METRICS}


def decompose(index, control):
    """Exact telescoping; NOT a causal mediation decomposition."""
    result = []
    for seed in SEEDS:
        w = lambda arm: index[(seed, *arm)]['values']['worst20_pct']
        total = w((4800, 'risk_rfa')) - w((240, control))
        risk = w((4800, 'risk_rfa')) - w((4800, control))
        batch = w((4800, control)) - w((240, control))
        if total != risk + batch:
            raise ValueError('Telescoping identity failed')
        result.append(dict(seed=seed, total=float(total), within_full_batch=float(risk),
                           erm_batch_change=float(batch)))
    return result


def build():
    audit_hash, ledger_hash = sha(AUDIT), sha(LEDGER)
    audit = json.loads(AUDIT.read_text())
    if not (audit['audit_passed'] and audit['valid_runs'] == audit['expected_runs'] == 24
            and audit['gate_evaluated'] and not audit['global_validation']):
        raise ValueError('Require complete verified V29 audit')
    if audit['decision']['clean_confirmation_passed'] or audit['attacks_evaluated']:
        raise ValueError('This diagnostic is for the recorded negative clean confirmation')
    records = audit['records']
    if len(records) != 24:
        raise ValueError('Missing records')
    index, input_hashes = {}, {str(AUDIT.relative_to(ROOT)): audit_hash,
                               str(LEDGER.relative_to(ROOT)): ledger_hash}
    for row in records:
        job = row['job']
        key = (job['seed'], job['batch'], job['method'])
        if key in index:
            raise ValueError('Duplicate run')
        directory = Path(row['source']).parent
        for name, expected in row['files'].items():
            path = directory / name
            if sha(path) != expected:
                raise ValueError(f'Post-audit modification: {path}')
            input_hashes[str(path.relative_to(ROOT))] = expected
        status = json.loads((directory / 'orchestration_status.json').read_text())
        if status['status'] != 'completed' or status['device'] != 'mps':
            raise ValueError('Invalid run status/device')
        values, recalls, counts = exact_view(row['test'])
        index[key] = dict(values=values, recalls=recalls, counts=counts, row=row)
    if set(index) != {(s, *a) for s in SEEDS for a in ARMS}:
        raise ValueError('Wrong run matrix')
    for seed in SEEDS:
        if any(index[(seed, *arm)]['counts'] != index[(seed, *ARMS[0])]['counts'] for arm in ARMS):
            raise ValueError('Class populations not paired')
        per_client = lambda arm: [c['class_count'] for c in index[(seed, *arm)]['row']['test']['clients']]
        if any(per_client(arm) != per_client(ARMS[0]) for arm in ARMS):
            raise ValueError('Client class populations not paired')
    grouped = []
    for batch, method in ARMS:
        group = [index[(s, batch, method)] for s in SEEDS]
        grouped.append(dict(batch=batch, method=method,
            metrics={k: summary(r['values'][k] for r in group) for k in METRICS},
            elapsed_minutes=summary(r['row']['elapsed_seconds'] / 60 for r in group),
            clipping=summary(r['row']['median_clipping_fraction'] for r in group),
            concentration=summary(r['row']['median_objective_concentration'] for r in group),
            gradient_examples_per_client=120 * batch))
    contrasts = {
        'risk_vs_erm_under_mean': contrast(index, (4800, 'risk_mean'), (4800, 'erm_mean')),
        'risk_vs_erm_under_rfa': contrast(index, (4800, 'risk_rfa'), (4800, 'erm_rfa')),
        'rfa_vs_mean_under_risk': contrast(index, (4800, 'risk_rfa'), (4800, 'risk_mean')),
        'rfa_vs_mean_under_erm': contrast(index, (4800, 'erm_rfa'), (4800, 'erm_mean')),
    }
    classes = []
    for i, label in enumerate(CLASSES):
        for control in ('erm_rfa', 'risk_mean'):
            values = [index[(s, 4800, 'risk_rfa')]['recalls'][i] -
                      index[(s, 4800, control)]['recalls'][i] for s in SEEDS]
            classes.append(dict(class_index=i, label=label, control=control, delta=summary(values)))
    result = dict(analysis_type='posthoc_descriptive_no_new_confirmation', seeds=SEEDS,
        audit_sha256=audit_hash, ledger_sha256=ledger_hash, inputs=input_hashes,
        groups=grouped, contrasts=contrasts, class_recall_deltas=classes,
        worst20_telescoping={c: decompose(index, c) for c in ('erm_mean', 'erm_rfa')},
        original_decision=audit['decision'], no_new_training=True, no_new_threshold=True,
        attacks_opened=False, global_validation=False)
    ledger = json.loads(LEDGER.read_text())
    wave = next(w for w in ledger['waves'] if w['wave'] == 1)
    wave.update(complete_runs=24, independent_audit_passed=True, clean_confirmation_passed=False,
                joint_objective_validated=False, attacks_evaluated=False,
                completion_evidence=AUDIT.name, completion_evidence_sha256=audit_hash)
    ledger.update(completion_snapshot_of=LEDGER.name, registration_snapshot_sha256=ledger_hash,
                  no_new_wave_registered=True)
    # Verify again before producing derived artifacts; never modify frozen inputs.
    if any(sha(ROOT / p) != digest for p, digest in input_hashes.items()):
        raise ValueError('Inputs changed while diagnosing')
    return result, ledger


def fmt(s):
    return f"{s['mean']:.4f} ± {s['sd']:.4f}"


def render(r):
    lines = ['# V29 terminé — diagnostic des limites, sans changement du verdict', '',
        '24/24 entraînements MPS, quatre seeds réservées, audit indépendant terminé. '
        '**Confirmation propre FAIL ; validation conjointe non acquise ; V30 non ouvert.**', '',
        'Les résultats ci-dessous sont des diagnostics descriptifs après confirmation. '
        'Ils ne changent ni candidate, ni seuil, ni endpoint (test au tour 120). '
        'Moyenne ± écart-type échantillonnal sur quatre seeds ; aucun nouvel IC confirmatoire.', '',
        '## 1. Résultats et coût', '',
        'ERM signifie demi-Brier non pondérée, pas cross-entropie. '
        'Risque-RFA reste la candidate ; risque-moyenne reste un témoin descriptif.', '',
        '| Batch | Méthode | Accuracy (%) | Worst-20 (%) | Gap B20–W20 (pp) | Variance (pp²) |',
        '|--:|:--|--:|--:|--:|--:|']
    for g in r['groups']:
        lines.append(f"| {g['batch']} | {g['method']} | " + ' | '.join(fmt(g['metrics'][k]) for k in METRICS[:4]) + ' |')
    lines += ['', '| Batch | Méthode | Balanced accuracy clients (%) | CE | Demi-Brier | Minutes/run |',
              '|--:|:--|--:|--:|--:|--:|']
    for g in r['groups']:
        lines.append(f"| {g['batch']} | {g['method']} | " + ' | '.join(fmt(g['metrics'][k]) for k in METRICS[4:]) + f" | {fmt(g['elapsed_minutes'])} |")
    lines += ['', 'Les temps sont observés, non un benchmark de calcul contrôlé. '
        'Le batch complet utilise 576 000 gradients individuels/client contre 28 800 : facteur 20. '
        'Les rapports de risque ajoutent des évaluations ; 20 blocs par tour ne sont pas 20 pas locaux.', '',
        '## 2. D’où vient le gain propre ?', '',
        '| Contraste à batch complet | Δ accuracy (pp) | Δ Worst-20 (pp) | Δ gap (pp) | Δ variance (pp²) |',
        '|:--|--:|--:|--:|--:|']
    for label, c in r['contrasts'].items():
        lines.append('| ' + label + ' | ' + ' | '.join(fmt(c[k]) for k in METRICS[:4]) + ' |')
    lines += ['', '« risk_vs_erm » change la pondération ET ajoute le canal de risque, '
        'avec un léger recalibrage du bruit des gradients à budget total fixé. '
        '« rfa_vs_mean » change la règle appliquée sur toute la trajectoire. '
        'Ce ne sont pas des interventions à messages identiques : les modèles divergent.', '',
        '**Inférence limitée :** les contrastes localisent le gain propre principalement dans '
        'la branche de risque privé, et non dans un avantage massif de RFA sur la moyenne pondérée. '
        'L’absence d’attaque interdit toute conclusion sur l’avantage robuste de RFA.', '',
        '## 3. Pourquoi la comparaison au petit batch échoue-t-elle ?', '',
        'Identité descriptive exacte, appliquée à Worst-20 :', '',
        '`[risque-RFA complet − ERM petit] = [risque-RFA complet − ERM complet] + [ERM complet − ERM petit]`', '',
        '| Témoin ERM | Seed | Gain au même batch | Effet du changement de batch ERM | Gain total |',
        '|:--|--:|--:|--:|--:|']
    for c, pairs in r['worst20_telescoping'].items():
        for p in pairs:
            lines.append(f"| {c} | {p['seed']} | {p['within_full_batch']:+.4f} | {p['erm_batch_change']:+.4f} | {p['total']:+.4f} |")
    lines += ['', 'Cette identité n’isole pas causalement le bruit : le batch change aussi '
        'l’échantillonnage, le calcul et les trajectoires. Le bruit plus faible du batch complet '
        'ne garantit pas un meilleur Worst-20. Aucun choix de seed ni de checkpoint n’est autorisé.', '',
        'Les échecs préenregistrés sont conservés : +0,30 pp face à ERM-moyenne B=240 '
        'sur 180702 et +0,95 pp face à ERM-RFA B=240 sur 180704, sous la marge +1. '
        'Les deux bornes inférieures du gain moyen face à B=240 sont également négatives. '
        'Un résultat non confirmé n’est ni une preuve de gain nul, ni une validation.', '',
        '## 4. Les classes ne sont pas les clients', '',
        'Balanced accuracy clients = moyenne, sur clients, des rappels des classes présentes '
        'dans chaque partition. Rappel regroupé = total des prédictions correctes d’une classe '
        'divisé par son effectif sur tous les clients. Les deux pondérations diffèrent. '
        'Les petites classes locales peuvent rendre la première métrique très variable.', '',
        '| Classe | Témoin complet | Δ rappel regroupé (pp) | Δ par seed (ordre 180701…180704) |',
        '|:--|:--|--:|:--|']
    for c in r['class_recall_deltas']:
        lines.append(f"| {c['label']} | {c['control']} | {fmt(c['delta'])} | " + ', '.join(f'{x:+.2f}' for x in c['delta']['values']) + ' |')
    lines += ['', '**Observation importante :** face à ERM-RFA complet, le rappel Shirt '
        'augmente sur les quatre seeds (+10,35 pp en moyenne), mais le rappel Pullover '
        'baisse sur chacune (−7,525 pp). La balanced accuracy moyenne des clients baisse '
        'de 0,3332 pp en moyenne. Le mécanisme améliore donc le Worst-20 sans améliorer '
        'uniformément les classes ; ce compromis n’est pas une preuve de transfert de '
        'confusions Pullover vers Shirt, car aucune matrice de confusion n’est sauvegardée ici. '
        'Il faut un diagnostic dédié avant d’attribuer une cause.', '',
        '## 5. Décision de recherche', '',
        '- Conserver FAIL dans un nouveau registre de clôture ; ne pas modifier le registre initial.',
        '- Ne pas lancer V30 : son préalable de confirmation propre complète est absent.',
        '- Ne pas remplacer la candidate par risque-moyenne après lecture du test.',
        '- Ne pas attribuer le résultat au seul bruit, ni confondre fairness client et fairness par classe.',
        '- Avant une nouvelle confirmation, expliquer sur calibration le compromis batch/bruit/sampling '
        'et le coût honnête de RFA. Les nouvelles analyses de V29 sont exploratoires et ces seeds '
        'ne sont plus réservées pour une candidate modifiée.', '',
        'La piste reste une recherche active, pas une méthode robuste, équitable et privée validée. '
        'Une éventuelle nouvelle vague conserve les marges et utilise le niveau prospectif 0,00625, '
        'sans enregistrement automatique ici. Réduire les exigences pour obtenir PASS est exclu.', '',
        '## Traçabilité et portée', '',
        'Recalcul des métriques à partir des comptages entiers, appariement des populations de test, '
        'contrôle des empreintes de tous les fichiers audités et des statuts MPS. '
        'Cette analyse utilise uniquement de l’arithmétique scalaire sur résultats existants ; '
        'aucun entraînement CPU ni nouvelle requête aux données privées.', '',
        'Privacy : exemple côté client, replace-one, ε≤4 et δ=10⁻⁵ par exécution. '
        'Diagnostics de recherche, comptages de test, checkpoints et oracles exclus du transcript certifié. '
        'Les bras appariés ne sont pas une publication conjointe à ε=4. La comptabilité B=240 '
        'est la calibration historique générique, pas la version gaussienne affinée.', '',
        '[Audit primaire](Full_Population_Private_Risk_Confirmation_V29_Analyse.md) · '
        '[Protocole figé](Full_Population_Private_Risk_Confirmation_V29_Protocol.md) · '
        '[Données du diagnostic](Full_Population_V29_Completion_Diagnostic.json) · '
        '[Registre de clôture](Private_Fairness_Robustness_Confirmation_Ledger_V29_Completed.json)', '']
    return '\n'.join(lines)


def main():
    result, ledger = build()
    for path, obj in ((DEST.with_suffix('.json'), result),
            (OUT / 'Private_Fairness_Robustness_Confirmation_Ledger_V29_Completed.json', ledger)):
        text = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
        if path.exists() and path.read_text() != text:
            raise RuntimeError(f'Refusing to overwrite a different diagnostic: {path}')
        path.write_text(text)
    DEST.with_suffix('.md').write_text(render(result))
    print(json.dumps(dict(records=24, decision='FAIL preserved', new_training=False,
                         attacks_opened=False, report=str(DEST.with_suffix('.md')))))


if __name__ == '__main__':
    main()
