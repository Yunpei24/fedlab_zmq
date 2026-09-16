"""Descriptive scalar audit of saved validation counts; no training or test access.

This is not another gate. Only completed independently audited V28 runs enter.
The public reports expose research validation, not the certified DP transcript.
"""
from pathlib import Path
import hashlib
import json
import math
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/ldp_gradient_far/full_population_private_risk_calibration_v28'
OUT = ROOT / 'output/analysis'
METHODS = ('erm_mean', 'erm_rfa', 'risk_mean', 'risk_rfa')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(clients):
    if not clients:
        raise ValueError('Empty cohort')
    k = len(clients[0]['class_count'])
    if not k:
        raise ValueError('No classes')
    supports, hits, local_ba, accuracies = [0]*k, [0]*k, [], []
    small_cells = present_cells = 0
    for row in clients:
        ns, hs = row['class_count'], row['class_hits']
        if len(ns) != k or len(hs) != k:
            raise ValueError('Inconsistent class dimensions')
        if any(not math.isfinite(v) or v < 0 or v != int(v) for v in ns + hs):
            raise ValueError('Counts must be finite nonnegative integers')
        if sum(ns) != row['N'] or row['N'] <= 0 or any(h > n for n, h in zip(ns, hs)):
            raise ValueError('Invalid counts')
        local_ba.append(100*st.mean(h/n for n, h in zip(ns, hs) if n))
        accuracies.append(100*sum(hs)/row['N'])
        for j, (n, h) in enumerate(zip(ns, hs)):
            supports[j] += int(n)
            hits[j] += int(h)
            present_cells += n > 0
            small_cells += 0 < n <= 5
    recalls = [100*h/n if n else None for n, h in zip(supports, hits)]
    tail = max(1, math.ceil(.2*len(clients)))
    ordered = sorted(accuracies)
    return dict(
        client_accuracy_pct=st.mean(accuracies),
        pooled_accuracy_pct=100*sum(hits)/sum(supports),
        mean_client_balanced_accuracy_pct=st.mean(local_ba),
        pooled_class_balanced_accuracy_pct=st.mean(x for x in recalls if x is not None),
        worst20_pct=st.mean(ordered[:tail]),
        gap_best20_worst20_pp=st.mean(ordered[-tail:])-st.mean(ordered[:tail]),
        variance_pp2=st.pvariance(accuracies),
        class_support=supports, class_hits=hits, class_recall_pct=recalls,
        client_balanced_accuracy_pct=local_ba, client_accuracy_values_pct=accuracies,
        present_client_class_cells=present_cells, cells_with_1_to_5_examples=small_cells,
    )


def main():
    records = []
    for seed in (170501, 170502):
        for method in METHODS:
            job = f'seed{seed}__{method}'
            directory = RESULTS / job
            status_file = directory / 'orchestration_status.json'
            audit_file = OUT / 'audit_full_population_private_risk_v28' / f'{job}.json'
            if not status_file.exists() or not audit_file.exists():
                continue
            status = json.loads(status_file.read_text())
            if status['status'] != 'completed':
                continue
            audit = json.loads(audit_file.read_text())
            metrics_file = directory / 'metrics.json'
            mh = digest(metrics_file)
            assert mh == status['metrics_sha256'] == audit['record']['files']['metrics.json']
            assert audit['record']['endpoint_round'] == status['round'] == 120
            assert status['device'] == 'mps'
            assert digest(status_file) == audit['record']['files']['orchestration_status.json']
            metrics = json.loads(metrics_file.read_text())
            validation = metrics['final']['validation']
            stats = summarize(validation['clients'])
            for key in ('client_accuracy_pct', 'worst20_pct', 'gap_best20_worst20_pp', 'variance_pp2'):
                assert abs(stats[key]-validation[key]) < 1e-6, key
            # The original saved balanced metric was calculated in MPS float32.
            assert abs(stats['mean_client_balanced_accuracy_pct']-validation['balanced_accuracy_pct']) < 1e-4
            records.append(dict(job=job, seed=seed, method=method, metrics_sha256=mh,
                                audit_sha256=digest(audit_file), **stats))
    pairs = []
    for seed in (170501, 170502):
        by_method = {r['method']: r for r in records if r['seed'] == seed}
        for name in ('risk_mean', 'risk_rfa'):
            control = name.replace('risk', 'erm')
            if name not in by_method or control not in by_method:
                continue
            a, b = by_method[name], by_method[control]
            assert a['class_support'] == b['class_support']
            pairs.append(dict(seed=seed, candidate=name, control=control,
                delta={key: a[key]-b[key] for key in (
                    'client_accuracy_pct', 'worst20_pct', 'gap_best20_worst20_pp',
                    'variance_pp2', 'mean_client_balanced_accuracy_pct',
                    'pooled_class_balanced_accuracy_pct')},
                class_recall_change_pp=[x-y if x is not None and y is not None else None
                                        for x, y in zip(a['class_recall_pct'], b['class_recall_pct'])]))
    report = dict(valid_runs=len(records), expected_runs=8, source_sha256=digest(Path(__file__)),
                  gate_changed=False, confirmation=False, test_accessed=False,
                  scope='descriptive saved validation counts; outside certified transcript',
                  records=records, paired_comparisons=pairs)
    stem = OUT / 'Full_Population_V28_Validation_Class_Counts'
    stem.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = [
        '# V28 — équité entre clients et entre classes : distinguer les moyennes', '',
        f'**{len(records)}/8 résultats de validation terminés et audités. Diagnostic descriptif.**', '',
        'Le critère V28 reste inchangé : Worst-20, accuracy, gap et variance sur les deux seeds. '
        'Ce complément ne sélectionne ni méthode ni checkpoint et ne consulte pas le test.', '',
        'La balanced accuracy déjà enregistrée est la moyenne entre clients de leur rappel '
        'moyen sur les classes présentes. Une classe représentée par un seul exemple chez un '
        'client y reçoit le même poids qu’une classe fréquente de ce client. La balanced '
        'accuracy globale ci-dessous regroupe d’abord les comptages de tous les clients, '
        'puis moyenne les rappels des classes présentes globalement. Les deux statistiques '
        'répondent à des questions différentes ; aucune ne remplace le Worst-20.', '',
        '| Seed | Méthode | Acc. clients (%) | Worst-20 (%) | BA moyenne des clients (%) | BA classes regroupées (%) |',
        '|--:|:--|--:|--:|--:|--:|',
    ]
    for r in records:
        lines.append(f"| {r['seed']} | {r['method']} | {r['client_accuracy_pct']:.4f} | "
                     f"{r['worst20_pct']:.4f} | {r['mean_client_balanced_accuracy_pct']:.4f} | "
                     f"{r['pooled_class_balanced_accuracy_pct']:.4f} |")
    for p in pairs:
        a = next(r for r in records if r['seed'] == p['seed'] and r['method'] == p['candidate'])
        b = next(r for r in records if r['seed'] == p['seed'] and r['method'] == p['control'])
        lines += ['', f"## Seed {p['seed']} — {p['candidate']} contre {p['control']}", '',
            f"Il y a {a['present_client_class_cells']} cellules client–classe présentes, dont "
            f"{a['cells_with_1_to_5_examples']} avec seulement 1 à 5 exemples. Ce comptage décrit "
            'le support, pas une significativité statistique.', '',
            '| Classe (index Fashion-MNIST) | Exemples regroupés | Rappel contrôle (%) | Rappel candidate (%) | Écart (pp) |',
            '|--:|--:|--:|--:|--:|']
        for j, change in enumerate(p['class_recall_change_pp']):
            if change is not None:
                lines.append(f"| {j} | {a['class_support'][j]} | {b['class_recall_pct'][j]:.4f} | "
                             f"{a['class_recall_pct'][j]:.4f} | {change:+.4f} |")
    lines += ['', 'Une hausse de Worst-20 ne prouve donc pas que chaque classe ou chaque client '
              's’améliore. Ces trajectoires partagent données et réalisations aléatoires standardisées, '
              'mais leurs états divergent. Deux seeds connues ne sont pas une confirmation indépendante. '
              'Aucune attaque n’est évaluée ici.', '',
              '[Protocole fixe](Full_Population_Private_Risk_Calibration_V28_Protocol.md) · '
              '[Comptages et calculs](Full_Population_V28_Validation_Class_Counts.json).']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(valid_runs=len(records), comparisons=pairs), allow_nan=False))


if __name__ == '__main__':
    main()
