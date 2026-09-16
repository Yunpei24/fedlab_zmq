"""Exact rational cross-check of the frozen V28 calibration gate.

Only integer validation counts and scalar arithmetic; no model/tensor work.
No evaluation before all eight completed runs have passed the main audit.
A disagreement blocks promotion; this program never changes the primary gate.
"""
from fractions import Fraction as Q
from pathlib import Path
import hashlib
import json
import math
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/ldp_gradient_far/full_population_private_risk_calibration_v28'
OLD = ROOT/'results/ldp_gradient_far/recursive_private_risk_calibration_v19'
REPORT = ROOT/'output/analysis/Full_Population_Private_Risk_Calibration_V28_Analyse.json'
DEST = ROOT/'output/analysis/Full_Population_V28_Exact_Count_Gate_Audit'
SEEDS = (170501, 170502)
METHODS = ('erm_mean', 'erm_rfa', 'risk_mean', 'risk_rfa')
KEYS = ('accuracy_pct', 'worst20_pct', 'gap_best20_worst20_pp', 'variance_pp2')
CONTROLS = ('full_erm_mean', 'full_erm_rfa', 'v19_fresh_erm_mean', 'v19_fresh_erm_rfa')
TEST = ROOT/'tests/test_v28_exact_count_gate.py'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_metrics(clients):
    if len(clients) != 10:
        raise ValueError('The fixed cohort has ten clients')
    accuracies = []
    for row in clients:
        counts, hits = row['class_count'], row['class_hits']
        if len(counts) != 10 or len(hits) != 10 or row['N'] != 1200:
            raise ValueError('The fixed validation split has 1200 examples per client')
        if any(not math.isfinite(x) or x < 0 or int(x) != x for x in counts+hits):
            raise ValueError('Finite integer counts required')
        if sum(counts) != 1200 or any(h > n for h, n in zip(hits, counts)):
            raise ValueError('Invalid per-class counts')
        accuracies.append(Q(100*int(sum(hits)), 1200))
    mean = sum(accuracies)/10
    ordered = sorted(accuracies)
    worst = sum(ordered[:2])/2
    return dict(accuracy_pct=mean, worst20_pct=worst,
                gap_best20_worst20_pp=sum(ordered[-2:])/2-worst,
                variance_pp2=sum((x-mean)**2 for x in accuracies)/10)


def exact_decision(current, historical):
    if set(current) != {(s, m) for s in SEEDS for m in METHODS}:
        raise ValueError('All eight unique current endpoints are required')
    if set(historical) != {(s, m) for s in SEEDS for m in ('erm_mean', 'erm_rfa')}:
        raise ValueError('All four historical controls are required')
    output = []
    for name in CONTROLS:
        table = current if name.startswith('full_') else historical
        method = name.removeprefix('full_').removeprefix('v19_fresh_')
        differences = [{k: current[s, 'risk_rfa'][k]-table[s, method][k] for k in KEYS} for s in SEEDS]
        if any(not isinstance(v, Q) for d in differences for v in d.values()):
            raise ValueError('This cross-check requires rational count-derived metrics')
        gates = dict(
            all_seeds_worst20=all(d['worst20_pct'] >= 1 for d in differences),
            all_seeds_accuracy=all(d['accuracy_pct'] >= -1 for d in differences),
            mean_gap_nonincreasing=sum(d['gap_best20_worst20_pp'] for d in differences) <= 0,
            mean_variance_nonincreasing=sum(d['variance_pp2'] for d in differences) <= 0,
        )
        output.append(dict(control=name, differences=differences, gates=gates, passed=all(gates.values())))
    return output


def encoded(value):
    return dict(numerator=value.numerator, denominator=value.denominator, decimal=float(value))


def load_record(record, historical):
    seed, method = record['job']['seed'], record['job']['method']
    name = f'seed{seed}__fresh__{method}' if historical else f'seed{seed}__{method}'
    directory = (OLD if historical else SOURCE)/name
    path, status_path = directory/'metrics.json', directory/'orchestration_status.json'
    status = json.loads(status_path.read_text())
    assert status['status'] == 'completed' and status['metrics_sha256'] == digest(path)
    if historical:
        assert record['source'] == str(path) and record['sha256'] == digest(path)
    else:
        assert record['files']['metrics.json'] == digest(path)
        assert record['files']['orchestration_status.json'] == digest(status_path)
    assert record['endpoint_round'] == 120 and record['test_evaluated'] is False
    metrics = json.loads(path.read_text())
    assert metrics['device'] == 'mps' and metrics['final']['round'] == 120
    validation = metrics['final']['validation']
    exact = exact_metrics(validation['clients'])
    for key in KEYS:
        assert abs(float(exact[key])-validation[key]) < 1e-10, key
        assert abs(float(exact[key])-record['validation'][key]) < 1e-10, key
    return (seed, method), exact, {str(p.relative_to(ROOT)): digest(p) for p in (path, status_path)}


def main():
    if not REPORT.exists():
        raise RuntimeError('No final main audit; the exact gate cannot be evaluated yet')
    report = json.loads(REPORT.read_text())
    assert report['audit_passed'] and report['valid_runs'] == report['expected_runs'] == 8
    assert report['gate_evaluated'] and not report['global_validation']
    assert report['decision']['primary_method'] == 'risk_rfa'
    state = json.loads((SOURCE/'status.json').read_text())
    assert state['status'] == 'completed' and state['valid_runs'] == 8
    assert len(report['records']) == 8 and len(report['historical']) == 4
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_v28_exact_count_gate', '-v'],
                           cwd=ROOT, capture_output=True, text=True)
    test_evidence = dict(passed=tests.returncode == 0, output=tests.stdout+tests.stderr,
                         test_sha256=digest(TEST), source_sha256=digest(Path(__file__)))
    DEST.with_name(DEST.name+'_Tests').with_suffix('.json').write_text(
        json.dumps(test_evidence, indent=2, allow_nan=False)+'\n')
    if not test_evidence['passed']:
        raise RuntimeError('Exact-count boundary tests failed; no gate cross-check')
    current, historical, inputs = {}, {}, {}
    for prior, rows, destination in ((False, report['records'], current), (True, report['historical'], historical)):
        for row in rows:
            key, numbers, hashes = load_record(row, prior)
            assert key not in destination
            destination[key] = numbers; inputs.update(hashes)
    decisions = exact_decision(current, historical)
    reference = {r['control']: r for r in report['decision']['comparisons']}
    assert len(reference) == 4 and set(reference) == set(CONTROLS)
    mismatches, records = [], []
    for decision in decisions:
        name = decision['control']; old = reference[name]
        if decision['gates'] != old['gates'] or decision['passed'] != old['passed']:
            mismatches.append(name)
        assert [p['seed'] for p in old['pairs']] == list(SEEDS)
        for index, d in enumerate(decision['differences']):
            for key in KEYS:
                assert abs(float(d[key])-old['pairs'][index]['delta'][key]) < 1e-10
        for key in KEYS:
            values = [d[key] for d in decision['differences']]
            mean = sum(values)/2
            sd = math.sqrt(float((values[0]-values[1])**2/2))
            assert abs(float(mean)-old['summaries'][key]['mean']) < 1e-10
            assert abs(sd-old['summaries'][key]['sd']) < 1e-10
            assert old['summaries'][key]['confidence_interval'] is None
        records.append(dict(control=name, gates=decision['gates'], passed=decision['passed'],
                            pairs=[dict(seed=seed, delta={k: encoded(v) for k, v in d.items()})
                                   for seed, d in zip(SEEDS, decision['differences'])]))
    exact_passed = all(d['passed'] for d in decisions)
    if exact_passed != report['decision']['calibration_passed']:
        mismatches.append('overall_verdict')
    result = dict(exact_calibration_passed=exact_passed, agrees_with_main_audit=not mismatches,
                  disagreements=mismatches, comparisons=records, expected_current_runs=8,
                  historical_controls=4, source_sha256=digest(Path(__file__)),
                  test_evidence=test_evidence,
                  input_stamp={**inputs, str(REPORT.relative_to(ROOT)): digest(REPORT)},
                  global_validation=False, primary_gate_overridden=False, statistical_confirmation=False)
    DEST.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# V28 — vérification exacte des critères de calibration', '',
             'Les quatre métriques primaires sont recalculées comme fractions à partir des '
             'nombres entiers de prédictions correctes de chaque client. Aucun modèle n’est '
             'réentraîné et aucune métrique de test n’est évaluée. Les critères restent : '
             'Δ Worst-20 ≥ 1 pp et Δ accuracy ≥ −1 pp sur chaque seed, puis moyennes '
             'des Δ gap et variance ≤ 0, contre chacun des quatre contrôles.', '',
             f'Accord avec l’audit principal : **{"oui" if not mismatches else "NON — promotion interdite"}**. '
             f'Gate exact de calibration : **{"PASS" if exact_passed else "FAIL"}**.', '',
             '| Contrôle | Seed | Δ accuracy (pp) | Δ Worst-20 (pp) | Δ gap (pp) | Δ variance (pp²) | Gate complet |',
             '|:--|--:|--:|--:|--:|--:|:--|']
    for r in records:
        for p in r['pairs']:
            values = [p['delta'][k]['decimal'] for k in KEYS]
            lines.append(f"| {r['control']} | {p['seed']} | "+' | '.join(f'{v:+.9f}' for v in values)+
                         (' | PASS |' if r['passed'] else ' | FAIL |'))
    lines += ['', 'Les fractions exactes figurent dans le JSON ; les décimales du tableau sont '
              'arrondies pour lecture. Ce contrôle n’améliore pas la puissance statistique et '
              'ne transforme pas deux seeds connues en confirmation. Un désaccord avec '
              'l’implémentation principale bloque la promotion ; aucun seuil n’est ajusté.', '',
              '[Protocole figé](Full_Population_Private_Risk_Calibration_V28_Protocol.md) · '
              '[Critères et audit principal](Full_Population_Private_Risk_Calibration_V28_Analyse.md) · '
              '[Fractions et empreintes](Full_Population_V28_Exact_Count_Gate_Audit.json).']
    DEST.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(exact_calibration_passed=exact_passed, agrees_with_main_audit=not mismatches)))
    if mismatches:
        raise RuntimeError('Exact-count and floating-point decisions disagree; no promotion')


if __name__ == '__main__':
    main()
