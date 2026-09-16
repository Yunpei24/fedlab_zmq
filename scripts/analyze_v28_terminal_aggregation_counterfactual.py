"""MPS-only, fixed-state mean/RFA intervention; never training or a gate."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
import torch
from scripts import run_fair_objective_screen as base
from scripts.analyze_full_population_error_attribution_v28b import source_evidence
from scripts.audit_full_population_error_attribution_v28b import polynomial_brier
from privacy.fair_objective import require_mps
from privacy.stable_weighted_rfa import weighted_rfa

SOURCE = ROOT/'results/ldp_gradient_far/full_population_private_risk_calibration_v28'
DIAG = ROOT/'results/ldp_gradient_far/full_population_error_attribution_v28b'
OUT = ROOT/'results/ldp_gradient_far/v28_terminal_aggregation_counterfactual'
DEST = ROOT/'output/analysis/Full_Population_V28_Terminal_Aggregation_Counterfactual'
PROTOCOL = DEST.with_name(DEST.name+'_Protocol').with_suffix('.md')
TEST = ROOT/'tests/test_v28_terminal_aggregation_counterfactual.py'
METRICS = ('accuracy_pct', 'worst20_pct', 'gap_best20_worst20_pp', 'variance_pp2', 'ce_loss', 'brier_loss')


def controls(messages, weights):
    require_mps()
    assert messages.device.type == weights.device.type == 'mps'
    assert messages.ndim == 2 and weights.shape == (len(messages),)
    assert bool(torch.isfinite(messages).all()) and bool(torch.isfinite(weights).all())
    assert float(weights.min()) > 0 and abs(float(weights.sum())-1) < 2e-6
    original, original_weights = messages.clone(), weights.clone()
    mean = (weights[:, None]*messages).sum(0)
    rfa, solver = weighted_rfa(messages, weights)
    assert torch.equal(messages, original) and torch.equal(weights, original_weights)
    return {'mean': mean, 'rfa': rfa}, solver


def objective(risks, method):
    if method.startswith('erm_'):
        return float(risks.mean())
    return float(torch.where(risks <= .5, risks+2*risks.square(), 3*risks-.5).mean())


@torch.no_grad()
def population_objective(model, data, method):
    risks = []
    for ids in data['train']:
        total = torch.zeros((), device='mps')
        for batch in ids.split(256):
            total += polynomial_brier(model(data['x'][batch]), data['y'][batch]).sum()/len(ids)
        risks.append(total)
    return objective(torch.stack(risks), method)


def run_one(job, profile, data, stamp, inputs):
    name = f"seed{job['seed']}__{job['method']}"
    folder = DIAG/name
    independent = ROOT/'output/analysis/audit_full_population_error_attribution_v28b'/f'{name}.json'
    if not (folder/'diagnostic.json').exists() or not independent.exists():
        return None
    d = json.loads((folder/'diagnostic.json').read_text())
    base.verify_stamp(d['source_stamp']); base.verify_stamp(d['input_stamp'])
    assert base.digest(folder/'oracle_vectors.pt') == d['vectors_sha256']
    audited = json.loads(independent.read_text())
    base.verify_stamp(audited['signature']['audit_stamp'])
    base.verify_stamp(audited['signature']['inputs'])
    dependencies = dict(inputs)
    for file in (folder/'diagnostic.json', folder/'oracle_vectors.pt', independent):
        dependencies[str(file.relative_to(ROOT))] = base.digest(file)
    signature = dict(source_stamp=stamp, inputs=dependencies)
    path = OUT/name/'diagnostic.json'
    if path.exists():
        cached = json.loads(path.read_text())
        assert cached['signature'] == signature
        return cached
    cp = torch.load(SOURCE/name/'checkpoint.pt', map_location='cpu', weights_only=True)
    vectors = torch.load(folder/'oracle_vectors.pt', map_location='cpu', weights_only=True)
    model = base.new_model(profile, job['seed'])
    model.load_state_dict(cp['pre_round_model'])
    before = base.evaluate(model, data, 'val')
    assert before == d['validation_before']
    before_j = population_objective(model, data, job['method'])
    assert abs(before_j-d['J_before']) < 4e-7
    messages = cp['last_private_messages'].to('mps')
    weights = torch.tensor(cp['rows'][-1]['aggregation']['objective_weights'], device='mps')
    candidates, solver = controls(messages, weights)
    target = vectors['target'].to('mps')
    gradient = target*vectors['objective_scale']
    eta = cp['rows'][-1]['aggregation']['eta']
    assert eta == .5
    rows = {}
    for kind, aggregate in candidates.items():
        model.load_state_dict(cp['pre_round_model'])
        with torch.no_grad():
            base.apply_gradient(model, aggregate, eta)
        after = base.evaluate(model, data, 'val')
        after_j = population_objective(model, data, job['method'])
        reproduced = kind == job['method'].split('_')[-1]
        if reproduced:
            assert torch.equal(eta*aggregate, cp['last_step'].to('mps'))
            assert all(torch.equal(v, cp['model'][k].to('mps')) for k, v in model.state_dict().items())
            assert after == cp['rows'][-1]['validation'] == d['validation_after']
            assert abs(after_j-d['J_after']) < 4e-7
        predicted = float(eta*torch.dot(gradient, aggregate))
        gain = before_j-after_j
        rows[kind] = dict(squared_error=float((aggregate-target).square().sum()),
                         predicted_objective_gain=predicted, actual_objective_gain=gain,
                         finite_step_remainder=predicted-gain, validation=after,
                         validation_delta={k: after[k]-before[k] for k in METRICS},
                         original_branch_reproduced=reproduced)
    difference = {k: rows['rfa'][k]-rows['mean'][k] for k in
                  ('squared_error', 'predicted_objective_gain', 'actual_objective_gain')}
    difference.update({k: rows['rfa']['validation'][k]-rows['mean']['validation'][k] for k in METRICS})
    result = dict(job=job, round=120, device='mps', signature=signature,
                  fixed_weights=True, fixed_messages=True, fixed_pre_state=True,
                  oracle_feeds_mechanism=False, test_evaluated=False, privacy_protected=False,
                  gate_changed=False, global_validation=False, J_before=before_j,
                  candidates=rows, rfa_minus_mean=difference, solver=solver)
    base.verify_stamp(dependencies); base.verify_stamp(stamp); base.save(path, result)
    print(f'{name}: same-message mean/RFA comparison, original branch bitwise PASS', flush=True)
    del model; torch.mps.empty_cache()
    return result


def write_report(records, stamp):
    dest = DEST.with_name(DEST.name+('' if len(records) == 8 else '_Partial'))
    base.save(dest.with_suffix('.json'), dict(records=records, audited_states=len(records), expected_states=8,
              source_stamp=stamp, global_validation=False, independent_seeds=2, gate_changed=False))
    lines = ['# V28 — RFA et moyenne appliquées aux mêmes messages privés', '',
             f'**{len(records)}/8 états pré-tour 120 audités. Intervention locale, pas une confirmation.**', '',
             'Dans chaque ligne, le modèle initial, les messages privés, les poids reçus et le pas '
             'sont identiques. La règle originale reproduit exactement le run. La seconde règle '
             'ne produit qu’un pas contrefactuel : elle ne réentraîne pas le modèle.', '',
             'Toutes les différences ci-dessous sont **RFA moins moyenne**. Une erreur² négative '
             'favorise RFA ; un gain de J, d’accuracy ou de Worst-20 positif favorise RFA. '
             'Un gap ou une variance négatifs favorisent RFA.', '',
             '| Seed | Trajectoire source | Δ erreur² | Δ gain réel J | Δ accuracy (pp) | Δ Worst-20 (pp) | Δ gap (pp) | Δ variance (pp²) |',
             '|--:|:--|--:|--:|--:|--:|--:|--:|']
    for r in records:
        x = r['rfa_minus_mean']
        values = [x[k] for k in ('squared_error', 'actual_objective_gain', 'accuracy_pct',
                                 'worst20_pct', 'gap_best20_worst20_pp', 'variance_pp2')]
        lines.append(f"| {r['job']['seed']} | {r['job']['method']} | "+' | '.join(f'{v:+.7g}' for v in values)+' |')
    lines += ['', 'J est la moyenne demi-Brier pour ERM et le potentiel équitable pour risque privé. '
              'L’erreur² vise le gradient honnête de population non clippé, avec poids oracle. '
              'Ces oracles servent seulement à l’analyse. Les erreurs sont celles d’une réalisation, '
              'pas des MSE moyennées sur le bruit.', '',
              'Les huit états ne sont pas indépendants : deux seeds connues, quatre trajectoires '
              'par seed. Un effet local ne prédit pas le résultat end-to-end, et aucun attaquant '
              'n’est présent. Ce diagnostic ne change ni la candidate ni les critères V28.', '',
              '[Protocole](Full_Population_V28_Terminal_Aggregation_Counterfactual_Protocol.md) · '
              f'[Résultats détaillés]({dest.name}.json).']
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def main(partial):
    require_mps(); OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'diagnostic.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = json.loads((SOURCE/'manifest.json').read_text())
        stamp = dict(original['source_stamp']); base.verify_stamp(stamp)
        extras = (Path(__file__), PROTOCOL, TEST, SOURCE/'manifest.json',
                  ROOT/'scripts/analyze_full_population_error_attribution_v28b.py',
                  ROOT/'scripts/audit_full_population_error_attribution_v28b.py')
        stamp.update({str(p.relative_to(ROOT)): base.digest(p) for p in extras})
        manifest = dict(source_stamp=stamp, expected_states=8, states='pre120',
                        alternatives=['mean', 'rfa'], training=False, gate_changed=False)
        if (OUT/'manifest.json').exists():
            assert json.loads((OUT/'manifest.json').read_text()) == manifest
        else:
            base.save(OUT/'manifest.json', manifest)
        if not (OUT/'tests.json').exists():
            p = subprocess.run([sys.executable, '-m', 'pytest', str(TEST), '-q'], cwd=ROOT,
                               capture_output=True, text=True)
            base.save(OUT/'tests.json', dict(passed=p.returncode == 0, source_stamp=stamp,
                                           output=p.stdout+p.stderr))
        tests = json.loads((OUT/'tests.json').read_text())
        assert tests['passed'] and tests['source_stamp'] == stamp
        records = []
        for seed in (170501, 170502):
            jobs = [(dict(seed=seed, method=method), source_evidence(dict(seed=seed, method=method)))
                    for method in ('erm_mean', 'erm_rfa', 'risk_mean', 'risk_rfa')]
            if not any(inputs is not None for _, inputs in jobs):
                continue
            data = base.prepare(original['profile'], seed)
            for job, inputs in jobs:
                if inputs is None:
                    continue
                result = run_one(job, original['profile'], data, stamp, inputs)
                if result is not None:
                    records.append(result)
            del data; torch.mps.empty_cache()
        if not partial:
            assert len(records) == 8, 'Eight independently audited states required'
        write_report(records, stamp); base.verify_stamp(stamp)
        base.save(OUT/'status.json', dict(status='completed' if len(records) == 8 else 'partial',
                  audited_states=len(records), expected_states=8, device='mps', gate_changed=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--resume', required=True, action='store_true')
    p.add_argument('--partial', action='store_true')
    main(p.parse_args().partial)
