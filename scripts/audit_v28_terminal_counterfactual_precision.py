"""Independent arithmetic-sensitivity check of V28 terminal objective gains.

All model/tensor calculations use MPS. math.fsum only accumulates scalar batch
totals. Two blockings are numerical checks, not statistical repetitions or CIs.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
import torch
from scripts import run_fair_objective_screen as base
from privacy.fair_objective import require_mps
from privacy.stable_weighted_rfa import weighted_rfa

SOURCE = ROOT/'results/ldp_gradient_far/full_population_private_risk_calibration_v28'
DIAG = ROOT/'results/ldp_gradient_far/v28_terminal_aggregation_counterfactual'
CACHE = ROOT/'output/analysis/audit_v28_terminal_counterfactual_precision'
DEST = ROOT/'output/analysis/Full_Population_V28_Terminal_Counterfactual_Numerical_Audit'
TEST = ROOT/'tests/test_v28_terminal_counterfactual_precision.py'
BLOCKS = (128, 512)


def squared_brier(logits, labels):
    assert logits.device.type == labels.device.type == 'mps'
    probs = logits.softmax(-1)
    target = torch.nn.functional.one_hot(labels, logits.shape[-1]).to(probs.dtype)
    return .5*(probs-target).square().sum(-1)


def scalar_potential(risk, fair):
    if not fair:
        return risk
    return risk+2*risk*risk if risk <= .5 else 3*risk-.5


@torch.no_grad()
def evaluate_objective(model, data, method, block):
    require_mps()
    assert block in BLOCKS and not model.training
    assert all(p.device.type == 'mps' for p in model.parameters())
    risks = []
    for ids in data['train']:
        totals = [float(squared_brier(model(data['x'][part]), data['y'][part]).sum())
                  for part in ids.split(block)]
        risks.append(math.fsum(totals)/len(ids))
    value = math.fsum(scalar_potential(r, method.startswith('risk_')) for r in risks)/len(risks)
    assert math.isfinite(value)
    return value


def audit_one(job, profile, data, stamp):
    name = f"seed{job['seed']}__{job['method']}"
    path = DIAG/name/'diagnostic.json'
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    assert record['job'] == job and record['round'] == 120 and record['device'] == 'mps'
    base.verify_stamp(record['signature']['source_stamp'])
    base.verify_stamp(record['signature']['inputs'])
    signature = dict(auditor_source_stamp=stamp, source_diagnostic_sha256=base.digest(path))
    cached = CACHE/f'{name}.json'
    if cached.exists():
        previous = json.loads(cached.read_text())
        assert previous['signature'] == signature
        return previous
    cp = torch.load(SOURCE/name/'checkpoint.pt', map_location='cpu', weights_only=True)
    messages = cp['last_private_messages'].to('mps')
    weights = torch.tensor(cp['rows'][-1]['aggregation']['objective_weights'], device='mps')
    # Same actual operators as the mechanism, but not the diagnostic's helper.
    aggregate = dict(mean=(weights[:, None]*messages).sum(0), rfa=weighted_rfa(messages, weights)[0])
    model = base.new_model(profile, job['seed'])
    eta = cp['rows'][-1]['aggregation']['eta']; assert eta == .5
    checks = []
    max_difference = 0.
    for block in BLOCKS:
        model.load_state_dict(cp['pre_round_model'])
        initial = evaluate_objective(model, data, job['method'], block)
        max_difference = max(max_difference, abs(initial-record['J_before']))
        gains = {}
        for kind in ('mean', 'rfa'):
            model.load_state_dict(cp['pre_round_model'])
            with torch.no_grad():
                offset = 0
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        count = parameter.numel()
                        parameter.sub_((eta*aggregate[kind][offset:offset+count]).view_as(parameter))
                        offset += count
                assert offset == aggregate[kind].numel()
            if job['method'].endswith(kind):
                assert all(torch.equal(v, cp['model'][k].to('mps')) for k, v in model.state_dict().items())
            final = evaluate_objective(model, data, job['method'], block)
            gains[kind] = initial-final
            max_difference = max(max_difference, abs(gains[kind]-record['candidates'][kind]['actual_objective_gain']))
        checks.append(dict(block_size=block, J_before=initial, gains=gains,
                           delta_gain_rfa_minus_mean=gains['rfa']-gains['mean']))
    # Existing V28b verification tolerance; not a confidence level or effect gate.
    assert max_difference < 4e-7, max_difference
    values = [x['delta_gain_rfa_minus_mean'] for x in checks]
    reported = record['rfa_minus_mean']['actual_objective_gain']
    result = dict(job=job, signature=signature, device='mps', checks=checks,
                  original_reported_delta=reported,
                  numerical_recalculation_min=min(values), numerical_recalculation_max=max(values),
                  maximum_objective_or_gain_discrepancy=max_difference,
                  sign_same_as_report=all((v > 0) == (reported > 0) and (v < 0) == (reported < 0) for v in values),
                  statistical_repetitions=False, error_bound_certificate=False,
                  global_validation=False, gate_changed=False, test_evaluated=False)
    base.verify_stamp(stamp); base.save(cached, result)
    print(f'{name}: independent one-hot Brier, block128/512 and original model replay PASS', flush=True)
    del model; torch.mps.empty_cache()
    return result


def main(partial):
    require_mps(); CACHE.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE/'manifest.json').read_text())
    stamp = dict(manifest['source_stamp']); base.verify_stamp(stamp)
    for p in (Path(__file__), TEST, SOURCE/'manifest.json', DIAG/'manifest.json'):
        stamp[str(p.relative_to(ROOT))] = base.digest(p)
    test_path = CACHE/'tests.json'
    if not test_path.exists():
        run = subprocess.run([sys.executable, '-m', 'pytest', str(TEST), '-q'], cwd=ROOT,
                             capture_output=True, text=True)
        base.save(test_path, dict(passed=run.returncode == 0, source_stamp=stamp, output=run.stdout+run.stderr))
    tests = json.loads(test_path.read_text()); assert tests['passed'] and tests['source_stamp'] == stamp
    records = []
    for seed in (170501, 170502):
        methods = [m for m in ('erm_mean', 'erm_rfa', 'risk_mean', 'risk_rfa')
                   if (DIAG/f'seed{seed}__{m}'/'diagnostic.json').exists()]
        if not methods:
            continue
        data = base.prepare(manifest['profile'], seed)
        for method in methods:
            records.append(audit_one(dict(seed=seed, method=method), manifest['profile'], data, stamp))
        del data; torch.mps.empty_cache()
    if not partial:
        assert len(records) == 8
    dest = DEST.with_name(DEST.name+('' if len(records) == 8 else '_Partial'))
    base.save(dest.with_suffix('.json'), dict(records=records, verified_states=len(records),
              expected_states=8, source_stamp=stamp, numerical_check_only=True, gate_changed=False))
    lines = ['# V28 — contrôle numérique indépendant des écarts de loss', '',
             f'**{len(records)}/8 états vérifiés sur MPS.**', '',
             'Le calcul utilise la distance au vecteur one-hot au lieu de l’expression polynomiale '
             'du Brier, des blocs de 128 puis 512 exemples au lieu de 256, et une sommation '
             'compensée des totaux scalaires. Les deux modèles après pas sont reconstruits '
             'depuis le même état ; le bras original reproduit bit à bit le modèle enregistré.', '',
             '| Seed | Source | Δ gain J publié (RFA − moyenne) | Recalcul blocs 128 | Recalcul blocs 512 | Écart maximal J/gain | Signe conservé |',
             '|--:|:--|--:|--:|--:|--:|:--|']
    for r in records:
        values = [r['original_reported_delta'], *[x['delta_gain_rfa_minus_mean'] for x in r['checks']],
                  r['maximum_objective_or_gain_discrepancy']]
        lines.append(f"| {r['job']['seed']} | {r['job']['method']} | "+' | '.join(f'{v:+.9g}' for v in values)+
                     (' | oui |' if r['sign_same_as_report'] else ' | non |'))
    lines += ['', 'Les deux tailles de bloc ne sont pas des réplications statistiques. Leur plage '
              'n’est ni un IC ni une borne certifiée d’arrondi global. Un signe conservé ne '
              'prouve pas que l’effet est substantiel, significatif ou qu’il se généralise. '
              'Les écarts très petits des contrôles ERM ne sont pas promus en avantage scientifique.', '',
              '[Protocole de l’intervention](Full_Population_V28_Terminal_Aggregation_Counterfactual_Protocol.md) · '
              f'[Détails numériques]({dest.name}.json).']
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    base.verify_stamp(stamp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--partial', action='store_true')
    main(parser.parse_args().partial)
