"""Bounded MPS single-step interventions on V28 calibration checkpoints only."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics as st
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
import torch
from scripts import run_fair_objective_screen as base
from scripts.analyze_full_population_error_attribution_v28b import source_evidence
from scripts.analyze_v28_terminal_aggregation_counterfactual import controls
from privacy.fair_objective import require_mps, per_example
from privacy.public_temporal_noise_confirmation_v25 import plan as small_plan

SOURCE = ROOT/'results/ldp_gradient_far/full_population_private_risk_calibration_v28'
VECTORS = ROOT/'results/ldp_gradient_far/full_population_error_attribution_v28b'
OUT = ROOT/'results/ldp_gradient_far/private_risk_batch_noise_diagnostic_v31'
DEST = ROOT/'output/analysis/Private_Risk_V31_Local_Batch_Noise_Diagnostic'
PROTOCOL = DEST.with_name(DEST.name+'_Protocol').with_suffix('.md')
TEST = ROOT/'tests/test_private_risk_batch_noise_v31.py'
SEEDS = (170501, 170502)
METRICS = ('accuracy_pct', 'worst20_pct', 'gap_best20_worst20_pp', 'variance_pp2',
           'balanced_accuracy_pct', 'ce_loss', 'brier_loss')


def components(aggregate, clean, target):
    require_mps()
    if any(t.device.type != 'mps' for t in (aggregate, clean, target)):
        raise ValueError('MPS only')
    bias, noise = clean-target, aggregate-clean
    error = float((aggregate-target).square().sum())
    bias2, noise2 = float(bias.square().sum()), float(noise.square().sum())
    cross = float(2*torch.dot(bias, noise))
    if abs(error-bias2-noise2-cross) > 3e-6 * max(1., error):
        raise ValueError('Error decomposition failed')
    return dict(error_squared=error, clean_error_squared=bias2,
                noise_displacement_squared=noise2, cross_term=cross)


@torch.no_grad()
def confusion(model, data):
    require_mps()
    result = torch.zeros((10, 10), device='mps', dtype=torch.int32)
    for ids in data['val']:
        for batch in ids.split(256):
            labels = data['y'][batch]
            predicted = model(data['x'][batch]).argmax(1)
            # Small fixed vocabulary: avoid unsupported MPS bincount fallback.
            for truth in range(10):
                for pred in range(10):
                    result[truth, pred] += ((labels == truth) & (predicted == pred)).sum().to(torch.int32)
    return result.cpu().tolist()


def validate_confusion(matrix, evaluation):
    counts = [sum(c['class_count'][k] for c in evaluation['clients']) for k in range(10)]
    hits = [sum(c['class_hits'][k] for c in evaluation['clients']) for k in range(10)]
    assert all(sum(row) == counts[k] and row[k] == hits[k] for k, row in enumerate(matrix))


def one_seed(seed, profile, stamp):
    job = dict(seed=seed, method='risk_rfa')
    name = f'seed{seed}__risk_rfa'
    inputs = source_evidence(job)
    assert inputs is not None, 'Independently audited parent required'
    diag = json.loads((VECTORS/name/'diagnostic.json').read_text())
    assert base.digest(VECTORS/name/'oracle_vectors.pt') == diag['vectors_sha256']
    audit_path = ROOT/'output/analysis/audit_full_population_error_attribution_v28b'/f'{name}.json'
    audited = json.loads(audit_path.read_text())
    assert audited['result']['audit_passed']
    base.verify_stamp(audited['signature']['inputs'])
    base.verify_stamp(audited['signature']['audit_stamp'])
    for p in (VECTORS/name/'diagnostic.json', VECTORS/name/'oracle_vectors.pt', audit_path):
        inputs[str(p.relative_to(ROOT))] = base.digest(p)
    signature = dict(stamp=stamp, inputs=inputs)
    data = base.prepare(profile, seed)
    cp = torch.load(SOURCE/name/'checkpoint.pt', map_location='cpu', weights_only=True)
    v = torch.load(VECTORS/name/'oracle_vectors.pt', map_location='cpu', weights_only=True)
    model = base.new_model(profile, seed)
    model.load_state_dict(cp['pre_round_model'])
    before = base.evaluate(model, data, 'val')
    assert before == diag['validation_before']
    full = cp['last_clean_means'].to('mps')
    torch.testing.assert_close(full, v['clipped_means'].to('mps'), rtol=6e-5, atol=4e-7)
    weights = v['received_weights'].to('mps')
    torch.testing.assert_close(weights, torch.tensor(cp['rows'][-1]['aggregation']['objective_weights'], device='mps'), rtol=0, atol=0)
    original, _ = controls(cp['last_private_messages'].to('mps'), weights)
    assert torch.equal(.5*original['rfa'], cp['last_step'].to('mps'))
    base.apply_gradient(model, original['rfa'], .5)
    assert all(torch.equal(value, cp['model'][k].to('mps')) for k, value in model.state_dict().items())
    assert base.evaluate(model, data, 'val') == cp['rows'][-1]['validation']
    target = v['target'].to('mps')
    gradient = target*v['objective_scale']
    low = json.loads((SOURCE/name/'metrics.json').read_text())['privacy']['gradient_std']
    sp = small_plan(0, 'risk_rfa')
    high = sp['sigma_early']
    assert high == sp['sigma_late'] and high > low > 0
    results = []
    for repetition in range(4):
        path = OUT/name/f'repetition{repetition}.json'
        if path.exists():
            cached = json.loads(path.read_text())
            assert cached['signature'] == signature and len(cached['rows']) == 12
            results.extend(cached['rows'])
            print(f'V31 seed {seed} repetition {repetition}: resumed 12/12', flush=True)
            continue
        model.load_state_dict(cp['pre_round_model'])
        small, draw_hashes = [], []
        for cid, population in enumerate(data['train']):
            ids = base.draw_indices(4800, 240, base.seed_for('v31-diagnostic', seed, repetition, cid, 'sampling'))
            assert len(ids.unique()) == 240
            draw_hashes.append(base.ids_hash(ids))
            batch = population[ids]
            _, rows, _, _ = per_example(model, data['x'][batch], data['y'][batch], clip_norm=2.)
            small.append(rows.mean(0))
        small = torch.stack(small)
        old_state = torch.mps.get_rng_state()
        try:
            torch.mps.manual_seed(base.seed_for('v31-diagnostic', seed, repetition, 'gaussian'))
            z = torch.randn(full.shape, device='mps')
        finally:
            torch.mps.set_rng_state(old_state)
        weighted_z = (weights[:, None]*z).sum(0)
        rows_out = []
        for batch, clean_messages in ((4800, full), (240, small)):
            clean_aggregates, _ = controls(clean_messages, weights)
            for noise_name, sigma in (('zero_oracle', 0.), ('full_plan', low), ('small_plan', high)):
                aggregates, solver = controls(clean_messages+sigma*z, weights)
                torch.testing.assert_close(aggregates['mean']-clean_aggregates['mean'], sigma*weighted_z,
                                           rtol=1e-4, atol=1e-7)
                for kind, aggregate in aggregates.items():
                    model.load_state_dict(cp['pre_round_model'])
                    base.apply_gradient(model, aggregate, .5)
                    evaluation = base.evaluate(model, data, 'val')
                    matrix = confusion(model, data)
                    validate_confusion(matrix, evaluation)
                    row = dict(seed=seed, repetition=repetition, batch=batch, noise=noise_name,
                        std=sigma, aggregation=kind, metrics=evaluation,
                        delta={k:evaluation[k]-before[k] for k in METRICS}, confusion=matrix,
                        projected_objective_gain=float(.5*torch.dot(gradient, aggregate)),
                        decomposition=components(aggregate, clean_aggregates[kind], target),
                        solver=solver if kind=='rfa' else None)
                    rows_out.append(row)
        base.verify_stamp(inputs); base.verify_stamp(stamp)
        base.save(path, dict(signature=signature, rows=rows_out, sampling_hashes=draw_hashes,
            device='mps', private_release=False, test_evaluated=False, fixed_weights=True,
            original_branch_bitwise=True, training=False))
        results.extend(rows_out)
        print(f'V31 seed {seed} repetition {repetition}: 12/12 virtual steps saved', flush=True)
        base.save(OUT/'status.json', dict(status='running', latest_seed=seed,
            latest_repetition=repetition, device='mps', no_confirmation_gate=True))
    del model, data, cp, v
    torch.mps.empty_cache()
    return results


def write_report(rows, stamp):
    assert len(rows) == 96
    result = dict(rows=rows, virtual_steps=96, calibration_seeds=SEEDS,
        independent_seeds=2, source_stamp=stamp, training=False, private_release=False,
        test_evaluated=False, global_validation=False, V29_changed=False, V30_opened=False)
    base.save(DEST.with_suffix('.json'), result)
    lines = ['# V31 — séparer batch et bruit sur des états de calibration', '',
        '**96/96 pas virtuels sur MPS ; deux modèles de calibration, pas 96 entraînements.**', '',
        'Chaque pas repart du modèle pré-tour 120 et conserve ses poids privés. '
        'Moyenne ± écart-type sur quatre répétitions conditionnelles au modèle ; '
        'pas d’IC de généralisation. Les répétitions sans bruit et batch complet sont identiques. '
        'Les deux modèles originaux ont été reproduits exactement avant intervention.', '',
        'Les amplitudes « full_plan » et « small_plan » sont celles des plans publics '
        'complet V28 et petit batch constant V25. Les croisements ne sont pas tous '
        'certifiés epsilon=4. « zero_oracle » est non privé. Toutes les évaluations '
        'sont sur validation, jamais sur le test de confirmation.', '',
        '| Seed | Batch | Bruit | Agrégateur | Δ accuracy (pp) | Δ Worst-20 (pp) | Erreur² vers gradient cible |',
        '|--:|--:|:--|:--|--:|--:|--:|']
    def fmt(values):
        return f'{st.mean(values):+.6g} ± {st.stdev(values):.3g}'
    for seed in SEEDS:
        for batch in (4800,240):
            for noise in ('zero_oracle','full_plan','small_plan'):
                for agg in ('mean','rfa'):
                    g=[r for r in rows if (r['seed'],r['batch'],r['noise'],r['aggregation'])==(seed,batch,noise,agg)]
                    lines.append(f'| {seed} | {batch} | {noise} | {agg} | '+
                        ' | '.join((fmt([r['delta']['accuracy_pct'] for r in g]),
                        fmt([r['delta']['worst20_pct'] for r in g]),
                        fmt([r['decomposition']['error_squared'] for r in g])))+' |')
    lines += ['', 'L’erreur² vise le gradient honnête équitable non clippé (oracle). '
        'Sa moyenne sur quatre tirages estime une MSE conditionnelle avec forte incertitude, '
        'pas une erreur d’accuracy. Matrices de confusion, metrics et décompositions '
        'par réalisation sont conservées dans le JSON.', '',
        'Aucun verdict de confirmation, aucun remplacement de la candidate et aucun '
        'lancement d’attaque. Ce diagnostic ne reconstruit pas les 120 tours ni le '
        'recalcul des poids. Un effet favorable au dernier pas ne démontre pas '
        'un gain end-to-end.', '',
        '[Protocole figé](Private_Risk_V31_Local_Batch_Noise_Diagnostic_Protocol.md) · '
        '[96 interventions détaillées](Private_Risk_V31_Local_Batch_Noise_Diagnostic.json)', '']
    DEST.with_suffix('.md').write_text('\n'.join(lines))


def main():
    require_mps()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'diagnostic.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent = json.loads((SOURCE/'manifest.json').read_text())
        stamp = dict(parent['source_stamp'])
        extras = [Path(__file__), TEST, PROTOCOL, SOURCE/'manifest.json',
            ROOT/'scripts/analyze_full_population_error_attribution_v28b.py',
            ROOT/'scripts/analyze_v28_terminal_aggregation_counterfactual.py',
            ROOT/'privacy/public_temporal_noise_confirmation_v25.py',
            ROOT/'privacy/public_temporal_noise_training_v24.py',
            ROOT/'privacy/public_temporal_noise_v23.py']
        stamp.update({str(p.relative_to(ROOT)):base.digest(p) for p in extras})
        base.verify_stamp(stamp)
        manifest = dict(stamp=stamp, seeds=SEEDS, repetitions=4, expected_virtual_steps=96,
                        no_training=True, no_promotion=True)
        if (OUT/'manifest.json').exists():
            assert json.loads((OUT/'manifest.json').read_text()) == json.loads(json.dumps(manifest))
        else:
            base.save(OUT/'manifest.json', manifest)
        tests = subprocess.run([sys.executable,'-m','pytest',str(TEST),'-q'], cwd=ROOT,
                               capture_output=True,text=True)
        base.save(OUT/'tests.json',dict(passed=tests.returncode==0,output=tests.stdout+tests.stderr,stamp=stamp))
        assert tests.returncode==0, tests.stdout+tests.stderr
        rows=[]
        for seed in SEEDS:
            rows.extend(one_seed(seed,parent['profile'],stamp))
        base.verify_stamp(stamp)
        write_report(rows,stamp)
        base.save(OUT/'status.json',dict(status='completed',device='mps',virtual_steps=96,
            global_validation=False,no_confirmation_gate=True))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true',required=True)
    parser.parse_args()
    main()
