#!/usr/bin/env python3
"""Exact MPS replay of two frozen calibration hosts, 96 isolated oracle probes."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
import torch
import yaml
from scripts import run_fair_objective_screen as base
from privacy.fair_objective import require_mps, per_example, release, wor_rdp
from privacy.split_risk_gradient import private_risk
from privacy.private_risk_message_safety import sanitize
from privacy.scheduled_private_risk import aggregate as host_aggregate
from privacy.capped_private_risk import potential
from privacy import private_risk_factorial_diagnostic_v26 as mechanism

NAME = 'private_risk_factorial_diagnostic_v26'
OUT = ROOT / 'results/ldp_gradient_far' / NAME
MATRIX = ROOT / 'configs/ldp_gradient_far' / f'{NAME}.yaml'
TEST = ROOT / 'tests/test_private_risk_factorial_diagnostic_v26.py'


def inputs():
    m = yaml.safe_load(MATRIX.read_text())
    assert m['campaign_id'] == NAME and m['device'] == 'mps'
    assert m['seeds'] == [170501, 170502] and m['pre_round_states'] == [2, 30, 60, 120]
    assert m['weights'] == list(mechanism.WEIGHTS) and m['messages'] == list(mechanism.MESSAGES)
    assert m['aggregators'] == list(mechanism.AGGREGATORS)
    assert (m['expected_states'], m['expected_probes'], m['source_rounds']) == (8, 96, 120)
    assert (m['source_mode'], m['source_method']) == ('fresh', 'risk_rfa')
    assert (m['population_block_size'], m['risk_scale'], m['median_iterations'], m['median_smoothing']) == (512, .5, 40, 1e-5)
    assert not any(m[k] for k in ('test_evaluated', 'privacy_protected', 'automatic_confirmation', 'automatic_attacks'))
    source = ROOT / m['source_campaign']
    original = json.loads((source / 'manifest.json').read_text())
    stamp = dict(original['source_stamp'])
    base.verify_stamp(stamp)
    profile = original['profile']
    assert (profile['model'], profile['num_clients'], profile['public_train_size'], profile['batch_size'], profile['rounds']) == ('lenet5_tanh', 10, 4800, 240, 120)
    paths = [Path(__file__), MATRIX, TEST, ROOT / m['protocol'],
             ROOT / 'privacy/private_risk_factorial_diagnostic_v26.py',
             ROOT / 'output/analysis/Public_Temporal_Noise_Confirmation_V25_Analyse.json',
             ROOT / 'output/analysis/Public_Temporal_Noise_Confirmation_V25_Decision.md',
             ROOT / 'output/analysis/Private_Fairness_Robustness_Prospective_Confirmation_Ledger.json',
             source / 'manifest.json']
    for seed in m['seeds']:
        d = source / f'seed{seed}__fresh__risk_rfa'
        s = json.loads((d / 'orchestration_status.json').read_text())
        r = json.loads((d / 'metrics.json').read_text())
        assert s['status'] == 'completed' and s['device'] == r['device'] == 'mps'
        assert s['metrics_sha256'] == base.digest(d / 'metrics.json')
        assert s['oracle_sha256'] == base.digest(d / 'simulator_oracle.json')
        assert r['source_stamp'] == original['source_stamp'] and not r['test_evaluated']
        assert r['privacy']['epsilon_realized'] <= 4 and r['privacy']['sampling'] == 'fixed_without_replacement'
        paths += [d / name for name in ('metrics.json', 'simulator_oracle.json', 'checkpoint.pt', 'orchestration_status.json')]
    stamp.update({str(p.relative_to(ROOT)): base.digest(p) for p in paths})
    return m, profile, stamp, source


def same(actual, expected, label):
    """No replay tolerances: JSON evidence must be identical to the source."""
    if actual != expected:
        raise AssertionError('Exact replay mismatch: ' + label)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def assert_model_equal(model, state):
    assert set(model.state_dict()) == set(state)
    for name, value in model.state_dict().items():
        if not torch.equal(value, state[name].to('mps')):
            raise AssertionError('Bitwise model mismatch: ' + name)


def progress(**fields):
    base.save(OUT / 'status.json', dict(status='running', device='mps', pid=os.getpid(),
              updated_unix=time.time(), valid_probes=len(list(OUT.glob('seed*/round*/probe_*.json'))),
              total_probes=96, **fields))


def validate_probe(path, stamp):
    r = json.loads(path.read_text())
    assert r['source_stamp'] == stamp and r['device'] == 'mps'
    assert not r['privacy_protected'] and not r['test_evaluated'] and not r['global_validation']
    assert r['vector_sha256'] == base.digest(path.with_suffix('.pt'))
    assert r['state_sha256'] == base.digest(path.parent / 'state.pt')
    assert r['treatment_id'] == mechanism.treatment_id(r['treatment'])
    return r


def probe_state(seed, round_number, model, messages, clean, reports, raw, step, host_diag,
                clients, data, profile, stamp):
    folder = OUT / f'seed{seed}' / f'round{round_number}'
    folder.mkdir(parents=True, exist_ok=True)
    node = folder / 'state.pt'
    before = cpu_state(model)
    if node.exists():
        snapshot = torch.load(node, map_location='cpu', weights_only=True)
        same(snapshot['source_stamp'], stamp, 'probe snapshot stamp')
        assert_model_equal(model, snapshot['before'])
        for key, current in [('private_messages', messages), ('clean_messages', clean),
                             ('private_reports', reports), ('raw_risks', raw), ('host_step', step)]:
            assert torch.equal(snapshot[key].to('mps'), current), key
        target = {k: v.to('mps') if isinstance(v, torch.Tensor) else v for k, v in snapshot['target'].items()}
        before_val = snapshot['validation_before']
    else:
        progress(seed=seed, replay_round=round_number, stage='population_gradient')
        risks, grads = mechanism.population(model, data['x'], data['y'], data['train'], with_gradient=True)
        # Different autograd/inference kernels may differ in final float32 bits;
        # host replay below is exact. This is a new, explicitly numerical oracle.
        torch.testing.assert_close(risks, raw, rtol=2e-5, atol=2e-7)
        target = mechanism.targets(raw, grads, clean)
        before_val = base.evaluate(model, data, 'val')
        snapshot = dict(before=before, private_messages=messages.cpu(), clean_messages=clean.cpu(),
                        private_reports=reports.cpu(), raw_risks=raw.cpu(), population_risks_autograd=risks.cpu(),
                        raw_population_gradients=grads.cpu(), host_step=step.cpu(), host_aggregation=host_diag,
                        clients=clients, validation_before=before_val,
                        target={k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in target.items()},
                        seed=seed, round=round_number, source_stamp=stamp, privacy_protected=False)
        base.checkpoint(node, snapshot)
    # A distinct model guarantees every intervention starts from the same host.
    probe = base.new_model(profile, seed)
    for t in mechanism.treatments():
        base.verify_stamp({k: v for k, v in stamp.items() if Path(k).suffix in ('.py', '.yaml', '.md')})
        name = mechanism.treatment_id(t)
        path = folder / f'probe_{name}.json'
        if path.exists():
            validate_probe(path, stamp)
            continue
        progress(seed=seed, replay_round=round_number, stage='one_step_probe', treatment=name)
        probe.load_state_dict(before)
        assert_model_equal(probe, before)
        A, diag = mechanism.aggregate(t, messages, clean, reports, raw)
        eta = host_diag['eta']
        applied = eta * A
        is_host = t == dict(weight='private', message='private', aggregator='rfa')
        if is_host:
            assert torch.equal(applied, step), 'Factorial host step differs from original'
        base.apply_gradient(probe, applied, 1.)
        after_risks, _ = mechanism.population(probe, data['x'], data['y'], data['train'], with_gradient=False)
        after_val = base.evaluate(probe, data, 'val')
        measures = mechanism.effects(A, target, eta=eta, J_after=float(potential(after_risks, .5).mean()))
        vec = path.with_suffix('.pt')
        base.checkpoint(vec, dict(aggregate=A.cpu(), after_risks=after_risks.cpu(), source_stamp=stamp,
                                 privacy_protected=False, treatment=t, seed=seed, round=round_number))
        base.save(path, dict(seed=seed, round=round_number, treatment=t, treatment_id=name, device='mps',
                  is_host_treatment=is_host, oracle_only_diagnostic=True, privacy_protected=False,
                  test_evaluated=False, global_validation=False, aggregation=diag, effects=measures,
                  validation_before=before_val, validation_after=after_val,
                  validation_change={k: after_val[k]-before_val[k] for k in ('accuracy_pct', 'worst20_pct',
                    'gap_best20_worst20_pp', 'variance_pp2', 'balanced_accuracy_pct', 'ce_loss', 'brier_loss')},
                  source_stamp=stamp, state_sha256=base.digest(node), vector_sha256=base.digest(vec)))
        print(f'V26 seed{seed} t{round_number} {name}: J gain={measures["actual_J_decrease"]:+.7f}', flush=True)
    assert_model_equal(model, before)
    base.save(folder / 'orchestration_status.json', dict(status='completed', device='mps', valid_probes=12,
              seed=seed, round=round_number, source_stamp=stamp, state_sha256=base.digest(node),
              probe_sha256={p.name: base.digest(p) for p in sorted(folder.glob('probe_*.json'))}))
    del probe
    torch.mps.empty_cache()


def replay(seed, m, profile, stamp, source, data, key):
    d = source / f'seed{seed}__fresh__risk_rfa'
    original = json.loads((d / 'metrics.json').read_text())
    oracle = json.loads((d / 'simulator_oracle.json').read_text())
    p = original['privacy']  # Exact stored calibration, no re-solving sigma.
    folder = OUT / f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    cp = folder / 'replay_checkpoint.pt'
    key_sha = base.digest(source / 'simulator_secret.json')
    model = base.new_model(profile, seed)
    unused_original_previous_model = base.new_model(profile, seed)
    start, replay_checks = 0, []
    same(data['splits'], original['splits'], 'dataset split')
    if cp.exists():
        saved = torch.load(cp, map_location='cpu', weights_only=True)
        same(saved['source_stamp'], stamp, 'replay stamp')
        assert saved['key_sha'] == key_sha and saved['seed'] == seed
        model.load_state_dict(saved['model'])
        start, replay_checks = saved['round'], saved['replay_checks']
        assert [r['round'] for r in replay_checks] == list(range(1, start+1))
    else:
        same(base.evaluate(model, data, 'val'), original['initial'], 'initial validation')
    del unused_original_previous_model
    code_stamp = {k: v for k, v in stamp.items() if Path(k).suffix in ('.py', '.yaml', '.md')}
    for t in range(start, 120):
        require_mps()
        base.verify_stamp(code_stamp)
        progress(seed=seed, replay_round=t+1, stage='host_replay')
        sent, clean, reports, raw_reports, local = [], [], [], [], []
        for cid, ids in enumerate(data['train']):
            rr, raw = private_risk(model, data['x'][ids], data['y'][ids], noise_std=p['risk_std'],
                                   seed=base.seed_for(key, seed, t, cid, 'risk'), N=4800)
            ix = base.draw_indices(4800, 240, base.seed_for(key, seed, t, cid, 'batch'))
            _, g, norms, _ = per_example(model, data['x'][ids[ix]], data['y'][ids[ix]], clip_norm=2.)
            gbar = g.mean(0)
            message = release(gbar, noise_std=p['gradient_std'], seed=base.seed_for(key, seed, t, cid, 'gaussian'))
            check = dict(client=cid, batch_hash=base.ids_hash(ix), gradient_clipped_count=int((norms>2).sum()),
                         batch_size=240, private_risk=float(rr), raw_risk=float(raw))
            previous = oracle['rounds'][t]['clients'][cid]
            same(check, {k: previous[k] for k in check}, f'seed{seed} round{t+1} client{cid}')
            sent.append(message); clean.append(gbar); reports.append(rr); raw_reports.append(raw); local.append(check)
        messages, r, safety = sanitize(torch.stack(sent), torch.stack(reports))
        assert safety['invalid_message_rows'] == safety['nonfinite_risk_reports'] == 0
        step, diag = host_aggregate(messages, r, kind='risk_rfa', round_number=t+1, horizon=120)
        diag['message_safety'] = safety
        same(diag, original['rounds'][t]['aggregation'], f'round{t+1} aggregation')
        epsilon = min((t+1)*wor_rdp(a, .05, p['gradient_z'])+(t+1)*a/(2*p['risk_z']**2)
                      +math.log(1e5)/(a-1) for a in range(2, 65))
        same(epsilon, original['rounds'][t]['epsilon_realized'], f'round{t+1} epsilon')
        if t+1 in m['pre_round_states']:
            probe_state(seed, t+1, model, messages, torch.stack(clean), r, torch.stack(raw_reports),
                        step, diag, local, data, profile, stamp)
        base.apply_gradient(model, step, 1.)
        val = base.evaluate(model, data, 'val') if t+1 in profile['evaluation_rounds'] else None
        same(val, original['rounds'][t]['validation'], f'round{t+1} validation')
        if t+1 in m['pre_round_states']:
            actual = json.loads((folder / f'round{t+1}' / 'probe_private__private__rfa.json').read_text())
            # Probe rounds 2/30/60/120 also get validation, even when original didn't.
            replay_val = val if val is not None else base.evaluate(model, data, 'val')
            same(actual['validation_after'], replay_val, 'host probe validation')
        replay_checks.append(dict(round=t+1, all_client_records_exact=True, aggregation_exact=True,
                                  stored_validation_exact=True, privacy_prefix_exact=True))
        base.checkpoint(cp, dict(seed=seed, round=t+1, model=cpu_state(model), replay_checks=replay_checks,
                                key_sha=key_sha, source_stamp=stamp, privacy_protected=False))
        if (t+1) % 10 == 0:
            print(f'V26 seed{seed} replay {t+1}/120 exact', flush=True)
    final = torch.load(d / 'checkpoint.pt', map_location='cpu', weights_only=True)
    assert_model_equal(model, final['model'])
    base.verify_stamp(stamp)
    base.save(folder / 'replay_audit.json', dict(seed=seed, device='mps', source_stamp=stamp,
              passed=True, initial_and_saved_evaluations_exact=True, all_rounds_exact=True,
              final_model_bitwise_equal=True, rounds_checked=120, source_checkpoint_sha256=base.digest(d / 'checkpoint.pt'),
              replay_checkpoint_sha256=base.digest(cp), key_sha=key_sha, test_evaluated=False))
    del model
    torch.mps.empty_cache()


def worker():
    require_mps()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            m, profile, stamp, source = inputs()
            manifest = dict(config=m, profile=profile, source_stamp=stamp,
                            torch_version=str(torch.__version__), device='mps', fallback=False)
            if (OUT / 'manifest.json').exists():
                same(json.loads((OUT / 'manifest.json').read_text()), manifest, 'manifest')
            else:
                base.save(OUT / 'manifest.json', manifest)
            if not (OUT / 'tests.json').exists():
                proc = subprocess.run([sys.executable, '-m', 'pytest', str(TEST), '-q'], cwd=ROOT, capture_output=True, text=True)
                base.save(OUT / 'tests.json', dict(passed=proc.returncode==0, output=proc.stdout+proc.stderr, source_stamp=stamp))
            tests = json.loads((OUT / 'tests.json').read_text())
            assert tests['passed'] and tests['source_stamp'] == stamp
            key = json.loads((source / 'simulator_secret.json').read_text())['key']
            for seed in m['seeds']:
                data = base.prepare(profile, seed)
                replay(seed, m, profile, stamp, source, data, key)
                del data
                torch.mps.empty_cache()
            paths = sorted(OUT.glob('seed*/round*/probe_*.json'))
            assert len(paths) == 96
            for path in paths:
                validate_probe(path, stamp)
            base.verify_stamp(stamp)
            base.save(OUT / 'status.json', dict(status='completed', device='mps', valid_probes=96,
                      states=8, replayed_hosts=2, replay_audits_passed=True, source_stamp=stamp,
                      global_validation=False, next_campaign_launched=False, test_evaluated=False))
        except Exception as exc:
            base.save(OUT / f'failure_{time.time_ns()}.json', dict(error=repr(exc), device='mps', pid=os.getpid()))
            base.save(OUT / 'status.json', dict(status='failed', error=repr(exc), device='mps', pid=os.getpid()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', required=True)
    parser.parse_args()
    worker()
