#!/usr/bin/env python3
"""Conditional V30 clean/attack/recovery runs, inheriting V29 exactly on MPS.

No automatic dispatch. --validate-only is read-only and rejects partial/negative
V29 evidence. A separately prepared independent audit program and its tests are
required before training can start. No tuning uses confirmation or attack data.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
import torch
import yaml
from scripts import run_fair_objective_screen as base
from scripts import run_full_population_private_risk_confirmation_v29 as parent
from scripts.analyze_private_risk_confirmation_v12 import verify
from privacy.fair_objective import require_mps
from privacy.full_population_private_risk_confirmation_v29 import plan, prefix, private_message
from privacy.full_population_private_risk_attacks_v30 import (
    SEEDS, METHODS, CONDITIONS, ENDPOINTS, jobs, exact_honest_test, require_clean_pass, decide)
from privacy.private_risk_attacks import inject
from privacy.split_risk_gradient import private_risk
from privacy.scheduled_private_risk import aggregate
from privacy.private_risk_message_safety import sanitize
from privacy.stable_weighted_rfa import stable_norm

NAME = 'full_population_private_risk_attacks_v30'
OUT = ROOT/'results/ldp_gradient_far'/NAME
MATRIX = ROOT/'configs/ldp_gradient_far'/f'{NAME}.yaml'
PARENT_AUDIT = ROOT/'output/analysis/Full_Population_Private_Risk_Confirmation_V29_Analyse.json'
REPORT = ROOT/'output/analysis/Full_Population_Private_Risk_Attacks_V30_Status.md'
AUDITOR = ROOT/'scripts/audit_full_population_private_risk_attacks_v30.py'
AUDITOR_TEST = ROOT/'tests/test_full_population_private_risk_attacks_v30_audit.py'
TESTS = ['tests/test_full_population_private_risk_attacks_v30.py',
    'tests/test_full_population_private_risk_attacks_v30_runner.py',
    'tests/test_private_risk_attacks.py', 'tests/test_private_risk_message_safety.py',
    'tests/test_stable_weighted_rfa.py', 'tests/test_scheduled_private_risk.py']


def config():
    c = yaml.safe_load(MATRIX.read_text())
    expected = dict(campaign_id=NAME, state_at_registration='conditional_preparation_during_v29',
        parent_campaign=parent.NAME, require_complete_independent_clean_pass=True,
        automatic_launch=False, device='mps', seeds=list(SEEDS), methods=list(METHODS),
        primary_method='risk_rfa', primary_controls=['erm_mean', 'erm_rfa'], batch_size=4800,
        public_train_size=4800, rounds=120, local_optimizer_steps=0, clip=2., risk_scale=.5,
        epsilon=4., delta=1e-5, privacy_plan='inherit_exact_v29_full_population_joint_plan',
        server_clip=None, solver_iterations=40, solver_smoothing=1e-5,
        eta_early=2., eta_late=.5, eta_switch_after_round=60, conditions=list(CONDITIONS),
        expected_runs=64, clean_replays=16, attacked_runs=48, byzantine_ids=[0, 1],
        honest_ids=list(range(2, 10)), forged_risk=1., attack_start=31, temporary_attack_end=90,
        bf_multiplier=10., alie_population_std_multiplier=1.5, ipm_ramp_end=60,
        ipm_final_multiplier=2., primary_endpoints=ENDPOINTS, test_rounds=[90, 120],
        validation_rounds=[1, *range(10, 121, 10)], minimum_worst20_gain_per_seed_pp=1.,
        maximum_accuracy_loss_per_seed_pp=1., maximum_own_clean_accuracy_loss_pp=5.,
        maximum_own_clean_worst20_loss_pp=5., maximum_solver_gap_diagnostic=.001,
        mean_gap_and_variance_must_not_increase=True, confirmation_wave=1,
        one_sided_alpha=.0125, paired_t_critical_df3=4.176534846104499,
        protocol='output/analysis/Full_Population_Private_Risk_Attacks_V30_Protocol.md')
    if c != expected:
        raise ValueError('V30 config differs from its fixed preregistration')
    return c


def identifier(j):
    return f"seed{j['seed']}__{j['method']}__{j['attack']}"


def clean_job(j):
    return dict(seed=j['seed'], method=j['method'], attack='none')


def parent_job(j):
    return dict(seed=j['seed'], batch=4800, method=j['method'])


def inputs():
    c = config()
    if not PARENT_AUDIT.exists():
        raise RuntimeError('V30 closed: independent final V29 audit is not available')
    audit = json.loads(PARENT_AUDIT.read_text())
    state = json.loads((parent.OUT/'status.json').read_text())
    require_clean_pass(audit, state, base.digest(parent.OUT/'manifest.json'))
    manifest = json.loads((parent.OUT/'manifest.json').read_text())
    base.verify_stamp(manifest['source_stamp']); base.verify_stamp(audit['source_stamp'])
    for j in parent.jobs():
        if not parent.completed(j, manifest['source_stamp']):
            raise RuntimeError('V30 closed: a parent completion or its hashes are invalid')
    # A runner-only pass never opens training without the independent audit code.
    if not AUDITOR.is_file() or not AUDITOR_TEST.is_file():
        raise RuntimeError('V30 closed: independent attack auditor and tests must be prepared first')
    stamp = dict(manifest['source_stamp'])
    stamp.update(audit['source_stamp'])
    files = [Path(__file__), MATRIX, ROOT/c['protocol'],
        ROOT/'privacy/full_population_private_risk_attacks_v30.py',
        ROOT/'privacy/private_risk_attacks.py', PARENT_AUDIT, AUDITOR, AUDITOR_TEST,
        parent.OUT/'manifest.json', parent.OUT/'status.json', *[ROOT/p for p in TESTS]]
    for j in parent.jobs():
        folder = parent.OUT/parent.identifier(j)
        files.extend(folder/name for name in ('metrics.json', 'simulator_oracle.json',
                                              'orchestration_status.json', 'checkpoint.pt'))
    stamp.update({str(p.relative_to(ROOT)): base.digest(p) for p in files})
    return c, manifest['profile'], stamp, manifest


def honest_metrics(full):
    clients = full['clients'][2:]
    if len(full['clients']) != 10 or len(clients) != 8:
        raise ValueError('Fixed ten-client evaluation required')
    acc = [c['accuracy'] for c in clients]; ordered = sorted(acc)
    h = dict(accuracy_pct=100*st.mean(acc), client_accuracy_pct=100*st.mean(acc),
        worst20_pct=100*st.mean(ordered[:2]),
        gap_best20_worst20_pp=100*(st.mean(ordered[-2:])-st.mean(ordered[:2])),
        gap_best_worst_pp=100*(max(acc)-min(acc)), variance_pp2=10000*st.pvariance(acc),
        ce_loss=st.mean(c['ce_loss'] for c in clients), brier_loss=st.mean(c['brier_loss'] for c in clients),
        balanced_accuracy_pct=100*st.mean(c['balanced_accuracy'] for c in clients),
        clients=clients, honest_ids=list(range(2, 10)), worst20_client_count=2)
    if all(c['N'] == 1000 for c in full['clients']):
        h.update({k: float(v) for k, v in exact_honest_test(full).items()})
        h['client_accuracy_pct'] = h['accuracy_pct']
    return h


def model_hash(model):
    # Covers all state entries, not just the update vector; tensors remain on MPS.
    return {k: base.ids_hash(v.detach()) for k, v in model.state_dict().items()}


def assert_same_prefix(row, local, old_row, old_local):
    """Exact public transcript/model replay while the current attack is inactive."""
    fields = ('epsilon_realized', 'epsilon_order', 'aggregation', 'validation',
              'private_messages_sha256', 'private_reports_sha256',
              'step_sha256', 'model_sha256', 'pre_model_sha256')
    if any(row[k] != old_row[k] for k in fields) or local != old_local:
        raise AssertionError('V30 clean/prefix replay differs; never silently resume a mismatched trajectory')


def completed(j, stamp):
    folder = OUT/identifier(j); path = folder/'orchestration_status.json'
    if not path.exists() or json.loads(path.read_text())['status'] != 'completed':
        return False
    status = json.loads(path.read_text()); r = json.loads((folder/'metrics.json').read_text())
    assert status['device'] == r['device'] == 'mps' and status['round'] == 120
    assert r['job'] == j and r['source_stamp'] == stamp
    for filename, field in [('metrics.json', 'metrics_sha256'), ('simulator_oracle.json', 'oracle_sha256'),
                            ('checkpoint.pt', 'checkpoint_sha256'), ('endpoint_090.pt', 'endpoint90_sha256'),
                            ('endpoint_120.pt', 'endpoint120_sha256')]:
        assert base.digest(folder/filename) == status[field]
    assert r['privacy'] == json.loads(json.dumps(plan(4800, j['method'])))
    assert r['privacy']['epsilon_realized'] <= 4 and r['privacy']['delta'] == 1e-5
    assert [x['round'] for x in r['rounds']] == list(range(1, 121))
    assert [x['round'] for x in r['rounds'] if x['test'] is not None] == [90, 120]
    assert r['test_evaluation_rounds'] == [90, 120] and r['final'] == r['rounds'][-1]
    assert r['local_optimizer_steps'] == 0 and r['gradient_examples_per_client'] == 120*4800
    for row in r['rounds']:
        assert row['device'] == 'mps' and row['epsilon_realized'] <= 4
        if row['test'] is not None:
            verify(row['test'])
            assert row['test_honest'] == honest_metrics(row['test'])
    return True


def progress(j, t, stage, client=None):
    s = dict(status='running', device='mps', active=identifier(j), round=t, total_rounds=120,
        stage=stage, client=client, pid=os.getpid(), updated_unix=time.time())
    base.save(OUT/'status.json', s); base.save(OUT/identifier(j)/'orchestration_status.json', s)


def train(j, c, profile, stamp, parent_manifest, data, key):
    if completed(j, stamp):
        return
    folder = OUT/identifier(j); folder.mkdir(parents=True, exist_ok=True)
    prior_folder = parent.OUT/parent.identifier(parent_job(j))
    prior = json.loads((prior_folder/'metrics.json').read_text())
    prior_oracle = json.loads((prior_folder/'simulator_oracle.json').read_text())['rounds']
    control = control_oracle = None
    if j['attack'] != 'none':
        if not completed(clean_job(j), stamp):
            raise RuntimeError('The matching clean V30 replay must finish first')
        control_folder = OUT/identifier(clean_job(j))
        control = json.loads((control_folder/'metrics.json').read_text())
        control_oracle = json.loads((control_folder/'simulator_oracle.json').read_text())['rounds']
    p = plan(4800, j['method']); model = base.new_model(profile, j['seed'])
    cp = folder/'checkpoint.pt'; key_sha = base.digest(parent.OUT/'simulator_secret.json')
    rows, oracles, start, elapsed = [], [], 0, 0.
    if cp.exists():
        saved = torch.load(cp, map_location='cpu', weights_only=True)
        assert saved['job'] == j and saved['source_stamp'] == stamp and saved['key_sha'] == key_sha
        assert saved['privacy'] == p and saved['round'] <= 120
        model.load_state_dict(saved['model'])
        rows, oracles, start, elapsed, initial = (saved[k] for k in ('rows', 'oracles', 'round', 'elapsed_seconds', 'initial'))
        assert [r['round'] for r in rows] == list(range(1, start+1))
        # If interruption occurred between checkpoint and endpoint snapshot, rebuild
        # the identical snapshot from the complete saved endpoint, not from new data.
        if start in (90, 120):
            ep = folder/f'endpoint_{start:03}.pt'
            if not ep.exists(): base.checkpoint(ep, saved)
    else:
        initial = base.evaluate(model, data, 'val')
    assert initial == prior['initial'] and data['splits'] == prior['splits']
    base.save(folder/'public_protocol.json', dict(config=c, profile=profile, job=j, privacy=p, source_stamp=stamp))
    code = {k: v for k, v in stamp.items() if Path(k).suffix in ('.py', '.yaml', '.md')}
    for t in range(start, 120):
        tick = time.monotonic(); require_mps(); base.verify_stamp(code)
        before = parent.cpu_state(model); before_hash = model_hash(model)
        messages, clean, reports, local = [], [], [], []
        for cid, ids in enumerate(data['train']):
            progress(j, t+1, 'private_gradients', cid); rr = raw = None
            if j['method'].startswith('risk_'):
                rr, raw = private_risk(model, data['x'][ids], data['y'][ids], noise_std=p['risk_std'],
                    seed=base.seed_for(key, j['seed'], t, cid, 'risk'), N=4800)
                reports.append(rr)
            ix = base.draw_indices(4800, 4800, base.seed_for(key, j['seed'], t, cid, 'batch'))
            assert len(torch.unique(ix)) == 4800
            ih, ph = base.ids_hash(ix), base.ids_hash(ix[:240]); old = prior_oracle[t]['clients'][cid]
            assert ih == old['indices_sha256'] and ph == old['batch_prefix240_sha256']
            message, query, gradient = private_message(model, data['x'][ids[ix]], data['y'][ids[ix]],
                batch=4800, noise_std=p['gradient_std'], seed=base.seed_for(key, j['seed'], t, cid, 'gaussian'))
            messages.append(message); clean.append(query)
            item = dict(client=cid, batch_size=4800, indices_sha256=ih, batch_prefix240_sha256=ph,
                private_message_sha256=base.ids_hash(message), clean_mean_sha256=base.ids_hash(query),
                private_risk=None if rr is None else float(rr), raw_risk=None if raw is None else float(raw), gradient=gradient)
            if j['attack'] == 'none' or t < 30:
                assert item == old, 'Parent clean query/report differs before any attack'
            local.append(item)
        assert model_hash(model) == before_hash
        generated = torch.stack(messages); generated_reports = None if not reports else torch.stack(reports)
        sent, r, threat = inject(generated, generated_reports, attack=j['attack'], round_number=t+1)
        sent, r, safety = sanitize(sent, r)
        if safety['invalid_message_rows'] or safety['nonfinite_risk_reports']:
            raise FloatingPointError('Nonfinite/out-of-domain message in fixed finite threat benchmark')
        step, agg = aggregate(sent, r, kind=j['method'], round_number=t+1, horizon=120)
        agg['message_safety'] = safety
        if not bool(torch.isfinite(step).all()): raise FloatingPointError('Nonfinite applied step')
        center = step/agg['eta']; target = torch.stack(clean)[2:].mean(0)
        weights = agg['objective_weights'] if agg['solver'] is None else agg['solver']['stationary_weights']
        effective = torch.tensor(weights, device='mps', dtype=torch.float32)
        reconstructed = (effective[:, None]*sent).sum(0)
        byzpart = (effective[:2, None]*sent[:2]).sum(0)
        assert sum(agg['objective_weights'][:2]) <= 3/7+1e-6
        oracle = dict(round=t+1, clients=local, aggregate=dict(
            squared_error_to_unnoised_clipped_honest_mean=float(stable_norm(center-target).square()),
            honest_target_norm=float(stable_norm(target)), aggregate_norm=float(stable_norm(center)),
            designated_objective_mass=sum(agg['objective_weights'][:2]),
            designated_effective_mass=sum(weights[:2]),
            active_byzantine_contribution_norm=float(stable_norm(byzpart)) if threat['active'] else 0.,
            reconstruction_error=float(stable_norm(center-reconstructed)),
            target='Mean of eight honest population gradients after per-example clipping, before DP noise'))
        base.apply_gradient(model, step, 1.); progress(j, t+1, 'evaluation_and_checkpoint')
        val = base.evaluate(model, data, 'val') if t+1 in c['validation_rounds'] else None
        test = base.evaluate(model, data, 'test') if t+1 in c['test_rounds'] else None
        if val is not None: verify(val)
        if test is not None: verify(test)
        eps, order = prefix(p, t+1)
        row = dict(round=t+1, device='mps', epsilon_realized=eps, epsilon_order=order,
            aggregation=agg, attack=threat, validation=val,
            validation_honest=None if val is None else honest_metrics(val),
            test=test, test_honest=None if test is None else honest_metrics(test),
            pre_model_sha256=before_hash, model_sha256=model_hash(model), step_sha256=base.ids_hash(step),
            private_messages_sha256=base.ids_hash(sent),
            private_reports_sha256=None if r is None else base.ids_hash(r))
        if j['attack'] == 'none':
            oldrow = prior['rounds'][t]
            assert all(row[k] == oldrow[k] for k in ('aggregation', 'validation', 'epsilon_realized', 'epsilon_order'))
            if t+1 == 120:
                assert test == prior['final']['test']
                last = torch.load(prior_folder/'checkpoint.pt', map_location='cpu', weights_only=True)
                assert all(torch.equal(v, last['model'][k].to('mps')) for k, v in model.state_dict().items())
        elif t < 30:
            assert_same_prefix(row, local, control['rounds'][t], control_oracle[t]['clients'])
        rows.append(row); oracles.append(oracle); elapsed += time.monotonic()-tick
        snapshot = dict(model=parent.cpu_state(model), pre_round_model=before,
            last_generated_messages=generated.detach().cpu(), last_sent_messages=sent.detach().cpu(),
            last_clean_means=torch.stack(clean).detach().cpu(),
            last_generated_reports=None if generated_reports is None else generated_reports.detach().cpu(),
            last_sent_reports=None if r is None else r.detach().cpu(), last_step=step.detach().cpu(),
            rows=rows, oracles=oracles, round=t+1, elapsed_seconds=elapsed, initial=initial,
            privacy=p, job=j, source_stamp=stamp, key_sha=key_sha, privacy_protected=False)
        base.checkpoint(cp, snapshot)
        if t+1 in (90, 120): base.checkpoint(folder/f'endpoint_{t+1:03}.pt', snapshot)
        if val is not None:
            v = row['validation_honest']
            print(f"{identifier(j)} {t+1}/120 honest_val_acc={v['accuracy_pct']:.4f} "
                  f"W20={v['worst20_pct']:.4f} elapsed={elapsed:.1f}s", flush=True)
    base.verify_stamp(stamp)
    base.save(folder/'metrics.json', dict(job=j, source_stamp=stamp, device='mps', privacy=p,
        initial=initial, rounds=rows, final=rows[-1], splits=data['splits'],
        test_evaluation_rounds=[90, 120], gradient_examples_per_client=120*4800,
        private_gradient_releases_per_client=120, local_optimizer_steps=0,
        validation_test_and_oracles_not_private=True, elapsed_seconds=elapsed))
    base.save(folder/'simulator_oracle.json', dict(privacy_protected=False, feeds_mechanism=False, rounds=oracles))
    base.save(folder/'orchestration_status.json', dict(status='completed', device='mps', job=j, round=120,
        metrics_sha256=base.digest(folder/'metrics.json'), oracle_sha256=base.digest(folder/'simulator_oracle.json'),
        checkpoint_sha256=base.digest(cp), endpoint90_sha256=base.digest(folder/'endpoint_090.pt'),
        endpoint120_sha256=base.digest(folder/'endpoint_120.pt'), gate_evaluated=False))
    del model; torch.mps.empty_cache()


def report(stamp):
    records = [json.loads((OUT/identifier(j)/'metrics.json').read_text()) for j in jobs() if completed(j, stamp)]
    decision = decide(records) if len(records) == 64 else None
    if decision is not None:
        base.save(OUT/'evidence_unverified.json', dict(decision=decision, source_stamp=stamp, independent_audit_passed=False))
    lines = ['# V30 — contrôles propres, attaques et récupération', '', f'**{len(records)}/64 runs terminés.**', '',
        'Tous les chiffres primaires portent sur les huit identités honnêtes 2–9. '
        'Une sortie du lanceur ne remplace pas son audit indépendant.', '',
        '| Seed | Condition | Méthode | Tour | Accuracy (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) |',
        '|--:|:--|:--|--:|--:|--:|--:|--:|']
    for r in records:
        j = r['job']
        for row in r['rounds']:
            if row['test'] is None: continue
            v = row['test_honest']
            lines.append(f"| {j['seed']} | {j['attack']} | {j['method']} | {row['round']} | "
                f"{v['accuracy_pct']:.4f} | {v['worst20_pct']:.4f} | {v['gap_best20_worst20_pp']:.4f} | {v['variance_pp2']:.4f} |")
    if decision is not None:
        lines += ['', f"Critères calculés, audit encore requis : {'PASS' if decision['attack_confirmation_passed'] else 'FAIL'}."]
    lines += ['', '[Protocole conditionnel](Full_Population_Private_Risk_Attacks_V30_Protocol.md).']
    REPORT.write_text('\n'.join(lines)+'\n')
    return len(records)


def worker():
    require_mps()
    c, profile, stamp, parent_manifest = inputs()  # Closed gate creates no campaign directory.
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            manifest = dict(config=c, profile=profile, source_stamp=stamp, device='mps',
                torch_version=str(torch.__version__), fallback=False,
                parent_manifest_sha256=base.digest(parent.OUT/'manifest.json'),
                simulator_key_sha256=base.digest(parent.OUT/'simulator_secret.json'))
            if (OUT/'manifest.json').exists(): assert json.loads((OUT/'manifest.json').read_text()) == manifest
            else: base.save(OUT/'manifest.json', manifest)
            if not (OUT/'tests.json').exists():
                tests = subprocess.run([sys.executable, '-m', 'pytest', *TESTS,
                    str(AUDITOR_TEST.relative_to(ROOT)), '-q'], cwd=ROOT, capture_output=True, text=True)
                base.save(OUT/'tests.json', dict(passed=tests.returncode == 0, source_stamp=stamp,
                    output=tests.stdout+tests.stderr))
            tests = json.loads((OUT/'tests.json').read_text())
            assert tests['passed'] and tests['source_stamp'] == stamp
            key = json.loads((parent.OUT/'simulator_secret.json').read_text())['key']
            for seed in SEEDS:
                data = base.prepare(profile, seed)
                for j in [j for j in jobs() if j['seed'] == seed]:
                    train(j, c, profile, stamp, parent_manifest, data, key)
                    print(f'V30 {report(stamp)}/64 completed; no independent gate yet', flush=True)
                del data; torch.mps.empty_cache()
            assert report(stamp) == 64; base.verify_stamp(stamp)
            base.save(OUT/'status.json', dict(status='completed', device='mps', valid_runs=64,
                independent_audit_passed=False, global_validation=False, source_stamp=stamp))
        except Exception as exc:
            failure = dict(status='failed', device='mps', error=repr(exc), pid=os.getpid())
            base.save(OUT/f'failure_{time.time_ns()}.json', failure); base.save(OUT/'status.json', failure)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--resume', action='store_true')
    group.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        inputs()
        print(json.dumps(dict(prerequisites_passed=True, training_launched=False)))
    else:
        worker()
