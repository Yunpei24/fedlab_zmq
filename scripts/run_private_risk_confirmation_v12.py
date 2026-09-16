#!/usr/bin/env python3
"""Frozen, clean, four-seed confirmation. Test is evaluated only at round 120.

All model operations run on MPS, no fallback. Host operations are scalar
accounting, evidence checks and checkpoint I/O. No automatic attack campaign.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import secrets
import statistics as st
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
from scripts import run_capped_private_risk_calibration_v7 as ledger
from scripts import run_public_horizon_private_risk_v11 as previous
from privacy.fair_objective import require_mps, per_example, release
from privacy.split_risk_gradient import private_risk
from privacy.scheduled_private_risk import aggregate, learning_rate
from privacy.private_risk_message_safety import sanitize

NAME = 'private_risk_confirmation_v12'
MATRIX = ROOT / 'configs/ldp_gradient_far' / f'{NAME}.yaml'
OUT = ROOT / 'results/ldp_gradient_far' / NAME
LOG = ROOT / 'logs' / f'{NAME}.log'
REPORT = ROOT / 'output/analysis/Private_Risk_Confirmation_V12_Status.md'
TESTS = ['tests/test_private_risk_confirmation_v12.py',
         'tests/test_private_risk_message_safety.py',
         'tests/test_scheduled_private_risk.py', 'tests/test_stable_weighted_rfa.py',
         'tests/test_split_risk_gradient.py', 'tests/test_fair_objective.py']


def config():
    m = yaml.safe_load(MATRIX.read_text())
    assert m['campaign_id'] == NAME and m['device'] == 'mps'
    assert m['confirmation_seeds'] == [170601, 170602, 170603, 170604]
    assert m['methods'] == ['erm_mean', 'erm_rfa', 'risk_mean', 'risk_rfa']
    assert m['rounds'] == 120 and m['expected_runs'] == 16
    assert m['clip'] == 2 and m['risk_scale'] == .5
    assert m['epsilon'] == 4 and m['delta'] == 1e-5
    assert m['test_final_only'] and m['attacks'] == 'none' and not m['automatic_attacks']
    assert m['minimum_worst20_advantage_pp_per_seed'] == 1
    assert m['maximum_accuracy_loss_pp_per_seed'] == 1
    assert m['confidence_level'] == .95 and m['paired_t_critical_df3'] == 3.182446305284263
    assert all(m[k] for k in ('require_positive_worst20_ci_lower',
        'require_accuracy_ci_lower_ge_minus_one', 'require_nonpositive_mean_gap_change',
        'require_nonpositive_mean_variance_change'))
    assert m['primary_controls'] == ['erm_mean', 'erm_rfa']
    return m


def jobs(m):
    return [dict(seed=s, method=k) for s in m['confirmation_seeds'] for k in m['methods']]


def identifier(j):
    return f"seed{j['seed']}__{j['method']}"


def inputs(m):
    old = json.loads((previous.OUT / 'manifest.json').read_text())
    base.verify_stamp(old['source_stamp'])
    status = json.loads((previous.OUT / 'status.json').read_text())
    assert status['status'] == 'completed' and status['selected_robust_candidate'] == 'risk_rfa'
    assert set(m['confirmation_seeds']).isdisjoint(old['config']['calibration_seeds'])
    assert m['confirmation_seeds'] == old['config']['reserved_confirmation_seeds']
    names = ('dataset', 'model', 'num_clients', 'partition', 'dirichlet_beta',
             'public_train_size', 'validation_size', 'batch_size', 'epsilon_risk',
             'local_optimizer_steps', 'evaluation_rounds')
    profile = {k: old['profile'][k] for k in names}
    profile.update(device='mps', rounds=120, epsilon=4., delta=1e-5,
                   clips=[2.], risk_scales=[.5], server_clip=None)
    assert profile['local_optimizer_steps'] == 0 and profile['evaluation_rounds'][-1] == 120
    files = [Path(__file__), MATRIX, ROOT / m['protocol'],
             ROOT / 'privacy/private_risk_message_safety.py',
             ROOT / 'output/analysis/Private_Risk_Message_Safety_Policy.md',
             *[ROOT / p for p in TESTS]]
    stamp = dict(old['source_stamp'])
    stamp.update({str(p.relative_to(ROOT)): base.digest(p) for p in files})
    return profile, stamp


def paired_summary(values, critical):
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise ValueError('Exactly four finite independent seed differences required')
    mean, sd = st.mean(values), st.stdev(values)
    radius = critical * sd / 2
    return dict(n=4, mean=mean, sd=sd, ci95=[mean-radius, mean+radius], df=3)


def decide(m, rows):
    expected = {(j['seed'], j['method']) for j in jobs(m)}
    index = {(r['job']['seed'], r['job']['method']): r['final']['test'] for r in rows}
    if len(rows) != len(expected) or len(index) != len(rows) or set(index) != expected:
        raise ValueError('Complete unique preregistered grid required; no selective stopping')
    metrics = ['accuracy_pct', 'worst20_pct', 'gap_best20_worst20_pp', 'variance_pp2']
    comparisons = {}
    for control in [*m['primary_controls'], 'risk_mean']:
        pairs = []
        for seed in m['confirmation_seeds']:
            a, b = index[seed, 'risk_rfa'], index[seed, control]
            delta = {k: a[k]-b[k] for k in metrics}
            pairs.append(dict(seed=seed, delta=delta,
                seed_gate=delta['worst20_pct'] >= m['minimum_worst20_advantage_pp_per_seed']
                and delta['accuracy_pct'] >= -m['maximum_accuracy_loss_pp_per_seed']))
        summaries = {k: paired_summary([p['delta'][k] for p in pairs], m['paired_t_critical_df3']) for k in metrics}
        gates = dict(all_seed_gates=all(p['seed_gate'] for p in pairs),
            worst20_ci_lower_positive=summaries['worst20_pct']['ci95'][0] > 0,
            accuracy_ci_lower_noninferior=summaries['accuracy_pct']['ci95'][0] >= -1,
            mean_gap_nonincreasing=summaries['gap_best20_worst20_pp']['mean'] <= 0,
            mean_variance_nonincreasing=summaries['variance_pp2']['mean'] <= 0)
        comparisons[control] = dict(primary=control in m['primary_controls'], pairs=pairs,
                                    summaries=summaries, gates=gates, passed=all(gates.values()))
    return dict(clean_confirmation_passed=all(comparisons[c]['passed'] for c in m['primary_controls']),
        comparisons=comparisons, primary_endpoint='fixed final test round 120',
        joint_privacy_fairness_robustness_validated=False, attacks_evaluated=False,
        caution='Four independent seed differences; marginal t intervals, not universal guarantees')


def audit_evaluation(v):
    """Independent scalar reconstruction from saved class counts, no tensor work."""
    acc = []
    balanced = []
    for c in v['clients']:
        counts, hits = c['class_count'], c['class_hits']
        assert len(counts) == len(hits) == 10 and sum(counts) == c['N']
        assert all(0 <= h <= n and int(h) == h and int(n) == n for h, n in zip(hits, counts))
        a = sum(hits)/c['N']
        assert abs(a-c['accuracy']) < 1e-7
        acc.append(a)
        balanced.append(st.mean(h/n for h, n in zip(hits, counts) if n > 0))
    tail = max(1, math.ceil(.2*len(acc)))
    ordered = sorted(acc)
    expected = dict(accuracy_pct=100*st.mean(acc), client_accuracy_pct=100*st.mean(acc),
        worst20_pct=100*st.mean(ordered[:tail]),
        gap_best20_worst20_pp=100*(st.mean(ordered[-tail:])-st.mean(ordered[:tail])),
        gap_best_worst_pp=100*(max(acc)-min(acc)), variance_pp2=10000*st.pvariance(acc),
        balanced_accuracy_pct=100*st.mean(balanced),
        ce_loss=st.mean(c['ce_loss'] for c in v['clients']),
        brier_loss=st.mean(c['brier_loss'] for c in v['clients']))
    assert all(math.isfinite(v[k]) and abs(v[k]-value) < 2e-5 for k, value in expected.items())
    return True


def completed(j, m, profile, stamp):
    d = OUT / identifier(j)
    path = d / 'orchestration_status.json'
    if not path.exists():
        return False
    status = json.loads(path.read_text())
    if status['status'] != 'completed':
        return False
    r = json.loads((d/'metrics.json').read_text())
    assert r['job'] == j and r['source_stamp'] == stamp and r['device'] == 'mps'
    assert status['metrics_sha256'] == base.digest(d/'metrics.json')
    assert status['oracle_sha256'] == base.digest(d/'simulator_oracle.json')
    assert r['test_evaluated'] and r['test_evaluation_rounds'] == [120]
    assert r['privacy']['epsilon_realized'] <= m['epsilon']
    assert r['privacy']['delta'] == m['delta']
    assert [t['round'] for t in r['rounds']] == list(range(1,121))
    assert [t['round'] for t in r['rounds'] if t['test'] is not None] == [120]
    assert [t['round'] for t in r['rounds'] if t['validation'] is not None] == profile['evaluation_rounds']
    assert r['final'] == r['rounds'][-1]
    for t in r['rounds']:
        assert t['device'] == 'mps' and t['aggregation']['eta'] == learning_rate(t['round'],120)
        assert t['epsilon_realized'] <= 4
        if t['validation'] is not None:
            audit_evaluation(t['validation'])
    audit_evaluation(r['final']['test'])
    return True


def train(m, profile, j, data, key, stamp):
    if completed(j,m,profile,stamp):
        return
    d = OUT / identifier(j)
    d.mkdir(parents=True,exist_ok=True)
    p = ledger.privacy(profile, previous.arm(j['method']))
    model = base.new_model(profile,j['seed'])
    rows, oracle, start, elapsed = [], [], 0, 0.
    cp = d/'checkpoint.pt'
    if cp.exists():
        state = torch.load(cp,map_location='cpu',weights_only=True)
        assert state['source_stamp'] == stamp and state['job'] == j
        assert state['key_digest'] == base.digest(OUT/'simulator_secret.json')
        model.load_state_dict(state['model'])
        rows, oracle, start, initial = state['rows'],state['oracle'],state['round'],state['initial']
        elapsed = state['elapsed_seconds']
    else:
        initial = base.evaluate(model,data,'val')
    base.save(d/'public_protocol.json',dict(config=m,profile=profile,job=j,privacy=p,source_stamp=stamp))
    for t in range(start,120):
        tic = time.monotonic()
        require_mps()
        base.verify_stamp(stamp)
        status = dict(status='running',device='mps',active=identifier(j),round=t+1,
                      total_rounds=120,pid=os.getpid(),updated_unix=time.time())
        base.save(OUT/'status.json',status)
        base.save(d/'orchestration_status.json',status)
        messages,reports,local = [],[],[]
        for cid,ids in enumerate(data['train']):
            rr,raw = None,None
            if j['method'].startswith('risk_'):
                rr,raw = private_risk(model,data['x'][ids],data['y'][ids],noise_std=p['risk_std'],
                    seed=base.seed_for(key,j['seed'],t,cid,'risk'),N=profile['public_train_size'])
                reports.append(rr)
            idx = base.draw_indices(len(ids),profile['batch_size'],base.seed_for(key,j['seed'],t,cid,'batch'))
            _,g,norms,_ = per_example(model,data['x'][ids[idx]],data['y'][ids[idx]],clip_norm=2.)
            messages.append(release(g.mean(0),noise_std=p['gradient_std'],
                                    seed=base.seed_for(key,j['seed'],t,cid,'gaussian')))
            local.append(dict(client=cid,batch_hash=base.ids_hash(idx),
                clip_fraction=float((norms>2).float().mean()),
                private_risk=None if rr is None else float(rr),
                raw_risk_research_only=None if raw is None else float(raw)))
        x,r,safety = sanitize(torch.stack(messages),None if not reports else torch.stack(reports))
        # This clean confirmation must not silently repair a numerical failure.
        if safety['invalid_message_rows'] or safety['nonfinite_risk_reports']:
            raise FloatingPointError('Invalid honest message in clean confirmation')
        u,diag = aggregate(x,r,kind=j['method'],round_number=t+1,horizon=120)
        diag['message_safety'] = safety
        if not bool(torch.isfinite(u).all()):
            raise FloatingPointError('Nonfinite model step')
        base.apply_gradient(model,u,1.)
        validation = base.evaluate(model,data,'val') if t+1 in profile['evaluation_rounds'] else None
        test = base.evaluate(model,data,'test') if t+1 == 120 else None
        rows.append(dict(round=t+1,device='mps',epsilon_realized=ledger.epsilon_at(p,t+1,profile,j['method']),
                         aggregation=diag,validation=validation,test=test))
        oracle.append(dict(round=t+1,clients=local))
        elapsed += time.monotonic()-tic
        base.checkpoint(cp,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            rows=rows,oracle=oracle,round=t+1,initial=initial,source_stamp=stamp,job=j,
            key_digest=base.digest(OUT/'simulator_secret.json'),elapsed_seconds=elapsed))
        if validation is not None:
            print(f"{identifier(j)} {t+1}/120 eta={diag['eta']:g} val_acc={validation['accuracy_pct']:.3f} W20={validation['worst20_pct']:.3f}",flush=True)
    base.save(d/'metrics.json',dict(job=j,source_stamp=stamp,device='mps',privacy=p,initial=initial,
        rounds=rows,final=rows[-1],splits=data['splits'],test_evaluated=True,test_evaluation_rounds=[120],
        evaluation_and_oracles_not_private=True,elapsed_seconds=elapsed))
    base.save(d/'simulator_oracle.json',dict(privacy_protected=False,feeds_mechanism=False,rounds=oracle))
    base.save(d/'orchestration_status.json',dict(status='completed',device='mps',job=j,round=120,
        metrics_sha256=base.digest(d/'metrics.json'),oracle_sha256=base.digest(d/'simulator_oracle.json')))


def report(m,profile,stamp):
    rows = [json.loads((OUT/identifier(j)/'metrics.json').read_text()) for j in jobs(m) if completed(j,m,profile,stamp)]
    lines = ['# V12 — confirmation propre indépendante','',f'**{len(rows)}/16 runs valides sur MPS.**', '',
        'Test au tour 120 uniquement. Critères inchangés ; aucune validation conjointe avec robustesse à ce stade.', '',
        '| Seed | Méthode | Test acc. (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) | Brier |',
        '|--:|:--|--:|--:|--:|--:|--:|']
    for r in rows:
        j,v = r['job'],r['final']['test']
        lines.append(f"| {j['seed']} | {j['method']} | {v['accuracy_pct']:.3f} | {v['worst20_pct']:.3f} | {v['gap_best20_worst20_pp']:.3f} | {v['variance_pp2']:.3f} | {v['brier_loss']:.5f} |")
    decision = decide(m,rows) if len(rows)==16 else None
    if decision is not None:
        base.save(OUT/'evidence.json',dict(decision=decision,source_stamp=stamp,
            runs=[dict(job=r['job'],metrics_sha256=base.digest(OUT/identifier(r['job'])/'metrics.json')) for r in rows]))
        lines += ['',f"Gate propre : **{'PASS' if decision['clean_confirmation_passed'] else 'FAIL'}**.",
                  '', 'Les attaques ne sont pas lancées automatiquement.']
        for c,e in decision['comparisons'].items():
            lines += ['',f'### Face à {c}', '',f"Critères satisfaits : {e['gates']}", '',
                '| Différence risque-RFA − contrôle | Moyenne | Écart-type | IC95 inférieur | IC95 supérieur |',
                '|:--|--:|--:|--:|--:|']
            for k,s in e['summaries'].items():
                lines.append(f"| {k} | {s['mean']:.4f} | {s['sd']:.4f} | {s['ci95'][0]:.4f} | {s['ci95'][1]:.4f} |")
    lines += ['', '[Protocole figé](Private_Risk_Confirmation_V12_Protocol.md).']
    REPORT.write_text('\n'.join(lines)+'\n')
    return len(rows),decision


@contextmanager
def lock():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'campaign.lock').open('a') as f:
        try:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Campaign already active; no duplicate')
        yield


def worker(m):
    require_mps()
    with lock():
        try:
            profile,stamp = inputs(m)
            manifest = OUT/'manifest.json'
            content = dict(config=m,profile=profile,source_stamp=stamp)
            if manifest.exists():
                assert json.loads(manifest.read_text())==content
            else:
                base.save(manifest,content)
            testpath = OUT/'tests.json'
            if not testpath.exists():
                proc = subprocess.run([sys.executable,'-m','pytest',*TESTS,'-q'],cwd=ROOT,capture_output=True,text=True)
                base.save(testpath,dict(passed=proc.returncode==0,output=proc.stdout+proc.stderr,source_stamp=stamp))
            testresult = json.loads(testpath.read_text())
            assert testresult['passed'] and testresult['source_stamp']==stamp
            secret = OUT/'simulator_secret.json'
            if not secret.exists():
                if any(OUT.glob('seed*/checkpoint.pt')):
                    raise RuntimeError('Missing simulation key; cannot change randomness on resume')
                base.save(secret,dict(key=secrets.token_hex(32),not_a_private_release=True))
            key = json.loads(secret.read_text())['key']
            base.save(OUT/'privacy_plans.json',{k:ledger.privacy(profile,previous.arm(k)) for k in m['methods']})
            for seed in m['confirmation_seeds']:
                js = [j for j in jobs(m) if j['seed']==seed]
                if all(completed(j,m,profile,stamp) for j in js):
                    continue
                data = base.prepare(profile,seed)
                for j in js:
                    train(m,profile,j,data,key,stamp)
                    report(m,profile,stamp)
                del data
                torch.mps.empty_cache()
            for seed in m['confirmation_seeds']:
                baseline,pattern = None,None
                for j in [j for j in jobs(m) if j['seed']==seed]:
                    d = OUT/identifier(j)
                    r = json.loads((d/'metrics.json').read_text())
                    o = json.loads((d/'simulator_oracle.json').read_text())
                    now = [[c['batch_hash'] for c in t['clients']] for t in o['rounds']]
                    if baseline is None:
                        baseline,pattern = r,now
                    assert r['initial']==baseline['initial'] and r['splits']==baseline['splits'] and now==pattern
            base.save(OUT/'pairing_audit.json',dict(passed=True,splits_initial_models_batches_paired=True,
                gaussian_draws_paired_by_frozen_domain_separated_key=True,shared_seed_not_shared_private_transcript=True))
            count,decision = report(m,profile,stamp)
            assert count==16
            base.save(OUT/'status.json',dict(status='completed',device='mps',valid_runs=count,
                clean_confirmation_passed=decision['clean_confirmation_passed'],no_attacks_launched=True))
        except Exception as exc:
            base.save(OUT/'status.json',dict(status='failed',device='mps',error=repr(exc),pid=os.getpid()))
            raise


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ('launch','worker','status'):
        group.add_argument('--'+flag,action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    m = config()
    if args.status:
        print((OUT/'status.json').read_text() if (OUT/'status.json').exists() else 'not started')
        return
    if not args.resume:
        parser.error('--resume required')
    if args.worker:
        worker(m)
        return
    require_mps()
    with lock():
        pass
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open('a') as stream:
        proc = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker','--resume'],cwd=ROOT,
            env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0'),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=proc.pid,device='mps',runs=16,log=str(LOG))))


if __name__ == '__main__':
    main()
