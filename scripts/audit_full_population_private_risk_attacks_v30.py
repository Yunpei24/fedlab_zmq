#!/usr/bin/env python3
"""Independent V30 audit: no import of its runner, attack generator or gate.

Reuses the independent V29 accounting/RNG/count primitives, dataset/model and
autodiff, and the shared stable geometric-median solver. Attacks, risk weights,
parameter updates, honest counts and attack/recovery decisions are rederived.
Shared PyTorch primitives mean this is not an independent numerical platform.
Only completed runs are read; raw oracles are outside the private transcript.
"""
import argparse
from fractions import Fraction as Q
import json
import math
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
import torch
import yaml
from scripts import run_fair_objective_screen as base
from scripts import audit_full_population_private_risk_confirmation_v29 as clean_audit
from scripts.analyze_private_risk_confirmation_v12 import verify
from privacy.fair_objective import require_mps, per_example
from privacy.stable_weighted_rfa import weighted_rfa

OUT = ROOT/'results/ldp_gradient_far/full_population_private_risk_attacks_v30'
PARENT = ROOT/'results/ldp_gradient_far/full_population_private_risk_confirmation_v29'
PARENT_AUDIT = ROOT/'output/analysis/Full_Population_Private_Risk_Confirmation_V29_Analyse.json'
DEST = ROOT/'output/analysis/Full_Population_Private_Risk_Attacks_V30_Analyse'
CACHE = ROOT/'output/analysis/audit_full_population_private_risk_attacks_v30'
TEST = ROOT/'tests/test_full_population_private_risk_attacks_v30_audit.py'
SEEDS = (180701, 180702, 180703, 180704)
METHODS = ('erm_mean', 'erm_rfa', 'risk_mean', 'risk_rfa')
ATTACKS = ('none', 'abrupt_bf', 'persistent_alie', 'slow_ipm')
ENDPOINTS = dict(abrupt_bf=90, persistent_alie=120, slow_ipm=90)
KEYS = ('accuracy_pct', 'worst20_pct', 'gap_best20_worst20_pp', 'variance_pp2')
EXTRA = ('balanced_accuracy_pct', 'ce_loss', 'brier_loss')
digest, close = clean_audit.digest, clean_audit.close


def name(j): return f"seed{j['seed']}__{j['method']}__{j['attack']}"
def parent_name(j): return f"seed{j['seed']}__b4800__{j['method']}"
def matrix(): return [dict(seed=s, method=m, attack=a) for s in SEEDS for a in ATTACKS for m in METHODS]
def model_hash(model): return {k: base.ids_hash(v.detach()) for k, v in model.state_dict().items()}


def active_at(attack, t):
    if attack not in ATTACKS or type(t) is not int or not 1 <= t <= 120:
        raise ValueError('Unknown attack or public round')
    return attack != 'none' and t >= 31 and (attack == 'persistent_alie' or t <= 90)


def independent_attack(messages, reports, attack, t):
    """Manual population moment for ALIE; no call to private_risk_attacks.inject."""
    require_mps()
    assert messages.device.type == 'mps' and messages.dtype == torch.float32 and len(messages) == 10
    assert bool(torch.isfinite(messages).all())
    sent = messages.clone(); r = None if reports is None else reports.clone()
    if not active_at(attack, t): return sent, r
    honest = messages[2:]; mu = honest.sum(0)/8
    if attack == 'abrupt_bf': vector = -10*mu
    elif attack == 'persistent_alie':
        # Distinct implementation of variance, division by 8 (not 7).
        population_variance = ((honest-mu).square()).sum(0)/8
        vector = mu+1.5*population_variance.sqrt()
    else:
        scale = 2*(t-31)/29 if t <= 60 else 2.
        vector = -scale*mu
    sent[0] = vector; sent[1] = vector
    if r is not None: r[:2] = 1.
    return sent, r


def honest_counts(full, population=1000):
    # Validate all ten records; the corruption mask is never inferred from accuracy.
    clean_audit.exact_counts(full, population)
    acc = [Q(100*int(sum(c['class_hits'])), population) for c in full['clients'][2:]]
    ss = sorted(acc); mean = sum(acc)/8; worst = (ss[0]+ss[1])/2
    return dict(accuracy_pct=mean, worst20_pct=worst,
        gap_best20_worst20_pp=(ss[6]+ss[7])/2-worst,
        variance_pp2=sum(a*a for a in acc)/8-mean*mean)


def verify_honest(full, stored, population):
    values = honest_counts(full, population)
    assert stored['honest_ids'] == list(range(2, 10)) and stored['worst20_client_count'] == 2
    assert stored['clients'] == full['clients'][2:]
    for k, v in values.items(): close(float(v), stored[k], True)
    close(float(values['accuracy_pct']), stored['client_accuracy_pct'], True)
    for metric in ('ce_loss', 'brier_loss'):
        close(stored[metric], st.mean(c[metric] for c in stored['clients']), True)
    close(stored['balanced_accuracy_pct'], 100*st.mean(c['balanced_accuracy'] for c in stored['clients']), True)
    acc = [c['accuracy'] for c in stored['clients']]
    close(stored['gap_best_worst_pp'], 100*(max(acc)-min(acc)), True)
    return values


def independent_decision(records):
    expected = {(j['seed'], j['method'], j['attack']) for j in matrix()}
    if len(records) != 64: raise ValueError('All 64 audited records required')
    index, gaps = {}, {}
    for r in records:
        j = r['job']; key = (j['seed'], j['method'], j['attack'])
        if key in index or key not in expected or set(r['endpoints']) != {'90', '120'}:
            raise ValueError('Unique fixed endpoint grid required')
        index[key] = {int(t): honest_counts(v['test']) for t, v in r['endpoints'].items()}
        if j['method'].endswith('_rfa'):
            value = r['max_solver_gap']
            if not math.isfinite(value) or value < 0: raise ValueError('Invalid solver gap')
            gaps[key] = value
    if set(index) != expected: raise ValueError('Missing fixed comparisons')
    results = {}
    for attack, t in ENDPOINTS.items():
        comparisons = {}
        for control in ('erm_mean', 'erm_rfa', 'risk_mean'):
            differences = [{k: index[s, 'risk_rfa', attack][t][k]-index[s, control, attack][t][k]
                            for k in KEYS} for s in SEEDS]
            summaries = {k: clean_audit.interval([d[k] for d in differences]) for k in KEYS}
            gates = dict(all_seed_gates=all(d['accuracy_pct'] >= -1 and d['worst20_pct'] >= 1 for d in differences),
                worst20_lower_positive=summaries['worst20_pct']['lower_one_sided_9875'] > 0,
                accuracy_lower_noninferior=summaries['accuracy_pct']['lower_one_sided_9875'] >= -1,
                mean_gap_nonincreasing=sum(d['gap_best20_worst20_pp'] for d in differences) <= 0,
                mean_variance_nonincreasing=sum(d['variance_pp2'] for d in differences) <= 0)
            comparisons[control] = dict(primary=control != 'risk_mean',
                pairs=[dict(seed=s, delta={k: float(v) for k, v in d.items()}) for s, d in zip(SEEDS, differences)],
                summaries=summaries, gates=gates, passed=all(gates.values()))
        clean = []
        for s in SEEDS:
            for ep in sorted({t, 120}):
                da = index[s, 'risk_rfa', attack][ep]['accuracy_pct']-index[s, 'risk_rfa', 'none'][ep]['accuracy_pct']
                dw = index[s, 'risk_rfa', attack][ep]['worst20_pct']-index[s, 'risk_rfa', 'none'][ep]['worst20_pct']
                clean.append(dict(seed=s, round=ep, accuracy_delta_pp=float(da), worst20_delta_pp=float(dw),
                    passed=min(da, dw) >= -5))
        top = max(gaps[s, 'risk_rfa', attack] for s in SEEDS)
        passed = (comparisons['erm_mean']['passed'] and comparisons['erm_rfa']['passed']
                  and all(c['passed'] for c in clean) and top <= .001)
        results[attack] = dict(primary_round=t, comparisons=comparisons,
            own_clean_comparisons=clean, solver_max_gap=top, solver_gate=top <= .001, passed=passed)
    top = max(gaps.values())
    return dict(attacks=results, all_rfa_solver_max_gap=top, all_rfa_solver_gate=top <= .001,
        attack_confirmation_passed=all(r['passed'] for r in results.values()) and top <= .001,
        independent_audit_required=True, joint_objective_validated=False, wave=1, one_sided_alpha=.0125,
        scope='Fixed full-population mechanism, four paired seeds, three fixed attacks only')


def compare_structures(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b)
        for k in a: compare_structures(a[k], b[k])
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b)
        for x, y in zip(a, b): compare_structures(x, y)
    elif type(a) is float:
        close(a, b, True)
    else: assert type(a) is type(b) and a == b, (a, b)


def verify_model_chain(rows, initial_hash):
    if not rows or not initial_hash: raise ValueError('Nonempty model chain required')
    previous = initial_hash
    for row in rows:
        assert row['pre_model_sha256'] == previous, 'Model reset or broken trajectory chain'
        assert set(row['model_sha256']) == set(initial_hash)
        previous = row['model_sha256']


def replay_endpoint(j, cp, record, raw, data, profile, key):
    ep = cp['round']; assert ep in (90, 120)
    row = record['rounds'][ep-1]; local = raw['rounds'][ep-1]['clients']; p = record['privacy']
    model = base.new_model(profile, j['seed']); model.load_state_dict(cp['pre_round_model'])
    before = {k: v.clone() for k, v in model.state_dict().items()}
    assert model_hash(model) == row['pre_model_sha256']
    generated = cp['last_generated_messages'].to('mps')
    sent = cp['last_sent_messages'].to('mps'); queries = cp['last_clean_means'].to('mps')
    risks = cp['last_generated_reports']; risks = None if risks is None else risks.to('mps')
    sent_risks = cp['last_sent_reports']; sent_risks = None if sent_risks is None else sent_risks.to('mps')
    assert generated.shape == sent.shape == queries.shape == (10, 61706)
    assert bool(torch.isfinite(generated).all()) and bool(torch.isfinite(sent).all())
    errors, report_errors = [], []
    for cid, ids in enumerate(data['train']):
        ix = clean_audit.random_draw(clean_audit.seed_for(key, j['seed'], ep-1, cid, 'batch'), permutation=True)
        q = torch.zeros(61706, device='mps'); clipped_count = 0
        for sub in ids[ix].split(120):
            _, clipped, norms, _ = per_example(model, data['x'][sub], data['y'][sub], kind='brier', clip_norm=2.)
            q += clipped.sum(0)/4800; clipped_count += int((norms > 2).sum())
        assert clipped_count == local[cid]['gradient']['per_example_clipped_count']
        assert base.ids_hash(queries[cid]) == local[cid]['clean_mean_sha256']
        assert base.ids_hash(generated[cid]) == local[cid]['private_message_sha256']
        torch.testing.assert_close(q, queries[cid], rtol=5e-5, atol=3e-7)
        z = clean_audit.random_draw(clean_audit.seed_for(key, j['seed'], ep-1, cid, 'gaussian'), dimension=(61706,))
        assert torch.equal(queries[cid]+p['gradient_std']*z, generated[cid])
        torch.testing.assert_close(q+p['gradient_std']*z, generated[cid], rtol=5e-5, atol=4e-7)
        errors.append(float((q-queries[cid]).abs().max()))
        if j['method'].startswith('risk_'):
            assert risks is not None and risks.shape == (10,)
            risk = torch.zeros(1, device='mps')
            with torch.no_grad():
                for sub in ids.split(256):
                    probabilities = model(data['x'][sub]).softmax(1)
                    correct = probabilities.gather(1, data['y'][sub, None]).squeeze(1)
                    values = (probabilities.square().sum(1)-2*correct+1)/2
                    risk += values.sum()/4800
            close(float(risk), local[cid]['raw_risk'])
            zr = clean_audit.random_draw(clean_audit.seed_for(key, j['seed'], ep-1, cid, 'risk'), dimension=(1,))
            rp = (risk+p['risk_std']*zr).clamp(0, 1).squeeze()
            torch.testing.assert_close(rp, risks[cid], rtol=3e-5, atol=4e-7)
            assert float(risks[cid]) == local[cid]['private_risk']
            report_errors.append(float((rp-risks[cid]).abs()))
    for k, v in model.state_dict().items(): assert torch.equal(v, before[k])
    expected, expected_risks = independent_attack(generated, risks, j['attack'], ep)
    torch.testing.assert_close(expected, sent, rtol=4e-6, atol=4e-7)
    assert torch.equal(sent[2:], generated[2:])
    if active_at(j['attack'], ep): assert torch.equal(sent[0], sent[1])
    else: assert torch.equal(sent, generated)
    if expected_risks is None: assert sent_risks is None
    else: assert torch.equal(expected_risks, sent_risks)
    assert base.ids_hash(sent) == row['private_messages_sha256']
    assert (None if sent_risks is None else base.ids_hash(sent_risks)) == row['private_reports_sha256']
    if j['method'].startswith('risk_'):
        a = 1+2*(sent_risks.clamp(0, 1)/.5).clamp(max=1); lam = a/a.sum()
    else:
        assert risks is sent_risks is None
        lam = torch.ones(10, device='mps')/10
    torch.testing.assert_close(lam, torch.tensor(row['aggregation']['objective_weights'], device='mps'), rtol=0, atol=0)
    if j['method'].endswith('_rfa'):
        A, solver = weighted_rfa(sent, lam, iterations=40, smoothing=1e-5)
        compare_structures(solver, row['aggregation']['solver'])
        weights = lam/lam.max(); weights = weights/weights.sum()
        distances = ((sent-A).square().sum(1)+1e-10).sqrt()
        effective = weights/distances; effective = effective/effective.sum()
        torch.testing.assert_close(effective, torch.tensor(solver['stationary_weights'], device='mps'), rtol=3e-5, atol=3e-7)
    else:
        A = (lam[:, None]*sent).sum(0); solver = None; effective = lam
    step = .5*A
    assert torch.equal(step, cp['last_step'].to('mps')) and base.ids_hash(step) == row['step_sha256']
    target = queries[2:].mean(0); o = raw['rounds'][ep-1]['aggregate']
    close(float((A-target).square().sum()), o['squared_error_to_unnoised_clipped_honest_mean'])
    close(float(target.square().sum().sqrt()), o['honest_target_norm'])
    close(float(A.square().sum().sqrt()), o['aggregate_norm'])
    close(float(lam[:2].sum()), o['designated_objective_mass'])
    close(float(effective[:2].sum()), o['designated_effective_mass'])
    contribution = float(((effective[:2, None]*sent[:2]).sum(0)).square().sum().sqrt()) if active_at(j['attack'], ep) else 0.
    close(contribution, o['active_byzantine_contribution_norm'])
    # Reconstruction uses an independently calculated effective-weight vector, so
    # a tiny floating-point norm may differ slightly from the source solver's.
    close(float((A-(effective[:, None]*sent).sum(0)).square().sum().sqrt()), o['reconstruction_error'])
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                n = parameter.numel(); parameter.sub_(step[offset:offset+n].reshape(parameter.shape)); offset += n
    assert offset == 61706
    assert all(torch.equal(v, cp['model'][k].to('mps')) for k, v in model.state_dict().items())
    assert model_hash(model) == row['model_sha256']
    validation = base.evaluate(model, data, 'val'); test = base.evaluate(model, data, 'test')
    assert validation == row['validation'] and test == row['test']
    verify(validation); verify(test)
    verify_honest(validation, row['validation_honest'], 1200)
    verify_honest(test, row['test_honest'], 1000)
    result = dict(round=ep, generated_gradient_release_bitwise=True, honest_messages_unchanged=True,
        independent_attack_max_absolute_error=float((expected-sent).abs().max()),
        independent_query_max_absolute_error=max(errors),
        independent_risk_max_absolute_error=max(report_errors, default=0.),
        applied_step_bitwise=True, model_bitwise=True, test_exact=True, validation_exact=True,
        shared_solver=solver is not None, independent_accumulation_block=120)
    del model; torch.mps.empty_cache(); return result


def row_checks(j, rows, oracle, p, draws, initial_hash):
    prefixes = clean_audit.privacy_check(p, dict(seed=j['seed'], batch=4800, method=j['method']))
    assert [r['round'] for r in rows] == [r['round'] for r in oracle] == list(range(1, 121))
    assert [r['round'] for r in rows if r['test'] is not None] == [90, 120]
    assert [r['round'] for r in rows if r['validation'] is not None] == [1, *range(10, 121, 10)]
    verify_model_chain(rows, initial_hash)
    previous = initial_hash; solver_gaps = []; masses = []
    for t, (row, raw) in enumerate(zip(rows, oracle), start=1):
        assert row['device'] == 'mps' and row['pre_model_sha256'] == previous
        previous = row['model_sha256']
        close(row['epsilon_realized'], prefixes[t-1][0], True)
        assert row['epsilon_order'] == prefixes[t-1][1]
        active = active_at(j['attack'], t); threat = row['attack']
        assert threat['attack'] == j['attack'] and threat['active'] == active and threat['byzantine_ids'] == [0, 1]
        multiplier = None if not active else (10. if j['attack'] == 'abrupt_bf' else
            (1.5 if j['attack'] == 'persistent_alie' else 2.*min(t-31, 29)/29))
        assert threat['multiplier'] == multiplier
        risk = j['method'].startswith('risk_')
        assert threat['forged_risk'] == (1. if active and risk else None)
        if active: assert math.isfinite(threat['forged_norm']) and threat['forged_norm'] >= 0
        else: assert threat['forged_norm'] is None
        assert [c['client'] for c in raw['clients']] == list(range(10))
        coeff = []
        for cid, item in enumerate(raw['clients']):
            assert item['indices_sha256'] == draws[t-1][cid][0]
            assert item['batch_prefix240_sha256'] == draws[t-1][cid][1] and item['batch_size'] == 4800
            g = item['gradient']
            assert g['gaussian_releases'] == 1 and g['local_optimizer_steps'] == 0
            assert g['accumulation_blocks'] == 20 and g['noise_added_after_complete_mean']
            close(g['replace_one_sensitivity'], 4/4800, True)
            assert 0 <= g['per_example_clipped_count'] <= 4800
            close(g['per_example_clipped_fraction'], g['per_example_clipped_count']/4800)
            if risk:
                assert 0 <= item['private_risk'] <= 1 and -1e-6 <= item['raw_risk'] <= 1+1e-6
                rp = 1. if active and cid < 2 else item['private_risk']
                coeff.append(1+2*min(rp/.5, 1))
            else:
                assert item['raw_risk'] is item['private_risk'] is None; coeff.append(1.)
        agg = row['aggregation']; expected = [a/sum(coeff) for a in coeff]
        assert agg['eta'] == (2. if t <= 60 else .5)
        assert agg['message_safety']['invalid_message_rows'] == agg['message_safety']['nonfinite_risk_reports'] == 0
        assert len(agg['objective_weights']) == 10
        for actual, ideal in zip(agg['objective_weights'], expected): close(actual, ideal)
        close(agg['max_weight'], max(expected)); close(agg['concentration'], 10*sum(x*x for x in expected))
        assert sum(agg['objective_weights'][:2]) <= 3/7+1e-6
        if j['method'].endswith('_rfa'):
            solver = agg['solver']; assert solver['iterations'] == 40 and solver['smoothing'] == 1e-5
            effective = solver['stationary_weights']
            assert len(effective) == 10 and all(math.isfinite(w) and w > 0 for w in effective)
            close(sum(effective), 1.)
            for k in ('unsmoothed_objective_gap_upper', 'residual_norm', 'stationary_reconstruction_error',
                      'cluster_radius', 'minimizer_radius_bound'):
                assert math.isfinite(solver[k]) and solver[k] >= 0
            solver_gaps.append(solver['unsmoothed_objective_gap_upper'])
        else:
            assert agg['solver'] is None; effective = expected
        o = raw['aggregate']
        for k in ('squared_error_to_unnoised_clipped_honest_mean', 'honest_target_norm', 'aggregate_norm',
                  'active_byzantine_contribution_norm', 'reconstruction_error'):
            assert math.isfinite(o[k]) and o[k] >= 0
        close(o['designated_objective_mass'], sum(expected[:2]))
        close(o['designated_effective_mass'], sum(effective[:2]))
        if not active: assert o['active_byzantine_contribution_norm'] == 0.
        masses.append(o['designated_effective_mass'])
        if row['validation'] is not None:
            verify(row['validation']); verify_honest(row['validation'], row['validation_honest'], 1200)
        else: assert row['validation_honest'] is None
        if row['test'] is not None:
            verify(row['test']); verify_honest(row['test'], row['test_honest'], 1000)
        else: assert row['test_honest'] is None
    return max(solver_gaps, default=None), masses


def audit_one(j, manifest, data, initial, initial_hash, draws, key, stamp):
    folder = OUT/name(j)
    filenames = ('metrics.json', 'orchestration_status.json', 'public_protocol.json',
        'simulator_oracle.json', 'checkpoint.pt', 'endpoint_090.pt', 'endpoint_120.pt')
    files = {f: digest(folder/f) for f in filenames}
    signature = dict(files=files, source_stamp=stamp, key_sha256=digest(PARENT/'simulator_secret.json'))
    cache = CACHE/(name(j)+'.json')
    if cache.exists():
        cached = json.loads(cache.read_text()); assert cached['signature'] == signature
        return cached['record']
    r = json.loads((folder/'metrics.json').read_text())
    status = json.loads((folder/'orchestration_status.json').read_text())
    raw = json.loads((folder/'simulator_oracle.json').read_text())
    public = json.loads((folder/'public_protocol.json').read_text())
    assert r['job'] == status['job'] == j and r['device'] == status['device'] == 'mps'
    assert status['status'] == 'completed' and status['round'] == 120
    assert r['source_stamp'] == manifest['source_stamp']
    for f, field in [('metrics.json', 'metrics_sha256'), ('simulator_oracle.json', 'oracle_sha256'),
            ('checkpoint.pt', 'checkpoint_sha256'), ('endpoint_090.pt', 'endpoint90_sha256'),
            ('endpoint_120.pt', 'endpoint120_sha256')]:
        assert files[f] == status[field]
    assert public == dict(config=manifest['config'], profile=manifest['profile'], job=j,
        privacy=r['privacy'], source_stamp=manifest['source_stamp'])
    assert r['initial'] == initial and r['splits'] == data['splits']
    assert r['final'] == r['rounds'][-1] and r['test_evaluation_rounds'] == [90, 120]
    assert r['gradient_examples_per_client'] == 576000 and r['local_optimizer_steps'] == 0
    assert r['private_gradient_releases_per_client'] == 120
    assert r['validation_test_and_oracles_not_private'] and not raw['privacy_protected'] and not raw['feeds_mechanism']
    prior_folder = PARENT/parent_name(j)
    prior = json.loads((prior_folder/'metrics.json').read_text())
    prior_raw = json.loads((prior_folder/'simulator_oracle.json').read_text())
    assert r['privacy'] == prior['privacy'] and initial == prior['initial'] and r['splits'] == prior['splits']
    largest_gap, masses = row_checks(j, r['rounds'], raw['rounds'], r['privacy'], draws, initial_hash)
    if j['attack'] == 'none':
        assert r['final']['test'] == prior['final']['test']
        assert [x['clients'] for x in raw['rounds']] == [x['clients'] for x in prior_raw['rounds']]
        for row, old in zip(r['rounds'], prior['rounds']):
            assert all(row[k] == old[k] for k in ('aggregation', 'validation', 'epsilon_realized', 'epsilon_order'))
    else:
        control_folder = OUT/name(dict(seed=j['seed'], method=j['method'], attack='none'))
        control = json.loads((control_folder/'metrics.json').read_text())
        control_raw = json.loads((control_folder/'simulator_oracle.json').read_text())
        assert json.loads((control_folder/'orchestration_status.json').read_text())['status'] == 'completed'
        for t in range(30):
            for k in ('aggregation', 'validation', 'epsilon_realized', 'epsilon_order',
                      'private_messages_sha256', 'private_reports_sha256',
                      'step_sha256', 'model_sha256', 'pre_model_sha256'):
                assert r['rounds'][t][k] == control['rounds'][t][k]
            assert raw['rounds'][t]['clients'] == control_raw['rounds'][t]['clients'] == prior_raw['rounds'][t]['clients']
    endpoints, replays = {}, {}
    for ep in (90, 120):
        cp = torch.load(folder/f'endpoint_{ep:03}.pt', map_location='cpu', weights_only=True)
        assert cp['round'] == ep and cp['job'] == j and cp['source_stamp'] == manifest['source_stamp']
        assert cp['key_sha'] == digest(PARENT/'simulator_secret.json') and not cp['privacy_protected']
        assert cp['rows'] == r['rounds'][:ep] and cp['oracles'] == raw['rounds'][:ep]
        assert cp['initial'] == initial and json.loads(json.dumps(cp['privacy'])) == r['privacy']
        if ep == 120:
            last = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=True)
            assert last['rows'] == cp['rows'] and last['oracles'] == cp['oracles']
            # Checkpoint deserialization is host storage; all tensor comparisons MPS.
            for k in ('model', 'pre_round_model'):
                assert set(last[k]) == set(cp[k])
                for field in cp[k]: assert torch.equal(last[k][field].to('mps'), cp[k][field].to('mps'))
            for k in ('last_generated_messages', 'last_sent_messages', 'last_clean_means',
                      'last_generated_reports', 'last_sent_reports', 'last_step'):
                if cp[k] is None: assert last[k] is None
                else: assert torch.equal(last[k].to('mps'), cp[k].to('mps'))
            if j['attack'] == 'none':
                original = torch.load(prior_folder/'checkpoint.pt', map_location='cpu', weights_only=True)
                assert all(torch.equal(v.to('mps'), original['model'][k].to('mps')) for k, v in cp['model'].items())
        replays[str(ep)] = replay_endpoint(j, cp, r, raw, data, manifest['profile'], key)
        row = r['rounds'][ep-1]
        endpoints[str(ep)] = dict(test=row['test'], honest=row['test_honest'])
        del cp; torch.mps.empty_cache()
    record = dict(job=j, endpoints=endpoints, replays=replays, files=files, privacy=r['privacy'],
        elapsed_seconds=r['elapsed_seconds'], max_solver_gap=largest_gap,
        median_effective_designated_mass=st.median(masses),
        exact_prefix_rounds=120 if j['attack'] == 'none' else 30,
        model_hash_chain_verified=True, source=str(folder/'metrics.json'))
    base.verify_stamp(stamp); base.save(cache, dict(signature=signature, record=record))
    print(f"V30 independent audit {name(j)}: privacy, paired prefix, attacks, endpoints90/120 PASS", flush=True)
    return record


def write(records, decision, stamp):
    target = DEST.with_name(DEST.name+('' if len(records) == 64 else '_Partial'))
    parent = json.loads(PARENT_AUDIT.read_text())
    payload = dict(audit_passed=True, valid_runs=len(records), expected_runs=64, records=records,
        source_stamp=stamp, manifest_sha256=digest(OUT/'manifest.json'), decision=decision,
        gate_evaluated=decision is not None, clean_confirmation_passed=True,
        clean_and_attacks_fixed_benchmark_passed=bool(decision and decision['attack_confirmation_passed']),
        global_validation=False, interpretation='Independent scoped benchmark evidence, not universal validation or novelty proof',
        parent_confirmation_manifest_sha256=parent['manifest_sha256'])
    base.save(target.with_suffix('.json'), payload)
    lines = ['# V30 — audit indépendant propre, dommage et récupération', '',
        f'**{len(records)}/64 runs vérifiés ; aucun gate statistique avant la matrice complète.**', '',
        'L’audit recompose la privacy, les attaques, les coefficients, la chaîne des modèles '
        'et les métriques honnêtes. Les gradients et modèles aux tours 90 et 120 sont reconstruits '
        'sur MPS. L’autodiff, le chargement des données et le solveur de médiane restent partagés.', '',
        '| Seed | Condition | Méthode | Tour | Accuracy honnête (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) |',
        '|--:|:--|:--|--:|--:|--:|--:|--:|']
    for r in records:
        for ep, data in r['endpoints'].items():
            v = data['honest']; j = r['job']
            lines.append(f"| {j['seed']} | {j['attack']} | {j['method']} | {ep} | "
                f"{v['accuracy_pct']:.4f} | {v['worst20_pct']:.4f} | {v['gap_best20_worst20_pp']:.4f} | {v['variance_pp2']:.4f} |")
    if decision is not None:
        lines += ['', f"**Gate des trois attaques : {'PASS' if decision['attack_confirmation_passed'] else 'FAIL'}.**", '',
            'Les comparaisons ci-dessous utilisent quatre différences appariées de seeds ; '
            'la borne inférieure est unilatérale à 98,75 %. Les marges propres, '
            'le dommage avant récupération et les contrôles numériques sont aussi obligatoires.', '',
            '| Attaque | Contrôle | Δ accuracy moyen (pp) | Borne inf. | Δ Worst-20 moyen (pp) | Borne inf. | Comparaison |',
            '|:--|:--|--:|--:|--:|--:|:--|']
        for attack, a in decision['attacks'].items():
            for control, c in a['comparisons'].items():
                x, w = c['summaries']['accuracy_pct'], c['summaries']['worst20_pct']
                lines.append(f"| {attack} | {control}{'' if c['primary'] else ' (diagnostic)'} | {x['mean']:.4f} | "
                    f"{x['lower_one_sided_9875']:.4f} | {w['mean']:.4f} | {w['lower_one_sided_9875']:.4f} | "
                    f"{'PASS' if c['passed'] else 'FAIL'} |")
            failed = [c for c in a['own_clean_comparisons'] if not c['passed']]
            lines += ['', f"{attack} : marges face au propre {'respectées' if not failed else 'NON respectées'} ; "
                f"précision candidate {'respectée' if a['solver_gate'] else 'NON respectée'}."]
        lines += ['', 'Les métriques all-clients et la balanced accuracy restent disponibles dans les calculs complets. '
            'Un passage porte uniquement sur cette matrice ; il ne prouve ni optimalité, ni nouveauté, '
            'ni une garantie de performance face à tous les adversaires.']
    lines += ['', '[Protocole](Full_Population_Private_Risk_Attacks_V30_Protocol.md) · '
        f'[Calculs et preuves enregistrées]({target.name}.json).']
    target.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(valid_runs=len(records), gate_evaluated=decision is not None,
        attack_confirmation_passed=None if decision is None else decision['attack_confirmation_passed'],
        global_validation=False)), flush=True)


def main(partial=False):
    require_mps()
    if not (OUT/'manifest.json').is_file():
        raise RuntimeError('V30 not launched: no attack results to audit')
    manifest = json.loads((OUT/'manifest.json').read_text()); stamp = dict(manifest['source_stamp'])
    base.verify_stamp(stamp)
    assert manifest['device'] == 'mps' and manifest['fallback'] is False
    assert manifest['torch_version'] == str(torch.__version__)
    source = [Path(__file__), TEST, Path(clean_audit.__file__)]
    stamp.update({str(p.relative_to(ROOT)): digest(p) for p in source})
    c = manifest['config']
    matrix_file = ROOT/'configs/ldp_gradient_far/full_population_private_risk_attacks_v30.yaml'
    assert c == yaml.safe_load(matrix_file.read_text())
    parent_manifest = json.loads((PARENT/'manifest.json').read_text())
    assert manifest['profile'] == parent_manifest['profile']
    assert manifest['parent_manifest_sha256'] == digest(PARENT/'manifest.json')
    assert c['seeds'] == list(SEEDS) and c['methods'] == list(METHODS) and c['conditions'] == list(ATTACKS)
    assert c['batch_size'] == c['public_train_size'] == 4800 and c['rounds'] == 120
    assert c['primary_endpoints'] == ENDPOINTS and c['one_sided_alpha'] == .0125 and c['confirmation_wave'] == 1
    assert c['byzantine_ids'] == [0, 1] and c['honest_ids'] == list(range(2, 10))
    assert c['maximum_solver_gap_diagnostic'] == .001
    parent = json.loads(PARENT_AUDIT.read_text())
    assert parent['audit_passed'] and parent['gate_evaluated'] and parent['valid_runs'] == 24
    assert parent['manifest_sha256'] == digest(PARENT/'manifest.json')
    assert clean_audit.independent_decision(parent['records'])['clean_confirmation_passed']
    parent_state = json.loads((PARENT/'status.json').read_text())
    assert parent_state['status'] == 'completed' and parent_state['valid_runs'] == 24
    tests = json.loads((OUT/'tests.json').read_text())
    assert tests['passed'] and tests['source_stamp'] == manifest['source_stamp']
    if not (CACHE/'tests.json').exists():
        result = subprocess.run([sys.executable, '-m', 'pytest', str(TEST), '-q'], cwd=ROOT, capture_output=True, text=True)
        base.save(CACHE/'tests.json', dict(passed=result.returncode == 0, source_stamp=stamp,
            output=result.stdout+result.stderr))
    tested = json.loads((CACHE/'tests.json').read_text())
    assert tested['passed'] and tested['source_stamp'] == stamp
    state = json.loads((OUT/'status.json').read_text())
    if not partial: assert state['status'] == 'completed' and state['valid_runs'] == 64
    key = json.loads((PARENT/'simulator_secret.json').read_text())['key']
    assert manifest['simulator_key_sha256'] == digest(PARENT/'simulator_secret.json')
    records = []
    for seed in SEEDS:
        completed = []
        for j in [j for j in matrix() if j['seed'] == seed]:
            path = OUT/name(j)/'orchestration_status.json'
            if path.is_file() and json.loads(path.read_text())['status'] == 'completed': completed.append(j)
        if not completed: continue
        data = base.prepare(manifest['profile'], seed)
        model = base.new_model(manifest['profile'], seed)
        initial = base.evaluate(model, data, 'val'); initial_hash = model_hash(model); del model
        draws = clean_audit.patterns(seed, key)
        for j in completed: records.append(audit_one(j, manifest, data, initial, initial_hash, draws, key, stamp))
        del data; torch.mps.empty_cache()
    decision = None
    if len(records) == 64:
        terminal = json.loads((OUT/'status.json').read_text())
        assert terminal['status'] == 'completed' and terminal['valid_runs'] == 64
        decision = independent_decision(records)
        runner = json.loads((OUT/'evidence_unverified.json').read_text())
        assert runner['source_stamp'] == manifest['source_stamp'] and not runner['independent_audit_passed']
        compare_structures(decision, runner['decision'])
    elif not partial: raise AssertionError('Missing completed attack runs')
    base.verify_stamp(stamp); write(records, decision, stamp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--partial', action='store_true')
    main(parser.parse_args().partial)
