#!/usr/bin/env python3
"""Independent V25 audit; never changes the running experiment or its admission rule."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
import torch
import yaml
from scripts import run_public_temporal_noise_confirmation_v25 as run
from scripts import run_fair_objective_screen as base
from scripts.analyze_private_risk_confirmation_v12 import verify
from scripts.audit_public_temporal_noise_v23 import independent_rdp
from privacy.fair_objective import require_mps, per_example, release
from privacy.split_risk_gradient import private_risk
from privacy.stable_weighted_rfa import weighted_rfa

KEYS = ('accuracy_pct', 'worst20_pct', 'gap_best20_worst20_pp', 'variance_pp2',
        'balanced_accuracy_pct', 'ce_loss', 'brier_loss')
DEST = ROOT / 'output/analysis/Public_Temporal_Noise_Confirmation_V25_Analyse'
SEEDS = (180601, 180602, 180603, 180604)
METHODS = ('erm_mean', 'erm_rfa', 'risk_mean', 'risk_rfa')


def close(a, b, *, scalar=False):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=2e-11 if scalar else 2e-5,
                        abs_tol=2e-11 if scalar else 2e-6), (a, b)


def interval(values):
    """Invert the exact df=3 CDF; no reuse of the runner's interval function."""
    if len(values) != 4 or not all(math.isfinite(x) for x in values):
        raise ValueError('Exactly four finite independent seed differences required')
    low, high = 0., 10.
    for _ in range(70):
        mid = (low + high) / 2
        u = mid / math.sqrt(3)
        cdf = .5 + (math.atan(u) + u / (1 + u*u)) / math.pi
        if cdf < .975:
            low = mid
        else:
            high = mid
    mean = sum(values) / 4
    sd = math.sqrt(sum((x - mean)**2 for x in values) / 3)
    radius = high * sd / 2
    return dict(n=4, mean=mean, sd=sd, ci95=[mean-radius, mean+radius], df=3)


def decide(records):
    expected = {(s, k, m) for s in SEEDS for k in (0, 13) for m in METHODS}
    index = {(r['job']['seed'], r['job']['grid_index'], r['job']['method']): r['final']['test']
             for r in records}
    if len(records) != 32 or len(index) != 32 or set(index) != expected:
        raise ValueError('Full 32-run, four-seed final TEST matrix required')
    contrasts = []
    for mode in (0, 13):
        for method in ('erm_mean', 'erm_rfa'):
            pairs = []
            for seed in SEEDS:
                a, b = index[seed, 13, 'risk_rfa'], index[seed, mode, method]
                delta = {k: a[k] - b[k] for k in KEYS[:4]}
                pairs.append(dict(seed=seed, delta=delta,
                    seed_gate=delta['accuracy_pct'] >= -1. and delta['worst20_pct'] >= 1.))
            summaries = {k: interval([p['delta'][k] for p in pairs]) for k in KEYS[:4]}
            gates = dict(all_seed_gates=all(p['seed_gate'] for p in pairs),
                worst20_ci_lower_positive=summaries['worst20_pct']['ci95'][0] > 0,
                accuracy_ci_lower_noninferior=summaries['accuracy_pct']['ci95'][0] >= -1.,
                mean_gap_nonincreasing=summaries['gap_best20_worst20_pp']['mean'] <= 0,
                mean_variance_nonincreasing=summaries['variance_pp2']['mean'] <= 0)
            contrasts.append(dict(control_grid_index=mode, control_method=method, pairs=pairs,
                                  summaries=summaries, gates=gates, passed=all(gates.values())))
    return dict(clean_confirmation_passed=all(c['passed'] for c in contrasts), contrasts=contrasts,
        selected_grid_index=13, selected_method='risk_rfa', endpoint='fixed final test round 120',
        prospective_confirmation_wave=0, one_sided_alpha=.025, attacks_evaluated=False,
        joint_privacy_fairness_robustness_validated=False)


def compare_evidence(actual, expected):
    """Tolerant scalar comparison, exact structural and Boolean comparison."""
    assert type(actual) is type(expected), (type(actual), type(expected))
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            compare_evidence(actual[key], expected[key])
    elif isinstance(actual, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            compare_evidence(a, b)
    elif isinstance(actual, float):
        close(actual, expected, scalar=True)
    else:
        assert actual == expected, (actual, expected)


def expected_batches(seed, key):
    # This independently reconstructs the draw for every round/client, once per seed.
    return [[base.ids_hash(base.draw_indices(4800, 240, base.seed_for(key, seed, t, cid, 'batch')))
             for cid in range(10)] for t in range(120)]


def audit_one(job, data, model, initial, stamp, profile, key, batches):
    path = run.OUT / run.identifier(job)
    r = json.loads((path/'metrics.json').read_text())
    status = json.loads((path/'orchestration_status.json').read_text())
    oracle = json.loads((path/'simulator_oracle.json').read_text())
    assert status['status'] == 'completed' and status['device'] == 'mps' and status['round'] == 120
    assert status['metrics_sha256'] == base.digest(path/'metrics.json')
    assert status['oracle_sha256'] == base.digest(path/'simulator_oracle.json')
    assert r['job'] == job and status['job'] == job and r['source_stamp'] == stamp and r['device'] == 'mps'
    assert r['test_evaluated'] and r['test_evaluation_rounds'] == [120]
    assert r['initial'] == initial and r['splits'] == data['splits']
    assert r['validation_test_and_oracles_not_private']
    assert r['final'] == r['rounds'][-1] and r['per_client_batch_gradient_evaluations'] == 120
    assert [t['round'] for t in r['rounds']] == list(range(1, 121))
    assert [t['round'] for t in r['rounds'] if t['validation'] is not None] == profile['evaluation_rounds']
    assert [t['round'] for t in r['rounds'] if t['test'] is not None] == [120]
    assert not oracle['privacy_protected'] and not oracle['feeds_mechanism'] and len(oracle['rounds']) == 120
    protocol = json.loads((path/'public_protocol.json').read_text())
    assert protocol['job'] == job and protocol['source_stamp'] == stamp and protocol['privacy'] == r['privacy']
    p = r['privacy']; risk = job['method'].startswith('risk_')
    assert p['with_risk'] == risk and p['grid_index'] == job['grid_index']
    assert p['adjacency'] == 'replace_one' and p['sampling'] == 'fixed_without_replacement'
    assert p['epsilon_cap'] == 4. and p['delta'] == 1e-5 and (p['N'], p['b'], p['T'], p['C']) == (4800, 240, 120, 2.)
    assert p['gradient_releases'] == 120 and p['risk_releases'] == (120 if risk else 0)
    close(p['gradient_sensitivity'], 4/240, scalar=True)
    assert p['risk_sensitivity'] == (1/4800 if risk else None)
    close(p['ratio'], 2**(job['grid_index']/16), scalar=True)
    close(p['sigma_early'], 4*p['z_early']/240, scalar=True)
    close(p['sigma_late'], 4*p['z_late']/240, scalar=True)
    close(p['z_late']/p['z_early'], p['ratio'], scalar=True)
    if risk:
        close(p['risk_std'], p['risk_z']/4800, scalar=True)
        risk_epsilon = min(120*a/(2*p['risk_z']**2) + math.log(1/(.5e-5))/(a-1) for a in range(2, 65))
        assert risk_epsilon <= .25 and p['epsilon_risk'] == .25
    else:
        assert p['risk_std'] == 0. and p['risk_z'] is None
    per_step_early = {a: independent_rdp(a, .05, p['z_early']) for a in range(2, 65)}
    per_step_late = {a: independent_rdp(a, .05, p['z_late']) for a in range(2, 65)}
    clip_counts, concentrations, maxweights, solver_gaps = [], [], [], []
    energy, linear_cost = 0., 0.
    for row, raw in zip(r['rounds'], oracle['rounds']):
        t = row['round']; assert raw['round'] == t and row['device'] == 'mps'
        assert len(raw['clients']) == 10 and [c['client'] for c in raw['clients']] == list(range(10))
        assert [c['batch_hash'] for c in raw['clients']] == batches[t-1]
        phase, eta = ('early', 2.) if t <= 60 else ('late', .5)
        assert row['noise_schedule']['phase'] == phase
        close(row['noise_schedule']['sigma'], p['sigma_'+phase], scalar=True)
        close(row['noise_schedule']['z'], p['z_'+phase], scalar=True)
        assert row['noise_schedule']['eta'] == row['aggregation']['eta'] == eta
        rdp = {a: min(t,60)*per_step_early[a] + max(0,t-60)*per_step_late[a]
               + (t*a/(2*p['risk_z']**2) if risk else 0.) for a in range(2, 65)}
        epsilon, order = min((v+math.log(1e5)/(a-1), a) for a,v in rdp.items())
        assert epsilon <= 4 and row['privacy_order'] == order
        close(epsilon, row['epsilon_realized'], scalar=True)
        if t == 120:
            assert epsilon >= 3.99999, 'A different, more conservative budget is not the frozen comparison'
            close(epsilon, p['epsilon_realized'], scalar=True)
            for a in rdp:
                close(rdp[a], p['rdp'][str(a)], scalar=True)
        for split in ('validation', 'test'):
            if row[split] is not None:
                verify(row[split])
        diag = row['aggregation']; safe = diag['message_safety']
        assert safe['invalid_message_rows'] == safe['nonfinite_risk_reports'] == 0
        coeff = [1+2*min(max(c['private_risk'],0)/.5,1) for c in raw['clients']] if risk else [1.]*10
        objective = [x/sum(coeff) for x in coeff]
        assert len(diag['objective_weights']) == 10
        for actual, expected in zip(diag['objective_weights'], objective):
            close(actual, expected)
        close(diag['max_weight'], max(objective))
        close(diag['concentration'], 10*sum(x*x for x in objective))
        concentrations.append(diag['concentration']); maxweights.append(diag['max_weight'])
        if job['method'].endswith('rfa'):
            assert diag['solver']['iterations'] == 40 and diag['solver']['smoothing'] == 1e-5
            solver_gaps.append(diag['solver']['unsmoothed_objective_gap_upper'])
        else:
            assert diag['solver'] is None
        for c in raw['clients']:
            assert c['batch_size'] == 240 and type(c['gradient_clipped_count']) is int and 0 <= c['gradient_clipped_count'] <= 240
            if risk:
                assert 0 <= c['private_risk'] <= 1
            else:
                assert c['private_risk'] is None and c['raw_risk'] is None
        clip_counts.append(sum(c['gradient_clipped_count'] for c in raw['clients'])/2400)
        energy += eta**2*p['sigma_'+phase]**2
        linear_cost += eta*p['sigma_'+phase]**2
    close(energy, p['energy_proxy'], scalar=True)
    cp = torch.load(path/'checkpoint.pt', map_location='cpu', weights_only=True)
    assert cp['source_stamp'] == stamp and cp['job'] == job and cp['round'] == 120
    assert cp['key_sha'] == base.digest(run.OUT/'simulator_secret.json')
    assert cp['rows'] == r['rounds'] and cp['oracles'] == oracle['rounds'] and cp['initial'] == initial
    model.load_state_dict(cp['model'])
    for split, slot in (('val', 'validation'), ('test', 'test')):
        endpoint = base.evaluate(model, data, split); verify(endpoint)
        for k in KEYS:
            close(endpoint[k], r['final'][slot][k])
        for actual, expected in zip(endpoint['clients'], r['final'][slot]['clients']):
            assert actual['class_count'] == expected['class_count'] and actual['class_hits'] == expected['class_hits']
    model.load_state_dict(cp['previous_model']); sent, reports = [], []
    for cid, ids in enumerate(data['train']):
        raw = oracle['rounds'][-1]['clients'][cid]
        if risk:
            report, unprotected = private_risk(model, data['x'][ids], data['y'][ids], noise_std=p['risk_std'],
                seed=base.seed_for(key,job['seed'],119,cid,'risk'), N=4800)
            close(float(report), raw['private_risk']); close(float(unprotected), raw['raw_risk'])
            reports.append(report)
        ix = base.draw_indices(4800, 240, base.seed_for(key,job['seed'],119,cid,'batch'))
        assert base.ids_hash(ix) == raw['batch_hash']
        _, grad, norms, _ = per_example(model,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)
        assert int((norms>2).sum()) == raw['gradient_clipped_count']
        sent.append(release(grad.mean(0),noise_std=p['sigma_late'],seed=base.seed_for(key,job['seed'],119,cid,'gaussian')))
    X = torch.stack(sent)
    if risk:
        a = 1+2*(torch.stack(reports)/.5).clamp(0,1); lam = a/a.sum()
    else:
        lam = torch.ones(10, device='mps')/10
    A = weighted_rfa(X,lam)[0] if job['method'].endswith('rfa') else (lam[:,None]*X).sum(0)
    base.apply_gradient(model, .5*A, 1.)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, cp['model'][name].to('mps'), rtol=0, atol=0)
    return r, dict(job=job,validation_and_test_recomputed=True,last_private_step_bitwise_replayed=True,
        all_accountant_prefixes_verified=True,all_batch_hashes_reconstructed=True,
        energy_proxy=energy,eta_weighted_proxy=linear_cost,
        median_local_clipped_fraction=st.median(clip_counts),median_concentration=st.median(concentrations),
        median_max_objective_weight=st.median(maxweights),
        max_numerical_solver_gap_upper=None if not solver_gaps else max(solver_gaps),
        metrics_sha256=base.digest(path/'metrics.json'),checkpoint_sha256=base.digest(path/'checkpoint.pt'))


def write_report(records, decision, details, dest):
    lines = ['# V25 — audit indépendant de la confirmation', '',
        f'**{len(records)}/32 résultats terminés audités sur MPS.**', '',
        'La seule candidate primaire est risque-RFA, allocation k13. Les résultats ci-dessous proviennent du test final au tour 120, sans sélection du checkpoint. Les validations intermédiaires ne servent pas au verdict.', '',
        '## Résultats par seed', '',
        'Accuracy et Worst-20 en %, gap B20–W20 en pp, variance en pp².', '',
        '| Seed | Allocation | Règle | Accuracy | Worst-20 | Gap | Variance | Balanced acc. | CE | Demi-Brier |',
        '|--:|:--|:--|--:|--:|--:|--:|--:|--:|--:|']
    for r in records:
        j, v = r['job'], r['final']['test']
        lines.append('| '+' | '.join([str(j['seed']), 'constante' if j['grid_index']==0 else 'k13', j['method'],
                                      *[f'{v[k]:.4f}' for k in KEYS]])+' |')
    if decision is None:
        lines += ['', '**Matrice incomplète : aucun verdict de confirmation.** Les runs restant à terminer sont obligatoires ; ce bilan ne justifie ni sélection ni arrêt anticipé favorable.']
    else:
        index = {(r['job']['seed'],r['job']['grid_index'],r['job']['method']):r['final']['test'] for r in records}
        lines += ['', '## Moyenne ± écart-type sur quatre seeds', '',
            '| Allocation | Règle | Accuracy | Worst-20 | Gap | Variance | Balanced acc. | CE | Demi-Brier |',
            '|:--|:--|--:|--:|--:|--:|--:|--:|--:|']
        for mode in (0,13):
            for method in METHODS:
                cells = [f'{st.mean(index[s,mode,method][k] for s in SEEDS):.4f} ± {st.stdev(index[s,mode,method][k] for s in SEEDS):.4f}' for k in KEYS]
                lines.append('| '+' | '.join(['constante' if mode==0 else 'k13', method, *cells])+' |')
        lines += ['', '## Décision primaire', '',
            f'Confirmation propre : **{"PASS" if decision["clean_confirmation_passed"] else "FAIL"}**.', '',
            'La différence est candidate moins contrôle. Chaque seed doit gagner ≥1 pp de Worst-20 et perdre au plus 1 pp d’accuracy. Les IC appariés doivent aussi respecter les bornes préenregistrées ; les changements moyens de gap et de variance doivent être ≤0. Tous les quatre contrôles sont obligatoires.', '',
            '| Contrôle | Δ accuracy [IC95] | Δ Worst-20 [IC95] | Critères échoués |',
            '|:--|:--|:--|:--|']
        for c in decision['contrasts']:
            cells = []
            for k in KEYS[:2]:
                v = c['summaries'][k]
                cells.append(f'{v["mean"]:+.4f} [{v["ci95"][0]:+.4f} ; {v["ci95"][1]:+.4f}]')
            failed = ', '.join(k for k,v in c['gates'].items() if not v) or 'aucun'
            lines.append('| '+' | '.join([f'k{c["control_grid_index"]}/{c["control_method"]}', *cells, failed])+' |')
        lines += ['', '## Effet propre de l’allocation : diagnostic, pas nouvelle sélection', '',
            'Différences test de risque-RFA k13 moins risque-RFA constant, à seed et tirages standards identiques.', '',
            '| Seed | Δ accuracy | Δ Worst-20 | Δ gap | Δ variance |', '|--:|--:|--:|--:|--:|']
        for s in SEEDS:
            a,b = index[s,13,'risk_rfa'],index[s,0,'risk_rfa']
            lines.append('| '+' | '.join([str(s),*[f'{a[k]-b[k]:+.4f}' for k in KEYS[:4]]])+' |')
    lines += ['', '## Comptabilité et portée de l’audit', '',
        'Chaque exécution alternative protège un exemple local sous adjacence replace-one, batch fixe sans remise : ε≤4, δ=10⁻⁵. Le canal du risque est recomposé avec le gradient ; il n’est pas gratuit. La constante et k13 ont des amplitudes différentes, mais les mêmes gaussiennes standard appariées. La publication conjointe de tous ces bras n’est pas certifiée ε=4.', '',
        'Tous les préfixes RDP, statistiques enregistrées et hashes de batch sont vérifiés. Les modèles finaux sont réévalués en validation et en test, puis le dernier pas privé est reconstruit exactement. Ce n’est pas une réexécution intégrale des trajectoires. Le code de l’accountant est recalculé séparément ; certaines primitives de gradient et RFA sont partagées avec l’implémentation. Les métriques, oracles, données et clés du simulateur ne sont pas des releases DP.', '',
        'Les quatre seeds sont les unités de réplication ; les clients et les tours ne sont pas des réplications indépendantes. Les IC de Student supposent un modèle approprié des différences entre seeds et restent fragiles avec n=4. La règle prospective de dépense du risque statistique ne corrige pas rétroactivement les explorations antérieures.', '',
        '**Aucune validation de robustesse byzantine ni de nouveauté n’est fournie ici.** Un PASS propre serait une étape nécessaire avant des attaques, pas la validation conjointe local-DP + fairness + robustesse + performance.', '',
        '[Protocole figé](Public_Temporal_Noise_Confirmation_V25_Protocol.md) · [Calibration V24](Public_Temporal_Noise_Training_V24_Analyse.md).']
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--completed-only', action='store_true', help='Audit a snapshot of complete jobs; never decide early')
    args = parser.parse_args(); require_mps()
    manifest = json.loads((run.OUT/'manifest.json').read_text())
    config,profile,stamp = manifest['config'],manifest['profile'],manifest['source_stamp']
    assert config == yaml.safe_load(run.MATRIX.read_text())
    expected_profile = dict(dataset='fashionmnist',model='lenet5_tanh',num_clients=10,
        partition='client_dirichlet_balanced',dirichlet_beta=.1,public_train_size=4800,
        validation_size=1200,rounds=120,evaluation_rounds=[1]+list(range(10,121,10)))
    assert profile == expected_profile
    assert tuple(config['confirmation_seeds']) == SEEDS and config['selected_grid_index'] == 13 and config['expected_runs'] == 32
    base.verify_stamp(stamp)
    tests = json.loads((run.OUT/'tests.json').read_text()); assert tests['passed'] and tests['source_stamp'] == stamp
    reservation = json.loads((run.OUT/'seed_reservation.json').read_text())
    assert reservation['seeds'] == list(SEEDS) and reservation['reserved_before_training'] and not reservation['prior_output_path_conflicts']
    selected = []
    for job in run.jobs(config):
        status_file = run.OUT/run.identifier(job)/'orchestration_status.json'
        if status_file.exists() and json.loads(status_file.read_text())['status'] == 'completed':
            selected.append(job)
    if not args.completed_only:
        state = json.loads((run.OUT/'status.json').read_text())
        assert state['status'] == 'completed' and state['valid_runs'] == len(selected) == 32
    assert selected, 'No complete job to audit'
    key = json.loads((run.OUT/'simulator_secret.json').read_text())['key']
    records,details = [],[]
    for seed in SEEDS:
        jobs = [j for j in selected if j['seed'] == seed]
        if not jobs:
            continue
        data = base.prepare(profile,seed); model = base.new_model(profile,seed)
        initial = base.evaluate(model,data,'val'); verify(initial)
        batches = expected_batches(seed,key)
        for job in jobs:
            r,d = audit_one(job,data,model,initial,stamp,profile,key,batches)
            records.append(r); details.append(d)
            print(f'{len(records)}/{len(selected)} complete jobs: ledgers, test/validation endpoints and exact last step verified', flush=True)
        del data,model; torch.mps.empty_cache()
    decision = None
    if not args.completed_only:
        decision = decide(records)
        evidence = json.loads((run.OUT/'evidence.json').read_text())
        assert evidence['source_stamp'] == stamp
        compare_evidence(decision, evidence['decision'])
        assert state['clean_confirmation_passed'] == decision['clean_confirmation_passed']
    base.verify_stamp(stamp)
    dest = DEST if not args.completed_only else DEST.with_name(DEST.name+'_Partial')
    base.save(dest.with_suffix('.json'),dict(audit_passed=True,device='mps',runs=len(records),
        snapshot_partial=args.completed_only,source_stamp=stamp,auditor_sha256=base.digest(Path(__file__)),
        auditor_test_sha256=base.digest(ROOT/'tests/test_public_temporal_noise_v25_audit.py'),
        details=details,decision=decision,final_test_records=[dict(job=r['job'],test=r['final']['test']) for r in records],
        scope='All accounting prefixes, all batch hash reconstruction, stored statistics, final validation/test reevaluation and bitwise final private step; not full trajectory replay'))
    write_report(records,decision,details,dest)
    print(json.dumps(dict(audit_passed=True,runs=len(records),decision=decision and decision['clean_confirmation_passed'])), flush=True)


if __name__ == '__main__':
    main()
