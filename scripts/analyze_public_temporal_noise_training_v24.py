#!/usr/bin/env python3
"""Independent accounting, endpoint and last-private-step verification of V24."""
import json
import math
import os
from pathlib import Path
import statistics as st
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.dont_write_bytecode = True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_public_temporal_noise_training_v24 as run
from scripts import run_fair_objective_screen as base
from scripts.analyze_private_risk_confirmation_v12 import verify
from scripts.audit_public_temporal_noise_v23 import independent_rdp
from privacy.fair_objective import require_mps, per_example, release
from privacy.split_risk_gradient import private_risk
from privacy.stable_weighted_rfa import weighted_rfa

KEYS = ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2',
        'balanced_accuracy_pct','ce_loss','brier_loss')
DEST = ROOT/'output/analysis/Public_Temporal_Noise_Training_V24_Analyse'


def close(a,b,*,strict=False):
    assert math.isclose(a,b,rel_tol=2e-11 if strict else 2e-5,abs_tol=2e-11 if strict else 2e-6),(a,b)


def audit_one(job, data, model, initial, stamp, profile, key):
    path = run.OUT/run.identifier(job)
    r = json.loads((path/'metrics.json').read_text())
    status = json.loads((path/'orchestration_status.json').read_text())
    oracle = json.loads((path/'simulator_oracle.json').read_text())
    assert status['status'] == 'completed' and status['metrics_sha256'] == base.digest(path/'metrics.json')
    assert status['oracle_sha256'] == base.digest(path/'simulator_oracle.json')
    assert r['job'] == job and r['source_stamp'] == stamp and r['device'] == 'mps'
    assert not r['test_evaluated'] and r['initial'] == initial and r['splits'] == data['splits']
    assert r['final'] == r['rounds'][-1] and r['per_client_batch_gradient_evaluations'] == 120
    assert [t['round'] for t in r['rounds']] == list(range(1,121))
    assert [t['round'] for t in r['rounds'] if t['validation'] is not None] == profile['evaluation_rounds']
    assert not oracle['privacy_protected'] and not oracle['feeds_mechanism'] and len(oracle['rounds']) == 120
    prior_path = run.PRIOR/f'seed{job["seed"]}__fresh__{job["method"]}'
    prior = json.loads((prior_path/'metrics.json').read_text())
    prior_oracle = json.loads((prior_path/'simulator_oracle.json').read_text())
    assert r['initial'] == prior['initial'] and r['splits'] == prior['splits']
    p = r['privacy']; risk = job['method'].startswith('risk_')
    assert p['with_risk'] == risk and p['grid_index'] == job['grid_index']
    assert p['adjacency'] == 'replace_one' and p['sampling'] == 'fixed_without_replacement'
    assert p['epsilon_cap'] == 4. and p['delta'] == 1e-5 and (p['N'],p['b'],p['T'],p['C']) == (4800,240,120,2.)
    assert p['risk_std'] == prior['privacy']['risk_std'] and p['gradient_releases'] == 120
    assert p['risk_releases'] == (120 if risk else 0)
    close(p['ratio'],2**(job['grid_index']/16),strict=True)
    close(p['sigma_early'],4*p['z_early']/240,strict=True)
    close(p['sigma_late'],4*p['z_late']/240,strict=True)
    close(p['z_late']/p['z_early'],p['ratio'],strict=True)
    if risk:
        close(p['risk_std'],p['risk_z']/4800,strict=True)
    energy, linear_cost, clip_counts, concentrations, maxweights, solver_gaps = 0.,0.,[],[],[],[]
    for row, raw, old_raw in zip(r['rounds'],oracle['rounds'],prior_oracle['rounds']):
        t = row['round']; assert raw['round'] == old_raw['round'] == t and row['device'] == 'mps'
        assert len(raw['clients']) == 10
        assert [c['client'] for c in raw['clients']] == list(range(10))
        assert [c['batch_hash'] for c in raw['clients']] == [c['batch_hash'] for c in old_raw['clients']]
        phase = 'early' if t <= 60 else 'late'; eta = 2. if t <= 60 else .5
        assert row['noise_schedule']['phase'] == phase
        close(row['noise_schedule']['sigma'],p['sigma_'+phase],strict=True)
        close(row['noise_schedule']['z'],p['z_'+phase],strict=True)
        assert row['noise_schedule']['eta'] == row['aggregation']['eta'] == eta
        early,late = min(t,60),max(0,t-60)
        rdp = {a: early*independent_rdp(a,.05,p['z_early'])+late*independent_rdp(a,.05,p['z_late'])
               +(t*a/(2*p['risk_z']**2) if risk else 0.) for a in range(2,65)}
        epsilon,order = min((v+math.log(1e5)/(a-1),a) for a,v in rdp.items())
        assert epsilon <= 4 and row['privacy_order'] == order
        close(epsilon,row['epsilon_realized'],strict=True)
        if t == 120:
            close(epsilon,p['epsilon_realized'],strict=True)
            for a in rdp:
                close(rdp[a],p['rdp'][str(a)],strict=True)
        if row['validation'] is not None:
            verify(row['validation'])
        diag = row['aggregation']; safe = diag['message_safety']
        assert safe['invalid_message_rows'] == safe['nonfinite_risk_reports'] == 0
        coeff = [1+2*min(max(c['private_risk'],0)/.5,1) for c in raw['clients']] if risk else [1.]*10
        objective = [x/sum(coeff) for x in coeff]
        assert len(diag['objective_weights']) == 10
        for actual,expected in zip(diag['objective_weights'],objective):
            close(actual,expected)
        close(diag['max_weight'],max(objective)); close(diag['concentration'],10*sum(x*x for x in objective))
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
    close(energy,p['energy_proxy'],strict=True)
    cp = torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert cp['source_stamp'] == stamp and cp['job'] == job and cp['round'] == 120
    assert cp['key_sha'] == base.digest(run.PRIOR/'simulator_secret.json')
    assert cp['rows'] == r['rounds'] and cp['oracles'] == oracle['rounds']
    model.load_state_dict(cp['model']); endpoint = base.evaluate(model,data,'val'); verify(endpoint)
    for k in KEYS:
        close(endpoint[k],r['final']['validation'][k])
    # Reconstruct private reports and clipped batch queries at the stored pre-round model.
    model.load_state_dict(cp['previous_model']); sent,reports = [],[]
    for cid,ids in enumerate(data['train']):
        raw = oracle['rounds'][-1]['clients'][cid]
        if risk:
            report,unprotected = private_risk(model,data['x'][ids],data['y'][ids],noise_std=p['risk_std'],
                seed=base.seed_for(key,job['seed'],119,cid,'risk'),N=4800)
            close(float(report),raw['private_risk']); close(float(unprotected),raw['raw_risk'])
            reports.append(report)
        ix = base.draw_indices(4800,240,base.seed_for(key,job['seed'],119,cid,'batch'))
        assert base.ids_hash(ix) == raw['batch_hash']
        _,grad,norms,_ = per_example(model,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)
        assert int((norms>2).sum()) == raw['gradient_clipped_count']
        sent.append(release(grad.mean(0),noise_std=p['sigma_late'],seed=base.seed_for(key,job['seed'],119,cid,'gaussian')))
    X = torch.stack(sent)
    if risk:
        coefficients = 1+2*(torch.stack(reports)/.5).clamp(0,1)
        lam = coefficients/coefficients.sum()
    else:
        lam = torch.ones(10,device='mps')/10
    A = weighted_rfa(X,lam)[0] if job['method'].endswith('rfa') else (lam[:,None]*X).sum(0)
    base.apply_gradient(model,.5*A,1.)
    for name,tensor in model.state_dict().items():
        torch.testing.assert_close(tensor,cp['model'][name].to('mps'),rtol=0,atol=0)
    detail = dict(job=job,endpoint_recomputed=True,last_private_step_bitwise_replayed=True,
        all_accountant_prefixes_verified=True,energy_proxy=energy,eta_weighted_proxy=linear_cost,
        median_local_clipped_fraction=st.median(clip_counts),median_concentration=st.median(concentrations),
        median_max_objective_weight=st.median(maxweights),
        max_numerical_solver_gap_upper=None if not solver_gaps else max(solver_gaps),
        metrics_sha256=base.digest(path/'metrics.json'))
    return r,detail


def decide_independently(records, config):
    index = {(r['job']['seed'],r['job']['grid_index'],r['job']['method']):r['final']['validation'] for r in records}
    for seed in config['seeds']:
        for method in run.METHODS:
            index[seed,'unchanged_v19',method] = json.loads((run.PRIOR/f'seed{seed}__fresh__{method}/metrics.json').read_text())['final']['validation']
        for method in ('erm_mean','erm_rfa'):
            index[seed,'historical_v11',method] = json.loads(run.controls.historical(seed,method).read_text())['final']['validation']
    candidates = []
    for candidate in (7,13):
        contrasts = []
        for control in (7,13,'unchanged_v19','historical_v11'):
            for method in ('erm_mean','erm_rfa'):
                pairs = []
                for seed in config['seeds']:
                    a,b = index[seed,candidate,'risk_rfa'],index[seed,control,method]
                    verify(a); verify(b)
                    delta = {k:a[k]-b[k] for k in KEYS[:4]}
                    pairs.append(dict(seed=seed,delta=delta,passed=delta['accuracy_pct']>=-1. and delta['worst20_pct']>=1.))
                passed = all(p['passed'] for p in pairs) and all(st.mean(p['delta'][k] for p in pairs)<=0 for k in KEYS[2:4])
                contrasts.append(dict(control_mode=control,control_method=method,pairs=pairs,passed=passed,
                                      standard_gaussians_paired=control!='historical_v11'))
        candidates.append(dict(grid_index=candidate,contrasts=contrasts,passed=all(c['passed'] for c in contrasts)))
    eligible = [c['grid_index'] for c in candidates if c['passed']]
    decision = dict(candidates=candidates,selected=None if not eligible else eligible[0],
                    eligible_for_independent_confirmation=bool(eligible),global_validation=False)
    recorded = json.loads((run.OUT/'evidence.json').read_text())['decision']
    assert decision == recorded
    return index,decision


def write_report(index, decision, details, config):
    admitted = decision['eligible_for_independent_confirmation']
    label = {7:'ratio 1,354256',13:'ratio 1,756252','unchanged_v19':'constant V19','historical_v11':'constant V11 (autre bruit)'}
    lines = ['# V24 — transfert au modèle de l’allocation temporelle du bruit','',
        f'**16/16 runs MPS vérifiés** ; admission à confirmation : **{"PASS" if admitted else "FAIL"}**.', '',
        'Seeds 170501/170502 déjà utilisées pour la calibration, ε≤4 et δ=10⁻⁵. Un PASS ne serait pas une confirmation indépendante ni une validation de robustesse.', '',
        '## Moyenne ± écart-type entre seeds','',
        'Statistiques descriptives, ddof=1 ; accuracy/Worst-20 en %, gap en pp, variance en pp². Pas de sélection de checkpoint.', '',
        '| Bruit | Règle | Accuracy | Worst-20 | Gap B20–W20 | Variance | Balanced acc. | CE | Brier |',
        '|:--|:--|--:|--:|--:|--:|--:|--:|--:|']
    for mode in ('unchanged_v19',7,13,'historical_v11'):
        for method in run.METHODS:
            if mode == 'historical_v11' and method.startswith('risk_'):
                continue
            values = [index[s,mode,method] for s in config['seeds']]
            cells = [f'{st.mean(v[k] for v in values):.4f} ± {st.stdev(v[k] for v in values):.4f}' for k in KEYS]
            lines.append('| '+' | '.join([label[mode],method,*cells])+' |')
    lines += ['','## Tous les contrastes primaires : risque-RFA moins ERM','',
        '| Candidate | Contrôle | Seed | Δ accuracy | Δ Worst-20 | Δ gap | Δ variance | Marges par seed |',
        '|:--|:--|--:|--:|--:|--:|--:|:--|']
    for c in decision['candidates']:
        for contrast in c['contrasts']:
            for pair in contrast['pairs']:
                cells = [f'{pair["delta"][k]:+.4f}' for k in KEYS[:4]]
                lines.append('| '+' | '.join([label[c['grid_index']],label[contrast['control_mode']]+'/'+contrast['control_method'],
                    str(pair['seed']),*cells,str(pair['passed'])])+' |')
    lines += ['','## Diagnostics, sans les confondre avec l’utilité','',
        '| Seed | Allocation | Règle | Ση²σ² | Σησ² | Clip local médian % | Concentration médiane | λ max médian | Borne numérique max du solveur |',
        '|--:|:--|:--|--:|--:|--:|--:|--:|--:|']
    for d in details:
        j = d['job']; gap = '—' if d['max_numerical_solver_gap_upper'] is None else f'{d["max_numerical_solver_gap_upper"]:.6g}'
        cells = [f'{d["energy_proxy"]:.6f}',f'{d["eta_weighted_proxy"]:.6f}',f'{100*d["median_local_clipped_fraction"]:.3f}',
                 f'{d["median_concentration"]:.5f}',f'{d["median_max_objective_weight"]:.5f}',gap]
        lines.append('| '+' | '.join([str(j['seed']),label[j['grid_index']],j['method'],*cells])+' |')
    lines += ['','## Portée du verdict','',
        'Les deux allocations viennent du calcul public, pas de l’accuracy. Les critères obligent la candidate à dépasser les deux ERM à chacune des allocations ainsi que les témoins forts V19 et V11. Les branches V19/V24 partagent les gaussiennes standards, pas leurs amplitudes ; V11 a une autre clé. Le risque-moyenne n’est pas un certificat de robustesse.', '',
        'Le proxy gaussien concerne une composante linéaire ; RFA peut modifier son biais et sa covariance. La borne numérique de solveur affichée est un diagnostic float32, pas une preuve universelle en arithmétique exacte. Un meilleur proxy ne suffit pas pour conclure à un meilleur modèle.', '',
        'Audit : tous les budgets et évaluations stockées, tous les hashes de batch, 16 réévaluations finales, reconstruction bitwise du dernier pas incluant les requêtes privées. Ce n’est pas un replay intégral des 16 trajectoires. Les oracles et métriques de validation ne sont pas des releases DP ; les bras appariés ne forment pas un transcript conjoint ε=4.', '',
        'Aucune attaque ni nouvelle seed indépendante n’est lancée automatiquement. La validation conjointe local-DP + fairness + robustesse + performance n’est pas acquise.', '',
        '[Protocole figé](Public_Temporal_Noise_Training_V24_Protocol.md) · [Audit JSON](Public_Temporal_Noise_Training_V24_Analyse.json) · [Calcul public V23](Public_Temporal_Noise_V23_Analyse.md).']
    DEST.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def main():
    require_mps()
    status = json.loads((run.OUT/'status.json').read_text())
    assert status['status'] == 'completed' and status['valid_runs'] == 16
    manifest = json.loads((run.OUT/'manifest.json').read_text())
    config,profile,stamp = manifest['config'],manifest['profile'],manifest['source_stamp']
    base.verify_stamp(stamp)
    tests = json.loads((run.OUT/'tests.json').read_text()); assert tests['passed'] and tests['source_stamp'] == stamp
    key = json.loads((run.PRIOR/'simulator_secret.json').read_text())['key']
    records,details = [],[]
    for seed in config['seeds']:
        data = base.prepare(profile,seed); model = base.new_model(profile,seed)
        initial = base.evaluate(model,data,'val')
        for job in [j for j in run.jobs(config) if j['seed'] == seed]:
            r,d = audit_one(job,data,model,initial,stamp,profile,key)
            records.append(r); details.append(d)
            print(f'{len(records)}/16 ledgers, endpoint models and exact final steps verified',flush=True)
        del data,model; torch.mps.empty_cache()
    assert len(records) == len({run.identifier(r['job']) for r in records}) == 16
    index,decision = decide_independently(records,config)
    base.verify_stamp(stamp)
    base.save(DEST.with_suffix('.json'),dict(audit_passed=True,device='mps',runs=16,source_stamp=stamp,
        auditor_sha256=base.digest(Path(__file__)),details=details,decision=decision,
        scope='All stored metric statistics and accounting prefixes, paired batch hashes, 16 endpoint reevaluations and bitwise private last-step replay; not full trajectory replay.'))
    write_report(index,decision,details,config)
    print(json.dumps(dict(audit_passed=True,selected=decision['selected'],eligible_for_independent_confirmation=decision['eligible_for_independent_confirmation'])),flush=True)


if __name__ == '__main__':
    main()
