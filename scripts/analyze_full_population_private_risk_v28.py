#!/usr/bin/env python3
"""Independent V28 audit and frozen calibration gate, never a training launcher.

The audit does not import the V28 query, accountant or runner. It re-derives
Gaussian composition, accumulates the final query in blocks of 120 instead
of 240, reconstructs private releases and the final server step on MPS.
Raw data/checkpoints and audit output remain outside the private transcript.
"""
import argparse
import hashlib
import json
import math
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
from scripts.analyze_private_risk_confirmation_v12 import verify
from privacy.fair_objective import require_mps, per_example, losses
from privacy.stable_weighted_rfa import weighted_rfa

OUT = ROOT/'results/ldp_gradient_far/full_population_private_risk_calibration_v28'
OLD = ROOT/'results/ldp_gradient_far/recursive_private_risk_calibration_v19'
DEST = ROOT/'output/analysis/Full_Population_Private_Risk_Calibration_V28_Analyse'
CACHE = ROOT/'output/analysis/audit_full_population_private_risk_v28'
TEST = ROOT/'tests/test_full_population_private_risk_v28_audit.py'
SEEDS = (170501, 170502)
METHODS = ('erm_mean', 'erm_rfa', 'risk_mean', 'risk_rfa')
KEYS = ('accuracy_pct', 'worst20_pct', 'gap_best20_worst20_pp', 'variance_pp2',
        'balanced_accuracy_pct', 'ce_loss', 'brier_loss')
CONTROLS = ('full_erm_mean', 'full_erm_rfa', 'v19_fresh_erm_mean', 'v19_fresh_erm_rfa')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, strict=False):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=2e-11 if strict else 3e-5,
                        abs_tol=2e-11 if strict else 3e-7), (a, b)


def moments(values):
    if len(values) != 2 or not all(math.isfinite(v) for v in values):
        raise ValueError('Two finite calibration seed values required')
    return dict(mean=st.mean(values), sd=st.stdev(values), n=2,
                confidence_interval=None, interpretation='descriptive known-seed calibration only')


def decide(records, historical):
    """No best checkpoint, no method switching, no new criterion after data."""
    def index(rows, methods):
        expected = {(s, m) for s in SEEDS for m in methods}
        result = {(r['job']['seed'], r['job']['method']): r['validation'] for r in rows}
        if len(rows) != len(expected) or set(result) != expected:
            raise ValueError('Full unique frozen matrix required before the gate')
        if any(r.get('endpoint_round') != 120 or r.get('test_evaluated') is not False for r in rows):
            raise ValueError('The endpoint is validation at round 120, not a selected checkpoint or test')
        return result
    current = index(records, METHODS)
    prior = index(historical, ('erm_mean', 'erm_rfa'))
    comparisons = []
    for control in CONTROLS:
        table = current if control.startswith('full_') else prior
        method = control.removeprefix('full_').removeprefix('v19_fresh_')
        pairs = []
        for seed in SEEDS:
            a, b = current[seed, 'risk_rfa'], table[seed, method]
            delta = {k: a[k]-b[k] for k in KEYS}
            if not all(math.isfinite(x) for x in delta.values()):
                raise ValueError('Non-finite paired endpoint')
            pairs.append(dict(seed=seed, delta=delta,
                worst20_margin=delta['worst20_pct'] >= 1.,
                accuracy_noninferiority=delta['accuracy_pct'] >= -1.))
        summaries = {k: moments([p['delta'][k] for p in pairs]) for k in KEYS}
        gates = dict(all_seeds_worst20=all(p['worst20_margin'] for p in pairs),
                     all_seeds_accuracy=all(p['accuracy_noninferiority'] for p in pairs),
                     mean_gap_nonincreasing=summaries['gap_best20_worst20_pp']['mean'] <= 0.,
                     mean_variance_nonincreasing=summaries['variance_pp2']['mean'] <= 0.)
        comparisons.append(dict(control=control, pairs=pairs, summaries=summaries,
                                 gates=gates, passed=all(gates.values()),
                                 equal_compute=control.startswith('full_')))
    return dict(primary_method='risk_rfa', calibration_passed=all(c['passed'] for c in comparisons),
                comparisons=comparisons, confirmation_passed=False, attacks_evaluated=False,
                global_validation=False, automatic_next_campaign=False)


def privacy_audit(plan, method):
    risk = method.startswith('risk_')
    assert (plan['N'], plan['b'], plan['T'], plan['C']) == (4800, 4800, 120, 2.)
    assert plan['adjacency'] == 'replace_one' and plan['sampling'] == 'full_population'
    assert plan['epsilon_cap'] == 4. and plan['delta'] == 1e-5
    assert plan['gradient_releases'] == 120 and plan['risk_releases'] == (120 if risk else 0)
    assert plan['accumulation_blocks_are_not_releases'] is True
    close(plan['gradient_sensitivity'], 4/4800, True)
    close(plan['gradient_std'], plan['gradient_z']*4/4800, True)
    if risk:
        assert plan['risk_sensitivity'] == 1/4800
        close(plan['risk_std'], plan['risk_z']/4800, True)
        assert plan['risk_calibration_epsilon'] == .25 and plan['risk_calibration_delta'] == 5e-6
        risk_epsilon = min(120*a/(2*plan['risk_z']**2)+math.log(2e5)/(a-1) for a in range(2,65))
        assert risk_epsilon <= .25
    else:
        assert plan['risk_z'] is None and plan['risk_std'] == 0. and plan['risk_sensitivity'] is None
    per_round = {a: a/(2*plan['gradient_z']**2)+(a/(2*plan['risk_z']**2) if risk else 0.)
                 for a in range(2,65)}
    prefixes = [min((t*v+math.log(1e5)/(a-1), a) for a,v in per_round.items()) for t in range(1,121)]
    assert all(a[0] <= b[0] for a,b in zip(prefixes, prefixes[1:])) and prefixes[-1][0] <= 4.
    close(prefixes[-1][0], plan['epsilon_realized'], True)
    assert prefixes[-1][1] == plan['order'] and prefixes[-1][0] >= 3.99999
    for a,v in per_round.items():
        close(120*v, plan['rdp'][str(a)], True)
    public = json.loads((ROOT/'output/analysis/Additive_Gaussian_WOR_V27_Public_Audit.json').read_text())
    family = 'private_risk_channel' if risk else 'erm_all_budget'
    audited = next(r['plan'] for r in public['rows'] if r['family']==family and r['plan']['batch']==4800)
    close(plan['gradient_std'], audited['gradient_std'], True)
    close(plan['epsilon_realized'], audited['epsilon_realized'], True)
    return prefixes


def gaussian_like(tensor, seed):
    state = torch.mps.get_rng_state()
    try:
        torch.mps.manual_seed(seed)
        return torch.randn_like(tensor)
    finally:
        torch.mps.set_rng_state(state)


def reconstruct_final(job, data, profile, cp, oracle, p, key):
    """Different gradient accumulation size; only the frozen numerical RFA solver is shared."""
    model = base.new_model(profile, job['seed'])
    model.load_state_dict(cp['pre_round_model'])
    validation_before = base.evaluate(model,data,'val');verify(validation_before)
    old_state = {k:v.clone() for k,v in model.state_dict().items()}
    saved_messages = cp['last_private_messages'].to('mps')
    saved_queries = cp['last_clean_means'].to('mps')
    saved_reports = None if cp['last_reports'] is None else cp['last_reports'].to('mps')
    risk = job['method'].startswith('risk_')
    query_errors, message_errors, report_errors = [], [], []
    for cid, ids in enumerate(data['train']):
        raw = oracle['rounds'][-1]['clients'][cid]
        ix = base.draw_indices(4800,4800,base.seed_for(key,job['seed'],119,cid,'batch'))
        query = torch.zeros(saved_queries.shape[1],device='mps')
        nclip = 0
        for batch in ids[ix].split(120):
            _, rows, norms, _ = per_example(model,data['x'][batch],data['y'][batch],clip_norm=2.)
            query += rows.sum(0)/4800
            nclip += int((norms>2.).sum())
        assert nclip == raw['gradient']['per_example_clipped_count']
        assert base.ids_hash(saved_queries[cid]) == raw['clean_mean_sha256']
        assert base.ids_hash(saved_messages[cid]) == raw['private_message_sha256']
        torch.testing.assert_close(query,saved_queries[cid],rtol=5e-5,atol=3e-7)
        noise = gaussian_like(query,base.seed_for(key,job['seed'],119,cid,'gaussian'))
        message = query + p['gradient_std']*noise
        torch.testing.assert_close(message,saved_messages[cid],rtol=5e-5,atol=4e-7)
        query_errors.append(float((query-saved_queries[cid]).abs().max()))
        message_errors.append(float((message-saved_messages[cid]).abs().max()))
        if risk:
            total = torch.zeros(1,device='mps')
            with torch.no_grad():
                for batch in ids.split(256):
                    total += losses(model(data['x'][batch]),data['y'][batch],'brier').sum()/4800
            close(float(total), raw['raw_risk'])
            report = (total+p['risk_std']*gaussian_like(total,base.seed_for(key,job['seed'],119,cid,'risk'))).clamp(0,1).squeeze()
            torch.testing.assert_close(report,saved_reports[cid],rtol=3e-5,atol=3e-7)
            report_errors.append(float((report-saved_reports[cid]).abs()))
    for k,v in model.state_dict().items():
        assert torch.equal(v,old_state[k]), 'Audit population queries changed model state'
    if risk:
        a = 1+2*(saved_reports.clamp(0,1)/.5).clamp(max=1)
        lam = a/a.sum()
    else:
        assert saved_reports is None
        lam = torch.ones(10,device='mps')/10
    if job['method'].endswith('rfa'):
        center,solver = weighted_rfa(saved_messages,lam)
        distances = ((center-saved_messages).square().sum(1)+1e-10).sqrt()
        w = lam/lam.max();w=w/w.sum()
        residual = (w[:,None]*(center-saved_messages)/distances[:,None]).sum(0)
        close(float(torch.linalg.vector_norm(residual)),solver['residual_norm'])
        nu = w/distances;nu=nu/nu.sum()
        torch.testing.assert_close(nu,torch.tensor(solver['stationary_weights'],device='mps'),rtol=3e-5,atol=3e-7)
    else:
        center=(lam[:,None]*saved_messages).sum(0);solver=None;nu=lam
    step=.5*center
    assert torch.equal(step,cp['last_step'].to('mps')), 'Final server step not exactly reconstructed'
    base.apply_gradient(model,step,1.)
    for k,v in model.state_dict().items():
        assert torch.equal(v,cp['model'][k].to('mps')), 'Final model differs after reconstructed update'
    val=base.evaluate(model,data,'val');verify(val)
    recorded=cp['rows'][-1]['validation']
    assert val==recorded, 'Saved terminal validation not reproducible'
    clean_weighted=(lam[:,None]*saved_queries).sum(0)
    private_weighted=(lam[:,None]*saved_messages).sum(0)
    noise_term=(nu[:,None]*(saved_messages-saved_queries)).sum(0)
    geometry_term=((nu-lam)[:,None]*saved_queries).sum(0)
    numerical_term=center-(nu[:,None]*saved_messages).sum(0)
    error=center-clean_weighted
    torch.testing.assert_close(error,noise_term+geometry_term+numerical_term,rtol=3e-5,atol=3e-7)
    squared={name:float(vector.square().sum()) for name,vector in
             [('noise',noise_term),('geometry',geometry_term),('numerical',numerical_term)]}
    cross=dict(noise_geometry=float(2*torch.dot(noise_term,geometry_term)),
               noise_numerical=float(2*torch.dot(noise_term,numerical_term)),
               geometry_numerical=float(2*torch.dot(geometry_term,numerical_term)))
    close(float(error.square().sum()),sum(squared.values())+sum(cross.values()))
    clients=[]
    for cid,raw in enumerate(oracle['rounds'][-1]['clients']):
        clients.append(dict(client=cid,training_brier_pre120=raw['gradient']['raw_population_brier_risk'],
            validation_accuracy_pre120=100*validation_before['clients'][cid]['accuracy'],
            validation_accuracy_post120=100*val['clients'][cid]['accuracy'],
            private_risk_pre120=raw['private_risk'],objective_weight=float(lam[cid]),
            effective_rfa_weight=float(nu[cid]),clipping_fraction_pre120=raw['gradient']['per_example_clipped_fraction']))
    result=dict(max_query_absolute_error=max(query_errors),max_message_absolute_error=max(message_errors),
                max_report_absolute_error=max(report_errors,default=0.),final_update_bitwise=True,
                final_model_bitwise=True,validation_exact=True,gradient_accumulation_block=120,
                final_private_mean_minus_clean_mean_norm=float(torch.linalg.vector_norm(private_weighted-clean_weighted)),
                final_aggregate_minus_clean_mean_norm=float(torch.linalg.vector_norm(center-clean_weighted)),
                final_clean_mean_norm=float(torch.linalg.vector_norm(clean_weighted)),
                final_error_decomposition=dict(target='same private-risk-weighted clipped population means at pre120',
                    squared_error=float(error.square().sum()),squared_terms=squared,cross_terms=cross,
                    interpretation='single realized conditional diagnostic, not MSE expectation or true fair population gradient',
                    solver_weights_depend_on_current_private_messages=solver is not None),
                clients_pre_post_last_step=clients,
                diagnostics_not_mechanism_inputs=True)
    del model;torch.mps.empty_cache()
    return result


def audit_one(job, manifest, data, initial, patterns, key):
    folder=OUT/f"seed{job['seed']}__{job['method']}"
    names=('metrics.json','simulator_oracle.json','checkpoint.pt','orchestration_status.json','public_protocol.json')
    files={name:digest(folder/name) for name in names}
    cache_signature=dict(files=files,manifest=digest(OUT/'manifest.json'),analysis=digest(__file__),test=digest(TEST),
                         key_sha256=digest(OLD/'simulator_secret.json'))
    cached=CACHE/(folder.name+'.json')
    if cached.exists():
        saved=json.loads(cached.read_text())
        if saved['cache_signature']==cache_signature:
            return saved['record']
    status=json.loads((folder/'orchestration_status.json').read_text())
    r=json.loads((folder/'metrics.json').read_text());raw=json.loads((folder/'simulator_oracle.json').read_text())
    public=json.loads((folder/'public_protocol.json').read_text())
    assert status['status']=='completed' and status['device']=='mps' and status['round']==120 and status['job']==job
    for name,field in [('metrics.json','metrics_sha256'),('simulator_oracle.json','oracle_sha256'),('checkpoint.pt','checkpoint_sha256')]:
        assert status[field]==files[name]
    assert not status['gate_evaluated'] and not r['test_evaluated'] and r['device']=='mps' and r['job']==job
    assert r['source_stamp']==manifest['source_stamp'] and r['splits']==data['splits'] and r['initial']==initial
    assert r['validation_and_oracles_not_private'] and not raw['privacy_protected'] and not raw['feeds_mechanism']
    assert r['gradient_examples_per_client']==576000 and r['private_gradient_releases_per_client']==120 and r['local_optimizer_steps']==0
    assert public['config']==manifest['config'] and public['profile']==manifest['profile'] and public['source_stamp']==manifest['source_stamp']
    assert public['job']==job and public['privacy']==r['privacy']
    assert [row['round'] for row in r['rounds']]==list(range(1,121)) and r['final']==r['rounds'][-1]
    assert [row['round'] for row in r['rounds'] if row['validation'] is not None]==[1,*range(10,121,10)]
    assert [row['round'] for row in raw['rounds']]==list(range(1,121))
    prefixes=privacy_audit(r['privacy'],job['method']);clip_fractions=[];concentrations=[];gaps=[]
    prior=json.loads((OLD/f"seed{job['seed']}__fresh__{job['method']}"/'simulator_oracle.json').read_text())
    verify(r['initial'])
    for row,oracle,pattern,priv in zip(r['rounds'],raw['rounds'],patterns,prefixes):
        t=row['round'];assert row['device']=='mps' and oracle['round']==t
        close(row['epsilon_realized'],priv[0],True);assert row['epsilon_order']==priv[1]
        assert [c['client'] for c in oracle['clients']]==list(range(10))
        assert [c['permutation_sha256'] for c in oracle['clients']]==[v[0] for v in pattern]
        assert [c['legacy_prefix_sha256'] for c in oracle['clients']]==[v[1] for v in pattern]
        assert [c['legacy_prefix_sha256'] for c in oracle['clients']]==[c['batch_hash'] for c in prior['rounds'][t-1]['clients']]
        coeff=[]
        for c in oracle['clients']:
            g=c['gradient']
            assert (g['population_size'],g['block_size'],g['accumulation_blocks'],g['gaussian_releases'])==(4800,240,20,1)
            assert g['local_optimizer_steps']==0 and g['clipping_before_averaging'] and g['noise_added_after_complete_mean']
            assert g['diagnostic_fields_not_private'] and type(g['per_example_clipped_count']) is int
            assert 0<=g['per_example_clipped_count']<=4800 and 0<=g['raw_population_brier_risk']<=1+1e-6
            close(g['per_example_clipped_fraction'],g['per_example_clipped_count']/4800,True)
            close(g['replace_one_sensitivity'],4/4800,True);close(g['gradient_noise_std'],r['privacy']['gradient_std'],True)
            assert 0<=g['query_norm']<=2+1e-5 and 0<=g['raw_gradient_norm_mean']<=g['raw_gradient_norm_max']+1e-5
            if job['method'].startswith('risk_'):
                assert 0<=c['private_risk']<=1 and 0<=c['raw_risk']<=1+1e-6
                close(c['raw_risk'],g['raw_population_brier_risk'])
                coeff.append(1+2*min(c['private_risk']/.5,1))
            else:
                assert c['private_risk'] is None and c['raw_risk'] is None;coeff.append(1.)
        agg=row['aggregation'];assert agg['eta']==(2. if t<=60 else .5)
        safe=agg['message_safety'];assert safe['invalid_message_rows']==safe['nonfinite_risk_reports']==0
        lam=[x/sum(coeff) for x in coeff]
        assert len(agg['objective_weights'])==10
        for actual,expected in zip(agg['objective_weights'],lam):close(actual,expected)
        close(agg['max_weight'],max(lam));close(agg['concentration'],10*sum(w*w for w in lam))
        if job['method'].endswith('rfa'):
            solver=agg['solver'];assert solver['iterations']==40 and solver['smoothing']==1e-5
            assert len(solver['stationary_weights'])==10 and all(w>0 for w in solver['stationary_weights'])
            close(sum(solver['stationary_weights']),1.)
            assert solver['unsmoothed_objective_gap_upper']>=0 and solver['stationary_reconstruction_error']>=0
            gaps.append(solver['unsmoothed_objective_gap_upper'])
        else:assert agg['solver'] is None
        if row['validation'] is not None:verify(row['validation'])
        clip_fractions.append(sum(c['gradient']['per_example_clipped_count'] for c in oracle['clients'])/48000)
        concentrations.append(agg['concentration'])
    cp=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert cp['source_stamp']==manifest['source_stamp'] and cp['job']==job and cp['round']==120
    assert cp['key_sha']==digest(OLD/'simulator_secret.json') and cp['rows']==r['rounds'] and cp['oracles']==raw['rounds']
    # JSON object keys are strings; the checkpoint preserves integer RDP orders.
    assert cp['initial']==initial and json.loads(json.dumps(cp['privacy']))==r['privacy'] and not cp['privacy_protected']
    replay=reconstruct_final(job,data,manifest['profile'],cp,raw,r['privacy'],key)
    result=dict(job=job,validation=r['final']['validation'],endpoint_round=120,test_evaluated=False,
                privacy=r['privacy'],elapsed_seconds=r['elapsed_seconds'],files=files,
                replay=replay,median_clipping_fraction=st.median(clip_fractions),
                median_objective_concentration=st.median(concentrations),maximum_solver_gap=max(gaps,default=None),
                source=str(folder/'metrics.json'))
    base.save(cached,dict(cache_signature=cache_signature,record=result))
    print(f"V28 audit {folder.name}: Gaussian ledger, 120 prefixes, 1200 messages, terminal replay PASS",flush=True)
    return result


def write(records,historical,decision,manifest):
    payload=dict(audit_passed=True,valid_runs=len(records),expected_runs=8,records=records,historical=historical,
                 decision=decision,gate_evaluated=decision is not None,global_validation=False,
                 manifest_sha256=digest(OUT/'manifest.json'),source_sha256=digest(__file__),test_sha256=digest(TEST))
    suffix='' if len(records)==8 else '_Partial'
    dest=DEST.with_name(DEST.name+suffix)
    base.save(dest.with_suffix('.json'),payload)
    verdict='non évalué (matrice incomplète)' if decision is None else ('PASS' if decision['calibration_passed'] else 'FAIL')
    lines=['# V28 — gradient privé de population complète','',
           f'**{len(records)}/8 runs MPS audités. Gate de calibration : {verdict}.**', '',
           'Le test de confirmation n’est pas consulté. Les deux seeds sont connues : les moyennes ± écarts-types '
           'ci-dessous sont descriptives, sans IC de confirmation. La robustesse aux attaques n’est pas évaluée.', '',
           'ERM désigne ici la même loss demi-Brier non pondérée, pas la cross-entropy. « risk » ajoute des poids '
           'issus du risque privé avec a = 1 + 2 min(R/0,5 ; 1), puis λ = a/somme(a). « mean » applique la moyenne '
           'pondérée ; « rfa » applique directement la médiane géométrique pondérée. RFA est donc l’agrégat final '
           'de cette candidate, et non une simple référence pour des poids FAR fondés sur les distances.', '',
           '## Résultats au tour fixé 120', '',
           '| Seed | Méthode | Accuracy (%) | Worst-20 (%) | Gap B20–W20 (pp) | Variance (pp²) | Balanced acc. (%) | CE | Demi-Brier |',
           '|--:|:--|--:|--:|--:|--:|--:|--:|--:|']
    for r in records:
        lines.append(f"| {r['job']['seed']} | {r['job']['method']} | "+' | '.join(f"{r['validation'][k]:.4f}" for k in KEYS)+' |')
    if len(records)==8:
        lines+=['','## Moyenne ± écart-type sur les deux seeds','',
                '| Méthode | Accuracy (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) | Balanced acc. (%) | CE | Demi-Brier |',
                '|:--|--:|--:|--:|--:|--:|--:|--:|']
        for method in METHODS:
            cells=[]
            for k in KEYS:
                q=moments([r['validation'][k] for r in records if r['job']['method']==method])
                cells.append(f"{q['mean']:.4f} ± {q['sd']:.4f}")
            lines.append('| '+method+' | '+' | '.join(cells)+' |')
        for comparison in decision['comparisons']:
            lines+=['',f"## Candidate risque/RFA − {comparison['control']}",'',
                    ('Calcul et budget identiques.' if comparison['equal_compute'] else
                     'Contrôle historique : batch 240 au lieu de 4800. Même budget, mais calcul et variance de sampling différents.'),'',
                    '| Seed | Δ accuracy (pp) | Δ Worst-20 (pp) | Δ gap (pp) | Δ variance (pp²) | Marges par seed |',
                    '|--:|--:|--:|--:|--:|:--|']
            for p in comparison['pairs']:
                lines.append(f"| {p['seed']} | "+' | '.join(f"{p['delta'][k]:+.4f}" for k in KEYS[:4])+f" | {'PASS' if p['worst20_margin'] and p['accuracy_noninferiority'] else 'FAIL'} |")
            lines+=['',f"Critères : {comparison['gates']}. Verdict : **{'PASS' if comparison['passed'] else 'FAIL'}**."]
    if records:
        lines+=['','## Diagnostic du dernier agrégat : une réalisation, pas une MSE moyenne','',
                'La cible de ce diagnostic est h = somme(λᵢ gᵢ), où gᵢ est le gradient de population déjà clippé '
                'et λᵢ le poids de risque effectivement appliqué. Ce n’est pas le gradient équitable exact non clippé. '
                'Si νᵢ sont les poids effectifs de RFA, la différence A − h se décompose en '
                'somme(νᵢ Zᵢ) + somme((νᵢ − λᵢ) gᵢ) + résidu numérique. Pour la moyenne, ν = λ.', '',
                '| Seed | Méthode | Erreur² totale | Bruit² | Déformation géométrique² | Résidu numérique² | Somme des termes croisés |',
                '|--:|:--|--:|--:|--:|--:|--:|']
        for r in records:
            d=r['replay']['final_error_decomposition'];q=d['squared_terms']
            values=[d['squared_error'],q['noise'],q['geometry'],q['numerical'],sum(d['cross_terms'].values())]
            lines.append(f"| {r['job']['seed']} | {r['job']['method']} | "+' | '.join(f'{x:.6g}' for x in values)+' |')
        lines+=['','Les termes croisés sont conservés, car le bruit et la déformation par RFA ne sont pas indépendants. '
                'Ces valeurs sont en unités de gradient au carré, pas en points d’accuracy. Les états des méthodes '
                'ont divergé pendant l’entraînement : cette table n’est pas, à elle seule, une intervention causale '
                'à modèle identique. Une réduction de cette erreur ne remplace aucun critère final d’accuracy ou de fairness.']
    lines+=['','## Ce qui a été vérifié','',
            'Pour chaque run terminé : sources et fichiers inchangés, 120 tours, sorties MPS, appariement des permutations '
            'et préfixes des anciens batchs, comptabilité des deux canaux et tous les préfixes ε. '
            'Le dernier gradient de population est recalculé avec des blocs de 120 au lieu de 240 ; '
            'un seul ajout gaussien par gradient est reconstruit. Le dernier pas, le modèle résultant et la validation sont reproduits.', '',
            'Le comptage par classe reconstitue l’accuracy, le Worst-20 et les gaps ; la variance est une variance '
            'de population entre les dix accuracies clientes, en pp². Le solveur RFA est une approximation numérique '
            'à 40 itérations, pas une médiane géométrique exacte certifiée en arithmétique réelle.', '',
            '## Privacy, coût et portée','',
            'Chaque client calcule au même modèle 4800 gradients demi-Brier, clippe chacun à C = 2, puis moyenne et '
            'ajoute un bruit gaussien une seule fois. Aucun pas local d’optimiseur. Les 20 blocs d’accumulation ne sont '
            'pas 20 libérations DP. Sensibilité replace-one = 4/4800 ; B = N, donc aucune amplification par sampling.', '',
            'ERM utilise tout ε = 4 pour les gradients. La candidate compose le canal de risque privé et le gradient '
            'au même ε total, δ = 10⁻⁵. Écart-type du gradient : 0,01183828 pour ERM et 0,01185582 pour la candidate. '
            'La garantie est au niveau exemple et par exécution/client, pas client-level, ni une garantie conjointe '
            'sur toutes les simulations appariées, validations, checkpoints ou oracles. Le RNG de recherche et '
            'l’arithmétique float32 ne constituent pas une implémentation cryptographique de production.', '',
            'Cette intervention multiplie par vingt les gradients individuels calculés par tour par rapport à B = 240. '
            'Un éventuel gain contre V19 ne peut donc être attribué au seul bruit DP. Une calibration positive '
            'n’autorise aucune revendication de nouveauté ou de robustesse : il reste une confirmation propre '
            'préenregistrée sur de nouvelles seeds, puis les attaques. Aucun lancement automatique ici.', '',
            '[Protocole figé](Full_Population_Private_Risk_Calibration_V28_Protocol.md) · '
            '[Frontière DP publique](Additive_Gaussian_WOR_V27_Public_Audit.md) · '
            f'[Evidence vérifiée]({dest.name}.json).']
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(valid_runs=len(records),gate_evaluated=decision is not None,
                         calibration_passed=None if decision is None else decision['calibration_passed'],
                         global_validation=False,report=str(dest.with_suffix('.md')))),flush=True)


def main(partial=False):
    require_mps();manifest=json.loads((OUT/'manifest.json').read_text());base.verify_stamp(manifest['source_stamp'])
    test_stamp={str(Path(__file__).relative_to(ROOT)):digest(__file__),str(TEST.relative_to(ROOT)):digest(TEST)}
    test_evidence=CACHE/'tests.json'
    previous=json.loads(test_evidence.read_text()) if test_evidence.exists() else None
    if previous is None or previous['source_stamp']!=test_stamp:
        proc=subprocess.run([sys.executable,'-m','pytest',str(TEST),'-q'],cwd=ROOT,capture_output=True,text=True)
        previous=dict(passed=proc.returncode==0,output=proc.stdout+proc.stderr,source_stamp=test_stamp)
        base.save(test_evidence,previous)
    assert previous['passed'],previous['output']
    assert manifest['device']=='mps' and not manifest['fallback']
    assert manifest['config']['seeds']==list(SEEDS) and manifest['config']['methods']==list(METHODS)
    assert manifest['config']['primary_method']=='risk_rfa'
    tests=json.loads((OUT/'tests.json').read_text());assert tests['passed'] and tests['source_stamp']==manifest['source_stamp']
    state=json.loads((OUT/'status.json').read_text())
    if not partial:assert state['status']=='completed' and state['valid_runs']==8 and not state['gate_evaluated']
    key=json.loads((OLD/'simulator_secret.json').read_text())['key']
    expected=[dict(seed=s,method=m) for s in SEEDS for m in METHODS]
    done=[]
    for j in expected:
        folder=OUT/f"seed{j['seed']}__{j['method']}"
        if (folder/'orchestration_status.json').exists() and json.loads((folder/'orchestration_status.json').read_text())['status']=='completed':
            done.append(j)
    if not partial:assert len(done)==8
    records=[];historical=[]
    for seed in SEEDS:
        selected=[j for j in done if j['seed']==seed]
        if not selected:continue
        data=base.prepare(manifest['profile'],seed);model=base.new_model(manifest['profile'],seed)
        initial=base.evaluate(model,data,'val');del model
        patterns=[]
        for t in range(120):
            row=[]
            for cid in range(10):
                ix=base.draw_indices(4800,4800,base.seed_for(key,seed,t,cid,'batch'))
                assert torch.equal(ix.sort().values,torch.arange(4800,device='mps'))
                row.append((base.ids_hash(ix),base.ids_hash(ix[:240])))
            patterns.append(row)
        for j in selected:records.append(audit_one(j,manifest,data,initial,patterns,key))
        for method in ('erm_mean','erm_rfa'):
            folder=OLD/f'seed{seed}__fresh__{method}'
            r=json.loads((folder/'metrics.json').read_text());s=json.loads((folder/'orchestration_status.json').read_text())
            assert s['status']=='completed' and s['metrics_sha256']==digest(folder/'metrics.json')
            assert r['initial']==initial and r['splits']==data['splits'] and r['device']=='mps'
            val=r['final']['validation'];verify(val)
            historical.append(dict(job=dict(seed=seed,method=method),validation=val,endpoint_round=120,
                                   test_evaluated=False,source=str(folder/'metrics.json'),sha256=digest(folder/'metrics.json')))
        del data;torch.mps.empty_cache()
    decision=decide(records,historical) if len(records)==8 else None
    base.verify_stamp(manifest['source_stamp']);write(records,historical,decision,manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true')
    main(parser.parse_args().partial)
