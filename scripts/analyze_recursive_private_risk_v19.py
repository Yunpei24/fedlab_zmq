#!/usr/bin/env python3
"""Independent endpoint, privacy, pairing and registered-gate audit for V19."""
import json
import math
import os
from pathlib import Path
import statistics as st
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_recursive_private_risk_calibration_v19 as run
from scripts import run_fair_objective_screen as base
from scripts.analyze_private_risk_confirmation_v12 import verify
from privacy.fair_objective import require_mps,wor_rdp
from privacy.scheduled_private_risk import aggregate


def close(a,b):
    assert math.isclose(a,b,rel_tol=2e-5,abs_tol=2e-6),(a,b)


def main():
    require_mps();out=run.OUT;manifest=json.loads((out/'manifest.json').read_text())
    m,profile,stamp=manifest['config'],manifest['profile'],manifest['source_stamp']
    base.verify_stamp(stamp)
    assert json.loads((out/'status.json').read_text())['status']=='completed'
    assert json.loads((out/'pairing_audit.json').read_text())['passed']
    assert json.loads((out/'tests.json').read_text())['passed']
    theta=1-math.sqrt(.5+1e-8);D=2*theta;records=[];details=[];patterns={};initials={};splits={}
    expected={(j['seed'],j['mode'],j['method']) for j in run.jobs(m)}
    for seed in m['seeds']:
        data=base.prepare(profile,seed);model=base.new_model(profile,seed)
        for j in [j for j in run.jobs(m) if j['seed']==seed]:
            folder=out/run.identifier(j);r=json.loads((folder/'metrics.json').read_text())
            status=json.loads((folder/'orchestration_status.json').read_text())
            assert status['status']=='completed' and status['metrics_sha256']==base.digest(folder/'metrics.json')
            assert status['oracle_sha256']==base.digest(folder/'simulator_oracle.json')
            assert r['source_stamp']==stamp and r['job']==j and r['device']=='mps' and not r['test_evaluated']
            assert r['final']==r['rounds'][-1] and [t['round'] for t in r['rounds']]==list(range(1,121))
            assert [t['round'] for t in r['rounds'] if t['validation'] is not None]==m['validation_rounds']
            assert r['per_client_batch_gradient_evaluations']==(120 if j['mode']=='fresh' else 239)
            oracle=json.loads((folder/'simulator_oracle.json').read_text());assert not oracle['privacy_protected'] and not oracle['feeds_mechanism']
            assert len(oracle['rounds'])==120
            p=r['privacy'];zg=p['gradient_z'];verify(r['initial'])
            close(p['gradient_std'],4*zg/240)
            if j['method'].startswith('risk_'):close(p['risk_std'],p['risk_z']/4800)
            else:assert p['risk_std']==0 and p['risk_releases']==0
            counts=[];biases=[];noises=[];solver=[]
            phases={name:dict(counts=[],biases=[]) for name in ('rounds_2_20','rounds_21_60','rounds_61_120')}
            for t,raw in zip(r['rounds'],oracle['rounds']):
                k=t['round'];assert raw['round']==k and t['device']=='mps' and len(raw['clients'])==10
                eps=min(k*wor_rdp(a,.05,zg)+(k*a/(2*p['risk_z']**2) if j['method'].startswith('risk_') else 0)
                    +math.log(1e5)/(a-1) for a in range(2,65))
                close(eps,t['epsilon_realized']);assert eps<=4
                eta=2. if k<=60 else .5;close(t['aggregation']['eta'],eta)
                safe=t['aggregation']['message_safety'];assert safe['invalid_message_rows']==safe['nonfinite_risk_reports']==0
                if t['validation'] is not None:verify(t['validation'])
                if t['aggregation']['solver'] is not None:solver.append(t['aggregation']['solver']['unsmoothed_objective_gap_upper'])
                for c in raw['clients']:
                    d=c['recursion'];assert type(c['gradient_clipped_count']) is int and 0<=c['gradient_clipped_count']<=240
                    recursive=j['mode']=='recursive' and k>1
                    effective=theta*2+(1-theta)*D if recursive else 2.
                    close(d['effective_C'],effective);close(d['query_sensitivity'],2*effective/240)
                    close(d['noise_std'],zg*2*effective/240)
                    assert d['initial']==(not recursive) and type(d['clipping_increment_count']) is int
                    assert 0<=d['clipping_increment_count']<=240 and d['batch_size']==240
                    if recursive:
                        close(d['theta'],theta);close(d['D'],D)
                        counts.append(d['clipping_increment_count']/240);biases.append(d['increment_bias_proxy_norm'])
                        phase='rounds_2_20' if k<=20 else ('rounds_21_60' if k<=60 else 'rounds_61_120')
                        phases[phase]['counts'].append(d['clipping_increment_count']/240)
                        phases[phase]['biases'].append(d['increment_bias_proxy_norm'])
                    noises.append(d['noise_std'])
            close(eps,p['epsilon_realized'])
            pattern=[[x['batch_hash'] for x in t['clients']] for t in oracle['rounds']]
            if seed in patterns:assert patterns[seed]==pattern and initials[seed]==r['initial'] and splits[seed]==r['splits']
            else:patterns[seed],initials[seed],splits[seed]=pattern,r['initial'],r['splits']
            cp=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
            assert cp['source_stamp']==stamp and cp['job']==j and cp['round']==120
            assert cp['key_sha']==base.digest(out/'simulator_secret.json')
            model.load_state_dict(cp['model']);validation=base.evaluate(model,data,'val');verify(validation)
            for key in ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2','balanced_accuracy_pct','ce_loss','brier_loss'):
                close(validation[key],r['final']['validation'][key])
            final_step_reconstructed=False
            if j['mode']=='recursive':
                # Final memory contains the actually sent private messages at t=120.
                messages=cp['memory'].to('mps')
                rr=None if j['method'].startswith('erm_') else torch.tensor([c['private_risk'] for c in oracle['rounds'][-1]['clients']],device='mps')
                step,diag=aggregate(messages,rr,kind=j['method'],round_number=120,horizon=120)
                model.load_state_dict(cp['previous_model']);base.apply_gradient(model,step,1.)
                for name,value in model.state_dict().items():
                    torch.testing.assert_close(value,cp['model'][name].to('mps'),rtol=0,atol=0)
                final_step_reconstructed=True
            else:assert cp['memory'] is None
            details.append(dict(job=j,final_validation_recomputed=True,final_step_reconstructed=final_step_reconstructed,
                increment_clipping_median=st.median(counts) if counts else None,
                increment_clipping_p90=sorted(counts)[math.ceil(.9*len(counts))-1] if counts else None,
                mean_increment_bias_proxy=st.mean(biases) if biases else None,
                phase_diagnostics={name:dict(client_rounds=len(v['counts']),
                    median_fraction_clipped=st.median(v['counts']),mean_increment_bias_proxy=st.mean(v['biases']))
                    for name,v in phases.items() if v['counts']},
                noise_std_range=[min(noises),max(noises)],max_solver_bound=max(solver) if solver else None,
                metrics_sha256=base.digest(folder/'metrics.json')))
            records.append(r);print(f'{len(records)}/16 runs independently checked',flush=True)
        del data,model;torch.mps.empty_cache()
    assert len(records)==16 and {(r['job']['seed'],r['job']['mode'],r['job']['method']) for r in records}==expected
    index={(r['job']['seed'],r['job']['mode'],r['job']['method']):r['final']['validation'] for r in records}
    contrasts=[]
    for mode in ('fresh','recursive','historical_v11'):
        for kind in ('erm_mean','erm_rfa'):
            pairs=[]
            for seed in m['seeds']:
                a=index[seed,'recursive','risk_rfa']
                b=(json.loads((run.screen.prior.prior.population.SOURCE/f'seed{seed}__{kind}'/'metrics.json').read_text())['final']['validation']
                   if mode=='historical_v11' else index[seed,mode,kind])
                verify(b)
                diff={k:a[k]-b[k] for k in ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2')}
                pairs.append(dict(seed=seed,delta=diff,passed=diff['accuracy_pct']>=-1 and diff['worst20_pct']>=1))
            passed=all(p['passed'] for p in pairs) and all(st.mean(p['delta'][k] for p in pairs)<=0 for k in ('gap_best20_worst20_pp','variance_pp2'))
            contrasts.append(dict(mode=mode,method=kind,pairs=pairs,passed=passed))
    admitted=all(c['passed'] for c in contrasts)
    assert admitted==json.loads((out/'evidence.json').read_text())['decision']['eligible_for_independent_confirmation']
    secondary=[]
    for seed in m['seeds']:
        a=index[seed,'recursive','risk_rfa']
        for mode,kind in [('fresh','risk_rfa'),('recursive','risk_mean')]:
            b=index[seed,mode,kind]
            secondary.append(dict(seed=seed,control=f'{mode}/{kind}',same_compute=mode=='recursive',
                delta={k:a[k]-b[k] for k in ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2')}))
    source=ROOT/'output/analysis/Recursive_Private_Risk_V19_Analyse'
    base.save(source.with_suffix('.json'),dict(audit_passed=True,device='mps',runs=16,details=details,contrasts=contrasts,secondary_contrasts=secondary,
        eligible_for_independent_confirmation=admitted,global_validation=False,
        source_sha256=base.digest(Path(__file__)),scope='Endpoint reevaluation, exact recursive final-step replay, scalar ledger/pairing/gates; not full replay of all training trajectories'))
    lines=['# V19 — analyse et audit indépendant','',f'**16/16 runs MPS vérifiés** ; calibration propre ; admission à confirmation : **{"PASS" if admitted else "FAIL"}**.', '',
        '| Seed | Requête | Règle | Accuracy (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) | Balanced acc. (%) | CE | Brier |',
        '|--:|:--|:--|--:|--:|--:|--:|--:|--:|--:|']
    for r in records:
        j,v=r['job'],r['final']['validation'];lines.append(f"| {j['seed']} | {j['mode']} | {j['method']} | {v['accuracy_pct']:.3f} | {v['worst20_pct']:.3f} | {v['gap_best20_worst20_pp']:.3f} | {v['variance_pp2']:.3f} | {v['balanced_accuracy_pct']:.3f} | {v['ce_loss']:.5f} | {v['brier_loss']:.5f} |")
    lines+=['','## Résumé sur les deux seeds de calibration','',
        'Moyenne ± écart-type inter-seeds (n=2, ddof=1), pas un intervalle de confiance ni une confirmation indépendante.', '',
        '| Requête | Règle | Accuracy (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) |',
        '|:--|:--|--:|--:|--:|--:|']
    for mode in m['modes']:
        for kind in m['methods']:
            vals=[index[s,mode,kind] for s in m['seeds']]
            cells=[f'{st.mean(v[k] for v in vals):.3f} ± {st.stdev(v[k] for v in vals):.3f}' for k in
                   ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2')]
            lines.append('| '+ ' | '.join([mode,kind,*cells])+' |')
    lines+=['','## Contrastes primaires, risque-RFA récursif moins contrôle','',
        '| Seed | Témoin | Δ acc. (pp) | Δ Worst-20 (pp) | Δ gap (pp) | Δ variance (pp²) | Marges par seed |',
        '|--:|:--|--:|--:|--:|--:|:--|']
    for c in contrasts:
        for p in c['pairs']:
            d=p['delta'];lines.append(f"| {p['seed']} | {c['mode']} / {c['method']} | {d['accuracy_pct']:+.3f} | {d['worst20_pct']:+.3f} | {d['gap_best20_worst20_pp']:+.3f} | {d['variance_pp2']:+.3f} | {p['passed']} |")
    lines+=['','## Contrastes secondaires, sans substitution aux ERM','',
        '| Seed | Témoin | Δ accuracy (pp) | Δ Worst-20 (pp) | Δ gap (pp) | Δ variance (pp²) |',
        '|--:|:--|--:|--:|--:|--:|']
    for p in secondary:
        d=p['delta'];lines.append(f"| {p['seed']} | {p['control']} | {d['accuracy_pct']:+.3f} | {d['worst20_pct']:+.3f} | {d['gap_best20_worst20_pp']:+.3f} | {d['variance_pp2']:+.3f} |")
    lines+=['','## Clipping des différences au fil de la trajectoire','',
        'Ces diagnostics descriptifs portent sur les blocs client-tour, pas sur un pourcentage de clients exclus. Le proxy est la norme du biais de la requête sur le batch courant, pas le vrai biais de population. Les fenêtres sont ajoutées pour diagnostiquer le transfert après observation des premiers runs ; elles ne modifient aucun critère primaire.', '',
        '| Seed | Règle récursive | Tours | Blocs client-tour | Fraction médiane clippée | Norme moyenne du proxy de biais |',
        '|--:|:--|:--|--:|--:|--:|']
    for d in details:
        for phase,v in d['phase_diagnostics'].items():
            lines.append(f"| {d['job']['seed']} | {d['job']['method']} | {phase} | {v['client_rounds']} | {100*v['median_fraction_clipped']:.3f} % | {v['mean_increment_bias_proxy']:.6f} |")
    lines+=['','Les sorties de validation ont été réévaluées sur MPS à partir de chaque checkpoint. Le dernier pas des huit bras récursifs a été reconstruit exactement depuis le modèle précédent et les messages privés mémorisés. Ce n’est pas un replay intégral des 16 trajectoires. Les rapports bruts de diagnostic et métriques ne sont pas couverts par le ledger des communications.', '',
        'Les deux seeds sont de calibration. Aucune robustesse sous attaque n’est établie ici, même si la règle contient RFA. Les bras récursifs coûtent239 évaluations de gradients de batch par client, contre120 pour les bras frais ; les comparaisons à coût égal restent nécessaires.', '',
        '[Protocole](Recursive_Private_Risk_Calibration_V19_Protocol.md). [Dérivation et limites](Recursive_Private_Risk_V19_What_Is_And_Is_Not_Proved.md).']
    source.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(audit_passed=True,eligible_for_independent_confirmation=admitted)))


if __name__=='__main__':main()
