#!/usr/bin/env python3
"""Independent endpoint, final-step, accountant and registered-margin audit."""
import json
import math
import os
from pathlib import Path
import statistics as st
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_fresh_private_risk_step_v22_r1 as run
from scripts import run_fair_objective_screen as base
from scripts.analyze_private_risk_confirmation_v12 import verify
from privacy.fair_objective import require_mps,per_example,release,wor_rdp
from privacy.split_risk_gradient import private_risk
from privacy.stable_weighted_rfa import weighted_rfa


def close(a,b):assert math.isclose(a,b,rel_tol=2e-5,abs_tol=2e-6),(a,b)


def main():
    require_mps();out=run.OUT;assert json.loads((out/'status.json').read_text())['status']=='completed'
    manifest=json.loads((out/'manifest.json').read_text());m=manifest['config'];profile=manifest['profile'];stamp=manifest['source_stamp'];base.verify_stamp(stamp)
    assert json.loads((out/'tests.json').read_text())['passed']
    key=json.loads((run.prior.OUT/'simulator_secret.json').read_text())['key'];records=[];details=[]
    expected={(j['seed'],j['step_control'],j['method']) for j in run.jobs(m)}
    for seed in m['seeds']:
        data=base.prepare(profile,seed);model=base.new_model(profile,seed)
        initial=base.evaluate(model,data,'val')
        for j in [x for x in run.jobs(m) if x['seed']==seed]:
            folder=out/run.identifier(j);r=json.loads((folder/'metrics.json').read_text());status=json.loads((folder/'orchestration_status.json').read_text())
            assert status['status']=='completed' and status['metrics_sha256']==base.digest(folder/'metrics.json')
            assert status['oracle_sha256']==base.digest(folder/'simulator_oracle.json')
            assert r['source_stamp']==stamp and r['job']==j and r['device']=='mps' and not r['test_evaluated']
            assert r['initial']==initial and r['splits']==data['splits'] and r['final']==r['rounds'][-1]
            duplicate=ROOT/'results/ldp_gradient_far/fresh_private_risk_step_v22'/run.identifier(j)
            if (duplicate/'metrics.json').exists():
                old_duplicate=json.loads((duplicate/'metrics.json').read_text())
                assert r['rounds']==old_duplicate['rounds'] and r['privacy']==old_duplicate['privacy']
            assert r['per_client_batch_gradient_evaluations']==120
            source=run.prior.OUT/f'seed{seed}__fresh__{j["method"]}'
            old=json.loads((source/'metrics.json').read_text());oracle=json.loads((folder/'simulator_oracle.json').read_text())
            oldoracle=json.loads((source/'simulator_oracle.json').read_text())
            assert not oracle['privacy_protected'] and not oracle['feeds_mechanism']
            assert r['privacy']==old['privacy'] and r['initial']==old['initial'] and r['splits']==old['splits']
            assert [t['round'] for t in r['rounds']]==list(range(1,121))
            assert [t['round'] for t in r['rounds'] if t['validation'] is not None]==profile['evaluation_rounds']
            assert len(oracle['rounds'])==120
            p=r['privacy'];close(p['gradient_std'],4*p['gradient_z']/240)
            factors=[];norms=[]
            for t,raw,oldraw in zip(r['rounds'],oracle['rounds'],oldoracle['rounds']):
                k=t['round'];assert raw['round']==oldraw['round']==k and t['device']=='mps'
                assert [c['batch_hash'] for c in raw['clients']]==[c['batch_hash'] for c in oldraw['clients']]
                assert len(raw['clients'])==10
                eta=2. if k<=60 else .5;d=t['aggregation']['step_control'];close(d['eta'],eta)
                factor=.5 if j['step_control']=='half' else min(1.,1./d['aggregate_norm']) if d['aggregate_norm'] else 1.
                close(factor,d['factor']);close(d['step_norm'],eta*factor*d['aggregate_norm'])
                if j['step_control']=='global_clip':assert d['step_norm']<=eta+3e-6
                safe=t['aggregation']['message_safety'];assert safe['invalid_message_rows']==safe['nonfinite_risk_reports']==0
                eps=min(k*wor_rdp(a,.05,p['gradient_z'])+(k*a/(2*p['risk_z']**2) if j['method'].startswith('risk_') else 0.)
                    +math.log(1e5)/(a-1) for a in range(2,65));close(eps,t['epsilon_realized']);assert eps<=4
                if t['validation'] is not None:verify(t['validation'])
                for c in raw['clients']:
                    assert type(c['gradient_clipped_count']) is int and 0<=c['gradient_clipped_count']<=240 and c['batch_size']==240
                    if j['method'].startswith('risk_'):assert 0<=c['private_risk']<=1
                    else:assert c['private_risk'] is None and c['raw_risk'] is None
                factors.append(d['factor']);norms.append(d['aggregate_norm'])
            close(eps,p['epsilon_realized'])
            cp=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
            assert cp['source_stamp']==stamp and cp['job']==j and cp['round']==120 and cp['key_sha']==base.digest(run.prior.OUT/'simulator_secret.json')
            model.load_state_dict(cp['model']);v=base.evaluate(model,data,'val');verify(v)
            if (duplicate/'checkpoint.pt').exists():
                original_cp=torch.load(duplicate/'checkpoint.pt',map_location='cpu',weights_only=True)
                for name,tensor in model.state_dict().items():torch.testing.assert_close(tensor,original_cp['model'][name].to('mps'),rtol=0,atol=0)
            for k in ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2','balanced_accuracy_pct','ce_loss','brier_loss'):
                close(v[k],r['final']['validation'][k])
            # Recreate the last private queries and update from pre-round model.
            model.load_state_dict(cp['previous_model']);messages=[];reports=[]
            for cid,ids in enumerate(data['train']):
                rr=None;raw=oracle['rounds'][-1]['clients'][cid]
                if j['method'].startswith('risk_'):
                    rr,unprotected=private_risk(model,data['x'][ids],data['y'][ids],noise_std=p['risk_std'],
                        seed=base.seed_for(key,seed,119,cid,'risk'),N=4800)
                    close(float(rr),raw['private_risk']);close(float(unprotected),raw['raw_risk']);reports.append(rr)
                ix=base.draw_indices(4800,240,base.seed_for(key,seed,119,cid,'batch'));assert base.ids_hash(ix)==raw['batch_hash']
                _,g,n,_=per_example(model,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)
                assert int((n>2).sum())==raw['gradient_clipped_count']
                messages.append(release(g.mean(0),noise_std=p['gradient_std'],seed=base.seed_for(key,seed,119,cid,'gaussian')))
            X=torch.stack(messages)
            if reports:
                coeff=1+2*(torch.stack(reports)/.5).clamp(0,1);lam=coeff/coeff.sum()
            else:lam=torch.ones(10,device='mps')/10
            A=weighted_rfa(X,lam)[0] if j['method'].endswith('rfa') else (lam[:,None]*X).sum(0)
            norm=float(torch.linalg.vector_norm(A));factor=.5 if j['step_control']=='half' else min(1.,1./norm) if norm else 1.
            step=.5*(factor*A);base.apply_gradient(model,step,1.)
            for name,tensor in model.state_dict().items():torch.testing.assert_close(tensor,cp['model'][name].to('mps'),rtol=0,atol=0)
            details.append(dict(job=j,endpoint_recomputed=True,last_private_queries_and_step_bitwise_replayed=True,
                factor_min=min(factors),factor_median=st.median(factors),factor_max=max(factors),factor_sd=st.stdev(factors),
                median_aggregate_norm=st.median(norms),metrics_sha256=base.digest(folder/'metrics.json')))
            records.append(r);print(f'{len(records)}/16 endpoints and exact final updates verified',flush=True)
        del data,model;torch.mps.empty_cache()
    assert {(r['job']['seed'],r['job']['step_control'],r['job']['method']) for r in records}==expected
    index={(r['job']['seed'],r['job']['step_control'],r['job']['method']):r['final']['validation'] for r in records}
    for seed in m['seeds']:
        for method in run.METHODS:
            index[seed,'unchanged_v19',method]=json.loads((run.prior.OUT/f'seed{seed}__fresh__{method}/metrics.json').read_text())['final']['validation']
    decisions=[]
    for candidate in ('global_clip','half'):
        contrasts=[]
        for control in ('half','global_clip','unchanged_v19','historical_v11'):
            for kind in ('erm_mean','erm_rfa'):
                pairs=[]
                for seed in m['seeds']:
                    a=index[seed,candidate,'risk_rfa']
                    b=json.loads(run.historical(seed,kind).read_text())['final']['validation'] if control=='historical_v11' else index[seed,control,kind]
                    verify(b);diff={k:a[k]-b[k] for k in ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2')}
                    pairs.append(dict(seed=seed,delta=diff,passed=diff['accuracy_pct']>=-1. and diff['worst20_pct']>=1.))
                passed=all(p['passed'] for p in pairs) and all(st.mean(p['delta'][k] for p in pairs)<=0 for k in ('gap_best20_worst20_pp','variance_pp2'))
                contrasts.append(dict(control=control,kind=kind,pairs=pairs,passed=passed))
        decisions.append(dict(candidate=candidate,contrasts=contrasts,passed=all(c['passed'] for c in contrasts)))
    admitted=[c['candidate'] for c in decisions if c['passed']];selected=admitted[0] if admitted else None
    prior_decision=json.loads((out/'evidence.json').read_text())['decision']
    assert selected==prior_decision['selected'] and bool(admitted)==prior_decision['eligible_for_independent_confirmation']
    for c,d in zip(decisions,prior_decision['candidates']):
        assert c['passed']==d['passed']
        for a,b in zip(c['contrasts'],d['contrasts']):assert a['passed']==b['passed'] and a['pairs']==b['pairs']
    base.verify_stamp(stamp)
    source=ROOT/'output/analysis/Fresh_Private_Risk_Step_V22_Analyse'
    base.save(source.with_suffix('.json'),dict(audit_passed=True,device='mps',runs=16,details=details,candidates=decisions,
        selected=selected,eligible_for_independent_confirmation=bool(admitted),global_validation=False,source_stamp=stamp,
        auditor_sha256=base.digest(Path(__file__)),scope='All stored evaluation statistics, all scalar ledgers and pairing hashes; 16 endpoint reevaluations and bitwise last-step private query replays. Not a full replay of every training trajectory.'))
    lines=['# V22 — effet end-to-end du contrôle du pas privé','',
        f'**16/16 entraînements MPS vérifiés**, seeds de calibration170501/170502, T=120 ; admission à confirmation : **{"PASS" if admitted else "FAIL"}**.', '',
        'Toutes les règles utilisent des gradients frais. Aucun avantage de mémoire n’est revendiqué. Les témoins non contrôlés V19 et historiques V11 restent dans la décision.', '',
        '## Moyenne ± écart-type entre les deux seeds','',
        'Statistiques descriptives (ddof=1), pas des IC de confirmation. Accuracy et Worst-20 en %, gap en pp, variance en pp².', '',
        '| Pas | Règle | Accuracy | Worst-20 | Gap B20–W20 | Variance | Balanced acc. | CE | Brier |',
        '|:--|:--|--:|--:|--:|--:|--:|--:|--:|']
    keys=('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2','balanced_accuracy_pct','ce_loss','brier_loss')
    for control in ('unchanged_v19','half','global_clip'):
        for method in run.METHODS:
            vs=[index[s,control,method] for s in m['seeds']]
            cells=[f'{st.mean(v[k] for v in vs):.4f} ± {st.stdev(v[k] for v in vs):.4f}' for k in keys]
            lines.append('| '+' | '.join([control,method,*cells])+' |')
    lines+=['','## Résultats individuels','',
        '| Seed | Pas | Règle | Accuracy | Worst-20 | Gap | Variance |','|--:|:--|:--|--:|--:|--:|--:|']
    for (seed,control,method),v in sorted(index.items()):
        lines.append(f'| {seed} | {control} | {method} | {v["accuracy_pct"]:.4f} | {v["worst20_pct"]:.4f} | {v["gap_best20_worst20_pp"]:.4f} | {v["variance_pp2"]:.4f} |')
    lines+=['','## Tous les contrastes primaires : risque-RFA moins témoin','',
        '| Candidate | Témoin | Seed | Δ accuracy | Δ Worst-20 | Δ gap | Δ variance | Marges par seed |',
        '|:--|:--|--:|--:|--:|--:|--:|:--|']
    for c in decisions:
        for cc in c['contrasts']:
            for p in cc['pairs']:
                d=p['delta'];cells=[f'{d[k]:+.4f}' for k in keys[:4]]
                lines.append('| '+' | '.join([c['candidate'],cc['control']+'/'+cc['kind'],str(p['seed']),*cells,str(p['passed'])])+' |')
    lines+=['','## Amplitude effective du pas','',
        'Le facteur multiplie le calendrier public η=2 puis0,5 ; il ne mesure pas la fraction de clients exclus.', '',
        '| Seed | Pas | Règle | Facteur min | Médiane | Max | Écart-type temporel | Norme médiane de A |',
        '|--:|:--|:--|--:|--:|--:|--:|--:|']
    for d in details:
        j=d['job'];cells=[f'{d[k]:.5f}' for k in ('factor_min','factor_median','factor_max','factor_sd','median_aggregate_norm')]
        lines.append('| '+' | '.join([str(j['seed']),j['step_control'],j['method'],*cells])+' |')
    lines+=['','## Interprétation autorisée et limites','',
        'Les critères portent sur la performance finale, pas sur un meilleur checkpoint. Une amélioration par rapport à un pas instable ne suffit pas : les deux candidates doivent aussi battre les ERM forts historiques aux marges inchangées. Le contrôle historique V11 partage les seeds mais pas la même clé de bruit.', '',
        'La privacy idéale sample-level côté client reste ε≤4, δ=10⁻⁵, replace-one et batch fixe sans remise. Le contrôle global est un post-traitement. Le test final, les attaques et de nouvelles seeds indépendantes ne sont pas évalués. Les seeds connues, les oracles et les branches à bruit partagé ne constituent pas une confirmation ni un transcript conjoint privé.', '',
        'Tous les derniers pas ont été rejoués exactement depuis le modèle pré-tour120, y compris gradients individuels et rapports privés. Ce n’est pas une réexécution intégrale des16 trajectoires.', '',
        '[Protocole figé](Fresh_Private_Risk_Step_V22_Protocol.md). [Audit et critères](Fresh_Private_Risk_Step_V22_Analyse.json). [Diagnostic V21](PriSMA_Global_Step_V21_Analyse.md).']
    source.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(audit_passed=True,eligible_for_independent_confirmation=bool(admitted),selected=selected)),flush=True)


if __name__=='__main__':main()
