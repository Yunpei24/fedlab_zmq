#!/usr/bin/env python3
"""Independent V29 audit, with no import of its runner/query/accountant/gate.

Shared components are the audited per-example autodiff primitive, dataset/model
loader and numerical RFA solver. Gaussian composition, WOR expansion, final
accumulation, parameter update, exact counts and statistics are re-derived here.
Only completed runs are read. Oracles remain outside the certified transcript.
"""
import argparse
from fractions import Fraction as Q
import hashlib
import json
import math
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_fair_objective_screen as base
from scripts.analyze_private_risk_confirmation_v12 import verify
from privacy.fair_objective import require_mps, per_example
from privacy.stable_weighted_rfa import weighted_rfa

OUT=ROOT/'results/ldp_gradient_far/full_population_private_risk_confirmation_v29'
DEST=ROOT/'output/analysis/Full_Population_Private_Risk_Confirmation_V29_Analyse'
CACHE=ROOT/'output/analysis/audit_full_population_private_risk_confirmation_v29'
TEST=ROOT/'tests/test_full_population_private_risk_confirmation_v29_audit.py'
SEEDS=(180701,180702,180703,180704)
METHODS=('erm_mean','erm_rfa','risk_mean','risk_rfa')
ARMS=tuple((4800,m) for m in METHODS)+((240,'erm_mean'),(240,'erm_rfa'))
CONTROLS=((4800,'erm_mean'),(4800,'erm_rfa'),(240,'erm_mean'),(240,'erm_rfa'))
KEYS=('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2')
EXTRA=('balanced_accuracy_pct','ce_loss','brier_loss')


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def name(j): return f"seed{j['seed']}__b{j['batch']}__{j['method']}"
def close(a,b,strict=False):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a,b,rel_tol=2e-11 if strict else 3e-5,abs_tol=2e-11 if strict else 4e-7),(a,b)


def seed_for(key,*parts):
    message=key+'/'+('/'.join(str(x) for x in parts))
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8],'big')%(2**63-1)


def random_draw(seed, *, permutation=False, dimension=None):
    require_mps(); state=torch.mps.get_rng_state()
    try:
        torch.mps.manual_seed(seed)
        if permutation: return torch.randperm(4800,device='mps')
        return torch.randn(dimension,device='mps',dtype=torch.float32)
    finally: torch.mps.set_rng_state(state)


def independent_wor(order,z):
    # Direct log-binomial sum for the fixed q=.05 generic WBK bound.
    v=1/(z*z); second=min(math.log(4*math.expm1(v)),math.log(2)+v)
    terms=[0.]+[math.log(math.comb(order,j))+j*math.log(.05)+
        (second if j==2 else math.log(2)+j*(j-1)*v/2) for j in range(2,order+1)]
    top=max(terms); logsum=top+math.log(math.fsum(math.exp(x-top) for x in terms))
    return min(order*v/2,logsum/(order-1))


def privacy_check(p,j):
    b=j['batch']; risk=j['method'].startswith('risk_')
    assert (b,j['method']) in ARMS and (p['N'],p['b'],p['T'],p['C'])==(4800,b,120,2.)
    assert p['adjacency']=='replace_one' and p['epsilon_cap']==4. and p['delta']==1e-5
    assert p['sampling']==('full_population' if b==4800 else 'fixed_without_replacement')
    assert p['gradient_releases']==120 and p['risk_releases']==(120 if risk else 0)
    assert p['gradient_z']>0 and p['gradient_std']>0
    close(p['gradient_sensitivity'],4/b,True); close(p['gradient_std'],4*p['gradient_z']/b,True)
    if risk:
        assert b==4800 and p['risk_sensitivity']==1/4800
        close(p['risk_std'],p['risk_z']/4800,True)
        assert p['risk_calibration_epsilon']==.25 and p['risk_calibration_delta']==5e-6
        assert min(120*a/(2*p['risk_z']**2)+math.log(2e5)/(a-1) for a in range(2,65))<=.25
    else:
        assert p['risk_z'] is None and p['risk_std']==0 and p['risk_sensitivity'] is None
    if b==4800: assert p['accumulation_blocks_are_not_releases'] is True
    rdps={a:(a/(2*p['gradient_z']**2) if b==4800 else independent_wor(a,p['gradient_z']))+
        (a/(2*p['risk_z']**2) if risk else 0.) for a in range(2,65)}
    assert {int(k) for k in p['rdp']}==set(range(2,65))
    for a,v in rdps.items(): close(120*v,p['rdp'][str(a)],True)
    prefixes=[min((t*v+math.log(1e5)/(a-1),a) for a,v in rdps.items()) for t in range(1,121)]
    assert all(x[0]<=y[0] for x,y in zip(prefixes,prefixes[1:]))
    assert 3.99999<=prefixes[-1][0]<=4. and prefixes[-1][1]==p['order']
    close(prefixes[-1][0],p['epsilon_realized'],True)
    return prefixes


def exact_counts(v, population=1000):
    if len(v['clients'])!=10: raise ValueError('Ten clients required')
    numbers=[]
    for c in v['clients']:
        counts,hits=c['class_count'],c['class_hits']
        if (len(counts)!=10 or len(hits)!=10 or c['N']!=population or sum(counts)!=population
                or any(not math.isfinite(x) or x<0 or x!=int(x) for x in counts+hits)
                or any(h>n for h,n in zip(hits,counts))): raise ValueError('Invalid exact counts')
        numbers.append(Q(100*int(sum(hits)),population))
    ordered=sorted(numbers); average=sum(numbers)/10
    worst=(ordered[0]+ordered[1])/2; best=(ordered[8]+ordered[9])/2
    result=dict(accuracy_pct=average,worst20_pct=worst,gap_best20_worst20_pp=best-worst,
        variance_pp2=sum(a*a for a in numbers)/10-average*average)
    for k,x in result.items():
        if not math.isfinite(v[k]) or abs(float(x)-v[k])>1e-9: raise ValueError('Counts/metrics disagree')
    return result


def interval(values):
    if len(values)!=4 or any(not isinstance(v,Q) for v in values): raise ValueError('Four exact differences required')
    lo,hi=0.,10.
    for _ in range(80):
        mid=(lo+hi)/2; u=mid/math.sqrt(3)
        cdf=.5+(math.atan(u)+u/(1+u*u))/math.pi
        if cdf<.9875: lo=mid
        else: hi=mid
    mean=sum(values)/4; variance=sum((v-mean)**2 for v in values)/3
    sd=math.sqrt(float(variance))
    return dict(n=4,mean=float(mean),sd=sd,lower_one_sided_9875=float(mean)-hi*sd/2,df=3,
        exact_mean=dict(numerator=mean.numerator,denominator=mean.denominator))


def independent_decision(records):
    expected={(s,b,m) for s in SEEDS for b,m in ARMS}; index={}
    if len(records)!=24: raise ValueError('No incomplete confirmation gate')
    for r in records:
        j=r['job']; key=(j['seed'],j['batch'],j['method'])
        if key in index or key not in expected or r['endpoint_round']!=120 or not r['test_evaluated']:
            raise ValueError('Unique fixed final test matrix required')
        index[key]=exact_counts(r['test'])
    if set(index)!=expected: raise ValueError('Missing controls')
    contrasts=[]
    for b,m in CONTROLS:
        ds=[{k:index[s,4800,'risk_rfa'][k]-index[s,b,m][k] for k in KEYS} for s in SEEDS]
        summaries={k:interval([d[k] for d in ds]) for k in KEYS}
        gates=dict(all_seed_gates=all(d['accuracy_pct']>=-1 and d['worst20_pct']>=1 for d in ds),
            worst20_lower_positive=summaries['worst20_pct']['lower_one_sided_9875']>0,
            accuracy_lower_noninferior=summaries['accuracy_pct']['lower_one_sided_9875']>=-1,
            mean_gap_nonincreasing=sum(d['gap_best20_worst20_pp'] for d in ds)<=0,
            mean_variance_nonincreasing=sum(d['variance_pp2'] for d in ds)<=0)
        contrasts.append(dict(control_batch=b,control_method=m,summaries=summaries,gates=gates,passed=all(gates.values()),
            pairs=[dict(seed=s,delta={k:float(v) for k,v in d.items()}) for s,d in zip(SEEDS,ds)]))
    return dict(clean_confirmation_passed=all(c['passed'] for c in contrasts),contrasts=contrasts,
        primary_method='risk_rfa',primary_batch=4800,wave=1,one_sided_alpha=.0125,
        endpoint='final_test_round_120',attacks_evaluated=False,joint_objective_validated=False)


def compare_runner(independent,runner):
    for k in ('clean_confirmation_passed','primary_method','primary_batch','wave','one_sided_alpha','endpoint'):
        assert independent[k]==runner[k],k
    assert len(independent['contrasts'])==len(runner['contrasts'])==4
    for a,b in zip(independent['contrasts'],runner['contrasts']):
        for k in ('control_batch','control_method','gates','passed','pairs'): assert a[k]==b[k],k
        for k in KEYS:
            for field in ('mean','sd','lower_one_sided_9875'): close(a['summaries'][k][field],b['summaries'][k][field],True)
            assert a['summaries'][k]['exact_mean']==b['summaries'][k]['exact_mean']


def patterns(seed,key):
    result=[]
    for t in range(120):
        row=[]
        for cid in range(10):
            ix=random_draw(seed_for(key,seed,t,cid,'batch'),permutation=True)
            assert torch.equal(ix.sort().values,torch.arange(4800,device='mps'))
            row.append((base.ids_hash(ix),base.ids_hash(ix[:240])))
        result.append(row)
    return result


def replay(j,data,profile,cp,oracle,p,key):
    model=base.new_model(profile,j['seed']); model.load_state_dict(cp['pre_round_model'])
    before={k:v.clone() for k,v in model.state_dict().items()}
    X=cp['last_private_messages'].to('mps'); queries=cp['last_clean_means'].to('mps')
    assert X.shape==queries.shape==(10,61706) and bool(torch.isfinite(X).all())
    risks=cp['last_reports']; risks=None if risks is None else risks.to('mps')
    errs=[]; private_errs=[]; risk_errs=[]
    for cid,ids in enumerate(data['train']):
        ix=random_draw(seed_for(key,j['seed'],119,cid,'batch'),permutation=True)[:j['batch']]
        raw=oracle['rounds'][-1]['clients'][cid]; q=torch.zeros(61706,device='mps'); nclip=0
        for sub in ids[ix].split(120):
            _,clipped,norms,_=per_example(model,data['x'][sub],data['y'][sub],kind='brier',clip_norm=2.)
            q+=clipped.sum(0)/j['batch']; nclip+=int((norms>2).sum())
        assert nclip==raw['gradient']['per_example_clipped_count']
        assert base.ids_hash(queries[cid])==raw['clean_mean_sha256'] and base.ids_hash(X[cid])==raw['private_message_sha256']
        torch.testing.assert_close(q,queries[cid],rtol=5e-5,atol=3e-7)
        z=random_draw(seed_for(key,j['seed'],119,cid,'gaussian'),dimension=(61706,))
        assert torch.equal(queries[cid]+p['gradient_std']*z,X[cid])
        torch.testing.assert_close(q+p['gradient_std']*z,X[cid],rtol=5e-5,atol=4e-7)
        errs.append(float((q-queries[cid]).abs().max()))
        private_errs.append(float((q+p['gradient_std']*z-X[cid]).abs().max()))
        if j['method'].startswith('risk_'):
            assert risks is not None and risks.shape==(10,)
            total=torch.zeros(1,device='mps')
            with torch.no_grad():
                for sub in ids.split(256):
                    prob=model(data['x'][sub]).softmax(1); lab=data['y'][sub]
                    # Independent polynomial expression of half-Brier, not the source helper.
                    rr=.5*(prob.square().sum(1)-2*prob.gather(1,lab[:,None]).squeeze(1)+1)
                    total+=rr.sum()/4800
            close(float(total),raw['raw_risk'])
            zr=random_draw(seed_for(key,j['seed'],119,cid,'risk'),dimension=(1,))
            released=(total+p['risk_std']*zr).clamp(0,1).squeeze()
            torch.testing.assert_close(released,risks[cid],rtol=3e-5,atol=4e-7)
            risk_errs.append(float((released-risks[cid]).abs()))
    for k,v in model.state_dict().items(): assert torch.equal(v,before[k])
    if j['method'].startswith('risk_'):
        a=1+2*(risks.clamp(0,1)/.5).clamp(max=1); lam=a/a.sum()
    else:
        assert risks is None; lam=torch.ones(10,device='mps')/10
    if j['method'].endswith('rfa'):
        A,solver=weighted_rfa(X,lam,iterations=40,smoothing=1e-5)
        d=((A-X).square().sum(1)+1e-10).sqrt(); w=lam/lam.max(); w=w/w.sum()
        effective=w/d; effective=effective/effective.sum()
        torch.testing.assert_close(effective,torch.tensor(solver['stationary_weights'],device='mps'),rtol=3e-5,atol=3e-7)
    else: A=(lam[:,None]*X).sum(0); solver=None
    step=.5*A
    assert torch.equal(step,cp['last_step'].to('mps'))
    offset=0
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                size=parameter.numel(); parameter.sub_(step[offset:offset+size].reshape(parameter.shape)); offset+=size
    assert offset==61706
    for k,v in model.state_dict().items(): assert torch.equal(v,cp['model'][k].to('mps'))
    val=base.evaluate(model,data,'val'); test=base.evaluate(model,data,'test')
    assert val==cp['rows'][-1]['validation'] and test==cp['rows'][-1]['test']
    verify(val); verify(test); exact_counts(val,1200); exact_counts(test,1000)
    result=dict(max_query_absolute_error=max(errs),max_message_absolute_error=max(private_errs),
        max_risk_absolute_error=max(risk_errs,default=0.),saved_query_release_bitwise=True,
        final_step_bitwise=True,final_model_bitwise=True,validation_exact=True,test_exact=True,
        independent_accumulation_block=120,raw_oracles_not_private=True,solver=solver)
    del model; torch.mps.empty_cache(); return result


def audit_one(j,manifest,data,initial,draws,key,stamp):
    folder=OUT/name(j)
    files={f:digest(folder/f) for f in ('metrics.json','simulator_oracle.json','orchestration_status.json','checkpoint.pt','public_protocol.json')}
    cache=CACHE/(name(j)+'.json'); signature=dict(files=files,source_stamp=stamp,key_sha256=digest(OUT/'simulator_secret.json'))
    if cache.exists():
        old=json.loads(cache.read_text()); assert old['signature']==signature
        return old['record']
    r=json.loads((folder/'metrics.json').read_text()); status=json.loads((folder/'orchestration_status.json').read_text())
    raw=json.loads((folder/'simulator_oracle.json').read_text()); public=json.loads((folder/'public_protocol.json').read_text())
    assert status['status']=='completed' and status['round']==120 and status['job']==r['job']==j
    assert status['device']==r['device']=='mps' and r['source_stamp']==manifest['source_stamp']
    for f,k in [('metrics.json','metrics_sha256'),('simulator_oracle.json','oracle_sha256'),('checkpoint.pt','checkpoint_sha256')]: assert files[f]==status[k]
    assert public==dict(config=manifest['config'],profile=manifest['profile'],job=j,privacy=r['privacy'],source_stamp=manifest['source_stamp'])
    assert r['initial']==initial and r['splits']==data['splits']
    assert r['test_evaluated'] and r['test_evaluation_rounds']==[120] and r['local_optimizer_steps']==0
    assert r['gradient_examples_per_client']==120*j['batch'] and r['private_gradient_releases_per_client']==120
    assert r['validation_test_and_oracles_not_private'] and not raw['privacy_protected'] and not raw['feeds_mechanism']
    prefixes=privacy_check(r['privacy'],j)
    assert r['final']==r['rounds'][-1]
    assert [v['round'] for v in r['rounds']]==[v['round'] for v in raw['rounds']]==list(range(1,121))
    assert [v['round'] for v in r['rounds'] if v['test'] is not None]==[120]
    assert [v['round'] for v in r['rounds'] if v['validation'] is not None]==[1,*range(10,121,10)]
    clipping=[]; concentration=[]
    for t,(v,oracle,ep) in enumerate(zip(r['rounds'],raw['rounds'],prefixes)):
        assert v['device']=='mps'; close(v['epsilon_realized'],ep[0],True); assert v['epsilon_order']==ep[1]
        assert [c['client'] for c in oracle['clients']]==list(range(10))
        coeff=[]; clipped=0
        for cid,c in enumerate(oracle['clients']):
            assert c['indices_sha256']==draws[t][cid][0 if j['batch']==4800 else 1]
            assert c['batch_prefix240_sha256']==draws[t][cid][1] and c['batch_size']==j['batch']
            g=c['gradient']; assert g['local_optimizer_steps']==0 and g['gaussian_releases']==1
            assert g['accumulation_blocks']==(20 if j['batch']==4800 else 1) and g['noise_added_after_complete_mean']
            assert 0<=g['per_example_clipped_count']<=j['batch']; clipped+=g['per_example_clipped_count']
            close(g['replace_one_sensitivity'],4/j['batch'],True)
            if j['method'].startswith('risk_'):
                assert 0<=c['private_risk']<=1 and -1e-6<=c['raw_risk']<=1+1e-6
                coeff.append(1+2*min(c['private_risk']/.5,1))
            else:
                assert c['private_risk'] is c['raw_risk'] is None; coeff.append(1.)
        agg=v['aggregation']; assert agg['eta']==(2. if t<60 else .5)
        safe=agg['message_safety']; assert safe['invalid_message_rows']==safe['nonfinite_risk_reports']==0
        expected=[a/sum(coeff) for a in coeff]; assert len(agg['objective_weights'])==10
        for a,b in zip(expected,agg['objective_weights']): close(a,b)
        close(agg['max_weight'],max(expected)); close(agg['concentration'],10*sum(x*x for x in expected))
        if j['method'].endswith('rfa'):
            s=agg['solver']; assert s['iterations']==40 and s['smoothing']==1e-5
            assert len(s['stationary_weights'])==10 and all(math.isfinite(x) and x>0 for x in s['stationary_weights'])
            close(sum(s['stationary_weights']),1.)
            assert all(math.isfinite(s[k]) and s[k]>=0 for k in ('unsmoothed_objective_gap_upper',
                'residual_norm','stationary_reconstruction_error','cluster_radius','minimizer_radius_bound'))
        else: assert agg['solver'] is None
        if v['validation'] is not None: verify(v['validation']); exact_counts(v['validation'],1200)
        clipping.append(clipped/(10*j['batch'])); concentration.append(agg['concentration'])
    verify(r['final']['test']); exact_counts(r['final']['test'])
    cp=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert cp['job']==j and cp['source_stamp']==manifest['source_stamp'] and cp['round']==120 and not cp['privacy_protected']
    assert cp['key_sha']==digest(OUT/'simulator_secret.json') and cp['rows']==r['rounds'] and cp['oracles']==raw['rounds']
    assert cp['initial']==initial and json.loads(json.dumps(cp['privacy']))==r['privacy']
    reconstruction=replay(j,data,manifest['profile'],cp,raw,r['privacy'],key)
    record=dict(job=j,endpoint_round=120,test_evaluated=True,test=r['final']['test'],validation=r['final']['validation'],
        privacy=r['privacy'],elapsed_seconds=r['elapsed_seconds'],files=files,replay=reconstruction,
        median_clipping_fraction=st.median(clipping),median_objective_concentration=st.median(concentration),
        source=str(folder/'metrics.json'))
    base.verify_stamp(stamp); base.save(cache,dict(signature=signature,record=record))
    print(f"V29 independent audit {name(j)}: ledger, 1200 draws, last query/noise/model/test PASS",flush=True)
    return record


def write(records,decision,stamp):
    target=DEST.with_name(DEST.name+('' if len(records)==24 else '_Partial'))
    payload=dict(audit_passed=True,valid_runs=len(records),expected_runs=24,records=records,decision=decision,
        gate_evaluated=decision is not None,source_stamp=stamp,manifest_sha256=digest(OUT/'manifest.json'),
        global_validation=False,attacks_evaluated=False,partial_results_never_select_candidate=True)
    base.save(target.with_suffix('.json'),payload)
    lines=['# V29 — confirmation indépendante : résultats audités','',f'**{len(records)}/24 runs MPS audités.**', '',
        'Résultat principal : test au tour 120. La candidate fixée est risque-RFA au batch complet. '
        'Les résultats partiels ne déclenchent aucune décision et les quatre nouvelles seeds sont '
        'des réplications sur le même benchmark, pas quatre datasets indépendants.', '',
        '| Seed | Batch | Méthode | Accuracy (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) | CE | Demi-Brier |',
        '|--:|--:|:--|--:|--:|--:|--:|--:|--:|']
    for r in records:
        j,v=r['job'],r['test']; values=[v[k] for k in (*KEYS,'ce_loss','brier_loss')]
        lines.append(f"| {j['seed']} | {j['batch']} | {j['method']} | "+' | '.join(f'{x:.6f}' for x in values)+' |')
    if decision is not None:
        lines+=['',f"Confirmation propre : **{'PASS' if decision['clean_confirmation_passed'] else 'FAIL'}**.", '',
            'Les bornes inférieures t sont à 98,75 % unilatéral (quatre différences appariées, df=3). '
            'Elles reposent sur les hypothèses du test t et leur puissance reste limitée. '
            'La réussite exige chaque comparaison et chaque critère, sans sélectionner la meilleure.', '',
            '| Témoin | Δ accuracy moyen | Borne inférieure | Δ Worst-20 moyen | Borne inférieure | Verdict |',
            '|:--|--:|--:|--:|--:|:--|']
        for c in decision['contrasts']:
            a,w=(c['summaries'][k] for k in KEYS[:2])
            lines.append(f"| B={c['control_batch']} {c['control_method']} | {a['mean']:+.6f} | {a['lower_one_sided_9875']:+.6f} | {w['mean']:+.6f} | {w['lower_one_sided_9875']:+.6f} | {'PASS' if c['passed'] else 'FAIL'} |")
        for c in decision['contrasts']:
            lines+=['',f"### Contre B={c['control_batch']} {c['control_method']}", '',
                '| Seed | Δ accuracy (pp) | Δ Worst-20 (pp) | Δ gap (pp) | Δ variance (pp²) |',
                '|--:|--:|--:|--:|--:|']
            for p in c['pairs']: lines.append(f"| {p['seed']} | "+' | '.join(f"{p['delta'][k]:+.6f}" for k in KEYS)+' |')
            lines+=['','Critères : '+', '.join(f"{k}: {'PASS' if v else 'FAIL'}" for k,v in c['gates'].items())+'.']
    lines+=['','## Portée de l’audit','',
        'Recalcul indépendant de la composition gaussienne ou fixed-WOR et des 120 préfixes ; '
        'vérification des comptages, tirages, coefficients et du modèle pré-tour ; dernière moyenne '
        'recalculée en blocs de 120 au lieu de 240 ; release à partir de la requête sauvegardée, '
        'pas et modèle final bit à bit, validation et test reproduits. Les primitives d’autodiff '
        'par exemple et le solveur numérique RFA restent partagés et sont explicitement testés ; '
        'cet audit n’est pas une seconde implémentation entièrement indépendante de PyTorch.', '',
        'Privacy au niveau exemple, replace-one, par exécution/client : les oracles et évaluations '
        'de recherche ne sont pas couverts. RFA n’est pas une preuve de robustesse du modèle. '
        'Même un PASS propre attend les attaques, et ne prouve ni nouveauté ni supériorité générale. '
        'Les témoins B=240 reproduisent la calibration générique historique, pas sa version affinée ; '
        'le nombre de gradients du batch complet est vingt fois supérieur.', '',
        '[Protocole préenregistré](Full_Population_Private_Risk_Confirmation_V29_Protocol.md) · '
        f'[Calculs et empreintes]({target.name}.json).']
    target.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(valid_runs=len(records),gate_evaluated=decision is not None,
        clean_confirmation_passed=None if decision is None else decision['clean_confirmation_passed'],global_validation=False)),flush=True)


def main(partial):
    require_mps(); CACHE.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((OUT/'manifest.json').read_text()); stamp=dict(manifest['source_stamp']); base.verify_stamp(stamp)
    assert manifest['device']=='mps' and not manifest['fallback']
    c=manifest['config']; assert c['seeds']==list(SEEDS) and c['full_methods']==list(METHODS)
    assert c['small_methods']==['erm_mean','erm_rfa'] and c['expected_runs']==24
    assert c['primary_batch']==4800 and c['primary_method']=='risk_rfa' and c['test_rounds']==[120]
    assert c['prospective_confirmation_wave']==1 and c['one_sided_alpha']==.0125
    assert c['minimum_worst20_advantage_pp']==c['maximum_accuracy_loss_pp']==1
    allowed={name(dict(seed=s,batch=b,method=m)) for s in SEEDS for b,m in ARMS}
    actual={p.name for p in OUT.glob('seed*') if p.is_dir()}
    assert actual<=allowed, 'Unexpected experimental arm in the confirmation directory'
    for p in (Path(__file__),TEST,OUT/'manifest.json'): stamp[str(p.relative_to(ROOT))]=digest(p)
    tests=json.loads((OUT/'tests.json').read_text()); assert tests['passed'] and tests['source_stamp']==manifest['source_stamp']
    path=CACHE/'tests.json'
    if not path.exists():
        proc=subprocess.run([sys.executable,'-m','pytest',str(TEST),'-q'],cwd=ROOT,capture_output=True,text=True)
        base.save(path,dict(passed=proc.returncode==0,source_stamp=stamp,output=proc.stdout+proc.stderr))
    audit_tests=json.loads(path.read_text()); assert audit_tests['passed'] and audit_tests['source_stamp']==stamp
    state=json.loads((OUT/'status.json').read_text())
    if not partial: assert state['status']=='completed' and state['valid_runs']==24
    key=json.loads((OUT/'simulator_secret.json').read_text())['key']; records=[]
    for seed in SEEDS:
        done=[]
        for b,m in ARMS:
            j=dict(seed=seed,batch=b,method=m); status=OUT/name(j)/'orchestration_status.json'
            if status.exists() and json.loads(status.read_text())['status']=='completed': done.append(j)
        if not done: continue
        data=base.prepare(manifest['profile'],seed); model=base.new_model(manifest['profile'],seed)
        initial=base.evaluate(model,data,'val'); del model; draw=patterns(seed,key)
        for j in done: records.append(audit_one(j,manifest,data,initial,draw,key,stamp))
        del data; torch.mps.empty_cache()
    decision=None
    if len(records)==24:
        # Completion must be authoritative even if --partial was used near a transition.
        final_state=json.loads((OUT/'status.json').read_text()); assert final_state['status']=='completed' and final_state['valid_runs']==24
        decision=independent_decision(records)
        evidence=json.loads((OUT/'evidence_unverified.json').read_text())
        assert evidence['source_stamp']==manifest['source_stamp'] and not evidence['independent_audit_passed']
        compare_runner(decision,evidence['decision'])
    elif not partial: raise AssertionError('Missing completed runs')
    base.verify_stamp(stamp); write(records,decision,stamp)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--partial',action='store_true')
    main(parser.parse_args().partial)
