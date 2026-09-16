"""Independent exact-count gates and final-step replay for V33.

Does not import the continuation runner or V32 winsor implementation. Shares
the established RFA primitive and data/model evaluation. Only the LAST step
is numerically replayed; earlier evaluations are count-audited, not replayed.
"""
from fractions import Fraction as Q
import argparse
import json
import math
import os
from pathlib import Path
import statistics as st
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_fair_objective_screen as base
from privacy.fair_objective import require_mps
from privacy.stable_weighted_rfa import weighted_rfa

OUT=ROOT/'results/ldp_gradient_far/private_risk_continuation_v33'
SOURCE=ROOT/'results/ldp_gradient_far/full_population_private_risk_calibration_v28'
DEST=ROOT/'output/analysis/Private_Risk_V33_Continuation_Audit'
SEEDS=(170501,170502)
CONDITIONS=('none','abrupt_bf','persistent_alie','slow_ipm')
METHODS=('risk_mean','risk_rfa','risk_winsor')

def exact(ev,ids):
    values=[]
    for i in ids:
        c=ev['clients'][i];ns,hs=c['class_count'],c['class_hits']
        assert c['N']==1200 and len(ns)==len(hs)==10 and sum(ns)==1200
        assert all(math.isfinite(x) and x==int(x) and x>=0 for x in ns+hs)
        assert all(h<=n for h,n in zip(hs,ns))
        values.append(Q(int(sum(hs)),12))
    mean=sum(values)/len(values);ordered=sorted(values);tail=math.ceil(len(values)/5)
    return dict(accuracy_pct=mean,worst20_pct=sum(ordered[:tail])/tail,
        gap_best20_worst20_pp=(sum(ordered[-tail:])-sum(ordered[:tail]))/tail,
        variance_pp2=sum((v-mean)**2 for v in values)/len(values))

def independently_decide(index,*,allow_incomplete=False):
    expected={(s,c,m) for s in SEEDS for c in CONDITIONS for m in METHODS}
    assert set(index)<=expected
    if not allow_incomplete:assert set(index)==expected
    comparisons=[]
    def get(s,c,m,k,ids):return exact(index[s,c,m]['rows'][k-1]['validation'],ids)
    for s in SEEDS:
        for k in (4,8,12):
            if (s,'none','risk_winsor') not in index or (s,'none','risk_mean') not in index:continue
            v=get(s,'none','risk_winsor',k,range(10));b=get(s,'none','risk_mean',k,range(10))
            da,dw=v['accuracy_pct']-b['accuracy_pct'],v['worst20_pct']-b['worst20_pct']
            comparisons.append(dict(seed=s,attack='none',step=k,control='risk_mean',accuracy_delta_pp=float(da),
                worst20_delta_pp=float(dw),passed=da>=Q(-1,10) and dw>=Q(-1,4)))
        for c in CONDITIONS[1:]:
            for k in (8,12):
                if (s,c,'risk_winsor') not in index:continue
                v=get(s,c,'risk_winsor',k,range(2,10))
                for a,m in (('none','risk_winsor'),(c,'risk_rfa')):
                    if (s,a,m) not in index:continue
                    b=get(s,a,m,k,range(2,10));da,dw=v['accuracy_pct']-b['accuracy_pct'],v['worst20_pct']-b['worst20_pct']
                    comparisons.append(dict(seed=s,attack=c,step=k,control=f'{a}/{m}',accuracy_delta_pp=float(da),
                        worst20_delta_pp=float(dw),passed=da>=-1 and dw>=-1))
    return comparisons

def main(*,partial_failure=False):
    require_mps()
    state=json.loads((OUT/'status.json').read_text())
    if partial_failure:
        assert state['status']=='failed'
    elif state['status']!='completed' or state.get('valid_runs')!=24:
        raise RuntimeError('Audit requires 24 completed trajectories; active jobs must not be restarted')
    manifest=json.loads((OUT/'manifest.json').read_text());stamp=manifest['stamp'];base.verify_stamp(stamp)
    parent=json.loads((SOURCE/'manifest.json').read_text())
    hashes={str(Path(__file__).relative_to(ROOT)):base.digest(Path(__file__))}
    paths=[OUT/'manifest.json',OUT/'tests.json',OUT/'status.json',ROOT/'tests/test_private_risk_continuation_v33_audit.py']
    if partial_failure:paths.extend((OUT/'failure.json',ROOT/'output/analysis/Private_Risk_V33_Stop_Diagnosis.json'))
    else:paths.append(OUT/'decision.json')
    for p in paths:
        hashes[str(p.relative_to(ROOT))]=base.digest(p)
    tests=json.loads((OUT/'tests.json').read_text());assert tests['passed'] and tests['source_stamp']==stamp
    plan=manifest['privacy_parent'];index={};replays=[];table=[];missing=[]
    if partial_failure:
        diagnosis=json.loads((ROOT/'output/analysis/Private_Risk_V33_Stop_Diagnosis.json').read_text())
        assert diagnosis['reproduced'] and diagnosis['no_update_applied'] and diagnosis['V33_modified'] is False
        for p,h in diagnosis['inputs'].items():assert base.digest(ROOT/p)==h
    for seed in SEEDS:
        data=base.prepare(parent['profile'],seed);model=base.new_model(parent['profile'],seed)
        for condition in CONDITIONS:
            for method in METHODS:
                folder=OUT/f'seed{seed}__{condition}__{method}'
                if partial_failure and not (folder/'metrics.json').exists():
                    assert seed==170502 and condition=='slow_ipm' and method in ('risk_rfa','risk_winsor')
                    entry=dict(seed=seed,attack=condition,method=method,completed_steps=0,status='not_started')
                    if method=='risk_rfa':
                        stopped=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
                        assert [r['step'] for r in stopped['rows']]==[1,2,3,4]
                        assert stopped['signature']['source_stamp']==stamp
                        entry.update(completed_steps=4,status='interrupted_before_applying_step5')
                        for fn in ('checkpoint.pt','orchestration_status.json'):
                            p=folder/fn;hashes[str(p.relative_to(ROOT))]=base.digest(p)
                    else:assert not folder.exists()
                    missing.append(entry);continue
                rec=json.loads((folder/'metrics.json').read_text());status=json.loads((folder/'orchestration_status.json').read_text())
                assert status['status']=='completed' and status['device']==rec['device']=='mps'
                assert status['metrics_sha256']==base.digest(folder/'metrics.json')
                assert status['checkpoint_sha256']==base.digest(folder/'checkpoint.pt')
                assert rec['signature']==dict(seed=seed,attack=condition,method=method,source_stamp=stamp)
                assert rec['test_evaluated'] is False and rec['global_validation'] is False
                assert rec['privacy']==plan and [r['step'] for r in rec['rows']]==list(range(1,13))
                cp=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
                assert cp['signature']==rec['signature'] and cp['rows']==rec['rows'] and cp['initial']==rec['initial']
                for fname in ('metrics.json','orchestration_status.json','checkpoint.pt'):
                    p=folder/fname;hashes[str(p.relative_to(ROOT))]=base.digest(p)
                for row in rec['rows']:
                    k=row['step'];e,a=min(((120+k)*j/2*(plan['gradient_z']**-2+plan['risk_z']**-2)
                        +math.log(1e5)/(j-1),j) for j in range(2,65))
                    assert math.isclose(e,row['epsilon_total'],rel_tol=1e-12) and a==row['epsilon_order']
                    assert row['device']=='mps' and row['oracle']['feeds_mechanism'] is False
                    active=condition!='none' and 5<=k<=(12 if condition=='persistent_alie' else 8)
                    assert row['attack']['active']==active
                    if active:
                        expected=10. if condition=='abrupt_bf' else 1.5 if condition=='persistent_alie' else .5*(k-4)
                        assert row['attack']['multiplier']==expected
                    if row['validation'] is not None:
                        assert k in (4,8,12)
                        for key,value in exact(row['validation'],range(10)).items():
                            assert math.isclose(float(value),row['validation'][key],rel_tol=1e-9,abs_tol=1e-8)
                        hs=exact(row['validation'],range(2,10))
                        table.append(dict(seed=seed,attack=condition,method=method,step=k,
                            honest={key:float(value) for key,value in hs.items()},
                            all_clients={key:float(value) for key,value in exact(row['validation'],range(10)).items()}))
                    else:assert k not in (4,8,12)
                    if method=='risk_winsor':
                        d=row['aggregation'];assert d['clipped_count']<=2 and d['center_norm']<=2+1e-5
                        assert d['pilot_solver']['unsmoothed_objective_gap_upper']<=.001
                    if method=='risk_rfa':assert row['aggregation']['solver']['unsmoothed_objective_gap_upper']<=.001
                assert math.isclose(e,rec['epsilon_total_132'],rel_tol=1e-12) and e>4
                # Independent final-step attacks: BF/IPM have recovered at step12.
                x=cp['last_private_messages'].to('mps');r=cp['last_reports'].to('mps')
                if condition=='persistent_alie':
                    mu=x[2:].mean(0);sd=((x[2:]-mu).square().mean(0)).sqrt()
                    x=x.clone();r=r.clone();x[:2]=mu+1.5*sd;r[:2]=1.
                coeff=1+4*r.clamp(0,.5);lam=coeff/coeff.sum()
                if method=='risk_mean':A=(lam[:,None]*x).sum(0)
                elif method=='risk_rfa':A,_=weighted_rfa(x,lam)
                else:
                    p,_=weighted_rfa(x,torch.full((10,),.1,device='mps'))
                    c=p*(2/torch.linalg.vector_norm(p).clamp_min(1e-30)).clamp(max=1)
                    norms=torch.linalg.vector_norm(x-c,dim=1)
                    radius=max(4.,float(torch.sort(norms).values[7]))
                    z=c+(x-c)*(radius/norms.clamp_min(1e-30)).clamp(max=1)[:,None]
                    A=(lam[:,None]*z).sum(0)
                    assert math.isclose(radius,rec['rows'][-1]['aggregation']['radius'],rel_tol=1e-6)
                torch.testing.assert_close(.5*A,cp['last_step'].to('mps'),rtol=2e-5,atol=2e-6)
                model.load_state_dict(cp['pre_round_model']);base.apply_gradient(model,A,.5)
                for key,value in model.state_dict().items():
                    torch.testing.assert_close(value,cp['model'][key].to('mps'),rtol=2e-5,atol=2e-6)
                val=base.evaluate(model,data,'val');old=rec['rows'][-1]['validation']
                for new,ref in zip(val['clients'],old['clients']):
                    assert new['class_hits']==ref['class_hits'] and new['class_count']==ref['class_count']
                for key in ('ce_loss','brier_loss','balanced_accuracy_pct'):
                    assert math.isclose(val[key],old[key],rel_tol=2e-6,abs_tol=2e-6)
                index[seed,condition,method]=rec
                replays.append(dict(seed=seed,attack=condition,method=method,step=12,exact_validation_counts=True))
                print(f'V33 final-step audit {len(replays)}/{22 if partial_failure else 24}',flush=True)
        del model,data;torch.mps.empty_cache()
    # Same pre-attack model and messages within each rule, same permutations across ALL branches.
    for s in SEEDS:
        for m in METHODS:
            clean=index[s,'none',m]
            for c in CONDITIONS:
                if (s,c,m) not in index:continue
                for k in range(4):
                    other=index[s,c,m]['rows'][k]
                    assert other['model_hash']==clean['rows'][k]['model_hash']
                    assert other['clients']==clean['rows'][k]['clients']
            for c in CONDITIONS:
                if (s,c,m) not in index:continue
                for k in range(12):
                    expected=index[s,'none','risk_mean']['rows'][k]['clients']
                    for v,w in zip(index[s,c,m]['rows'][k]['clients'],expected):
                        assert v['permutation_hash']==w['permutation_hash']
    comparisons=independently_decide(index,allow_incomplete=partial_failure)
    if partial_failure:
        assert len(replays)==22 and len(missing)==2 and len(comparisons)==26
        # Incomplete data cannot certify PASS; a known failed necessary criterion can certify rejection.
        gate=False if any(not c['passed'] for c in comparisons) else None
    else:
        runner=json.loads((OUT/'decision.json').read_text())
        assert comparisons==runner['comparisons']
        gate=all(c['passed'] for c in comparisons);assert gate==runner['local_gate_passed']
    base.verify_stamp(stamp)
    for p,h in hashes.items():assert base.digest(ROOT/p)==h
    result=dict(audit_passed=True,local_gate_passed=gate,global_validation=False,device='mps',
        source_stamp=stamp,inputs=hashes,comparisons=comparisons,table=table,final_step_replays=replays,
        earlier_steps_numerically_replayed=False,epsilon_total_132=manifest['epsilon_total_132'],
        independent_seeds=False,automatic_promotion=False,test_evaluated=False,
        audit_scope='completed trajectories only' if partial_failure else '24 completed trajectories',
        campaign_complete=not partial_failure,missing_trajectories=missing,available_comparisons=len(comparisons),required_comparisons=30)
    dest=DEST.with_name(DEST.name+'_Partial_Failure') if partial_failure else DEST
    base.save(dest.with_suffix('.json'),result)
    lines=['# V33 — accumulation sur 12 pas supplémentaires','',
        '**Calibration sur deux seeds déjà connues ; pas de confirmation finale.**', '',
        f"{len(replays)}/24 continuations MPS auditées ; critère local : **{'PASS' if gate is True else 'FAIL' if gate is False else 'non évaluable'}**.", '',
        f"Budget total parent + continuation : epsilon={manifest['epsilon_total_132']:.6f}, delta=10⁻⁵ par alternative. "
        'Les branches appariées ne sont pas conjointement privées à ce budget.', '',
        'Les contrôles comptent les mêmes huit clients honnêtes dans toutes les conditions. '
        'BF/IPM : attaque aux pas5–8, récupération9–12 ; ALIE : attaque5–12. '
        'La moyenne ± écart-type ci-dessous est descriptive ; n indique le nombre de seeds disponibles. '
        'Un résultat n=1 n’a pas d’écart-type inter-seeds.', '',
        '| Condition | Règle | Pas | n seeds | Acc honnête (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) |',
        '|:--|:--|--:|--:|--:|--:|--:|--:|']
    for condition in CONDITIONS:
        for method in METHODS:
            for k in (4,8,12):
                values=[v['honest'] for v in table if v['attack']==condition and v['method']==method and v['step']==k]
                cells=[]
                for field in ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2'):
                    nums=[v[field] for v in values]
                    cells.append(f'{st.mean(nums):.3f} ± {st.stdev(nums):.3f}' if len(nums)>1 else f'{nums[0]:.3f}')
                lines.append(f'| {condition} | {method} | {k} | {len(values)} | '+ ' | '.join(cells)+' |')
    if partial_failure:
        lines+=['','## Manquants et arrêt conservé','',
            'Seed170502, IPM : RFA interrompue avant l’application du pas5 (4/12 pas sauvegardés) ; '
            'V32 non lancée. Aucun résultat n’est imputé. Seulement26 des30 comparaisons prévues '
            'sont évaluables. L’échec d’un critère nécessaire suffit au rejet ; la campagne '
            'reste incomplète. Le statut original failed n’est pas modifié.', '',
            '[Diagnostic de l’arrêt numérique](Private_Risk_V33_Stop_Diagnosis.md)']
    lines+=['','## Critères appariés préenregistrés','',
        '| Seed | Condition | Pas | Contrôle | Δ Acc (pp) | Δ Worst-20 (pp) | Verdict |',
        '|--:|:--|--:|:--|--:|--:|:--|']
    for c in comparisons:
        lines.append(f"| {c['seed']} | {c['attack']} | {c['step']} | {c['control']} | {c['accuracy_delta_pp']:+.4f} | {c['worst20_delta_pp']:+.4f} | {'PASS' if c['passed'] else 'FAIL'} |")
    lines+=['','## Portée de l’audit','',
        'Empreintes, statuts, budgets, comptages de validation et appariement avant attaque vérifiés. '
        'Dernière mise à jour rejouée sur MPS par une seconde écriture des attaques et du clipping. '
        'RFA et les primitives modèle/évaluation sont partagées. Les onze pas antérieurs ne sont pas '
        'numériquement rejoués. Aucun test final ni nouvelle seed de confirmation consommés. '
        'V29 reste FAIL et V30 fermé ; un éventuel PASS local ne lance rien.', '',
        f'[Protocole](Private_Risk_V33_Continuation_Protocol.md) · [Audit détaillé]({dest.name}.json)']
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(audit_passed=True,local_gate_passed=gate,campaign_complete=not partial_failure,global_validation=False)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--partial-failure',action='store_true')
    main(partial_failure=parser.parse_args().partial_failure)
