"""128 frozen local model probes of risk-weighted spectral subset means."""
import fcntl
from fractions import Fraction as Q
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_fair_objective_screen as base
from scripts.analyze_full_population_error_attribution_v28b import source_evidence
from scripts.audit_private_risk_continuation_v33 import exact
from scripts.diagnose_private_risk_alie_model_v34 import summary, validation_gradient, aggregate as controls
from privacy.fair_objective import require_mps
from privacy.capped_private_risk import weights
from privacy.private_risk_spectral_v35 import risk_smea, spectral_intervals

PARENT=ROOT/'results/ldp_gradient_far/full_population_private_risk_calibration_v28'
V33=ROOT/'results/ldp_gradient_far/private_risk_continuation_v33'
OUT=ROOT/'results/ldp_gradient_far/private_risk_spectral_v35'
DEST=ROOT/'output/analysis/Private_Risk_V35_Fixed_State_Screen'
DOC=ROOT/'output/analysis/Private_Risk_V35_Fixed_State_Protocol.md'
SEEDS=(170501,170502);STATES=('parent_clean','alie_pre12')
ATTACKS=('none','abrupt_bf','persistent_alie','slow_ipm')
METHODS=('risk_mean','risk_rfa','risk_winsor','risk_smea')
ETAS=(.125,.5)


def decide(rows):
    ix={(r['seed'],r['state'],r['attack'],r['method'],r['eta']):r for r in rows}
    assert len(rows)==len(ix)==128
    checks=[]
    for s in SEEDS:
        for eta in ETAS:
            v=ix[s,'parent_clean','none','risk_smea',eta]
            b=ix[s,'parent_clean','none','risk_mean',eta]
            requests=[('parent_clean','none','risk_mean',v,b,range(10),Q(-1,10),Q(-1,4))]
            for state in STATES:
                for a in ATTACKS[1:]:
                    v=ix[s,state,a,'risk_smea',eta]
                    requests.extend((state,a,control,v,b,range(2,10),Q(-1),Q(-1)) for control,b in
                        (('same_state/no_current_attack',ix[s,state,'none','risk_smea',eta]),
                         ('attacked_RFA',ix[s,state,a,'risk_rfa',eta])))
            for state,a,c,v,b,ids,ma,mw in requests:
                d=dict(seed=s,state=state,attack=a,eta=eta,control=c)
                if not v['evaluated'] or not b['evaluated']:
                    d.update(passed=False,missing=True)
                else:
                    va,ba=exact(v['evaluation'],ids),exact(b['evaluation'],ids)
                    da,dw=va['accuracy_pct']-ba['accuracy_pct'],va['worst20_pct']-ba['worst20_pct']
                    d.update(passed=da>=ma and dw>=mw,accuracy_delta_pp=float(da),worst20_delta_pp=float(dw))
                checks.append(d)
    assert len(checks)==52
    return dict(local_gate_passed=all(c['passed'] for c in checks) and all(r['evaluated'] for r in rows),
                comparisons=checks,global_validation=False,automatic_promotion=False)


def main():
    require_mps();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'screen.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        parent=json.loads((PARENT/'manifest.json').read_text())
        sourceaudit=ROOT/'output/analysis/Private_Risk_V34_Direction_Step_Audit.json'
        audit=json.loads(sourceaudit.read_text());assert audit['passed']
        stamp=dict(audit['source_stamp'])
        for s in SEEDS:stamp.update(source_evidence(dict(seed=s,method='risk_rfa')))
        for p in (Path(__file__),DOC,sourceaudit,ROOT/'privacy/private_risk_spectral_v35.py',
                  ROOT/'tests/test_private_risk_spectral_v35.py',ROOT/'output/analysis/Private_Risk_V35_Weighted_Spectral_Construction.md'):
            stamp[str(p.relative_to(ROOT))]=base.digest(p)
        base.verify_stamp(stamp)
        sig=dict(source_stamp=stamp,seeds=list(SEEDS),states=list(STATES),attacks=list(ATTACKS),
                 methods=list(METHODS),etas=list(ETAS),expected=128,device='mps')
        if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==sig
        else:base.save(OUT/'manifest.json',sig)
        rows=[]
        for seed in SEEDS:
            data=base.prepare(parent['profile'],seed);model=base.new_model(parent['profile'],seed)
            for state in STATES:
                cache=OUT/f'seed{seed}__{state}.json'
                if cache.exists():
                    old=json.loads(cache.read_text());assert old['signature']==sig
                    rows.extend(old['rows']);continue
                path=(PARENT/f'seed{seed}__risk_rfa/checkpoint.pt' if state=='parent_clean'
                      else V33/f'seed{seed}__persistent_alie__risk_winsor/checkpoint.pt')
                cp=torch.load(path,map_location='cpu',weights_only=True)
                x0,r0=cp['last_private_messages'].to('mps'),cp['last_reports'].to('mps')
                mu=cp['last_clean_means'].to('mps')
                model.load_state_dict(cp['pre_round_model']);before=base.evaluate(model,data,'val')
                jgrad=validation_gradient(model,data,before)
                batch=[]
                for attack in ATTACKS:
                    x,r=x0.clone(),r0.clone()
                    if attack!='none':
                        center=x0[2:].mean(0)
                        if attack=='abrupt_bf':x[:2]=-10*center
                        elif attack=='persistent_alie':x[:2]=center+1.5*x0[2:].std(0,correction=0)
                        else:x[:2]=-2*center
                        r[:2]=1.
                    assert torch.equal(x[2:],x0[2:]) and torch.equal(r[2:],r0[2:])
                    lam,_=weights(r,.5);hp=lam[2:]/lam[2:].sum()
                    target=(hp[:,None]*mu[2:]).sum(0)
                    noisymean=(hp[:,None]*x[2:]).sum(0)
                    hb=hp.sqrt()[:,None]*(x[2:]-noisymean)
                    _,hupper=spectral_intervals((hb@hb.T)[None])
                    for method in METHODS:
                        a=None;di={};missing=None
                        try:
                            if method=='risk_smea':a,di=risk_smea(x,r,f_budget=2)
                            else:
                                a,di=controls(x,r,method)
                                if di['solver_gap']>.001:missing='original_RFA_certificate_exceeds_0.001'
                        except (RuntimeError,ValueError,FloatingPointError) as exc:
                            missing=repr(exc)
                        oracle=None
                        if a is not None and missing is None:
                            assert bool(torch.isfinite(a).all())
                            oracle=dict(applied_error_squared=float((a-target).square().sum()),
                                oracle_alignment=float((jgrad*a).sum()),honest_covariance_upper=float(hupper),
                                feeds_mechanism=False)
                            if method=='risk_smea':
                                eff=torch.tensor(di['effective_weights'],device='mps')
                                oracle.update(reserved_pair_effective_mass=float(eff[:2].sum()),
                                    removed_honest_ids=[i for i in di['removed_ids'] if i>=2],
                                    removed_honest_nominal_mass=sum(float(lam[i]) for i in di['removed_ids'] if i>=2),
                                    noisy_honest_mean_error_squared=float((a-noisymean).square().sum()),
                                    loose_spectral_error_upper=4*(float(hupper)+di['selection_gap_upper']))
                                assert oracle['noisy_honest_mean_error_squared']<=oracle['loose_spectral_error_upper']+2e-4
                        for eta in ETAS:
                            record=dict(seed=seed,state=state,attack=attack,method=method,eta=eta,
                                evaluated=missing is None,diagnostics=di,device='mps',oracle=oracle)
                            if missing is not None:record['missing_reason']=missing
                            else:
                                model.load_state_dict(cp['pre_round_model']);base.apply_gradient(model,a,eta)
                                ev=base.evaluate(model,data,'val');record['evaluation']=ev;record['honest']=summary(ev)
                            rows.append(record);batch.append(record)
                            desc=missing if missing else f"acc={record['honest']['accuracy_pct']:.3f} W={record['honest']['worst20_pct']:.3f}"
                            print(f'V35 {len(rows)}/128 {seed} {state} {attack} {method} eta={eta} {desc}',flush=True)
                            base.save(OUT/'status.json',dict(status='running',pid=os.getpid(),device='mps',
                                attempted=len(rows),expected=128,active_state=state,updated_unix=time.time()))
                assert len(batch)==32
                base.verify_stamp(stamp)
                base.save(cache,dict(signature=sig,rows=batch,before=before))
                del cp,jgrad;torch.mps.empty_cache()
            del data,model;torch.mps.empty_cache()
        base.verify_stamp(stamp);decision=decide(rows)
        result=dict(signature=sig,rows=rows,decision=decision,device='mps',global_validation=False,
            test_evaluated=False,research_exports_private=False,new_private_gradient_draws=0)
        base.save(OUT/'results.json',result);base.save(DEST.with_suffix('.json'),result)
        base.save(OUT/'status.json',dict(status='completed',device='mps',pid=os.getpid(),attempted=128,
            evaluated=sum(r['evaluated'] for r in rows),local_gate_passed=decision['local_gate_passed'],
            results_sha256=base.digest(OUT/'results.json')))
        report(result)


def report(r):
    lines=['# V35 — écran spectral pondéré sur gradients privés réels','',
        f"**128 points tentés ; {sum(v['evaluated'] for v in r['rows'])} évalués. Critère local : {'PASS' if r['decision']['local_gate_passed'] else 'FAIL'}.**",'',
        'Un seul pas virtuel par point ; deux seeds de calibration. parent_clean : historique sans attaque '
        'avant le tour 120 ; alie_pre12 : historique déjà affecté par ALIE, avant le pas 12. '
        'Toutes les règles partagent modèles/messages/rapports au sein de chaque intervention. '
        'Les valeurs ci-dessous portent sur les mêmes huit identités honnêtes, même pour le contrefactuel none.', '',
        '| Seed | État | Attaque courante | Règle | Eta | Acc H (%) | Worst-20 H (%) | Gap H (pp) | J_val |',
        '|--:|:--|:--|:--|--:|--:|--:|--:|--:|']
    for v in r['rows']:
        prefix=f"| {v['seed']} | {v['state']} | {v['attack']} | {v['method']} | {v['eta']} | "
        lines.append(prefix+(' | '.join(f"{v['honest'][k]:.6f}" for k in
            ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','J_val')) if v['evaluated'] else 'non évalué | — | — | —')+' |')
    lines+=['','## Critères non satisfaits','',
        '| Seed | État | Attaque | Eta | Contrôle | Δ Acc (pp) | Δ Worst-20 (pp) |',
        '|--:|:--|:--|--:|:--|--:|--:|']
    for c in r['decision']['comparisons']:
        if not c['passed']:
            lines.append(f"| {c['seed']} | {c['state']} | {c['attack']} | {c['eta']} | {c['control']} | "
                +(f"{c['accuracy_delta_pp']:+.6f} | {c['worst20_delta_pp']:+.6f}" if not c.get('missing') else 'manquant | manquant')+' |')
    lines+=['','La sélection utilise uniquement les messages et rapports privés. Les identités honnêtes, '
        'erreurs et gradient de validation sont des oracles hors mécanisme. Aucune revendication DP '
        'pour cet export de recherche. Les coûts sans attaque du gate sont calculés sur dix clients '
        'au modèle parent, contrairement au tableau descriptif à huit clients.', '',
        '[Protocole gelé](Private_Risk_V35_Fixed_State_Protocol.md) · '
        '[Construction et preuves](Private_Risk_V35_Weighted_Spectral_Construction.md) · '
        '[Résultats complets](Private_Risk_V35_Fixed_State_Screen.json)']
    DEST.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
