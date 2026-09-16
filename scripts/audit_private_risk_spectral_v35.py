"""V35 exact counts/gates; independent Gram and power bound, 32 model replays."""
import fcntl
from fractions import Fraction as Q
import itertools
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_fair_objective_screen as base
from scripts.audit_private_risk_continuation_v33 import exact
from privacy.fair_objective import require_mps

SOURCE=ROOT/'results/ldp_gradient_far/private_risk_spectral_v35'
OUT=ROOT/'results/ldp_gradient_far/private_risk_v35_independent_audit'
DEST=ROOT/'output/analysis/Private_Risk_V35_Independent_Audit'


def power_bounds(gram):
    """1024th Schatten upper bound and Rayleigh lower, all on MPS.

    Scaling at every squaring avoids overflow. Independent of Jacobi rotations.
    Float32 diagnostic bounds only, no interval rounding claim.
    """
    require_mps()
    scale=gram.abs().amax(dim=(1,2)).clamp_min(1e-30)
    b=gram/scale[:,None,None];logscale=scale.log()
    for _ in range(10):
        b=b.bmm(b)
        s=b.abs().amax(dim=(1,2)).clamp_min(1e-30)
        b=b/s[:,None,None];logscale=2*logscale+s.log()
    upper=((logscale+b.diagonal(dim1=1,dim2=2).sum(1).clamp_min(1e-30).log())/1024).exp()
    col=b.square().sum(1).argmax(1)
    v=b[torch.arange(len(b),device='mps'),:,col]
    denom=v.square().sum(1).clamp_min(1e-30)
    lower=(v*gram.bmm(v[:,:,None]).squeeze(2)).sum(1)/denom
    zero=gram.abs().amax(dim=(1,2))==0
    return torch.where(zero,0.,lower),torch.where(zero,0.,upper)


def main():
    require_mps();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'audit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=json.loads((SOURCE/'results.json').read_text());status=json.loads((SOURCE/'status.json').read_text())
        assert status['status']=='completed' and status['evaluated']==128
        assert status['results_sha256']==base.digest(SOURCE/'results.json')
        stamp=dict(result['signature']['source_stamp']);base.verify_stamp(stamp)
        for p in (Path(__file__),SOURCE/'results.json',SOURCE/'status.json',ROOT/'tests/test_private_risk_v35_audit.py'):
            stamp[str(p.relative_to(ROOT))]=base.digest(p)
        ix={}
        for row in result['rows']:
            key=(row['seed'],row['state'],row['attack'],row['method'],row['eta'])
            assert key not in ix and row['evaluated'] and row['device']=='mps';ix[key]=row
            for k,v in exact(row['evaluation'],range(2,10)).items():
                assert math.isclose(float(v),row['honest'][k],rel_tol=1e-10,abs_tol=1e-10)
        assert len(ix)==128
        # Recompute each frozen comparison from integer counts, not reported deltas.
        for c in result['decision']['comparisons']:
            key=(c['seed'],c['state'],c['attack'],'risk_smea',c['eta'])
            if c['control']=='risk_mean':
                bk=(c['seed'],c['state'],'none','risk_mean',c['eta']);ids=range(10);limits=(Q(-1,10),Q(-1,4))
            elif c['control']=='same_state/no_current_attack':
                bk=(c['seed'],c['state'],'none','risk_smea',c['eta']);ids=range(2,10);limits=(-1,-1)
            else:
                assert c['control']=='attacked_RFA'
                bk=(c['seed'],c['state'],c['attack'],'risk_rfa',c['eta']);ids=range(2,10);limits=(-1,-1)
            v,b=exact(ix[key]['evaluation'],ids),exact(ix[bk]['evaluation'],ids)
            da,dw=v['accuracy_pct']-b['accuracy_pct'],v['worst20_pct']-b['worst20_pct']
            assert float(da)==c['accuracy_delta_pp'] and float(dw)==c['worst20_delta_pp']
            assert c['passed']==(da>=limits[0] and dw>=limits[1])
        assert len(result['decision']['comparisons'])==52
        gate=all(c['passed'] for c in result['decision']['comparisons'])
        assert gate==result['decision']['local_gate_passed']
        parent=ROOT/'results/ldp_gradient_far/full_population_private_risk_calibration_v28'
        v33=ROOT/'results/ldp_gradient_far/private_risk_continuation_v33'
        profile=json.loads((parent/'manifest.json').read_text())['profile']
        subsets=list(itertools.combinations(range(10),8));subix=torch.tensor(subsets,device='mps')
        checks=[];replays=0
        for seed in (170501,170502):
            data=base.prepare(profile,seed);model=base.new_model(profile,seed)
            for state in ('parent_clean','alie_pre12'):
                path=(parent/f'seed{seed}__risk_rfa/checkpoint.pt' if state=='parent_clean'
                      else v33/f'seed{seed}__persistent_alie__risk_winsor/checkpoint.pt')
                cp=torch.load(path,map_location='cpu',weights_only=True)
                for attack in ('none','abrupt_bf','persistent_alie','slow_ipm'):
                    x=cp['last_private_messages'].to('mps');r=cp['last_reports'].to('mps')
                    if attack!='none':
                        mean=x[2:].mean(0)
                        if attack=='abrupt_bf':poison=-10*mean
                        elif attack=='slow_ipm':poison=-2*mean
                        else:poison=mean+1.5*((x[2:]-mean).square().mean(0)).sqrt()
                        x=x.clone();r=r.clone();x[:2]=poison;r[:2]=1
                    a=1+4*r.clamp(0,.5);sel=a[subix];p=sel/sel.sum(1,keepdim=True)
                    # Direct 45*k*d residual matrices, unlike candidate's n*n Gram identity.
                    vs=x[subix];mus=(p[:,:,None]*vs).sum(1)
                    b=(vs-mus[:,None,:])*p.sqrt()[:,:,None]
                    gram=b.bmm(b.transpose(1,2));lo,up=power_bounds(gram)
                    record=ix[seed,state,attack,'risk_smea',.125]
                    ids=record['diagnostics']['selected_ids'];j=subsets.index(tuple(ids))
                    spectral_gap=max(0.,float(up[j]-lo.min()))
                    # Independent power enclosure is less tight than the Jacobi one.
                    assert spectral_gap<=.01*max(1.,float(up[j]))
                    saved=record['diagnostics']['selected_eigenvalue_upper']
                    assert float(lo[j])-2e-4<=saved<=float(up[j])+2e-4
                    A=mus[j]
                    original=record['diagnostics']['effective_weights']
                    expected=torch.zeros(10,device='mps');expected[subix[j]]=p[j]
                    torch.testing.assert_close(expected,torch.tensor(original,device='mps'),rtol=2e-5,atol=2e-6)
                    for eta in (.125,.5):
                        model.load_state_dict(cp['pre_round_model']);base.apply_gradient(model,A,eta)
                        ev=base.evaluate(model,data,'val');ref=ix[seed,state,attack,'risk_smea',eta]['evaluation']
                        for new,old in zip(ev['clients'],ref['clients']):
                            assert new['class_count']==old['class_count'] and new['class_hits']==old['class_hits']
                        for key in ('ce_loss','brier_loss'):
                            assert math.isclose(ev[key],ref[key],rel_tol=2e-6,abs_tol=2e-6)
                        replays+=1
                    checks.append(dict(seed=seed,state=state,attack=attack,selected_ids=ids,
                        discarded_ids=[i for i in range(10) if i not in ids],
                        independent_selection_gap_upper=spectral_gap,power_lower=float(lo[j]),power_upper=float(up[j]),
                        jacobi_selected_upper=saved,exact_prediction_counts=True))
                    print(f'V35 independent subsets {len(checks)}/16; model replays {replays}/32',flush=True)
                del cp
            del data,model;torch.mps.empty_cache()
        base.verify_stamp(stamp)
        evidence=dict(audit_passed=True,source_stamp=stamp,device='mps',counts_audited=128,
            comparisons_audited=52,candidate_model_replays=replays,baseline_models_replayed=False,
            spectral_cases=checks,local_gate_passed=gate,global_validation=False)
        base.save(OUT/'results.json',evidence);base.save(DEST.with_suffix('.json'),evidence)
        lines=['# V35 — audit indépendant des comptes et de la sélection spectrale','',
            '**128 comptages, 52 comparaisons, 16 sélections et 32 pas du candidat vérifiés sur MPS.**','',
            f"Critère local : {'PASS' if gate else 'FAIL'}. Les baselines sont auditées par comptages, pas rejouées par ce second auditeur.", '',
            'Le candidat utilise Jacobi sur un Gram obtenu par centrage algébrique. Cet auditeur reconstruit '
            'directement les résidus pondérés puis utilise Rayleigh et une puissance 1024 pour encadrer '
            'la valeur propre. Ce sont deux calculs float32, pas une preuve par arrondi dirigé.', '',
            '| Seed | État | Attaque courante | Identités supprimées | Borne indépendante de sous-optimalité spectrale |',
            '|--:|:--|:--|:--|--:|']
        for c in checks:
            lines.append(f"| {c['seed']} | {c['state']} | {c['attack']} | {c['discarded_ids']} | {c['independent_selection_gap_upper']:.7f} |")
        lines+=['','Les identités 0 et 1 sont malveillantes uniquement dans les interventions actives ; '
            'none utilise les dix messages privés originaux. Leur exclusion dans ce cas ne signifie '
            'pas une détection de Byzantine. Modèle et évaluateur sont partagés avec le premier calcul.', '',
            '[Evidence et empreintes](Private_Risk_V35_Independent_Audit.json) · '
            '[Écran complet](Private_Risk_V35_Fixed_State_Screen.md)']
        DEST.with_suffix('.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps(dict(audit_passed=True,local_gate_passed=gate,candidate_replays=replays)))


if __name__=='__main__':main()
