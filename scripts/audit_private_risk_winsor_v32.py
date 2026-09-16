"""MPS replay with independent attacks, winsor formula and exact-count gate.

Shares only model/data loading, evaluated model metrics, and the audited RFA
primitive. Does not import the V32 candidate or screen/gate implementation.
"""
from fractions import Fraction as Q
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_fair_objective_screen as base
from privacy.fair_objective import require_mps
from privacy.stable_weighted_rfa import weighted_rfa

SOURCE=ROOT/'results/ldp_gradient_far/full_population_private_risk_calibration_v28'
SCREEN=ROOT/'results/ldp_gradient_far/private_risk_winsor_screen_v32'
REPORT=ROOT/'output/analysis/Private_Risk_V32_Fixed_State_Screen.json'
DEST=ROOT/'output/analysis/Private_Risk_V32_Independent_Replay.json'


def exact(ev,ids):
    scores=[]
    for i in ids:
        c=ev['clients'][i];ns,hs=c['class_count'],c['class_hits']
        assert c['N']==1200 and len(ns)==len(hs)==10 and sum(ns)==1200
        assert all(math.isfinite(x) and x==int(x) and x>=0 for x in ns+hs)
        assert all(h<=n for h,n in zip(hs,ns))
        scores.append(Q(int(sum(hs)),12))
    return sum(scores)/len(scores),sum(sorted(scores)[:2])/2


def main():
    require_mps()
    evidence=json.loads(REPORT.read_text());base.verify_stamp(evidence['source_stamp'])
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    assert not evidence['global_validation'] and not evidence['training'] and not evidence['test_evaluated']
    index={(r['seed'],r['attack'],r['method']):r for r in evidence['rows']}
    conditions=('none','abrupt_bf','persistent_alie','slow_ipm')
    methods=('risk_mean','risk_rfa','risk_winsor')
    assert len(evidence['rows'])==len(index)==24
    assert set(index)=={(s,a,m) for s in (170501,170502) for a in conditions for m in methods}
    hashes={str(REPORT.relative_to(ROOT)):base.digest(REPORT),str(Path(__file__).relative_to(ROOT)):base.digest(Path(__file__))}
    checks=[];gate=[]
    for seed in (170501,170502):
        saved_path=SCREEN/f'seed{seed}.json'
        saved=json.loads(saved_path.read_text())
        base.verify_stamp(saved['signature']['inputs']);base.verify_stamp(saved['signature']['stamp'])
        hashes[str(saved_path.relative_to(ROOT))]=base.digest(saved_path)
        cp=torch.load(SOURCE/f'seed{seed}__risk_rfa'/'checkpoint.pt',map_location='cpu',weights_only=True)
        model=base.new_model(manifest['profile'],seed);data=base.prepare(manifest['profile'],seed)
        clean=cp['last_private_messages'].to('mps');clean_reports=cp['last_reports'].to('mps')
        honest=clean[2:];mu=honest.mean(0)
        sd=((honest-mu).square().mean(0)).sqrt()
        for attack in conditions:
            x=clean.clone();reports=clean_reports.clone()
            if attack!='none':
                forged={'abrupt_bf':-10*mu,'persistent_alie':mu+1.5*sd,'slow_ipm':-2*mu}[attack]
                x[:2]=forged;reports[:2]=1
            coeff=1+4*reports.clamp(0,.5);lam=coeff/coeff.sum()
            mean=(lam[:,None]*x).sum(0)
            rfa,_=weighted_rfa(x,lam)
            pilot,_=weighted_rfa(x,torch.full((10,),.1,device='mps'))
            norm=torch.linalg.vector_norm(pilot)
            center=pilot if float(norm)<=2 else 2*pilot/norm
            ds=torch.linalg.vector_norm(x-center,dim=1)
            tau=max(4.,float(torch.kthvalue(ds,8).values))
            transformed=center+(x-center)*(tau/ds.clamp_min(1e-30)).clamp(max=1)[:,None]
            candidate=(lam[:,None]*transformed).sum(0)
            row=index[seed,attack,'risk_winsor'];diag=row['candidate_diagnostics']
            assert abs(tau-diag['radius'])<1e-5
            assert int((ds>tau).sum())==diag['clipped_count']
            beta=float(lam[:2].sum());H=(lam[2:,None]*x[2:]).sum(0)/(1-beta)
            removal=(lam[2:,None]*(x[2:]-transformed[2:])).sum(0)
            byz=(lam[:2,None]*(transformed[:2]-center)).sum(0)
            torch.testing.assert_close(candidate-H,-removal+beta*(center-H)+byz,rtol=1e-5,atol=2e-6)
            assert float(torch.linalg.vector_norm(byz))<=beta*tau+1e-5
            for kind,A in (('risk_mean',mean),('risk_rfa',rfa),('risk_winsor',candidate)):
                model.load_state_dict(cp['pre_round_model']);base.apply_gradient(model,A,.5)
                ev=base.evaluate(model,data,'val');ref=index[seed,attack,kind]['evaluation']
                for a,b in zip(ev['clients'],ref['clients']):
                    assert a['class_hits']==b['class_hits'] and a['class_count']==b['class_count']
                for k in ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2',
                          'balanced_accuracy_pct','ce_loss','brier_loss'):
                    assert math.isclose(ev[k],ref[k],rel_tol=2e-6,abs_tol=2e-6),(seed,attack,kind,k)
                if attack=='none' and kind=='risk_rfa':
                    assert all(torch.equal(v,cp['model'][k].to('mps')) for k,v in model.state_dict().items())
                checks.append(dict(seed=seed,attack=attack,method=kind,counts_exact=True,metrics_agree=True))
            print(f'V32 independent replay seed {seed} {attack}: 3/3 PASS',flush=True)
        def vals(a,m,all_clients=False):return exact(index[seed,a,m]['evaluation'],range(10) if all_clients else range(2,10))
        ca,cw=vals('none','risk_winsor',True);ma,mw=vals('none','risk_mean',True)
        gate.append(ca-ma>=Q(-1,10) and cw-mw>=Q(-1,4))
        clean_a,clean_w=vals('none','risk_winsor')
        for a in conditions[1:]:
            va,vw=vals(a,'risk_winsor');ra,rw=vals(a,'risk_rfa')
            gate.extend((va-clean_a>=-1 and vw-clean_w>=-1,va-ra>=-1 and vw-rw>=-1))
        del model,data,cp
        torch.mps.empty_cache()
    assert all(gate)==evidence['decision']['local_feasibility_passed']
    for p,h in hashes.items():assert base.digest(ROOT/p)==h
    base.verify_stamp(evidence['source_stamp'])
    base.save(DEST,dict(checked_steps=len(checks),checks=checks,device='mps',
        independent_gate_passed=all(gate),audit_passed=True,inputs=hashes,
        source_stamp=evidence['source_stamp'],test_evaluated=False,global_validation=False,
        shared_primitives=['data/model loader','RFA numerical primitive','evaluation forward helper'],
        independently_rederived=['attack vectors','risk weights','pilot projection',
            'order-statistic radius','radial clipping','mean reconstruction','count gate']))
    print(json.dumps(dict(audit_passed=True,checked_steps=24,local_gate_passed=all(gate),global_validation=False)),flush=True)


if __name__=='__main__':main()
