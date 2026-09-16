"""MPS checks of projection fidelity versus aggregate error; no training."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_fair_objective_screen as base
from privacy.fair_objective import require_mps
from privacy.private_risk_winsor_v32 import winsor

DEST=ROOT/'output/analysis/Private_Risk_V32_Projection_Theory_Checks.json'
DOC=ROOT/'output/analysis/Private_Risk_V32_Projection_vs_Aggregate_Error.md'

def diagnostic(x,mu,reports,*,C=1.,honest=tuple(range(10))):
    _,d,v=winsor(x,reports,C=C,f_budget=2)
    ids=list(honest);y=v['transformed'][ids];xx=x[ids];u=mu[ids]
    assert bool((torch.linalg.vector_norm(u,dim=1)<=C+1e-6).all())
    noise=xx-u;removed=xx-y
    before=noise.square().sum(1);after=(y-u).square().sum(1)
    residual=after+removed.square().sum(1)-before
    tolerance=2e-5*(1+float(before.max()))
    assert float(residual.max())<=tolerance
    p=v['weights'][ids];p=p/p.sum()
    zb=(p[:,None]*noise).sum(0);rh=(p[:,None]*removed).sum(0)
    hb=(p[:,None]*(xx-u)).sum(0);ha=(p[:,None]*(y-u)).sum(0)
    delta=ha.square().sum()-hb.square().sum()
    cost=rh.square().sum();alignment=2*(zb*rh).sum()
    torch.testing.assert_close(delta,cost-alignment,rtol=3e-5,atol=tolerance)
    jensen=(p*after).sum()
    assert float(ha.square().sum()-jensen)<=tolerance
    return dict(individual_error_before_sum=float(before.sum()),individual_error_after_sum=float(after.sum()),
        firm_projection_residual_max=float(residual.max()),tolerance=tolerance,
        honest_mean_error_before=float(hb.square().sum()),honest_mean_error_after=float(ha.square().sum()),
        aggregate_error_difference=float(delta),removed_mean_squared=float(cost),twice_noise_removal_inner=float(alignment),
        clipping_count=d['clipped_count'],radius=d['radius'],center_norm=d['center_norm'],
        honest_ids_for_oracle_only=ids)

def main():
    require_mps()
    stamp={str(p.relative_to(ROOT)):base.digest(p) for p in (Path(__file__),DOC,
        ROOT/'privacy/private_risk_winsor_v32.py',ROOT/'privacy/stable_weighted_rfa.py',ROOT/'privacy/capped_private_risk.py')}
    state=torch.mps.get_rng_state();cases=[]
    try:
        for dim in (1,7,61,1000):
            for sigma in (.1,1.,5.):
                torch.mps.manual_seed(330000+dim*100+int(10*sigma))
                mu=torch.randn(10,dim,device='mps')
                mu=mu/torch.linalg.vector_norm(mu,dim=1).clamp_min(1.)[:,None]
                mu*=torch.linspace(.1,.9,10,device='mps')[:,None]
                x=mu+sigma*torch.randn(10,dim,device='mps');r=torch.rand(10,device='mps')
                cases.append(dict(dim=dim,sigma=sigma,attack=False,**diagnostic(x,mu,r)))
                forged=x.clone();forged[:2]=1e3
                reports=r.clone();reports[:2]=1.
                cases.append(dict(dim=dim,sigma=sigma,attack=True,
                    **diagnostic(forged,mu,reports,honest=tuple(range(2,10)))))
        x=torch.tensor([[1.]]*9+[[-9.]],device='mps');mu=torch.zeros_like(x);r=torch.zeros(10,device='mps')
        counter=diagnostic(x,mu,r)
        assert counter['honest_mean_error_before']<1e-10 and abs(counter['honest_mean_error_after']-.64)<1e-4
        assert counter['individual_error_after_sum']<counter['individual_error_before_sum']
        assert abs(counter['removed_mean_squared']-.64)<1e-4
        assert counter['clipping_count']==1
    finally:torch.mps.set_rng_state(state)
    base.verify_stamp(stamp)
    result=dict(device='mps',numerical_checks_passed=True,random_cases=cases,counterexample=counter,source_stamp=stamp,
        changes_to_V33=False,global_validation=False,proof_is_analytical=True,
        disclaimer='Finite float32 tests, not interval certificates. Counterexample is pathwise, not expected Gaussian risk.')
    base.save(DEST,result)
    print(json.dumps(dict(passed=True,random_cases=len(cases),counterexample=counter,global_validation=False)))

if __name__=='__main__':main()
