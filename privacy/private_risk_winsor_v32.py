"""V32 calibration candidate: risk-weighted centered clipping, not new DP.

No raw data, honest identities or clean gradients enter this postprocessor.
No claim of novelty or universal fairness/Byzantine convergence.
"""
import math
import torch
from privacy.fair_objective import require_mps
from privacy.capped_private_risk import weights
from privacy.stable_weighted_rfa import weighted_rfa, stable_norm


def winsor(messages, reports, *, C, f_budget):
    require_mps()
    if (messages.device.type!='mps' or messages.dtype!=torch.float32 or messages.ndim!=2
            or min(messages.shape)<1 or not bool(torch.isfinite(messages).all())
            or not math.isfinite(C) or C<=0 or type(f_budget)!=int
            or not 0<=f_budget<len(messages)/2):
        raise ValueError('Finite MPS matrix, positive public C and 0<=f<n/2 required')
    if reports.device.type!='mps' or reports.shape!=(len(messages),):
        raise ValueError('One private report per message required')
    lam,_=weights(reports,.5)
    # Unweighted pilot: forged risk reports cannot increase pilot objective mass.
    pilot,solver=weighted_rfa(messages,torch.ones(len(messages),device='mps')/len(messages))
    center=pilot*torch.minimum(torch.ones((),device='mps'),C/stable_norm(pilot).clamp_min(1e-30))
    residuals=messages-center
    distances=stable_norm(residuals,dim=1)
    kth=distances.sort().values[len(messages)-f_budget-1]
    radius=torch.maximum(torch.tensor(2*C,device='mps'),kth)
    factors=torch.minimum(torch.ones_like(distances),radius/distances.clamp_min(1e-30))
    clipped=residuals*factors[:,None]
    transformed=center+clipped
    aggregate=(lam[:,None]*transformed).sum(0)
    if not bool(torch.isfinite(aggregate).all()):
        raise FloatingPointError('Invalid postprocessed aggregate')
    diag=dict(center_norm=float(stable_norm(center)),radius=float(radius),
        order_statistic=float(kth),f_budget=f_budget,local_clip=C,
        clipped_count=int((distances>radius).sum()),objective_weights=lam.cpu().tolist(),
        factors=factors.cpu().tolist(),pilot_solver=solver,
        center_projection_active=bool(stable_norm(pilot)>C),
        method='risk-weighted centered clipping; projected unweighted RFA pilot',
        raw_data_used=False,honest_identities_used=False)
    return aggregate,diag,dict(center=center,clipped_residuals=clipped,
        transformed=transformed,weights=lam,removed=messages-transformed)
