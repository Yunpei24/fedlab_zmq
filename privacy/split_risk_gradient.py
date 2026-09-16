"""Separate sample-private risk/gradient channels; aggregation is post-processing.

This is a mechanism-control experiment, not a novelty or utility claim.
All private tensors stay on MPS. The returned RDP ledger is host scalar arithmetic.
"""
import math
import torch
from privacy.fair_objective import calibrate, wor_rdp, require_mps, release, losses


def plan(*, N=4800, b=240, T=60, C=1., epsilon=4., delta=1e-5, epsilon_risk=.25):
    if (not all(math.isfinite(v) for v in (N,b,T,C,epsilon,delta,epsilon_risk))
        or not 2 <= b <= N or any(int(v)!=v for v in (N,b,T)) or T < 1 or C <= 0
        or not 0 < epsilon_risk < epsilon or not 0 < delta < 1):
        raise ValueError('Invalid public privacy plan')
    if epsilon_risk <= math.log(2/delta)/63:
        raise ValueError('Risk calibration needs a wider order grid for this small epsilon')
    zr = calibrate(q=1., steps=T, epsilon=epsilon_risk, delta=delta/2)
    def joint(z):
        return min((T*wor_rdp(a,b/N,z)+T*a/(2*zr*zr)+math.log(1/delta)/(a-1),a)
                   for a in range(2,65))
    low,high = .01,1.
    while joint(high)[0] > epsilon:
        high *= 2
    for _ in range(70):
        mid = (low+high)/2
        if joint(mid)[0] > epsilon:
            low = mid
        else:
            high = mid
    zg = high*(1+1e-8)
    ledger = {a: T*wor_rdp(a,b/N,zg)+T*a/(2*zr*zr) for a in range(2,65)}
    realized, order = min((v+math.log(1/delta)/(a-1),a) for a,v in ledger.items())
    assert realized <= epsilon
    return dict(N=N,b=b,T=T,C=C,epsilon_cap=epsilon,delta=delta,epsilon_realized=realized,
        order=order,gradient_z=zg,risk_z=zr,gradient_std=zg*2*C/b,risk_std=zr/N,
        gradient_sensitivity=2*C/b,risk_sensitivity=1/N,
        risk_calibration_epsilon=epsilon_risk,risk_calibration_delta=delta/2,
        calibration='gradient z solved against joint risk+gradient RDP at total epsilon/delta',
        rdp=ledger,adjacency='replace_one',sampling='fixed_without_replacement',
        risk_releases=T,gradient_releases=T,scope='sample-level per-client per-run messages only')


@torch.no_grad()
def private_risk(model,x,y,*,noise_std,seed,N,batch_size=512):
    require_mps()
    if (len(y) != N or len(x) != N or N < 1 or batch_size < 1
        or x.device.type != 'mps' or y.device.type != 'mps'):
        raise ValueError('Fixed public-size local dataset required on MPS')
    risk = torch.zeros(1,device='mps')
    for start in range(0,N,batch_size):
        rr = losses(model(x[start:start+batch_size]),y[start:start+batch_size],'brier')
        if not bool(torch.isfinite(rr).all()) or float(rr.min()) < -1e-6 or float(rr.max()) > 1+1e-6:
            raise ValueError('Risk loss outside certified range')
        risk += rr.sum()/N
    noisy = release(risk,noise_std=noise_std,seed=seed).clamp(0,1)
    return noisy.squeeze(0),risk.squeeze(0)  # raw second value is an explicitly private research oracle


def risk_weights(reports,beta=2.):
    if reports.device.type != 'mps' or reports.ndim != 1 or reports.numel()==0 or not math.isfinite(beta) or beta < 0:
        raise ValueError('MPS vector and nonnegative beta required')
    if not bool(torch.isfinite(reports).all()):
        raise ValueError('Non-finite report')
    coeff = 1+beta*reports.clamp(0,1)
    return coeff/coeff.sum(),coeff.mean()


def weighted_rfa(vectors,weights,*,iterations=40,smoothing=1e-5):
    """Fixed-iteration smooth geometric median, with an a posteriori gap bound.

    The gradient-residual certificate uses convexity and the diameter of the
    public-message convex hull; it is not an oracle accuracy certificate.
    """
    if vectors.device.type != 'mps' or weights.device.type != 'mps':
        raise ValueError('MPS tensors required')
    if vectors.ndim != 2 or weights.shape != (len(vectors),) or iterations < 1 or smoothing <= 0:
        raise ValueError('Invalid dimensions/solver parameters')
    if not bool(torch.isfinite(vectors).all()) or not bool(torch.isfinite(weights).all()) or float(weights.min()) <= 0:
        raise ValueError('Finite vectors, positive weights required')
    w = weights/weights.sum()
    point = (w[:,None]*vectors).sum(0)
    for _ in range(iterations):
        distances = ((vectors-point).square().sum(1)+smoothing*smoothing).sqrt()
        reweight = w/distances
        point = (reweight[:,None]*vectors).sum(0)/reweight.sum()
    diff = point-vectors
    dist = (diff.square().sum(1)+smoothing*smoothing).sqrt()
    residual = ((w/dist)[:,None]*diff).sum(0)
    # Both point and a minimizer lie in the convex hull; this encloses its diameter.
    diameter_bound = 2*torch.linalg.vector_norm(vectors-point,dim=1).max()
    gap = float(torch.linalg.vector_norm(residual)*diameter_bound)+smoothing
    return point,dict(iterations=iterations,smoothing=smoothing,
        unsmoothed_objective_gap_upper=gap,
        residual_norm=float(torch.linalg.vector_norm(residual)))


def aggregate(messages,reports,*,method,beta=2.,eta_erm=2.,eta_fair=2/1.9):
    """Returns the complete parameter step, not an unscaled gradient."""
    if messages.device.type != 'mps' or messages.ndim != 2 or not bool(torch.isfinite(messages).all()):
        raise ValueError('Finite MPS messages required')
    if len(messages)==0 or eta_erm<=0 or eta_fair<=0 or not all(map(math.isfinite,(eta_erm,eta_fair))):
        raise ValueError('Positive finite learning rates and nonempty messages required')
    uniform = torch.ones(len(messages),device='mps')/len(messages)
    diagnostics = {}
    if method == 'erm_full':
        return eta_erm*messages.mean(0),dict(eta=eta_erm,weights=uniform.cpu().tolist(),solver=None)
    if reports is None or reports.shape != (len(messages),):
        raise ValueError('One private risk report per client required')
    w,mean_coeff = risk_weights(reports,beta)
    eta = eta_fair*mean_coeff
    chosen = w if method.startswith('risk_') else uniform
    if method not in ('matched_mean','risk_mean','matched_rfa','risk_rfa'):
        raise ValueError('Unknown aggregation method')
    if method.endswith('rfa'):
        center,diagnostics = weighted_rfa(messages,chosen)
    else:
        center = (chosen[:,None]*messages).sum(0)
    return eta*center,dict(eta=float(eta),weights=chosen.cpu().tolist(),solver=diagnostics,
        max_weight=float(chosen.max()),concentration=float(len(messages)*chosen.square().sum()))
