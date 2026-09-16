"""Bounded-risk potential: vary risk emphasis without raising Byzantine mass cap."""
import math
import torch
from privacy.split_risk_gradient import weighted_rfa


def potential(risk,scale):
    if not math.isfinite(scale) or scale<=0:raise ValueError('Positive public risk scale')
    return risk+torch.where(risk<=scale,risk.square()/scale,2*risk-scale)


def weights(reports,scale):
    if reports.device.type!='mps' or reports.ndim!=1 or not len(reports):raise ValueError('MPS reports required')
    if not math.isfinite(scale) or scale<=0 or not bool(torch.isfinite(reports).all()):raise ValueError('Invalid risk inputs')
    a=1+2*(reports.clamp(0,1)/scale).clamp(max=1)
    return a/a.sum(),a


def aggregate(messages,reports,*,kind,scale=None,eta=2.):
    if messages.device.type!='mps' or messages.ndim!=2 or not len(messages) or not bool(torch.isfinite(messages).all()):
        raise ValueError('Finite MPS messages required')
    if kind not in ('erm_mean','erm_rfa','risk_mean','risk_rfa') or not math.isfinite(eta) or eta<=0:raise ValueError('Invalid aggregation')
    n=len(messages);lam=torch.ones(n,device='mps')/n
    coeff=None
    if kind.startswith('risk_'):
        if reports is None or reports.shape!=(n,):raise ValueError('One private risk per client')
        lam,coeff=weights(reports,scale)
    mean=(lam[:,None]*messages).sum(0)
    center=mean;solver=None;effective=lam;reconstruction_error=0.
    if kind.endswith('rfa'):
        center,solver=weighted_rfa(messages,lam)
        # Weights at the returned point, not the prior loss weights. Their
        # reconstruction discrepancy quantifies the finite-solver approximation.
        distances=((messages-center).square().sum(1)+solver['smoothing']**2).sqrt()
        effective=lam/distances;effective/=effective.sum()
        reconstruction_error=float(torch.linalg.vector_norm((effective[:,None]*messages).sum(0)-center))
    diag=dict(eta=eta,objective_weights=lam.cpu().tolist(),stationary_weights=effective.cpu().tolist(),
        stationary_reconstruction_error=reconstruction_error,
        reference_minus_weighted_mean_norm=float(torch.linalg.vector_norm(center-mean)),solver=solver,
        coefficient_range=None if coeff is None else [float(coeff.min()),float(coeff.max())],
        saturated_coefficient_fraction=None if coeff is None else float((coeff>=3-1e-7).float().mean()))
    return eta*center,diag
