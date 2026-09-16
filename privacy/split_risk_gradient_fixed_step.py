"""V6: only the global step differs from V5; frozen V5 is not modified."""
import math
import torch
from privacy.split_risk_gradient import risk_weights, weighted_rfa


def aggregate(messages,reports,*,method,beta=2.,eta_erm=2.,eta_fair=2.):
    if messages.device.type!='mps' or messages.ndim!=2 or not len(messages):
        raise ValueError('Nonempty MPS messages required')
    if not bool(torch.isfinite(messages).all()):raise ValueError('Nonfinite messages')
    if not all(math.isfinite(x) and x>0 for x in (eta_erm,eta_fair)) or eta_erm!=eta_fair:
        raise ValueError('V6 requires identical fixed steps in all arms')
    uniform=torch.ones(len(messages),device='mps')/len(messages)
    if method=='erm_full':chosen=uniform
    else:
        if reports is None or reports.shape!=(len(messages),):raise ValueError('One private report per client')
        weights,_=risk_weights(reports,beta)
        chosen=weights if method.startswith('risk_') else uniform
    if method not in ('erm_full','matched_mean','risk_mean','matched_rfa','risk_rfa'):
        raise ValueError('Unknown method')
    if method.endswith('rfa'):
        direction,solver=weighted_rfa(messages,chosen)
    else:
        direction=(chosen[:,None]*messages).sum(0);solver=None
    return eta_erm*direction,dict(eta=eta_erm,weights=chosen.cpu().tolist(),solver=solver,
        max_weight=float(chosen.max()),concentration=float(len(messages)*chosen.square().sum()),fixed_step=True)
