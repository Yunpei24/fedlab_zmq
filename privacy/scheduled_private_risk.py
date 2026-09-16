"""Public late-stage step reduction; independent of private observations."""
import torch
from privacy.capped_private_risk import weights
from privacy.stable_weighted_rfa import weighted_rfa


def learning_rate(round_number, horizon=60):
    if type(round_number) is not int or type(horizon) is not int or horizon < 2 or not 1 <= round_number <= horizon:
        raise ValueError('Valid public round/horizon required')
    return 2. if round_number <= horizon//2 else .5


def aggregate(messages, reports, *, kind, round_number, horizon=60):
    if messages.device.type != 'mps' or messages.ndim != 2 or len(messages)==0 or not bool(torch.isfinite(messages).all()):
        raise ValueError('Finite MPS messages required')
    if kind not in ('erm_mean','erm_rfa','risk_mean','risk_rfa'):
        raise ValueError('Unknown method')
    if kind.startswith('risk_'):
        if reports is None or reports.shape != (len(messages),):
            raise ValueError('One private report per message required')
        lam,_=weights(reports,.5)
    else:
        lam=torch.ones(len(messages),device='mps')/len(messages)
    if kind.endswith('rfa'):
        center, solver=weighted_rfa(messages,lam)
    else:
        center=(lam[:,None]*messages).sum(0);solver=None
    eta=learning_rate(round_number,horizon)
    return eta*center,dict(eta=eta,objective_weights=lam.cpu().tolist(),solver=solver,
                           max_weight=float(lam.max()),concentration=float(len(lam)*lam.square().sum()))
