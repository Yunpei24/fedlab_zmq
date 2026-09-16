"""Predeclared upload attacks. Never consumes raw data or unnoised gradients."""
import torch
from privacy.stable_weighted_rfa import stable_norm


def inject(messages, reports, *, attack, round_number):
    if (messages.device.type!='mps' or messages.dtype!=torch.float32
            or messages.ndim!=2 or len(messages)!=10 or not bool(torch.isfinite(messages).all())):
        raise ValueError('Ten finite parsed MPS messages required')
    if reports is not None and (reports.device.type!='mps' or reports.shape!=(10,)):
        raise ValueError('Ten MPS reports required')
    if attack not in ('none','abrupt_bf','persistent_alie','slow_ipm') or type(round_number)!=int or not 1<=round_number<=120:
        raise ValueError('Unknown attack/public round')
    active = attack!='none' and 31<=round_number<=(120 if attack=='persistent_alie' else 90)
    out = messages.clone()
    rr = None if reports is None else reports.clone()
    multiplier = None
    if active:
        honest = messages[2:]
        mean = honest.mean(0)
        if attack=='abrupt_bf':
            forged = -10.*mean
            multiplier = 10.
        elif attack=='persistent_alie':
            forged = mean+1.5*honest.std(dim=0,correction=0)
            multiplier = 1.5
        else:
            multiplier = 2.*min(round_number-31,29)/29
            forged = -multiplier*mean
        out[:2] = forged
        if rr is not None:
            rr[:2] = 1.
    return out,rr,dict(attack=attack,active=active,byzantine_ids=[0,1],
        multiplier=multiplier,forged_risk=1. if active and reports is not None else None,
        forged_norm=float(stable_norm(out[0])) if active else None,
        source='current private honest uploads only; no unnoised oracle')
