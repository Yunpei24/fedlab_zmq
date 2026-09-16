"""Gaussian norm compatibility gate followed by risk-SMEA, exploratory only.

Passing this gate does NOT authenticate a client. Bounded-mean Gaussian
adversaries pass the same necessary consistency test; they are tested explicitly.
"""
import math
import torch
from privacy.fair_objective import require_mps
from privacy.capped_private_risk import weights
from privacy.stable_weighted_rfa import stable_norm
from privacy.private_risk_spectral_v35 import risk_smea


def norm_bounds(*, dimension, C, std, n, horizon=132, false_intervention=.01):
    if (type(dimension) is not int or dimension < 1 or type(n) is not int or n < 1
            or type(horizon) is not int or horizon < 1
            or not math.isfinite(C) or C <= 0 or not math.isfinite(std) or std < 0
            or not math.isfinite(false_intervention) or not 0 < false_intervention < 1):
        raise ValueError('Valid public dimension, C, noise, n, horizon and probability required')
    x=math.log(2*n*horizon/false_intervention)
    lower2=max(0.,std*std*(dimension-2*math.sqrt(dimension*x)))
    upper2=C*C+dimension*std*std+2*std*math.sqrt((dimension*std*std+2*C*C)*x)+2*std*std*x
    return dict(lower=math.sqrt(lower2),upper=math.sqrt(upper2),x=x,
        false_intervention_per_honest_trajectory_upper=false_intervention,
        per_message_two_tails_upper=false_intervention/(n*horizon))


def gated_risk_smea(messages,reports,*,C,stds,f_budget=2,horizon=132,false_intervention=.01):
    require_mps()
    if (messages.device.type!='mps' or messages.dtype!=torch.float32 or messages.ndim!=2
            or reports.device.type!='mps' or reports.shape!=(len(messages),)
            or len(stds)!=len(messages) or not bool(torch.isfinite(messages).all())):
        raise ValueError('Finite MPS messages, one report and public std per client required')
    lam,_=weights(reports,.5)
    bounds=[norm_bounds(dimension=messages.shape[1],C=C,std=float(s),n=len(messages),
                        horizon=horizon,false_intervention=false_intervention) for s in stds]
    norms=stable_norm(messages,dim=1)
    low=torch.tensor([b['lower'] for b in bounds],device='mps')
    high=torch.tensor([b['upper'] for b in bounds],device='mps')
    flags=(norms<low)|(norms>high)
    incompatible=bool(flags.any())
    if incompatible:
        aggregate,spectral=risk_smea(messages,reports,f_budget=f_budget)
    else:
        aggregate=(lam[:,None]*messages).sum(0);spectral=None
    return aggregate,dict(branch='spectral' if incompatible else 'fair_mean',
        incompatible_ids=torch.where(flags)[0].cpu().tolist(),norms=norms.cpu().tolist(),
        bounds=bounds,spectral=spectral,false_intervention=false_intervention,horizon=horizon,
        decision_uses_oracle=False,authenticates_clients=False,global_validation=False)


def bounded_gaussian_attack(messages,*,C,stds,seed,target_coordinate=None):
    """Byzantines know their peers: adverse bounded mean plus independent noise.

    No honest clean mean, labels, evaluation losses or oracle gradients used.
    No retry/normalization to force acceptance. Two fresh public-seeded noise draws.
    """
    require_mps()
    if len(messages)!=10 or len(stds)!=10 or type(seed) is not int:
        raise ValueError('Frozen n=10 attack with a public integer seed required')
    if target_coordinate is None:
        direction=messages[2:].mean(0)
        norm=stable_norm(direction)
        badmean=-C*direction/norm.clamp_min(1e-30)
    else:
        if type(target_coordinate) is not int or not 0<=target_coordinate<messages.shape[1]:
            raise ValueError('Public model coordinate out of range')
        badmean=torch.zeros(messages.shape[1],device='mps');badmean[target_coordinate]=C
    state=torch.mps.get_rng_state()
    try:
        torch.mps.manual_seed(seed)
        z=torch.randn((2,messages.shape[1]),device='mps',dtype=torch.float32)
    finally:
        torch.mps.set_rng_state(state)
    result=messages.clone()
    for i in range(2):result[i]=badmean+float(stds[i])*z[i]
    return result,dict(mean_norm=float(stable_norm(badmean)),mean_radius=C,
        target_coordinate=target_coordinate,
        public_seed=seed,noise_std_byzantine=[float(s) for s in stds[:2]],
        independent_byzantine_noise=True,retries=0,uses_clean_honest_oracle=False)
