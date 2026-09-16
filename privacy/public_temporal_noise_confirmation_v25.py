"""Constant and fixed V24-promoted privacy plans for independent confirmation."""
from functools import lru_cache
import math
import statistics as st
from privacy.public_temporal_noise_v23 import plan as scalar_plan
from privacy.public_temporal_noise_training_v24 import METHODS


@lru_cache(maxsize=8,typed=True)
def plan(grid_index,method):
    if type(grid_index) is not int or grid_index not in (0,13) or method not in METHODS:
        raise ValueError('Only constant or the frozen promoted schedule is permitted')
    risk = method.startswith('risk_'); p = scalar_plan(2**(grid_index/16),risk)
    p.update(grid_index=grid_index,N=4800,b=240,T=120,C=2.,epsilon_cap=4.,boundary_round=60,
        risk_std=p['risk_z']/4800 if risk else 0.,gradient_releases=120,risk_releases=120 if risk else 0,
        gradient_sensitivity=4/240,risk_sensitivity=1/4800 if risk else None,
        privacy_scope='sample-level per-client per-run ideal Gaussian mechanism',epsilon_risk=.25 if risk else None)
    return p


def paired_summary(values):
    if len(values) != 4 or not all(math.isfinite(x) for x in values):
        raise ValueError('Exactly four finite seed differences required')
    mean,sd = st.mean(values),st.stdev(values); radius = 3.182446305284263*sd/2
    return dict(n=4,mean=mean,sd=sd,ci95=[mean-radius,mean+radius],df=3)


def compare(pairs):
    if len(pairs) != 4:
        raise ValueError('Four seeds required')
    keys = ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2')
    summaries = {k:paired_summary([p[k] for p in pairs]) for k in keys}
    gates = dict(all_seed_gates=all(p['accuracy_pct']>=-1. and p['worst20_pct']>=1. for p in pairs),
        worst20_ci_lower_positive=summaries['worst20_pct']['ci95'][0]>0,
        accuracy_ci_lower_noninferior=summaries['accuracy_pct']['ci95'][0]>=-1.,
        mean_gap_nonincreasing=summaries['gap_best20_worst20_pp']['mean']<=0,
        mean_variance_nonincreasing=summaries['variance_pp2']['mean']<=0)
    return dict(summaries=summaries,gates=gates,passed=all(gates.values()))
