"""Two publicly selected noise schedules; unchanged fresh private queries."""
from functools import lru_cache
import math
from privacy.public_temporal_noise_v23 import plan as scalar_plan
from privacy.fair_objective import wor_rdp

METHODS = ('erm_mean', 'erm_rfa', 'risk_mean', 'risk_rfa')
GRID_INDICES = (7, 13)


@lru_cache(maxsize=8, typed=True)
def plan(grid_index, method):
    if type(grid_index) is not int or grid_index not in GRID_INDICES or method not in METHODS:
        raise ValueError('Frozen public grid and one of four methods required')
    risk = method.startswith('risk_')
    p = scalar_plan(2**(grid_index/16), risk)
    p.update(grid_index=grid_index, N=4800, b=240, T=120, C=2., epsilon_cap=4.,
             boundary_round=60, risk_std=p['risk_z']/4800 if risk else 0.,
             gradient_releases=120, risk_releases=120 if risk else 0,
             gradient_sensitivity=4/240, risk_sensitivity=1/4800 if risk else None,
             privacy_scope='sample-level per-client per-run; ideal Gaussian mechanism',
             epsilon_risk=0.25 if risk else None)
    return p


def round_parameters(p, round_number):
    if type(round_number) is not int or not 1 <= round_number <= 120:
        raise ValueError('Round must be in 1..120')
    phase = 'early' if round_number <= 60 else 'late'
    return dict(phase=phase, z=p[f'z_{phase}'], sigma=p[f'sigma_{phase}'],
                eta=2. if phase == 'early' else .5)


def prefix_ledger(p, rounds):
    if type(rounds) is not int or not 0 <= rounds <= 120:
        raise ValueError('Prefix must be in 0..120')
    early, late = min(rounds, 60), max(0, rounds-60)
    rdp = {a: early*wor_rdp(a,.05,p['z_early'])+late*wor_rdp(a,.05,p['z_late'])
           +(rounds*a/(2*p['risk_z']**2) if p['with_risk'] else 0.) for a in range(2,65)}
    epsilon, order = min((cost+math.log(1e5)/(a-1),a) for a,cost in rdp.items())
    return dict(epsilon=0. if rounds == 0 else epsilon, order=order, rdp=rdp,
                gradient_releases=rounds, risk_releases=rounds if p['with_risk'] else 0)


def contrast_passes(pairs):
    if len(pairs) != 2:
        raise ValueError('Exactly the two registered calibration seeds are required')
    return (all(d['accuracy_pct'] >= -1. and d['worst20_pct'] >= 1. for d in pairs)
            and all(sum(d[k] for d in pairs) <= 0. for k in ('gap_best20_worst20_pp','variance_pp2')))
