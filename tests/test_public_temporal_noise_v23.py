import math
import pytest
from privacy.public_temporal_noise_v23 import composed,simplified_gaussian


def test_simplified_cauchy_schwarz_optimum():
    r=simplified_gaussian()
    assert r['early_variance_ratio']==.625 and r['late_variance_ratio']==2.5
    assert math.isclose(r['inverse_variance_cost'],120.)
    assert r['energy']==187.5 and r['uniform_energy']==255.
    assert math.isclose(r['energy']*r['inverse_variance_cost'],150**2)


def test_sampled_composition_not_simplified_gaussian():
    from privacy.fair_objective import wor_rdp
    ep,order,r=composed(1.8,2.,260.)
    for a in r:
        assert math.isclose(r[a],60*wor_rdp(a,.05,1.8)+60*wor_rdp(a,.05,3.6)+120*a/(2*260**2))
    assert ep==min(r[a]+math.log(1e5)/(a-1) for a in r)


def test_constant_noise_and_invalid_inputs():
    from privacy.fair_objective import wor_rdp
    ep,order,r=composed(1.8,1.)
    for a in r:assert r[a]==120*wor_rdp(a,.05,1.8)
    for z in [0.,-1.,float('nan')]:
        with pytest.raises(ValueError):composed(z,1.)


def test_exact_risk_concentration_ceiling():
    candidates=[(10+8*k)/(10+2*k)**2 for k in range(11)]
    assert max(candidates)==34/256 and candidates.index(max(candidates))==3
