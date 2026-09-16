import math
import mpmath as mp
import pytest
from privacy.additive_gaussian_wor_v27 import profile
from privacy.fair_objective import wor_rdp


def test_boundaries_and_exact_order_two():
    assert profile(q=0, z=2., orders=(2,7)) == {2:0.,7:0.}
    for a, value in profile(q=1, z=2., orders=(2,7)).items():
        assert value >= a/8 and math.isclose(value, a/8, rel_tol=1e-14)
    q,z=.05,1.74
    expected=math.log1p(q*q*min(4*math.expm1(1/z**2),2*math.exp(1/z**2)))
    assert math.isclose(profile(q=q,z=z,orders=(2,))[2], expected, rel_tol=1e-13)


def test_precision_crosscheck_and_generic_envelope():
    for q,z in ((.05,1.68),(.1,3.04),(.2,5.9),(.5,14.3)):
        lo=profile(q=q,z=z,orders=tuple(range(2,65)),precision=160)
        hi=profile(q=q,z=z,orders=tuple(range(2,65)),precision=320)
        for a in lo:
            assert math.isclose(lo[a],hi[a],rel_tol=2e-14,abs_tol=1e-15)
            assert lo[a] <= wor_rdp(a,q,z)*(1+1e-12)+1e-14
            assert lo[a] > 0


def test_order_four_independent_closed_moment():
    with mp.workdps(100):
        q,z=mp.mpf(.05),mp.mpf(2.)
        u=1/z**2
        B2=mp.exp(u)-1
        B4=mp.exp(6*u)-4*mp.exp(3*u)+6*mp.exp(u)-3
        t2=min(4*B2,2*mp.exp(u))
        t3=min(4*mp.sqrt(B2*B4),2*mp.exp(3*u))
        t4=min(4*B4,2*mp.exp(6*u))
        exact=min(mp.log(1+6*q**2*t2+4*q**3*t3+q**4*t4)/3,2/z**2)
        reported=profile(q=float(q),z=float(z),orders=(4,))[4]
        assert mp.mpf(reported) >= exact
        assert math.isclose(reported,float(exact),rel_tol=1e-14)


def test_domain_and_precision_fail_closed():
    for kwargs in (dict(q=.1,z=1000),dict(q=.1,z=2,precision=30),dict(q=.1,z=2,orders=(65,)),dict(q=-1,z=2)):
        with pytest.raises(ValueError):profile(**kwargs)
