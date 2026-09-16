import math
import pytest
from privacy.recursive_public_calibration import variance_ratio,largest_increment_ratio


def test_public_optimizer_global_minimum_and_maximum_feasible_radius():
    for d in (.01,.1,.2,.292893,.4,.8,1.):
        expected=d*(2-d)
        assert math.isclose(variance_ratio(d,d),expected,rel_tol=1e-12)
        for k in range(1,1001):assert variance_ratio(k/1000,d)>=expected-1e-12
    for v in (.1,.25,.5,.75):
        p=largest_increment_ratio(C=2.,variance_ceiling=v)
        assert p['stationary_ratio']<=v and abs(p['stationary_ratio']-(v-1e-8))<1e-12
        assert p['D_over_C']==p['theta'] and p['D']==2*p['theta']
        # Any strictly larger d beyond the un-margined boundary is infeasible.
        d=1-math.sqrt(1-v)+1e-5
        assert variance_ratio(d,d)>v


def test_derivative_and_transient():
    for theta,d in ((.2,.1),(.1,.4),(.5,.2),(.8,.8)):
        eps=1e-6
        numerical=(variance_ratio(theta+eps,d)-variance_ratio(theta-eps,d))/(2*eps)
        exact=2*(d+(1-d)*theta)*(theta-d)/(theta**2*(2-theta)**2)
        assert math.isclose(numerical,exact,rel_tol=1e-6,abs_tol=1e-8)
    p=largest_increment_ratio(C=2.)
    assert p['stationary_ratio']<.5 and p['finite_ratio_t20']<.55


def test_bad_inputs_rejected():
    for C in (0.,-1.,float('nan')):
        with pytest.raises(ValueError):largest_increment_ratio(C=C)
