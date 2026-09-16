"""Public optimization of the linear Gaussian recurrence, not an utility claim."""
import math


def variance_ratio(theta, d):
    if not all(math.isfinite(x) for x in (theta,d)) or not 0<theta<=1 or not 0<d<=1:
        raise ValueError('0<theta,d<=1 required')
    return (theta+(1-theta)*d)**2/(theta*(2-theta))


def largest_increment_ratio(*, C, variance_ceiling=.5, numerical_margin=1e-8):
    if not all(math.isfinite(x) for x in (C,variance_ceiling,numerical_margin)) or C<=0 or not 0<numerical_margin<variance_ceiling<1:
        raise ValueError('Positive C, 0<margin<ceiling<1 required')
    target=variance_ceiling-numerical_margin
    d=1-math.sqrt(1-target)
    theta=d
    return dict(C=C,D=C*d,D_over_C=d,theta=theta,variance_ceiling=variance_ceiling,
                numerical_margin=numerical_margin,stationary_ratio=variance_ratio(theta,d),
                finite_ratio_t20=(1-theta)**40+variance_ratio(theta,d)*(1-(1-theta)**40))
