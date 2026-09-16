"""Isolated Gaussian-specific WOR accounting audit, not wired into training.

WBK2019 supplementary Theorem21 (Theorem27 in the arXiv version), Gaussian
moments and the generic envelope. Fixed public N/B, replace-one sensitivity
2C/B. Interval arithmetic prevents cancellation from underestimating moments.
No applicability claim for batch-dependent preprocessing is made here.
"""
import math
import mpmath as mp


def _upper_float(x):
    value = float(x.b)
    if not math.isfinite(value) or value < 0:
        raise ArithmeticError('Invalid RDP upper endpoint')
    return math.nextafter(value, math.inf)


def _min_upper(a, b):
    # Each argument bounds the same quantity; its upper endpoint suffices.
    return a.b if a.b < b.b else b.b


def profile(*, q, z, orders=tuple(range(2, 65)), precision=160):
    if (not math.isfinite(q) or not 0 <= q <= 1 or not math.isfinite(z) or not .25 <= z <= 64
            or not orders or any(type(a) is not int or not 2 <= a <= 64 for a in orders)
            or type(precision) is not int or precision < 160):
        raise ValueError('Audited domain: q in [0,1], z in [.25,64], integer orders 2..64, >=160 digits')
    previous = mp.iv.dps
    try:
        mp.iv.dps = precision
        iv = mp.iv
        qq, zz = iv.mpf(q), iv.mpf(z)
        u = 1/(zz*zz)
        if q == 0:
            return {a: 0. for a in orders}
        if q == 1:
            return {a: _upper_float(a*u/2) for a in orders}
        top = max(orders) + max(orders) % 2
        moments = {}
        exponential = [iv.exp(j*(j-1)*u/2) for j in range(top+1)]
        for k in range(2, top+1, 2):
            value = iv.mpf(0)
            for j in range(k+1):
                value += ((-1)**(k-j))*math.comb(k, j)*exponential[j]
            if not value.a > 0:
                raise ArithmeticError('Even Gaussian moment not enclosed away from zero; increase precision')
            moments[k] = value
        bounds = {}
        for a in orders:
            total = iv.mpf(1)
            for j in range(2, a+1):
                if j == 2:
                    term = _min_upper(4*(iv.exp(u)-1), 2*iv.exp(u))
                else:
                    lo, hi = j-(j%2), j+(j%2)
                    tight = 4*iv.sqrt(moments[lo]*moments[hi])
                    term = _min_upper(tight, 2*exponential[j])
                total += math.comb(a, j)*qq**j*term
            sampled = iv.log(total)/(a-1)
            bounds[a] = _upper_float(_min_upper(sampled, a*u/2))
        return bounds
    finally:
        mp.iv.dps = previous


def plan(*, batch, risk_z=None, N=4800, T=120, C=2., epsilon=4., delta=1e-5):
    if (type(batch) is not int or not 2 <= batch <= N or N != 4800 or T != 120
            or C != 2. or epsilon != 4. or delta != 1e-5
            or (risk_z is not None and (not math.isfinite(risk_z) or risk_z <= 0))):
        raise ValueError('Only the frozen V27 public frontier is supported')
    orders = tuple(range(2, 65))
    def evaluate(z):
        rdp = profile(q=batch/N, z=z, orders=orders)
        # The risk channel stays at the original public Gaussian calibration.
        full = {a: T*rdp[a]+(T*a/(2*risk_z**2) if risk_z else 0.) for a in orders}
        eps, order = min((v+math.log(1/delta)/(a-1), a) for a, v in full.items())
        return eps, order, full
    lo, hi = .25, 32.
    if evaluate(lo)[0] <= epsilon:
        raise ArithmeticError('Lower calibration bracket must be infeasible')
    assert evaluate(hi)[0] < epsilon
    for _ in range(50):
        mid = (lo+hi)/2
        if evaluate(mid)[0] > epsilon:
            lo = mid
        else:
            hi = mid
    z = hi*(1+1e-8)
    e, order, ledger = evaluate(z)
    assert e <= epsilon
    return dict(N=N, batch=batch, rounds=T, clip=C, gradient_sensitivity=2*C/batch,
                gradient_z=z, gradient_std=z*2*C/batch, risk_z=risk_z,
                risk_std=risk_z/N if risk_z else None, epsilon_realized=e, delta=delta,
                order=order, rdp=ledger, interval_precision=160,
                method='Gaussian WOR moments + generic + unsampled envelopes',
                sampling='fixed_without_replacement', adjacency='replace_one',
                training_launched=False, global_validation=False)
