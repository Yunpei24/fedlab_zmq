"""Public scalar accountant calculation, without private data or model scores."""
import math
from privacy.fair_objective import calibrate,wor_rdp

ORDERS=tuple(range(2,65))


def composed(z_early,ratio,risk_z=None):
    if not math.isfinite(z_early) or z_early<=0 or not math.isfinite(ratio) or ratio<=0:
        raise ValueError('Positive finite noise multipliers required')
    if risk_z is not None and (not math.isfinite(risk_z) or risk_z<=0):raise ValueError('Invalid risk multiplier')
    ledger={a:60*wor_rdp(a,.05,z_early)+60*wor_rdp(a,.05,ratio*z_early)
            +(120*a/(2*risk_z*risk_z) if risk_z is not None else 0.) for a in ORDERS}
    epsilon,order=min((cost+math.log(1e5)/(a-1),a) for a,cost in ledger.items())
    return epsilon,order,ledger


def plan(ratio,with_risk):
    if ratio<1 or ratio>4 or type(with_risk) is not bool:raise ValueError('Public domain ratio[1,4], bool channel')
    risk_z=calibrate(q=1.,steps=120,epsilon=.25,delta=.5e-5) if with_risk else None
    low,high=.01,1.
    while composed(high,ratio,risk_z)[0]>4:high*=2
    for _ in range(70):
        mid=(low+high)/2
        if composed(mid,ratio,risk_z)[0]>4:low=mid
        else:high=mid
    z=high*(1+1e-8);eps,order,rdp=composed(z,ratio,risk_z);assert eps<=4
    std0=z*4/240;std1=ratio*std0
    return dict(ratio=ratio,with_risk=with_risk,z_early=z,z_late=z*ratio,risk_z=risk_z,
        sigma_early=std0,sigma_late=std1,energy_proxy=60*4*std0**2+60*.25*std1**2,
        epsilon_realized=eps,delta=1e-5,order=order,rdp=rdp,sensitivity=4/240,
        sampling='fixed_without_replacement',adjacency='replace_one',scope='Public Gaussian component proxy, not model MSE or RFA covariance')


def simplified_gaussian():
    eta=[2.]*60+[.5]*60;normalizer=sum(eta)/len(eta)
    variance=[normalizer/x for x in eta]
    energy=sum(x*x*v for x,v in zip(eta,variance));uniform=sum(x*x for x in eta)
    return dict(early_variance_ratio=variance[0],late_variance_ratio=variance[-1],
        inverse_variance_cost=sum(1/v for v in variance),energy=energy,uniform_energy=uniform,ratio=energy/uniform)
