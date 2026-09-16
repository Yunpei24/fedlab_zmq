import pytest
import torch
from privacy.fair_objective import require_mps
from privacy.private_risk_attacks import inject
from privacy.scheduled_private_risk import aggregate
from privacy.private_risk_message_safety import sanitize
from privacy.capped_private_risk import weights
from privacy.stable_weighted_rfa import weighted_rfa


def data():
    require_mps()
    return torch.arange(50,device='mps',dtype=torch.float32).reshape(10,5)/100,torch.linspace(0,.5,10,device='mps')


@pytest.mark.parametrize('attack',['none','abrupt_bf','persistent_alie','slow_ipm'])
def test_inactive_is_exact_identity_and_never_mutates_inputs(attack):
    x,r = data()
    xx,rr = x.clone(),r.clone()
    y,s,d = inject(x,r,attack=attack,round_number=30)
    assert torch.equal(x,y) and torch.equal(r,s) and not d['active']
    assert torch.equal(x,xx) and torch.equal(r,rr)
    y,s,d = inject(x,r,attack=attack,round_number=120)
    if attack!='persistent_alie':
        assert torch.equal(x,y) and torch.equal(r,s) and not d['active']
    else:
        assert d['active']


def test_exact_attack_formula_and_population_std():
    x,r = data()
    mean = x[2:].mean(0)
    for attack,expected in [('abrupt_bf',-10*mean),('persistent_alie',mean+1.5*x[2:].std(0,correction=0)),('slow_ipm',-2*mean)]:
        y,s,d = inject(x,r,attack=attack,round_number=60)
        torch.testing.assert_close(y[0],expected)
        assert torch.equal(y[0],y[1]) and torch.equal(y[2:],x[2:])
        assert torch.equal(s[2:],r[2:]) and bool((s[:2]==1).all()) and d['active']
    y,_,d = inject(x,None,attack='slow_ipm',round_number=31)
    assert bool((y[:2]==0).all()) and d['multiplier']==0
    _,_,d = inject(x,None,attack='slow_ipm',round_number=90)
    assert d['multiplier']==2


def test_private_risk_objective_mass_bounded_despite_forgery():
    x,r = data()
    for attack in ('abrupt_bf','persistent_alie','slow_ipm'):
        y,s,_ = inject(x,r,attack=attack,round_number=60)
        y,s,_ = sanitize(y,s)
        a,d = aggregate(y,s,kind='risk_rfa',round_number=60,horizon=120)
        assert bool(torch.isfinite(a).all())
        assert sum(d['objective_weights'][:2])<=3/7+1e-6
        # A finite 40-iteration solve does not promise a universal 1e-3 bound.
        # The real-campaign precision gate remains independently enforced.
        assert d['solver']['unsmoothed_objective_gap_upper']>=0


def test_constructed_alie_exposes_nonuniversal_40_iteration_certificate():
    x,r = data()
    y,s,_ = inject(x,r,attack='persistent_alie',round_number=60)
    w,_ = weights(s,.5)
    _,d40 = weighted_rfa(y,w,iterations=40)
    _,d160 = weighted_rfa(y,w,iterations=160)
    assert d40['unsmoothed_objective_gap_upper']>1e-3
    assert d160['unsmoothed_objective_gap_upper']<1e-3
    # This diagnostic is not an authorization to change the frozen candidate.


def test_attack_never_uses_corrupted_client_inputs():
    x,r = data()
    y,_,_ = inject(x,r,attack='persistent_alie',round_number=75)
    xx=x.clone();xx[:2]=12345
    yy,_,_=inject(xx,r,attack='persistent_alie',round_number=75)
    assert torch.equal(y,yy)


def test_bad_round_is_rejected():
    x,r = data()
    with pytest.raises(ValueError):
        inject(x,r,attack='abrupt_bf',round_number=0)
