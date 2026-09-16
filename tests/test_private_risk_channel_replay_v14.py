import itertools
import math
import torch
from privacy.fair_objective import require_mps
from privacy.capped_private_risk import weights
from privacy.split_risk_gradient import plan
from scripts import run_private_risk_channel_replay_v14 as run


def test_exact_public_concentration_constant_on_vertices():
    require_mps()
    a=torch.tensor(list(itertools.product((1.,3.),repeat=10)),device='mps')
    q=a/a.sum(1,keepdim=True)
    assert math.isclose(float(q.square().sum(1).max()),34/256,abs_tol=1e-7)


def test_normalized_risk_error_identity_and_cauchy_bound():
    require_mps();torch.mps.manual_seed(3401)
    g=torch.randn(10,8,device='mps');g=2*g/torch.linalg.vector_norm(g,dim=1)[:,None].clamp_min(1e-9)
    r=torch.linspace(0,1,10,device='mps');xi=.2*torch.randn(10,device='mps')
    w,a=weights(r,.5);ww,aa=weights((r+xi).clamp(0,1),.5)
    h=(w[:,None]*g).sum(0)
    difference=((ww-w)[:,None]*g).sum(0)
    reconstructed=((aa-a)[:,None]*(g-h)).sum(0)/aa.sum()
    torch.testing.assert_close(difference,reconstructed,atol=2e-7,rtol=2e-6)
    bound=(aa-a).square().sum()*(g-h).square().sum()/100
    assert float(difference.square().sum())<=float(bound)+1e-7
    assert bool(((aa-a).abs()<=4*xi.abs()+1e-6).all())


def test_joint_ledger_and_calibration_only_sources():
    a=plan(T=120,C=2,epsilon_risk=.25)
    b=plan(T=120,C=2,epsilon_risk=.5)
    assert a['epsilon_realized']<=4 and b['epsilon_realized']<=4
    assert b['risk_std']<a['risk_std'] and b['gradient_std']>a['gradient_std']
    assert run.SEEDS==[170501,170502] and run.METHODS==['original','reallocated','oracle_risk']
    profile,stamp=run.inputs()
    assert all('17060' not in path for path in stamp if 'checkpoint' in path)
    assert profile['model']=='lenet5_tanh' and profile['rounds']==120
