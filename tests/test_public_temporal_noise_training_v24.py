import json
import math
import pytest
import torch
from privacy.fair_objective import require_mps, release
from privacy.public_temporal_noise_v23 import plan as scalar_plan
from privacy.public_temporal_noise_training_v24 import plan, round_parameters, prefix_ledger, contrast_passes
from privacy.scheduled_private_risk import aggregate
from scripts.audit_public_temporal_noise_v23 import independent_rdp
from scripts.run_public_temporal_noise_training_v24 import jobs, identifier, canonical, inputs
from scripts.run_private_clipping_step_diagnostic_v16 import ledger


def test_frozen_inputs_and_complete_matrix():
    c,p,_ = inputs()
    js = jobs(c)
    assert len(js) == len({identifier(j) for j in js}) == 16
    assert p['model'] == 'lenet5_tanh' and p['rounds'] == 120
    assert p['evaluation_rounds'] == [1]+list(range(10,121,10))
    assert 'reserved_confirmation_seeds' not in p


@pytest.mark.parametrize('risk', [False,True])
@pytest.mark.parametrize('index', [7,13])
def test_independent_composition_every_prefix_and_boundary(index, risk):
    p = plan(index,'risk_rfa' if risk else 'erm_mean')
    previous = 0.
    for t in range(121):
        expected = {a: min(t,60)*independent_rdp(a,.05,p['z_early'])
                    + max(t-60,0)*independent_rdp(a,.05,p['z_late'])
                    + (t*a/(2*p['risk_z']**2) if risk else 0.) for a in range(2,65)}
        actual = prefix_ledger(p,t)
        for a in expected:
            assert math.isclose(actual['rdp'][a],expected[a],rel_tol=2e-12,abs_tol=2e-12)
        eps = min(v+math.log(1e5)/(a-1) for a,v in expected.items()) if t else 0.
        assert math.isclose(actual['epsilon'],eps,rel_tol=2e-12,abs_tol=2e-12)
        assert previous <= actual['epsilon'] <= 4.
        previous = actual['epsilon']
    assert math.isclose(previous,p['epsilon_realized'],abs_tol=1e-12)
    assert round_parameters(p,60) == dict(phase='early',z=p['z_early'],sigma=p['sigma_early'],eta=2.)
    assert round_parameters(p,61) == dict(phase='late',z=p['z_late'],sigma=p['sigma_late'],eta=.5)
    assert p['sigma_early'] == p['z_early']*4/240
    assert math.isclose(p['sigma_late'],p['z_late']*4/240,rel_tol=1e-15)
    assert canonical(p) == json.loads(json.dumps(p))


@pytest.mark.parametrize('risk', [False,True])
def test_constant_allocation_recovers_original_budget(risk):
    p = scalar_plan(1.,risk)
    original = ledger(2.,'risk_rfa' if risk else 'erm_mean')
    assert math.isclose(p['sigma_early'],original['gradient_std'],rel_tol=2e-12)
    assert math.isclose(p['epsilon_realized'],original['epsilon_realized'],rel_tol=2e-12)
    if risk:
        assert p['risk_z']/4800 == original['risk_std']


def test_only_public_choices_and_valid_rounds():
    for k in (0,8,12,16,True,7.):
        with pytest.raises(ValueError):
            plan(k,'risk_rfa')
    with pytest.raises(ValueError):
        plan(7,'unregistered_method')
    p = plan(7,'erm_mean')
    for t in (-1,0,121,1.,True):
        with pytest.raises(ValueError):
            round_parameters(p,t)
    for t in (-1,121,True):
        with pytest.raises(ValueError):
            prefix_ledger(p,t)


def test_gate_not_average_only():
    good = dict(accuracy_pct=-.8,worst20_pct=1.2,gap_best20_worst20_pp=-1.,variance_pp2=-2.)
    assert contrast_passes([good,good])
    assert not contrast_passes([good,dict(good,worst20_pct=.99)])
    assert not contrast_passes([good,dict(good,accuracy_pct=-1.01)])
    assert not contrast_passes([good,dict(good,variance_pp2=3.)])
    with pytest.raises(ValueError):
        contrast_passes([good])


def test_mps_noise_pairing_and_unchanged_mean_step():
    require_mps()
    x = torch.zeros((10,19),device='mps')
    a,b = plan(7,'risk_mean'),plan(13,'risk_mean')
    n1 = release(x,noise_std=a['sigma_early'],seed=777)
    n2 = release(x,noise_std=b['sigma_early'],seed=777)
    assert torch.allclose(n1/a['sigma_early'],n2/b['sigma_early'],rtol=1e-6,atol=1e-6)
    rr = torch.linspace(0,1,10,device='mps')
    weights = 1+2*(rr/.5).clamp(max=1); weights /= weights.sum()
    for t in (60,61):
        step,diag = aggregate(n1,rr,kind='risk_mean',round_number=t,horizon=120)
        eta = 2. if t == 60 else .5
        assert diag['eta'] == eta
        assert torch.allclose(step,eta*(weights[:,None]*n1).sum(0),atol=1e-8,rtol=1e-6)
        assert step.device.type == 'mps'
