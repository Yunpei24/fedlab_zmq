import math
import pytest
import torch
from privacy.fair_objective import require_mps
from privacy.capped_private_risk import weights,potential,aggregate
from privacy.split_risk_gradient_fixed_step import aggregate as old


@pytest.fixture(scope='module',autouse=True)
def mps():require_mps()


@pytest.mark.parametrize('scale',[.25,.5,1.])
def test_derivative_potential_bounded_weights_and_byzantine_mass(scale):
    r=torch.tensor([0.,.1,.2,.3,.8],device='mps',requires_grad=True)
    phi=potential(r,scale).sum();g=torch.autograd.grad(phi,r)[0]
    lam,a=weights(r,scale)
    torch.testing.assert_close(g,a)
    assert float(a.detach().min())>=1 and float(a.detach().max())<=3
    aa=weights(torch.tensor([0.]*8+[1.,1.],device='mps'),scale)[0]
    assert float(aa[-2:].sum())==pytest.approx(3/7,abs=1e-7)


def test_scale_one_reproduces_the_v6_direction_and_erm_uses_no_report():
    x=torch.tensor([[1.,0.],[0.,2.],[-.5,1.]],device='mps');r=torch.tensor([.1,.7,.9],device='mps')
    for kind in ('risk_mean','risk_rfa'):
        a,d=aggregate(x,r,kind=kind,scale=1)
        b,_=old(x,r,method=kind)
        torch.testing.assert_close(a,b)
        assert d['eta']==2 and abs(sum(d['stationary_weights'])-1)<1e-6
        if kind.endswith('rfa'):
            # Finite 40-step solve: certify the exact residual identity, not an
            # unjustified universal numerical accuracy threshold.
            lam=torch.tensor(d['objective_weights'],device='mps')
            center=a/2
            dist=((x-center).square().sum(1)+d['solver']['smoothing']**2).sqrt()
            expected=d['solver']['residual_norm']/float((lam/dist).sum())
            assert d['stationary_reconstruction_error']==pytest.approx(expected,abs=2e-7,rel=1e-3)
    a,_=aggregate(x,None,kind='erm_mean');torch.testing.assert_close(a,2*x.mean(0))
    a,_=aggregate(x,None,kind='erm_rfa');assert bool(torch.isfinite(a).all())


def test_all_saturated_risks_become_uniform_without_illegal_gradient_access():
    w,a=weights(torch.tensor([.4,.6,.8],device='mps'),.25)
    torch.testing.assert_close(w,torch.ones(3,device='mps')/3)
    assert float(a.max())==3
    with pytest.raises(ValueError):weights(torch.tensor([.2],device='mps'),0)


def fake_rows(m):
    from scripts.run_capped_private_risk_calibration_v7 import jobs
    return [dict(job=j,final=dict(validation=dict(accuracy_pct=70.,worst20_pct=50.))) for j in jobs(m)]


def test_grid_privacy_does_not_depend_on_risk_scale_and_strong_controls_remain():
    from scripts import run_capped_private_risk_calibration_v7 as run
    m=run.config();assert len(run.jobs(m))==24 and len(run.arms(m))==12
    for C in (1,2):
        p=run.privacy(m,f'risk_mean_C{C}_scale0.5')
        assert p==run.privacy(m,f'risk_rfa_C{C}_scale0.25')
        assert p['epsilon_realized']<=4
        base=run.privacy(m,f'erm_mean_C{C}');assert base==run.privacy(m,f'erm_rfa_C{C}')
        assert base['gradient_std']<p['gradient_std']
    assert run.privacy(m,'risk_rfa_C2_scale0.5')['gradient_std']==2*run.privacy(m,'risk_rfa_C1_scale0.5')['gradient_std']
    assert set(m['screen']['controls'])=={'erm_mean_C1','erm_mean_C2','erm_rfa_C1','erm_rfa_C2'}


def test_no_cherry_picking_across_seeds_and_deterministic_selection():
    from scripts import run_capped_private_risk_calibration_v7 as run
    m=run.config();rows=fake_rows(m)
    # A large average benefit must not hide one seed missing the 1-point target.
    for r in rows:
        if r['job']['arm'].startswith('risk_'):
            r['final']['validation']['worst20_pct']=50.5 if r['job']['seed']==170501 else 60.
    assert run.decide(m,rows)['selected_robust_candidate'] is None
    for r in rows:
        if r['job']['arm'].startswith('risk_'):r['final']['validation']['worst20_pct']=52.
    assert run.decide(m,rows)['selected_robust_candidate']=='risk_rfa_C1_scale0.5'
    for r in rows:
        if r['job']['arm']=='erm_mean_C2':r['final']['validation']['accuracy_pct']=73.
    assert run.decide(m,rows)['selected_robust_candidate'] is None
