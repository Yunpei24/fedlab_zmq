"""MPS-only audit of the split risk/gradient lane before any training."""
import math
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import pytest
import torch
from privacy.fair_objective import require_mps, losses, wor_rdp, release, calibrate
from privacy.split_risk_gradient import plan, private_risk, risk_weights, weighted_rfa, aggregate


@pytest.fixture(scope='module',autouse=True)
def mps_required():require_mps()


def test_joint_accountant_spends_total_budget_and_composes_every_release():
    p=plan()
    values=[60*wor_rdp(a,.05,p['gradient_z'])+60*a/(2*p['risk_z']**2)
            +math.log(1e5)/(a-1) for a in range(2,65)]
    assert min(values)==pytest.approx(p['epsilon_realized'],abs=1e-12)
    assert 3.99999<p['epsilon_realized']<=4
    lower=[60*wor_rdp(a,.05,.999*p['gradient_z'])+60*a/(2*p['risk_z']**2)
           +math.log(1e5)/(a-1) for a in range(2,65)]
    assert min(lower)>4
    assert p['risk_releases']==p['gradient_releases']==60
    assert p['gradient_sensitivity']==2/240 and p['risk_sensitivity']==1/4800
    assert p['risk_std']==p['risk_z']/4800
    assert p['gradient_std']==p['gradient_z']*2/240
    baseline_z=calibrate(q=.05,steps=60,epsilon=4,delta=1e-5)
    assert baseline_z<p['gradient_z']<1.01*baseline_z
    assert all(p['rdp'][a]>60*wor_rdp(a,.05,p['gradient_z']) for a in p['rdp'])


@pytest.mark.parametrize('kw',[{'T':0},{'N':200},{'delta':0},{'epsilon_risk':.01},
    {'b':2.5},{'C':float('nan')},{'epsilon_risk':4}])
def test_invalid_privacy_plans_fail_fast(kw):
    with pytest.raises(ValueError):plan(**kw)


def test_risk_release_on_mps_is_bounded_sensitivity_one_over_N_and_restores_rng():
    model=torch.nn.Identity().to('mps')
    x=torch.tensor([[3.,-2.],[-1.,1.],[2.,0.],[0.,1.]],device='mps')
    y=torch.tensor([0,1,1,0],device='mps')
    state=torch.mps.get_rng_state().clone()
    report,raw=private_risk(model,x,y,noise_std=.1,seed=203,N=4,batch_size=2)
    assert torch.equal(state,torch.mps.get_rng_state())
    target=losses(x,y,'brier').mean()
    torch.testing.assert_close(raw,target)
    torch.testing.assert_close(report,release(target.reshape(1),noise_std=.1,seed=203).clamp(0,1).squeeze(0))
    xx=x.clone();yy=y.clone();xx[0]=torch.tensor([-40.,40.],device='mps');yy[0]=0
    _,changed=private_risk(model,xx,yy,noise_std=0,seed=11,N=4)
    assert abs(float(changed-raw))<=1/4+1e-7
    assert report.device.type=='mps' and 0<=float(report)<=1
    with pytest.raises(ValueError):private_risk(model,x,y,noise_std=.1,seed=1,N=3)


def test_risk_weights_bound_byzantine_mass_even_for_forged_reports():
    reports=torch.tensor([-1.]*8+[100.,100.],device='mps')
    w,mean=risk_weights(reports)
    assert float(w[-2:].sum())==pytest.approx(3/7,abs=2e-7)
    assert float(mean)==pytest.approx(1.4,abs=2e-7)
    assert float(w.sum())==pytest.approx(1,abs=1e-7)
    with pytest.raises(ValueError):risk_weights(torch.tensor([float('nan')],device='mps'))


@pytest.mark.parametrize('weights',[[.1,.2,.7],[1.,1.,1.]])
def test_weighted_rfa_gap_certificate_against_known_scalar_optimum(weights):
    x=torch.tensor([[-2.],[1.],[5.]],device='mps')
    w=torch.tensor(weights,device='mps');w/=w.sum()
    center,diag=weighted_rfa(x,w)
    # A scalar weighted median is one of these observed values; exact enumeration.
    costs=torch.stack([(w*torch.abs(x[:,0]-p)).sum() for p in x[:,0]])
    excess=float((w*torch.abs(x[:,0]-center[0])).sum()-costs.min())
    assert excess<=diag['unsmoothed_objective_gap_upper']+2e-6
    assert float(x.min())<=float(center)<=float(x.max())
    if weights[2]/sum(weights)>.5:assert float(center)>4.99


def test_rfa_translation_permutation_and_constant_inputs():
    x=torch.tensor([[0.,1.],[-2.,0.],[1.,.5],[2.,-1.]],device='mps')
    w=torch.tensor([.1,.2,.4,.3],device='mps')
    a,_=weighted_rfa(x,w)
    idx=torch.tensor([2,0,3,1],device='mps');shift=torch.tensor([1.,2.],device='mps')
    b,_=weighted_rfa(x[idx]+shift,w[idx])
    torch.testing.assert_close(b,a+shift,atol=2e-6,rtol=2e-5)
    same,diag=weighted_rfa(torch.ones(10,3,device='mps'),torch.ones(10,device='mps'))
    torch.testing.assert_close(same,torch.ones(3,device='mps'))
    assert diag['unsmoothed_objective_gap_upper']<1.01e-5


def test_matched_controls_have_identical_step_but_different_relative_weights():
    x=torch.tensor([[1.,0.],[0.,2.],[-.5,1.]],device='mps')
    reports=torch.tensor([.1,.7,.9],device='mps')
    fair,df=aggregate(x,reports,method='risk_mean')
    control,dc=aggregate(x,reports,method='matched_mean')
    a=1+2*reports
    torch.testing.assert_close(fair,(2/1.9)*(a[:,None]*x).mean(0))
    torch.testing.assert_close(control,(2/1.9)*a.mean()*x.mean(0))
    assert df['eta']==dc['eta']
    assert df['weights']!=dc['weights']
    for method in ('risk_rfa','matched_rfa'):
        _,d=aggregate(x,reports,method=method)
        assert d['eta']==df['eta']
    erm,_=aggregate(x,None,method='erm_full')
    torch.testing.assert_close(erm,2*x.mean(0))
    with pytest.raises(ValueError):aggregate(x,reports[:2],method='risk_mean')


def test_equal_risks_give_the_same_matched_and_weighted_aggregate():
    x=torch.tensor([[1.,0.],[0.,2.],[-.5,1.]],device='mps')
    reports=torch.full((3,),.5,device='mps')
    for suffix in ('mean','rfa'):
        a,_=aggregate(x,reports,method='risk_'+suffix)
        b,_=aggregate(x,reports,method='matched_'+suffix)
        torch.testing.assert_close(a,b)


def test_runner_no_raw_risk_in_server_and_calibration_scope():
    import inspect
    from scripts import run_split_risk_gradient_v5 as runner
    m=runner.config();jobs=runner.jobs(m)
    assert len(jobs)==10 and set(j['seed'] for j in jobs)=={170501,170502}
    assert not set(m['calibration_seeds'])&{170301,170302,170401,170402,170403,170404}
    assert m['screen']['all_seeds_required'] and not m['automatic_attacks']
    assert m['attacks']=='none' and not m['test_evaluated']
    src=inspect.getsource(runner.train)
    assert 'reports.append(noisy_risk)' in src and 'reports.append(raw_risk)' not in src
    assert "'test'" not in src
    p=runner.privacy(m,'risk_mean')
    assert runner.epsilon_at(p,60,m,'risk_mean')==pytest.approx(p['epsilon_realized'])
    assert runner.epsilon_at(p,30,m,'risk_mean')<4
    q=runner.privacy(m,'erm_full')
    assert q['risk_releases']==0 and q['gradient_std']<p['gradient_std']
    assert q['epsilon_realized']<=4
