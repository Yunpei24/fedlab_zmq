import math
import torch
from privacy.fair_objective import require_mps,release,per_example
from privacy.clipped_recursive_private_gradient import recursive_release
from privacy.recursive_public_calibration import largest_increment_ratio
from scripts import run_recursive_private_risk_calibration_v19 as run


def sample():
    require_mps();torch.mps.manual_seed(819)
    x=.1*torch.randn(8,5,device='mps');y=x+.03*torch.randn_like(x)
    return x,y,torch.randn(5,device='mps')


def test_initial_and_theta_one_reproduce_fresh_mechanism_exactly():
    x,y,m=sample()
    p=largest_increment_ratio(C=2.)
    a,d=recursive_release(x,None,None,C=2.,D=p['D'],theta=p['theta'],base_noise_std=.03,seed=3)
    torch.testing.assert_close(a,release(x.mean(0),noise_std=.03,seed=3),rtol=0,atol=0)
    b,diag=recursive_release(x,y,m,C=2.,D=p['D'],theta=1.,base_noise_std=.03,seed=3)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert d['initial'] and not diag['initial']


def test_same_noise_multiplier_and_private_memory_postprocessing():
    x,y,m=sample();p=largest_increment_ratio(C=2.)
    a,d=recursive_release(x,y,m,C=2.,D=p['D'],theta=p['theta'],base_noise_std=.03,seed=4)
    b,_=recursive_release(x,y,m+.2,C=2.,D=p['D'],theta=p['theta'],base_noise_std=.03,seed=4)
    torch.testing.assert_close(b-a,torch.full_like(a,(1-p['theta'])*.2),rtol=1e-5,atol=1e-6)
    assert math.isclose(d['noise_std']/d['query_sensitivity'],.03/(4/8),rel_tol=1e-12)
    assert isinstance(d['clipping_increment_count'],int)


def test_two_model_batch_reconstruction_and_checkpoint_history():
    require_mps();torch.mps.manual_seed(820)
    model=torch.nn.Linear(3,2).to('mps');previous=torch.nn.Linear(3,2).to('mps')
    previous.load_state_dict(model.state_dict());x=torch.randn(10,3,device='mps');y=torch.arange(10,device='mps')%2
    _,g0,_,_=per_example(model,x,y,clip_norm=2.)
    pp=largest_increment_ratio(C=2.)
    mem,_=recursive_release(g0,None,None,C=2.,D=pp['D'],theta=pp['theta'],base_noise_std=.03,seed=5)
    with torch.no_grad():
        for p in model.parameters():p.add_(.01)
    _,g1,_,_=per_example(model,x,y,clip_norm=2.)
    old=per_example(previous,x,y,clip_norm=2.)[1]
    torch.testing.assert_close(g0,old,rtol=0,atol=0)
    a,_=recursive_release(g1,old,mem,C=2.,D=pp['D'],theta=pp['theta'],base_noise_std=.03,seed=6)
    restored=mem.detach().cpu().to('mps')
    b,_=recursive_release(g1,old,restored,C=2.,D=pp['D'],theta=pp['theta'],base_noise_std=.03,seed=6)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_registration_and_explicit_cost_and_no_test_access():
    m,profile,stamp=run.inputs()
    assert len(run.jobs(m))==16 and set(j['seed'] for j in run.jobs(m))=={170501,170502}
    assert profile['local_optimizer_steps']==0 and profile['model']=='lenet5_tanh'
    assert not m['test_evaluated'] and m['minimum_worst20_gain_pp']==m['maximum_accuracy_loss_pp']==1
    assert not any('17060' in path for path in stamp if path.endswith('checkpoint.pt'))
