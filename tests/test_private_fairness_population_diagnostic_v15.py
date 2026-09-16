import torch
from privacy.fair_objective import require_mps,losses,per_example
from privacy.capped_private_risk import potential
from scripts import run_private_fairness_population_diagnostic_v15 as run


def test_population_fair_direction_matches_autograd_objective():
    require_mps();torch.mps.manual_seed(234)
    model=torch.nn.Linear(3,2).to('mps')
    x=torch.randn(8,3,device='mps');y=torch.arange(8,device='mps')%2
    risk=[];grad=[]
    for ix in (slice(0,4),slice(4,8)):
        rr,_,_,g=per_example(model,x[ix],y[ix],clip_norm=2.)
        risk.append(rr.mean());grad.append(g)
    lam,GJ,target=run.fair_directions(torch.stack(risk),torch.stack(grad))
    model.zero_grad()
    full=torch.stack([losses(model(x[:4]),y[:4]).mean(),losses(model(x[4:]),y[4:]).mean()])
    potential(full,.5).mean().backward()
    exact=torch.cat([p.grad.flatten() for p in model.parameters()])
    torch.testing.assert_close(GJ,exact,rtol=2e-5,atol=2e-6)
    assert float(torch.dot(GJ,target))>=0 and abs(float(lam.sum())-1)<1e-6


def test_only_calibration_checkpoints_and_fixed_number_of_interventions():
    profile,stamp=run.inputs()
    assert run.SEEDS==[170501,170502]
    assert 2*(len(run.POPULATION)+4*len(run.SAMPLED))==40
    assert all('17060' not in p for p in stamp if p.endswith('checkpoint.pt'))
    assert profile['model']=='lenet5_tanh'
