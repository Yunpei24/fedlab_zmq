import ast
from pathlib import Path
import pytest
import torch
from metrics.fair_direction_audit import (flat_parameters, set_parameters, mean_brier_gradient,
    clip_rows, directional_metrics, actual_metrics)
from privacy.fair_objective import losses, require_mps
from scripts import run_fair_direction_diagnostic_v4 as run


@pytest.fixture(scope='module', autouse=True)
def only_mps():
    require_mps()


def test_plan_excludes_confirmation_seeds_and_test():
    m = run.config()
    assert len(run.states(m)) == 4 and len(m['conditions']) == 10
    assert {j['seed'] for j in run.states(m)} == {170301, 170302}
    assert not m['use_test'] and m['oracle_only']
    assert not m['automatic_training']
    assert 4*m['batch_replays_per_state']*len(m['conditions']) == 320
    tree = ast.parse(Path(run.__file__).read_text())
    calls = [x for x in ast.walk(tree) if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)
             and x.func.attr == 'evaluate']
    assert len(calls) == 2 and all(x.args[-1].value == 'val' for x in calls)


def tiny_model():
    torch.manual_seed(6201)
    return torch.nn.Sequential(torch.nn.Linear(3,4), torch.nn.Tanh(), torch.nn.Linear(4,2)).to('mps').eval()


def test_gradient_matches_full_autograd_and_public_step_restores():
    model = tiny_model()
    x = torch.randn(9,3,device='mps')
    y = torch.arange(9,device='mps')%2
    loss = losses(model(x),y).mean()
    expected = torch.cat([a.flatten() for a in torch.autograd.grad(loss, model.parameters())])
    risk, g = mean_brier_gradient(model,x,y,batch_size=4)
    assert risk == pytest.approx(float(loss.detach()), abs=1e-7)
    torch.testing.assert_close(g,expected,atol=1e-7,rtol=2e-5)
    w = flat_parameters(model)
    set_parameters(model,w-.1*g)
    after = float(losses(model(x),y).mean().detach())
    assert after < risk
    set_parameters(model,w)
    assert torch.equal(flat_parameters(model),w)


def test_finite_difference_direction_and_equitable_gradient():
    model = tiny_model()
    x = torch.randn(10,3,device='mps')
    y = torch.arange(10,device='mps')%2
    r,g = mean_brier_gradient(model,x,y)
    direction = g / torch.linalg.vector_norm(g)
    w = flat_parameters(model)
    h = .001
    set_parameters(model,w+h*direction)
    plus = float(losses(model(x),y).mean().detach())
    set_parameters(model,w-h*direction)
    minus = float(losses(model(x),y).mean().detach())
    set_parameters(model,w)
    assert (plus-minus)/(2*h) == pytest.approx(float(torch.dot(g,direction)), abs=3e-5)
    d = directional_metrics(direction,g[None,:],[r],[0])
    assert d['all_clients']['cosine'] == pytest.approx(1., abs=1e-5)
    assert d['equitable_J2']['predicted_loss_gain'] == pytest.approx((1+2*r)*float(torch.dot(g,direction)), rel=1e-5)


def test_clipping_is_samplewise_and_noise_pairing_is_exact():
    gs = [torch.tensor([[3.,4.],[0.,2.]],device='mps'),
          torch.tensor([[-1.,0.],[2.,1.]],device='mps')]
    risks = [torch.tensor([.1,.5],device='mps'),torch.tensor([.4,.7],device='mps')]
    noise = torch.tensor([[.3,-.1],[.2,.8]],device='mps')
    plans = {f'dp{C}_{n}':dict(std=.1*C*(4 if n=='fair' else 1)) for C in (1,2) for n in ('erm','fair')}
    a = run.matched_aggregates(risks,gs,noise,plans)
    for C in (1,2):
        assert float(torch.linalg.vector_norm(clip_rows(gs[0],C),dim=1).max()) <= C+1e-6
        for mode in ('erm','fair'):
            torch.testing.assert_close(a[f'dp{C}_{mode}']-a[f'clip{C}_{mode}'],
                plans[f'dp{C}_{mode}']['std']*noise.mean(0),atol=3e-7,rtol=1e-5)
    expected = torch.stack([(1+2*r.mean())*g.mean(0) for r,g in zip(risks,gs)]).mean(0)
    torch.testing.assert_close(a['raw_fair'],expected)


def test_zero_direction_no_nan():
    g = torch.ones(2,4,device='mps')
    d = directional_metrics(torch.zeros(4,device='mps'),g,[.2,.3],[0])
    assert d['all_clients']['predicted_loss_gain'] == 0
    assert d['all_clients']['cosine'] is None
    assert d['all_clients']['gain_per_unit_step'] is None


def test_cpu_rejected():
    with pytest.raises(ValueError):
        clip_rows(torch.ones(2,3),1.)
    with pytest.raises(ValueError):
        mean_brier_gradient(tiny_model(),torch.ones(2,3),torch.zeros(2,dtype=torch.long))


def test_actual_metrics_fixed_group_and_dynamic_tail_distinct():
    def make(risks,acc,worst):
        return dict(clients=[dict(brier_loss=r,accuracy=a) for r,a in zip(risks,acc)],
                    accuracy_pct=50,worst20_pct=worst,gap_best20_worst20_pp=10,variance_pp2=5)
    before = make([.3,.2],[.4,.6],40)
    after = make([.25,.1],[.5,.5],50)
    actual = actual_metrics(before,after,[0],dict(per_client_predicted_gain=[.06,.08]))
    assert actual['fixed_hard_loss_gain'] == pytest.approx(.05)
    assert actual['fixed_hard_accuracy_gain_pp'] == pytest.approx(10.)
    assert actual['per_client_first_order_remainder'] == pytest.approx([.01,-.02])


def test_source_checkpoints_and_frozen_manifests_intact():
    old, sources = run.verify_sources(run.config())
    assert len(sources) == 4 and old['config']['rounds'] == 60
    assert all(set(v) == {'checkpoint.pt','metrics.json'} for v in sources.values())


def test_privacy_plans_keep_the_cost_of_actual_query():
    old,_ = run.verify_sources(run.config())
    plans = run.privacy_plans(old,run.config())
    assert all(x['epsilon'] <= 4 and x['steps'] == 60 for x in plans.values())
    assert plans['dp1_fair']['std']/plans['dp1_erm']['std'] == pytest.approx(4.)
