import torch
from scripts.analyze_v28_terminal_aggregation_counterfactual import controls, objective
from privacy.fair_objective import require_mps


def test_equal_messages_give_same_control_and_no_mutation():
    require_mps()
    x = torch.tensor([[.25, -.5]]*10, device='mps')
    w = torch.ones(10, device='mps')/10
    values, _ = controls(x, w)
    torch.testing.assert_close(values['mean'], x[0], rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(values['rfa'], values['mean'], rtol=2e-5, atol=1e-6)
    assert torch.equal(x, torch.tensor([[.25, -.5]]*10, device='mps'))


def test_robustification_is_not_an_alias_for_the_mean():
    require_mps()
    x = torch.tensor([[0.]]*8+[[10.]]*2, device='mps')
    w = torch.ones(10, device='mps')/10
    values, _ = controls(x, w)
    assert abs(float(values['mean'])-2) < 1e-6
    assert abs(float(values['rfa'])) < 1e-3


def test_piecewise_objective_matches_integrated_risk_weight():
    require_mps()
    risks = torch.tensor([0., .25, .5, .75, 1.], device='mps')
    expected = torch.tensor([0., .375, 1., 1.75, 2.5], device='mps')
    assert abs(objective(risks, 'risk_rfa')-float(expected.mean())) < 1e-6
    assert abs(objective(risks, 'erm_mean')-.5) < 1e-6
