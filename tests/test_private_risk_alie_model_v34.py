import math
import pytest
import torch
from scripts.diagnose_private_risk_alie_model_v34 import phi, contrast, validation_gradient


def test_phi_derivative_and_continuity():
    assert phi(.5) == 1.
    for r in (.1, .4, .5, .8):
        h = 1e-6
        assert math.isclose((phi(r+h)-phi(r-h))/(2*h), 1+4*min(r,.5), abs_tol=3e-6)
    with pytest.raises(ValueError):
        phi(float('nan'))


def test_factorial_signs_and_no_imputation():
    c = contrast({'00':1., '10':3., '01':4., '11':8.})
    assert c == dict(gradient_original_report=2., report_original_gradient=3.,
                     report_forged_gradient=5., gradient_forged_report=4., interaction=2.)
    with pytest.raises(ValueError):
        contrast({'00':1.})


def test_oracle_gradient_is_gradient_of_mean_transformed_client_risks_on_mps():
    from privacy.fair_objective import require_mps, losses
    require_mps()
    torch.manual_seed(9134)
    model = torch.nn.Linear(2, 10).to('mps').eval()
    x = torch.randn(12000, 2, device='mps')
    y = torch.arange(12000, device='mps') % 10
    ids = list(torch.arange(12000, device='mps').split(1200))
    data = dict(x=x, y=y, val=ids)
    with torch.no_grad():
        rs = [float(losses(model(x[ix]), y[ix], 'brier').mean()) for ix in ids]
    ev = dict(clients=[dict(brier_loss=r) for r in rs])
    got = validation_gradient(model, data, ev)
    risks = torch.stack([losses(model(x[ix]), y[ix], 'brier').mean() for ix in ids[2:]])
    obj = torch.where(risks <= .5, risks+2*risks.square(), 3*risks-.5).mean()
    ref = torch.cat([g.flatten() for g in torch.autograd.grad(obj, list(model.parameters()))])
    torch.testing.assert_close(got, ref, rtol=2e-5, atol=2e-6)
