import math
import torch
from privacy.fair_objective import require_mps
from scripts.audit_v28_terminal_counterfactual_precision import scalar_potential, squared_brier


def test_one_hot_matches_polynomial_brier_on_mps():
    require_mps()
    logits = torch.tensor([[1., -2., .1], [3., 1., -4.]], device='mps')
    y = torch.tensor([1, 0], device='mps')
    p = logits.softmax(1)
    polynomial = .5*(p.square().sum(1)-2*p.gather(1, y[:, None]).squeeze(1)+1)
    torch.testing.assert_close(squared_brier(logits, y), polynomial, rtol=1e-5, atol=1e-7)


def test_fair_potential_continuity_and_risk_coefficient():
    for r in (0., .1, .25, .5, .75, 1.):
        assert scalar_potential(r, False) == r
        h = 1e-6
        derivative = (scalar_potential(r+h, True)-scalar_potential(r-h, True))/(2*h)
        assert math.isclose(derivative, 1+2*min(r/.5, 1), abs_tol=2e-6)
