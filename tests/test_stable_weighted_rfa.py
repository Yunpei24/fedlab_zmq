import math
import pytest
import torch
from privacy.fair_objective import require_mps
from privacy.stable_weighted_rfa import weighted_rfa, stable_norm


@pytest.fixture(scope='module', autouse=True)
def mps():
    require_mps()


def test_stable_norm_large_small_zero_and_validation():
    x = torch.tensor([[3e30, 4e30], [0., 0.], [3e-20, 4e-20]], device='mps')
    n = stable_norm(x, 1)
    assert float(n[0]) == pytest.approx(5e30, rel=1e-6)
    assert float(n[1]) == 0.
    assert float(n[2]) == pytest.approx(5e-20, rel=1e-6, abs=0.)
    with pytest.raises(ValueError):
        weighted_rfa(x, torch.ones(3, device='mps'))
    with pytest.raises(ValueError):
        weighted_rfa(torch.zeros(2, 1, device='mps'), torch.tensor([0., 1.], device='mps'))


@pytest.mark.parametrize('scale', [1., 1e3, 1e8, 1e18, 1e30])
def test_large_minority_does_not_poison_initialization(scale):
    torch.mps.manual_seed(171015)
    h = .01 * torch.randn(8, 16, device='mps')
    x = torch.cat([h, torch.full((2, 16), scale, device='mps')])
    w = torch.tensor([1.] * 8 + [3.] * 2, device='mps')
    z, d = weighted_rfa(x, w)
    assert bool(torch.isfinite(z).all())
    assert math.isfinite(d['unsmoothed_objective_gap_upper'])
    assert float(stable_norm(z - h.mean(0))) < .08
    assert d['unsmoothed_objective_gap_upper'] < .001
    assert d['localized_mass'] > .5
    assert d['stationary_reconstruction_error'] < .001


def test_convexity_bound_against_grid_and_equivariance():
    x = torch.tensor([[-2.], [-.1], [0.], [.2], [4.]], device='mps')
    w = torch.tensor([1., 2., 2., 1., 1.], device='mps')
    z, d = weighted_rfa(x, w, iterations=8)
    grid = torch.linspace(-2, 4, 12001, device='mps')
    cost = ((x[:, 0, None] - grid[None, :]).abs() * (w/w.sum())[:, None]).sum(0)
    value = float(((x[:, 0] - z[0]).abs() * w/w.sum()).sum())
    assert value - float(cost.min()) <= d['unsmoothed_objective_gap_upper'] + 1e-5
    p = torch.tensor([4, 1, 0, 3, 2], device='mps')
    zz, _ = weighted_rfa(x[p] + 2, w[p], iterations=8)
    torch.testing.assert_close(zz, z + 2, atol=1e-6, rtol=1e-5)


def test_honest_minority_fairness_failure_is_not_hidden_by_numerical_fix():
    x = torch.tensor([[.25]] * 8 + [[-1.]] * 2, device='mps')
    w = torch.tensor([1.4] * 8 + [3.] * 2, device='mps')
    z, _ = weighted_rfa(x, w)
    weighted_mean = (x[:, 0] * w).sum() / w.sum()
    assert float(z[0]) > .24
    assert float(weighted_mean) < 0
