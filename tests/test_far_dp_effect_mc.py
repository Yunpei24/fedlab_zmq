"""MPS numerical audit; host tests only handle scalar metadata/statistics."""
import math
from pathlib import Path
import statistics

import pytest
import torch
import yaml

from privacy.far_dp_effect_mc import (
    stream_seed, sampled_indices, gradient_release, far_aggregate,
    privacy_plan, nested_summary, mps_rng,
)


def test_streams_separate_levels_and_have_no_arm_argument():
    seed = stream_seed('test-only-entropy', 42, 0, 'batch', 12, 4)
    assert seed == stream_seed('test-only-entropy', 42, 0, 'batch', 12, 4)
    assert len({seed, stream_seed('test-only-entropy', 42, 1, 'batch', 12, 4),
                stream_seed('test-only-entropy', 42, 0, 'noise', 12, 4),
                stream_seed('test-only-entropy', 72, 0, 'batch', 12, 4)}) == 4


def test_nested_sd_is_not_sem_and_no_pseudoreplication():
    r = nested_summary([[0., 2., 4.], [10., 12., 14.]])
    assert r['mean'] == 7
    assert r['outer_mean_sd'] == statistics.stdev([2., 12.])
    assert r['within_mc_rms_sd'] == 2
    assert r['outer_count'] == 2 and r['mc_count'] == 3
    with pytest.raises(ValueError):
        nested_summary([[1, 2], [3]])


def test_privacy_scale_and_strict_budget():
    args = dict(epsilon=4., delta=1e-5, n_local=4800, batch=300, rounds=100)
    p = privacy_plan(**args, clip=2.)
    p8 = privacy_plan(**args, clip=8.)
    assert p['epsilon_bound'] <= 4
    assert p['std'] == pytest.approx(p['sigma_C']*2/300)
    assert p['z_sensitivity'] == p['sigma_C']/2
    assert p8['sigma_C'] == p['sigma_C']
    assert p8['std'] == 4*p['std']
    long = privacy_plan(**{**args, 'rounds': 600}, clip=2.)
    assert long['sigma_C'] > p['sigma_C']


def test_matrix_is_explicit_and_not_accidentally_launchable():
    p = Path(__file__).resolve().parents[1]/'configs/ldp_gradient_far/far_dp_effect_mc_v1.yaml'
    m = yaml.safe_load(p.read_text())
    assert len(set(m['outer_seeds'])) == 10
    assert not set(m['outer_seeds']) & set(m['calibration_seeds'])
    assert m['alpha_values'] == sorted(set(m['alpha_values']))
    assert set(m['alpha_values']) == {-x for x in m['alpha_values']}
    assert len(m['alpha_values']) == 15
    assert m['server_clipping_modes'] == ['off', 'on']
    assert m['primary_rounds'] is None
    assert not m['execution']['automatic_full_grid_launch']


@pytest.fixture
def toy():
    if not torch.backends.mps.is_available():
        pytest.fail('MPS unavailable: rerun outside sandbox, no CPU substitution')
    with mps_rng(123):
        model = torch.nn.Linear(3, 2, bias=True, device='mps').eval()
        x = torch.randn((8, 3), device='mps')
        y = torch.arange(8, device='mps') % 2
    return model, x, y


def test_sampling_exact_unique_and_rng_restoration(toy):
    before = torch.mps.get_rng_state()
    a = sampled_indices(100, 25, 442)
    assert len(a.unique()) == 25
    assert torch.equal(a, sampled_indices(100, 25, 442))
    assert torch.equal(before, torch.mps.get_rng_state())
    assert not torch.equal(a, sampled_indices(100, 25, 443))


def test_unclipped_release_equals_full_batch_autograd_no_model_step(toy):
    model, x, y = toy
    before = [p.detach().clone() for p in model.parameters()]
    expected = torch.autograd.grad(torch.nn.functional.cross_entropy(model(x), y), tuple(model.parameters()))
    expected = torch.cat([g.flatten() for g in expected])
    actual, _ = gradient_release(model, x, y, local_clip=None, noise_std=0, noise_seed=4)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)
    assert all(torch.equal(p, old) for p, old in zip(model.parameters(), before))
    with pytest.raises(ValueError):
        gradient_release(model, x, y, local_clip=None, noise_std=.1, noise_seed=4)


def test_per_example_clip_and_noise_after_sum_microbatch_invariance(toy):
    model, x, y = toy
    c = .15
    hand = []
    for j in range(len(x)):
        grad = torch.autograd.grad(torch.nn.functional.cross_entropy(model(x[j:j+1]), y[j:j+1]),
                                   tuple(model.parameters()))
        g = torch.cat([v.flatten() for v in grad])
        hand.append(g * (c/g.norm().clamp_min(1e-12)).clamp(max=1))
    expected = torch.stack(hand).mean(0)
    out, d = gradient_release(model, x, y, local_clip=c, noise_std=.03, noise_seed=27, microbatch=3)
    other, _ = gradient_release(model, x, y, local_clip=c, noise_std=.03, noise_seed=27, microbatch=8)
    assert torch.allclose(d['clean_oracle'], expected, atol=2e-6, rtol=2e-5)
    assert torch.allclose(out, expected + .03*d['noise_oracle'], atol=2e-6, rtol=2e-5)
    assert torch.allclose(out, other, atol=2e-6, rtol=2e-5)
    assert float(d['clean_oracle'].norm()) <= c+1e-6


@pytest.mark.parametrize('reference', ['rfa', 'coordinate_median', 'trimmed_mean', 'centered_clipping'])
def test_alpha_zero_exact_mean_for_every_reference(toy, reference):
    x = torch.arange(30, device='mps', dtype=torch.float32).reshape(10, 3)/10
    a, d = far_aggregate(x, alpha=0, reference=reference)
    assert torch.allclose(a, x.mean(0), atol=1e-6)
    assert torch.allclose(d['weights'], torch.full((10,), .1, device='mps'))
    assert d['server_clip_fraction'] == 0


def test_server_clip_off_on_and_raw_distance_formula(toy):
    x = torch.arange(30, device='mps', dtype=torch.float32).reshape(10, 3)/10
    a, d = far_aggregate(x, alpha=3.4, reference='coordinate_median', server_clip=.5)
    expected_x = x * (.5/x.norm(dim=1)).clamp(max=1)[:, None]
    expected_ref = torch.quantile(expected_x, .5, dim=0, interpolation='midpoint')
    lam = torch.softmax(3.4*(expected_x-expected_ref).norm(dim=1), dim=0)
    assert torch.allclose(a, (lam[:, None]*expected_x).sum(0), atol=1e-6)
    b, _ = far_aggregate(x, alpha=3.4, reference='coordinate_median', server_clip=None)
    assert not torch.allclose(a, b)
    # Radius larger than every norm is a genuine no-op.
    huge, _ = far_aggregate(x, alpha=3.4, reference='coordinate_median', server_clip=100)
    assert torch.allclose(b, huge, atol=1e-6)
