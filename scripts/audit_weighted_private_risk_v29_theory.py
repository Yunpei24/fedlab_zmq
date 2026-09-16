"""MPS algebra checks only; no training, tuning, gate, or empirical validation.

Proofs are in Weighted_Private_Risk_V29_Theory_Bridge.md. Finite checks are
regression tests, not substitutes for proofs. Frozen campaign sources are read.
"""
import itertools
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
import torch
from privacy.fair_objective import require_mps
from scripts import run_fair_objective_screen as base

DEST = ROOT/'output/analysis/Weighted_Private_Risk_V29_Theory_Checks.json'
NOTE = ROOT/'output/analysis/Weighted_Private_Risk_V29_Theory_Bridge.md'


def close(a, b, tol=3e-5):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, abs_tol=tol, rel_tol=tol), (a, b)


def main():
    require_mps()
    assert os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] == '0'
    records = []
    for n, maximum in ((8, 4/3), (10, 85/64)):
        a = torch.tensor(list(itertools.product((1., 3.), repeat=n)), device='mps')
        p = a/a.sum(dim=1, keepdim=True)
        concentration = n*p.square().sum(dim=1)
        close(float(concentration.max()), maximum)
        assert float(concentration.min()) >= 1-2e-6
        assert float(concentration.max()) <= 4/3+2e-6
        close(float(p.max()), 3/(n+2))
        records.append(dict(check='all_coefficient_vertices', clients=n,
                            vertices=len(a), observed_max=float(concentration.max()),
                            theoretical_max=maximum, passed=True))

    # A symmetric finite ensemble has exactly covariance s² I_(nd).
    # It is not Gaussian and not a new DP experiment; the covariance identity
    # used in the note depends on its moments, not on its distribution.
    n, d, s = 10, 4, .07
    coeff = torch.tensor([3., 3., 3., 1., 1., 1., 1., 1., 1., 1.], device='mps')
    p = coeff/coeff.sum()
    basis = torch.eye(n*d, device='mps').reshape(n*d, n, d)
    noise = torch.cat((basis, -basis), dim=0)*(s*math.sqrt(n*d))
    weighted_noise = (noise*p[None, :, None]).sum(dim=1)
    observed = float(weighted_noise.square().sum(dim=1).mean())
    target = float(d*s*s*p.square().sum())
    close(observed, target, 2e-6)
    assert float(weighted_noise.mean(dim=0).abs().max()) < 1e-7
    # Fixed clipping and risk biases are not separately squared: the cross
    # term between the two biases remains in the formula.
    g = torch.arange(n*d, device='mps', dtype=torch.float32).reshape(n, d)/40-.4
    c = g*torch.minimum(torch.ones(n, device='mps'), .25/torch.linalg.vector_norm(g, dim=1))[:, None]
    true_coeff = torch.linspace(1., 3., n, device='mps')
    true_p = true_coeff/true_coeff.sum()
    honest_target = (true_p[:, None]*g).sum(dim=0)
    bias_clip = (true_p[:, None]*(c-g)).sum(dim=0)
    bias_risk = ((p-true_p)[:, None]*c).sum(dim=0)
    mean = (p[None, :, None]*(c[None, :, :]+noise)).sum(dim=1)
    observed_error = float((mean-honest_target).square().sum(dim=1).mean())
    target_error = float((bias_clip+bias_risk).square().sum())+target
    close(observed_error, target_error, 2e-6)
    records.append(dict(check='conditional_variance_and_bias_identity',
        ensemble='symmetric coordinate ensemble, same second moments; not DP Gaussian releases',
        realizations=len(noise), observed_noise_mse=observed, expected_noise_mse=target,
        observed_total_mse=observed_error, expected_total_mse=target_error, passed=True))

    # Scalar counterexample to dropping dependence for arbitrary adaptive weights.
    z = torch.tensor(list(itertools.product((-1., 1.), repeat=2)), device='mps')
    a = torch.ones_like(z)
    a[:, 0] = torch.where(z[:, 0] > 0, 3., 1.)
    adaptive_p = a/a.sum(dim=1, keepdim=True)
    adaptive_mean = float((adaptive_p*z).sum(dim=1).mean())
    close(adaptive_mean, .125, 2e-6)
    records.append(dict(check='arbitrary_same_noise_weights_not_centered',
        expected_message_mean=adaptive_mean, is_rfa_counterexample=False,
        is_real_training_result=False, passed=True))

    # Exhaustive mass inequality for the fixed eight-honest/two-Byzantine setup.
    a = torch.tensor(list(itertools.product((1., 3.), repeat=10)), device='mps')
    lam = a/a.sum(dim=1, keepdim=True)
    beta = lam[:, :2].sum(dim=1)
    close(float(beta.max()), 3/7)
    mass = 1-beta
    close(float((2*mass/(2*mass-1)).max()), 8.)
    close(float((1/(2*mass-1)).max()), 7.)
    records.append(dict(check='all_mass_vertices', vertices=len(a), byzantines=2,
        maximum_byzantine_mass=float(beta.max()), dispersion_factor=8.,
        numerical_error_factor=7., passed=True))

    # Check the pathwise normalized-objective descent inequality on convex
    # quadratics, deliberately using arbitrary biased errors. No training data.
    state = torch.mps.get_rng_state()
    try:
        torch.mps.manual_seed(190901)
        w = torch.randn((256, 4), device='mps')
        error = torch.randn((256, 4), device='mps')+.3
        mean_coeff = 1+2*torch.rand(256, device='mps')
    finally:
        torch.mps.set_rng_state(state)
    L = 1.7
    eta = torch.linspace(.01, 1/L, 256, device='mps')
    gradient = L*w
    agg = gradient/mean_coeff[:, None]+error
    nxt = w-eta[:, None]*agg
    actual = .5*L*nxt.square().sum(dim=1)
    bound = .5*L*w.square().sum(dim=1)-eta/(2*mean_coeff)*gradient.square().sum(dim=1)+eta*mean_coeff/2*error.square().sum(dim=1)
    assert float((actual-bound).max()) < 2e-5
    loose_bound = .5*L*w.square().sum(dim=1)-eta/6*gradient.square().sum(dim=1)+3*eta/2*error.square().sum(dim=1)
    assert float((bound-loose_bound).max()) < 2e-5
    records.append(dict(check='normalized_objective_quadratic_descent', cases=256,
        public_test_seed=190901, max_slack_violation=float((actual-bound).max()), passed=True,
        certifies_neural_training_step_sizes=False))

    folder = ROOT/'results/ldp_gradient_far/full_population_private_risk_confirmation_v29'
    manifest = json.loads((folder/'manifest.json').read_text())
    base.verify_stamp(manifest['source_stamp'])
    # Only the predeclared release scales are read, never accuracy or losses.
    sources = [folder/'seed180701__b4800__risk_rfa/public_protocol.json',
               folder/'seed180701__b4800__erm_mean/public_protocol.json']
    scales = [json.loads(p.read_text())['privacy']['gradient_std'] for p in sources]
    model = base.new_model(manifest['profile'], 190901)
    dimension = sum(p.numel() for p in model.parameters())
    assert dimension == 61706
    del model
    v = dimension*scales[0]**2/10
    base.save(DEST, dict(algebra_checks_passed=True, device='mps', fallback=False,
        torch_version=str(torch.__version__), records=records,
        public_variance_example=dict(dimension=dimension, clients=10,
            risk_gradient_std=scales[0], erm_gradient_std=scales[1],
            uniform_variance_at_risk_sigma=v, bounded_weight_variance_upper=v*85/64,
            ratio_to_erm_variance_upper=(scales[0]/scales[1])**2*85/64),
        source_stamp={str(p.relative_to(ROOT)):base.digest(p) for p in [Path(__file__), NOTE, *sources]},
        empirical_seeds_added=0, training_runs_added=0, confirmation_gate_evaluated=False,
        universal_proofs_replaced_by_checks=False, robustness_validated=False,
        neural_step_size_certified=False, global_validation=False))
    print(json.dumps(dict(algebra_checks_passed=True, groups=len(records), device='mps',
                         training_runs_added=0, global_validation=False)), flush=True)


if __name__ == '__main__':
    main()
