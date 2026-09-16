import math
from pathlib import Path
import pytest
import torch
from privacy.fair_objective import require_mps, losses
from privacy.capped_private_risk import potential
from privacy.scheduled_private_risk import aggregate as host_aggregate
from privacy import private_risk_factorial_diagnostic_v26 as m
from scripts import run_private_risk_factorial_diagnostic_v26 as run


def model_data():
    require_mps()
    torch.manual_seed(5123); torch.mps.manual_seed(5123)
    model = torch.nn.Linear(3, 2).to('mps').eval()
    x = torch.randn(12, 3, device='mps')
    y = torch.arange(12, device='mps') % 2
    ids = [torch.arange(0, 5, device='mps'), torch.arange(5, 12, device='mps')]
    return model, x, y, ids


def test_population_gradient_and_target_equal_autograd_full_objective():
    model, x, y, ids = model_data()
    risks, gradients = m.population(model, x, y, ids, with_gradient=True, block_size=3)
    full = torch.stack([losses(model(x[ii]), y[ii]).mean() for ii in ids])
    J = potential(full, .5).mean()
    exact = torch.cat([g.flatten() for g in torch.autograd.grad(J, tuple(model.parameters()))])
    target = m.targets(risks, gradients, gradients)
    torch.testing.assert_close(target['GJ'], exact, rtol=2e-5, atol=2e-6)
    r, no_gradient = m.population(model, x, y, ids, with_gradient=False, block_size=3)
    torch.testing.assert_close(r, risks, rtol=2e-6, atol=2e-7)
    assert no_gradient is None and all(p.grad is None for p in model.parameters())
    torch.testing.assert_close(target['h'], target['h_batch'])


def test_potential_derivative_and_junction():
    require_mps()
    r = torch.tensor([.1, .499, .5, .501, .9], device='mps', requires_grad=True)
    d = torch.autograd.grad(potential(r, .5).sum(), r)[0]
    torch.testing.assert_close(d, 1+2*(r/.5).clamp(max=1))
    assert float(potential(r, .5)[2].detach()) == 1.


def test_factorial_has_twelve_unique_cells_and_three_controls():
    ts = m.treatments()
    assert len(ts) == len({m.treatment_id(t) for t in ts}) == 12
    assert len(ts)*2*4 == 96
    with pytest.raises(ValueError):
        m.treatment_id(dict(weight='ema', message='private', aggregator='rfa'))


def test_host_step_bitwise_identical_and_factors_isolated():
    require_mps(); torch.mps.manual_seed(382)
    x = torch.randn(10, 17, device='mps')
    clean = .2*x
    reports = torch.linspace(.1, .8, 10, device='mps')
    raw = torch.linspace(.7, .1, 10, device='mps')
    t = dict(weight='private', message='private', aggregator='rfa')
    A, diag = m.aggregate(t, x, clean, reports, raw)
    step, original = host_aggregate(x, reports, kind='risk_rfa', round_number=30, horizon=120)
    assert torch.equal(2*A, step) and diag['solver'] == original['solver']
    uniform = dict(weight='uniform', message='private', aggregator='mean')
    u, _ = m.aggregate(uniform, x, clean, reports, raw)
    torch.testing.assert_close(u, x.mean(0))
    clean_case = dict(weight='uniform', message='clean_clipped_oracle', aggregator='mean')
    uc, _ = m.aggregate(clean_case, x, clean, reports, raw)
    torch.testing.assert_close(uc, clean.mean(0))
    # No private weights consult the raw-risk oracle.
    altered, _ = m.aggregate(t, x, clean, reports, 1-raw)
    assert torch.equal(A, altered)


def test_effect_signs_and_exact_finite_step_identity():
    require_mps()
    A = torch.tensor([2., 1.], device='mps')
    target = dict(GJ=torch.tensor([3., 4.], device='mps'), h=A, h_batch=A+1, J=5.)
    d = m.effects(A, target, eta=.5, J_after=6.)
    assert d['predicted_J_decrease'] == 5 and d['actual_J_decrease'] == -1
    assert d['finite_step_remainder'] == 6 and d['error_to_population_target_sq'] == 0
    assert d['error_to_clipped_batch_target_sq'] == 2
    assert d['predicted_J_decrease']-d['finite_step_remainder'] == d['actual_J_decrease']


def test_isolated_model_step_does_not_change_host():
    model, x, y, ids = model_data()
    before = run.cpu_state(model)
    probe = torch.nn.Linear(3, 2).to('mps').eval()
    probe.load_state_dict(before)
    direction = torch.ones(sum(p.numel() for p in probe.parameters()), device='mps')
    run.base.apply_gradient(probe, direction, .1)
    run.assert_model_equal(model, before)
    with pytest.raises(AssertionError):
        run.assert_model_equal(probe, before)
    probe.load_state_dict(before)
    run.assert_model_equal(probe, before)


def test_exact_replay_comparator_rejects_any_numeric_mismatch():
    run.same({'x': 1.}, {'x': 1.}, 'equal')
    with pytest.raises(AssertionError):
        run.same({'x': 1.}, {'x': 1.+1e-15}, 'not exact')


def test_inputs_frozen_calibration_only_and_no_test_eval():
    config, profile, stamp, source = run.inputs()
    assert config['seeds'] == [170501, 170502] and not config['privacy_protected']
    assert profile['rounds'] == 120 and 'v19' in str(source)
    # The transitive stamp also preserves earlier diagnostic artifacts. Only
    # these two V19 checkpoints are model inputs to this replay.
    paths = [p for p in stamp if p.endswith('checkpoint.pt') and str(source.relative_to(run.ROOT)) in p]
    assert set(paths) == {str((source/f'seed{s}__fresh__risk_rfa/checkpoint.pt').relative_to(run.ROOT))
                          for s in (170501, 170502)}
    text = Path(run.__file__).read_text()
    assert "evaluate(model, data, 'test')" not in text and "evaluate(probe, data, 'test')" not in text
    assert 'previous_model\']' not in text


def test_reject_training_models_and_cpu_tensors():
    model, x, y, ids = model_data()
    model.train()
    with pytest.raises(ValueError):
        m.population(model, x, y, ids, with_gradient=True)
    model.eval()
    with pytest.raises(ValueError):
        m.population(model, x.cpu(), y, ids, with_gradient=True)
