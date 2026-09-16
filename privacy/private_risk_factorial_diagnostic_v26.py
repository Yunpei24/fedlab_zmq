"""Oracle-only causal probes; never a private/deployable training mechanism.

All vector work stays on MPS. Population gradients are ordinary gradients of
the full empirical risk, accumulated without a per-example gradient matrix.
"""
from itertools import product
import math
import torch
from privacy.fair_objective import require_mps, losses
from privacy.capped_private_risk import weights, potential
from privacy.stable_weighted_rfa import weighted_rfa

WEIGHTS = ('private', 'oracle', 'uniform')
MESSAGES = ('private', 'clean_clipped_oracle')
AGGREGATORS = ('mean', 'rfa')


def treatments():
    return [dict(weight=w, message=m, aggregator=a)
            for w, m, a in product(WEIGHTS, MESSAGES, AGGREGATORS)]


def treatment_id(t):
    if t not in treatments():
        raise ValueError('Unknown frozen treatment')
    return f"{t['weight']}__{t['message']}__{t['aggregator']}"


def _validate(model, x, y, client_ids, block_size):
    require_mps()
    if (x.device.type != 'mps' or y.device.type != 'mps' or not client_ids
            or type(block_size) is not int or block_size < 1
            or any(len(ids) == 0 or ids.device.type != 'mps' for ids in client_ids)
            or any(p.device.type != 'mps' for p in model.parameters())):
        raise ValueError('Nonempty MPS population and positive block size required')
    if model.training or any(isinstance(m, (torch.nn.modules.batchnorm._BatchNorm,
                                           torch.nn.modules.dropout._DropoutNd))
                             for m in model.modules()):
        raise ValueError('Deterministic, batch-independent evaluation model required')


def population(model, x, y, client_ids, *, with_gradient, block_size=512):
    """R_i and optionally grad R_i, without clipping or privacy protection."""
    _validate(model, x, y, client_ids, block_size)
    params = [p for p in model.parameters() if p.requires_grad]
    risks, gradients = [], []
    for ids in client_ids:
        # Match the original full-risk query's chunking and float32 accumulation.
        risk = torch.zeros(1, device='mps')
        total = torch.zeros(sum(p.numel() for p in params), device='mps') if with_gradient else None
        for block in ids.split(block_size):
            with torch.set_grad_enabled(with_gradient):
                r = losses(model(x[block]), y[block], 'brier').sum() / len(ids)
                if with_gradient:
                    gs = torch.autograd.grad(r, params)
                    total += torch.cat([g.detach().flatten() for g in gs])
            risk += r.detach()
        risks.append(risk.squeeze(0))
        if with_gradient:
            gradients.append(total)
    risks = torch.stack(risks)
    if not bool(torch.isfinite(risks).all()) or float(risks.min()) < 0 or float(risks.max()) > 1 + 1e-6:
        raise FloatingPointError('Invalid population risks')
    grads = torch.stack(gradients) if with_gradient else None
    if grads is not None and not bool(torch.isfinite(grads).all()):
        raise FloatingPointError('Invalid population gradients')
    return risks, grads


def targets(raw_risks, raw_gradients, clipped_batch_means):
    if any(t.device.type != 'mps' for t in (raw_risks, raw_gradients, clipped_batch_means)):
        raise ValueError('MPS targets required')
    lam, a = weights(raw_risks, .5)
    GJ = (a[:, None] * raw_gradients).mean(0)
    h = (lam[:, None] * raw_gradients).sum(0)
    h_batch = (lam[:, None] * clipped_batch_means).sum(0)
    torch.testing.assert_close(GJ / a.mean(), h, rtol=2e-5, atol=2e-7)
    return dict(oracle_weights=lam, coefficients=a, GJ=GJ, h=h, h_batch=h_batch,
                J=float(potential(raw_risks, .5).mean()))


def aggregate(treatment, private_messages, clean_messages, private_reports, raw_risks):
    treatment_id(treatment)
    xx = private_messages if treatment['message'] == 'private' else clean_messages
    if xx.device.type != 'mps':
        raise ValueError('MPS messages required')
    if treatment['weight'] == 'uniform':
        lam = torch.ones(len(xx), device='mps') / len(xx)
    else:
        lam, _ = weights(private_reports if treatment['weight'] == 'private' else raw_risks, .5)
    if treatment['aggregator'] == 'mean':
        A, solver = (lam[:, None] * xx).sum(0), None
    else:
        A, solver = weighted_rfa(xx, lam, iterations=40, smoothing=1e-5)
    return A, dict(objective_weights=lam.cpu().tolist(), solver=solver,
                   concentration=float(len(lam) * lam.square().sum()), max_weight=float(lam.max()))


def effects(A, target, *, eta, J_after):
    if not math.isfinite(eta) or eta <= 0 or not math.isfinite(J_after):
        raise ValueError('Finite actual step required')
    predicted = eta * float(torch.dot(target['GJ'], A))
    change = J_after - target['J']
    return dict(error_to_population_target_sq=float((A-target['h']).square().sum()),
                error_to_clipped_batch_target_sq=float((A-target['h_batch']).square().sum()),
                eta=eta, step_norm=float(torch.linalg.vector_norm(eta*A)),
                predicted_J_decrease=predicted, actual_J_decrease=-change,
                J_before=target['J'], J_after=J_after,
                finite_step_remainder=change+predicted)
