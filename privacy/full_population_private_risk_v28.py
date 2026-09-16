"""Streaming per-example clipping followed by ONE full-population Gaussian.

This research lane has no local optimizer step, no Poisson amplification and
no privacy release per accumulation block. Tensor operations require MPS.
"""
import math
import torch
from privacy.fair_objective import require_mps, per_example, release, calibrate

METHODS = ('erm_mean', 'erm_rfa', 'risk_mean', 'risk_rfa')


def prefix_epsilon(plan, completed_rounds):
    if type(completed_rounds) is not int or not 0 <= completed_rounds <= plan['T']:
        raise ValueError('Invalid public prefix')
    if completed_rounds == 0:
        return 0., 2
    z, zr = plan['gradient_z'], plan['risk_z']
    return min((completed_rounds*a/(2*z*z)
                +(completed_rounds*a/(2*zr*zr) if zr is not None else 0.)
                +math.log(1/plan['delta'])/(a-1), a) for a in range(2, 65))


def ledger(method):
    if method not in METHODS:
        raise ValueError('Unknown frozen method')
    risk = method.startswith('risk_')
    zr = calibrate(q=1., steps=120, epsilon=.25, delta=5e-6) if risk else None
    candidates = []
    for a in range(2, 65):
        available = 4-math.log(1e5)/(a-1)-(120*a/(2*zr*zr) if risk else 0.)
        if available > 0:
            candidates.append(math.sqrt(120*a/(2*available)))
    z = min(candidates)*(1+1e-8)
    p = dict(N=4800, b=4800, T=120, C=2., gradient_sensitivity=4/4800,
             gradient_z=z, gradient_std=z*4/4800, risk_z=zr,
             risk_std=zr/4800 if risk else 0., risk_sensitivity=1/4800 if risk else None,
             risk_calibration_epsilon=.25 if risk else None,
             risk_calibration_delta=5e-6 if risk else None,
             gradient_releases=120, risk_releases=120 if risk else 0,
             epsilon_cap=4., delta=1e-5, adjacency='replace_one', sampling='full_population',
             accounting='ordinary Gaussian RDP, q=1; joint channels; basic conversion; orders 2..64',
             scope='sample-level per-client per-run messages only',
             accumulation_blocks_are_not_releases=True)
    e, a = prefix_epsilon(p, 120)
    assert e <= 4
    p.update(epsilon_realized=e, order=a,
             rdp={a:120*a/(2*z*z)+(120*a/(2*zr*zr) if risk else 0.) for a in range(2,65)})
    return p


def clipped_population_mean(model, x, y, *, C=2., block_size=240):
    require_mps()
    if (x.device.type != 'mps' or y.device.type != 'mps' or len(x) != len(y) or len(x) < 1
            or not math.isfinite(C) or C <= 0 or type(block_size) is not int or block_size < 1
            or model.training or any(p.device.type != 'mps' for p in model.parameters())):
        raise ValueError('Fixed MPS model and nonempty population required')
    total = torch.zeros(sum(p.numel() for p in model.parameters() if p.requires_grad), device='mps')
    loss_sum = torch.zeros((), device='mps')
    norm_sum = torch.zeros((), device='mps')
    clipped = 0
    norm_max = 0.
    blocks = 0
    for start in range(0, len(x), block_size):
        losses, rows, norms, _ = per_example(model, x[start:start+block_size], y[start:start+block_size],
                                            kind='brier', clip_norm=C)
        total += rows.sum(0)
        loss_sum += losses.sum()
        norm_sum += norms.sum()
        norm_max = max(norm_max, float(norms.max()))
        clipped += int((norms>C).sum())
        blocks += 1
    query = total/len(x)
    if not bool(torch.isfinite(query).all()):
        raise FloatingPointError('Invalid population query')
    return query, dict(population_size=len(x), block_size=block_size, accumulation_blocks=blocks,
                       per_example_clipped_count=clipped, per_example_clipped_fraction=clipped/len(x),
                       raw_gradient_norm_mean=float(norm_sum/len(x)), raw_gradient_norm_max=norm_max,
                       raw_population_brier_risk=float(loss_sum/len(x)),
                       local_optimizer_steps=0, query_norm=float(torch.linalg.vector_norm(query)),
                       clipping_before_averaging=True, diagnostic_fields_not_private=True)


def private_population_gradient(model, x, y, *, C=2., block_size=240, noise_std, seed):
    if not math.isfinite(noise_std) or noise_std <= 0:
        raise ValueError('This lane always adds positive Gaussian noise')
    query, diag = clipped_population_mean(model, x, y, C=C, block_size=block_size)
    message = release(query, noise_std=noise_std, seed=seed)
    diag.update(gaussian_releases=1, gradient_noise_std=noise_std,
                replace_one_sensitivity=2*C/len(x), noise_added_after_complete_mean=True)
    return message, query, diag  # query/diagnostics are research oracles, never server inputs.
