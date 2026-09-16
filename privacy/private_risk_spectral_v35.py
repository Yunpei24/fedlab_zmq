"""Exploratory risk-weighted SMEA, exact subset enumeration, MPS only.

SMEA is an existing aggregation principle, not our novelty claim. This version
uses bounded risk weights inside both means and covariances. Numeric intervals
are float32 diagnostics, not outward-rounded certificates.
"""
import itertools
import math
import torch
from privacy.fair_objective import require_mps
from privacy.capped_private_risk import weights


def spectral_intervals(matrices, *, sweeps=12):
    """Cyclic Jacobi on small batched symmetric matrices, without CPU eigensolve.

    Exact arithmetic: max diagonal is a lower bound, and maximum Gershgorin
    endpoint an upper bound, on the largest eigenvalue after orthogonal rotation.
    Float32 bounds are approximate; caller must check width and finiteness.
    """
    require_mps()
    if (matrices.device.type != 'mps' or matrices.dtype != torch.float32
            or matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]
            or type(sweeps) is not int or sweeps < 1):
        raise ValueError('Batched square MPS float32 matrices required')
    if not bool(torch.isfinite(matrices).all()):
        raise ValueError('Finite matrices required')
    d = (matrices + matrices.transpose(1, 2))*.5
    n = d.shape[1]
    eye = torch.eye(n, device='mps').expand(len(d), n, n)
    for _ in range(sweeps):
        for p in range(n):
            for q in range(p+1, n):
                off = d[:, p, q]
                theta = .5*torch.atan2(2*off, d[:, q, q]-d[:, p, p])
                theta = torch.where(off == 0, torch.zeros_like(theta), theta)
                c, s = theta.cos(), theta.sin()
                rot = eye.clone()
                rot[:, p, p] = c; rot[:, q, q] = c
                rot[:, p, q] = s; rot[:, q, p] = -s
                d = rot.transpose(1, 2).bmm(d).bmm(rot)
                d = (d+d.transpose(1, 2))*.5
    diag = d.diagonal(dim1=1, dim2=2)
    offsum = d.abs().sum(2)-diag.abs()
    lower = diag.max(1).values
    upper = (diag+offsum.clamp_min(0)).max(1).values
    if not bool(torch.isfinite(upper).all()):
        raise FloatingPointError('Jacobi overflow')
    return lower, upper


def subset_covariance_grams(messages, coeff, subsets):
    """Build weighted centered Gram matrices; n*n dot products, no d*d matrix."""
    anchor = messages.median(dim=0).values
    centered = messages-anchor
    scale = centered.abs().max().clamp_min(1e-20)
    unit = centered/scale
    full = unit @ unit.T
    ix = torch.tensor(subsets, dtype=torch.long, device='mps')
    cs = coeff[ix]; ps = cs/cs.sum(1, keepdim=True)
    raw = full[ix[:, :, None], ix[:, None, :]]
    h = torch.eye(ix.shape[1], device='mps')[None, :, :]-ps[:, None, :]
    centered_gram = h.bmm(raw).bmm(h.transpose(1, 2))
    gram = centered_gram*ps.sqrt()[:, :, None]*ps.sqrt()[:, None, :]
    return gram, ps, scale


def risk_smea(messages, reports, *, f_budget, max_subsets=10000):
    require_mps()
    if (messages.device.type != 'mps' or messages.dtype != torch.float32
            or messages.ndim != 2 or min(messages.shape) < 1
            or reports.device.type != 'mps' or reports.shape != (len(messages),)
            or type(f_budget) is not int or not 0 <= f_budget < len(messages)/2):
        raise ValueError('MPS message matrix, reports, and 0 <= f < n/2 required')
    if not bool(torch.isfinite(messages).all()) or float(messages.abs().max()) > 1e30:
        raise ValueError('Messages outside audited finite coordinate domain')
    n = len(messages); k = n-f_budget
    if math.comb(n, k) > max_subsets:
        raise ValueError('Combinatorial budget exceeded; no silent approximation')
    lam, _ = weights(reports, .5)
    # Relative weights suffice; the public coefficient ratio is at most three.
    subsets = list(itertools.combinations(range(n), k))
    grams, ps, scale = subset_covariance_grams(messages, lam, subsets)
    lower, upper = spectral_intervals(grams)
    width = (upper-lower).clamp_min(0)
    tol = 2e-5*upper.abs().clamp_min(1.)
    if not bool((width <= tol).all()):
        raise RuntimeError('Jacobi interval did not converge at frozen 12 sweeps')
    best = int(upper.argmin())
    ids = subsets[best]
    ix = torch.tensor(ids, dtype=torch.long, device='mps')
    selected = ps[best]
    result = (selected[:, None]*messages[ix]).sum(0)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError('Nonfinite spectral mean')
    scale2 = float(scale)**2
    effective = torch.zeros(n, device='mps'); effective[ix] = selected
    info = dict(method='risk-weighted SMEA; existing spectral subset principle',
        selected_ids=list(ids), removed_ids=[i for i in range(n) if i not in ids],
        effective_weights=effective.cpu().tolist(), subsets=len(subsets), f_budget=f_budget,
        selected_eigenvalue_lower=float(lower[best])*scale2,
        selected_eigenvalue_upper=float(upper[best])*scale2,
        selection_gap_upper=max(0.,float(upper[best]-lower.min()))*scale2,
        max_interval_width=float(width.max())*scale2,
        bound_arithmetic='MPS float32, not interval-certified rounding',
        input_dependence='private messages and private reports only',
        sweeps=12, global_validation=False)
    return result, info
