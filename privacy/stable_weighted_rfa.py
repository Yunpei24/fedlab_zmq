"""Numerically guarded smooth weighted geometric median, MPS float32 only.

This independent diagnostic module does not alter any frozen experiment.
The reported convexity bound is evaluated in floating point, not an interval
arithmetic certificate. Finite iterations do not promise an exact median.
"""
import math
import torch


def stable_norm(x, dim=-1):
    """Euclidean norm without squaring coordinates at their original scale."""
    scale = x.abs().amax(dim=dim, keepdim=True)
    divisor = torch.where(scale > 0, scale, torch.ones_like(scale))
    return scale.squeeze(dim) * (x / divisor).square().sum(dim=dim).sqrt()


def _smooth_distance(diff, smoothing):
    scale = diff.abs().amax(dim=1).clamp_min(smoothing)
    return scale * ((diff / scale[:, None]).square().sum(1)
                    + (smoothing / scale).square()).sqrt()


def weighted_rfa(vectors, weights, *, iterations=40, smoothing=1e-5):
    """Robust initializer and scale-safe distances; explicit residual bound.

    A ball containing any subset of objective mass W > 1/2 localizes a minimizer
    of the smooth objective within W*(r+sqrt(r*r+s*s))/(2*W-1) of its center.
    This controls the distance used in the convexity certificate independently
    of very distant minority messages. It does not identify honest clients.
    """
    if (vectors.device.type != 'mps' or weights.device.type != 'mps'
            or vectors.dtype != torch.float32 or weights.dtype != torch.float32):
        raise ValueError('MPS float32 tensors required')
    if (vectors.ndim != 2 or min(vectors.shape) < 1
            or weights.shape != (len(vectors),) or type(iterations) is not int
            or iterations < 1 or not math.isfinite(smoothing) or smoothing <= 0):
        raise ValueError('Invalid solver dimensions or public parameters')
    if (not bool(torch.isfinite(vectors).all()) or not bool(torch.isfinite(weights).all())
            or float(weights.min()) <= 0):
        raise ValueError('Finite vectors and positive weights required')
    # Explicit numerical domain, not a statistical or DP clipping threshold.
    # Beyond it we reject, rather than silently emit a non-finite aggregate.
    if float(vectors.abs().max()) > 1e30 * (1 + 1e-6):
        raise ValueError('Coordinate exceeds the audited float32 numerical domain 1e30')
    w = weights / weights.max()
    w = w / w.sum()
    anchor = vectors.median(dim=0).values
    dist0 = stable_norm(vectors - anchor, dim=1)
    radii, order = dist0.sort()
    masses = w[order].cumsum(0)
    eligible = masses > .5 + 1e-6
    # Always includes all messages, since weights sum to approximately one.
    r = radii[eligible]
    W = masses[eligible]
    hyp = torch.stack([r, torch.full_like(r, smoothing)], dim=1)
    bounds = W * (r + stable_norm(hyp, dim=1)) / (2 * W - 1)
    k = int(bounds.argmin())
    radius = bounds[k]
    if not bool(torch.isfinite(radius)):
        raise FloatingPointError('Non-finite localization bound')
    point = anchor.clone()
    for _ in range(iterations):
        distances = _smooth_distance(vectors - point, smoothing)
        reweight = w / distances
        normalized = reweight / reweight.sum()
        point = (normalized[:, None] * vectors).sum(0)
        # The ball contains a smooth minimizer in exact arithmetic. Projection
        # prevents an intermediate iterate escaping that bounded search region.
        displacement = point - anchor
        length = stable_norm(displacement)
        point = anchor + displacement * torch.minimum(torch.ones_like(length), radius / length.clamp_min(1e-30))
        if not bool(torch.isfinite(point).all()):
            raise FloatingPointError('Non-finite smooth median iterate')
    diff = point - vectors
    distances = _smooth_distance(diff, smoothing)
    residual = (w[:, None] * (diff / distances[:, None])).sum(0)
    residual_norm = stable_norm(residual)
    distance_to_minimizer_bound = stable_norm(point - anchor) + radius
    gap = residual_norm * distance_to_minimizer_bound + smoothing
    nu = w / distances
    nu = nu / nu.sum()
    reconstruction = stable_norm(point - (nu[:, None] * vectors).sum(0))
    diag = dict(iterations=iterations, smoothing=smoothing,
                initializer='coordinate_median', localized_mass=float(W[k]),
                cluster_radius=float(r[k]), minimizer_radius_bound=float(radius),
                residual_norm=float(residual_norm),
                unsmoothed_objective_gap_upper=float(gap),
                bound_arithmetic='float32 evaluation, no interval rounding certificate',
                stationary_weights=nu.cpu().tolist(),
                stationary_reconstruction_error=float(reconstruction))
    return point, diag
