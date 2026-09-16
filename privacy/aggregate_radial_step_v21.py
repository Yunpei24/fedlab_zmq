"""Public whole-aggregate step controls, not a DP mechanism by themselves."""
import math
import torch


def controlled_step(aggregate, eta, mode, radius=1.):
    if (aggregate.device.type != 'mps' or aggregate.ndim != 1
            or not bool(torch.isfinite(aggregate).all()) or aggregate.numel() == 0):
        raise ValueError('A finite nonempty MPS aggregate is required')
    if not math.isfinite(eta) or eta <= 0 or not math.isfinite(radius) or radius <= 0:
        raise ValueError('Positive finite public eta and radius required')
    norm = float(torch.linalg.vector_norm(aggregate))
    if mode == 'unchanged':
        factor = 1.
    elif mode == 'half':
        factor = .5
    elif mode == 'global_clip':
        factor = min(1., radius/norm) if norm else 1.
    else:
        raise ValueError('Unknown control')
    step = eta * (factor * aggregate)
    return step, dict(eta=eta, radius=radius, factor=factor,
                     aggregate_norm=norm, step_norm=float(torch.linalg.vector_norm(step)),
                     global_clipped=mode == 'global_clip' and norm > radius)
