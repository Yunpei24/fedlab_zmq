"""Isolated contracts for the proposed FAR noise-effect study (not a runner).

Historical algorithms and frozen experiments are deliberately untouched.
All numerical tensor operations require MPS. Host-only statistics are scalar.
Sampling is independent fixed-cardinality WOR each round, NOT random reshuffling.
Seeds/indices and research diagnostics are not a production DP transcript.
"""
from contextlib import contextmanager
import hashlib
import math
import statistics

import torch

from privacy.local_dpsgd import _per_sample_grads_vectorized
from privacy.rdp import RDPAccountant, calibrate_sampled_without_replacement_gaussian_noise
from robustness.aggregators import aggregate_vectors


def stream_seed(private_root, outer_seed, repeat, channel, round_index=0, client=0):
    if not private_root or channel not in {"batch", "noise", "init", "split"}:
        raise ValueError("explicit entropy root and named channel required")
    payload = f"{private_root}/{outer_seed}/{repeat}/{channel}/{round_index}/{client}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**63 - 1)


def require_mps(*values):
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required; CPU fallback forbidden")
    if any(value.device.type != "mps" for value in values):
        raise ValueError("all tensors must be on MPS")


@contextmanager
def mps_rng(seed):
    require_mps()
    before = torch.mps.get_rng_state()
    try:
        torch.mps.manual_seed(seed)
        yield
    finally:
        torch.mps.set_rng_state(before)


def sampled_indices(n, b, seed):
    require_mps()
    if not (0 < b <= n):
        raise ValueError("0 < B <= N required")
    with mps_rng(seed):
        return torch.randperm(n, device="mps")[:b]


def gradient_release(model, x, y, *, local_clip, noise_std, noise_seed, microbatch=32):
    """One CE gradient, optional per-example clipping, then ONE Gaussian.

    x/y already contain the sampled batch. No local optimiser update. Model
    must be deterministic across examples (no dropout/BatchNorm). The same
    flat Gaussian draw is used for different microbatch sizes. Diagnostics
    returned here are simulator-only and MUST NOT be sent as DP messages.
    """
    require_mps(x, y, *model.parameters())
    if any(isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout))
           for m in model.modules()):
        raise ValueError("batch-coupled or stochastic layers need a separate audit")
    if len(x) != len(y) or len(x) < 1 or microbatch < 1:
        raise ValueError("nonempty matching batch and positive microbatch required")
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("finite nonnegative noise std required")
    if local_clip is None and noise_std != 0:
        raise ValueError("unclipped gradients cannot claim this DP mechanism")
    if local_clip is not None and (not math.isfinite(local_clip) or local_clip <= 0):
        raise ValueError("finite positive clipping threshold required")
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    sums = [torch.zeros_like(p) for _, p in params]
    clipped = 0
    # Each microbatch is purely a memory split, not a separate private release.
    for xx, yy in zip(x.split(microbatch), y.split(microbatch)):
        grads, _ = _per_sample_grads_vectorized(model, [name for name, _ in params], xx, yy)
        norm2 = sum(g.reshape(len(xx), -1).square().sum(1) for g in grads)
        factors = (torch.ones_like(norm2) if local_clip is None else
                   (local_clip / norm2.sqrt().clamp_min(1e-12)).clamp(max=1))
        clipped += int((factors < 1).sum().item())
        for accumulator, g in zip(sums, grads):
            accumulator.add_((g * factors.view((len(xx),) + (1,) * (g.ndim - 1))).sum(0))
    clean = torch.cat([g.reshape(-1) for g in sums]) / len(x)
    with mps_rng(noise_seed):
        z = torch.randn(clean.shape, device="mps", dtype=clean.dtype)
    return clean + noise_std * z, {"clean_oracle": clean, "noise_oracle": z,
                                  "clip_fraction_oracle": clipped / len(x),
                                  "privacy_protected_diagnostics": False}


def far_aggregate(messages, *, alpha, reference, server_clip=None, anchor=None,
                  reference_radius=1.0, f_budget=2):
    require_mps(messages)
    if messages.ndim != 2 or not torch.isfinite(messages).all() or not math.isfinite(alpha):
        raise ValueError("finite matrix and alpha required")
    if reference not in {"rfa", "coordinate_median", "trimmed_mean", "centered_clipping"}:
        raise ValueError("unsupported reference")
    if server_clip is not None and (not math.isfinite(server_clip) or server_clip <= 0):
        raise ValueError("finite positive server radius required")
    x = messages
    factors = torch.ones(len(x), device="mps")
    if server_clip is not None:
        factors = (server_clip / x.norm(dim=1).clamp_min(1e-12)).clamp(max=1)
        x = x * factors[:, None]
    if anchor is None:
        anchor = torch.zeros(x.shape[1], device="mps")
    require_mps(anchor)
    ref = aggregate_vectors(x, method=reference, f=f_budget, anchor=anchor,
                            tau=reference_radius, max_iter=100, tol=1e-6)
    # No output projection, score clipping, alpha cap, or hidden normalisation.
    distances = (x - ref).norm(dim=1)
    weights = torch.softmax(alpha * distances, dim=0)
    return (weights[:, None] * x).sum(0), {
        "weights": weights, "reference": ref, "distances": distances,
        "server_clip_fraction": float((factors < 1).float().mean().item()),
    }


def privacy_plan(*, epsilon, delta, n_local, batch, rounds, clip):
    if not (0 < batch <= n_local) or rounds < 1 or clip <= 0 or not 0 < delta < 1:
        raise ValueError("invalid public privacy parameters")
    if epsilon is None:
        return dict(enabled=False, std=0.0, sigma_C=0.0, epsilon_bound=None)
    sigma = calibrate_sampled_without_replacement_gaussian_noise(
        target_epsilon=epsilon, delta=delta, sampling_rate=batch/n_local,
        steps=rounds, sensitivity_multiplier=2.0)
    def bound(s):
        acc = RDPAccountant()
        acc.add_sampled_without_replacement_gaussian(channel="gradient", sampling_rate=batch/n_local,
                                                    noise_multiplier=s/2, steps=rounds)
        return acc.epsilon(delta)[0]
    # Old calibration may return slightly ABOVE epsilon within its tolerance.
    # Keep frozen history intact, tighten only this new plan's multiplier.
    for _ in range(100):
        value = bound(sigma)
        if value <= epsilon:
            return dict(enabled=True, sigma_C=sigma, z_sensitivity=sigma/2,
                        std=sigma*clip/batch, sensitivity=2*clip/batch,
                        epsilon_bound=value, delta=delta, q=batch/n_local,
                        steps=rounds, accountant="fixed_wor_replace_one")
        sigma *= 1.0001
    raise RuntimeError("could not achieve one-sided epsilon bound")


def nested_summary(values):
    """values[outer_seed][MC] at ONE configuration and evaluated round."""
    if len(values) < 2 or len({len(v) for v in values}) != 1 or len(values[0]) < 2:
        raise ValueError("balanced >=2 outer seeds and >=2 MC repetitions required")
    if not all(math.isfinite(x) for row in values for x in row):
        raise ValueError("no missing/nonfinite values or imputation")
    means = [statistics.mean(row) for row in values]
    return {"mean": statistics.mean(means), "outer_mean_sd": statistics.stdev(means),
            "within_mc_rms_sd": math.sqrt(statistics.mean(statistics.variance(row) for row in values)),
            "all_trajectory_sd": statistics.stdev([x for row in values for x in row]),
            "outer_count": len(values), "mc_count": len(values[0])}
