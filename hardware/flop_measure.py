"""
hardware/flop_measure.py
========================
Measured-FLOPs interface backed by torch.utils.flop_counter.FlopCounterMode.

Why this module exists
----------------------
The analytic FLOP estimator in hardware/profiles.py uses
    flops_per_step = 3 * 2 * num_params * batch_size
which is a Kaplan/Hoffmann-style proxy that under-counts CNNs (spatial maps
are ignored) by ~2 orders of magnitude in absolute value, and — more
importantly for FedPart/FedPartBE — it assumes the cost of a layer-group
backward pass is proportional to the group's *size*. In reality it depends
on the group's *position* (depth from the output): training a shallow group
forces backprop through every downstream layer, so the cost is dominated by
position, not by parameter count.

This module provides three primitives:
    1. measure_fwd_flops(model, input_shape, batch_size)
    2. measure_fwd_bwd_full_flops(model, input_shape, batch_size)
    3. measure_fwd_bwd_per_group_flops(model, groups, input_shape, batch_size)

All measurements are cached by
    (id(model.__class__), input_shape, batch_size, group_idx_or_None)
so they run exactly once per (architecture, shape) and never inside the
per-round training loop.

Convention
----------
FlopCounterMode in PyTorch 2.x counts true FLOPs (mul+add separately).
A ResNet-50 forward at 224x224 measures ~8.2 GFLOPs, not the literature
"~4.1 GMACs". The factor stored here is therefore 1.0, which matches
DeviceProfile.compute.peak_gflops (also FP32 FLOPs/s with mul+add
separate). The two must agree for time = flops / throughput to be correct.

Do NOT wrap a torch.compile'd model — FlopCounterMode hooks do not fire
under the inductor backend. Always pass the eager model.
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode


# ─────────────────────────────────────────────────────────────────────────────
# Convention
# ─────────────────────────────────────────────────────────────────────────────

# FlopCounterMode reports true FLOPs (verified: ResNet-50 fwd@224 -> 8.18e9).
# DeviceProfile.compute.peak_gflops is also in FP32 FLOP/s (mul+add separate).
# So measured FLOPs can be divided by peak_gflops*1e9 directly to get seconds.
FLOPCOUNTER_TO_FLOPS_FACTOR: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Internal cache
# ─────────────────────────────────────────────────────────────────────────────

# Keyed by (model class id, input_shape, batch_size, group_idx)
# group_idx == -1 means "no freezing" (full fwd+bwd or fwd-only).
# group_idx == -2 means "forward only".
_MEASURE_CACHE: dict[tuple, int] = {}

# One-shot warning state for the recalibration nudge.
_MEASURED_WARNED: dict[str, bool] = {"done": False}


def _cache_key(model: nn.Module, input_shape: tuple, batch_size: int,
               group_idx: int) -> tuple:
    return (id(type(model)), tuple(input_shape), int(batch_size), int(group_idx))


def clear_cache() -> None:
    """Drop all cached measurements (test helper)."""
    _MEASURE_CACHE.clear()
    _MEASURED_WARNED["done"] = False


def cache_size() -> int:
    """Number of cached measurements (test helper)."""
    return len(_MEASURE_CACHE)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _build_dummy_input(input_shape: tuple, batch_size: int,
                       device: torch.device) -> torch.Tensor:
    # input_shape is (N, C, H, W) — we replace N by batch_size so callers can
    # reuse the same shape constant they pass to _compute_group_flops.
    shape = (batch_size,) + tuple(input_shape[1:])
    return torch.zeros(shape, device=device, dtype=torch.float32)


def _freeze_all_except_group(model: nn.Module,
                             groups: list[list[str]],
                             active_idx: int) -> None:
    """Mirror of algorithms.fedpart._freeze_all_except_group.

    Duplicated here (instead of imported) because hardware/ must not depend
    on algorithms/. The semantics are the contract:
        active_idx == -1 -> all params trainable (full bwd)
        active_idx >=  0 -> only params in groups[active_idx] trainable
    """
    active_names: set[str] = set()
    if active_idx >= 0:
        active_names = set(groups[active_idx])
    for name, param in model.named_parameters():
        if active_idx < 0 or name in active_names:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)


def _restore_all_grad(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(True)


def _is_compiled(model: nn.Module) -> bool:
    # torch.compile wraps the module — FlopCounterMode hooks won't fire.
    return getattr(model, "_orig_mod", None) is not None


# ─────────────────────────────────────────────────────────────────────────────
# Public API — single-pass measurements
# ─────────────────────────────────────────────────────────────────────────────

def measure_fwd_flops(model: nn.Module,
                      input_shape: tuple = (1, 3, 32, 32),
                      batch_size: int = 32) -> int:
    """Measure forward-only FLOPs for one mini-batch of `batch_size` samples.

    Cached. Does not change model.training state semantically: it forces
    eval() during measurement to avoid BN-statistics side effects, then
    restores the original mode.
    """
    if _is_compiled(model):
        raise RuntimeError(
            "measure_fwd_flops received a torch.compile()'d model. "
            "FlopCounterMode does not see ops behind the inductor backend. "
            "Pass the underlying eager model (model._orig_mod) instead."
        )

    key = _cache_key(model, input_shape, batch_size, -2)
    cached = _MEASURE_CACHE.get(key)
    if cached is not None:
        return cached

    was_training = model.training
    model.eval()
    device = _model_device(model)
    x = _build_dummy_input(input_shape, batch_size, device)

    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            model(x)
        flops = int(fc.get_total_flops() * FLOPCOUNTER_TO_FLOPS_FACTOR)

    if was_training:
        model.train()

    _MEASURE_CACHE[key] = flops
    return flops


def measure_fwd_bwd_full_flops(model: nn.Module,
                               input_shape: tuple = (1, 3, 32, 32),
                               batch_size: int = 32) -> int:
    """Measure forward + full backward FLOPs for one mini-batch.

    "Full" = nothing frozen, every parameter receives a gradient.
    Used as the FedAvg baseline cost per step.
    """
    if _is_compiled(model):
        raise RuntimeError(
            "measure_fwd_bwd_full_flops received a torch.compile()'d model."
        )

    key = _cache_key(model, input_shape, batch_size, -1)
    cached = _MEASURE_CACHE.get(key)
    if cached is not None:
        return cached

    was_training = model.training
    model.train()
    device = _model_device(model)

    # Snapshot requires_grad so we can restore exactly the pre-call state.
    saved_rg = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(True)

    x = _build_dummy_input(input_shape, batch_size, device)
    # Dummy targets — only the FLOPs matter, not the loss value.
    # Use a small classification head: output of model is logits over C classes.
    with FlopCounterMode(display=False) as fc:
        out = model(x)
        # Reduce to scalar loss to trigger a backward pass.
        # Using out.sum() is the cheapest and covers every output element.
        loss = out.float().sum()
        loss.backward()
    flops = int(fc.get_total_flops() * FLOPCOUNTER_TO_FLOPS_FACTOR)

    # Drop the gradients we just produced.
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    for n, p in model.named_parameters():
        p.requires_grad_(saved_rg.get(n, True))

    if not was_training:
        model.eval()

    _MEASURE_CACHE[key] = flops
    return flops


def measure_fwd_bwd_per_group_flops(model: nn.Module,
                                    groups: list[list[str]],
                                    input_shape: tuple = (1, 3, 32, 32),
                                    batch_size: int = 32) -> list[int]:
    """For each group g, measure (forward + backward with everything frozen
    except group g). Returns a list aligned with `groups`.

    Cached per (architecture, input_shape, batch_size, group_idx). Subsequent
    calls with the same group structure hit the cache without re-running.
    """
    if _is_compiled(model):
        raise RuntimeError(
            "measure_fwd_bwd_per_group_flops received a torch.compile()'d model."
        )

    # Cheap path: every group already cached.
    cached_all = []
    miss_any = False
    for g_idx in range(len(groups)):
        k = _cache_key(model, input_shape, batch_size, g_idx)
        if k in _MEASURE_CACHE:
            cached_all.append(_MEASURE_CACHE[k])
        else:
            miss_any = True
            break
    if not miss_any and cached_all:
        return cached_all

    was_training = model.training
    model.train()
    device = _model_device(model)
    saved_rg = {n: p.requires_grad for n, p in model.named_parameters()}

    results: list[int] = []
    for g_idx in range(len(groups)):
        key = _cache_key(model, input_shape, batch_size, g_idx)
        if key in _MEASURE_CACHE:
            results.append(_MEASURE_CACHE[key])
            continue

        _freeze_all_except_group(model, groups, g_idx)
        x = _build_dummy_input(input_shape, batch_size, device)

        with FlopCounterMode(display=False) as fc:
            out = model(x)
            loss = out.float().sum()
            try:
                loss.backward()
            except RuntimeError as exc:
                # No parameters require_grad in this group (degenerate group):
                # fall back to forward-only FLOPs so we never return zero.
                warnings.warn(
                    f"[flop_measure] group {g_idx} produced no grad path "
                    f"({exc}); falling back to forward-only FLOPs.",
                    stacklevel=2,
                )
        flops = int(fc.get_total_flops() * FLOPCOUNTER_TO_FLOPS_FACTOR)

        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

        _MEASURE_CACHE[key] = flops
        results.append(flops)

    # Restore original requires_grad and train/eval mode.
    for n, p in model.named_parameters():
        p.requires_grad_(saved_rg.get(n, True))
    if not was_training:
        model.eval()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher: analytic vs measured FLOPs
# ─────────────────────────────────────────────────────────────────────────────

# Sentinel used when the algorithm has no notion of partial training.
# Algorithms that already partition the model (FedPart, FedPartBE) pass an
# integer >= 0 to select a specific group; FedAvg / FedProx / etc pass None.
_FULL = -1


def compute_training_flops(
    model: nn.Module,
    profile,                       # hardware.profiles.DeviceProfile
    dataloader: torch.utils.data.DataLoader,
    local_epochs: int,
    config: dict,
    *,
    active_group_idx: Optional[int] = None,
    groups: Optional[list[list[str]]] = None,
    group_flops_analytic: Optional[list[float]] = None,
    beta_fraction: Optional[float] = None,
) -> float:
    """Return the total training FLOPs to be charged for one client round.

    Behaviour
    ─────────
    Default (config.use_measured_flops is False or absent):
        Identical to the legacy analytic formula. For algorithms that pass
        an `active_group_idx` and an analytic `group_flops_analytic`, returns
            full_flops * (1/3 + 2/3 * group_flops_analytic[g] / sum(group_flops_analytic))
        Otherwise returns full_flops (FedAvg / FedProx).

    use_measured_flops = True:
        Forces FLOPs to come from FlopCounterMode (cached). The contract is:
            * No active_group_idx       -> fwd_bwd_full * steps
            * active_group_idx == -1    -> fwd_bwd_full * steps (warmup row)
            * active_group_idx >= 0     -> fwd_bwd_per_group[g] * steps

    Steps are always (dataset_size // batch_size) * local_epochs — unchanged
    from the existing convention.

    Notes
    ─────
    * Never measured per round. The cache in this module makes the second
      and following calls O(dict lookup).
    * Never replaces the energy_scale_factor logic, which lives in the algo
      and is calibration, not measurement.
    """
    batch_size = dataloader.batch_size
    dataset_size = len(dataloader.dataset)
    steps = (dataset_size // batch_size) * local_epochs

    use_measured = bool(config.get("use_measured_flops", False))

    if use_measured and not _MEASURED_WARNED["done"]:
        warnings.warn(
            "[flop_measure] use_measured_flops=True: measured FLOPs are typically "
            "~2 orders of magnitude larger than the analytic estimator "
            "(3*2*num_params*B*S). The energy_scale_factor (e.g. 12.6) was "
            "calibrated against the analytic estimator and will need to be "
            "recalibrated separately if you want to keep the same per-round "
            "energy budgets in Joules.",
            stacklevel=2,
        )
        _MEASURED_WARNED["done"] = True

    if not use_measured:
        # Legacy path — strictly unchanged behaviour.
        num_params = sum(p.numel() for p in model.parameters())
        full_flops = profile.flops_for_model(
            num_params, batch_size, local_epochs, dataset_size
        )
        if beta_fraction is not None:
            # Legacy server_mask/fed_resonance/ccsEF heuristic:
            #   effective = (1 + beta) / 2 * full_flops
            # — half the cost at beta=0 (forward-only), full at beta=1.
            return float(full_flops * 0.5 * (1.0 + float(beta_fraction)))
        if active_group_idx is None or active_group_idx == _FULL:
            return float(full_flops)
        if group_flops_analytic is None or not group_flops_analytic:
            return float(full_flops)
        active_fraction = (
            group_flops_analytic[active_group_idx] / sum(group_flops_analytic)
        )
        # The 1/3 + 2/3*phi split is the per-step decomposition (fwd is
        # always full; bwd shrinks to the active group). The analytic
        # full_flops already bakes in the (fwd + bwd + opt-step) factor of 3,
        # so we apply the split to the pre-step total directly.
        return float(full_flops * (1.0 / 3.0 + 2.0 / 3.0 * active_fraction))

    # Measured path.
    input_shape = config.get("input_shape", (1, 3, 32, 32))

    if beta_fraction is not None:
        fwd_only = measure_fwd_flops(model, input_shape, batch_size)
        fwd_bwd  = measure_fwd_bwd_full_flops(model, input_shape, batch_size)
        # Interpolate between fwd-only (beta=0) and full fwd+bwd (beta=1).
        per_step = fwd_only + float(beta_fraction) * max(0, fwd_bwd - fwd_only)
        return float(per_step * steps)

    if active_group_idx is None or active_group_idx == _FULL:
        per_step = measure_fwd_bwd_full_flops(model, input_shape, batch_size)
        return float(per_step * steps)

    if groups is None:
        raise ValueError(
            "compute_training_flops(use_measured_flops=True, active_group_idx>=0) "
            "requires `groups` so that per-group FLOPs can be measured."
        )

    per_group = measure_fwd_bwd_per_group_flops(
        model, groups, input_shape, batch_size
    )
    g = int(active_group_idx)
    if g < 0 or g >= len(per_group):
        # Should not happen — the caller pre-clamps active_group_idx — but if
        # it does, fall back to the full-bwd measurement rather than crashing.
        per_step = measure_fwd_bwd_full_flops(model, input_shape, batch_size)
        return float(per_step * steps)
    return float(per_group[g] * steps)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Verify the MAC vs FLOP convention against a textbook reference.
    from torchvision.models import resnet50

    m = resnet50().eval()
    x = torch.randn(1, 3, 224, 224)
    with FlopCounterMode(display=False) as fc:
        m(x)
    flops = fc.get_total_flops()
    print(f"ResNet-50 fwd @ 224x224, batch=1: {flops:.3e}")
    print(f"  In GFLOPs: {flops/1e9:.3f}")
    if 7.5e9 <= flops <= 9.0e9:
        print("  -> FlopCounterMode reports true FLOPs (factor = 1.0).")
    elif 3.5e9 <= flops <= 4.7e9:
        print("  -> FlopCounterMode reports MACs (factor = 2.0 needed!).")
    else:
        print("  -> Unexpected magnitude — investigate.")
    print(f"  Module factor in use: {FLOPCOUNTER_TO_FLOPS_FACTOR}")
