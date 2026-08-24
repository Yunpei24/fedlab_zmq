"""
hardware/flop_cost.py
=====================
Single source of truth for the per-round compute cost (FLOPs) of an FL client.

Why this module exists
----------------------
Before this module, every algorithm reimplemented its own variant of the
"forward + (partial) backward" cost:

  - FedPart      : full * (1/3 + 2/3 * group_flops[g]/sum(group_flops))   ← phi-model
  - FedStep    : phi-model for ENERGY ACCOUNTING but corrected costs for tier
                   ASSIGNMENT — internally inconsistent.
  - ServerMaskFL : (1 + beta) / 2 * full_flops
  - ccsEF        : (d + 2 * |primary|) * 2 * B * S  (freeze_secondary branch)
  - FedAvg / FedResonance : just full

The validation against torch.utils.flop_counter.FlopCounterMode confirmed the
phi-model under-counts CNNs by ~150–300× AND inverts the rank of shallow
groups (the stem appears the cheapest but is actually the most expensive in
real backward).

The unified primitive here is:

    measure_fwd_bwd_flops(model, trainable_names, input_shape, batch_size)

For any algorithm — contiguous group (FedPart), masked tensors (ServerMaskFL),
partial backward (ccsEF), or full model (FedAvg) — the per-step cost is the
output of FlopCounterMode with exactly the requires_grad mask the algo would
set during real training. Forward is always full; backward depends on the
trainable set.

A dispatcher `round_compute_flops` selects between three cost models, all
sharing the same caller contract (trainable_names + optional legacy hints):

    cost_model = "measured"  (default) — FlopCounterMode, cached.
    cost_model = "corrected"           — full * (1/3 + 2/3 * corrected_fraction)
                                         for contiguous groups; falls back to
                                         measured for non-group masks.
    cost_model = "phi"       (legacy)  — bit-for-bit reproduction of today's
                                         per-algo formulas (regression-test
                                         baseline).

`phi` is the regression mode; switching to "corrected" or "measured" shifts
absolute energy by ~2 orders of magnitude — `energy_scale_factor=12.6` will
need to be recalibrated separately (a one-shot warning is emitted on first
non-phi call).

Convention
----------
FlopCounterMode in PyTorch 2.x reports true FLOPs (mul+add separately).
`calibrate_convention()` measures ResNet-50 forward at 224×224 once: ~8.2 GFLOPs
=> factor 1.0; ~4.1 GFLOPs (rare) => factor 2.0. The factor is cached and
applied to every measurement. DeviceProfile.compute.peak_gflops is also in
FP32 FLOPs/s (mul+add separate), so the two must agree for
`t = flops / peak_flops_per_sec` to be a correct time.

Caveats
-------
* Do NOT pass a torch.compile()'d model — FlopCounterMode hooks do not see
  inductor-backend ops. Use the eager module (`model._orig_mod`).
* Measurements are cached by `(arch_fingerprint, input_shape, batch_size,
  frozenset(trainable_names))`. For algorithms whose mask changes per round
  per client (e.g. ServerMaskFL), this still works — each unique mask is
  measured once and reused. A warning fires if the miss rate is suspiciously
  high (sign of a never-converging mask).
* Every measurement temporarily forces train() mode + saves/restores the
  per-parameter requires_grad mask. Safe to call from inside client_update().
"""

from __future__ import annotations

import warnings
from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

# ─────────────────────────────────────────────────────────────────────────────
# Convention calibration (lazy, cached)
# ─────────────────────────────────────────────────────────────────────────────

_CONVENTION_FACTOR: Optional[float] = None


def calibrate_convention() -> float:
    """Return the multiplier needed to convert FlopCounterMode output to
    "true" FLOPs (the unit used by DeviceProfile.peak_gflops).

    Measured ONCE per process by running a torchvision ResNet-50 forward at
    224×224. The verdict:

        ~8.2e9 → FlopCounterMode already counts true FLOPs → factor 1.0
        ~4.1e9 → FlopCounterMode counts MACs            → factor 2.0

    For PyTorch 2.x (current default) the answer is 1.0. The function still
    measures rather than hard-coding so a future PyTorch convention change
    surfaces here, not as silently-wrong energy numbers downstream.
    """
    global _CONVENTION_FACTOR
    if _CONVENTION_FACTOR is not None:
        return _CONVENTION_FACTOR

    try:
        from torchvision.models import resnet50
    except ImportError:
        # No torchvision — assume true FLOPs (PyTorch 2.x default).
        _CONVENTION_FACTOR = 1.0
        return _CONVENTION_FACTOR

    m = resnet50().eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            m(x)
    flops = fc.get_total_flops()

    if 7.5e9 <= flops <= 9.0e9:
        _CONVENTION_FACTOR = 1.0
    elif 3.5e9 <= flops <= 4.7e9:
        _CONVENTION_FACTOR = 2.0
    else:
        # Unexpected magnitude — default to 1.0 but loudly warn.
        warnings.warn(
            f"[flop_cost] calibrate_convention(): ResNet-50 fwd@224 returned "
            f"{flops:.3e} FLOPs, expected ~8.2e9 (true) or ~4.1e9 (MACs). "
            f"Defaulting factor to 1.0 — verify peak_gflops convention.",
            stacklevel=2,
        )
        _CONVENTION_FACTOR = 1.0

    return _CONVENTION_FACTOR


# ─────────────────────────────────────────────────────────────────────────────
# Cache + miss-rate accounting
# ─────────────────────────────────────────────────────────────────────────────

_CACHE: dict[tuple, float] = {}
_HITS: int = 0
_MISSES: int = 0
_HIGH_MISS_WARNED: bool = False
_RECALIBRATE_WARNED: bool = False


def cache_size() -> int:
    return len(_CACHE)


def cache_stats() -> dict:
    return {"hits": _HITS, "misses": _MISSES, "size": len(_CACHE)}


def clear_cache() -> None:
    global _HITS, _MISSES, _HIGH_MISS_WARNED, _RECALIBRATE_WARNED
    _CACHE.clear()
    _HITS = 0
    _MISSES = 0
    _HIGH_MISS_WARNED = False
    _RECALIBRATE_WARNED = False


def _arch_fingerprint(model: nn.Module) -> int:
    """Hash of the architecture: class name + sorted (name, shape) tuples.

    Two instances of the same class with identical parameter shapes share the
    same fingerprint — that's the intent (cache hits across client instances
    of the same model). Different classes or different shapes (e.g. CIFAR-10
    vs CIFAR-100 head) produce distinct fingerprints.
    """
    cls = type(model).__name__
    shapes = tuple(
        sorted((name, tuple(p.shape)) for name, p in model.named_parameters())
    )
    return hash((cls, shapes))


def _cache_key(
    model: nn.Module, trainable_names: frozenset, input_shape: tuple, batch_size: int
) -> tuple:
    return (
        _arch_fingerprint(model),
        tuple(input_shape),
        int(batch_size),
        trainable_names,
    )


def _maybe_warn_high_miss_rate() -> None:
    """Log once if cache misses are dominating after enough measurements.

    Triggered for algorithms whose trainable mask varies wildly across rounds
    (e.g. ServerMaskFL with an evolving gradient-importance mask). The user
    should know the cache is doing little work — the algorithm itself is the
    bottleneck, not this module.
    """
    global _HIGH_MISS_WARNED
    if _HIGH_MISS_WARNED:
        return
    total = _HITS + _MISSES
    if total >= 30 and (_MISSES / max(total, 1)) > 0.5:
        warnings.warn(
            f"[flop_cost] cache miss rate is {_MISSES/total:.0%} "
            f"({_MISSES} misses / {total} calls). Likely the trainable "
            f"parameter mask varies frequently (ServerMaskFL-style). Each "
            f"unique mask costs one FlopCounterMode run.",
            stacklevel=3,
        )
        _HIGH_MISS_WARNED = True


# ─────────────────────────────────────────────────────────────────────────────
# Freeze / restore (generalised to arbitrary trainable name sets)
# ─────────────────────────────────────────────────────────────────────────────


def freeze_to_trainable(model: nn.Module, trainable_names: Iterable[str]) -> None:
    """Set requires_grad=True on parameters in `trainable_names`, False on all
    others. Mirror of the old per-algo `_freeze_all_except_group`, but keyed
    by an explicit name set instead of a group index.
    """
    names_set = frozenset(trainable_names)
    for name, param in model.named_parameters():
        param.requires_grad_(name in names_set)


def restore_grad(model: nn.Module) -> None:
    """Re-enable gradients for every parameter (post-training cleanup)."""
    for p in model.parameters():
        p.requires_grad_(True)


# ─────────────────────────────────────────────────────────────────────────────
# Group cost helpers (moved from algorithms/fedpart and fedstep)
# ─────────────────────────────────────────────────────────────────────────────


def compute_group_flops(
    groups: list[list[str]],
    model: nn.Module,
    input_shape: tuple = (1, 3, 32, 32),
) -> list[int]:
    """Per-layer-group FLOPs estimated via a dummy forward pass with hooks.

    For Conv2d:  FLOPs = 2 * k_h * k_w * C_in * C_out * H_out * W_out
    For Linear:  FLOPs = 2 * in_features * out_features
    For BN/ReLU: 0 (negligible)

    Falls back to param count if the dummy forward raises (e.g. unexpected
    input shape). Result is guaranteed strictly positive.
    """
    module_flops: dict[str, int] = {}
    hooks = []

    def _make_hook(mod_name: str):
        def _hook(module, inp, output):
            if isinstance(module, nn.Conv2d):
                kh, kw = (
                    module.kernel_size
                    if isinstance(module.kernel_size, tuple)
                    else (module.kernel_size, module.kernel_size)
                )
                H_out = output.shape[2] if output.dim() >= 3 else 1
                W_out = output.shape[3] if output.dim() >= 4 else 1
                module_flops[mod_name] = int(
                    2
                    * kh
                    * kw
                    * module.in_channels
                    * module.out_channels
                    * H_out
                    * W_out
                )
            elif isinstance(module, nn.Linear):
                module_flops[mod_name] = int(
                    2 * module.in_features * module.out_features
                )
            else:
                module_flops[mod_name] = 0

        return _hook

    for mod_name, mod in model.named_modules():
        if isinstance(
            mod, (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d, nn.ReLU)
        ):
            hooks.append(mod.register_forward_hook(_make_hook(mod_name)))

    was_training = model.training
    model.eval()
    try:
        _device = next(model.parameters()).device
    except StopIteration:
        _device = torch.device("cpu")
    dummy = torch.zeros(input_shape, device=_device)
    try:
        with torch.no_grad():
            model(dummy)
    except Exception:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()
        sd = model.state_dict()
        return [max(sum(sd[k].numel() for k in g if k in sd), 1) for g in groups]

    for h in hooks:
        h.remove()
    if was_training:
        model.train()

    def _param_to_mod(key: str) -> str:
        parts = key.rsplit(".", 1)
        return parts[0] if (len(parts) == 2 and parts[1] in ("weight", "bias")) else key

    result = []
    for group in groups:
        seen: set[str] = set()
        total = 0
        for key in group:
            mod_name = _param_to_mod(key)
            if mod_name not in seen:
                seen.add(mod_name)
                total += module_flops.get(mod_name, 0)
        result.append(max(total, 1))
    return result


def compute_corrected_group_costs(group_flops: list[float]) -> list[float]:
    """Position-aware analytic cost: each group pays its own backward FLOPs
    plus half the FLOPs of every downstream (frozen but grad-traversed) group.

        corrected_p = group_flops[p] + 0.5 * Σ_{i > p} group_flops[i]

    Rationale: with only group p trainable, autograd must propagate grad_input
    through every layer downstream of p. The shallow groups therefore carry a
    large hidden cost that the naive group_flops[p]/sum(group_flops) formula
    misses, and the FlopCounterMode measurement makes explicit.
    """
    corrected: list[float] = []
    for p in range(len(group_flops)):
        downstream = sum(group_flops[p + 1 :])
        corrected.append(group_flops[p] + 0.5 * downstream)
    return corrected


# ─────────────────────────────────────────────────────────────────────────────
# Core measurement
# ─────────────────────────────────────────────────────────────────────────────


def _is_compiled(model: nn.Module) -> bool:
    return getattr(model, "_orig_mod", None) is not None


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def measure_fwd_bwd_flops(
    model: nn.Module,
    trainable_names: Iterable[str],
    input_shape: tuple = (1, 3, 32, 32),
    batch_size: int = 32,
    device: str = "cpu",
) -> float:
    """Forward + backward FLOPs for one mini-batch, with backward restricted
    to `trainable_names`. Result is cached per
    (arch_fingerprint, input_shape, batch_size, frozenset(trainable_names)).

    The forward pass is always full (the model owns its computation graph;
    requires_grad is what controls which parameters get a gradient, not which
    activations are computed). FlopCounterMode therefore captures:

        full_forward + backward_through_downstream + bwd_op_for_trainable

    — exactly the cost of one training step.

    Notes
    -----
    * `trainable_names` may be empty: backward then short-circuits and the
      measurement collapses to forward-only FLOPs. Useful for sanity checks
      but not a normal call path.
    * Idempotent: requires_grad state is saved and restored, gradients are
      cleared, train/eval mode is restored. No visible state mutation.
    """
    if _is_compiled(model):
        raise RuntimeError(
            "measure_fwd_bwd_flops received a torch.compile()'d model. "
            "FlopCounterMode hooks do not see ops behind inductor — pass "
            "the eager module (model._orig_mod) instead."
        )

    names_set = frozenset(trainable_names)
    key = _cache_key(model, names_set, input_shape, batch_size)
    cached = _CACHE.get(key)
    if cached is not None:
        global _HITS
        _HITS += 1
        return cached

    global _MISSES
    _MISSES += 1
    _maybe_warn_high_miss_rate()

    factor = calibrate_convention()
    saved_rg = {n: p.requires_grad for n, p in model.named_parameters()}
    was_training = model.training
    model.train()

    try:
        for n, p in model.named_parameters():
            p.requires_grad_(n in names_set)

        m_device = _model_device(model)
        x = torch.randn(
            (batch_size,) + tuple(input_shape[1:]),
            device=m_device,
            dtype=torch.float32,
        )

        with FlopCounterMode(display=False) as fc:
            out = model(x)
            loss = out.float().sum()
            try:
                loss.backward()
            except RuntimeError:
                # Degenerate case: no parameters in the trainable set, or no
                # grad path. The forward-only FLOPs are still recorded.
                pass
        flops = float(fc.get_total_flops()) * float(factor)

        # Drop gradients we just produced.
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
    finally:
        for n, p in model.named_parameters():
            p.requires_grad_(saved_rg.get(n, True))
        if not was_training:
            model.eval()

    _CACHE[key] = flops
    return flops


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher: cost_model = phi | corrected | measured
# ─────────────────────────────────────────────────────────────────────────────

VALID_COST_MODELS = ("phi", "corrected", "measured")


def round_compute_flops(
    model: nn.Module,
    trainable_names: Iterable[str],
    config: dict,
    profile,  # hardware.profiles.DeviceProfile
    dataloader: torch.utils.data.DataLoader,
    local_epochs: int,
    *,
    groups: Optional[list[list[str]]] = None,
    active_group_idx: Optional[int] = None,
    group_flops_analytic: Optional[list[float]] = None,
    beta_fraction: Optional[float] = None,
    num_primary_params: Optional[int] = None,
) -> float:
    """Per-round compute FLOPs for one client.

    The algorithm passes:
      - `trainable_names`: the actual set of parameter names that will receive
        gradients during local training (the requires_grad mask).
      - Optional legacy hints — only consulted by the "phi" branch — that
        replay the algorithm's pre-refactor formula bit-for-bit.

    cost_model is read from `config["cost_model"]` (default "measured").

    Returns
    -------
    Total compute FLOPs for one client round (per-step cost × steps).
    `steps = (dataset_size // batch_size) * local_epochs`.
    """
    cost_model = str((config or {}).get("cost_model", "measured")).lower()
    if cost_model not in VALID_COST_MODELS:
        raise ValueError(
            f"cost_model must be one of {VALID_COST_MODELS}, got {cost_model!r}."
        )

    batch_size = dataloader.batch_size
    dataset_size = len(dataloader.dataset)
    steps = (dataset_size // batch_size) * local_epochs

    # One-shot recalibration warning for the non-phi paths.
    global _RECALIBRATE_WARNED
    if cost_model != "phi" and not _RECALIBRATE_WARNED:
        warnings.warn(
            f"[flop_cost] cost_model={cost_model!r}: absolute compute FLOPs "
            f"are ~2 orders of magnitude larger than the analytic 'phi' "
            f"estimator (3*2*num_params*B*S). The energy_scale_factor "
            f"(e.g. 12.6) was calibrated against the analytic model and will "
            f"need a separate recalibration. Relative ordering between "
            f"algorithms is preserved.",
            stacklevel=2,
        )
        _RECALIBRATE_WARNED = True

    # ── PHI: bit-exact legacy reproduction ───────────────────────────────────
    if cost_model == "phi":
        num_params = sum(p.numel() for p in model.parameters())
        full_flops = profile.flops_for_model(
            num_params, batch_size, local_epochs, dataset_size
        )
        # Priority of legacy formulas matches the per-algo paths that existed
        # before this refactor; each algo passes exactly one set of hints.
        if num_primary_params is not None:
            # ccsEF (freeze_secondary): (d + 2|primary|) * 2 * B * S
            return float(2 * batch_size * steps * (num_params + 2 * num_primary_params))
        if beta_fraction is not None:
            # server_mask: (1 + beta) / 2 * full_flops
            return float(full_flops * 0.5 * (1.0 + float(beta_fraction)))
        if (
            active_group_idx is not None
            and active_group_idx >= 0
            and group_flops_analytic
        ):
            # fedpart / fedstep (PNU round)
            frac = group_flops_analytic[active_group_idx] / sum(group_flops_analytic)
            return float(full_flops * (1.0 / 3.0 + 2.0 / 3.0 * frac))
        # FedAvg, FedResonance, warmup paths → full
        return float(full_flops)

    # ── CORRECTED: full * (1/3 + 2/3 * corrected_g / sum(group_flops)) ───────
    if cost_model == "corrected":
        # 1. Contiguous-group active round → position-aware analytic formula.
        if (
            groups is not None
            and active_group_idx is not None
            and active_group_idx >= 0
            and group_flops_analytic
        ):
            num_params = sum(p.numel() for p in model.parameters())
            full_flops = profile.flops_for_model(
                num_params, batch_size, local_epochs, dataset_size
            )
            corrected = compute_corrected_group_costs(group_flops_analytic)
            denom = max(sum(group_flops_analytic), 1.0)
            frac = min(1.0, corrected[active_group_idx] / denom)
            return float(full_flops * (1.0 / 3.0 + 2.0 / 3.0 * frac))
        # 2. Full-train round (warmup or non-group algo with no partial-bwd
        #    hint) → analytic full_flops (the "frac = 1" case of the same
        #    formula: 1/3 + 2/3*1 = 1). Holds whether or not `groups` was
        #    passed alongside active_group_idx = -1.
        if (
            (active_group_idx is None or active_group_idx < 0)
            and beta_fraction is None
            and num_primary_params is None
        ):
            num_params = sum(p.numel() for p in model.parameters())
            return float(
                profile.flops_for_model(
                    num_params,
                    batch_size,
                    local_epochs,
                    dataset_size,
                )
            )
        # 3. Scattered mask / partial backward without group structure: the
        #    analytic position-aware model is not defined here. Fall through
        #    to the measured branch below — documented behaviour.

    # ── MEASURED (default) ──────────────────────────────────────────────────
    input_shape = (config or {}).get("input_shape", (1, 3, 32, 32))
    per_step = measure_fwd_bwd_flops(
        model,
        trainable_names,
        input_shape,
        batch_size,
    )
    return float(per_step * steps)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    f = calibrate_convention()
    print(f"FlopCounterMode -> true-FLOPs conversion factor: {f}")
    print(f"Cache stats: {cache_stats()}")
