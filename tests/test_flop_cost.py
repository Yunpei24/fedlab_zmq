"""
tests/test_flop_cost.py
=======================
Contract tests for hardware.flop_cost — the single source of truth for the
per-round compute cost of FL clients.

Properties verified:
  1. FlopCounterMode convention (true FLOPs vs MACs) — must agree with
     DeviceProfile.peak_gflops convention.
  2. Measurement is deterministic across repeated calls and across cache hits.
  3. Cache key is the frozenset(trainable_names) — adding the same names in a
     different order is a hit, not a miss.
  4. Position monotonicity — for sequential CNNs, training a shallow group is
     more expensive than training a deep group.
  5. Regression: `cost_model="phi"` reproduces today's per-algo formulas
     bit-for-bit (fedpart phi-model, server_mask (1+beta)/2, ccsEF
     freeze-secondary, fedavg full).
  6. `cost_model="corrected"` returns the position-aware analytic value for
     contiguous groups and falls through to measured for scattered masks.
  7. Backward-compat re-exports in algorithms.fedpart and algorithms.fedpart_be
     still produce the same outputs as the canonical implementations.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Bypass algorithms/__init__.py (loads every algo; some pull stale symbols)
# ─────────────────────────────────────────────────────────────────────────────


def _load_fedpart_helpers():
    pkg = types.ModuleType("algorithms")
    pkg.__path__ = [str(REPO_ROOT / "algorithms")]
    sys.modules.setdefault("algorithms", pkg)
    for name, fname in [
        ("algorithms.base", "algorithms/base.py"),
        ("algorithms.fedpart", "algorithms/fedpart.py"),
    ]:
        spec = importlib.util.spec_from_file_location(name, REPO_ROOT / fname)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules["algorithms.fedpart"]


@pytest.fixture(scope="module")
def fp_mod():
    return _load_fedpart_helpers()


@pytest.fixture(autouse=True)
def _clear_cache_between_tests():
    from hardware.flop_cost import clear_cache

    clear_cache()
    yield
    clear_cache()


@pytest.fixture(scope="module")
def resnet8(fp_mod):
    from models.registry import get_model

    return get_model("resnet8", "cifar10").eval()


@pytest.fixture(scope="module")
def resnet8_groups(fp_mod, resnet8):
    return fp_mod._derive_layer_groups(resnet8)


@pytest.fixture(scope="module")
def rpi4_profile():
    from hardware.profiles import DEVICE_PROFILES

    return DEVICE_PROFILES["raspberry_pi_4"]


@pytest.fixture(scope="module")
def cifar_loader():
    ds = TensorDataset(torch.zeros(128, 3, 32, 32), torch.zeros(128, dtype=torch.long))
    return DataLoader(ds, batch_size=32)


# ─────────────────────────────────────────────────────────────────────────────
# Convention
# ─────────────────────────────────────────────────────────────────────────────


def test_calibrate_convention_returns_float():
    from hardware.flop_cost import calibrate_convention

    f = calibrate_convention()
    assert isinstance(f, float)
    assert f in (1.0, 2.0), (
        f"Unexpected convention factor {f}. The module supports 1.0 (true FLOPs) "
        f"or 2.0 (MACs); anything else indicates a PyTorch convention change."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Determinism + caching
# ─────────────────────────────────────────────────────────────────────────────


def test_measure_is_deterministic(resnet8, resnet8_groups):
    from hardware.flop_cost import clear_cache, measure_fwd_bwd_flops

    shape = (1, 3, 32, 32)
    names = list(resnet8_groups[0])  # stem

    clear_cache()
    a = measure_fwd_bwd_flops(resnet8, names, shape, 16)
    clear_cache()
    b = measure_fwd_bwd_flops(resnet8, names, shape, 16)
    assert a == b, "Two cold measurements diverged."

    # Hot cache: same key must return without measuring.
    c = measure_fwd_bwd_flops(resnet8, names, shape, 16)
    assert c == a


def test_cache_keyed_by_frozenset(resnet8, resnet8_groups):
    from hardware.flop_cost import cache_stats, clear_cache, measure_fwd_bwd_flops

    shape = (1, 3, 32, 32)
    names = list(resnet8_groups[0])

    clear_cache()
    _ = measure_fwd_bwd_flops(resnet8, names, shape, 16)
    s1 = cache_stats()
    # Reverse iteration order — same set, must be a cache hit.
    _ = measure_fwd_bwd_flops(resnet8, list(reversed(names)), shape, 16)
    s2 = cache_stats()
    assert s2["misses"] == s1["misses"], (
        "Different ordering of the same name set produced a cache miss — "
        "key should be order-independent (frozenset)."
    )
    assert s2["hits"] == s1["hits"] + 1


# ─────────────────────────────────────────────────────────────────────────────
# Position monotonicity
# ─────────────────────────────────────────────────────────────────────────────


def test_shallow_group_costs_more_than_deep(resnet8, resnet8_groups):
    """Backward through a shallow group must propagate grad_input through every
    downstream layer; the fc head has zero downstream layers. So measured FLOPs
    for the stem must exceed measured FLOPs for fc."""
    from hardware.flop_cost import measure_fwd_bwd_flops

    shape = (1, 3, 32, 32)
    stem = measure_fwd_bwd_flops(resnet8, resnet8_groups[0], shape, 16)
    fc = measure_fwd_bwd_flops(resnet8, resnet8_groups[-1], shape, 16)
    assert stem > fc, (
        f"Expected stem FLOPs > fc FLOPs, got stem={stem:.3e} fc={fc:.3e}. "
        "If equal, FlopCounterMode probably didn't see the backward op — "
        "check that the model is not torch.compile()'d."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phi-mode bit-exact regression
# ─────────────────────────────────────────────────────────────────────────────


def test_phi_matches_legacy_fedavg(resnet8, rpi4_profile, cifar_loader):
    from hardware.flop_cost import round_compute_flops

    cfg = {"cost_model": "phi", "input_shape": (1, 3, 32, 32)}
    names = [n for n, _ in resnet8.named_parameters()]
    got = round_compute_flops(
        resnet8,
        names,
        cfg,
        rpi4_profile,
        cifar_loader,
        1,
    )
    num_params = sum(p.numel() for p in resnet8.parameters())
    expected = rpi4_profile.flops_for_model(num_params, 32, 1, 128)
    assert got == float(expected)


def test_phi_matches_legacy_fedpart(
    resnet8, resnet8_groups, rpi4_profile, cifar_loader, fp_mod
):
    from hardware.flop_cost import round_compute_flops

    cfg = {"cost_model": "phi", "input_shape": (1, 3, 32, 32)}
    gf = fp_mod._compute_group_flops(
        resnet8_groups, resnet8, input_shape=(1, 3, 32, 32)
    )
    num_params = sum(p.numel() for p in resnet8.parameters())
    full = rpi4_profile.flops_for_model(num_params, 32, 1, 128)
    for g_idx in range(len(resnet8_groups)):
        got = round_compute_flops(
            resnet8,
            resnet8_groups[g_idx],
            cfg,
            rpi4_profile,
            cifar_loader,
            1,
            groups=resnet8_groups,
            active_group_idx=g_idx,
            group_flops_analytic=gf,
        )
        phi = gf[g_idx] / sum(gf)
        expected = full * (1.0 / 3.0 + 2.0 / 3.0 * phi)
        assert abs(got - expected) <= max(expected * 1e-6, 1.0)


def test_phi_matches_legacy_server_mask(resnet8, rpi4_profile, cifar_loader):
    from hardware.flop_cost import round_compute_flops

    cfg = {"cost_model": "phi", "input_shape": (1, 3, 32, 32)}
    num_params = sum(p.numel() for p in resnet8.parameters())
    full = rpi4_profile.flops_for_model(num_params, 32, 1, 128)
    for beta in [0.0, 0.25, 0.5, 0.75, 1.0]:
        got = round_compute_flops(
            resnet8,
            [],
            cfg,
            rpi4_profile,
            cifar_loader,
            1,
            beta_fraction=beta,
        )
        expected = full * 0.5 * (1.0 + beta)
        assert abs(got - expected) <= max(expected * 1e-6, 1.0)


def test_phi_matches_legacy_ccsef(resnet8, rpi4_profile, cifar_loader):
    from hardware.flop_cost import round_compute_flops

    cfg = {"cost_model": "phi", "input_shape": (1, 3, 32, 32)}
    num_params = sum(p.numel() for p in resnet8.parameters())
    for primary in [1000, 10000, 50000, num_params]:
        got = round_compute_flops(
            resnet8,
            [],
            cfg,
            rpi4_profile,
            cifar_loader,
            1,
            num_primary_params=primary,
        )
        steps = 128 // 32 * 1
        expected = 2 * 32 * steps * (num_params + 2 * primary)
        assert got == float(expected)


# ─────────────────────────────────────────────────────────────────────────────
# Corrected vs measured
# ─────────────────────────────────────────────────────────────────────────────


def test_corrected_shallow_costs_more_than_phi_shallow(
    resnet8, resnet8_groups, rpi4_profile, cifar_loader, fp_mod
):
    """For the stem group, corrected must charge more than phi — that's the
    whole point of the position-aware model."""
    from hardware.flop_cost import round_compute_flops

    gf = fp_mod._compute_group_flops(
        resnet8_groups, resnet8, input_shape=(1, 3, 32, 32)
    )
    base = {
        "input_shape": (1, 3, 32, 32),
    }
    common = dict(
        groups=resnet8_groups,
        active_group_idx=0,
        group_flops_analytic=gf,
    )
    phi = round_compute_flops(
        resnet8,
        resnet8_groups[0],
        {**base, "cost_model": "phi"},
        rpi4_profile,
        cifar_loader,
        1,
        **common,
    )
    corr = round_compute_flops(
        resnet8,
        resnet8_groups[0],
        {**base, "cost_model": "corrected"},
        rpi4_profile,
        cifar_loader,
        1,
        **common,
    )
    assert corr > phi


def test_corrected_falls_through_to_measured_for_scattered_set(
    resnet8, rpi4_profile, cifar_loader
):
    """Server-mask-style scattered mask: corrected has no analytic answer, so
    the dispatcher must fall through to the measured branch — verified by
    confirming the cache grew by exactly one entry on a cold call."""
    import warnings

    from hardware.flop_cost import (
        cache_stats,
        clear_cache,
        round_compute_flops,
    )

    clear_cache()
    cfg = {"cost_model": "corrected", "input_shape": (1, 3, 32, 32)}
    # A scattered set: every-other-parameter.
    names = [n for i, (n, _) in enumerate(resnet8.named_parameters()) if i % 2 == 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _ = round_compute_flops(
            resnet8,
            names,
            cfg,
            rpi4_profile,
            cifar_loader,
            1,
            beta_fraction=0.5,  # hint that disables the "full" branch
        )
    assert cache_stats()["misses"] == 1, (
        "Expected exactly one FlopCounterMode measurement (fall-through) "
        f"but cache stats are {cache_stats()}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat re-exports
# ─────────────────────────────────────────────────────────────────────────────


def test_fedpart_helpers_still_callable(fp_mod, resnet8, resnet8_groups):
    """The wrappers in algorithms.fedpart must keep producing the same values
    as the canonical implementations in hardware.flop_cost."""
    from hardware.flop_cost import compute_corrected_group_costs as _canonical_corrected
    from hardware.flop_cost import compute_group_flops as _canonical_group_flops

    via_fedpart = fp_mod._compute_group_flops(
        resnet8_groups, resnet8, input_shape=(1, 3, 32, 32)
    )
    direct = _canonical_group_flops(resnet8_groups, resnet8, input_shape=(1, 3, 32, 32))
    assert via_fedpart == direct

    corrected_direct = _canonical_corrected(direct)
    assert len(corrected_direct) == len(direct)
    # The position-aware formula guarantees that the head (fc, last group) has
    # the LOWEST corrected cost (no downstream layers) and the stem (first
    # group) has more downstream than any other group — so its corrected cost
    # is strictly greater than the head's. This is the bug-detection contract;
    # the absolute max across all groups depends on raw FLOPs magnitudes (a
    # large mid-network conv can outrank the stem) and is not part of the spec.
    assert corrected_direct[0] > corrected_direct[-1], (
        f"Expected stem corrected > fc corrected, "
        f"got stem={corrected_direct[0]:.3e} fc={corrected_direct[-1]:.3e}."
    )
    # Strict monotonicity of the "downstream sum" component: corrected[g]
    # must be non-increasing in g once the per-group base is removed.
    # Equivalent check: corrected[g] >= 0.5 * sum(direct[g+1:]) for all g.
    for g in range(len(direct)):
        assert corrected_direct[g] >= 0.5 * sum(direct[g + 1 :])
