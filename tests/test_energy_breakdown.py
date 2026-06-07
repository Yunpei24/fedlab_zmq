"""
tests/test_energy_breakdown.py
==============================
Guardrail for DeviceProfile.round_energy_breakdown — the single source of the
per-round energy split (compute / uplink / downlink) and of how the
energy_scale_factor (alpha) is applied.

Contract:
  1. alpha_applies_to="total" reproduces the legacy `round_energy_j(...) * alpha`
     bit-for-bit (backward compatibility).
  2. alpha_applies_to="compute" (the default for all reported results) scales
     ONLY the compute term; uplink/downlink keep their own physics.
  3. The three components always sum to "total".
  4. At alpha=1 both modes equal round_energy_j.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def profile():
    from hardware.profiles import DEVICE_PROFILES

    return DEVICE_PROFILES["esp32_s3"]


F, U, D = 1.0e9, 1_000_000, 5_000_000


def test_total_mode_reproduces_legacy(profile):
    base = profile.round_energy_j(F, U, D)
    for alpha in (1.0, 2.5, 12.6):
        bd = profile.round_energy_breakdown(F, U, D, alpha, "total")
        assert abs(bd["total"] - base * alpha) <= 1e-9 * max(base * alpha, 1.0)


def test_compute_mode_scales_compute_only(profile):
    up = profile.comm_energy_j(U, "uplink")
    dn = profile.comm_energy_j(D, "downlink")
    comp = profile.compute_energy_j(F)
    for alpha in (1.0, 5.0, 20.0):
        bd = profile.round_energy_breakdown(F, U, D, alpha, "compute")
        assert abs(bd["compute"] - comp * alpha) <= 1e-9 * max(comp * alpha, 1.0)
        assert abs(bd["uplink"] - up) <= 1e-12
        assert abs(bd["downlink"] - dn) <= 1e-12
        assert abs(bd["total"] - (comp * alpha + up + dn)) <= 1e-9 * max(bd["total"], 1.0)


def test_components_sum_to_total(profile):
    for mode in ("compute", "total"):
        bd = profile.round_energy_breakdown(F, U, D, 7.0, mode)
        assert abs((bd["compute"] + bd["uplink"] + bd["downlink"]) - bd["total"]) <= 1e-9


def test_alpha_one_equals_round_energy_j(profile):
    base = profile.round_energy_j(F, U, D)
    for mode in ("compute", "total"):
        bd = profile.round_energy_breakdown(F, U, D, 1.0, mode)
        assert abs(bd["total"] - base) <= 1e-9 * max(base, 1.0)


def test_default_is_compute(profile):
    """The default alpha_applies_to must be 'compute' (reported-results default)."""
    default = profile.round_energy_breakdown(F, U, D, 5.0)
    compute = profile.round_energy_breakdown(F, U, D, 5.0, "compute")
    assert default == compute
