"""
hardware/energy_model.py
========================
Physics-based **communication** energy model for federated learning.

This module is the single source for wireless link energy (Shannon-capacity
channel model with Friis path loss + bandwidth-capped fallback). Compute
energy and the analytic FLOPs estimator live in `hardware/profiles.py` on
`DeviceProfile` and must not be duplicated here.

Two computation modes for COMMUNICATION:
  1. Shannon mode  — uses carrier frequency, transmit power, noise floor,
                     and path loss to derive a realistic bitrate, then
                     computes transmission time and energy from that rate.
  2. Fallback mode — uses the fixed uplink_mbps / downlink_mbps from the
                     CommSpec when Shannon parameters are absent.

Selection is automatic in `comm_energy_j`: Shannon if `profile.channel_params`
is set, fallback otherwise.

Reference:
  - Goldsmith, A. (2005). *Wireless Communications*. Cambridge University Press.
  - Pereira et al. (2025). LeanFed, Eq. (2).
  - Shi et al. (2020). IEEE Comm. Surveys & Tutorials.
  - Free-space path loss (Friis, 1946).

Authors: J. Nikiema & E. Amhoud — UM6P
"""

import math
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────

_SPEED_OF_LIGHT_M_S: float = 3.0e8          # m/s
_THERMAL_NOISE_DENSITY_W_HZ: float = 4.0e-21  # N0 = kT @ ~290 K (W/Hz)


# ─────────────────────────────────────────────────────────────────────────────
# Shannon channel parameters (added to device profiles)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChannelParams:
    """
    RF channel parameters for Shannon-capacity energy model.

    Fields:
        carrier_freq_hz   : carrier frequency in Hz  (e.g. 2.4e9 for WiFi 2.4 GHz)
        bandwidth_hz      : channel bandwidth in Hz  (e.g. 20e6 for 20 MHz WiFi channel)
        tx_power_w        : RF transmit power in Watts (radiated, PA output)
        p_circuit_w       : circuit power in Watts (PA + baseband + RF chain overhead)
        noise_density_w_hz: one-sided noise PSD N0 in W/Hz (default: kT ≈ 4e-21 W/Hz)

    Example (WiFi 5 / RPi4):
        ChannelParams(
            carrier_freq_hz=2.4e9,
            bandwidth_hz=20e6,
            tx_power_w=0.20,
            p_circuit_w=0.80,
        )

    Calibration note (T_Correction paper):
        RPi4 measured total WiFi draw ≈ 1.0 W during active TX.
        Split as: P_tx = 0.20 W (radiated) + P_circuit = 0.80 W (RF chain).
        This matches the T_Correction energy model (Nikiema & Amhoud, 2025).
    """
    carrier_freq_hz: float      = 2.4e9      # WiFi 2.4 GHz default
    bandwidth_hz: float         = 20.0e6     # 20 MHz channel
    tx_power_w: float           = 0.20       # radiated power
    p_circuit_w: float          = 0.80       # RF chain overhead
    noise_density_w_hz: float   = _THERMAL_NOISE_DENSITY_W_HZ

    # Per-device overrides for downlink (if the device is the receiver)
    rx_p_circuit_w: float       = 0.60       # RX chain power (less than TX)

    @property
    def total_tx_power_w(self) -> float:
        """Total transmission power = radiated + circuit (uplink)."""
        return self.tx_power_w + self.p_circuit_w

    @property
    def total_rx_power_w(self) -> float:
        """Total reception power (downlink side)."""
        return self.rx_p_circuit_w


# ─────────────────────────────────────────────────────────────────────────────
# Predefined channel parameter sets (calibrated per technology)
# ─────────────────────────────────────────────────────────────────────────────

# RPi4 / Jetson Nano: WiFi5 802.11ac @ 2.4 GHz band, 20 MHz channel
CHANNEL_WIFI5_2G = ChannelParams(
    carrier_freq_hz=2.4e9,
    bandwidth_hz=20.0e6,
    tx_power_w=0.20,       # P_tx (radiated)
    p_circuit_w=0.80,      # P_circuit (RF chain)
    rx_p_circuit_w=0.60,
)

# RPi Zero2W / ESP32: WiFi4 802.11n @ 2.4 GHz band, 20 MHz channel
CHANNEL_WIFI4_2G = ChannelParams(
    carrier_freq_hz=2.4e9,
    bandwidth_hz=20.0e6,
    tx_power_w=0.10,
    p_circuit_w=0.50,
    rx_p_circuit_w=0.35,
)

# ESP32 low-power: narrower effective bandwidth
CHANNEL_ESP32 = ChannelParams(
    carrier_freq_hz=2.4e9,
    bandwidth_hz=20.0e6,
    tx_power_w=0.05,       # 50 mW radiated (ESP32 max ~0.05 W @ 20 dBm → 15 dBm mode)
    p_circuit_w=0.28,      # Espressif datasheet
    rx_p_circuit_w=0.10,
)

# Smartphone LTE @ 1.8 GHz
CHANNEL_LTE = ChannelParams(
    carrier_freq_hz=1.8e9,
    bandwidth_hz=10.0e6,   # 10 MHz LTE channel
    tx_power_w=0.25,
    p_circuit_w=1.25,
    rx_p_circuit_w=0.55,
)

# Smartphone 5G @ 3.5 GHz
CHANNEL_5G = ChannelParams(
    carrier_freq_hz=3.5e9,
    bandwidth_hz=100.0e6,  # 100 MHz NR channel
    tx_power_w=0.50,
    p_circuit_w=1.50,
    rx_p_circuit_w=0.70,
)

# Ethernet (wired — no path loss, minimal overhead; fallback only)
CHANNEL_ETHERNET = ChannelParams(
    carrier_freq_hz=1.0e9,   # nominal (not used in Shannon calc for wired)
    bandwidth_hz=10.0e9,     # 10G
    tx_power_w=0.50,
    p_circuit_w=4.50,
    rx_p_circuit_w=4.50,
)


# ─────────────────────────────────────────────────────────────────────────────
# Path loss model (free-space, indoor)
# ─────────────────────────────────────────────────────────────────────────────

def friis_path_loss(distance_m: float, carrier_freq_hz: float) -> float:
    """
    Free-space path loss (Friis transmission equation).

    PL = (4 * pi * d * fc / c)^2

    This is a lower bound on path loss (no walls, reflections).
    For indoor scenarios with d = 10 m it gives a pessimistic but
    tractable SNR that is consistent with the T_Correction energy model.

    Args:
        distance_m      : link distance in metres
        carrier_freq_hz : carrier frequency in Hz

    Returns:
        Path loss (dimensionless, > 1).  Multiply by received power to
        get received signal power:  P_rx = P_tx / PL
    """
    wavelength = _SPEED_OF_LIGHT_M_S / carrier_freq_hz
    return (4.0 * math.pi * distance_m / wavelength) ** 2


# ─────────────────────────────────────────────────────────────────────────────
# Core Shannon energy computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_shannon_rate_bps(
    channel: ChannelParams,
    distance_m: float = 10.0,
) -> float:
    """
    Shannon-capacity bitrate for a given link distance.

    R = B * log2(1 + SNR)

    where:
        SNR  = P_tx / (N0 * B * PL)
        PL   = Friis path loss at distance_m

    Args:
        channel    : ChannelParams for the link
        distance_m : link distance in metres (default 10 m = nominal indoor)

    Returns:
        Achievable bitrate in bits/second.
    """
    pl = friis_path_loss(distance_m, channel.carrier_freq_hz)
    noise_power_w = channel.noise_density_w_hz * channel.bandwidth_hz
    snr = channel.tx_power_w / (noise_power_w * pl)
    snr = max(snr, 1e-9)   # numerical guard — avoid log2(0+)
    rate_bps = channel.bandwidth_hz * math.log2(1.0 + snr)
    return rate_bps


def compute_comm_energy(
    payload_bytes: float,
    channel: ChannelParams,
    direction: str = "uplink",
    distance_m: float = 10.0,
) -> float:
    """
    Energy consumed for wireless transmission using the Shannon channel model.

    Uplink (client → server):
        R    = B * log2(1 + SNR)       [bits/s]
        T_tx = payload_bits / R        [s]
        E    = (P_tx + P_circuit) * T_tx

    Downlink (server → client):
        Uses the same Shannon rate (symmetric channel assumption),
        but power is P_circuit_rx (receiver chain only — no radiated power).
        E    = P_circuit_rx * T_rx

    Args:
        payload_bytes : number of bytes to transmit
        channel       : ChannelParams (RF parameters for this link)
        direction     : "uplink" (client TX) or "downlink" (client RX)
        distance_m    : link distance in metres (default 10 m, indoor)

    Returns:
        Energy in Joules.
    """
    rate_bps = compute_shannon_rate_bps(channel, distance_m)
    payload_bits = payload_bytes * 8.0
    t_s = payload_bits / rate_bps   # transmission time in seconds

    if direction == "uplink":
        power_w = channel.total_tx_power_w   # P_tx + P_circuit
    else:
        power_w = channel.total_rx_power_w   # RX chain only

    return power_w * t_s


# ─────────────────────────────────────────────────────────────────────────────
# Unified API — works with or without ChannelParams
# ─────────────────────────────────────────────────────────────────────────────

def comm_energy_j(
    payload_bytes: float,
    profile,                          # hardware.profiles.DeviceProfile
    direction: str = "uplink",
    distance_m: float = 10.0,
) -> float:
    """
    Compute communication energy, selecting Shannon or fallback model.

    Priority:
        1. If profile has a `channel_params` attribute (ChannelParams),
           use the full Shannon model.
        2. Otherwise fall back to the fixed-bandwidth model from CommSpec
           and the PowerSpec tx_w / rx_w values.

    This function is the single entry-point for all energy accounting
    in run_experiment.py and the algorithm implementations.

    Args:
        payload_bytes : bytes to transmit / receive
        profile       : DeviceProfile (hardware.profiles)
        direction     : "uplink" or "downlink"
        distance_m    : link distance (metres); only used in Shannon mode

    Returns:
        Energy in Joules.
    """
    channel: Optional[ChannelParams] = getattr(profile, "channel_params", None)

    if channel is not None:
        return compute_comm_energy(payload_bytes, channel, direction, distance_m)
    else:
        # Fallback: fixed-bandwidth model (original profiles.py logic)
        return profile.comm_energy_j(payload_bytes, direction)


# ─────────────────────────────────────────────────────────────────────────────
# Removed: round_energy_j() and flops_for_model().
#
# These previously lived here as duplicates of DeviceProfile.round_energy_j /
# DeviceProfile.flops_for_model in hardware/profiles.py. Worse, the local
# round_energy_j applied an additional × training_flop_factor (= 3) to its
# `num_flops` argument while every caller already passed FLOPs produced by
# profile.flops_for_model — which already bakes in the × 3 factor. That made
# this function a × 9 double-counter masquerading as "the same model".
#
# Nothing in the repository called these. run_experiment.py imported
# round_energy_j as _round_energy_j but never invoked it. The algorithm
# implementations all go through profile.round_energy_j() in profiles.py.
#
# The Shannon channel model (ChannelParams, compute_shannon_rate_bps,
# compute_comm_energy, comm_energy_j) above is still the live communication
# energy path, used lazily by DeviceProfile.comm_energy_j and exposed here
# as the single entry-point. Compute energy lives exclusively in
# DeviceProfile.compute_energy_j / .round_energy_j.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check / demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))
    from hardware.profiles import DEVICE_PROFILES

    print("Shannon channel model — sanity check")
    print("=" * 60)

    # Standalone Shannon rate check for RPi4 @ 10 m
    ch = CHANNEL_WIFI5_2G
    rate = compute_shannon_rate_bps(ch, distance_m=10.0)
    print(f"Shannon rate (WiFi5 @ 10m): {rate/1e6:.1f} Mbps")

    # 1 MB uplink
    payload_mb = 1.0
    e_ul = compute_comm_energy(payload_mb * 1e6, ch, "uplink", 10.0)
    e_dl = compute_comm_energy(payload_mb * 1e6, ch, "downlink", 10.0)
    print(f"1 MB uplink energy:   {e_ul*1000:.2f} mJ")
    print(f"1 MB downlink energy: {e_dl*1000:.2f} mJ")

    print()
    print("Device comparison (1 MB up + 1 MB down, d=10m):")
    print(f"  {'Device':<28} {'Rate (Mbps)':>12} {'E_up (mJ)':>12} {'E_dn (mJ)':>12}")
    print("  " + "-" * 66)

    channel_map = {
        "raspberry_pi_4":       CHANNEL_WIFI5_2G,
        "raspberry_pi_zero2w":  CHANNEL_WIFI4_2G,
        "jetson_nano":          CHANNEL_WIFI5_2G,
        "esp32_s3":             CHANNEL_ESP32,
        "smartphone_midrange":  CHANNEL_LTE,
        "smartphone_highend":   CHANNEL_5G,
    }

    for key, ch_params in channel_map.items():
        if key in DEVICE_PROFILES:
            r = compute_shannon_rate_bps(ch_params, distance_m=10.0)
            eu = compute_comm_energy(1e6, ch_params, "uplink", 10.0)
            ed = compute_comm_energy(1e6, ch_params, "downlink", 10.0)
            profile = DEVICE_PROFILES[key]
            print(f"  {profile.name:<28} {r/1e6:>12.1f} {eu*1000:>12.2f} {ed*1000:>12.2f}")

    print()
    print("Training FLOPs check (ResNet18 proxy, 10 clients) via DeviceProfile.flops_for_model:")
    num_params = 11_173_962   # ResNet18 parameter count
    rpi = DEVICE_PROFILES["raspberry_pi_4"]
    total_flops = rpi.flops_for_model(
        num_params=num_params,
        batch_size=32,
        local_epochs=1,
        dataset_size=5000,
    )
    print(f"  Total FLOPs (factor=3): {total_flops/1e9:.2f} GFLOPs")
    print(f"  (vs factor=1 fwd-only): {total_flops/3/1e9:.2f} GFLOPs")
