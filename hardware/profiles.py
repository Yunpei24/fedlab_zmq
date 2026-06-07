"""
hardware/profiles.py
====================
Realistic device profiles for federated learning simulation.

Each DeviceProfile captures:
  - Compute specs (CPU/GPU frequency, cores, precision)
  - Memory constraints
  - Communication specs (WiFi / BLE / LoRa / LTE uplink+downlink)
  - Power consumption model
  - Battery capacity
  - Optional ChannelParams for the Shannon energy model

Values sourced from:
  - Raspberry Pi datasheets (raspberrypi.com)
  - ESP32 datasheet (espressif.com)
  - NVIDIA Jetson benchmarks
  - IEEE 802.11 / 3GPP power models
  - Literature: Pereira et al. (2025), Shi et al. (2020)
  - T_Correction energy calibration (Nikiema & Amhoud, 2025)
"""

from dataclasses import dataclass, field
from typing import Optional

# ChannelParams is imported lazily to avoid circular imports;
# type annotation uses string forward-reference.
# The actual class lives in hardware.energy_model.


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ComputeSpec:
    """CPU/GPU compute characteristics."""

    name: str  # e.g. "ARM Cortex-A72"
    num_cores: int  # physical cores available to FL process
    freq_ghz: float  # operating frequency in GHz
    float_ops_per_cycle: float  # FLOPs per clock cycle per core (FP32)
    # e.g., ARM Cortex-A72: 2 FMA/cycle → 4 FLOP/cycle; GPU: much higher
    precision_bits: int = 32  # FP32 default; set 16 for FP16 devices
    has_gpu: bool = False
    gpu_tflops: Optional[float] = None  # GPU peak TFLOP/s (FP32)

    @property
    def peak_gflops(self) -> float:
        """Peak CPU compute in GFLOP/s."""
        cpu = self.num_cores * self.freq_ghz * self.float_ops_per_cycle
        if self.has_gpu and self.gpu_tflops:
            return max(cpu, self.gpu_tflops * 1000)
        return cpu


@dataclass
class PowerSpec:
    """Power consumption in Watts under different workloads."""

    idle_w: float  # idle power draw
    compute_w: float  # active training power draw
    tx_w: float  # transmit power (WiFi/cellular radio active)
    rx_w: float  # receive power
    sleep_w: float = 0.001  # deep sleep (mW range for IoT)


@dataclass
class BatterySpec:
    """Battery specification."""

    capacity_mah: float  # mAh
    voltage_v: float  # nominal voltage
    initial_soc: float = 1.0  # state of charge ∈ (0, 1]

    @property
    def capacity_j(self) -> float:
        """Total capacity in Joules."""
        return self.capacity_mah * self.voltage_v * 3.6  # mAh * V * 3.6 = J

    @property
    def initial_energy_j(self) -> float:
        return self.initial_soc * self.capacity_j


@dataclass
class CommSpec:
    """Communication link characteristics."""

    technology: str  # "WiFi6", "WiFi5", "BLE5", "LoRa", "LTE", "5G"
    uplink_mbps: float  # uplink bandwidth in Mbit/s
    downlink_mbps: float  # downlink bandwidth in Mbit/s
    latency_ms: float  # one-way latency
    packet_loss_rate: float = 0.0  # ∈ [0, 1]
    tx_power_dbm: float = 20.0  # transmit power in dBm

    @property
    def uplink_bytes_per_sec(self) -> float:
        return self.uplink_mbps * 1e6 / 8

    @property
    def downlink_bytes_per_sec(self) -> float:
        return self.downlink_mbps * 1e6 / 8


@dataclass
class DeviceProfile:
    """
    Complete hardware profile for a federated learning client.

    Usage:
        profile = DEVICE_PROFILES["raspberry_pi_4"]
        energy = profile.compute_energy(num_flops=1e9)

    Optional field:
        channel_params: hardware.energy_model.ChannelParams
            When present, hardware.energy_model.comm_energy_j() will use the
            full Shannon channel model instead of the fixed-bandwidth fallback.
            Set to None (default) to keep the original fixed-rate behaviour.
    """

    name: str
    description: str
    compute: ComputeSpec
    power: PowerSpec
    battery: BatterySpec
    comm: CommSpec
    ram_mb: int  # available RAM for model + data
    storage_mb: int  # available local storage
    # Optional Shannon channel parameters (hardware.energy_model.ChannelParams)
    channel_params: Optional[object] = field(default=None, repr=False)

    # ── Energy computation models ──────────────────────────────────────────

    def compute_energy_j(self, num_flops: float) -> float:
        """
        Energy consumed during local training (Joules).

        Model: E_comp = P_compute * T_compute
               T_compute = num_flops / (peak_gflops * 1e9)

        Ref: Shi et al. (2020), IEEE Comm. Surveys & Tutorials
        """
        peak_flops_per_sec = self.compute.peak_gflops * 1e9
        t_compute_s = num_flops / peak_flops_per_sec
        return self.power.compute_w * t_compute_s

    def comm_energy_j(self, num_bytes: float, direction: str = "uplink") -> float:
        """
        Energy consumed for wireless transmission (Joules).

        Model: E_comm = P_tx * T_tx  (uplink) ; P_tx is the transmit power in Watts
                      = P_rx * T_rx  (downlink) ; P_rx is the receive power in Watts
               T_tx = num_bytes / uplink_bytes_per_sec ; T_tx is the transmit time in seconds
               T_rx = num_bytes / downlink_bytes_per_sec ; T_rx is the receive time in seconds

        Ref: Pereira et al. (2025) LeanFed, Eq. (2)
             IEEE 802.11ax power model
        """
        if direction == "uplink":
            bps = self.comm.uplink_bytes_per_sec
            power_w = self.power.tx_w
        else:
            bps = self.comm.downlink_bytes_per_sec
            power_w = self.power.rx_w
        t_s = num_bytes / bps
        return power_w * t_s

    def round_energy_j(
        self,
        num_flops: float,
        uplink_bytes: float,
        downlink_bytes: float,
    ) -> float:
        """Total energy for one FL round (compute + uplink + downlink)."""
        return (
            self.compute_energy_j(num_flops)
            + self.comm_energy_j(uplink_bytes, "uplink")
            + self.comm_energy_j(downlink_bytes, "downlink")
        )

    def round_energy_breakdown(
        self,
        num_flops: float,
        uplink_bytes: float,
        downlink_bytes: float,
        alpha: float = 1.0,
        alpha_applies_to: str = "compute",
    ) -> dict:
        """
        Per-round energy split into {compute, uplink, downlink, total}, with the
        energy_scale_factor ``alpha`` applied per ``alpha_applies_to``:

          - "compute" (default): alpha multiplies ONLY the compute term. This is
            the physically correct reading — alpha is the compute utilization
            gap (1/sustained-utilization >= 1); a radio transmission is not
            affected by the lack of an FPU/SIMD unit. Comm keeps its own physics
            (per-device bandwidth + tx/rx power).
          - "total": alpha multiplies the whole round (compute+uplink+downlink).
            This reproduces the legacy behavior `round_energy_j(...) * alpha`
            bit-for-bit and exists ONLY for backward reproduction.

        The breakdown's "total" is what actually drains the battery, so callers
        must consume `total` (not recompute energy elsewhere) to keep survival
        dynamics consistent with the reported split.
        """
        compute = self.compute_energy_j(num_flops)
        uplink = self.comm_energy_j(uplink_bytes, "uplink")
        downlink = self.comm_energy_j(downlink_bytes, "downlink")
        if alpha_applies_to == "total":
            compute *= alpha
            uplink *= alpha
            downlink *= alpha
        else:  # "compute" — alpha is a compute-only utilization gap
            compute *= alpha
        return {
            "compute": compute,
            "uplink": uplink,
            "downlink": downlink,
            "total": compute + uplink + downlink,
        }

    def flops_for_model(
        self,
        num_params: int,
        batch_size: int,
        local_epochs: int,
        dataset_size: int,
        training_flop_factor: int = 3,
    ) -> float:
        """
        Estimate FLOPs for local training.

        Convention:
            FLOPs per step = training_flop_factor * 2 * num_params * batch_size

        The factor of 2 converts MACs to FLOPs (standard convention).
        The training_flop_factor accounts for:
            - 1× forward pass
            - 1× backward pass  (gradient computation)
            - 1× optimizer step (SGD momentum / Adam)
        Default training_flop_factor=3 follows the T_Correction energy model
        (Nikiema & Amhoud, 2025) and is consistent with Kaplan et al. (2020).

        Args:
            num_params           : model parameter count
            batch_size           : mini-batch size
            local_epochs         : local training epochs
            dataset_size         : number of training samples for this client
            training_flop_factor : fwd+bwd+update multiplier (default 3)

        Returns:
            Total FLOPs as float.
        """
        steps = (dataset_size // batch_size) * local_epochs
        flops_per_step = training_flop_factor * 2 * num_params * batch_size
        return float(flops_per_step * steps)

    def can_run_model(self, model_size_mb: float, dataset_size_mb: float) -> bool:
        """Check if device has enough RAM for model + one batch."""
        required_mb = model_size_mb * 3 + dataset_size_mb  # model + grads + optimizer
        return required_mb <= self.ram_mb

    def __repr__(self):
        return (
            f"DeviceProfile({self.name!r} | "
            f"{self.compute.peak_gflops:.1f} GFLOPS | "
            f"↑{self.comm.uplink_mbps}Mbps ↓{self.comm.downlink_mbps}Mbps | "
            f"🔋{self.battery.capacity_j:.0f}J)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Device Catalog
# Values verified against official datasheets + published FL benchmarks
# ─────────────────────────────────────────────────────────────────────────────


def _channel_params():
    """
    Lazy import of ChannelParams presets from hardware.energy_model.

    Uses a try/except to support both:
      - Package import:  from hardware.energy_model import ...  (normal usage)
      - Direct script:   from energy_model import ...           (python profiles.py)
    """
    try:
        from hardware.energy_model import (
            CHANNEL_5G,
            CHANNEL_ESP32,
            CHANNEL_ETHERNET,
            CHANNEL_LTE,
            CHANNEL_WIFI4_2G,
            CHANNEL_WIFI5_2G,
        )
    except ModuleNotFoundError:
        from energy_model import (  # fallback when run as: python hardware/profiles.py
            CHANNEL_5G,
            CHANNEL_ESP32,
            CHANNEL_ETHERNET,
            CHANNEL_LTE,
            CHANNEL_WIFI4_2G,
            CHANNEL_WIFI5_2G,
        )
    return {
        "wifi5": CHANNEL_WIFI5_2G,
        "wifi4": CHANNEL_WIFI4_2G,
        "esp32": CHANNEL_ESP32,
        "lte": CHANNEL_LTE,
        "5g": CHANNEL_5G,
        "ethernet": CHANNEL_ETHERNET,
    }


def _build_profiles() -> dict:
    """Build device profiles with Shannon channel parameters."""
    ch = _channel_params()

    return {
        # ── Raspberry Pi 4 Model B (4GB) ─────────────────────────────────────
        # SoC: BCM2711, 4× Cortex-A72 @ 1.5GHz
        # Measured idle: ~3.4W, load: ~6.4W (official RPi measurements)
        # WiFi: 802.11ac (WiFi5) dual-band
        # Shannon split: P_tx=0.20W (radiated) + P_circuit=0.80W (RF chain)
        # Total WiFi draw ≈ 1.0 W, calibrated from T_Correction measurements.
        # Legacy PowerSpec.tx_w kept at 1.0 W for backward-compat fallback.
        "raspberry_pi_4": DeviceProfile(
            name="Raspberry Pi 4B",
            description="Quad-core ARM Cortex-A72 @ 1.5GHz, 4GB LPDDR4",
            compute=ComputeSpec(
                name="BCM2711 ARM Cortex-A72",
                num_cores=4,
                freq_ghz=1.5,
                float_ops_per_cycle=4.0,  # 2 FMA → 4 FLOP/cycle (FP32, NEON)
                precision_bits=32,
            ),
            power=PowerSpec(
                idle_w=3.4,
                compute_w=6.4,
                # P_tx_total = P_tx_radiated(0.20) + P_circuit(0.80) = 1.00 W
                # Kept as tx_w for fallback compatibility; Shannon mode uses
                # channel_params which has the explicit 0.20/0.80 split.
                tx_w=1.0,
                rx_w=0.6,
            ),
            battery=BatterySpec(
                capacity_mah=10000,  # 10Ah powerbank (common setup)
                voltage_v=5.1,
                initial_soc=1.0,
            ),
            comm=CommSpec(
                technology="WiFi5",
                uplink_mbps=50.0,
                downlink_mbps=100.0,
                latency_ms=5.0,
                packet_loss_rate=0.001,
                tx_power_dbm=20.0,
            ),
            ram_mb=3900,
            storage_mb=16000,
            channel_params=ch["wifi5"],  # Shannon model: P_tx=0.20W, P_cir=0.80W
        ),
        # ── Raspberry Pi Zero 2W ──────────────────────────────────────────────
        # SoC: RP3A0, 4× Cortex-A53 @ 1.0GHz
        # Very constrained — typical IoT sensing node
        "raspberry_pi_zero2w": DeviceProfile(
            name="Raspberry Pi Zero 2W",
            description="Quad-core ARM Cortex-A53 @ 1.0GHz, 512MB LPDDR2",
            compute=ComputeSpec(
                name="RP3A0 ARM Cortex-A53",
                num_cores=4,
                freq_ghz=1.0,
                float_ops_per_cycle=2.0,  # no FP64, limited NEON
                precision_bits=32,
            ),
            power=PowerSpec(
                idle_w=0.4,
                compute_w=1.0,
                tx_w=0.60,  # P_tx(0.10) + P_circuit(0.50) = 0.60 W
                rx_w=0.35,
            ),
            battery=BatterySpec(
                capacity_mah=3000,
                voltage_v=5.0,
                initial_soc=1.0,
            ),
            comm=CommSpec(
                technology="WiFi4",
                uplink_mbps=20.0,
                downlink_mbps=40.0,
                latency_ms=8.0,
                packet_loss_rate=0.002,
            ),
            ram_mb=480,
            storage_mb=8000,
            channel_params=ch["wifi4"],
        ),
        # ── NVIDIA Jetson Nano ────────────────────────────────────────────────
        # 128-core Maxwell GPU + 4× Cortex-A57 @ 1.43GHz
        # GPU: ~472 GFLOPS (FP32)
        "jetson_nano": DeviceProfile(
            name="NVIDIA Jetson Nano",
            description="128-core Maxwell GPU + ARM Cortex-A57, 4GB LPDDR4",
            compute=ComputeSpec(
                name="Maxwell GPU + Cortex-A57",
                num_cores=4,
                freq_ghz=1.43,
                float_ops_per_cycle=4.0,
                precision_bits=32,
                has_gpu=True,
                gpu_tflops=0.472,  # 472 GFLOPS
            ),
            power=PowerSpec(
                idle_w=2.0,
                compute_w=10.0,  # 10W mode
                tx_w=1.0,
                rx_w=0.6,
            ),
            battery=BatterySpec(
                capacity_mah=20000,
                voltage_v=5.0,
                initial_soc=1.0,
            ),
            comm=CommSpec(
                technology="WiFi5",
                uplink_mbps=80.0,
                downlink_mbps=150.0,
                latency_ms=3.0,
                packet_loss_rate=0.001,
            ),
            ram_mb=3800,
            storage_mb=64000,
            channel_params=ch["wifi5"],
        ),
        # ── ESP32-S3 ──────────────────────────────────────────────────────────
        # Xtensa LX7 dual-core @ 240MHz — microcontroller class
        # Very constrained: 512KB SRAM, only tiny models
        "esp32_s3": DeviceProfile(
            name="ESP32-S3",
            description="Dual Xtensa LX7 @ 240MHz, 512KB SRAM + 8MB PSRAM",
            compute=ComputeSpec(
                name="Xtensa LX7",
                num_cores=2,
                freq_ghz=0.24,
                float_ops_per_cycle=1.0,  # basic FPU, no SIMD
                precision_bits=32,
            ),
            power=PowerSpec(
                idle_w=0.03,
                compute_w=0.50,
                tx_w=0.33,  # P_tx(0.05) + P_circuit(0.28) = 0.33 W (Espressif datasheet)
                rx_w=0.10,
            ),
            battery=BatterySpec(
                capacity_mah=1000,
                voltage_v=3.7,
                initial_soc=1.0,
            ),
            comm=CommSpec(
                technology="WiFi4",
                uplink_mbps=5.0,  # uplink_mbps represents the bandwidth of the channel
                downlink_mbps=10.0,
                latency_ms=15.0,
                packet_loss_rate=0.01,
                tx_power_dbm=15.0,
            ),
            ram_mb=8,  # 8MB PSRAM (with spiram), 512KB SRAM only otherwise
            storage_mb=16,
            channel_params=ch["esp32"],
        ),
        # ── Smartphone (mid-range Android, e.g. Samsung A54) ─────────────────
        # Exynos 1380, 8-core, with NPU
        "smartphone_midrange": DeviceProfile(
            name="Smartphone Mid-range",
            description="Octa-core SoC @ 2.4GHz, 6GB RAM, LTE+WiFi6",
            compute=ComputeSpec(
                name="Exynos 1380",
                num_cores=4,  # using efficiency cores for background FL
                freq_ghz=2.0,
                float_ops_per_cycle=8.0,  # NEON + partial GPU offload
                precision_bits=32,
                has_gpu=True,
                gpu_tflops=0.3,
            ),
            power=PowerSpec(
                idle_w=0.5,
                compute_w=3.5,
                tx_w=1.5,  # P_tx(0.25) + P_circuit(1.25) = 1.50 W (LTE uplink)
                rx_w=0.55,
            ),
            battery=BatterySpec(
                capacity_mah=5000,
                voltage_v=3.85,
                initial_soc=0.8,  # typically not fully charged
            ),
            comm=CommSpec(
                technology="LTE",
                uplink_mbps=25.0,
                downlink_mbps=80.0,
                latency_ms=20.0,
                packet_loss_rate=0.005,
            ),
            ram_mb=5500,
            storage_mb=32000,
            channel_params=ch["lte"],
        ),
        # ── Smartphone (high-end, e.g. Pixel 8 / iPhone 15) ──────────────────
        "smartphone_highend": DeviceProfile(
            name="Smartphone High-end",
            description="Octa-core @ 3.2GHz, 12GB RAM, 5G, on-device NPU",
            compute=ComputeSpec(
                name="Tensor G3 / A17",  # This represents the CPU part of the SoC, not the NPU
                num_cores=4,
                freq_ghz=3.0,
                float_ops_per_cycle=16.0,  # NPU-assisted
                precision_bits=16,  # FP16 native
                has_gpu=True,
                gpu_tflops=1.4,
            ),
            power=PowerSpec(
                idle_w=0.4,
                compute_w=5.0,
                tx_w=2.0,  # P_tx(0.50) + P_circuit(1.50) = 2.0 W (5G)
                rx_w=0.7,
            ),
            battery=BatterySpec(
                capacity_mah=4700,
                voltage_v=3.85,
                initial_soc=0.9,
            ),
            comm=CommSpec(
                technology="5G",
                uplink_mbps=100.0,
                downlink_mbps=500.0,
                latency_ms=5.0,
                packet_loss_rate=0.001,
            ),
            ram_mb=11000,
            storage_mb=128000,
            channel_params=ch["5g"],
        ),
        # ── Generic "cloud" node (for benchmark reference) ────────────────────
        "cloud_node": DeviceProfile(
            name="Cloud Node (V100 baseline)",
            description="Server-class GPU node, unconstrained energy",
            compute=ComputeSpec(
                name="NVIDIA V100",
                num_cores=1,
                freq_ghz=1.53,
                float_ops_per_cycle=1.0,
                precision_bits=32,
                has_gpu=True,
                gpu_tflops=14.0,
            ),
            power=PowerSpec(
                idle_w=50.0,
                compute_w=300.0,
                tx_w=5.0,
                rx_w=5.0,
            ),
            battery=BatterySpec(
                capacity_mah=999999,  # effectively unlimited
                voltage_v=12.0,
                initial_soc=1.0,
            ),
            comm=CommSpec(
                technology="Ethernet10G",
                uplink_mbps=10000.0,
                downlink_mbps=10000.0,
                latency_ms=0.5,
                packet_loss_rate=0.0,
            ),
            ram_mb=32000,
            storage_mb=1000000,
            # No channel_params: wired Ethernet uses fallback fixed-bandwidth model
        ),
    }  # end _build_profiles() return dict


# Module-level dict — built once at import time.
# All device profiles include channel_params for the Shannon energy model.
DEVICE_PROFILES: dict = _build_profiles()


# ─────────────────────────────────────────────────────────────────────────────
# Heterogeneous fleet factory
# ─────────────────────────────────────────────────────────────────────────────


def make_fleet(
    spec: list[tuple[str, int]],
    battery_noise_std: float = 0.1,
    battery_dist: str = "gaussian",
    battery_params: Optional[dict] = None,
    seed: Optional[int] = None,
) -> list[DeviceProfile]:
    """
    Create a heterogeneous fleet of devices with configurable battery distributions.

    Args:
        spec             : list of (device_type, count)
        battery_noise_std: std for Gaussian SOC noise (used when battery_dist="gaussian")
        battery_dist     : one of "gaussian", "weibull_soc", "uniform_soc"
        battery_params   : distribution-specific parameters (see below)
        seed             : optional RNG seed for reproducibility

    Distribution parameters (battery_params keys):
        "gaussian"     : uses battery_noise_std (legacy behaviour, default)
        "weibull_soc"  : {"kappa": 2.0, "lambda_soc": 0.7,
                          "min_soc": 0.05, "max_soc": 0.95}
            SOC ~ Weibull(shape=kappa, scale=lambda_soc), clipped to [min_soc, max_soc]
        "uniform_soc"  : {"min_soc": 0.05, "max_soc": 0.95}
            SOC ~ Uniform(min_soc, max_soc)

    Returns:
        List of DeviceProfile instances with heterogeneous initial batteries.

    Example (FedPartBE paper setup — Weibull batteries):
        fleet = make_fleet(
            [("raspberry_pi_4", 10), ("smartphone_midrange", 10),
             ("jetson_nano", 5), ("esp32_s3", 5)],
            battery_dist="weibull_soc",
            battery_params={"kappa": 2.0, "lambda_soc": 0.7,
                            "min_soc": 0.05, "max_soc": 0.95},
            seed=42,
        )
    """
    import copy

    import numpy as _np

    rng = _np.random.default_rng(seed)
    params = battery_params or {}

    fleet = []
    for device_type, count in spec:
        base = DEVICE_PROFILES[device_type]
        for _ in range(count):
            device = copy.deepcopy(base)

            if battery_dist == "weibull_soc":
                kappa = float(params.get("kappa", 2.0))
                lambda_soc = float(params.get("lambda_soc", 0.7))
                min_soc = float(params.get("min_soc", 0.05))
                max_soc = float(params.get("max_soc", 0.95))
                # numpy weibull(a) ~ Weibull(shape=a, scale=1); scale by lambda_soc
                raw = rng.weibull(kappa) * lambda_soc
                soc = float(max(min_soc, min(max_soc, raw)))

            elif battery_dist == "uniform_soc":
                min_soc = float(params.get("min_soc", 0.05))
                max_soc = float(params.get("max_soc", 0.95))
                soc = float(rng.uniform(min_soc, max_soc))

            else:  # "gaussian" — legacy default
                noise = float(rng.normal(0.0, battery_noise_std))
                soc = float(max(0.1, min(1.0, base.battery.initial_soc + noise)))

            device.battery.initial_soc = soc
            fleet.append(device)

    return fleet


def list_profiles() -> None:
    """Print available device profiles."""
    print(
        f"{'Name':<30} {'Peak GFLOPS':>12} {'RAM MB':>8} "
        f"{'↑Mbps':>8} {'↓Mbps':>8} {'Battery J':>10}"
    )
    print("─" * 80)
    for key, p in DEVICE_PROFILES.items():
        print(
            f"{p.name:<30} {p.compute.peak_gflops:>12.1f} "
            f"{p.ram_mb:>8} {p.comm.uplink_mbps:>8.1f} "
            f"{p.comm.downlink_mbps:>8.1f} "
            f"{p.battery.initial_energy_j:>10.0f}"
        )


if __name__ == "__main__":
    list_profiles()
    print()
    fleet = make_fleet([("raspberry_pi_4", 2), ("esp32_s3", 1), ("jetson_nano", 1)])
    for d in fleet:
        print(d)
