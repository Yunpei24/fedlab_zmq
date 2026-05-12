"""
dashboard/app.py
================
FedLab ZMQ — Research Dashboard
Energy-Efficient Federated Learning Framework

Mohammed VI Polytechnic University (UM6P)
J. Nikiema & E. Amhoud

Launch:
    streamlit run dashboard/app.py
    # or from any directory:
    streamlit run /path/to/fedlab_zmq/dashboard/app.py
"""

import re
import streamlit as st
import pandas as pd
import json
import pathlib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def ema_smooth(values, alpha: float = 0.15):
    """
    Exponential Moving Average smoothing for convergence curves.

    lr_t = alpha * x_t + (1 - alpha) * lr_{t-1}

    alpha=0.15 → smooth trend visible, ~6-round memory.
    alpha=1.0  → no smoothing (raw values returned as-is).

    Does NOT modify the underlying data — purely for display.
    Standard practice in FL papers (FedProx, SCAFFOLD, FedNova, etc.)
    Caption should mention 'EMA smoothed (α=X) for clarity'.
    """
    arr = np.array(values, dtype=float)
    if len(arr) == 0 or alpha >= 1.0:
        return arr
    result = np.empty_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1.0 - alpha) * result[i - 1]
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FedLab ZMQ",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Typography & base ───────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hero gradient header ────────────────────────────────────────────────── */
.hero-section {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 16px;
    padding: 48px 40px;
    margin-bottom: 32px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero-accent {
    color: #38bdf8;
}
.hero-tagline {
    font-size: 1.15rem;
    color: #94a3b8;
    margin: 0 0 24px 0;
    font-weight: 400;
}
.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}
.hero-badge {
    background: rgba(56,189,248,0.15);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    font-weight: 500;
}
.hero-badge-green {
    background: rgba(52,211,153,0.15);
    border: 1px solid rgba(52,211,153,0.3);
    color: #34d399;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    font-weight: 500;
}

/* ── Section titles ──────────────────────────────────────────────────────── */
.section-title {
    font-size: 1.35rem;
    font-weight: 600;
    color: #1e293b;
    margin: 0 0 4px 0;
}
.section-subtitle {
    font-size: 0.9rem;
    color: #64748b;
    margin: 0 0 20px 0;
}

/* ── Algorithm cards ─────────────────────────────────────────────────────── */
.algo-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    height: 100%;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.algo-card:hover {
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    transform: translateY(-2px);
}
.algo-card-proposed {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border: 1.5px solid #f59e0b;
    border-radius: 12px;
    padding: 20px;
    height: 100%;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
    box-shadow: 0 2px 8px rgba(245,158,11,0.15);
}
.algo-card-proposed:hover {
    box-shadow: 0 8px 24px rgba(245,158,11,0.25);
    transform: translateY(-2px);
}
.algo-badge-proposed {
    display: inline-block;
    background: linear-gradient(135deg, #f59e0b, #d97706);
    color: white;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 3px 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    text-transform: uppercase;
}
.algo-badge-baseline {
    display: inline-block;
    background: #3b82f6;
    color: white;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    text-transform: uppercase;
}
.algo-badge-energy {
    display: inline-block;
    background: #10b981;
    color: white;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    text-transform: uppercase;
}
.algo-badge-new-proposed {
    display: inline-block;
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    color: white;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 3px 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    text-transform: uppercase;
}
.algo-card-new-proposed {
    background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
    border: 1.5px solid #a855f7;
    border-radius: 12px;
    padding: 20px;
    height: 100%;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
    box-shadow: 0 2px 8px rgba(168,85,247,0.15);
}
.algo-card-new-proposed:hover {
    box-shadow: 0 8px 24px rgba(168,85,247,0.25);
    transform: translateY(-2px);
}
.algo-name {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e293b;
    margin: 0 0 4px 0;
}
.algo-ref {
    font-size: 0.78rem;
    color: #94a3b8;
    margin: 0 0 10px 0;
    font-style: italic;
}
.algo-desc {
    font-size: 0.875rem;
    color: #475569;
    line-height: 1.5;
    margin: 0 0 12px 0;
}
.algo-key {
    font-size: 0.8rem;
    color: #64748b;
    background: #f1f5f9;
    padding: 6px 10px;
    border-radius: 6px;
    font-family: monospace;
}

/* ── Device cards ────────────────────────────────────────────────────────── */
.device-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px;
    transition: box-shadow 0.2s ease;
}
.device-card:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
}
.device-icon {
    font-size: 2rem;
    margin-bottom: 8px;
}
.device-name {
    font-size: 1rem;
    font-weight: 700;
    color: #1e293b;
    margin: 0 0 2px 0;
}
.device-category {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 0 0 10px 0;
}
.device-spec {
    font-size: 0.82rem;
    color: #475569;
    line-height: 1.7;
}
.device-spec strong {
    color: #334155;
}

/* ── Skeleton placeholder cards ─────────────────────────────────────────── */
.skeleton-card {
    background: linear-gradient(90deg, #f1f5f9 25%, #e2e8f0 50%, #f1f5f9 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 12px;
    padding: 24px;
    height: 100px;
    border: 1px solid #e2e8f0;
}
@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
.skeleton-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
    font-weight: 600;
}
.skeleton-value {
    font-size: 1.6rem;
    color: #cbd5e1;
    font-weight: 700;
}

/* ── Getting started steps ───────────────────────────────────────────────── */
.step-card {
    background: #f8fafc;
    border-left: 4px solid #38bdf8;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.step-number {
    font-size: 0.75rem;
    font-weight: 700;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.step-text {
    font-size: 0.9rem;
    color: #334155;
    margin: 0;
}
.step-code {
    font-family: 'Courier New', monospace;
    background: #1e293b;
    color: #7dd3fc;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.82rem;
}

/* ── Metric callout ──────────────────────────────────────────────────────── */
.metric-callout {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.metric-callout-value {
    font-size: 2rem;
    font-weight: 700;
    color: #0f172a;
}
.metric-callout-label {
    font-size: 0.82rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}
.metric-callout-delta-pos {
    font-size: 0.85rem;
    color: #10b981;
    font-weight: 600;
    margin-top: 4px;
}
.metric-callout-delta-neg {
    font-size: 0.85rem;
    color: #f59e0b;
    font-weight: 600;
    margin-top: 4px;
}

/* ── About page ──────────────────────────────────────────────────────────── */
.about-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
}
.about-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e293b;
    margin: 0 0 12px 0;
}
.bibtex-box {
    background: #1e293b;
    color: #7dd3fc;
    border-radius: 8px;
    padding: 16px;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    line-height: 1.7;
    overflow-x: auto;
    white-space: pre;
}

/* ── Sidebar styling ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0f172a;
}
[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── No-results banner ───────────────────────────────────────────────────── */
.no-results-banner {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 24px;
}
.no-results-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1e40af;
    margin: 0 0 8px 0;
}
.no-results-text {
    font-size: 0.9rem;
    color: #3b82f6;
    margin: 0;
    line-height: 1.5;
}

/* ── Results page summary bar ────────────────────────────────────────────── */
.results-header {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 24px;
    border: 1px solid rgba(255,255,255,0.08);
}
.results-header-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 4px 0;
}
.results-header-sub {
    font-size: 0.88rem;
    color: #94a3b8;
    margin: 0;
}

/* ── Divider with text ───────────────────────────────────────────────────── */
.section-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 32px 0 24px 0;
}
.section-divider-line {
    flex: 1;
    height: 1px;
    background: #e2e8f0;
}
.section-divider-text {
    font-size: 0.8rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    white-space: nowrap;
}

/* ── General spacing ─────────────────────────────────────────────────────── */
.spacer-sm { height: 12px; }
.spacer-md { height: 24px; }
.spacer-lg { height: 40px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants — algorithm and device metadata
# ─────────────────────────────────────────────────────────────────────────────

ALGORITHMS = [
    # ── Row 1 : Proposed + Level-1 baselines ────────────────────────────────
    {
        "name": "FedPart_BE",
        "full_name": "Federated Partial Network Updates with Battery-Energy Stratification",
        "ref": "Nikiema & EL Amhoud, UM6P 2026",
        "badge": "proposed",
        "desc": "Assigns layer groups to clients by energy tier: low-battery clients train cheap (early) layers; high-battery clients train expensive (deep) layers. Adds representation proximal regularisation and staleness-aware group priority to prevent inter-layer drift.",
        "key": "energy tiering, rep-proximal, Gauss-Seidel agg",
    },
    {
        "name": "FedAvg",
        "full_name": "Federated Averaging",
        "ref": "McMahan et al., AISTATS 2017",
        "badge": "baseline",
        "desc": "The canonical federated learning algorithm. Clients train locally for E epochs; server averages updates weighted by dataset size. No compression, full gradients.",
        "key": "local_epochs=1, weighted average",
    },
    {
        "name": "FedProx",
        "full_name": "Federated Proximal",
        "ref": "Li et al., MLSys 2020",
        "badge": "baseline",
        "desc": "Adds a proximal regularization term μ‖w − w_global‖² to the local objective to limit client drift in heterogeneous settings.",
        "key": "mu=0.01, proximal regularization",
    },
    {
        "name": "SCAFFOLD",
        "full_name": "Stochastic Controlled Averaging",
        "ref": "Karimireddy et al., ICML 2020",
        "badge": "baseline",
        "desc": "Uses client control variates to correct for client drift without requiring full device participation. Achieves linear speedup under heterogeneous data.",
        "key": "control variates, variance reduction",
    },
    # ── Row 2 : Level-2 energy-aware baselines ───────────────────────────────
    {
        "name": "FedPart",
        "full_name": "Federated Partial Network Updates",
        "ref": "Wang et al., NeurIPS 2024",
        "badge": "energy",
        "desc": "Trains and aggregates only one layer group per round (round-robin schedule). Reduces communication by 1/M and computation by ~1/3 vs FedAvg. Fixes layer mismatch via partial frozen-anchor training.",
        "key": "M=10 groups, rpl=2, sequential rotation",
    },
    {
        "name": "LeanFed",
        "full_name": "Lean Federated Learning",
        "ref": "Pereira et al., 2025",
        "badge": "energy",
        "desc": "Battery-proportional data subsampling: devices with low battery train on fewer local samples, preserving energy without discarding updates entirely.",
        "key": "subsample_ratio = B_t^k / B_max",
    },
    {
        "name": "FedBacys",
        "full_name": "Battery-Aware Cyclic Scheduling",
        "ref": "Jeong et al., 2025",
        "badge": "energy",
        "desc": "Cyclically schedules clients based on battery state, activating only devices above an energy threshold per round. Maximizes device longevity.",
        "key": "cyclic scheduling, threshold activation",
    },
    {
        "name": "Vaishnav",
        "full_name": "Channel-Adaptive TopK + EF",
        "ref": "Vaishnav et al., 2024",
        "badge": "energy",
        "desc": "Adapts gradient sparsity based on wireless channel quality rather than battery. Devices with poor channels compress more to avoid retransmissions.",
        "key": "channel-aware TopK, error feedback",
    },
    {
        "name": "FedSparQ",
        "full_name": "Federated Sparse Quantization",
        "ref": "Medjadji et al., 2025",
        "badge": "energy",
        "desc": "Combines EMA-based dynamic sparsity threshold with FP16 quantization and error feedback. Delivers significant communication savings on bandwidth-limited devices.",
        "key": "EMA threshold + fp16 + EF",
    },
    {
        "name": "E-CEFFL",
        "full_name": "Energy-Constrained Error-Feedback Federated Learning",
        "ref": "Nikiema, EL Amhoud, ELHAMMOUTI & KISSAMI, UM6P 2025",
        "badge": "energy",
        "desc": "Adaptive sparsification driven by remaining battery: β_t^k = β_min + (β_max − β_min) · B_t^k / B_max. Combines TopK gradient compression with error feedback for convergence guarantees.",
        "key": "beta_min=0.01, lr=0.01, TopK+EF",
    },
    # ── Row 3 : New proposed algorithms — UM6P 2026 ──────────────────────────
    {
        "name": "Fed-Resonance",
        "full_name": "Battery-Aware Adaptive Spectral Compression",
        "ref": "Nikiema & Amhoud, UM6P 2026",
        "badge": "new_proposed",
        "desc": "Per-layer adaptive switching between truncated SVD, subspace projection, and dense transmission. Rank adapts to preserve ε=90% gradient energy; battery-critical clients force SVD mode.",
        "key": "ε=0.90, τ_rank=0.85, error feedback",
    },
    {
        "name": "Fed-Osmosis",
        "full_name": "Thermodynamic Distribution Alignment",
        "ref": "Nikiema & Amhoud, UM6P 2026",
        "badge": "new_proposed",
        "desc": "Adds an osmotic pressure regularizer (KL divergence from local to global activation distribution) to the local training loss. Aligns feature representations across heterogeneous clients without extra communication.",
        "key": "γ=0.1, dense gradients + activation stats",
    },
    {
        "name": "Fed-Resonance-Osmosis",
        "full_name": "Spectral-Thermodynamic Hybrid FL",
        "ref": "Nikiema & Amhoud, UM6P 2026",
        "badge": "new_proposed",
        "desc": "Two-stage pipeline: (1) osmotic training aligns local feature distributions; (2) resonance spectral compression transmits the resulting low-rank gradients. The two stages reinforce each other.",
        "key": "γ=0.1 + ε=0.90 + EF, adaptive rank",
    },
]

DEVICES = [
    {
        "icon": "🔴",
        "name": "Raspberry Pi 4B",
        "category": "Edge SBC",
        "cpu": "ARM Cortex-A72 @ 1.5 GHz (4 cores)",
        "ram": "4 GB LPDDR4",
        "comm": "WiFi 802.11ac / Gigabit Ethernet",
        "battery": "10,000 mAh @ 5.1 V → 183,600 J",
        "power": "3.5 W idle / 6.4 W compute",
        "color": "#ef4444",
    },
    {
        "icon": "🟢",
        "name": "Jetson Nano",
        "category": "Edge GPU",
        "cpu": "ARM Cortex-A57 @ 1.43 GHz + Maxwell GPU",
        "ram": "4 GB LPDDR4",
        "comm": "WiFi 802.11ac",
        "battery": "20,000 mAh @ 5 V → 360,000 J",
        "power": "2.0 W idle / 10.0 W GPU compute",
        "color": "#22c55e",
    },
    {
        "icon": "🟡",
        "name": "Raspberry Pi Zero 2W",
        "category": "Ultra-Low Power SBC",
        "cpu": "ARM Cortex-A53 @ 1.0 GHz (4 cores)",
        "ram": "512 MB LPDDR2",
        "comm": "WiFi 802.11b/g/n",
        "battery": "5,000 mAh @ 5 V → 90,000 J",
        "power": "0.4 W idle / 1.5 W compute",
        "color": "#eab308",
    },
    {
        "icon": "🔵",
        "name": "ESP32",
        "category": "Microcontroller (IoT)",
        "cpu": "Xtensa LX6 @ 240 MHz (dual core)",
        "ram": "520 KB SRAM",
        "comm": "WiFi 802.11b/g/n / Bluetooth 4.2",
        "battery": "1,000 mAh @ 3.7 V → 13,320 J",
        "power": "0.03 W idle / 0.16 W active",
        "color": "#3b82f6",
    },
    {
        "icon": "🟣",
        "name": "Smartphone (Mid)",
        "category": "Mobile Device",
        "cpu": "ARM Cortex-A75 @ 2.0 GHz (8 cores)",
        "ram": "4 GB LPDDR4X",
        "comm": "LTE Cat.12 / WiFi 802.11ac",
        "battery": "3,500 mAh @ 3.85 V → 48,510 J",
        "power": "0.3 W idle / 2.8 W compute",
        "color": "#8b5cf6",
    },
    {
        "icon": "⚫",
        "name": "Smartphone (High)",
        "category": "Mobile Device",
        "cpu": "ARM Cortex-A78 @ 2.8 GHz (8 cores) + GPU",
        "ram": "8 GB LPDDR5",
        "comm": "5G NR / WiFi 6 (802.11ax)",
        "battery": "5,000 mAh @ 3.85 V → 69,300 J",
        "power": "0.5 W idle / 5.0 W compute",
        "color": "#374151",
    },
]

COLOR_MAP = [
    "#f59e0b",  # amber      — E-CEFFL
    "#3b82f6",  # blue       — FedAvg
    "#8b5cf6",  # purple     — FedProx
    "#10b981",  # green      — SCAFFOLD
    "#ef4444",  # red        — LeanFed
    "#06b6d4",  # cyan       — FedBacys
    "#f97316",  # orange     — Vaishnav
    "#6366f1",  # indigo     — FedSparQ
    "#a855f7",  # violet     — Fed-Resonance
    "#14b8a6",  # teal       — Fed-Osmosis
    "#ec4899",  # pink       — Fed-Resonance-Osmosis
]

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_DASHBOARD_DIR  = pathlib.Path(__file__).parent
_PROJECT_ROOT   = _DASHBOARD_DIR.parent
_DEFAULT_RESULTS = _PROJECT_ROOT / "results"
if not _DEFAULT_RESULTS.exists():
    _DEFAULT_RESULTS = _DASHBOARD_DIR / "results"

# Allow overriding via ?results_dir=... query param or sidebar input (set below)
_qp = st.query_params.get("results_dir", None)
RESULTS_DIR = pathlib.Path(_qp) if _qp else _DEFAULT_RESULTS

# ─────────────────────────────────────────────────────────────────────────────
# Data loading (preserved from previous session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def _get_exp_label(d: pathlib.Path, results_dir: pathlib.Path) -> str:
    """
    Generate a descriptive label for an experiment directory by reading its metrics.json.

    Format: [algo] <folder_name> | param1=v1, param2=v2, ...

    The folder_name is the parent directory of the metrics.json parent
    (i.e. the "named" experiment folder, e.g. ccsEF_e3_full or fedavg_e3).
    Key params are algorithm-specific.
    """
    metrics_path = d / "metrics.json"

    # Determine the "named" folder (parent of the run dir)
    try:
        rel = str(d.relative_to(results_dir))
        path_parts = rel.replace("\\", "/").split("/")
        folder_name = path_parts[-2] if len(path_parts) >= 2 else path_parts[0]
    except ValueError:
        folder_name = d.parent.name if d.parent != d else d.name

    if not metrics_path.exists():
        return folder_name

    try:
        with open(metrics_path, "r") as f:
            data = json.load(f)

        cfg  = data.get("config", {})
        algo = cfg.get("algorithm", data.get("algorithm", "???"))

        # algo_config holds the hyperparams (nested under config in server format)
        ac = cfg.get("algo_config", cfg)

        # ── Algorithm-specific param display ──────────────────────────────────
        algo_lower = (algo or "").lower()
        param_parts = []

        if "ccsEf" in algo or "ccsef" in algo_lower:
            E     = ac.get("local_epochs", cfg.get("local_epochs", "?"))
            T_rot = ac.get("T_rot", "?")
            frz   = ac.get("freeze_secondary_backward", False)
            clip  = ac.get("max_grad_norm", None)
            lr    = ac.get("lr", cfg.get("lr", "?"))
            param_parts.append(f"E={E}")
            param_parts.append(f"T_rot={T_rot}")
            if frz:
                param_parts.append("frozen")
            if clip is not None:
                param_parts.append(f"clip={clip}")
            param_parts.append(f"lr={lr}")

        elif "fedavg" in algo_lower:
            E  = ac.get("local_epochs", cfg.get("local_epochs", "?"))
            lr = ac.get("lr", cfg.get("lr", "?"))
            param_parts.append(f"E={E}")
            param_parts.append(f"lr={lr}")

        elif "fedpart" in algo_lower:
            M    = ac.get("num_tiers", "?")
            mu_r = ac.get("mu_repr", None)
            mu_w = ac.get("mu_weight", None)
            strat = ac.get("assignment_strategy", "")
            E    = ac.get("local_epochs", cfg.get("local_epochs", "?"))
            param_parts.append(f"M={M}")
            param_parts.append(f"E={E}")
            if mu_r is not None: param_parts.append(f"μr={mu_r}")
            if mu_w is not None: param_parts.append(f"μw={mu_w}")
            if strat: param_parts.append(strat)

        elif "eceffl" in algo_lower or "e-ceffl" in algo_lower:
            E      = ac.get("local_epochs", cfg.get("local_epochs", "?"))
            b_min  = ac.get("beta_min", "?")
            lr     = ac.get("lr", cfg.get("lr", "?"))
            param_parts.append(f"E={E}")
            param_parts.append(f"β_min={b_min}")
            param_parts.append(f"lr={lr}")

        elif "fedsparq" in algo_lower or "sparq" in algo_lower:
            E  = ac.get("local_epochs", cfg.get("local_epochs", "?"))
            lr = ac.get("lr", cfg.get("lr", "?"))
            param_parts.append(f"E={E}")
            param_parts.append(f"lr={lr}")

        else:
            # Generic fallback: show E and lr if present
            E  = ac.get("local_epochs", cfg.get("local_epochs", None))
            lr = ac.get("lr", cfg.get("lr", None))
            if E  is not None: param_parts.append(f"E={E}")
            if lr is not None: param_parts.append(f"lr={lr}")

        params_str = ", ".join(param_parts)
        label = f"[{algo}] {folder_name}"
        if params_str:
            label += f" | {params_str}"
        return label

    except Exception:
        return folder_name


_GLABEL_RE_ALGO = re.compile(r'\[([^\]]+)\]')
_GLABEL_RE_M    = re.compile(r'\bM=(\S+?)(?:[,\s\(|]|$)')

def _short_graph_label(full_label: str, max_len: int = 36) -> str:
    """
    Concise legend/axis label for graph traces.
    '[FedPartBE], M=3, μr=0.1 | rotation_30/run_001'
    → 'FedPartBE · M=3 · rotation_30'

    Components:
      - Algorithm name from [AlgoName]
      - M= (num_tiers) if present
      - rotation_period = name of the second-to-last folder
        (the folder containing the folder that directly holds metrics.json),
        extracted from the path suffix appended after '|'
    Falls back gracefully when brackets or suffix are absent.
    """
    # ── Extract rotation_period from the path suffix (part after '|') ─────────
    halves = full_label.split("|", 1)
    rotation = ""
    if len(halves) == 2:
        suffix_parts = halves[1].strip().split("/")
        if len(suffix_parts) >= 2:
            rotation = suffix_parts[0]   # avant-dernier dossier

    # ── Build the short label ──────────────────────────────────────────────────
    m_algo = _GLABEL_RE_ALGO.match(full_label.strip())
    if not m_algo:
        base = halves[0].strip()[:max_len]
        label = f"{base} · {rotation}" if rotation else base
    else:
        algo = m_algo.group(1)
        # Extract the folder name (2nd folder up from metrics.json) sitting after [algo] in halves[0]
        folder_part = halves[0].split(']', 1)[-1].strip() if ']' in halves[0] else ""
        m_val = _GLABEL_RE_M.search(full_label)
        label = algo
        if folder_part:
            label += f" · {folder_part}"
        if m_val:
            label += f" · M={m_val.group(1)}"
        if rotation:
            label += f" · {rotation}"

    return label if len(label) <= max_len else label[:max_len - 1] + "…"


@st.cache_data
def load_experiment(exp_dir: str) -> tuple[dict, pd.DataFrame]:
    # exp_dir may be a relative name (old behaviour) or an absolute path
    p = pathlib.Path(exp_dir)
    if p.is_absolute() and p.exists():
        path = p if p.name == "metrics.json" else p / "metrics.json"
    else:
        path = RESULTS_DIR / exp_dir / "metrics.json"
    with open(path) as f:
        data = json.load(f)

    df = pd.DataFrame(data["rounds"])

    # Flatten algo_metrics column if present (ZMQ framework writes this)
    if "algo_metrics" in df.columns:
        algo_df = pd.json_normalize(df["algo_metrics"])
        df = pd.concat([df.drop("algo_metrics", axis=1), algo_df], axis=1)

    # Normalise column names
    if "total_bytes" not in df.columns and "total_bytes_sent" in df.columns:
        df["total_bytes"] = df["total_bytes_sent"]

    # Extract experiment-level metadata
    if "experiment" in data:
        experiment_meta = data["experiment"]
    else:
        experiment_meta = {
            "algorithm": data.get("algorithm", exp_dir),
            "config":    data.get("config", {}),
            "summary":   data.get("summary", {}),
        }
        experiment_meta["training"] = {
            "algorithm": data.get("algorithm", exp_dir)
        }

    return experiment_meta, df


def _round_col(df: pd.DataFrame) -> pd.Series:
    for col in ("round_num", "round", "t"):
        if col in df.columns:
            return df[col]
    return pd.Series(range(1, len(df) + 1))


def _fmt_seconds(seconds) -> str:
    """Format a duration in seconds as 'Xs' or 'Xmin Ys'."""
    if seconds is None:
        return "—"
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "—"
    if s < 60:
        return f"{s:.1f}s"
    minutes = int(s // 60)
    rem = s - minutes * 60
    return f"{minutes}min {rem:.0f}s"


def _get_result_dirs():
    root = RESULTS_DIR
    if not root.exists():
        return []
    # rglob scans all depths — supports results/exp/, results/group/run/exp/, etc.
    return sorted(
        [p.parent for p in root.rglob("metrics.json")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

result_dirs = _get_result_dirs()
n_experiments = len(result_dirs)

with st.sidebar:
    st.markdown(
        '<div style="padding: 4px 0 20px 0;">'
        '<div style="font-size:1.3rem; font-weight:700; color:#f8fafc; letter-spacing:-0.5px;">'
        '⚡ FedLab ZMQ'
        '</div>'
        '<div style="font-size:0.78rem; color:#64748b; margin-top:2px;">'
        'Energy-Efficient FL Framework'
        '</div>'
        '<div style="font-size:0.72rem; color:#475569; margin-top:6px;">'
        'UM6P — J. Nikiema, EL Amhoud, H. ELHAMMOUTI &amp; I. KISSAMI'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Results root folder picker ────────────────────────────────────────────
    with st.expander("Results folder", expanded=False):
        custom_dir = st.text_input(
            "Root results directory",
            value=str(RESULTS_DIR),
            help="Absolute path to the folder that contains your experiments. "
                 "Scanned recursively — nested subdirectories are found automatically.",
        )
        if st.button("Load", width="stretch"):
            p = pathlib.Path(custom_dir)
            if p.exists():
                st.query_params["results_dir"] = str(p)
                st.rerun()
            else:
                st.error(f"Path not found: {custom_dir}")
        st.caption(f"Currently scanning: `{RESULTS_DIR}`")

    st.divider()

    # Navigation
    page = st.radio(
        "Navigation",
        ["Home", "Results", "Compare Algorithms", "Survival & Fairness"],
        label_visibility="collapsed",
        help="Results: single-experiment view. Compare: multi-algorithm benchmarks. Survival: battery & fairness.",
    )

    st.divider()

    # Experiment status
    if n_experiments == 0:
        st.markdown("""
        <div style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3);
                    border-radius:8px; padding:12px; text-align:center;">
          <div style="font-size:1.4rem; font-weight:700; color:#fca5a5;">0</div>
          <div style="font-size:0.72rem; color:#94a3b8; text-transform:uppercase;
                      letter-spacing:0.5px; margin-top:2px;">Experiments Loaded</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.78rem; color:#475569; margin-top:12px; line-height:1.6;">
        Run your first experiment:<br>
        <code style="background:#1e293b; color:#7dd3fc; padding:2px 6px;
               border-radius:4px; font-size:0.75rem;">
        python run_experiment.py --algo fedpart_be --rounds 10 --clients 4
        </code>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3);
                    border-radius:8px; padding:12px; text-align:center;">
          <div style="font-size:1.4rem; font-weight:700; color:#6ee7b7;">{n_experiments}</div>
          <div style="font-size:0.72rem; color:#94a3b8; text-transform:uppercase;
                      letter-spacing:0.5px; margin-top:2px;">
            Experiment{'s' if n_experiments > 1 else ''} Available
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Experiment selector (shown on relevant pages)
    selected_dirs = []
    if page in ("Results", "Compare Algorithms", "Survival & Fairness") and result_dirs:
        # Build display labels: path relative to RESULTS_DIR for readability
        # Build display labels: lookup data from metrics.json for personalisation
        dir_labels = {
            _get_exp_label(d, RESULTS_DIR): str(d)
            for d in result_dirs
        }
        selected_labels = st.multiselect(
            "SELECT EXPERIMENTS",
            options=list(dir_labels.keys()),
            default=None,
            help="Select one or more experiment result folders to visualize. "
                 "Labels are personalized from metrics.json.",
        )
        selected_dirs = [dir_labels[lbl] for lbl in selected_labels]

    st.markdown(
        '<div style="padding:16px 0 4px 0; font-size:0.7rem; color:#475569; line-height:1.6;">'
        'Mohammed VI Polytechnic University<br>Benguerir, Morocco'
        '</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Page: HOME
# ─────────────────────────────────────────────────────────────────────────────

if page == "Home":

    # ── Hero ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-section">
      <div class="hero-title">
        FedLab <span class="hero-accent">ZMQ</span> — Energy-Efficient Federated Learning
      </div>
      <div class="hero-tagline">
        Research framework for battery-aware FL algorithms on heterogeneous IoT fleets.
        Two papers in preparation — UM6P 2026.
      </div>
      <div class="hero-badges">
        <span class="hero-badge">ZeroMQ Transport</span>
        <span class="hero-badge">msgpack Serialization</span>
        <span class="hero-badge">Shannon Energy Model</span>
        <span class="hero-badge-green">CCS-EF — Paper 1</span>
        <span class="hero-badge-green">FedPartBE — Paper 2</span>
        <span class="hero-badge">Jain Fairness Index</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Research context ─────────────────────────────────────────────────────
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown("""
        <div class="metric-callout">
          <div class="metric-callout-value">2</div>
          <div class="metric-callout-label">Papers in preparation</div>
          <div class="metric-callout-delta-pos">CCS-EF · FedPartBE</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="metric-callout">
          <div class="metric-callout-value">30</div>
          <div class="metric-callout-label">IoT clients (ESP32-S3)</div>
          <div class="metric-callout-delta-pos">SOC [5%, 95%] — Dirichlet α=0.5</div>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="metric-callout">
          <div class="metric-callout-value">78.9%</div>
          <div class="metric-callout-label">Best acc — CCS-EF Full</div>
          <div class="metric-callout-delta-pos">T_rot=3 · CIFAR-10 · rd 165</div>
        </div>
        """, unsafe_allow_html=True)
    with col_d:
        st.markdown("""
        <div class="metric-callout">
          <div class="metric-callout-value">−60%</div>
          <div class="metric-callout-label">Bytes vs FedAvg</div>
          <div class="metric-callout-delta-pos">0.61 GB vs 1.54 GB · same survival</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)

    # ── Two papers overview ───────────────────────────────────────────────────
    st.markdown("""
    <div class="section-divider">
      <div class="section-divider-line"></div>
      <div class="section-divider-text">Research Papers</div>
      <div class="section-divider-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("""
        <div class="algo-card-new-proposed" style="min-height:220px;">
          <div><span class="algo-badge-new-proposed">✦ Paper 1 — In preparation</span></div>
          <div class="algo-name">CCS-EF</div>
          <div class="algo-ref">Complementary Cluster Split with Error Feedback</div>
          <div class="algo-desc">
            Splits clients into two complementary clusters (C1/C2) that alternate primary
            layer group every T_rot rounds. Error feedback ensures convergence with biased
            top-K compression. Partial backward option (freeze_secondary) reduces E_comp ~33%.
          </div>
          <div class="algo-key">
            Best: <b>78.90%</b> (T_rot=3) · −60% bytes · 19/30 alive rd200 · Jain=0.633
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col_p2:
        st.markdown("""
        <div class="algo-card-proposed" style="min-height:220px;">
          <div><span class="algo-badge-proposed">⭐ Paper 2 — In preparation</span></div>
          <div class="algo-name">FedPartBE</div>
          <div class="algo-ref">Federated Partial Training with Battery-Aware Energy Tiering</div>
          <div class="algo-desc">
            Assigns clients to energy tiers (M=3) based on remaining battery. Low-battery
            clients train only cheap early layers; high-battery clients train deep layers.
            Extends device lifetime and ensures universal participation via energy-proportional
            layer assignment.
          </div>
          <div class="algo-key">
            Extends device lifetime · FedAvg baseline · Jain fairness · SOC-aware scheduling
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)

    # ── Current best preliminary results ─────────────────────────────────────
    st.markdown("""
    <div class="section-divider">
      <div class="section-divider-line"></div>
      <div class="section-divider-text">Preliminary Results — seed=42, CIFAR-10, ResNet-8, 30 clients</div>
      <div class="section-divider-line"></div>
    </div>
    """, unsafe_allow_html=True)

    prelim_data = {
        "Algorithm":    ["FedAvg (E=3)", "CCS-EF Full T_rot=1", "CCS-EF Frozen T_rot=1",
                         "CCS-EF Full T_rot=3 ✦", "CCS-EF Frozen T_rot=3 ✦"],
        "Best Acc":     ["81.63%", "76.35%", "76.07%", "78.90%", "78.63%"],
        "Alive rd200":  ["19/30", "19/30", "22/30", "19/30", "22/30"],
        "1st death":    ["rd 37", "rd 39", "rd 56", "rd 39", "rd 55"],
        "Energy (kJ)":  ["155.9", "151.3", "113.2", "151.3", "113.0"],
        "Bytes (GB)":   ["1.544", "0.613", "0.684", "0.612", "0.682"],
        "Jain":         ["0.633", "0.633", "0.733", "0.633", "0.733"],
    }
    import pandas as _pd
    st.dataframe(_pd.DataFrame(prelim_data), width="stretch", hide_index=True)
    st.caption("✦ T_rot=3 + EF buffer flush fix + gradient clipping (max_grad_norm=10) — seed=42 only, not final.")

    # ── Quick links ───────────────────────────────────────────────────────────
    st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-divider">
      <div class="section-divider-line"></div>
      <div class="section-divider-text">Run commands</div>
      <div class="section-divider-line"></div>
    </div>
    """, unsafe_allow_html=True)
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("""
        **CCS-EF Full (T_rot=3)**
        ```bash
        python run_experiment.py \\
          --config configs/ccsEF_soc_wide.yaml \\
          --algo ccsEF \\
          --output results/Algorithms_fare/ccsEF_Trot3_full \\
          --device mps
        ```
        **CCS-EF Frozen (T_rot=3)**
        ```bash
        python run_experiment.py \\
          --config configs/ccsEF_frozen.yaml \\
          --algo ccsEF \\
          --output results/Algorithms_fare/ccsEF_Trot3_frozen \\
          --device mps
        ```
        """)
    with col_r2:
        st.markdown("""
        **FedAvg baseline**
        ```bash
        python run_experiment.py \\
          --config configs/ccsEF_soc_wide.yaml \\
          --algo fedavg \\
          --output results/Algorithms_fare/fedavg_e3 \\
          --device mps
        ```
        **FedPartBE**
        ```bash
        python run_experiment.py \\
          --config configs/algo_comparison.yaml \\
          --algo fedpart_be \\
          --output results/Algorithms_fare/fedpart_be \\
          --device mps
        ```
        """)

# ─────────────────────────────────────────────────────────────────────────────
# Page: RESULTS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Results":

    st.markdown("""
    <div class="results-header">
      <div class="results-header-title">Experiment Results</div>
      <div class="results-header-sub">
        Training curves, energy consumption, battery evolution, and fairness metrics.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not result_dirs:
        st.markdown("""
        <div class="no-results-banner">
          <div class="no-results-title">No experiments found</div>
          <div class="no-results-text">
            Run <code style="background:rgba(59,130,246,0.1); padding:2px 6px;
            border-radius:4px;">python run_experiment.py --algo fedpart_be --rounds 10 --clients 4</code>
            to generate your first results. The dashboard will auto-detect them here.
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    if not selected_dirs:
        st.info("Select at least one experiment from the sidebar to view results.")
        st.stop()

    # Load selected experiments
    experiments = {}
    for name in selected_dirs:
        config, df = load_experiment(name)
        # Use personalized label from metrics.json helper
        label = _get_exp_label(pathlib.Path(name), RESULTS_DIR)
        experiments[label] = {"config": config, "df": df}

    # Summary metrics bar
    sum_cols = st.columns(len(experiments))
    for col, (label, exp) in zip(sum_cols, experiments.items()):
        df = exp["df"]
        best_acc  = df["test_accuracy"].max() * 100 if "test_accuracy" in df.columns else 0.0
        total_gb  = df["total_bytes"].sum() / 1e9   if "total_bytes" in df.columns else 0.0
        final_j   = df["jain_index"].iloc[-1]        if "jain_index" in df.columns and len(df) > 0 else 0.0
        col.metric(label, f"{best_acc:.1f}%", f"{total_gb:.3f} GB | J={final_j:.3f}")

    st.divider()

    tab_acc, tab_energy, tab_battery, tab_fair, tab_table, tab_convspeed, tab_lm = st.tabs([
        "Accuracy & Loss",
        "Energy per Round",
        "Battery & Survival",
        "Bytes & Fairness",
        "Summary Table",
        "Convergence Speed",
        "Layer Mismatch",
    ])

    # ── Accuracy & Loss ──────────────────────────────────────────────────────
    with tab_acc:
        # ── Smoothing controls ────────────────────────────────────────────────
        _sc1, _sc2 = st.columns([1, 3])
        with _sc1:
            _smooth_on = st.toggle("EMA smoothing", value=True,
                                   help="Exponential Moving Average — shows trend without changing data. "
                                        "Raw curve shown transparently in background.")
        with _sc2:
            _ema_alpha = st.slider("α (smoothing strength)",
                                   min_value=0.05, max_value=1.0, value=0.15, step=0.05,
                                   disabled=not _smooth_on,
                                   help="α=0.05 → very smooth (30-round memory). "
                                        "α=0.30 → light smoothing. α=1.0 → raw data.")

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Accuracy (%)", "Loss"])
        for i, (label, exp) in enumerate(experiments.items()):
            df = exp["df"]
            color = COLOR_MAP[i % len(COLOR_MAP)]
            rounds = _round_col(df)
            legend_label = _short_graph_label(label)

            if "test_accuracy" in df.columns:
                y_raw = df["test_accuracy"] * 100
                y_plot = ema_smooth(y_raw, _ema_alpha) if _smooth_on else y_raw
                if _smooth_on:
                    # Raw curve: thin + transparent background
                    fig.add_trace(go.Scatter(
                        x=rounds, y=y_raw,
                        line=dict(color=color, width=0.8),
                        opacity=0.2, showlegend=False, hoverinfo="skip",
                    ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=rounds, y=y_plot,
                    name=legend_label, line=dict(color=color, width=2.5),
                    showlegend=True,
                ), row=1, col=1)

            if "test_loss" in df.columns:
                y_raw = df["test_loss"]
                y_plot = ema_smooth(y_raw, _ema_alpha) if _smooth_on else y_raw
                if _smooth_on:
                    fig.add_trace(go.Scatter(
                        x=rounds, y=y_raw,
                        line=dict(color=color, width=0.8),
                        opacity=0.2, showlegend=False, hoverinfo="skip",
                    ), row=1, col=2)
                fig.add_trace(go.Scatter(
                    x=rounds, y=y_plot,
                    name=legend_label, line=dict(color=color, width=2.5),
                    showlegend=False,
                ), row=1, col=2)

        fig.update_xaxes(title_text="Round")
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        fig.update_layout(
            height=450, template="plotly_white",
            legend=dict(
                orientation="v",
                xanchor="left", x=1.02,
                yanchor="middle", y=0.5,
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#e2e8f0", borderwidth=1,
            ),
            font=dict(family="Inter"),
            margin=dict(r=160),
        )
        st.plotly_chart(fig, width="stretch")
        if _smooth_on:
            st.caption(
                f"EMA smoothed (α={_ema_alpha:.2f}) — raw values shown transparently. Underlying data unchanged."
            )
        st.caption(
            "**How to read:** X-axis = communication round (one global aggregation). "
            "**Accuracy** = fraction of the test set correctly classified (higher = better). "
            "**Loss** = cross-entropy on the test set (lower = better). "
            "A flat or rising loss with falling accuracy = divergence. "
            "Each colour = one experiment (algorithm / configuration)."
        )

        # Beta dynamics (E-CEFFL specific)
        if any("avg_beta" in exp["df"].columns for exp in experiments.values()):
            st.markdown("**Sparsification ratio β over rounds** (lower = more compressed gradient)")
            fig_b = go.Figure()
            for i, (label, exp) in enumerate(experiments.items()):
                df = exp["df"]
                if "avg_beta" in df.columns:
                    fig_b.add_trace(go.Scatter(
                        x=_round_col(df), y=df["avg_beta"],
                        name=_short_graph_label(label), line=dict(color=COLOR_MAP[i % len(COLOR_MAP)], width=2)
                    ))
            fig_b.update_layout(
                xaxis_title="Round", yaxis_title="β (avg across clients)",
                template="plotly_white", height=300,
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_b, width="stretch")
            st.caption(
                "**β (sparsification ratio)** = fraction of gradient coordinates transmitted each round. "
                "β=1.0 → full dense gradient (full-battery client). "
                "β=0.01 → 1% of gradient sent (dying client). "
                "Formula: β_t^k = β_min + (β_max − β_min) × B_t^k / B_max. "
                "A falling β curve means the fleet is losing battery over time."
            )

        # Osmotic pressure (Fed-Osmosis / Fed-Resonance-Osmosis)
        osmosis_cols = ["avg_osmotic_pressure", "osmotic_pressure", "mean_osmotic_pressure"]
        osmosis_col_found = next(
            (c for c in osmosis_cols
             if any(c in exp["df"].columns for exp in experiments.values())),
            None,
        )
        if osmosis_col_found:
            st.markdown(
                "**Osmotic pressure Π over rounds** — measures distribution divergence "
                "between local and global activation statistics (lower = better aligned clients)"
            )
            fig_os = go.Figure()
            for i, (label, exp) in enumerate(experiments.items()):
                df = exp["df"]
                if osmosis_col_found in df.columns:
                    fig_os.add_trace(go.Scatter(
                        x=_round_col(df), y=df[osmosis_col_found],
                        name=_short_graph_label(label), line=dict(color=COLOR_MAP[i % len(COLOR_MAP)], width=2)
                    ))
            fig_os.update_layout(
                xaxis_title="Round", yaxis_title="Osmotic Pressure Π",
                template="plotly_white", height=300,
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_os, width="stretch")
            st.caption(
                "Osmotic pressure = KL divergence from local to global activation distribution. "
                "Decreasing Π indicates that Fed-Osmosis is successfully aligning client feature spaces."
            )

        # Spectral rank (Fed-Resonance / Fed-Resonance-Osmosis)
        rank_cols = ["avg_rank", "mean_rank", "avg_svd_rank"]
        rank_col_found = next(
            (c for c in rank_cols
             if any(c in exp["df"].columns for exp in experiments.values())),
            None,
        )
        if rank_col_found:
            st.markdown(
                "**Adaptive spectral rank over rounds** — average truncated rank r* used "
                "per layer (lower = more compression, preserving ε=90% gradient energy)"
            )
            fig_rk = go.Figure()
            for i, (label, exp) in enumerate(experiments.items()):
                df = exp["df"]
                if rank_col_found in df.columns:
                    fig_rk.add_trace(go.Scatter(
                        x=_round_col(df), y=df[rank_col_found],
                        name=_short_graph_label(label), line=dict(color=COLOR_MAP[i % len(COLOR_MAP)], width=2)
                    ))
            fig_rk.update_layout(
                xaxis_title="Round", yaxis_title="Avg Rank r*",
                template="plotly_white", height=300,
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_rk, width="stretch")
            st.caption(
                "Fed-Resonance selects rank r* = argmin{r : Σσᵢ²[:r]/Σσᵢ² ≥ ε}. "
                "A decreasing rank over rounds means gradients become increasingly low-rank "
                "as the model converges — a sign of successful training."
            )

    # ── Energy per Round ─────────────────────────────────────────────────────
    with tab_energy:

        # ── Accuracy vs Cumulative Energy ─────────────────────────────────────
        st.markdown(
            "##### Accuracy vs Cumulative Energy &nbsp;&nbsp;"
            "<span style='background:#1f77b4;color:white;border-radius:4px;"
            "padding:2px 8px;font-size:12px;'>Métrique principale</span>",
            unsafe_allow_html=True,
        )
        _ae_sc1, _ae_sc2 = st.columns([1, 3])
        with _ae_sc1:
            _ae_smooth = st.toggle("EMA smoothing", value=True,
                                   key="ae_smooth",
                                   help="Lisse la courbe accuracy pour réduire le bruit "
                                        "sans changer les données sous-jacentes.")
        with _ae_sc2:
            _ae_alpha = st.slider("α (lissage)", min_value=0.05, max_value=1.0,
                                  value=0.2, step=0.05,
                                  key="ae_alpha",
                                  disabled=not _ae_smooth)

        fig_ae = go.Figure()
        for i, (label, exp) in enumerate(experiments.items()):
            df    = exp["df"]
            color = COLOR_MAP[i % len(COLOR_MAP)]
            lbl   = _short_graph_label(label)
            if "cumulative_energy_j" not in df.columns or "test_accuracy" not in df.columns:
                continue
            x_vals = df["cumulative_energy_j"]
            y_raw  = df["test_accuracy"] * 100
            y_plot = ema_smooth(y_raw, _ae_alpha) if _ae_smooth else y_raw

            if _ae_smooth:
                fig_ae.add_trace(go.Scatter(
                    x=x_vals, y=y_raw,
                    line=dict(color=color, width=0.8),
                    opacity=0.2, showlegend=False, hoverinfo="skip",
                ))
            fig_ae.add_trace(go.Scatter(
                x=x_vals, y=y_plot,
                name=lbl,
                line=dict(color=color, width=2.5),
                mode="lines",
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Energy: %{x:,.0f} J<br>"
                    "Accuracy: %{y:.2f}%<extra></extra>"
                ),
            ))

        fig_ae.update_layout(
            title="Accuracy (%) vs Cumulative Energy Consumed (J)",
            xaxis_title="Cumulative Energy (J)",
            yaxis_title="Test Accuracy (%)",
            template="plotly_white",
            height=440,
            font=dict(family="Inter"),
            legend=dict(
                orientation="v",
                xanchor="left", x=1.02,
                yanchor="middle", y=0.5,
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#e2e8f0", borderwidth=1,
            ),
            margin=dict(r=180),
        )
        st.plotly_chart(fig_ae, width="stretch")
        if _ae_smooth:
            st.caption(
                f"Smoothed EMA (α={_ae_alpha:.2f}) — raw values in transparency. "
                "Underlying data unchanged."
            )
        st.caption(
            "**Why this curve?** It compares algorithms fairly "
            "independently of the number of local epochs or the per-round convergence rate. "
            "The X-axis represents the energy actually spent by the fleet — the true constrained resource. "
            "A better algorithm achieves higher accuracy for the same energy budget. "
            "This is the central metric for battery-limited IoT systems."
        )

        st.divider()
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            fig = go.Figure()
            for i, (label, exp) in enumerate(experiments.items()):
                df = exp["df"]
                rounds = _round_col(df)
                if "cumulative_energy_j" in df.columns:
                    y_vals = df["cumulative_energy_j"]
                elif "total_energy_j" in df.columns:
                    y_vals = df["total_energy_j"].cumsum()
                else:
                    continue
                fig.add_trace(go.Scatter(
                    x=rounds, y=y_vals,
                    name=_short_graph_label(label),
                    line=dict(color=COLOR_MAP[i % len(COLOR_MAP)], width=2.5),
                    fill="tozeroy", fillcolor=COLOR_MAP[i % len(COLOR_MAP)].replace(")", ",0.05)").replace("rgb", "rgba") if "rgb" in COLOR_MAP[i % len(COLOR_MAP)] else "rgba(56,189,248,0.05)",
                ))
            fig.update_layout(
                title="Cumulative Energy Consumed (J)",
                xaxis_title="Round", yaxis_title="Total Energy (J)",
                template="plotly_white", height=350,
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "**Cumulative energy** = sum of E_round for all rounds up to t. "
                "E_round = E_compute + E_uplink (modelled via the battery profile of each device). "
                "Lower total = more energy-efficient algorithm. "
                "A steeper slope = more energy spent per round."
            )

        with col_e2:
            fig = go.Figure()
            for i, (label, exp) in enumerate(experiments.items()):
                df = exp["df"]
                rounds = _round_col(df)
                if "total_energy_j" in df.columns:
                    fig.add_trace(go.Scatter(
                        x=rounds, y=df["total_energy_j"],
                        name=_short_graph_label(label),
                        line=dict(color=COLOR_MAP[i % len(COLOR_MAP)], width=2),
                    ))
            fig.update_layout(
                title="Energy Consumed per Round (J)",
                xaxis_title="Round", yaxis_title="Energy (J)",
                template="plotly_white", height=350,
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "**Energy per round** = sum over all alive clients of their per-round consumption. "
                "A downward trend is expected as clients die (battery reaches 0). "
                "A sharp drop = multiple clients died simultaneously."
            )

    # ── Battery Evolution ────────────────────────────────────────────────────
    with tab_battery:
        fig = go.Figure()
        for i, (label, exp) in enumerate(experiments.items()):
            df = exp["df"]
            rounds = _round_col(df)
            if "avg_battery_j" in df.columns:
                fig.add_trace(go.Scatter(
                    x=rounds, y=df["avg_battery_j"],
                    name=_short_graph_label(label),
                    line=dict(color=COLOR_MAP[i % len(COLOR_MAP)], width=2.5),
                ))
        fig.update_layout(
            title="Average Client Battery Remaining (J) — Higher is Better",
            xaxis_title="Round", yaxis_title="Avg Battery (J)",
            template="plotly_white", height=400,
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "**Avg Battery (J)** = mean remaining battery across all clients still alive at round t. "
            "Initial battery = B_max × SOC_0 (depends on device profile and initial SOC). "
            "For ESP32-S3: B_max = 13 320 J (1 000 mAh × 3.7 V × 3.6). "
            "A faster drop = algorithm drains batteries quicker. "
            "When the curve reaches 0, the last client has died."
        )

        # Communication bytes
        fig2 = make_subplots(rows=1, cols=2,
                             subplot_titles=["Cumulative Communication (GB)",
                                            "Total Communication Comparison"])
        for i, (label, exp) in enumerate(experiments.items()):
            df = exp["df"]
            rounds = _round_col(df)
            if "cumulative_bytes" in df.columns:
                y_vals = df["cumulative_bytes"] / 1e9
            elif "total_bytes" in df.columns:
                y_vals = df["total_bytes"].cumsum() / 1e9
            else:
                continue
            fig2.add_trace(go.Scatter(
                x=rounds, y=y_vals,
                name=_short_graph_label(label),
                line=dict(color=COLOR_MAP[i % len(COLOR_MAP)], width=2),
                showlegend=True,
            ), row=1, col=1)

        bar_labels, bar_vals = [], []
        for label, exp in experiments.items():
            df = exp["df"]
            if "total_bytes" in df.columns:
                bar_labels.append(_short_graph_label(label))
                bar_vals.append(df["total_bytes"].sum() / 1e9)
        if bar_labels:
            fig2.add_trace(go.Bar(
                x=bar_labels, y=bar_vals,
                marker_color=COLOR_MAP[:len(bar_labels)],
                showlegend=False,
            ), row=1, col=2)

        fig2.update_xaxes(title_text="Round", row=1, col=1)
        fig2.update_yaxes(title_text="GB", row=1, col=1)
        fig2.update_yaxes(title_text="GB", row=1, col=2)
        fig2.update_layout(
            height=380, template="plotly_white",
            legend=dict(x=1.02, xanchor="left", y=1.0, yanchor="top"),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig2, width="stretch")
        st.caption(
            "**Communication (GB)** = total bytes uploaded by all clients. "
            "Uplink = compressed gradient sent to the server each round. "
            "With sparsification (β < 1), only β × |θ| floats are sent instead of the full model. "
            "Lower total bytes = less communication overhead — key metric for bandwidth-constrained IoT."
        )

        # ── Total Energy Consumed per Round ──────────────────────────────────
        st.markdown("---")
        st.markdown("**Total Energy Consumed per Round**")
        has_energy_per_round = any(
            "total_energy_j" in exp["df"].columns for exp in experiments.values()
        )
        if has_energy_per_round:
            fig_enrnd = go.Figure()
            for i, (label, exp) in enumerate(experiments.items()):
                df = exp["df"]
                if "total_energy_j" not in df.columns:
                    continue
                fig_enrnd.add_trace(go.Bar(
                    x=_round_col(df),
                    y=df["total_energy_j"],
                    name=_short_graph_label(label),
                    marker_color=COLOR_MAP[i % len(COLOR_MAP)],
                    opacity=0.82,
                ))
            fig_enrnd.update_layout(
                title="Total Energy Consumed per Round (J) — all clients combined",
                xaxis_title="Round",
                yaxis_title="Energy (J)",
                barmode="group",
                template="plotly_white",
                height=400,
                font=dict(family="Inter"),
                legend=dict(
                    orientation="v", xanchor="left", x=1.02,
                    yanchor="middle", y=0.5,
                    font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#e2e8f0", borderwidth=1,
                ),
                margin=dict(r=180),
            )
            st.plotly_chart(fig_enrnd, width="stretch")
            st.caption(
                "**Energy per round (J)** = sum of compute + uplink energy over all active clients for that round. "
                "E_compute = P_active × training_time; E_uplink = model_bytes / link_rate × P_tx. "
                "Algorithms that skip clients (cyclic scheduling) show lower per-round energy but may need more rounds."
            )
        else:
            st.info("No `total_energy_j` column found in the selected experiments.")

        # ── Total Energy Consumed (full run) ─────────────────────────────────
        st.markdown("---")
        st.markdown("**Total Energy Consumed — Full Run**")
        total_en_labels, total_en_vals = [], []
        for label, exp in experiments.items():
            df = exp["df"]
            if "cumulative_energy_j" in df.columns:
                val = float(df["cumulative_energy_j"].iloc[-1])
            elif "total_energy_j" in df.columns:
                val = float(df["total_energy_j"].sum())
            else:
                continue
            total_en_labels.append(_short_graph_label(label))
            total_en_vals.append(val)

        if total_en_labels:
            fig_total_en = go.Figure(go.Bar(
                x=total_en_labels,
                y=total_en_vals,
                marker_color=COLOR_MAP[: len(total_en_labels)],
                marker_line_color="white",
                marker_line_width=1.5,
                text=[f"{v:.0f} J" for v in total_en_vals],
                textposition="outside",
            ))
            fig_total_en.update_layout(
                title="Total Energy Consumed over All Rounds (J)",
                yaxis_title="Energy (J)",
                template="plotly_white",
                height=400,
                font=dict(family="Inter"),
                xaxis_tickangle=-20,
                uniformtext_minsize=9,
                uniformtext_mode="hide",
            )
            st.plotly_chart(fig_total_en, width="stretch")
            st.caption(
                "**Total energy (J)** = cumulative compute + uplink energy summed over all rounds and all clients. "
                "Uses `cumulative_energy_j` (last round) when available, otherwise sums `total_energy_j`. "
                "Lower = more energy-efficient over the full experiment."
            )
        else:
            st.info("No energy data found in the selected experiments.")

        # ── Energy & Time to Target Accuracy ─────────────────────────────────
        st.markdown("---")
        st.markdown("**Energy Consumption & Computation Time to Target Accuracy**")
        _target_acc = st.slider(
            "Target accuracy (%)",
            min_value=10, max_value=100, value=70, step=5,
            key="battery_target_acc",
            help="Vary this threshold to see how much energy and simulated time each algorithm needs to first reach it.",
        )

        _ta_labels, _ta_energy, _ta_time, _ta_reached = [], [], [], []
        for label, exp in experiments.items():
            df = exp["df"]
            if "test_accuracy" not in df.columns:
                continue
            acc_series = df["test_accuracy"] * 100
            mask = acc_series >= _target_acc
            if mask.any():
                rnd_idx = int(mask.values.argmax())
                reached = True
            else:
                rnd_idx = len(df) - 1
                reached = False

            energy_val = None
            if "cumulative_energy_j" in df.columns:
                energy_val = float(df["cumulative_energy_j"].iloc[rnd_idx])
            elif "total_energy_j" in df.columns:
                energy_val = float(df["total_energy_j"].iloc[: rnd_idx + 1].sum())

            time_val = None
            if "cumulative_sim_time_s" in df.columns:
                time_val = float(df["cumulative_sim_time_s"].iloc[rnd_idx])
            elif "sim_round_time_s" in df.columns:
                time_val = float(df["sim_round_time_s"].iloc[: rnd_idx + 1].sum())

            _ta_labels.append(_short_graph_label(label))
            _ta_energy.append(energy_val if energy_val is not None else 0.0)
            _ta_time.append(time_val if time_val is not None else 0.0)
            _ta_reached.append(reached)

        if _ta_labels:
            _bar_colors = [
                COLOR_MAP[i % len(COLOR_MAP)] if r else "rgba(180,180,180,0.6)"
                for i, r in enumerate(_ta_reached)
            ]
            _ta_c1, _ta_c2 = st.columns(2)
            with _ta_c1:
                fig_eta = go.Figure(go.Bar(
                    x=_ta_labels,
                    y=_ta_energy,
                    marker_color=_bar_colors,
                    marker_line_color="white",
                    marker_line_width=1.5,
                    text=[f"{v:.0f} J" for v in _ta_energy],
                    textposition="outside",
                ))
                fig_eta.update_layout(
                    title=f"Energy to reach {_target_acc}% accuracy",
                    yaxis_title="Energy (J)",
                    template="plotly_white",
                    height=400,
                    font=dict(family="Inter"),
                    xaxis_tickangle=-20,
                    uniformtext_minsize=9,
                    uniformtext_mode="hide",
                )
                st.plotly_chart(fig_eta, width="stretch")
                _grey_note = " Grey = threshold never reached (full-run energy)." if not all(_ta_reached) else ""
                st.caption(
                    f"Total energy (J) from round 1 until first reaching {_target_acc}% test accuracy.{_grey_note} "
                    "Lower = more energy-efficient."
                )
            with _ta_c2:
                if any(t > 0 for t in _ta_time):
                    fig_tta_bat = go.Figure(go.Bar(
                        x=_ta_labels,
                        y=_ta_time,
                        marker_color=_bar_colors,
                        marker_line_color="white",
                        marker_line_width=1.5,
                        text=[f"{v:.0f}s" if v > 0 else "—" for v in _ta_time],
                        textposition="outside",
                    ))
                    fig_tta_bat.update_layout(
                        title=f"Simulated time to reach {_target_acc}% accuracy",
                        yaxis_title="Time (s)",
                        template="plotly_white",
                        height=400,
                        font=dict(family="Inter"),
                        xaxis_tickangle=-20,
                        uniformtext_minsize=9,
                        uniformtext_mode="hide",
                    )
                    st.plotly_chart(fig_tta_bat, width="stretch")
                    st.caption(
                        f"Cumulative simulated wall-clock time (s) to first reach {_target_acc}% accuracy. "
                        "Based on max(client training times) per round — lower = faster convergence."
                    )
                else:
                    st.info(
                        "No simulated time data found. Re-run experiments with a version of the "
                        "framework that records `sim_round_time_s` per round."
                    )
        else:
            st.info("No accuracy or energy data available in the selected experiments.")

    # ── Fairness ─────────────────────────────────────────────────────────────
    with tab_fair:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            fig = go.Figure()
            for i, (label, exp) in enumerate(experiments.items()):
                df = exp["df"]
                rounds = _round_col(df)
                if "jain_index" in df.columns:
                    fig.add_trace(go.Scatter(
                        x=rounds, y=df["jain_index"],
                        name=_short_graph_label(label),
                        line=dict(color=COLOR_MAP[i % len(COLOR_MAP)], width=2.5),
                    ))
            fig.add_hline(y=1.0, line_dash="dot", line_color="#22c55e",
                          annotation_text="Perfect fairness (J=1.0)")
            fig.update_layout(
                title="Participation Fairness — Jain Index",
                xaxis_title="Round", yaxis_title="Jain Index",
                yaxis_range=[0, 1.05],
                template="plotly_white", height=380,
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "**Jain Index J = (Σ E_k)² / (K · Σ E_k²)** — measures how evenly energy is shared. "
                "J=1.0 → all clients consume exactly the same energy (perfect fairness). "
                "J→0 → one client does all the work. "
                "J≥0.8 is the accepted threshold in FL fairness literature. "
                "Computed on cumulative energy per client at each round."
            )

        with col_f2:
            fig = go.Figure()
            for i, (label, exp) in enumerate(experiments.items()):
                df = exp["df"]
                rounds = _round_col(df)
                if "participation_rate" in df.columns:
                    fig.add_trace(go.Scatter(
                        x=rounds, y=df["participation_rate"] * 100,
                        name=_short_graph_label(label),
                        line=dict(color=COLOR_MAP[i % len(COLOR_MAP)], width=2),
                    ))
            fig.update_layout(
                title="Client Participation Rate (%)",
                xaxis_title="Round", yaxis_title="%",
                template="plotly_white", height=380,
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "**Participation rate** = alive clients / total clients at each round. "
                "A client is considered dead when its battery reaches 0 J. "
                "100% → full fleet active. A falling curve = clients dying over time. "
                "Algorithms that preserve participation longer are more robust for long FL campaigns."
            )

    # ── Summary Table ────────────────────────────────────────────────────────
    with tab_table:
        rows = []
        for label, exp in experiments.items():
            df   = exp["df"]
            meta = exp["config"]
            summary = meta.get("summary", {})

            best_acc = (
                f"{summary['best_accuracy']*100:.2f}"
                if "best_accuracy" in summary
                else (f"{df['test_accuracy'].max()*100:.2f}" if "test_accuracy" in df.columns else "—")
            )
            final_acc = (
                f"{summary['final_accuracy']*100:.2f}"
                if "final_accuracy" in summary
                else (f"{df['test_accuracy'].iloc[-1]*100:.2f}" if "test_accuracy" in df.columns and len(df) > 0 else "—")
            )
            total_gb = (
                f"{summary['total_bytes_gb']:.3f}"
                if "total_bytes_gb" in summary
                else (f"{df['total_bytes'].sum()/1e9:.3f}" if "total_bytes" in df.columns else "—")
            )
            total_energy = (
                f"{summary['total_energy_j']:.1f}"
                if "total_energy_j" in summary
                else (f"{df['total_energy_j'].sum():.1f}" if "total_energy_j" in df.columns else "—")
            )
            final_jain = (
                f"{summary['final_jain_index']:.4f}"
                if "final_jain_index" in summary
                else (f"{df['jain_index'].iloc[-1]:.4f}" if "jain_index" in df.columns and len(df) > 0 else "—")
            )
            final_battery = (
                f"{summary['final_avg_battery_j']:.1f}"
                if "final_avg_battery_j" in summary
                else (f"{df['avg_battery_j'].iloc[-1]:.1f}" if "avg_battery_j" in df.columns and len(df) > 0 else "—")
            )

            # ── Clients alive at end of training ─────────────────────────────
            if len(df) > 0:
                if "K_alive" in df.columns:
                    # direct count
                    k_alive_end = int(df["K_alive"].iloc[-1])
                    k_total     = int(df["K_alive"].max())          # round 1 should be max
                    clients_alive = f"{k_alive_end} / {k_total}"
                elif "num_alive_clients" in df.columns:
                    k_alive_end = int(df["num_alive_clients"].iloc[-1])
                    k_total     = int(df["num_alive_clients"].max())
                    clients_alive = f"{k_alive_end} / {k_total}"
                elif "participation_rate" in df.columns:
                    # ratio only — show as percentage
                    rate = float(df["participation_rate"].iloc[-1])
                    clients_alive = f"{rate*100:.0f}%"
                else:
                    clients_alive = "—"
            else:
                clients_alive = "—"

            total_sim = summary.get("total_sim_time_s", None)
            if total_sim is None and "sim_round_time_s" in df.columns:
                total_sim = float(df["sim_round_time_s"].sum())
            r2best = summary.get("rounds_to_best_acc", None)
            t2best = summary.get("sim_time_to_best_acc", None)

            rows.append({
                "Experiment":           label,
                "Best Acc. (%)":        best_acc,
                "Final Acc. (%)":       final_acc,
                "Clients Alive (end)":  clients_alive,
                "Total Comm. (GB)":     total_gb,
                "Total Energy (J)":     total_energy,
                "Final Jain J":         final_jain,
                "Final Battery (J)":    final_battery,
                "Rounds":               len(df),
                "Total Sim Time":       _fmt_seconds(total_sim),
                "Rounds→Best Acc":      int(r2best) if r2best is not None else "—",
                "SimTime→Best Acc":     _fmt_seconds(t2best),
            })

        summary_df = pd.DataFrame(rows)
        st.dataframe(summary_df, width="stretch", hide_index=True)
        st.caption(
            "**Column definitions — ** "
            "**Best Acc. (%)**: peak test accuracy across all rounds. "
            "**Final Acc. (%)**: test accuracy at the last completed round. "
            "**Clients Alive (end)**: number of clients with battery > 0 at the last round, "
            "shown as `alive / total` (e.g. `19 / 30` means 11 clients ran out of battery). "
            "A high value means the algorithm preserved the fleet well. "
            "**Total Comm. (GB)**: total bytes uploaded by all clients over the whole run. "
            "**Total Energy (J)**: total joules consumed by all clients (compute + uplink). "
            "**Final Jain J**: Jain fairness index at the last round — J=(ΣE_k)²/(K·ΣE_k²), range [0,1], higher=fairer. "
            "**Final Battery (J)**: average remaining battery per client at the last round — "
            "how much energy was left in the fleet when training ended (higher = less drained). "
            "**Rounds**: total rounds completed. "
            "**Rounds→Best Acc**: round at which best accuracy was first reached. "
            "**Total/SimTime→Best Acc**: simulated wall-clock time to best accuracy (based on device profiles)."
        )
        csv = summary_df.to_csv(index=False)
        st.download_button("Download CSV", csv, "fedlab_comparison.csv", "text/csv")

        # ── Fleet Lifetime Metrics ────────────────────────────────────────────
        st.divider()
        st.markdown(
            "#### Fleet Lifetime Metrics"
            " &nbsp;<span style='background:#0f172a;color:white;border-radius:4px;"
            "padding:2px 8px;font-size:11px;letter-spacing:.5px;'>Paper KPIs</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "Survival metrics designed for FL systems with limited battery life."
            "A client is **alive** if its battery `B_k > 0`.",
        )

        # Jain evaluation round selector
        _jain_round = st.number_input(
            "Evaluate Jain Index at round:",
            min_value=1, max_value=500, value=50, step=5,
            help="Round at which Jain Index is extracted. "
                 "The presentation uses round 50 as a reference.",
        )

        lifetime_rows = []
        for label, exp in experiments.items():
            df = exp["df"]
            if df.empty:
                continue

            # ── survival_ratio column ─────────────────────────────────────────
            if "survival_ratio" in df.columns:
                surv = df["survival_ratio"].values
            elif "num_alive_clients" in df.columns:
                k_max = df["num_alive_clients"].max()
                surv  = df["num_alive_clients"].values / max(k_max, 1)
            else:
                surv = None

            rounds_col = df["round_num"].values if "round_num" in df.columns else np.arange(1, len(df) + 1)

            # system_lifetime = first round where survival_ratio ≤ 0.5
            system_lifetime = "—"
            acc_at_half     = "—"
            if surv is not None:
                idx_half = np.where(surv <= 0.5)[0]
                if len(idx_half) > 0:
                    system_lifetime = int(rounds_col[idx_half[0]])
                    if "test_accuracy" in df.columns:
                        acc_val = df["test_accuracy"].iloc[idx_half[0]]
                        acc_at_half = f"{acc_val * 100:.1f}%"

            # all_dead_round = first round where survival_ratio == 0
            # = the round during which the last client's battery hit 0
            # survival_ratio is recorded at END of round (after deaths),
            # so surv==0 at round t means the last client died during round t.
            all_dead_round = "—"
            if surv is not None:
                dead_rounds = np.where(surv == 0)[0]
                if len(dead_rounds) > 0:
                    all_dead_round = int(rounds_col[dead_rounds[0]])

            # Jain index at selected round
            jain_at_round = "—"
            if "jain_index" in df.columns and "round_num" in df.columns:
                jain_rows = df[df["round_num"] <= _jain_round]
                if not jain_rows.empty:
                    jain_val = jain_rows["jain_index"].iloc[-1]
                    jain_at_round = f"{jain_val:.3f}"

            # Best accuracy overall
            best_acc_val = df["test_accuracy"].max() * 100 if "test_accuracy" in df.columns else None
            best_acc_str = f"{best_acc_val:.1f}%" if best_acc_val is not None else "—"

            lifetime_rows.append({
                "Experiment":       label,
                "system_lifetime":  system_lifetime,
                "acc@half_drop":    acc_at_half,
                "all_dead_round":   all_dead_round,
                f"Jain (rd {_jain_round})": jain_at_round,
                "Best Acc.":        best_acc_str,
            })

        if lifetime_rows:
            lifetime_df = pd.DataFrame(lifetime_rows)
            # Normalize mixed int/"—" object columns to str so Arrow can serialize them
            for _col in ("system_lifetime", "all_dead_round"):
                if _col in lifetime_df.columns:
                    lifetime_df[_col] = lifetime_df[_col].apply(
                        lambda v: str(v) if isinstance(v, int) else v
                    )
            st.dataframe(lifetime_df, width="stretch", hide_index=True)

            # ── Formulas block ────────────────────────────────────────────────
            st.caption(
                "**system\\_lifetime** `= min{t : survival_ratio(t) ≤ 0.5}` — "
                "round when 50% of the fleet has died. Later = better survival. "
                "**acc@half\\_drop** — accuracy *at the moment* half the fleet has died : "
                "measures graceful degradation under client loss. "
                "**all\\_dead\\_round** `= min{t : survival_ratio(t) = 0}` — "
                "round when the last client dies (total fleet lifetime). "
                f"**Jain (rd {_jain_round})** "
                "`J = (ΣE_k)² / (K · ΣE_k²)` — "
                "fairness of energy consumption across clients. "
                "J=1 : all clients drain exactly the same energy. "
                "J→0 : only one client does all the work."
            )

            # ── Visual: system_lifetime bar chart ─────────────────────────────
            sl_data  = [(r["Experiment"], r["system_lifetime"])
                        for r in lifetime_rows if isinstance(r["system_lifetime"], int)]
            adr_data = [(r["Experiment"], r["all_dead_round"])
                        for r in lifetime_rows if isinstance(r["all_dead_round"], int)]

            if sl_data or adr_data:
                fig_lt = go.Figure()
                colors = [COLOR_MAP[i % len(COLOR_MAP)] for i in range(len(lifetime_rows))]

                if sl_data:
                    fig_lt.add_trace(go.Bar(
                        name="system_lifetime (50% mort)",
                        x=[d[0] for d in sl_data],
                        y=[d[1] for d in sl_data],
                        marker_color=[colors[i] for i, r in enumerate(lifetime_rows)
                                      if isinstance(r["system_lifetime"], int)],
                        opacity=0.65,
                        text=[str(d[1]) for d in sl_data],
                        textposition="outside",
                    ))
                if adr_data:
                    fig_lt.add_trace(go.Bar(
                        name="all_dead_round (100% mort)",
                        x=[d[0] for d in adr_data],
                        y=[d[1] for d in adr_data],
                        marker_color=[colors[i] for i, r in enumerate(lifetime_rows)
                                      if isinstance(r["all_dead_round"], int)],
                        opacity=1.0,
                        text=[str(d[1]) for d in adr_data],
                        textposition="outside",
                    ))

                fig_lt.update_layout(
                    title="Fleet Lifetime — system_lifetime vs all_dead_round (higher = better)",
                    xaxis_title="Experiment",
                    yaxis_title="Round",
                    barmode="group",
                    template="plotly_white",
                    height=380,
                    font=dict(family="Inter"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1),
                )
                st.plotly_chart(fig_lt, width="stretch")

            csv_lt = lifetime_df.to_csv(index=False)
            st.download_button("Download Fleet Lifetime CSV", csv_lt,
                               "fleet_lifetime_metrics.csv", "text/csv",
                               key="dl_lifetime")

    # ── Convergence Speed ────────────────────────────────────────────────────
    with tab_convspeed:
        st.markdown(
            "**Convergence Speed** — time-to-accuracy and per-round simulated wall-clock time. "
            "These plots use `cumulative_sim_time_s` (x-axis) and `sim_round_time_s` (bar chart), "
            "which are available in experiments that record simulated parallel training times. "
            "Old results without these columns are handled gracefully."
        )

        # ── Time-to-Accuracy curve ────────────────────────────────────────────
        has_sim_time = any(
            "cumulative_sim_time_s" in exp["df"].columns
            for exp in experiments.values()
        )

        if has_sim_time:
            st.markdown("**Time-to-Accuracy curve** — x-axis = cumulative simulated time (s), "
                        "y-axis = test accuracy (%). Curves shifted left = faster convergence.")
            fig_tta = go.Figure()
            for i, (label, exp) in enumerate(experiments.items()):
                df = exp["df"]
                if "cumulative_sim_time_s" not in df.columns or "test_accuracy" not in df.columns:
                    continue
                legend_label = _short_graph_label(label)
                fig_tta.add_trace(go.Scatter(
                    x=df["cumulative_sim_time_s"],
                    y=df["test_accuracy"] * 100,
                    mode="lines",
                    name=legend_label,
                    line=dict(color=COLOR_MAP[i % len(COLOR_MAP)], width=2.5),
                ))
            fig_tta.update_layout(
                title="Time-to-Accuracy (simulated parallel time)",
                xaxis_title="Cumulative Simulated Time (s)",
                yaxis_title="Test Accuracy (%)",
                template="plotly_white",
                height=460,
                font=dict(family="Inter"),
                legend=dict(
                    orientation="v", xanchor="left", x=1.02,
                    yanchor="middle", y=0.5,
                    font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#e2e8f0", borderwidth=1,
                ),
                margin=dict(r=180),
            )
            st.plotly_chart(fig_tta, width="stretch")
            st.caption(
                "sim_round_time_s = max(client training times) for each round — "
                "models the true wall-clock time of a distributed round where clients train in parallel. "
                "cumulative_sim_time_s is the running sum from round 1."
            )
        else:
            st.info(
                "No `cumulative_sim_time_s` column found in selected experiments. "
                "Re-run experiments with a version of the framework that records "
                "`sim_round_time_s` per round."
            )

        st.markdown("---")

        # ── Simulated round time bar / line chart ─────────────────────────────
        has_round_time = any(
            "sim_round_time_s" in exp["df"].columns
            for exp in experiments.values()
        )

        if has_round_time:
            st.markdown("**Simulated round time per round** — "
                        "bar height = max client training time for that round (seconds). "
                        "Heavier rounds indicate synchronisation bottlenecks.")
            fig_rt = go.Figure()
            for i, (label, exp) in enumerate(experiments.items()):
                df = exp["df"]
                if "sim_round_time_s" not in df.columns:
                    continue
                legend_label = _short_graph_label(label)
                fig_rt.add_trace(go.Bar(
                    x=_round_col(df),
                    y=df["sim_round_time_s"],
                    name=legend_label,
                    marker_color=COLOR_MAP[i % len(COLOR_MAP)],
                    opacity=0.75,
                ))
            fig_rt.update_layout(
                title="Simulated Round Time (s) per Round — max(client training times)",
                xaxis_title="Round",
                yaxis_title="Round Time (s)",
                barmode="group",
                template="plotly_white",
                height=380,
                font=dict(family="Inter"),
                legend=dict(
                    orientation="v", xanchor="left", x=1.02,
                    yanchor="middle", y=0.5,
                    font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#e2e8f0", borderwidth=1,
                ),
                margin=dict(r=180),
            )
            st.plotly_chart(fig_rt, width="stretch")
            st.caption(
                "Algorithms that skip clients (e.g. FedBacys cyclic scheduling) "
                "may show lower round times because fewer clients train per round. "
                "FedPart_BE should exhibit more stable round times vs FedPart."
            )

            # Summary table: total sim time, rounds to best acc, sim time to best acc
            st.markdown("---")
            st.markdown("**Convergence speed summary**")
            cs_rows = []
            for label, exp in experiments.items():
                df   = exp["df"]
                meta = exp["config"]
                summary = meta.get("summary", {})
                total_sim = summary.get("total_sim_time_s", None)
                if total_sim is None and "sim_round_time_s" in df.columns:
                    total_sim = float(df["sim_round_time_s"].sum())
                r2best   = summary.get("rounds_to_best_acc", None)
                t2best   = summary.get("sim_time_to_best_acc", None)
                cs_rows.append({
                    "Experiment":              label,
                    "Total Sim Time":          _fmt_seconds(total_sim),
                    "Rounds to Best Acc":      str(int(r2best)) if r2best is not None else "—",
                    "Sim Time to Best Acc":    _fmt_seconds(t2best),
                })
            if cs_rows:
                st.dataframe(pd.DataFrame(cs_rows), width="stretch", hide_index=True)
        else:
            st.info("No `sim_round_time_s` column found in selected experiments.")

    # ── Layer Mismatch ───────────────────────────────────────────────────────
    with tab_lm:
        st.markdown(
            "**Layer Mismatch Diagnostic** — measures inter-layer alignment after FedAvg aggregation. "
            "A high score (→ 1) signals that layers from different clients don't cooperate: "
            "the aggregated global model has layers trained on incompatible feature spaces. "
            "Enable `layer_mismatch: true` in the YAML config to collect this metric."
        )

        # Check which experiments have layer_mismatch data
        lm_experiments = {
            label: exp for label, exp in experiments.items()
            if "layer_mismatch" in exp["df"].columns
            and exp["df"]["layer_mismatch"].notna().any()
        }

        if not lm_experiments:
            st.info(
                "No layer mismatch data found in the selected experiments. "
                "Enable `layer_mismatch: true` in your YAML config and re-run the experiment."
            )
            st.code(
                "# In your config YAML:\n"
                "layer_mismatch: true\n"
                "layer_mismatch_config:\n"
                "  freq: 5          # compute every 5 rounds\n"
                "  layers: null     # auto-detect Conv2d + Linear\n"
                "  metrics: [drift, loss_jump, cka]",
                language="yaml"
            )
        else:
            # ── KPI cards ─────────────────────────────────────────────────────
            st.markdown("**Average mismatch score per algorithm** (lower = better alignment)")
            kpi_cols = st.columns(min(len(lm_experiments), 4))
            for i, (label, exp) in enumerate(lm_experiments.items()):
                lm_series = exp["df"]["layer_mismatch"].dropna()
                avg_lm = float(lm_series.mean()) if len(lm_series) > 0 else float("nan")
                max_lm = float(lm_series.max()) if len(lm_series) > 0 else float("nan")
                col = kpi_cols[i % len(kpi_cols)]
                col.metric(
                    label=_short_graph_label(label),
                    value=f"{avg_lm:.3f}" if not (avg_lm != avg_lm) else "—",
                    delta=f"max {max_lm:.3f}" if not (max_lm != max_lm) else None,
                    delta_color="inverse",
                )

            st.divider()

            # ── Mismatch score over rounds ────────────────────────────────────
            st.markdown("**Mismatch score over rounds** — gaps = rounds skipped by `freq` setting")
            fig_lm = go.Figure()
            for i, (label, exp) in enumerate(lm_experiments.items()):
                df = exp["df"]
                rounds = _round_col(df)
                lm_col = df["layer_mismatch"]
                color = COLOR_MAP[i % len(COLOR_MAP)]
                legend_label = _short_graph_label(label)

                # Plot with connectgaps=False so None values show as gaps
                fig_lm.add_trace(go.Scatter(
                    x=rounds,
                    y=lm_col,
                    mode="lines+markers",
                    name=legend_label,
                    line=dict(color=color, width=2.5),
                    marker=dict(size=5, color=color),
                    connectgaps=False,
                ))

            fig_lm.add_hline(
                y=0.3, line_dash="dot", line_color="orange",
                annotation_text="Moderate mismatch (0.3)",
                annotation_position="bottom right",
            )
            fig_lm.add_hline(
                y=0.6, line_dash="dot", line_color="red",
                annotation_text="Severe mismatch (0.6)",
                annotation_position="bottom right",
            )
            fig_lm.update_layout(
                xaxis_title="Round",
                yaxis_title="Mismatch Score [0, 1]",
                yaxis=dict(range=[0, 1]),
                template="plotly_white",
                height=460,
                font=dict(family="Inter"),
                legend=dict(
                    orientation="v", xanchor="left", x=1.02,
                    yanchor="middle", y=0.5,
                    font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#e2e8f0", borderwidth=1,
                ),
                margin=dict(r=180),
            )
            st.plotly_chart(fig_lm, width="stretch")
            st.caption(
                "Score ∈ [0, 1] — composite of three sub-metrics averaged: "
                "**Representation Drift** (‖h_before − h_after‖_F / ‖h_before‖_F after FedAvg), "
                "**Loss Jump** (L_i(w_global) − L_i(w_local)), "
                "**CKA** (1 − CKA similarity between client representations). "
                "Gaps in the curve = rounds not computed due to `freq > 1` setting."
            )

            st.divider()

            # ── Per-algorithm summary table ───────────────────────────────────
            st.markdown("**Summary table**")
            lm_rows = []
            for label, exp in lm_experiments.items():
                lm_series = exp["df"]["layer_mismatch"].dropna()
                n_computed = int(lm_series.count())
                lm_rows.append({
                    "Algorithm":       label,
                    "Rounds computed": n_computed,
                    "Mean score":      f"{lm_series.mean():.4f}" if n_computed > 0 else "—",
                    "Max score":       f"{lm_series.max():.4f}" if n_computed > 0 else "—",
                    "Min score":       f"{lm_series.min():.4f}" if n_computed > 0 else "—",
                    "Last score":      f"{lm_series.iloc[-1]:.4f}" if n_computed > 0 else "—",
                })
            st.dataframe(pd.DataFrame(lm_rows), hide_index=True, width="stretch")
            st.caption(
                "Mean score averaged over all computed rounds (None rounds excluded). "
                "A decreasing trend over rounds indicates the aggregation scheme is "
                "progressively reducing inter-layer misalignment."
            )

            # ── Sub-metric breakdown (if individual metrics stored) ───────────
            sub_metrics = {
                "lm_drift":      ("Representation Drift", "orange"),
                "lm_loss_jump":  ("Loss Jump",            "crimson"),
                "lm_cka":        ("1 − CKA Similarity",   "steelblue"),
            }
            available_sub = [
                (col, name, color)
                for col, (name, color) in sub_metrics.items()
                if any(col in exp["df"].columns for exp in lm_experiments.values())
            ]
            if available_sub:
                st.divider()
                st.markdown("**Sub-metric breakdown**")
                fig_sub = go.Figure()
                for col, name, sub_color in available_sub:
                    for i, (label, exp) in enumerate(lm_experiments.items()):
                        df = exp["df"]
                        if col not in df.columns:
                            continue
                        rounds = _round_col(df)
                        legend_label = f"{_short_graph_label(label)} / {name}"
                        fig_sub.add_trace(go.Scatter(
                            x=rounds, y=df[col],
                            mode="lines",
                            name=legend_label,
                            line=dict(color=COLOR_MAP[i % len(COLOR_MAP)],
                                      width=1.5, dash="dot"),
                            connectgaps=False,
                        ))
                fig_sub.update_layout(
                    xaxis_title="Round",
                    yaxis_title="Sub-metric value",
                    template="plotly_white",
                    height=380,
                    font=dict(family="Inter"),
                    legend=dict(orientation="v", xanchor="left", x=1.02,
                                yanchor="middle", y=0.5,
                                bgcolor="rgba(255,255,255,0.8)",
                                bordercolor="#e2e8f0", borderwidth=1),
                    margin=dict(r=220),
                )
                st.plotly_chart(fig_sub, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# Page: COMPARE ALGORITHMS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Compare Algorithms":

    st.markdown("""
    <div class="results-header">
      <div class="results-header-title">Algorithm Comparison</div>
      <div class="results-header-sub">
        Accuracy vs. energy trade-offs, communication savings, and head-to-head benchmarks.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not result_dirs:
        st.markdown("""
        <div class="no-results-banner">
          <div class="no-results-title">No experiments available for comparison</div>
          <div class="no-results-text">
            Run the full benchmark suite to populate this page:<br>
            <code style="background:rgba(59,130,246,0.1); padding:2px 6px; border-radius:4px;">
            python run_experiment.py --benchmark --rounds 100 --output results/comparison</code>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Placeholder comparison table showing expected metrics
        st.markdown("**Expected comparison table format (populated after experiments):**")
        placeholder_data = {
            "Algorithm":        ["E-CEFFL", "FedAvg", "FedProx", "SCAFFOLD",
                                 "LeanFed", "FedBacys", "Vaishnav", "FedSparQ",
                                 "Fed-Resonance", "Fed-Osmosis", "Fed-Resonance-Osmosis"],
            "Best Acc. (%)":    ["—"] * 11,
            "Total Energy (J)": ["—"] * 11,
            "Comm. (GB)":       ["—"] * 11,
            "Jain Index":       ["—"] * 11,
            "Status":           ["Proposed", "Baseline", "Baseline", "Baseline",
                                 "Energy-Aware", "Energy-Aware", "Energy-Aware", "Energy-Aware",
                                 "New · 2026", "New · 2026", "New · 2026"],
        }
        st.dataframe(pd.DataFrame(placeholder_data), width="stretch", hide_index=True)
        st.stop()

    if not selected_dirs:
        st.info("Select at least two experiments from the sidebar for comparison.")
        st.stop()

    experiments = {}
    for name in selected_dirs:
        config, df = load_experiment(name)
        label = _get_exp_label(pathlib.Path(name), RESULTS_DIR)
        experiments[label] = {"config": config, "df": df}

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_scatter, tab_bars, tab_convergence = st.tabs([
        "Accuracy vs Energy",
        "Communication Savings",
        "Convergence Speed",
    ])

    with tab_scatter:
        scatter_data = []
        for i, (label, exp) in enumerate(experiments.items()):
            df = exp["df"]
            if "test_accuracy" in df.columns:
                best_acc = df["test_accuracy"].max() * 100
            else:
                continue
            total_energy = df["total_energy_j"].sum() if "total_energy_j" in df.columns else 0
            total_gb = df["total_bytes"].sum() / 1e9 if "total_bytes" in df.columns else 0
            scatter_data.append({
                "Algorithm": label,
                "Best Accuracy (%)": best_acc,
                "Total Energy (J)": total_energy,
                "Total Comm. (GB)": total_gb,
                "Color": COLOR_MAP[i % len(COLOR_MAP)],
            })

        if scatter_data:
            sc_df = pd.DataFrame(scatter_data)
            fig = px.scatter(
                sc_df, x="Total Energy (J)", y="Best Accuracy (%)",
                color="Algorithm",
                size="Total Comm. (GB)",
                size_max=30,
                text="Algorithm",
                color_discrete_sequence=COLOR_MAP,
                template="plotly_white",
                title="Accuracy vs Total Energy — bubble size = communication volume",
            )
            fig.update_traces(textposition="top center", marker=dict(line=dict(width=1.5, color="white")))
            fig.update_layout(height=500, font=dict(family="Inter"), showlegend=False)
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "**How to read:** X-axis = total energy (J) spent by all clients over the run — lower is better. "
                "Y-axis = best test accuracy (%) — higher is better. "
                "Bubble size = total communication volume (GB) — smaller bubble = less bandwidth used. "
                "**Upper-left + small bubble** = ideal algorithm: accurate, energy-efficient, and communication-light."
            )
        else:
            st.info("No accuracy or energy data found in selected experiments.")

    with tab_bars:
        col_b1, col_b2 = st.columns(2)

        with col_b1:
            # Communication comparison bar
            bar_labels, bar_comm, bar_energy = [], [], []
            for label, exp in experiments.items():
                df = exp["df"]
                if "total_bytes" in df.columns:
                    bar_labels.append(_short_graph_label(label))
                    bar_comm.append(df["total_bytes"].sum() / 1e9)
                    bar_energy.append(df["total_energy_j"].sum() if "total_energy_j" in df.columns else 0)

            if bar_labels:
                fig = go.Figure(go.Bar(
                    x=bar_labels, y=bar_comm,
                    marker_color=COLOR_MAP[:len(bar_labels)],
                    marker_line_color="white",
                    marker_line_width=1.5,
                ))
                fig.update_layout(
                    title="Total Communication Cost (GB)",
                    yaxis_title="GB",
                    template="plotly_white", height=380,
                    font=dict(family="Inter"),
                    xaxis_tickangle=-20,
                )
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    "**Total communication (GB)** = cumulative bytes uploaded by all clients over all rounds. "
                    "With gradient sparsification (β < 1), only β × |θ| parameters are sent instead of the full model. "
                    "Shorter bar = less bandwidth used = better for low-bandwidth IoT deployments."
                )

        with col_b2:
            if bar_labels and bar_energy:
                fig2 = go.Figure(go.Bar(
                    x=bar_labels, y=bar_energy,
                    marker_color=COLOR_MAP[:len(bar_labels)],
                    marker_line_color="white",
                    marker_line_width=1.5,
                ))
                fig2.update_layout(
                    title="Total Energy Consumption (J)",
                    yaxis_title="Joules",
                    template="plotly_white", height=380,
                    font=dict(family="Inter"),
                    xaxis_tickangle=-20,
                )
                st.plotly_chart(fig2, width="stretch")
                st.caption(
                    "**Total energy (J)** = cumulative compute + uplink energy over all rounds and all clients. "
                    "Energy = E_compute (local training) + E_uplink (model transmission). "
                    "For ESP32-S3: E_compute ≈ 0.38 W × training time; E_uplink ≈ model size / link rate × P_tx. "
                    "Shorter bar = more energy-efficient = clients survive longer."
                )

    with tab_convergence:
        # Rounds to reach accuracy thresholds
        thresholds = [50.0, 60.0, 70.0, 80.0, 90.0]
        conv_rows = []
        for label, exp in experiments.items():
            df = exp["df"]
            if "test_accuracy" not in df.columns:
                continue
            row = {"Algorithm": label}
            acc_series = df["test_accuracy"] * 100
            rounds_series = _round_col(df)
            for thr in thresholds:
                mask = acc_series >= thr
                if mask.any():
                    row[f"{thr:.0f}%"] = int(rounds_series[mask].iloc[0])
                else:
                    row[f"{thr:.0f}%"] = None  # None is Arrow-compatible; "—" is not
            conv_rows.append(row)

        if conv_rows:
            st.markdown("**Rounds needed to reach accuracy threshold** (lower is better)")
            st.dataframe(pd.DataFrame(conv_rows), width="stretch", hide_index=True)
            st.caption(
                "Each cell = number of rounds before test accuracy first exceeded that threshold. "
                "Empty (—) = accuracy was never reached. "
                "Lower value = faster convergence. "
                "An algorithm that reaches 70% in round 30 vs 80 is 2.7× faster to converge."
            )
        else:
            st.info("No accuracy data available in the selected experiments.")

        st.markdown("---")

        # ── Time-to-Accuracy curve (cumulative simulated time) ────────────────
        has_sim_time_cmp = any(
            "cumulative_sim_time_s" in exp["df"].columns
            for exp in experiments.values()
        )
        if has_sim_time_cmp:
            st.markdown(
                "**Time-to-Accuracy curve** — x-axis = cumulative simulated wall-clock time (s). "
                "This is the primary convergence speed metric: a curve shifted left converges faster."
            )
            _cmp_sc1, _cmp_sc2 = st.columns([1, 3])
            with _cmp_sc1:
                _cmp_smooth = st.toggle("EMA smoothing", value=True, key="cmp_smooth_tta",
                                        help="Smooth convergence curves for clarity.")
            with _cmp_sc2:
                _cmp_alpha = st.slider("α", min_value=0.05, max_value=1.0, value=0.15,
                                       step=0.05, key="cmp_alpha_tta",
                                       disabled=not _cmp_smooth)

            fig_tta_cmp = go.Figure()
            for i, (label, exp) in enumerate(experiments.items()):
                df = exp["df"]
                if "cumulative_sim_time_s" not in df.columns or "test_accuracy" not in df.columns:
                    continue
                legend_label = _short_graph_label(label)
                color = COLOR_MAP[i % len(COLOR_MAP)]
                y_raw  = df["test_accuracy"] * 100
                y_plot = ema_smooth(y_raw, _cmp_alpha) if _cmp_smooth else y_raw
                if _cmp_smooth:
                    fig_tta_cmp.add_trace(go.Scatter(
                        x=df["cumulative_sim_time_s"], y=y_raw,
                        mode="lines", line=dict(color=color, width=0.8),
                        opacity=0.2, showlegend=False, hoverinfo="skip",
                    ))
                fig_tta_cmp.add_trace(go.Scatter(
                    x=df["cumulative_sim_time_s"],
                    y=y_plot,
                    mode="lines",
                    name=legend_label,
                    line=dict(color=color, width=2.5),
                ))
            fig_tta_cmp.update_layout(
                title="Time-to-Accuracy (simulated parallel time) — left-shifted = faster",
                xaxis_title="Cumulative Simulated Time (s)",
                yaxis_title="Test Accuracy (%)",
                template="plotly_white",
                height=460,
                font=dict(family="Inter"),
                legend=dict(
                    orientation="v", xanchor="left", x=1.02,
                    yanchor="middle", y=0.5,
                    font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#e2e8f0", borderwidth=1,
                ),
                margin=dict(r=180),
            )
            st.plotly_chart(fig_tta_cmp, width="stretch")
            _tta_caption = (
                "sim_round_time_s = max(client training times) per round — "
                "models true distributed wall-clock time. cumulative_sim_time_s is the running total."
            )
            if _cmp_smooth:
                _tta_caption += (
                    f" | 📈 EMA smoothed (α={_cmp_alpha:.2f}) for clarity — raw data shown transparently."
                )
            st.caption(_tta_caption)

        # ── Simulated round time bar chart ────────────────────────────────────
        has_round_time_cmp = any(
            "sim_round_time_s" in exp["df"].columns
            for exp in experiments.values()
        )
        if has_round_time_cmp:
            st.markdown(
                "**Simulated round time per round** — "
                "reveals whether certain algorithms have heavier rounds due to stragglers."
            )
            fig_rt_cmp = go.Figure()
            for i, (label, exp) in enumerate(experiments.items()):
                df = exp["df"]
                if "sim_round_time_s" not in df.columns:
                    continue
                legend_label = _short_graph_label(label)
                fig_rt_cmp.add_trace(go.Bar(
                    x=_round_col(df),
                    y=df["sim_round_time_s"],
                    name=legend_label,
                    marker_color=COLOR_MAP[i % len(COLOR_MAP)],
                    opacity=0.75,
                ))
            fig_rt_cmp.update_layout(
                title="Simulated Round Time (s) — max(client training times) per round",
                xaxis_title="Round",
                yaxis_title="Round Time (s)",
                barmode="group",
                template="plotly_white",
                height=380,
                font=dict(family="Inter"),
                legend=dict(
                    orientation="v", xanchor="left", x=1.02,
                    yanchor="middle", y=0.5,
                    font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#e2e8f0", borderwidth=1,
                ),
                margin=dict(r=180),
            )
            st.plotly_chart(fig_rt_cmp, width="stretch")

        # ── Summary: sim time KPIs ────────────────────────────────────────────
        if has_sim_time_cmp or has_round_time_cmp:
            st.markdown("**Convergence speed summary (simulated time)**")
            cs_cmp_rows = []
            for label, exp in experiments.items():
                df    = exp["df"]
                meta  = exp["config"]
                summary = meta.get("summary", {})
                total_sim = summary.get("total_sim_time_s", None)
                if total_sim is None and "sim_round_time_s" in df.columns:
                    total_sim = float(df["sim_round_time_s"].sum())
                r2best = summary.get("rounds_to_best_acc", None)
                t2best = summary.get("sim_time_to_best_acc", None)
                cs_cmp_rows.append({
                    "Experiment":           label,
                    "Total Sim Time":       _fmt_seconds(total_sim),
                    "Rounds to Best Acc":   str(int(r2best)) if r2best is not None else "—",
                    "Sim Time to Best Acc": _fmt_seconds(t2best),
                })
            if cs_cmp_rows:
                st.dataframe(pd.DataFrame(cs_cmp_rows), width="stretch", hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page: SURVIVAL & FAIRNESS  (FedPartBE evaluation)
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Survival & Fairness":

    st.markdown("""
    <div class="results-header">
      <div class="results-header-title">Survival &amp; Fairness Analysis</div>
      <div class="results-header-sub">
        System lifetime, client survival curves, energy fairness (Jain index),
        and layer staleness — key metrics for FedPartBE evaluation.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not result_dirs:
        st.info("Run experiments with heterogeneous clients to use this page.")
        st.code(
            "python run_experiment.py --benchmark \\\n"
            "    --algos fedavg,fedprox,fedpart,heterofl,fjord,fedpart_be \\\n"
            "    --rounds 200 --clients 30 --alpha 0.5 \\\n"
            "    --output results/fedpartbe_comparison",
            language="bash"
        )
        st.stop()

    if not selected_dirs:
        st.info("Select experiments from the sidebar.")
        st.stop()

    experiments_sf = {}
    for name in selected_dirs:
        config, df = load_experiment(name)
        label = _get_exp_label(pathlib.Path(name), RESULTS_DIR)
        experiments_sf[label] = {"config": config, "df": df}

    tab_surv, tab_lifetime, tab_pareto, tab_fair = st.tabs([
        "Survival Curves",
        "System Lifetime Table",
        "Energy-Accuracy Pareto",
        "Bytes & Fairness",
    ])

    # ── Survival Curves ───────────────────────────────────────────────────────
    with tab_surv:
        st.markdown(
            "**Client survival curves** — fraction of clients still alive (battery > 0) "
            "at each round. Uses `survival_ratio` when available (falls back to `participation_rate`). "
            "A higher / flatter curve = better resource management."
        )

        # Prefer survival_ratio (battery-based), fallback to num_alive_clients, then participation_rate
        def _survival_col(df):
            if "survival_ratio" in df.columns:
                return df["survival_ratio"]
            if "num_alive_clients" in df.columns:
                # normalise to 0-1 using max value
                max_alive = df["num_alive_clients"].max()
                if max_alive > 0:
                    return df["num_alive_clients"] / max_alive
            if "participation_rate" in df.columns:
                return df["participation_rate"]
            return None

        has_survival = any(
            _survival_col(exp["df"]) is not None
            for exp in experiments_sf.values()
        )

        if not has_survival:
            st.info(
                "No `survival_ratio`, `num_alive_clients`, or `participation_rate` column found. "
                "Make sure you are using the latest version of `run_experiment.py`."
            )
        else:
            # ── Per-algorithm color assignment (by label keyword matching) ─────
            _ALGO_COLORS = {
                "fedavg":    "#ef4444",   # red
                "fedpart_be": "#f97316",  # orange
                "fedpart":   "#06b6d4",   # cyan
                "fedprox":   "#22c55e",   # green
                "heterofl":  "#8b5cf6",   # purple
                "fjord":     "#92400e",   # brown
            }

            def _algo_color(label_str, fallback_idx):
                lw = label_str.lower()
                for key, col in _ALGO_COLORS.items():
                    if key in lw:
                        return col
                return COLOR_MAP[fallback_idx % len(COLOR_MAP)]

            # ── Compute milestones for reference lines ─────────────────────
            # Collect global first_drop, half_drop, all_dead across all algorithms
            _global_first_drop = None
            _global_half_drop  = None
            _global_all_dead   = None
            for _lbl, _exp in experiments_sf.items():
                _s = _survival_col(_exp["df"])
                if _s is None:
                    continue
                _rds = _round_col(_exp["df"])
                _fd = next((int(r) for r, v in zip(_rds, _s) if v < 1.0), None)
                _hd = next((int(r) for r, v in zip(_rds, _s) if v <= 0.5), None)
                _ad = next((int(r) for r, v in zip(_rds, _s) if v == 0.0), None)
                if _fd is not None and (_global_first_drop is None or _fd < _global_first_drop):
                    _global_first_drop = _fd
                if _hd is not None and (_global_half_drop is None or _hd < _global_half_drop):
                    _global_half_drop = _hd
                if _ad is not None and (_global_all_dead is None or _ad < _global_all_dead):
                    _global_all_dead = _ad

            fig_surv = go.Figure()
            for i, (label, exp) in enumerate(experiments_sf.items()):
                df = exp["df"]
                s = _survival_col(df)
                if s is None:
                    continue
                color = _algo_color(label, i)
                fig_surv.add_trace(go.Scatter(
                    x=_round_col(df),
                    y=s * 100,
                    name=_short_graph_label(label),
                    line=dict(color=color, width=3),
                ))
                # ── Best-accuracy-before-death annotation ──────────────────
                if "test_accuracy" in df.columns:
                    try:
                        best_acc_val = float(df["test_accuracy"].max()) * 100
                        best_acc_idx = df["test_accuracy"].idxmax()
                        best_acc_round = int(_round_col(df).iloc[best_acc_idx])
                        best_acc_surv = float(s.iloc[best_acc_idx]) * 100
                        fig_surv.add_trace(go.Scatter(
                            x=[best_acc_round],
                            y=[best_acc_surv],
                            mode="markers+text",
                            name=f"{_short_graph_label(label)} best acc",
                            marker=dict(symbol="star", size=14, color=color,
                                        line=dict(width=1, color="white")),
                            text=[f"Best: {best_acc_val:.1f}%"],
                            textposition="top center",
                            showlegend=False,
                        ))
                    except (TypeError, ValueError, IndexError):
                        pass

            # ── Vertical reference lines at key survival milestones ────────
            if _global_first_drop is not None:
                fig_surv.add_vline(
                    x=_global_first_drop, line_dash="dash", line_color="#64748b", line_width=1.5,
                    annotation_text=f"First dropout (r={_global_first_drop})",
                    annotation_position="top right",
                    annotation_font=dict(size=10, color="#64748b"),
                )
            if _global_half_drop is not None:
                fig_surv.add_vline(
                    x=_global_half_drop, line_dash="dash", line_color="#f59e0b", line_width=1.5,
                    annotation_text=f"50% dropout (r={_global_half_drop})",
                    annotation_position="top right",
                    annotation_font=dict(size=10, color="#d97706"),
                )
            if _global_all_dead is not None:
                fig_surv.add_vline(
                    x=_global_all_dead, line_dash="dash", line_color="#ef4444", line_width=1.5,
                    annotation_text=f"All dead (r={_global_all_dead})",
                    annotation_position="top right",
                    annotation_font=dict(size=10, color="#dc2626"),
                )

            fig_surv.update_layout(
                title="% Clients Alive per Round  (Survival Curve)",
                xaxis_title="Round",
                yaxis_title="% Clients Alive",
                yaxis_range=[0, 115],
                template="plotly_white",
                height=480,
                font=dict(family="Inter"),
                legend=dict(orientation="v", xanchor="left", x=1.02, y=0.5,
                            font=dict(size=11), bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="#e2e8f0", borderwidth=1),
                margin=dict(r=240),
            )
            st.plotly_chart(fig_surv, width="stretch")
            st.caption(
                "`survival_ratio` = clients with battery > 0 / total clients. "
                "`num_alive_clients` is normalised by the max value when `survival_ratio` is absent. "
                "Stars mark the round with the best test accuracy. "
                "FedPartBE should maintain a flatter curve than FedPart by assigning "
                "cheapest layer groups to low-battery clients. "
                "Colors: red=FedAvg, cyan=FedPart, orange=FedPartBE, green=FedProx, purple=HeteroFL, brown=FjORD."
            )

            # ── Summary table ──────────────────────────────────────────────
            st.markdown("##### Summary: Algorithm | First Dropout | 50% Dropout Round | All Dead Round | Best Acc")
            milestone_rows = []
            for label, exp in experiments_sf.items():
                df = exp["df"]
                s = _survival_col(df)
                if s is None:
                    continue
                rounds = _round_col(df)
                first_drop = next((int(r) for r, v in zip(rounds, s) if v < 1.0), None)
                half_drop  = next((int(r) for r, v in zip(rounds, s) if v <= 0.5), None)
                all_dead   = next((int(r) for r, v in zip(rounds, s) if v == 0.0), None)
                best_acc_ms = None
                if "test_accuracy" in df.columns:
                    try:
                        best_acc_ms = round(float(df["test_accuracy"].max()) * 100, 2)
                    except (TypeError, ValueError):
                        pass
                milestone_rows.append({
                    "Algorithm": label,
                    "First Dropout (round)": first_drop if first_drop is not None else "—",
                    "50% Dropout Round":     half_drop  if half_drop  is not None else ">max",
                    "All Dead Round":        all_dead   if all_dead   is not None else ">max",
                    "Best Acc (%)":          best_acc_ms if best_acc_ms is not None else "—",
                })
            if milestone_rows:
                st.dataframe(pd.DataFrame(milestone_rows), hide_index=True, width="stretch")

            # ── Jain fairness over rounds (if available) ───────────────────
            has_jain_surv = any("jain_index" in exp["df"].columns for exp in experiments_sf.values())
            if has_jain_surv:
                st.markdown("##### Jain Fairness Index over Rounds")
                fig_jain_surv = go.Figure()
                for i, (label, exp) in enumerate(experiments_sf.items()):
                    df = exp["df"]
                    if "jain_index" not in df.columns:
                        continue
                    fig_jain_surv.add_trace(go.Scatter(
                        x=_round_col(df), y=df["jain_index"],
                        name=_short_graph_label(label),
                        line=dict(color=_algo_color(label, i), width=2),
                    ))
                fig_jain_surv.add_hline(y=1.0, line_dash="dot", line_color="#22c55e",
                                        annotation_text="Perfect fairness")
                fig_jain_surv.add_hline(y=0.8, line_dash="dot", line_color="#f59e0b",
                                        annotation_text="0.8 threshold")
                fig_jain_surv.update_layout(
                    xaxis_title="Round", yaxis_title="Jain Index",
                    yaxis_range=[0, 1.05], template="plotly_white",
                    height=320, font=dict(family="Inter"),
                    legend=dict(orientation="v", xanchor="left", x=1.02, y=0.5,
                                font=dict(size=11), bgcolor="rgba(255,255,255,0.8)",
                                bordercolor="#e2e8f0", borderwidth=1),
                    margin=dict(r=220),
                )
                st.plotly_chart(fig_jain_surv, width="stretch")

        # ── Accuracy × Survival joint view ───────────────────────────────────
        st.markdown("---")
        st.markdown("**Accuracy × Participation tradeoff** — does higher survival come at an accuracy cost?")
        joint_rows = []
        for i, (label, exp) in enumerate(experiments_sf.items()):
            df = exp["df"]
            final_acc = float(df["test_accuracy"].iloc[-1]) * 100 if "test_accuracy" in df.columns else 0.0
            final_part = float(df["participation_rate"].iloc[-1]) * 100 if "participation_rate" in df.columns else 100.0
            best_acc = float(df["test_accuracy"].max()) * 100 if "test_accuracy" in df.columns else 0.0
            joint_rows.append({
                "Algorithm": label,
                "Final Accuracy (%)": round(final_acc, 2),
                "Best Accuracy (%)": round(best_acc, 2),
                "Final Participation (%)": round(final_part, 1),
                "Color": COLOR_MAP[i % len(COLOR_MAP)],
            })
        if joint_rows:
            jdf = pd.DataFrame(joint_rows)
            fig_joint = px.scatter(
                jdf, x="Final Participation (%)", y="Best Accuracy (%)",
                color="Algorithm", text="Algorithm",
                color_discrete_sequence=COLOR_MAP,
                template="plotly_white",
                title="Best Accuracy vs Final Participation Rate",
            )
            fig_joint.update_traces(textposition="top center", marker_size=12)
            fig_joint.update_layout(height=420, font=dict(family="Inter"), showlegend=False)
            st.plotly_chart(fig_joint, width="stretch")
            st.caption("Upper-right = ideal: high accuracy AND high participation (clients survive longer).")

    # ── Energy Fairness ───────────────────────────────────────────────────────
    with tab_fair:
        st.markdown(
            "**Energy fairness (Jain index)** — measures how evenly energy is "
            "consumed across clients. Jain index = 1 → perfect fairness (all clients "
            "consume equally). Jain → 0 → extreme unfairness (few clients do all the work). "
            "FedPartBE targets a higher Jain index than FedPart."
        )

        has_jain = any("jain_index" in exp["df"].columns for exp in experiments_sf.values())

        if not has_jain:
            st.info("No `jain_index` column found in selected experiments.")
        else:
            # Jain index over rounds
            fig_jain = go.Figure()
            for i, (label, exp) in enumerate(experiments_sf.items()):
                df = exp["df"]
                if "jain_index" not in df.columns:
                    continue
                fig_jain.add_trace(go.Scatter(
                    x=_round_col(df), y=df["jain_index"],
                    name=_short_graph_label(label),
                    line=dict(color=COLOR_MAP[i % len(COLOR_MAP)], width=2.5),
                ))
            fig_jain.add_hline(y=1.0, line_dash="dot", line_color="#22c55e",
                               annotation_text="Perfect fairness")
            fig_jain.add_hline(y=0.8, line_dash="dot", line_color="#f59e0b",
                               annotation_text="Fairness threshold (0.8)")
            fig_jain.update_layout(
                title="Jain Fairness Index per Round (energy)",
                xaxis_title="Round", yaxis_title="Jain Index",
                yaxis_range=[0, 1.05],
                template="plotly_white", height=400, font=dict(family="Inter"),
                legend=dict(orientation="v", xanchor="left", x=1.02, y=0.5,
                            font=dict(size=11), bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="#e2e8f0", borderwidth=1),
                margin=dict(r=200),
            )
            st.plotly_chart(fig_jain, width="stretch")

            # Final Jain index bar chart
            st.markdown("**Final Jain fairness index — algorithm comparison**")
            jain_bars = [
                (label, float(exp["df"]["jain_index"].iloc[-1]))
                for label, exp in experiments_sf.items()
                if "jain_index" in exp["df"].columns
            ]
            if jain_bars:
                jain_bars.sort(key=lambda x: -x[1])
                fig_jbar = go.Figure(go.Bar(
                    x=[x[0] for x in jain_bars],
                    y=[x[1] for x in jain_bars],
                    marker_color=[COLOR_MAP[i % len(COLOR_MAP)] for i in range(len(jain_bars))],
                    marker_line_color="white", marker_line_width=1.5,
                    text=[f"{v:.3f}" for _, v in jain_bars],
                    textposition="outside",
                ))
                fig_jbar.add_hline(y=0.8, line_dash="dot", line_color="#f59e0b",
                                   annotation_text="0.8 threshold")
                fig_jbar.update_layout(
                    title="Final Jain Fairness Index (higher = more fair)",
                    yaxis_range=[0, 1.1], template="plotly_white",
                    height=380, font=dict(family="Inter"), xaxis_tickangle=-20,
                )
                st.plotly_chart(fig_jbar, width="stretch")

        # Battery evolution
        st.markdown("---")
        st.markdown("**Average battery remaining per round**")
        has_batt = any("avg_battery_j" in exp["df"].columns for exp in experiments_sf.values())
        if has_batt:
            fig_batt = go.Figure()
            for i, (label, exp) in enumerate(experiments_sf.items()):
                df = exp["df"]
                if "avg_battery_j" not in df.columns:
                    continue
                b0 = float(df["avg_battery_j"].iloc[0]) or 1.0
                fig_batt.add_trace(go.Scatter(
                    x=_round_col(df),
                    y=df["avg_battery_j"] / b0 * 100,
                    name=_short_graph_label(label),
                    line=dict(color=COLOR_MAP[i % len(COLOR_MAP)], width=2),
                ))
            fig_batt.update_layout(
                title="Average Battery Remaining (% of initial)",
                xaxis_title="Round", yaxis_title="Battery (%)",
                yaxis_range=[0, 105], template="plotly_white",
                height=380, font=dict(family="Inter"),
                legend=dict(orientation="v", xanchor="left", x=1.02, y=0.5,
                            bgcolor="rgba(255,255,255,0.8)", bordercolor="#e2e8f0", borderwidth=1),
                margin=dict(r=200),
            )
            st.plotly_chart(fig_batt, width="stretch")
            st.caption("A slower drain = more energy-efficient algorithm. FedPartBE should drain more slowly than FedPart on expensive groups.")

    # ── System Lifetime Table ─────────────────────────────────────────────────
    with tab_lifetime:
        st.markdown(
            "**System lifetime table** — for a given target accuracy, how many rounds "
            "does it take, how many clients are still alive, and how much energy was spent?"
        )

        # ── Table A — Paper-ready survival summary ────────────────────────
        st.markdown("#### Table A — Paper-ready Survival Summary")
        st.caption(
            "**Column definitions — ** "
            "**Best Acc (%)**: peak test accuracy across all 200 rounds. "
            "**@ Round**: round at which that peak was reached. "
            "**1st Dropout**: first round a client died (battery = 0). "
            "**50% Alive**: round when half the fleet has died — system lifetime indicator. "
            "**All Dead**: last round where at least one client is alive. "
            "**Total Energy (kJ)**: cumulative energy consumed by the whole fleet. "
            "**Total Comm (MB)**: total bytes uploaded to the server. "
            "**Jain@BestAcc**: Jain fairness index J=(ΣE_k)²/(K·ΣE_k²) at the best-accuracy round — "
            "closer to 1.0 = more equitable energy distribution."
        )

        def _survival_col(df):
            if "survival_ratio" in df.columns:
                return df["survival_ratio"]
            if "participation_rate" in df.columns:
                return df["participation_rate"]
            return None

        table_a_rows = []
        for label, exp in experiments_sf.items():
            df = exp["df"]
            s = _survival_col(df)
            rounds = _round_col(df)

            # Survival milestones
            first_drop = next((int(r) for r, v in zip(rounds, s) if v < 1.0), None) if s is not None else None
            half_drop  = next((int(r) for r, v in zip(rounds, s) if v <= 0.5), None) if s is not None else None
            all_dead   = next((int(r) for r, v in zip(rounds, s) if v == 0.0), None) if s is not None else None

            # Accuracy
            best_acc   = float(df["test_accuracy"].max() * 100) if "test_accuracy" in df.columns else None
            best_round = int(df.loc[df["test_accuracy"].idxmax(), rounds.name]) if "test_accuracy" in df.columns else None

            # Total energy & comm at last round
            total_energy = float(df["cumulative_energy_j"].iloc[-1]) if "cumulative_energy_j" in df.columns else None
            total_comm   = float(df["cumulative_bytes"].iloc[-1]) / 1e6 if "cumulative_bytes" in df.columns else None

            # Jain at best_acc round
            if best_round is not None and "jain_index" in df.columns:
                mask = rounds == best_round
                jain_at_best = float(df.loc[mask, "jain_index"].iloc[0]) if mask.any() else None
            else:
                jain_at_best = None

            table_a_rows.append({
                "Algorithm":          label,
                "Best Acc (%)":       f"{best_acc:.2f}" if best_acc is not None else "—",
                "@ Round":            best_round if best_round is not None else "—",
                "1st Dropout":        first_drop if first_drop is not None else "—",
                "50% Alive":          half_drop  if half_drop  is not None else ">200",
                "All Dead":           all_dead   if all_dead   is not None else ">200",
                "Total Energy (kJ)":  f"{total_energy/1000:.1f}" if total_energy is not None else "—",
                "Total Comm (MB)":    f"{total_comm:.1f}" if total_comm is not None else "—",
                "Jain@BestAcc":       f"{jain_at_best:.3f}" if jain_at_best is not None else "—",
            })

        if table_a_rows:
            ta_df = pd.DataFrame(table_a_rows)
            st.dataframe(ta_df, hide_index=True, width="stretch")
            # CSV download
            csv_a = ta_df.to_csv(index=False).encode()
            st.download_button(
                "Download Table A (CSV)",
                csv_a,
                file_name="table_A_survival_summary.csv",
                mime="text/csv",
            )

        st.markdown("---")

        target_sf = st.slider("Target accuracy (%)", 10.0, 99.0, 70.0, step=5.0, key="sf_target")

        lt_rows = []
        for label, exp in experiments_sf.items():
            df = exp["df"]
            if "test_accuracy" not in df.columns:
                continue
            acc = df["test_accuracy"] * 100
            mask = acc >= target_sf
            row = {"Algorithm": label}
            if mask.any():
                idx = mask.idxmax()
                row["Rounds to target"] = int(_round_col(df).loc[idx])
                row["Energy (J)"] = round(float(df["cumulative_energy_j"].loc[idx]), 1) if "cumulative_energy_j" in df.columns else None
                row["Bytes (MB)"] = round(float(df["cumulative_bytes"].loc[idx]) / 1e6, 2) if "cumulative_bytes" in df.columns else None
                row["Participation at target (%)"] = round(float(df["participation_rate"].loc[idx]) * 100, 1) if "participation_rate" in df.columns else None
                row["Jain at target"] = round(float(df["jain_index"].loc[idx]), 3) if "jain_index" in df.columns else None
                row["Reached"] = True
            else:
                row.update({"Rounds to target": None, "Energy (J)": None,
                            "Bytes (MB)": None, "Participation at target (%)": None,
                            "Jain at target": None, "Reached": False})
            lt_rows.append(row)

        if lt_rows:
            lt_df = pd.DataFrame(lt_rows)
            reached = lt_df[lt_df["Reached"] == True].drop(columns=["Reached"])
            not_reached = lt_df[lt_df["Reached"] == False]["Algorithm"].tolist()

            if not reached.empty:
                reached = reached.sort_values("Energy (J)")
                st.dataframe(reached, hide_index=True, width="stretch")
                st.caption(
                    "Participation at target = % of original 30 clients still active when accuracy X% is reached. "
                    "Higher = better system lifetime. Jain at target = energy fairness at that round."
                )

            if not_reached:
                st.warning(f"Did not reach {target_sf:.0f}%: {', '.join(not_reached)}")

    # ── Energy-Accuracy Pareto (NeurIPS key figure) ───────────────────────────
    with tab_pareto:
        st.markdown(
            "**Energy-Accuracy Pareto Frontier** — the key figure for the paper. "
            "Each bubble = one algorithm. "
            "**Ideal position: upper-left** (high accuracy, low total energy). "
            "**Bubble size** ∝ system lifetime (round when 50% of clients are still alive) — "
            "larger bubble = algorithm keeps the fleet alive longer."
        )
        st.caption(
            "**How to read:** X-axis = total energy consumed by the fleet over 200 rounds (kJ). "
            "Y-axis = best test accuracy (%). "
            "Bubble size = survival_50pct round (larger = longer fleet lifetime). "
            "An algorithm is Pareto-dominant if no other algorithm is simultaneously more accurate AND more efficient."
        )

        # Collect Pareto data
        pareto_data = []
        for i, (label, exp) in enumerate(experiments_sf.items()):
            df = exp["df"]
            meta = exp["config"]

            if "test_accuracy" not in df.columns:
                continue

            best_acc = float(df["test_accuracy"].max() * 100)

            # Total energy in kJ
            if "cumulative_energy_j" in df.columns:
                total_energy_kj = float(df["cumulative_energy_j"].iloc[-1]) / 1000
            elif "total_energy_j" in df.columns:
                total_energy_kj = float(df["total_energy_j"].sum()) / 1000
            else:
                total_energy_kj = 0.0

            # survival_50pct as bubble size proxy
            s_col = None
            if "survival_ratio" in df.columns:
                s_col = df["survival_ratio"]
            elif "participation_rate" in df.columns:
                s_col = df["participation_rate"]

            if s_col is not None:
                rounds_series = _round_col(df)
                mask_50 = s_col <= 0.5
                survival_50 = int(rounds_series[mask_50].iloc[0]) if mask_50.any() else len(df)
            else:
                survival_50 = len(df)

            pareto_data.append({
                "Algorithm": label,
                "Total Energy (kJ)": total_energy_kj,
                "Best Accuracy (%)": best_acc,
                "Survival 50% Round": survival_50,
                "Color": COLOR_MAP[i % len(COLOR_MAP)],
            })

        if not pareto_data:
            st.info("No accuracy or energy data found in selected experiments.")
        else:
            p_df = pd.DataFrame(pareto_data)

            # Normalize bubble size to [8, 40]
            s_min = p_df["Survival 50% Round"].min()
            s_max = p_df["Survival 50% Round"].max()
            s_range = max(s_max - s_min, 1)
            p_df["bubble_size"] = 8 + 32 * (p_df["Survival 50% Round"] - s_min) / s_range

            fig_par = go.Figure()
            for _, row in p_df.iterrows():
                fig_par.add_trace(go.Scatter(
                    x=[row["Total Energy (kJ)"]],
                    y=[row["Best Accuracy (%)"]],
                    mode="markers+text",
                    name=row["Algorithm"],
                    text=[row["Algorithm"].split("/")[0]],
                    textposition="top center",
                    textfont=dict(size=11, color="#1e293b"),
                    marker=dict(
                        color=row["Color"],
                        size=row["bubble_size"],
                        line=dict(width=2, color="white"),
                        opacity=0.85,
                    ),
                    showlegend=True,
                    customdata=[[
                        row["Total Energy (kJ)"],
                        row["Best Accuracy (%)"],
                        row["Survival 50% Round"],
                    ]],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Energy: %{customdata[0]:.1f} kJ<br>"
                        "Best Acc: %{customdata[1]:.1f}%<br>"
                        "Survival 50%: round %{customdata[2]}<extra></extra>"
                    ),
                ))

            fig_par.update_layout(
                title="Energy-Accuracy Pareto Frontier — 30 ESP32-S3, CIFAR-10",
                xaxis_title="Total Energy Consumed (kJ)",
                yaxis_title="Best Test Accuracy (%)",
                template="plotly_white",
                height=520,
                font=dict(family="Inter"),
                legend=dict(
                    orientation="v", xanchor="left", x=1.02, y=0.5,
                    font=dict(size=11), bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#e2e8f0", borderwidth=1,
                ),
                margin=dict(r=220, t=70),
            )
            st.plotly_chart(fig_par, width="stretch")
            st.caption(
                "Bubble size = system lifetime (round when 50% clients survive). "
                "Larger bubble = algorithm keeps clients alive longer. "
                "Upper-left + large bubble = best algorithm: high accuracy, low energy, long lifetime. "
                "FedPartUniversal should appear in the upper-left quadrant with the largest bubble."
            )

            # Summary table for paper
            st.markdown("**Summary table (paper-ready)**")
            p_table = p_df[["Algorithm", "Best Accuracy (%)", "Total Energy (kJ)",
                            "Survival 50% Round"]].copy()
            p_table = p_table.sort_values("Best Accuracy (%)", ascending=False)
            p_table["Best Accuracy (%)"] = p_table["Best Accuracy (%)"].map("{:.2f}".format)
            p_table["Total Energy (kJ)"] = p_table["Total Energy (kJ)"].map("{:.1f}".format)
            st.dataframe(p_table, hide_index=True, width="stretch")
            csv_par = p_table.to_csv(index=False).encode()
            st.download_button(
                "Download Pareto Table (CSV)",
                csv_par,
                file_name="pareto_energy_accuracy.csv",
                mime="text/csv",
            )

# ─────────────────────────────────────────────────────────────────────────────
# Footer (all pages)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-top:40px; padding-top:16px; border-top:1px solid #e2e8f0;
            text-align:center; font-size:0.75rem; color:#94a3b8;">
  FedLab ZMQ &nbsp;|&nbsp;
  Energy-Efficient Federated Learning Research &nbsp;|&nbsp;
  J. Nikiema, EL Amhoud, ELHAMMOUTI & KISSAMI &nbsp;|&nbsp;
  Mohammed VI Polytechnic University (UM6P), Benguerir, Morocco
</div>
""", unsafe_allow_html=True)
