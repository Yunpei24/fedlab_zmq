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

try:
    from dashboard.multiseed_summary import (
        load_run_summary,
        style_best_values,
        summarize_runs,
    )
    from dashboard.home import render_home
except ModuleNotFoundError:  # ``streamlit run dashboard/app.py`` from dashboard/
    from multiseed_summary import (  # type: ignore[no-redef]
        load_run_summary,
        style_best_values,
        summarize_runs,
    )
    from home import render_home  # type: ignore[no-redef]


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
    page_title="FedLab ZMQ · Research Console",
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

</style>
""", unsafe_allow_html=True)

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

        elif algo_lower in {"far", "dpfar", "dp_far"}:
            alpha = ac.get("far_alpha")
            E = ac.get("local_epochs", cfg.get("local_epochs", "?"))
            lr = ac.get("lr", cfg.get("lr", "?"))
            if alpha is not None:
                param_parts.append(f"α={alpha:g}")
            param_parts.append(f"E={E}")
            param_parts.append(f"lr={lr}")

            # Reproduction trees place the method variant two levels above
            # the run directory: .../far_alpha_0p4/seed36/<run>. Showing that
            # folder prevents distinct FAR variants from sharing a label.
            if d.parent.name.startswith("seed") and d.parent.parent != d.parent:
                folder_name = d.parent.parent.name

        elif algo_lower == "fedfair":
            fairness_lambda = ac.get("fairness_lambda")
            E = ac.get("local_epochs", cfg.get("local_epochs", "?"))
            if fairness_lambda is not None:
                param_parts.append(f"λ={fairness_lambda:g}")
            param_parts.append(f"E={E}")
            if d.parent.name.startswith("seed") and d.parent.parent != d.parent:
                folder_name = d.parent.parent.name

        elif algo_lower in {"qffl", "q-ffl"}:
            q_value = ac.get("q")
            E = ac.get("local_epochs", cfg.get("local_epochs", "?"))
            if q_value is not None:
                param_parts.append(f"q={q_value:g}")
            param_parts.append(f"E={E}")
            if d.parent.name.startswith("seed") and d.parent.parent != d.parent:
                folder_name = d.parent.parent.name

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
        # Seed disambiguation: multi-seed runs share the same named folder and
        # hyperparams — without the seed, their labels are identical and the
        # sidebar becomes a guessing game. Extracted from the run dir name
        # (…_s43) or the config as fallback.
        _seed_m = re.search(r"_s(\d+)$", d.name)
        _parent_seed_m = re.fullmatch(r"seed(\d+)", d.parent.name)
        _seed = (
            _seed_m.group(1)
            if _seed_m
            else (_parent_seed_m.group(1) if _parent_seed_m else cfg.get("seed"))
        )
        if _seed is not None and f"s{_seed}" not in label:
            label += f" | seed {_seed}"
        return label

    except Exception:
        return folder_name


_GLABEL_RE_ALGO = re.compile(r'\[([^\]]+)\]')
_GLABEL_RE_M    = re.compile(r'\bM=(\S+?)(?:[,\s\(|]|$)')

def _short_graph_label(full_label: str, max_len: int = 36) -> str:
    """
    Concise legend/axis label for graph traces.
    '[FedStep], M=3, μr=0.1 | rotation_30/run_001'
    → 'FedStep · M=3 · rotation_30'

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
    if p.exists():
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


def _has_round_metric(experiments: dict, key: str) -> bool:
    """Return True when at least one selected run contains usable values."""

    return any(
        key in exp["df"].columns and exp["df"][key].notna().any()
        for exp in experiments.values()
    )


def _round_metric_figure(
    experiments: dict,
    metric_specs: list[tuple[str, str, float]],
    *,
    title: str,
    yaxis_title: str,
    height: int = 390,
) -> go.Figure:
    """Build a comparable round curve for one or more scalar diagnostics."""

    fig = go.Figure()
    dash_styles = ["solid", "dash", "dot", "dashdot"]
    for exp_idx, (label, exp) in enumerate(experiments.items()):
        df = exp["df"]
        rounds = _round_col(df)
        color = COLOR_MAP[exp_idx % len(COLOR_MAP)]
        for metric_idx, (key, metric_label, scale) in enumerate(metric_specs):
            if key not in df.columns or not df[key].notna().any():
                continue
            values = pd.to_numeric(df[key], errors="coerce") * scale
            fig.add_trace(
                go.Scatter(
                    x=rounds,
                    y=values,
                    mode="lines+markers",
                    marker=dict(size=4),
                    name=f"{_short_graph_label(label)} · {metric_label}",
                    line=dict(
                        color=color,
                        width=2.3,
                        dash=dash_styles[metric_idx % len(dash_styles)],
                    ),
                )
            )
    fig.update_layout(
        title=title,
        xaxis_title="Round",
        yaxis_title=yaxis_title,
        template="plotly_white",
        height=height,
        font=dict(family="Inter"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


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
        'Federated Learning Research Console'
        '</div>'
        '<div style="font-size:0.72rem; color:#475569; margin-top:6px;">'
        'Fairness · Robustness · Privacy · Energy'
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
        ["Home", "Results", "Compare Algorithms", "Survival & Fairness", "σ² Estimator"],
        label_visibility="collapsed",
        help="Results: single-experiment view. Compare: multi-algorithm benchmarks. Survival: battery & fairness. σ²: gradient variance & optimal M*.",
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
        python run_experiment.py --algo fedstep --rounds 10 --clients 4
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
    selected_label_by_dir = {}
    if page in ("Results", "Compare Algorithms", "Survival & Fairness") and result_dirs:
        # Never construct this mapping with a dict comprehension: two runs can
        # legitimately share most metadata. A duplicate key would silently hide
        # one of them (this previously collapsed FAR alpha variants).
        dir_labels = {}
        for d in result_dirs:
            base_label = _get_exp_label(d, RESULTS_DIR)
            label = base_label
            if label in dir_labels:
                try:
                    relative = d.relative_to(RESULTS_DIR)
                except ValueError:
                    relative = d
                label = f"{base_label} | {relative}"
            duplicate_index = 2
            while label in dir_labels:
                label = f"{base_label} | duplicate {duplicate_index}"
                duplicate_index += 1
            dir_labels[label] = str(d)
        selected_labels = st.multiselect(
            "SELECT EXPERIMENTS",
            options=list(dir_labels.keys()),
            default=None,
            help="Select one or more experiment result folders to visualize. "
                 "Labels are personalized from metrics.json.",
        )
        selected_dirs = [dir_labels[lbl] for lbl in selected_labels]
        selected_label_by_dir = {dir_labels[lbl]: lbl for lbl in selected_labels}

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
    render_home(result_dirs, RESULTS_DIR)

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
            border-radius:4px;">python run_experiment.py --algo fedstep --rounds 10 --clients 4</code>
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
        label = selected_label_by_dir.get(
            str(name), _get_exp_label(pathlib.Path(name), RESULTS_DIR)
        )
        experiments[label] = {"config": config, "df": df, "run_dir": str(name)}

    # Summary metrics bar
    sum_cols = st.columns(len(experiments))
    for col, (label, exp) in zip(sum_cols, experiments.items()):
        df = exp["df"]
        best_acc  = df["test_accuracy"].max() * 100 if "test_accuracy" in df.columns else 0.0
        total_gb  = df["total_bytes"].sum() / 1e9   if "total_bytes" in df.columns else 0.0
        final_j   = df["jain_index"].iloc[-1]        if "jain_index" in df.columns and len(df) > 0 else 0.0
        col.metric(label, f"{best_acc:.1f}%", f"{total_gb:.3f} GB | J={final_j:.3f}")

    st.divider()

    (
        tab_acc,
        tab_energy,
        tab_battery,
        tab_fair,
        tab_robust,
        tab_privacy,
        tab_table,
        tab_convspeed,
        tab_lm,
        tab_sigma2,
        tab_ee,
    ) = st.tabs([
        "Accuracy & Loss",
        "Energy per Round",
        "Battery & Survival",
        "Client Fairness",
        "Robustness & FAR",
        "Privacy",
        "Summary Table",
        "Convergence Speed",
        "Layer Mismatch",
        "σ² Evolution",
        "Early-Exit & CCVR",
    ])

    # ── Early-Exit & CCVR ────────────────────────────────────────────────────
    # (a) Per-exit accuracy curves for runs that log exit_acc_1/2/3 (all runs
    #     after the per-exit eval landed). (b) CCVR post-hoc calibration
    #     before/after, read from results/ccvr_perexit_*.json emitted by
    #     scripts/ccvr_per_exit.py.
    with tab_ee:
        st.subheader("Accuracy par exit (entraînement)")
        _ee_exps = {lbl: e for lbl, e in experiments.items()
                    if "exit_acc_1" in e["df"].columns}
        if not _ee_exps:
            st.info("Aucune expérience sélectionnée ne logge exit_acc_1/2/3 "
                    "(runs antérieurs à l'éval par exit).")
        else:
            _sel = st.selectbox("Expérience", list(_ee_exps.keys()), key="ee_sel")
            _df = _ee_exps[_sel]["df"]
            _rounds = _round_col(_df)
            fig_ee = go.Figure()
            for d, name, color in ((1, "exit 1 (41% FLOPs)", "#8ecae6"),
                                   (2, "exit 2 (71% FLOPs)", "#219ebc"),
                                   (3, "tête finale (100%)", "#023047")):
                col = f"exit_acc_{d}"
                if col in _df.columns:
                    fig_ee.add_trace(go.Scatter(
                        x=_rounds, y=_df[col] * 100, name=name,
                        mode="lines", line=dict(color=color, width=2)))
            fig_ee.update_layout(xaxis_title="Round",
                                 yaxis_title="Test accuracy (%)",
                                 height=380, legend=dict(orientation="h"))
            st.plotly_chart(fig_ee, width="stretch")

        st.divider()
        st.subheader("Calibration CCVR par exit (post-hoc)")
        _ccvr_files = sorted(RESULTS_DIR.glob("ccvr_perexit_*.json"))
        if not _ccvr_files:
            st.info("Aucun résultat CCVR (results/ccvr_perexit_*.json). "
                    "Générer avec scripts/ccvr_per_exit.py.")
        else:
            _rows = []
            for p in _ccvr_files:
                try:
                    _c = json.loads(p.read_text())
                    _seed = _c.get("seed", p.stem.split("_")[-1])
                    for d in ("1", "2", "3"):
                        _rows.append({"seed": f"s{_seed}",
                                      "exit": f"exit {d}",
                                      "avant": _c["before"][d] * 100,
                                      "après": _c["after"][d] * 100})
                except Exception as exc:
                    st.warning(f"{p.name}: {exc}")
            if _rows:
                _cdf = pd.DataFrame(_rows)
                _cdf["Δ"] = _cdf["après"] - _cdf["avant"]
                fig_c = go.Figure()
                _x = [f"{r['seed']} · {r['exit']}" for _, r in _cdf.iterrows()]
                fig_c.add_trace(go.Bar(name="avant CCVR", x=_x,
                                       y=_cdf["avant"], marker_color="#adb5bd"))
                fig_c.add_trace(go.Bar(name="après CCVR", x=_x,
                                       y=_cdf["après"], marker_color="#2a9d8f",
                                       text=[f"{v:+.1f}" for v in _cdf["Δ"]],
                                       textposition="outside"))
                fig_c.update_layout(barmode="group", height=420,
                                    yaxis_title="Test accuracy (%)",
                                    legend=dict(orientation="h"))
                st.plotly_chart(fig_c, width="stretch")
                st.dataframe(
                    _cdf.round(1), width="stretch", hide_index=True)

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
        _zoom_bat = st.toggle(
            "Zoom y-axis to highlight differences",
            value=False,
            key="battery_zoom",
            help="Starts the y-axis near the minimum value so small differences between algorithms become visible. "
                 "Applies to all bar/histogram charts in this tab.",
        )
        _zoom_note = " | Zoom actif — axe Y tronqué pour mettre en évidence les écarts." if _zoom_bat else ""

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
                text=[f"{v:.3f} GB" for v in bar_vals],
                textposition="outside",
                showlegend=False,
            ), row=1, col=2)

        fig2.update_xaxes(title_text="Round", row=1, col=1)
        fig2.update_yaxes(title_text="GB", row=1, col=1)
        _comm_range = [min(bar_vals) * 0.97, max(bar_vals) * 1.03] if _zoom_bat and len(bar_vals) >= 2 else None
        fig2.update_yaxes(title_text="GB", row=1, col=2,
                          **({'range': _comm_range} if _comm_range else {}))
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
            f"Lower total bytes = less communication overhead — key metric for bandwidth-constrained IoT.{_zoom_note}"
        )

        # ── Total Energy Consumed per Round ──────────────────────────────────
        st.markdown("---")
        st.markdown("**Total Energy Consumed per Round**")
        has_energy_per_round = any(
            "total_energy_j" in exp["df"].columns for exp in experiments.values()
        )
        if has_energy_per_round:
            _all_rnd_vals = [
                v for exp in experiments.values()
                if "total_energy_j" in exp["df"].columns
                for v in exp["df"]["total_energy_j"].dropna().tolist()
            ]
            _enrnd_range = (
                [min(_all_rnd_vals) * 0.97, max(_all_rnd_vals) * 1.03]
                if _zoom_bat and _all_rnd_vals else None
            )
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
                yaxis=dict(range=_enrnd_range) if _enrnd_range else {},
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
                f"Algorithms that skip clients (cyclic scheduling) show lower per-round energy but may need more rounds.{_zoom_note}"
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
            _en_range = (
                [min(total_en_vals) * 0.97, max(total_en_vals) * 1.03]
                if _zoom_bat and len(total_en_vals) >= 2 else None
            )
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
                yaxis=dict(range=_en_range) if _en_range else {},
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
                f"Lower = more energy-efficient over the full experiment.{_zoom_note}"
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
        st.markdown(
            "Client-level fairness is evaluated on the distribution of the global "
            "model's performance across honest clients. These are **oracle research "
            "metrics**: Byzantine client identities are excluded only for evaluation, "
            "never supplied to the training algorithm."
        )

        _has_tail = any(
            _has_round_metric(experiments, key)
            for key in (
                "worst20_accuracy_pct",
                "best20_accuracy_pct",
                "worst20_balanced_accuracy_pct",
            )
        )
        _has_dispersion = any(
            _has_round_metric(experiments, key)
            for key in (
                "best20_worst20_gap_pct",
                "client_accuracy_variance_pct2",
                "balanced_performance_fairness",
                "client_balanced_accuracy_variance_pct2",
                "best_worst_balanced_accuracy_gap_pct",
            )
        )
        if _has_tail or _has_dispersion:
            _fair_col1, _fair_col2 = st.columns(2)
            with _fair_col1:
                if _has_tail:
                    st.plotly_chart(
                        _round_metric_figure(
                            experiments,
                            [
                                ("worst20_accuracy_pct", "Worst-20", 1.0),
                                ("best20_accuracy_pct", "Best-20", 1.0),
                            ],
                            title="Best-20 and Worst-20 client accuracy",
                            yaxis_title="Accuracy (%)",
                        ),
                        width="stretch",
                    )
                    st.caption(
                        "Worst-20 is the mean accuracy of the lowest-performing 20% "
                        "of evaluated honest clients. A higher Worst-20 and a smaller "
                        "Best-20/Worst-20 gap indicate better client-level inclusion."
                    )
                else:
                    st.info("No Best-20/Worst-20 metrics in the selected runs.")
            with _fair_col2:
                if _has_dispersion:
                    st.plotly_chart(
                        _round_metric_figure(
                            experiments,
                            [
                                ("best20_worst20_gap_pct", "Tail gap", 1.0),
                                ("client_accuracy_variance_pct2", "Accuracy variance", 1.0),
                            ],
                            title="Across-client performance dispersion",
                            yaxis_title="Percentage-point scale",
                        ),
                        width="stretch",
                    )
                    st.caption(
                        "Tail gap is measured in percentage points. Accuracy variance is "
                        "reported in percentage-points squared; both should decrease when "
                        "performance becomes more uniform across clients."
                    )
                else:
                    st.info("No across-client dispersion metrics in the selected runs.")

            if _has_round_metric(experiments, "balanced_performance_fairness"):
                st.plotly_chart(
                    _round_metric_figure(
                        experiments,
                        [
                            ("balanced_performance_fairness", "Weighted loss variance", 1.0),
                            ("client_loss_variance", "Unweighted loss variance", 1.0),
                        ],
                        title="FedFDP-style balanced performance fairness",
                        yaxis_title="Loss variance (lower is fairer)",
                    ),
                    width="stretch",
                )
                st.caption(
                    "Balanced performance fairness is the client-data-size-weighted "
                    "variance of local test losses. It complements FAR's tail metrics: "
                    "variance measures uniformity, while Worst-20 exposes the disadvantaged tail."
                )

            _distribution_rows = []
            for label, exp in experiments.items():
                df = exp["df"]
                key = "client_accuracy_values_oracle"
                if key not in df.columns:
                    continue
                usable = df[df[key].apply(lambda value: isinstance(value, list) and len(value) > 0)]
                if usable.empty:
                    continue
                for value in usable.iloc[-1][key]:
                    if value is None:
                        continue
                    _distribution_rows.append(
                        {"Experiment": _short_graph_label(label), "Client accuracy (%)": 100.0 * float(value)}
                    )
            if _distribution_rows:
                _distribution_df = pd.DataFrame(_distribution_rows)
                fig_distribution = px.box(
                    _distribution_df,
                    x="Experiment",
                    y="Client accuracy (%)",
                    points="all",
                    title="Per-client accuracy distribution at the last evaluated round",
                )
                fig_distribution.update_layout(template="plotly_white", height=410)
                st.plotly_chart(fig_distribution, width="stretch")
                st.caption(
                    "Each point is one honest client's local test accuracy. This detailed "
                    "distribution is stored only as an oracle experiment diagnostic."
                )

            _dmd_metric_keys = (
                "worst20_balanced_accuracy_pct",
                "client_balanced_accuracy_variance_pct2",
                "best_worst_balanced_accuracy_gap_pct",
                "canonical_cb_deficit_mean",
                "canonical_cb_deficit_cvar20",
                "fixed_zero_deficit_upper_semivariance",
                "pre_fixed_zero_deficit_upper_semivariance",
                "stale_usv_value",
            )
            if any(_has_round_metric(experiments, key) for key in _dmd_metric_keys):
                st.divider()
                st.markdown("#### Decision-Margin Deficit (DMD)")
                st.caption(
                    "DMD evaluates client fairness in decision space. Balanced accuracy "
                    "gives each observed class equal importance; deficit statistics measure "
                    "negative decision margins. These prototype exports are oracle research "
                    "diagnostics and are not privacy-preserving telemetry."
                )
                _dmd_col1, _dmd_col2 = st.columns(2)
                with _dmd_col1:
                    st.plotly_chart(
                        _round_metric_figure(
                            experiments,
                            [
                                ("worst20_balanced_accuracy_pct", "Worst-20 BA", 1.0),
                                ("mean_client_balanced_accuracy_pct", "Mean BA", 1.0),
                            ],
                            title="DMD client tail performance",
                            yaxis_title="Balanced accuracy (%)",
                        ),
                        width="stretch",
                    )
                    st.caption(
                        "Worst-20 BA is the mean balanced accuracy of the bottom 20% "
                        "of clients. Higher values indicate a better-served client tail."
                    )
                with _dmd_col2:
                    st.plotly_chart(
                        _round_metric_figure(
                            experiments,
                            [
                                (
                                    "client_balanced_accuracy_variance_pct2",
                                    "BA variance",
                                    1.0,
                                ),
                                (
                                    "best_worst_balanced_accuracy_gap_pct",
                                    "Best-Worst BA gap",
                                    1.0,
                                ),
                            ],
                            title="DMD inter-client dispersion",
                            yaxis_title="Percentage-point scale",
                        ),
                        width="stretch",
                    )
                    st.caption(
                        "The Best-Worst curve is the best-client minus worst-client gap, "
                        "not a Best-20 minus Worst-20 gap. Lower variance and gap mean "
                        "more homogeneous client performance."
                    )

                _dmd_deficit_specs = [
                    ("canonical_cb_deficit_mean", "Mean DMD-CB", 1.0),
                    ("canonical_cb_deficit_cvar20", "DMD-CB CVaR-20", 1.0),
                    (
                        "fixed_zero_deficit_upper_semivariance",
                        "Deficit upper-semivariance",
                        1.0,
                    ),
                    (
                        "pre_fixed_zero_deficit_upper_semivariance",
                        "Pre-training deficit upper-semivariance",
                        1.0,
                    ),
                    ("stale_usv_value", "Training stale-USV", 1.0),
                ]
                if any(
                    _has_round_metric(experiments, key)
                    for key, _, _ in _dmd_deficit_specs
                ):
                    st.plotly_chart(
                        _round_metric_figure(
                            experiments,
                            _dmd_deficit_specs,
                            title="Decision-deficit risk over rounds",
                            yaxis_title="Quadratic deficit",
                            height=430,
                        ),
                        width="stretch",
                    )
                    st.caption(
                        "Mean DMD-CB controls the average class-balanced decision deficit; "
                        "CVaR-20 targets its upper tail. Upper-semivariance penalizes only "
                        "clients above the delayed cohort reference. Lower is better."
                    )

                _dmd_distribution_rows = []
                for label, exp in experiments.items():
                    df = exp["df"]
                    key = "client_balanced_accuracy_values_oracle"
                    if key not in df.columns:
                        continue
                    usable = df[
                        df[key].apply(
                            lambda value: isinstance(value, list) and len(value) > 0
                        )
                    ]
                    if usable.empty:
                        continue
                    for value in usable.iloc[-1][key]:
                        if value is None:
                            continue
                        _dmd_distribution_rows.append(
                            {
                                "Experiment": _short_graph_label(label),
                                "Client balanced accuracy (%)": 100.0 * float(value),
                            }
                        )
                if _dmd_distribution_rows:
                    _dmd_distribution_df = pd.DataFrame(_dmd_distribution_rows)
                    _dmd_distribution_fig = px.box(
                        _dmd_distribution_df,
                        x="Experiment",
                        y="Client balanced accuracy (%)",
                        points="all",
                        title="Per-client balanced accuracy at the last evaluated round",
                    )
                    _dmd_distribution_fig.update_layout(
                        template="plotly_white", height=410
                    )
                    st.plotly_chart(_dmd_distribution_fig, width="stretch")

                _dmd_joint_rows = []
                for label, exp in experiments.items():
                    df = exp["df"]
                    required = (
                        "client_balanced_accuracy_values_oracle",
                        "client_dmd_cb_values_oracle",
                    )
                    if not all(key in df.columns for key in required):
                        continue
                    usable = df[
                        df.apply(
                            lambda row: all(
                                isinstance(row[key], list) and len(row[key]) > 0
                                for key in required
                            ),
                            axis=1,
                        )
                    ]
                    if usable.empty:
                        continue
                    row = usable.iloc[-1]
                    for balanced_accuracy, deficit in zip(
                        row["client_balanced_accuracy_values_oracle"],
                        row["client_dmd_cb_values_oracle"],
                    ):
                        if balanced_accuracy is None or deficit is None:
                            continue
                        _dmd_joint_rows.append(
                            {
                                "Experiment": _short_graph_label(label),
                                "Client balanced accuracy (%)": 100.0
                                * float(balanced_accuracy),
                                "DMD-CB deficit": float(deficit),
                            }
                        )
                if _dmd_joint_rows:
                    _dmd_joint_df = pd.DataFrame(_dmd_joint_rows)
                    _dmd_dist_col, _dmd_scatter_col = st.columns(2)
                    with _dmd_dist_col:
                        _dmd_deficit_box = px.box(
                            _dmd_joint_df,
                            x="Experiment",
                            y="DMD-CB deficit",
                            points="all",
                            title="Per-client DMD-CB deficit at final evaluation",
                        )
                        _dmd_deficit_box.update_layout(
                            template="plotly_white", height=410
                        )
                        st.plotly_chart(_dmd_deficit_box, width="stretch")
                    with _dmd_scatter_col:
                        _dmd_scatter = px.scatter(
                            _dmd_joint_df,
                            x="DMD-CB deficit",
                            y="Client balanced accuracy (%)",
                            color="Experiment",
                            title="Decision deficit versus client performance",
                        )
                        _dmd_scatter.update_layout(template="plotly_white", height=410)
                        st.plotly_chart(_dmd_scatter, width="stretch")
                    st.caption(
                        "The scatter diagnoses whether the decision-space signal is aligned "
                        "with the disadvantaged performance tail. It is an oracle analysis, "
                        "not a statistic exposed by a privacy-preserving deployment."
                    )

            st.divider()
            st.markdown("#### Resource and participation fairness")

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
                "**Participation rate** = fraction of the fleet that performed a **real "
                "training update** this round (not merely alive). For full-participation "
                "methods it equals the alive fraction; for selection methods (FedLE) it is "
                "the selected fraction. See the survival-vs-contribution panel below."
            )

        # ── Survie vs Contribution (l'« empty survival ») ────────────────────
        st.markdown("#### Survie (batterie > 0) vs Contribution (entraînement réel)")
        fig_sc = go.Figure()
        contrib_rows = []
        for i, (label, exp) in enumerate(experiments.items()):
            df = exp["df"]
            rounds = _round_col(df)
            col = COLOR_MAP[i % len(COLOR_MAP)]
            surv = df["survival_ratio"] * 100 if "survival_ratio" in df.columns else None
            contrib = df["participation_rate"] * 100 if "participation_rate" in df.columns else None
            if surv is not None:
                fig_sc.add_trace(go.Scatter(
                    x=rounds, y=surv, name=f"{_short_graph_label(label)} · survie",
                    line=dict(color=col, width=2.5)))
            if contrib is not None:
                fig_sc.add_trace(go.Scatter(
                    x=rounds, y=contrib, name=f"{_short_graph_label(label)} · contribution",
                    line=dict(color=col, width=2, dash="dash")))
            if surv is not None and contrib is not None:
                gap = float((surv - contrib).clip(lower=0).mean())
                contrib_rows.append({
                    "Algorithme": _short_graph_label(label),
                    "Survie finale (%)": round(float(surv.iloc[-1]), 1),
                    "Contribution moy. (%)": round(float(contrib.mean()), 1),
                    "Écart survie−contribution (pts)": round(gap, 1),
                })
        fig_sc.update_layout(
            title="Trait plein = survie (batterie>0) · pointillé = contribution (a vraiment entraîné)",
            xaxis_title="Round", yaxis_title="%", template="plotly_white", height=420,
            font=dict(family="Inter"), yaxis_range=[0, 105])
        st.plotly_chart(fig_sc, width="stretch")
        if contrib_rows:
            st.dataframe(pd.DataFrame(contrib_rows), width="stretch", hide_index=True)
        st.caption(
            "**Survie vs contribution.** Un client peut être *vivant* (batterie > 0) sans "
            "jamais *contribuer* (s'entraîner). Les méthodes de sélection (FedLE) préservent "
            "la batterie des clients faibles en ne les sélectionnant pas : ils survivent mais "
            "leur donnée n'entre jamais dans le modèle — une **survie « vide »**, écart "
            "survie−contribution élevé. FedSTEP-EE garde les clients faibles *en train de "
            "contribuer* (à profondeur réduite) : survie et contribution coïncident (écart ≈ 0). "
            "C'est la comparaison équitable entre survie-par-sélection et survie-par-façonnage-du-calcul."
        )

    # ── Robustness, attacks and FAR diagnostics ──────────────────────────────
    with tab_robust:
        st.markdown(
            "This view separates the **threat configured for the round** from "
            "the **response of the aggregation rule**. Byzantine labels and "
            "Byzantine weight mass are oracle diagnostics used only after training."
        )

        _attack_rows = []
        for label, exp in experiments.items():
            df = exp["df"]
            if df.empty:
                continue
            last = df.iloc[-1]
            _attack_rows.append(
                {
                    "Experiment": _short_graph_label(label),
                    "Attack": last.get("attack_name", "not logged"),
                    "Byzantine clients": last.get(
                        "num_byzantine_oracle", last.get("far_num_byzantine_oracle", "—")
                    ),
                    "Byzantine fraction": (
                        f"{100.0 * float(last['byzantine_fraction_oracle']):.1f}%"
                        if pd.notna(last.get("byzantine_fraction_oracle"))
                        else "—"
                    ),
                    "Robust reference": last.get(
                        "robust_reference", last.get("robust_aggregator", "—")
                    ),
                }
            )
        if _attack_rows:
            st.dataframe(pd.DataFrame(_attack_rows), width="stretch", hide_index=True)

        _has_weights = any(
            _has_round_metric(experiments, key)
            for key in (
                "max_client_weight",
                "byzantine_weight_mass_oracle",
                "effective_num_clients",
                "weight_entropy",
            )
        )
        _has_far_geometry = any(
            _has_round_metric(experiments, key)
            for key in ("far_mean_distance", "far_max_distance", "far_logit_range")
        )
        if _has_weights:
            _rob_col1, _rob_col2 = st.columns(2)
            with _rob_col1:
                st.plotly_chart(
                    _round_metric_figure(
                        experiments,
                        [
                            ("max_client_weight", "Maximum client weight", 1.0),
                            ("byzantine_weight_mass_oracle", "Byzantine weight mass", 1.0),
                        ],
                        title="Influence assigned by the aggregator",
                        yaxis_title="Weight mass",
                    ),
                    width="stretch",
                )
                st.caption(
                    "Maximum weight exposes concentration on one client. Byzantine weight "
                    "mass is the total weight assigned to oracle-labelled attackers. It is "
                    "an evaluation metric, not information available to FAR."
                )
            with _rob_col2:
                st.plotly_chart(
                    _round_metric_figure(
                        experiments,
                        [
                            ("effective_num_clients", "Effective clients", 1.0),
                            ("weight_entropy", "Weight entropy", 1.0),
                        ],
                        title="Weight diversity",
                        yaxis_title="Diagnostic value",
                    ),
                    width="stretch",
                )
                st.caption(
                    "Effective clients is exp(weight entropy). It equals n for uniform "
                    "weights and approaches 1 when one client dominates. Entropy is shown "
                    "alongside it as the underlying concentration statistic."
                )

        if _has_far_geometry:
            _far_col1, _far_col2 = st.columns(2)
            with _far_col1:
                st.plotly_chart(
                    _round_metric_figure(
                        experiments,
                        [
                            ("far_mean_distance", "Mean distance", 1.0),
                            ("far_max_distance", "Maximum distance", 1.0),
                        ],
                        title="Deviation from FAR's robust reference",
                        yaxis_title="L2 distance",
                    ),
                    width="stretch",
                )
                st.caption(
                    "Distances are computed between each client update and g_F, the output "
                    "of the selected Byzantine-robust aggregation rule."
                )
            with _far_col2:
                st.plotly_chart(
                    _round_metric_figure(
                        experiments,
                        [
                            ("far_logit_range", "FAR logit range", 1.0),
                            ("far_weight_ratio", "Max/min weight ratio", 1.0),
                        ],
                        title="Exponential-tilting amplification",
                        yaxis_title="Amplification diagnostic",
                    ),
                    width="stretch",
                )
                st.caption(
                    "The logit range is alpha times the distance range. The max/min "
                    "weight ratio is exp(logit range). Large values indicate concentrated "
                    "tilting and motivate sensitivity-controlled FAR."
                )

        if not _has_weights and not _has_far_geometry:
            st.info(
                "No FAR/robustness diagnostics were recorded in the selected runs. "
                "Run FAR or a Byzantine-robust reference with the updated logger."
            )

    # ── Differential privacy diagnostics ─────────────────────────────────────
    with tab_privacy:
        st.markdown(
            "Privacy metrics are shown only when the algorithm logs a privacy "
            "accountant. For FedFDP, epsilon is cumulative per client and includes "
            "both the model-update channel and the private-loss channel."
        )
        _has_epsilon = any(
            _has_round_metric(experiments, key)
            for key in ("privacy_epsilon_max", "privacy_epsilon_mean")
        )
        if _has_epsilon:
            _privacy_col1, _privacy_col2 = st.columns(2)
            with _privacy_col1:
                st.plotly_chart(
                    _round_metric_figure(
                        experiments,
                        [
                            ("privacy_epsilon_max", "Maximum epsilon", 1.0),
                            ("privacy_epsilon_mean", "Mean epsilon", 1.0),
                        ],
                        title="Cumulative privacy loss",
                        yaxis_title="Epsilon (lower is more private)",
                    ),
                    width="stretch",
                )
                st.caption(
                    "Maximum epsilon is the conservative fleet-level report: it is the "
                    "largest cumulative epsilon among participating clients at the chosen delta."
                )
            with _privacy_col2:
                st.plotly_chart(
                    _round_metric_figure(
                        experiments,
                        [
                            ("fedfdp_clip_rate", "Gradient clip rate", 100.0),
                            ("privacy_sampling_rate_mean", "Sampling rate", 100.0),
                        ],
                        title="Clipping and local sampling",
                        yaxis_title="Rate (%)",
                    ),
                    width="stretch",
                )
                st.caption(
                    "Clip rate is the fraction of sample gradients limited by the norm "
                    "bound. Sampling rate is minibatch size divided by local dataset size."
                )

            _privacy_rows = []
            for label, exp in experiments.items():
                df = exp["df"]
                valid = df[df["privacy_epsilon_max"].notna()] if "privacy_epsilon_max" in df.columns else df.iloc[0:0]
                if valid.empty:
                    continue
                last = valid.iloc[-1]
                _privacy_rows.append(
                    {
                        "Experiment": _short_graph_label(label),
                        "epsilon max": f"{float(last['privacy_epsilon_max']):.4f}",
                        "delta": (
                            f"{float(last['privacy_delta']):.2e}"
                            if pd.notna(last.get("privacy_delta"))
                            else "—"
                        ),
                        "RDP order": (
                            f"{float(last['privacy_best_order_mean']):.1f}"
                            if pd.notna(last.get("privacy_best_order_mean"))
                            else "—"
                        ),
                        "model sigma": last.get("privacy_model_noise_multiplier", "—"),
                        "loss sigma": last.get("privacy_loss_noise_multiplier", "—"),
                        "model steps (mean)": last.get("privacy_model_steps_mean", "—"),
                        "loss clip": last.get("fedfdp_loss_clip_mean", "—"),
                        "accounting assumption": last.get(
                            "privacy_accounting_assumption", "—"
                        ),
                    }
                )
            if _privacy_rows:
                st.dataframe(pd.DataFrame(_privacy_rows), width="stretch", hide_index=True)

            if _has_round_metric(experiments, "fedfdp_global_private_loss"):
                st.plotly_chart(
                    _round_metric_figure(
                        experiments,
                        [("fedfdp_global_private_loss", "Private global loss", 1.0)],
                        title="FedFDP private loss signal",
                        yaxis_title="Noisy clipped loss",
                    ),
                    width="stretch",
                )
        else:
            st.info(
                "No formal privacy-accounting columns were found. A run without a "
                "logged accountant must not be interpreted as having an epsilon guarantee."
            )

    # ── Summary Table ────────────────────────────────────────────────────────
    with tab_table:
        st.markdown("### Synthèse multi-seeds — accuracy et fairness")
        st.caption(
            "Chaque ligne représente une méthode avec ses hyperparamètres structurants. "
            "Les valeurs proviennent du dernier round terminé de chaque run, puis sont "
            "agrégées sur les seeds sélectionnés. Les variantes de FAR sont séparées par α."
        )

        try:
            _selected_run_summaries = [load_run_summary(path) for path in selected_dirs]
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            _selected_run_summaries = []
            st.warning(f"La synthèse multi-seeds ne peut pas être calculée : {exc}")

        if _selected_run_summaries:
            _conditions = sorted({run.condition for run in _selected_run_summaries})
            if len(_conditions) > 1:
                _condition = st.selectbox(
                    "Protocole expérimental comparable",
                    options=_conditions,
                    help=(
                        "Les seeds ne sont moyennés qu'à protocole identique. Cette sélection "
                        "évite de mélanger, par exemple, R1 sans attaque et R2 avec attaque."
                    ),
                    key="multiseed_summary_condition",
                )
            else:
                _condition = _conditions[0]

            _show_seed_std = st.toggle(
                "Afficher moyenne ± écart-type entre seeds",
                value=True,
                help=(
                    "L'écart-type est l'écart-type échantillonnal (ddof=1). "
                    "Avec un seul seed, seule la valeur observée est affichée."
                ),
                key="multiseed_summary_std",
            )
            _condition_runs = [
                run for run in _selected_run_summaries if run.condition == _condition
            ]
            _multiseed_df, _method_seeds = summarize_runs(
                _condition_runs,
                include_std=_show_seed_std,
                decimals=3,
            )
            st.dataframe(
                style_best_values(_multiseed_df),
                width="stretch",
                hide_index=True,
            )

            _seed_details = "; ".join(
                f"{method}: {', '.join(map(str, seeds)) if seeds else 'seed non identifié'}"
                for method, seeds in _method_seeds.items()
            )
            st.caption(
                f"Protocole : `{_condition}`. Seeds incluses — {_seed_details}. "
                "Var est exprimée en points de pourcentage au carré (pp²), soit "
                "10 000 × la variance calculée sur les accuracies dans [0,1]. "
                "Les meilleures valeurs sont en gras : maximum pour les accuracies et "
                "Worst-20, minimum pour Var et Gap Δk."
            )
            st.download_button(
                "Télécharger la synthèse multi-seeds (CSV)",
                _multiseed_df.to_csv(index=False),
                "fedlab_multiseed_fairness_summary.csv",
                "text/csv",
                key="download_multiseed_summary",
            )
        else:
            st.info(
                "Les runs sélectionnés ne contiennent pas encore les métriques client nécessaires."
            )

        st.divider()
        st.markdown("### Détail de chaque run sélectionné")
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

        st.divider()
        st.markdown("#### Round metric explorer")
        st.caption(
            "Plot any scalar stored in metrics.json. This keeps newly added algorithm "
            "diagnostics visible without requiring another dashboard redesign."
        )
        _excluded_explorer = {
            "round",
            "round_num",
            "t",
            "evaluated_client_ids_oracle",
            "client_accuracy_values_oracle",
            "client_loss_values_oracle",
        }
        _numeric_metric_names = sorted(
            {
                column
                for exp in experiments.values()
                for column in exp["df"].columns
                if column not in _excluded_explorer
                and pd.api.types.is_numeric_dtype(exp["df"][column])
            }
        )
        _default_explorer = [
            key
            for key in (
                "test_accuracy",
                "worst20_accuracy",
                "max_client_weight",
                "privacy_epsilon_max",
            )
            if key in _numeric_metric_names
        ][:3]
        _chosen_metrics = st.multiselect(
            "Metrics to plot",
            options=_numeric_metric_names,
            default=_default_explorer,
            key="round_metric_explorer",
        )
        if _chosen_metrics:
            st.plotly_chart(
                _round_metric_figure(
                    experiments,
                    [(key, key, 1.0) for key in _chosen_metrics],
                    title="Selected round diagnostics",
                    yaxis_title="Recorded value",
                    height=460,
                ),
                width="stretch",
            )
        elif _numeric_metric_names:
            st.info("Select one or more round metrics to plot.")
        else:
            st.info("No numeric round metrics were found in the selected runs.")

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

    # ── σ² Evolution ─────────────────────────────────────────────────────────
    with tab_sigma2:
        st.markdown("""
        **Variance inter-client des deltas** — proxy de σ² (variance des gradients), calculée à coût zéro
        depuis les deltas déjà envoyés au serveur à chaque round.

        - **σ²_update** : variance brute des deltas clients `||Δ_k − Δ̄||²`
        - **σ²_grad ≈ σ²_update / (E × lr)²** : proxy normalisé vers l'espace gradient (Δ_k ≈ E·lr·∇f_k)
        - **Warmup** (fond gris) : FedAvg complet — estimation la plus fiable (deltas complets)
        - **PNU** : estimation intra-tier (deltas partiels — même groupe par tier)
        """)

        has_sigma2 = any(
            "sigma2_update" in exp["df"].columns and exp["df"]["sigma2_update"].notna().any()
            for exp in experiments.values()
        )

        if not has_sigma2:
            st.info(
                "σ² non disponible dans ces expériences. "
                "Relancez une expérience avec la version mise à jour de `fedpart_be.py` "
                "pour activer le logging automatique."
            )
        else:
            _s2_smooth_col, _ = st.columns([1, 3])
            with _s2_smooth_col:
                _s2_smooth = st.toggle("EMA smoothing", value=True, key="s2_smooth")

            fig_s2 = go.Figure()
            for label, exp in experiments.items():
                df = exp["df"]
                if "sigma2_gradient_approx" not in df.columns:
                    continue
                col_u = df["sigma2_update"].dropna()
                col_g = df["sigma2_gradient_approx"].dropna()
                if col_g.empty:
                    continue

                rounds_g = df.loc[col_g.index, _round_col(df)]
                vals_g   = col_g.values
                if _s2_smooth:
                    vals_g = ema_smooth(vals_g, alpha=0.2)

                fig_s2.add_trace(go.Scatter(
                    x=rounds_g, y=vals_g,
                    mode="lines", name=f"{label} — σ²_grad",
                    line=dict(width=2),
                ))

                rounds_u = df.loc[col_u.index, _round_col(df)]
                vals_u   = col_u.values
                if _s2_smooth:
                    vals_u = ema_smooth(vals_u, alpha=0.2)

                fig_s2.add_trace(go.Scatter(
                    x=rounds_u, y=vals_u,
                    mode="lines", name=f"{label} — σ²_update",
                    line=dict(width=1.5, dash="dot"),
                    visible="legendonly",
                ))

            # Shade warmup region
            for label, exp in experiments.items():
                df = exp["df"]
                if "is_warmup_round" in df.columns:
                    warmup_rounds = df[df["is_warmup_round"] == True][_round_col(df)]
                    if not warmup_rounds.empty:
                        fig_s2.add_vrect(
                            x0=warmup_rounds.min(), x1=warmup_rounds.max(),
                            fillcolor="gray", opacity=0.1, layer="below",
                            line_width=0, annotation_text="warmup",
                            annotation_position="top left",
                        )
                        break

            fig_s2.update_layout(
                xaxis_title="Round",
                yaxis_title="σ² (proxy gradient variance)",
                template="plotly_white",
                height=420,
                margin=dict(l=50, r=20, t=40, b=40),
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig_s2, use_container_width=True)

            # Summary stats table
            st.markdown("**Statistiques σ²_grad par expérience**")
            rows_s2 = []
            for label, exp in experiments.items():
                df = exp["df"]
                if "sigma2_gradient_approx" not in df.columns:
                    continue
                col = df["sigma2_gradient_approx"].dropna()
                if col.empty:
                    continue
                warmup_mask = df.loc[col.index, "is_warmup_round"] if "is_warmup_round" in df.columns else None
                pnu_vals = col[~warmup_mask].values if warmup_mask is not None else col.values
                wu_vals  = col[warmup_mask].values  if warmup_mask is not None else []
                rows_s2.append({
                    "Expérience": label,
                    "σ²_grad warmup (moy)": f"{float(np.mean(wu_vals)):.3f}" if len(wu_vals) > 0 else "—",
                    "σ²_grad PNU (moy)":    f"{float(np.mean(pnu_vals)):.3f}" if len(pnu_vals) > 0 else "—",
                    "σ²_grad min":          f"{float(col.min()):.3f}",
                    "σ²_grad max":          f"{float(col.max()):.3f}",
                })
            if rows_s2:
                st.dataframe(pd.DataFrame(rows_s2).set_index("Expérience"), use_container_width=True)

            st.caption(
                "σ²_grad ≈ σ²_update / (E·lr)² avec E=local_epochs, lr=learning_rate. "
                "Pendant PNU : variance calculée intra-tier (même groupe). "
                "σ²_update (tirets) = variance brute des deltas."
            )

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
        label = selected_label_by_dir.get(
            str(name), _get_exp_label(pathlib.Path(name), RESULTS_DIR)
        )
        experiments[label] = {"config": config, "df": df, "run_dir": str(name)}

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
        _zoom_cmp = st.toggle(
            "Zoom y-axis to highlight differences",
            value=False,
            key="cmp_zoom",
            help="Starts the y-axis near the minimum value so small differences between algorithms become visible.",
        )

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
                _comm_range_cmp = (
                    [min(bar_comm) * 0.97, max(bar_comm) * 1.03]
                    if _zoom_cmp and len(bar_comm) >= 2 else None
                )
                fig = go.Figure(go.Bar(
                    x=bar_labels, y=bar_comm,
                    marker_color=COLOR_MAP[:len(bar_labels)],
                    marker_line_color="white",
                    marker_line_width=1.5,
                    text=[f"{v:.3f} GB" for v in bar_comm],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Total Communication Cost (GB)",
                    yaxis_title="GB",
                    yaxis=dict(range=_comm_range_cmp) if _comm_range_cmp else {},
                    template="plotly_white", height=380,
                    font=dict(family="Inter"),
                    xaxis_tickangle=-20,
                    uniformtext_minsize=9,
                    uniformtext_mode="hide",
                )
                st.plotly_chart(fig, width="stretch")
                _zoom_note_cmp = " | Zoom actif — axe Y tronqué." if _zoom_cmp else ""
                st.caption(
                    "**Total communication (GB)** = cumulative bytes uploaded by all clients over all rounds. "
                    "With gradient sparsification (β < 1), only β × |θ| parameters are sent instead of the full model. "
                    f"Shorter bar = less bandwidth used = better for low-bandwidth IoT deployments.{_zoom_note_cmp}"
                )

        with col_b2:
            if bar_labels and bar_energy:
                _en_range_cmp = (
                    [min(bar_energy) * 0.97, max(bar_energy) * 1.03]
                    if _zoom_cmp and len(bar_energy) >= 2 else None
                )
                fig2 = go.Figure(go.Bar(
                    x=bar_labels, y=bar_energy,
                    marker_color=COLOR_MAP[:len(bar_labels)],
                    marker_line_color="white",
                    marker_line_width=1.5,
                    text=[f"{v:.0f} J" for v in bar_energy],
                    textposition="outside",
                ))
                fig2.update_layout(
                    title="Total Energy Consumption (J)",
                    yaxis_title="Joules",
                    yaxis=dict(range=_en_range_cmp) if _en_range_cmp else {},
                    template="plotly_white", height=380,
                    font=dict(family="Inter"),
                    xaxis_tickangle=-20,
                    uniformtext_minsize=9,
                    uniformtext_mode="hide",
                )
                st.plotly_chart(fig2, width="stretch")
                st.caption(
                    "**Total energy (J)** = cumulative compute + uplink energy over all rounds and all clients. "
                    "Energy = E_compute (local training) + E_uplink (model transmission). "
                    "For ESP32-S3: E_compute ≈ 0.38 W × training time; E_uplink ≈ model size / link rate × P_tx. "
                    f"Shorter bar = more energy-efficient = clients survive longer.{_zoom_note_cmp}"
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
# Page: SURVIVAL & FAIRNESS  (FedStep evaluation)
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Survival & Fairness":

    st.markdown("""
    <div class="results-header">
      <div class="results-header-title">Survival &amp; Fairness Analysis</div>
      <div class="results-header-sub">
        System lifetime, client survival curves, energy fairness (Jain index),
        and layer staleness — key metrics for FedStep evaluation.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not result_dirs:
        st.info("Run experiments with heterogeneous clients to use this page.")
        st.code(
            "python run_experiment.py --benchmark \\\n"
            "    --algos fedavg,fedprox,fedpart,heterofl,fjord,fedstep \\\n"
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
        label = selected_label_by_dir.get(
            str(name), _get_exp_label(pathlib.Path(name), RESULTS_DIR)
        )
        experiments_sf[label] = {"config": config, "df": df, "run_dir": str(name)}

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
                "fedstep": "#f97316",  # orange
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
                "FedStep should maintain a flatter curve than FedPart by assigning "
                "cheapest layer groups to low-battery clients. "
                "Colors: red=FedAvg, cyan=FedPart, orange=FedStep, green=FedProx, purple=HeteroFL, brown=FjORD."
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
            "FedStep targets a higher Jain index than FedPart."
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
            st.caption("A slower drain = more energy-efficient algorithm. FedStep should drain more slowly than FedPart on expensive groups.")

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
# Page: σ² ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

elif page == "σ² Estimator":

    st.markdown("""
    <div class="results-header">
      <div class="results-header-title">Gradient Variance Estimator — σ²</div>
      <div class="results-header-sub">
        Mesure empirique de la variance inter-client des gradients.
        Valide la formule M* = (2G³K/σ²)<sup>1/6</sup> sur vos données réelles.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Import du module de calcul ────────────────────────────────────────────
    import sys as _sys, pathlib as _pathlib
    _proj = str(_pathlib.Path(__file__).resolve().parents[1])
    if _proj not in _sys.path:
        _sys.path.insert(0, _proj)

    try:
        from diagnostics.gradient_variance import (
            measure_gradient_variance, GradVarianceResult
        )
        from datasets.registry import NUM_CLASSES
        _import_ok = True
    except Exception as _e:
        st.error(f"Import error: {_e}")
        _import_ok = False

    if _import_ok:

        # ── Configuration ─────────────────────────────────────────────────────
        st.subheader("Configuration")
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

        with col_cfg1:
            sv_dataset = st.selectbox(
                "Dataset", ["cifar10", "cifar100", "mnist", "fashionmnist"],
                index=0,
            )
            sv_model = st.selectbox(
                "Modèle",
                ["resnet8", "resnet18", "lenet5", "mlp", "vit_tiny"],
                index=0,
            )
            sv_data_root = st.text_input("data_root", value="./data")

        with col_cfg2:
            sv_clients = st.slider("Clients K", 5, 100, 30, step=5)
            sv_alpha = st.select_slider(
                "Dirichlet α",
                options=[0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0],
                value=0.5,
            )
            sv_seed = st.number_input("Seed", value=42, step=1)

        with col_cfg3:
            sv_batches = st.slider(
                "Batches par client",
                min_value=1, max_value=10, value=3,
                help="Plus de batches = estimation plus précise mais plus lente."
            )
            sv_batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
            sv_device = st.selectbox("Device", ["cpu", "mps", "cuda"], index=0)
            sv_G = st.slider(
                "G (groupes de couches du modèle)",
                min_value=2, max_value=20, value=10,
                help="Pour ResNet-8 : G=10. Utilisé dans la formule M*.",
            )

        st.info(
            f"Estimation : {sv_clients} clients × {sv_batches} batches × {sv_batch_size} samples "
            f"= ~{sv_clients * sv_batches * sv_batch_size} forward/backward passes. "
            f"Temps attendu sur CPU : 15–90s selon le modèle."
        )

        run_btn = st.button("▶ Lancer la mesure", type="primary", use_container_width=True)

        # ── Calcul ────────────────────────────────────────────────────────────
        if run_btn or "sv_result" in st.session_state:

            if run_btn:
                # Barre de progression
                prog_bar = st.progress(0, text="Initialisation...")
                status_txt = st.empty()

                def _cb(k, total):
                    pct = k / total if total > 0 else 0
                    prog_bar.progress(pct, text=f"Client {k}/{total} …")

                with st.spinner("Calcul en cours…"):
                    try:
                        result: GradVarianceResult = measure_gradient_variance(
                            dataset=sv_dataset,
                            model_name=sv_model,
                            num_clients=sv_clients,
                            alpha=sv_alpha,
                            batches_per_client=sv_batches,
                            batch_size=sv_batch_size,
                            seed=int(sv_seed),
                            device=sv_device,
                            data_root=sv_data_root,
                            progress_cb=_cb,
                        )
                        st.session_state["sv_result"] = result
                        prog_bar.empty()
                    except Exception as ex:
                        st.error(f"Erreur : {ex}")
                        st.stop()

            result: GradVarianceResult = st.session_state["sv_result"]

            st.divider()

            # ── Métriques clés ────────────────────────────────────────────────
            m_star_cont = result.optimal_m(sv_G, result.num_clients)
            m_star_int = result.rounded_m(sv_G, result.num_clients)

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("σ² mesuré", f"{result.sigma2:,.1f}",
                       help="Variance inter-client des gradients (norme L2 carrée, moyennée sur K clients).")
            mc2.metric("M* (continu)", f"{m_star_cont:.3f}",
                       help=f"M* = (2×G³×K/σ²)^{{1/6}} avec G={sv_G}, K={result.num_clients}.")
            mc3.metric("M* (arrondi)", str(m_star_int),
                       help="round(M*) — nombre de tiers recommandé.")
            mc4.metric("Temps", f"{result.elapsed_s:.1f}s")

            # Interprétation
            if m_star_int == 1:
                interp_color = "#fef3c7"
                interp_msg = f"σ²={result.sigma2:.0f} est élevé → peu de tiers optimal. M*=1 signifie que FedPart standard (1 groupe/round) est optimal. La variance inter-client domine le staleness."
            elif m_star_int <= 3:
                interp_color = "#d1fae5"
                interp_msg = f"σ²={result.sigma2:.0f} → M*={m_star_int} tiers. Configuration favorable pour FedStep : les {m_star_int} tiers simultanés réduisent le staleness sans exploser la variance."
            else:
                interp_color = "#fee2e2"
                interp_msg = f"σ²={result.sigma2:.0f} est faible (données peu hétérogènes) → M*={m_star_int} tiers. Risque : K/M={result.num_clients}/{m_star_int}={result.num_clients//m_star_int} clients/tier, gradient bruité."

            st.markdown(f"""
            <div style="background:{interp_color}; border-radius:8px; padding:12px 16px;
                        font-size:0.88rem; line-height:1.6; margin:8px 0;">
              <b>Interprétation :</b> {interp_msg}
            </div>
            """, unsafe_allow_html=True)

            st.divider()

            # ── Tabs ─────────────────────────────────────────────────────────
            tab_bar, tab_table, tab_dist, tab_formula = st.tabs([
                "Normes par client",
                "Table M* (G × K)",
                "Distribution σ²",
                "Formule & Validation",
            ])

            with tab_bar:
                st.markdown("**Norme du gradient de chaque client ||∇f_k||** et écart au gradient moyen **||∇f_k − ∇f̄||**")

                fig_bar = go.Figure()
                client_ids = list(range(len(result.grad_norm_per_client)))
                sq_diffs = [v**0.5 for v in result.sigma2_per_client]  # ||∇f_k - ∇f̄||

                fig_bar.add_trace(go.Bar(
                    x=client_ids,
                    y=result.grad_norm_per_client,
                    name="||∇f_k||",
                    marker_color="#3b82f6",
                    opacity=0.7,
                ))
                fig_bar.add_trace(go.Bar(
                    x=client_ids,
                    y=sq_diffs,
                    name="||∇f_k − ∇f̄||",
                    marker_color="#ef4444",
                    opacity=0.8,
                ))
                fig_bar.add_hline(
                    y=result.mean_grad_norm,
                    line_dash="dash", line_color="#1d4ed8",
                    annotation_text=f"||∇f̄|| = {result.mean_grad_norm:.4f}",
                    annotation_position="top right",
                )
                fig_bar.update_layout(
                    barmode="group",
                    xaxis_title="Client ID",
                    yaxis_title="Norme L2",
                    legend=dict(orientation="h", y=1.1),
                    height=400,
                    margin=dict(l=40, r=20, t=40, b=40),
                    template="plotly_white",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with tab_table:
                st.markdown(
                    f"**Table de M\\* = (2G³K/σ²)¹/⁶** avec σ² = {result.sigma2:.1f} mesuré "
                    f"({sv_dataset}, α={result.alpha}, K={result.num_clients})"
                )

                import pandas as pd
                rows_display = []
                for G_val in [5, 10, 18]:
                    row = {"G": G_val}
                    for K_val in [10, 30, 50, 100]:
                        m = result.optimal_m(G_val, K_val)
                        m_r = result.rounded_m(G_val, K_val)
                        row[f"K={K_val}"] = f"{m:.2f} → {m_r}"
                    rows_display.append(row)

                df_table = pd.DataFrame(rows_display).set_index("G")

                # Highlight cell matching current setup
                st.dataframe(
                    df_table,
                    use_container_width=True,
                )
                st.caption(
                    f"Cellule de référence : G={sv_G}, K={result.num_clients} "
                    f"→ M*={m_star_cont:.3f} ≈ {m_star_int}"
                )

            with tab_dist:
                st.markdown("**Distribution de ||∇f_k − ∇f̄||² sur les clients**")

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=result.sigma2_per_client,
                    nbinsx=max(8, len(result.sigma2_per_client) // 3),
                    marker_color="#8b5cf6",
                    opacity=0.8,
                    name="||∇f_k − ∇f̄||²",
                ))
                fig_hist.add_vline(
                    x=result.sigma2,
                    line_dash="dash", line_color="#dc2626",
                    annotation_text=f"σ² = {result.sigma2:.1f}",
                    annotation_position="top right",
                )
                fig_hist.update_layout(
                    xaxis_title="||∇f_k − ∇f̄||²",
                    yaxis_title="Nombre de clients",
                    height=380,
                    margin=dict(l=40, r=20, t=30, b=40),
                    template="plotly_white",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                # Stats
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("min ||∇f_k − ∇f̄||²", f"{min(result.sigma2_per_client):.1f}")
                sc2.metric("médiane", f"{float(np.median(result.sigma2_per_client)):.1f}")
                sc3.metric("max", f"{max(result.sigma2_per_client):.1f}")

            with tab_formula:
                st.markdown(f"""
**Formule du tier optimal :**

$$M^* = \\left(\\frac{{2\\,G^3\\,K}}{{\\sigma^2}}\\right)^{{1/6}}$$

**Avec vos paramètres mesurés :**

| Symbole | Valeur | Source |
|---|---|---|
| σ² | **{result.sigma2:.1f}** | Mesuré empiriquement ({result.batches_per_client} batches/client) |
| G | {sv_G} | Choisi (ResNet-8 : 10 groupes) |
| K | {result.num_clients} | Nombre de clients |
| **M\\*** | **{m_star_cont:.3f} → {m_star_int}** | Résultat |

**Calcul :**
$$M^* = \\left(\\frac{{2 \\times {sv_G}^3 \\times {result.num_clients}}}{{{result.sigma2:.0f}}}\\right)^{{1/6}}
= \\left({2 * sv_G**3 * result.num_clients / result.sigma2:.1f}\\right)^{{1/6}}
\\approx {m_star_cont:.3f} \\approx {m_star_int}$$

**Robustesse à l'estimation de σ² (puissance 1/6) :**

| σ² | M* (continu) | M* (arrondi) |
|---|---|---|
| {result.sigma2 * 0.25:.0f} (×0.25) | {result.optimal_m(sv_G)*(result.sigma2/(result.sigma2*0.25))**(1/6):.2f} | {result.rounded_m(sv_G, result.num_clients) if True else '?'} |
| **{result.sigma2:.0f} (mesuré)** | **{m_star_cont:.3f}** | **{m_star_int}** |
| {result.sigma2 * 4:.0f} (×4) | {(2*sv_G**3*result.num_clients/(result.sigma2*4))**(1/6):.2f} | {max(1,round((2*sv_G**3*result.num_clients/(result.sigma2*4))**(1/6)))} |

*Multiplier σ² par 4 ne change M* que d'un facteur 4^{{1/6}} ≈ 1.26 → prédiction robuste.*
                """)

# ─────────────────────────────────────────────────────────────────────────────
# Footer (all pages)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-top:40px; padding-top:16px; border-top:1px solid #e2e8f0;
            text-align:center; font-size:0.75rem; color:#94a3b8;">
  FedLab ZMQ &nbsp;|&nbsp;
  Fairness, Robustness, Privacy &amp; Energy-Efficient FL &nbsp;|&nbsp;
  Mohammed VI Polytechnic University (UM6P), Benguerir, Morocco
</div>
""", unsafe_allow_html=True)
