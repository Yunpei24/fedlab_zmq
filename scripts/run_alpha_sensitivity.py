#!/usr/bin/env python3
"""
scripts/run_alpha_sensitivity.py
================================
Sensitivity sweep over the compute energy_scale_factor (alpha).

The hand-calibrated alpha = 12.6 (chosen so the fleet dies at T* ~ 80 rounds)
is circular: it was tuned against the analytic FLOPs it is meant to convert.
Instead of trusting one value, this driver sweeps alpha over a hardware-
justified range and reports, for each paper conclusion, whether it holds
across the WHOLE grid (and on what sub-range otherwise).

Why a full re-run per alpha (no post-hoc rescaling)
---------------------------------------------------
alpha multiplies per-round COMPUTE energy. It cancels in RELATIVE (cross-algo)
energy ratios, but NOT in ABSOLUTE survival: once a battery reaches zero, alpha
decides who dies and when; under battery-stratified group assignment that
changes the assignment, the deaths, and hence the whole training trajectory.
So each alpha needs a fresh full simulation. (If battery death is inactive,
alpha is a pure post-hoc multiplier and the relative metrics are analytically
invariant — flat lines; we confirm this empirically and ALERT if a relative
metric is NOT flat, which would indicate a bug.)

One-command usage
-----------------
    python scripts/run_alpha_sensitivity.py --grid smoke   # ~minutes, sanity
    python scripts/run_alpha_sensitivity.py --grid full    # the real sweep
    python scripts/run_alpha_sensitivity.py --grid full --jobs 4   # parallel
    python scripts/run_alpha_sensitivity.py --grid full --dry-run  # plan only

Outputs (under results/alpha_sensitivity/<grid>/)
-------------------------------------------------
    aggregated_wide.csv    one row per (algo, alpha), all metrics
    aggregated_long.csv    tidy: one row per (algo, alpha, metric, value)
    relative.csv           per-alpha cross-algo ratios (expected ~flat)
    figures/               PNGs: relative block + absolute block, vline @12.6
    robustness_summary.md  per-conclusion verdict + paper-ready sentence
    <algo>__a<alpha>/      per-run dir (resolved_config.yaml, metrics.json,
                           manifest.json, survival.csv)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = ROOT / "configs/alpha_sensitivity_base.yaml"
DEFAULT_OUT = ROOT / "results/alpha_sensitivity"

ALGOS = ["fedavg", "fedpart", "fedpart_be"]

# alpha grid. Justification (see configs/alpha_sensitivity_base.yaml):
# an FPU/SIMD-less MCU (ESP32-S3 class) sustains ~5-30% of peak FLOP/s on real
# conv/BN, so effective energy/analytic-FLOP is inflated by ~1/utilization,
# i.e. alpha ~ 3-20. The grid brackets this with headroom and includes 12.6
# (the hand-calibrated reference) as a marker.
ALPHA_GRID_FULL = [2.0, 3.0, 5.0, 8.0, 12.6, 20.0, 30.0]
ALPHA_GRID_SMOKE = [3.0, 12.6, 30.0]
ALPHA_REFERENCE = 12.6

# Accuracy thresholds for the cost-to-accuracy metric.
ACC_THRESHOLDS = [0.60, 0.70, 0.75]

# Rough a-priori per-run wall-time model (seconds), refined live after run 1.
# per_run ~= FIXED_OVERHEAD_SEC + rounds*clients*epochs*sec_cre. measured-mode
# caches the FLOP measurement, so the training term is ~linear in
# rounds x clients x epochs. Both terms are DEVICE-DEPENDENT (CPU is far slower
# than mps/cuda); the default sec_cre is a CPU-class ballpark calibrated from
# the smoke grid. Tune via --est-sec-per-run; the estimate is refined live.
FIXED_OVERHEAD_SEC = 60.0
DEFAULT_SEC_PER_CLIENT_ROUND_EPOCH = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# Config resolution — write a fully-resolved YAML per run (reproducible, and
# sidesteps the fact that --clients/--rounds do not override a --config).
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_config(
    base: dict,
    algo: str,
    alpha: float,
    grid: str,
    seed: int,
    device: str,
    run_dir: Path,
) -> dict:
    cfg = copy.deepcopy(base)
    cfg["seed"] = seed
    cfg["device"] = device
    cfg["cost_model"] = "measured"  # frozen for the whole sweep
    cfg["output_dir"] = str(run_dir)

    cfg.setdefault("training", {})
    cfg["training"]["algorithm"] = algo
    ac = cfg["training"].setdefault("algo_config", {})
    ac["energy_scale_factor"] = float(alpha)
    # mirror cost_model into algo_config (flop_cost reads it from there)
    ac["cost_model"] = "measured"

    if grid == "smoke":
        # Fast pipeline check: 3 clients, 20 rounds, 3 alphas, light epochs.
        cfg["training"]["num_rounds"] = 20
        ac["local_epochs"] = 1
        cfg.setdefault("clients", {})
        cfg["clients"]["num_clients"] = 3
        cfg["clients"]["fleet"] = [{"type": "esp32_s3", "count": 3}]
        cfg["clients"]["min_clients"] = 1
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Run one (algo, alpha)
# ─────────────────────────────────────────────────────────────────────────────


def _run_one(
    base: dict,
    algo: str,
    alpha: float,
    grid: str,
    seed: int,
    device: str,
    base_out: Path,
    dry_run: bool,
) -> dict:
    run_dir = base_out / f"{algo}__a{alpha:g}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = _resolve_config(base, algo, alpha, grid, seed, device, run_dir)
    cfg_path = run_dir / "resolved_config.yaml"
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    cmd = [sys.executable, str(ROOT / "run_experiment.py"), "--config", str(cfg_path)]
    if dry_run:
        return {
            "algo": algo,
            "alpha": alpha,
            "cmd": " ".join(cmd),
            "run_dir": str(run_dir),
        }

    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    dt = time.time() - t0
    ok = res.returncode == 0
    if not ok:
        tail = (res.stderr or res.stdout or "").strip().splitlines()[-3:]
        print(
            f"  [warn] {algo} a={alpha:g} failed (rc={res.returncode}): "
            f"{' / '.join(tail)}"
        )
    return {
        "algo": algo,
        "alpha": alpha,
        "run_dir": str(run_dir),
        "seconds": dt,
        "ok": ok,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metric extraction
# ─────────────────────────────────────────────────────────────────────────────


def _find_metrics_json(run_dir: Path):
    c = list(Path(run_dir).glob("**/metrics.json"))
    return c[0] if c else None


def _cost_to_accuracy(rounds: list, thresh: float) -> dict:
    """rounds / cumulative energy / cumulative bits to first reach `thresh`.
    Returns inf for whichever is never reached."""
    for r in rounds:
        acc = r.get("test_accuracy")
        if acc is not None and acc >= thresh:
            return {
                "rounds": r.get("round_num"),
                "energy_j": r.get("cumulative_energy_j", math.inf),
                "bits": (r.get("cumulative_bytes", math.inf) or math.inf) * 8,
            }
    return {"rounds": math.inf, "energy_j": math.inf, "bits": math.inf}


def _extract_metrics(algo: str, alpha: float, run_dir: Path) -> dict:
    row = {"algo": algo, "alpha": alpha}
    mp = _find_metrics_json(run_dir)
    if mp is None or not mp.exists():
        row["error"] = "metrics.json not found"
        return row
    with open(mp) as fh:
        data = json.load(fh)
    summary = data.get("summary", {}) or {}
    survival = data.get("survival", {}) or {}
    rounds = data.get("rounds", []) or []
    num_clients = summary.get("num_clients")
    num_rounds = summary.get("num_rounds") or (len(rounds) or None)

    # Accuracy
    row["best_acc"] = summary.get("best_accuracy")
    row["final_acc"] = summary.get("final_accuracy")
    # Survival
    row["alive_final"] = rounds[-1].get("num_alive_clients") if rounds else None
    row["num_clients"] = num_clients
    row["median_lifetime"] = survival.get("median_lifetime")
    row["round_5th_death"] = survival.get("round_of_5th_death")
    row["round_10th_death"] = survival.get("round_of_10th_death")
    row["round_15th_death"] = survival.get("round_of_15th_death")
    row["survival_auc"] = survival.get("survival_auc")
    row["participation_frac"] = survival.get("participation_frac")
    # Energy
    e_tot = summary.get("total_energy_j")
    row["total_energy_j"] = e_tot
    row["avg_energy_per_round_j"] = (
        e_tot / num_rounds if (e_tot is not None and num_rounds) else None
    )
    gb = summary.get("total_bytes_gb")
    row["uplink_mb"] = gb * 1024.0 if gb is not None else None
    # Cost-to-accuracy
    for x in ACC_THRESHOLDS:
        c = _cost_to_accuracy(rounds, x)
        row[f"rounds_to_{x:.2f}"] = c["rounds"]
        row[f"energy_to_{x:.2f}_j"] = c["energy_j"]
        row[f"bits_to_{x:.2f}"] = c["bits"]
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation, figures, robustness
# ─────────────────────────────────────────────────────────────────────────────

WIDE_FIELDS = [
    "grid",
    "algo",
    "alpha",
    "best_acc",
    "final_acc",
    "alive_final",
    "num_clients",
    "median_lifetime",
    "round_5th_death",
    "round_10th_death",
    "round_15th_death",
    "survival_auc",
    "participation_frac",
    "total_energy_j",
    "avg_energy_per_round_j",
    "uplink_mb",
    "rounds_to_0.60",
    "energy_to_0.60_j",
    "bits_to_0.60",
    "rounds_to_0.70",
    "energy_to_0.70_j",
    "bits_to_0.70",
    "rounds_to_0.75",
    "energy_to_0.75_j",
    "bits_to_0.75",
    "run_seconds",
    "error",
]


def _write_csvs(rows: list[dict], base_out: Path):
    import pandas as pd

    df = pd.DataFrame(rows)
    for f in WIDE_FIELDS:
        if f not in df.columns:
            df[f] = None
    df = df[WIDE_FIELDS].sort_values(["algo", "alpha"])
    wide = base_out / "aggregated_wide.csv"
    df.to_csv(wide, index=False)

    metric_cols = [
        c for c in WIDE_FIELDS if c not in ("grid", "algo", "alpha", "error")
    ]
    long = df.melt(
        id_vars=["grid", "algo", "alpha"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    long_path = base_out / "aggregated_long.csv"
    long.to_csv(long_path, index=False)
    return df, wide, long_path


def _pivot(df, metric):
    """{algo: {alpha: value}} ignoring NaN/inf for ordering where needed."""
    out = {}
    for algo in df["algo"].unique():
        sub = df[df["algo"] == algo].sort_values("alpha")
        out[algo] = list(zip(sub["alpha"].tolist(), sub[metric].tolist()))
    return out


def _relative_table(df):
    """Per-alpha cross-algo ratios (BE/FedPart). Expected ~flat."""
    import pandas as pd

    rows = []
    for alpha in sorted(df["alpha"].unique()):
        sl = df[df["alpha"] == alpha]

        def get(algo, col):
            v = sl[sl["algo"] == algo][col]
            return v.iloc[0] if len(v) else None

        be_e, fp_e = get("fedpart_be", "total_energy_j"), get(
            "fedpart", "total_energy_j"
        )
        be_u, fp_u = get("fedpart_be", "uplink_mb"), get("fedpart", "uplink_mb")
        # Use `is not None` (not truthiness): a legitimate 0.0 energy/uplink
        # (e.g. a fleet that dies immediately) must yield ratio 0.0, not None.
        rows.append(
            {
                "alpha": float(alpha),
                "energy_ratio_be_over_fedpart": (
                    (be_e / fp_e) if (be_e is not None and fp_e) else None
                ),
                "uplink_ratio_be_over_fedpart": (
                    (be_u / fp_u) if (be_u is not None and fp_u) else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _cv(values) -> float | None:
    vs = [
        v
        for v in values
        if v is not None
        and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
    ]
    if len(vs) < 2:
        return None
    mean = sum(vs) / len(vs)
    if mean == 0:
        return None
    var = sum((v - mean) ** 2 for v in vs) / len(vs)
    return math.sqrt(var) / abs(mean)


def _make_figures(df, rel_df, fig_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"fedavg": "#888", "fedpart": "#1f77b4", "fedpart_be": "#d62728"}

    def _abs_plot(metric, ylabel, fname, title):
        fig, ax = plt.subplots(figsize=(6, 4))
        piv = _pivot(df, metric)
        for algo, pts in piv.items():
            xs = [a for a, _ in pts]
            ys = [(v if v is not None else float("nan")) for _, v in pts]
            ax.plot(xs, ys, "o-", label=algo, color=colors.get(algo))
        ax.axvline(
            ALPHA_REFERENCE,
            ls="--",
            color="k",
            alpha=0.5,
            label=f"alpha={ALPHA_REFERENCE} (ref)",
        )
        ax.set_xscale("log")
        ax.set_xlabel("energy_scale_factor alpha (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=130)
        plt.close(fig)

    def _rel_plot(col, ylabel, fname, title):
        fig, ax = plt.subplots(figsize=(6, 4))
        sub = rel_df.dropna(subset=[col])
        ax.plot(sub["alpha"], sub[col], "s-", color="#2ca02c")
        ax.axvline(ALPHA_REFERENCE, ls="--", color="k", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("energy_scale_factor alpha (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title + "  (EXPECTED FLAT)")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=130)
        plt.close(fig)

    # Block (b) ABSOLUTE — move with alpha, ORDER must be preserved.
    _abs_plot(
        "alive_final",
        "clients alive @ final round",
        "absolute_alive_at_end.png",
        "Survivors vs alpha",
    )
    _abs_plot(
        "median_lifetime",
        "median client lifetime (rounds)",
        "absolute_median_lifetime.png",
        "Median lifetime vs alpha",
    )
    _abs_plot(
        "total_energy_j",
        "total fleet energy (J)",
        "absolute_total_energy.png",
        "Total energy vs alpha",
    )
    _abs_plot(
        "best_acc",
        "best accuracy",
        "absolute_best_accuracy.png",
        "Best accuracy vs alpha",
    )

    # Block (a) RELATIVE — should be flat (alpha cancels in ratios).
    _rel_plot(
        "energy_ratio_be_over_fedpart",
        "energy ratio BE / FedPart",
        "relative_energy_ratio.png",
        "Energy ratio BE/FedPart",
    )
    _rel_plot(
        "uplink_ratio_be_over_fedpart",
        "uplink ratio BE / FedPart",
        "relative_uplink_ratio.png",
        "Uplink ratio BE/FedPart",
    )

    # Survival gap BE - FedPart vs alpha.
    fig, ax = plt.subplots(figsize=(6, 4))
    be = dict(_pivot(df, "alive_final")["fedpart_be"])
    fp = dict(_pivot(df, "alive_final")["fedpart"])
    xs = sorted(set(be) & set(fp))
    gap = [
        (be[a] - fp[a]) if (be[a] is not None and fp[a] is not None) else float("nan")
        for a in xs
    ]
    ax.plot(xs, gap, "o-", color="#9467bd")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(ALPHA_REFERENCE, ls="--", color="k", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("energy_scale_factor alpha (log scale)")
    ax.set_ylabel("alive@end gap  (FedPartBE - FedPart)")
    ax.set_title("Survival gap FedPartBE - FedPart vs alpha")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "gap_survival_be_minus_fedpart.png", dpi=130)
    plt.close(fig)


def _robustness(df, rel_df, base_out: Path, grid: str, seed: int):
    """Per-conclusion verdict + paper sentence. Returns the markdown string."""
    # Cast to plain Python float so the summary text doesn't show np.float64(...).
    alphas = sorted(float(a) for a in df["alpha"].unique())

    def at(alpha, algo, col):
        s = df[(df["alpha"] == alpha) & (df["algo"] == algo)][col]
        return s.iloc[0] if len(s) else None

    # Is battery death active anywhere? (alive_final < num_clients)
    death_active = False
    for _, r in df.iterrows():
        if (
            r.get("alive_final") is not None
            and r.get("num_clients")
            and r["alive_final"] < r["num_clients"]
        ):
            death_active = True
            break

    def _ordering_holds(metric, order, ge_first_pair=False):
        """order = [a,b,c] expects metric[a] >= metric[b] >= metric[c] (>=);
        ge_first_pair: first relation is >= (ties ok), rest strict-ish (>=)."""
        per_alpha = {}
        for alpha in alphas:
            vals = [at(alpha, a, metric) for a in order]
            if any(v is None for v in vals):
                per_alpha[alpha] = None
                continue
            ok = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
            per_alpha[alpha] = bool(ok)
        return per_alpha

    surv = _ordering_holds("survival_auc", ["fedpart_be", "fedpart", "fedavg"])
    energy_order = {}
    for alpha in alphas:
        vals = [
            at(alpha, a, "total_energy_j") for a in ["fedpart_be", "fedpart", "fedavg"]
        ]
        # bool(...) — a chained numpy comparison returns numpy.bool_, which
        # fails an `is True` identity check downstream.
        energy_order[alpha] = (
            None
            if any(v is None for v in vals)
            else bool(vals[0] <= vals[1] <= vals[2])
        )

    # Survival gap BE - FedPart (using survival_auc) sign & magnitude.
    gaps = {}
    for alpha in alphas:
        be, fp = at(alpha, "fedpart_be", "survival_auc"), at(
            alpha, "fedpart", "survival_auc"
        )
        gaps[alpha] = (be - fp) if (be is not None and fp is not None) else None

    def _holds_range(per_alpha):
        # Truthiness (not `is True`) so numpy.bool_ values classify correctly.
        good = [float(a) for a, v in per_alpha.items() if v is not None and bool(v)]
        bad = [float(a) for a, v in per_alpha.items() if v is not None and not bool(v)]
        na = [float(a) for a, v in per_alpha.items() if v is None]
        if not bad and good:
            return True, good, bad, na
        return False, good, bad, na

    surv_all, surv_good, surv_bad, surv_na = _holds_range(surv)
    en_all, en_good, en_bad, en_na = _holds_range(energy_order)

    # Relative-metric flatness check (the bug detector).
    e_cv = _cv(rel_df["energy_ratio_be_over_fedpart"].tolist())
    u_cv = _cv(rel_df["uplink_ratio_be_over_fedpart"].tolist())
    FLAT_TOL = 0.02  # 2%

    def _flat_verdict(name, cv):
        if cv is None:
            return f"- **{name}**: insufficient data to assess flatness."
        pct = cv * 100
        if cv <= FLAT_TOL:
            return (
                f"- **{name}**: flat across alpha (CV={pct:.2f}% <= "
                f"{FLAT_TOL*100:.0f}%). alpha cancels in the ratio, as expected."
            )
        if not death_active:
            return (
                f"- 🚨 **ALERT — {name} NOT flat** (CV={pct:.2f}%) while "
                f"battery death is INACTIVE: alpha should cancel exactly in "
                f"the ratio. This indicates a BUG in the energy accounting."
            )
        return (
            f"- **{name}**: varies by CV={pct:.2f}% across alpha. Battery "
            f"death is ACTIVE, so this reflects alpha-induced trajectory "
            f"divergence (different deaths -> different participation), not "
            f"a pure-multiplier violation."
        )

    g0 = [a for a, v in gaps.items() if v is not None]
    gap_signs = {a: gaps[a] for a in g0}
    all_nonneg = all(v >= 0 for v in gap_signs.values()) if gap_signs else False

    lines = []
    lines.append(f"# alpha sensitivity — robustness summary ({grid} grid)\n")
    lines.append(f"- seed: **{seed}** (fixed across the whole grid)")
    lines.append("- cost_model: **measured** (frozen)")
    lines.append(f"- alpha grid: {alphas}  (reference marker: {ALPHA_REFERENCE})")
    lines.append(
        f"- battery death active in this scenario: "
        f"**{'YES' if death_active else 'NO'}**"
    )
    if not death_active:
        lines.append(
            "  - With death inactive, alpha is a pure post-hoc "
            "multiplier: relative metrics are analytically invariant "
            "(flat lines), confirmed empirically below."
        )
    lines.append("")
    lines.append("## (a) Relative metrics — expected FLAT (alpha cancels)\n")
    lines.append(_flat_verdict("energy ratio BE/FedPart", e_cv))
    lines.append(_flat_verdict("uplink ratio BE/FedPart", u_cv))
    lines.append("")
    lines.append("## (b) Absolute conclusions — order must be preserved\n")

    def _range_str(good, bad, na):
        s = f"holds at alpha={sorted(good)}" if good else "holds nowhere"
        if bad:
            s += f"; FAILS at alpha={sorted(bad)}"
        if na:
            s += f"; n/a at alpha={sorted(na)}"
        return s

    lines.append(
        f"**C1 — survival order FedPartBE >= FedPart > FedAvg** "
        f"(by survival AUC): "
        f"{'HOLDS across the WHOLE grid' if surv_all else 'PARTIAL'} — "
        f"{_range_str(surv_good, surv_bad, surv_na)}."
    )
    lines.append("")
    lines.append(
        f"**C2 — total energy FedPartBE <= FedPart <= FedAvg**: "
        f"{'HOLDS across the WHOLE grid' if en_all else 'PARTIAL'} — "
        f"{_range_str(en_good, en_bad, en_na)}."
    )
    lines.append("")
    gap_desc = ", ".join(f"alpha={a:g}: {gaps[a]:+.1f}" for a in g0) or "n/a"
    lines.append(
        f"**C3 — survival gap (FedPartBE - FedPart, survival AUC)**: "
        f"{'non-negative for all tested alpha' if all_nonneg else 'changes sign'} "
        f"(magnitude vs alpha: {gap_desc})."
    )
    lines.append("")

    # Paper-ready verdict sentence, populated empirically.
    lines.append("## Verdict (paper-ready)\n")
    rel_ok = (e_cv is not None and e_cv <= FLAT_TOL) and (
        u_cv is None or u_cv <= FLAT_TOL
    )
    amin, amax = (min(alphas), max(alphas)) if alphas else (None, None)

    # Relative clause — honest about WHY the ratios do/don't move.
    if rel_ok:
        rel_clause = (
            f"Relative conclusions are invariant to alpha (cross-algo ratios "
            f"flat within {FLAT_TOL*100:.0f}%, i.e. alpha cancels as expected)"
        )
    elif death_active:
        rel_clause = (
            "Cross-algo energy ratios shift with alpha because battery death is "
            "active (alpha changes who dies and when, so it does not cancel "
            "post-hoc); the robust claims are therefore the absolute orderings "
            "below, re-derived per alpha"
        )
    else:
        rel_clause = (
            "Cross-algo ratios are NOT flat although battery death is inactive — "
            "see the bug ALERT above (alpha should cancel exactly here)"
        )

    def _span(good, all_flag):
        if all_flag:
            return f"all alpha in [{amin:g}, {amax:g}]"
        return f"alpha={sorted(good)}" if good else "no tested alpha"

    if surv_all and en_all:
        abs_clause = (
            f"the survival order (FedPartBE >= FedPart > FedAvg) and the energy "
            f"order (FedPartBE <= FedPart <= FedAvg) both hold for every alpha in "
            f"[{amin:g}, {amax:g}] tested, so the findings do not depend on the "
            f"hand-calibrated alpha=12.6"
        )
    else:
        abs_clause = (
            f"over [{amin:g}, {amax:g}] the survival order holds for "
            f"{_span(surv_good, surv_all)} and the energy order holds for "
            f"{_span(en_good, en_all)} (see C1/C2)"
        )

    verdict = f"{rel_clause}; {abs_clause}."
    lines.append("> " + verdict)
    lines.append("")

    md = "\n".join(lines)
    (base_out / "robustness_summary.md").write_text(md)
    return md


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def _estimate_seconds(n_runs, base, grid, jobs, sec_cre):
    if grid == "smoke":
        rounds, clients, epochs = 20, 3, 1
    else:
        rounds = base.get("training", {}).get("num_rounds", 200)
        clients = base.get("clients", {}).get("num_clients", 30)
        epochs = base.get("training", {}).get("algo_config", {}).get("local_epochs", 8)
    # per_run = fixed overhead (torch import + dataset load + FlopCounterMode
    # calibration) + linear training term. Both are device-dependent and only a
    # ballpark — refined live after the first real run.
    per_run = FIXED_OVERHEAD_SEC + rounds * clients * epochs * sec_cre
    return n_runs * per_run / max(1, jobs), per_run


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--grid", choices=["full", "smoke"], default="full")
    p.add_argument("--config", default=str(DEFAULT_BASE_CONFIG))
    p.add_argument(
        "--output",
        default=None,
        help="Base output dir (default results/alpha_sensitivity/<grid>)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--jobs", type=int, default=1, help="Parallel runs (same seed everywhere)."
    )
    p.add_argument("--algos", nargs="+", default=ALGOS)
    p.add_argument(
        "--alpha-grid", nargs="+", type=float, default=None, help="Override alpha grid."
    )
    p.add_argument(
        "--est-sec-per-run",
        type=float,
        default=DEFAULT_SEC_PER_CLIENT_ROUND_EPOCH,
        help="Seconds per (client x round x epoch) for the ETA estimate.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip simulation; re-aggregate existing run dirs into the "
        "CSVs / figures / robustness summary.",
    )
    args = p.parse_args()

    with open(args.config) as fh:
        base = yaml.safe_load(fh)

    alphas = args.alpha_grid or (
        ALPHA_GRID_SMOKE if args.grid == "smoke" else ALPHA_GRID_FULL
    )
    base_out = Path(args.output) if args.output else DEFAULT_OUT / args.grid
    base_out.mkdir(parents=True, exist_ok=True)

    jobs = [(a, al) for a in args.algos for al in alphas]
    n = len(jobs)

    # Duration estimate BEFORE launching.
    eta_s, per_run = _estimate_seconds(
        n, base, args.grid, args.jobs, args.est_sec_per_run
    )
    print("=" * 70)
    print(f"alpha sensitivity sweep — grid={args.grid}")
    print(f"  algos     : {args.algos}")
    print(f"  alpha grid: {alphas}   (reference {ALPHA_REFERENCE})")
    print(f"  runs      : {n}  ({len(args.algos)} algos x {len(alphas)} alphas)")
    print(f"  parallel  : {args.jobs} job(s)")
    print(
        f"  est. time : ~{eta_s/60:.0f} min  (rough: ~{per_run/60:.1f} min/run "
        f"x {n} / {args.jobs}; refined live after run 1)"
    )
    print(f"  output    : {base_out}")
    print("=" * 70)

    if args.dry_run:
        for a, al in jobs:
            r = _run_one(
                base, a, al, args.grid, args.seed, args.device, base_out, dry_run=True
            )
            print(f"[dry-run] {a} a={al:g}: {r['cmd']}")
        return

    # Launch (optionally parallel). Same seed everywhere.
    timings = []
    results = []

    def _do(job):
        a, al = job
        return _run_one(
            base, a, al, args.grid, args.seed, args.device, base_out, dry_run=False
        )

    t_start = time.time()
    if args.aggregate_only:
        print("\n--aggregate-only: skipping simulation, re-reading existing runs.")
    elif args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_do, j): j for j in jobs}
            done = 0
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                done += 1
                if r.get("seconds"):
                    timings.append(r["seconds"])
                avg = sum(timings) / len(timings) if timings else per_run
                remaining = (n - done) / args.jobs * avg
                print(
                    f"  [{done}/{n}] {r['algo']} a={r['alpha']:g} "
                    f"({r.get('seconds', 0):.0f}s)  ETA ~{remaining/60:.0f} min"
                )
    else:
        for i, job in enumerate(jobs, 1):
            a, al = job
            print(f"\n[{i}/{n}] >>> {a} / alpha={al:g}")
            r = _do(job)
            results.append(r)
            if r.get("seconds"):
                timings.append(r["seconds"])
            avg = sum(timings) / len(timings) if timings else per_run
            remaining = (n - i) * avg
            print(
                f"      done in {r.get('seconds', 0):.0f}s  "
                f"ETA ~{remaining/60:.0f} min"
            )
    if not args.aggregate_only:
        print(f"\nAll {n} runs finished in {(time.time()-t_start)/60:.1f} min.")

    # Aggregate metrics.
    rows = []
    by_key = {(r["algo"], r["alpha"]): r for r in results}
    for a, al in jobs:
        m = _extract_metrics(a, al, base_out / f"{a}__a{al:g}")
        m["grid"] = args.grid
        rr = by_key.get((a, al), {})
        m["run_seconds"] = round(rr.get("seconds", 0), 1) if rr.get("seconds") else None
        rows.append(m)

    df, wide_path, long_path = _write_csvs(rows, base_out)
    rel_df = _relative_table(df)
    rel_df.to_csv(base_out / "relative.csv", index=False)

    if not args.no_figures:
        try:
            _make_figures(df, rel_df, base_out / "figures")
            print(f"Figures: {base_out / 'figures'}")
        except Exception as exc:
            print(f"[warn] figures skipped: {exc}")

    md = _robustness(df, rel_df, base_out, args.grid, args.seed)

    print(f"\nCSV (wide): {wide_path}")
    print(f"CSV (long): {long_path}")
    print(f"CSV (relative ratios): {base_out / 'relative.csv'}")
    print(f"Robustness summary: {base_out / 'robustness_summary.md'}\n")
    print(md)


if __name__ == "__main__":
    main()
