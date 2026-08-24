#!/usr/bin/env python3
"""
scripts/make_paper_figures.py
=============================
Generate the four IoT figures for the FedSTEP IoT-J paper from experiment
metrics. Output PDFs land in papers_fedpartbe/figures/ with the exact filenames
the LaTeX placeholders expect:

  fig_acc_vs_energy.pdf          (fig:acc_energy)  E1: acc vs cumulative energy
  fig_acc_vs_bytes.pdf           (fig:acc_bytes)   E1: acc vs cumulative uplink
  fig_survival_jain.pdf          (fig:survival)    E1: alive + Jain vs round
  fig_e2_breakdown_survival.pdf  (fig:e2)          E2: energy split + survival

Sources:
  E1  -> results/E1_a1/<algo>/*/metrics.json   (per-round 'rounds' list)
  E2  -> results/E2/aggregated_wide.csv        (one row per profile,algo)

Standalone & idempotent: re-run any time. Missing inputs are skipped with a
warning (so E1 figures still build if E2 is not done yet).
"""
from __future__ import annotations
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E1_DIR = os.path.join(ROOT, "results", "E1_a1")
E2_CSV = os.path.join(ROOT, "results", "E2", "aggregated_wide.csv")
OUT = os.path.join(ROOT, "papers_fedpartbe", "figures")
os.makedirs(OUT, exist_ok=True)

# Consistent algo identity (order = legend/z-order). fedpart_be == FedSTEP.
# fed_resonance is intentionally excluded from this paper (parallel work).
# NOTE (2026-07): the algorithm was renamed fedpart_be -> fedstep (registry
# alias kept). Internal keys stay "fedpart_be" so HISTORICAL result dirs load;
# DIR_ALIASES lets the loaders pick up NEW runs written under "fedstep".
ALGOS = ["fedavg", "fedpart", "fedpart_be"]
DIR_ALIASES = {"fedpart_be": ["fedpart_be", "fedstep"]}
def _norm_algo(name: str) -> str:
    return "fedpart_be" if name == "fedstep" else name
LABEL = {"fedavg": "FedAvg", "fedpart": "FedPart",
         "fed_resonance": "FedResonance", "fedpart_be": "FedSTEP"}
COLOR = {"fedavg": "#d62728", "fedpart": "#1f77b4",
         "fed_resonance": "#2ca02c", "fedpart_be": "#000000"}
STYLE = {"fedavg": "--", "fedpart": "-.", "fed_resonance": ":", "fedpart_be": "-"}
MARK = {"fedavg": "v", "fedpart": "s", "fed_resonance": "D", "fedpart_be": "o"}

plt.rcParams.update({
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.35,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "figure.dpi": 200, "savefig.bbox": "tight",
})

from matplotlib.ticker import MaxNLocator, AutoMinorLocator


def _style_xy(ax, xmax=None):
    """Denser, more readable axes: more major ticks + minor ticks + dual grid."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which="major", alpha=0.35)
    ax.grid(which="minor", alpha=0.15, linewidth=0.5)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if xmax is not None:
        ax.set_xlim(left=0, right=xmax)


def load_e1():
    """algo -> list of per-round dicts (sorted by round_num)."""
    out = {}
    for a in ALGOS:
        fs = []
        for d_name in DIR_ALIASES.get(a, [a]):
            fs = glob.glob(os.path.join(E1_DIR, d_name, "*", "metrics.json"))
            if fs:
                break
        if not fs:
            print(f"[E1] WARN: no metrics for {a}", file=sys.stderr)
            continue
        d = json.load(open(fs[0]))
        out[a] = sorted(d["rounds"], key=lambda r: r["round_num"])
    return out


def fig_acc_vs_energy(e1):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    xmax = 0
    for a in ALGOS:
        if a not in e1:
            continue
        R = e1[a]
        x = [r["cumulative_energy_j"] / 1e6 for r in R]      # MJ
        y = [r["test_accuracy"] * 100 for r in R]            # %
        xmax = max(xmax, x[-1])
        ax.plot(x, y, STYLE[a], color=COLOR[a], lw=1.7, label=LABEL[a],
                marker=MARK[a], markevery=max(1, len(x) // 10), ms=4)
    ax.set_xlabel("cumulative fleet energy (MJ)")
    ax.set_ylabel("test accuracy (%)")
    _style_xy(ax, xmax=xmax * 1.02)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.savefig(os.path.join(OUT, "fig_acc_vs_energy.pdf"))
    plt.close(fig)
    print("wrote fig_acc_vs_energy.pdf")


def fig_acc_vs_bytes(e1):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    xmax = 0
    for a in ALGOS:
        if a not in e1:
            continue
        R = e1[a]
        x = [r["cumulative_bytes"] / 1e6 for r in R]         # MB uplink
        y = [r["test_accuracy"] * 100 for r in R]            # %
        xmax = max(xmax, x[-1])
        ax.plot(x, y, STYLE[a], color=COLOR[a], lw=1.7, label=LABEL[a],
                marker=MARK[a], markevery=max(1, len(x) // 10), ms=4)
    ax.set_xlabel("cumulative uplink (MB)")
    ax.set_ylabel("test accuracy (%)")
    _style_xy(ax, xmax=xmax * 1.02)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.savefig(os.path.join(OUT, "fig_acc_vs_bytes.pdf"))
    plt.close(fig)
    print("wrote fig_acc_vs_bytes.pdf")


def fig_survival_jain(e1):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for a in ALGOS:
        if a not in e1:
            continue
        R = e1[a]
        rd = [r["round_num"] for r in R]
        axL.plot(rd, [r["num_alive_clients"] for r in R], STYLE[a],
                 color=COLOR[a], lw=1.6, label=LABEL[a])
        axR.plot(rd, [r["jain_index"] for r in R], STYLE[a],
                 color=COLOR[a], lw=1.6, label=LABEL[a])
    axL.set_xlabel("round"); axL.set_ylabel("clients alive")
    axR.set_xlabel("round"); axR.set_ylabel("Jain energy fairness")
    axL.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_survival_jain.pdf"))
    plt.close(fig)
    print("wrote fig_survival_jain.pdf")


# Profile capability order (constrained -> capable).
# Capability ladder ordered by peak GFLOPS (constrained -> capable):
# ESP32 0.5 < RPi Zero2W 8 < RPi 4 24 < Phone-mid 300 < Jetson 472 < Phone-high 1400.
PROF_ORDER = ["esp32_s3", "raspberry_pi_zero2w", "raspberry_pi_4",
              "smartphone_midrange", "jetson_nano", "smartphone_highend"]
PROF_LABEL = {"esp32_s3": "ESP32-S3", "raspberry_pi_zero2w": "RPi Zero2W",
              "raspberry_pi_4": "RPi 4", "smartphone_midrange": "Phone-mid",
              "jetson_nano": "Jetson", "smartphone_highend": "Phone-high"}


def fig_e2(rows):
    if not rows:
        print("[E2] WARN: no aggregated_wide.csv rows; skipping fig:e2",
              file=sys.stderr)
        return
    # index: (profile, algo) -> row
    idx = {(r["profile"], r["algo"]): r for r in rows}
    profs = [p for p in PROF_ORDER if any(p == r["profile"] for r in rows)]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.8, 2.6))

    # Panel A: FedSTEP energy breakdown across profiles (100%-stacked compute/comm).
    fb = [idx.get((p, "fedpart_be")) for p in profs]
    comp = [float(r["compute_fraction"]) * 100 if r else 0 for r in fb]
    comm = [float(r["comm_fraction"]) * 100 if r else 0 for r in fb]
    xlab = [PROF_LABEL[p] for p in profs]
    xpos = range(len(profs))
    axA.bar(xpos, comp, color="#4c72b0", label="compute")
    axA.bar(xpos, comm, bottom=comp, color="#dd8452", label="comm (up+down)")
    axA.set_xticks(list(xpos)); axA.set_xticklabels(xlab, rotation=30, ha="right")
    axA.set_ylabel("FedSTEP energy share (%)")
    axA.set_ylim(0, 100); axA.legend(loc="lower left", fontsize=7)
    # annotate the (tiny) comm fraction
    for i, c in enumerate(comm):
        if c > 0:
            axA.text(i, 101, f"{c:.2f}%", ha="center", va="bottom", fontsize=6)

    # Panel B: comm energy fraction vs device capability (log-y), per algo.
    # Shows the compute<->comm balance shifting toward comm on capable devices
    # (as the per-device compute gap alpha shrinks). Compute still dominates.
    for a in ALGOS:
        ys, xs = [], []
        for j, p in enumerate(profs):
            r = idx.get((p, a))
            if r and float(r["comm_fraction"]) > 0:
                xs.append(j)
                ys.append(float(r["comm_fraction"]) * 100)   # percent
        if xs:
            axB.plot(xs, ys, STYLE[a], color=COLOR[a], lw=1.6,
                     marker=MARK[a], ms=4, label=LABEL[a])
    axB.set_yscale("log")
    axB.set_xticks(list(range(len(profs))))
    axB.set_xticklabels(xlab, rotation=30, ha="right")
    axB.set_ylabel("comm energy share (%)")
    axB.legend(loc="upper left", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_e2_breakdown_survival.pdf"))
    plt.close(fig)
    print("wrote fig_e2_breakdown_survival.pdf")


def load_e2():
    if not os.path.exists(E2_CSV):
        return []
    with open(E2_CSV) as f:
        rows = list(csv.DictReader(f))
    for r in rows:  # normalize new-name runs onto the historical key
        if "algo" in r:
            r["algo"] = _norm_algo(r["algo"])
        if "algorithm" in r:
            r["algorithm"] = _norm_algo(r["algorithm"])
    return rows


def main():
    e1 = load_e1()
    if e1:
        fig_acc_vs_energy(e1)
        fig_acc_vs_bytes(e1)
        fig_survival_jain(e1)
    else:
        print("[E1] no data; skipped E1 figures", file=sys.stderr)
    fig_e2(load_e2())
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
