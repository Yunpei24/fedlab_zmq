#!/usr/bin/env python3
"""
scripts/compare_results.py
===========================
Compare multiple experiment results and generate:
  1. Summary comparison table (terminal)
  2. LaTeX table (paper-ready)
  3. Matplotlib plots (accuracy, energy, Jain index, battery)

Usage:
  python scripts/compare_results.py --results results/benchmark/

  # With custom metric
  python scripts/compare_results.py \\
    --results results/eceffl results/fedavg results/fedprox \\
    --metric test_accuracy

  # LaTeX output
  python scripts/compare_results.py --results results/benchmark/ --latex

  # Save plots
  python scripts/compare_results.py --results results/benchmark/ --plot --save-dir figures/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_results(results_dir: str) -> dict[str, dict]:
    """
    Load all experiment results from a directory.
    Handles both:
      - Single metrics.json (one algorithm)
      - Subdirectory structure (benchmark output)
    """
    p = Path(results_dir)
    all_results = {}

    if (p / "metrics.json").exists():
        # Single experiment
        with open(p / "metrics.json") as f:
            data = json.load(f)
        algo = data.get("algorithm", p.name)
        all_results[algo] = data

    elif (p / "comparison_summary.json").exists():
        # Benchmark output with individual subdirs
        for subdir in sorted(p.iterdir()):
            if subdir.is_dir() and (subdir / "metrics.json").exists():
                with open(subdir / "metrics.json") as f:
                    data = json.load(f)
                algo = data.get("algorithm", subdir.name)
                all_results[algo] = data

    else:
        # Try loading from multiple paths passed directly
        for subdir in sorted(p.iterdir()):
            if subdir.is_dir() and (subdir / "metrics.json").exists():
                with open(subdir / "metrics.json") as f:
                    data = json.load(f)
                algo = data.get("algorithm", subdir.name)
                all_results[algo] = data

    return all_results


def load_from_paths(paths: list[str]) -> dict[str, dict]:
    """Load results from a list of directory paths."""
    all_results = {}
    for path in paths:
        p = Path(path)
        mf = p / "metrics.json"
        if not mf.exists():
            print(f"[WARN] No metrics.json in {path}")
            continue
        with open(mf) as f:
            data = json.load(f)
        algo = data.get("algorithm", p.name)
        all_results[algo] = data
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_series(result: dict, metric: str) -> list[float]:
    """Extract per-round metric series from result dict."""
    rounds = result.get("rounds", [])
    return [r.get(metric, 0.0) for r in rounds]


def summarize(results: dict[str, dict]) -> dict[str, dict]:
    """Compute summary statistics for each algorithm."""
    summaries = {}
    for algo, result in results.items():
        rounds = result.get("rounds", [])
        if not rounds:
            continue

        accs = [r.get("test_accuracy", 0) for r in rounds]
        energies = [r.get("total_energy_j", 0) for r in rounds]
        bytes_sent = [r.get("total_bytes", 0) for r in rounds]
        jains = [r.get("jain_index", 0) for r in rounds]
        batteries = [r.get("avg_battery_j", 0) for r in rounds]

        # Energy to reach 70% accuracy (or best achievable)
        target_acc = 0.70
        e_to_target = None
        cum_e = 0.0
        for r in rounds:
            cum_e += r.get("total_energy_j", 0)
            if r.get("test_accuracy", 0) >= target_acc:
                e_to_target = cum_e
                break

        summaries[algo] = {
            "best_accuracy":       max(accs),
            "final_accuracy":      accs[-1],
            "rounds_to_best":      accs.index(max(accs)) + 1,
            "total_bytes_gb":      sum(bytes_sent) / 1e9,
            "total_energy_j":      sum(energies),
            "avg_energy_per_round": np.mean(energies),
            "final_jain_index":    jains[-1],
            "avg_jain_index":      np.mean(jains),
            "final_battery_j":     batteries[-1],
            "battery_depletion":   (batteries[0] - batteries[-1]) / max(batteries[0], 1)
                                   if batteries else 0.0,
            "energy_to_70pct":     e_to_target,
            "num_rounds":          len(rounds),
        }

    return summaries


# ─────────────────────────────────────────────────────────────────────────────
# Terminal table
# ─────────────────────────────────────────────────────────────────────────────

def print_table(summaries: dict[str, dict]):
    """Print comparison table in terminal."""
    print(f"\n{'='*110}")
    print(f"  FL Algorithm Comparison — CIFAR-10 | Dirichlet(α=0.5) | 10 clients | 100 rounds")
    print(f"{'='*110}")
    header = (
        f"  {'Algorithm':<14} {'Best Acc':>10} {'Final Acc':>10} "
        f"{'Comm GB':>9} {'Energy J':>12} {'Jain':>8} "
        f"{'Batt Dep%':>10} {'E@70%':>10}"
    )
    print(header)
    print(f"  {'-'*98}")

    # Sort by best accuracy descending
    sorted_algos = sorted(summaries.items(),
                          key=lambda x: x[1]["best_accuracy"], reverse=True)

    for algo, s in sorted_algos:
        e70 = f"{s['energy_to_70pct']:.0f}" if s["energy_to_70pct"] else "N/A"
        bdep = f"{s['battery_depletion']*100:.1f}%"
        print(
            f"  {algo:<14} {s['best_accuracy']*100:>9.2f}% "
            f"{s['final_accuracy']*100:>9.2f}% "
            f"{s['total_bytes_gb']:>9.3f} "
            f"{s['total_energy_j']:>12.0f} "
            f"{s['final_jain_index']:>8.4f} "
            f"{bdep:>10} "
            f"{e70:>10}"
        )

    print(f"{'='*110}")


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table
# ─────────────────────────────────────────────────────────────────────────────

ALGO_DISPLAY_NAMES = {
    "fedavg":   "FedAvg",
    "eceffl":   "\\textbf{E-CEFFL} (ours)",
    "leanfed":  "LeanFed",
    "fedbacys": "FedBacys",
    "vaishnav": "Vaishnav",
    "fedsparq": "FedSparQ",
    "fedprox":  "FedProx",
    "scaffold": "SCAFFOLD",
}


def generate_latex_table(summaries: dict[str, dict]) -> str:
    """Generate paper-ready LaTeX table."""

    # Preferred ordering
    order = ["fedavg", "fedprox", "scaffold", "leanfed", "fedbacys",
             "vaishnav", "fedsparq", "eceffl"]
    ordered = [(a, summaries[a]) for a in order if a in summaries]
    # Add any remaining
    for a, s in summaries.items():
        if a not in order:
            ordered.append((a, s))

    best_acc     = max(s["best_accuracy"] for _, s in ordered)
    min_energy   = min(s["total_energy_j"] for _, s in ordered)
    min_bytes    = min(s["total_bytes_gb"] for _, s in ordered)
    max_jain     = max(s["final_jain_index"] for _, s in ordered)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison of FL algorithms on CIFAR-10 (Dirichlet $\alpha=0.5$, "
        r"10 clients, 100 rounds). \textbf{Bold} = best in column. "
        r"E@70\% = energy (J) to reach 70\% test accuracy.}",
        r"\label{tab:comparison_cifar10}",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Algorithm & Best Acc (\%) & Final Acc (\%) & Comm (GB) & "
        r"Energy (J) & Jain & E@70\% (J) \\",
        r"\midrule",
    ]

    for algo, s in ordered:
        name = ALGO_DISPLAY_NAMES.get(algo, algo)

        def fmt_best(val, best, fmt_fn, higher_is_better=True):
            if higher_is_better:
                return f"\\textbf{{{fmt_fn(val)}}}" if abs(val - best) < 1e-9 else fmt_fn(val)
            else:
                return f"\\textbf{{{fmt_fn(val)}}}" if abs(val - best) < 1e-9 else fmt_fn(val)

        acc_best  = fmt_best(s["best_accuracy"]*100,  best_acc*100,  lambda v: f"{v:.2f}")
        acc_final = f"{s['final_accuracy']*100:.2f}"
        comm      = fmt_best(s["total_bytes_gb"], min_bytes, lambda v: f"{v:.3f}", False)
        energy    = fmt_best(s["total_energy_j"],  min_energy, lambda v: f"{v:.0f}", False)
        jain      = fmt_best(s["final_jain_index"], max_jain, lambda v: f"{v:.4f}")
        e70       = f"{s['energy_to_70pct']:.0f}" if s["energy_to_70pct"] else "--"

        lines.append(
            f"{name} & {acc_best} & {acc_final} & {comm} & {energy} & {jain} & {e70} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(results: dict[str, dict], save_dir: str = None):
    """Generate 4-panel comparison figure."""
    try:
        import matplotlib
        matplotlib.use("Agg" if save_dir else "TkAgg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping plots")
        return

    # Color palette (colorblind-friendly)
    COLORS = {
        "fedavg":   "#808080",   # gray
        "fedprox":  "#1f77b4",   # blue
        "scaffold": "#aec7e8",   # light blue
        "leanfed":  "#ff7f0e",   # orange
        "fedbacys": "#ffbb78",   # light orange
        "vaishnav": "#2ca02c",   # green
        "fedsparq": "#98df8a",   # light green
        "eceffl":   "#d62728",   # red (ours)
    }
    LINESTYLES = {
        "fedavg":   "--",
        "fedprox":  "-.",
        "scaffold": ":",
        "leanfed":  "--",
        "fedbacys": "-.",
        "vaishnav": ":",
        "fedsparq": "--",
        "eceffl":   "-",
    }
    LINEWIDTHS = {k: (2.5 if k == "eceffl" else 1.5) for k in COLORS}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("FL Algorithm Comparison — CIFAR-10 | Dirichlet(α=0.5) | 10 clients",
                 fontsize=13, fontweight="bold")

    ax_acc, ax_bytes, ax_energy, ax_jain = axes.flat

    # Preferred display order
    order = ["fedavg", "fedprox", "scaffold", "leanfed", "fedbacys",
             "vaishnav", "fedsparq", "eceffl"]
    ordered_algos = [a for a in order if a in results]
    ordered_algos += [a for a in results if a not in order]

    for algo in ordered_algos:
        result = results[algo]
        rounds = result.get("rounds", [])
        if not rounds:
            continue
        xs = list(range(1, len(rounds) + 1))
        color = COLORS.get(algo, "black")
        ls    = LINESTYLES.get(algo, "-")
        lw    = LINEWIDTHS.get(algo, 1.5)
        label = ALGO_DISPLAY_NAMES.get(algo, algo)

        accs    = [r.get("test_accuracy", 0) * 100 for r in rounds]
        # Cumulative bytes (GB)
        cbytes  = np.cumsum([r.get("total_bytes", 0) for r in rounds]) / 1e9
        # Cumulative energy (J)
        cenergy = np.cumsum([r.get("total_energy_j", 0) for r in rounds])
        jains   = [r.get("jain_index", 1.0) for r in rounds]

        ax_acc.plot(xs, accs, color=color, ls=ls, lw=lw, label=label)
        ax_bytes.plot(xs, cbytes, color=color, ls=ls, lw=lw, label=label)
        ax_energy.plot(xs, cenergy, color=color, ls=ls, lw=lw, label=label)
        ax_jain.plot(xs, jains, color=color, ls=ls, lw=lw, label=label)

    # Formatting
    for ax, ylabel, title in [
        (ax_acc,    "Test Accuracy (%)",    "Test Accuracy vs Rounds"),
        (ax_bytes,  "Cumulative Comm (GB)", "Communication Cost"),
        (ax_energy, "Cumulative Energy (J)","Energy Consumption"),
        (ax_jain,   "Jain Fairness Index",  "Participation Fairness"),
    ]:
        ax.set_xlabel("Round", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right" if "Accuracy" in ylabel else "upper left")

    ax_jain.set_ylim(0, 1.05)

    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        path = Path(save_dir) / "comparison_figure.pdf"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Figure saved: {path}")
        # Also save PNG
        fig.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    else:
        plt.show()

    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Compare FL experiment results and generate tables/figures"
    )
    p.add_argument("--results", type=str, nargs="+", required=True,
                   help="Path(s) to result directories")
    p.add_argument("--latex",    action="store_true",
                   help="Output LaTeX table")
    p.add_argument("--plot",     action="store_true",
                   help="Generate comparison plots")
    p.add_argument("--save-dir", type=str, default=None,
                   help="Save figures to this directory")
    p.add_argument("--save-latex", type=str, default=None,
                   help="Save LaTeX table to this file")

    args = p.parse_args()

    # Load results
    if len(args.results) == 1:
        results = load_results(args.results[0])
    else:
        results = load_from_paths(args.results)

    if not results:
        print("[ERROR] No results found.")
        sys.exit(1)

    print(f"\nLoaded results for: {list(results.keys())}")

    summaries = summarize(results)
    print_table(summaries)

    if args.latex:
        latex = generate_latex_table(summaries)
        print("\n--- LaTeX Table ---\n")
        print(latex)
        if args.save_latex:
            with open(args.save_latex, "w") as f:
                f.write(latex + "\n")
            print(f"\nLaTeX saved: {args.save_latex}")

    if args.plot:
        plot_comparison(results, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
