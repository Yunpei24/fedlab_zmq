"""Phase-1 DMD campaign: late-window levels and seed-paired contrasts.

    python3 scripts/analyze_phase1.py results/fmnist_phase1
    python3 scripts/analyze_phase1.py results/fmnist_phase1 \\
        --calibration configs/dmd_fmnist/dmdcb_hi_calibration.json

Per run, each metric is averaged over the rounds that measured it inside the
late window (rounds 121-150 by default).  Arms are compared seed by seed: the
seed fixes both the partition and the initialisation, so B - A on one seed is a
paired difference.  With n=4 seeds the exact sign-flip test cannot go below
p=0.125, so "4/4 seeds" is p=0.125; the paired t-test is reported next to it.
Read-only: nothing is written.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev

# key -> (saved field, scale, label, desirable direction)
METRICS = {
    "acc": ("test_accuracy", 100.0, "Acc %", "higher"),
    "mean_ba": ("mean_client_balanced_accuracy_pct", 1.0, "MeanBA %", "higher"),
    "worst20": ("worst20_balanced_accuracy_pct", 1.0, "Worst-20 BA %", "higher"),
    "var_ba": ("client_balanced_accuracy_variance", 1.0, "VarBA", "lower"),
    "gap": ("best_worst_balanced_accuracy_gap_pct", 1.0, "Gap B20-W20 pp", "lower"),
}
ARMS = ["fedavg", "dmdcb", "dmdcb_hi", "usv025", "tail_emp", "tail_dual"]
CONTRASTS = [
    ("dmdcb", "fedavg", "DMD-CB vs FedAvg"),
    ("usv025", "dmdcb", "terme USV, mu_M fixe"),
    ("dmdcb_hi", "dmdcb", "mu_M releve a l'intensite effective de USV"),
    ("usv025", "dmdcb_hi", "Q1: USV a intensite egale (~0 => USV = mu_M plus fort)"),
    ("tail_dual", "usv025", "Q2: CVaR a seuil appris vs USV"),
    ("tail_emp", "dmdcb", "Q3: CVaR empirique vs DMD-CB (~0 si inerte)"),
]


def t_two_sided_p(t: float, df: int) -> float:
    """P(|T| >= |t|) for Student's t with integer df (Abramowitz-Stegun 26.7.3-4)."""

    theta = math.atan(abs(t) / math.sqrt(df))
    s, c = math.sin(theta), math.cos(theta)
    if df % 2 == 1:
        total = term = c if df > 1 else 0.0
        for k in range(3, df - 1, 2):
            term *= c * c * (k - 1) / k
            total += term
        inside = 2.0 / math.pi * (theta + s * total)
    else:
        total = term = 1.0
        for k in range(2, df - 1, 2):
            term *= c * c * (k - 1) / k
            total += term
        inside = s * total
    return min(1.0, max(0.0, 1.0 - inside))


def sign_flip_p(deltas: list[float]) -> float:
    """Exact two-sided sign-flip permutation p-value of the mean difference."""

    observed = abs(mean(deltas))
    flips = list(itertools.product((1.0, -1.0), repeat=len(deltas)))
    extreme = sum(
        abs(mean(sign * d for sign, d in zip(signs, deltas))) >= observed - 1e-12
        for signs in flips
    )
    return extreme / len(flips)


def late_mean(rounds: list[dict], field: str, lo: int, hi: int) -> float | None:
    values = [
        r[field] for r in rounds
        if lo <= r["round_num"] <= hi and r.get(field) is not None
        # test_accuracy is carried forward between global evaluations
        and (field != "test_accuracy" or r.get("global_eval_measured", True))
    ]
    return mean(values) if values else None


def run_row(path: Path, lo: int, hi: int) -> dict:
    """Late-window metrics plus the DMD intensity audit of one finished run."""

    result = json.loads(path.read_text())
    rounds = result["rounds"]
    row = {key: late_mean(rounds, field, lo, hi)
           for key, (field, _, _, _) in METRICS.items()}
    context = [r for r in rounds if r.get("avg_local_dmd_effective_mu") is not None]
    row["eff_mu"] = (mean(r["avg_local_dmd_effective_mu"] for r in context)
                     if context else None)
    # Every DMD variant logs the cohort eta audit; it only drives the penalty
    # for the tail variant.
    is_tail = result["summary"].get("algorithm") == "dmd_tail"
    row["tail_frac"] = (mean(r["dmd_cvar_tail_fraction_above_eta"] for r in context)
                        if context and is_tail else None)
    row["rounds"] = rounds[-1]["round_num"]
    return row


def load(root: Path, lo: int, hi: int) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = {}
    for path in sorted(root.glob("*/*/metrics.json")):
        seed = re.search(r"_s(\d+)$", path.parent.name)
        if seed is None:
            continue
        runs.setdefault(path.parent.parent.name, {})[int(seed.group(1))] = run_row(
            path, lo, hi)
    return runs


def fmt(values: list[float], key: str) -> str:
    scale = METRICS[key][1] if key in METRICS else 1.0
    digits = 4 if key in ("var_ba", "eff_mu", "tail_frac") else 2
    values = [v * scale for v in values]
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{mean(values):.{digits}f} ± {stdev(values):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("--window", type=int, nargs=2, default=(121, 150),
                        metavar=("FIRST", "LAST"))
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    lo, hi = args.window
    runs = load(args.results, lo, hi)
    arms = [arm for arm in ARMS if arm in runs] + sorted(set(runs) - set(ARMS))

    print(f"# Phase 1: {args.results}\n")
    print(f"Fenetre tardive: rounds {lo}-{hi}. Moyenne ± ecart-type sur les seeds.\n")
    if args.calibration and args.calibration.exists():
        cal = json.loads(args.calibration.read_text())
        print(f"dmdcb_hi: mean_mu = {cal['dmdcb_hi_mean_mu']} "
              f"(intensite effective mesuree sur usv025, "
              f"par seed: {', '.join(f'{v:.4f}' for v in cal['per_run'].values())})\n")

    extra = [("eff_mu", "mu effectif"), ("tail_frac", "frac. > eta")]
    header = ["bras", "seeds"] + [METRICS[k][2] for k in METRICS] + [lab for _, lab in extra]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for arm in arms:
        seeds = runs[arm]
        cells = [arm, ",".join(str(s) for s in sorted(seeds))]
        for key in list(METRICS) + [k for k, _ in extra]:
            values = [row[key] for row in seeds.values() if row[key] is not None]
            cells.append(fmt(values, key) if values else "-")
        print("| " + " | ".join(cells) + " |")

    print("\n## Contrastes apparies par seed (B - A)\n")
    print("Favorable = dans le sens souhaite (Acc, MeanBA, Worst-20 en hausse; "
          "VarBA, Gap en baisse). p_t: t-test apparie; p_signe: permutation exacte.\n")
    for b, a, label in CONTRASTS:
        if a not in runs or b not in runs:
            continue
        seeds = sorted(set(runs[a]) & set(runs[b]))
        print(f"### {b} - {a}: {label} (n={len(seeds)})\n")
        print("| metrique | delta moyen ± e.t. | favorable | p_t | p_signe |")
        print("|---|---|---|---|---|")
        for key, (_, scale, name, direction) in METRICS.items():
            deltas = [(runs[b][s][key] - runs[a][s][key]) * scale for s in seeds
                      if runs[b][s][key] is not None and runs[a][s][key] is not None]
            if len(deltas) < 2:
                continue
            sd = stdev(deltas)
            t = mean(deltas) / (sd / math.sqrt(len(deltas))) if sd > 0 else (
                math.inf if mean(deltas) != 0 else 0.0)
            good = sum((d > 0) if direction == "higher" else (d < 0) for d in deltas)
            digits = 4 if key == "var_ba" else 2
            print(f"| {name} | {mean(deltas):+.{digits}f} ± {sd:.{digits}f} | "
                  f"{good}/{len(deltas)} | {t_two_sided_p(t, len(deltas) - 1):.3f} | "
                  f"{sign_flip_p(deltas):.3f} |")
        print()


if __name__ == "__main__":
    main()
