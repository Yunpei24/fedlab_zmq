"""Phase-2 DMD campaigns: calibration, levels, contrasts.

    python3 scripts/analyze_phase2.py results/fmnist_phase2 \\
        --selection configs/dmd_fmnist_p2/phase2a_selection.json
    python3 scripts/analyze_phase2.py results/fmnist_cbmargin \\
        --experiment cbce_margin \\
        --selection configs/dmd_fmnist_cbm/phase2a_selection.json

Runs live at <root>/<arm>/p<partition>/<run>_s<init>/metrics.json.  Each metric
is averaged over the late window per run.  Runs sharing a partition are not
independent, so the partition is the unit of inference: a contrast pairs runs
on the same (partition, init), averages those deltas within each partition and
tests over partitions (n=3 on the default grid, where the exact sign-flip test
cannot go below p=0.25).  The partition/init variance split is reported
because measuring it is one reason phase 2 decouples the two seeds.
Read-only: nothing is written.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev

from analyze_phase1 import METRICS, fmt, run_row, sign_flip_p, t_two_sided_p

ARMS = ["fedavg", "cbce", "dmdcb_prob", "dmdcb_ptau", "dmdcb_norm", "tail_dual_p"]
CONTRASTS = [
    ("cbce", "fedavg", "class-balancing seul vs FedAvg"),
    ("dmdcb_prob", "fedavg", "DMD-CB probability vs FedAvg"),
    ("dmdcb_prob", "cbce", "marge vs class-balancing seul (cb_ce doit etre battu)"),
    ("dmdcb_ptau", "cbce", "marge satisficing vs class-balancing seul"),
    ("dmdcb_norm", "cbce", "marge normalisee vs class-balancing seul"),
    ("dmdcb_ptau", "dmdcb_prob", "cible tau=0.3 vs tau=0"),
    ("dmdcb_norm", "dmdcb_prob", "espace normalized vs probability"),
    ("tail_dual_p", "dmdcb_prob", "Option B en espace borne vs DMD-CB"),
]
# Every cbce_margin arm is read against the same client at mean_mu=0, which is
# CB-CE exactly, so each contrast is the margin penalty alone.
VIEWS = {
    "bounded_margin": (ARMS, CONTRASTS),
    "cbce_margin": (
        ["cbce_ref", "cbdmd_prob", "cbdmd_ptau", "cbdmd_norm"],
        [
            ("cbdmd_prob", "cbce_ref", "marge probability par-dessus CB-CE"),
            ("cbdmd_ptau", "cbce_ref", "marge satisficing par-dessus CB-CE"),
            ("cbdmd_norm", "cbce_ref", "marge normalisee par-dessus CB-CE"),
        ],
    ),
    "cbce_margin_mc": (
        ["cbce_ref", "cbdmd_norm", "cbdmd_norm_hi"],
        [("cbdmd_norm", "cbce_ref",
          "marge normalisee (mu=0.3) par-dessus CB-CE, Dirichlet par client, "
          "tirages communs"),
         ("cbdmd_norm_hi", "cbce_ref",
          "marge normalisee forte (mu=3) par-dessus CB-CE, tirages communs")],
    ),
    "norm_margin_mc": (
        ["cbce_ref", "cbdmd_norm", "cbdmd_norm_hi"],
        [("cbdmd_norm", "cbce_ref",
          "marge normalisee (mu calibre) par-dessus CB-CE, Dirichlet par client, "
          "tirages communs"),
         ("cbdmd_norm_hi", "cbce_ref",
          "marge normalisee forte (mu=10) par-dessus CB-CE, tirages communs")],
    ),
    # cbce_ref comes from the cbce_margin_mc / norm_margin_mc root (--import-arm).
    "label_skew_mc": (
        ["ce_ref", "cbce_ref", "cbloss", "bsm", "fedlc"],
        [("cbce_ref", "ce_ref", "CB-CE vs CE (FedAvg)"),
         ("cbloss", "cbce_ref", "CB-loss (beta calibre) vs CB-CE"),
         ("bsm", "cbce_ref", "Balanced Softmax (tau calibre) vs CB-CE"),
         ("fedlc", "cbce_ref", "FedLC (tau calibre) vs CB-CE")],
    ),
}
PARAM_NAMES = {"mean_mu": "mu", "label_skew_beta": "beta", "label_skew_tau": "tau"}


def load(root: Path, lo: int, hi: int) -> dict[str, dict[tuple[int, int], dict]]:
    runs: dict[str, dict[tuple[int, int], dict]] = {}
    for path in sorted(root.glob("*/p*/*/metrics.json")):
        init = re.search(r"_s(\d+)$", path.parent.name)
        partition = re.fullmatch(r"p(\d+)", path.parent.parent.name)
        if init is None or partition is None:
            continue
        key = (int(partition.group(1)), int(init.group(1)))
        runs.setdefault(path.parent.parent.parent.name, {})[key] = run_row(path, lo, hi)
    return runs


def partition_means(cells: dict[tuple[int, int], dict], key: str) -> dict[int, float]:
    by_partition: dict[int, list[float]] = {}
    for (partition, _), row in cells.items():
        if row[key] is not None:
            by_partition.setdefault(partition, []).append(row[key])
    return {p: mean(values) for p, values in sorted(by_partition.items())}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("--window", type=int, nargs=2, default=(121, 150),
                        metavar=("FIRST", "LAST"))
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--experiment", choices=sorted(VIEWS), default="bounded_margin")
    parser.add_argument(
        "--import-arm", action="append", default=[], metavar="ARM=ROOT",
        help="read ARM's runs from another results root; valid for pairing only "
             "when that run used the same config, partitions and inits",
    )
    args = parser.parse_args()
    lo, hi = args.window
    runs = load(args.results, lo, hi)
    for spec in args.import_arm:
        arm, root = spec.split("=", 1)
        imported = load(Path(root), lo, hi)
        if arm not in imported:
            raise SystemExit(f"--import-arm: no {arm} runs under {root}")
        runs[arm] = imported[arm]
    arm_order, contrasts = VIEWS[args.experiment]
    arms = [arm for arm in arm_order if arm in runs]

    print(f"# Phase 2, {args.experiment}: {args.results}\n")
    if args.selection and args.selection.exists():
        selection = json.loads(args.selection.read_text())
        print(f"## Calibration 2a (partitions {selection['calibration_partitions']})\n")
        print(f"Critere: {selection['criterion']}.\n")
        print("| bras | valeur | runs | Acc % | MeanBA % | Worst-20 BA % | VarBA |")
        print("|---|---|---|---|---|---|---|")
        for arm, rows in selection["table"].items():
            for row in rows:
                param = row.get("param", "mean_mu")
                value = row[param]
                if row.get("reference"):
                    mark = " (reference)"
                else:
                    mark = " **retenu**" if value == selection["picks"][arm] else ""
                cells = [f"{row[k]:.2f}" if row[k] is not None else "-"
                         for k in ("acc", "mean_ba", "worst20")]
                var = f"{row['var_ba']:.4f}" if row["var_ba"] is not None else "-"
                print(f"| {arm} | {PARAM_NAMES.get(param, param)}={value:g}{mark} | "
                      f"{row['runs']} | {' | '.join(cells)} | {var} |")
        print(f"\nValeurs de test: {selection.get('test_params', selection.get('test_mean_mu'))}\n")

    print(f"## Niveaux 2b (rounds {lo}-{hi})\n")
    print("Moyenne ± ecart-type des moyennes par partition.\n")
    extra = [("eff_mu", "mu effectif"), ("tail_frac", "frac. > eta")]
    header = ["bras", "runs"] + [METRICS[k][2] for k in METRICS] + [lab for _, lab in extra]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for arm in arms:
        cells = [arm, str(len(runs[arm]))]
        for key in list(METRICS) + [k for k, _ in extra]:
            values = list(partition_means(runs[arm], key).values())
            cells.append(fmt(values, key) if values else "-")
        print("| " + " | ".join(cells) + " |")

    print("\n## Part de variance: partition vs init\n")
    print("e.t. entre partitions (moyennes sur les inits) et e.t. moyen entre inits "
          "d'une meme partition.\n")
    print("| bras | Worst-20 BA: partitions | Worst-20 BA: inits | Acc: partitions | Acc: inits |")
    print("|---|---|---|---|---|")
    for arm in arms:
        cells = [arm]
        for key in ("worst20", "acc"):
            scale = METRICS[key][1]
            means = list(partition_means(runs[arm], key).values())
            spreads = []
            for partition in {p for p, _ in runs[arm]}:
                values = [row[key] * scale for (p, _), row in runs[arm].items()
                          if p == partition and row[key] is not None]
                if len(values) > 1:
                    spreads.append(stdev(values))
            cells.append(f"{stdev(means) * scale:.2f}" if len(means) > 1 else "-")
            cells.append(f"{mean(spreads):.2f}" if spreads else "-")
        print("| " + " | ".join(cells) + " |")

    print("\n## Contrastes apparies (B - A), unite = partition\n")
    print("Favorable = dans le sens souhaite (Acc, MeanBA, Worst-20 en hausse; "
          "VarBA, Gap en baisse).\n")
    for b, a, label in contrasts:
        if a not in runs or b not in runs:
            continue
        cells = sorted(set(runs[a]) & set(runs[b]))
        print(f"### {b} - {a}: {label} ({len(cells)} paires)\n")
        print("| metrique | delta moyen ± e.t. | favorable | p_t | p_signe |")
        print("|---|---|---|---|---|")
        for key, (_, scale, name, direction) in METRICS.items():
            per_partition: dict[int, list[float]] = {}
            for cell in cells:
                if runs[b][cell][key] is not None and runs[a][cell][key] is not None:
                    per_partition.setdefault(cell[0], []).append(
                        (runs[b][cell][key] - runs[a][cell][key]) * scale)
            deltas = [mean(values) for _, values in sorted(per_partition.items())]
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
