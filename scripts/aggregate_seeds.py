#!/usr/bin/env python
"""Aggregate multi-seed FL runs into mean +/- std (CSV + paste-ready LaTeX rows).

Each run writes <...>/metrics.json with a "summary" block. This tool groups runs by
(algorithm, dataset) — pooling over seeds — and reports mean +/- std for the headline
metrics, so a table cell becomes e.g. "69.8 +/- 0.4".

Usage
-----
  # all algos under a directory tree (recurses, finds every metrics.json)
  python scripts/aggregate_seeds.py results/E5/femnist_natural

  # several roots; only these metrics; LaTeX rows too
  python scripts/aggregate_seeds.py results/E1_a1 results/E1_a1_s43 --latex

  # group key controls what is pooled (default: algorithm,dataset)
  python scripts/aggregate_seeds.py results/E3 --group-by algorithm

Notes
-----
- Pools EVERY metrics.json found under the given roots, grouped by --group-by. Put the
  seeds you want aggregated under the same root(s); the seed itself is read from the
  summary and only used to count n and warn on duplicates.
- best_acc/final_acc/Jain/alive vary with seed (Dirichlet partition) -> report +/-std.
  energy/uplink are near-deterministic -> std is tiny (reported anyway).
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

# metric_key_in_summary -> (display name, scale, unit, decimals)
METRICS = [
    ("best_accuracy",     "best_acc",  100.0, "%",  2),
    ("final_accuracy",    "final_acc", 100.0, "%",  2),
    ("total_energy_j",    "energy_kJ", 1e-3,  "kJ", 1),
    ("uplink_mb",         "uplink_MB", 1.0,   "MB", 1),
    ("final_jain_index",  "jain",      1.0,   "",   3),
    ("final_survival_ratio", "alive_frac", 1.0, "", 3),
]


def _summary(path):
    try:
        d = json.load(open(path))
    except Exception as e:
        print(f"  [skip] {path}: {e}", file=sys.stderr)
        return None
    s = d.get("summary", d)
    # uplink_mb may live only in survival/rounds; derive from total_bytes_gb if absent
    if "uplink_mb" not in s and "total_bytes_gb" in s:
        # decimal MB (bytes/1e6); total_bytes_gb is bytes/1e9, so x1000 (NOT x1024).
        # Matches the dashboard's MB convention.
        s = {**s, "uplink_mb": s["total_bytes_gb"] * 1000.0}
    return s


def _stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return mean, std, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="dirs to scan for metrics.json (recursive)")
    ap.add_argument("--group-by", default="algorithm,dataset",
                    help="comma-sep summary keys to pool over (default: algorithm,dataset)")
    ap.add_argument("--latex", action="store_true", help="also print paste-ready LaTeX rows")
    ap.add_argument("--out", default=None, help="write the CSV here (default: stdout only)")
    ap.add_argument("--seeds-only", action="store_true",
                    help="only count runs whose path has a /seed<N>/ component "
                         "(ignores sibling experiments like IID1_lr003 living under "
                         "the same root)")
    args = ap.parse_args()

    gkeys = [k.strip() for k in args.group_by.split(",")]
    files = []
    for r in args.roots:
        files += glob.glob(os.path.join(r, "**", "metrics.json"), recursive=True)
    if args.seeds_only:
        _seed_dir = re.compile(r"(?:^|/)seed\d+(?:/|$)")
        files = [f for f in files if _seed_dir.search(os.path.dirname(f))]
    if not files:
        print("No metrics.json found under: " + ", ".join(args.roots), file=sys.stderr)
        sys.exit(1)

    groups = defaultdict(list)   # key tuple -> list of (seed, summary)
    for f in sorted(files):
        s = _summary(f)
        if s is None:
            continue
        key = tuple(str(s.get(k, "?")) for k in gkeys)
        groups[key].append((s.get("seed", "?"), s))

    rows = []
    for key, runs in sorted(groups.items()):
        seeds = [seed for seed, _ in runs]
        if len(set(seeds)) != len(seeds):
            print(f"  [warn] {key}: duplicate seeds {seeds} (pooling all)", file=sys.stderr)
        row = {gkeys[i]: key[i] for i in range(len(gkeys))}
        row["n_seeds"] = len(runs)
        row["seeds"] = "|".join(str(s) for s in seeds)
        for mkey, disp, scale, unit, dec in METRICS:
            vals = [s[mkey] * scale for _, s in runs if mkey in s and s[mkey] is not None]
            if not vals:
                row[disp] = ""
                continue
            mean, std, _ = _stats(vals)
            row[disp] = f"{mean:.{dec}f}+/-{std:.{dec}f}"
        rows.append(row)

    fieldnames = gkeys + ["n_seeds", "seeds"] + [m[1] for m in METRICS]
    # ---- CSV (stdout) ----
    w = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)
    if args.out:
        with open(args.out, "w", newline="") as fh:
            ww = csv.DictWriter(fh, fieldnames=fieldnames)
            ww.writeheader(); ww.writerows(rows)
        print(f"\n# CSV written to {args.out}", file=sys.stderr)

    # ---- LaTeX rows (acc / energy / uplink / jain / alive) ----
    if args.latex:
        print("\n% --- paste-ready LaTeX rows (mean $\\pm$ std) ---")
        for r in rows:
            label = " ".join(r[k] for k in gkeys)
            def cell(disp):
                v = r.get(disp, "")
                return v.replace("+/-", r" $\pm$ ") if v else r"\TBD"
            print(f"{label} & {cell('best_acc')} & {cell('energy_kJ')} & "
                  f"{cell('uplink_MB')} & {cell('jain')} & {cell('alive_frac')} "
                  f"\\\\  % n={r['n_seeds']} seeds={r['seeds']}")


if __name__ == "__main__":
    main()
