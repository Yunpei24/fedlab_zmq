#!/usr/bin/env python3
"""
scripts/run_costmodel_ablation.py
=================================
Run {fedavg, fedpart, fedpart_be} x {phi, corrected, measured} on the same
CIFAR-10 / K=30 ESP32 / SoC[5,95]% / α=1 / E=8 / 200 rounds setup and produce
a comparison CSV.

Goal: does the FedPartBE-vs-FedPart gap *widen* when we move from the phi
cost model (legacy, under-counts shallow groups) to the corrected (analytic
position-aware) or measured (FlopCounterMode) model?

Why a subprocess runner: the experiment loop in run_experiment.py owns its
own torch / CUDA / DataLoader state. Calling it 9 times in one process leaks
that state between runs. Subprocess isolates each run cleanly and reuses the
fully-debugged CLI path.

Usage
-----
    python scripts/run_costmodel_ablation.py [--rounds 200] [--seed 42]

Output
------
    results/costmodel_ablation/
        comparison.csv          ← Best Acc, Final Acc, Alive@N, median lifetime,
                                  survival AUC, total energy, total uplink
        <algo>__<cost_model>/   ← per-run output dir from run_experiment.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/fedpartbe_survival_wide_cifar10.yaml"
DEFAULT_OUT = ROOT / "results/costmodel_ablation"


ALGOS = ["fedavg", "fedpart", "fedpart_be"]
COST_MODELS = ["phi", "corrected", "measured"]


def _build_command(algo: str, cost_model: str, args, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable, str(ROOT / "run_experiment.py"),
        "--config", str(args.config),
        "--algo", algo,
        "--cost-model", cost_model,
        "--rounds", str(args.rounds),
        "--epochs", str(args.epochs),
        "--seed", str(args.seed),
        "--device", args.device,
        "--output", str(out_dir),
    ]
    return cmd


def _run_one(algo: str, cost_model: str, args, base_out: Path) -> Path:
    out_dir = base_out / f"{algo}__{cost_model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_command(algo, cost_model, args, out_dir)
    print(f"\n[ablation] >>> {algo} / cost_model={cost_model}")
    print(f"           cmd: {' '.join(shlex.quote(c) for c in cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(ROOT))
    if res.returncode != 0:
        print(f"           [warn] returncode={res.returncode}")
    else:
        print(f"           done in {time.time()-t0:.0f}s")
    return out_dir


def _find_metrics_json(run_dir: Path) -> Path | None:
    candidates = list(run_dir.glob("**/metrics.json"))
    return candidates[0] if candidates else None


def _extract_row(algo: str, cost_model: str, metrics_path: Path | None) -> dict:
    row: dict = {
        "algo":        algo,
        "cost_model":  cost_model,
        "best_acc":    None,
        "final_acc":   None,
        "alive_final": None,
        "median_lifetime": None,
        "survival_auc":    None,
        "total_energy_j":  None,
        "total_bytes_gb":  None,
    }
    if metrics_path is None or not metrics_path.exists():
        row["error"] = "metrics.json not found"
        return row
    with open(metrics_path) as fh:
        data = json.load(fh)
    summary = data.get("summary", {})
    survival = data.get("survival", {})
    row["best_acc"]        = summary.get("best_accuracy")
    row["final_acc"]       = summary.get("final_accuracy")
    row["total_energy_j"]  = summary.get("total_energy_j")
    row["total_bytes_gb"]  = summary.get("total_bytes_gb")
    row["median_lifetime"] = survival.get("median_lifetime")
    row["survival_auc"]    = survival.get("survival_auc")
    rounds = data.get("rounds", [])
    if rounds:
        row["alive_final"] = rounds[-1].get("num_alive_clients")
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    p.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--algos", nargs="+", default=ALGOS,
                   help="Subset of algos to run (default: fedavg fedpart fedpart_be)")
    p.add_argument("--cost-models", nargs="+", default=COST_MODELS,
                   help="Subset of cost_models to run (default: phi corrected measured)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned commands without launching anything.")
    args = p.parse_args()

    base_out = Path(args.output)
    base_out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for algo in args.algos:
        for cm in args.cost_models:
            run_dir = base_out / f"{algo}__{cm}"
            if args.dry_run:
                cmd = _build_command(algo, cm, args, run_dir)
                print(f"[dry-run] {algo} / {cm}: {' '.join(shlex.quote(c) for c in cmd)}")
                continue
            run_dir = _run_one(algo, cm, args, base_out)
            metrics_json = _find_metrics_json(run_dir)
            rows.append(_extract_row(algo, cm, metrics_json))

    if args.dry_run:
        return

    # Comparison CSV
    csv_path = base_out / "comparison.csv"
    fields = ["algo", "cost_model", "best_acc", "final_acc", "alive_final",
              "median_lifetime", "survival_auc", "total_energy_j", "total_bytes_gb"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})

    # Pretty print to stdout for the human reader.
    print(f"\n=== Cost-model ablation summary ({csv_path}) ===")
    header = f"{'algo':<12} {'cost_model':<10} {'best_acc':>9} {'final_acc':>10} {'alive':>6} {'med_life':>9} {'surv_auc':>9} {'energy_J':>10} {'bytes_GB':>9}"
    print(header)
    print("-" * len(header))
    for row in rows:
        def _fmt(v, w=9, p=4):
            if v is None: return f"{'N/A':>{w}}"
            if isinstance(v, float): return f"{v:>{w}.{p}f}"
            return f"{v:>{w}}"
        print(
            f"{row['algo']:<12} {row['cost_model']:<10}"
            f" {_fmt(row['best_acc'])} {_fmt(row['final_acc'], 10)}"
            f" {_fmt(row['alive_final'], 6)}"
            f" {_fmt(row['median_lifetime'])}"
            f" {_fmt(row['survival_auc'], 9, 0)}"
            f" {_fmt(row['total_energy_j'], 10, 1)}"
            f" {_fmt(row['total_bytes_gb'], 9, 3)}"
        )


if __name__ == "__main__":
    main()
