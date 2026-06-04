"""
metrics/survival.py
===================
Fleet survival + Pareto metrics for FL energy experiments.

Inputs the per-round metrics already produced by every algorithm
(``server_aggregate`` returns dicts with ``avg_battery_j``, ``num_clients``,
``total_energy_j``, etc.), plus the final ClientState snapshots, and produces
a small CSV per algorithm with:

  Per-round (one row per round)
  ------------------------------
    round, num_alive, cum_energy_j, best_acc, mean_acc

  Summary (separate JSON-encoded extra row, or written separately)
  ----------------------------------------------------------------
    median_lifetime
    round_of_nth_death  (N = 5, 10, 15)
    survival_auc        (Σ num_alive)
    participation_frac  (Σ num_alive) / (num_clients * num_rounds)

A client's "lifetime" = the round index at which its battery first crossed
zero. Clients alive at end of run have lifetime = num_rounds + 1 (acts as
inf for ranking purposes; the median is robust to this).

This module does not call into any algorithm — every input is produced by
the existing runner. Drop it into run_experiment.py as a final step.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Optional


def _safe_get(d: dict, *keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def derive_per_round_rows(
    rounds_metrics: list[dict],
    accuracy_history: Optional[list[float]] = None,
    num_clients_total: int = 0,
) -> list[dict]:
    """Build the per-round table for the CSV.

    `rounds_metrics[r]` is whatever the algorithm's `server_aggregate`
    returned for round r (dict). `accuracy_history[r]` is the test-set
    accuracy after round r (added by `run_experiment.py`).

    The cumulative-energy column is the running sum of
    `total_energy_j` over rounds, so a single column captures both the
    instantaneous draw and the cumulative budget burn.
    """
    rows: list[dict] = []
    cum_energy = 0.0
    best_acc = 0.0
    acc_hist = accuracy_history or []

    for r, m in enumerate(rounds_metrics):
        energy = float(m.get("total_energy_j", 0.0) or 0.0)
        cum_energy += energy
        num_alive = m.get("num_alive_clients", None)
        if num_alive is None:
            # Fallback: best-effort using num_clients which is K_round (selected
            # for the round), not necessarily fleet-alive count.
            num_alive = m.get("num_clients", num_clients_total)
        acc = acc_hist[r] if r < len(acc_hist) else None
        if acc is not None:
            best_acc = max(best_acc, float(acc))
        rows.append(
            {
                "round": r,
                "num_alive": int(num_alive),
                "cum_energy_j": cum_energy,
                "best_acc": best_acc if acc is not None else "",
                "mean_acc": float(acc) if acc is not None else "",
            }
        )
    return rows


def derive_lifetimes(
    client_states: list,
    num_rounds: int,
) -> list[int]:
    """Return per-client lifetime in rounds.

    A client whose battery is still > 0 at end of run gets lifetime
    = num_rounds + 1 — large enough to act as "no death" without being a
    NaN that breaks the median.

    If a client carries a `death_round` attribute (e.g. populated by the
    server when battery first hit zero), use it; otherwise fall back to
    end-of-run state: 0 battery => death at last round, >0 => alive.
    """
    lifetimes: list[int] = []
    for st in client_states:
        # Support both dict snapshots (from run_experiment.py) and live
        # ClientState dataclasses (programmatic callers).
        if isinstance(st, dict):
            dr = st.get("death_round")
            battery = float(st.get("battery_j", 0.0) or 0.0)
        else:
            dr = getattr(st, "death_round", None)
            if dr is None:
                custom = getattr(st, "custom", None) or {}
                dr = custom.get("death_round") if isinstance(custom, dict) else None
            battery = float(getattr(st, "battery_j", 0.0) or 0.0)
        if isinstance(dr, int) and dr >= 0:
            lifetimes.append(dr)
            continue
        if battery <= 0:
            # Best-effort: place death at the last round that ran.
            lifetimes.append(num_rounds)
        else:
            lifetimes.append(num_rounds + 1)
    return lifetimes


def summary_from(
    per_round_rows: list[dict],
    lifetimes: list[int],
    num_clients_total: int,
    num_rounds: int,
) -> dict:
    """Aggregate scalar metrics suitable for a comparison table.

    Returns
    -------
    dict with keys:
      median_lifetime, round_of_5th_death, round_of_10th_death,
      round_of_15th_death, survival_auc, participation_frac,
      cum_energy_j (final), best_acc, final_acc
    """
    summary: dict[str, Any] = {}

    if lifetimes:
        sorted_l = sorted(lifetimes)
        n = len(sorted_l)
        if n % 2 == 0:
            summary["median_lifetime"] = 0.5 * (sorted_l[n // 2 - 1] + sorted_l[n // 2])
        else:
            summary["median_lifetime"] = float(sorted_l[n // 2])
    else:
        summary["median_lifetime"] = float("nan")

    # Round at which the Nth death occurred. None if fewer than N deaths.
    sorted_deaths = sorted(
        x for x in lifetimes if x <= num_rounds  # alive sentinel excluded
    )
    for N in (5, 10, 15):
        key = f"round_of_{N}th_death"
        if len(sorted_deaths) >= N:
            summary[key] = int(sorted_deaths[N - 1])
        else:
            summary[key] = None

    summary["survival_auc"] = sum(int(r["num_alive"]) for r in per_round_rows)
    denom = float(max(num_clients_total * num_rounds, 1))
    summary["participation_frac"] = summary["survival_auc"] / denom

    summary["cum_energy_j"] = (
        float(per_round_rows[-1]["cum_energy_j"]) if per_round_rows else 0.0
    )
    # best_acc / final_acc lifted from the last row that has an accuracy entry.
    best, final = 0.0, 0.0
    for row in per_round_rows:
        if row["mean_acc"] != "" and row["mean_acc"] is not None:
            v = float(row["mean_acc"])
            best = max(best, v)
            final = v
    summary["best_acc"] = best
    summary["final_acc"] = final

    return summary


def write_csv(
    output_dir: str,
    per_round_rows: list[dict],
    summary: dict,
    filename: str = "survival.csv",
) -> str:
    """Write per-round rows + a final '# summary: ...' line to CSV.

    The summary is written as commented-out key=value pairs in a trailing
    line so the file remains a valid CSV when read with pandas / csv.DictReader
    (those skip lines starting with '#').
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fieldnames = ["round", "num_alive", "cum_energy_j", "best_acc", "mean_acc"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in per_round_rows:
            w.writerow(row)
        # Trailing summary
        fh.write(
            "# summary: " + ", ".join(f"{k}={v}" for k, v in summary.items()) + "\n"
        )
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: full pipeline from runner state
# ─────────────────────────────────────────────────────────────────────────────


def emit_survival_artifacts(
    output_dir: str,
    rounds_metrics: list[dict],
    accuracy_history: Optional[list[float]],
    client_states: list,
    num_clients_total: int,
    num_rounds: int,
) -> dict:
    """One-shot: derive rows + lifetimes + summary, write the CSV, return
    the summary dict (caller can fold it into metrics.json).
    """
    rows = derive_per_round_rows(rounds_metrics, accuracy_history, num_clients_total)
    lifetimes = derive_lifetimes(client_states, num_rounds)
    summary = summary_from(rows, lifetimes, num_clients_total, num_rounds)
    path = write_csv(output_dir, rows, summary)
    summary["_csv_path"] = path
    summary["_lifetimes"] = lifetimes
    return summary
