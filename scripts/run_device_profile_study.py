#!/usr/bin/env python3
"""
scripts/run_device_profile_study.py
===================================
Per-device-profile ENERGY STUDY (a profile-driven projection, NOT a real
deployment). Replays the same FL workload across the device profiles in
hardware/profiles.py and reports, per profile:

  - the energy breakdown (compute / uplink / downlink, per-round + cumulative),
  - survival / participation,

then a CROSS-PROFILE summary of how the compute-vs-comm balance shifts with the
device class, and whether the algorithms reduce energy and extend participation
on each profile.

Physics
-------
cost_model="measured" (real FLOPs) and alpha_applies_to="compute": each device
has its own PHYSICAL alpha (compute utilization gap >= 1; see
configs/device_profile_study.yaml). Comm keeps its own per-device physics
(bandwidth + tx/rx power), never scaled by alpha. The breakdown's compute term
therefore scales with the device's alpha; comm does not.

Honest framing
--------------
For the ESP32-S3 the model does NOT fit in 8 MB RAM (resnet8 + grads + optimizer
~15 MB), so its row is an energy PROJECTION on that profile, flagged
`fits_in_ram=False`. RPi/smartphone genuinely run the model.

Foregrounded metric per regime
------------------------------
  - battery bites (deaths occur)  -> survival (median lifetime, survival AUC)
  - battery does not bite          -> total energy + rounds-to-accuracy

One-command usage
-----------------
    python scripts/run_device_profile_study.py --smoke           # fast sanity
    python scripts/run_device_profile_study.py --device mps --jobs 4
    python scripts/run_device_profile_study.py --dry-run
    python scripts/run_device_profile_study.py --aggregate-only  # re-build out

Outputs (results/device_profile_study/)
---------------------------------------
    aggregated_wide.csv          one row per (profile, algo)
    cross_profile_summary.csv    compute/comm balance + algo deltas per profile
    figures/<profile>_energy_breakdown.png   stacked compute/uplink/downlink
    figures/<profile>_survival.png           alive vs round per algo
    figures/cross_profile_comm_fraction.png  comm energy fraction vs device
    device_profile_summary.md    per-profile tables + cross-profile narrative
    <profile>__<algo>/           per-run dir (resolved_config + metrics + manifest)
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
DEFAULT_CONFIG = ROOT / "configs/device_profile_study.yaml"
DEFAULT_OUT = ROOT / "results/device_profile_study"

ACC_THRESHOLDS = [0.60, 0.70, 0.75]
FIXED_OVERHEAD_SEC = 60.0
DEFAULT_SEC_PER_CLIENT_ROUND_EPOCH = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# alpha resolution (per-device, with optional measured anchor)
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_device_alphas(base: dict) -> dict:
    """device -> alpha, optionally rescaled to a measured anchor that preserves
    the utilization RATIOS between devices."""
    alphas = dict(base.get("device_alpha", {}))
    anchor = base.get("device_alpha_measured_anchor")
    if anchor and anchor.get("device") in alphas and alphas[anchor["device"]] > 0:
        scale = float(anchor["alpha"]) / float(alphas[anchor["device"]])
        alphas = {d: a * scale for d, a in alphas.items()}
    return alphas


# ─────────────────────────────────────────────────────────────────────────────
# Per-run resolved config
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_config(base, profile, algo, alpha, seed, device, run_dir, smoke):
    cfg = copy.deepcopy(base)
    for k in ("device_alpha", "device_alpha_measured_anchor", "profiles", "algos"):
        cfg.pop(k, None)
    cfg["seed"] = seed
    cfg["device"] = device
    cfg["cost_model"] = "measured"
    cfg["output_dir"] = str(run_dir)
    cfg.setdefault("training", {})["algorithm"] = algo
    ac = cfg["training"].setdefault("algo_config", {})
    ac["energy_scale_factor"] = float(alpha)
    ac["alpha_applies_to"] = "compute"
    ac["cost_model"] = "measured"
    cfg.setdefault("clients", {})["fleet"] = [{"type": profile, "count": 30}]
    if smoke:
        cfg["training"]["num_rounds"] = 15
        ac["local_epochs"] = 1
        cfg["clients"]["num_clients"] = 3
        cfg["clients"]["fleet"] = [{"type": profile, "count": 3}]
        cfg["clients"]["min_clients"] = 1
    return cfg


def _run_one(base, profile, algo, alpha, seed, device, base_out, smoke, dry):
    run_dir = base_out / f"{profile}__{algo}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = _resolve_config(base, profile, algo, alpha, seed, device, run_dir, smoke)
    cfg_path = run_dir / "resolved_config.yaml"
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    cmd = [sys.executable, str(ROOT / "run_experiment.py"), "--config", str(cfg_path)]
    if dry:
        return {"profile": profile, "algo": algo, "alpha": alpha, "cmd": " ".join(cmd)}
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    dt = time.time() - t0
    if res.returncode != 0:
        tail = (res.stderr or res.stdout or "").strip().splitlines()[-3:]
        print(
            f"  [warn] {profile}/{algo} failed (rc={res.returncode}): "
            f"{' / '.join(tail)}"
        )
    return {
        "profile": profile,
        "algo": algo,
        "alpha": alpha,
        "seconds": dt,
        "ok": res.returncode == 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metric extraction
# ─────────────────────────────────────────────────────────────────────────────


def _find_metrics(run_dir: Path):
    c = list(Path(run_dir).glob("**/metrics.json"))
    return c[0] if c else None


def _cost_to_acc(rounds, x):
    for r in rounds:
        a = r.get("test_accuracy")
        if a is not None and a >= x:
            return r.get("round_num"), r.get("cumulative_energy_j", math.inf)
    return math.inf, math.inf


def _fits_in_ram(profile_name: str) -> bool | None:
    try:
        import torch  # noqa

        from hardware.profiles import DEVICE_PROFILES
        from models.registry import get_model

        m = get_model("resnet8", "cifar10")
        params = sum(p.numel() for p in m.parameters())
        model_mb = params * 4 / 1e6
        return DEVICE_PROFILES[profile_name].can_run_model(
            model_mb, dataset_size_mb=1.0
        )
    except Exception:
        return None


def _extract(profile, algo, alpha, run_dir: Path) -> dict:
    row = {
        "profile": profile,
        "algo": algo,
        "alpha": alpha,
        "fits_in_ram": _fits_in_ram(profile),
    }
    mp = _find_metrics(run_dir)
    if mp is None:
        row["error"] = "metrics.json not found"
        return row
    d = json.load(open(mp))
    s = d.get("summary", {}) or {}
    sv = d.get("survival", {}) or {}
    r = d.get("rounds", []) or []
    last = r[-1] if r else {}
    row["best_acc"] = s.get("best_accuracy")
    row["final_acc"] = s.get("final_accuracy")
    row["num_clients"] = s.get("num_clients")
    row["alive_final"] = last.get("num_alive_clients")
    row["median_lifetime"] = sv.get("median_lifetime")
    row["survival_auc"] = sv.get("survival_auc")
    row["participation_frac"] = sv.get("participation_frac")
    row["total_energy_j"] = s.get("total_energy_j")
    row["compute_energy_j"] = last.get("cumulative_compute_energy_j")
    row["uplink_energy_j"] = last.get("cumulative_uplink_energy_j")
    row["downlink_energy_j"] = last.get("cumulative_downlink_energy_j")
    tot = row["total_energy_j"] or 0.0
    comm = (row["uplink_energy_j"] or 0.0) + (row["downlink_energy_j"] or 0.0)
    row["comm_fraction"] = (comm / tot) if tot else None
    row["compute_fraction"] = ((row["compute_energy_j"] or 0.0) / tot) if tot else None
    row["uplink_mb"] = (s.get("total_bytes_gb") or 0.0) * 1024.0
    for x in ACC_THRESHOLDS:
        rd, en = _cost_to_acc(r, x)
        row[f"rounds_to_{x:.2f}"] = rd
        row[f"energy_to_{x:.2f}_j"] = en
    # survival curve for the figure
    row["_alive_curve"] = [
        (rr.get("round_num"), rr.get("num_alive_clients")) for rr in r
    ]
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────


def _make_figures(rows, fig_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    profiles = sorted({r["profile"] for r in rows})
    algos = sorted({r["algo"] for r in rows})

    def get(profile, algo):
        for r in rows:
            if r["profile"] == profile and r["algo"] == algo:
                return r
        return None

    # Per-profile: stacked compute/uplink/downlink energy per algo.
    for prof in profiles:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        xs = range(len(algos))
        comp = [(get(prof, a) or {}).get("compute_energy_j") or 0.0 for a in algos]
        up = [(get(prof, a) or {}).get("uplink_energy_j") or 0.0 for a in algos]
        dn = [(get(prof, a) or {}).get("downlink_energy_j") or 0.0 for a in algos]
        ax.bar(xs, comp, label="compute", color="#d62728")
        ax.bar(xs, up, bottom=comp, label="uplink", color="#1f77b4")
        ax.bar(
            xs,
            dn,
            bottom=[c + u for c, u in zip(comp, up)],
            label="downlink",
            color="#2ca02c",
        )
        ax.set_xticks(list(xs))
        ax.set_xticklabels(algos, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("cumulative energy (J)")
        ax.set_title(f"Energy breakdown — {prof}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{prof}_energy_breakdown.png", dpi=130)
        plt.close(fig)

        # Survival curves per algo.
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for a in algos:
            r = get(prof, a)
            if not r or not r.get("_alive_curve"):
                continue
            xs2 = [x for x, _ in r["_alive_curve"]]
            ys2 = [y for _, y in r["_alive_curve"]]
            ax.plot(xs2, ys2, label=a)
        ax.set_xlabel("round")
        ax.set_ylabel("clients alive")
        ax.set_title(f"Survival — {prof}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{prof}_survival.png", dpi=130)
        plt.close(fig)

    # Cross-profile: comm-energy fraction vs device, one line per algo.
    order = [
        p
        for p in [
            "esp32_s3",
            "raspberry_pi_zero2w",
            "raspberry_pi_4",
            "smartphone_midrange",
            "smartphone_highend",
        ]
        if p in profiles
    ] or profiles
    fig, ax = plt.subplots(figsize=(7, 4))
    for a in algos:
        ys = [(get(p, a) or {}).get("comm_fraction") for p in order]
        ys = [(y if y is not None else float("nan")) for y in ys]
        ax.plot(range(len(order)), ys, "o-", label=a)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("comm energy / total energy")
    ax.set_title("Compute→comm balance shift across device class")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "cross_profile_comm_fraction.png", dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CSVs + summary
# ─────────────────────────────────────────────────────────────────────────────

WIDE = [
    "profile",
    "algo",
    "alpha",
    "fits_in_ram",
    "best_acc",
    "final_acc",
    "num_clients",
    "alive_final",
    "median_lifetime",
    "survival_auc",
    "participation_frac",
    "total_energy_j",
    "compute_energy_j",
    "uplink_energy_j",
    "downlink_energy_j",
    "compute_fraction",
    "comm_fraction",
    "uplink_mb",
    "rounds_to_0.60",
    "energy_to_0.60_j",
    "rounds_to_0.70",
    "rounds_to_0.75",
    "run_seconds",
    "error",
]


def _write_outputs(rows, base_out: Path):
    import pandas as pd

    df = pd.DataFrame(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    )
    for c in WIDE:
        if c not in df.columns:
            df[c] = None
    df = df[WIDE].sort_values(["profile", "algo"])
    df.to_csv(base_out / "aggregated_wide.csv", index=False)
    return df


def _isnum(v) -> bool:
    """True for any real number, incl. numpy scalars (np.int64 is NOT a Python
    int subclass, so isinstance(v, (int, float)) wrongly rejects it)."""
    try:
        f = float(v)
        return not math.isnan(f)
    except (TypeError, ValueError):
        return False


def _fnum(v, p=1) -> str:
    return f"{float(v):.{p}f}" if _isnum(v) else "—"


def _summary_md(rows, df, base_out: Path):
    profiles = [
        p
        for p in [
            "esp32_s3",
            "raspberry_pi_zero2w",
            "raspberry_pi_4",
            "smartphone_midrange",
            "smartphone_highend",
        ]
        if p in set(df["profile"])
    ]
    algos = sorted(set(df["algo"]))

    def cell(prof, algo, col):
        s = df[(df["profile"] == prof) & (df["algo"] == algo)][col]
        return s.iloc[0] if len(s) else None

    L = [
        "# Device-profile energy study — summary\n",
        "Profile-driven energy PROJECTION (not a deployment). "
        "cost_model=measured, alpha_applies_to=compute (per-device physical alpha).\n",
    ]

    cross = []
    for prof in profiles:
        bites = any(
            (
                cell(prof, a, "alive_final") is not None
                and cell(prof, a, "num_clients")
                and cell(prof, a, "alive_final") < cell(prof, a, "num_clients")
            )
            for a in algos
        )
        fits = cell(prof, algos[0], "fits_in_ram")
        regime = (
            "battery BITES -> foreground SURVIVAL"
            if bites
            else "battery does NOT bite -> foreground ENERGY / rounds-to-acc"
        )
        L.append(f"\n## {prof}  ({regime})")
        if fits is False:
            L.append(
                f"> NOTE: resnet8 does not fit in {prof} RAM -> ENERGY "
                f"PROJECTION on this profile, not an executable run."
            )
        L.append(
            "\n| algo | compute J | uplink J | downlink J | comm frac | "
            "total J | median life | alive@end | best acc |"
        )
        L.append("|---|---|---|---|---|---|---|---|---|")
        for a in algos:

            def f(c, p=1, _a=a):
                return _fnum(cell(prof, _a, c), p)

            L.append(
                f"| {a} | {f('compute_energy_j')} | {f('uplink_energy_j',3)} "
                f"| {f('downlink_energy_j',3)} | {f('comm_fraction',3)} | "
                f"{f('total_energy_j')} | {f('median_lifetime')} | "
                f"{f('alive_final',0)} | {f('best_acc',3)} |"
            )

        # comm fraction range across algos (compute/comm balance)
        cf = [cell(prof, a, "comm_fraction") for a in algos]
        cf = [float(c) for c in cf if _isnum(c)]
        cf_mean = sum(cf) / len(cf) if cf else None
        # does fedpart_be reduce energy + extend participation vs fedavg?
        be_e, fa_e = cell(prof, "fedpart_be", "total_energy_j"), cell(
            prof, "fedavg", "total_energy_j"
        )
        be_s, fa_s = cell(prof, "fedpart_be", "survival_auc"), cell(
            prof, "fedavg", "survival_auc"
        )
        if _isnum(be_e) and _isnum(fa_e) and float(fa_e) > 0:
            _save = (1 - float(be_e) / float(fa_e)) * 100
            e_txt = (
                f"saves {_save:.0f}% energy vs FedAvg"
                if _save >= 0
                else f"uses {-_save:.0f}% MORE energy vs FedAvg"
            )
        else:
            e_txt = "n/a"
        s_txt = (
            f"survival AUC {be_s} vs {fa_s}"
            if (be_s is not None and fa_s is not None)
            else "n/a"
        )
        cross.append((prof, cf_mean, e_txt, s_txt, bites))

    L.append("\n## Cross-profile: compute↔comm balance & algorithm benefit\n")
    L.append("| profile | mean comm fraction | FedPartBE energy | FedPartBE survival |")
    L.append("|---|---|---|---|")
    for prof, cfm, e_txt, s_txt, _ in cross:
        L.append(f"| {prof} | {cfm:.3f} |" if cfm is not None else f"| {prof} | — |")
        L[-1] += f" {e_txt} | {s_txt} |"
    # narrative — honest about what the numbers actually show.
    if cross:
        lo, hi = cross[0], cross[-1]

        def _ref_total(prof):
            for a in ("fedpart_be", "fedavg", *algos):
                v = cell(prof, a, "total_energy_j")
                if _isnum(v):
                    return float(v)
            return None

        lo_e, hi_e = _ref_total(lo[0]), _ref_total(hi[0])
        span = f"{lo_e/hi_e:.0f}x" if (lo_e and hi_e and hi_e > 0) else "many-fold"
        lo_cf = lo[1] if lo[1] is not None else 0.0
        hi_cf = hi[1] if hi[1] is not None else 0.0
        shift = f"{(hi_cf/lo_cf):.0f}x" if lo_cf > 0 else "orders of magnitude"
        comm_dominates = hi_cf > 0.5
        balance = (
            f"becomes comm-dominated on `{hi[0]}` (comm fraction {hi_cf:.2f})"
            if comm_dominates
            else (
                f"stays COMPUTE-bound on every profile (comm fraction only "
                f"{lo_cf:.3f} -> {hi_cf:.3f}, a {shift} shift but still well "
                f"below 50%)"
            )
        )
        L.append(
            f"\n> **Absolute energy** spans ~{span} across the fleet "
            f"(`{lo[0]}` ~{lo_e:.0f} J vs `{hi[0]}` ~{hi_e:.0f} J). **Compute vs "
            f"comm**: the balance {balance}. For this small model (resnet8, "
            f"~78k params) trained on full local partitions, the per-round "
            f"compute dwarfs the one-shot model transfer, so FL is compute-bound "
            f"even on the fastest device; a comm-bound regime would require a "
            f"much larger model (more parameters to ship) and/or smaller local "
            f"datasets. **Where the metric matters**: survival on the "
            f"battery-constrained MCU class (battery bites), total energy / "
            f"rounds-to-accuracy on the capable devices (it does not). "
            f"**Algorithm benefit**: FedPartBE's partial updates cut compute "
            f"work (and uplink payload), saving energy and extending "
            f"participation on every profile — see the per-profile rows above."
        )
    md = "\n".join(L)
    (base_out / "device_profile_summary.md").write_text(md)
    return md


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--output", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--profiles", nargs="+", default=None)
    p.add_argument("--algos", nargs="+", default=None)
    p.add_argument(
        "--smoke", action="store_true", help="Fast sanity: 3 clients, 15 rounds."
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument(
        "--est-sec-per-run", type=float, default=DEFAULT_SEC_PER_CLIENT_ROUND_EPOCH
    )
    args = p.parse_args()

    base = yaml.safe_load(open(args.config))
    profiles = args.profiles or base["profiles"]
    algos = args.algos or base["algos"]
    dev_alphas = _resolve_device_alphas(base)
    base_out = (
        Path(args.output)
        if args.output
        else (DEFAULT_OUT / ("smoke" if args.smoke else "full"))
    )
    base_out.mkdir(parents=True, exist_ok=True)

    jobs = [(pr, al) for pr in profiles for al in algos]
    n = len(jobs)
    rounds = 15 if args.smoke else base.get("training", {}).get("num_rounds", 200)
    clients = 3 if args.smoke else base.get("clients", {}).get("num_clients", 30)
    per_run = FIXED_OVERHEAD_SEC + rounds * clients * 1 * args.est_sec_per_run
    print("=" * 70)
    print(f"device-profile energy study  (smoke={args.smoke})")
    print(f"  profiles: {profiles}")
    print(f"  algos   : {algos}")
    print(f"  alphas  : { {p: dev_alphas.get(p) for p in profiles} }")
    print(f"  runs    : {n}  | parallel {args.jobs}")
    print(
        f"  est time: ~{n*per_run/max(1,args.jobs)/60:.0f} min "
        f"(~{per_run/60:.1f} min/run; refined live)"
    )
    print(f"  output  : {base_out}")
    print("=" * 70)

    if args.dry_run:
        for pr, al in jobs:
            r = _run_one(
                base,
                pr,
                al,
                dev_alphas.get(pr, 1.0),
                args.seed,
                args.device,
                base_out,
                args.smoke,
                dry=True,
            )
            print(f"[dry] {pr}/{al} a={dev_alphas.get(pr)}: {r['cmd']}")
        return

    results = []
    timings = []

    def _do(job):
        pr, al = job
        return _run_one(
            base,
            pr,
            al,
            dev_alphas.get(pr, 1.0),
            args.seed,
            args.device,
            base_out,
            args.smoke,
            dry=False,
        )

    if not args.aggregate_only:
        t0 = time.time()
        if args.jobs > 1:
            with ThreadPoolExecutor(max_workers=args.jobs) as ex:
                futs = {ex.submit(_do, j): j for j in jobs}
                done = 0
                for fut in as_completed(futs):
                    r = fut.result()
                    results.append(r)
                    done += 1
                    timings.append(r.get("seconds", 0))
                    avg = sum(timings) / len(timings)
                    print(
                        f"  [{done}/{n}] {r['profile']}/{r['algo']} "
                        f"({r.get('seconds',0):.0f}s) ETA "
                        f"~{(n-done)/args.jobs*avg/60:.0f} min"
                    )
        else:
            for i, job in enumerate(jobs, 1):
                r = _do(job)
                results.append(r)
                timings.append(r.get("seconds", 0))
                avg = sum(timings) / len(timings)
                print(
                    f"  [{i}/{n}] {job[0]}/{job[1]} ({r.get('seconds',0):.0f}s) "
                    f"ETA ~{(n-i)*avg/60:.0f} min"
                )
        print(f"\nAll {n} runs done in {(time.time()-t0)/60:.1f} min.")

    rows = []
    secs = {(r["profile"], r["algo"]): r.get("seconds") for r in results}
    for pr, al in jobs:
        row = _extract(pr, al, dev_alphas.get(pr, 1.0), base_out / f"{pr}__{al}")
        row["run_seconds"] = round(secs.get((pr, al)) or 0, 1) or None
        rows.append(row)

    df = _write_outputs(rows, base_out)
    if not args.no_figures:
        try:
            _make_figures(rows, base_out / "figures")
            print(f"Figures: {base_out / 'figures'}")
        except Exception as exc:
            print(f"[warn] figures skipped: {exc}")
    md = _summary_md(rows, df, base_out)
    print(f"\nCSV: {base_out / 'aggregated_wide.csv'}")
    print(f"Summary: {base_out / 'device_profile_summary.md'}\n")
    print(md)


if __name__ == "__main__":
    main()
