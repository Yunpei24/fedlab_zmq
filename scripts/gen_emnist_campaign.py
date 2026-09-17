"""Generate the phase-1 DMD campaign configs (EMNIST/ByClass by default).

Protocol mirrors the DMD-CB conceptual framework: 10 non-IID clients
(Dirichlet 0.1), CNN-GN, 500 train + 150 anchor per client, 150 rounds,
5 selected / 1 pre-training dropout, SGD lr 0.03, batch 64, 1 local epoch,
2 CE warmup rounds, fixed-zero margin reference.

    python3 scripts/gen_emnist_campaign.py        # EMNIST, as published
    python3 scripts/gen_emnist_campaign.py --dataset fashionmnist --device mps
    python3 scripts/gen_emnist_campaign.py --dataset fashionmnist --device mps \\
        --calibrate-dmdcb-hi results/fmnist_phase1/usv025

Off EMNIST, the matched-intensity control ``dmdcb_hi`` is only generated once
its intensity has been measured on that dataset (see ``ARMS``).
"""

import argparse
import json
from pathlib import Path

import yaml

SEEDS = [91, 92, 93, 24]
# dataset -> (tag naming the config and results directories, number of classes)
DATASETS = {
    "emnist": ("emnist", 62),
    "fashionmnist": ("fmnist", 10),
    "mnist": ("mnist", 10),
}


def base_algo(num_classes: int, device: str) -> dict:
    return {
        "lr": 0.03, "momentum": 0.9, "weight_decay": 1.0e-4,
        "local_epochs": 1, "batch_size": 64, "num_classes": num_classes,
        "class_weight_mode": "uniform", "reference_mode": "fixed_zero",
        "reference_method": "median", "min_reference_clients": 2,
        "context_policy": "one_round_stale", "warmup_rounds": 2,
        "anchor_fraction": 0.23, "anchor_batch_size": 256,
        "max_train_samples": 500, "max_anchor_samples": 150, "min_train_samples": 50,
        "require_anchor_dataloader": True,
        "client_metrics_every": 5, "global_eval_every": 5,
        "client_eval_max_batches": 12,
        "device": device, "device_profile": None,
    }


# mean_mu=0.2027 is the MEAN EFFECTIVE INTENSITY the paper reports for USV-0.25
# on EMNIST (mu_M + 2*mu_V*E[D-a]_+ = 0.2027).  If this plain arm reproduces
# USV's variance reduction, the upper-semivariance term is just a slightly
# larger mu_M.  E[D-a]_+ is in deficit units, so the number is only valid on
# EMNIST; on another dataset it is re-measured from that dataset's own usv025
# runs (--calibrate-dmdcb-hi) or the control compares against nothing.
ARMS = {
    "fedavg":    ("fedavg",   {}),
    "dmdcb":     ("dmd_mean", {"mean_mu": 0.1875}),
    "dmdcb_hi":  ("dmd_mean", {"mean_mu": 0.2027}),
    "usv025":    ("dmd_usv",  {"mean_mu": 0.1875, "dispersion_mu": 0.046875}),
    "tail_emp":  ("dmd_tail", {"mean_mu": 0.1875, "dispersion_mu": 0.046875,
                               "cvar_tail_mass": 0.25, "cvar_eta_mode": "empirical"}),
    "tail_dual": ("dmd_tail", {"mean_mu": 0.1875, "dispersion_mu": 0.046875,
                               "cvar_tail_mass": 0.25, "cvar_eta_mode": "dual",
                               "cvar_eta_lr": 0.1}),
}


def config_root(dataset: str) -> Path:
    return Path(f"configs/dmd_{DATASETS[dataset][0]}")


def build(
    arm: str,
    seed: int,
    *,
    dataset: str = "emnist",
    device: str = "cpu",
    dmdcb_hi_mu: float | None = None,
) -> Path:
    tag, num_classes = DATASETS[dataset]
    algorithm, overrides = ARMS[arm]
    if arm == "dmdcb_hi" and dmdcb_hi_mu is not None:
        overrides = {**overrides, "mean_mu": dmdcb_hi_mu}
    algo = {**base_algo(num_classes, device), **overrides}
    if algorithm == "fedavg":
        algo = {k: v for k, v in algo.items()
                if k in {"lr", "momentum", "weight_decay", "local_epochs",
                         "batch_size", "num_classes", "client_metrics_every",
                         "global_eval_every", "client_eval_max_batches",
                         "max_train_samples", "min_train_samples",
                         "anchor_fraction", "anchor_batch_size",
                         "max_anchor_samples", "device", "device_profile"}}
    cfg = {
        "seed": seed,
        "output_dir": f"./results/{tag}_phase1/{arm}",
        "device": device,
        "cost_model": "phi",
        "layer_mismatch": False,
        # partition_seed == seed keeps partition and init coupled, exactly as in
        # the paper.  Decoupling them is a phase-2 change, deliberately not made
        # here so this campaign stays comparable to the published tables.
        "data": {"dataset": dataset, "model": "cnn_gn", "partition": "dirichlet",
                 "partition_seed": seed, "alpha": 0.1, "data_root": "./data"},
        "model": {"architecture": "cnn_gn"},
        "training": {"num_rounds": 150, "algorithm": algorithm, "algo_config": algo},
        "clients": {"num_clients": 10, "sample_fraction": 0.5,
                    "sampling_strategy": "round_robin", "min_clients": 5,
                    "dropout_rate": 0.2,
                    "fleet": [{"type": "raspberry_pi_4", "count": 10}]},
    }
    root = config_root(dataset)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{arm}_s{seed}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def measured_effective_intensity(results_dir: Path, min_runs: int = len(SEEDS)) -> dict:
    """Effective DMD intensity of finished runs: per run, then over runs.

    Per run it is the mean of ``avg_local_dmd_effective_mu`` over the rounds
    where the penalty was applied; the campaign value is the mean over seeds.
    A diverged run leaves no metrics.json, so ``min_runs`` below the seed
    count is how a campaign calibrates on the seeds that converged.
    """

    per_run = {}
    for path in sorted(results_dir.glob("*/metrics.json")):
        rounds = json.loads(path.read_text())["rounds"]
        values = [r["avg_local_dmd_effective_mu"] for r in rounds
                  if r.get("avg_local_dmd_effective_mu") is not None]
        if not values:
            raise SystemExit(f"{path}: no avg_local_dmd_effective_mu logged; the "
                             "run predates the intensity probe, rerun it")
        per_run[path.parent.name] = sum(values) / len(values)
    if len(per_run) < min_runs:
        raise SystemExit(f"{results_dir}: need {min_runs} finished runs to "
                         f"calibrate, found {len(per_run)}")
    return {"mean": sum(per_run.values()) / len(per_run), "per_run": per_run}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="emnist")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument(
        "--calibrate-dmdcb-hi", type=Path, metavar="USV_RESULTS",
        help="generate only dmdcb_hi, with mean_mu set to the effective intensity "
             "measured by the finished usv025 runs in this directory",
    )
    parser.add_argument(
        "--min-runs", type=int, default=len(SEEDS),
        help="finished usv025 runs the calibration requires (default: every seed)",
    )
    args = parser.parse_args()

    arms = list(args.arms)
    dmdcb_hi_mu = None
    if args.calibrate_dmdcb_hi is not None:
        measured = measured_effective_intensity(args.calibrate_dmdcb_hi, args.min_runs)
        dmdcb_hi_mu = round(measured["mean"], 4)
        arms = ["dmdcb_hi"]
        root = config_root(args.dataset)
        root.mkdir(parents=True, exist_ok=True)
        record = {
            "definition": "mean over seeds of the per-run mean, over rounds with "
                          "the penalty active, of d(penalty)/dD (mu_M + 2*mu_V*[D-a]_+)",
            "source": str(args.calibrate_dmdcb_hi),
            "measured_mean": measured["mean"],
            "runs_used": len(measured["per_run"]),
            "seeds_planned": len(SEEDS),
            "per_run": measured["per_run"],
            "dmdcb_hi_mean_mu": dmdcb_hi_mu,
        }
        (root / "dmdcb_hi_calibration.json").write_text(json.dumps(record, indent=2))
        print(f"dmdcb_hi mean_mu = {dmdcb_hi_mu} (measured on {args.calibrate_dmdcb_hi})")
    elif args.dataset != "emnist" and "dmdcb_hi" in arms:
        arms.remove("dmdcb_hi")
        print("dmdcb_hi skipped: its intensity has to be measured on this dataset "
              "first (--calibrate-dmdcb-hi)")

    paths = [
        build(arm, seed, dataset=args.dataset, device=args.device,
              dmdcb_hi_mu=dmdcb_hi_mu)
        for arm in arms for seed in SEEDS
    ]
    print("\n".join(str(p) for p in paths))
    print(f"\n{len(paths)} configs ({len(arms)} bras x {len(SEEDS)} seeds)")


if __name__ == "__main__":
    main()
