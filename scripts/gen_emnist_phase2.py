"""Generate the phase-2 EMNIST campaign: bounded margin, in two steps.

Phase 2 changes three things relative to phase 1, each fixing a specific defect
documented in docs/dmd_audit_option_b.md.

1. Bounded margin space.  The logit margin is unbounded and not scale
   invariant: contracting every logit by 4 divides the quadratic deficit by 16
   with every decision unchanged, and weight decay pushes in exactly that
   direction.  It also makes deficits incomparable across clients, which is
   fatal because the USV/CVaR threshold compares them, and it leaves the DP
   report channel with a sensitivity larger than its own signal.  Measured
   signal/sensitivity for the report scalar: 0.27 for logits clipped at M=10,
   8.81 in probability space, 4.48 in normalized space.

2. Intensity recalibrated on SEPARATE seeds.  Changing the margin space changes
   the scale of the deficit by roughly two orders of magnitude, so mu_M cannot
   carry over.  More importantly, the published USV-0.25 ratio was chosen after
   looking at the same four seeds used to report it; here the intensity is
   selected on calibration seeds and only then measured on disjoint test seeds.

3. Partition and initialisation seeds decoupled.  In phase 1 a single seed
   drives the Dirichlet partition, the initialisation and the participation
   schedule at once, so their variance contributions cannot be separated.  With
   ten clients at alpha=0.1 the partition draw is expected to dominate; a
   partitions x inits grid measures that instead of assuming it.

Step 2a (``--step calibrate``) sweeps mean_mu over CALIBRATION_PARTITIONS at
CALIBRATE_ROUNDS rounds.  Step 2b (``--step test``) runs the full protocol on
the disjoint TEST_PARTITIONS with ``--mean-mu`` fixed to the winner.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path("configs/dmd_emnist_p2")

# Disjoint by construction: nothing selected on the calibration grid is ever
# also reported from it.
CALIBRATION_PARTITIONS = [201, 202]
CALIBRATION_INITS = [1]
TEST_PARTITIONS = [301, 302, 303]
TEST_INITS = [1, 2]

CALIBRATE_ROUNDS = 60
TEST_ROUNDS = 150

# Probability-space deficits run orders of magnitude below logit-space ones, so
# the phase-1 value of 0.1875 would make the penalty negligible next to a
# cross-entropy of ~4.  Measured on EMNIST at the start of training: ~1e-5
# rising to ~2e-3 over the first four rounds (logit space: ~30).  The deficit
# peaks once the model is confident enough to be confidently wrong on some
# classes, then falls.  This span brackets "negligible" to "dominant"; step 2a
# picks from it on the calibration partitions rather than by guessing a ratio.
MEAN_MU_SWEEP = [1.0, 3.0, 10.0, 30.0, 100.0]

BASE_ALGO = {
    "lr": 0.03, "momentum": 0.9, "weight_decay": 1.0e-4,
    "local_epochs": 1, "batch_size": 64, "num_classes": 62,
    "class_weight_mode": "uniform", "reference_mode": "fixed_zero",
    "reference_method": "median", "min_reference_clients": 2,
    "context_policy": "one_round_stale", "warmup_rounds": 2,
    "anchor_fraction": 0.23, "anchor_batch_size": 256,
    "max_train_samples": 500, "max_anchor_samples": 150, "min_train_samples": 50,
    "require_anchor_dataloader": True,
    "client_metrics_every": 5, "global_eval_every": 5,
    "client_eval_max_batches": 12,
    "device": "cpu", "device_profile": None,
}

PLAIN_KEYS = {
    "lr", "momentum", "weight_decay", "local_epochs", "batch_size", "num_classes",
    "client_metrics_every", "global_eval_every", "client_eval_max_batches",
    "max_train_samples", "min_train_samples", "anchor_fraction",
    "anchor_batch_size", "max_anchor_samples", "device", "device_profile",
}

# mean_mu is filled in from the calibration result; None means "sweep me".
ARMS = {
    "fedavg":      ("fedavg",   {}),
    # The control the study was missing: class balancing without any margin.
    "cbce":        ("cb_ce",    {}),
    "dmdcb_prob":  ("dmd_mean", {"margin_space": "probability", "mean_mu": None}),
    # Satisficing target: clear the boundary by 0.3 of probability mass, then
    # stop.  Only meaningful in a bounded space.
    "dmdcb_ptau":  ("dmd_mean", {"margin_space": "probability",
                                 "margin_target": 0.3, "mean_mu": None}),
    "dmdcb_norm":  ("dmd_mean", {"margin_space": "normalized", "mean_mu": None}),
    # Option B carried into the bounded space.
    "tail_dual_p": ("dmd_tail", {"margin_space": "probability", "mean_mu": None,
                                 "cvar_tail_mass": 0.25, "cvar_eta_mode": "dual",
                                 "cvar_eta_lr": 0.1}),
}


def build(arm, partition_seed, init_seed, rounds, mean_mu, out_root, suffix=""):
    algorithm, overrides = ARMS[arm]
    algo = {**BASE_ALGO, **{k: v for k, v in overrides.items() if v is not None}}
    if overrides.get("mean_mu", "absent") is None:
        if mean_mu is None:
            raise ValueError(f"arm {arm} needs an explicit mean_mu")
        algo["mean_mu"] = mean_mu
        # Keep the phase-1 dispersion ratio mu_V/mu_M = 0.25 so the tail layer
        # is compared at the same relative strength as the published USV-0.25.
        if algorithm == "dmd_tail":
            algo["dispersion_mu"] = 0.25 * mean_mu
    if algorithm in {"fedavg", "cb_ce"}:
        algo = {k: v for k, v in algo.items() if k in PLAIN_KEYS}
    name = f"{arm}{suffix}_p{partition_seed}_i{init_seed}"
    cfg = {
        "seed": init_seed,
        "output_dir": f"./{out_root}/{arm}{suffix}",
        "device": "cpu",
        "cost_model": "phi",
        "layer_mismatch": False,
        # partition_seed is independent of seed: one drives the shards, the
        # other the initialisation and the participation draw.
        "data": {"dataset": "emnist", "model": "cnn_gn", "partition": "dirichlet",
                 "partition_seed": partition_seed, "alpha": 0.1,
                 "data_root": "./data"},
        "model": {"architecture": "cnn_gn"},
        "training": {"num_rounds": rounds, "algorithm": algorithm,
                     "algo_config": algo},
        "clients": {"num_clients": 10, "sample_fraction": 0.5,
                    "sampling_strategy": "round_robin", "min_clients": 5,
                    "dropout_rate": 0.2,
                    "fleet": [{"type": "raspberry_pi_4", "count": 10}]},
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["calibrate", "test"], required=True)
    parser.add_argument("--mean-mu", type=float,
                        help="winner of step 2a; required for --step test")
    args = parser.parse_args()

    paths = []
    if args.step == "calibrate":
        # Only the arms whose intensity is unknown need calibrating.
        for arm in ("dmdcb_prob", "dmdcb_ptau", "dmdcb_norm"):
            for mu in MEAN_MU_SWEEP:
                for partition in CALIBRATION_PARTITIONS:
                    for init in CALIBRATION_INITS:
                        paths.append(build(
                            arm, partition, init, CALIBRATE_ROUNDS, mu,
                            "results/emnist_phase2_cal", suffix=f"_mu{mu:g}",
                        ))
    else:
        if args.mean_mu is None:
            parser.error("--step test requires --mean-mu from step 2a")
        for arm in ARMS:
            for partition in TEST_PARTITIONS:
                for init in TEST_INITS:
                    paths.append(build(arm, partition, init, TEST_ROUNDS,
                                       args.mean_mu, "results/emnist_phase2"))
    print(f"{len(paths)} configs ecrites dans {ROOT}/")
    for path in paths[:4]:
        print(f"  {path}")
    if len(paths) > 4:
        print(f"  ... (+{len(paths) - 4})")


if __name__ == "__main__":
    main()
