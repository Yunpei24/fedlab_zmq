"""Generate the EMNIST/ByClass phase-1 campaign configs.

Protocol mirrors the DMD-CB conceptual framework: EMNIST/ByClass, 10 non-IID
clients (Dirichlet 0.1), CNN-GN, 500 train + 150 anchor per client, 150 rounds,
5 selected / 1 pre-training dropout, SGD lr 0.03, batch 64, 1 local epoch,
2 CE warmup rounds, fixed-zero margin reference.
"""

from pathlib import Path
import yaml

SEEDS = [91, 92, 93, 24]
ROOT = Path("configs/dmd_emnist")

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

# mean_mu=0.2027 is the MEAN EFFECTIVE INTENSITY the paper reports for USV-0.25
# (mu_M + 2*mu_V*E[D-a]_+ = 0.2027).  If this plain arm reproduces USV's
# variance reduction, the upper-semivariance term is just a slightly larger mu_M.
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


def build(arm: str, seed: int) -> Path:
    algorithm, overrides = ARMS[arm]
    algo = {**BASE_ALGO, **overrides}
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
        "output_dir": f"./results/emnist_phase1/{arm}",
        "device": "cpu",
        "cost_model": "phi",
        "layer_mismatch": False,
        # partition_seed == seed keeps partition and init coupled, exactly as in
        # the paper.  Decoupling them is a phase-2 change, deliberately not made
        # here so this campaign stays comparable to the published tables.
        "data": {"dataset": "emnist", "model": "cnn_gn", "partition": "dirichlet",
                 "partition_seed": seed, "alpha": 0.1, "data_root": "./data"},
        "model": {"architecture": "cnn_gn"},
        "training": {"num_rounds": 150, "algorithm": algorithm, "algo_config": algo},
        "clients": {"num_clients": 10, "sample_fraction": 0.5,
                    "sampling_strategy": "round_robin", "min_clients": 5,
                    "dropout_rate": 0.2,
                    "fleet": [{"type": "raspberry_pi_4", "count": 10}]},
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / f"{arm}_s{seed}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


if __name__ == "__main__":
    paths = [build(arm, seed) for arm in ARMS for seed in SEEDS]
    print("\n".join(str(p) for p in paths))
    print(f"\n{len(paths)} configs ({len(ARMS)} bras x {len(SEEDS)} seeds)")
