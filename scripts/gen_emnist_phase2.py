"""Generate the phase-2 DMD campaigns: bounded margin, in two steps.

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

Two experiments share that protocol (``--experiment``):

- ``bounded_margin``: the margin penalty on plain cross-entropy, against FedAvg
  and the class-balanced CE control (CB-CE).
- ``cbce_margin``: the same penalty on top of CB-CE, against CB-CE itself run
  through the same client at mean_mu=0, so each contrast isolates the margin.
  On Fashion-MNIST, bounded_margin found CB-CE ahead of every margin arm, which
  leaves this as the one setting where the margin can still add something.

Step 2a (``--step calibrate``) sweeps mean_mu over the calibration partitions at
CALIBRATE_ROUNDS rounds.  ``--step select`` fixes each arm's intensity from that
sweep by the criterion documented in ``select``, stated before any calibration
result existed.  Step 2b (``--step test``) runs the full protocol on the
disjoint test partitions with those intensities, or with one shared --mean-mu.

    python3 scripts/gen_emnist_phase2.py --step calibrate          # EMNIST
    python3 scripts/gen_emnist_phase2.py --dataset fashionmnist --device mps \\
        --feasible-partitions --step calibrate
    python3 scripts/gen_emnist_phase2.py --dataset fashionmnist \\
        --feasible-partitions --step select
    python3 scripts/gen_emnist_phase2.py --dataset fashionmnist --device mps \\
        --feasible-partitions --step test \\
        --selection configs/dmd_fmnist_p2/phase2a_selection.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

import yaml

from analyze_phase1 import METRICS, late_mean
from gen_emnist_campaign import DATASETS, base_algo

# Disjoint by construction: nothing selected on the calibration grid is ever
# also reported from it.
CALIBRATION_PARTITIONS = [201, 202]
CALIBRATION_INITS = [1]
TEST_PARTITIONS = [301, 302, 303]
TEST_INITS = [1, 2]

CALIBRATE_ROUNDS = 60
TEST_ROUNDS = 150
NUM_CLIENTS = 10
ALPHA = 0.1

# Probability-space deficits run orders of magnitude below logit-space ones, so
# the phase-1 value of 0.1875 would make the penalty negligible next to a
# cross-entropy of ~4.  Measured on EMNIST at the start of training: ~1e-5
# rising to ~2e-3 over the first four rounds (logit space: ~30).  The deficit
# peaks once the model is confident enough to be confidently wrong on some
# classes, then falls.  This span brackets "negligible" to "dominant"; step 2a
# picks from it on the calibration partitions rather than by guessing a ratio.
# Fashion-MNIST, rounds 3-10: probability deficits ~2e-3..7e-2, so the penalty
# is ~1% of cross-entropy at mu=1 and up to ~40% at mu=100; normalized deficits
# ~0.08..0.26 already weigh 20-50% at mu=10.  Same span, still bracketing.
MEAN_MU_SWEEP = [1.0, 3.0, 10.0, 30.0, 100.0]

PLAIN_KEYS = {
    "lr", "momentum", "weight_decay", "local_epochs", "batch_size", "num_classes",
    "client_metrics_every", "global_eval_every", "client_eval_max_batches",
    "max_train_samples", "min_train_samples", "anchor_fraction",
    "anchor_batch_size", "max_anchor_samples", "device", "device_profile",
    "round_client_seeding",
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
CALIBRATED_ARMS = ("dmdcb_prob", "dmdcb_ptau", "dmdcb_norm")
# The calibrated arm whose intensity each test arm uses.  tail_dual_p shares
# dmdcb_prob's margin space and deficit scale, so it inherits that intensity.
MU_SOURCE = {"dmdcb_prob": "dmdcb_prob", "dmdcb_ptau": "dmdcb_ptau",
             "dmdcb_norm": "dmdcb_norm", "tail_dual_p": "dmdcb_prob"}

# Every arm trains on the CB-CE objective through the DMD client; the control
# is that client at mean_mu=0, which is cb_ce update for update
# (tests/dmd/test_contracts_and_adapter.py), so a contrast is the penalty alone.
CB = {"ce_class_weighting": "inverse_frequency"}
CBCE_MARGIN_ARMS = {
    "cbce_ref":   ("dmd_mean", {**CB, "margin_space": "probability", "mean_mu": 0.0}),
    "cbdmd_prob": ("dmd_mean", {**CB, "margin_space": "probability", "mean_mu": None}),
    "cbdmd_ptau": ("dmd_mean", {**CB, "margin_space": "probability",
                                "margin_target": 0.3, "mean_mu": None}),
    "cbdmd_norm": ("dmd_mean", {**CB, "margin_space": "normalized", "mean_mu": None}),
}
# The bounded_margin sweep put the optimum at 1-10 on plain CE, collapse at
# 30-100, and normalized on the grid's lower edge.  One step down: 100 killed
# every arm, 0.3 brackets normalized from below.  That sweep ran on the same
# calibration partitions, so the shift reads nothing from the test partitions.
CBCE_MARGIN_SWEEP = [0.3, 1.0, 3.0, 10.0, 30.0]

# Monte Carlo confirmation of the one margin that stayed ahead of CB-CE
# (normalized, mean_mu=0.3 as selected by cbce_margin's calibration), on more
# partitions and with the variance sources pinned down:
# - client-centric Dirichlet (datasets/partitioner.py): each client draws its
#   own label profile pi_i ~ Dir(alpha), every client gets the same size, so no
#   partition seed is ever infeasible;
# - round_client_seeding: each local update's torch draws (flips, dropout) are
#   fixed by (seed, round, client), so both arms face identical draws;
# - the two arms share every setting but mean_mu, margin space included.
CRN = {"round_client_seeding": True}
# cbdmd_norm_hi tests the margin where it actually weighs.  At the selected 0.3
# the penalty is 0.2% of the CE late in training, so a null there says little
# about the margin itself.  "hi" is the strongest intensity of the calibration
# sweep that still kept >= 80% of the control's Worst-20 before the collapse:
# 3 on Fashion-MNIST (84%, penalty ~1.8% of CE at rounds 41-60; 10 fell to 46%)
# and 10 on EMNIST (83%, ~1.2%; 30 fell to 6%).  Same rule, comparable weight.
# Added after the first 32 runs; resuming skips them and CRN keeps every new run
# paired with its control.
CBCE_MARGIN_MC_ARMS = {
    "cbce_ref":   ("dmd_mean", {**CB, **CRN, "margin_space": "normalized",
                                "mean_mu": 0.0}),
    "cbdmd_norm": ("dmd_mean", {**CB, **CRN, "margin_space": "normalized",
                                "mean_mu": 0.3}),
    "cbdmd_norm_hi": ("dmd_mean", {**CB, **CRN, "margin_space": "normalized",
                                   "mean_mu": 3.0}),
}
# The same Monte Carlo design with the intensity selected on the dataset at
# hand, for any dataset but Fashion-MNIST, where 0.3 was selected: intensities do
# not carry over (phase 1 measured USV's effective intensity at 0.2193 on
# Fashion-MNIST against 0.2027 reported on EMNIST).  The sweep runs on
# calibration partitions from the same client-centric Dirichlet, with the
# control alongside.
NORM_MARGIN_MC_ARMS = {
    "cbce_ref":   CBCE_MARGIN_MC_ARMS["cbce_ref"],
    "cbdmd_norm": ("dmd_mean", {**CB, **CRN, "margin_space": "normalized",
                                "mean_mu": None}),
    # Read off EMNIST's calibration table; see the rule above CBCE_MARGIN_MC_ARMS.
    "cbdmd_norm_hi": ("dmd_mean", {**CB, **CRN, "margin_space": "normalized",
                                   "mean_mu": 10.0}),
}

# With the margin closed, what CB-CE is worth against the established label-skew
# losses of algorithms/label_skew.py.  Every arm runs through the DMD client at
# mean_mu=0 in the Monte Carlo design, so all share the anchor pass and the
# random draws of the cbce_ref control, which already ran on these partitions and
# inits in cbce_margin_mc / norm_margin_mc with this exact config and is read
# from there rather than rerun.  ce_ref is plain CE, i.e. FedAvg.  The loss
# hyperparameters are calibrated over the full 150 rounds: for the margin a
# 60-round calibration misjudged long-run effects in both directions.
LS = {**CRN, "margin_space": "normalized", "mean_mu": 0.0}
LABEL_SKEW_ARMS = {
    "ce_ref":   ("dmd_mean", {**LS, "base_loss": "ce"}),
    "cbce_ref": CBCE_MARGIN_MC_ARMS["cbce_ref"],
    "cbloss":   ("dmd_mean", {**LS, "base_loss": "effective_number",
                              "label_skew_beta": None}),
    "bsm":      ("dmd_mean", {**LS, "base_loss": "balanced_softmax",
                              "label_skew_tau": None}),
    "fedlc":    ("dmd_mean", {**LS, "base_loss": "fedlc", "label_skew_tau": None}),
}
# arm -> (config key, name tag, values).  At 500 local examples beta=0.9 weighs
# classes almost uniformly and 0.999 almost like CB-CE; tau=1 is Balanced
# Softmax proper; FedLC's shift spans ~0.8*tau between a client's commonest and
# rarest observed classes.
LABEL_SKEW_SWEEPS = {
    "cbloss": ("label_skew_beta", "b", [0.9, 0.99, 0.999]),
    "bsm":    ("label_skew_tau", "t", [0.5, 1.0, 2.0]),
    "fedlc":  ("label_skew_tau", "t", [0.3, 1.0, 3.0]),
}


@dataclass(frozen=True)
class Experiment:
    configs: str                # configs/dmd_<tag>_<configs>
    results: str                # results/<tag>_<results>, and <..>_cal for 2a
    arms: dict
    calibrated: tuple
    sweep: list
    mu_source: dict
    # Arm with a fixed mean_mu also run on the calibration partitions, so the
    # sweep can be read against the control and not only ranked within itself.
    reference: str | None = None
    partition: str = "dirichlet"
    test_partitions: tuple = tuple(TEST_PARTITIONS)
    test_inits: tuple = tuple(TEST_INITS)
    # Calibrated arms swept over something other than mean_mu, as
    # arm -> (config key, name tag, values); the others sweep mean_mu over sweep.
    sweeps: dict = field(default_factory=dict)
    extra_references: tuple = ()
    calibration_rounds: int = CALIBRATE_ROUNDS
    calibration_window: int = 20        # selection reads the last this-many rounds
    test_arms: tuple | None = None      # None: every arm

    def sweep_of(self, arm: str) -> tuple[str, str, list]:
        return self.sweeps.get(arm, ("mean_mu", "mu", self.sweep))

    def reference_arms(self) -> tuple:
        return ((self.reference,) if self.reference else ()) + self.extra_references


EXPERIMENTS = {
    "bounded_margin": Experiment("p2", "phase2", ARMS, CALIBRATED_ARMS,
                                 MEAN_MU_SWEEP, MU_SOURCE),
    "cbce_margin": Experiment("cbm", "cbmargin", CBCE_MARGIN_ARMS,
                              ("cbdmd_prob", "cbdmd_ptau", "cbdmd_norm"),
                              CBCE_MARGIN_SWEEP,
                              {arm: arm for arm in ("cbdmd_prob", "cbdmd_ptau",
                                                    "cbdmd_norm")},
                              reference="cbce_ref"),
    "cbce_margin_mc": Experiment("cbmmc", "cbmargin_mc", CBCE_MARGIN_MC_ARMS,
                                 (), [], {},
                                 partition="client_dirichlet_balanced",
                                 test_partitions=tuple(range(301, 309))),
    "norm_margin_mc": Experiment("nmmc", "normmargin_mc", NORM_MARGIN_MC_ARMS,
                                 ("cbdmd_norm",), CBCE_MARGIN_SWEEP,
                                 {"cbdmd_norm": "cbdmd_norm"},
                                 reference="cbce_ref",
                                 partition="client_dirichlet_balanced",
                                 test_partitions=tuple(range(301, 309))),
    "label_skew_mc": Experiment("lsmc", "labelskew_mc", LABEL_SKEW_ARMS,
                                tuple(LABEL_SKEW_SWEEPS), [],
                                {arm: arm for arm in LABEL_SKEW_SWEEPS},
                                reference="ce_ref", extra_references=("cbce_ref",),
                                partition="client_dirichlet_balanced",
                                test_partitions=tuple(range(301, 309)),
                                sweeps=LABEL_SKEW_SWEEPS,
                                calibration_rounds=TEST_ROUNDS, calibration_window=30,
                                test_arms=("ce_ref", "cbloss", "bsm", "fedlc")),
}


def config_root(dataset: str, experiment: Experiment = EXPERIMENTS["bounded_margin"]) -> Path:
    return Path(f"configs/dmd_{DATASETS[dataset][0]}_{experiment.configs}")


def results_root(dataset: str, step: str,
                 experiment: Experiment = EXPERIMENTS["bounded_margin"]) -> str:
    base = f"results/{DATASETS[dataset][0]}_{experiment.results}"
    return f"{base}_cal" if step == "calibrate" else base


def feasible_partitions(dataset: str, first: int, count: int,
                        partition: str = "dirichlet") -> list[int]:
    """First ``count`` partition seeds from ``first`` up whose every client
    holds the nominal split, max_train_samples + max_anchor_samples examples.

    A Dirichlet(0.1) draw over ten clients can leave a client nearly empty on a
    small dataset.  On Fashion-MNIST seed 202 leaves one client 17 examples,
    which the anchor split refuses, and seed 302 leaves one 148, which runs but
    on a fraction of the protocol's 500 + 150.  The rule reads shard sizes only,
    through the loader the runner itself uses, and never a training outcome.
    """

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from datasets.registry import get_dataloader

    algo = base_algo(DATASETS[dataset][1], "cpu")
    needed = algo["max_train_samples"] + algo["max_anchor_samples"]
    seeds, seed = [], first
    while len(seeds) < count:
        smallest = min(
            len(get_dataloader(
                dataset_name=dataset, split="train", partition=partition,
                client_id=cid, num_clients=NUM_CLIENTS, alpha=ALPHA,
                batch_size=algo["batch_size"], seed=seed, data_root="./data",
                matched_dirichlet=algo["client_metrics_every"] > 0,
            ).dataset)
            for cid in range(NUM_CLIENTS)
        )
        if smallest >= needed:
            seeds.append(seed)
        seed += 1
    return seeds


def partition_seeds(dataset: str, feasible: bool,
                    experiment: Experiment = EXPERIMENTS["bounded_margin"],
                    ) -> tuple[list[int], list[int]]:
    """(calibration, test) partition seeds, filtered by feasibility on request."""

    if not feasible:
        return CALIBRATION_PARTITIONS, list(experiment.test_partitions)
    calibration = feasible_partitions(
        dataset, CALIBRATION_PARTITIONS[0], len(CALIBRATION_PARTITIONS),
        experiment.partition)
    test = feasible_partitions(dataset, experiment.test_partitions[0],
                               len(experiment.test_partitions), experiment.partition)
    if max(calibration) >= min(test):
        raise SystemExit("calibration partition seeds ran into the test range")
    return calibration, test


def build(arm, partition_seed, init_seed, rounds, params, out_root, *,
          dataset="emnist", device="cpu", suffix="",
          experiment: Experiment = EXPERIMENTS["bounded_margin"]):
    """Write one config; ``params`` fills the arm's overrides left at None."""

    num_classes = DATASETS[dataset][1]
    algorithm, overrides = experiment.arms[arm]
    algo = {**base_algo(num_classes, device),
            **{k: v for k, v in overrides.items() if v is not None}}
    for key in [k for k, v in overrides.items() if v is None]:
        if params.get(key) is None:
            raise ValueError(f"arm {arm} needs an explicit {key}")
        algo[key] = params[key]
    # Keep the phase-1 dispersion ratio mu_V/mu_M = 0.25 so the tail layer is
    # compared at the same relative strength as the published USV-0.25.
    if overrides.get("mean_mu", "absent") is None and algorithm == "dmd_tail":
        algo["dispersion_mu"] = 0.25 * algo["mean_mu"]
    if algorithm in {"fedavg", "cb_ce"}:
        algo = {k: v for k, v in algo.items() if k in PLAIN_KEYS}
    name = f"{arm}{suffix}_p{partition_seed}_i{init_seed}"
    cfg = {
        "seed": init_seed,
        # One directory per partition: run_experiment names a run after its
        # training seed only, so partitions sharing an init seed would write the
        # same metrics.json and each would silently replace the previous one.
        "output_dir": f"./{out_root}/{arm}{suffix}/p{partition_seed}",
        "device": device,
        "cost_model": "phi",
        "layer_mismatch": False,
        # partition_seed is independent of seed: one drives the shards, the
        # other the initialisation and the participation draw.
        "data": {"dataset": dataset, "model": "cnn_gn", "partition": experiment.partition,
                 "partition_seed": partition_seed, "alpha": ALPHA,
                 "data_root": "./data"},
        "model": {"architecture": "cnn_gn"},
        "training": {"num_rounds": rounds, "algorithm": algorithm,
                     "algo_config": algo},
        "clients": {"num_clients": NUM_CLIENTS, "sample_fraction": 0.5,
                    "sampling_strategy": "round_robin", "min_clients": 5,
                    "dropout_rate": 0.2,
                    "fleet": [{"type": "raspberry_pi_4", "count": 10}]},
    }
    root = config_root(dataset, experiment)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def _calibration_row(root: Path, directory: str, key: str, value: float,
                     calibration_partitions: list[int], lo: int, hi: int) -> dict:
    expected = len(calibration_partitions) * len(CALIBRATION_INITS)
    paths = sorted(
        path for part in calibration_partitions
        for path in (root / directory / f"p{part}").glob("*/metrics.json")
    )
    runs = [json.loads(p.read_text())["rounds"] for p in paths]
    row = {key: value, "runs": len(runs), "eligible": len(runs) == expected}
    if key != "mean_mu":
        row["param"] = key
    for key, (field, scale, _, _) in METRICS.items():
        values = [late_mean(rounds, field, lo, hi) for rounds in runs]
        values = [v * scale for v in values if v is not None]
        row[key] = mean(values) if values else None
    row["eligible"] = row["eligible"] and row["worst20"] is not None
    return row


def select(dataset: str, calibration_partitions: list[int],
           experiment: Experiment = EXPERIMENTS["bounded_margin"]) -> dict:
    """Fix each calibrated arm's swept value from the finished 2a sweep.

    Criterion, stated before any calibration run existed: the value (mean_mu,
    or the arm's own swept parameter) with the highest Worst-20 client balanced
    accuracy, averaged over the last ``calibration_window`` rounds and then over
    the calibration partitions.  Worst-20 is the tail DMD exists to lift, and it
    also collapses once the penalty overwhelms cross-entropy, so it guards both
    ends of the sweep.  Ties go to the smaller value.  A value missing any run
    (a diverged run writes no metrics.json) is not eligible.  Reference arms are
    tabulated for reading and never selected.
    """

    root = Path(results_root(dataset, "calibrate", experiment))
    hi = experiment.calibration_rounds
    lo = hi - experiment.calibration_window + 1
    table, picks = {}, {}
    for reference in experiment.reference_arms():
        mu = experiment.arms[reference][1]["mean_mu"]
        table[reference] = [
            {**_calibration_row(root, f"{reference}_mu{mu:g}", "mean_mu", mu,
                                calibration_partitions, lo, hi), "reference": True}
        ]
    for arm in experiment.calibrated:
        key, tag, values = experiment.sweep_of(arm)
        rows = [_calibration_row(root, f"{arm}_{tag}{value:g}", key, value,
                                 calibration_partitions, lo, hi) for value in values]
        eligible = [row for row in rows if row["eligible"]]
        if not eligible:
            raise SystemExit(f"{arm}: no {key} finished on every calibration partition")
        picks[arm] = max(eligible, key=lambda row: row["worst20"])[key]
        table[arm] = rows
    only_mu = all(experiment.sweep_of(arm)[0] == "mean_mu" for arm in experiment.calibrated)
    word = "mean_mu" if only_mu else "value"
    selection = {
        "criterion": "max Worst-20 client balanced accuracy, mean over rounds "
                     f"{lo}-{hi} then over calibration partitions; ties to the "
                     f"smaller {word}; {word} with a missing run ineligible",
        "calibration_partitions": calibration_partitions,
        "calibration_inits": CALIBRATION_INITS,
        "picks": picks,
    }
    if only_mu:
        selection["test_mean_mu"] = {arm: picks[source]
                                     for arm, source in experiment.mu_source.items()}
    else:
        selection["test_params"] = {
            arm: {experiment.sweep_of(source)[0]: picks[source]}
            for arm, source in experiment.mu_source.items()
        }
    selection["table"] = table
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="emnist")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS),
                        default="bounded_margin")
    parser.add_argument("--step", required=True,
                        choices=["calibrate", "select", "test", "describe"])
    parser.add_argument("--mean-mu", type=float,
                        help="one intensity for every DMD arm of --step test")
    parser.add_argument("--selection", type=Path,
                        help="per-arm intensities written by --step select")
    parser.add_argument("--feasible-partitions", action="store_true",
                        help="replace partition seeds whose smallest client cannot "
                             "hold the nominal train + anchor split (see "
                             "feasible_partitions); needed off EMNIST")
    args = parser.parse_args()
    experiment = EXPERIMENTS[args.experiment]
    root = config_root(args.dataset, experiment)

    if args.step == "describe":
        # The paths a campaign script needs, from the one place that defines them.
        # The last field is the number of arms to calibrate: 0 means every arm
        # has a fixed mean_mu and the campaign goes straight to 2b.
        tag = DATASETS[args.dataset][0]
        print(root, results_root(args.dataset, "calibrate", experiment),
              results_root(args.dataset, "test", experiment),
              f"{tag}_{experiment.results}", len(experiment.calibrated))
        return

    calibration_partitions, test_partitions = partition_seeds(
        args.dataset, args.feasible_partitions, experiment)

    if args.step == "select":
        selection = select(args.dataset, calibration_partitions, experiment)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "phase2a_selection.json"
        path.write_text(json.dumps(selection, indent=2))
        print(f"valeurs retenues: {selection['picks']} -> {path}")
        return

    paths = []
    common = {"dataset": args.dataset, "device": args.device, "experiment": experiment}
    if args.step == "calibrate":
        out_root = results_root(args.dataset, "calibrate", experiment)
        rounds = experiment.calibration_rounds
        # Only the arms whose value is unknown need calibrating.
        for arm in experiment.calibrated:
            key, tag, values = experiment.sweep_of(arm)
            for value in values:
                for partition in calibration_partitions:
                    for init in CALIBRATION_INITS:
                        paths.append(build(arm, partition, init, rounds, {key: value},
                                           out_root, suffix=f"_{tag}{value:g}", **common))
        for reference in experiment.reference_arms():
            mu = experiment.arms[reference][1]["mean_mu"]
            for partition in calibration_partitions:
                for init in CALIBRATION_INITS:
                    paths.append(build(reference, partition, init, rounds, {},
                                       out_root, suffix=f"_mu{mu:g}", **common))
    else:
        if args.selection is not None:
            selection = json.loads(args.selection.read_text())
            test_params = selection.get("test_params") or {
                arm: {"mean_mu": mu} for arm, mu in selection["test_mean_mu"].items()
            }
        elif args.mean_mu is not None:
            test_params = {arm: {"mean_mu": args.mean_mu} for arm in experiment.mu_source}
        elif experiment.calibrated:
            parser.error("--step test requires --selection or --mean-mu from step 2a")
        else:
            test_params = {}  # every arm carries its own fixed values
        for arm in experiment.test_arms or experiment.arms:
            for partition in test_partitions:
                for init in experiment.test_inits:
                    paths.append(build(arm, partition, init, TEST_ROUNDS,
                                       test_params.get(arm, {}),
                                       results_root(args.dataset, "test", experiment),
                                       **common))
    used = calibration_partitions if args.step == "calibrate" else test_partitions
    print(f"{len(paths)} configs ecrites dans {root}/ (partitions {used})")
    for path in paths[:4]:
        print(f"  {path}")
    if len(paths) > 4:
        print(f"  ... (+{len(paths) - 4})")


if __name__ == "__main__":
    main()
