# FAR / FedFDP internship reproduction matrices

These files distinguish two lanes that must not be pooled in one result table.

## Faithful lane

The three `exp*.yaml` files encode the internship report protocol:

- MNIST, LeNet-5-style CNN, 10 clients, Dirichlet 0.1;
- 40 rounds, 2 local epochs, batch size 64, learning rate 0.01;
- seeds 28, 36 and 54;
- client fairness evaluation every 2 rounds;
- Experiment 1: no attack;
- Experiment 2: IPM and Bit/Sign Flip with two Byzantine clients;
- Experiment 3: epsilon in `{3.56, infinity}`.

The current corrected revision is stored under
`faithful/algorithm_fidelity_v4/`. This revision fixes propagation of the shared
Dirichlet `data.partition_seed`; previous `algorithm_fidelity_v3` and
`far_prox_v2` outputs are retained
as an audit trail and must not be pooled with this revision. The corrected
revision uses FAR's proximal local objective with `far_prox_mu: 0.01` and
retains the report's two-local-epoch protocol through
`far_update_mode: multi_epoch_delta`.  A scientifically distinct
`single_step_gradient` mode is also implemented for paper-style analyses: it
computes one empirical gradient at the received global model and applies the
explicit server learning rate `far_server_lr`.  The proximal gradient is zero
at that anchor, so `mu` has no numerical effect in that one-gradient mode;
this is reported explicitly rather than hidden.

The same revision also fixes three baseline semantics:

- **FedFair** implements Algorithm 1's batch-level dynamic learning rate and
  has no per-example clipping or Gaussian mechanism;
- **FedFDP** implements Algorithm 2's per-example fair clipping, model-noise
  channel and private post-update loss channel. Its epsilon-infinity ablation
  keeps fair clipping but disables both noises, rather than being relabelled
  FedFair;
- **q-FedAvg** uses the unweighted numerator/denominator sums from Algorithm 2.
  Dataset-size weighting remains available only as the explicitly named
  `aggregation_prior: dataset_size` ablation.

The exact notebook referenced by the report is not available. The matrices
therefore use seed 28 as an explicit shared Dirichlet partition seed for all
training seeds. This preserves the reported shared-partition design, but the
partition, exact CNN details and epsilon-to-noise calibration cannot be
claimed as bit-for-bit reproductions until that notebook is recovered.

## Paper-grade extension lane

`extension_paper_grade.yaml` is a new confirmatory protocol, not part of the
internship report. It adds:

- epsilon in `{1, 2, 3, 4}`;
- ALIE, Min-Max and Min-Sum attacks;
- Sensitivity-Controlled Partial FAR under DP;
- privacy, fairness, robustness and efficiency telemetry.

## Commands

Validate and inspect the deterministic matrix:

```bash
python3 scripts/run_internship_far_fedfdp.py --validate --lane all
python3 scripts/run_internship_far_fedfdp.py --list --lane faithful
```

Run one faithful task by index:

```bash
python3 scripts/run_internship_far_fedfdp.py \
  --lane faithful --job-index 0 --device cuda
```

Run a short software pilot without confusing it with a protocol result:

```bash
python3 scripts/run_internship_far_fedfdp.py --run \
  --lane faithful --scenario exp3_privacy_fairness --job-index 0 \
  --pilot-rounds 1 --pilot-local-batches 1 \
  --output-root /tmp/internship-far-pilot --device cpu
```

Run all currently ready Experiment-1 tasks locally:

```bash
python3 scripts/run_internship_far_fedfdp.py \
  --scenario exp1_fairness_no_attack --skip-unavailable
```

Finite-epsilon tasks are rejected unless their registered algorithm exposes a
`target_epsilon` calibration contract. This prevents a numeric target epsilon
from being recorded when the noise multiplier was never calibrated.
