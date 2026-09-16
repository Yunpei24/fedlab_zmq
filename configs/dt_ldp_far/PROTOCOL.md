# DT-LDP-FAR experimental protocol

The registered algorithm is `dt_ldp_far`. It must not be reported as either
`dpfar` (current-round FAR weights over local-DP updates) or `scfar_dp`
(central user-level Gaussian DP).

## Primary privacy semantics

The private DT-LDP-FAR lane is frozen to **sample-level add/remove
adjacency** inside each client.  At every local DP step, every record in the
client dataset is included independently with public probability `q=0.05`.
This is genuine Poisson subsampling, not a shuffled fixed-size minibatch later
treated as Poisson by the accountant.

The exact realised local cardinality is not released in this add/remove lane.
Each scenario instead declares a public padded capacity `N_i^pub` (6001 for
the balanced FMNIST clients and 5001 for the balanced CIFAR-10 clients). The
noisy sum is divided by the public constant `q*N_i^pub`. Hence two neighboring
datasets use the same normalization and server metadata even though one
contains one additional active record.

For the corrected v2 E0--E6 and common-Poisson E8 configurations, one
communication round uses five Poisson DP steps. Within each sampled subset,
the implementation computes
one gradient per sampled example, clips each gradient, sums the clipped
gradients, adds one Gaussian vector to the sum, divides by the public expected
batch size `q*N_i^pub`, and applies one optimizer step.  It therefore does not
make a complete pass over the client's dataset and does not perform one
optimizer step per example.

The matrix field `local_epochs` is retained for compatibility with the common
experiment runner.  Under `sampling_scheme: poisson`, it denotes the number
of independent Poisson DP steps per communication round, not the number of
complete data epochs.  E7 explicitly varies this value in `{1,2,5}` and
recalibrates privacy accounting for the resulting number of mechanisms. The
one-step setting is no longer the default comparison lane because the v1
pilot showed that it severely under-trains every common-Poisson method.

The resulting RDP ledger uses the same public `q`, noise multiplier, number of
Poisson steps, and add/remove adjacency as the executed mechanism.  Fixed
minibatch runs remain available only as legacy/reproduction lanes and are
labelled `poisson_approximation_for_fixed_minibatches` rather than certified
as the primary SGM lane.

This explicit choice also removes an ambiguity in the FedFDP manuscript: its
general DP definition places both neighboring datasets in `X^n` and says they
differ in one sample, which reads as bounded/replace-one adjacency, whereas
its privacy theorem imports the Poisson SGM mixture whose cited source uses
add/remove adjacency. DT-LDP-FAR does not combine those two conventions: the
executed sampler, adjacency, sensitivity normalization, and accountant are
all the add/remove Poisson-SGM lane.

## Executable smoke test

```bash
python3 run_experiment.py \
  --config configs/dt_ldp_far/smoke_mnist.yaml \
  --device cpu
```

The smoke test validates execution only. It is not included in scientific
tables. A paper run must set `privacy_num_rounds`, calibrate
`target_epsilon`, remove `max_local_batches`, use at least three training and
partition seeds, and record the exact sampling/accountant assumption.

## Executable E1--E8 matrices

The bounded campaign is frozen in three YAML files:

- `common.yaml` declares shared scenarios and profiles;
- `pilot_e1_e8.yaml` contains 37 tasks over one paired seed;
- `full_e1_e8.yaml` contains 318 tasks over three paired seeds.

The v2 pilot exercises every experiment block and software path before the
full campaign is submitted. It does not reuse v1 output directories. Both
matrices use full participation and no dropout;
the main privacy accounting lane therefore makes no client-sampling
amplification claim.

Validate, inspect and dry-run without training:

```bash
python scripts/run_dt_ldp_far.py --validate \
  --matrix configs/dt_ldp_far/pilot_e1_e8.yaml

python scripts/run_dt_ldp_far.py --list \
  --matrix configs/dt_ldp_far/pilot_e1_e8.yaml

python scripts/run_dt_ldp_far.py --dry-run \
  --matrix configs/dt_ldp_far/pilot_e1_e8.yaml \
  --job-index 0 --device cpu
```

`--pilot-rounds 2` creates a short infrastructure check and recalibrates the
accounting horizon. Such output is explicitly labelled “not paper evidence”.
The runner checks per-block and total task counts, rejects a campaign above
its default 512-task safety limit, and resumes only complete `metrics.json`
outputs.

Toubkal launch commands are:

```bash
sbatch --account="$FEDLAB_ACCOUNT" hpc/run_dt_ldp_far_pilot_cpu.slurm
sbatch --account="$FEDLAB_ACCOUNT" hpc/run_dt_ldp_far_full_gpu.slurm
```

The full GPU command is used only after the 37 pilot tasks are accepted.
The frozen full matrix covers both FMNIST heterogeneity levels. The declared
CIFAR-10/AlexNet scenario is a promotion lane, not silently included in the
318-task core campaign; it should be activated only after FMNIST go/no-go.

## Frozen causal order

For every round `t`:

1. clients execute sample-level local DP-SGD with add/remove adjacency and
   genuine Poisson sampling, then send `Y_i,t`;
2. the server computes `X_i,t = Clip_U(Y_i,t)`;
3. the server aggregates with weights constructed from scores at `t-1`;
4. only after the current aggregate is fixed, it computes `F_t`, distances,
   bounded scores, and the weights that will be used at `t+1`.

Weights are indexed by `client_id`, never by message position. The main theory
uses full participation. Partial-cohort runs are implementation stress tests,
not evidence for the full-participation theorem.

## E0-E8 campaign blocks

| ID | Scientific question | Required comparisons / axes | Primary diagnostics |
|---|---|---|---|
| E0 | Which frozen public geometry makes tilting observable without violating the cap? | public score scale `D_score` in {0.02, 0.05, 0.10}; tau at 0.5/1.0 of analytic maximum | score/logit span, saturation rate, max weight, entropy, clipping rate |
| E1 | Does delayed tilting retain non-private FAR utility? | native FedAvg/FAR reported separately; clipping-matched no-noise Poisson DPFedAvg/DPFAR/DT with five steps | accuracy, Worst-20, gap, score margin, entropy |
| E2 | Does delay prevent same-noise self-weighting? | bounded-normalised current-round `dpfar` vs instrumented `dt_ldp_far`, identical clipping, `D_score`, tau, five Poisson steps and DP calibration | delayed/current weight L1, score-noise correlation oracle, noise amplification factor |
| E3 | Privacy-utility frontier | core: epsilon in {1, 2, 4, 8, infinity}, FMNIST Dirichlet 0.1/0.5; promotion: MNIST/CIFAR-10 | realised epsilon, accuracy, Worst-20, convergence |
| E4 | Robustness under Byzantine clients | core: none, BF, IPM, ALIE, Min-Max, Min-Sum at fraction 0.2; promotion: fraction 0.1 | attack success, Byzantine weight mass, reference error oracle |
| E5 | Role of robust reference F | centered clipping main; CM(NNM), trMean(NNM), RFA, Huber ablations | reference drift, honest-center error, false outlier rate |
| E6 | Cost of staleness | current weights, one-round delay, optional longer delays; model/data drift | score drift, staleness aggregate norm, fairness loss |
| E7 | Multiple local DP steps | core: Poisson steps 1/2/5; promotion: learning-rate and clipping sweep | clip oracle in diagnostic-only runs, update bias proxies, final utility and realised epsilon |
| E8 | End-to-end paper comparison | common true-Poisson lane for DP-FedAvg, DP-qFFL, bounded `dpfar`, DT; separate FedFDP-native and fixed-minibatch five-step lanes | realised (not nominal-label) epsilon, utility, client fairness, time/energy |

## Comparability and reporting policy

Native non-private FedAvg/FAR use their ordinary minibatch optimisers.  They
are not compared as if they shared the local per-example clipping used by the
local-DP methods.  E1 therefore adds a distinct no-noise control in which
DPFedAvg, bounded-score DPFAR and DT-LDP-FAR all execute the same five Poisson
steps and the same per-example clipping, with only the server rule changing.

The v1 geometry (`server_clip_norm=1`, `D_score=2`) produced scores around
0.005 and essentially uniform weights.  A first v2 smoke test with five
Poisson steps then showed distances around 0.027--0.033, making `D_score=0.01`
saturate every score at one. V2 therefore freezes a public calibration grid
before the campaign: server clip 0.05 and `D_score` in {0.02, 0.05, 0.10}.
No current-round private message is allowed to adapt these values.  The core
v2 lane uses `D_score=0.05` and the analytic boundary tau; E0 must still verify
that this yields non-trivial, non-saturated weights.

FedFDP is not presented as belonging to the true-Poisson DT lane.  Its native
implementation uses shuffled fixed minibatches and an explicitly labelled
Poisson-SGM accounting approximation.  E8 reports it twice: its native full
local pass and a five-minibatch compute-matched diagnostic. Calibration and
online accounting now use the same sampling rate, and tables must report the
realised epsilon from `metrics.json`; profile names such as `eps4` are never
treated as measured privacy values.

## Reference policy

`dt_reference` is configurable. `centered_clipping` is the main candidate
because its robustness-quality and temporal-drift terms can be bounded under
explicit anchor assumptions. Its replace-one stability is **not** used to
obtain local DP. CM(NNM), trimmed mean and RFA are legitimate post-processing
ablations but need their own quality assumptions when used in convergence or
fairness claims.

## Metrics emitted every round

- DP ledger: `privacy_epsilon_max`, `privacy_delta`, noise multiplier, steps;
- delayed mechanism: source round, previous-score coverage, score drift,
  delayed/current weight L1 distance and staleness aggregate norm;
- concentration: max/min weight, entropy, effective clients, analytical cap,
  `n * sum_i omega_i^2` noise-amplification factor;
- geometry: distance range, reference norm and drift, anchor norm;
- robustness/fairness: Byzantine mass (oracle), reference-to-honest-center
  error (oracle), framework client accuracy variance, Worst-20 and gap.

Server-side oracle diagnostics derived from already-private uploads require
`enable_oracle_diagnostics: true` and are not published by the mechanism.
Client-internal loss, clipping and realised-noise diagnostics require the
separate `enable_private_client_oracle_diagnostics: true`.  That second flag
is enabled only for the instrumented E2 mechanism diagnostic: its transcript
is **not** used as DP evidence, and its model trajectory is used only to study
the causal noise/weight interaction.  All privacy--utility and end-to-end
DT-LDP-FAR tasks keep that flag disabled.
