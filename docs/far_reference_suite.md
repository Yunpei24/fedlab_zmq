# FAR, fairness and privacy reference suite

This directory documents the reference implementations added to `fedlab_zmq`
for the study of **Sensitivity-Controlled Partial FAR under DP**.  The code is
written for readability and auditable experiments.  It does not claim to
reproduce a paper result until the paper-specific hyperparameters, seeds and
threat model have been matched.

## What is implemented

| Component | Registry/config name | Role |
|---|---|---|
| q-FedAvg | `qffl`, `qfedavg` | q-FFL fairness baseline |
| Client-level TERM | `term` | loss-tilted fairness baseline |
| FAR | `far` | distance-to-robust-reference reweighting |
| FedFair | `fedfair` | non-private fair-clipping ablation |
| FedFDP | `fedfdp` | sample-level local DP, private model and loss channels |
| Robust aggregation | `robustfedavg` | standalone CM(NNM), trMean(NNM), NBS, RFA or CMLS |

The Byzantine attacks live in `attacks/`.  The robust rules live in
`robustness/`.  They are independent of FAR so that a robust baseline and its
FAR-wrapped counterpart can share the same local training and attack.

## Algorithm equations

### q-FedAvg

Client `k` evaluates its loss at the received global model, trains locally to
obtain `w_bar_k`, and returns

```text
Delta_w_k = L (w_t - w_bar_k)
Delta_k   = F_k(w_t)^q Delta_w_k
h_k       = q F_k(w_t)^(q-1) ||Delta_w_k||² + L F_k(w_t)^q.
```

The server performs

```text
w_(t+1) = w_t - sum_k p_k Delta_k / sum_k p_k h_k.
```

`q=0` recovers the FedAvg-style update under the selected prior `p_k`.  Larger
positive `q` increases the relative influence of high-loss clients.  This
algorithm is not differentially private: the server consumes client loss and
curvature metadata in cleartext.

### Client-level TERM

The server forms stable loss-tilted weights

```text
omega_k = p_k exp(t F_k) / sum_j p_j exp(t F_j)
```

and aggregates local model deltas with `omega_k`.  Positive `t` emphasizes
high-loss clients, negative `t` attenuates them, and `t=0` recovers the prior.
This is the federated group-level interpretation used as a reference; it is
not a private loss channel.

### FAR

FAR first asks a Byzantine-robust rule `F` for a reference:

```text
g_F      = F(g_1, ..., g_n)
d_i      = ||g_i - g_F||_2
lambda_i = softmax_i(alpha d_i)
g_FAR    = sum_i lambda_i g_i.
```

The robust rule supplies only the reference.  FAR's final output reintroduces
all submitted updates.  With positive `alpha`, distant updates receive more
weight; this supports honest statistical outliers but is not a universal
Byzantine defense.  The maximum weight, entropy, effective number of clients
and oracle Byzantine weight mass are logged to expose concentration.

### FedFDP

For each training example `j`, the client computes a gradient and the
fair-clipping factor

```text
fair_ij = 1 + lambda (loss_ij - global_loss)
scale_ij = min(fair_ij, C / ||gradient_ij||_2).
```

The clipped per-example gradients are summed, Gaussian noise is added and the
batch average is used for SGD.  A separately clipped and noised loss is
released for the next round's fairness reference.  The RDP accountant charges
the model and loss channels separately and composes them over local steps and
rounds.

The finite-sum accountant implements the Poisson-sampled Gaussian expression
used in the paper.  The current generic FedLab loader draws shuffled,
fixed-size minibatches; consequently, the displayed epsilon is a diagnostic
Poisson-accountant value rather than a deployment certificate until a matching
Poisson sampler (or a fixed-size without-replacement accountant) is selected.

This is a **sample-level local-DP reference**.  It is deliberately different
from the planned SC-Partial-FAR mechanism, whose first target is user-level
central Gaussian DP after aggregation.  The explicit per-example loop is
pedagogical and slow; do not interpret wall-clock measurements from this
backend as an optimized DP-SGD result.

## Robust aggregation baselines

Set `training.algo_config.robust_aggregator` to:

* `cm_nnm`: nearest-neighbour mixing followed by coordinate median;
* `trmean_nnm`: nearest-neighbour mixing followed by coordinate trimmed mean;
* `nbs`: remove the largest update norms, then average;
* `rfa`: geometric median via a smoothed Weiszfeld iteration;
* `cmls`: coordinate-median reference plus inverse-distance reintroduction of
  submitted vectors.

For NNM and trimmed mean, `num_byzantine=f` is an assumed attack budget, not
an oracle detection result.  The validity of `f` must be part of the
experimental threat model.

## Byzantine attacks

The `attack` block accepts `alie`, `minmax`, `minsum`, `bf`/`sign_flip`, or
`ipm`.  Example:

```yaml
attack:
  enabled: true
  name: minmax
  num_byzantine: 5
  client_ids: [0, 1, 2, 3, 4]
```

Explicit IDs are recommended.  If omitted, the runner fixes the first
`num_byzantine` global client IDs so that attackers remain consistent under
client sampling.  Attack labels are oracle metadata for evaluation only.

## Metrics needed by the research direction

The standalone runner evaluates the common global model on client-specific
held-out partitions.  Byzantine clients are excluded from FAR fairness
metrics by an oracle evaluation mask, but they remain in training.

* `client_accuracy_variance`: equal-client variance in `[0,1]²`;
* `client_accuracy_variance_pct2`: the same variance in percentage-points²,
  matching FAR-style tables;
* `worst20_accuracy`, `best20_accuracy`, `best20_worst20_gap`;
* `client_loss_variance`: equal-client loss variance;
* `balanced_performance_fairness`: data-size-weighted client loss variance,
  corresponding to the FedFDP definition;
* `max_client_weight`, `weight_entropy`, `effective_num_clients`;
* `byzantine_weight_mass_oracle`: evaluation-only mass assigned to attackers;
* `privacy_epsilon_max/mean`, `privacy_delta`, clipping rate and RDP order;
* global accuracy/loss, bytes, joules and survival metrics already present in
  `fedlab_zmq`.

For the future SC-Partial-FAR algorithm, also log the active group, group
dimension, user clipping rate, empirical output change in neighbouring-client
audits, Gaussian noise norm and Joules-to-Accuracy.

## Running the references

```bash
python run_experiment.py --config configs/far_references/qffl_fmnist.yaml
python run_experiment.py --config configs/far_references/term_fmnist.yaml
python run_experiment.py --config configs/far_references/far_fmnist_alie.yaml
python run_experiment.py --config configs/far_references/robust_baseline_fmnist.yaml
python run_experiment.py --config configs/far_references/fedfdp_fmnist.yaml
python run_experiment.py --config configs/far_references/far_cifar10_alexnet.yaml
```

Before a large sweep, use multiple seeds and construct a factorial matrix:

```text
defense  x  attack  x  Dirichlet alpha  x  FAR alpha  x  seed
```

At minimum, compare no attack, BF, IPM, ALIE, Min-Max and Min-Sum against
CM(NNM), trMean(NNM), NBS, RFA and CMLS, each with and without FAR.

## Interpretation constraints

1. **DP is not Byzantine integrity.** Gaussian noise limits information in a
   release; it does not certify honest local training or honest loss reports.
2. **Robustness is not DP.** A robust reference can tolerate poisoned vectors
   without hiding training records from an observer.
3. **FAR is not automatically robust.** Positive tilting can amplify a
   malicious update if it is distant yet survives the reference rule.
4. **Fairness requires client test partitions.** Pooled test accuracy cannot
   establish Worst-20 or inter-client variance.
5. **FedFDP has two privacy channels.** Reporting only the model accountant
   while ignoring the private-loss release understates privacy loss.
6. **Sensitivity-Controlled FAR is not implemented here yet.** A cap such as
   `lambda_i <= kappa/n` suggests a direct-influence scale, but an
   `O(kappa C/n)` end-to-end sensitivity claim also needs a proof controlling
   the stability of the robust reference and of all data-dependent weights.
