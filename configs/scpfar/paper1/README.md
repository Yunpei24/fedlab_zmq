# SC-FAR-DP paper 1: frozen full-update protocol

This directory is the executable pre-registration of S1–S4. The common
protocol protects one complete client contribution with replace-one
user-level central DP. All 25 clients participate at every round; client
dropout, partial training and privacy amplification by client sampling are
excluded.

## Frozen design choices

### Reference, anchor and Byzantine grid

- `tau/C`: `0.25, 0.5, 1, 2` in the clean screening lane; `0.5, 1` under
  attack.
- anchors: fixed zero, EMA of the previous public release with rates `0.1`
  and `0.5`, and the previous public release itself.
- anchor cadence ablation: update every `1` or `5` rounds in a targeted S1
  submatrix.
- `n=25`; `b=0,2,5`, corresponding exactly to Byzantine fractions
  `0, 0.08, 0.20`.
- attacks: BF, IPM, ALIE, Min-Max and Min-Sum.

The primary reference is one-step public-anchor centered clipping. A
regularized Huber M-estimator with ten public fixed iterations is an ablation.

### Controlled tilting and honest outliers

For scores in `[0,1]`, the runner derives

```text
alpha = r alpha_max(n,kappa_w),
alpha_max(n,kappa_w) = log[kappa_w(n-1)/(n-kappa_w)].
```

The frozen grid is:

```text
kappa_w in {1.25, 2, 5}
r = alpha/alpha_max in {0, 0.25, 0.5, 0.75, 1}
```

The runner uses `alpha_bound_policy=error`: an invalid task is rejected rather
than silently clipped. For `n=25`, the three alpha maxima are approximately
`0.233615`, `0.735707` and `1.791759`.

Honest outliers are pre-registered before training as 20% (four of twenty) of
the always-honest clients with the largest Jensen-Shannon divergence between their label
distribution and the global label distribution. Their IDs are oracle-only
evaluation metadata and never influence training.

BF, IPM and ALIE use intensity factors `0.5, 1, 2`. Min-Max and Min-Sum use
`0.5, 1`: factor `1` is the canonical optimized stealth point; a factor above
one would no longer inherit the attack's stealth constraint.

## Matrices

| Matrix | Purpose | Tasks |
|---|---|---:|
| `s1_reference_tradeoff.yaml` | `tau`, anchor, cadence and robustness-bias | 1,908 |
| `s2_full_update_ablations.yaml` | full-update ablation chain | 360 |
| `s3_inclusion_attacks.yaml` | inclusion, fairness and Byzantine robustness | 1,224 |
| `s4_central_dp.yaml` | epsilon and sensitivity calibration | 720 |

Every task crosses three partition seeds (`101,102,103`) with three training
seeds (`28,36,54`). FashionMNIST/LeNet-5 and CIFAR-10/AlexNet use the
client-balanced Dirichlet constructor at beta `0.1` and `0.5`.

## Validation and execution

```bash
python scripts/run_scfar_paper1.py \
  --validate \
  --matrix configs/scpfar/paper1/s2_full_update_ablations.yaml
```

The runner rejects partial-training keys, incomplete participation, dropout,
invalid alpha bounds, unsupported references and overlaps between Byzantine
and honest-outlier IDs. It writes one resolved YAML and one immutable manifest
per task, supports resume, and verifies after training that no client died and
that every declared round was completed.

Toubkal uses `hpc/run_scfar_paper1_cpu.slurm` or
`hpc/run_scfar_paper1_gpu.slurm`; the array range is provided at submission
time because the matrices have different cardinalities.
