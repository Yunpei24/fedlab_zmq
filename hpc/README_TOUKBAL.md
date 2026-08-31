# Toubkal experiment launchers

For the complete CPU installation and validation procedure, see
[`SETUP_TOUKBAL_CPU.md`](SETUP_TOUKBAL_CPU.md).

The launchers are split by scientific purpose so that results from different
protocols are never pooled accidentally.

| Level | Launcher | Purpose |
|---|---|---|
| R | `run_r1_r2_r3.slurm` | 135 internship-report R1/R2/R3 tasks |
| P | `run_level_p_paper_fidelity.slurm` | Paper-aligned FAR/FedFDP references and baselines |
| S | `run_level_s_scfar_validation.slurm` | Main SC-Partial-FAR-DP training validation |
| S | `run_level_s_sensitivity_audit.slurm` | CPU replace-one sensitivity falsification audit |
| S0.5 | `run_s0_5_reference_comparators_cpu.slurm` | CPU audit of mean, CM, trimmed mean and RFA |
| S-P1 | `run_scfar_paper1_cpu.slurm` | Worker CPU des matrices full-update S1–S4 |
| S-P1 | `run_scfar_paper1_gpu.slurm` | Worker GPU des matrices full-update S1–S4 |

CPU-first equivalents are provided as:

- `run_r1_r2_r3_cpu.slurm`;
- `run_level_p_paper_fidelity_cpu.slurm`;
- `run_level_s_scfar_validation_cpu.slurm`;
- `run_dmd_phase_a_cpu.slurm` for the DMD Phase A matrix.

The older `run_internship_far_fedfdp.slurm` is the generic R/P matrix engine.
`run_r1_r2_r3.slurm` is the explicit Toubkal entry point requested for Level R.
The older `run_level_s_scfar_validation*` launchers target the partial-training
extension. They must not be used for the full-update paper-1 protocol.

## SC-FAR-DP paper 1: frozen full-update matrices

The common protocol is defined in `configs/scpfar/paper1/common.yaml` and is
checked before every run: 25/25 clients participate, dropout is zero,
user-level clipping is enabled, no partial-training key is accepted, adjacency
is replace-one and the Gaussian accountant claims no client-sampling
amplification (`q=1`).

| Matrix | Tasks | Slurm indices |
|---|---:|---:|
| `s1_reference_tradeoff.yaml` | 1,908 | `0-1907` |
| `s2_full_update_ablations.yaml` | 360 | `0-359` |
| `s3_inclusion_attacks.yaml` | 1,224 | `0-1223` |
| `s4_central_dp.yaml` | 720 | `0-719` |

Validate a matrix on the login node without starting training:

```bash
python scripts/run_scfar_paper1.py \
  --validate \
  --matrix configs/scpfar/paper1/s2_full_update_ablations.yaml
```

Submit S2 on CPU, with at most eight simultaneous tasks:

```bash
cd "$HOME/fedlab_zmq/slurm_logs"
sbatch \
  --account=MANAPY-UM6P-ST-MSDA-1WABCJWE938-DEFAULT-CPU \
  --array=0-359%8 \
  --export=ALL,SCFAR_MATRIX=s2_full_update_ablations.yaml \
  ../hpc/run_scfar_paper1_cpu.slurm
```

For a one-round preflight, add
`SCFAR_PILOT_ROUNDS=1` to `--export`. The resulting manifest is explicitly
marked `pilot_not_full_protocol`, so pilot results cannot be mistaken for the
100-round scientific campaign.

The q=1 accountant can be checked independently from the ledger object with:

```bash
python scripts/validate_scfar_gaussian_accountant.py \
  --targets 1,3,6,10 \
  --steps 100 \
  --delta 1e-5 \
  --output "$WORK_ROOT/results/scfar_paper1/accountant_preflight.json"
```

If Opacus is installed in a dedicated validation environment, add
`--require-opacus` to make the external-library check mandatory.

## Before submitting

Create the Conda environment named `fedlab-zmq`, clone the repository into
`$HOME/fedlab_zmq`, and create a local directory for Slurm text logs:

```bash
mkdir -p "$HOME/fedlab_zmq/slurm_logs"
cd "$HOME/fedlab_zmq/slurm_logs"
```

Submit from that directory so the `%A_%a` log files remain on the backed-up
home filesystem.  Pass the Slurm account explicitly; it is intentionally not
hard-coded in the scripts.

```bash
sbatch --account=<PROJECT>-DEFAULT-GPU ../hpc/run_r1_r2_r3.slurm
sbatch --account=<PROJECT>-DEFAULT-GPU ../hpc/run_level_p_paper_fidelity.slurm
sbatch --account=<PROJECT>-DEFAULT-GPU ../hpc/run_level_s_scfar_validation.slurm
sbatch --account=<PROJECT>-DEFAULT-CPU ../hpc/run_level_s_sensitivity_audit.slurm
sbatch --account=<PROJECT>-DEFAULT-CPU ../hpc/run_s0_5_reference_comparators_cpu.slurm
```

For a CPU-first campaign, use the `_cpu.slurm` entry points with the
`<PROJECT>-DEFAULT-CPU` account.  Their array concurrency is deliberately
bounded so that the initial validation does not monopolise the CPU project.

For example, the complete DMD Phase A CPU matrix is submitted with:

```bash
sbatch --account=<PROJECT>-DEFAULT-CPU hpc/run_dmd_phase_a_cpu.slurm
```

Environment variables such as `REPO_DIR`, `PROJECT_DIR`, `WORK_ROOT`,
`DATA_ROOT`, `OUTPUT_ROOT`, and `CONDA_ENV` can be overridden with
`sbatch --export=ALL,NAME=value`.
