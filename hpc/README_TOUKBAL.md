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

CPU-first equivalents are provided as:

- `run_r1_r2_r3_cpu.slurm`;
- `run_level_p_paper_fidelity_cpu.slurm`;
- `run_level_s_scfar_validation_cpu.slurm`.

The older `run_internship_far_fedfdp.slurm` is the generic R/P matrix engine.
`run_r1_r2_r3.slurm` is the explicit Toubkal entry point requested for Level R.

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
```

For a CPU-first campaign, use the `_cpu.slurm` entry points with the
`<PROJECT>-DEFAULT-CPU` account.  Their array concurrency is deliberately
bounded so that the initial validation does not monopolise the CPU project.

Environment variables such as `REPO_DIR`, `PROJECT_DIR`, `WORK_ROOT`,
`DATA_ROOT`, `OUTPUT_ROOT`, and `CONDA_ENV` can be overridden with
`sbatch --export=ALL,NAME=value`.
