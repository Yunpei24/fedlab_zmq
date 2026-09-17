#!/bin/bash
#SBATCH --job-name=far_dp_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Account / premium QoS / partition are intentionally not guessed here.
set -euo pipefail
: "${FEDLAB_REPO:?Set absolute repository path}"
: "${FAR_MANIFEST:?Set the prepared manifest path}"
: "${FAR_DATA_ROOT:?Set shared dataset path}"

if [[ -n "${CONDA_ENV:-}" ]]; then
    module load "${CONDA_MODULE:-Anaconda3/2025.06-1}"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "${FEDLAB_REPO}"
TASK_INDEX=$(( ${JOB_OFFSET:-0} + ${SLURM_ARRAY_TASK_ID:-0} ))

exec srun --cpu-bind=cores "${PYTHON_BIN}" -u scripts/run_far_dp_effect_cpu.py run \
    --manifest "${FAR_MANIFEST}" \
    --job-index "${TASK_INDEX}" \
    --threads "${SLURM_CPUS_PER_TASK:-8}" \
    --data-root "${FAR_DATA_ROOT}" \
    --resume
