#!/bin/bash
#SBATCH --job-name=dtldp_pilot_cpu
#SBATCH --array=0-36%6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=compute
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Submit with: sbatch --account="$FEDLAB_ACCOUNT" hpc/run_dt_ldp_far_pilot_cpu.slurm
set -euo pipefail

REPO_DIR="${REPO_DIR:-${HOME}/fedlab_zmq}"
PROJECT_DIR="${PROJECT_DIR:-${HOME}/lustre/manapy-um6p-st-msda-1wabcjwe938}"
WORK_ROOT="${WORK_ROOT:-${PROJECT_DIR}/users/${USER}/fedlab_zmq}"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/results/dt_ldp_far}"
CONDA_ENV="${CONDA_ENV:-fedlab-zmq}"

module purge
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

mkdir -p "${DATA_ROOT}" "${OUTPUT_ROOT}"
cd "${REPO_DIR}"

python -u scripts/run_dt_ldp_far.py \
  --run \
  --matrix configs/dt_ldp_far/pilot_e1_e8.yaml \
  --job-index "${SLURM_ARRAY_TASK_ID}" \
  --device cpu \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --resume
