#!/bin/bash
#SBATCH --job-name=dtldp_full_gpu
#SBATCH --array=0-317%8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --qos=default-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Submit only after pilot acceptance:
# sbatch --account="$FEDLAB_ACCOUNT" hpc/run_dt_ldp_far_full_gpu.slurm
set -euo pipefail

REPO_DIR="${REPO_DIR:-${HOME}/fedlab_zmq}"
PROJECT_DIR="${PROJECT_DIR:-${HOME}/lustre/manapy-um6p-st-msda-1wabcjwe938}"
WORK_ROOT="${WORK_ROOT:-${PROJECT_DIR}/users/${USER}/fedlab_zmq}"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/results/dt_ldp_far}"
CONDA_ENV="${CONDA_ENV:-fedlab-zmq-gpu}"

module purge
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${DATA_ROOT}" "${OUTPUT_ROOT}"
cd "${REPO_DIR}"

python -u scripts/run_dt_ldp_far.py \
  --run \
  --matrix configs/dt_ldp_far/full_e1_e8.yaml \
  --job-index "${SLURM_ARRAY_TASK_ID}" \
  --device cuda \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --resume
