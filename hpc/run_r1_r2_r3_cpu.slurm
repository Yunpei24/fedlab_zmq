#!/bin/bash
#SBATCH --job-name=far_r123_cpu
#SBATCH --array=0-134%4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# CPU-first Level R campaign. Submit with the project's DEFAULT-CPU account.

set -euo pipefail

REPO_DIR="${REPO_DIR:-${HOME}/fedlab_zmq}"
PROJECT_DIR="${PROJECT_DIR:-${HOME}/lustre/manapy-um6p-st-msda-1wabcjwe938}"
WORK_ROOT="${WORK_ROOT:-${PROJECT_DIR}/users/${USER}/fedlab_zmq}"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/results/internship_far_fedfdp_cpu}"
CONDA_ENV="${CONDA_ENV:-fedlab-zmq}"

module purge
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "${DATA_ROOT}" "${OUTPUT_ROOT}"
cd "${REPO_DIR}"

python -u scripts/run_internship_far_fedfdp.py \
  --run \
  --lane faithful \
  --scenario exp1_fairness_no_attack \
  --scenario exp2_fairness_robustness \
  --scenario exp3_privacy_fairness \
  --job-index "${SLURM_ARRAY_TASK_ID}" \
  --device cpu \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --resume
