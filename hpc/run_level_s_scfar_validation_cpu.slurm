#!/bin/bash
#SBATCH --job-name=scfar_level_s_cpu
#SBATCH --array=0-2%3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# CPU-first Level S main validation on three independent seeds.

set -euo pipefail

REPO_DIR="${REPO_DIR:-${HOME}/fedlab_zmq}"
PROJECT_DIR="${PROJECT_DIR:-${HOME}/lustre/manapy-um6p-st-msda-1wabcjwe938}"
WORK_ROOT="${WORK_ROOT:-${PROJECT_DIR}/users/${USER}/fedlab_zmq}"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/results/level_s_scfar_cpu}"
CONDA_ENV="${CONDA_ENV:-fedlab-zmq}"
SEEDS=(28 36 54)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
RUN_OUTPUT="${OUTPUT_ROOT}/main/seed${SEED}"

module purge
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p "${DATA_ROOT}" "${RUN_OUTPUT}"
cd "${REPO_DIR}"

python -u run_experiment.py \
  --config configs/scpfar/cifar10_resnet18_main.yaml \
  --device cpu \
  --seed "${SEED}" \
  --data-root "${DATA_ROOT}" \
  --output "${RUN_OUTPUT}"
