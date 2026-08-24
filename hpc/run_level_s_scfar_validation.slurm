#!/bin/bash
#SBATCH --job-name=scfar_level_s
#SBATCH --array=0-2%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --qos=default-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Level S: main SC-Partial-FAR-DP validation on three independent seeds.
# The separate CPU sensitivity audit is launched with
# hpc/run_level_s_sensitivity_audit.slurm.

set -euo pipefail

REPO_DIR="${REPO_DIR:-${HOME}/fedlab_zmq}"
PROJECT_DIR="${PROJECT_DIR:-${HOME}/lustre/manapy-um6p-st-msda-1wabcjwe938}"
WORK_ROOT="${WORK_ROOT:-${PROJECT_DIR}/users/${USER}/fedlab_zmq}"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/results/level_s_scfar}"
CONDA_ENV="${CONDA_ENV:-fedlab-zmq}"
SEEDS=(28 36 54)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

module purge
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

RUN_OUTPUT="${OUTPUT_ROOT}/main/seed${SEED}"
mkdir -p "${DATA_ROOT}" "${RUN_OUTPUT}"
cd "${REPO_DIR}"

python -u run_experiment.py \
  --config configs/scpfar/cifar10_resnet18_main.yaml \
  --device cuda \
  --seed "${SEED}" \
  --data-root "${DATA_ROOT}" \
  --output "${RUN_OUTPUT}"
