#!/bin/bash
#SBATCH --job-name=far_level_p
#SBATCH --array=0-17%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --qos=default-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Level P: paper-aligned FAR/FedFDP reference protocols and their baselines.
# This is a confirmatory implementation audit, not a claim of bit-for-bit
# reproduction when original partitions or private author code are absent.
# Six configurations x three training seeds = 18 independent tasks.

set -euo pipefail

REPO_DIR="${REPO_DIR:-${HOME}/fedlab_zmq}"
PROJECT_DIR="${PROJECT_DIR:-${HOME}/lustre/manapy-um6p-st-msda-1wabcjwe938}"
WORK_ROOT="${WORK_ROOT:-${PROJECT_DIR}/users/${USER}/fedlab_zmq}"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/results/level_p_paper_fidelity}"
CONDA_ENV="${CONDA_ENV:-fedlab-zmq}"

CONFIGS=(
  configs/far_references/far_cifar10_alexnet.yaml
  configs/far_references/far_fmnist_alie.yaml
  configs/far_references/fedfdp_fmnist.yaml
  configs/far_references/qffl_fmnist.yaml
  configs/far_references/robust_baseline_fmnist.yaml
  configs/far_references/term_fmnist.yaml
)
SEEDS=(28 36 54)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
CONFIG_INDEX=$((TASK_ID / ${#SEEDS[@]}))
SEED_INDEX=$((TASK_ID % ${#SEEDS[@]}))
CONFIG="${CONFIGS[$CONFIG_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
CONFIG_NAME="$(basename "${CONFIG}" .yaml)"
RUN_OUTPUT="${OUTPUT_ROOT}/${CONFIG_NAME}/seed${SEED}"

module purge
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${DATA_ROOT}" "${RUN_OUTPUT}"
cd "${REPO_DIR}"

python -u run_experiment.py \
  --config "${CONFIG}" \
  --device cuda \
  --seed "${SEED}" \
  --data-root "${DATA_ROOT}" \
  --output "${RUN_OUTPUT}"
