#!/bin/bash
#SBATCH --job-name=far_fedfdp_repro
#SBATCH --array=0-134
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

# Default submission: the 135 faithful internship-report runs.
#
#   sbatch hpc/run_internship_far_fedfdp.slurm
#
# Paper-grade extension (96 runs, intentionally separate):
#
#   sbatch --array=0-95 --export=ALL,LANE=paper-grade \
#     hpc/run_internship_far_fedfdp.slurm
#
# One scenario only: first list its compact indices locally, then submit the
# corresponding array and export SCENARIO. Example:
#
#   python3 scripts/run_internship_far_fedfdp.py \
#     --list --lane faithful --scenario exp3_privacy_fairness
#   sbatch --array=0-53 \
#     --export=ALL,LANE=faithful,SCENARIO=exp3_privacy_fairness \
#     hpc/run_internship_far_fedfdp.slurm

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LANE="${LANE:-faithful}"
DEVICE="${DEVICE:-cuda}"
DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/results/reproductions/internship_far_fedfdp}"

cd "${REPO_DIR}"

# Cluster environments differ. Supply VENV_ACTIVATE or MODULE_SETUP from the
# submission command rather than hard-coding Toubkal-specific module names.
if [[ -n "${MODULE_SETUP:-}" ]]; then
  eval "${MODULE_SETUP}"
fi
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  source "${VENV_ACTIVATE}"
fi

SCENARIO_ARGS=()
if [[ -n "${SCENARIO:-}" ]]; then
  SCENARIO_ARGS=(--scenario "${SCENARIO}")
fi

echo "============================================================"
echo "FedLab FAR/FedFDP reproduction"
echo "Job       : ${SLURM_JOB_ID:-local}[${SLURM_ARRAY_TASK_ID:-0}]"
echo "Lane      : ${LANE}"
echo "Scenario  : ${SCENARIO:-all}"
echo "Device    : ${DEVICE}"
echo "Data root : ${DATA_ROOT}"
echo "Output    : ${OUTPUT_ROOT}"
echo "============================================================"

"${PYTHON_BIN}" scripts/run_internship_far_fedfdp.py \
  --run \
  --lane "${LANE}" \
  "${SCENARIO_ARGS[@]}" \
  --job-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --device "${DEVICE}" \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --resume

