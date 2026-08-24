#!/bin/bash
#SBATCH --job-name=dmd_phase_a
#SBATCH --array=0-179
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-dmd-%A_%a.out
#SBATCH --error=slurm-dmd-%A_%a.err

# Default: Phase A full participation, 180 tasks (one method per task).
#
#   sbatch hpc/run_dmd_phase_a.slurm
#
# Smoke (24 tasks):
#
#   sbatch --array=0-23 --export=ALL,STAGE=smoke hpc/run_dmd_phase_a.slurm
#
# Partial follow-ups (120 tasks each):
#
#   sbatch --array=0-119 --export=ALL,STAGE=phase_a_partial_no_dropout hpc/run_dmd_phase_a.slurm
#   sbatch --array=0-119 --export=ALL,STAGE=phase_a_partial_dropout hpc/run_dmd_phase_a.slurm

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DMD_MATRIX="${DMD_MATRIX:-${REPO_DIR}/configs/dmd/toubkal_phase_a.yaml}"
STAGE="${STAGE:-phase_a_full}"
DEVICE="${DEVICE:-cuda}"
DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/results/toubkal_dmd}"
DASHBOARD_OUTPUT_ROOT="${DASHBOARD_OUTPUT_ROOT:-${OUTPUT_ROOT}/dashboard_exports}"

cd "${REPO_DIR}"

if [[ -n "${MODULE_SETUP:-}" ]]; then
  eval "${MODULE_SETUP}"
fi
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  source "${VENV_ACTIVATE}"
fi

PILOT_ARGS=()
if [[ -n "${PILOT_ROUNDS:-}" ]]; then
  PILOT_ARGS=(--pilot-rounds "${PILOT_ROUNDS}")
fi

echo "============================================================"
echo "DMD non-private Toubkal campaign"
echo "Job       : ${SLURM_JOB_ID:-local}[${SLURM_ARRAY_TASK_ID:-0}]"
echo "Stage     : ${STAGE}"
echo "Device    : ${DEVICE}"
echo "Data root : ${DATA_ROOT}"
echo "Output    : ${OUTPUT_ROOT}"
echo "============================================================"

"${PYTHON_BIN}" scripts/run_dmd_toubkal.py \
  --run \
  --matrix "${DMD_MATRIX}" \
  --stage "${STAGE}" \
  --job-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --device "${DEVICE}" \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --dashboard-output-root "${DASHBOARD_OUTPUT_ROOT}" \
  --python-bin "${PYTHON_BIN}" \
  --resume \
  "${PILOT_ARGS[@]}"

