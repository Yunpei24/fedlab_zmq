#!/bin/bash
#SBATCH --job-name=dmd_phase_a_cpu
#SBATCH --array=0-629%4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=compute
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# CPU-first DMD Phase A campaign.
#
# Full participation, 630 tasks (5 partitions x 3 training seeds x 7 arms):
#
#   sbatch --account="$FEDLAB_ACCOUNT" hpc/run_dmd_phase_a_cpu.slurm
#
# Smoke, 28 tasks:
#
#   sbatch --account="$FEDLAB_ACCOUNT" \
#     --array=0-27%4 \
#     --export=ALL,STAGE=smoke \
#     hpc/run_dmd_phase_a_cpu.slurm
#
# Partial follow-ups, 420 tasks each:
#
#   sbatch --account="$FEDLAB_ACCOUNT" \
#     --array=0-419%4 \
#     --export=ALL,STAGE=phase_a_partial_no_dropout \
#     hpc/run_dmd_phase_a_cpu.slurm
#
#   sbatch --account="$FEDLAB_ACCOUNT" \
#     --array=0-419%4 \
#     --export=ALL,STAGE=phase_a_partial_dropout \
#     hpc/run_dmd_phase_a_cpu.slurm

set -euo pipefail

REPO_DIR="${REPO_DIR:-${HOME}/fedlab_zmq}"
PROJECT_DIR="${PROJECT_DIR:-${HOME}/lustre/manapy-um6p-st-msda-1wabcjwe938}"
WORK_ROOT="${WORK_ROOT:-${PROJECT_DIR}/users/${USER}/fedlab_zmq}"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/results/toubkal_dmd_cpu}"
DASHBOARD_OUTPUT_ROOT="${DASHBOARD_OUTPUT_ROOT:-${OUTPUT_ROOT}/dashboard_exports}"
DMD_MATRIX="${DMD_MATRIX:-${REPO_DIR}/configs/dmd/toubkal_phase_a.yaml}"
STAGE="${STAGE:-phase_a_full}"
CONDA_ENV="${CONDA_ENV:-fedlab-zmq}"
PYTHON_BIN="${PYTHON_BIN:-python}"

module purge
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "${DATA_ROOT}" "${OUTPUT_ROOT}" "${DASHBOARD_OUTPUT_ROOT}"
cd "${REPO_DIR}"

PILOT_ARGS=()
if [[ -n "${PILOT_ROUNDS:-}" ]]; then
  PILOT_ARGS=(--pilot-rounds "${PILOT_ROUNDS}")
fi

echo "============================================================"
echo "DMD non-private Toubkal CPU campaign"
echo "Job       : ${SLURM_JOB_ID:-local}[${SLURM_ARRAY_TASK_ID:-0}]"
echo "Stage     : ${STAGE}"
echo "Device    : cpu"
echo "Data root : ${DATA_ROOT}"
echo "Output    : ${OUTPUT_ROOT}"
echo "============================================================"

"${PYTHON_BIN}" -u scripts/run_dmd_toubkal.py \
  --run \
  --matrix "${DMD_MATRIX}" \
  --stage "${STAGE}" \
  --job-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --device cpu \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --dashboard-output-root "${DASHBOARD_OUTPUT_ROOT}" \
  --python-bin "${PYTHON_BIN}" \
  --resume \
  "${PILOT_ARGS[@]}"
