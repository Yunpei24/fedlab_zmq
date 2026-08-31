#!/bin/bash
#SBATCH --job-name=scfar_p1_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Full-update paper-1 worker. The array range is deliberately supplied at
# submission time because S1, S2, S3 and S4 have different cardinalities.

set -euo pipefail

REPO_DIR="${REPO_DIR:-${HOME}/fedlab_zmq}"
PROJECT_DIR="${PROJECT_DIR:-${HOME}/lustre/manapy-um6p-st-msda-1wabcjwe938}"
WORK_ROOT="${WORK_ROOT:-${PROJECT_DIR}/users/${USER}/fedlab_zmq}"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/results/scfar_paper1}"
CONDA_ENV="${CONDA_ENV:-fedlab-zmq}"
SCFAR_MATRIX="${SCFAR_MATRIX:-s1_reference_tradeoff.yaml}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Submit this worker with sbatch --array=<first>-<last>%<concurrency>." >&2
  exit 2
fi

module purge
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p "${DATA_ROOT}" "${OUTPUT_ROOT}"
cd "${REPO_DIR}"

ARGS=(
  --run
  --matrix "configs/scpfar/paper1/${SCFAR_MATRIX}"
  --job-index "${SLURM_ARRAY_TASK_ID}"
  --device cpu
  --data-root "${DATA_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --resume
)
if [[ -n "${SCFAR_PILOT_ROUNDS:-}" ]]; then
  ARGS+=(--pilot-rounds "${SCFAR_PILOT_ROUNDS}")
fi

python -u scripts/run_scfar_paper1.py "${ARGS[@]}"
