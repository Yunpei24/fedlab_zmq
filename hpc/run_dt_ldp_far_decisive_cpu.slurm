#!/bin/bash
# Generic staged launcher. Submit with an explicit, validated array bound, e.g.:
# sbatch --account="$FEDLAB_ACCOUNT" --array=0-11%4 \
#   --export=ALL,MATRIX=configs/dt_ldp_far/decisive_stage1_alpha_delay_n25.yaml \
#   hpc/run_dt_ldp_far_decisive_cpu.slurm
#
# The file intentionally has no #SBATCH --array directive: every decisive
# stage has a different frozen cardinality and must be submitted explicitly.
#SBATCH --job-name=dtldp_decisive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --qos=default
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${HOME}/fedlab_zmq}"
PROJECT_DIR="${PROJECT_DIR:-${HOME}/lustre/manapy-um6p-st-msda-1wabcjwe938}"
WORK_ROOT="${WORK_ROOT:-${PROJECT_DIR}/users/${USER}/fedlab_zmq}"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/results/dt_ldp_far/decisive}"
CONDA_ENV="${CONDA_ENV:-fedlab-zmq}"
MATRIX="${MATRIX:?MATRIX must name one validated decisive YAML matrix}"

module purge
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${DATA_ROOT}" "${OUTPUT_ROOT}"
cd "${REPO_DIR}"

python -u scripts/run_dt_ldp_far.py \
  --run \
  --matrix "${MATRIX}" \
  --job-index "${SLURM_ARRAY_TASK_ID}" \
  --device cpu \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --resume
