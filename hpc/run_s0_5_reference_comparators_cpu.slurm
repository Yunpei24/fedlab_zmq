#!/bin/bash
#SBATCH --job-name=scfar_s0_5_refs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=compute
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# S0.5: CPU-only replace-one audit of mean, CM, trimmed mean and RFA.

set -euo pipefail

REPO_DIR="${REPO_DIR:-${HOME}/fedlab_zmq}"
PROJECT_DIR="${PROJECT_DIR:-${HOME}/lustre/manapy-um6p-st-msda-1wabcjwe938}"
WORK_ROOT="${WORK_ROOT:-${PROJECT_DIR}/users/${USER}/fedlab_zmq}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/results/scfar/sensitivity_s0_5}"
CONDA_ENV="${CONDA_ENV:-fedlab-zmq}"

module purge
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_DIR}"

python -u scripts/run_scfar_reference_comparators.py \
  --n 5,10,20,40 \
  --methods mean,cm,trmean,rfa \
  --dimension 64 \
  --trials 200 \
  --clip 1.0 \
  --distance-clip 2.0 \
  --alpha 1.0 \
  --kappa-w 2.0 \
  --byzantine-fraction 0.2 \
  --anchor-mode zero \
  --seed 42 \
  --output "${OUTPUT_ROOT}"
