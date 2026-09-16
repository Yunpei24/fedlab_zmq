#!/bin/zsh

# Paired R3-style FedFDP diagnostic on Apple MPS. The raw arm is deliberately
# not DP-certified; see the matrix and emitted privacy_guarantee_status.

set -euo pipefail

FEDLAB_REPO_DIR="${0:A:h:h}"
FEDLAB_PYTHON="${FEDLAB_REPO_DIR}/venv/bin/python"
FEDLAB_LOG_DIR="${FEDLAB_REPO_DIR}/results/reproductions/internship_far_fedfdp/logs"
FEDLAB_LOG_FILE="${FEDLAB_LOG_DIR}/fedfdp_negative_scale_ablation_v1_mps.log"

mkdir -p "${FEDLAB_LOG_DIR}"
cd "${FEDLAB_REPO_DIR}"

exec >> "${FEDLAB_LOG_FILE}" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting FedFDP negative-scale ablation on MPS"
"${FEDLAB_PYTHON}" -u scripts/run_internship_far_fedfdp.py \
  --run \
  --lane paper-grade \
  --scenario exp3_fedfdp_negative_scale_ablation \
  --device mps \
  --resume
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Campaign completed"
