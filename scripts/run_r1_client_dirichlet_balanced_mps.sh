#!/bin/zsh

# Paired R1 partition ablation.  All v4 R1 choices are preserved except for
# the client-centric Dirichlet split with controlled local dataset sizes.

set -euo pipefail

FEDLAB_REPO_DIR="${0:A:h:h}"
FEDLAB_PYTHON="${FEDLAB_REPO_DIR}/venv/bin/python"
FEDLAB_LOG_DIR="${FEDLAB_REPO_DIR}/results/reproductions/internship_far_fedfdp/logs"
FEDLAB_LOG_FILE="${FEDLAB_LOG_DIR}/r1_client_dirichlet_balanced_v1_mps.log"

mkdir -p "${FEDLAB_LOG_DIR}"
cd "${FEDLAB_REPO_DIR}"

exec >> "${FEDLAB_LOG_FILE}" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting balanced client-Dirichlet R1 on MPS"
"${FEDLAB_PYTHON}" -u scripts/run_internship_far_fedfdp.py \
  --run \
  --lane faithful \
  --scenario exp1_fairness_no_attack_client_dirichlet_balanced \
  --device mps \
  --resume
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Campaign completed"

