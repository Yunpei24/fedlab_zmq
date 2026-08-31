#!/bin/zsh

# Persistent local launcher for the corrected internship reproduction.
# The scenario order is deliberate: R3, then R2, then R1.

set -euo pipefail

FEDLAB_REPO_DIR="${0:A:h:h}"
FEDLAB_PYTHON="${FEDLAB_REPO_DIR}/venv/bin/python"
FEDLAB_LOG_DIR="${FEDLAB_REPO_DIR}/results/reproductions/internship_far_fedfdp/logs"
FEDLAB_LOG_FILE="${FEDLAB_LOG_DIR}/algorithm_fidelity_v4_r3_r2_r1_mps.log"

mkdir -p "${FEDLAB_LOG_DIR}"
cd "${FEDLAB_REPO_DIR}"

exec >> "${FEDLAB_LOG_FILE}" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting algorithm_fidelity_v4 on MPS"

for FEDLAB_SCENARIO in \
  exp3_privacy_fairness \
  exp2_fairness_robustness \
  exp1_fairness_no_attack
do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${FEDLAB_SCENARIO}"
  "${FEDLAB_PYTHON}" -u scripts/run_internship_far_fedfdp.py \
    --run \
    --lane faithful \
    --scenario "${FEDLAB_SCENARIO}" \
    --device mps \
    --resume
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed ${FEDLAB_SCENARIO}"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Campaign completed"
