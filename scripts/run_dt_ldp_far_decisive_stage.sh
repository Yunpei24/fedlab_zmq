#!/bin/bash
# Run one frozen decisive matrix sequentially.  The stage-0 gate is the only
# matrix that should be run before a geometry is accepted.
set -euo pipefail

MATRIX_PATH="${1:-configs/dt_ldp_far/decisive_stage0_geometry.yaml}"
DEVICE_NAME="${DEVICE:-mps}"
DATA_PATH="${DATA_ROOT:-data}"
RESULT_PATH="${OUTPUT_ROOT:-results/dt_ldp_far/decisive}"
PYTHON_PATH="${PYTHON_BIN:-venv/bin/python}"

mkdir -p logs/dt_ldp_far
TASK_COUNT="$(${PYTHON_PATH} scripts/run_dt_ldp_far.py --list --matrix "${MATRIX_PATH}" | wc -l | tr -d ' ')"
if [[ "${TASK_COUNT}" -lt 1 ]]; then
  echo "No task found in ${MATRIX_PATH}" >&2
  exit 2
fi

MATRIX_STEM="$(basename "${MATRIX_PATH}" .yaml)"
LOG_PATH="logs/dt_ldp_far/${MATRIX_STEM}_${DEVICE_NAME}.log"
for ((TASK_INDEX=0; TASK_INDEX<TASK_COUNT; TASK_INDEX++)); do
  "${PYTHON_PATH}" -u scripts/run_dt_ldp_far.py \
    --run \
    --matrix "${MATRIX_PATH}" \
    --job-index "${TASK_INDEX}" \
    --device "${DEVICE_NAME}" \
    --data-root "${DATA_PATH}" \
    --output-root "${RESULT_PATH}" \
    --resume 2>&1 | tee -a "${LOG_PATH}"
done
