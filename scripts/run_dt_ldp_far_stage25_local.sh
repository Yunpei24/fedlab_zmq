#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python}"
DEVICE="${DEVICE:-mps}"
MATRIX="configs/dt_ldp_far/stage25_robust_anchor_containment_validation.yaml"
OUTPUT_ROOT="results/dt_ldp_far/stage25_robust_anchor_containment_validation_v1"
LOG_DIR="logs/dt_ldp_far"
LOG_FILE="$LOG_DIR/stage25_robust_anchor_containment_validation_mps.log"

mkdir -p "$LOG_DIR"

for job_index in $(seq 0 35); do
  "$PYTHON_BIN" -u scripts/run_dt_ldp_far.py \
    --run \
    --matrix "$MATRIX" \
    --job-index "$job_index" \
    --device "$DEVICE" \
    --data-root data \
    --output-root "$OUTPUT_ROOT" \
    --resume 2>&1 | tee -a "$LOG_FILE"
done
