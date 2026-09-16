#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python}"
DEVICE="${DEVICE:-mps}"
MATRIX="configs/dt_ldp_far/stage26b_trmean_nnm_anchor_confirmatory.yaml"
OUTPUT_ROOT="results/dt_ldp_far/stage26b_trmean_nnm_anchor_confirmatory_v1"
LOG_DIR="logs/dt_ldp_far"
LOG_FILE="$LOG_DIR/stage26b_trmean_nnm_anchor_confirmatory_mps.log"

mkdir -p "$LOG_DIR"

for job_index in $(seq 0 26); do
  "$PYTHON_BIN" -u scripts/run_dt_ldp_far.py \
    --run \
    --matrix "$MATRIX" \
    --job-index "$job_index" \
    --device "$DEVICE" \
    --data-root data \
    --output-root "$OUTPUT_ROOT" \
    --resume 2>&1 | tee -a "$LOG_FILE"
done
