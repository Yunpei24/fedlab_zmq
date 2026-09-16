#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python}"
MATRIX="configs/scpfar/paper1/s0_step1b_fmnist_geometry_dev.yaml"
OUTPUT_ROOT="results/scfar_paper1_fmnist_step1b_geometry_dev_v1"

"$PYTHON_BIN" -u scripts/run_scfar_paper1.py \
  --validate \
  --matrix "$MATRIX"

for job_index in $(seq 0 9); do
  "$PYTHON_BIN" -u scripts/run_scfar_paper1.py \
    --run \
    --matrix "$MATRIX" \
    --pilot-rounds 20 \
    --job-index "$job_index" \
    --device mps \
    --data-root data \
    --output-root "$OUTPUT_ROOT" \
    --resume
done
