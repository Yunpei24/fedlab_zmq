#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python}"
MATRIX="configs/scpfar/paper1/s2_full_update_ablations.yaml"
OUTPUT_ROOT="results/scfar_paper1_fmnist_pilot_v1"

for job_index in $(seq 0 8); do
  "$PYTHON_BIN" -u scripts/run_scfar_paper1.py \
    --run \
    --matrix "$MATRIX" \
    --experiment s2b_central_dp_chain \
    --scenario fmnist_lenet5_b01 \
    --method central_dp_fedavg_exact,scfar_dp_2c,scfar_dp_certified \
    --partition-seed 101 \
    --training-seed 28,36,54 \
    --pilot-rounds 20 \
    --job-index "$job_index" \
    --device mps \
    --data-root data \
    --output-root "$OUTPUT_ROOT" \
    --resume
done
