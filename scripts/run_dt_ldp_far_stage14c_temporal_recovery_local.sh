#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATRIX="${ROOT}/configs/dt_ldp_far/decisive_stage14c_temporal_poison_recovery_screen.yaml"
OUTPUT="${ROOT}/results/dt_ldp_far/stage14c_temporal_poison_recovery_screen_v1"
LOG_DIR="${ROOT}/logs/dt_ldp_far"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

"${ROOT}/venv/bin/python" -c \
  'import torch; assert torch.backends.mps.is_available(); print(f"torch={torch.__version__} mps_available=True")'

for job_index in $(seq 0 8); do
  "${ROOT}/venv/bin/python" -u scripts/run_dt_ldp_far.py \
    --run \
    --matrix "${MATRIX}" \
    --job-index "${job_index}" \
    --device mps \
    --data-root data \
    --output-root "${OUTPUT}" \
    --resume 2>&1 | tee -a "${LOG_DIR}/stage14c_temporal_recovery_mps.log"
done
