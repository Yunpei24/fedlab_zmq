#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT}/venv/bin/python"
MATRIX="${ROOT}/configs/dt_ldp_far/stage20r_fmnist_mechanism_replication.yaml"
OUTPUT="${ROOT}/results/dt_ldp_far/stage20r_fmnist_mechanism_replication_v1"
LOG_DIR="${ROOT}/logs/dt_ldp_far"
LOG_FILE="${LOG_DIR}/stage20r_fmnist_mechanism_replication_mps.log"

mkdir -p "${LOG_DIR}"

"${PYTHON_BIN}" - <<'PY'
import torch

if not torch.backends.mps.is_available():
    raise SystemExit("MPS is not available in this execution environment")
print(f"torch={torch.__version__} mps_available=True")
PY

for job_index in $(seq 0 11); do
  "${PYTHON_BIN}" -u "${ROOT}/scripts/run_dt_ldp_far.py" \
    --run \
    --matrix "${MATRIX}" \
    --job-index "${job_index}" \
    --device mps \
    --data-root "${ROOT}/data" \
    --output-root "${OUTPUT}" \
    --resume 2>&1 | tee -a "${LOG_FILE}"
done
