#!/usr/bin/env bash
set -euo pipefail

# Sequential gate: execute only after the Step-1D analysis has frozen which
# score/tilt candidates are eligible for private confirmation.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-$repo_root/venv/bin/python}"
matrix="configs/scpfar/paper1/s0_step1e_fmnist_central_dp_raw_confirmation.yaml"
output_root="${SCFAR_STEP1E_OUTPUT_ROOT:-results/scfar_paper1_fmnist_step1e_dp_v1}"
device="${SCFAR_STEP1E_DEVICE:-mps}"

"$python_bin" -u scripts/run_scfar_paper1.py --validate --matrix "$matrix"

for job_index in $(seq 0 14); do
  "$python_bin" -u scripts/run_scfar_paper1.py \
    --run \
    --matrix "$matrix" \
    --pilot-rounds 20 \
    --job-index "$job_index" \
    --device "$device" \
    --data-root data \
    --output-root "$output_root" \
    --resume
done
