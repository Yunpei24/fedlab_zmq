#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-$repo_root/venv/bin/python}"
matrix="configs/scpfar/paper1/s0_step1h_fmnist_masked_dpfedavg_eps10_screen.yaml"
output_root="${SCFAR_STEP1H_OUTPUT_ROOT:-results/scfar_paper1_fmnist_step1h_masked_dpfedavg_v1}"
device="${SCFAR_STEP1H_DEVICE:-mps}"

"$python_bin" -u scripts/run_scfar_paper1.py --validate --matrix "$matrix"

for job_index in $(seq 0 5); do
  "$python_bin" -u scripts/run_scfar_paper1.py \
    --run \
    --matrix "$matrix" \
    --job-index "$job_index" \
    --device "$device" \
    --data-root data \
    --output-root "$output_root" \
    --resume
done
