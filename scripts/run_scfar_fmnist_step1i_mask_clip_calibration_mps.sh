#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-$repo_root/venv/bin/python}"
matrix="configs/scpfar/paper1/s0_step1i_fmnist_mask_clip_calibration.yaml"
output_root="${SCFAR_STEP1I_OUTPUT_ROOT:-results/scfar_paper1_fmnist_step1i_mask_clip_calibration_v1}"
device="${SCFAR_STEP1I_DEVICE:-mps}"

"$python_bin" -u scripts/run_scfar_paper1.py --validate --matrix "$matrix"

for job_index in $(seq 0 7); do
  "$python_bin" -u scripts/run_scfar_paper1.py \
    --run \
    --matrix "$matrix" \
    --job-index "$job_index" \
    --device "$device" \
    --data-root data \
    --output-root "$output_root" \
    --resume
done
