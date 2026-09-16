#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-$repo_root/venv/bin/python}"
matrix="configs/scpfar/paper1/s0_step1ir_fmnist_mask_clip_refinement.yaml"
output_root="${SCFAR_STEP1IR_OUTPUT_ROOT:-results/scfar_paper1_fmnist_step1ir_mask_clip_refinement_v1}"
device="${SCFAR_STEP1IR_DEVICE:-mps}"

"$python_bin" -u scripts/analyze_scfar_mask_clip_calibration.py
"$python_bin" -u scripts/prepare_scfar_mask_clip_refinement.py --output "$matrix"
"$python_bin" -u scripts/run_scfar_paper1.py --validate --matrix "$matrix"

task_count="$($python_bin -u scripts/run_scfar_paper1.py \
  --list --matrix "$matrix" --output-root "$output_root" | wc -l | tr -d ' ')"

for ((job_index = 0; job_index < task_count; job_index++)); do
  "$python_bin" -u scripts/run_scfar_paper1.py \
    --run \
    --matrix "$matrix" \
    --job-index "$job_index" \
    --device "$device" \
    --data-root data \
    --output-root "$output_root" \
    --resume
done

"$python_bin" -u scripts/analyze_scfar_mask_clip_calibration.py \
  --matrix "$matrix" \
  --results-root "$output_root" \
  --report output/analysis/SC_FAR_Step1IR_Mask_Clip_Refinement.md \
  --selection "$output_root/selection.json"
