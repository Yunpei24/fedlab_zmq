#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-$repo_root/venv/bin/python}"
matrix="configs/scpfar/paper1/s0_step1j_fmnist_joint_mask_clip_horizon.yaml"
output_root="${SCFAR_STEP1J_OUTPUT_ROOT:-results/scfar_paper1_fmnist_step1j_joint_horizon_v1}"
device="${SCFAR_STEP1J_DEVICE:-mps}"

"$python_bin" -u scripts/analyze_scfar_mask_clip_calibration.py \
  --matrix configs/scpfar/paper1/s0_step1ir_fmnist_mask_clip_refinement.yaml \
  --results-root results/scfar_paper1_fmnist_step1ir_mask_clip_refinement_v1 \
  --report output/analysis/SC_FAR_Step1IR_Mask_Clip_Refinement.md \
  --selection results/scfar_paper1_fmnist_step1ir_mask_clip_refinement_v1/selection.json
"$python_bin" -u scripts/prepare_scfar_joint_horizon_screen.py --output "$matrix"
"$python_bin" -u scripts/run_scfar_paper1.py --validate --matrix "$matrix"

task_count="$($python_bin -u scripts/run_scfar_paper1.py \
  --list --matrix "$matrix" --output-root "$output_root" | wc -l | tr -d ' ')"

for rounds in 5 10 20; do
  for ((job_index = 0; job_index < task_count; job_index++)); do
    "$python_bin" -u scripts/run_scfar_paper1.py \
      --run \
      --matrix "$matrix" \
      --job-index "$job_index" \
      --pilot-rounds "$rounds" \
      --device "$device" \
      --data-root data \
      --output-root "$output_root" \
      --resume
  done
done

"$python_bin" -u scripts/analyze_scfar_joint_horizon.py \
  --matrix "$matrix" \
  --results-root "$output_root" \
  --report output/analysis/SC_FAR_Step1J_Joint_Horizon_Screen.md \
  --selection "$output_root/selection.json"
