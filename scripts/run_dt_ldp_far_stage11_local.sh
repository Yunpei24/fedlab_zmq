#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${DT_STAGE11_PYTHON:-$repo_root/venv/bin/python}"
matrix="configs/dt_ldp_far/decisive_stage11_generation4_scores_n25_screen.yaml"
output_root="${DT_STAGE11_OUTPUT_ROOT:-results/dt_ldp_far/decisive}"
log_path="${DT_STAGE11_LOG_PATH:-logs/dt_ldp_far/stage11_generation4_scores_n25_mps.log}"
start_index="${DT_STAGE11_START_INDEX:-0}"
end_index="${DT_STAGE11_END_INDEX:-31}"
device="${DT_STAGE11_DEVICE:-mps}"

mkdir -p "$(dirname "$log_path")"
for job_index in $(seq "$start_index" "$end_index"); do
  "$python_bin" -u scripts/run_dt_ldp_far.py \
    --run \
    --matrix "$matrix" \
    --job-index "$job_index" \
    --device "$device" \
    --data-root data \
    --output-root "$output_root" \
    --resume 2>&1 | tee -a "$log_path"
done
