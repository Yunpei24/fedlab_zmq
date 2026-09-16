#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${DT_STAGE9_PYTHON:-$repo_root/venv/bin/python}"
matrix="configs/dt_ldp_far/decisive_stage9_fixed_without_replacement_n25.yaml"
output_root="${DT_STAGE9_OUTPUT_ROOT:-results/dt_ldp_far/decisive_stage9_fixed_wor_mps}"
log_path="${DT_STAGE9_LOG_PATH:-logs/dt_ldp_far/stage9_fixed_wor_n25_mps.log}"
start_index="${DT_STAGE9_START_INDEX:-0}"
end_index="${DT_STAGE9_END_INDEX:-35}"
device="${DT_STAGE9_DEVICE:-mps}"

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
