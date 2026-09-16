#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${DT_STAGE12_PYTHON:-$repo_root/venv/bin/python}"
matrix="configs/dt_ldp_far/decisive_stage12_generation5_tier_rank_n25_screen.yaml"
output_root="${DT_STAGE12_OUTPUT_ROOT:-results/dt_ldp_far/decisive}"
log_path="${DT_STAGE12_LOG_PATH:-logs/dt_ldp_far/stage12_generation5_tier_rank_n25.log}"
start_index="${DT_STAGE12_START_INDEX:-0}"
end_index="${DT_STAGE12_END_INDEX:-31}"
device="${DT_STAGE12_DEVICE:-mps}"

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
