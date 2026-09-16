#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${DT_STAGE13A_PYTHON:-$repo_root/venv/bin/python}"
matrix="configs/dt_ldp_far/decisive_stage13a_generation6_temporal_score_n25_screen.yaml"
output_root="${DT_STAGE13A_OUTPUT_ROOT:-results/dt_ldp_far/decisive}"
log_path="${DT_STAGE13A_LOG_PATH:-logs/dt_ldp_far/stage13a_generation6_temporal_score_n25.log}"
start_index="${DT_STAGE13A_START_INDEX:-0}"
end_index="${DT_STAGE13A_END_INDEX:-7}"
device="${DT_STAGE13A_DEVICE:-mps}"

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
