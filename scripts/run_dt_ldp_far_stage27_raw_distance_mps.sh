#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${DT_STAGE27_PYTHON:-$repo_root/venv/bin/python}"
matrix="configs/dt_ldp_far/stage27_raw_distance_n25.yaml"
output_root="${DT_STAGE27_OUTPUT_ROOT:-results/dt_ldp_far/decisive}"
log_path="${DT_STAGE27_LOG_PATH:-logs/dt_ldp_far/stage27_raw_distance_n25_mps.log}"
device="${DT_STAGE27_DEVICE:-mps}"

mkdir -p "$(dirname "$log_path")"
"$python_bin" -u scripts/run_dt_ldp_far.py --validate --matrix "$matrix"

for job_index in $(seq 0 14); do
  "$python_bin" -u scripts/run_dt_ldp_far.py \
    --run \
    --matrix "$matrix" \
    --job-index "$job_index" \
    --device "$device" \
    --data-root data \
    --output-root "$output_root" \
    --resume 2>&1 | tee -a "$log_path"
done
