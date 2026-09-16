#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${DT_STAGE6_PYTHON:-$repo_root/venv/bin/python}"
matrix="configs/dt_ldp_far/decisive_stage6_noise_aware_score_n25.yaml"
output_root="results/dt_ldp_far/decisive"
log_path="logs/dt_ldp_far/stage6_noise_aware_score_n25_mps.log"
start_index="${DT_STAGE6_START_INDEX:-0}"
end_index="${DT_STAGE6_END_INDEX:-29}"

mkdir -p "$(dirname "$log_path")"
for job_index in $(seq "$start_index" "$end_index"); do
  "$python_bin" -u scripts/run_dt_ldp_far.py \
    --run \
    --matrix "$matrix" \
    --job-index "$job_index" \
    --device mps \
    --data-root data \
    --output-root "$output_root" \
    --resume 2>&1 | tee -a "$log_path"
done
