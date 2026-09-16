#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

python_bin="${PYTHON_BIN:-${repo_root}/venv/bin/python}"
device="${DEVICE:-mps}"
matrix="configs/dt_ldp_far/decisive_stage5_end_to_end_stress_discovery_n25.yaml"
output_root="results/dt_ldp_far/decisive"
log_dir="logs/dt_ldp_far"
log_file="${log_dir}/stage5_end_to_end_stress_mps.log"

mkdir -p "${log_dir}"

"${python_bin}" scripts/run_dt_ldp_far.py \
  --validate \
  --matrix "${matrix}" \
  --output-root "${output_root}"

for job_index in $(seq 0 23); do
  "${python_bin}" -u scripts/run_dt_ldp_far.py \
    --run \
    --matrix "${matrix}" \
    --job-index "${job_index}" \
    --device "${device}" \
    --data-root data \
    --output-root "${output_root}" \
    --python-bin "${python_bin}" \
    --resume 2>&1 | tee -a "${log_file}"
done

set +e
"${python_bin}" scripts/analyze_dt_ldp_far_stage5_stress.py 2>&1 | tee -a "${log_file}"
analysis_status=${PIPESTATUS[0]}
set -e
if [[ ${analysis_status} -eq 2 ]]; then
  echo "Stage-5 screen completed: no candidate passed every pre-registered seed gate." \
    | tee -a "${log_file}"
elif [[ ${analysis_status} -ne 0 ]]; then
  exit "${analysis_status}"
fi
