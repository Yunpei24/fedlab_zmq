#!/usr/bin/env bash
set -euo pipefail

# Conditional stage-4 queue. The 80/120-round matrix is launched only when
# the n=25 end-to-end transfer gate finds a genuinely informative paired cell.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_root}/venv/bin/python}"
data_root="${DATA_ROOT:-${repo_root}/data}"
output_root="${OUTPUT_ROOT:-${repo_root}/results/dt_ldp_far/decisive}"
log_dir="${repo_root}/logs/dt_ldp_far"
log_file="${log_dir}/decisive_stage4_queue_mps.log"

mkdir -p "${log_dir}"

run_matrix() {
  local matrix="$1"
  local count
  count="$(${python_bin} "${repo_root}/scripts/run_dt_ldp_far.py" \
    --list --matrix "${repo_root}/${matrix}" | wc -l | tr -d ' ')"
  "${python_bin}" "${repo_root}/scripts/run_dt_ldp_far.py" \
    --validate --matrix "${repo_root}/${matrix}" | tee -a "${log_file}"
  for ((index=0; index<count; index++)); do
    "${python_bin}" -u "${repo_root}/scripts/run_dt_ldp_far.py" \
      --run \
      --matrix "${repo_root}/${matrix}" \
      --job-index "${index}" \
      --device mps \
      --data-root "${data_root}" \
      --output-root "${output_root}" \
      --resume 2>&1 | tee -a "${log_file}"
  done
}

run_selected_matrix() {
  local matrix="$1"
  local experiment="$2"
  local geometry="$3"
  local tilt="${4:-}"
  local list_args=(
    --list
    --matrix "${repo_root}/${matrix}"
    --experiment "${experiment}"
    --geometry "${geometry}"
  )
  if [[ -n "${tilt}" ]]; then
    list_args+=(--tilt "${tilt}")
  fi
  "${python_bin}" "${repo_root}/scripts/run_dt_ldp_far.py" \
    --validate --matrix "${repo_root}/${matrix}" >/dev/null
  while IFS=$'\t' read -r index _; do
    [[ -n "${index}" ]] || continue
    local run_args=(
      -u "${repo_root}/scripts/run_dt_ldp_far.py"
      --run
      --matrix "${repo_root}/${matrix}"
      --experiment "${experiment}"
      --geometry "${geometry}"
      --job-index "${index}"
      --device mps
      --data-root "${data_root}"
      --output-root "${output_root}"
      --resume
    )
    if [[ -n "${tilt}" ]]; then
      run_args+=(--tilt "${tilt}")
    fi
    "${python_bin}" "${run_args[@]}" 2>&1 | tee -a "${log_file}"
  done < <("${python_bin}" "${repo_root}/scripts/run_dt_ldp_far.py" "${list_args[@]}")
}

screen_matrix="configs/dt_ldp_far/decisive_stage4_mechanistic_transfer_screen_n25.yaml"
server_clip_matrix="configs/dt_ldp_far/decisive_stage4_server_clip_ablation_n25.yaml"
references_matrix="configs/dt_ldp_far/decisive_stage4_current_delay_references_n25.yaml"
long_matrix="configs/dt_ldp_far/decisive_stage4_long_horizon_n25.yaml"

if [[ "${SKIP_SCREEN:-0}" != "1" ]]; then
  run_matrix "${screen_matrix}"
fi

set +e
"${python_bin}" "${repo_root}/scripts/analyze_dt_ldp_far_n25_mechanistic_transfer.py" \
  2>&1 | tee -a "${log_file}"
gate_status=${PIPESTATUS[0]}
set -e
if [[ "${gate_status}" -ne 0 && "${gate_status}" -ne 2 ]]; then
  exit "${gate_status}"
fi

# These two scientific ablations do not require a positive delay gate.
run_matrix "${server_clip_matrix}"
run_matrix "${references_matrix}"

if [[ "${gate_status}" -eq 0 ]]; then
  gate_summary="${repo_root}/output/analysis/dt_ldp_far_n25_mechanistic_transfer_v1/summary.json"
  selected_geometry="$(jq -r '.selected_cell.geometry' "${gate_summary}")"
  selected_tilt="$(jq -r '.selected_cell.tilt' "${gate_summary}")"
  run_selected_matrix \
    "${long_matrix}" \
    "E11_uniform_same_path_t80_t120_n25" \
    "${selected_geometry}"
  run_selected_matrix \
    "${long_matrix}" \
    "E11_alpha_stress_long_current_delay_n25" \
    "${selected_geometry}" \
    "${selected_tilt}"
else
  echo "Transfer gate negative: the 80/120-round matrix was not launched." \
    | tee -a "${log_file}"
fi
