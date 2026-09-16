#!/usr/bin/env bash
set -euo pipefail

# Sequential local queue for the five promoted DT-LDP-FAR screens.
# The order is deliberate: partition robustness is checked at n=10 before
# spending compute on the three n=25 screens. Every task is independently
# resumable through its metrics.json artifact.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${repo_root}/venv/bin/python"
data_root="${repo_root}/data"
output_root="${repo_root}/results/dt_ldp_far/decisive"
log_dir="${repo_root}/logs/dt_ldp_far"
log_file="${log_dir}/decisive_partition_and_n25_queue_mps.log"

mkdir -p "${log_dir}"

matrices=(
  "configs/dt_ldp_far/decisive_stage0_geometry_final_confirm_class_dirichlet.yaml"
  "configs/dt_ldp_far/decisive_stage0_alpha_stress_n10_c4_u28_class_dirichlet.yaml"
  "configs/dt_ldp_far/decisive_stage1_alpha_delay_n25.yaml"
  "configs/dt_ldp_far/decisive_stage1_server_clip_ablation_n25_c4.yaml"
  "configs/dt_ldp_far/decisive_stage1_references_attacks_n25.yaml"
)

task_counts=(6 8 12 6 30)

for matrix_index in "${!matrices[@]}"; do
  matrix="${matrices[$matrix_index]}"
  task_count="${task_counts[$matrix_index]}"

  printf '\n[%s] Validating %s (%s tasks)\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "${matrix}" "${task_count}" \
    | tee -a "${log_file}"

  "${python_bin}" "${repo_root}/scripts/run_dt_ldp_far.py" \
    --validate \
    --matrix "${repo_root}/${matrix}" \
    | tee -a "${log_file}"

  for task_index in $(seq 0 $((task_count - 1))); do
    printf '[%s] %s task %s/%s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" \
      "${matrix}" \
      "$((task_index + 1))" \
      "${task_count}" \
      | tee -a "${log_file}"

    "${python_bin}" -u "${repo_root}/scripts/run_dt_ldp_far.py" \
      --run \
      --matrix "${repo_root}/${matrix}" \
      --job-index "${task_index}" \
      --device mps \
      --data-root "${data_root}" \
      --output-root "${output_root}" \
      --resume \
      2>&1 | tee -a "${log_file}"
  done
done

printf '\n[%s] Sequential DT-LDP-FAR queue completed.\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" \
  | tee -a "${log_file}"
