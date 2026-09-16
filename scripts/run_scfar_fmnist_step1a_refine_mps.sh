#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-$repo_root/venv/bin/python}"
matrix="configs/scpfar/paper1/s0_step1a_fmnist_health_refine_dev.yaml"
output_root="results/scfar_paper1_fmnist_step1a_refine_dev_v1"

"$python_bin" -u scripts/run_scfar_paper1.py \
  --validate \
  --matrix "$matrix"

for job_index in $(seq 0 3); do
  "$python_bin" -u scripts/run_scfar_paper1.py \
    --run \
    --matrix "$matrix" \
    --pilot-rounds 20 \
    --job-index "$job_index" \
    --device mps \
    --data-root data \
    --output-root "$output_root" \
    --resume
done
