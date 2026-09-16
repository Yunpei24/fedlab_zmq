#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mkdir -p logs/dt_ldp_far
for job_index in $(seq 0 35); do
  venv/bin/python -u scripts/run_dt_ldp_far.py \
    --run \
    --matrix configs/dt_ldp_far/stage24_multikrum_admissibility_validation.yaml \
    --job-index "$job_index" \
    --device mps \
    --data-root data \
    --output-root results/dt_ldp_far/stage24_multikrum_admissibility_validation_v1 \
    --resume 2>&1 | tee -a logs/dt_ldp_far/stage24_multikrum_admissibility_validation_mps.log
done
