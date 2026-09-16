#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mkdir -p logs/dt_ldp_far

./venv/bin/python -u scripts/run_dt_ldp_far_stage14a_effective_moments_audit.py \
  2>&1 | tee logs/dt_ldp_far/stage14a_effective_null_moments.log
