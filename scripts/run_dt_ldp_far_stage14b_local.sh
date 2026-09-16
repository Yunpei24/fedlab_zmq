#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mkdir -p logs/dt_ldp_far
python_bin="${DT_LDP_FAR_PYTHON:-$repo_root/venv/bin/python}"

"$python_bin" -u scripts/run_dt_ldp_far_stage14b_reference_trust_audit.py \
  2>&1 | tee logs/dt_ldp_far/stage14b_reference_trust_guard.log
