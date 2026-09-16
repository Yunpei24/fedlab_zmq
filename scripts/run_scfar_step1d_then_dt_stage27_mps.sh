#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# The first launcher is resumable.  Stage 27 starts only after all 39 SC-FAR
# screening tasks have completed successfully, avoiding concurrent MPS jobs.
bash scripts/run_scfar_fmnist_step1d_multiseed_mps.sh
bash scripts/run_dt_ldp_far_stage27_raw_distance_mps.sh
