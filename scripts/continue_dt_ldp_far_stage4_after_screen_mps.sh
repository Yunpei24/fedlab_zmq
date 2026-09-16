#!/usr/bin/env bash
set -euo pipefail

# Wait for the already-running 26-task transfer screen, then continue the
# preregistered conditional queue without launching a competing screen run.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
campaign_root="${repo_root}/results/dt_ldp_far/decisive/dt_ldp_far_decisive_stage4_mechanistic_transfer_screen_n25_v1"
expected=26

while true; do
  completed="$(find "${campaign_root}" -name metrics.json 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${completed}" -ge "${expected}" ]]; then
    break
  fi
  sleep 30
done

SKIP_SCREEN=1 bash "${repo_root}/scripts/run_dt_ldp_far_stage4_queue_mps.sh"
