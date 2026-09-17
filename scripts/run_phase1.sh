#!/usr/bin/env bash
# Phase-1 EMNIST campaign. See scripts/run_campaign.sh for the serial-run
# rationale; relaunch this any time, completed runs are skipped.
set -u
export FEDLAB_NUM_WORKERS=0
exec scripts/run_campaign.sh 'configs/dmd_emnist/*_s[0-9]*.yaml' \
  results/emnist_phase1 emnist_phase1 1 4
