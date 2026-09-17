#!/usr/bin/env bash
# Wait for the phase-1 campaign to finish, then launch the phase-2a intensity
# calibration sweep.
#
# Phase 2a cannot start earlier: it would contend for the same four cores and
# slow both campaigns. It also cannot simply be queued behind phase 1 in one
# xargs run, because its configs do not exist until generated.
#
# Completion is detected from the progress file rather than from the process
# table, so this survives being restarted itself: relaunch it any time and it
# picks up where the state on disk says things are.
set -u
PHASE1_PROGRESS="logs/emnist_phase1/_progress.txt"
POLL=${POLL:-120}

while true; do
  if [ -f "$PHASE1_PROGRESS" ] && grep -q "CAMPAGNE TERMINEE" "$PHASE1_PROGRESS"; then
    break
  fi
  sleep "$POLL"
done

echo "$(date +%H:%M:%S) phase 1 terminee, generation de la calibration 2a" \
  >> logs/_chain.txt
python3 scripts/gen_emnist_phase2.py --step calibrate >> logs/_chain.txt 2>&1

# 0 DataLoader workers: four single-threaded runs on four cores otherwise sit at
# a load average above five waiting on each other, and the raw dataset is cached
# in memory so the workers buy nothing.
export FEDLAB_NUM_WORKERS=0
echo "$(date +%H:%M:%S) lancement 2a" >> logs/_chain.txt
scripts/run_campaign.sh 'configs/dmd_emnist_p2/*_mu*_p*.yaml' \
  results/emnist_phase2_cal phase2_cal 1 4 >> logs/_chain.txt 2>&1
echo "$(date +%H:%M:%S) 2a terminee" >> logs/_chain.txt
