#!/usr/bin/env bash
# Phase-1 DMD campaign on a local machine, GPU by default (Apple MPS).
#   scripts/run_phase1_local.sh [dataset] [device] [pool] [threads]
#   scripts/run_phase1_local.sh fashionmnist mps 4 2      # the defaults
#
# Three stages, because the matched-intensity control dmdcb_hi is only
# meaningful at the effective intensity USV-0.25 reaches on THIS dataset:
#   1. usv025, 4 seeds;
#   2. measure its mean effective intensity and generate dmdcb_hi from it;
#   3. every arm, dmdcb_hi included.
# Relaunch any time: finished runs are skipped (see scripts/run_campaign.sh),
# and the calibration is recomputed from the same usv025 runs. It needs all
# four seeds unless MIN_USV_RUNS says otherwise: a usv025 seed that diverges
# leaves no metrics.json, and calibrating on the others must be a choice.
#
# On MPS, four concurrent runs gave 2.6x the serial throughput and results
# bit-identical to serial runs. Never mix these runs with CPU ones in a table:
# dropout draws differ between the CPU and MPS generators.
set -euo pipefail
DATASET="${1:-fashionmnist}"; DEVICE="${2:-mps}"; POOL="${3:-4}"; THREADS="${4:-2}"
case "$DATASET" in
  fashionmnist) TAG=fmnist ;;
  *) TAG="$DATASET" ;;
esac
CONFIGS="configs/dmd_${TAG}"
RESULTS="results/${TAG}_phase1"
LOGTAG="${TAG}_phase1"
PROGRESS="logs/${LOGTAG}/_progress.txt"
mkdir -p "logs/${LOGTAG}"
export FEDLAB_NUM_WORKERS=0

note() { echo "$(date +%H:%M:%S) === $*" | tee -a "$PROGRESS"; }

# Keep the Mac from idle-sleeping for as long as this script runs.
if command -v caffeinate > /dev/null; then
  caffeinate -i -w $$ &
fi

# Download once up front: concurrent first runs would race on the same files.
python3 -c "
from datasets.registry import _load_raw_dataset
for split in ('train', 'test'):
    _load_raw_dataset('${DATASET}', split, './data')
"

note "phase 1 ${DATASET} on ${DEVICE}, pool=${POOL} threads=${THREADS}"
python3 scripts/gen_emnist_campaign.py --dataset "$DATASET" --device "$DEVICE" > /dev/null

note "stage 1/3: usv025"
scripts/run_campaign.sh "${CONFIGS}/usv025_s[0-9]*.yaml" "$RESULTS" "$LOGTAG" "$POOL" "$THREADS"

note "stage 2/3: dmdcb_hi calibration"
calibration=$(python3 scripts/gen_emnist_campaign.py --dataset "$DATASET" \
  --device "$DEVICE" --calibrate-dmdcb-hi "${RESULTS}/usv025" \
  --min-runs "${MIN_USV_RUNS:-4}")
echo "$calibration" | sed -n 1p | tee -a "$PROGRESS"

note "stage 3/3: all arms"
scripts/run_campaign.sh "${CONFIGS}/*_s[0-9]*.yaml" "$RESULTS" "$LOGTAG" "$POOL" "$THREADS"

done_count=$(find "logs/${LOGTAG}" -name '*.done' | wc -l | tr -d ' ')
note "PHASE 1 DONE: ${done_count}/24 runs"
