#!/usr/bin/env bash
# Run pending campaign configs in the FOREGROUND until a wall-clock budget runs
# out, then exit cleanly.
#
#   scripts/run_until_deadline.sh <config-glob> <results-root> <log-tag> <budget-s>
#
# Background jobs do not reliably make progress in this sandbox: it restarts
# every ~30 minutes and processes launched with nohup have been observed to stop
# advancing between turns. Foreground execution inside a bounded call always
# runs. Each completed run is banked by its .done marker, so successive calls
# resume rather than restart.
set -u
GLOB="$1"; RESULTS_ROOT="$2"; TAG="$3"; BUDGET="$4"
# Do not start a run that cannot finish: a truncated run banks nothing and the
# time is simply lost. Measured ~400s for a 150-round EMNIST run on four cores.
MIN_START="${5:-450}"
LOGDIR="logs/${TAG}"
mkdir -p "$LOGDIR"
DEADLINE=$(( $(date +%s) + BUDGET ))

for cfg in $GLOB; do
  name=$(basename "$cfg" .yaml)
  [ -f "${LOGDIR}/${name}.done" ] && continue
  remaining=$(( DEADLINE - $(date +%s) ))
  if [ "$remaining" -lt "$MIN_START" ]; then
    echo "budget epuise, arret propre"
    break
  fi
  echo "-> ${name} (budget restant ${remaining}s)"
  FEDLAB_NUM_WORKERS=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    timeout "$remaining" python3 run_experiment.py --config "$cfg" \
    > "${LOGDIR}/${name}.log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    touch "${LOGDIR}/${name}.done"
    echo "   OK"
    echo "$(date +%H:%M:%S) OK   ${name}" >> "${LOGDIR}/_progress.txt"
  else
    echo "   INCOMPLET rc=${rc}"
    echo "$(date +%H:%M:%S) FAIL ${name} rc=${rc}" >> "${LOGDIR}/_progress.txt"
  fi
done
echo "bankes : $(ls ${LOGDIR}/*.done 2>/dev/null | wc -l)"
