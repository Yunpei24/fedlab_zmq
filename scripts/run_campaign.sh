#!/usr/bin/env bash
# Generic resumable campaign runner.
#   scripts/run_campaign.sh <config-glob> <results-root> <log-tag> [pool] [threads]
#
# Defaults to ONE run at a time using every core, not several single-threaded
# runs. Total throughput is the same, but a serial run finishes in ~13 minutes
# instead of ~60, and this sandbox restarts roughly every 35 minutes, taking
# every background process with it. A 60-minute run never reaches its
# metrics.json and is redone from scratch forever; a 13-minute one completes and
# is banked. Wall-clock per run matters more than parallelism when the ceiling
# on uninterrupted time is fixed.
#
# Resumable by design. The sandbox can restart and take every background process
# with it, so a run is treated as complete when its metrics.json exists and is
# skipped on relaunch: a restart costs the in-flight runs only. Delete the
# results root for a clean re-run.
set -u
GLOB="$1"; RESULTS_ROOT="$2"; TAG="$3"; POOL="${4:-1}"; THREADS="${5:-4}"
LOGDIR="logs/${TAG}"
mkdir -p "$LOGDIR"
export RESULTS_ROOT LOGDIR THREADS

run_one() {
  local cfg="$1" name arm seed rc done_glob
  name=$(basename "$cfg" .yaml)
  # Config names are <arm>_p<partition>_i<init> or <arm>_s<seed>; the results
  # subdirectory is the arm, which is everything before the seed marker.
  arm=$(echo "$name" | sed -E 's/_(p[0-9]+_i[0-9]+|s[0-9]+)$//')
  seed=$(echo "$name" | sed -E 's/.*_(p[0-9]+_i[0-9]+|s[0-9]+)$/\1/')
  # A partition/init run sits one level deeper, under its partition, and its
  # run directory is named after the init seed (scripts/gen_emnist_phase2.py).
  case "$seed" in
    p*) done_glob="${RESULTS_ROOT}/${arm}/${seed%%_*}/*_s${seed##*_i}/metrics.json" ;;
    *)  done_glob="${RESULTS_ROOT}/${arm}/*/metrics.json" ;;
  esac
  if compgen -G "$done_glob" > /dev/null && [ -f "${LOGDIR}/${name}.done" ]; then
    echo "$(date +%H:%M:%S) SKIP ${name}" >> "${LOGDIR}/_progress.txt"
    return 0
  fi
  OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
    python3 run_experiment.py --config "$cfg" > "${LOGDIR}/${name}.log" 2>&1
  rc=$?
  # Read the status before anything else runs: a killed worker exits non-zero
  # and leaves a truncated log, which must not be reported as success.
  if [ "$rc" -eq 0 ]; then
    touch "${LOGDIR}/${name}.done"
    echo "$(date +%H:%M:%S) OK   ${name}" >> "${LOGDIR}/_progress.txt"
  else
    echo "$(date +%H:%M:%S) FAIL ${name} rc=${rc}" >> "${LOGDIR}/_progress.txt"
  fi
}
export -f run_one

ls $GLOB | xargs -P "$POOL" -I{} bash -c 'run_one "$@"' _ {}
echo "$(date +%H:%M:%S) CAMPAGNE TERMINEE ($TAG)" >> "${LOGDIR}/_progress.txt"
