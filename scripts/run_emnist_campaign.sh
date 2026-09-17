#!/usr/bin/env bash
# Run the phase-1 EMNIST campaign with a fixed-size worker pool.
#
# One OMP thread per worker: four single-threaded runs beat one four-threaded
# run here because the bottleneck is many small forward passes, not matmul size.
#
# Resumable by design. The sandbox this runs in can restart and take every
# background process with it, so a completed run is recorded by the presence of
# its metrics.json and is skipped on relaunch: a restart costs the in-flight
# runs only, not the whole campaign. Delete the results directory for a clean
# re-run.
set -u
POOL=${POOL:-4}
mkdir -p logs/emnist_phase1

run_one() {
  local cfg="$1" name arm seed out
  name=$(basename "$cfg" .yaml)
  arm=${name%_s*}
  seed=${name##*_s}
  out="results/emnist_phase1/${arm}"
  if compgen -G "${out}/*_s${seed}/metrics.json" > /dev/null; then
    echo "$(date +%H:%M:%S) SKIP ${name} (deja complet)" >> logs/emnist_phase1/_progress.txt
    return 0
  fi
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 run_experiment.py --config "$cfg" \
    > "logs/emnist_phase1/${name}.log" 2>&1
  local rc=$?
  # Capture the status BEFORE anything else runs: a killed worker exits non-zero
  # and leaves a truncated log, which must not be reported as success.
  if [ "$rc" -eq 0 ]; then
    echo "$(date +%H:%M:%S) OK   ${name}" >> logs/emnist_phase1/_progress.txt
  else
    echo "$(date +%H:%M:%S) FAIL ${name} rc=${rc}" >> logs/emnist_phase1/_progress.txt
  fi
}
export -f run_one

ls configs/dmd_emnist/*_s*.yaml | grep -v _smoke \
  | xargs -P "$POOL" -I{} bash -c 'run_one "$@"' _ {}
echo "$(date +%H:%M:%S) CAMPAGNE TERMINEE" >> logs/emnist_phase1/_progress.txt
