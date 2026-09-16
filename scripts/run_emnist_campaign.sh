#!/usr/bin/env bash
# Run the phase-1 EMNIST campaign with a fixed-size worker pool.
# One OMP thread per worker: four single-threaded runs beat one four-threaded
# run here because the bottleneck is many small forward passes, not matmul size.
set -u
POOL=${POOL:-4}
mkdir -p logs/emnist_phase1
run_one() {
  local cfg="$1" name
  name=$(basename "$cfg" .yaml)
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 run_experiment.py --config "$cfg" \
    > "logs/emnist_phase1/${name}.log" 2>&1
  echo "$(date +%H:%M:%S) done ${name} rc=$?" >> logs/emnist_phase1/_progress.txt
}
export -f run_one
ls configs/dmd_emnist/*_s*.yaml | grep -v _smoke \
  | xargs -P "$POOL" -I{} bash -c 'run_one "$@"' _ {}
echo "CAMPAGNE TERMINEE" >> logs/emnist_phase1/_progress.txt
