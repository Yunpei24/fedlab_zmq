#!/bin/bash
# =====================================================================
# E5 — Heterogeneity stress test (severe non-IID, Dirichlet alpha=0.1).
# Same fleet/budget/cost as E1; only alpha 1.0 -> 0.1. 3 algos x seeds.
# Out: results/E5/<algo>/seed<S>/metrics.json
#
# Usage:   bash scripts/run_e5.sh
#   SEEDS, ALGOS env-overridable.
# =====================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=venv/bin/python
CFG=configs/e5_severe_niid.yaml
SEEDS="${SEEDS:-42 43 44}"
ALGOS="${ALGOS:-fedavg fedpart fedpart_be}"

for S in $SEEDS; do
  for A in $ALGOS; do
    OUT="results/E5/${A}/seed${S}"
    echo "===== E5 ${A} seed${S} START $(date) ====="
    caffeinate -si $PY run_experiment.py --config "$CFG" --algo "$A" \
        --seed "$S" --output "$OUT" --cost-model measured --device mps \
        > "/tmp/e5_${A}_s${S}.log" 2>&1 || echo "  FAILED rc=$? (see /tmp/e5_${A}_s${S}.log)"
    echo "===== E5 ${A} seed${S} DONE $(date) ====="
  done
done

echo "===== E5 AGGREGATE (mean +/- std over seeds) ====="
$PY scripts/aggregate_seeds.py results/E5 --latex || true
