#!/bin/bash
# =====================================================================
# E1 — Main comparison (iso-setup): FedAvg / FedPart / FedSTEP on
# 40xRPi-4, CIFAR-10, Dirichlet alpha=1, E=8, T=200, measured cost.
# Multi-seed: each algo x seed -> results/E1/<algo>/seed<S>/metrics.json
# Aggregate with scripts/aggregate_seeds.py (mean +/- std).
#
# Usage:   bash scripts/run_e1.sh
#   SEEDS="42 43 44"  ALGOS="fedavg fedpart fedstep"   (env overridable)
# =====================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=venv/bin/python
CFG=configs/e1_main.yaml
SEEDS="${SEEDS:-42 43 44}"
ALGOS="${ALGOS:-fedavg fedpart fedstep}"

for S in $SEEDS; do
  for A in $ALGOS; do
    OUT="results/E1/${A}/seed${S}"
    echo "===== E1 ${A} seed${S} START $(date) ====="
    caffeinate -si $PY run_experiment.py --config "$CFG" --algo "$A" \
        --seed "$S" --output "$OUT" --cost-model measured --device mps \
        > "/tmp/e1_${A}_s${S}.log" 2>&1 || echo "  FAILED rc=$? (see /tmp/e1_${A}_s${S}.log)"
    echo "===== E1 ${A} seed${S} DONE $(date) ====="
  done
done

echo "===== E1 AGGREGATE (mean +/- std over seeds) ====="
$PY scripts/aggregate_seeds.py results/E1 --seeds-only --latex || true
