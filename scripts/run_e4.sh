#!/bin/bash
# =====================================================================
# E4 / Table III — Heterogeneity stress test across non-IID severity AND datasets.
# Sweeps Dirichlet alpha in {0.1, 0.5} x dataset in {cifar10, cifar100, femnist}
# x algo in {fedavg, fedpart, fedpart_be} x seeds. lr=0.003, measured cost.
# Per-dataset base config (alpha sed-overridden):
#   cifar10  -> configs/e1_main.yaml          (RPi-4, E=8, Dirichlet)
#   cifar100 -> configs/e5_cifar100.yaml      (RPi-4, E=8, Dirichlet)
#   femnist  -> configs/e5_emnist.yaml        (EMNIST-ByClass Dirichlet stand-in,
#               Jetson, E=1; the true LEAF FEMNIST is per-writer natural with no alpha)
# Out: results/E4/<dataset>_a<alpha>/<algo>/seed<S>/metrics.json
#
# Usage:  bash scripts/run_e4.sh    # SEEDS / ALPHAS / DATASETS / ALGOS env-overridable
# =====================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=venv/bin/python
SEEDS="${SEEDS:-42 43 44}"
ALPHAS="${ALPHAS:-0.1 0.5}"
DATASETS="${DATASETS:-cifar10 cifar100 femnist}"
ALGOS="${ALGOS:-fedavg fedpart fedpart_be}"

base_for() { case "$1" in
  cifar10)  echo configs/e1_main.yaml ;;
  cifar100) echo configs/e5_cifar100.yaml ;;
  femnist)  echo configs/e5_emnist.yaml ;;   # EMNIST-ByClass Dirichlet (Jetson, E=1)
  *) echo "" ;;
esac; }

for S in $SEEDS; do
  for D in $DATASETS; do
    B="$(base_for "$D")"; [ -f "$B" ] || { echo "SKIP $D (no base)"; continue; }
    for AL in $ALPHAS; do
      TMP="/tmp/e5_${D}_a${AL}.yaml"
      sed "s/^  alpha: .*/  alpha: ${AL}/" "$B" > "$TMP"
      for ALG in $ALGOS; do
        OUT="results/E4/${D}_a${AL}/${ALG}/seed${S}"
        echo "===== E4 ${D} alpha=${AL} ${ALG} seed${S} START $(date) ====="
        caffeinate -si $PY run_experiment.py --config "$TMP" --algo "$ALG" \
            --seed "$S" --output "$OUT" --cost-model measured --device mps \
            > "/tmp/e5_${D}_a${AL}_${ALG}_s${S}.log" 2>&1 \
            || echo "  FAILED rc=$? (/tmp/e5_${D}_a${AL}_${ALG}_s${S}.log)"
      done
    done
  done
done

echo "===== E4 AGGREGATE (per dataset x alpha; mean +/- std) ====="
for D in $DATASETS; do for AL in $ALPHAS; do
  echo "## ${D} alpha=${AL}"; $PY scripts/aggregate_seeds.py "results/E4/${D}_a${AL}" --latex 2>/dev/null || true
done; done
