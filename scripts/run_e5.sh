#!/bin/bash
# =====================================================================
# E5 — Generalisation across datasets. 3 algos x seeds per dataset.
# local_epochs comes from each YAML (NOT overridden): cifar100/femnist=8,
# emnist=1 (EMNIST-ByClass is data-heavy and runs on a Jetson fleet at E=1).
# Datasets: cifar100, femnist_natural (LEAF), emnist (ByClass stand-in).
# Tiny-ImageNet deferred. Out: results/E5/<dataset>/<algo>/seed<S>/metrics.json
#
# Usage:   bash scripts/run_e5.sh
#   SEEDS, DATASETS, ALGOS env-overridable.
#   (femnist_natural needs `bash scripts/prepare_femnist.sh` first.)
# =====================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=venv/bin/python
SEEDS="${SEEDS:-42 43 44}"
DATASETS="${DATASETS:-cifar100 femnist_natural emnist}"
ALGOS="${ALGOS:-fedavg fedpart fedstep}"

cfg_for() {  # dataset -> config path (macOS bash 3.2: no assoc arrays)
  case "$1" in
    cifar100)        echo configs/e5_cifar100.yaml ;;
    femnist_natural) echo configs/e5_femnist_natural.yaml ;;
    emnist)          echo configs/e5_emnist.yaml ;;
    tiny|tiny_imagenet) echo configs/e5_tinyimagenet.yaml ;;
    *) echo "" ;;
  esac
}

for S in $SEEDS; do
  for D in $DATASETS; do
    CFG="$(cfg_for "$D")"
    if [ -z "$CFG" ] || [ ! -f "$CFG" ]; then echo "  SKIP $D (no config)"; continue; fi
    for A in $ALGOS; do
      OUT="results/E5/${D}/${A}/seed${S}"
      echo "===== E5 ${D} ${A} seed${S} START $(date) ====="
      caffeinate -si $PY run_experiment.py --config "$CFG" --algo "$A" \
          --seed "$S" --output "$OUT" --cost-model measured --device mps \
          > "/tmp/e6_${D}_${A}_s${S}.log" 2>&1 || echo "  FAILED rc=$? (see /tmp/e6_${D}_${A}_s${S}.log)"
    done
  done
done

echo "===== E5 AGGREGATE (per dataset; mean +/- std over seeds) ====="
for D in $DATASETS; do
  echo "## ${D}"
  $PY scripts/aggregate_seeds.py "results/E5/${D}" --latex 2>/dev/null || true
done
