#!/bin/bash
# =====================================================================
# E3 — Ablations of FedSTEP (algo fedpart_be), one explicit YAML per variant
# (configs/e3_<variant>.yaml). Baseline "Full FedSTEP" = E1 fedpart_be (run_e1.sh),
# not re-run here. Variants: no_rotation, no_repr, M2, M3, Tr2, Tr4, Tr5.
# Out: results/E3/<variant>/seed<S>/metrics.json
#
# Usage:   bash scripts/run_e3.sh
#   SEEDS, VARIANTS env-overridable.
# =====================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=venv/bin/python
SEEDS="${SEEDS:-42 43 44}"
VARIANTS="${VARIANTS:-no_rotation no_repr M2 M3 Tr2 Tr4 Tr5}"

for S in $SEEDS; do
  for V in $VARIANTS; do
    OUT="results/E3/${V}/seed${S}"
    echo "===== E3 ${V} seed${S} START $(date) ====="
    caffeinate -si $PY run_experiment.py --config "configs/e3_${V}.yaml" --algo fedpart_be \
        --seed "$S" --output "$OUT" --cost-model measured --device mps \
        > "/tmp/e3_${V}_s${S}.log" 2>&1 || echo "  FAILED rc=$? (see /tmp/e3_${V}_s${S}.log)"
    echo "===== E3 ${V} seed${S} DONE $(date) ====="
  done
done

echo "===== E3 AGGREGATE (per variant; mean +/- std over seeds) ====="
for V in $VARIANTS; do
  echo "## ${V}"
  $PY scripts/aggregate_seeds.py "results/E3/${V}" 2>/dev/null || true
done
