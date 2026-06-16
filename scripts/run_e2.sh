#!/bin/bash
# =====================================================================
# E2 — Device heterogeneity & compute<->comm balance. Same workload
# (CIFAR-10, alpha=1, E=8) replayed across device classes; fed_resonance
# uses E=3. The device is the only varied axis, so run_e2.sh sed-swaps
# clients.fleet type in configs/e2_base.yaml per device.
# Out: results/E2/<device>__<algo>/seed<S>/metrics.json (matches prior E2 layout).
#
# Usage:   bash scripts/run_e2.sh
#   SEEDS, DEVICES, ALGOS env-overridable.
# Note: ESP32-S3 is expected to die in warmup (physical infeasibility finding).
# =====================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=venv/bin/python
BASE=configs/e2_base.yaml
SEEDS="${SEEDS:-42 43 44}"
DEVICES="${DEVICES:-esp32_s3 raspberry_pi_4 raspberry_pi_zero2w smartphone_midrange smartphone_highend}"
ALGOS="${ALGOS:-fedavg fedpart fedpart_be fed_resonance}"

for S in $SEEDS; do
  for DEV in $DEVICES; do
    TMP="/tmp/e2_${DEV}.yaml"
    sed "s|^    - type: .*|    - type: ${DEV}|" "$BASE" > "$TMP"
    for A in $ALGOS; do
      EP=8; [ "$A" = "fed_resonance" ] && EP=3
      OUT="results/E2/${DEV}__${A}/seed${S}"
      echo "===== E2 ${DEV} ${A} (E=${EP}) seed${S} START $(date) ====="
      caffeinate -si $PY run_experiment.py --config "$TMP" --algo "$A" --epochs "$EP" \
          --seed "$S" --output "$OUT" --cost-model measured --device mps \
          > "/tmp/e2_${DEV}_${A}_s${S}.log" 2>&1 || echo "  FAILED rc=$? (see /tmp/e2_${DEV}_${A}_s${S}.log)"
    done
  done
done

echo "===== E2 AGGREGATE (per device x algo; mean +/- std over seeds) ====="
for DEV in $DEVICES; do
  for A in $ALGOS; do
    D="results/E2/${DEV}__${A}"
    [ -d "$D" ] || continue
    echo "## ${DEV} / ${A}"
    $PY scripts/aggregate_seeds.py "$D" 2>/dev/null | tail -n +1 || true
  done
done
