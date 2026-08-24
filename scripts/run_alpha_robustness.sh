#!/bin/bash
# =====================================================================
# alpha-robustness (App. D) — Robustness to the compute utilisation gap alpha (energy_scale_factor).
# alpha is fixed PHYSICALLY per device (1/sustained-utilisation), never tuned to a
# survival horizon. To show the conclusions do not hinge on a particular alpha, we
# re-run the E1 scenario (CIFAR-10, 40xRPi-4, E=8, lr=0.003, measured) over a
# physically-justified grid of alpha, each a FULL re-simulation (alpha decides who
# dies once a battery hits zero, which changes the whole trajectory).
# Two checks: (a) cross-algo RELATIVE metrics (energy/uplink ratios) are alpha-
# invariant; (b) the absolute order FedSTEP >= FedPart > FedAvg (survival) and the
# energy/uplink savings hold across the WHOLE grid.
# Out: results/alpha_robustness/alpha<a>/<algo>/seed<S>/metrics.json
#
# Usage:  bash scripts/run_alpha_robustness.sh    # SEEDS / ALPHAS / ALGOS env-overridable
# =====================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=venv/bin/python
BASE=configs/e1_main.yaml            # RPi-4 x40, CIFAR-10, E=8, lr=0.003, measured
SEEDS="${SEEDS:-42 43 44}"
ALPHAS="${ALPHAS:-1.5 3 5 8}"        # around the physical RPi-4 estimate (~3)
ALGOS="${ALGOS:-fedavg fedpart fedstep}"

for S in $SEEDS; do
  for A in $ALPHAS; do
    TMP="/tmp/e4_alpha${A}.yaml"
    sed "s/^    energy_scale_factor: .*/    energy_scale_factor: ${A}/" "$BASE" > "$TMP"
    for ALG in $ALGOS; do
      OUT="results/alpha_robustness/alpha${A}/${ALG}/seed${S}"
      echo "===== alpha-robustness (App. D) alpha=${A} ${ALG} seed${S} START $(date) ====="
      caffeinate -si $PY run_experiment.py --config "$TMP" --algo "$ALG" \
          --seed "$S" --output "$OUT" --cost-model measured --device mps \
          > "/tmp/e4_a${A}_${ALG}_s${S}.log" 2>&1 \
          || echo "  FAILED rc=$? (/tmp/e4_a${A}_${ALG}_s${S}.log)"
    done
  done
done

echo "===== alpha-robustness (App. D) AGGREGATE (per alpha; mean +/- std) ====="
for A in $ALPHAS; do
  echo "## alpha=${A}"; $PY scripts/aggregate_seeds.py "results/alpha_robustness/alpha${A}" --latex 2>/dev/null || true
done
