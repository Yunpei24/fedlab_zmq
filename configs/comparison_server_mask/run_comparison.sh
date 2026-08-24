#!/bin/bash
# ============================================================
# Server-Mask FL — Full Comparison Suite
# 11 algorithms × 200 rounds × 30 ESP32-S3 clients
# Scenario: SOC uniform [5%, 95%], CIFAR-10, Dirichlet α=1.0
# Controlled: local_epochs=5, seed=42 for ALL algorithms
# ============================================================
# Usage:
#   bash configs/comparison_server_mask/run_comparison.sh
#   bash configs/comparison_server_mask/run_comparison.sh fedavg fedstep server_mask
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# All algorithms in order (fastest → slowest expected)
ALL_ALGOS=(fedavg fedprox fedsparq leanfed heterofl fjord fedpart fedstep fedmask hermes server_mask)

# If arguments provided, run only those
if [ $# -gt 0 ]; then
    ALGOS=("$@")
else
    ALGOS=("${ALL_ALGOS[@]}")
fi

echo "============================================================"
echo "  Server-Mask Comparison Suite"
echo "  Algorithms: ${ALGOS[*]}"
echo "  Project: $PROJECT_DIR"
echo "============================================================"

FAILED=()
for ALGO in "${ALGOS[@]}"; do
    CONFIG="$SCRIPT_DIR/${ALGO}.yaml"
    if [ ! -f "$CONFIG" ]; then
        echo "  [SKIP] $ALGO — config not found: $CONFIG"
        continue
    fi
    echo ""
    echo "────────────────────────────────────────────────────────"
    echo "  Running: $ALGO"
    echo "  Config:  $CONFIG"
    echo "────────────────────────────────────────────────────────"
    START=$(date +%s)
    if python run_experiment.py --config "$CONFIG" --device mps; then
        END=$(date +%s)
        echo "  ✓ $ALGO completed in $((END - START))s"
    else
        END=$(date +%s)
        echo "  ✗ $ALGO FAILED after $((END - START))s"
        FAILED+=("$ALGO")
    fi
done

echo ""
echo "============================================================"
echo "  Comparison complete"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  FAILED: ${FAILED[*]}"
else
    echo "  All algorithms succeeded"
fi
echo "  Results: $PROJECT_DIR/results/comparison_server_mask/"
echo "============================================================"
