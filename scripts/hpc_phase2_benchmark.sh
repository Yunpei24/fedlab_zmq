#!/bin/bash
# ============================================================
# Phase 2 — Benchmark final (SLURM array, 8 runs)
# ============================================================
# Baselines  : FedAvg | FedPart | HeteroFL | FjORD | FedProx
# Proposés   : FedStep | ccsEF full | ccsEF frozen
#
# Prérequis : Phase 1 terminée.
# Après find_best_config.py, mettre à jour CONFIGS[] ci-dessous
# avec les EPOCHS et paramètres optimaux trouvés.
#
# Soumettre :
#   sbatch scripts/hpc_phase2_benchmark.sh
#
# Analyse :
#   python analyze_benchmark.py results/benchmark/ --save figures/
# ============================================================

#SBATCH --job-name=fl_benchmark
#SBATCH --array=0-7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/benchmark_%A_%a.out
#SBATCH --error=logs/benchmark_%A_%a.err

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

module purge
module load python/3.12 cuda/12.1

source venv/bin/activate

# ============================================================
# !! METTRE À JOUR après find_best_config.py !!
#
# Format : "ALGO_NAME EPOCHS EXTRA_KEY EXTRA_VAL FREEZE TAG"
#   EXTRA_KEY/VAL : paramètre specifique (num_tiers, T_rot, mu…)
#                   "none none" si pas de paramètre extra
#   FREEZE        : 1 = --freeze-secondary, 0 = non
# ============================================================
CONFIGS=(
    # ── Baselines ──────────────────────────────────────────────────────────
    # ID 0 — FedAvg
    "fedavg     5  none     none  0  fedavg_opt"

    # ID 1 — FedPart
    "fedpart    5  none     none  0  fedpart_opt"

    # ID 2 — HeteroFL
    "heterofl   5  none     none  0  heterofl_opt"

    # ID 3 — FjORD
    "fjord      5  none     none  0  fjord_opt"

    # ID 4 — FedProx
    "fedprox    5  mu       0.01  0  fedprox_opt"

    # ── Proposés ───────────────────────────────────────────────────────────
    # ID 5 — FedStep  (optimal Phase 1)
    "fedstep 8  num_tiers  3   0  fedpartbe_opt"

    # ID 6 — ccsEF full  (optimal Phase 1)
    "ccsEF      3  T_rot      3   0  ccsEF_full_opt"

    # ID 7 — ccsEF frozen  (optimal Phase 1)
    "ccsEF      3  T_rot      3   1  ccsEF_frozen_opt"
)

CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read -r ALGO_NAME EPOCHS EXTRA_KEY EXTRA_VAL FREEZE TAG <<< "$CFG"

echo "========================================"
echo "  Job    : $SLURM_JOB_ID[$SLURM_ARRAY_TASK_ID]"
echo "  Algo   : $ALGO_NAME"
echo "  Epochs : $EPOCHS"
echo "  Extra  : $EXTRA_KEY=$EXTRA_VAL"
echo "  Freeze : $FREEZE"
echo "  Tag    : $TAG"
echo "========================================"

# ── Override paramètre extra via YAML temporaire ─────────────
TEMP_YAML="configs/tmp_bench_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"
cp configs/benchmark_final.yaml "$TEMP_YAML"

if [ "$EXTRA_KEY" != "none" ]; then
    python - <<PYEOF
import yaml
with open("$TEMP_YAML") as f:
    cfg = yaml.safe_load(f)
cfg["training"]["algo_config"]["$EXTRA_KEY"] = $EXTRA_VAL
with open("$TEMP_YAML", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
PYEOF
fi

FREEZE_FLAG=""
[ "$FREEZE" = "1" ] && FREEZE_FLAG="--freeze-secondary"

python run_experiment.py \
    --config "$TEMP_YAML" \
    --algo   "$ALGO_NAME" \
    --epochs "$EPOCHS" \
    --output "results/benchmark/$TAG" \
    --device cuda \
    $FREEZE_FLAG

STATUS=$?
rm -f "$TEMP_YAML"
echo "Exit status: $STATUS"
exit $STATUS
