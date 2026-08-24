#!/bin/bash
# ============================================================
# Phase 1 — Tuning hyperparamètres (SLURM array, 45 runs)
# ============================================================
# Baselines  : FedAvg(3) + FedPart(3) + HeteroFL(3) + FjORD(3) + FedProx(6)
# Proposés   : FedStep(9) + ccsEF_full(9) + ccsEF_frozen(9)
# Total      : 45 runs  (--array=0-44)
#
# Hors comparaison (non publiés) : eceffl, fed_resonance
#
# Soumettre :
#   sbatch scripts/hpc_phase1_tuning.sh
#
# Après exécution :
#   python scripts/find_best_config.py results/tuning/
# ============================================================

#SBATCH --job-name=fl_tuning
#SBATCH --array=0-44
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/tuning_%A_%a.out
#SBATCH --error=logs/tuning_%A_%a.err

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

module purge
module load python/3.12 cuda/12.1   # adapter à ton HPC

source venv/bin/activate

# ============================================================
# Table des 45 configurations
# Format : "ALGO_NAME EPOCHS EXTRA_KEY EXTRA_VAL FREEZE TAG"
#   ALGO_NAME  : nom dans le registre (ccsEF pour full et frozen)
#   EXTRA_KEY  : paramètre à override dans le YAML (none si aucun)
#   EXTRA_VAL  : valeur numérique du paramètre
#   FREEZE     : 1 = --freeze-secondary, 0 = non
#   TAG        : nom du sous-dossier résultat
# ============================================================
CONFIGS=(
    # ── Baselines : FedAvg E ∈ {3,5,8} ── IDs 0-2
    "fedavg      3  none none 0  fedavg_e3"
    "fedavg      5  none none 0  fedavg_e5"
    "fedavg      8  none none 0  fedavg_e8"

    # ── Baselines : FedPart E ∈ {3,5,8} ── IDs 3-5
    "fedpart     3  none none 0  fedpart_e3"
    "fedpart     5  none none 0  fedpart_e5"
    "fedpart     8  none none 0  fedpart_e8"

    # ── Baselines : HeteroFL E ∈ {3,5,8} ── IDs 6-8
    "heterofl    3  none none 0  heterofl_e3"
    "heterofl    5  none none 0  heterofl_e5"
    "heterofl    8  none none 0  heterofl_e8"

    # ── Baselines : FjORD E ∈ {3,5,8} ── IDs 9-11
    "fjord       3  none none 0  fjord_e3"
    "fjord       5  none none 0  fjord_e5"
    "fjord       8  none none 0  fjord_e8"

    # ── Baselines : FedProx (E,mu) ∈ {3,5,8}×{0.01,0.1} ── IDs 12-17
    "fedprox     3  mu 0.01 0  fedprox_e3_mu001"
    "fedprox     3  mu 0.1  0  fedprox_e3_mu01"
    "fedprox     5  mu 0.01 0  fedprox_e5_mu001"
    "fedprox     5  mu 0.1  0  fedprox_e5_mu01"
    "fedprox     8  mu 0.01 0  fedprox_e8_mu001"
    "fedprox     8  mu 0.1  0  fedprox_e8_mu01"

    # ── Proposé : FedStep (E,num_tiers) ∈ {3,5,8}×{2,3,4} ── IDs 18-26
    "fedstep  3  num_tiers 2 0  fedpartbe_e3_t2"
    "fedstep  3  num_tiers 3 0  fedpartbe_e3_t3"
    "fedstep  3  num_tiers 4 0  fedpartbe_e3_t4"
    "fedstep  5  num_tiers 2 0  fedpartbe_e5_t2"
    "fedstep  5  num_tiers 3 0  fedpartbe_e5_t3"
    "fedstep  5  num_tiers 4 0  fedpartbe_e5_t4"
    "fedstep  8  num_tiers 2 0  fedpartbe_e8_t2"
    "fedstep  8  num_tiers 3 0  fedpartbe_e8_t3"
    "fedstep  8  num_tiers 4 0  fedpartbe_e8_t4"

    # ── Proposé : ccsEF full (E,T_rot) ∈ {1,3,5}×{1,3,5} ── IDs 27-35
    "ccsEF       1  T_rot 1 0  ccsEF_full_e1_r1"
    "ccsEF       1  T_rot 3 0  ccsEF_full_e1_r3"
    "ccsEF       1  T_rot 5 0  ccsEF_full_e1_r5"
    "ccsEF       3  T_rot 1 0  ccsEF_full_e3_r1"
    "ccsEF       3  T_rot 3 0  ccsEF_full_e3_r3"
    "ccsEF       3  T_rot 5 0  ccsEF_full_e3_r5"
    "ccsEF       5  T_rot 1 0  ccsEF_full_e5_r1"
    "ccsEF       5  T_rot 3 0  ccsEF_full_e5_r3"
    "ccsEF       5  T_rot 5 0  ccsEF_full_e5_r5"

    # ── Proposé : ccsEF frozen (E,T_rot) ∈ {1,3,5}×{1,3,5} ── IDs 36-44
    "ccsEF       1  T_rot 1 1  ccsEF_frozen_e1_r1"
    "ccsEF       1  T_rot 3 1  ccsEF_frozen_e1_r3"
    "ccsEF       1  T_rot 5 1  ccsEF_frozen_e1_r5"
    "ccsEF       3  T_rot 1 1  ccsEF_frozen_e3_r1"
    "ccsEF       3  T_rot 3 1  ccsEF_frozen_e3_r3"
    "ccsEF       3  T_rot 5 1  ccsEF_frozen_e3_r5"
    "ccsEF       5  T_rot 1 1  ccsEF_frozen_e5_r1"
    "ccsEF       5  T_rot 3 1  ccsEF_frozen_e5_r3"
    "ccsEF       5  T_rot 5 1  ccsEF_frozen_e5_r5"
)

# ── Extraire la config du job courant ────────────────────────
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

# ── Override du paramètre extra via YAML temporaire ─────────
TEMP_YAML="configs/tmp_tuning_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"
cp configs/tuning_hp.yaml "$TEMP_YAML"

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

# ── Lancer l'expérience ──────────────────────────────────────
FREEZE_FLAG=""
[ "$FREEZE" = "1" ] && FREEZE_FLAG="--freeze-secondary"

python run_experiment.py \
    --config "$TEMP_YAML" \
    --algo   "$ALGO_NAME" \
    --epochs "$EPOCHS" \
    --output "results/tuning/$TAG" \
    --device cuda \
    $FREEZE_FLAG

STATUS=$?
rm -f "$TEMP_YAML"
echo "Exit status: $STATUS"
exit $STATUS
