#!/bin/bash
# ============================================================
# HPC SLURM — FedPartBE Full Experiment Suite
# Submits all experiments as independent array jobs
#
# Usage (from project root):
#   bash hpc/run_fedpartbe_all.sh
#
# Requirements:
#   module load python/3.11 (or conda activate your_env)
#   Adjust --partition, --account, --time as needed
# ============================================================

set -euo pipefail
PROJ="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
PYTHON="${PYTHON:-python3}"

echo "Project root: $PROJ"
echo "Python: $PYTHON"

# ── Slurm defaults (override with env vars) ──────────────────
PARTITION="${SLURM_PARTITION:-gpu}"
ACCOUNT="${SLURM_ACCOUNT:-um6p_fl}"
CPUS="${SLURM_CPUS:-8}"
MEM="${SLURM_MEM:-32G}"
TIME_MAIN="${SLURM_TIME_MAIN:-12:00:00}"
TIME_ABL="${SLURM_TIME_ABL:-6:00:00}"
DEVICE="${FL_DEVICE:-cpu}"   # set to "cuda" or "mps" if available


# ── Helper: submit one job ─────────────────────────────────────
submit() {
    local name="$1"
    local time="$2"
    local cmd="$3"
    sbatch \
        --job-name="fedpartbe_${name}" \
        --partition="${PARTITION}" \
        --account="${ACCOUNT}" \
        --cpus-per-task="${CPUS}" \
        --mem="${MEM}" \
        --time="${time}" \
        --output="${PROJ}/logs/fedpartbe_${name}_%j.out" \
        --error="${PROJ}/logs/fedpartbe_${name}_%j.err" \
        --wrap="cd ${PROJ} && ${PYTHON} ${cmd}"
    echo "  Submitted: fedpartbe_${name}"
}

mkdir -p "${PROJ}/logs"

echo ""
echo "============================================================"
echo "  Submitting FedPartBE experiments"
echo "============================================================"


# ── A. Survival curve: FedPartBE vs all baselines (6 algos × 1 run) ──────────
echo ""
echo "=== A. Survival Curve ==="

ALGOS_SURVIVAL="fedavg fedprox heterofl fjord fedpart fedpart_be"
for ALGO in $ALGOS_SURVIVAL; do
    submit "survival_${ALGO}" "${TIME_MAIN}" \
        "run_experiment.py --algo ${ALGO} \
            --dataset cifar10 --model resnet8 \
            --rounds 200 --clients 30 --alpha 0.3 \
            --partition dirichlet \
            --lr 0.01 --epochs 8 --batch-size 32 \
            --device ${DEVICE} --seed 42 \
            --output results/fedpartbe_survival/${ALGO}"
done


# ── B. Full benchmark Table 1 (all algos, same setup) ─────────────────────────
echo ""
echo "=== B. Full Benchmark (Table 1) ==="

ALGOS_BENCH="fedavg fedprox scaffold leanfed fedbacys vaishnav fedsparq heterofl fjord fedpart fedpart_be"
for ALGO in $ALGOS_BENCH; do
    submit "bench_${ALGO}" "${TIME_MAIN}" \
        "run_experiment.py --algo ${ALGO} \
            --dataset cifar10 --model resnet8 \
            --rounds 200 --clients 30 --alpha 0.3 \
            --partition dirichlet \
            --lr 0.01 --epochs 8 --batch-size 32 \
            --device ${DEVICE} --seed 42 \
            --output results/fedpartbe_benchmark/${ALGO}"
done


# ── C. Ablations ──────────────────────────────────────────────────────────────
echo ""
echo "=== C. Ablations ==="

# C1. No representation proximal (mu_repr=0)
submit "ablation_no_repr" "${TIME_ABL}" \
    "run_experiment.py --config configs/fedpartbe_ablation_no_repr.yaml \
        --device ${DEVICE}"

# C2. Single tier M=1 (= FedPart-like)
submit "ablation_m1" "${TIME_ABL}" \
    "run_experiment.py --config configs/fedpartbe_ablation_m1.yaml \
        --device ${DEVICE}"

# C3. No staleness priority
submit "ablation_no_staleness" "${TIME_ABL}" \
    "run_experiment.py --config configs/fedpartbe_ablation_no_staleness.yaml \
        --device ${DEVICE}"


# ── D. Sensitivity: Non-IID alpha ─────────────────────────────────────────────
echo ""
echo "=== D. Sensitivity: alpha ==="

for ALGO in fedpart fedpart_be; do
    for ALPHA in 0.1 0.5 1.0; do
        ALPHA_TAG="${ALPHA//./_}"
        submit "sens_alpha${ALPHA_TAG}_${ALGO}" "${TIME_ABL}" \
            "run_experiment.py --algo ${ALGO} \
                --dataset cifar10 --model resnet8 \
                --rounds 200 --clients 30 --alpha ${ALPHA} \
                --partition dirichlet \
                --lr 0.01 --epochs 8 --batch-size 32 \
                --device ${DEVICE} --seed 42 \
                --output results/fedpartbe_sensitivity/alpha_${ALPHA_TAG}_${ALGO}"
    done
done


# ── E. Sensitivity: Number of tiers M ─────────────────────────────────────────
echo ""
echo "=== E. Sensitivity: num_tiers ==="

for M in 2 3 5; do
    submit "sens_tiers_m${M}" "${TIME_ABL}" \
        "run_experiment.py --algo fedpart_be \
            --dataset cifar10 --model resnet8 \
            --rounds 200 --clients 30 --alpha 0.3 \
            --partition dirichlet \
            --lr 0.01 --epochs 8 --batch-size 32 \
            --device ${DEVICE} --seed 42 \
            --output results/fedpartbe_sensitivity/tiers_m${M}"
done


# ── F. Reproducibility: 3 seeds (main experiment only) ───────────────────────
echo ""
echo "=== F. Reproducibility (seeds 42, 123, 2025) ==="

for ALGO in fedpart fedpart_be; do
    for SEED in 42 123 2025; do
        submit "repro_${ALGO}_s${SEED}" "${TIME_MAIN}" \
            "run_experiment.py --algo ${ALGO} \
                --dataset cifar10 --model resnet8 \
                --rounds 200 --clients 30 --alpha 0.3 \
                --partition dirichlet \
                --lr 0.01 --epochs 8 --batch-size 32 \
                --device ${DEVICE} --seed ${SEED} \
                --output results/fedpartbe_repro/${ALGO}_s${SEED}"
    done
done

echo ""
echo "============================================================"
echo "  All jobs submitted. Monitor with: squeue -u \$USER"
echo "  Results will be in: ${PROJ}/results/"
echo "============================================================"
