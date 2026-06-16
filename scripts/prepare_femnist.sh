#!/bin/bash
# =====================================================================
# Generate LEAF-preprocessed FEMNIST (natural per-writer split) into
#   data/femnist/train/*.json  and  data/femnist/test/*.json
# which datasets/femnist_loader.py reads (partition: natural).
#
# This clones the LEAF benchmark and runs its preprocess.sh, which DOWNLOADS
# the raw NIST by_write data (several GB) on first run and groups it by writer.
#
# Requirements: git, bash, wget/curl, unzip, python3 with numpy + pillow.
#
# Usage:
#   bash scripts/prepare_femnist.sh            # 5% of writers (fast, ~hundreds of clients)
#   SF=1.0 bash scripts/prepare_femnist.sh     # full FEMNIST (~3500 writers, large)
#   SF=0.05 K=64 TF=0.9 bash scripts/prepare_femnist.sh
# Env vars:
#   SF  data/writer fraction to keep (default 0.05). Use 1.0 for the full set.
#   K   minimum samples per writer to keep a writer (default 0).
#   TF  per-writer train fraction (default 0.9 -> 90% train / 10% test).
# =====================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/data/femnist"
WORK="${LEAF_WORKDIR:-${TMPDIR:-/tmp}/leaf_femnist}"
SF="${SF:-0.05}"
K="${K:-0}"
TF="${TF:-0.9}"

echo "[femnist] LEAF workdir: ${WORK}"
echo "[femnist] params: SF=${SF} (writer fraction)  K=${K} (min samples)  TF=${TF} (train frac)"
mkdir -p "${WORK}"

if [ ! -d "${WORK}/leaf" ]; then
  echo "[femnist] cloning LEAF ..."
  git clone --depth 1 https://github.com/TalwalkarLab/leaf.git "${WORK}/leaf"
fi

cd "${WORK}/leaf/data/femnist"
echo "[femnist] running LEAF preprocess.sh (downloads raw NIST on first run; this is slow) ..."
# -s niid: keep the natural (per-writer) non-IID grouping
# -t sample: split train/test by sample WITHIN each writer (writer present in both)
./preprocess.sh -s niid --sf "${SF}" -k "${K}" -t sample --tf "${TF}"

echo "[femnist] copying JSON into ${DEST} ..."
mkdir -p "${DEST}/train" "${DEST}/test"
cp data/train/*.json "${DEST}/train/"
cp data/test/*.json  "${DEST}/test/"

echo "[femnist] DONE. Files:"
echo "  train: $(ls "${DEST}/train" | wc -l | tr -d ' ') json"
echo "  test : $(ls "${DEST}/test"  | wc -l | tr -d ' ') json"
echo "[femnist] Now run with partition: natural, e.g.:"
echo "  python run_experiment.py --config configs/femnist_fedstep.yaml --algo fedavg"
echo "  (set data.partition: natural in the config to use the per-writer split)"
