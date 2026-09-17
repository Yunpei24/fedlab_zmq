#!/bin/bash
# Prepare manifests beforehand. Dry-run by default; --submit actually submits.
# One array chunk per invocation: never launches an entire 88k-run grid.
set -euo pipefail
: "${FEDLAB_REPO:?Set FEDLAB_REPO}"
export FEDLAB_ACCOUNT="${FEDLAB_ACCOUNT:-manapy-1wabcjwe938-premium-cpu}"
export FEDLAB_PARTITION="${FEDLAB_PARTITION:-compute}"
export FEDLAB_QOS="${FEDLAB_QOS:-premium-cpu}"
export PYTHONNOUSERSITE=1
: "${FAR_MANIFEST:?Set FAR_MANIFEST}"
: "${FAR_DATA_ROOT:?Set FAR_DATA_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
offset="${JOB_OFFSET:-0}"
chunk="${ARRAY_CHUNK:-1000}"
parallel="${MAX_PARALLEL:-4}"
for number in "$offset" "$chunk" "$parallel"; do
    [[ "$number" =~ ^[0-9]+$ ]] || { echo 'Array settings must be nonnegative integers' >&2; exit 2; }
done
(( chunk > 0 && parallel > 0 )) || exit 2
total=$("${PYTHON_BIN}" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["tasks"]))' "$FAR_MANIFEST")
(( offset < total )) || { echo "Offset $offset outside $total tasks" >&2; exit 2; }
count=$(( total - offset )); (( count <= chunk )) || count="$chunk"
export FEDLAB_REPO FAR_MANIFEST FAR_DATA_ROOT JOB_OFFSET="$offset"
args=(--account="$FEDLAB_ACCOUNT" --array="0-$((count-1))%${parallel}" --export=ALL)
[[ -z "${FEDLAB_PARTITION:-}" ]] || args+=(--partition="$FEDLAB_PARTITION")
[[ -z "${FEDLAB_QOS:-}" ]] || args+=(--qos="$FEDLAB_QOS")
[[ -z "${FAR_TIME_LIMIT:-}" ]] || args+=(--time="$FAR_TIME_LIMIT")
[[ -z "${FAR_CPUS:-}" ]] || args+=(--cpus-per-task="$FAR_CPUS")
[[ -z "${FAR_MEMORY:-}" ]] || args+=(--mem="$FAR_MEMORY")
cd "$FEDLAB_REPO"
printf 'Stage tasks: %s; submitting indices %s..%s; concurrency %s\n' "$total" "$offset" "$((offset+count-1))" "$parallel"
printf '%q ' sbatch "${args[@]}" hpc/run_far_dp_effect_cpu.slurm
printf '\n'
if [[ "${1:-}" == '--submit' ]]; then
    sbatch "${args[@]}" hpc/run_far_dp_effect_cpu.slurm
elif [[ -n "${1:-}" ]]; then
    echo 'Only --submit is accepted; omit it for a dry-run' >&2; exit 2
else
    echo 'Dry-run only. Add --submit to submit this chunk.'
fi
