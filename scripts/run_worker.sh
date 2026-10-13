#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
worker_id="${1:?Usage: bash scripts/run_worker.sh WORKER_ID PLAN_FILE [run options]}"
plan_file="${2:?PLAN_FILE is required}"
shift 2
if [[ ! "$worker_id" =~ ^[0-9]+$ ]]; then echo "WORKER_ID must be an integer" >&2; exit 2; fi
worker_id=$((10#$worker_id))
python_bin="${PYTHON_BIN:-python3}"
printf 'Runner starting on %s; Python=%s; plan=%s\n' "$(hostname)" "$python_bin" "$plan_file"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-8}"
export TOKENIZERS_PARALLELISM=false
unset CURRENT_TRIAL_ID
study_name="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["study_name"])' "$plan_file")"
mkdir -p "logs/$study_name"
printf -v worker_label '%02d' "$worker_id"
"$python_bin" -u experiments/cluster.py run --plan "$plan_file" --worker-id "$worker_id" "$@" \
  2>&1 | tee -a "logs/$study_name/worker_$worker_label.log"
