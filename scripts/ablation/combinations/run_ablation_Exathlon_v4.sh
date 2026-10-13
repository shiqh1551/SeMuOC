#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-1}"
export TOKENIZERS_PARALLELISM=false
for arg in "$@"; do
  case "$arg" in --study_name|--study_name=*|--seeds|--seeds=*|--baseline_config|--baseline_config=*|--selected_dataset|--selected_dataset=*)
    echo "Do not override dataset/study/seeds/baseline; use a separate version." >&2; exit 2;;
  esac
done
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p logs/ablation_exact_locks
exec 9>logs/ablation_exact_locks/Exathlon_ablation_exact_v4.lock
flock 9
pids=()
for seed in 1 2 3 4 5; do
  python3 SemSphere_ablation_method.py \
    --selected_dataset Exathlon \
    --baseline_config configs/ablation_exact/v4/Exathlon.json \
    --study_name "ablation_exact_v4_seed${seed}" \
    --seeds "$seed" --device cuda --njobs 1 --run_combination_ablation \
    --num_epoch 5 \
    --window_size 64 \
    --time_step 64 \
    --vus_window 64 \
    --d_model 32 \
    --tcn_layers 2 \
    --attention_heads 4 \
    --project_dim 32 \
    --batch_size 128 \
    --lambda_recon 1.0 \
    --lambda_align 7.389436599280714e-05 \
    --lambda_oc 63.05143340184449 \
    --score_alpha 1.0 \
    --score_beta 0.03 \
    "$@" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
pids=()
if [[ "$failed" != 0 ]]; then echo "A seed failed; completed seed results are retained." >&2; exit 1; fi
python3 scripts/ablation/exact/merge.py --case_group combinations --version v4 --dataset Exathlon
flock -u 9
exec 9>&-
