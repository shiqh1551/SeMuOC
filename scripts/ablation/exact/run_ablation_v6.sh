#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-1}"
export TOKENIZERS_PARALLELISM=false
selected=all
if [[ $# -gt 0 && "$1" != --* ]]; then selected="$1"; shift; fi
[[ "$selected" != KPI ]] || selected=IOpsCompetition
case "$selected" in all|IOpsCompetition|ASD|Exathlon|WADI) ;; *) echo "Dataset must be KPI, ASD, Exathlon, WADI, or all" >&2; exit 2;; esac
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

if [[ "$selected" == all || "$selected" == IOpsCompetition ]]; then
  mkdir -p logs/ablation_exact_locks
  exec 9>logs/ablation_exact_locks/IOpsCompetition_ablation_exact_v6.lock
  flock 9
  pids=()
  for seed in 1 2 3 4 5; do
    python3 SemSphere_ablation_method.py \
      --selected_dataset IOpsCompetition \
      --baseline_config configs/ablation_exact/v6/IOpsCompetition.json \
      --study_name "ablation_exact_v6_seed${seed}" \
      --seeds "$seed" --device cuda --njobs 1 --run_all_ablation \
      --num_epoch 30 \
      --window_size 64 \
      --time_step 64 \
      --vus_window 64 \
      --d_model 32 \
      --tcn_layers 2 \
      --attention_heads 4 \
      --project_dim 32 \
      --batch_size 512 \
      --lambda_recon 1.0 \
      --lambda_align 1.882832579474493e-05 \
      --lambda_oc 5.18546123341844e-05 \
      --score_alpha 1.0 \
      --score_beta 0.03 \
      "$@" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  pids=()
  if [[ "$failed" != 0 ]]; then echo "A seed failed; completed seed results are retained." >&2; exit 1; fi
  python3 scripts/ablation/exact/merge.py --version v6 --dataset IOpsCompetition
  flock -u 9
  exec 9>&-
fi

if [[ "$selected" == all || "$selected" == ASD ]]; then
  mkdir -p logs/ablation_exact_locks
  exec 9>logs/ablation_exact_locks/ASD_ablation_exact_v6.lock
  flock 9
  pids=()
  for seed in 1 2 3 4 5; do
    python3 SemSphere_ablation_method.py \
      --selected_dataset ASD \
      --baseline_config configs/ablation_exact/v6/ASD.json \
      --study_name "ablation_exact_v6_seed${seed}" \
      --seeds "$seed" --device cuda --njobs 1 --run_all_ablation \
      --num_epoch 10 \
      --window_size 32 \
      --time_step 16 \
      --vus_window 32 \
      --d_model 96 \
      --tcn_layers 3 \
      --attention_heads 8 \
      --project_dim 32 \
      --batch_size 128 \
      --lambda_recon 1.0 \
      --lambda_align 0.00010907212655624029 \
      --lambda_oc 0.0005665070554617389 \
      --score_alpha 1.0 \
      --score_beta 0.8500000000000001 \
      "$@" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  pids=()
  if [[ "$failed" != 0 ]]; then echo "A seed failed; completed seed results are retained." >&2; exit 1; fi
  python3 scripts/ablation/exact/merge.py --version v6 --dataset ASD
  flock -u 9
  exec 9>&-
fi

if [[ "$selected" == all || "$selected" == Exathlon ]]; then
  mkdir -p logs/ablation_exact_locks
  exec 9>logs/ablation_exact_locks/Exathlon_ablation_exact_v6.lock
  flock 9
  pids=()
  for seed in 1 2 3 4 5; do
    python3 SemSphere_ablation_method.py \
      --selected_dataset Exathlon \
      --baseline_config configs/ablation_exact/v6/Exathlon.json \
      --study_name "ablation_exact_v6_seed${seed}" \
      --seeds "$seed" --device cuda --njobs 1 --run_all_ablation \
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
      --lambda_align 1.583770205572279e-05 \
      --lambda_oc 0.007458056678534306 \
      --score_alpha 1.0 \
      --score_beta 0.06 \
      "$@" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  pids=()
  if [[ "$failed" != 0 ]]; then echo "A seed failed; completed seed results are retained." >&2; exit 1; fi
  python3 scripts/ablation/exact/merge.py --version v6 --dataset Exathlon
  flock -u 9
  exec 9>&-
fi

if [[ "$selected" == all || "$selected" == WADI ]]; then
  mkdir -p logs/ablation_exact_locks
  exec 9>logs/ablation_exact_locks/WADI_ablation_exact_v6.lock
  flock 9
  pids=()
  for seed in 1 2 3 4 5; do
    python3 SemSphere_ablation_method.py \
      --selected_dataset WADI \
      --baseline_config configs/ablation_exact/v6/WADI.json \
      --study_name "ablation_exact_v6_seed${seed}" \
      --seeds "$seed" --device cuda --njobs 1 --run_all_ablation \
      --num_epoch 10 \
      --window_size 128 \
      --time_step 128 \
      --vus_window 128 \
      --d_model 32 \
      --tcn_layers 2 \
      --attention_heads 4 \
      --project_dim 32 \
      --batch_size 128 \
      --lambda_recon 1.0 \
      --lambda_align 0.20327053901876696 \
      --lambda_oc 6.642354259129604e-05 \
      --score_alpha 1.0 \
      --score_beta 0.8 \
      "$@" &
    pids+=("$!")
    if [[ ${#pids[@]} -ge 2 ]]; then
      failed=0
      for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
      pids=()
      if [[ "$failed" != 0 ]]; then echo "A seed failed; completed seed results are retained." >&2; exit 1; fi
    fi
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  pids=()
  if [[ "$failed" != 0 ]]; then echo "A seed failed; completed seed results are retained." >&2; exit 1; fi
  python3 scripts/ablation/exact/merge.py --version v6 --dataset WADI
  flock -u 9
  exec 9>&-
fi
