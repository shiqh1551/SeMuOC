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
  exec 9>logs/ablation_exact_locks/IOpsCompetition_ablation_exact_v7.lock
  flock 9
  pids=()
  for seed in 1 2 3 4 5; do
    python3 SemSphere_ablation_method.py \
      --selected_dataset IOpsCompetition \
      --baseline_config configs/ablation_exact/v7/IOpsCompetition.json \
      --study_name "ablation_exact_v7_seed${seed}" \
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
      --lambda_align 9.524762810221545e-05 \
      --lambda_oc 0.0002284044740855875 \
      --score_alpha 1.0 \
      --score_beta 0.09 \
      "$@" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  pids=()
  if [[ "$failed" != 0 ]]; then echo "A seed failed; completed seed results are retained." >&2; exit 1; fi
  python3 scripts/ablation/exact/merge.py --version v7 --dataset IOpsCompetition
  flock -u 9
  exec 9>&-
fi

if [[ "$selected" == all || "$selected" == ASD ]]; then
  mkdir -p logs/ablation_exact_locks
  exec 9>logs/ablation_exact_locks/ASD_ablation_exact_v7.lock
  flock 9
  pids=()
  for seed in 1 2 3 4 5; do
    python3 SemSphere_ablation_method.py \
      --selected_dataset ASD \
      --baseline_config configs/ablation_exact/v7/ASD.json \
      --study_name "ablation_exact_v7_seed${seed}" \
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
      --lambda_align 0.0029237611243148934 \
      --lambda_oc 0.0003103174399434219 \
      --score_alpha 1.0 \
      --score_beta 1.3 \
      "$@" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  pids=()
  if [[ "$failed" != 0 ]]; then echo "A seed failed; completed seed results are retained." >&2; exit 1; fi
  python3 scripts/ablation/exact/merge.py --version v7 --dataset ASD
  flock -u 9
  exec 9>&-
fi

if [[ "$selected" == all || "$selected" == Exathlon ]]; then
  mkdir -p logs/ablation_exact_locks
  exec 9>logs/ablation_exact_locks/Exathlon_ablation_exact_v7.lock
  flock 9
  pids=()
  for seed in 1 2 3 4 5; do
    python3 SemSphere_ablation_method.py \
      --selected_dataset Exathlon \
      --baseline_config configs/ablation_exact/v7/Exathlon.json \
      --study_name "ablation_exact_v7_seed${seed}" \
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
      --lambda_align 0.242806895670078 \
      --lambda_oc 0.7834840381449714 \
      --score_alpha 1.0 \
      --score_beta 0.03 \
      "$@" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  pids=()
  if [[ "$failed" != 0 ]]; then echo "A seed failed; completed seed results are retained." >&2; exit 1; fi
  python3 scripts/ablation/exact/merge.py --version v7 --dataset Exathlon
  flock -u 9
  exec 9>&-
fi

if [[ "$selected" == all || "$selected" == WADI ]]; then
  mkdir -p logs/ablation_exact_locks
  exec 9>logs/ablation_exact_locks/WADI_ablation_exact_v7.lock
  flock 9
  pids=()
  for seed in 1 2 3 4 5; do
    python3 SemSphere_ablation_method.py \
      --selected_dataset WADI \
      --baseline_config configs/ablation_exact/v7/WADI.json \
      --study_name "ablation_exact_v7_seed${seed}" \
      --seeds "$seed" --device cuda --njobs 1 --run_all_ablation \
      --num_epoch 5 \
      --window_size 128 \
      --time_step 128 \
      --vus_window 128 \
      --d_model 64 \
      --tcn_layers 3 \
      --attention_heads 4 \
      --project_dim 32 \
      --batch_size 512 \
      --lambda_recon 1.0 \
      --lambda_align 0.16789034732053 \
      --lambda_oc 0.14596258094306092 \
      --score_alpha 1.0 \
      --score_beta 1.7000000000000002 \
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
  python3 scripts/ablation/exact/merge.py --version v7 --dataset WADI
  flock -u 9
  exec 9>&-
fi
