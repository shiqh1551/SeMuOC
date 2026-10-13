#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-8}"
export TOKENIZERS_PARALLELISM=false
python optuna_SemSphere.py \
  --selected_dataset WADI \
  --study_name v2 \
  --device cuda \
  --njobs 5 \
  --trials 50 \
  --num_workers 0 \
  --num_epoch 5 \
  --d_model 64 \
  --tcn_layers 3 \
  --window_size 128 \
  --time_step 128 \
  --vus_window 128 \
  --seeds 1,2,3,4,5 \
  --batch_size 512 \
  --lambda_align_min 0.0001 --lambda_align_max 1 \
  --lambda_oc_min 0.0001 --lambda_oc_max 1 \
  --score_beta_min 0 --score_beta_max 2 --score_beta_step 0.1 \
  "$@"
