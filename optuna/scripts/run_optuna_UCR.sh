#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-8}"
export TOKENIZERS_PARALLELISM=false

python optuna_SemSphere.py \
  --selected_dataset UCR \
  --study_name v2 \
  --device cuda \
  --njobs 1 \
  --trials 20 \
  --num_workers 0 \
  --num_epoch 20 \
  --d_model 32 \
  --tcn_layers 2 \
  --window_size 64 \
  --time_step 64 \
  --vus_window 64 \
  --seeds 1,2,3,4,5 \
  --lambda_align_min 0.0001 --lambda_align_max 1 \
  --lambda_oc_min 0.0001 --lambda_oc_max 1 \
  --score_beta_min 0 --score_beta_max 2 --score_beta_step 0.1 \
  "$@"

python optuna_SemSphere.py \
  --selected_dataset UCR \
  --study_name v3 \
  --device cuda \
  --njobs 1 \
  --trials 20 \
  --num_workers 0 \
  --num_epoch 10 \
  --d_model 32 \
  --tcn_layers 2 \
  --window_size 128 \
  --time_step 64 \
  --vus_window 64 \
  --seeds 1,2,3,4,5 \
  --lambda_align_min 0.0001 --lambda_align_max 1 \
  --lambda_oc_min 0.0001 --lambda_oc_max 1 \
  --score_beta_min 0 --score_beta_max 2 --score_beta_step 0.1 \
  "$@"

python optuna_SemSphere.py \
  --selected_dataset UCR \
  --study_name v4 \
  --device cuda \
  --njobs 1 \
  --trials 20 \
  --num_workers 0 \
  --num_epoch 10 \
  --d_model 32 \
  --tcn_layers 2 \
  --window_size 256 \
  --time_step 64 \
  --vus_window 64 \
  --seeds 1,2,3,4,5 \
  --lambda_align_min 0.0001 --lambda_align_max 1 \
  --lambda_oc_min 0.0001 --lambda_oc_max 1 \
  --score_beta_min 0 --score_beta_max 2 --score_beta_step 0.1 \
  "$@"
