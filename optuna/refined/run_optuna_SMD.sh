#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../scripts"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-8}"
export TOKENIZERS_PARALLELISM=false


python optuna_SemSphere.py \
  --selected_dataset SMD \
  --study_name v16 \
  --device cuda \
  --njobs 5 \
  --trials 20 \
  --num_workers 0 \
  --num_epoch 5 \
  --d_model 32 \
  --tcn_layers 2 \
  --attention_heads 4 \
  --project_dim 32 \
  --batch_size 128 \
  --window_size 128 \
  --time_step 258 \
  --vus_window 258 \
  --lr 0.0003 \
  --weight 0.0005 \
  --dropout 0.1 \
  --text_model intfloat/e5-base-v2 \
  --seeds 1,2,3,4,5 \
  --lambda_align_min 0.001 \
  --lambda_align_max 0.2 \
  --lambda_oc_min 0.01 \
  --lambda_oc_max 1 \
  --score_beta_min 0 \
  --score_beta_max 0.1 \
  --score_beta_step 0.01 \
  "$@"


python optuna_SemSphere.py \
  --selected_dataset SMD \
  --study_name v17 \
  --device cuda \
  --njobs 5 \
  --trials 20 \
  --num_workers 0 \
  --num_epoch 5 \
  --d_model 64 \
  --tcn_layers 2 \
  --attention_heads 4 \
  --project_dim 32 \
  --batch_size 128 \
  --window_size 128 \
  --time_step 258 \
  --vus_window 258 \
  --lr 0.0003 \
  --weight 0.0005 \
  --dropout 0.1 \
  --text_model intfloat/e5-base-v2 \
  --seeds 1,2,3,4,5 \
  --lambda_align_min 0.001 \
  --lambda_align_max 0.2 \
  --lambda_oc_min 0.01 \
  --lambda_oc_max 1 \
  --score_beta_min 0 \
  --score_beta_max 0.2 \
  --score_beta_step 0.01 \
  "$@"


python optuna_SemSphere.py \
  --selected_dataset SMD \
  --study_name v18 \
  --device cuda \
  --njobs 5 \
  --trials 20 \
  --num_workers 0 \
  --num_epoch 5 \
  --d_model 32 \
  --tcn_layers 2 \
  --attention_heads 4 \
  --project_dim 32 \
  --batch_size 128 \
  --window_size 256 \
  --time_step 258 \
  --vus_window 258 \
  --lr 0.0003 \
  --weight 0.0005 \
  --dropout 0.1 \
  --text_model intfloat/e5-base-v2 \
  --seeds 1,2,3,4,5 \
  --lambda_align_min 0.001 \
  --lambda_align_max 0.2 \
  --lambda_oc_min 0.01 \
  --lambda_oc_max 1 \
  --score_beta_min 0 \
  --score_beta_max 0.2 \
  --score_beta_step 0.01 \
  "$@"
