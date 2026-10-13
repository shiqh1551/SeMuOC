#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../scripts"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-8}"
export TOKENIZERS_PARALLELISM=false


python optuna_SemSphere.py \
  --selected_dataset IOpsCompetition \
  --study_name v8 \
  --device cuda \
  --njobs 10 \
  --trials 20 \
  --num_workers 0 \
  --num_epoch 20 \
  --d_model 32 \
  --tcn_layers 2 \
  --attention_heads 4 \
  --project_dim 32 \
  --batch_size 512 \
  --window_size 64 \
  --time_step 64 \
  --vus_window 64 \
  --lr 0.0003 \
  --weight 0.0005 \
  --dropout 0.1 \
  --text_model intfloat/e5-base-v2 \
  --seeds 1,2,3,4,5 \
  --lambda_align_min 0.0001 \
  --lambda_align_max 0.1 \
  --lambda_oc_min 1e-05 \
  --lambda_oc_max 0.005 \
  --score_beta_min 0 \
  --score_beta_max 0.3 \
  --score_beta_step 0.01 \
  "$@"


python optuna_SemSphere.py \
  --selected_dataset IOpsCompetition \
  --study_name v9 \
  --device cuda \
  --njobs 10 \
  --trials 20 \
  --num_workers 0 \
  --num_epoch 20 \
  --d_model 32 \
  --tcn_layers 2 \
  --attention_heads 4 \
  --project_dim 32 \
  --batch_size 512 \
  --window_size 64 \
  --time_step 64 \
  --vus_window 64 \
  --lr 0.0003 \
  --weight 0.0005 \
  --dropout 0.1 \
  --text_model intfloat/e5-base-v2 \
  --seeds 1,2,3,4,5 \
  --lambda_align_min 1e-05 \
  --lambda_align_max 1 \
  --lambda_oc_min 1e-06 \
  --lambda_oc_max 0.01 \
  --score_beta_min 0 \
  --score_beta_max 0.4 \
  --score_beta_step 0.02 \
  "$@"


python optuna_SemSphere.py \
  --selected_dataset IOpsCompetition \
  --study_name v10 \
  --device cuda \
  --njobs 10 \
  --trials 20 \
  --num_workers 0 \
  --num_epoch 30 \
  --d_model 32 \
  --tcn_layers 2 \
  --attention_heads 4 \
  --project_dim 32 \
  --batch_size 512 \
  --window_size 64 \
  --time_step 64 \
  --vus_window 64 \
  --lr 0.0003 \
  --weight 0.0005 \
  --dropout 0.1 \
  --text_model intfloat/e5-base-v2 \
  --seeds 1,2,3,4,5 \
  --lambda_align_min 1e-05 \
  --lambda_align_max 0.1 \
  --lambda_oc_min 1e-05 \
  --lambda_oc_max 0.01 \
  --score_beta_min 0 \
  --score_beta_max 0.3 \
  --score_beta_step 0.01 \
  "$@"
