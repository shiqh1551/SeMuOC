#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../scripts"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-8}"
export TOKENIZERS_PARALLELISM=false


python optuna_SemSphere.py \
  --selected_dataset ASD \
  --study_name v21 \
  --device cuda \
  --njobs 10 \
  --trials 20 \
  --num_workers 0 \
  --num_epoch 10 \
  --d_model 96 \
  --tcn_layers 3 \
  --attention_heads 8 \
  --project_dim 32 \
  --batch_size 128 \
  --window_size 32 \
  --time_step 16 \
  --vus_window 32 \
  --lr 0.0003 \
  --weight 0.0005 \
  --dropout 0.2 \
  --text_model intfloat/e5-base-v2 \
  --seeds 1,2,3,4,5 \
  --lambda_align_min 1e-05 \
  --lambda_align_max 10 \
  --lambda_oc_min 1e-05 \
  --lambda_oc_max 100 \
  --score_beta_min 0.4 \
  --score_beta_max 2 \
  --score_beta_step 0.05 \
  --old_best v11 \
  "$@"

python optuna_SemSphere.py \
  --selected_dataset ASD \
  --study_name v22 \
  --device cuda \
  --njobs 10 \
  --trials 20 \
  --num_workers 0 \
  --num_epoch 10 \
  --d_model 96 \
  --tcn_layers 3 \
  --attention_heads 8 \
  --project_dim 32 \
  --batch_size 128 \
  --window_size 32 \
  --time_step 16 \
  --vus_window 32 \
  --lr 0.0003 \
  --weight 0.0005 \
  --dropout 0.2 \
  --text_model intfloat/e5-base-v2 \
  --seeds 1,2,3,4,5 \
  --lambda_align_min 1e-05 \
  --lambda_align_max 0.03 \
  --lambda_oc_min 1 \
  --lambda_oc_max 100 \
  --score_beta_min 0.4 \
  --score_beta_max 2 \
  --score_beta_step 0.05 \
  "$@"
