#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-8}"
export TOKENIZERS_PARALLELISM=false

python SemSphere_method.py \
  --selected_dataset ASD \
  --study_name vis \
  --device cpu \
  --seeds 1,2,3,4,5 \
  --njobs 5 \
  --num_workers 0 \
  --num_epoch 20 \
  --d_model 32 \
  --tcn_layers 2 \
  --window_size 64 \
  --time_step 64 \
  --lambda_recon 1 \
  --lambda_align 0.1 \
  --lambda_oc 0.1 \
  --score_alpha 1 \
  --score_beta 0.5 \
  "$@"
