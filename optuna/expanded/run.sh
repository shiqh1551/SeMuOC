#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for dataset in KPI ASD Exathlon WADI; do
  bash "run_optuna_${dataset}.sh" "$@"
done
