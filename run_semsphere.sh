#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
selected_dataset="${1:-ASD}"
selected_device="${2:-cuda}"
for seed in 1 2 3 4 5; do
  python SemSphere_method.py --selected_dataset "$selected_dataset" --device "$selected_device" \
    --seed "$seed" --study_name semsphere_v1 --experiment_description semsphere_v1 "${@:3}"
done
