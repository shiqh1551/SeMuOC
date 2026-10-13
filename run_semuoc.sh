#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
selected_dataset="${1:-ASD}"
selected_device="${2:-cuda}"
for seed in 1 2 3 4 5; do
  python SeMuOC.py --selected_dataset "$selected_dataset" --device "$selected_device" \
    --seed "$seed" "${@:3}"
done
