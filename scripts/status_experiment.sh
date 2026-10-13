#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
plan_file="${1:?Usage: bash scripts/status_experiment.sh PLAN_FILE}"
shift
"${PYTHON_BIN:-python3}" experiments/cluster.py status --plan "$plan_file" "$@"
