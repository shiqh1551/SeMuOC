#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
version=v9
if [[ $# -gt 0 && "$1" != --* ]]; then version="$1"; shift; fi
case "$version" in
  v4|v5|v6|v7|v8|v9)
    exec bash "$script_dir/run_ablation_WADI_${version}.sh" "$@" ;;
  all)
    for version in v4 v5 v6 v7 v8 v9; do
      bash "$script_dir/run_ablation_WADI_${version}.sh" "$@"
    done ;;
  *) echo "Version must be v4, v5, v6, v7, v8, v9 (recommended/default), or all" >&2; exit 2 ;;
esac
