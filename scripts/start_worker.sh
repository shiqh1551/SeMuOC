#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
background=false
if [[ "${1:-}" == "--background" ]]; then
  background=true
  shift
fi
worker_id="${1:?Usage: bash scripts/start_worker.sh [--background] WORKER_ID PLAN_FILE [run options]}"
plan_file="${2:?PLAN_FILE is required}"
shift 2
if [[ ! "$worker_id" =~ ^[0-9]+$ ]]; then echo "WORKER_ID must be an integer" >&2; exit 2; fi
worker_id=$((10#$worker_id))
python_bin="${PYTHON_BIN:-python3}"
study_name="$("$python_bin" -c 'import json,sys; p=json.load(open(sys.argv[1])); n=int(sys.argv[2]); assert 1<=n<=p["nodes"], "Invalid WORKER_ID"; print(p["study_name"])' "$plan_file" "$worker_id")"
printf -v worker_label '%02d' "$worker_id"
mkdir -p "logs/$study_name"
launch_log="logs/$study_name/worker_$worker_label.launch.log"
if [[ "$background" == false ]]; then
  printf 'Running worker %s in foreground; this command waits for the worker to exit.\n' "$worker_label"
  exec bash scripts/run_worker.sh "$worker_id" "$plan_file" "$@"
fi
printf 'Background mode: the platform must keep this server allocated after this command returns.\n'
nohup bash scripts/run_worker.sh "$worker_id" "$plan_file" "$@" >> "$launch_log" 2>&1 < /dev/null &
launcher_pid=$!
printf 'Started worker %s (launcher PID %s). Log: %s\n' "$worker_label" "$launcher_pid" "$launch_log"
