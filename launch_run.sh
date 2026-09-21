#!/usr/bin/env bash
# Outer log boundary: captures failures before Python can import run_logger.py.
set -uo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: $0 COMMAND [ARG ...]" >&2
  exit 64
fi

timestamp="$(date '+%Y%m%dT%H%M%S.%N%z')"
entry_name="$(basename -- "$1")"
entry_name="${entry_name//[^[:alnum:]_.-]/_}"
run_id="${MOLECULE_RUN_ID:-${timestamp}_${entry_name}_pid$$}"
run_root="${MOLECULE_RUN_ROOT:-./runs}"
run_dir="${MOLECULE_RUN_DIR:-${run_root}/${run_id}}"

mkdir -p -- "$(dirname -- "$run_dir")"
if ! mkdir -- "$run_dir"; then
  echo "refusing to reuse run directory: $run_dir" >&2
  exit 73
fi
run_dir="$(cd -- "$run_dir" && pwd)"
launcher_log="$run_dir/launcher_${timestamp}.log"
launcher_meta="$run_dir/launcher.meta"
export MOLECULE_RUN_ID="$run_id"
export MOLECULE_RUN_DIR="$run_dir"

exec > >(tee -a -- "$launcher_log") 2>&1
{
  printf 'run_id=%s\n' "$run_id"
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'pid=%s\n' "$$"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'cwd=%s\n' "$PWD"
  printf 'command='
  printf '%q ' "$@"
  printf '\nlauncher_log=%s\n' "$launcher_log"
  printf 'launcher_sha256=%s\n' "$(sha256sum -- "${BASH_SOURCE[0]}" | cut -d' ' -f1)"
} | tee -- "$launcher_meta"

"$@"
exit_code=$?
{
  printf 'ended_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'exit_code=%s\n' "$exit_code"
} | tee -a -- "$launcher_meta"
exit "$exit_code"
