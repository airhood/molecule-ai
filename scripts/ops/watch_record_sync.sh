#!/usr/bin/env bash
# Periodically pull logs/docs/records. Checkpoints remain excluded by sync_records.sh.
set -u
interval="${1:-120}"
case "$interval" in (*[!0-9]*|'') echo "interval must be an integer number of seconds" >&2; exit 64;; esac
if (( interval < 30 )); then echo "interval must be at least 30 seconds" >&2; exit 64; fi
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
lock_file="$repo_root/records/record_sync.lock"
mkdir -p "$repo_root/records" "$repo_root/logs/sync"
exec 9>"$lock_file"
if ! flock -n 9; then
  echo "record sync watcher is already running" >&2
  exit 73
fi
printf 'watcher_started_at=%s interval_seconds=%s pid=%s\n' "$(date --iso-8601=seconds)" "$interval" "$$"
while true; do
  "$repo_root/scripts/ops/sync_records.sh" || \
    printf 'sync_failed_at=%s exit=%s\n' "$(date --iso-8601=seconds)" "$?" >&2
  sleep "$interval"
done
