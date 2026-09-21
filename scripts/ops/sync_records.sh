#!/usr/bin/env bash
# Synchronize reproducibility records immediately; never transfer checkpoints/data.
set -euo pipefail
remote_host="${1:-${MOLECULE_REMOTE_HOST:-cbgpu@100.100.136.37}}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
stamp="$(date '+%Y%m%dT%H%M%S%z')"
host_label="${remote_host//@/_at_}"; host_label="${host_label//[^[:alnum:]_.-]/_}"
log_dest="$repo_root/logs/remote/$host_label"
doc_dest="$repo_root/docs/remote/$host_label"
record_dest="$repo_root/records/remote/$host_label"
log_history="$repo_root/logs/remote_history/$stamp"
doc_history="$repo_root/docs/remote_history/$stamp"
record_history="$repo_root/records/remote_history/$stamp"
sync_log="$repo_root/logs/sync/${stamp}.log"
mkdir -p "$log_dest" "$doc_dest" "$record_dest" "$log_history" "$doc_history" "$record_history" "$(dirname -- "$sync_log")"
roots=(molecule-AI molecule-AI-control molecule-AI-c1-feature molecule-AI-c1-control molecule-AI-pec)
common_excludes=(--exclude='/data/***' --exclude='/checkpoints*/***' --exclude='*.pt' --exclude='/.git/***' --exclude='/.venv/***')
{
  printf 'sync_id=%s\nremote=%s\nstarted_at=%s\n' "$stamp" "$remote_host" "$(date --iso-8601=seconds)"
  for name in "${roots[@]}"; do
    printf '\n[%s logs]\n' "$name"; mkdir -p "$log_dest/$name"
    rsync -a --prune-empty-dirs --partial --itemize-changes --backup --backup-dir="$log_history/$name" \
      "${common_excludes[@]}" --include='*/' --include='*.log' --exclude='*' \
      "$remote_host:/home/cbgpu/$name/" "$log_dest/$name/"

    printf '[%s docs]\n' "$name"; mkdir -p "$doc_dest/$name"
    rsync -a --prune-empty-dirs --partial --itemize-changes --backup --backup-dir="$doc_history/$name" \
      "${common_excludes[@]}" --include='/docs/***' --include='/*.md' --exclude='*' \
      "$remote_host:/home/cbgpu/$name/" "$doc_dest/$name/"

    printf '[%s records]\n' "$name"; mkdir -p "$record_dest/$name"
    rsync -a --prune-empty-dirs --partial --itemize-changes --backup --backup-dir="$record_history/$name" \
      "${common_excludes[@]}" --exclude='/docs/***' --exclude='/logs/***' --exclude='*.log' \
      --include='*/' --include='*.json' --include='*.jsonl' --include='*.csv' --include='*.tsv' \
      --include='*.txt' --include='*.py' --include='*.sh' --include='*.ipynb' --include='pid' \
      --exclude='*' "$remote_host:/home/cbgpu/$name/" "$record_dest/$name/"
  done
  printf '\nended_at=%s\n' "$(date --iso-8601=seconds)"
} 2>&1 | tee "$sync_log"
printf 'logs: %s\ndocs: %s\nrecords: %s\n' "$log_dest" "$doc_dest" "$record_dest"
