#!/usr/bin/env bash
set -euo pipefail
if [[ ! -f "$HOME/.config/systemd/user/molecule-ai-record-sync.timer" ]]; then
  echo "timer is not installed; run scripts/ops/install_record_sync_timer.sh first" >&2
  exit 69
fi
systemctl --user enable --now molecule-ai-record-sync.timer
systemctl --user start --no-block molecule-ai-record-sync.service
echo "molecule-ai-record-sync.timer started"
