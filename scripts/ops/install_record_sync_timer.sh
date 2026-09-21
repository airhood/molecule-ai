#!/usr/bin/env bash
set -euo pipefail
interval="${1:-2min}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
unit_dir="$HOME/.config/systemd/user"
service="$unit_dir/molecule-ai-record-sync.service"
timer="$unit_dir/molecule-ai-record-sync.timer"
mkdir -p "$unit_dir"
cat > "$service" <<EOF
[Unit]
Description=Synchronize molecule-AI logs, documents, and records (no checkpoints)

[Service]
Type=oneshot
WorkingDirectory=$repo_root
ExecStart=/usr/bin/bash "$repo_root/scripts/ops/sync_records.sh"
EOF
cat > "$timer" <<EOF
[Unit]
Description=Run molecule-AI record synchronization every $interval

[Timer]
OnBootSec=30s
OnUnitActiveSec=$interval
AccuracySec=10s
Unit=molecule-ai-record-sync.service

[Install]
WantedBy=timers.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now molecule-ai-record-sync.timer
systemctl --user start --no-block molecule-ai-record-sync.service
echo "installed: $service"
echo "installed: $timer"
