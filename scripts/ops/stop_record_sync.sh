#!/usr/bin/env bash
set -euo pipefail
systemctl --user stop molecule-ai-record-sync.timer
echo "molecule-ai-record-sync.timer stopped"
