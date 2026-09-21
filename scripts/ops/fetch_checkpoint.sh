#!/usr/bin/env bash
# Fetch one explicitly selected, stable checkpoint and verify its expected SHA-256.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 REMOTE_PATH EXPECTED_SHA256 [REMOTE_HOST]" >&2
  exit 64
fi
remote_path="$1"
expected_sha="$2"
remote_host="${3:-${MOLECULE_REMOTE_HOST:-cbgpu@100.100.136.37}}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
basename="$(basename -- "$remote_path")"
destination_dir="$repo_root/records/checkpoints/files/$expected_sha"
destination="$destination_dir/$basename"
mkdir -p "$destination_dir"
rsync -a --partial "$remote_host:$remote_path" "$destination"
actual_sha="$(sha256sum -- "$destination" | cut -d' ' -f1)"
if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "checkpoint hash mismatch: expected=$expected_sha actual=$actual_sha" >&2
  exit 65
fi
printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$remote_host:$remote_path" "$expected_sha" "$destination" \
  >> "$repo_root/records/checkpoints/fetched.tsv"
echo "verified checkpoint: $destination"
