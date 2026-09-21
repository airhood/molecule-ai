#!/usr/bin/env python3
"""Inventory remote checkpoints without copying them; hash only new/changed files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

ROOTS = [
    "/home/cbgpu/molecule-AI",
    "/home/cbgpu/molecule-AI-control",
    "/home/cbgpu/molecule-AI-c1-feature",
    "/home/cbgpu/molecule-AI-c1-control",
    "/home/cbgpu/molecule-AI-pec",
]


def inventory(host: str) -> dict[str, dict]:
    roots = " ".join(shlex.quote(root) for root in ROOTS)
    command = f"find {roots} -type f -name '*.pt' -printf '%p\\t%s\\t%T@\\n'"
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", host, command],
                            capture_output=True, text=True, check=True)
    found = {}
    for line in result.stdout.splitlines():
        path, size, mtime = line.split("\t")
        # Processed dataset chunks also use .pt, but they are data rather than
        # model weights and must never enter the checkpoint registry.
        if "/data/" in path:
            continue
        found[path] = {"size": int(size), "mtime": mtime}
    return found


def hashes(host: str, paths: list[str]) -> dict[str, str]:
    if not paths:
        return {}
    command = "sha256sum -- " + " ".join(shlex.quote(path) for path in paths)
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", host, command],
                            capture_output=True, text=True, check=True)
    output = {}
    for line in result.stdout.splitlines():
        digest, path = line.split(None, 1)
        output[path] = digest
    return output


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="cbgpu@100.100.136.37")
    parser.add_argument("--output-dir", default="records/checkpoints")
    parser.add_argument("--skip-hash", action="store_true",
                        help="Update size/mtime only; leave new files unhashed.")
    parser.add_argument("--seed-snapshot", default=None,
                        help="Verified backup root containing SHA256SUMS and trees/. "
                             "A hash is reused only when size and mtime still match.")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "registry.json"
    previous = json.loads(registry_path.read_text()) if registry_path.exists() else {"checkpoints": {}}
    old_entries = previous.get("checkpoints", {})

    snapshot_entries = {}
    if args.seed_snapshot:
        snapshot = Path(args.seed_snapshot).resolve()
        sums = snapshot / "SHA256SUMS"
        marker = "/trees/"
        if sums.exists():
            for line in sums.read_text().splitlines():
                digest, stored = line.split(None, 1)
                if marker not in stored:
                    continue
                suffix = stored.split(marker, 1)[1]
                worktree, relative = suffix.split("/", 1)
                local = snapshot / "trees" / worktree / relative
                if not local.exists():
                    continue
                stat = local.stat()
                snapshot_entries[f"/home/cbgpu/{worktree}/{relative}"] = {
                    "sha256": digest, "size": stat.st_size, "mtime": stat.st_mtime,
                }

    before = inventory(args.host)
    seeded = {
        path: value for path, value in snapshot_entries.items()
        if path in before and value["size"] == before[path]["size"]
        and abs(value["mtime"] - float(before[path]["mtime"])) < 0.001
    }
    changed = [path for path, meta in before.items()
               if path not in seeded and (path not in old_entries
               or old_entries[path].get("size") != meta["size"]
               or old_entries[path].get("mtime") != meta["mtime"]
               or not old_entries[path].get("sha256"))]
    digests = {} if args.skip_hash else hashes(args.host, changed)
    after = inventory(args.host)
    now = datetime.now().astimezone().isoformat()
    entries = {}
    for path, meta in after.items():
        old = old_entries.get(path, {})
        stable = before.get(path) == meta
        digest = digests.get(path) if stable else None
        hash_source = "remote_sha256" if digest else None
        if not digest and stable and path in seeded:
            digest = seeded[path]["sha256"]
            hash_source = "verified_20260920_backup"
        if not digest and old.get("size") == meta["size"] and old.get("mtime") == meta["mtime"]:
            digest = old.get("sha256")
            hash_source = old.get("hash_source")
        entries[path] = {
            **meta,
            "sha256": digest,
            "hash_source": hash_source,
            "status": "stable" if stable and digest else ("stable_unhashed" if stable else "changing"),
            "first_seen": old.get("first_seen", now),
            "last_seen": now,
        }

    registry = {"schema_version": 2, "host": args.host, "updated_at": now,
                "checkpoints": entries}
    atomic_json(registry_path, registry)
    with (output_dir / "registry.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["remote_path", "size", "mtime", "sha256", "hash_source", "status", "first_seen", "last_seen"])
        for path, meta in sorted(entries.items()):
            writer.writerow([path, meta["size"], meta["mtime"], meta.get("sha256") or "",
                             meta.get("hash_source") or "", meta["status"],
                             meta["first_seen"], meta["last_seen"]])
    print(f"checkpoint registry: {len(entries)} files, {len(changed)} new/changed")
    print(registry_path)


if __name__ == "__main__":
    main()
