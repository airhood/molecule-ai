#!/usr/bin/env python3
"""Build a chronological model/run/checkpoint/log catalog from preserved evidence."""
from __future__ import annotations

import argparse
import json
import re
import shlex
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HOST_LABEL = "cbgpu_at_100.100.136.37"
REMOTE_PREFIX = "/home/cbgpu/"
EPOCH_RE = re.compile(r"epoch(\d+)")


def iso_from_epoch(value: str | float) -> str:
    return datetime.fromtimestamp(float(value)).astimezone().isoformat()


def load_backup_hashes(path: Path) -> dict[str, str]:
    hashes = {}
    if not path.exists():
        return hashes
    marker = "/trees/"
    for line in path.read_text().splitlines():
        digest, stored = line.split(None, 1)
        if marker not in stored:
            continue
        suffix = stored.split(marker, 1)[1]
        worktree, relative = suffix.split("/", 1)
        hashes[f"{REMOTE_PREFIX}{worktree}/{relative}"] = digest
    return hashes


def remote_parts(remote_dir: str) -> tuple[str, str]:
    relative = remote_dir.removeprefix(REMOTE_PREFIX)
    worktree, _, rest = relative.partition("/")
    return worktree, rest


def local_checkpoint_path(snapshot: Path, remote_path: str) -> str | None:
    relative = remote_path.removeprefix(REMOTE_PREFIX)
    candidate = snapshot / "trees" / relative
    return str(candidate) if candidate.exists() else None


def log_record(repo: Path, worktree: str, name: str) -> dict:
    local = repo / "logs" / "remote" / HOST_LABEL / worktree / name
    remote = f"{REMOTE_PREFIX}{worktree}/{name}"
    return {
        "role": "stdout" if "stdout" in name else "named_log",
        "remote_path": remote,
        "local_path": str(local),
        "present_locally": local.exists(),
        "size": local.stat().st_size if local.exists() else None,
        "mtime": datetime.fromtimestamp(local.stat().st_mtime).astimezone().isoformat() if local.exists() else None,
        "evidence": "exact_path_from_override"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="records/checkpoints/registry.json")
    parser.add_argument("--overrides", default="records/model_catalog_overrides.json")
    parser.add_argument("--snapshot", default="../molecule-AI_server_snapshot_20260920T123633+0000")
    parser.add_argument("--output", default="records/model_catalog.json")
    args = parser.parse_args()
    repo = Path.cwd()
    registry = json.loads(Path(args.registry).read_text())
    overrides = json.loads(Path(args.overrides).read_text()).get("models", {})
    snapshot = Path(args.snapshot).resolve()
    backup_hashes = load_backup_hashes(snapshot / "SHA256SUMS")

    groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    auxiliary: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for remote_path, meta in registry["checkpoints"].items():
        path = Path(remote_path)
        if path.parent.name.startswith("checkpoints"):
            groups[str(path.parent)].append((remote_path, meta))
        else:
            worktree, _ = remote_parts(str(path.parent))
            # Root .pt files mix regressors, old weights, and generated tensors.
            # Keep them visible but do not silently call all of them model weights.
            auxiliary[worktree].append((remote_path, meta))

    manifests = []
    manifest_root = repo / "records" / "remote" / HOST_LABEL
    for path in manifest_root.rglob("manifest.json") if manifest_root.exists() else []:
        try:
            value = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "run_id" in value and "command" in value:
            manifests.append((path, value))

    models = []
    for remote_dir, files in groups.items():
        worktree, directory = remote_parts(remote_dir)
        key = f"{worktree}/{directory}"
        override = overrides.get(key, {})
        checkpoints = []
        for remote_path, meta in sorted(files, key=lambda item: (EPOCH_RE.search(Path(item[0]).name) is None,
                                                                  int(EPOCH_RE.search(Path(item[0]).name).group(1)) if EPOCH_RE.search(Path(item[0]).name) else 10**9,
                                                                  Path(item[0]).name)):
            match = EPOCH_RE.search(Path(remote_path).name)
            digest = meta.get("sha256")
            backup_path = local_checkpoint_path(snapshot, remote_path)
            checkpoints.append({
                "filename": Path(remote_path).name,
                "epoch": int(match.group(1)) if match else None,
                "kind": "periodic" if match else ("best" if Path(remote_path).name == "best.pt" else "named"),
                "remote_path": remote_path,
                "local_backup_path": backup_path,
                "local_backup_matches_current": meta.get("hash_source") == "verified_20260920_backup",
                "size": meta["size"],
                "mtime": iso_from_epoch(meta["mtime"]),
                "sha256": digest,
                "hash_evidence": meta.get("hash_source") or ("checkpoint_registry" if meta.get("sha256") else "not_yet_hashed")
            })
        earliest = min(float(meta["mtime"]) for _, meta in files)
        linked_manifests = []
        for local_manifest, manifest in manifests:
            command = manifest.get("command", "")
            try:
                command_argv = shlex.split(command)
            except ValueError:
                command_argv = []
            save_dirs = [command_argv[i + 1] for i, token in enumerate(command_argv[:-1])
                         if token == "--save-dir"]
            if any(Path(value).name == directory for value in save_dirs):
                linked_manifests.append({
                    "run_id": manifest.get("run_id"),
                    "started_at": manifest.get("started_at"),
                    "command": command,
                    "parent_checkpoint": manifest.get("parent_checkpoint"),
                    "parent_checkpoint_sha256": manifest.get("parent_checkpoint_sha256"),
                    "source_hashes": {k: v for k, v in manifest.items() if k.endswith("_sha256") and k != "parent_checkpoint_sha256"},
                    "local_manifest_path": str(local_manifest),
                    "evidence": "exact_manifest"
                })
        models.append({
            "model_id": key.replace("/", "::"),
            "display_name": override.get("display_name", directory),
            "worktree": worktree,
            "checkpoint_directory": remote_dir,
            "first_checkpoint_at": iso_from_epoch(earliest),
            "chronology_basis": "earliest_checkpoint_mtime",
            "status": override.get("status", "unknown"),
            "parent_lineage": {
                "checkpoint": override.get("parent"),
                "evidence": "devlog_and_filename_mapping" if override.get("parent") else "unknown"
            },
            "training_sessions": linked_manifests or [{
                "run_id": None,
                "evidence": "legacy_run_without_manifest",
                "note": "Session boundaries and exact launch command are not recoverable from a run manifest."
            }],
            "source_snapshot_paths": {
                name: {
                    "remote_path": f"{REMOTE_PREFIX}{worktree}/{name}",
                    "local_path": str(repo / "records" / "remote" / HOST_LABEL / worktree / name),
                    "evidence": "current_or_sync_time_snapshot_not_proven_run_time"
                } for name in ("model3.py", "train3.py", "dataset2.py")
            },
            "logs": [log_record(repo, worktree, name) for name in override.get("logs", [])],
            "checkpoints": checkpoints,
            "notes": override.get("notes", [])
        })

    models.sort(key=lambda model: model["first_checkpoint_at"])
    for index, model in enumerate(models, 1):
        model["chronological_order"] = index

    auxiliary_records = []
    for worktree, files in sorted(auxiliary.items()):
        items = []
        for remote_path, meta in sorted(files):
            digest = meta.get("sha256")
            backup_path = local_checkpoint_path(snapshot, remote_path)
            items.append({
                "filename": Path(remote_path).name,
                "remote_path": remote_path,
                "local_backup_path": backup_path,
                "local_backup_matches_current": meta.get("hash_source") == "verified_20260920_backup",
                "size": meta["size"], "mtime": iso_from_epoch(meta["mtime"]),
                "sha256": digest,
                "classification": "auxiliary_or_legacy_pt_requires_manual_classification",
                "hash_evidence": meta.get("hash_source") or ("checkpoint_registry" if meta.get("sha256") else "not_yet_hashed")
            })
        auxiliary_records.append({"worktree": worktree, "artifacts": items})

    output = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "ordering": "models sorted by earliest checkpoint mtime; checkpoints sorted by epoch then name",
        "evidence_policy": {
            "exact_manifest": "Recorded by a run manifest at launch time.",
            "verified_backup": "Binary hash verified in the 2026-09-20 point-in-time backup.",
            "legacy_run_without_manifest": "Artifacts exist, but exact session boundaries/config are incomplete.",
            "inference_warning": "Directory/log name associations come from explicit overrides and must be reviewed."
        },
        "models": models,
        "auxiliary_pt_artifacts": auxiliary_records
    }
    target = Path(args.output)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(target)
    print(f"model catalog: {len(models)} models, {sum(len(m['checkpoints']) for m in models)} checkpoints")
    print(f"auxiliary/root pt artifacts: {sum(len(x['artifacts']) for x in auxiliary_records)}")
    print(target)


if __name__ == "__main__":
    main()
