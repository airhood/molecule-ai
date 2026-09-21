# Logs

This directory contains console and runtime logs only.

- `experiments/`: preserved historical experiment logs formerly mixed into a documentation subdirectory.
- `remote/`: logs synchronized from GPU worktrees.
- `sync/`: logs produced by the synchronization process itself.
- `legacy_root/`: old root-level logs retained for provenance.

Logs are append-only evidence. Human analysis belongs in `docs/`; manifests, hashes, results, and source snapshots belong in `records/`.
