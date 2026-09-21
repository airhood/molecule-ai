# Experiment records

GPU artifacts are split by role: runtime logs go to `logs/remote/`, human documents to `docs/remote/`, and manifests, source snapshots, and machine-readable results to `records/remote/`. `scripts/ops/sync_records.sh` updates them without deleting old local files; changed copies are retained in the corresponding role-specific history directory.

Checkpoint binaries are excluded from record sync. `scripts/ops/checkpoint_registry.py` records their remote path, size, modification time, and SHA-256 in `records/checkpoints/registry.{json,tsv}`. Only explicitly selected checkpoints are fetched with `scripts/ops/fetch_checkpoint.sh REMOTE_PATH SHA256`.

Every training launch should use `./launch_run.sh python -u train3.py ...`. The launcher captures failures before Python starts, while `RunLogger` records stdout, stderr, arguments, source hashes, Git state, and parent checkpoint hashes. The historical `--log-file` output remains enabled.

## Model catalog

`model_catalog.json` groups model checkpoints by training lineage in chronological order and links each model to logs, manifests, source snapshots, parent lineage, and both remote and local paths. Rebuild it with `scripts/ops/build_model_catalog.py` after updating `records/checkpoints/registry.json`. `model_catalog_overrides.json` contains reviewed legacy associations; inferred links are labeled rather than presented as exact.

## Automatic record sync

Install the 2-minute user timer with `scripts/ops/install_record_sync_timer.sh 2min`. Control it with `start_record_sync.sh` and `stop_record_sync.sh`. The timer runs `sync_records.sh`; checkpoints and datasets remain excluded.
