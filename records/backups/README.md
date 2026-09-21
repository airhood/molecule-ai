# Backup locations

`locations.json` records the absolute path, host, filesystem device, size, file count, and verification evidence for backups stored outside the repository workspace.

The two local copies are on host `airhood`, on the same `/dev/nvme1n1p2` volume as the repository. They are valid pre-change copies but are not protection against physical failure of that volume. The GPU-server backup at `/home/cbgpu/molecule-AI_backup_pre_restructure_20260920T123633+0000` is the physically separate copy.

Checksum manifests and the complete `sha256sum -c` outputs are copied here so a reviewer restricted to the repository can inspect the evidence without traversing sibling directories.
