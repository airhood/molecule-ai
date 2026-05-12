"""
processed.zip 안에 meta.json이 없을 때 추가하는 마이그레이션 스크립트.
zip 파일 구조는 유지되고 meta.json만 추가됨.

Usage:
    python migrate_meta.py processed.zip
    python migrate_meta.py  # 현재 디렉토리의 processed.zip 사용
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import torch

sys.stdout.reconfigure(line_buffering=True)


def main():
    zip_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("processed.zip")

    if not zip_path.exists():
        print(f"Error: {zip_path} not found.")
        sys.exit(1)

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()

        if "meta.json" in all_names:
            print("meta.json already exists in zip. Nothing to do.")
            sys.exit(0)

        chunk_names = sorted(
            n for n in all_names
            if Path(n).name.startswith("data_chunk_") and n.endswith(".pt")
        )
        if not chunk_names:
            print("No data_chunk_*.pt files found in zip.")
            sys.exit(1)

        print(f"Found {len(chunk_names)} chunk files. Reading sizes ...")
        chunk_sizes = []
        for name in chunk_names:
            raw = zf.read(name)
            chunk = torch.load(io.BytesIO(raw), weights_only=False)
            chunk_sizes.append(len(chunk))
            del chunk, raw
            print(f"  {Path(name).name}: {chunk_sizes[-1]:,} items")

    meta = {
        "chunk_files": [Path(n).name for n in chunk_names],
        "chunk_sizes": chunk_sizes,
    }

    with zipfile.ZipFile(zip_path, "a") as zf:
        zf.writestr("meta.json", json.dumps(meta))

    print(f"\nmeta.json added to {zip_path}")
    print(f"Total items: {sum(chunk_sizes):,}")


if __name__ == "__main__":
    main()
