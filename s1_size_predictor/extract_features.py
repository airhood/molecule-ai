"""[S-1, P1] astra_review_20260922.md P1 권고 반영 -- 학습 그래프 청크
(청크당 500MB, 5만 분자)를 매 epoch 통째로 다시 읽는 대신, S-1에
필요한 7개 정규화 물성값 + heavy atom count만 딱 한 번 뽑아서 작은
feature table로 저장한다. 500k행 기준 7개 float32+label이면 대략
16MB -- 이후 모든 S-1 학습/ablation은 이 파일만 메모리에 올려서 쓴다
(QMugsDataset의 500MB 청크 lru_cache를 매 epoch 다시 태울 필요 없음).

전체 데이터셋(39개 청크, ~19.5GB)을 한 번 순회하는 이 스크립트 자체는
시간이 걸리지만(청크당 ~24초 x 39 ~= 15분), 이후 S-1 학습은 이 비용을
다시 내지 않는다.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

COND_PROP_INDICES = [0, 1, 5, 6, 8, 9, 10]
COND_PROP_NAMES = ["HOMO", "LUMO", "LogP", "TPSA", "HBA", "RotBonds", "AromaticRings"]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="../data/processed_ext7")
    parser.add_argument("--out-dir", default="./features")
    parser.add_argument("--split-ratio", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processed_path = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(processed_path / "meta.json") as f:
        meta = json.load(f)
    chunk_names = meta["chunk_files"]

    stats_path = processed_path / "stats.json"
    with open(stats_path) as f:
        stats = json.load(f)
    prop_cols = list(stats.keys())
    mean = torch.tensor([stats[c]["mean"] for c in prop_cols], dtype=torch.float)
    std = torch.tensor([stats[c]["std"] for c in prop_cols], dtype=torch.float)
    for idx, name in zip(COND_PROP_INDICES, COND_PROP_NAMES):
        print(f"  index {idx}: stats.json='{prop_cols[idx]}'  기대='{name}'")

    # QMugsDataset.__init__과 정확히 같은 split 배정(같은 seed의 permutation) --
    # 이 순서를 벗어나면 이 feature table의 train/val이 실제 학습 코드가 보는
    # split과 달라져서 조용히 틀린 실험이 된다.
    chunk_sizes = meta["chunk_sizes"]
    total = sum(chunk_sizes)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(total)
    train_end = int(total * args.split_ratio[0])
    val_end = train_end + int(total * args.split_ratio[1])
    split_of = np.empty(total, dtype=np.int8)  # 0=train,1=val,2=test
    split_of[perm[:train_end]] = 0
    split_of[perm[train_end:val_end]] = 1
    split_of[perm[val_end:]] = 2

    chunk_offsets = np.cumsum([0] + chunk_sizes[:-1])

    buffers = {0: [], 1: [], 2: []}  # each: list of (p7, n_atoms, global_idx)
    for ci, chunk_name in enumerate(chunk_names):
        print(f"  chunk {ci+1}/{len(chunk_names)}: {chunk_name}")
        data_list = torch.load(processed_path / chunk_name, weights_only=False)
        base = int(chunk_offsets[ci])
        for local_idx, item in enumerate(data_list):
            global_idx = base + local_idx
            split = int(split_of[global_idx])
            p = (item.p_raw - mean) / std
            p7 = p[COND_PROP_INDICES].numpy().astype(np.float32)
            n_atoms = int((item.z != 1).sum())
            buffers[split].append((p7, n_atoms, global_idx))

    manifest = {
        "source_processed_dir": str(processed_path.resolve()),
        "stats_sha256": _sha256(stats_path),
        "meta_sha256": _sha256(processed_path / "meta.json"),
        "seed": args.seed,
        "split_ratio": list(args.split_ratio),
        "cond_prop_indices": COND_PROP_INDICES,
        "cond_prop_names": COND_PROP_NAMES,
    }

    split_names = {0: "train", 1: "val", 2: "test"}
    seen_global_idx = set()
    for split_id, name in split_names.items():
        rows = buffers[split_id]
        p7 = np.stack([r[0] for r in rows]) if rows else np.zeros((0, 7), dtype=np.float32)
        n_atoms = np.array([r[1] for r in rows], dtype=np.int16)
        global_idx = np.array([r[2] for r in rows], dtype=np.int64)
        # train/val/test 누출 검증 -- 같은 global_idx가 두 split에 동시에
        # 들어가면 split_of 배정 로직 자체가 깨진 것이므로 바로 에러.
        overlap = seen_global_idx & set(global_idx.tolist())
        if overlap:
            raise RuntimeError(f"split 누출 발견: {name}에 이미 다른 split에서 "
                                f"본 global_idx {len(overlap)}개 포함")
        seen_global_idx |= set(global_idx.tolist())
        torch.save({
            "p7": torch.from_numpy(p7),
            "n_atoms": torch.from_numpy(n_atoms.astype(np.int64)),
            "global_idx": torch.from_numpy(global_idx),
        }, out_dir / f"{name}.pt")
        manifest[f"{name}_count"] = len(rows)
        print(f"  {name}: {len(rows):,}개, {out_dir / f'{name}.pt'}")

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n완료. manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
