"""[S-1, P1 재작업] astra_review_20260924.md §1 반영 -- extract_features.py는
행(row) 단위로 무작위 분리했는데, 같은 분자(chembl_id)가 conformer별로
여러 행에 나뉘어 들어가 있어서 같은 분자가 train과 val에 동시에 들어가는
누출이 있었다(첫 청크 기준 val의 96.3%가 train과 같은 분자). 이 버전은:

1. chembl_id 단위로 묶어서 split -- 같은 분자의 모든 행은 항상 같은 split.
2. 정규화 통계(mean/std)를 train으로 배정된 행에서만 계산(기존은 split
   이전 전체 summary에서 계산됨, astra_review_20260924.md §1 후반 지적).
3. feature table에 chembl_id도 같이 저장 -- 이후 감사가 분자 단위로
   직접 검증할 수 있게.
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
    parser.add_argument("--out-dir", default="./features_v2")
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
        stats_cols = list(json.load(f).keys())
    for idx, name in zip(COND_PROP_INDICES, COND_PROP_NAMES):
        print(f"  index {idx}: stats.json='{stats_cols[idx]}'  기대='{name}'")

    # 1차 패스: 모든 행의 chembl_id/p_raw(12차원, 정규화 전 원본)/n_atoms를
    # 메모리에 모은다(스칼라+짧은 문자열이라 190만행이어도 가벼움).
    chembl_ids, p_raws, n_atoms_list = [], [], []
    for ci, chunk_name in enumerate(chunk_names):
        print(f"  [읽기] chunk {ci+1}/{len(chunk_names)}: {chunk_name}")
        data_list = torch.load(processed_path / chunk_name, weights_only=False)
        for item in data_list:
            chembl_ids.append(item.chembl_id)
            p_raws.append(item.p_raw.numpy().astype(np.float32))
            n_atoms_list.append(int((item.z != 1).sum()))

    p_raw_all = np.stack(p_raws)
    n_atoms_all = np.array(n_atoms_list, dtype=np.int64)
    total = len(chembl_ids)
    print(f"  총 {total:,}행, 고유 분자 {len(set(chembl_ids)):,}개")

    # 2. 분자(chembl_id) 단위로 split 배정 -- 같은 분자의 모든 행(conformer)은
    # 항상 같은 split에 들어간다.
    unique_ids = sorted(set(chembl_ids))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(unique_ids))
    n_ids = len(unique_ids)
    train_end = int(n_ids * args.split_ratio[0])
    val_end = train_end + int(n_ids * args.split_ratio[1])
    id_split = {}
    for i, idx in enumerate(perm):
        if i < train_end:
            id_split[unique_ids[idx]] = 0
        elif i < val_end:
            id_split[unique_ids[idx]] = 1
        else:
            id_split[unique_ids[idx]] = 2
    split_of = np.array([id_split[cid] for cid in chembl_ids], dtype=np.int8)

    # 검증: 같은 chembl_id가 두 split에 걸치지 않는지 직접 확인(분자 단위 누출 검사).
    id_to_splits = {}
    for cid, sp in zip(chembl_ids, split_of):
        id_to_splits.setdefault(cid, set()).add(int(sp))
    leaked = {cid: sps for cid, sps in id_to_splits.items() if len(sps) > 1}
    if leaked:
        raise RuntimeError(f"분자 단위 split 누출 발견: {len(leaked)}개 분자가 "
                            f"여러 split에 걸쳐 있음(버그)")
    print(f"  분자 단위 누출 검사 통과: 고유 분자 {len(id_to_splits):,}개 전부 단일 split")

    # 3. 정규화 통계는 train으로 배정된 행에서만 계산(astra_review_20260924.md
    # §1 후반 -- "정규화 통계도 새 train에서만 산출한다").
    train_mask = split_of == 0
    mean = p_raw_all[train_mask].mean(axis=0)
    std = p_raw_all[train_mask].std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)  # 상수 컬럼 0-division 방지
    print(f"  train-only 정규화 통계 계산 완료(행 {train_mask.sum():,}개 기준)")

    p_normalized = (p_raw_all - mean) / std
    p7 = p_normalized[:, COND_PROP_INDICES].astype(np.float32)

    split_names = {0: "train", 1: "val", 2: "test"}
    for split_id, name in split_names.items():
        m = split_of == split_id
        torch.save({
            "p7": torch.from_numpy(p7[m]),
            "n_atoms": torch.from_numpy(n_atoms_all[m]),
            "chembl_id": [chembl_ids[i] for i in np.where(m)[0]],
        }, out_dir / f"{name}.pt")
        n_unique = len(set(chembl_ids[i] for i in np.where(m)[0]))
        print(f"  {name}: {m.sum():,}행, 고유분자 {n_unique:,}개, {out_dir / f'{name}.pt'}")

    manifest = {
        "source_processed_dir": str(processed_path.resolve()),
        "stats_sha256": _sha256(stats_path),
        "meta_sha256": _sha256(processed_path / "meta.json"),
        "seed": args.seed,
        "split_ratio": list(args.split_ratio),
        "split_unit": "chembl_id (molecule-level group split, not row-level)",
        "normalization": "train-only mean/std (computed after split, not full-dataset summary)",
        "cond_prop_indices": COND_PROP_INDICES,
        "cond_prop_names": COND_PROP_NAMES,
        "total_rows": total,
        "total_unique_molecules": len(unique_ids),
        "train_count": int((split_of == 0).sum()),
        "val_count": int((split_of == 1).sum()),
        "test_count": int((split_of == 2).sum()),
        "train_unique_molecules": len(set(chembl_ids[i] for i in np.where(split_of == 0)[0])),
        "val_unique_molecules": len(set(chembl_ids[i] for i in np.where(split_of == 1)[0])),
        "test_unique_molecules": len(set(chembl_ids[i] for i in np.where(split_of == 2)[0])),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n완료. manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
