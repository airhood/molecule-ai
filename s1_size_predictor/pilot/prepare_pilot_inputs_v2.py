"""[S-1 pilot v2 confirmatory, CPU only] astra_review_20261001.md §5 Q2 권고 반영.

pilot_v1의 prepare_pilot_inputs.py와 동일한 로직이되, 딱 하나 다르다: target
후보 풀에서 pilot_v1이 이미 쓴 target_id를 미리 제외한 뒤 뽑는다(완전히
disjoint한 새 target 집합 보장). 그 외(정규화 게이트, prior 계산, manifest
구조)는 전부 동일 -- 로직을 포크가 아니라 복붙한 이유는 pilot_v1의
prepare_pilot_inputs.py가 이미 Astra 검토를 통과한 코드라 변경 범위를 "exclude
인자 추가" 하나로만 좁혀서 재검토 부담을 최소화하기 위함이다.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
from run_logger import RunLogger

MIN_ATOMS, MAX_ATOMS = 2, 50
N_CLASSES = MAX_ATOMS - MIN_ATOMS + 1
COND_PROP_INDICES = [0, 1, 5, 6, 8, 9, 10]
COND_PROP_NAMES = ["HOMO", "LUMO", "LogP", "TPSA", "HBA", "RotBonds", "AromaticRings"]
UNITS = {"HOMO": "Hartree", "LUMO": "Hartree", "LogP": "dimensionless", "TPSA": "A^2",
         "HBA": "count", "RotBonds": "count", "AromaticRings": "count"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_prior(n_atoms):
    n = np.asarray(n_atoms)
    in_range = (n >= MIN_ATOMS) & (n <= MAX_ATOMS)
    counts = np.bincount(n[in_range] - MIN_ATOMS, minlength=N_CLASSES).astype(np.int64)
    probs = counts / counts.sum()
    return counts, probs, int(in_range.sum()), int((~in_range).sum())


def select_targets(chembl_ids, global_idx, n_atoms, k, seed, exclude_ids=frozenset()):
    """pilot_v1과 동일하되 exclude_ids(이전 pilot이 이미 쓴 target_id)를 후보 풀에서
    미리 빼고 뽑는다 -- disjoint 보장이 선택 알고리즘 자체에 들어가야, "뽑고 나서
    겹치면 재추첨" 같은 사후 개입(결과에 의존하는 선택) 없이 결정적으로 보장된다."""
    rep = {}
    for cid, g, n in zip(chembl_ids, global_idx, n_atoms):
        if not (MIN_ATOMS <= int(n) <= MAX_ATOMS):
            continue
        if cid in exclude_ids:
            continue
        g = int(g)
        if cid not in rep or g < rep[cid]:
            rep[cid] = g
    sorted_ids = sorted(rep.keys())
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(sorted_ids), size=k, replace=False).tolist())
    return [(sorted_ids[i], rep[sorted_ids[i]]) for i in pick], len(sorted_ids)


def locate(global_idx, chunk_sizes):
    offsets = np.cumsum([0] + list(chunk_sizes[:-1]))
    c = int(np.searchsorted(offsets, global_idx, side="right") - 1)
    return c, int(global_idx - offsets[c])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--exclude-targets-json", required=True,
                        help="이전 pilot(v1)의 targets.json -- 여기 target_id 전부를 후보 풀에서 제외")
    parser.add_argument("--features-dir", default="./features_v2_fixed")
    parser.add_argument("--processed-dir", default="../data/processed_ext7")
    parser.add_argument("--out-root", default="./pilot_inputs")
    args = parser.parse_args()

    run_logger = RunLogger(__file__, source_paths=(Path(args.config), Path(args.exclude_targets_json))).start()
    run_logger.record_arguments(vars(args))

    cfg = json.load(open(args.config))
    features_dir = Path(args.features_dir)
    processed = Path(args.processed_dir)
    out_dir = Path(args.out_root) / run_logger.run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    print(f"out_dir: {out_dir}")

    exclude_doc = json.load(open(args.exclude_targets_json))
    exclude_ids = frozenset(t["target_id"] for t in exclude_doc["targets"])
    exclude_sha = sha256_file(args.exclude_targets_json)
    print(f"이전 pilot target {len(exclude_ids)}개 후보 풀에서 제외: {sorted(exclude_ids)}")

    # --- 정규화 artifact 검증(pilot_v1과 동일) ---
    norm_path = features_dir / "normalization.json"
    norm_sha = sha256_file(norm_path)
    if norm_sha != cfg["s1_models"]["normalization_sha256"]:
        raise RuntimeError(f"normalization SHA 불일치: {norm_sha}")
    norm = json.load(open(norm_path))
    assert norm["cond_prop_names"] == COND_PROP_NAMES and norm["cond_prop_indices"] == COND_PROP_INDICES
    s1_mean = np.array(norm["mean_cond7"], dtype=np.float32)
    s1_std = np.array(norm["std_cond7"], dtype=np.float32)
    assert np.isfinite(s1_mean).all() and np.isfinite(s1_std).all() and (s1_std > 0).all()

    stats_path = processed / "stats.json"
    stats_sha = sha256_file(stats_path)
    if stats_sha != cfg["frozen_a1"]["stats_json_sha256"]:
        raise RuntimeError(f"A-1 stats.json SHA 불일치: {stats_sha}")
    stats = json.load(open(stats_path))
    stat_cols = list(stats.keys())
    a1_mean = np.array([stats[stat_cols[i]]["mean"] for i in COND_PROP_INDICES], dtype=np.float32)
    a1_std = np.array([stats[stat_cols[i]]["std"] for i in COND_PROP_INDICES], dtype=np.float32)

    # --- train prior(pilot_v1과 동일 계산, target 선택과 무관) ---
    train_path = features_dir / "train.pt"
    train = torch.load(train_path, weights_only=False)
    counts, probs, n_in, n_out = compute_prior(train["n_atoms"].numpy())
    assert abs(probs.sum() - 1.0) < 1e-9
    prior = {
        "definition": cfg["prior"]["definition"],
        "support": [MIN_ATOMS, MAX_ATOMS], "n_classes": N_CLASSES,
        "counts": counts.tolist(), "probs": probs.tolist(),
        "train_rows_in_range": n_in, "train_rows_out_of_range_excluded": n_out,
        "train_pt_sha256": sha256_file(train_path),
        "mode_atoms": int(counts.argmax() + MIN_ATOMS),
    }
    json.dump(prior, open(out_dir / "prior.json", "w"), indent=2)
    print(f"prior: 행 {n_in:,} (범위 밖 {n_out:,} 제외), 최빈 원자수 {prior['mode_atoms']}")

    # --- target 선정(exclude 적용) ---
    val_path = features_dir / "val.pt"
    val = torch.load(val_path, weights_only=False)
    tcfg = cfg["targets"]
    chosen, n_unique_in_range = select_targets(
        val["chembl_id"], val["global_idx"].numpy(), val["n_atoms"].numpy(),
        tcfg["n_targets"], tcfg["target_selection_seed"], exclude_ids=exclude_ids)
    print(f"val 고유 ID(범위 내, 이전 pilot {len(exclude_ids)}개 제외 후) {n_unique_in_range:,}개 중 {len(chosen)}개 선정")

    chosen_ids = {cid for cid, _ in chosen}
    overlap = chosen_ids & exclude_ids
    assert not overlap, f"disjoint 보장 실패(있으면 안 됨): {overlap}"

    meta = json.load(open(processed / "meta.json"))
    chunk_sizes, chunk_files = meta["chunk_sizes"], meta["chunk_files"]
    val_gidx = val["global_idx"].numpy()
    val_row = {int(g): i for i, g in enumerate(val_gidx)}
    targets, cache = [], {}
    for cid, g in chosen:
        c, local = locate(g, chunk_sizes)
        if c not in cache:
            cache[c] = torch.load(processed / chunk_files[c], weights_only=False)
        item = cache[c][local]
        assert item.chembl_id == cid, f"chembl_id 불일치 {item.chembl_id} != {cid}"
        n_atoms_raw = int((item.z != 1).sum())
        row = val_row[g]
        assert n_atoms_raw == int(val["n_atoms"][row]), "heavy atom 수 불일치"
        raw12 = item.p_raw.numpy().astype(np.float32)
        raw7 = raw12[COND_PROP_INDICES]
        s1_norm = ((raw12 - np.array([norm["mean_all12"][i] for i in range(12)], dtype=np.float32))
                   / np.array([norm["std_all12"][i] for i in range(12)], dtype=np.float32))[COND_PROP_INDICES]
        assert np.array_equal(s1_norm.astype(np.float32), val["p7"][row].numpy()), \
            f"S-1 정규화 재현 불일치 (global_idx={g})"
        a1_norm = (raw7 - a1_mean) / a1_std
        targets.append({
            "target_id": cid, "global_idx": int(g), "n_atoms_true_NOT_USED_FOR_SIZE": n_atoms_raw,
            "raw7": {n: float(v) for n, v in zip(COND_PROP_NAMES, raw7)},
            "s1_normalized7": [float(x) for x in s1_norm],
            "a1_normalized7": [float(x) for x in a1_norm],
        })
    targets_doc = {
        "units": UNITS, "prop_order": COND_PROP_NAMES,
        "selection_seed": tcfg["target_selection_seed"], "n_unique_ids_in_pool": n_unique_in_range,
        "val_pt_sha256": sha256_file(val_path), "normalization_sha256": norm_sha,
        "a1_stats_json_sha256": stats_sha, "targets": targets,
        "excluded_from_pilot_v1": sorted(exclude_ids), "exclude_targets_json_sha256": exclude_sha,
    }
    (out_dir / "targets.json").write_text(json.dumps(targets_doc, indent=2))
    targets_sha = sha256_file(out_dir / "targets.json")

    manifest = {
        "config_sha256": sha256_file(Path(args.config)),
        "config_version": cfg["config_version"],
        "prior_json_sha256": sha256_file(out_dir / "prior.json"),
        "targets_json_sha256": targets_sha,
        "features_train_pt_sha256": prior["train_pt_sha256"],
        "features_val_pt_sha256": targets_doc["val_pt_sha256"],
        "normalization_sha256": norm_sha, "a1_stats_json_sha256": stats_sha,
        "meta_json_sha256": sha256_file(processed / "meta.json"),
        "exclude_targets_json_sha256": exclude_sha,
        "gates_passed": [
            "normalization/stats SHA가 config와 일치",
            "raw->S-1 정규화가 val feature table p7과 target 전부 bit-exact 일치",
            "target chembl_id/heavy atom 수가 원본 청크와 일치",
            "prior 확률 합 = 1",
            "선정된 target이 pilot_v1의 16개와 disjoint(교집합 0)",
        ],
        "gates_NOT_yet_checked": [
            "A-1 checkpoint/source SHA와 strict load", "S-1 checkpoint SHA/strict load",
            "evaluator SHA/RDKit 버전", "mask 불변성", "RNG 재현성",
        ],
    }
    (out_dir / "inputs_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"완료. {out_dir}")
    run_logger.finish("completed", out_dir=str(out_dir), targets_sha256=targets_sha)


if __name__ == "__main__":
    main()
