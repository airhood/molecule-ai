"""[A-2] p_raw를 5차원(HOMO,LUMO,GAP,E_total,Dipole)에서 12차원으로 확장
(+ LogP,TPSA,HBD,HBA,RotBonds,AromaticRings,SAscore). SDF/conformer 재파싱
없이 캐시된 data.smiles만 사용 -- 그래프(z/edge_index/bond_type/...)는 그대로
clone. 원본 data/processed/는 건드리지 않고 새 디렉터리에 저장.
39개 청크를 코어 수만큼 병렬 처리."""
import sys, os, json, time
sys.path.insert(0, ".")

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, rdMolDescriptors, RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

from multiprocessing import Pool

RDLogger.DisableLog("rdApp.*")

SRC_DIR = "data/processed"
DST_DIR = "data/processed_ext7"
NEW_PROP_NAMES = ["LogP", "TPSA", "HBD", "HBA", "RotBonds", "AromaticRings", "SAscore"]
N_WORKERS = 32


def compute_new_props(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        float(rdMolDescriptors.CalcNumHBD(mol)),
        float(rdMolDescriptors.CalcNumHBA(mol)),
        float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        sascorer.calculateScore(mol),
    ]


def process_chunk(args):
    import numpy as np
    chunk_name, idx, total = args
    t0 = time.time()
    items = torch.load(os.path.join(SRC_DIR, chunk_name), weights_only=False)
    n_fail = 0
    out = []
    vals_accum = [[] for _ in NEW_PROP_NAMES]  # stats.json 계산용 -- 재로딩 없이 이 패스에서 같이 수집
    for d in items:
        props = compute_new_props(d.smiles)
        if props is None:
            n_fail += 1
            new_p = torch.cat([d.p_raw, torch.full((len(NEW_PROP_NAMES),), float("nan"))])
        else:
            new_p = torch.cat([d.p_raw, torch.tensor(props, dtype=torch.float)])
            for i, v in enumerate(props):
                vals_accum[i].append(v)
        d.p_raw = new_p
        out.append(d)
    torch.save(out, os.path.join(DST_DIR, chunk_name))
    dt = time.time() - t0
    stat = [(float(np.sum(v)), float(np.sum(np.square(v))), len(v)) for v in vals_accum]
    return chunk_name, len(items), n_fail, dt, stat


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    with open(os.path.join(SRC_DIR, "meta.json")) as f:
        meta = json.load(f)
    chunk_names = meta["chunk_files"]

    tasks = [(name, i, len(chunk_names)) for i, name in enumerate(chunk_names)]
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        results = []
        for r in pool.imap_unordered(process_chunk, tasks):
            results.append(r)
            name, n, n_fail, dt, _ = r
            print(f"[{len(results)}/{len(tasks)}] {name}: n={n} fail={n_fail} ({dt:.1f}s)", flush=True)
    print(f"all chunks done in {time.time()-t0:.1f}s", flush=True)

    total_fail = sum(r[2] for r in results)
    total_n = sum(r[1] for r in results)
    print(f"total: n={total_n} smiles_parse_fail={total_fail}", flush=True)

    # meta.json / sizes.npy는 그래프 구조 무관(순서/크기 그대로) -> 그대로 복사
    import shutil
    shutil.copy(os.path.join(SRC_DIR, "meta.json"), os.path.join(DST_DIR, "meta.json"))
    sizes_path = os.path.join(SRC_DIR, "sizes.npy")
    if os.path.exists(sizes_path):
        shutil.copy(sizes_path, os.path.join(DST_DIR, "sizes.npy"))

    # stats.json: 기존 5개 컬럼 순서 유지 + 새 7개 컬럼 통계 추가(같은 순서로 p_raw에 이어붙였으므로)
    with open(os.path.join(SRC_DIR, "stats.json")) as f:
        old_stats = json.load(f)
    old_cols = list(old_stats.keys())
    print(f"기존 prop_cols(순서 유지): {old_cols}")

    # 새 통계는 process_chunk가 이미 이 패스에서 같이 모아온 sum/sumsq/count로 계산
    # (청크를 다시 로딩하지 않음 -- 19GB+ 재I/O 절약)
    import numpy as np
    n_props = len(NEW_PROP_NAMES)
    sums = [0.0] * n_props
    sumsqs = [0.0] * n_props
    counts = [0] * n_props
    for r in results:
        _, _, _, _, stat = r
        for i in range(n_props):
            s, sq, c = stat[i]
            sums[i] += s
            sumsqs[i] += sq
            counts[i] += c
    new_stats = {}
    for i, name in enumerate(NEW_PROP_NAMES):
        mean = sums[i] / counts[i]
        var = sumsqs[i] / counts[i] - mean ** 2
        new_stats[name] = {"mean": float(mean), "std": float(np.sqrt(max(var, 0.0)))}
    combined_stats = dict(old_stats)
    combined_stats.update(new_stats)
    with open(os.path.join(DST_DIR, "stats.json"), "w") as f:
        json.dump(combined_stats, f, indent=2)
    print("stats.json 저장 완료:", combined_stats)


if __name__ == "__main__":
    main()
