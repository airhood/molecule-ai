"""[Exp A2] "물성 7개면 충분한가" -- Exp A의 실데이터 통제군 방법론을
확장된 조건 벡터(HOMO,LUMO,LogP,TPSA,RotBonds,AromaticRings,SAscore, 총 7개)로
재실행. 2개일 때보다 창(다차원 박스)이 좁아질수록 구조 다양성이 얼마나 더
줄어드는지 비교. data/processed_ext7 사용, GPU/체크포인트 불필요.
"""
import sys, random, time
sys.path.insert(0, ".")

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from dataset2 import QMugsDataset

RDLogger.DisableLog("rdApp.*")

# p_raw 인덱스: 0 HOMO, 1 LUMO, 2 GAP(제외), 3 E_total(제외), 4 Dipole(제외),
# 5 LogP, 6 TPSA, 7 HBD(제외,중복), 8 HBA(제외,중복), 9 RotBonds, 10 AromaticRings, 11 SAscore
PROP_IDX = [0, 1, 5, 6, 9, 10, 11]
PROP_NAMES = ["HOMO", "LUMO", "LogP", "TPSA", "RotBonds", "AromaticRings", "SAscore"]
QUANTILES = [10, 30, 50, 70, 90]   # LUMO 기준 분위수(기존과 동일 정의)로 타겟 분자 선택
WIDTHS = [0.05, 0.10, 0.20, 0.30, 0.40]  # 7차원 박스라 2차원보다 넓게 잡아야 표본이 남음
N_BASELINE_DRAWS = 20
MAX_FP_PAIRS = 1500
SEED = 21


def diversity_metrics(idxs, heavy, rings, scaffold, fps, valid_mask, rng):
    idxs = [i for i in idxs if valid_mask[i]]
    k = len(idxs)
    if k < 2:
        return None
    scaf_set = {scaffold[i] for i in idxs if scaffold[i] is not None}
    scaf_unique = len(scaf_set)
    scaf_ratio = scaf_unique / k
    heavy_std = float(np.std(heavy[idxs]))
    ring_std = float(np.std(rings[idxs]))
    pairs_n = min(MAX_FP_PAIRS, k * (k - 1) // 2)
    sims, seen, tries = [], set(), 0
    while len(sims) < pairs_n and tries < pairs_n * 8:
        a, b = rng.sample(idxs, 2)
        key = (a, b) if a < b else (b, a)
        tries += 1
        if key in seen:
            continue
        seen.add(key)
        sims.append(DataStructs.TanimotoSimilarity(fps[a], fps[b]))
    mean_tanimoto = float(np.mean(sims)) if sims else float("nan")
    return dict(n=k, scaf_unique=scaf_unique, scaf_ratio=scaf_ratio, heavy_std=heavy_std,
                ring_std=ring_std, mean_tanimoto=mean_tanimoto)


def rarefied_unique_count(idxs, m, n_reps, scaffold, valid_mask, rng):
    idxs = [i for i in idxs if valid_mask[i] and scaffold[i] is not None]
    if len(idxs) < m:
        return None
    counts = []
    for _ in range(n_reps):
        samp = rng.sample(idxs, m)
        counts.append(len({scaffold[i] for i in samp}))
    return float(np.mean(counts)), float(np.std(counts))


def main(pool_n):
    t0 = time.time()
    ds = QMugsDataset("./data/processed_ext7", split="train", max_samples=pool_n, cache_chunks=8)
    n = len(ds)
    P = np.empty((n, len(PROP_IDX)))
    smiles = [None] * n
    for i in range(n):
        global_idx = int(ds.indices[i])
        chunk_idx = int(ds._chunk_of[i])
        local_idx = global_idx - int(ds._chunk_offsets[chunk_idx])
        raw = ds._load_chunk(chunk_idx)[local_idx]
        P[i] = [float(raw.p_raw[j]) for j in PROP_IDX]
        smiles[i] = raw.smiles
    print(f"loaded n={n} in {time.time()-t0:.1f}s", flush=True)

    ranges = np.percentile(P, 90, axis=0) - np.percentile(P, 10, axis=0)
    print("property p90-p10 ranges:", dict(zip(PROP_NAMES, ranges.tolist())), flush=True)

    lumo = P[:, 1]
    order = np.argsort(lumo)
    targets = []
    for q in QUANTILES:
        idx = int(n * q / 100)
        ci = int(order[idx])
        targets.append((P[ci].copy(), ci))

    t0 = time.time()
    heavy = np.full(n, np.nan)
    rings = np.full(n, np.nan)
    scaffold = [None] * n
    fps = [None] * n
    valid_mask = np.zeros(n, dtype=bool)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        valid_mask[i] = True
        heavy[i] = m.GetNumHeavyAtoms()
        rings[i] = m.GetRingInfo().NumRings()
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(m)
            scaffold[i] = Chem.MolToSmiles(scaf)
        except Exception:
            scaffold[i] = None
        fps[i] = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
    print(f"descriptors n_valid={valid_mask.sum()}/{n} in {time.time()-t0:.1f}s", flush=True)

    rng = random.Random(SEED)
    print(f"[Exp A2] multi-property({PROP_NAMES}) real-data control, pool_n={n}", flush=True)
    for (tvec, target_idx), q in zip(targets, QUANTILES):
        for w in WIDTHS:
            half = w * ranges
            diff = np.abs(P - tvec[None, :])
            window_idx = np.where(np.all(diff <= half[None, :], axis=1))[0]
            window_idx = window_idx[window_idx != target_idx]
            wm = diversity_metrics(window_idx.tolist(), heavy, rings, scaffold, fps, valid_mask, rng)
            if wm is None:
                print(f"q={q:>2} w={w:.2f}  n_window={len(window_idx):<5} (표본 부족, 스킵)", flush=True)
                continue
            k = wm["n"]
            base_runs = []
            for _ in range(N_BASELINE_DRAWS):
                samp = rng.sample(range(n), min(k, n))
                bm = diversity_metrics(samp, heavy, rings, scaffold, fps, valid_mask, rng)
                if bm:
                    base_runs.append(bm)

            def m_sd(key):
                vals = [b[key] for b in base_runs]
                return float(np.mean(vals)), float(np.std(vals))

            b_tan_m, b_tan_sd = m_sd("mean_tanimoto")
            m_fixed = min(100, k)
            n_reps = 50
            rare_win = rarefied_unique_count(window_idx.tolist(), m_fixed, n_reps, scaffold, valid_mask, rng)
            rare_base = rarefied_unique_count(list(range(n)), m_fixed, n_reps, scaffold, valid_mask, rng)
            print(
                f"q={q:>2} w={w:.2f} n={k:<6} "
                f"[tanimoto] window={wm['mean_tanimoto']:.3f} baseline={b_tan_m:.3f}±{b_tan_sd:.3f} "
                f"| rarefied(m={m_fixed}) window={rare_win[0]:.1f}±{rare_win[1]:.1f} "
                f"baseline={rare_base[0]:.1f}±{rare_base[1]:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    pool_n = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    main(pool_n)
