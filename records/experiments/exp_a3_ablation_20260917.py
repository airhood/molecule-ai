"""[Exp A3] 7개 물성 중 구조 제약이 정수형 위상 지표(RotBonds,AromaticRings)
때문인지 연속형 전자/물리화학 지표(HOMO,LUMO,LogP,TPSA,SAscore) 때문인지
분리. 같은 pool/descriptor를 한 번만 로딩해서 세 조건(A:위상만, B:연속만,
C:전체 7개)을 비교. data/processed_ext7 사용, GPU/체크포인트 불필요.
"""
import sys, random, time
sys.path.insert(0, ".")

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from dataset2 import QMugsDataset

RDLogger.DisableLog("rdApp.*")

# p_raw 인덱스: 0 HOMO,1 LUMO,2 GAP,3 E_total,4 Dipole,5 LogP,6 TPSA,7 HBD,8 HBA,
# 9 RotBonds,10 AromaticRings,11 SAscore
CONDITIONS = {
    "A_topological(RotBonds+AromaticRings)": [9, 10],
    "B_continuous(HOMO+LUMO+LogP+TPSA+SAscore)": [0, 1, 5, 6, 11],
    "C_all7": [0, 1, 5, 6, 9, 10, 11],
}
ALL_NAMES = ["HOMO", "LUMO", "GAP", "E_total", "Dipole", "LogP", "TPSA", "HBD", "HBA",
             "RotBonds", "AromaticRings", "SAscore"]
QUANTILES = [10, 30, 50, 70, 90]
WIDTHS = [0.10, 0.20, 0.30, 0.40]
N_BASELINE_DRAWS = 20
MAX_FP_PAIRS = 1500
SEED = 21


def diversity_metrics(idxs, heavy, rings, scaffold, fps, valid_mask, rng):
    idxs = [i for i in idxs if valid_mask[i]]
    k = len(idxs)
    if k < 2:
        return None
    scaf_set = {scaffold[i] for i in idxs if scaffold[i] is not None}
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
    return dict(n=k, scaf_unique=len(scaf_set), heavy_std=heavy_std,
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
    P = np.empty((n, len(ALL_NAMES)))
    smiles = [None] * n
    for i in range(n):
        global_idx = int(ds.indices[i])
        chunk_idx = int(ds._chunk_of[i])
        local_idx = global_idx - int(ds._chunk_offsets[chunk_idx])
        raw = ds._load_chunk(chunk_idx)[local_idx]
        P[i] = [float(raw.p_raw[j]) for j in range(len(ALL_NAMES))]
        smiles[i] = raw.smiles
    print(f"loaded n={n} in {time.time()-t0:.1f}s", flush=True)

    ranges_all = np.percentile(P, 90, axis=0) - np.percentile(P, 10, axis=0)

    # 타겟 분자: 기존과 동일하게 LUMO 정렬 분위수로 선정 (전체 물성 벡터를 타겟으로 사용)
    lumo = P[:, 1]
    order = np.argsort(lumo)
    targets = []
    for q in QUANTILES:
        idx = int(n * q / 100)
        ci = int(order[idx])
        targets.append((ci, q))

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
    for cond_name, prop_idx in CONDITIONS.items():
        print(f"\n=== condition {cond_name} ===", flush=True)
        ranges = ranges_all[prop_idx]
        for target_idx, q in targets:
            tvec = P[target_idx, prop_idx]
            for w in WIDTHS:
                half = w * ranges
                diff = np.abs(P[:, prop_idx] - tvec[None, :])
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
