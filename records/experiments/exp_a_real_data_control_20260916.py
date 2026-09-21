"""[Exp A] under-determined 가설 통제군: 실제 QMugs 분자 중 (HOMO,LUMO)가
타겟 근처인 것들의 구조 다양성을, 같은 크기의 무작위 표본과 비교.
GPU/체크포인트 불필요 -- 순수 데이터셋 분석. QM 물성(gap, E_total 등)이
아니라 구조 지표(scaffold, 중원자 수, 고리 수, Morgan FP Tanimoto)만 사용.
"""
import sys, random, time
sys.path.insert(0, ".")

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from dataset2 import QMugsDataset

RDLogger.DisableLog("rdApp.*")

RANGE_HOMO, RANGE_LUMO = 0.0468, 0.0678
QUANTILES = [10, 30, 50, 70, 90]
WIDTHS = [0.02, 0.05, 0.10, 0.15, 0.20]   # 분포범위(p90-p10) 대비 박스 반폭.
# 0.15/0.20 추가 이유: 모델의 실측 conditioning 오차(median|err|, 축별
# 15~20%)가 박스 창 폭(양 축 동시 제약)보다 원래 느슨한 기준이라, w=0.10에서
# 이미 거의 닫힌 격차가 모델의 실제 운용 정밀도에서 극외삽 없이 어떻게
# 이어지는지 직접 측정하기 위함.
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
    sims = []
    seen = set()
    tries = 0
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
    """window/baseline 표본 크기(모집단 크기) 차이로 인한 unique-ratio 포화
    아티팩트를 제거하기 위해, 둘 다 같은 고정 크기 m으로 재표집(rarefaction)해서
    unique scaffold 개수를 비교."""
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
    ds = QMugsDataset("./data/processed", split="train", max_samples=pool_n, cache_chunks=8)
    n = len(ds)
    homo = np.empty(n)
    lumo = np.empty(n)
    smiles = [None] * n
    # __getitem__은 heavy-atom 마스킹 등 그래프 편집을 하는데 여기선 p_raw/smiles만
    # 필요하므로 우회해서 직접 청크에서 읽음 (5000개에 25s -> 훨씬 빠르게).
    for i in range(n):
        global_idx = int(ds.indices[i])
        chunk_idx = int(ds._chunk_of[i])
        local_idx = global_idx - int(ds._chunk_offsets[chunk_idx])
        raw = ds._load_chunk(chunk_idx)[local_idx]
        homo[i] = float(raw.p_raw[0])
        lumo[i] = float(raw.p_raw[1])
        smiles[i] = raw.smiles
    print(f"loaded n={n} in {time.time()-t0:.1f}s", flush=True)

    order = np.argsort(lumo)
    targets = []
    for q in QUANTILES:
        idx = int(n * q / 100)
        ci = int(order[idx])
        targets.append((float(homo[ci]), float(lumo[ci]), ci))

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
    print(f"CKPT-free real-data control, pool_n={n}", flush=True)
    for (th, tl, target_idx), q in zip(targets, QUANTILES):
        for w in WIDTHS:
            hw, lw = w * RANGE_HOMO, w * RANGE_LUMO
            window_idx = np.where((np.abs(homo - th) <= hw) & (np.abs(lumo - tl) <= lw))[0]
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

            b_scaf_u_m, b_scaf_u_sd = m_sd("scaf_unique")
            b_scaf_m, b_scaf_sd = m_sd("scaf_ratio")
            b_heavy_m, b_heavy_sd = m_sd("heavy_std")
            b_ring_m, b_ring_sd = m_sd("ring_std")
            b_tan_m, b_tan_sd = m_sd("mean_tanimoto")

            # rarefaction: population 크기 차이(window=제약된 소집단, baseline=
            # 200000 전체에서 추출) 때문에 생기는 unique-ratio 포화 아티팩트 제거.
            m_fixed = min(100, k)
            n_reps = 50
            rare_win = rarefied_unique_count(window_idx.tolist(), m_fixed, n_reps, scaffold, valid_mask, rng)
            rare_base = rarefied_unique_count(list(range(n)), m_fixed, n_reps, scaffold, valid_mask, rng)

            print(
                f"q={q:>2} w={w:.2f} n={k:<5} "
                f"[scaffold] unique window={wm['scaf_unique']} baseline={b_scaf_u_m:.0f}±{b_scaf_u_sd:.0f} "
                f"| ratio window={wm['scaf_ratio']:.3f} baseline={b_scaf_m:.3f}±{b_scaf_sd:.3f} "
                f"| rarefied(m={m_fixed}) window={rare_win[0]:.1f}±{rare_win[1]:.1f} "
                f"baseline={rare_base[0]:.1f}±{rare_base[1]:.1f}",
                flush=True,
            )
            print(
                f"        [heavy_std] window={wm['heavy_std']:.2f} baseline={b_heavy_m:.2f}±{b_heavy_sd:.2f} "
                f"| [ring_std] window={wm['ring_std']:.2f} baseline={b_ring_m:.2f}±{b_ring_sd:.2f} "
                f"| [tanimoto] window={wm['mean_tanimoto']:.3f} baseline={b_tan_m:.3f}±{b_tan_sd:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    pool_n = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    main(pool_n)
