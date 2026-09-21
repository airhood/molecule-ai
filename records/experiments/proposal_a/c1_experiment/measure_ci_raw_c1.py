"""[C-1] 8단계 준비: CI 기반 conditioning 정밀도 평가 + 분자별 원본 오차를
**타겟(quantile)별로 층화 저장**(Δ-CI 계산 시 5개 타겟을 풀링하지 않고
층화 재표집하기 위함 -- astra_review_20260919.md/review19.md 지적 반영,
advisor 지적: 나중에 재구성 불가능하므로 지금부터 저장). 이산 속성
(HBA/RotBonds/AromaticRings)의 exact-hit/±1-hit 계산을 위해 raw 예측값과
타겟값도 함께 저장(median만으론 계단형이라 둔감 -- astra 지적)."""
import sys, random, os, json
sys.path.insert(0, ".")
import torch
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, rdMolDescriptors

import model3 as m
from train3 import analyze_molecule, build_size_distribution
from dataset2 import QMugsDataset
import qm_props

RDLogger.DisableLog("rdApp.*")

N_PER_Q = 48
QUANTILES = [10, 30, 50, 70, 90]
SEED = 21
CKPT = os.environ.get("CKPT")
OUT = os.environ.get("OUT", "raw_errors.json")
GUIDANCE_W = 1.0
DISCRETE_PROPS = {"HBA", "RotBonds", "AromaticRings"}

RDKIT_COMPUTE = {
    2: lambda mol: Crippen.MolLogP(mol),
    3: lambda mol: rdMolDescriptors.CalcTPSA(mol),
    4: lambda mol: float(rdMolDescriptors.CalcNumHBA(mol)),
    5: lambda mol: float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
    6: lambda mol: float(rdMolDescriptors.CalcNumAromaticRings(mol)),
}


def boot_ci(errs, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    errs = np.asarray(errs)
    meds = [np.median(errs[rng.integers(0, len(errs), len(errs))]) for _ in range(n_boot)]
    return np.median(errs), np.percentile(meds, 2.5), np.percentile(meds, 97.5)


def main():
    device = torch.device("cuda")
    sd = torch.load(CKPT, map_location=device)
    model = m.MoleculeGraphDiffusion(sd["schedule.m_X"].clone(), sd["schedule.m_E"].clone())
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    print(f"missing={missing} unexpected={unexpected}", flush=True)

    val_ds = QMugsDataset("./data/processed_ext7", split="val", max_samples=30000)
    mean, std = val_ds.mean[m.COND_PROP_INDICES], val_ds.std[m.COND_PROP_INDICES]
    sizes = build_size_distribution(val_ds, n_sample=2000)

    all_p = torch.stack([val_ds[i].p_raw[m.COND_PROP_INDICES] for i in range(len(val_ds))])
    ranges = {}
    for i, name in enumerate(m.COND_PROP_NAMES):
        ranges[name] = float(torch.quantile(all_p[:, i], 0.9) - torch.quantile(all_p[:, i], 0.1))

    lumo_vals = all_p[:, 1]
    order = torch.argsort(lumo_vals)
    targets = []
    for q in QUANTILES:
        idx = int(len(val_ds) * q / 100)
        ci = order[idx].item()
        targets.append(all_p[ci].clone())

    print(f"CKPT={CKPT} guidance_w={GUIDANCE_W} N_PER_Q={N_PER_Q}", flush=True)

    random.seed(SEED)
    torch.manual_seed(SEED)

    # 타겟(quantile)별로 층화 저장: errs_by_prop[name][q_idx] = [abs_err, ...]
    # vals_by_prop[name][q_idx] = [(raw_pred, target), ...] (exact-hit용)
    n_q = len(QUANTILES)
    errs_by_prop = {name: [[] for _ in range(n_q)] for name in m.COND_PROP_NAMES}
    vals_by_prop = {name: [[] for _ in range(n_q)] for name in m.COND_PROP_NAMES}
    n_valid_total = 0
    for q_idx, tvec in enumerate(targets):
        norm = (tvec - mean) / std
        n_atoms = random.choice(sizes)
        cond = norm.unsqueeze(0).expand(N_PER_Q, -1).contiguous().to(device)
        with torch.no_grad():
            X, E = model.sample(N_PER_Q, n_atoms, device, cond=cond, guidance_w=GUIDANCE_W)
        X_cpu, E_cpu = X.cpu(), E.cpu()
        for i in range(N_PER_Q):
            info = analyze_molecule(X_cpu[i], E_cpu[i])
            if not info["strict_valid"]:
                continue
            mol = info["mol"]
            n_valid_total += 1
            for idx, fn in RDKIT_COMPUTE.items():
                name = m.COND_PROP_NAMES[idx]
                try:
                    val = fn(mol)
                    tgt = tvec[idx].item()
                    errs_by_prop[name][q_idx].append(abs(val - tgt))
                    vals_by_prop[name][q_idx].append((val, tgt))
                except Exception:
                    pass
            est = qm_props.gnn_homo_lumo(mol)
            if est is not None:
                errs_by_prop["HOMO"][q_idx].append(abs(est[0] - tvec[0].item()))
                errs_by_prop["LUMO"][q_idx].append(abs(est[1] - tvec[1].item()))
                vals_by_prop["HOMO"][q_idx].append((est[0], tvec[0].item()))
                vals_by_prop["LUMO"][q_idx].append((est[1], tvec[1].item()))

    print(f"\n총 valid 분자 수: {n_valid_total} (목표 {N_PER_Q*len(QUANTILES)}개 시도)", flush=True)
    print(f"\n{'물성':<14} {'n':>5} {'median|err|':>12} {'CI_low':>10} {'CI_high':>10}"
          f"{'exact-hit':>11}{'±1-hit':>9}", flush=True)
    for name in m.COND_PROP_NAMES:
        errs_flat = [e for q in errs_by_prop[name] for e in q]
        if len(errs_flat) < 5:
            print(f"{name:<14} n={len(errs_flat)} (표본 부족)", flush=True)
            continue
        med, lo, hi = boot_ci(errs_flat)
        extra = ""
        if name in DISCRETE_PROPS:
            vals_flat = [v for q in vals_by_prop[name] for v in q]
            exact = sum(1 for val, tgt in vals_flat if round(val) == round(tgt)) / len(vals_flat)
            pm1 = sum(1 for val, tgt in vals_flat if abs(round(val) - round(tgt)) <= 1) / len(vals_flat)
            extra = f"{exact:>11.3f}{pm1:>9.3f}"
        print(f"{name:<14} {len(errs_flat):>5} {med:>12.4f} {lo:>10.4f} {hi:>10.4f}{extra}", flush=True)

    with open(OUT, "w") as f:
        json.dump({
            "ckpt": CKPT, "ranges": ranges, "quantiles": QUANTILES,
            "errs_by_target": errs_by_prop, "vals_by_target": vals_by_prop,
            "n_per_q": N_PER_Q, "seed": SEED, "guidance_w": GUIDANCE_W,
        }, f)
    print(f"\n저장: {OUT}", flush=True)


if __name__ == "__main__":
    main()
