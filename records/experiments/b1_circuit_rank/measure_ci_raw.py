"""[Astra 4/6번] CI 기반 conditioning 정밀도 평가 + 분자별 원본 오차 배열을
JSON으로 저장(Δ-CI 계산용). review18 [B-3] 반영: N_PER_Q 32->48로 늘려
n>=128 목표."""
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
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

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

    errs_by_prop = {name: [] for name in m.COND_PROP_NAMES}
    n_valid_total = 0
    for tvec in targets:
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
                    errs_by_prop[name].append(abs(val - tvec[idx].item()))
                except Exception:
                    pass
            est = qm_props.gnn_homo_lumo(mol)
            if est is not None:
                errs_by_prop["HOMO"].append(abs(est[0] - tvec[0].item()))
                errs_by_prop["LUMO"].append(abs(est[1] - tvec[1].item()))

    print(f"\n총 valid 분자 수: {n_valid_total} (목표 {N_PER_Q*len(QUANTILES)}개 시도)", flush=True)
    print(f"\n{'물성':<14} {'n':>5} {'median|err|':>12} {'CI_low':>10} {'CI_high':>10}", flush=True)
    for name in m.COND_PROP_NAMES:
        errs = errs_by_prop[name]
        if len(errs) < 5:
            print(f"{name:<14} n={len(errs)} (표본 부족)", flush=True)
            continue
        med, lo, hi = boot_ci(errs)
        print(f"{name:<14} {len(errs):>5} {med:>12.4f} {lo:>10.4f} {hi:>10.4f}", flush=True)

    with open(OUT, "w") as f:
        json.dump({"ckpt": CKPT, "ranges": ranges, "errs": errs_by_prop}, f)
    print(f"\n저장: {OUT}", flush=True)


if __name__ == "__main__":
    main()
