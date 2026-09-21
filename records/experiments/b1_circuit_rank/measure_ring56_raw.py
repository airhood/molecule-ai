"""ring56Rate의 Δ-CI 검증. evaluate_validity와 동일하게 무조건부(cond=None)
생성 후, 분자별 ring_sizes 리스트를 저장(분자 단위 재표집을 위해 -- 같은
분자 안의 고리들은 독립이 아니므로 고리 단위가 아니라 분자 단위로 resample).
"""
import sys, random, os, json
sys.path.insert(0, ".")
import torch

import model3 as m
from train3 import analyze_molecule, build_size_distribution, mol_diagnostics
from dataset2 import QMugsDataset

N_EVAL = 256
SEED = 21
CKPT = os.environ.get("CKPT")
OUT = os.environ.get("OUT", "ring_raw.json")


def main():
    device = torch.device("cuda")
    sd = torch.load(CKPT, map_location=device)
    model = m.MoleculeGraphDiffusion(sd["schedule.m_X"].clone(), sd["schedule.m_E"].clone())
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    val_ds = QMugsDataset("./data/processed_ext7" if os.path.isdir("./data/processed_ext7") else "./data", split="val", max_samples=30000)
    sizes = build_size_distribution(val_ds, n_sample=2000)

    random.seed(SEED)
    torch.manual_seed(SEED)

    from collections import Counter
    chosen_sizes = random.choices(sizes, k=N_EVAL)
    size_counts = Counter(chosen_sizes)

    per_mol_ring_sizes = []  # 분자별 ring_sizes 리스트(고리 없는 분자는 빈 리스트)
    n_strict = 0
    for n_atoms, count in size_counts.items():
        if n_atoms > m.MAX_ATOMS:
            continue
        X, E = model.sample(count, n_atoms, device)
        X_cpu, E_cpu = X.cpu(), E.cpu()
        for i in range(count):
            info = analyze_molecule(X_cpu[i], E_cpu[i])
            if not info["strict_valid"]:
                continue
            n_strict += 1
            d = mol_diagnostics(info["mol"])
            per_mol_ring_sizes.append(d["ring_sizes"])

    print(f"CKPT={CKPT} n_strict_valid={n_strict}/{N_EVAL}", flush=True)
    all_rings = [s for mol_rings in per_mol_ring_sizes for s in mol_rings]
    ring56 = sum(1 for s in all_rings if s in (5, 6))
    print(f"ring56Rate(점추정) = {ring56}/{len(all_rings)} = {ring56/len(all_rings) if all_rings else float('nan'):.4f}", flush=True)

    with open(OUT, "w") as f:
        json.dump({"ckpt": CKPT, "per_mol_ring_sizes": per_mol_ring_sizes}, f)
    print(f"저장: {OUT}", flush=True)


if __name__ == "__main__":
    main()
