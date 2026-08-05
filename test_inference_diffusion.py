import argparse
import random
from collections import Counter
from pathlib import Path
import torch
from rdkit import Chem, RDLogger

from dataset2 import QMugsDataset
from model2 import ATOMIC_NUM_TO_CLS, K_E, K_X, MAX_ATOMS, MoleculeGraphDiffusion

RDLogger.DisableLog("rdApp.*")

_BOND_TYPE = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.AROMATIC,
}
_CLS_TO_ANUM = {v: k for k, v in ATOMIC_NUM_TO_CLS.items()}


def tensors_to_mol(X, E):
    mol = Chem.RWMol()
    real = (X > 0).nonzero(as_tuple=True)[0].tolist()
    if len(real) < 2:
        return None
    
    idx_map = {}
    for ai in real:
        anum = _CLS_TO_ANUM.get(int(X[ai]))
        if anum is None:
            return None
        idx_map[ai] = mol.AddAtom(Chem.Atom(anum))

    for i, ai in enumerate(real):
        for aj in real[i + 1:]:
            bc = int(E[ai, aj])
            if bc == 0:
                continue
            btype = _BOND_TYPE.get(bc)
            if btype is None:
                continue
            mol.AddBond(idx_map[ai], idx_map[aj], btype)

    try:
        Chem.SanitizeMol(mol)
        return mol.GetMol()
    except Exception:
        return None


def build_size_distribution(dataset, n_sample=1000):
    sizes = []
    ci = 0
    while len(sizes) < n_sample and ci < len(dataset._chunk_names):
        for data in dataset._load_chunk(ci):
            n = int((data.z != 1).sum())
            if 0 < n <= MAX_ATOMS:
                sizes.append(n)
            if len(sizes) >= n_sample:
                break
        ci += 1
    return sizes


@torch.no_grad()
def evaluate_validity(model, size_dist, n_eval, device):
    model.eval()
    sizes = random.choices(size_dist, k=n_eval)
    size_counts = Counter(sizes)

    valid_mols = []
    for n_atoms, count in size_counts.items():
        if n_atoms > MAX_ATOMS:
            continue
        X, E = model.sample(count, n_atoms, device)
        for i in range(count):
            mol = tensors_to_mol(X[i], E[i])
            if mol is not None:
                valid_mols.append(mol)

    validity = len(valid_mols) / n_eval
    if valid_mols:
        smiles = {Chem.MolToSmiles(m) for m in valid_mols}
        uniqueness = len(smiles) / len(valid_mols)
    else:
        uniqueness = 0.0

    return validity, uniqueness, valid_mols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./checkpoints_diffusion/ckpt_epoch0070.pt")
    parser.add_argument("--processed-dir", default="./data/processed")
    parser.add_argument("--n-eval", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MoleculeGraphDiffusion().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    if isinstance(ckpt, dict) and "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        print(f"Loaded EMA weights from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")
    elif isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        print(f"Loaded model weights from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")
    else:
        model.load_state_dict(ckpt)
        print(f"Loaded state dict from {args.checkpoint}")

    print("Loading dataset for size distribution...")
    dataset = QMugsDataset(args.processed_dir, split="val")
    size_dist = build_size_distribution(dataset)
    print(f"Size distribution build complete. Sample size: {len(size_dist)}")

    print(f"Evaluating {args.n_eval} molecules...")
    validity, uniqueness, valid_mols = evaluate_validity(model, size_dist, args.n_eval, device)

    print(f"\nValidity:   {validity:.2%}")
    print(f"Uniqueness: {uniqueness:.2%}")
    print(f"Valid Molecules count: {len(valid_mols)}")

    if valid_mols:
        print("\nGenerated SMILES (up to 10):")
        for mol in random.sample(valid_mols, min(10, len(valid_mols))):
            print(f"  {Chem.MolToSmiles(mol)}")


if __name__ == "__main__":
    main()
