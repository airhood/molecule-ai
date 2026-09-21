import argparse
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from rdkit import Chem, RDLogger
from rdkit.Chem import RWMol

RDLogger.DisableLog("rdApp.*")

from dataset import QMugsDataset
from model import MoleculeCVAE

BOND_TYPE_MAP = {
    0: Chem.rdchem.BondType.SINGLE,
    1: Chem.rdchem.BondType.DOUBLE,
    2: Chem.rdchem.BondType.TRIPLE,
    3: Chem.rdchem.BondType.AROMATIC,
}


def build_mol(n_atoms, edge_index, bond_types, atom_types):
    mol = RWMol()
    for an in atom_types[:n_atoms]:
        mol.AddAtom(Chem.rdchem.Atom(int(an)))

    added = set()
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        if u < v and (u, v) not in added:
            bt = BOND_TYPE_MAP.get(int(bond_types[i]), Chem.rdchem.BondType.SINGLE)
            mol.AddBond(u, v, bt)
            added.add((u, v))

    try:
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def evaluate(model, loader, device, n_batches, threshold):
    model.eval()
    total = 0
    valid_count = 0
    unique_smiles = set()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            batch = batch.to(device)
            B = batch.num_graphs
            p = batch.p.view(B, -1)
            a_bin = batch.a_bin.view(B, -1)

            results = model.generate(p, a_bin, threshold=threshold)

            for b, r in enumerate(results):
                total += 1
                mol = build_mol(
                    r["n_atoms"],
                    r["edge_index"].cpu(),
                    r["bond_types"].cpu(),
                    r["atom_types"],
                )
                if mol is not None:
                    valid_count += 1
                    unique_smiles.add(Chem.MolToSmiles(mol))

    validity = valid_count / total if total > 0 else 0.0
    uniqueness = len(unique_smiles) / valid_count if valid_count > 0 else 0.0

    return {
        "validity": round(validity, 4),
        "uniqueness": round(uniqueness, 4),
        "n_valid": valid_count,
        "n_total": total,
        "n_unique": len(unique_smiles),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="체크포인트 경로 (.pt)")
    parser.add_argument("--processed-dir", default="./data/processed")
    parser.add_argument("--n-batches", type=int, default=10, help="평가할 배치 수")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5, help="결합 예측 임계값")
    parser.add_argument("--output", default=None, help="결과 저장 경로 (.json)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MoleculeCVAE().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    print(f"Loaded: {args.checkpoint}")

    val_set = QMugsDataset(
        args.processed_dir, split="val",
        max_samples=args.n_batches * args.batch_size,
    )
    loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Evaluating {args.n_batches} batches (threshold={args.threshold}) ...")
    metrics = evaluate(model, loader, device, args.n_batches, args.threshold)

    print(f"\nValidity  : {metrics['validity']:.2%}  ({metrics['n_valid']}/{metrics['n_total']})")
    print(f"Uniqueness: {metrics['uniqueness']:.2%}  ({metrics['n_unique']} unique SMILES)")

    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
