import argparse
import json
import math
import multiprocessing
import os
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolTransforms
from torch_geometric.data import Data
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
RDLogger.DisableLog("rdApp.warning")

PROP_COLS = [
    "DFT_HOMO_ENERGY",
    "DFT_LUMO_ENERGY",
    "DFT_HOMO_LUMO_GAP",
    "DFT_TOTAL_ENERGY",
    "DFT_DIPOLE_TOT",
]
ELEMENTS = ["C", "H", "O", "N", "S", "P", "F", "Cl", "Br", "I"]

CHUNK_SIZE = 50_000

BOND_TYPE_MAP = {
    1.0: [1, 0, 0, 0],  # SINGLE
    2.0: [0, 1, 0, 0],  # DOUBLE
    3.0: [0, 0, 1, 0],  # TRIPLE
    1.5: [0, 0, 0, 1],  # AROMATIC
}


def _build_stereo_map():
    from rdkit.Chem import rdchem
    return {
        rdchem.BondStereo.STEREONONE:  [1, 0, 0, 0, 0],
        rdchem.BondStereo.STEREOCIS:   [0, 1, 0, 0, 0],
        rdchem.BondStereo.STEREOTRANS: [0, 0, 1, 0, 0],
        rdchem.BondStereo.STEREOE:     [0, 0, 0, 1, 0],
        rdchem.BondStereo.STEREOZ:     [0, 0, 0, 0, 1],
    }


STEREO_MAP = _build_stereo_map()
STEREO_DEFAULT = [1, 0, 0, 0, 0]


def _pick_neighbor(mol, center_idx: int, exclude_idx: int):
    """Highest atomic-number neighbor of center_idx (excluding exclude_idx).
    Tie-breaks by smallest atom index."""
    neighbors = [
        a for a in mol.GetAtomWithIdx(center_idx).GetNeighbors()
        if a.GetIdx() != exclude_idx
    ]
    if not neighbors:
        return None
    return max(neighbors, key=lambda a: (a.GetAtomicNum(), -a.GetIdx())).GetIdx()


def _compute_composition(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    a = [counts.get(e, 0) for e in ELEMENTS]
    a.append(mol.GetNumAtoms())
    return a


def _process_sdf(args):
    """Worker: parse one SDF file and return a plain dict or None.
    Tensors are NOT created here to avoid expensive pickle serialization over IPC."""
    sdf_path_str, row = args

    chembl_id = Path(sdf_path_str).parent.name

    try:
        supplier = Chem.SDMolSupplier(sdf_path_str, removeHs=False, sanitize=True)
        mol = supplier[0] if supplier and len(supplier) > 0 else None
    except Exception:
        return None

    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()
    if n_atoms < 3 or n_atoms > 100:
        return None

    if mol.GetNumConformers() == 0:
        return None

    conf = mol.GetConformer()
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    z_list, charge_list, chirality_list = [], [], []
    for atom in mol.GetAtoms():
        z_list.append(atom.GetAtomicNum())
        charge_list.append(float(atom.GetFormalCharge()))
        if atom.HasProp("_CIPCode"):
            cip = atom.GetProp("_CIPCode")
            chirality_list.append([1, 0, 0] if cip == "R" else [0, 1, 0])
        else:
            chirality_list.append([0, 0, 1])

    src_list, dst_list = [], []
    bt_list, bs_list, dih_list = [], [], []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()

        bt_val = bond.GetBondTypeAsDouble()
        bt = BOND_TYPE_MAP.get(bt_val, [1, 0, 0, 0])
        bs = STEREO_MAP.get(bond.GetStereo(), STEREO_DEFAULT)

        dih = 0.0
        if bt_val == 1.0 and not bond.IsInRing():
            i = _pick_neighbor(mol, u, v)
            j = _pick_neighbor(mol, v, u)
            if i is not None and j is not None:
                try:
                    deg = rdMolTransforms.GetDihedralDeg(conf, i, u, v, j)
                    dih = float(deg) * math.pi / 180.0
                except Exception:
                    dih = 0.0

        for s, d in ((u, v), (v, u)):
            src_list.append(s)
            dst_list.append(d)
            bt_list.append(bt)
            bs_list.append(bs)
            dih_list.append(dih)

    return {
        "z":          z_list,
        "pos":        conf.GetPositions().tolist(),
        "charge":     charge_list,
        "chirality":  chirality_list,
        "src":        src_list,
        "dst":        dst_list,
        "bond_type":  bt_list,
        "bond_stereo":bs_list,
        "dihedral":   dih_list,
        "p_raw":      row["p_raw"],
        "a":          row["a"],
        "chembl_id":  chembl_id,
        "smiles":     row["smiles"],
    }


def _dict_to_data(d: dict) -> Data:
    return Data(
        z=torch.tensor(d["z"], dtype=torch.long),
        pos=torch.tensor(d["pos"], dtype=torch.float),
        charge=torch.tensor(d["charge"], dtype=torch.float),
        chirality=torch.tensor(d["chirality"], dtype=torch.float),
        edge_index=torch.tensor([d["src"], d["dst"]], dtype=torch.long),
        bond_type=torch.tensor(d["bond_type"], dtype=torch.float),
        bond_stereo=torch.tensor(d["bond_stereo"], dtype=torch.float),
        dihedral=torch.tensor(d["dihedral"], dtype=torch.float),
        p_raw=torch.tensor(d["p_raw"], dtype=torch.float),
        a=torch.tensor(d["a"], dtype=torch.float),
        chembl_id=d["chembl_id"],
        smiles=d["smiles"],
    )


def _print_stats(data_list, stats: dict) -> None:
    if not data_list:
        return

    total_atoms = 0
    elem_counts: Counter = Counter()
    for d in data_list:
        total_atoms += int(d.a[-1].item())
        for i, e in enumerate(ELEMENTS):
            elem_counts[e] += int(d.a[i].item())

    print("\n원소 분포:")
    for e in ELEMENTS:
        pct = 100.0 * elem_counts[e] / total_atoms if total_atoms else 0.0
        print(f"  {e:2s}: {pct:.1f}%")

    p_raws = torch.stack([d.p_raw for d in data_list])
    prop_labels = ["HOMO", "LUMO", "gap", "total", "dipole"]
    print("\n물성 분포 (원본값):")
    for i, label in enumerate(prop_labels):
        col = p_raws[:, i]
        print(f"  {label:6s}: mean={col.mean():.4f}  std={col.std():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess QMugs dataset")
    parser.add_argument("--raw-dir", default="./data/raw", dest="raw_dir")
    parser.add_argument("--out-dir", default="./data/processed", dest="out_dir")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--chunksize", type=int, default=256)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load and process summary.csv ---
    print("Loading summary.csv ...")
    df = pd.read_csv(
        raw_dir / "summary.csv",
        usecols=["chembl_id", "smiles"] + PROP_COLS,
    )
    df = df.dropna()
    print(f"  {len(df):,} rows after NaN removal")

    stats = {
        col: {"mean": float(df[col].mean()), "std": float(df[col].std())}
        for col in PROP_COLS
    }
    stats_path = out_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats -> {stats_path}")

    # Build per-molecule lookup dict
    print("Computing composition vectors ...")
    mol_data: dict = {}
    skipped_smiles = 0
    for _, row in tqdm(df.iterrows(), total=len(df), leave=False):
        a = _compute_composition(row["smiles"])
        if a is None:
            skipped_smiles += 1
            continue
        mol_data[row["chembl_id"]] = {
            "smiles": row["smiles"],
            "p_raw": [float(row[c]) for c in PROP_COLS],
            "a": a,
        }
    print(f"  {len(mol_data):,} molecules ({skipped_smiles} invalid SMILES skipped)")

    # --- Build task list ---
    structures_dir = raw_dir / "structures"
    print(f"Scanning {structures_dir} for SDF files ...")
    sdf_paths = sorted(structures_dir.rglob("*.sdf"))
    print(f"  Found {len(sdf_paths):,} SDF files")

    tasks = []
    skipped_no_match = 0
    for p in sdf_paths:
        cid = p.parent.name
        if cid in mol_data:
            tasks.append((str(p), mol_data[cid]))
        else:
            skipped_no_match += 1
    print(f"  {len(tasks):,} tasks ({skipped_no_match:,} skipped: no CSV match)")

    # --- Parallel SDF parsing ---
    print(f"Processing with {args.workers} workers (chunksize={args.chunksize}) ...")
    data_list = []
    filtered = 0

    with multiprocessing.Pool(args.workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_sdf, tasks, chunksize=args.chunksize),
            total=len(tasks),
        ):
            if result is not None:
                data_list.append(_dict_to_data(result))
            else:
                filtered += 1

    total_skipped = filtered + skipped_no_match + skipped_smiles
    print(f"\n총 conformer 수:  {len(tasks):>10,}")
    print(f"필터링 제거:      {total_skipped:>10,}")
    print(f"최종 저장:        {len(data_list):>10,}")

    # --- Save chunks + meta.json ---
    print()
    chunk_sizes = []
    for i in range(0, len(data_list), CHUNK_SIZE):
        chunk = data_list[i:i + CHUNK_SIZE]
        chunk_path = out_dir / f"data_chunk_{len(chunk_sizes):04d}.pt"
        torch.save(chunk, chunk_path)
        chunk_sizes.append(len(chunk))
        print(f"  Saved {chunk_path.name} ({len(chunk):,} items)")

    meta = {
        "chunk_files": [f"data_chunk_{i:04d}.pt" for i in range(len(chunk_sizes))],
        "chunk_sizes": chunk_sizes,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f)
    print(f"  Saved meta.json ({len(chunk_sizes)} chunks, {sum(chunk_sizes):,} total)")

    _print_stats(data_list, stats)


if __name__ == "__main__":
    main()
