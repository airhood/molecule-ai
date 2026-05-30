import argparse
import copy
import io
import sys
import time
from pathlib import Path

import torch
from rdkit import Chem, RDLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

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


class _Tee(io.TextIOBase):

    def __init__(self, stream, file_path):
        self._stream = stream
        self._f = open(file_path, "a", encoding="utf-8")

    def write(self, text):
        self._stream.write(text)
        self._f.write(text)
        return len(text)

    def flush(self):
        self._stream.flush()
        self._f.flush()

    def close(self):
        self._f.close()


class EMA:

    def __init__(self, model, decay = 0.9999):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for ema_p, p in zip(self.model.parameters(), model.parameters()):
                ema_p.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)


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


def build_size_distribution(dataset, n_sample = 2000):
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


def run_epoch(model, loader, device, train, epoch, optimizer=None, ema=None):
    model.train(train)
    totals = {
        "loss": 0.0,
        "loss_x": 0.0,
        "loss_e": 0.0
    }

    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
            batch = batch.to(device)
            result = model(batch)

            if train:
                optimizer.zero_grad()
                result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
                optimizer.step()
                if ema is not None:
                    ema.update(model)

            for k in totals:
                totals[k] += result[k].item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate_validity(model, size_dist, n_eval, device):
    import random
    from collections import Counter

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

    return validity, uniqueness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir",  default="./data/processed")
    parser.add_argument("--save-dir",       default="./checkpoints_diffusion")
    parser.add_argument("--epochs",         type=int,   default=200)
    parser.add_argument("--batch-size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--ema-decay",      type=float, default=0.9999)
    parser.add_argument("--save-every",     type=int,   default=10)
    parser.add_argument("--eval-every",     type=int,   default=10,
                        help="Validity evaluation interval (epochs). 0 = disable.")
    parser.add_argument("--n-eval",         type=int,   default=256,
                        help="Number of molecules to sample per validity evaluation.")
    parser.add_argument("--num-workers",    type=int,   default=0)
    parser.add_argument("--max-samples",    type=int,   default=None)
    parser.add_argument("--log-file",       default="./train_diffusion.log")
    parser.add_argument("--resume",         default=None,
                        help="체크포인트 경로 (.pt). model/ema/optimizer/scheduler/epoch 복원.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tee = _Tee(sys.stdout, args.log_file)
    sys.stdout = tee

    print(f"Device: {device}")
    print(f"Log: {args.log_file}")

    print("Loading datasets ...")
    train_set = QMugsDataset(
        args.processed_dir, split="train",
        max_samples=args.max_samples
    )
    val_set = QMugsDataset(
        args.processed_dir, split="val",
        max_samples=args.max_samples // 5 if args.max_samples else None
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    print(f"  Train: {len(train_set):,}  Val: {len(val_set):,}")

    print("Building size distribution ...")
    size_dist = build_size_distribution(train_set)
    print(f"  Size range: {min(size_dist)} ~ {max(size_dist)} atoms  "
          f"(median {sorted(size_dist)[len(size_dist)//2]})")
    
    model = MoleculeGraphDiffusion().to(device)
    ema = EMA(model, decay=args.ema_decay)
    ema.model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float('inf'))
        print(f"  Resumed from epoch {ckpt['epoch']}  (best val loss: {best_val_loss:.4f})")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    print(f"\n{'Epoch':>6}  {'T-loss':>8} {'T-x':>8} {'T-e':>8}  "
          f"{'V-loss':>8} {'V-x':>8} {'V-e':>8}  {'Valid':>7} {'Uniq':>7}  Time")
    print("-" * 105)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_set.reshuffle_indices()
        
        train = run_epoch(model, train_loader, device, train=True, epoch=epoch, optimizer=optimizer, ema=ema)
        val = run_epoch(ema.model, val_loader, device, train=False, epoch=epoch)

        scheduler.step()

        validity = uniqueness = float("nan")
        if args.eval_every > 0 and epoch % args.eval_every == 0:
            validity, uniqueness = evaluate_validity(ema.model, size_dist, args.n_eval, device)

        elapsed = time.time() - t0
        print(
            f"{epoch:>6}  "
            f"{train['loss']:>8.4f} {train['loss_x']:>8.4f} {train['loss_e']:>8.4f}  "
            f"{val['loss']:>8.4f} {val['loss_x']:>8.4f} {val['loss_e']:>8.4f}  "
            f"{validity:>7.3f} {uniqueness:>7.3f}  {elapsed:>5.1f}s"
        )

        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            torch.save(ema.model.state_dict(), save_dir / "best.pt")

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss
            }, save_dir / f"ckpt_epoch{epoch:04d}.pt")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    sys.stdout = tee._stream
    tee.close()


if __name__ == "__main__":
    main()
