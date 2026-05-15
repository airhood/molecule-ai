import argparse
import io
import math
import sys
import time
from pathlib import Path
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from dataset import QMugsDataset
from model import MoleculeCVAE


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


def cyclical_beta(step, total_steps, n_cycles=4, max_beta=1.0, ratio=0.5, warmup_steps=0):
    if step < warmup_steps:
        return 0.0
    effective_step = step - warmup_steps
    effective_total = total_steps - warmup_steps
    cycle_steps = effective_total / n_cycles
    cycle_pos = effective_step % cycle_steps
    anneal_steps = cycle_steps * ratio
    if cycle_pos < anneal_steps:
        t = cycle_pos / anneal_steps
        return max_beta * 0.5 * (1 - math.cos(math.pi * t))
    return max_beta


def run_epoch(model, loader, optimizer, beta, device, train, epoch):
    model.train(train)
    totals = {
        "loss": 0,
        "reconstruction_loss": 0,
        "exist_loss": 0,
        "type_loss": 0,
        "kl_loss": 0,
        "a_pred_loss": 0
    }

    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
            batch = batch.to(device)
            model.beta = beta

            loss_result = model(batch)

            if train:
                optimizer.zero_grad()
                loss_result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            for k in totals:
                totals[k] += loss_result[k].item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="./data/processed")
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-cycles", type=int, default=4)
    parser.add_argument("--max-beta", type=float, default=0.5)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-file", default="./train.log", help="로그 파일 경로")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file) if args.log_file else save_dir / "train.log"
    tee = _Tee(sys.stdout, log_path)
    sys.stdout = tee

    print(f"Device: {device}")
    print(f"Log: {log_path}")

    print("Loading datasets ...")

    train_set = QMugsDataset(args.processed_dir, split="train", max_samples=args.max_samples)
    val_set = QMugsDataset(
        args.processed_dir, split="val",
        max_samples=args.max_samples // 5 if args.max_samples else None
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"  Train: {len(train_set):,}  Val: {len(val_set):,}")

    model = MoleculeCVAE().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    step = 0
    best_fixed_eval = float("inf")

    print(f"  Warmup: {args.warmup_epochs} epochs ({warmup_steps:,} steps), then {args.n_cycles} cosine cycles")
    print(f"\n{'Epoch':>6} {'Beta':>6} {'T-loss':>8} {'T-recon':>8} "
          f"{'T-exist':>8} {'T-type':>8} {'T-kl':>8} {'T-apred':>8} "
          f"{'V-loss':>8} {'V-recon':>8} {'V-kl':>8} {'V-apred':>8}  Time")
    print("-" * 120)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        beta = cyclical_beta(step, total_steps, args.n_cycles, args.max_beta, warmup_steps=warmup_steps)

        train_result = run_epoch(model, train_loader, optimizer, beta, device, train=True, epoch=epoch)
        val_result = run_epoch(model, val_loader, optimizer, beta, device, train=False, epoch=epoch)

        step += len(train_loader)
        scheduler.step()

        time_elapsed = time.time() - t0
        print(
            f"{epoch:>6} {beta:>6.3f} "
            f"{train_result['loss']:>8.4f} {train_result['reconstruction_loss']:>8.4f} "
            f"{train_result['exist_loss']:>8.4f} {train_result['type_loss']:>8.4f} "
            f"{train_result['kl_loss']:>8.4f} {train_result['a_pred_loss']:>8.4f} "
            f"{val_result['loss']:>8.4f} {val_result['reconstruction_loss']:>8.4f} "
            f"{val_result['kl_loss']:>8.4f} {val_result['a_pred_loss']:>8.4f}  {time_elapsed:>5.1f}s"
        )

        fixed_eval = (val_result["reconstruction_loss"]
                      + val_result["a_pred_loss"]
                      + args.max_beta * val_result["kl_loss"])
        if fixed_eval < best_fixed_eval:
            best_fixed_eval = fixed_eval
            torch.save(model.state_dict(), save_dir / "best.pt")

        if args.save_every != 0 and epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_fixed_eval": best_fixed_eval
            }, save_dir / f"ckpt_epoch{epoch:04d}.pt")

    print(f"\ntrain finished! Best fixed-beta eval: {best_fixed_eval:.4f}")
    sys.stdout = tee._stream
    tee.close()


if __name__ == "__main__":
    main()
