import argparse
import math
import os
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from dataset import QMugsDataset
from model_ import MolCVAE


# ── Cyclical Beta 스케줄러 ─────────────────────────────────────────────────────
def cyclical_beta(step: int, total_steps: int, n_cycles: int = 4,
                  max_beta: float = 1.0, ratio: float = 0.5) -> float:
    """
    학습 전체를 n_cycles개 주기로 나누고,
    각 주기의 앞 ratio 비율 동안 cosine으로 0 → max_beta 부드럽게 증가,
    나머지는 max_beta 고정, 다음 주기 시작 시 0으로 리셋.
    """
    cycle_steps = total_steps / n_cycles
    cycle_pos   = step % cycle_steps
    anneal_end  = cycle_steps * ratio
    if cycle_pos < anneal_end:
        t = cycle_pos / anneal_end
        return max_beta * 0.5 * (1 - math.cos(math.pi * t))
    return max_beta


# ── 학습 / 검증 한 에폭 ────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, beta, device, train: bool):
    model.train(train)
    totals = {"loss": 0, "recon_loss": 0, "exist_loss": 0, "type_loss": 0, "kl_loss": 0}

    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(device)
            model.beta = beta

            losses = model(batch)

            if train:
                optimizer.zero_grad()
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            for k in totals:
                totals[k] += losses[k].item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


# ── 메인 ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train MolCVAE")
    parser.add_argument("--processed-dir", default="./data/processed")
    parser.add_argument("--save-dir",      default="./checkpoints")
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--n-cycles",      type=int,   default=4)
    parser.add_argument("--max-beta",      type=float, default=1.0)
    parser.add_argument("--save-every",    type=int,   default=10)
    parser.add_argument("--num-workers",   type=int,   default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터셋 ──────────────────────────────────────────────────────────────
    print("Loading dataset ...")
    train_set = QMugsDataset(args.processed_dir, split="train")
    val_set   = QMugsDataset(args.processed_dir, split="val")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"  Train: {len(train_set):,}  Val: {len(val_set):,}")

    # ── 모델 / 옵티마이저 ─────────────────────────────────────────────────────
    model     = MolCVAE().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    total_steps = args.epochs * len(train_loader)
    step        = 0
    best_val    = float("inf")

    # ── 학습 루프 ─────────────────────────────────────────────────────────────
    print(f"\n{'Epoch':>6} {'Beta':>6} {'T-loss':>8} {'T-recon':>8} "
          f"{'T-exist':>8} {'T-type':>8} {'T-kl':>8} "
          f"{'V-loss':>8} {'V-recon':>8}  Time")
    print("-" * 95)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # beta: 에폭 시작 시점의 step 기준으로 계산
        beta = cyclical_beta(step, total_steps, args.n_cycles, args.max_beta)

        train_metrics = run_epoch(model, train_loader, optimizer, beta, device, train=True)
        val_metrics   = run_epoch(model, val_loader,   optimizer, beta, device, train=False)

        step += len(train_loader)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"{epoch:>6} {beta:>6.3f} "
            f"{train_metrics['loss']:>8.4f} {train_metrics['recon_loss']:>8.4f} "
            f"{train_metrics['exist_loss']:>8.4f} {train_metrics['type_loss']:>8.4f} "
            f"{train_metrics['kl_loss']:>8.4f} "
            f"{val_metrics['loss']:>8.4f} {val_metrics['recon_loss']:>8.4f} "
            f" {elapsed:>5.1f}s"
        )

        # 최고 성능 모델 저장
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(model.state_dict(), save_dir / "best.pt")

        # 주기적 체크포인트
        if epoch % args.save_every == 0:
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "best_val":   best_val,
            }, save_dir / f"ckpt_epoch{epoch:04d}.pt")

    print(f"\n학습 완료. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
