"""[S-1] SizePredictor 학습. model3.py의 A-1 conditioning과 동일한 물성
선택(COND_PROP_INDICES)/정규화(dataset2.py의 z-score)/두 단계 dropout
(COND_DROPOUT_P_FULL/PARTIAL)을 그대로 재사용해, 실제 생성 시 만날
부분조건 분포와 학습 분포를 맞춘다. denoiser(model3.py)는 전혀 건드리지
않는 완전히 독립된 작은 분류기 학습 -- "크기 선택이 conditioning 오차의
구조적 하한인가"를 저비용으로 먼저 검증하는 용도(astra_architecture_
proposals_20260920.md §2).
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset2 import QMugsDataset
from size_predictor import SizePredictor, MIN_ATOMS, MAX_ATOMS, N_SIZE_CLASSES, COND_DIM

# model3.py와 동일 -- 두 모델이 물성 조건을 다르게 해석하면 S-1 검증
# 자체가 무의미해지므로 반드시 일치시킨다.
COND_PROP_INDICES = [0, 1, 5, 6, 8, 9, 10]
COND_DROPOUT_P_FULL = 0.15
COND_DROPOUT_P_PARTIAL = 0.2


def collate(batch):
    p = torch.stack([b.p for b in batch])[:, COND_PROP_INDICES]
    n_atoms = torch.tensor([b.num_nodes for b in batch], dtype=torch.long)
    return p, n_atoms


def make_cond_mask(B, device, train: bool):
    if not train:
        return torch.ones(B, COND_DIM, device=device)
    full_drop = torch.rand(B, device=device) < COND_DROPOUT_P_FULL
    partial_drop = torch.rand(B, COND_DIM, device=device) < COND_DROPOUT_P_PARTIAL
    cond_mask = (~partial_drop).float()
    cond_mask[full_drop] = 0.0
    return cond_mask


def run_epoch(model, loader, device, train, optimizer=None):
    model.train(train)
    total_loss, total_correct, total_n = 0.0, 0, 0
    for p, n_atoms in loader:
        p, n_atoms = p.to(device), n_atoms.to(device)
        B = p.shape[0]
        cond_mask = make_cond_mask(B, device, train)
        target = (n_atoms - MIN_ATOMS).clamp(0, N_SIZE_CLASSES - 1)

        logits = model(p, cond_mask)
        loss = F.cross_entropy(logits, target)

        if train:
            optimizer.zero_grad()
            loss.backward()
            # [안정화, 2026-09-21] 첫 시도에서 epoch5 부근 T-loss 급등(1.31->2.46)
            # + val이 계속 단일 클래스로 collapse(입력 무관 argmax 고정)하는
            # 패턴 관찰 -- property z-score의 꼬리값(TPSA/HBA/RotBonds 등 카운트성
            # 지표는 최대 z~7-8까지 감) 몇 개가 큰 gradient를 만들어 최적화가
            # 튀는 것으로 보여 clipping 추가.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * B
        total_correct += (logits.argmax(-1) == target).sum().item()
        total_n += B
    return total_loss / total_n, total_correct / total_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="../data/processed_ext7")
    parser.add_argument("--save-dir", default="./checkpoints_s1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="train.py(CVAE)의 KL beta warmup과 같은 원칙 -- "
                             "1 step째부터 최대 LR로 시작하면 분류 헤드가 "
                             "임의 클래스에 꽂혔다가 늦게 빠져나오는 진동이 "
                             "생김(2026-09-21 관찰). 처음 이 epoch 수 동안 "
                             "LR을 선형으로 낮은 값에서 --lr까지 올린 뒤 "
                             "코사인 감쇠로 넘어간다.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print("Loading datasets ...")
    # [2026-09-22] QMugsDataset 기본 cache_chunks=2는 model3.py처럼 epoch 수가
    # 적고 batch당 연산이 무거운 워크로드를 가정한 값. reshuffle_indices()가
    # 매 epoch마다 청크 순서를 다시 섞는데, 캐시가 2개뿐이면 --max-samples로
    # 필요한 청크 전부(500MB/개)를 매 epoch마다 디스크에서 다시 읽게 됨 --
    # S-1처럼 연산이 가볍고 epoch을 많이 도는 워크로드에선 이 재로딩 비용이
    # 학습 자체보다 압도적으로 커진다(실측: epoch1이 20분 넘게 걸림, GPU
    # 사용률 0%). --max-samples로 필요한 청크 수만큼 캐시를 키워 첫 epoch
    # 이후로는 전부 메모리에 남게 한다(청크당 500MB, 넉넉히 15개=7.5GB
    # 잡아도 시스템 메모리 380GB에 비해 무시할 수준).
    chunk_size = 50000
    cache_chunks = max(2, (args.max_samples // chunk_size + 2)) if args.max_samples else 40
    print(f"  cache_chunks={cache_chunks} (청크당 ~500MB)")
    train_set = QMugsDataset(args.processed_dir, split="train", max_samples=args.max_samples,
                              reshuffle_seed=args.seed, cache_chunks=cache_chunks)
    val_set = QMugsDataset(args.processed_dir, split="val",
                            max_samples=args.max_samples // 5 if args.max_samples else None,
                            cache_chunks=cache_chunks)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate)
    print(f"  Train: {len(train_set):,}  Val: {len(val_set):,}")

    # size 범위 밖 분자 비율 확인 -- SizePredictor는 MIN_ATOMS..MAX_ATOMS만
    # 표현 가능하므로, 벗어나는 분자가 많으면 평가가 왜곡된다.
    train_sizes = train_set._sizes
    out_of_range = ((train_sizes < MIN_ATOMS) | (train_sizes > MAX_ATOMS)).mean()
    print(f"  size range [{MIN_ATOMS},{MAX_ATOMS}] 밖 비율: {out_of_range:.2%}")

    model = SizePredictor().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # [안정화, 2026-09-21] 15만 샘플/30epoch에서 train acc는 계속 오르는데
    # val은 epoch1부터 계속 나빠지는 전형적 과적합 관찰 -- weight_decay를
    # 키우고(--max-samples를 늘려 같이 완화) 대응.
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    warmup_epochs = min(args.warmup_epochs, args.epochs - 1) if args.epochs > 1 else 0
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                                     total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs,
                                              eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                  milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    print(f"  Warmup: {warmup_epochs} epochs, then cosine decay over "
          f"{args.epochs - warmup_epochs} epochs")

    best_val_acc = 0.0
    print(f"\n{'Epoch':>6}  {'T-loss':>8} {'T-acc':>8}  {'V-loss':>8} {'V-acc':>8}")
    print("-" * 50)
    for epoch in range(1, args.epochs + 1):
        train_set.reshuffle_indices()
        t_loss, t_acc = run_epoch(model, train_loader, device, train=True, optimizer=optimizer)
        with torch.no_grad():
            v_loss, v_acc = run_epoch(model, val_loader, device, train=False)
        scheduler.step()
        print(f"{epoch:>6}  {t_loss:>8.4f} {t_acc:>8.2%}  {v_loss:>8.4f} {v_acc:>8.2%}")
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), save_dir / "best.pt")

    print(f"\nDone. Best val acc: {best_val_acc:.2%}")


if __name__ == "__main__":
    main()
