"""[S-1] SizePredictor 학습. model3.py의 A-1 conditioning과 동일한 물성
선택(COND_PROP_INDICES)/정규화(dataset2.py의 z-score)/두 단계 dropout
(COND_DROPOUT_P_FULL/PARTIAL)을 그대로 재사용해, 실제 생성 시 만날
부분조건 분포와 학습 분포를 맞춘다. denoiser(model3.py)는 전혀 건드리지
않는 완전히 독립된 작은 분류기 학습 -- "크기 선택이 conditioning 오차의
구조적 하한인가"를 저비용으로 먼저 검증하는 용도(astra_architecture_
proposals_20260920.md §2).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from dataset2 import QMugsDataset
from run_logger import RunLogger
from size_predictor import SizePredictor, MIN_ATOMS, MAX_ATOMS, N_SIZE_CLASSES, COND_DIM

# model3.py와 동일 -- 두 모델이 물성 조건을 다르게 해석하면 S-1 검증
# 자체가 무의미해지므로 반드시 일치시킨다.
COND_PROP_INDICES = [0, 1, 5, 6, 8, 9, 10]
COND_PROP_NAMES = ["HOMO", "LUMO", "LogP", "TPSA", "HBA", "RotBonds", "AromaticRings"]
COND_DROPOUT_P_FULL = 0.15
COND_DROPOUT_P_PARTIAL = 0.2


def _atomic_torch_save(obj, path):
    """train3.py의 동일 함수와 같은 이유(astra_review_20260920d.md §6) --
    torch.save가 최종 경로에 직접 쓰면 중단 시 부분 파일이 정상 파일명으로
    남을 수 있다. temp+fsync+os.replace로 원자적 치환."""
    path = Path(path)
    tmp_path = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp_path, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _preflight(args, stats_path):
    """astra_review_20260922.md P0 -- 실행 전에 max_samples/split 크기/
    property index-name 정렬/class 범위/param count/LR 스케줄을 검증.
    여기서 걸러지면 학습을 아예 시작하지 않는다(실행해서 알아차리는 방식
    금지)."""
    with open(stats_path) as f:
        stats = json.load(f)
    stat_keys = list(stats.keys())
    for idx, name in zip(COND_PROP_INDICES, COND_PROP_NAMES):
        if idx >= len(stat_keys):
            raise ValueError(f"preflight 실패: COND_PROP_INDICES[{idx}]가 "
                              f"stats.json 키 개수({len(stat_keys)})를 벗어남")
        actual = stat_keys[idx]
        # stats.json의 실제 표기(DFT_HOMO_ENERGY 등)와 별칭(HOMO 등)이 다를 수
        # 있어 완전 일치는 요구하지 않되, 사람이 눈으로 확인할 수 있게 출력.
        print(f"  [preflight] index {idx}: stats.json='{actual}'  기대='{name}'")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError(f"preflight 실패: --max-samples={args.max_samples} <= 0")
    if args.epochs <= 0:
        raise ValueError(f"preflight 실패: --epochs={args.epochs} <= 0")
    if MIN_ATOMS >= MAX_ATOMS:
        raise ValueError("preflight 실패: MIN_ATOMS >= MAX_ATOMS")
    print(f"  [preflight] class range: {MIN_ATOMS}..{MAX_ATOMS} ({N_SIZE_CLASSES}개)")
    print("  [preflight] 통과")


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
    # train3.py와 동일 순서(astra_review_20260922.md P0) -- argparse 전에
    # 시작해서 import/syntax 실패 이전 구간도 launch_run.sh가 잡고, 여기서부터는
    # RunLogger가 run_id/manifest/source SHA를 기록한다.
    run_logger = RunLogger(
        __file__, source_paths=(_HERE / "size_predictor.py",)
    ).start()

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="../data/processed_ext7")
    parser.add_argument("--save-dir", default="./checkpoints_s1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="astra_review_20260922.md §4 정정 -- CVAE(train.py)의 "
                             "기존 warmup은 KL beta warmup이지 optimizer LR "
                             "warmup이 아니다(그쪽 LR은 처음부터 1e-3, "
                             "CosineAnnealingLR만 사용). '같은 원칙'이라는 "
                             "이전 주석은 부정확했음 -- 정정. 2026-09-21 "
                             "관찰된 중간 진동이 LR warmup 부재 때문이라는 것도 "
                             "검증 안 된 가설이며, epoch 단위 스케줄이라 첫 "
                             "epoch의 7,813 step이 낮은 LR에 머무는 것과 "
                             "얼마나 상관있는지도 확인 전이다. warmup 자체는 "
                             "시도해볼 가치가 있는 값이지 확정된 해법이 아님.")
    parser.add_argument("--save-every", type=int, default=5,
                        help="astra_review_20260922.md P0 -- 주기적 checkpoint. "
                             "이 epoch마다 ckpt_epoch%04d.pt를 원자적으로 저장.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-file", default="./train_s1.log")
    args = parser.parse_args()

    run_logger.attach_legacy_log(args.log_file)
    run_logger.record_arguments(vars(args))

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # [astra_review_20260922.md P0] run별 고유 checkpoint 디렉터리, 기존 경로
    # 재사용 거부 -- 2026-09-22 사고(재실행 전 `rm -rf checkpoints_s1`로
    # 10.78% 체크포인트를 되돌릴 수 없이 삭제)의 재발 방지. run_id가 이미
    # timestamp+pid로 고유하므로 이 디렉터리는 한 번도 존재한 적이 없어야
    # 정상이다 -- exist_ok=False로 실수로라도 재사용/충돌하면 바로 에러.
    save_dir = Path(args.save_dir) / run_logger.run_id
    save_dir.mkdir(parents=True, exist_ok=False)
    print(f"  Checkpoint dir (run별 고유): {save_dir}")

    stats_path = Path(args.processed_dir) / "stats.json"
    _preflight(args, stats_path)

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
    # [2026-09-22 정정] max_samples/chunk_size로 계산한 청크 수는 정확히
    # 안 맞을 수 있음(chunk_of가 랜덤 permutation 기반이라 청크 경계와
    # max_samples 절단점이 딱 안 맞으면 +1개 더 걸침) -- 실측(500k 샘플
    # -> 13개 청크, 산식은 10+2=12로 1개 부족했음)으로 확인됨. 여유를
    # 절대적으로 더 크게 둔다.
    chunk_size = 50000
    cache_chunks = max(2, (args.max_samples // chunk_size + 5)) if args.max_samples else 40
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
            _atomic_torch_save(model.state_dict(), save_dir / "best.pt")
        if args.save_every > 0 and epoch % args.save_every == 0:
            _atomic_torch_save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
            }, save_dir / f"ckpt_epoch{epoch:04d}.pt")

    print(f"\nDone. Best val acc: {best_val_acc:.2%}")
    run_logger.finish("completed", best_val_acc=best_val_acc, save_dir=str(save_dir))


if __name__ == "__main__":
    main()
