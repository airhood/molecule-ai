"""[S-1, P1] astra_review_20260922.md P1 반영 -- train_s1.py는 매 epoch
QMugsDataset의 500MB 그래프 청크를 다시 읽어(청크 재로딩 병목으로 20epoch
학습이 8시간+ 걸림). 이 버전은 extract_features.py가 한 번 뽑아놓은 작은
feature table(features/{train,val,test}.pt, 7개 정규화 물성+heavy atom
count만 담음, 전체 합쳐도 <100MB)만 메모리에 올려서 학습한다 -- 청크
로딩이 아예 없으므로 50epoch도 수 분 내로 끝나야 정상이다.

train_s1.py와 다른 점(둘 다 astra_review_20260922.md 지적 반영):
1. 데이터: feature table 직접 로드 (dataset2.py/QMugsDataset 안 씀)
2. 2~50 범위 밖 샘플은 clamp(잘못 라벨링)가 아니라 학습/평가에서 제외
3. 경험적 prior(최빈값 정확도, prior 분포 NLL)를 학습 시작 전에 계산해
   baseline으로 같이 출력 -- "랜덤 대비"가 아니라 "prior 대비" 비교
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from run_logger import RunLogger
from size_predictor import (SizePredictor, SimpleConcatMLP, MIN_ATOMS, MAX_ATOMS,
                             N_SIZE_CLASSES, COND_DIM)

MODEL_CLASSES = {"size_predictor": SizePredictor, "concat_mlp": SimpleConcatMLP}

COND_DROPOUT_P_FULL = 0.15
COND_DROPOUT_P_PARTIAL = 0.2


def _atomic_torch_save(obj, path):
    import os
    path = Path(path)
    tmp_path = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp_path, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _preflight(features_dir):
    """feature table이 이 스크립트가 기대하는 property index/이름과
    일치하는지 실행 전에 확인 -- extract_features.py가 다른 설정으로
    다시 돌아간 파일을 실수로 가리키면 여기서 바로 걸러진다."""
    manifest_path = Path(features_dir) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    expected_names = ["HOMO", "LUMO", "LogP", "TPSA", "HBA", "RotBonds", "AromaticRings"]
    if manifest.get("cond_prop_names") != expected_names:
        raise ValueError(f"preflight 실패: {manifest_path}의 cond_prop_names="
                          f"{manifest.get('cond_prop_names')}가 기대값과 다름")
    for split in ("train", "val", "test"):
        if manifest.get(f"{split}_count", 0) <= 0:
            raise ValueError(f"preflight 실패: manifest에 {split}_count가 없거나 0")
    print(f"  [preflight] feature table manifest 확인됨: "
          f"train={manifest['train_count']:,} val={manifest['val_count']:,}")
    print(f"  [preflight] 통과")


def load_split(features_dir, name, device):
    """feature table 로드 + 2~50 범위 밖 샘플 제외(astra_review_20260922.md
    §"먼저 2~50 범위 밖 샘플을 train/val에서 제외..."). clamp로 2/50에
    잘못 라벨링하던 이전 방식 대신."""
    d = torch.load(Path(features_dir) / f"{name}.pt", weights_only=False)
    p7, n_atoms = d["p7"], d["n_atoms"]
    in_range = (n_atoms >= MIN_ATOMS) & (n_atoms <= MAX_ATOMS)
    n_dropped = (~in_range).sum().item()
    p7, n_atoms = p7[in_range], n_atoms[in_range]
    target = (n_atoms - MIN_ATOMS).clamp(0, N_SIZE_CLASSES - 1)
    print(f"  {name}: {len(p7):,}개 (범위 밖 {n_dropped}개 제외, "
          f"{n_dropped / (len(p7) + n_dropped):.2%})")
    return p7.to(device), target.to(device)


def empirical_prior_baseline(train_target, n_classes):
    """astra_review_20260922.md -- 균등 랜덤(1/n_classes)이 아니라 학습
    분포의 최빈값/전체 분포를 기준선으로 삼아야 한다. 반환: (최빈값
    정확도로 잰 baseline accuracy 함수는 val에서 별도 계산, prior 확률
    벡터, prior가 val에 대해 내는 NLL 계산용 log-prob)."""
    counts = torch.bincount(train_target, minlength=n_classes).float()
    probs = counts / counts.sum()
    log_probs = torch.log(probs.clamp(min=1e-12))
    mode_class = counts.argmax().item()
    return probs, log_probs, mode_class


def make_cond_mask(B, device, train: bool):
    if not train:
        return torch.ones(B, COND_DIM, device=device)
    full_drop = torch.rand(B, device=device) < COND_DROPOUT_P_FULL
    partial_drop = torch.rand(B, COND_DIM, device=device) < COND_DROPOUT_P_PARTIAL
    cond_mask = (~partial_drop).float()
    cond_mask[full_drop] = 0.0
    return cond_mask


def run_epoch(model, p7, target, device, batch_size, train, optimizer=None, generator=None):
    model.train(train)
    n = p7.shape[0]
    perm = torch.randperm(n, generator=generator, device="cpu").to(device) if train \
        else torch.arange(n, device=device)
    total_loss, total_correct = 0.0, 0
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        p, t = p7[idx], target[idx]
        B = p.shape[0]
        cond_mask = make_cond_mask(B, device, train)
        logits = model(p, cond_mask)
        loss = F.cross_entropy(logits, t)
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item() * B
        total_correct += (logits.argmax(-1) == t).sum().item()
    return total_loss / n, total_correct / n


def main():
    run_logger = RunLogger(__file__, source_paths=(_HERE / "size_predictor.py",)).start()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CLASSES.keys()), default="size_predictor",
                        help="astra_review_20260922.md §1 -- concat_mlp는 강한 단순 "
                             "기준선(SimpleConcatMLP), 동일 train/val/seed/budget으로 "
                             "size_predictor와 비교하기 위함.")
    parser.add_argument("--features-dir", default="./features")
    parser.add_argument("--save-dir", default="./checkpoints_s1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-file", default="./train_s1_v2.log")
    args = parser.parse_args()

    run_logger.attach_legacy_log(args.log_file)
    run_logger.record_arguments(vars(args))

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir) / run_logger.run_id
    save_dir.mkdir(parents=True, exist_ok=False)
    print(f"  Checkpoint dir (run별 고유): {save_dir}")
    print(f"Device: {device}")

    _preflight(args.features_dir)

    print("Loading feature table ...")
    train_p7, train_target = load_split(args.features_dir, "train", device)
    val_p7, val_target = load_split(args.features_dir, "val", device)

    probs, log_probs, mode_class = empirical_prior_baseline(train_target.cpu(), N_SIZE_CLASSES)
    prior_val_acc = (mode_class == val_target.cpu()).float().mean().item()
    prior_val_nll = -log_probs[val_target.cpu()].mean().item()
    print(f"  [baseline] 경험적 prior: 최빈 class={mode_class + MIN_ATOMS}(원자수), "
          f"val acc={prior_val_acc:.2%}, val NLL={prior_val_nll:.4f}")

    model = MODEL_CLASSES[args.model]().to(device)
    print(f"  Model: {args.model} ({MODEL_CLASSES[args.model].__name__})")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

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

    gen = torch.Generator().manual_seed(args.seed)
    # [astra_review_20260924.md §4-1 반영] 이전엔 val accuracy로 best.pt를
    # 고르고, 보고할 땐 최종 epoch의 NLL을 갖다 붙여서 서로 다른 epoch의
    # 지표를 한 checkpoint인 것처럼 섞어 보고하는 실수가 있었다. 이제
    # 선택 기준을 val NLL(분포 전체를 보는 지표, exact match보다 우선)로
    # 고정하고, 그 "선택된 epoch"의 acc/NLL을 같이 저장해 섞이지 않게 한다.
    best_val_nll = float("inf")
    best_epoch, best_val_acc_at_best = None, None
    print(f"\n{'Epoch':>6}  {'T-loss':>8} {'T-acc':>8}  {'V-loss':>8} {'V-acc':>8}")
    print("-" * 50)
    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = run_epoch(model, train_p7, train_target, device, args.batch_size,
                                   train=True, optimizer=optimizer, generator=gen)
        with torch.no_grad():
            v_loss, v_acc = run_epoch(model, val_p7, val_target, device, args.batch_size,
                                       train=False)
        scheduler.step()
        print(f"{epoch:>6}  {t_loss:>8.4f} {t_acc:>8.2%}  {v_loss:>8.4f} {v_acc:>8.2%}")
        if v_loss < best_val_nll:
            best_val_nll = v_loss
            best_epoch = epoch
            best_val_acc_at_best = v_acc
            _atomic_torch_save({"model": model.state_dict(), "epoch": epoch,
                                 "val_nll": v_loss, "val_acc": v_acc},
                                save_dir / "best.pt")
        if args.save_every > 0 and epoch % args.save_every == 0:
            _atomic_torch_save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "val_nll": v_loss, "val_acc": v_acc,
            }, save_dir / f"ckpt_epoch{epoch:04d}.pt")

    print(f"\nDone. Best checkpoint: epoch {best_epoch}, val NLL {best_val_nll:.4f}, "
          f"val acc {best_val_acc_at_best:.2%}  "
          f"(prior baseline: {prior_val_acc:.2%}, prior NLL: {prior_val_nll:.4f})")
    run_logger.finish("completed", best_epoch=best_epoch, best_val_nll=best_val_nll,
                       best_val_acc=best_val_acc_at_best,
                       prior_val_acc=prior_val_acc, prior_val_nll=prior_val_nll,
                       save_dir=str(save_dir))


if __name__ == "__main__":
    main()
