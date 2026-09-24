"""[S-1, P1] astra_review_20260922.md §3 -- "전체 관측 accuracy 하나는
대표 지표가 아니다." 학습된 checkpoint를 고정된 마스크 패널들에서
각각 평가한다:

1. 7개 전부 관측
2. HOMO/LUMO만 관측
3. 구조/약물형 속성 5개(LogP/TPSA/HBA/RotBonds/AromaticRings)만 관측
4. 속성별 leave-one-out(7개, 그 속성 하나만 가림)
5. 모두 미관측(경험적 prior와 동일 조건)

각 패널에 NLL/exact hit/±1 hit/MAE/80% coverage(모델이 스스로 80%
확률질량이라 주장하는 예측집합이 실제로 80% 커버하는지)를 보고한다.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from size_predictor import SizePredictor, SimpleConcatMLP, MIN_ATOMS, MAX_ATOMS, N_SIZE_CLASSES, COND_DIM

MODEL_CLASSES = {"size_predictor": SizePredictor, "concat_mlp": SimpleConcatMLP}
COND_PROP_NAMES = ["HOMO", "LUMO", "LogP", "TPSA", "HBA", "RotBonds", "AromaticRings"]


def build_panels():
    """이름 -> 관측(mask=1)할 속성 인덱스 리스트."""
    panels = {}
    panels["1_full_observed"] = list(range(COND_DIM))
    panels["2_homo_lumo_only"] = [0, 1]
    panels["3_structural5_only"] = [2, 3, 4, 5, 6]
    for i, name in enumerate(COND_PROP_NAMES):
        observed = [j for j in range(COND_DIM) if j != i]
        panels[f"4_leave_out_{name}"] = observed
    panels["5_all_unobserved"] = []
    return panels


@torch.no_grad()
def evaluate_panel(model, p7, target, observed_indices, device, coverage_level=0.8):
    B = p7.shape[0]
    cond_mask = torch.zeros(B, COND_DIM, device=device)
    if observed_indices:
        cond_mask[:, observed_indices] = 1.0
    logits = model(p7, cond_mask)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    nll = F.nll_loss(log_probs, target).item()
    pred = logits.argmax(-1)
    exact_hit = (pred == target).float().mean().item()
    pm1_hit = ((pred - target).abs() <= 1).float().mean().item()
    mae = (pred - target).abs().float().mean().item()

    # coverage: 확률 내림차순으로 누적해 coverage_level을 처음 넘기는 지점까지를
    # 예측 신뢰구간(집합)으로 삼고, 실제 target이 그 집합 안에 있는 비율을 잰다.
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cum = sorted_probs.cumsum(dim=-1)
    # 각 행에서 cum >= coverage_level을 처음 만족하는 위치(포함)까지가 집합
    in_set_rank = (cum < coverage_level).sum(dim=-1) + 1  # 몇 번째까지 포함하는지
    in_set_rank = in_set_rank.clamp(max=N_SIZE_CLASSES)
    target_rank = (sorted_idx == target.unsqueeze(-1)).float().argmax(dim=-1) + 1
    covered = (target_rank <= in_set_rank).float().mean().item()
    avg_set_size = in_set_rank.float().mean().item()

    return {
        "n": B, "nll": nll, "exact_hit": exact_hit, "pm1_hit": pm1_hit, "mae": mae,
        f"coverage_at_{coverage_level}": covered, "avg_credible_set_size": avg_set_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=list(MODEL_CLASSES.keys()), default="size_predictor")
    parser.add_argument("--features-dir", default="./features")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", default=None, help="결과 JSON 저장 경로(생략 시 stdout만)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = torch.load(Path(args.features_dir) / f"{args.split}.pt", weights_only=False)
    p7, n_atoms = d["p7"], d["n_atoms"]
    in_range = (n_atoms >= MIN_ATOMS) & (n_atoms <= MAX_ATOMS)
    p7, n_atoms = p7[in_range], n_atoms[in_range]
    target = (n_atoms - MIN_ATOMS).clamp(0, N_SIZE_CLASSES - 1).to(device)
    p7 = p7.to(device)
    print(f"{args.split}: {len(p7):,}개 (범위 밖 {(~in_range).sum().item()}개 제외)")

    model = MODEL_CLASSES[args.model]().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print(f"checkpoint 로드: {args.checkpoint}")

    panels = build_panels()
    results = {}
    print(f"\n{'panel':<28} {'n':>8} {'NLL':>8} {'exact':>8} {'+-1':>8} {'MAE':>7} {'cov80':>7} {'setsz':>7}")
    print("-" * 90)
    for name, observed in panels.items():
        r = evaluate_panel(model, p7, target, observed, device)
        results[name] = r
        print(f"{name:<28} {r['n']:>8,} {r['nll']:>8.4f} {r['exact_hit']:>8.2%} "
              f"{r['pm1_hit']:>8.2%} {r['mae']:>7.2f} {r['coverage_at_0.8']:>7.2%} "
              f"{r['avg_credible_set_size']:>7.1f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "model": args.model,
                       "split": args.split, "panels": results}, f, indent=2)
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
