import sys, json, time
sys.path.insert(0, "/home/cbgpu/molecule-AI")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool

from dataset2 import QMugsDataset

PROP_NAMES = ["HOMO", "LUMO", "GAP", "E_total", "Dipole"]
# [C-1] review8.md 지시: 조성 선형회귀 R²(하한) -- 대조용, 학습 없이 이미 나온 값
COMPOSITION_R2 = {"HOMO": 0.151, "LUMO": 0.344, "GAP": 0.344, "E_total": 1.0000, "Dipole": 0.130}

HIDDEN = 128
LAYERS = 3
N_BOND_FEAT = 3  # dataset2.py bond_type 실측 shape [E,3]
N_ATOM_TYPES = 128  # 원자번호 상한 여유 (I=53까지 실측 확인, model_.py 관례와 동일)

device = torch.device("cuda")


class PropRegressor(nn.Module):
    """순수 2D 그래프(원자종 + 결합종)만 보는 GINEConv 회귀기.
    [C-2] 조성 누설 방지 원칙과 동일하게, 조성 벡터(batch.a)나 원자 개수는
    입력에 절대 넣지 않는다 -- 오직 atom embedding + 결합 그래프 구조뿐."""

    def __init__(self, n_props=5):
        super().__init__()
        self.atom_emb = nn.Embedding(N_ATOM_TYPES, HIDDEN)
        self.convs = nn.ModuleList([
            GINEConv(
                nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN)),
                edge_dim=N_BOND_FEAT,
            )
            for _ in range(LAYERS)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(HIDDEN) for _ in range(LAYERS)])
        self.head = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, n_props)
        )

    def forward(self, batch):
        h = self.atom_emb(batch.z)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.relu(conv(h, batch.edge_index, batch.bond_type.float())))
        g = global_mean_pool(h, batch.batch)
        return self.head(g)


def r2_score(pred, target):
    ss_res = ((pred - target) ** 2).sum(dim=0)
    ss_tot = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    return 1 - ss_res / ss_tot.clamp(min=1e-8)


def mae(pred, target):
    return (pred - target).abs().mean(dim=0)


def main():
    train_set = QMugsDataset("./data/processed", split="train", max_samples=100_000)
    val_set = QMugsDataset("./data/processed", split="val", max_samples=8_000)

    # dataset2.py 자체가 이미 청크 국소성 있는 순서로 인덱스를 구성해두므로
    # (주석: "DataLoader shuffle=False로 cache hit 보장") shuffle=True를 쓰면
    # 전역 무작위 접근이 되어 lru_cache(maxsize=2)가 매 아이템마다 청크를
    # 갈아치우게 된다 -- 디스크 I/O가 병목이 되어 사실상 멈춘 것처럼 보임.
    # epoch마다 다양성을 주려면 dataset.reshuffle_indices()를 쓴다(청크 내부만 섞음).
    train_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0)

    model = PropRegressor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)

    history = []
    N_EPOCHS = 15
    for epoch in range(1, N_EPOCHS + 1):
        train_set.reshuffle_indices()
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch)
            target = batch.p.view(pred.shape[0], -1)
            loss = F.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        sched.step()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                target = batch.p.view(pred.shape[0], -1)
                val_preds.append(pred.cpu())
                val_targets.append(target.cpu())
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)

        # 정규화된 공간 R²는 스케일 무관하게 원래 공간과 동일 (평행이동/척도 불변)
        r2 = r2_score(val_preds, val_targets)
        # MAE는 원래 물성 단위로 환산해서 보고
        std = val_set.std
        mean = val_set.std * 0 + val_set.std  # placeholder not needed
        mae_norm = mae(val_preds, val_targets)
        mae_raw = mae_norm * val_set.std

        train_loss = total_loss / n_batches
        elapsed = time.time() - t0
        row = {
            "epoch": epoch, "train_loss": train_loss,
            "val_r2": r2.tolist(), "val_mae_raw": mae_raw.tolist(),
            "elapsed": elapsed,
        }
        history.append(row)
        r2_str = " ".join(f"{n}={v:.3f}" for n, v in zip(PROP_NAMES, r2.tolist()))
        mae_str = " ".join(f"{n}={v:.4f}" for n, v in zip(PROP_NAMES, mae_raw.tolist()))
        print(f"[epoch {epoch:2d}] train_loss={train_loss:.4f}  time={elapsed:.1f}s", flush=True)
        print(f"    R2:  {r2_str}", flush=True)
        print(f"    MAE: {mae_str}", flush=True)

    print("\n=== FINAL (held-out val) ===", flush=True)
    final_r2 = history[-1]["val_r2"]
    final_mae = history[-1]["val_mae_raw"]
    print(f"{'prop':10s} {'comp_R2(floor)':>15s} {'GNN_R2(ceiling)':>16s} {'GNN_MAE':>10s} {'beats_floor':>12s}", flush=True)
    for i, name in enumerate(PROP_NAMES):
        beats = "YES" if final_r2[i] > COMPOSITION_R2[name] else "NO"
        print(f"{name:10s} {COMPOSITION_R2[name]:15.4f} {final_r2[i]:16.4f} {final_mae[i]:10.4f} {beats:>12s}", flush=True)

    with open("c1_history.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save({
        "model": model.state_dict(),
        "val_mean": val_set.mean,
        "val_std": val_set.std,
    }, "c1_regressor.pt")
    print("saved c1_regressor.pt", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
