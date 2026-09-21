"""제안 A 4단계(2차): clean + noisy(t=10/40/75/110/150) QMugs 분자에서
3/4/5/6원 simple cycle 개수 분포 측정 -- node-level까지, 3개 고정 seed로
noise draw 반복(astra_review_20260919c.md §4 지시 반영).
결과 JSON에 seed/실제 표본 수/샘플 index 해시/m_E/device/dtype/dataset
경로까지 기록(재현성 -- astra 지적 §4-3).
RDKit SSSR은 정확성 oracle로 쓰지 않음(ring basis 계열이라 정의가 다름).
"""
import hashlib
import json
import sys
import time

import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, ".")
from dataset2 import QMugsDataset
from model3 import ATOMIC_NUM_TO_CLS, K_E, K_X, MoleculeGraphDiffusion
from cycle_features_dev import compute_ring_size_features

N_MOL_TARGET = 2000
BATCH_SIZE = 32
T_POINTS = [10, 40, 75, 110, 150]
NOISE_SEEDS = [21, 22, 23]  # 3개 고정 seed로 반복(heavy tail 안정성 확인)
PROCESSED_DIR = "./data/processed_ext7"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("데이터셋 로딩 ...", flush=True)
train_set = QMugsDataset(PROCESSED_DIR, split="train")

print("marginal 분포 계산 (train3.py와 동일 절차) ...", flush=True)
torch.manual_seed(0)
x_counts = torch.zeros(K_X, dtype=torch.float)
n_weight_samples = min(5000, len(train_set))
for i in range(n_weight_samples):
    data = train_set[i]
    for anum in data.z:
        cls_idx = ATOMIC_NUM_TO_CLS.get(int(anum), 0)
        x_counts[cls_idx] += 1
x_counts[0] = 0.0
m_X = x_counts / x_counts.sum().clamp(min=1.0)

e_counts = torch.zeros(K_E, dtype=torch.float)
n_edge_samples = min(1000, len(train_set))
for i in range(n_edge_samples):
    data = train_set[i]
    btype = data.bond_type.argmax(dim=-1) + 1
    for bt in btype:
        e_counts[int(bt)] += 1
    n_atoms = len(data.z)
    total_possible = n_atoms * (n_atoms - 1)
    no_bond = total_possible - len(btype)
    e_counts[0] += no_bond
m_E = e_counts / e_counts.sum().clamp(min=1.0)

model = MoleculeGraphDiffusion(m_X, m_E).to(device)
model.eval()

# 데이터로더 -- shuffle=False 필수(dataset2.py 청크 로더가 LRU cache
# maxsize=2라 shuffle=True면 39개 대용량 청크 전체에서 무작위 접근하며
# 캐시가 거의 매번 미스나 극심히 느려짐; 이전 측정에서 실제로 겪은 버그).
loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)

# 채널: node[c3,c4,c5] + graph[c3,c4,c5,c6] = 7개 전부 기록
NODE_NAMES = ("node_c3", "node_c4", "node_c5")
GRAPH_NAMES = ("graph_c3", "graph_c4", "graph_c5", "graph_c6")

results = {}
for key in ["clean"] + [f"t{t}" for t in T_POINTS]:
    results[key] = {name: [] for name in NODE_NAMES + GRAPH_NAMES}

sample_index_log = []
n_done = 0
t0 = time.time()
with torch.no_grad():
    for batch in loader:
        if n_done >= N_MOL_TARGET:
            break
        batch = batch.to(device)
        X0, E0, node_mask = model._to_dense(batch)
        B = X0.shape[0]
        sample_index_log.extend(range(n_done, n_done + B))

        node_c, graph_c = compute_ring_size_features(E0, node_mask, validate=True)
        nm = node_mask.float()
        n_real = nm.sum(-1).clamp(min=1.0)
        for k, name in enumerate(NODE_NAMES):
            per_mol_mean = (node_c[:, :, k] * nm).sum(-1) / n_real
            results["clean"][name].extend(per_mol_mean.cpu().tolist())
        for k, name in enumerate(GRAPH_NAMES):
            results["clean"][name].extend(graph_c[:, k].cpu().tolist())

        for t in T_POINTS:
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            # 3개 고정 seed로 noise draw 반복 -- 매 seed 결과를 그대로 이어붙여
            # (반복 draw 자체가 분포 안정성 검증의 핵심이므로 평균내지 않음)
            for seed in NOISE_SEEDS:
                # q_sample은 별도 generator 인자를 안 받음(torch.multinomial 전역 RNG
                # 사용) -- 고정 seed 재현을 위해 전역 RNG를 매 draw 전에 설정.
                torch.manual_seed(seed * 100000 + t + n_done)
                E_t = model.schedule.q_sample(E0, t_tensor, model.schedule.m_E)
                E_t = model._sym(E_t)
                e_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
                E_t = torch.where(e_mask, E_t, torch.zeros_like(E_t))

                node_c_t, graph_c_t = compute_ring_size_features(E_t, node_mask, validate=True)
                for k, name in enumerate(NODE_NAMES):
                    per_mol_mean = (node_c_t[:, :, k] * nm).sum(-1) / n_real
                    results[f"t{t}"][name].extend(per_mol_mean.cpu().tolist())
                for k, name in enumerate(GRAPH_NAMES):
                    results[f"t{t}"][name].extend(graph_c_t[:, k].cpu().tolist())

        n_done += B

elapsed = time.time() - t0
print(f"\n분자 {n_done}개 처리(시드 {len(NOISE_SEEDS)}개 반복 포함 t당 {n_done*len(NOISE_SEEDS)}개 샘플), "
      f"총 소요 {elapsed:.1f}s", flush=True)

summary = {}
for key, d in results.items():
    summary[key] = {}
    for name, vals in d.items():
        vals_t = torch.tensor(vals)
        summary[key][name] = {
            "n": len(vals),
            "mean": vals_t.mean().item(),
            "p95": torch.quantile(vals_t, 0.95).item(),
            "p99": torch.quantile(vals_t, 0.99).item(),
            "max": vals_t.max().item(),
            "frac_gt_10": (vals_t > 10).float().mean().item(),
        }

sample_hash = hashlib.sha256(str(sample_index_log[:n_done]).encode()).hexdigest()[:16]

meta = {
    "n_mol": n_done,
    "n_mol_target": N_MOL_TARGET,
    "batch_size": BATCH_SIZE,
    "noise_seeds": NOISE_SEEDS,
    "t_points": T_POINTS,
    "sample_index_hash": sample_hash,
    "sample_index_range": [min(sample_index_log[:n_done]), max(sample_index_log[:n_done])],
    "m_E": m_E.tolist(),
    "device": str(device),
    "dtype": "float32",
    "dataset_path": PROCESSED_DIR,
    "elapsed_sec": elapsed,
}

out = {"meta": meta, "summary": summary}
print(json.dumps(meta, indent=2, ensure_ascii=False))
with open("/tmp/cycle_saturation_results_v2.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\n저장: /tmp/cycle_saturation_results_v2.json")
