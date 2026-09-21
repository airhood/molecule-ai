"""[C-1] 5단계: 비용 벤치마크 (astra_review_20260919c.md 지시 반영).
- validate=False(production 경로)로 측정, CUDA event로 시간 측정.
- (a) feature kernel 단독: 실제 원자수 분포 + N=50 worst-case
- (b) 학습 batch forward+backward + peak memory (monkeypatch로 실제
  net.forward 안에 삽입된 것과 동일한 비용을 baseline 대비 측정)
- (c) CFG sampling 1-step + 전체 150-step 생성, feature 유무 samples/s
- (d) CFG의 conditional/unconditional 배치가 E_t를 중복(cat)하는데,
  구조 feature가 이 중복된 배치 전체에 대해 다시 계산되는 중복비용을
  정량화(astra 지적)
"""
import sys
import time

import torch

sys.path.insert(0, ".")
import model3 as m
from model3 import MoleculeGraphDiffusion, MAX_ATOMS
from cycle_features_dev import compute_ring_size_features
from dataset2 import QMugsDataset
from torch_geometric.loader import DataLoader

device = torch.device("cuda")
BATCH_SIZE = 12  # 프로젝트 표준 배치 (batch=32 롤백 사건 이후 확정값)
N_WARMUP = 5
N_TIMED = 20


def cuda_time(fn, n_warmup=N_WARMUP, n_timed=N_TIMED):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_timed):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_timed  # ms/call


def make_dummy_E(B, N, density=0.15, device=device):
    adj = (torch.rand(B, N, N, device=device) < density).float()
    triu = torch.triu(torch.ones(N, N, device=device), diagonal=1)
    adj = adj * triu
    E = adj + adj.transpose(-1, -2)
    return E


print("=" * 60)
print("(a) feature kernel 단독 -- 실제 원자수 분포 vs N=50 worst-case")
print("=" * 60)

train_set = QMugsDataset("./data/processed_ext7", split="train")
loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
batch = next(iter(loader)).to(device)

x_counts = torch.zeros(m.K_X, dtype=torch.float)
for i in range(min(2000, len(train_set))):
    for anum in train_set[i].z:
        x_counts[m.ATOMIC_NUM_TO_CLS.get(int(anum), 0)] += 1
x_counts[0] = 0
m_X = x_counts / x_counts.sum().clamp(min=1)
e_counts = torch.zeros(m.K_E, dtype=torch.float); e_counts[0] = 1
m_E = e_counts / e_counts.sum()
model = MoleculeGraphDiffusion(m_X, m_E).to(device)
model.eval()

X0, E0, node_mask = model._to_dense(batch)
real_N = E0.shape[1]
print(f"실제 배치: B={E0.shape[0]}, N(batch-local max)={real_N}")

t_ms = cuda_time(lambda: compute_ring_size_features(E0, node_mask, validate=False))
print(f"실제 분포 (B={BATCH_SIZE}, N={real_N}): {t_ms:.3f} ms/call")

E50 = make_dummy_E(BATCH_SIZE, MAX_ATOMS)
mask50 = torch.ones(BATCH_SIZE, MAX_ATOMS, dtype=torch.bool, device=device)
t_ms_50 = cuda_time(lambda: compute_ring_size_features(E50, mask50, validate=False))
print(f"worst-case (B={BATCH_SIZE}, N={MAX_ATOMS}): {t_ms_50:.3f} ms/call")

print()
print("=" * 60)
print("(b) 학습 batch forward+backward + peak memory (baseline vs +feature)")
print("=" * 60)

t_tensor = torch.randint(1, model.T + 1, (BATCH_SIZE,), device=device)
X_t = model.schedule.q_sample(X0, t_tensor, model.schedule.m_X)
E_t = model._sym(model.schedule.q_sample(E0, t_tensor, model.schedule.m_E))


def train_step_baseline():
    model.zero_grad(set_to_none=True)
    out = model(batch)
    out["loss"].backward()


def train_step_with_feature():
    model.zero_grad(set_to_none=True)
    # net.forward 안에서 _connectivity_features 다음에 실제로 삽입될
    # 위치와 동일한 지점 -- E_t/node_mask 계산 직후, 결과는 향후 cycle_proj
    # 입력이 될 것이므로 여기서는 계산 비용만 정직하게 추가(투영 레이어
    # 자체는 아직 미도입이라 결과는 버림).
    out = model(batch)
    _ = compute_ring_size_features(E_t, node_mask, validate=False)
    out["loss"].backward()


torch.cuda.reset_peak_memory_stats()
t_base = cuda_time(train_step_baseline, n_warmup=3, n_timed=10)
mem_base = torch.cuda.max_memory_allocated() / 1e6

torch.cuda.reset_peak_memory_stats()
t_feat = cuda_time(train_step_with_feature, n_warmup=3, n_timed=10)
mem_feat = torch.cuda.max_memory_allocated() / 1e6

print(f"baseline:    {t_base:.2f} ms/step, peak mem {mem_base:.1f} MB")
print(f"+feature:    {t_feat:.2f} ms/step, peak mem {mem_feat:.1f} MB")
print(f"오버헤드:     +{t_feat - t_base:.2f} ms/step ({(t_feat/t_base - 1)*100:.2f}%), "
      f"+{mem_feat - mem_base:.1f} MB")

print()
print("=" * 60)
print("(c) CFG sampling 1-step + 전체 150-step 생성")
print("=" * 60)

N_GEN = 8
N_ATOMS_GEN = 40
cond = torch.zeros(N_GEN, m.COND_DIM, device=device)

with torch.no_grad():
    node_mask_gen = torch.ones(N_GEN, N_ATOMS_GEN, dtype=torch.bool, device=device)
    X_gen = torch.randint(1, m.K_X, (N_GEN, N_ATOMS_GEN), device=device)
    # E_t는 e_embed(nn.Embedding)에 들어가므로 정수 결합클래스(0..K_E-1)여야 함
    bond_cls = torch.randint(0, m.K_E, (N_GEN, N_ATOMS_GEN, N_ATOMS_GEN), device=device)
    triu = torch.triu(torch.ones(N_ATOMS_GEN, N_ATOMS_GEN, device=device, dtype=torch.bool), diagonal=1)
    bond_cls = torch.where(triu.unsqueeze(0), bond_cls, torch.zeros_like(bond_cls))
    E_gen = bond_cls + bond_cls.transpose(-1, -2)
    t_gen = torch.full((N_GEN,), 75, dtype=torch.long, device=device)

    def cfg_step_baseline():
        null_cond = torch.zeros(N_GEN, m.COND_DIM, device=device)
        X2 = torch.cat([X_gen, X_gen], dim=0)
        E2 = torch.cat([E_gen, E_gen], dim=0)
        mask2 = torch.cat([node_mask_gen, node_mask_gen], dim=0)
        t2 = torch.cat([t_gen, t_gen], dim=0)
        cond2 = torch.cat([cond, null_cond], dim=0)
        model.net(X2, E2, mask2, t2, cond2, torch.ones_like(cond2))

    t_cfg_ms = cuda_time(cfg_step_baseline, n_warmup=3, n_timed=10)
    print(f"CFG 1-step (net.forward, 2x{N_GEN} 배치, N={N_ATOMS_GEN}): {t_cfg_ms:.3f} ms")

    # (d) 중복 계산 비용 정량화: E_t를 n개로 한 번만 계산 vs 2n(중복)으로 계산
    t_feat_n = cuda_time(lambda: compute_ring_size_features(E_gen, node_mask_gen, validate=False))
    E_gen2 = torch.cat([E_gen, E_gen], dim=0)
    mask_gen2 = torch.cat([node_mask_gen, node_mask_gen], dim=0)
    t_feat_2n = cuda_time(lambda: compute_ring_size_features(E_gen2, mask_gen2, validate=False))
    print(f"\n(d) CFG 중복계산 비용: feature(n={N_GEN})={t_feat_n:.3f}ms, "
          f"feature(2n={2*N_GEN}, 중복내용)={t_feat_2n:.3f}ms "
          f"(비율 {t_feat_2n/t_feat_n:.2f}x -- 2배 가까우면 dedup으로 최대 {t_feat_2n-t_feat_n:.3f}ms/step 절약 가능)")

    print(f"\n전체 150-step 생성 예상 추가 비용: {t_feat_2n * 150 / 1000:.3f}s "
          f"(dedup 시 {t_feat_n * 150 / 1000:.3f}s)")

    def full_gen_baseline():
        model.sample(N_GEN, N_ATOMS_GEN, device, cond=cond, guidance_w=1.0)

    t0 = time.time()
    torch.cuda.synchronize()
    full_gen_baseline()
    torch.cuda.synchronize()
    t_full = time.time() - t0
    print(f"\n전체 150-step 생성 실측(baseline, feature 없음): {t_full:.2f}s, "
          f"{N_GEN/t_full:.2f} molecules/s")
    print(f"feature 추가 시 예상: {t_full + t_feat_2n*150/1000:.2f}s "
          f"({(t_feat_2n*150/1000/t_full)*100:.2f}% 증가, dedup 시 "
          f"{(t_feat_n*150/1000/t_full)*100:.2f}% 증가)")

print("\n완료")
