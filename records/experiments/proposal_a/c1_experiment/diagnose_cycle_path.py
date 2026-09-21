"""[진단] astra_review_20260920.md §7 제안: cycle_proj 경로를 실제로
배웠는지, 배웠다면 도움이 됐는지 구분. 재학습 불필요, checkpoints_c4_c1_feature
사용.
1. weight norm 비교: cycle_proj vs conn_proj/struct_proj/cond_proj.
2. 실제 배치에서 각 경로의 RMS(h에 더해지는 벡터 크기) 비교.
3. cycle path를 0으로 끈 뒤(monkeypatch) validation loss / 같은 시드
   생성 출력이 바뀌는지 측정 -- 안 바뀌면 "안 배웠다", 바뀌는데 최종
   지표(다른 실험에서 이미 확인된 8개 metric)가 그대로면 "배웠지만
   목적에 도움 안 됨".
"""
import sys
sys.path.insert(0, ".")
import torch
import model3 as m
from dataset2 import QMugsDataset
from torch_geometric.loader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = "./checkpoints_c4_c1_feature/best.pt"

sd = torch.load(CKPT, map_location=device)
model = m.MoleculeGraphDiffusion(sd["schedule.m_X"].clone(), sd["schedule.m_E"].clone())
missing, unexpected = model.load_state_dict(sd, strict=False)
model.to(device).eval()
print(f"missing={missing} unexpected={unexpected}")

net = model.net

print("\n=== 1. weight norm 비교 ===")
for name in ["conn_proj", "cycle_proj", "struct_proj", "cond_proj"]:
    layer = getattr(net, name)
    w_norm = layer.weight.norm().item()
    b_norm = layer.bias.norm().item()
    print(f"{name:<12} weight_norm={w_norm:.4f}  bias_norm={b_norm:.4f}  "
          f"weight_shape={tuple(layer.weight.shape)}")

print("\n=== 2. 실제 배치에서 각 struct 벡터 RMS 비교 ===")
train_set = QMugsDataset("./data/processed_ext7", split="train", max_samples=1000)
loader = DataLoader(train_set, batch_size=16, shuffle=False)
batch = next(iter(loader)).to(device)
X0, E0, node_mask = model._to_dense(batch)
t = torch.full((X0.shape[0],), 75, dtype=torch.long, device=device)

# forward를 직접 재현해서 각 struct 항을 개별적으로 뽑아냄
with torch.no_grad():
    cur_valence = E0.float().sum(dim=-1, keepdim=True) / 4.0
    atomic_weight = net.atomic_weight[X0.clamp(min=0, max=m.K_X - 1)].unsqueeze(-1) / 100.0
    struct_feat = net.struct_proj(torch.cat([cur_valence, atomic_weight], dim=-1))

    # c1_feature의 model3.py는 circuit-rank-free 버전이라 2-return
    try:
        n_components, comp_size_ratio = m._connectivity_features(E0, node_mask)
        n_comp_bcast = (n_components / 10.0).unsqueeze(1).expand(-1, X0.shape[1])
        conn_feat = torch.stack([n_comp_bcast, comp_size_ratio], dim=-1)
    except ValueError:
        n_components, comp_size_ratio, circuit_rank = m._connectivity_features(E0, node_mask)
        n_comp_bcast = (n_components / 10.0).unsqueeze(1).expand(-1, X0.shape[1])
        circuit_rank_bcast = (circuit_rank / 5.0).unsqueeze(1).expand(-1, X0.shape[1])
        conn_feat = torch.stack([n_comp_bcast, comp_size_ratio, circuit_rank_bcast], dim=-1)
    conn_struct = net.conn_proj(conn_feat)

    node_cycles, graph_cycles = m._ring_cycle_features(E0, node_mask, validate=False)
    node_cycles_scaled = m._scale_cycle_features(node_cycles)
    graph_cycles_scaled = m._scale_cycle_features(graph_cycles).unsqueeze(1).expand(-1, X0.shape[1], -1)
    cycle_feat = torch.cat([node_cycles_scaled, graph_cycles_scaled], dim=-1)
    cycle_struct = net.cycle_proj(cycle_feat)

    cond = torch.zeros(X0.shape[0], m.COND_DIM, device=device)
    cond_mask = torch.zeros(X0.shape[0], m.COND_DIM, device=device)
    cond_full = torch.cat([cond * cond_mask, cond_mask], dim=-1)
    cond_struct = net.cond_proj(cond_full).unsqueeze(1).expand(-1, X0.shape[1], -1)

    x_embed_out = net.x_embed(X0)

    nm = node_mask.unsqueeze(-1).float()
    def masked_rms(t):
        return ((t * nm).pow(2).sum() / nm.sum() / t.shape[-1]).sqrt().item()

    print(f"x_embed      RMS={masked_rms(x_embed_out):.4f}")
    print(f"struct_feat  RMS={masked_rms(struct_feat):.4f}")
    print(f"conn_struct  RMS={masked_rms(conn_struct):.4f}")
    print(f"cycle_struct RMS={masked_rms(cycle_struct):.4f}   <-- 이게 0에 가까우면 안 배운 것")
    print(f"cond_struct  RMS={masked_rms(cond_struct):.4f} (cond=None이라 학습된 bias만)")

print("\n=== 3. cycle path on/off 비교 (같은 입력, 같은 seed) ===")
with torch.no_grad():
    xl_on, el_on, pp_on = net(X0, E0, node_mask, t, cond=None, cond_mask=None)

    orig_cycle_proj_weight = net.cycle_proj.weight.data.clone()
    orig_cycle_proj_bias = net.cycle_proj.bias.data.clone()
    net.cycle_proj.weight.data.zero_()
    net.cycle_proj.bias.data.zero_()
    xl_off, el_off, pp_off = net(X0, E0, node_mask, t, cond=None, cond_mask=None)
    net.cycle_proj.weight.data.copy_(orig_cycle_proj_weight)
    net.cycle_proj.bias.data.copy_(orig_cycle_proj_bias)

    xl_diff = (xl_on - xl_off).abs()
    el_diff = (el_on - el_off).abs()
    pp_diff = (pp_on - pp_off).abs()
    print(f"x_logits  on/off 차이: mean={xl_diff.mean().item():.6f} max={xl_diff.max().item():.6f}")
    print(f"e_logits  on/off 차이: mean={el_diff.mean().item():.6f} max={el_diff.max().item():.6f}")
    print(f"prop_pred on/off 차이: mean={pp_diff.mean().item():.6f} max={pp_diff.max().item():.6f}")

    # validation loss 비교(전체 모델 forward, loss까지) -- model.forward()가
    # 내부에서 t/노이즈를 새로 무작위 샘플하므로, on/off 호출 사이에 같은
    # 시드로 재설정해야 진짜 ablation이 됨(안 그러면 다른 노이즈를 비교하게 됨).
    torch.manual_seed(123)
    out_on = model(batch)
    net.cycle_proj.weight.data.zero_()
    net.cycle_proj.bias.data.zero_()
    torch.manual_seed(123)
    out_off = model(batch)
    net.cycle_proj.weight.data.copy_(orig_cycle_proj_weight)
    net.cycle_proj.bias.data.copy_(orig_cycle_proj_bias)
    print(f"\nloss on={out_on['loss'].item():.4f}  loss off={out_off['loss'].item():.4f}  (같은 시드로 노이즈 고정)")

print("\n완료")
