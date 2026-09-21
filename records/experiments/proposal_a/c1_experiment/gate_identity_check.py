"""[C-1] 6단계 identity gate: checkpoints_c4_a1 로드 직후 cycle_proj(zero-init)
추가 전/후 logits이 완전히 동일한지 확인. molecule-AI-c1-control(cycle_proj
없음)과 molecule-AI-c1-feature(cycle_proj 있음, zero-init) 양쪽에서 각각
실행해서 비교한다."""
import sys
sys.path.insert(0, ".")
import torch
import model3 as m
from dataset2 import QMugsDataset
from torch_geometric.loader import DataLoader

device = torch.device("cuda")
CKPT = "./checkpoints_c4_a1/best.pt"

sd = torch.load(CKPT, map_location=device)
model = m.MoleculeGraphDiffusion(sd["schedule.m_X"].clone(), sd["schedule.m_E"].clone())
missing, unexpected = model.load_state_dict(sd, strict=False)
model.to(device).eval()

print(f"missing keys: {missing}")
print(f"unexpected keys: {unexpected}")

train_set = QMugsDataset("./data/processed_ext7", split="train")
loader = DataLoader(train_set, batch_size=8, shuffle=False)
batch = next(iter(loader)).to(device)

X0, E0, node_mask = model._to_dense(batch)
t = torch.full((X0.shape[0],), 75, dtype=torch.long, device=device)

with torch.no_grad():
    xl, el, pp = model.net(X0, E0, node_mask, t, cond=None, cond_mask=None)

torch.save({"xl": xl.cpu(), "el": el.cpu(), "pp": pp.cpu()}, "/tmp/gate_logits.pt")
print(f"xl sum={xl.sum().item():.6f} el sum={el.sum().item():.6f} pp sum={pp.sum().item():.6f}")
print("저장: /tmp/gate_logits.pt")
