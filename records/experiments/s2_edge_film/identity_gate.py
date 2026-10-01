"""[S-2 1단계] zero-init edge FiLM 추가가 pilot_v1이 쓰는 동일한 frozen A-1
(circuit-rank 없는 model3_pec_base_snapshot.py, checkpoints_c4_a1/best.pt)
대비 checkpoint 로드 직후 출력을 bitwise로 보존하는지 확인하는 identity gate.
학습은 전혀 하지 않음, CPU only.
"""
import sys
import torch

# frozen 원본(수정 전): records/experiments/pec/model3_pec_base_snapshot.py와
# 바이트 단위로 동일한 파일을 import 가능한 이름으로 어딘가에 둔 뒤 그 디렉터리를
# sys.path에 넣는다(예: cp model3_pec_base_snapshot.py /tmp/s2_test_old/model3_pec_base_snapshot.py).
# 서버에서는 SHA가 같은 /home/cbgpu/molecule-AI-pec/model3.py를 그대로 써도 된다.
sys.path.insert(0, "/tmp/s2_test_old")       # frozen 원본(수정 전) -- 위 주석 참고해 준비
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))  # S-2 수정판(이 파일과 같은 디렉터리의 model3_s2.py)

import model3_pec_base_snapshot as old_model3  # noqa: E402
import model3_s2 as new_model3  # noqa: E402

torch.manual_seed(0)
CKPT = "/home/cbgpu/molecule-AI/checkpoints_c4_a1/best.pt"
sd = torch.load(CKPT, map_location="cpu", weights_only=False)

old = old_model3.MoleculeGraphDiffusion(sd["schedule.m_X"].clone(), sd["schedule.m_E"].clone())
res_old = old.load_state_dict(sd, strict=True)
assert not res_old.missing_keys and not res_old.unexpected_keys, res_old

new = new_model3.MoleculeGraphDiffusion(sd["schedule.m_X"].clone(), sd["schedule.m_E"].clone())
res_new = new.load_state_dict(sd, strict=False)
expected_missing = sorted(k for k in dict(new.named_parameters()).keys() if "edge_cond_mod" in k)
actual_missing = sorted(res_new.missing_keys)
assert actual_missing == expected_missing, (actual_missing, expected_missing)
assert not res_new.unexpected_keys, res_new.unexpected_keys
print(f"missing_keys(기대한 것만): {len(actual_missing)}개, 전부 edge_cond_mod -- OK")

for name, p in new.named_parameters():
    if "edge_cond_mod" in name:
        assert torch.all(p == 0), f"{name}이 0이 아님!"
print("edge_cond_mod 전부 0 확인 -- OK")

old.eval()
new.eval()

B, N = 2, 12
X_t = torch.randint(1, old_model3.K_X, (B, N))
E_t = torch.randint(0, old_model3.K_E, (B, N, N))
E_t = torch.triu(E_t, diagonal=1)
E_t = E_t + E_t.transpose(1, 2)
node_mask = torch.ones(B, N, dtype=torch.bool)
t = torch.randint(1, old_model3.T_STEPS, (B,))
cond = torch.randn(B, old_model3.COND_DIM)
cond_mask = (torch.rand(B, old_model3.COND_DIM) > 0.5).float()

with torch.no_grad():
    x_old, e_old, p_old = old.net(X_t, E_t, node_mask, t, cond, cond_mask)
    x_new, e_new, p_new = new.net(X_t, E_t, node_mask, t, cond, cond_mask)

x_match = torch.equal(x_old, x_new)
e_match = torch.equal(e_old, e_new)
p_match = torch.equal(p_old, p_new)
print(f"x_logits bitwise identical: {x_match}")
print(f"e_logits bitwise identical: {e_match}")
print(f"prop_pred bitwise identical: {p_match}")
assert x_match and e_match and p_match, "identity gate 실패 -- zero-init이 항등을 안 지킴"
print("\n[PASS] S-2 1단계(zero-init edge FiLM) identity gate: 체크포인트 로드 직후 출력 bitwise 동일")

with torch.no_grad():
    x_none_old, e_none_old, _ = old.net(X_t, E_t, node_mask, t, None, None)
    x_none_new, e_none_new, _ = new.net(X_t, E_t, node_mask, t, None, None)
assert torch.equal(x_none_old, x_none_new) and torch.equal(e_none_old, e_none_new)
print("[PASS] cond=None 경로도 bitwise 동일")

# edge_cond_mod가 실제로 edge에 영향을 줄 수 있는 경로인지(0이 아닌 값을 넣으면
# 출력이 달라지는지) 확인 -- 이게 안 바뀌면 배선이 죽어있다는 뜻.
with torch.no_grad():
    for p in new.net.layers[0].edge_cond_mod.parameters():
        p.add_(0.01)
    _, e_perturbed, _ = new.net(X_t, E_t, node_mask, t, cond, cond_mask)
changed = not torch.equal(e_old, e_perturbed)
print(f"[{'PASS' if changed else 'FAIL'}] edge_cond_mod 비0 값에서 e_logits 실제로 변함: {changed}")
assert changed
