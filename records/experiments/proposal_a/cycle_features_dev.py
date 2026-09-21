"""DiGress KNodeCycles 포팅.
k3/k4/k6: main 브랜치(docs/reference/digress_extra_features.py, main).
k5: **fixed_bug 브랜치**(docs/reference/digress_extra_features_fixed_bug.py,
commit 2a7c9b7b1c578a30293ec4fdbd4ea9ba48d7aaf1) — main의 k5는 삼각형+가지
(tailed-triangle) 패턴에서 오답(브루트포스 대조로 확인, astra_review_20260919b.md).
walk-count 함정 회피: 인접행렬 거듭제곱(A^2~A^6)의 trace/diagonal을 조합한
닫힌 형태 공식으로 3/4/5/6원 simple cycle 개수를 계산(단순 도보 수가 아님).
음수/비정수 clamp는 tolerance 기반(정수에서 1e-3 초과 벗어나면 버그로 간주해 assert).
"""
import math

import torch

# Astra 사전 등록 스케일(astra_review_20260919c.md): 모든 7채널(node c3/c4/c5,
# graph c3/c4/c5/c6)에 log1p(count)/log(11) 적용, 클리핑 없음.
# count=0 -> 0, count=10 -> 1(DiGress의 /10 기준점 유지), 10 초과도 순서 보존.
CYCLE_SCALE_LOG_BASE = math.log(11.0)


def _batch_trace(X):
    return torch.diagonal(X, dim1=-2, dim2=-1).sum(-1)


def _batch_diagonal(X):
    return torch.diagonal(X, dim1=-2, dim2=-1)


def scale_cycle_features(x):
    """사전 등록된 스케일: log1p(raw_count) / log(11), 클리핑 없음."""
    return torch.log1p(x) / CYCLE_SCALE_LOG_BASE


def compute_ring_size_features(E_t, node_mask, validate=True, clamp_tol=1e-3):
    """
    E_t: (B, N, N) 결합 차수/원-핫 등, > 0 이면 결합 존재
    node_mask: (B, N) bool, 실제 원자 여부
    validate: True면 음수/비정수 assert 검사 수행(테스트용, CUDA sync 다수 발생).
              학습/생성 production 경로에서는 False로 호출(astra_review_20260919c.md
              -- 검증용 .item() 호출이 CUDA에서 host-device sync를 다수 유발해
              비용 벤치마크를 오염시킴).
    반환:
      node_cycles: (B, N, 3) -- [3원, 4원, 5원] 참여 카운트 (노드별, 원본 DiGress처럼 절반 중복 보정 포함)
      graph_cycles: (B, 4) -- [3원, 4원, 5원, 6원] 그래프 전체 개수
    둘 다 raw count(정수값, round까지만 적용) -- 스케일은 scale_cycle_features()로 별도 적용.
    """
    B, N, _ = E_t.shape
    both_real = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
    eye = torch.eye(N, dtype=torch.bool, device=E_t.device).unsqueeze(0)
    adj = ((E_t > 0) & both_real & ~eye).float()  # 결합 존재 여부(차수 무시), 패딩/자기루프 완전 배제

    d = adj.sum(-1)
    k1 = adj
    k2 = k1 @ adj
    k3 = k2 @ adj
    k4 = k3 @ adj
    k5 = k4 @ adj
    k6 = k5 @ adj

    diag2 = _batch_diagonal(k2)
    diag3 = _batch_diagonal(k3)
    diag4 = _batch_diagonal(k4)
    diag5 = _batch_diagonal(k5)

    # --- 3원 ---
    c3_node = diag3 / 2
    c3_graph = diag3.sum(-1) / 6

    # --- 4원 ---
    c4 = diag4 - d * (d - 1) - (adj @ d.unsqueeze(-1)).squeeze(-1)
    c4_node = c4 / 2
    c4_graph = c4.sum(-1) / 8

    # --- 5원 (fixed_bug 브랜치 공식 — main의 tailed-triangle 오류 수정판) ---
    triangles = diag3 / 2
    joint_cycles = k2 * adj  # (i,j)가 공유하는 삼각형 수(존재하는 변에 대해서만)
    prod = 2 * (joint_cycles @ d.unsqueeze(-1)).squeeze(-1)
    prod2 = 2 * (adj @ triangles.unsqueeze(-1)).squeeze(-1)
    c5 = diag5 - prod - 4 * d * triangles - prod2 + 10 * triangles
    c5_node = c5 / 2
    c5_graph = c5.sum(-1) / 10

    # --- 6원 (그래프 단위만, DiGress 원본과 동일) ---
    term_1 = _batch_trace(k6)
    term_2 = (diag3 ** 2).sum(-1)  # batch_trace(k3**2)의 원소별-제곱 trace와 동치(대각만 남음)
    term_3 = torch.sum(adj * k2.pow(2), dim=[-2, -1])
    term_4 = (diag2 * diag4).sum(-1)
    term_5 = _batch_trace(k4)
    term_6 = _batch_trace(k3)
    term_7 = diag2.pow(3).sum(-1)
    term_8 = torch.sum(k3, dim=[-2, -1])
    term_9 = diag2.pow(2).sum(-1)
    term_10 = _batch_trace(k2)

    c6_graph = (term_1 - 3 * term_2 + 9 * term_3 - 6 * term_4 + 6 * term_5
                - 4 * term_6 + 4 * term_7 + 3 * term_8 - 12 * term_9 + 4 * term_10) / 12

    # Astra 지적(astra_review_20260919b.md §4-8): 음수뿐 아니라 "정수에서
    # 얼마나 벗어났는지"까지 확인 -- 0.4 같은 비정수 오류는 min>=0 검사만으론
    # 못 잡음. tolerance 넘는 음수/비정수는 clamp로 숨기지 말고 assert로 실패.
    # validate=False(production 경로)에서는 이 .item() 기반 검사를 건너뛴다
    # -- CUDA에서 채널당 2회씩 host-device sync가 발생해 비용 벤치마크를
    # 오염시키기 때문(astra_review_20260919c.md).
    if validate:
        named = [("c3_node", c3_node), ("c4_node", c4_node), ("c5_node", c5_node),
                 ("c3_graph", c3_graph), ("c4_graph", c4_graph), ("c5_graph", c5_graph),
                 ("c6_graph", c6_graph)]
        for name, t in named:
            min_val = t.min().item() if t.numel() > 0 else 0.0
            assert min_val >= -clamp_tol, f"{name} min={min_val} (tolerance={clamp_tol} 초과 음수 -- 버그 의심)"
            resid = (t - t.round()).abs()
            max_resid = resid.max().item() if resid.numel() > 0 else 0.0
            assert max_resid <= clamp_tol, f"{name} max |x-round(x)|={max_resid} (tolerance={clamp_tol} 초과 비정수 -- 버그 의심)"

    c3_node, c4_node, c5_node = (t.round() for t in (c3_node, c4_node, c5_node))
    c3_graph, c4_graph, c5_graph, c6_graph = (t.round() for t in (c3_graph, c4_graph, c5_graph, c6_graph))

    node_mask_f = node_mask.float()
    c3_node = c3_node.clamp(min=0) * node_mask_f
    c4_node = c4_node.clamp(min=0) * node_mask_f
    c5_node = c5_node.clamp(min=0) * node_mask_f
    c3_graph = c3_graph.clamp(min=0)
    c4_graph = c4_graph.clamp(min=0)
    c5_graph = c5_graph.clamp(min=0)
    c6_graph = c6_graph.clamp(min=0)

    node_cycles = torch.stack([c3_node, c4_node, c5_node], dim=-1)
    graph_cycles = torch.stack([c3_graph, c4_graph, c5_graph, c6_graph], dim=-1)
    return node_cycles, graph_cycles
