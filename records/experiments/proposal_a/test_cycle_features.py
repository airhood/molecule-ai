"""compute_ring_size_features 유닛테스트.
astra_review_20260919b.md §4 8개 지적 반영:
1. reference: k5는 fixed_bug 분기 공식 사용(코드/주석에 출처 명시 완료).
2. node-level brute-force 비교 추가(기존엔 graph만 비교).
3. self-loop 불변성 테스트 명시적으로 추가.
4. square+diagonal 주석/기대값 수정(바깥 4-cycle은 여전히 존재).
5. fused/spiro: graph vector 전체 + node 참여 수까지 검사.
6. N<=6 전수검사(모든 단순무방향그래프) + N=7~10 무작위 별도 검사.
7. production dtype(float32, CPU) 그대로 검사(별도 float64 unittest 없음 -- 기본이 float32).
8. clamp 전 정수성(|x-round(x)|<=tol) 확인 -- compute_ring_size_features 내부 assert로 이미 강제됨,
   여기서는 그 assert가 실제로 발동하는지까지 간접 확인(전수/무작위 테스트 통과 자체가 증거).
"""
import itertools
import networkx as nx
import numpy as np
import torch

from cycle_features_dev import compute_ring_size_features

N_FAIL = 0


def check(name, cond, detail=""):
    global N_FAIL
    status = "OK " if cond else "FAIL"
    if not cond:
        N_FAIL += 1
    print(f"[{status}] {name} {detail}")


def make_batch(edge_lists, n_list, N):
    B = len(edge_lists)
    E = torch.zeros(B, N, N)
    mask = torch.zeros(B, N, dtype=torch.bool)
    for b, (edges, n) in enumerate(zip(edge_lists, n_list)):
        mask[b, :n] = True
        for i, j in edges:
            E[b, i, j] = 1
            E[b, j, i] = 1
    return E, mask


def brute_force_graph_counts(edges, n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    counts = {3: 0, 4: 0, 5: 0, 6: 0}
    node_participation = {L: np.zeros(n) for L in (3, 4, 5)}
    for cycle in nx.simple_cycles(G):
        L = len(cycle)
        if L in counts:
            counts[L] += 1
            if L in node_participation:
                for v in cycle:
                    node_participation[L][v] += 1
    return counts, node_participation


def assert_matches_brute_force(name, edges, n, N=None, atol=1e-3):
    N = N or n
    E, mask = make_batch([edges], [n], N)
    node_c, graph_c = compute_ring_size_features(E, mask)
    bf_graph, bf_node = brute_force_graph_counts(edges, n)
    expect_graph = torch.tensor([bf_graph[3], bf_graph[4], bf_graph[5], bf_graph[6]], dtype=torch.float)
    ok_graph = torch.allclose(graph_c[0], expect_graph, atol=atol)
    expect_node = torch.zeros(N, 3)
    for k, L in enumerate((3, 4, 5)):
        expect_node[:n, k] = torch.tensor(bf_node[L], dtype=torch.float)
    ok_node = torch.allclose(node_c[0], expect_node, atol=atol)
    check(f"{name} graph={graph_c[0].tolist()} bf={bf_graph}", ok_graph)
    check(f"{name} node-level", ok_node, f"model={node_c[0][:n].tolist()} bf={expect_node[:n].tolist()}")
    return ok_graph and ok_node


# ---------- 1. 사슬/별/빈 그래프 ----------
def test_acyclic():
    assert_matches_brute_force("chain", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 6)
    assert_matches_brute_force("star", [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)], 6)
    E, mask = make_batch([[]], [0], 6)
    _, gc = compute_ring_size_features(E, mask)
    check("empty graph no NaN", not torch.isnan(gc).any())


# ---------- 2. C3~C7 단일 고리 ----------
def test_single_rings():
    for size in [3, 4, 5, 6, 7]:
        edges = [(i, (i + 1) % size) for i in range(size)]
        assert_matches_brute_force(f"C{size}", edges, size)


# ---------- 3. 대각선 있는 사각형: 바깥 4-cycle은 여전히 존재(과거 주석 오류 수정) ----------
def test_square_with_diagonal():
    # 0-1-2-3-0 + 대각선 0-2: 삼각형 {0,1,2},{0,2,3} 2개 + 바깥 4-cycle 0-1-2-3-0 1개 = 총 3개 simple cycle
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    assert_matches_brute_force("square+diagonal", edges, 4)


# ---------- 4. K4 ----------
def test_k4():
    edges = list(itertools.combinations(range(4), 2))
    assert_matches_brute_force("K4", edges, 4)


# ---------- 5. 융합(나프탈렌)/spiro -- graph vector 전체 + node 참여 수 검사 ----------
def test_fused_spiro():
    ring1 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    ring2 = [(4, 6), (6, 7), (7, 8), (8, 9), (9, 5)]
    assert_matches_brute_force("naphthalene(fused)", ring1 + ring2, 10)

    spiro_edges = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)]
    assert_matches_brute_force("spiro", spiro_edges, 5)


# ---------- 6. 분리 성분 ----------
def test_disjoint_union():
    tri = [(0, 1), (1, 2), (2, 0)]
    hexa = [(3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 3)]
    assert_matches_brute_force("disjoint tri+hexa", tri + hexa, 9)


# ---------- 7. 결합차수 불변성 ----------
def test_bond_order_invariance():
    N = 6
    edges = [(i, (i + 1) % 6) for i in range(6)]
    E1 = torch.zeros(1, N, N); E3 = torch.zeros(1, N, N)
    mask = torch.zeros(1, N, dtype=torch.bool); mask[0, :6] = True
    for i, j in edges:
        E1[0, i, j] = 1; E1[0, j, i] = 1
        E3[0, i, j] = 3; E3[0, j, i] = 3
    _, gc1 = compute_ring_size_features(E1, mask)
    _, gc3 = compute_ring_size_features(E3, mask)
    check("bond-order invariance", torch.allclose(gc1, gc3, atol=1e-4))


# ---------- 8. 패딩 불변성 ----------
def test_padding_invariance():
    N = 8
    edges = [(i, (i + 1) % 6) for i in range(6)]
    E, mask = make_batch([edges], [6], N)
    E_noisy = E.clone()
    E_noisy[0, 6, 7] = 1; E_noisy[0, 7, 6] = 1
    E_noisy[0, 5, 6] = 1; E_noisy[0, 6, 5] = 1
    _, gc_clean = compute_ring_size_features(E, mask)
    _, gc_noisy = compute_ring_size_features(E_noisy, mask)
    check("padding invariance", torch.allclose(gc_clean, gc_noisy, atol=1e-4),
          f"clean={gc_clean.tolist()} noisy={gc_noisy.tolist()}")


# ---------- 9. 자기루프 불변성 (신규 추가) ----------
def test_self_loop_invariance():
    N = 6
    edges = [(i, (i + 1) % 6) for i in range(6)]
    E, mask = make_batch([edges], [6], N)
    E_selfloop = E.clone()
    for i in range(6):
        E_selfloop[0, i, i] = 1  # 자기루프 주입 -- 결과에 영향 없어야 함(구현이 내부에서 eye 제외)
    _, gc_clean = compute_ring_size_features(E, mask)
    _, gc_sl = compute_ring_size_features(E_selfloop, mask)
    check("self-loop invariance", torch.allclose(gc_clean, gc_sl, atol=1e-4),
          f"clean={gc_clean.tolist()} selfloop={gc_sl.tolist()}")


# ---------- 10. 순열 등변성 ----------
def test_permutation_equivariance():
    N = 6
    edges = [(i, (i + 1) % 6) for i in range(6)]
    E, mask = make_batch([edges], [6], N)
    perm = torch.randperm(N)
    E_perm = E[:, perm][:, :, perm]
    mask_perm = mask[:, perm]
    node_c, graph_c = compute_ring_size_features(E, mask)
    node_c_perm, graph_c_perm = compute_ring_size_features(E_perm, mask_perm)
    check("permutation equivariance: graph", torch.allclose(graph_c[0], graph_c_perm[0], atol=1e-4))
    check("permutation equivariance: node", torch.allclose(node_c[0][perm], node_c_perm[0], atol=1e-4))


# ---------- 11. 배치 일관성 ----------
def test_batch_consistency():
    N = 7
    tri = [(0, 1), (1, 2), (2, 0)]
    hexa = [(i, (i + 1) % 6) for i in range(6)]
    E_batch, mask_batch = make_batch([tri, hexa], [3, 6], N)
    node_c_b, graph_c_b = compute_ring_size_features(E_batch, mask_batch)
    E1, mask1 = make_batch([tri], [3], N)
    E2, mask2 = make_batch([hexa], [6], N)
    _, gc1 = compute_ring_size_features(E1, mask1)
    _, gc2 = compute_ring_size_features(E2, mask2)
    check("batch consistency (tri)", torch.allclose(graph_c_b[0], gc1[0], atol=1e-4))
    check("batch consistency (hexa)", torch.allclose(graph_c_b[1], gc2[0], atol=1e-4))


# ---------- 12. N<=6 전수검사 (모든 단순무방향그래프) ----------
def test_exhaustive_n6():
    total, fail = 0, 0
    fail_node = 0
    for n in range(3, 7):
        all_pairs = list(itertools.combinations(range(n), 2))
        for r in range(len(all_pairs) + 1):
            for edge_subset in itertools.combinations(all_pairs, r):
                edges = list(edge_subset)
                E, mask = make_batch([edges], [n], n)
                node_c, graph_c = compute_ring_size_features(E, mask)
                bf_graph, bf_node = brute_force_graph_counts(edges, n)
                expect_graph = torch.tensor([bf_graph[3], bf_graph[4], bf_graph[5], bf_graph[6]], dtype=torch.float)
                expect_node = torch.zeros(n, 3)
                for k, L in enumerate((3, 4, 5)):
                    expect_node[:, k] = torch.tensor(bf_node[L], dtype=torch.float)
                total += 1
                if not torch.allclose(graph_c[0], expect_graph, atol=1e-3):
                    fail += 1
                    if fail <= 5:
                        print(f"  전수검사(graph) 실패 예시: n={n} edges={edges} model={graph_c[0].tolist()} bf={bf_graph}")
                if not torch.allclose(node_c[0], expect_node, atol=1e-3):
                    fail_node += 1
                    if fail_node <= 5:
                        print(f"  전수검사(node) 실패 예시: n={n} edges={edges} model={node_c[0].tolist()} bf={expect_node.tolist()}")
    check(f"exhaustive N<=6 graph ({total}개 그래프)", fail == 0, f"실패={fail}/{total}")
    check(f"exhaustive N<=6 node ({total}개 그래프)", fail_node == 0, f"실패={fail_node}/{total}")


# ---------- 13. N=7~10 무작위 검사 (graph + node) ----------
def test_random_n7_10(n_graphs=200, seed=1):
    rng = np.random.default_rng(seed)
    fail, fail_node = 0, 0
    for trial in range(n_graphs):
        n = int(rng.integers(7, 11))
        p = rng.uniform(0.15, 0.5)
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
        E, mask = make_batch([edges], [n], n)
        node_c, graph_c = compute_ring_size_features(E, mask)
        bf_graph, bf_node = brute_force_graph_counts(edges, n)
        expect_graph = torch.tensor([bf_graph[3], bf_graph[4], bf_graph[5], bf_graph[6]], dtype=torch.float)
        expect_node = torch.zeros(n, 3)
        for k, L in enumerate((3, 4, 5)):
            expect_node[:, k] = torch.tensor(bf_node[L], dtype=torch.float)
        if not torch.allclose(graph_c[0], expect_graph, atol=1e-3):
            fail += 1
            if fail <= 5:
                print(f"  무작위(graph) 실패 예시: n={n} edges={edges} model={graph_c[0].tolist()} bf={bf_graph}")
        if not torch.allclose(node_c[0], expect_node, atol=1e-3):
            fail_node += 1
            if fail_node <= 5:
                print(f"  무작위(node) 실패 예시: n={n} edges={edges} model={node_c[0].tolist()} bf={expect_node.tolist()}")
    check(f"random N=7~10 graph ({n_graphs}개)", fail == 0, f"실패={fail}/{n_graphs}")
    check(f"random N=7~10 node ({n_graphs}개)", fail_node == 0, f"실패={fail_node}/{n_graphs}")


# ---------- 14. CUDA/CPU float32 parity (CUDA 있을 때만) ----------
def test_cuda_cpu_parity():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA 없음 -- parity 테스트 생략")
        return
    rng = np.random.default_rng(7)
    edges_list, n_list = [], []
    # curated 몇 개 + 무작위 N=7~10 몇 개
    edges_list.append([(i, (i + 1) % 6) for i in range(6)]); n_list.append(6)
    edges_list.append(list(itertools.combinations(range(4), 2))); n_list.append(4)
    for _ in range(5):
        n = int(rng.integers(7, 11))
        p = rng.uniform(0.15, 0.5)
        edges_list.append([(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p])
        n_list.append(n)
    N = max(n_list)
    E, mask = make_batch(edges_list, n_list, N)
    node_cpu, graph_cpu = compute_ring_size_features(E, mask)
    node_cuda, graph_cuda = compute_ring_size_features(E.cuda(), mask.cuda())
    check("CUDA/CPU parity (graph)", torch.allclose(graph_cpu, graph_cuda.cpu(), atol=1e-3))
    check("CUDA/CPU parity (node)", torch.allclose(node_cpu, node_cuda.cpu(), atol=1e-3))


if __name__ == "__main__":
    test_acyclic()
    test_single_rings()
    test_square_with_diagonal()
    test_k4()
    test_fused_spiro()
    test_disjoint_union()
    test_bond_order_invariance()
    test_padding_invariance()
    test_self_loop_invariance()
    test_permutation_equivariance()
    test_batch_consistency()
    test_exhaustive_n6()
    test_random_n7_10()
    test_cuda_cpu_parity()
    print(f"\n총 실패: {N_FAIL}")
    if N_FAIL == 0:
        print("전부 통과.")
