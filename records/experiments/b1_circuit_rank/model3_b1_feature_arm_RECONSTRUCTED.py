"""[RECONSTRUCTED, not a training-time snapshot]
records/experiments/b1_circuit_rank/model3_b1_feature_arm_RECONSTRUCTED.py

molecule-AI(메인 트렁크)의 현재 model3.py에서 [C-1](cycle_proj/_ring_cycle_features/
_scale_cycle_features 및 forward의 cycle_struct 계산+합산)만 손으로 제거해
B-1 feature arm(circuit_rank 포함, cycle_proj 없음) 상태를 복원한 파일이다.

**checkpoints_c4_b1_fixed를 실제로 학습시킨 코드와 byte-identical하지 않다** --
메인 트렁크는 B-1 학습 이후에도 계속 진화했고 그 시점의 독립 스냅샷은 보존되지
않았다(review 없이 트렁크에 직접 누적하는 워크플로우였음). 2026-09-21에
model3.py(당시 SHA256은 devlog.md 참조)에서 [C-1] 태그가 붙은 블록만 제거해
재구성했으며, [B-1]/[A-1] 이전 코드와 골격이 동일함은 grep으로 확인함.

대조군(B-1 control arm)은 반대로 **진짜 학습 당시 스냅샷이 그대로 보존**되어
있다 -- molecule-AI-control/model3.py는 C-1 시점까지 수정되지 않아
records/experiments/proposal_a/c1_experiment/model3_c1_control_snapshot.py와
SHA256이 완전히 동일하다(efef1ee4474f2d8867358b997b0c0a3a6cb02bf79209d5fde8b3b3d5e42523ad).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch

# 2026-07-27: absorbing([MASK]) noise -> marginal-transition noise로 교체.
# 근거: Vignac et al., "DiGress: Discrete Denoising diffusion for graph
# generation", ICLR 2023 (arXiv:2209.14734), Sec 4.1 / Theorem 4.1.
# t=T(완전 노이즈)에서도 그래프가 학습 데이터의 marginal 분포(대부분 no-bond)로
# 수렴하도록 설계 -> 모델이 "맥락 없을 때 기본값은 no-bond"를 노이즈 구조 자체에서
# 배움. MASK 토큰 방식은 이 사전 정보가 없어 맥락 부족 시 결합 과다예측 편향 발생
# (devlog.md 2026-07-26/27 진단 참조). K_X/K_E에 MASK 클래스 불필요 -> 원복.
#
# 2026-08-03: 방향족(aromatic) 클래스 제거. ground-truth 재구성 검증에서 방향족을
# 독립 클래스로 두는 표현이 이론적 validity 상한을 80.2%로 제한함을 실측 확인
# (전역 속성인 방향족성을 결합별 독립 예측으로는 일관되게 맞추기 어려움).
# preprocess.py에서 kekulize로 방향족 -> 명시적 단일/이중 교대 변환 후 재전처리.
# K_X: 0(padding), 1~9(C, O, N, S, P, F, Cl, Br, I) -> Total 10
# K_E: 0(no bond/padding), 1~3(Single, Double, Triple) -> Total 4
MAX_ATOMS = 50
K_X = 10
K_E = 4
T_STEPS = 150

ATOMIC_NUM_TO_CLS = {
    6: 1,  # C
    8: 2,  # O
    7: 3,  # N
    16: 4, # S
    15: 5, # P
    9: 6,  # F
    17: 7, # Cl
    35: 8, # Br
    53: 9  # I
}

# [R-5] 2026-08-28: DiGress Appendix B.2 구조 feature 중 valency/분자량 복원.
# 2026-08-03 "명확한 차이 없음 REJECT" 판정은 review.md §2에 따라 검정력 없는
# 표본(7/84 vs 8/84)에서 나온 것이라 무효 -- 다시 검증.
# 표준 원자량(RDKit GetAtomicWeight 실측, g/mol). index 0(padding)은 0.
_ATOMIC_WEIGHT = {
    6: 12.011, 8: 15.999, 7: 14.007, 16: 32.067, 15: 30.974,
    9: 18.998, 17: 35.453, 35: 79.904, 53: 126.904,
}

NODE_DIM = 256
EDGE_DIM = 128
N_HEADS = 8
N_LAYERS = 6
FF_DIM = 512
T_DIM = 128
# [G-5] 2026-09-13: AdaLN류 조건 재주입에서 GTLayer가 참조할 수 있도록
# 모듈 레벨로 승격(원래 ScoreNetwork.__init__ 지역변수였음).
# [A-1] 2026-09-17: review16.md 권고로 2차원(HOMO/LUMO)에서 확장.
# data/processed_ext7의 batch.p(12차원, stats.json 순서 HOMO,LUMO,GAP,
# E_total,Dipole,LogP,TPSA,HBD,HBA,RotBonds,AromaticRings,SAscore)에서
# 아래 인덱스만 조건으로 사용. GAP/E_total/Dipole 제외는 review9.md
# 그대로(선형종속/조성만으로 R²=1.0/2D로는 GNN R²=0.303). QED는 넣지
# 않음 -- review16.md: 나머지의 (비선형) 결정론적 함수라 GAP과 같은
# 부류의 함정. HBD는 TPSA/HBA와 상관 0.72~0.76으로 중복이라 제외,
# HBA만 유지(review16.md §3).
# [A-1 수정] 2026-09-17(야간): SAscore 제거(8->7차원). review13.md/
# review_followup_20260912.md §2가 이미 "SAscore는 타겟을 지정하는
# 물성이 아니라 낮을수록 좋은 최적화 대상이라 조건부(conditioning)가
# 아니라 RL/loss로 가야 한다"고 판정하고 "다음 프로젝트로" 보류해뒀던
# 걸(TODO.md 아이디어 표), 이번 물성 확장 때 그 이력을 다시 안 찾아보고
# 조건 후보에 그대로 넣는 실수를 함(review_followup_20260917.md에
# 이 맥락을 안 담아서 review16.md도 못 잡아낸 것 — 상관관계 중복만
# 스크리닝했지 "타겟팅 가능 물성 vs 최적화 대상"이라는 축 자체를
# 못 받았음). HOMO/LUMO/LogP/TPSA/HBA/RotBonds/AromaticRings는 전부
# 신약 설계에서 실제로 특정 값을 조준하는 게 말이 되는 물성이라 유지.
# SAscore는 conditioning에서 빼고 향후 RL 단계로 재분류(TODO.md).
COND_PROP_INDICES = [0, 1, 5, 6, 8, 9, 10]
COND_PROP_NAMES = ["HOMO", "LUMO", "LogP", "TPSA", "HBA", "RotBonds", "AromaticRings"]
COND_DIM = len(COND_PROP_INDICES)


def _cosine_ac(T, s = 0.008):
    steps = torch.arange(T+1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    return (f / f[0]).float()[1:]


# [R-6] 2026-09-04: docs/review7.md [Z-4] 1순위 방침 -- Laplacian 고유값 0의
# 중복도 = 연결 성분 개수이지만, 매 step [N,N] 고유분해(DiGress 원논문 방식,
# 2순위)를 쓰지 않고 인접행렬의 불리언 전이적 폐포(transitive closure)로
# 대체한다. N<=50이므로 ceil(log2(N))<=6번의 (B,N,N) 행렬곱이면 충분 -- 사실상
# 비용 0. 학습 가능한 파라미터 없이 매 forward 호출 시 현재 E_t에서 즉석 계산
# (R-5와 동일 원칙, 고정 통계 아님).
def _connectivity_features(E_t, node_mask):
    """반환: n_components (B,) 그래프별 연결 성분 개수(실제 노드만 집계),
    comp_size_ratio (B,N) 노드별 자기 성분 크기/실제 노드 수,
    circuit_rank (B,) 독립 고리(cycle) 개수.

    [B-1] 2026-09-18: review13.md가 원래 진단한 "국소 메시지 패싱이
    원거리 연결성/고리 크기를 못 본다"는 한계를 겨냥해, R-6(연결
    성분)와 같은 원칙(학습 파라미터 없이 매 forward 즉석 계산)으로
    순환 랭크(circuit rank = E - V + C, 그래프 이론의 독립 사이클 수)
    를 추가. review18.md §4가 지적한 두 함정을 피해서 구현:
    (1) self-loop 포함 전(reachability용으로 위에서 eye를 더하기 전)
        원본 엣지만 세야 함 -- 안 그러면 실제 노드 수(V)만큼 값이
        부풀려짐.
    (2) 결합 차수(단일/이중/삼중)가 아니라 결합 "존재 여부"만 세야
        함 -- 이중/삼중결합을 엣지 여러 개로 세면 고리가 없는
        분자도 순환 랭크가 잘못 양수로 나옴.
    """
    B, N, _ = E_t.shape
    eye = torch.eye(N, dtype=torch.bool, device=E_t.device).unsqueeze(0)
    both_real = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)

    edge_exists = (E_t > 0) & both_real  # self-loop 추가 전, 결합 차수 무시 -- [B-1] 순환 랭크용 원본

    adj = edge_exists | eye  # 자기 자신 포함 (실/패딩 공통) -- 패딩 행은 이후에도 자기자신만 유지
    adj = adj & (both_real | eye)  # 실-패딩 교차 결합 제거, 패딩 자기 루프는 유지

    reach = adj.float()
    n_iter = max(1, math.ceil(math.log2(max(N, 2))))
    for _ in range(n_iter):
        reach = (reach @ reach) > 0
        reach = reach.float()
    reach_b = reach > 0  # (B,N,N)

    idx = torch.arange(N, device=E_t.device)
    canon = reach.argmax(dim=-1)  # 행별 첫 True 인덱스 (성분 내 최솟값 노드)
    is_repr = (canon == idx.unsqueeze(0)) & node_mask
    n_components = is_repr.float().sum(-1)  # (B,)

    comp_size = (reach_b & node_mask.unsqueeze(1)).float().sum(-1)  # (B,N)
    n_real = node_mask.float().sum(-1, keepdim=True).clamp(min=1.0)
    comp_size_ratio = comp_size / n_real

    # [B-1] circuit_rank = E - V + C. 무방향 그래프라 인접행렬이 대칭 ->
    # 상삼각만 세야 엣지를 두 번 안 셈(패딩 고립 노드는 E,V,C에 +0,+1,+1로
    # 순변화 0이라 마스킹 걱정 없음, review18.md §4).
    triu = torch.triu(torch.ones(N, N, dtype=torch.bool, device=E_t.device), diagonal=1).unsqueeze(0)
    e_count = (edge_exists & triu).float().sum((-1, -2))  # (B,)
    v_count = node_mask.float().sum(-1)  # (B,)
    circuit_rank = e_count - v_count + n_components  # (B,)

    return n_components, comp_size_ratio, circuit_rank


class MarginalNoiseSchedule(nn.Module):
    """Marginal-transition discrete noise schedule (DiGress Sec 4.1).

    Q^t = alpha^t I + beta^t * 1 m'  (m = 학습 데이터 marginal 클래스 분포)
    q(x^t | x^0=c) = ac_t * onehot(c) + (1-ac_t) * m
    Posterior q(x^{t-1}=i | x^t=j, x^0=c)는 Bayes rule로 직접 유도(closed form):
      propto [alpha_t*[i==j] + beta_t*m_j] * [ac_tm1*[i==c] + beta_tm1*m_i]
    (m_j는 관측값 j=x_t에서 계산한 스칼라. i에 대해 변하는 벡터가 아님에 주의 —
    이 부분 인덱싱 실수로 1차 로컬 검증에서 반증됐다가 수정 후 재검증됨.)
    이후 x0_probs(네트워크 예측)로 c에 대해 mixing (DiGress Eq.5).
    """

    def __init__(self, m_X, m_E, T = T_STEPS):
        super().__init__()
        ac = _cosine_ac(T)
        self.register_buffer("ac", ac)
        self.register_buffer("ac_prev", torch.cat([torch.ones(1), ac[:-1]]))
        self.register_buffer("m_X", m_X)  # [K_X], sums to 1, index 0(padding)=0
        self.register_buffer("m_E", m_E)  # [K_E], sums to 1
        self.T = T

    def q_sample(self, x0, t, m):
        shape = x0.shape
        B = t.shape[0]
        K = m.shape[-1]
        ac_t = self.ac[t-1].view(B, *([1] * (len(shape) - 1)))
        x0c = x0.clamp(min=0, max=K - 1)
        x0_onehot = F.one_hot(x0c, K).float()
        probs = ac_t.unsqueeze(-1) * x0_onehot + (1.0 - ac_t).unsqueeze(-1) * m
        probs = probs.clamp(min=0.0)
        probs = probs / probs.sum(-1, keepdim=True).clamp(min=1e-8)
        x_t = torch.multinomial(probs.reshape(-1, K), 1).view(shape)
        return x_t

    def posterior_sample(self, x_t, x0_probs, t, m):
        B = t.shape[0]
        shape = x_t.shape
        K = m.shape[-1]
        view = (B,) + (1,) * (len(shape) - 1)

        ac_t = self.ac[t-1].view(view)
        ac_tm1 = self.ac_prev[t-1].view(view)
        alpha_t = (ac_t / ac_tm1.clamp(min=1e-8)).clamp(max=1.0)
        beta_t = 1.0 - alpha_t

        # [Q^t]_{ij} = alpha_t*delta(i,j) + beta_t*m_j -- m_j는 관측값 j=x_t에서
        # 계산한 스칼라(i에 대해 상수), 벡터 m 전체가 아님.
        xt_clamped = x_t.clamp(min=0, max=K - 1)
        xt_onehot = F.one_hot(xt_clamped, K).float()
        m_xt = m[xt_clamped]
        term1 = (alpha_t.unsqueeze(-1) * xt_onehot + beta_t.unsqueeze(-1) * m_xt.unsqueeze(-1)).unsqueeze(-2)

        # [Q̄^{t-1}]_{ci} = ac_tm1*delta(c,i) + beta_tm1*m_i -- 여기는 i에 대해
        # 변하는 게 맞음 (i가 목적 상태이므로).
        eye = torch.eye(K, device=x_t.device, dtype=torch.float)
        beta_tm1 = 1.0 - ac_tm1
        term2 = ac_tm1.unsqueeze(-1).unsqueeze(-1) * eye + beta_tm1.unsqueeze(-1).unsqueeze(-1) * m.view(*([1]*len(view)), 1, K)

        unnorm = (term1 * term2).clamp(min=0.0)
        norm_c = unnorm.sum(-1, keepdim=True).clamp(min=1e-8)
        post_given_c = unnorm / norm_c

        result = (x0_probs.unsqueeze(-1) * post_given_c).sum(-2)
        # 2026-09-03 (Y-1 진단 중 발견): x0_probs가 모든 실제 클래스(c=1..K-1)에서
        # 0인 위치(예: 패딩) -- 원래 sample()은 패딩 없는 입력에서만 쓰여
        # 노출된 적 없던 경로 -- 에서는 result 행 전체가 정확히 0이 될 수 있고,
        # multinomial이 "sum of probabilities <= 0"으로 CUDA device-side assert를
        # 던져 이후 모든 연산이 cublas 오류로 오염됨. 실제 분포에는 영향 없는
        # 작은 균등 바닥을 더해 이런 행도 항상 유효한 분포가 되도록 함
        # (해당 위치는 호출부에서 어차피 node_mask로 덮어써짐).
        result = result.clamp(min=0.0) + 1e-8
        result = result / result.sum(-1, keepdim=True)

        stoch = torch.multinomial(result.reshape(-1, K), 1).view(shape)
        determ = x0_probs.argmax(dim=-1)
        t_mask = (t == 1).view(view).expand(shape)
        return torch.where(t_mask, determ, stoch)


class TimeEmbedding(nn.Module):

    def __init__(self, dim = T_DIM):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / (half - 1)
        )
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)


class GTLayer(nn.Module):

    def __init__(self):
        super().__init__()
        assert NODE_DIM % N_HEADS == 0

        self.qkv = nn.Linear(NODE_DIM, NODE_DIM * 3, bias=False)
        self.e_bias = nn.Linear(EDGE_DIM, N_HEADS)
        self.attn_out = nn.Linear(NODE_DIM, NODE_DIM)
        self.norm1 = nn.LayerNorm(NODE_DIM)
        self.ff = nn.Sequential(
            nn.Linear(NODE_DIM, FF_DIM), nn.GELU(), nn.Linear(FF_DIM, NODE_DIM)
        )
        self.norm2 = nn.LayerNorm(NODE_DIM)
        # h_i+h_j만으로는 원자 쌍의 상호작용(어느 쌍이 결합하는지)을 구별하기 어려움
        # h_i*h_j를 추가해 쌍별 상호작용 신호를 명시적으로 제공 (대칭 유지)
        self.edge_ff = nn.Sequential(
            nn.Linear(NODE_DIM * 2 + EDGE_DIM, FF_DIM),
            nn.GELU(),
            nn.Linear(FF_DIM, EDGE_DIM)
        )
        self.enorm = nn.LayerNorm(EDGE_DIM)

        # [G-5] 2026-09-13: AdaLN류 조건 재주입 -- 지금까지 cond는 입력
        # 단(ScoreNetwork.forward)에서 한 번만 h에 더해지고, 이후 6개
        # GTLayer를 통과하며 다시 주입되지 않았다(review13 이후 논의,
        # conditioning 정밀도 격차의 원인 후보). 매 레이어 끝에서 cond로
        # scale/shift를 예측해 h를 다시 변조. zero-init으로 학습 시작
        # 시점엔 항등변환(gamma=0,beta=0 -> h 그대로) -- 기존
        # checkpoints_c4/best.pt에서 이어받아도 초기 성능 저하 없음.
        # [A-1] 2026-09-17: 입력이 (마스크된 값, 마스크) concat이라 2*COND_DIM.
        # 입력 차원이 늘어도 zero-init은 그대로 출력을 0으로 만들어 안전
        # (review16.md §4에서 확인됨).
        self.cond_mod = nn.Linear(2 * COND_DIM, NODE_DIM * 2)
        nn.init.zeros_(self.cond_mod.weight)
        nn.init.zeros_(self.cond_mod.bias)

    def forward(self, h, e, pad_mask, cond=None):
        B, N, _ = h.shape
        nh = N_HEADS
        dh = NODE_DIM // N_HEADS

        Q, K, V = self.qkv(h).chunk(3, dim=-1)
        Q = Q.view(B, N, nh, dh).transpose(1, 2) # (B, nh, N, dh)
        K = K.view(B, N, nh, dh).transpose(1, 2)
        V = V.view(B, N, nh, dh).transpose(1, 2)

        scores = torch.einsum("bhid,bhjd->bhij", Q, K) / math.sqrt(dh)  # (B, nh, N, N)
        scores = scores + self.e_bias(e).permute(0, 3, 1, 2)
        scores = scores.masked_fill(pad_mask[:, None, None, :], float('-inf'))

        attn = scores.softmax(dim=-1)
        out  = torch.einsum("bhij,bhjd->bhid", attn, V)
        out = out.transpose(1, 2).reshape(B, N, -1)
        h = self.norm1(h + self.attn_out(out))
        h = self.norm2(h + self.ff(h))

        # [G-5] AdaLN류 재주입 -- 노드·엣지 업데이트 둘 다에 영향(엣지는
        # 아래에서 이 h로부터 hi/hj를 만들므로 자동 전파).
        if cond is not None:
            gamma, beta = self.cond_mod(cond).chunk(2, dim=-1)
            h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        hi = h.unsqueeze(2).expand(-1, -1, N, -1) # (B, N, N, D)
        hj = h.unsqueeze(1).expand(-1, N, -1, -1) # (B, N, N, D)
        e = self.enorm(e + self.edge_ff(torch.cat([hi + hj, hi * hj, e], dim=-1)))

        h = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        e_pad = pad_mask.unsqueeze(1) | pad_mask.unsqueeze(2)
        e = e.masked_fill(e_pad.unsqueeze(-1), 0.0)

        return h, e


class ScoreNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.x_embed = nn.Embedding(K_X, NODE_DIM)
        self.e_embed = nn.Embedding(K_E, EDGE_DIM)
        self.t_embed = TimeEmbedding(T_DIM)
        self.t_proj  = nn.Linear(T_DIM, NODE_DIM)
        self.layers  = nn.ModuleList([GTLayer() for _ in range(N_LAYERS)])

        # [R-5] 매 denoising step마다 현재(노이즈) 그래프 상태로부터 재계산되는
        # 구조 feature: 현재 원자가(E_t 행 합, 결합 차수 그대로) + 원자량(X_t 기준).
        # 학습 데이터 통계가 아닌 매 forward 호출 시 즉석 계산 -- 고정 입력 아님.
        weight_table = torch.zeros(K_X)
        for _anum, _cls in ATOMIC_NUM_TO_CLS.items():
            weight_table[_cls] = _ATOMIC_WEIGHT[_anum]
        self.register_buffer("atomic_weight", weight_table)
        self.struct_proj = nn.Linear(2, NODE_DIM)

        # [R-6] 연결성 feature 투영 (n_components 그래프 스칼라를 노드마다
        # broadcast + comp_size_ratio 노드별 값). review7.md [Z-4] 1순위.
        # [B-1] 2026-09-18: circuit_rank(순환 랭크, 그래프 스칼라 broadcast)
        # 추가로 3차원 -> review13.md가 진단한 "원거리 연결성/고리 크기를
        # 못 본다"는 한계를 R-6와 같은 원칙(학습 파라미터 없는 즉석 계산)
        # 으로 확장. review18.md §4/§5 검증 거침.
        self.conn_proj = nn.Linear(3, NODE_DIM)

        self.cond_proj = nn.Linear(2 * COND_DIM, NODE_DIM)

        # 원자: padding(0) 제외한 K_X-1개 실제 원소 예측.
        # 결합: MASK 클래스가 없으므로 no-bond(0) 포함 전체 K_E개 예측.
        self.x_head  = nn.Sequential(nn.LayerNorm(NODE_DIM), nn.Linear(NODE_DIM, K_X - 1))
        self.e_head  = nn.Sequential(nn.LayerNorm(EDGE_DIM), nn.Linear(EDGE_DIM, K_E))

        # [G-6] 2026-09-13: 명시적 물성 loss(사용자 제안, review_followup_
        # 20260913.md 이후 논의) -- 지금까지 conditioning은 "이 물성값을
        # 보이며 정답 분자를 복원하라"는 간접(재구성) 손실뿐이었다.
        # prop_head는 최종 hidden h(전체 원자 평균 pool)에서 HOMO/LUMO를
        # 직접 예측 -- backbone이 물성을 실제로 복원 가능한 형태로 담고
        # 있는지를 별도 supervised 신호로 강제한다. GNN 서로게이트(RDKit
        # 필요)와 달리 h가 이미 미분 가능한 표현이라 곧바로 MSE로 학습.
        self.prop_head = nn.Sequential(
            nn.LayerNorm(NODE_DIM), nn.Linear(NODE_DIM, NODE_DIM), nn.GELU(),
            nn.Linear(NODE_DIM, COND_DIM)
        )

    def forward(self, X_t, E_t, node_mask, t, cond=None, cond_mask=None):
        # [R-5] 현재 원자가(no-bond=0..triple=3, 값 자체가 결합 차수) + 원자량.
        # 스케일이 서로/다른 입력과 크게 달라 대략적인 크기로만 정규화(학습 가능한
        # struct_proj가 나머지를 흡수).
        cur_valence = E_t.float().sum(dim=-1, keepdim=True) / 4.0
        atomic_weight = self.atomic_weight[X_t.clamp(min=0, max=K_X - 1)].unsqueeze(-1) / 100.0
        struct_feat = self.struct_proj(torch.cat([cur_valence, atomic_weight], dim=-1))

        # [R-6] 연결 성분 개수/노드별 성분 크기 비율 -- 학습 불가 파라미터 없이
        # 현재 E_t에서 즉석 계산 (review7.md [Z-4]).
        # [B-1] circuit_rank(독립 고리 개수)도 같은 방식으로 계산해 추가.
        n_components, comp_size_ratio, circuit_rank = _connectivity_features(E_t, node_mask)
        n_comp_bcast = (n_components / 10.0).unsqueeze(1).expand(-1, X_t.shape[1])
        circuit_rank_bcast = (circuit_rank / 5.0).unsqueeze(1).expand(-1, X_t.shape[1])
        conn_feat = torch.stack([n_comp_bcast, comp_size_ratio, circuit_rank_bcast], dim=-1)
        conn_struct = self.conn_proj(conn_feat)

        # [A-1] 2026-09-17: cond가 안 주어지면 "완전 무조건"(값=0,마스크=0).
        # cond는 주어졌는데 cond_mask가 없으면 기존 호출부와의 하위호환으로
        # "전부 안다"(마스크=1)로 취급. 마스크된 위치의 값은 반드시 0으로
        # 지워서(cond*cond_mask) 모델이 마스크를 무시하고 실제 값을 새는
        # 경로로 학습하지 못하게 한다.
        B = X_t.shape[0]
        if cond is None:
            cond = torch.zeros(B, COND_DIM, device=X_t.device)
            cond_mask = torch.zeros(B, COND_DIM, device=X_t.device)
        elif cond_mask is None:
            cond_mask = torch.ones_like(cond)
        cond_full = torch.cat([cond * cond_mask, cond_mask], dim=-1)
        cond_struct = self.cond_proj(cond_full).unsqueeze(1).expand(-1, X_t.shape[1], -1)

        h   = self.x_embed(X_t) + self.t_proj(self.t_embed(t)).unsqueeze(1) + struct_feat + conn_struct + cond_struct
        e   = self.e_embed(E_t)
        pad = ~node_mask

        for layer in self.layers:
            h, e = layer(h, e, pad, cond_full)

        x_logits = self.x_head(h)
        e_logits = self.e_head(e)
        e_logits = (e_logits + e_logits.transpose(1, 2)) / 2

        # [G-6] 실제 원자에 대해서만 평균 pool(패딩 제외) 후 물성 예측.
        node_mask_f = node_mask.unsqueeze(-1).float()
        h_pooled = (h * node_mask_f).sum(dim=1) / node_mask_f.sum(dim=1).clamp(min=1.0)
        prop_pred = self.prop_head(h_pooled)

        return x_logits, e_logits, prop_pred


class MoleculeGraphDiffusion(nn.Module):

    # [R-4] 2026-08-27: docs/review.md 지적 -- 0.2의 도입 근거("엣지 수가 N배라
    # gradient 독점")가 실제로는 성립하지 않음 (F.cross_entropy가 이미 mean
    # reduction이라 쌍 개수로 인한 독점이 없음). DiGress 원논문은 반대로 λ=5를
    # 사용. 08-26 스모크런(diagnostic_fixed_t)에서 t=140(맥락 거의 없음)의
    # bond_acc가 EMA/RAW/두 epoch 전부 0.000으로 나온 것이 "결합 예측이 병목인데
    # 정확히 거기에 가장 적은 gradient 비중을 주고 있었다"는 이 가설과 일치.
    # T-x가 H(m_X)=0.8258 위로 되올라가면 과한 신호이므로 2.0으로 완화할 것.
    # [C-3] 2026-09-05: 학습 중 확률 COND_DROPOUT_P로 조건을 null_cond로
    # 대체(classifier-free guidance 준비, review8.md [C-4] 지시). a_bin
    # 스모크 테스트와 향후 물성 conditioning이 재학습 없이 이 경로를 공유.
    # [G-4] 2026-09-13: 0.15->0.10 파인튜닝 시도 -- GPU를 [G-5](AdaLN)에
    # 넘기려고 학습 중단, 미채택. **원복 완료**: G-5 실행 중 이 값이
    # 0.10으로 남아있던 걸 뒤늦게 발견 -- AdaLN 재주입 효과와 dropout
    # 변화가 뒤섞여 결과를 해석할 수 없게 되는 오염이었음. 0.15(원본,
    # checkpoints_c4/ 학습 시 값)로 되돌리고 G-5를 재시작함.
    # [A-1] 2026-09-17: 물성이 여러 개로 늘면서 review16.md §4가 지적한 문제
    # -- 속성별 독립 드롭아웃만 쓰면 "전부 동시에 null"(guidance의 uncond
    # 분기가 의존하는 기준점) 노출 확률이 COND_DROPOUT_P_PARTIAL^COND_DIM
    # 으로 지수적으로 사라진다(COND_DIM=7이어도 0.15^7 ≈ 1.7e-6로 여전히
    # 희귀). 그래서 2단계로 나눈다:
    # (1) 먼저 확률 COND_DROPOUT_P_FULL(기존 0.15 그대로)로 전체를 한
    # 번에 null 처리해 uncond 분기를 여전히 충분히 학습시키고,
    # (2) 나머지 경우에만 속성별 독립 드롭아웃(COND_DROPOUT_P_PARTIAL)을
    # 적용해 부분조건 조합을 학습시킨다. 이러면 완전-null 노출 빈도가
    # 물성 개수와 무관하게 항상 COND_DROPOUT_P_FULL로 유지된다.
    COND_DROPOUT_P_FULL = 0.15
    COND_DROPOUT_P_PARTIAL = 0.2

    # [G-6] 2026-09-13: 명시적 물성 loss 가중치/적용 범위.
    # - PROP_LOSS_WEIGHT: 재구성 loss(loss_x+loss_e)를 지배하지 않도록
    #   작게 시작. 구조 품질을 해치면서 물성만 맞추는 shortcut(reward
    #   hacking과 유사한 위험)을 피하려는 목적.
    # - PROP_LOSS_T_MAX: t가 크면(노이즈 심함) 그래프가 거의 무작위라
    #   물성 예측 자체가 무의미 -- 노이즈가 적은 구간(t 작음)에서만 적용.
    #   diagnostic_fixed_t가 보는 격자(10/40/75/110) 중 t=40까지 포함되게
    #   여유를 둠.
    PROP_LOSS_WEIGHT = 0.1
    PROP_LOSS_T_MAX = 50

    def __init__(self, m_X, m_E, T = T_STEPS, edge_loss_coeff = 5.0, class_weights = None, edge_class_weights = None):
        super().__init__()
        self.schedule = MarginalNoiseSchedule(m_X, m_E, T)
        self.net = ScoreNetwork()
        self.T = T
        self.edge_loss_coeff = edge_loss_coeff
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float))
        else:
            self.register_buffer("class_weights", torch.ones(K_X - 1, dtype=torch.float))

        if edge_class_weights is not None:
            self.register_buffer("edge_class_weights", torch.tensor(edge_class_weights, dtype=torch.float))
        else:
            self.register_buffer("edge_class_weights", torch.ones(K_E, dtype=torch.float))

    def forward(self, batch):
        X0, E0, node_mask = self._to_dense(batch)
        B, N = X0.shape
        device = X0.device

        t = torch.randint(1, self.T + 1, (B,), device=device)

        # Marginal transition noise 주입
        X_t = self.schedule.q_sample(X0, t, self.schedule.m_X)
        E_t = self._sym(self.schedule.q_sample(E0, t, self.schedule.m_E))

        # 패딩 영역 복원 (패딩된 위치는 항상 0 유지)
        X_t = torch.where(node_mask, X_t, torch.zeros_like(X_t))
        e_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        E_t = torch.where(e_mask, E_t, torch.zeros_like(E_t))

        # [A-1] 물성 COND_DIM개 조건 + CFG 학습용 2단계 랜덤 드롭(전체/속성별).
        cond = self._extract_property_cond(batch, B, device)
        if self.training:
            full_drop = torch.rand(B, device=device) < self.COND_DROPOUT_P_FULL
            partial_drop = torch.rand(B, COND_DIM, device=device) < self.COND_DROPOUT_P_PARTIAL
            cond_mask = (~partial_drop).float()
            cond_mask[full_drop] = 0.0
        else:
            full_drop = torch.zeros(B, dtype=torch.bool, device=device)
            cond_mask = torch.ones(B, COND_DIM, device=device)

        x_logits, e_logits, prop_pred = self.net(X_t, E_t, node_mask, t, cond, cond_mask)

        # 원자: 실제 클래스 1..9 -> head 출력 0..8로 매핑
        nm = node_mask.view(-1)
        loss_x = F.cross_entropy(
            x_logits.view(-1, K_X - 1)[nm],
            (X0.view(-1)[nm] - 1).clamp(min=0),
            weight=self.class_weights
        )

        triu = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        em = (node_mask.unsqueeze(2) & node_mask.unsqueeze(1)) & triu.unsqueeze(0)
        loss_e = F.cross_entropy(
            e_logits.view(-1, K_E)[em.view(-1)],
            E0.view(-1)[em.view(-1)],
            weight=self.edge_class_weights
        )

        # [G-6] 명시적 물성 loss -- 재구성 loss(간접)와 달리 h가 실제로
        # 물성을 복원 가능한 형태로 담고 있는지를 직접 감독한다.
        # 적용 대상: (1) 이번 스텝에서 조건이 전부 null로 드롭되지 않은
        # 예시만(완전히 드롭된 예시는 모델이 애초에 타겟을 하나도 모르므로
        # 벌점이 무의미하고 CFG의 무조건부 학습과 충돌함) -- [A-1] 이후
        # 속성 일부만 드롭된 예시는 그대로 포함한다(숨겨진 속성까지
        # 맞히도록 강제하는 게 오히려 backbone이 구조-물성 상관을 잘
        # 담게 만드는 유효한 신호), (2) 노이즈가 적은 t만(PROP_LOSS_T_MAX).
        prop_target = self._extract_property_cond(batch, B, device)
        gate = (~full_drop) & (t <= self.PROP_LOSS_T_MAX) if self.training else (t <= self.PROP_LOSS_T_MAX)
        if gate.any():
            loss_prop = F.mse_loss(prop_pred[gate], prop_target[gate])
        else:
            loss_prop = torch.zeros((), device=device)

        loss = loss_x + self.edge_loss_coeff * loss_e + self.PROP_LOSS_WEIGHT * loss_prop
        return {
            "loss": loss,
            "loss_x": loss_x,
            "loss_e": loss_e,
            "loss_prop": loss_prop,
        }

    @staticmethod
    def _extract_heavy_a_bin(batch, B, device):
        """[C-3] dataset2.py의 a_bin(10차원, 순서 C,H,O,N,S,P,F,Cl,Br,I)에서
        H를 제외한 heavy 9종만 뽑아 ATOMIC_NUM_TO_CLS(class 1..9) 순서로 정렬.
        스모크 테스트 전용, [C-4] 이후로는 _extract_property_cond를 쓴다."""
        a_bin10 = batch.a_bin.view(B, -1)
        idx = torch.tensor([0, 2, 3, 4, 5, 6, 7, 8, 9], device=device)
        return a_bin10[:, idx]

    @staticmethod
    def _extract_property_cond(batch, B, device):
        """[A-1] dataset2.py의 batch.p(12차원 정규화값, data/processed_ext7
        stats.json 순서 HOMO,LUMO,GAP,E_total,Dipole,LogP,TPSA,HBD,HBA,
        RotBonds,AromaticRings,SAscore)에서 COND_PROP_INDICES(현재 7개:
        HOMO,LUMO,LogP,TPSA,HBA,RotBonds,AromaticRings)만 뽑는다.
        GAP/E_total/Dipole 제외는 review9.md 그대로, QED 미포함/HBD 제외는
        review16.md §3, SAscore 제외는 위 COND_PROP_INDICES 주석 참고
        (타겟팅 물성이 아니라 최적화 대상이라 RL로 재분류)."""
        p = batch.p.view(B, -1)
        return p[:, COND_PROP_INDICES]

    @torch.no_grad()
    def sample(self, n, n_atoms, device, cond=None, cond_mask=None, guidance_w=None, step_cb=None):
        """[D-3] review10.md §4 Q1 지시로 classifier-free guidance 추가.
        guidance_w가 None이면 기존과 동일(순수 조건부 forward 1회, w=1과
        동치). guidance_w가 주어지면(cond 필수) 매 step에서 조건부/무조건부
        로짓을 함께 계산해 로짓 공간에서 외삽한다:
            guided = uncond + w * (cond - uncond)
        w=0 -> 순수 무조건부, w=1 -> 순수 조건부(위와 동일), w>1 -> 조건 신호
        증폭. 배치를 2배로 합쳐 한 번의 forward로 처리(속도 손실 최소화).

        [A-1] 2026-09-17: cond_mask(n,COND_DIM, 0/1)로 부분 조건 지원 --
        1이면 그 속성을 안다(값 사용), 0이면 모른다(값 무시). cond는 주는데
        cond_mask를 안 주면 기존 호출부와의 하위호환으로 "전부 안다"로
        취급(net.forward와 동일 규칙). guidance의 "무조건부" 기준점은
        이제 학습된 null_cond가 아니라 값=0,마스크=0 자체다.
        """
        node_mask = torch.ones(n, n_atoms, dtype=torch.bool, device=device)
        if cond is not None:
            cond = cond.to(device)
            cond_mask = torch.ones_like(cond) if cond_mask is None else cond_mask.to(device)
        if guidance_w is not None:
            assert cond is not None, "guidance_w는 cond가 있을 때만 의미가 있다"
        m_X, m_E = self.schedule.m_X, self.schedule.m_E

        # 초기 상태를 marginal 분포에서 샘플링 (DiGress Algorithm 2)
        X_t = torch.multinomial(m_X.expand(n * n_atoms, -1), 1).view(n, n_atoms).to(device)
        E_t = torch.multinomial(m_E.expand(n * n_atoms * n_atoms, -1), 1).view(n, n_atoms, n_atoms).to(device)

        E_t = self._sym(E_t)
        diag_mask = ~torch.eye(n_atoms, dtype=torch.bool, device=device).unsqueeze(0)
        E_t = torch.where(diag_mask, E_t, torch.zeros_like(E_t))

        for step in range(self.T, 0, -1):
            if step_cb is not None:
                step_cb(self.T - step, self.T)  # [서빙] 진행률/취소 훅 -- 계산에는 영향 없음, 콜백이 예외를 던지면 여기서 중단됨
            t = torch.full((n,), step, dtype=torch.long, device=device)

            if guidance_w is None:
                xl, el, _ = self.net(X_t, E_t, node_mask, t, cond, cond_mask)
            else:
                null_cond = torch.zeros(n, COND_DIM, device=device)
                null_mask = torch.zeros(n, COND_DIM, device=device)
                X2 = torch.cat([X_t, X_t], dim=0)
                E2 = torch.cat([E_t, E_t], dim=0)
                mask2 = torch.cat([node_mask, node_mask], dim=0)
                t2 = torch.cat([t, t], dim=0)
                cond2 = torch.cat([cond, null_cond], dim=0)
                cond_mask2 = torch.cat([cond_mask, null_mask], dim=0)
                xl2, el2, _ = self.net(X2, E2, mask2, t2, cond2, cond_mask2)
                xl_cond, xl_uncond = xl2[:n], xl2[n:]
                el_cond, el_uncond = el2[:n], el2[n:]
                xl = xl_uncond + guidance_w * (xl_cond - xl_uncond)
                el = el_uncond + guidance_w * (el_cond - el_uncond)

            # x_head는 실제 원자 클래스(1..9)만 예측 -> K_X 차원으로 복원 후 posterior 계산
            x0_probs_full = torch.zeros(n, n_atoms, K_X, device=device)
            x0_probs_full[..., 1:] = xl.softmax(-1)

            X_t = self.schedule.posterior_sample(X_t, x0_probs_full, t, m_X)
            E_t = self._sym(self.schedule.posterior_sample(E_t, el.softmax(-1), t, m_E))
            E_t = torch.where(diag_mask, E_t, torch.zeros_like(E_t))

        return X_t, E_t

    @staticmethod
    def _sym(E):
        N = E.shape[-1]
        triu = torch.triu(torch.ones(N, N, device=E.device, dtype=torch.bool), diagonal=1)
        E = E * triu
        return E + E.transpose(-1, -2)

    @staticmethod
    def _to_dense(batch):
        X0_raw, node_mask = to_dense_batch(batch.z, batch.batch)
        N = X0_raw.size(1)  # batch-local max atom count
        X0 = torch.zeros_like(X0_raw)
        for anum, cls in ATOMIC_NUM_TO_CLS.items():
            X0[X0_raw == anum] = cls

        ei = batch.edge_index
        uv = ei[0] < ei[1]
        bc = batch.bond_type[uv].argmax(dim=-1) + 1
        E0 = to_dense_adj(
            ei[:, uv], batch=batch.batch,
            edge_attr=bc.unsqueeze(-1).float(),
            max_num_nodes=N,
        ).squeeze(-1).long()
        E0 = E0 + E0.transpose(1, 2)

        return X0, E0, node_mask
