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
    comp_size_ratio (B,N) 노드별 자기 성분 크기/실제 노드 수."""
    B, N, _ = E_t.shape
    eye = torch.eye(N, dtype=torch.bool, device=E_t.device).unsqueeze(0)
    both_real = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)

    adj = (E_t > 0) & both_real
    adj = adj | eye  # 자기 자신 포함 (실/패딩 공통) -- 패딩 행은 이후에도 자기자신만 유지
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

    return n_components, comp_size_ratio


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

    def forward(self, h, e, pad_mask):
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
        self.conn_proj = nn.Linear(2, NODE_DIM)

        # [C-3] 2026-09-05: review8.md 지시로 conditioning 배선을 물성보다
        # 먼저 a_bin(9차원)으로 스모크 테스트해 통과함(off-distribution 타겟
        # 4개 전부 CI 비겹침, docs/review_followup_20260905b.md §3).
        # [C-4] 2026-09-05: review9.md 지시로 물성 conditioning 본 실험 착수.
        # 같은 메커니즘(cond_proj + 학습 가능한 null_cond로 CFG 드롭 시 대입 +
        # COND_DROPOUT_P)을 재사용하되 입력을 a_bin(9차원)에서 HOMO/LUMO
        # 정규화 값(2차원, batch.p 순서 0,1)으로 교체 -- GAP은 LUMO-HOMO로
        # 선형종속(review9.md §0, 상관계수 1.0)이라 조건에서 제외, E_total은
        # 조성만으로 R²=1.0(§0/[C-2])이라 제외, Dipole은 GNN_R²=0.303으로
        # 2D 그래프로는 70%가 설명 안 돼 조건 경로에 잡음만 주입하므로 제외.
        COND_DIM = 2
        self.cond_proj = nn.Linear(COND_DIM, NODE_DIM)
        self.null_cond = nn.Parameter(torch.zeros(COND_DIM))

        # 원자: padding(0) 제외한 K_X-1개 실제 원소 예측.
        # 결합: MASK 클래스가 없으므로 no-bond(0) 포함 전체 K_E개 예측.
        self.x_head  = nn.Sequential(nn.LayerNorm(NODE_DIM), nn.Linear(NODE_DIM, K_X - 1))
        self.e_head  = nn.Sequential(nn.LayerNorm(EDGE_DIM), nn.Linear(EDGE_DIM, K_E))

    def forward(self, X_t, E_t, node_mask, t, cond=None):
        # [R-5] 현재 원자가(no-bond=0..triple=3, 값 자체가 결합 차수) + 원자량.
        # 스케일이 서로/다른 입력과 크게 달라 대략적인 크기로만 정규화(학습 가능한
        # struct_proj가 나머지를 흡수).
        cur_valence = E_t.float().sum(dim=-1, keepdim=True) / 4.0
        atomic_weight = self.atomic_weight[X_t.clamp(min=0, max=K_X - 1)].unsqueeze(-1) / 100.0
        struct_feat = self.struct_proj(torch.cat([cur_valence, atomic_weight], dim=-1))

        # [R-6] 연결 성분 개수/노드별 성분 크기 비율 -- 학습 불가 파라미터 없이
        # 현재 E_t에서 즉석 계산 (review7.md [Z-4]).
        n_components, comp_size_ratio = _connectivity_features(E_t, node_mask)
        n_comp_bcast = (n_components / 10.0).unsqueeze(1).expand(-1, X_t.shape[1])
        conn_feat = torch.stack([n_comp_bcast, comp_size_ratio], dim=-1)
        conn_struct = self.conn_proj(conn_feat)

        # [C-3] cond가 안 주어지면(기존 무조건부 호출과의 하위호환)
        # null_cond를 배치 전체에 broadcast -- 학습된 "조건 없음" 임베딩.
        if cond is None:
            cond = self.null_cond.unsqueeze(0).expand(X_t.shape[0], -1)
        cond_struct = self.cond_proj(cond).unsqueeze(1).expand(-1, X_t.shape[1], -1)

        h   = self.x_embed(X_t) + self.t_proj(self.t_embed(t)).unsqueeze(1) + struct_feat + conn_struct + cond_struct
        e   = self.e_embed(E_t)
        pad = ~node_mask

        for layer in self.layers:
            h, e = layer(h, e, pad)

        x_logits = self.x_head(h)
        e_logits = self.e_head(e)
        e_logits = (e_logits + e_logits.transpose(1, 2)) / 2
        return x_logits, e_logits


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
    COND_DROPOUT_P = 0.15

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

        # [C-4] HOMO/LUMO 물성 조건 + CFG 학습용 랜덤 드롭.
        cond = self._extract_property_cond(batch, B, device)
        if self.training:
            drop = torch.rand(B, device=device) < self.COND_DROPOUT_P
            cond = torch.where(drop.unsqueeze(-1), self.net.null_cond.unsqueeze(0).expand(B, -1), cond)

        x_logits, e_logits = self.net(X_t, E_t, node_mask, t, cond)

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

        loss = loss_x + self.edge_loss_coeff * loss_e
        return {
            "loss": loss,
            "loss_x": loss_x,
            "loss_e": loss_e
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
        """[C-4] dataset2.py의 batch.p(5차원 정규화값, stats.json 순서
        HOMO,LUMO,GAP,E_total,Dipole)에서 HOMO/LUMO(인덱스 0,1)만 뽑는다.
        GAP=LUMO-HOMO 선형종속/E_total 조성 누설/Dipole 2D 예측불가로 제외
        (review9.md §0, §4)."""
        p = batch.p.view(B, -1)
        return p[:, [0, 1]]

    @torch.no_grad()
    def sample(self, n, n_atoms, device, cond=None, guidance_w=None):
        """[D-3] review10.md §4 Q1 지시로 classifier-free guidance 추가.
        guidance_w가 None이면 기존과 동일(순수 조건부 forward 1회, w=1과
        동치). guidance_w가 주어지면(cond 필수) 매 step에서 조건부/무조건부
        (null_cond) 로짓을 함께 계산해 로짓 공간에서 외삽한다:
            guided = uncond + w * (cond - uncond)
        w=0 -> 순수 무조건부, w=1 -> 순수 조건부(위와 동일), w>1 -> 조건 신호
        증폭. 배치를 2배로 합쳐 한 번의 forward로 처리(속도 손실 최소화).
        """
        node_mask = torch.ones(n, n_atoms, dtype=torch.bool, device=device)
        if cond is not None:
            cond = cond.to(device)
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
            t = torch.full((n,), step, dtype=torch.long, device=device)

            if guidance_w is None:
                xl, el = self.net(X_t, E_t, node_mask, t, cond)
            else:
                null_cond = self.net.null_cond.unsqueeze(0).expand(n, -1)
                X2 = torch.cat([X_t, X_t], dim=0)
                E2 = torch.cat([E_t, E_t], dim=0)
                mask2 = torch.cat([node_mask, node_mask], dim=0)
                t2 = torch.cat([t, t], dim=0)
                cond2 = torch.cat([cond, null_cond], dim=0)
                xl2, el2 = self.net(X2, E2, mask2, t2, cond2)
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
