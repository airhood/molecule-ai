import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch


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

def _connectivity_features(E_t, node_mask):
    B, N, _ = E_t.shape
    eye = torch.eye(N, dtype=torch.bool, device=E_t.device).unsqueeze(0)
    both_real = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)

    adj = (E_t > 0) & both_real
    adj = adj | eye  # 자기 자신 포함 (실/패딩 공통)
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

        xt_clamped = x_t.clamp(min=0, max=K - 1)
        xt_onehot = F.one_hot(xt_clamped, K).float()
        m_xt = m[xt_clamped]
        term1 = (alpha_t.unsqueeze(-1) * xt_onehot + beta_t.unsqueeze(-1) * m_xt.unsqueeze(-1)).unsqueeze(-2)

        eye = torch.eye(K, device=x_t.device, dtype=torch.float)
        beta_tm1 = 1.0 - ac_tm1
        term2 = ac_tm1.unsqueeze(-1).unsqueeze(-1) * eye + beta_tm1.unsqueeze(-1).unsqueeze(-1) * m.view(*([1]*len(view)), 1, K)

        unnorm = (term1 * term2).clamp(min=0.0)
        norm_c = unnorm.sum(-1, keepdim=True).clamp(min=1e-8)
        post_given_c = unnorm / norm_c

        result = (x0_probs.unsqueeze(-1) * post_given_c).sum(-2)
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
        Q = Q.view(B, N, nh, dh).transpose(1, 2)  # (B, nh, N, dh)
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

        hi = h.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, D)
        hj = h.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D)
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

        weight_table = torch.zeros(K_X)
        for _anum, _cls in ATOMIC_NUM_TO_CLS.items():
            weight_table[_cls] = _ATOMIC_WEIGHT[_anum]
        self.register_buffer("atomic_weight", weight_table)
        self.struct_proj = nn.Linear(2, NODE_DIM)

        self.conn_proj = nn.Linear(2, NODE_DIM)

        COND_DIM = 2
        self.cond_proj = nn.Linear(COND_DIM, NODE_DIM)
        self.null_cond = nn.Parameter(torch.zeros(COND_DIM))

        self.x_head  = nn.Sequential(nn.LayerNorm(NODE_DIM), nn.Linear(NODE_DIM, K_X - 1))
        self.e_head  = nn.Sequential(nn.LayerNorm(EDGE_DIM), nn.Linear(EDGE_DIM, K_E))

    def forward(self, X_t, E_t, node_mask, t, cond=None):
        cur_valence = E_t.float().sum(dim=-1, keepdim=True) / 4.0
        atomic_weight = self.atomic_weight[X_t.clamp(min=0, max=K_X - 1)].unsqueeze(-1) / 100.0
        struct_feat = self.struct_proj(torch.cat([cur_valence, atomic_weight], dim=-1))

        n_components, comp_size_ratio = _connectivity_features(E_t, node_mask)
        n_comp_bcast = (n_components / 10.0).unsqueeze(1).expand(-1, X_t.shape[1])
        conn_feat = torch.stack([n_comp_bcast, comp_size_ratio], dim=-1)
        conn_struct = self.conn_proj(conn_feat)

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

        X_t = self.schedule.q_sample(X0, t, self.schedule.m_X)
        E_t = self._sym(self.schedule.q_sample(E0, t, self.schedule.m_E))

        X_t = torch.where(node_mask, X_t, torch.zeros_like(X_t))
        e_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        E_t = torch.where(e_mask, E_t, torch.zeros_like(E_t))

        cond = self._extract_property_cond(batch, B, device)
        if self.training:
            drop = torch.rand(B, device=device) < self.COND_DROPOUT_P
            cond = torch.where(drop.unsqueeze(-1), self.net.null_cond.unsqueeze(0).expand(B, -1), cond)

        x_logits, e_logits = self.net(X_t, E_t, node_mask, t, cond)

        nm = node_mask.view(-1)
        loss_x = F.cross_entropy(  # 원자: 실제 클래스 1..9 -> head 출력 0..8로 매핑
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
        a_bin10 = batch.a_bin.view(B, -1)
        idx = torch.tensor([0, 2, 3, 4, 5, 6, 7, 8, 9], device=device)
        return a_bin10[:, idx]

    @staticmethod
    def _extract_property_cond(batch, B, device):
        p = batch.p.view(B, -1)
        return p[:, [0, 1]]

    @torch.no_grad()
    def sample(self, n, n_atoms, device, cond=None, guidance_w=None):
        node_mask = torch.ones(n, n_atoms, dtype=torch.bool, device=device)
        if cond is not None:
            cond = cond.to(device)
        if guidance_w is not None:
            assert cond is not None, "guidance_w는 cond가 있을 때만 의미가 있다"
        m_X, m_E = self.schedule.m_X, self.schedule.m_E

        X_t = torch.multinomial(m_X.expand(n * n_atoms, -1), 1).view(n, n_atoms).to(device)  # marginal 분포에서 초기 샘플링 (DiGress Algorithm 2)
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

            x0_probs_full = torch.zeros(n, n_atoms, K_X, device=device)  # x_head는 클래스 1..9만 예측 -> K_X 차원으로 복원
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
