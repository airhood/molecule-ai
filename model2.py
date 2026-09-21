# [Stage 2/3] 초기 diffusion 모델(absorbing-state 이전). model3.py의
# Absorbing Discrete Graph Diffusion으로 대체되어 더 이상 학습/개발되지
# 않는 참조용 코드. 계보: docs/model_lineup.md.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch


MAX_ATOMS = 50
K_X = 10
K_E = 5
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

NODE_DIM = 256
EDGE_DIM = 64
N_HEADS = 8
N_LAYERS = 6
FF_DIM = 512
T_DIM = 128


def _cosine_ac(T, s = 0.008):
    steps = torch.arange(T+1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    return (f / f[0]).float()[1:]


class DiscreteNoiseSchedule(nn.Module):

    def __init__(self, T = T_STEPS):
        super().__init__()
        ac = _cosine_ac(T)
        self.register_buffer("ac", ac) # alpha_bar[1] ... alpha_bar[T]
        self.register_buffer("ac_prev", torch.cat([torch.ones(1), ac[:-1]])) # alpha_bar[0] = 1, alpha_bar[1] ... alpha_bar[T-1]
        self.T = T

    def q_sample(self, x0, t, K):
        shape = x0.shape
        B = t.shape[0]
        ac_t = self.ac[t-1].view(B, *([1] * (len(shape) - 1)))
        keep = torch.bernoulli(ac_t.expand_as(x0.float())).bool()
        return torch.where(keep, x0, torch.randint(0, K, shape, device=x0.device))
    
    def posterior_sample(self, x_t, x0_probs, t, K):
        shape = x_t.shape
        B = t.shape[0]
        view = (B,) + (1,) * (len(shape) - 1)   # for t_mask: (B,1) atoms | (B,1,1) edges
        view_k = (B,) + (1,) * len(shape)        # one extra dim for K after one_hot

        ac_t = self.ac[t-1].view(view_k)
        ac_tm1 = self.ac_prev[t-1].view(view_k)
        alpha_t = (ac_t / ac_tm1.clamp(min=1e-8)).clamp(max=1.0)

        likelihood = alpha_t * F.one_hot(x_t, K).float() + (1 - alpha_t) / K
        prior = ac_tm1 * x0_probs + (1 - ac_tm1) / K

        post = likelihood * prior
        post = post / post.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        stoch = torch.multinomial(post.view(-1, K), 1).view(shape)
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
            nn.Linear(NODE_DIM + EDGE_DIM, FF_DIM),
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
        e = self.enorm(e + self.edge_ff(torch.cat([hi + hj, e], dim=-1)))

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
        self.x_head  = nn.Sequential(nn.LayerNorm(NODE_DIM), nn.Linear(NODE_DIM, K_X))
        self.e_head  = nn.Sequential(nn.LayerNorm(EDGE_DIM), nn.Linear(EDGE_DIM, K_E))

    def forward(self, X_t, E_t, node_mask, t):
        h   = self.x_embed(X_t) + self.t_proj(self.t_embed(t)).unsqueeze(1)
        e   = self.e_embed(E_t)
        pad = ~node_mask

        for layer in self.layers:
            h, e = layer(h, e, pad)

        x_logits = self.x_head(h)
        e_logits = self.e_head(e)
        e_logits = (e_logits + e_logits.transpose(1, 2)) / 2
        return x_logits, e_logits
    

class MoleculeGraphDiffusion(nn.Module):

    def __init__(self, T = T_STEPS):
        super().__init__()
        self.schedule = DiscreteNoiseSchedule(T)
        self.net = ScoreNetwork()
        self.T = T

    def forward(self, batch):
        X0, E0, node_mask = self._to_dense(batch)
        B, N = X0.shape
        device = X0.device

        t = torch.randint(1, self.T + 1, (B,), device=device)
        X_t = self.schedule.q_sample(X0, t, K_X)
        E_t = self._sym(self.schedule.q_sample(E0, t, K_E))

        x_logits, e_logits = self.net(X_t, E_t, node_mask, t)

        nm = node_mask.view(-1)
        loss_x = F.cross_entropy(x_logits.view(-1, K_X)[nm], X0.view(-1)[nm])

        triu = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        em = (node_mask.unsqueeze(2) & node_mask.unsqueeze(1)) & triu.unsqueeze(0)
        loss_e = F.cross_entropy(
            e_logits.view(-1, K_E)[em.view(-1)],
            E0.view(-1)[em.view(-1)]
        )

        loss = loss_x + loss_e
        return {
            "loss": loss,
            "loss_x": loss_x,
            "loss_e": loss_e
        }
    
    @torch.no_grad()
    def sample(self, n, n_atoms, device):
        node_mask = torch.ones(n, n_atoms, dtype=torch.bool, device=device)
        X_t = torch.randint(0, K_X, (n, n_atoms), device=device)
        E_t = self._sym(torch.randint(0, K_E, (n, n_atoms, n_atoms), device=device))

        for step in range(self.T, 0, -1):
            t = torch.full((n,), step, dtype=torch.long, device=device)
            xl, el = self.net(X_t, E_t, node_mask, t)
            X_t = self.schedule.posterior_sample(X_t, xl.softmax(-1), t, K_X)
            E_t = self._sym(self.schedule.posterior_sample(E_t, el.softmax(-1), t, K_E))

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