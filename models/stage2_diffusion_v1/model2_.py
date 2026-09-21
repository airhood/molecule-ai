"""
model2_.py — DiGress-style discrete graph diffusion for molecule generation.

Forward process:  q(x_t | x_0) = Cat(ᾱ_t · x_0 + (1-ᾱ_t)/K · 1)   (uniform noise)
Reverse process:  p_θ(x_{t-1}|x_t) from predicted x̂_0 via closed-form posterior
Network:          Graph Transformer with bidirectional node ↔ edge updates
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch

# ── Dataset-level constants ────────────────────────────────────────────────
MAX_ATOMS = 64
K_X       = 11   # atom classes: 0=virtual/pad, 1=C, 2=H, 3=O, 4=N, 5=S, 6=P, 7=F, 8=Cl, 9=Br, 10=I
K_E       = 5    # bond classes: 0=no-bond, 1=single, 2=double, 3=triple, 4=aromatic
T_STEPS   = 500

ATOMIC_NUM_TO_CLS = {6: 1, 1: 2, 8: 3, 7: 4, 16: 5, 15: 6, 9: 7, 17: 8, 35: 9, 53: 10}

# ── Network hyperparameters ────────────────────────────────────────────────
NODE_DIM = 256
EDGE_DIM = 64
N_HEADS  = 8
N_LAYERS = 6
FF_DIM   = 512
T_DIM    = 128


# ── Noise schedule ─────────────────────────────────────────────────────────

def _cosine_ac(T: int, s: float = 0.008) -> torch.Tensor:
    """Returns ᾱ_t for t = 1..T (cosine schedule)."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    return (f / f[0]).float()[1:]


class DiscreteNoiseSchedule(nn.Module):
    """
    Uniform categorical noise:
        q(x_t | x_0) = Cat(ᾱ_t · one_hot(x_0) + (1-ᾱ_t)/K · 1)

    Reverse posterior (closed form):
        q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) · q(x_{t-1} | x_0)
    """

    def __init__(self, T: int = T_STEPS):
        super().__init__()
        ac = _cosine_ac(T)
        self.register_buffer("ac",      ac)                                  # ᾱ_1..ᾱ_T
        self.register_buffer("ac_prev", torch.cat([torch.ones(1), ac[:-1]])) # ᾱ_0=1, ᾱ_1..ᾱ_{T-1}
        self.T = T

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, K: int) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x_0).  x0: (...) long → x_t: (...) long."""
        shape = x0.shape
        B = t.shape[0]
        ac_t = self.ac[t - 1].view(B, *([1] * (len(shape) - 1)))
        keep = torch.bernoulli(ac_t.expand_as(x0.float())).bool()
        return torch.where(keep, x0, torch.randint(0, K, shape, device=x0.device))

    def posterior_sample(
        self,
        x_t: torch.Tensor,
        x0_probs: torch.Tensor,
        t: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """
        Sample x_{t-1} ~ q(x_{t-1} | x_t, x̂_0).
        At t=1 returns argmax(x0_probs) deterministically.

        x_t      : (...) long
        x0_probs : (..., K) float  — softmax output of network
        """
        shape = x_t.shape
        B = t.shape[0]
        view = (B,) + (1,) * (len(shape) - 1)

        ac_t    = self.ac[t - 1].view(view)
        ac_tm1  = self.ac_prev[t - 1].view(view)
        alpha_t = (ac_t / ac_tm1.clamp(min=1e-8)).clamp(max=1.0)  # α_t = ᾱ_t / ᾱ_{t-1}

        # likelihood: q(x_t | x_{t-1} = c) for observed x_t
        likelihood = alpha_t * F.one_hot(x_t, K).float() + (1 - alpha_t) / K  # (..., K)
        # prior: q(x_{t-1} = c | x̂_0)
        prior = ac_tm1 * x0_probs + (1 - ac_tm1) / K                           # (..., K)

        post = likelihood * prior
        post = post / post.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        stoch  = torch.multinomial(post.view(-1, K), 1).view(shape)
        determ = x0_probs.argmax(dim=-1)
        t_mask = (t == 1).view(view).expand(shape)
        return torch.where(t_mask, determ, stoch)


# ── Time embedding ─────────────────────────────────────────────────────────

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = T_DIM):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) long → (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / (half - 1)
        )
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)


# ── Graph Transformer layer ─────────────────────────────────────────────────

class GTLayer(nn.Module):
    """
    Bidirectional Graph Transformer layer.

    Node update: multi-head self-attention with edge features injected as attention bias.
    Edge update: MLP on (h_i, h_j, e_ij) after node update.
    """

    def __init__(self):
        super().__init__()
        assert NODE_DIM % N_HEADS == 0
        # Node attention
        self.qkv      = nn.Linear(NODE_DIM, NODE_DIM * 3, bias=False)
        self.e_bias   = nn.Linear(EDGE_DIM, N_HEADS)         # edge → per-head attention bias
        self.attn_out = nn.Linear(NODE_DIM, NODE_DIM)
        self.norm1    = nn.LayerNorm(NODE_DIM)
        self.ff       = nn.Sequential(
            nn.Linear(NODE_DIM, FF_DIM), nn.GELU(), nn.Linear(FF_DIM, NODE_DIM)
        )
        self.norm2    = nn.LayerNorm(NODE_DIM)
        # Edge update — symmetric: hi+hj so (i,j) and (j,i) get identical input
        self.edge_ff  = nn.Sequential(
            nn.Linear(NODE_DIM + EDGE_DIM, FF_DIM),
            nn.GELU(),
            nn.Linear(FF_DIM, EDGE_DIM),
        )
        self.enorm    = nn.LayerNorm(EDGE_DIM)

    def forward(
        self,
        h: torch.Tensor,        # (B, N, NODE_DIM)
        e: torch.Tensor,        # (B, N, N, EDGE_DIM)
        pad_mask: torch.Tensor, # (B, N) True = padding
    ):
        B, N, _ = h.shape
        nh = N_HEADS
        dh = NODE_DIM // N_HEADS

        # ── Node self-attention ───────────────────────────────────────────
        Q, K, V = self.qkv(h).chunk(3, dim=-1)
        Q = Q.view(B, N, nh, dh).transpose(1, 2)   # (B, nh, N, dh)
        K = K.view(B, N, nh, dh).transpose(1, 2)
        V = V.view(B, N, nh, dh).transpose(1, 2)

        scores = torch.einsum("bhid,bhjd->bhij", Q, K) / math.sqrt(dh)  # (B, nh, N, N)
        scores = scores + self.e_bias(e).permute(0, 3, 1, 2)             # + edge bias
        scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))

        attn = scores.softmax(dim=-1)
        out  = torch.einsum("bhij,bhjd->bhid", attn, V)
        out  = out.transpose(1, 2).reshape(B, N, -1)
        h = self.norm1(h + self.attn_out(out))
        h = self.norm2(h + self.ff(h))

        # ── Edge update ───────────────────────────────────────────────────
        hi = h.unsqueeze(2).expand(-1, -1, N, -1)   # (B, N, N, D)
        hj = h.unsqueeze(1).expand(-1, N, -1, -1)   # (B, N, N, D)
        e  = self.enorm(e + self.edge_ff(torch.cat([hi + hj, e], dim=-1)))

        # Zero out padding positions to prevent garbage accumulation across layers
        h = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        e_pad = pad_mask.unsqueeze(1) | pad_mask.unsqueeze(2)   # (B, N, N)
        e = e.masked_fill(e_pad.unsqueeze(-1), 0.0)

        return h, e


# ── Score network ───────────────────────────────────────────────────────────

class ScoreNetwork(nn.Module):
    """
    Predicts x̂_0 (clean atom types and bond types) from noisy (X_t, E_t, t).

    Input
    -----
    X_t       : (B, N)       long  — noisy atom class indices
    E_t       : (B, N, N)    long  — noisy bond class indices (symmetric)
    node_mask : (B, N)       bool  — True = real atom
    t         : (B,)         long  — diffusion timestep

    Output
    ------
    x_logits  : (B, N, K_X)
    e_logits  : (B, N, N, K_E)
    """

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
        e_logits = (e_logits + e_logits.transpose(1, 2)) / 2   # enforce symmetry
        return x_logits, e_logits


# ── Full diffusion model ────────────────────────────────────────────────────

class MoleculeGraphDiffusion(nn.Module):

    def __init__(self, T: int = T_STEPS):
        super().__init__()
        self.schedule = DiscreteNoiseSchedule(T)
        self.net      = ScoreNetwork()
        self.T        = T

    # ── Training ──────────────────────────────────────────────────────────
    def forward(self, batch) -> dict:
        """
        batch: PyG Batch with z, edge_index, bond_type, batch fields.
        Returns {'loss', 'loss_x', 'loss_e'}.
        """
        X0, E0, node_mask = self._to_dense(batch)
        B, N = X0.shape
        device = X0.device

        t   = torch.randint(1, self.T + 1, (B,), device=device)
        X_t = self.schedule.q_sample(X0, t, K_X)
        E_t = self._sym(self.schedule.q_sample(E0, t, K_E))

        x_logits, e_logits = self.net(X_t, E_t, node_mask, t)

        # Node loss — real atoms only
        nm     = node_mask.view(-1)
        loss_x = F.cross_entropy(x_logits.view(-1, K_X)[nm], X0.view(-1)[nm])

        # Edge loss — upper triangle of real atom pairs only
        triu   = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        em     = (node_mask.unsqueeze(2) & node_mask.unsqueeze(1)) & triu.unsqueeze(0)
        loss_e = F.cross_entropy(
            e_logits.view(-1, K_E)[em.view(-1)],
            E0.view(-1)[em.view(-1)],
        )

        loss = loss_x + loss_e
        return {"loss": loss, "loss_x": loss_x, "loss_e": loss_e}

    # ── Sampling ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, n: int, n_atoms: int, device: torch.device):
        """
        Generate n molecules each with n_atoms atoms.
        n_atoms should be sampled from the training set size distribution in practice.
        Returns X (n, n_atoms) and E (n, n_atoms, n_atoms) as class indices.
        """
        node_mask = torch.ones(n, n_atoms, dtype=torch.bool, device=device)
        X_t = torch.randint(0, K_X, (n, n_atoms), device=device)
        E_t = self._sym(torch.randint(0, K_E, (n, n_atoms, n_atoms), device=device))

        for step in range(self.T, 0, -1):
            t       = torch.full((n,), step, dtype=torch.long, device=device)
            xl, el  = self.net(X_t, E_t, node_mask, t)
            X_t     = self.schedule.posterior_sample(X_t, xl.softmax(-1), t, K_X)
            E_t     = self._sym(self.schedule.posterior_sample(E_t, el.softmax(-1), t, K_E))

        return X_t, E_t

    # ── Helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _sym(E: torch.Tensor) -> torch.Tensor:
        """Upper-triangle only, then symmetrize, zero diagonal."""
        N = E.shape[-1]
        triu = torch.triu(torch.ones(N, N, device=E.device, dtype=torch.bool), diagonal=1)
        E = E * triu                    # zero lower triangle + diagonal
        return E + E.transpose(-1, -2)  # symmetrize

    @staticmethod
    def _to_dense(batch):
        """PyG batch → dense (X0, E0, node_mask)."""
        X0_raw, node_mask = to_dense_batch(batch.z, batch.batch, max_num_nodes=MAX_ATOMS)
        X0 = torch.zeros_like(X0_raw)
        for anum, cls in ATOMIC_NUM_TO_CLS.items():
            X0[X0_raw == anum] = cls

        ei  = batch.edge_index
        uv  = ei[0] < ei[1]
        bc  = batch.bond_type[uv].argmax(dim=-1) + 1   # one-hot → class 1..4
        E0  = to_dense_adj(
            ei[:, uv], batch=batch.batch,
            edge_attr=bc.unsqueeze(-1).float(),
            max_num_nodes=MAX_ATOMS,
        ).squeeze(-1).long()
        E0  = E0 + E0.transpose(1, 2)  # upper-tri only → full symmetric

        return X0, E0, node_mask
