import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, VGAE, global_mean_pool
from torch_geometric.utils import to_dense_adj

# ── 상수 ───────────────────────────────────────────────────────────────────────
# 조성 벡터 순서: C, H, O, N, S, P, F, Cl, Br, I
ELEM_ATOMIC_NUMS = [6, 1, 8, 7, 16, 15, 9, 17, 35, 53]

LATENT_DIM   = 128
HIDDEN_DIM   = 256
ATOM_EMB_DIM = 64
COND_DIM     = 128
ENC_LAYERS   = 4
DEC_LAYERS   = 4
N_BOND_TYPES = 4   # 단일 / 이중 / 삼중 / 컨쥬게이션

# 노드 피처 차원: atom_emb(64) + charge(1) + chirality(3)
NODE_DIM = ATOM_EMB_DIM + 1 + 3   # 68

# 엣지 피처 차원: bond_type(4) + bond_stereo(5) + dihedral(1)
EDGE_DIM = 4 + 5 + 1              # 10


# ── 유틸 ───────────────────────────────────────────────────────────────────────
def mlp(dims: list[int], dropout: float = 0.0) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.ReLU(), nn.Dropout(dropout)] if dropout else [nn.ReLU()]
    return nn.Sequential(*layers)



def a_to_atom_tensors(
    a: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    조성 벡터 [B, 11] → 패딩된 원자 번호 행렬 [B, N_max] + 마스크 [B, N_max].
    마스크: True = 실제 원자, False = 패딩.
    """
    atom_lists = []
    for row in a:
        atoms = []
        for atomic_num, cnt in zip(ELEM_ATOMIC_NUMS, row[:10].long().tolist()):
            atoms.extend([atomic_num] * cnt)
        atom_lists.append(atoms)

    N_max = max(len(al) for al in atom_lists)
    B = len(atom_lists)

    atom_idx = torch.zeros(B, N_max, dtype=torch.long, device=device)
    mask = torch.zeros(B, N_max, dtype=torch.bool, device=device)
    for b, atoms in enumerate(atom_lists):
        n = len(atoms)
        atom_idx[b, :n] = torch.tensor(atoms, dtype=torch.long, device=device)
        mask[b, :n] = True

    return atom_idx, mask


# ── 인코더 (학습 전용) ─────────────────────────────────────────────────────────
class GraphEncoder(nn.Module):
    """실제 분자 그래프 → 임베딩 벡터 [B, HIDDEN_DIM]. 학습 중에만 사용."""

    def __init__(self):
        super().__init__()
        self.atom_emb  = nn.Embedding(120, ATOM_EMB_DIM)
        self.atom_proj = mlp([NODE_DIM, HIDDEN_DIM])

        # GINEConv: 엣지 피처(bond_type + bond_stereo + dihedral) 포함
        self.convs = nn.ModuleList([
            GINEConv(
                mlp([HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM]),
                edge_dim=EDGE_DIM,
            )
            for _ in range(ENC_LAYERS)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(HIDDEN_DIM) for _ in range(ENC_LAYERS)])

        self.pool_proj = mlp([HIDDEN_DIM + 5 + 11, HIDDEN_DIM])

    def forward(self, batch) -> torch.Tensor:
        h = torch.cat(
            [self.atom_emb(batch.z), batch.charge.unsqueeze(-1), batch.chirality],
            dim=-1,
        )
        h = self.atom_proj(h)

        edge_attr = torch.cat(
            [batch.bond_type, batch.bond_stereo, batch.dihedral.unsqueeze(-1)],
            dim=-1,
        )                                                        # [E, EDGE_DIM]

        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.relu(conv(h, batch.edge_index, edge_attr)))

        g = global_mean_pool(h, batch.batch)                     # [B, HIDDEN]
        return self.pool_proj(torch.cat([g, batch.p, batch.a], dim=-1))  # [B, HIDDEN]


# ── 디코더 (핵심 생성 모듈) ────────────────────────────────────────────────────
class MolDecoder(nn.Module):
    """
    (z, p, a) → 분자 그래프 (결합 유무 + 결합 종류).

    1. 원자별 초기 임베딩 생성 (조성 + 전역 조건)
    2. Transformer self-attention (완전 그래프 = 모든 원자 쌍이 메시지 교환)
    3. 대칭 쌍 특징 → 결합 예측
    """

    def __init__(self):
        super().__init__()
        self.atom_emb  = nn.Embedding(120, ATOM_EMB_DIM)
        self.cond_proj = mlp([LATENT_DIM + 5 + 11, COND_DIM])
        self.init_proj = mlp([ATOM_EMB_DIM + COND_DIM, HIDDEN_DIM])

        # 완전 그래프 메시지 패싱 → Transformer self-attention
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(HIDDEN_DIM, num_heads=8, batch_first=True)
            for _ in range(DEC_LAYERS)
        ])
        self.attn_norms = nn.ModuleList([nn.LayerNorm(HIDDEN_DIM) for _ in range(DEC_LAYERS)])
        self.ffns = nn.ModuleList([
            mlp([HIDDEN_DIM, HIDDEN_DIM * 2, HIDDEN_DIM])
            for _ in range(DEC_LAYERS)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(HIDDEN_DIM) for _ in range(DEC_LAYERS)])

        # 쌍 특징: (h_i + h_j) || (h_i * h_j) → 대칭성 보장
        pair_dim = HIDDEN_DIM * 2
        self.bond_exist = mlp([pair_dim, HIDDEN_DIM, 1])
        self.bond_type  = mlp([pair_dim, HIDDEN_DIM, N_BOND_TYPES])

    def forward(
        self, z: torch.Tensor, p: torch.Tensor, a: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        반환:
            bond_exist_logit : [B, N_max, N_max]
            bond_type_logit  : [B, N_max, N_max, 4]
            atom_mask        : [B, N_max]  True = 실제 원자
        """
        device = z.device
        N_max: int

        c = self.cond_proj(torch.cat([z, p, a], dim=-1))        # [B, COND_DIM]

        atom_idx, atom_mask = a_to_atom_tensors(a, device)      # [B, N_max]
        N_max = atom_idx.size(1)

        h_atom = self.atom_emb(atom_idx)                        # [B, N_max, ATOM_EMB]
        c_exp  = c.unsqueeze(1).expand(-1, N_max, -1)           # [B, N_max, COND_DIM]
        h = self.init_proj(torch.cat([h_atom, c_exp], dim=-1))  # [B, N_max, HIDDEN]

        key_pad_mask = ~atom_mask                               # True = 패딩 (PyTorch 규약)
        for attn, a_norm, ffn, f_norm in zip(
            self.attn_layers, self.attn_norms, self.ffns, self.ffn_norms
        ):
            h2, _ = attn(h, h, h, key_padding_mask=key_pad_mask)
            h = a_norm(h + h2)
            h = f_norm(h + ffn(h))

        # 대칭 쌍 특징
        hi = h.unsqueeze(2).expand(-1, -1, N_max, -1)          # [B, N, N, H]
        hj = h.unsqueeze(1).expand(-1, N_max, -1, -1)          # [B, N, N, H]
        pair = torch.cat([hi + hj, hi * hj], dim=-1)           # [B, N, N, 2H]

        return self.bond_exist(pair).squeeze(-1), self.bond_type(pair), atom_mask


# ── 손실 계산 ──────────────────────────────────────────────────────────────────
def recon_loss(
    bond_exist_logit: torch.Tensor,
    bond_type_logit: torch.Tensor,
    atom_mask: torch.Tensor,
    batch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = bond_exist_logit.device
    B, N_max, _ = bond_exist_logit.shape

    # 정답 인접 행렬 [B, N_max, N_max]
    true_adj = to_dense_adj(batch.edge_index, batch=batch.batch, max_num_nodes=N_max)

    # 정답 결합 종류 (클래스 인덱스) [B, N_max, N_max]
    bond_class = batch.bond_type.argmax(dim=-1).float()
    true_bond_type = to_dense_adj(
        batch.edge_index, batch=batch.batch,
        edge_attr=bond_class.unsqueeze(-1),
        max_num_nodes=N_max,
    ).squeeze(-1).long()

    # 상삼각 마스크 (무방향, 자기 루프 제외)
    tri = torch.triu(torch.ones(N_max, N_max, dtype=torch.bool, device=device), diagonal=1)
    pair_mask = atom_mask.unsqueeze(2) & atom_mask.unsqueeze(1) & tri  # [B, N, N]

    # 결합 유무 손실 (결합 없는 쌍이 훨씬 많으므로 pos_weight로 보정)
    n_pairs    = pair_mask.sum().float()
    n_pos      = (true_adj * pair_mask).sum().float().clamp(min=1)
    pos_weight = ((n_pairs - n_pos) / n_pos).clamp(max=50.0)

    exist_loss = F.binary_cross_entropy_with_logits(
        bond_exist_logit[pair_mask],
        true_adj[pair_mask],
        pos_weight=pos_weight,
    )

    # 결합 종류 손실 (실제 결합 위치만)
    pos_mask  = pair_mask & (true_adj > 0.5)
    type_loss = (
        F.cross_entropy(bond_type_logit[pos_mask], true_bond_type[pos_mask])
        if pos_mask.any()
        else bond_exist_logit.new_tensor(0.0)
    )

    return exist_loss + type_loss, exist_loss, type_loss


# ── 전체 모델 ──────────────────────────────────────────────────────────────────
class MolCVAE(VGAE):
    """
    VGAE 상속 → reparametrize(), kl_loss() 재사용.
    mu_head / logstd_head 는 VAE 구조의 일부로 여기에 위치.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__(encoder=GraphEncoder())
        self.mu_head     = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.logstd_head = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.mol_decoder = MolDecoder()
        self.beta        = beta

    def encode(self, batch) -> torch.Tensor:
        h = self.encoder(batch)                                  # [B, HIDDEN]
        self.__mu__     = self.mu_head(h)
        self.__logstd__ = self.logstd_head(h).clamp(max=10.0)
        return self.reparametrize(self.__mu__, self.__logstd__)  # [B, LATENT]

    def forward(self, batch) -> dict[str, torch.Tensor]:
        """학습용 forward. loss dict 반환."""
        z = self.encode(batch)

        exist_logit, type_logit, atom_mask = self.mol_decoder(z, batch.p, batch.a)
        r_loss, e_loss, t_loss = recon_loss(exist_logit, type_logit, atom_mask, batch)
        kl = self.kl_loss()

        return {
            "loss":       r_loss + self.beta * kl,
            "recon_loss": r_loss,
            "exist_loss": e_loss,
            "type_loss":  t_loss,
            "kl_loss":    kl,
        }

    @torch.no_grad()
    def generate(
        self,
        p: torch.Tensor,
        a: torch.Tensor,
        threshold: float = 0.5,
    ) -> list[dict]:
        """
        분자 그래프 생성.
        p : [B, 5]  정규화된 물성 벡터 (stats.json 기준 z-score)
        a : [B, 11] 원자 조성 벡터
        반환: [{"edge_index": Tensor, "bond_types": Tensor, "n_atoms": int}, ...]
        """
        self.eval()
        z = torch.randn(p.size(0), LATENT_DIM, device=p.device)

        exist_logit, type_logit, atom_mask = self.mol_decoder(z, p, a)
        bond_prob = exist_logit.sigmoid()

        results = []
        for b in range(p.size(0)):
            n = atom_mask[b].sum().item()
            prob  = bond_prob[b, :n, :n]
            btype = type_logit[b, :n, :n].argmax(dim=-1)

            tri  = torch.triu(torch.ones(n, n, dtype=torch.bool, device=p.device), diagonal=1)
            edge = (prob > threshold) & tri

            src, dst = edge.nonzero(as_tuple=True)
            edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
            bond_types = torch.cat([btype[src, dst], btype[dst, src]])

            results.append({
                "edge_index": edge_index,
                "bond_types": bond_types,
                "n_atoms":    n,
            })

        return results
