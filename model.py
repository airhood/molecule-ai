import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, VGAE, global_mean_pool
from torch_geometric.utils import to_dense_adj

# C, H, O, N, S, P, F, Cl, Br, I
ELEM_ATOMIC_NUMS = [6, 1, 8, 7, 16, 15, 9, 17, 35, 53]
N_ELEM = 10

LATENT_DIM = 128
HIDDEN_DIM = 256
ATOM_EMB_DIM = 64
COND_DIM = 128
ENCODER_LAYERS = 4
DECODER_LAYERS = 4
N_BOND_TYPES = 4 # 1중 / 2중 / 3중 / 컨쥬게이션

# atom_embedding + charge + chirality
NODE_DIM = ATOM_EMB_DIM + 1 + 3
# bond_type + bond_stereo + dihedral
EDGE_DIM = 4 + 5 + 1


def mlp(dims, dropout=0.0):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:
            if dropout:
                layers += [nn.ReLU(), nn.Dropout(dropout)]
            else:
                layers += [nn.ReLU()]
    return nn.Sequential(*layers)

def to_atom_tensors(a_counts: torch.Tensor, device: torch.device):
    atom_lists = []
    for row in a_counts:
        atoms = []
        for atomic_num, cnt in zip(ELEM_ATOMIC_NUMS, row[:N_ELEM].long().tolist()):
            atoms.extend([atomic_num] * max(cnt, 0))
        if not atoms:
            atoms = [6]  # fallback: 빈 분자 방지
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


class GraphEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.atom_emb = nn.Embedding(120, ATOM_EMB_DIM)
        self.atom_proj = mlp([NODE_DIM, HIDDEN_DIM])

        self.convs = nn.ModuleList([
            GINEConv(
                mlp([HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM]),
                edge_dim=EDGE_DIM,
            )
            for _ in range(ENCODER_LAYERS)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(HIDDEN_DIM) for _ in range(ENCODER_LAYERS)])

        self.pool_proj = mlp([HIDDEN_DIM + 5 + N_ELEM, HIDDEN_DIM])

    def forward(self, batch):
        h = torch.cat(
            [self.atom_emb(batch.z), batch.charge.unsqueeze(-1), batch.chirality],
            dim=-1,
        )
        h = self.atom_proj(h)

        edge_attr = torch.cat(
            [batch.bond_type, batch.bond_stereo, batch.dihedral.unsqueeze(-1)],
            dim=-1,
        )

        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.relu(conv(h, batch.edge_index, edge_attr)))

        g = global_mean_pool(h, batch.batch)
        B = g.size(0)

        return self.pool_proj(torch.cat([g, batch.p.view(B, -1), batch.a_bin.view(B, -1)], dim=-1))


class MoleculeDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.atom_emb = nn.Embedding(120, ATOM_EMB_DIM)
        self.cond_proj = mlp([LATENT_DIM + 5 + N_ELEM, COND_DIM])
        self.init_proj = mlp([ATOM_EMB_DIM + COND_DIM, HIDDEN_DIM])

        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(HIDDEN_DIM, num_heads=8, batch_first=True)
            for _ in range(DECODER_LAYERS)
        ])
        self.attn_norms = nn.ModuleList([nn.LayerNorm(HIDDEN_DIM) for _ in range(DECODER_LAYERS)])
        self.ffns = nn.ModuleList([
            mlp([HIDDEN_DIM, HIDDEN_DIM * 2, HIDDEN_DIM])
            for _ in range(DECODER_LAYERS)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(HIDDEN_DIM) for _ in range(DECODER_LAYERS)])

        # Positional encoding: 동일 원소 원자들을 위치로 구별
        self.pos_emb = nn.Embedding(200, HIDDEN_DIM)

        # AdaLN: z → (γ_attn, β_attn, γ_ffn, β_ffn) per layer
        # zero-init: 초기에는 γ=0, β=0 → 표준 LayerNorm과 동일하게 시작
        self.z_adaln = nn.ModuleList([
            nn.Linear(LATENT_DIM, HIDDEN_DIM * 4)
            for _ in range(DECODER_LAYERS)
        ])
        for layer in self.z_adaln:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        pair_dim = HIDDEN_DIM * 2
        self.bond_exist = mlp([pair_dim, HIDDEN_DIM, 1])
        self.bond_type = mlp([pair_dim, HIDDEN_DIM, N_BOND_TYPES])

    def forward(self, z, p, a_bin, a_counts, a_bin_cond=None):
        device = z.device
        if a_bin_cond is None:
            a_bin_cond = a_bin

        c = self.cond_proj(torch.cat([z, p, a_bin_cond], dim=-1))  # [B, COND_DIM]

        atom_idx, atom_mask = to_atom_tensors(a_counts, device)  # [B, N_max]
        N_max = atom_idx.size(1)

        h_atom = self.atom_emb(atom_idx)  # [B, N_max, ATOM_EMB_DIM]
        c_exp = c.unsqueeze(1).expand(-1, N_max, -1)  # [B, N_max, COND_DIM]
        h = self.init_proj(torch.cat([h_atom, c_exp], dim=-1))  # [B, N_max, HIDDEN_DIM]

        h = h + self.pos_emb(torch.arange(N_max, device=device))

        key_pad_mask = ~atom_mask
        for adaln, attn, a_norm, ffn, f_norm in zip(
            self.z_adaln, self.attn_layers, self.attn_norms, self.ffns, self.ffn_norms):
            g_attn, b_attn, g_ffn, b_ffn = adaln(z).chunk(4, dim=-1)  # 각 [B, HIDDEN]
            h2, _ = attn(h, h, h, key_padding_mask=key_pad_mask)
            h = a_norm(h + h2) * (1 + g_attn.unsqueeze(1)) + b_attn.unsqueeze(1)
            h = f_norm(h + ffn(h)) * (1 + g_ffn.unsqueeze(1)) + b_ffn.unsqueeze(1)

        hi = h.unsqueeze(2).expand(-1, -1, N_max, -1)  # [B, N_max, N_max, H]
        hj = h.unsqueeze(1).expand(-1, N_max, -1, -1)
        pair = torch.cat([hi + hj, hi * hj], dim=-1)

        return self.bond_exist(pair).squeeze(-1), self.bond_type(pair), atom_mask


def reconstruction_loss(
    bond_exist_logit,
    bond_type_logit,
    atom_mask,
    batch
):
    device = bond_exist_logit.device
    B, N_max, _ = bond_exist_logit.shape

    # PyG to_dense_adj는 중복 엣지(u,v), (v,u)의 속성을 합산하므로 한쪽 방향(u < v)만 추출
    edge_mask = batch.edge_index[0] < batch.edge_index[1]
    half_ei = batch.edge_index[:, edge_mask]
    
    # 0(결합없음)과 구분하기 위해 클래스에 1을 더해줌 (1:단일, 2:이중, 3:삼중, 4:방향족)
    half_bt = batch.bond_type[edge_mask].argmax(dim=-1) + 1

    # [B, N_max, N_max] 크기의 정답 행렬 생성
    true_adj_type = to_dense_adj(half_ei, batch=batch.batch, edge_attr=half_bt, max_num_nodes=N_max).squeeze(-1)
    
    true_adj = (true_adj_type > 0).float() # 결합 존재 여부
    true_bond_type = (true_adj_type - 1).clamp(min=0).long() # 원래 클래스(0~3)로 복구

    tri = torch.triu(torch.ones(N_max, N_max, dtype=torch.bool, device=device), diagonal=1)
    pair_mask = atom_mask.unsqueeze(2) & atom_mask.unsqueeze(1) & tri  # [B, N_max, N_max]

    n_pairs = pair_mask.sum().float()
    n_pos = (true_adj * pair_mask).sum().float().clamp(min=1)
    pos_weight = ((n_pairs - n_pos) / n_pos).clamp(max=50)

    exist_loss = F.binary_cross_entropy_with_logits(
        bond_exist_logit[pair_mask],
        true_adj[pair_mask],
        pos_weight=pos_weight,
    )

    pos_mask = pair_mask & (true_adj > 0.5)
    type_loss = (
        F.cross_entropy(bond_type_logit[pos_mask], true_bond_type[pos_mask])
        if pos_mask.any()
        else bond_exist_logit.new_tensor(0.0)
    )

    return exist_loss + type_loss, exist_loss, type_loss


def free_bits_kl(mu, logstd, lambda_=0.1):
    # 차원별 KL을 lambda_ 이상으로 강제해 posterior collapse 방지
    kl_per_dim = -0.5 * (1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp())
    return kl_per_dim.clamp(min=lambda_).sum(dim=-1).mean()


class MoleculeCVAE(VGAE):

    def __init__(self, beta=1.0, a_dropout=0.15, a_pred_weight=1.0):
        super().__init__(encoder=GraphEncoder())
        self.mu_head = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.logstd_head = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.a_pred_head = mlp([LATENT_DIM + 5 + N_ELEM, HIDDEN_DIM, N_ELEM])
        self.molecule_decoder = MoleculeDecoder()
        self.beta = beta
        self.a_dropout = a_dropout
        self.a_pred_weight = a_pred_weight

    def encode(self, batch):
        h = self.encoder(batch)
        self.__mu__ = self.mu_head(h)
        self.__logstd__ = self.logstd_head(h).clamp(max=10.0)
        return self.reparametrize(self.__mu__, self.__logstd__)

    def forward(self, batch):
        h = self.encoder(batch)
        self.__mu__ = self.mu_head(h)
        self.__logstd__ = self.logstd_head(h).clamp(max=10.0)
        z = self.__mu__ if self.beta == 0 else self.reparametrize(self.__mu__, self.__logstd__)
        B = z.size(0)
        p = batch.p.view(B, -1)
        a_bin = batch.a_bin.view(B, -1)           # [B, 10] — 원소 종류 (0/1)
        a_counts = batch.a.view(B, -1)[:, :N_ELEM] # [B, 10] — 원소 개수 (teacher forcing)

        a_bin_cond = torch.zeros_like(a_bin) if (self.training and torch.rand(1).item() < self.a_dropout) else a_bin

        a_pred = self.a_pred_head(torch.cat([z, p, a_bin], dim=-1))  # [B, 10]
        a_pred_loss = F.mse_loss(a_pred, a_counts.float())

        exist_logit, type_logit, atom_mask = self.molecule_decoder(z, p, a_bin, a_counts, a_bin_cond=a_bin_cond)
        r_loss, e_loss, t_loss = reconstruction_loss(exist_logit, type_logit, atom_mask, batch)
        kl = free_bits_kl(self.__mu__, self.__logstd__)

        return {
            "loss": r_loss + self.beta * kl + self.a_pred_weight * a_pred_loss,
            "reconstruction_loss": r_loss,
            "exist_loss": e_loss,
            "type_loss": t_loss,
            "kl_loss": kl,
            "a_pred_loss": a_pred_loss
        }

    @torch.no_grad()
    def generate(self, p, a_bin, threshold=0.5):
        self.eval()
        B = p.size(0)
        z = torch.randn(B, LATENT_DIM, device=p.device)

        a_pred = self.a_pred_head(torch.cat([z, p, a_bin], dim=-1))
        a_counts = a_pred.round().clamp(min=0)  # [B, 10]

        atom_types_list = []
        for row in a_counts:
            atoms = []
            for atomic_num, cnt in zip(ELEM_ATOMIC_NUMS, row.long().tolist()):
                atoms.extend([atomic_num] * max(cnt, 0))
            if not atoms:
                atoms = [6]
            atom_types_list.append(atoms)

        exist_logit, type_logit, atom_mask = self.molecule_decoder(z, p, a_bin, a_counts)
        bond_prob = exist_logit.sigmoid()

        results = []
        for b in range(B):
            n = atom_mask[b].sum().item()
            prob = bond_prob[b, :n, :n]
            btype = type_logit[b, :n, :n].argmax(dim=-1)

            tri = torch.triu(torch.ones(n, n, dtype=torch.bool, device=p.device), diagonal=1)
            edge = (prob > threshold) & tri

            u, v = edge.nonzero(as_tuple=True)
            edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])])
            bond_types = torch.cat([btype[u, v], btype[v, u]])

            results.append({
                "edge_index": edge_index,
                "bond_types": bond_types,
                "n_atoms": n,
                "atom_types": atom_types_list[b][:n]
            })

        return results
