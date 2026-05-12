import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, VGAE, global_mean_pool
from torch_geometric.utils import to_dense_adj

# C, H, O, N, S, P, F, Cl, Br, I
ELEM_ATOMIC_NUMS = [6, 1, 8, 7, 16, 15, 9, 17, 35, 53]

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


def mlp(dims, dropout = 0.0):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:
            if dropout:
                layers += [nn.ReLU(), nn.Dropout(dropout)]
            else:
                layers += [nn.ReLU()]
    return nn.Sequential(*layers)

def to_atom_tensors(a: torch.Tensor, device: torch.device):
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

        self.pool_proj = mlp([HIDDEN_DIM + 5 + 11, HIDDEN_DIM])

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

        return self.pool_proj(torch.cat([g, batch.p.view(B, -1), batch.a.view(B, -1)], dim=-1))
    

class MoleculeDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.atom_emb = nn.Embedding(120, ATOM_EMB_DIM)
        self.cond_proj = mlp([LATENT_DIM + 5 + 11, COND_DIM])
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

        pair_dim = HIDDEN_DIM * 2
        self.bond_exist = mlp([pair_dim, HIDDEN_DIM, 1])
        self.bond_type = mlp([pair_dim, HIDDEN_DIM, N_BOND_TYPES])

    def forward(self, z, p, a):
        device = z.device

        c = self.cond_proj(torch.cat([z, p, a], dim=-1)) # [B, COND_DIM]


        atom_idx, atom_mask = to_atom_tensors(a, device) # [B, N_max]
        N_max = atom_idx.size(1)

        h_atom = self.atom_emb(atom_idx) # [B, N_max, ATOM_EMB_DIM]
        c_exp = c.unsqueeze(1).expand(-1, N_max, -1) # [B, N_max, COND_DIM]
        h = self.init_proj(torch.cat([h_atom, c_exp], dim=-1)) # [B, N_max, HIDDEN_DIM]

        # atom_mask : True(원자), False(패딩)
        # key_pad_mask (MultiHeadAttention) : True(패딩), False(원자)
        key_pad_mask = ~atom_mask
        for attn, a_norm, ffn, f_norm in zip(
            self.attn_layers, self.attn_norms, self.ffns, self.ffn_norms):
            h2, _ = attn(h, h, h, key_padding_mask=key_pad_mask)
            h = a_norm(h + h2)
            h = f_norm(h + ffn(h))

        hi = h.unsqueeze(2).expand(-1, -1, N_max, -1) # [B, N_max, N_max, H]
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

    true_adj = to_dense_adj(batch.edge_index, batch=batch.batch, max_num_nodes=N_max)

    # (B, N_max, N_max)
    bond_class = batch.bond_type.argmax(dim=-1).float()
    true_bond_type = to_dense_adj(
        batch.edge_index, batch=batch.batch,
        edge_attr=bond_class.unsqueeze(-1),
        max_num_nodes=N_max,
    ).squeeze(-1).long()

    tri = torch.triu(torch.ones(N_max, N_max, dtype=torch.bool, device=device), diagonal=1)
    pair_mask = atom_mask.unsqueeze(2) & atom_mask.unsqueeze(1) & tri # [B, N_max, N_max]

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

    def __init__(self, beta = 1.0):
        super().__init__(encoder=GraphEncoder())
        self.mu_head = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.logstd_head = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.molecule_decoder = MoleculeDecoder()
        self.beta = beta

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
        a = batch.a.view(B, -1)

        exist_logit, type_logit, atom_mask = self.molecule_decoder(z, p, a)
        r_loss, e_loss, t_loss = reconstruction_loss(exist_logit, type_logit, atom_mask, batch)
        kl = free_bits_kl(self.__mu__, self.__logstd__)

        return {
            "loss": r_loss + self.beta * kl,
            "reconstruction_loss": r_loss,
            "exist_loss": e_loss,
            "type_loss": t_loss,
            "kl_loss": kl
        }
    
    @torch.no_grad()
    def generate(self, p, a, threshold = 0.5):
        self.eval()
        B = p.size(0)
        z = torch.randn(B, LATENT_DIM, device=p.device)

        exist_logit, type_logit, atom_mask = self.molecule_decoder(z, p, a)
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
                "n_atoms": n
            })

        return results
