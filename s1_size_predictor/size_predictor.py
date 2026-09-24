"""[S-1] 조건부 원자 수 모델. astra_architecture_proposals_20260920.md §2 스펙.
기존 denoiser(A-1)는 전혀 건드리지 않는 완전히 독립적인 작은 모델 --
property values + mask -> p(n_atoms). 학습/생성 파이프라인에 통합하기
전에, 먼저 "크기 선택이 conditioning 오차의 구조적 하한인가"를
검증하는 용도.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

COND_DIM = 7  # model3.COND_DIM과 동일(HOMO,LUMO,LogP,TPSA,HBA,RotBonds,AromaticRings)
MAX_ATOMS = 50
MIN_ATOMS = 2
N_SIZE_CLASSES = MAX_ATOMS - MIN_ATOMS + 1  # 2..50 -> 49 classes
HIDDEN = 128


class SizePredictor(nn.Module):
    """property-token encoder(속성별 개별 임베딩+mask) -> mixer -> 2-layer MLP
    -> n_atoms(2..50) 위의 분포. 이산/연속 속성 구분해서 인코딩."""

    def __init__(self, cond_dim=COND_DIM, hidden=HIDDEN):
        super().__init__()
        self.cond_dim = cond_dim
        # 속성별 독립 scalar encoder(작은 MLP) + null(mask=0) embedding
        self.value_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            for _ in range(cond_dim)
        ])
        self.type_embed = nn.Embedding(cond_dim, hidden)
        self.null_embed = nn.Parameter(torch.zeros(cond_dim, hidden))
        self.mixer = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, N_SIZE_CLASSES)
        )

    def forward(self, cond, cond_mask):
        """cond: (B, cond_dim) 정규화된 물성값(관측 안 된 자리는 값 무관, mask로 처리).
        cond_mask: (B, cond_dim) 0/1. 반환: logits (B, N_SIZE_CLASSES)."""
        B = cond.shape[0]
        tokens = []
        for j in range(self.cond_dim):
            v = self.value_encoders[j](cond[:, j:j + 1])
            t = v + self.type_embed(torch.full((B,), j, dtype=torch.long, device=cond.device))
            m = cond_mask[:, j:j + 1]
            null_j = self.null_embed[j].unsqueeze(0).expand(B, -1)
            tok = torch.where(m.bool(), t, null_j)
            tokens.append(tok)
        stacked = torch.stack(tokens, dim=1)  # (B, cond_dim, hidden)
        pooled = stacked.mean(dim=1)  # masked_mean 아님 -- null_embed 자체가 "모른다"를 표현하므로 단순 평균으로 충분
        z = self.mixer(pooled)
        return self.head(z)

    def sample(self, cond, cond_mask, temperature=1.0, top_k=None):
        """온도/top-k 샘플링으로 다양성 유지(astra 지적 -- argmax 고정 금지)."""
        logits = self.forward(cond, cond_mask) / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            thresh = v[:, -1:].expand_as(logits)
            logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)
        probs = F.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, 1).squeeze(-1)
        return idx + MIN_ATOMS  # class index -> 실제 원자 수


class SimpleConcatMLP(nn.Module):
    """astra_review_20260922.md §1 -- SizePredictor(속성별 encoder+mean
    pooling, 175K params)가 이 문제에 정말 필요한지 판단할 강한 단순
    기준선. concat([properties * mask, mask]) -> 2-layer MLP. model3.py의
    cond_proj/cond_mod가 쓰는 입력 형식(값*마스크, 마스크 concat)과
    동일한 관례."""

    def __init__(self, cond_dim=COND_DIM, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * cond_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, N_SIZE_CLASSES),
        )

    def forward(self, cond, cond_mask):
        x = torch.cat([cond * cond_mask, cond_mask], dim=-1)
        return self.net(x)

    def sample(self, cond, cond_mask, temperature=1.0, top_k=None):
        logits = self.forward(cond, cond_mask) / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            thresh = v[:, -1:].expand_as(logits)
            logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)
        probs = F.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, 1).squeeze(-1)
        return idx + MIN_ATOMS
