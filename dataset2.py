import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class QMugsDataset(Dataset):
    def __init__(
        self,
        processed_path,
        split: str = "train",
        split_ratio: tuple = (0.8, 0.1, 0.1),
        seed: int = 42,
        cache_chunks: int = 2,
        max_samples: int = None,
    ):
        processed_path = Path(processed_path)
        self._processed_path = processed_path

        with open(processed_path / "meta.json") as f:
            meta = json.load(f)

        self._chunk_names = meta["chunk_files"]
        chunk_sizes = meta["chunk_sizes"]

        self._chunk_offsets = np.cumsum([0] + chunk_sizes[:-1])
        self._total = sum(chunk_sizes)

        rng = np.random.default_rng(seed)
        perm = rng.permutation(self._total)

        train_end = int(self._total * split_ratio[0])
        val_end = train_end + int(self._total * split_ratio[1])

        splits = {
            "train": perm[:train_end],
            "val":   perm[train_end:val_end],
            "test":  perm[val_end:],
        }
        if split not in splits:
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")
        split_indices = splits[split]

        # 청크 순서 셔플 + 청크 내 셔플 → DataLoader shuffle=False로 cache hit 보장
        chunk_of = np.searchsorted(self._chunk_offsets, split_indices, side="right") - 1
        chunk_order = rng.permutation(len(self._chunk_names))
        ordered = []
        for c in chunk_order:
            mask = chunk_of == c
            if mask.any():
                items = split_indices[mask].copy()
                rng.shuffle(items)
                ordered.append(items)
        self.indices = np.concatenate(ordered) if ordered else split_indices
        if max_samples is not None:
            self.indices = self.indices[:max_samples]

        with open(processed_path / "stats.json") as f:
            stats = json.load(f)
        prop_cols = list(stats.keys())
        self.mean = torch.tensor([stats[c]["mean"] for c in prop_cols], dtype=torch.float)
        self.std  = torch.tensor([stats[c]["std"]  for c in prop_cols], dtype=torch.float)

        self._load_chunk = lru_cache(maxsize=cache_chunks)(self._load_chunk_uncached)

        # 원자 수 사전계산 — size bucketing용 (init 1회, 이후 reshuffle에서 재사용)
        sizes_path = processed_path / "sizes.npy"
        if sizes_path.exists():
            all_sizes = np.load(sizes_path)
            self._sizes = all_sizes[self.indices]
        else:
            self._sizes = self._build_sizes()

        # reshuffle용 chunk 소속 캐시 — 매 에폭 searchsorted 재계산 방지
        self._chunk_of = np.searchsorted(self._chunk_offsets, self.indices, side="right") - 1

    def _load_chunk_uncached(self, chunk_idx: int) -> list:
        return torch.load(
            self._processed_path / self._chunk_names[chunk_idx],
            weights_only=False,
        )

    def _build_sizes(self) -> np.ndarray:
        chunk_of = np.searchsorted(self._chunk_offsets, self.indices, side="right") - 1
        sizes = np.empty(len(self.indices), dtype=np.int32)
        for c in np.unique(chunk_of):
            chunk_data = self._load_chunk(c)
            for pos in np.where(chunk_of == c)[0]:
                local = int(self.indices[pos]) - int(self._chunk_offsets[c])
                sizes[pos] = int((chunk_data[local].z != 1).sum())
        return sizes

    def reshuffle_indices(self):
        # 청크 순서는 랜덤, 청크 내부는 원자 수 오름차순 → size bucketing
        chunk_order = np.random.permutation(len(self._chunk_names))
        new_order = []
        for c in chunk_order:
            mask = self._chunk_of == c
            if mask.any():
                pos = np.where(mask)[0]
                sort = np.argsort(self._sizes[pos], kind="stable")
                new_order.append(pos[sort])
        if new_order:
            perm = np.concatenate(new_order)
            self.indices = self.indices[perm]
            self._sizes = self._sizes[perm]
            self._chunk_of = self._chunk_of[perm]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        global_idx = int(self.indices[idx])
        chunk_idx = int(np.searchsorted(self._chunk_offsets, global_idx, side="right") - 1)
        local_idx = global_idx - int(self._chunk_offsets[chunk_idx])
        data = self._load_chunk(chunk_idx)[local_idx].clone()
        data.p = (data.p_raw - self.mean) / self.std
        data.a_bin = (data.a[:10] > 0).float()

        # Heavy-atom only: remove H (atomic num 1)
        heavy_mask = data.z != 1
        if not heavy_mask.any():
            heavy_mask[0] = True
        heavy_idx = heavy_mask.nonzero(as_tuple=True)[0]

        old_to_new = torch.full((len(data.z),), -1, dtype=torch.long)
        old_to_new[heavy_idx] = torch.arange(len(heavy_idx), dtype=torch.long)

        ei = data.edge_index
        edge_mask = heavy_mask[ei[0]] & heavy_mask[ei[1]]
        data.z = data.z[heavy_idx]
        data.edge_index = old_to_new[ei[:, edge_mask]]
        data.bond_type = data.bond_type[edge_mask]
        data.num_nodes = len(heavy_idx)

        return data
