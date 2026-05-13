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

    def _load_chunk_uncached(self, chunk_idx: int) -> list:
        return torch.load(
            self._processed_path / self._chunk_names[chunk_idx],
            weights_only=False,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        global_idx = int(self.indices[idx])
        chunk_idx = int(np.searchsorted(self._chunk_offsets, global_idx, side="right") - 1)
        local_idx = global_idx - int(self._chunk_offsets[chunk_idx])
        data = self._load_chunk(chunk_idx)[local_idx].clone()
        data.p = (data.p_raw - self.mean) / self.std
        data.a_bin = (data.a[:10] > 0).float()
        return data
