"""[S-1 pilot] 공용 함수: SHA, RNG key 유도, 크기 추첨(inverse CDF), 고정 모듈 import.

pilot_config.json의 rng/size_sampling 규칙을 그대로 구현한다. Python의 hash()는
프로세스마다 달라지므로 쓰지 않는다.
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

MIN_ATOMS, MAX_ATOMS = 2, 50
N_CLASSES = MAX_ATOMS - MIN_ATOMS + 1
PROP_ORDER = ["HOMO", "LUMO", "LogP", "TPSA", "HBA", "RotBonds", "AromaticRings"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_key(master_seed, target_id, mask_name, repeat, purpose):
    """안정된 64bit 정수 key. purpose in {'size_uniform','denoise'}.
    arm은 key에 넣지 않는다 -> 같은 (target,mask,repeat)의 모든 arm이 같은 U와 같은
    denoising seed를 공유(pairing, pilot_config.json size_sampling.pairing)."""
    s = f"{master_seed}|{target_id}|{mask_name}|{repeat}|{purpose}"
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big") % (2 ** 63)


def draw_uniform(key):
    """크기 전용 CPU generator로 U in [0,1) 한 개. 전역 RNG를 건드리지 않는다."""
    g = torch.Generator(device="cpu")
    g.manual_seed(key)
    return float(torch.rand(1, generator=g, dtype=torch.float64).item())


def size_from_uniform(probs, u):
    """probs: 길이 49(원자수 2..50)의 확률. inverse CDF로 원자수 하나를 결정."""
    cdf = np.cumsum(np.asarray(probs, dtype=np.float64))
    idx = int(np.searchsorted(cdf, u, side="left"))
    idx = min(idx, N_CLASSES - 1)  # cdf[-1]이 부동소수점으로 1 미만일 때 끝 class로 클램프
    return idx + MIN_ATOMS


def mask_vector(mask_spec_observed):
    v = [1.0 if n in set(mask_spec_observed) else 0.0 for n in PROP_ORDER]
    return v


def import_pinned(frozen_dir):
    """고정 디렉터리에서 model3/train3/qm_props를 import하고 (모듈들, 원래 cwd)를 반환.
    qm_props는 import 시 cwd 기준 상대경로 'c1_regressor.pt'를 읽고 sys.path를
    바꾸므로 cwd를 고정 디렉터리로 잠깐 옮겼다가 복원한다. train3를 qm_props보다
    먼저 import해야 qm_props의 sys.path 삽입이 train3 해석을 바꾸지 않는다."""
    frozen_dir = str(frozen_dir)
    orig = os.getcwd()
    sys.path.insert(0, frozen_dir)
    import model3
    import train3
    os.chdir(frozen_dir)
    try:
        import qm_props
    finally:
        os.chdir(orig)
    import c1_prop_regressor
    return model3, train3, qm_props, c1_prop_regressor
