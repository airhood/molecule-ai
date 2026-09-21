# molecule-AI

제72회 충북과학전람회 — 물성 기반 분자 구조 예측 딥러닝 모델

QMugs 데이터셋을 전처리하여 PyTorch Geometric `Dataset`으로 변환하고, 물성 조건을 입력받아 분자 그래프를 생성하는 Conditional VAE를 학습합니다.

---

## 모델 구조

**MoleculeCVAE** — GINEConv 인코더 + Transformer 디코더 기반 Conditional VAE

```
입력 분자 그래프
     │
     ▼
GraphEncoder (GINEConv × 4)
  원자 임베딩(64) + formal charge(1) + chirality(3) → HIDDEN(256)
  결합 특성: bond_type(4) + bond_stereo(5) + dihedral(1)
  global_mean_pool → 물성 p(5) + 원자 조성 a(11) 결합
     │
     ▼
mu_head / logstd_head → z ∈ R^128  (reparametrize)
     │
     ▼
MoleculeDecoder (MultiheadAttention × 4)
  z + p + a → 조건 벡터(128)
  원자 시퀀스 임베딩 + 조건 결합 → Transformer → 쌍별 특징
     │
     ├── bond_exist  : 결합 유무 이진 분류 (N×N)
     └── bond_type   : 결합 종류 4-class 분류 (N×N)
```

| 하이퍼파라미터 | 값 |
|---|---|
| Latent dim | 128 |
| Hidden dim | 256 |
| Atom emb dim | 64 |
| Encoder layers | 4 |
| Decoder layers | 4 |
| Bond types | 4 (단일/이중/삼중/방향족) |

---

## 요구사항

```
torch >= 2.0
torch_geometric >= 2.4
rdkit >= 2023.03
pandas >= 2.0
numpy >= 1.24
tqdm
requests
```

```bash
pip install torch torch_geometric rdkit pandas numpy tqdm requests
```

---

## 디렉토리 구조

```
molecule-AI/
├── download.py        # QMugs 데이터 다운로드
├── preprocess.py      # 전처리 → 청크 .pt 생성
├── dataset.py         # QMugsDataset 클래스
├── model.py           # MoleculeCVAE 모델 정의
├── train.py           # 학습 스크립트 (CLI)
├── train.ipynb        # 학습 노트북 (Colab용)
├── data/
│   ├── raw/
│   │   ├── summary.csv          # 물성 메타데이터
│   │   └── structures/          # 분자별 3D conformer SDF
│   └── processed/
│       ├── chunk_*.pt           # 전처리 완료 청크 파일
│       ├── meta.json            # 청크 목록 및 크기
│       └── stats.json           # 물성 z-score 정규화 통계
└── checkpoints/
    ├── best.pt                  # 최저 val loss 모델 가중치
    └── ckpt_epoch####.pt        # 주기 체크포인트
```

---

## 실행 순서

### 1단계 — 데이터 다운로드

```bash
python download.py --data-dir ./data/raw
```

- `summary.csv` (~수백 MB) 와 `structures.tar.gz` (~7 GB) 를 다운로드합니다.
- 이미 파일이 있으면 자동으로 건너뜁니다 (재실행 안전).
- 압축 해제 후 `structures.tar.gz` 는 자동 삭제됩니다.

### 2단계 — 전처리

```bash
python preprocess.py --raw-dir ./data/raw --out-dir ./data/processed
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--raw-dir` | `./data/raw` | 원본 데이터 경로 |
| `--out-dir` | `./data/processed` | 출력 경로 |
| `--workers` | CPU 코어 수 (최대 8) | 병렬 처리 워커 수 |

완료 시 아래 파일이 생성됩니다:

- `data/processed/chunk_*.pt` — torch_geometric `Data` 객체 청크 파일들
- `data/processed/meta.json` — 청크 목록 및 크기
- `data/processed/stats.json` — 물성 정규화 통계 (mean / std)

### 3단계 — 학습

```bash
python train.py \
    --processed-dir ./data/processed \
    --save-dir ./checkpoints \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-3 \
    --n-cycles 4 \
    --max-beta 1.0 \
    --warmup-epochs 10 \
    --save-every 10
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--epochs` | 100 | 총 학습 epoch 수 |
| `--batch-size` | 128 | 배치 크기 |
| `--lr` | 1e-3 | AdamW 학습률 |
| `--n-cycles` | 4 | KL beta 사이클 수 |
| `--max-beta` | 1.0 | KL beta 최대값 |
| `--warmup-epochs` | 10 | KL warmup epoch 수 (beta=0) |
| `--save-every` | 10 | 체크포인트 저장 주기 (0=비활성) |
| `--max-samples` | None | 빠른 실험용 데이터 수 제한 |

**KL 스케줄**: warmup 동안 beta=0 (순수 오토인코더 학습), 이후 코사인 사이클 방식으로 beta를 0→max_beta까지 반복 상승시킵니다.

Colab 환경에서는 `train.ipynb` 사용을 권장합니다.

---

## Dataset 사용법

```python
from dataset import QMugsDataset  # models_legacy/stage1_cvae/dataset.py (2026-09-21 이동)
from torch_geometric.loader import DataLoader

train_set = QMugsDataset("./data/processed", split="train")
val_set   = QMugsDataset("./data/processed", split="val")
test_set  = QMugsDataset("./data/processed", split="test")

loader = DataLoader(train_set, batch_size=128, shuffle=False, num_workers=0)

for batch in loader:
    print(batch.z)          # 원자 번호        [N]
    print(batch.charge)     # formal charge    [N]
    print(batch.chirality)  # R/S/none one-hot [N, 3]
    print(batch.edge_index) # 결합 인덱스       [2, E]
    print(batch.bond_type)  # bond type one-hot [E, 4]
    print(batch.bond_stereo)# stereo one-hot    [E, 5]
    print(batch.dihedral)   # 이면각 (rad)      [E]
    print(batch.p)          # 물성 (정규화)     [5]
    print(batch.p_raw)      # 물성 (원본)       [5]
    print(batch.a)          # 원자 조성         [11]
    break
```

### 조건 벡터 `p` (5차원)

| 인덱스 | 항목 | 단위 |
|---|---|---|
| 0 | E_HOMO (`DFT_HOMO_ENERGY`) | eV |
| 1 | E_LUMO (`DFT_LUMO_ENERGY`) | eV |
| 2 | E_gap (`DFT_HOMO_LUMO_GAP`) | eV |
| 3 | E_total (`DFT_TOTAL_ENERGY`) | Hartree |
| 4 | Dipole (`DFT_DIPOLE_TOT`) | Debye |

`p` 는 `__getitem__` 에서 `stats.json` 기준으로 z-score 정규화된 값입니다.  
`p_raw` 는 정규화 전 원본값입니다.

### 원자 조성 벡터 `a` (11차원)

```
[n_C, n_H, n_O, n_N, n_S, n_P, n_F, n_Cl, n_Br, n_I, N_total]
```

### split 설정

```python
QMugsDataset(processed_path, split="train", split_ratio=(0.8, 0.1, 0.1), seed=42)
```

- 기본 비율: train 80% / val 10% / test 10%
- `seed` 고정으로 재현 가능한 분할
- `max_samples` 로 빠른 실험용 데이터 수 제한 가능

---

## 분자 생성

```python
import torch
from model import MoleculeCVAE  # models_legacy/stage1_cvae/model.py (2026-09-21 이동)

model = MoleculeCVAE()
model.load_state_dict(torch.load("checkpoints/best.pt"))
model.eval()

# p: [B, 5] 물성 조건 (정규화), a: [B, 11] 원자 조성
results = model.generate(p, a, threshold=0.5)

for r in results:
    print(r["edge_index"])   # [2, E] 결합 인덱스
    print(r["bond_types"])   # [E]    결합 종류
    print(r["n_atoms"])      # int    원자 수
```

## Experiment records and safe launches

Start new training through the outer launcher so failures before Python startup are also recorded:

```bash
./launch_run.sh python -u train3.py [arguments...]
```

`RunLogger` writes a unique timestamped raw log and runtime manifest while preserving the existing `--log-file`. GPU logs, docs, source snapshots, manifests, and results are synchronized without checkpoints:

```bash
scripts/ops/start_record_sync.sh 120
scripts/ops/stop_record_sync.sh
```

Checkpoint binaries are copied only by an explicit manual decision. Update the remote inventory and chronological model catalog with:

```bash
scripts/ops/checkpoint_registry.py --skip-hash
scripts/ops/build_model_catalog.py
```

See `records/README.md` and `docs/project_structure_20260920.md` for the artifact layout and evidence policy.
