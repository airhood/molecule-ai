# 논문용 코드 계보 지도

`docs/model_lineup.md`(전체 개발사, gitignored)의 요약판. 여기 나온 파일들은
전부 git으로 추적된다. 각 항목에 **보존 등급**을 표기했다:

- **T1 (증명됨)**: 실행 당시 명령/체크포인트/소스 해시가 매니페스트로 남아
  코드가 그 실행과 정확히 일치함이 증명됨.
- **T2 (검증됨)**: 별도 디렉터리에 고정된 파일로, mtime 또는 git 상태로
  학습 완료 시점 이후 수정되지 않았음이 확인됨. 매니페스트는 없음.
- **T3 (재구성)**: 학습 시점 스냅샷이 없어, 현재 코드에서 이후 추가분을
  손으로 제거해 되돌린 것. 원본과 byte-identical 아님 — 파일 자체에도
  명시.

## 1. 모델 계보 (선형 — 최종 코드만 보존하면 되는 단계)

**2026-09-21: 실제로 디렉터리 분리함** (이전엔 주석만 추가하고 파일 위치는
그대로 둬서 "정리"가 아니었음 — 지적받고 수정).

| 단계 | 위치 | 등급 | 비고 |
|---|---|---|---|
| 1. CVAE | `models/stage1_cvae/` (`model.py`, `train.py`, `dataset.py`, `test_inference.py`, `model_.py`, `train_.py`) | T2 | git 커밋됨, 미수정. `model_.py`/`train_.py`는 별도 변형(class명 `MolCVAE`), 용도 불명확하지만 자기완결적이라 같이 이동. |
| 2. 초기 diffusion | `models/stage2_diffusion_v1/` (`model2.py`, `train2.py`, `test_inference_diffusion.py`, `model2_.py`, `train2_.py`) | T2 | git 커밋됨, 미수정. `dataset2.py`는 저장소 루트에 공유 파일로 남아있어(model3.py도 사용) 이 폴더의 `train2*.py`/`test_inference_diffusion.py`에 `sys.path` 보정 3줄 추가함. |
| 데이터셋 재모델링 | `dataset_remodeling/` (`preprocess.py`, `data_preprocess.ipynb`) | T2 | 방향족→단/이중/삼중 kekulize 수정(commit `5732605`), 미수정, 이동만 함. |
| 3. Absorbing Discrete Graph Diffusion (R-4~C-4~A-1~B-1~C-1 누적) | 저장소 루트 `model3.py` | 현재 활성 개발 중 | **이동 안 함** — `train3.py` 및 `records/experiments/`의 다수 스크립트가 루트 기준 경로로 import해서 옮기면 서버 학습 파이프라인이 깨짐. 이 파일의 최신 상태가 선형 발전의 최종본. 중간 지점 참고용 스냅샷: `model3_clean.py`(G-5+G-6, A-1 이전). |

`dataset2.py`, `train3.py`, `test_inference_absorbing.py`도 같은 이유로
루트에 남겨둠 — model3.py와 공유되거나 그 자체가 활성 파이프라인.

## 2. 조건부 생성 개선 실험 (A-1 이후 분기)

| 실험 | 결론 | 코드 | 등급 |
|---|---|---|---|
| A-1 (부분조건화, HOMO/LUMO 2개→7개 물성 확장) | **채택됨** — 실패 사례 아니라 현재 아키텍처의 기반. 별도 대조군 코드 불필요(`model3.py` 자체가 그 결과물). | `model3.py` (현재) | — |
| B-1 (circuit_rank 구조 feature) | null (Δ-CI 8개 지표 전부 CI 겹침) | `records/experiments/b1_circuit_rank/` | control: **T2**, feature arm: **T2** (2026-09-21 정정 — 서버 메인 트렁크가 C-1 이후 갱신 안 돼 B-1 시점 그대로 남아있던 진짜 스냅샷 발견, 재구성본 대체) |
| D-1 (A-1 vs control Δ-CI 재검증, CI-겹침 오류 수정 후) | 가설 기각(무조건부 물성분포가 데이터와 큰 차이 없음) | `records/experiments/b1_circuit_rank/diff_ci_a1_vs_control.py` (신규 학습 없음, B-1 체크포인트 재사용) | T2 |
| C-1 (고리 크기별 cycle feature) | null | `records/experiments/proposal_a/c1_experiment/` | **T2** (feature/control 모두 학습 시점 스냅샷) |
| E-1 (C-1 control을 20→40 epoch 연장, TPSA 재현 시도) | TPSA 재현 안 됨 — D-1 원신호는 multiple-comparisons artifact로 잠정 결론 | `records/experiments/proposal_a/c1_experiment/`(C-1 코드 재사용) + `runs/`(매니페스트) | **T1** |
| PEC (propose-evaluate-correct, 새 아키텍처) | 진행 중, 설계 1단계만 완료 — 결론 없음 | `records/experiments/pec/` | 착수 전 |
| S-1 (조건부 원자 수 예측기) | 착수 전(모듈 작성만 완료, 학습/평가 없음) | 미배치 | 착수 전 |

## 3. 읽는 순서 제안 (논문 Methods/Experiments 절 대응)

1. `models/stage1_cvae/model.py` → `models/stage2_diffusion_v1/model2.py` → `model3.py`: 아키텍처 발전사.
2. `dataset_remodeling/preprocess.py`: 데이터 전처리(kekulization) 별도 절.
3. `records/experiments/b1_circuit_rank/README.md`, `.../proposal_a/c1_experiment/README.md`:
   각 실험의 가설/코드/결과 — 이 둘이 "무엇을 시도했고 왜 실패로 결론 내렸는가" 절의 근거.
4. E-1: `docs/devlog.md`(gitignored, 로컬 참고용) "2026-09-21(새벽): [E-1] 완료" 절 + `runs/` 매니페스트.

## 4. 알려진 한계

- `model3.py`는 계속 진화 중이므로, 이 문서의 "현재 상태" 서술은 커밋 시점 기준.
- **서버 메인 트렁크(`~/molecule-AI/model3.py`) 자체가 낡음**: C-1(2026-09-19)
  이후 한 번도 갱신되지 않아 B-1 시점(circuit_rank만 있음) 그대로 멈춰있다.
  이번엔 그 덕분에 B-1 feature arm 진짜 스냅샷을 거기서 건졌지만, 반대로
  다음에 메인 트렁크에서 새로 학습을 돌리면 C-1 코드가 빠진 채로 돌아갈
  위험이 있다 — 로컬 `model3.py`를 서버 메인 트렁크에 반영할지는 아직
  사용자 확인 대기 중.
