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

| 단계 | 파일 | 등급 | 비고 |
|---|---|---|---|
| 1. CVAE | `model.py` | T2 | git 커밋됨, 미수정. 상단에 stage 주석 추가함. |
| 2. 초기 diffusion | `model2.py` | T2 | git 커밋됨, 미수정. |
| 데이터셋 재모델링 | `preprocess.py`, `data_preprocess.ipynb` | T2 | 방향족→단/이중/삼중 kekulize 수정(commit `5732605`), 미수정. |
| 3. Absorbing Discrete Graph Diffusion (R-4~C-4~A-1~B-1~C-1 누적) | `model3.py` | 현재 활성 개발 중 | 이 파일의 **최신 상태**가 선형 발전의 최종본. 중간 지점 참고용 스냅샷: `model3_clean.py`(G-5+G-6, A-1 이전). |

## 2. 조건부 생성 개선 실험 (A-1 이후 분기)

| 실험 | 결론 | 코드 | 등급 |
|---|---|---|---|
| A-1 (부분조건화, HOMO/LUMO 2개→7개 물성 확장) | **채택됨** — 실패 사례 아니라 현재 아키텍처의 기반. 별도 대조군 코드 불필요(`model3.py` 자체가 그 결과물). | `model3.py` (현재) | — |
| B-1 (circuit_rank 구조 feature) | null (Δ-CI 8개 지표 전부 CI 겹침) | `records/experiments/b1_circuit_rank/` | control: **T2**, feature arm: **T3(재구성)** |
| D-1 (A-1 vs control Δ-CI 재검증, CI-겹침 오류 수정 후) | 가설 기각(무조건부 물성분포가 데이터와 큰 차이 없음) | `records/experiments/b1_circuit_rank/diff_ci_a1_vs_control.py` (신규 학습 없음, B-1 체크포인트 재사용) | T2 |
| C-1 (고리 크기별 cycle feature) | null | `records/experiments/proposal_a/c1_experiment/` | **T2** (feature/control 모두 학습 시점 스냅샷) |
| E-1 (C-1 control을 20→40 epoch 연장, TPSA 재현 시도) | TPSA 재현 안 됨 — D-1 원신호는 multiple-comparisons artifact로 잠정 결론 | `records/experiments/proposal_a/c1_experiment/`(C-1 코드 재사용) + `runs/`(매니페스트) | **T1** |
| PEC (propose-evaluate-correct, 새 아키텍처) | 진행 중, 설계 1단계만 완료 — 결론 없음 | `records/experiments/pec/` | 착수 전 |
| S-1 (조건부 원자 수 예측기) | 착수 전(모듈 작성만 완료, 학습/평가 없음) | 미배치 | 착수 전 |

## 3. 읽는 순서 제안 (논문 Methods/Experiments 절 대응)

1. `model.py` → `model2.py` → `model3.py`: 아키텍처 발전사.
2. `preprocess.py`: 데이터 전처리(kekulization) 별도 절.
3. `records/experiments/b1_circuit_rank/README.md`, `.../proposal_a/c1_experiment/README.md`:
   각 실험의 가설/코드/결과 — 이 둘이 "무엇을 시도했고 왜 실패로 결론 내렸는가" 절의 근거.
4. E-1: `docs/devlog.md`(gitignored, 로컬 참고용) "2026-09-21(새벽): [E-1] 완료" 절 + `runs/` 매니페스트.

## 4. 알려진 한계

- B-1의 feature arm은 재구성이며 학습 당시 코드와 byte-identical하지 않음
  (기능적으로 동등함은 diff로 확인했으나, 이 사실 자체를 논문/부록에 명시할 것).
- `model3.py`는 계속 진화 중이므로, 이 문서의 "현재 상태" 서술은 커밋 시점 기준.
