# B-1 실험 재현 자료 (circuit_rank 구조 feature)

A-1(부분조건화) 채택 직후 진행된 첫 구조-feature 추가 실험. `circuit_rank`
(독립 고리 개수, `E - V + C`)를 `conn_proj` 입력에 3번째 채널로 추가했을 때
conditioning 오차가 개선되는지 검증 — 결과는 null(Δ-CI 8개 지표 전부
CI 겹침), 다음 실험인 C-1(고리 "크기"별 세부 feature)로 이어짐.

## 코드

- `model3_b1_control_arm_snapshot.py`: **진짜 학습 시점 스냅샷** —
  `molecule-AI-control/model3.py`를 그대로 복사(SHA256
  `efef1ee4474f2d8867358b997b0c0a3a6cb02bf79209d5fde8b3b3d5e42523ad`).
  A-1 베이스 그대로, circuit_rank 없음. C-1 실험 때도 이 코드가 그대로
  재사용됐고(`molecule-AI-c1-control/model3.py`와 해시 완전히 동일),
  `records/experiments/proposal_a/c1_experiment/model3_c1_control_snapshot.py`와
  같은 파일이다 — 중복 보관이 아니라 같은 원본을 가리키는 두 경로.
- `model3_b1_feature_arm_RECONSTRUCTED.py`: **재구성 파일, 학습 시점
  스냅샷 아님.** circuit_rank를 실제로 켠 feature arm(`checkpoints_c4_b1_fixed`)은
  메인 트렁크(`molecule-AI/model3.py`)에서 직접 학습됐는데, 그 트렁크는
  B-1 이후에도 C-1이 바로 이어 붙어 계속 진화해 독립 스냅샷이 남지 않았다.
  이 파일은 2026-09-21에 **현재** `model3.py`에서 `[C-1]` 태그가 붙은
  블록(cycle_proj 초기화, `_ring_cycle_features`/`_scale_cycle_features`,
  forward의 cycle_struct 계산 및 `h` 합산)만 손으로 제거해 되돌린 것이다.
  파일 상단 docstring에 재구성 근거를 명시했다. 논문에는 "재구성, 원본
  학습 코드와 byte-identical 아님"으로 반드시 표기할 것.
- `diff_ci_a1_vs_control.py`는 D-1(A-1 vs control 재검증, review19.md 지시로
  CI-겹침 오류 수정 후 재계산 — devlog.md 2026-09-19 "[D-1] A-1 vs control
  Δ-CI 재검증")의 평가 스크립트도 겸한다. D-1은 새로 학습하지 않고 A-1/B-1
  control 체크포인트를 그대로 재사용했으므로 별도 model3.py 스냅샷 없음.
- `diff_ci.py`, `measure_ci_raw.py`,
  `measure_ring56_raw.py`, `ring56_diff_ci.py`: 평가/Δ-CI 스크립트.
- `raw_errors_a1.json`, `raw_errors_b1fixed.json`, `raw_errors_control.json`,
  `ring_raw_b1fixed.json`, `ring_raw_control.json`: 원본 오차 데이터.

## 의존성

`measure_ci_raw.py`는 `../c1_prop_regressor.py`(독립 GNN 회귀기, `PropRegressor`)를
import한다 — 이 파일은 B-1/C-1/D-1 평가 스크립트가 공유하는 의존성이라
`records/experiments/` 바로 아래(실험별 폴더 밖)에 둔다. 서버의 5개
워크트리 전부에서 SHA256 동일(`70cd21ef...`)함을 확인함 — 학습 코드와
달리 이 평가용 회귀기는 트리마다 갈라지지 않고 공유됨.

## 결과 요약

circuit_rank 추가는 conditioning 정확도(8개 물성 Δ-CI)와 ring56Rate 어느
쪽도 통계적으로 유의한 개선을 보이지 않음(null). 코드는 "구조 feature가
있어서 해될 게 없다"는 판단으로 트렁크에 유지되어 이후 C-1의 베이스가 됨.
