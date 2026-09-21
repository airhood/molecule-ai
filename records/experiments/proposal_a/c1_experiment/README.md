# C-1 실험 재현 자료

astra_review_20260920.md §6 지적 반영 — 재현에 필요한 자료 전부 보존.

## 코드 스냅샷 (학습 시점)

- `model3_c1_feature_snapshot.py`: `molecule-AI-c1-feature/model3.py`
  (circuit-rank-free A-1 베이스 + `cycle_proj` zero-init 추가)
- `model3_c1_control_snapshot.py`: `molecule-AI-c1-control/model3.py`
  (circuit-rank-free A-1 베이스, 변경 없음)
- `train3_c1_snapshot.py`: 두 트리 공통(모델 구성 직후 재시드 포함,
  단 **reshuffle RNG 버그는 이 스냅샷엔 아직 있음** — astra_review_20260920.md
  §1에서 발견, `dataset2.py`/`train3.py`의 현재 버전에서 수정됨.
  이 실험 자체는 수정 전 버전으로 돌아갔다는 뜻)
- `gate_identity_check.py`: 6단계 identity gate 스크립트
- `../../c1_prop_regressor.py` (**추가, 2026-09-21**): `measure_ci_raw_c1.py`가
  import하는 독립 GNN 회귀기(`PropRegressor`). B-1/D-1 평가와 공유하는
  파일이라 실험별 폴더 밖 `records/experiments/`에 둔다. 이전엔 서버에만
  있고 로컬에 없었음 — 5개 서버 워크트리 전부 SHA256 동일 확인 후 추가.
- `measure_ci_raw_c1.py`: 8단계 conditioning 평가(타겟 층화 저장판,
  `errs_by_target`/`vals_by_target` 형식)
- `measure_ring56_raw_c1.py`: ring56Rate 평가(env `CKPT`/`OUT`로 사용,
  코드 자체는 B-1 때와 동일)
- `diff_ci_stratified.py`: 타겟 층화 Δ-CI 계산 스크립트(주의: docstring의
  "MAE" 표현은 부정확 — 실제로는 median absolute error임,
  astra_review_20260920.md §4 지적)

## 학습 로그

- `train_c1_feature.log`, `train_c1_control.log`: epoch별 T/V-loss
  (`--max-samples 150000` 빠뜨렸던 최초 실행의 흔적이 앞부분에 남아있음
  — "Train: 150,000" 두 번째 블록 이후가 실제 유효 실행)

## 실행 커맨드 (둘 다 동일 seed, 동일 하이퍼파라미터)

```bash
# feature (GPU0)
python -u train3.py --processed-dir ./data/processed_ext7 --max-samples 150000 \
  --init-weights ./checkpoints_c4_a1/best.pt --save-dir ./checkpoints_c4_c1_feature \
  --log-file ./train_c1_feature.log --epochs 20 --batch-size 12 --lr 3e-5 --seed 60

# control (GPU2, molecule-AI-c1-control/ 디렉토리, model3.py는 cycle_proj 없음)
python -u train3.py --processed-dir ./data/processed_ext7 --max-samples 150000 \
  --init-weights ./checkpoints_c4_a1/best.pt --save-dir ./checkpoints_c4_c1_control \
  --log-file ./train_c1_control.log --epochs 20 --batch-size 12 --lr 3e-5 --seed 60
```

## 체크포인트 SHA256 (2026-09-20 확인)

```
fbf5039b1b268c268034814582b76bdfc18ba08fa14c5e6a1eb43a8208eb0370  checkpoints_c4_c1_feature/best.pt
c14e1515f34eb01b453ed24a763b5e7cdc1eae648f8f6720a8c7bb61fbade1dd  checkpoints_c4_c1_control/best.pt
c6041051a844cb84f2f61653667387d392650038bf86ac8946a440e46dd22f41  checkpoints_c4_a1/best.pt (warm-start 원본)
```

## 알려진 한계 (astra_review_20260920.md 기준, 중요)

1. **reshuffle RNG 버그**: `train_set.reshuffle_indices()`가 이 실행
   당시 전역 NumPy RNG를 썼고, `torch.manual_seed(args.seed)`로는 안
   잡혀서 feature/control 두 프로세스의 청크 순서가 1에폭부터
   달랐다. 따라서 "동일 seed로 완전히 매치된 학습 루프"라는 주장은
   **부정확** — 실제로는 "학습 seed 하나만 형식적으로 맞춘, 여전히
   교란 가능성이 있는 비교"다. `dataset2.py`/`train3.py`의 현재
   버전(`reshuffle_seed` 인자 추가)은 이 버그를 고쳤지만, **이
   실험(c1_feature/c1_control) 자체는 고치기 전 버전으로 돌았다.**
2. 학습 seed 쌍이 하나뿐이라 architecture 간 차이와 학습 seed
   변동성을 분리할 수 없다 — 일반화하려면 최소 3개 seed 쌍 필요.
3. 8단계 평가 표에 validity/전체 고리수 등 사전 약속했던 지표가
   빠져 있었음 — astra_review_20260920.md §3에 보완 계산 있음(전부
   null 재확인).

**결론적 지위**: 이 실험은 "고리 크기 feature를 계속 우선순위 높게
탐색할 근거는 약하다"는 판단에는 충분하지만, "구조 feature 접근이
실패했다"를 확증하는 실험은 아니다.
