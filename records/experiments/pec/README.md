# PEC (propose-evaluate-correct) — 설계 1단계, 결론 없음

Astra 제안 새 아키텍처. 2026-09-21 기준 서버에 전용 워크트리
(`molecule-AI-pec`)만 만들어졌고, 설계 확정 단계에서 3개 근본 결함
(원본 제안의 "달성된 물성" 라벨 오류, 조건 워터마크 누설 위험,
teacher-forced vs reverse-chain 분포 불일치)이 발견되어 TPSA-단일로
스코프를 좁혀 재설계 중 — 아직 학습 코드 작성/실행 전.

## 파일

- `manifest.json`: PEC 기반 모듈 해시(설계 1단계 산출물).
- `model3_pec_base_snapshot.py`, `train3_pec_base_snapshot.py`: PEC
  워크트리의 **현재** `model3.py`/`train3.py` — 아직 PEC 고유 수정이
  없어 `model3.py`는 B-1 control arm(=C-1 control arm)과 SHA256
  완전히 동일(`efef1ee4...`). PEC 코드가 실제로 추가되기 전 baseline
  기준점으로 보존.

## 의존성 (참고, PEC 전용 아님)

`molecule-AI-pec`에는 `c1_prop_regressor.py`(→ `../c1_prop_regressor.py`로
이미 보존)와 `qm_props.py`(→ `../qm_props.py`)도 있었지만, 확인 결과
5개 서버 워크트리 전부에 동일하게 존재하는 `web_server.py`/`viewer.html`
지원 유틸리티(GNN 대리모델 물성 추정 + DFT + 시각화)라 PEC이 만든 코드가
아님. PEC 폴더 밖(`records/experiments/`)에 공용으로 둔다.
