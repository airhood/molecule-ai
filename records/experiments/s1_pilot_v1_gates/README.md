# pilot_v1 launch gate 실행 기록 (비생성 게이트, CPU 전용)

- `20260927T002424...`: 첫 실행. 화면의 개별 게이트는 전부 PASS였지만 최종 판정이 FAILED로
  기록됨 -- `passed=None`(not_run) 항목을 실패로 세는 **게이트 코드 버그** 때문. 결과를
  바꾸지 않고 그대로 보존한다(`all_non_generation_gates_passed=false`는 버그 산물).
- `20260927T002456...`: 버그 수정(None 제외, not_run 별도 목록) 후 재실행. 비생성 게이트 전부 PASS,
  not_run = rng_reproducibility_same_attempt, forced_size_prior_vs_s1_same_a1_output
  (sampling 필요 -> Astra 재검토/사용자 승인 후 별도 실행). `launch_ready=false`.
