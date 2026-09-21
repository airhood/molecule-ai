"""[Astra 리뷰 4번] Δ = median(|err_new|) - median(|err_control|)의 부트스트랩
95% CI를 직접 계산. 개별 CI 겹침 여부(잘못된 검정)가 아니라 차이 자체의 CI로
판정. 두 표본은 서로 다른 생성 분자들이라 독립(unpaired) 재표집."""
import sys
import numpy as np
import json


def diff_ci(errs_new, errs_control, n_boot=5000, seed=0):
    """독립 두 표본 median 차이의 부트스트랩 CI. 반환: (point_diff, ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    errs_new = np.asarray(errs_new)
    errs_control = np.asarray(errs_control)
    point = np.median(errs_new) - np.median(errs_control)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        rs_new = errs_new[rng.integers(0, len(errs_new), len(errs_new))]
        rs_ctrl = errs_control[rng.integers(0, len(errs_control), len(errs_control))]
        diffs[i] = np.median(rs_new) - np.median(rs_ctrl)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, lo, hi


def verdict(lo, hi):
    if hi < 0:
        return "new가 유의하게 더 낮음(개선, CI가 0 미포함)"
    if lo > 0:
        return "new가 유의하게 더 높음(악화, CI가 0 미포함)"
    return "차이 유의하지 않음(CI가 0 포함)"


if __name__ == "__main__":
    # 자체 검증: 두 정규분포에서 뽑은 합성 데이터로 방법론이 맞게 동작하는지 확인
    rng = np.random.default_rng(42)
    print("=== 자체 검증(synthetic) ===")
    # 케이스 1: 두 표본이 진짜 다름(new가 낮음) -> CI가 0을 배제해야 함
    a = np.abs(rng.normal(0.10, 0.03, 150))
    b = np.abs(rng.normal(0.18, 0.05, 150))
    p, lo, hi = diff_ci(a, b)
    print(f"case1(진짜 다름) point={p:.4f} CI=[{lo:.4f},{hi:.4f}]  {verdict(lo, hi)}")
    # 케이스 2: 두 표본이 같은 분포 -> CI가 0을 포함해야 함
    c = np.abs(rng.normal(0.15, 0.04, 150))
    d = np.abs(rng.normal(0.15, 0.04, 150))
    p, lo, hi = diff_ci(c, d)
    print(f"case2(같은 분포)  point={p:.4f} CI=[{lo:.4f},{hi:.4f}]  {verdict(lo, hi)}")
    # 케이스 3: 개별 CI는 겹치지만 차이 CI는 유의한 전형적 반례 상황 재현
    # (표본이 크고 분산이 작으면 개별 CI 겹침 + 차이 유의 동시 발생 가능)
    e = np.abs(rng.normal(0.100, 0.020, 400))
    f = np.abs(rng.normal(0.115, 0.020, 400))
    p, lo, hi = diff_ci(e, f)
    print(f"case3(개별CI 겹침 반례 재현) point={p:.4f} CI=[{lo:.4f},{hi:.4f}]  {verdict(lo, hi)}")
