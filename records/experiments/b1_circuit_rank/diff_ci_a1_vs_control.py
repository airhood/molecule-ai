"""[D-1] review19.md 지시: A-1 vs control(circuit rank 없이 동일 예산
추가 학습만) Δ-CI 재평가 -- 재학습 없이 새 샘플 생성+채점만(raw_errors_a1.json,
raw_errors_control.json)으로 수행. Δ(A1-ctl) point>0 이면 A-1 오차가 더
크다는 뜻 -- 즉 control(추가 학습)이 더 정확함."""
import json

import numpy as np


def diff_ci(errs_new, errs_control, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    errs_new = np.asarray(errs_new)
    errs_control = np.asarray(errs_control)
    point = np.median(errs_new) - np.median(errs_control)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        rn = errs_new[rng.integers(0, len(errs_new), len(errs_new))]
        rc = errs_control[rng.integers(0, len(errs_control), len(errs_control))]
        diffs[i] = np.median(rn) - np.median(rc)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, lo, hi


if __name__ == "__main__":
    a1 = json.load(open("raw_errors_a1.json"))
    ctl = json.load(open("raw_errors_control.json"))

    print(f'{"물성":<14}{"A-1 med":>10}{"ctl med":>10}{"Δ(A1-ctl)":>11}{"Δ CI":>24}   판정')
    for name in a1["errs"]:
        ea1, ec = a1["errs"][name], ctl["errs"][name]
        if len(ea1) < 5 or len(ec) < 5:
            continue
        p, lo, hi = diff_ci(ea1, ec)
        if lo <= 0 <= hi:
            verdict = "차이 없음(0 포함)"
        elif hi < 0:
            verdict = "A-1이 유의하게 더 정확(control 악화)"
        else:
            verdict = "control이 유의하게 더 정확(추가학습으로 개선)"
        print(f"{name:<14}{np.median(ea1):>10.4f}{np.median(ec):>10.4f}{p:>11.4f}   [{lo:.4f},{hi:.4f}]   {verdict}")
