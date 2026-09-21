"""[C-1] 8단계: 타겟(quantile) 층화 Δ-CI. astra_review_20260919.md/review19.md
지적 반영 -- 5개 LUMO 분위수 타겟을 풀링하지 않고, 타겟별로 독립 재표집한 뒤
사전 지정 동일 가중치(단순 평균)로 결합. 이산 속성은 MAE(=|err| median과 동일
정보) 외에 exact-hit/±1-hit rate도 함께 보고(median의 계단형 둔감성 보완)."""
import json
import sys

import numpy as np


def stratified_point_and_ci(errs_by_target_new, errs_by_target_ctl, n_boot=5000, seed=0):
    """errs_by_target_*: list of list[float] (5개 타겟 그룹).
    각 bootstrap iter마다 타겟별로 독립 재표집 -> 타겟별 median -> 5개 평균."""
    rng = np.random.default_rng(seed)
    n_q = len(errs_by_target_new)

    def agg_median(groups):
        meds = [np.median(g) for g in groups if len(g) > 0]
        return np.mean(meds) if meds else np.nan

    point_new = agg_median(errs_by_target_new)
    point_ctl = agg_median(errs_by_target_ctl)
    point = point_new - point_ctl

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        rs_new = [np.asarray(g)[rng.integers(0, len(g), len(g))] if len(g) > 0 else np.array([])
                  for g in errs_by_target_new]
        rs_ctl = [np.asarray(g)[rng.integers(0, len(g), len(g))] if len(g) > 0 else np.array([])
                  for g in errs_by_target_ctl]
        diffs[i] = agg_median(rs_new) - agg_median(rs_ctl)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, lo, hi


def exact_pm1_hit(vals_by_target):
    flat = [v for q in vals_by_target for v in q]
    if not flat:
        return float("nan"), float("nan")
    exact = sum(1 for val, tgt in flat if round(val) == round(tgt)) / len(flat)
    pm1 = sum(1 for val, tgt in flat if abs(round(val) - round(tgt)) <= 1) / len(flat)
    return exact, pm1


DISCRETE_PROPS = {"HBA", "RotBonds", "AromaticRings"}


def main(path_new, path_ctl):
    new = json.load(open(path_new))
    ctl = json.load(open(path_ctl))

    print(f'{"물성":<14}{"new(전체n)":>11}{"ctl(전체n)":>11}{"Δ point":>10}{"Δ CI":>22}   판정')
    for name in new["errs_by_target"]:
        eg_new = new["errs_by_target"][name]
        eg_ctl = ctl["errs_by_target"][name]
        n_new = sum(len(g) for g in eg_new)
        n_ctl = sum(len(g) for g in eg_ctl)
        if n_new < 5 or n_ctl < 5:
            print(f"{name:<14} 표본 부족")
            continue
        p, lo, hi = stratified_point_and_ci(eg_new, eg_ctl)
        if lo <= 0 <= hi:
            verdict = "차이 없음(0 포함)"
        elif hi < 0:
            verdict = "new(feature)가 유의하게 더 정확"
        else:
            verdict = "ctl(control)이 유의하게 더 정확"
        extra = ""
        if name in DISCRETE_PROPS:
            ex_new, pm1_new = exact_pm1_hit(new["vals_by_target"][name])
            ex_ctl, pm1_ctl = exact_pm1_hit(ctl["vals_by_target"][name])
            extra = f"  [exact-hit new={ex_new:.3f} ctl={ex_ctl:.3f}, ±1-hit new={pm1_new:.3f} ctl={pm1_ctl:.3f}]"
        print(f"{name:<14}{n_new:>11}{n_ctl:>11}{p:>10.4f}   [{lo:.4f},{hi:.4f}]   {verdict}{extra}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
