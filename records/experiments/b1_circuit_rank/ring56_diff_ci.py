"""ring56Rate(5/6원 고리 비율, 고리 단위 pooled ratio)의 Δ-CI.
분자 단위로 재표집(resample) -- 같은 분자 안 고리들은 독립이 아니므로."""
import json, sys
import numpy as np


def ring56_ratio(per_mol_ring_sizes):
    all_rings = [s for mol in per_mol_ring_sizes for s in mol]
    if not all_rings:
        return float("nan")
    return sum(1 for s in all_rings if s in (5, 6)) / len(all_rings)


def diff_ci_ring56(mols_new, mols_control, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    point = ring56_ratio(mols_new) - ring56_ratio(mols_control)
    n1, n2 = len(mols_new), len(mols_control)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx1 = rng.integers(0, n1, n1)
        idx2 = rng.integers(0, n2, n2)
        r1 = ring56_ratio([mols_new[j] for j in idx1])
        r2 = ring56_ratio([mols_control[j] for j in idx2])
        diffs[i] = r1 - r2
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, lo, hi


if __name__ == "__main__":
    b1 = json.load(open("/tmp/ring_raw_b1fixed.json"))
    ctl = json.load(open("/tmp/ring_raw_control.json"))
    mols_new = b1["per_mol_ring_sizes"]
    mols_ctl = ctl["per_mol_ring_sizes"]
    print(f"B1-fixed: n_valid_mols={len(mols_new)}  ring56Rate={ring56_ratio(mols_new):.4f}")
    print(f"control:  n_valid_mols={len(mols_ctl)}  ring56Rate={ring56_ratio(mols_ctl):.4f}")
    p, lo, hi = diff_ci_ring56(mols_new, mols_ctl)
    print(f"\nΔ(ring56Rate) = {p:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    if hi < 0:
        print("판정: B1-fixed가 유의하게 낮음(control이 더 좋음)")
    elif lo > 0:
        print("판정: B1-fixed가 유의하게 높음(circuit rank가 ring56Rate 개선)")
    else:
        print("판정: 차이 유의하지 않음(CI가 0 포함)")
