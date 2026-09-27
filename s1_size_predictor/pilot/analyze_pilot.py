"""[S-1 pilot] 분석 스크립트. 결과를 보기 전에 작성/고정한다(pilot_config.json의
evaluation 규칙 + astra_review_20260926.md §5 그대로). ledger.jsonl의 원본 attempt
기록만 입력으로 쓰며, 결과를 본 뒤 지표/분모/집계를 바꾸지 않는다.

규칙 요약
- validity: strict/single 모두 분모 = 전체 '완료된' 생성 시도(오류는 별도 보고).
- property 오차: single-valid 분자 전체에서 계산(주). strict-valid는 보조.
- 구조5개는 |오차|/train SD(normalization.json std_cond7)의 평균(고정 요약 점수),
  HOMO/LUMO는 GNN proxy raw MAE(Hartree)로 따로 표시. 부분 mask는 관측된 속성만.
- target별로 먼저 계산 -> target 간 동일 가중. 유효 생성물 0인 target은 결측(NaN)이며
  arm별 개수를 보고한다(조용히 삭제하지 않음).
- CI: target 단위 paired bootstrap(모든 arm에 같은 resample), arm별 독립 CI 겹침으로
  판정하지 않는다. S-1 3개 seed는 동일 가중 평균도 함께 보고하고 좋은 seed만 고르지 않는다.
"""
import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

# 관측되지 않은 속성의 지표는 의도적으로 전부 NaN(결측)이라 nanmean이 경고를 낸다 -- 예상된 동작.
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

STRUCT5 = ["LogP", "TPSA", "HBA", "RotBonds", "AromaticRings"]
PROXY = ["HOMO", "LUMO"]
BOOT_SEED = 20260926
N_BOOT = 10000


def load_completed(ledger_path):
    done, errors, started = {}, [], set()
    for line in open(ledger_path, encoding="utf-8"):
        ev = json.loads(line)
        if ev["event"] == "started":
            started.add(ev["attempt_id"])
        elif ev["event"] == "completed":
            done[ev["attempt_id"]] = ev          # 같은 id가 두 번이면 마지막(재개) 값
        elif ev["event"] == "error":
            errors.append(ev)
    return list(done.values()), errors, started


def per_target_metrics(records, targets_by_id, mask_observed, sd, mask_name):
    """records: 한 (arm, mask)의 completed 기록들. target별 지표 dict 반환."""
    obs = set(mask_observed)
    by_t = defaultdict(list)
    for r in records:
        by_t[r["target_id"]].append(r)
    out = {}
    for tid, rs in by_t.items():
        raw = targets_by_id[tid]["raw7"]
        n = len(rs)
        strict = [r for r in rs if r["strict_valid"]]
        single = [r for r in rs if r["single_valid"]]
        m = {"n_attempts": n, "strict_rate": len(strict) / n, "single_rate": len(single) / n,
             "n_single": len(single), "n_strict": len(strict)}

        def prop_err(rr, prop):
            v = (rr.get("props") or {}).get(prop)
            return None if v is None else abs(v - raw[prop])

        for label, pool in (("single", single), ("strict", strict)):
            per_prop = {}
            for p in STRUCT5 + PROXY:
                if p not in obs:
                    continue
                errs = [e for e in (prop_err(r, p) for r in pool) if e is not None]
                per_prop[p] = float(np.mean(errs)) if errs else float("nan")
            struct_obs = [p for p in STRUCT5 if p in obs]
            if struct_obs:
                per_mol = []
                for r in pool:
                    es = [prop_err(r, p) for p in struct_obs]
                    if all(e is not None for e in es):
                        per_mol.append(np.mean([e / sd[p] for e, p in zip(es, struct_obs)]))
                m[f"struct5_std_err_{label}"] = float(np.mean(per_mol)) if per_mol else float("nan")
            else:
                m[f"struct5_std_err_{label}"] = float("nan")
            for p, v in per_prop.items():
                m[f"mae_{p}_{label}"] = v
            for p in PROXY:
                if p in obs:
                    m[f"proxy_{p}_mae_{label}"] = per_prop.get(p, float("nan"))
        out[tid] = m
    return out


def target_mean(values):
    v = np.array(values, dtype=float)
    ok = ~np.isnan(v)
    return (float(v[ok].mean()) if ok.any() else float("nan")), int((~ok).sum())


def paired_bootstrap(arm_vals, target_ids, arms, baseline, key, n_boot=N_BOOT, seed=BOOT_SEED):
    """arm_vals[arm][tid] -> 지표 값(NaN 가능). target 단위 resample, 모든 arm에 같은 index.
    반환: {arm: {'diff_mean','ci_low','ci_high'}} (arm - baseline)."""
    rng = np.random.default_rng(seed)
    T = len(target_ids)
    mat = {a: np.array([arm_vals[a].get(t, np.nan) for t in target_ids], dtype=float) for a in arms}
    idx = rng.integers(0, T, size=(n_boot, T))
    res = {}
    def boot_mean(v):
        s = v[idx]                                  # (n_boot, T)
        with np.errstate(invalid="ignore"):
            return np.nanmean(s, axis=1)
    base = boot_mean(mat[baseline])
    obs_base = np.nanmean(mat[baseline])
    for a in arms:
        if a == baseline:
            continue
        b = boot_mean(mat[a])
        d = b - base
        d = d[~np.isnan(d)]
        res[a] = {"diff_mean_observed": float(np.nanmean(mat[a]) - obs_base),
                  "ci_low": float(np.percentile(d, 2.5)) if len(d) else float("nan"),
                  "ci_high": float(np.percentile(d, 97.5)) if len(d) else float("nan"),
                  "n_boot_valid": int(len(d))}
    return res


def analyze(records, errors, targets_doc, cfg, sd, arms):
    tby = {t["target_id"]: t for t in targets_doc["targets"]}
    masks = {cfg["masks"]["primary"]["name"]: cfg["masks"]["primary"]["observed"]}
    for m in cfg["masks"]["auxiliary"]:
        masks[m["name"]] = m["observed"]
    out = {"per_mask": {}, "errors": {"n": len(errors), "by_type": defaultdict(int)}}
    for e in errors:
        out["errors"]["by_type"][e["error_type"]] += 1
    out["errors"]["by_type"] = dict(out["errors"]["by_type"])

    for mname, observed in masks.items():
        recs_mask = [r for r in records if r["mask"] == mname]
        if not recs_mask:
            continue
        target_ids = sorted({r["target_id"] for r in recs_mask})
        per_arm_t = {}
        for arm in arms:
            rs = [r for r in recs_mask if r["arm"] == arm]
            per_arm_t[arm] = per_target_metrics(rs, tby, observed, sd, mname)
        metric_names = sorted({k for a in arms for t in per_arm_t[a].values() for k in t
                               if k not in ("n_attempts", "n_single", "n_strict")})
        summary = {}
        for metric in metric_names:
            vals = {a: {t: per_arm_t[a][t].get(metric, np.nan) for t in per_arm_t[a]} for a in arms}
            row = {}
            for a in arms:
                mean, n_missing = target_mean([vals[a].get(t, np.nan) for t in target_ids])
                row[a] = {"mean_over_targets": mean, "n_targets_missing": n_missing}
            s1_arms = [a for a in arms if a.startswith("s1_")]
            if s1_arms:
                eq = [row[a]["mean_over_targets"] for a in s1_arms]
                row["s1_equal_weight_mean_of_seeds"] = float(np.nanmean(eq)) if not np.all(np.isnan(eq)) else float("nan")
            row["paired_bootstrap_vs_prior"] = paired_bootstrap(vals, target_ids, arms, "prior", metric)
            summary[metric] = row
        # 크기/다양성/비용
        size_info = {}
        for arm in arms:
            rs = [r for r in recs_mask if r["arm"] == arm]
            sm = {"n_completed": len(rs),
                  "requested_size_mean": float(np.mean([r["requested_n_atoms"] for r in rs])),
                  "actual_heavy_mean_of_strict": float(np.mean([r["actual_heavy_atoms"] for r in rs
                                                                if r.get("actual_heavy_atoms") is not None]))
                  if any(r.get("actual_heavy_atoms") is not None for r in rs) else None,
                  "unique_valid_smiles": len({r["smiles"] for r in rs if r["single_valid"] and r.get("smiles")}),
                  "n_single_valid": sum(1 for r in rs if r["single_valid"]),
                  "elapsed_s_sum": float(sum(r.get("elapsed_s", 0) for r in rs))}
            size_info[arm] = sm
        out["per_mask"][mname] = {"target_ids": target_ids, "summary": summary, "size_and_cost": size_info}
    out["notes"] = ["결과는 pilot(탐색적 개입 실험)이며 검정력은 보장되지 않는다.",
                    "target은 S-1 val 분자라 A-1이 이미 학습한 분자일 수 있다.",
                    "HOMO/LUMO는 GNN proxy 오차이며 DFT 검증 오차가 아니다.",
                    "MAE만으로 채택하지 않는다: validity/오류/결측 target 수와 함께 해석."]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--normalization", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    targets = json.load(open(a.targets))
    norm = json.load(open(a.normalization))
    sd = dict(zip(norm["cond_prop_names"], norm["std_cond7"]))
    records, errors, started = load_completed(Path(a.run_dir) / "ledger.jsonl")
    res = analyze(records, errors, targets, cfg, sd, cfg["arms"])
    res["n_completed"] = len(records)
    res["n_started_not_completed"] = len(started - {r["attempt_id"] for r in records})
    out = Path(a.out) if a.out else Path(a.run_dir) / "analysis.json"
    if out.exists():
        sys.exit(f"이미 존재: {out} (덮어쓰기 금지)")
    out.write_text(json.dumps(res, indent=2, default=lambda o: None if o != o else o))
    for mname, blk in res["per_mask"].items():
        print(f"\n=== mask={mname} ===")
        for metric in ("strict_rate", "single_rate", "struct5_std_err_single"):
            if metric in blk["summary"]:
                row = blk["summary"][metric]
                print(f"{metric}: " + "  ".join(f"{a_}={row[a_]['mean_over_targets']:.4f}(miss={row[a_]['n_targets_missing']})"
                                                for a_ in cfg["arms"]))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
