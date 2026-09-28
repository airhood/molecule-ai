"""[S-1 pilot] 분석 스크립트. 결과를 보기 전에 작성/고정한다(pilot_config.json의
evaluation 규칙 + astra_review_20260926.md §5 반영, astra_review_20260927.md P1-4/P2
수정 포함). ledger.jsonl의 원본 attempt 기록만 입력으로 쓴다.

astra_review_20260927.md P1-4 수정 사항:
1. target 전체 모집단은 targets.json(사전 선정된 16개)에서 고정 -- "완료 기록이 하나라도
   있는 target"이 아니라, 완료가 0개인 target도 결측으로 명시 카운트한다.
2. arm별 "descriptive" 평균(자기 own 평가 가능 target 기준)과, baseline(prior) 대비
   "paired" 비교(두 arm 모두 평가 가능한 공통 target에서만 계산)를 분리한다. 공통 target이
   0개면 diff/CI를 계산하지 않고 None으로 둔다.
3. S-1 3-seed 동일 가중 평균은 "세 seed 모두 평가 가능한" 공통 target에서만 계산하고,
   그 target 수/제외된 seed를 명시한다(참여 seed 수를 몰래 줄이지 않음).
4. validity는 pooled rate(전체 완료 수 대비)와 target-macro rate(target별 비율의 평균)를
   구분해서 둘 다 보고한다.
5. planned(=config로부터 결정적으로 계산)/completed/errored/missing을 마스크별로 집계하고,
   전부 완료되지 않았으면 status="partial"로 표시한다.

P2 수정: launch_manifest identity 검증, 재귀적 NaN/Inf -> null 정규화 후 allow_nan=False로 직렬화,
분석기 자체 SHA/입력 SHA/timestamp를 결과에 기록.
"""
import argparse
import hashlib
import json
import math
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

STRUCT5 = ["LogP", "TPSA", "HBA", "RotBonds", "AromaticRings"]
PROXY = ["HOMO", "LUMO"]
BOOT_SEED = 20260926
N_BOOT = 10000


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitize(obj):
    """재귀적으로 non-finite float(NaN/Inf)를 None으로 바꾼다(P2 -- json.dumps의
    default=는 이미 float인 NaN을 가로채지 않아 비표준 JSON이 나오던 문제)."""
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def load_ledger_safely(ledger_path):
    """P2 -- 마지막 줄이 중간에 끊겨도(전원 장애 등) 그 앞까지는 그대로 읽는다.
    끊긴 줄은 버리지 않고 별도로 보고한다(조용히 무시 금지)."""
    lines = open(ledger_path, encoding="utf-8").read().split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    events, corrupt_tail = [], None
    for i, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                corrupt_tail = line
            else:
                raise   # 중간 줄 손상은 조용히 넘기지 않고 즉시 실패
    return events, corrupt_tail


def load_completed(ledger_path):
    done, errors, started = {}, [], set()
    events, corrupt_tail = load_ledger_safely(ledger_path)
    for ev in events:
        if ev["event"] == "started":
            started.add(ev["attempt_id"])
        elif ev["event"] == "completed":
            done[ev["attempt_id"]] = ev
        elif ev["event"] == "error":
            errors.append(ev)
    return list(done.values()), errors, started, corrupt_tail


def per_target_metrics(records, targets_by_id, mask_observed, sd):
    """records: 한 (arm, mask)의 completed 기록들. target별 지표 dict 반환(완료 기록이
    있는 target에 한함 -- 완료 0개 target은 여기 안 나타나고 상위에서 전체 target
    universe와 대조해 결측으로 처리)."""
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


def descriptive_mean(all_target_ids, values_by_target):
    """자기 own 평가 가능 target(NaN 아닌 값)에서만 평균 -- 다른 arm과 무관한 기술 통계."""
    vals = np.array([values_by_target.get(t, np.nan) for t in all_target_ids], dtype=float)
    ok = ~np.isnan(vals)
    n_evaluable = int(ok.sum())
    return {
        "mean_over_own_evaluable_targets": float(vals[ok].mean()) if n_evaluable else None,
        "n_targets_evaluable": n_evaluable,
        "n_targets_missing": int(len(all_target_ids) - n_evaluable),
    }


def paired_vs_baseline(all_target_ids, values_by_arm, arm, baseline, n_boot=N_BOOT, seed=BOOT_SEED):
    """[astra_review_20260927.md P1-4 수정] arm과 baseline 둘 다 평가 가능한(NaN 아닌)
    공통 target에서만 target-단위 paired bootstrap. 공통 target이 0개면 계산하지 않고
    None + n_common_targets=0을 반환 -- 서로 다른 성공 집단의 평균 차이를 paired 효과로
    보고하지 않는다(Astra 반례: diff=10, CI=[10,10] with 0 common targets)."""
    a = np.array([values_by_arm[arm].get(t, np.nan) for t in all_target_ids], dtype=float)
    b = np.array([values_by_arm[baseline].get(t, np.nan) for t in all_target_ids], dtype=float)
    common_mask = ~np.isnan(a) & ~np.isnan(b)
    common_targets = [t for t, ok in zip(all_target_ids, common_mask) if ok]
    n_common = len(common_targets)
    if n_common == 0:
        return {"n_common_targets": 0, "common_target_ids": [], "diff_mean": None,
                "ci_low": None, "ci_high": None, "n_boot_valid": 0}
    diffs = a[common_mask] - b[common_mask]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_common, size=(n_boot, n_common))
    boot = diffs[idx].mean(axis=1)
    return {"n_common_targets": n_common, "common_target_ids": common_targets,
            "diff_mean": float(diffs.mean()), "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)), "n_boot_valid": int(n_boot)}


def s1_equal_weight_mean(all_target_ids, values_by_arm, s1_arms):
    """[P1-4 수정] 세 seed 모두 평가 가능한(NaN 아닌) 공통 target에서만 계산. 어떤
    seed든 그 target에서 결측이면 그 target은 통째로 빠진다(참여 seed 수를 몰래
    줄이지 않음 -- Astra 지적: nanmean이 결측 seed를 조용히 빼던 문제)."""
    mats = {a: np.array([values_by_arm[a].get(t, np.nan) for t in all_target_ids], dtype=float)
            for a in s1_arms}
    common_mask = np.all([~np.isnan(mats[a]) for a in s1_arms], axis=0)
    n_common = int(common_mask.sum())
    fully_missing_seeds = [a for a in s1_arms if np.isnan(mats[a]).all()]
    if n_common == 0:
        return {"mean": None, "n_common_targets": 0, "n_seeds": len(s1_arms),
                "fully_missing_seeds": fully_missing_seeds}
    per_target_seed_mean = np.mean([mats[a][common_mask] for a in s1_arms], axis=0)
    return {"mean": float(per_target_seed_mean.mean()), "n_common_targets": n_common,
            "n_seeds": len(s1_arms), "fully_missing_seeds": fully_missing_seeds}


def validity_rates(all_target_ids, records_by_arm_mask):
    """[P1-4 수정] pooled(전체 완료 수 대비)와 target-macro(target별 비율의 평균)를
    분리해서 보고 -- 중단/오류로 target별 완료 수가 다르면 두 값이 달라질 수 있다."""
    rs = records_by_arm_mask
    n = len(rs)
    if n == 0:
        return {"pooled_strict_rate": None, "pooled_single_rate": None,
                "target_macro_strict_rate": None, "target_macro_single_rate": None,
                "n_completed_total": 0}
    pooled_strict = sum(1 for r in rs if r["strict_valid"]) / n
    pooled_single = sum(1 for r in rs if r["single_valid"]) / n
    by_t = defaultdict(list)
    for r in rs:
        by_t[r["target_id"]].append(r)
    strict_rates = [sum(1 for r in v if r["strict_valid"]) / len(v) for v in by_t.values()]
    single_rates = [sum(1 for r in v if r["single_valid"]) / len(v) for v in by_t.values()]
    return {"pooled_strict_rate": pooled_strict, "pooled_single_rate": pooled_single,
            "target_macro_strict_rate": float(np.mean(strict_rates)) if strict_rates else None,
            "target_macro_single_rate": float(np.mean(single_rates)) if single_rates else None,
            "n_completed_total": n, "n_targets_with_any_completion": len(by_t)}


def analyze(records, errors, all_target_ids, targets_doc, cfg, sd, arms, repeats):
    tby = {t["target_id"]: t for t in targets_doc["targets"]}
    masks = {cfg["masks"]["primary"]["name"]: cfg["masks"]["primary"]["observed"]}
    for m in cfg["masks"]["auxiliary"]:
        masks[m["name"]] = m["observed"]
    out = {"per_mask": {}, "errors": {"n": len(errors), "by_type": defaultdict(int)}}
    for e in errors:
        out["errors"]["by_type"][e["error_type"]] += 1
    out["errors"]["by_type"] = dict(out["errors"]["by_type"])
    s1_arms = [a for a in arms if a.startswith("s1_")]

    for mname, observed in masks.items():
        recs_mask = [r for r in records if r["mask"] == mname]
        per_arm_t = {arm: per_target_metrics([r for r in recs_mask if r["arm"] == arm], tby, observed, sd)
                     for arm in arms}
        metric_names = sorted({k for a in arms for t in per_arm_t[a].values() for k in t
                               if k not in ("n_attempts", "n_single", "n_strict")})
        summary = {}
        for metric in metric_names:
            vals = {a: {t: per_arm_t[a][t][metric] for t in per_arm_t[a] if metric in per_arm_t[a][t]}
                    for a in arms}
            row = {a: descriptive_mean(all_target_ids, vals[a]) for a in arms}
            row["paired_vs_prior"] = {a: paired_vs_baseline(all_target_ids, vals, a, "prior")
                                      for a in arms if a != "prior"}
            if s1_arms:
                row["s1_equal_weight"] = s1_equal_weight_mean(all_target_ids, vals, s1_arms)
            summary[metric] = row

        # validity: pooled vs target-macro, 별도 처리(다른 metric들과 통계 형태가 달라 분리)
        validity = {arm: validity_rates(all_target_ids, [r for r in recs_mask if r["arm"] == arm])
                    for arm in arms}

        planned_per_arm = len(all_target_ids) * repeats
        counts = {}
        for arm in arms:
            n_completed = sum(1 for r in recs_mask if r["arm"] == arm)
            # attempt_id 형식: f"{target_id}|{mask}|r{repeat}|{arm}" (build_plan 참조)
            n_errored = sum(1 for e in errors if e["attempt_id"].split("|")[1:2] == [mname]
                            and e["attempt_id"].split("|")[-1] == arm)
            counts[arm] = {"planned": planned_per_arm, "completed": n_completed, "errored": n_errored,
                          "missing": planned_per_arm - n_completed - n_errored}
        status = "complete" if all(c["completed"] + c["errored"] == c["planned"] for c in counts.values()) else "partial"

        size_info = {}
        for arm in arms:
            rs = [r for r in recs_mask if r["arm"] == arm]
            size_info[arm] = {
                "n_completed": len(rs),
                "requested_size_mean": float(np.mean([r["requested_n_atoms"] for r in rs])) if rs else None,
                "actual_heavy_mean_of_valid": (float(np.mean([r["actual_heavy_atoms"] for r in rs
                                                if r.get("actual_heavy_atoms") is not None]))
                                                if any(r.get("actual_heavy_atoms") is not None for r in rs) else None),
                "unique_valid_smiles": len({r["smiles"] for r in rs if r["single_valid"] and r.get("smiles")}),
                "n_single_valid": sum(1 for r in rs if r["single_valid"]),
                "elapsed_s_sum": float(sum(r.get("elapsed_s", 0) for r in rs)),
            }
        out["per_mask"][mname] = {
            "status": status, "counts_per_arm": counts, "target_universe_size": len(all_target_ids),
            "validity": validity, "summary": summary, "size_and_cost": size_info,
        }
    out["notes"] = ["결과는 pilot(탐색적 개입 실험)이며 검정력은 보장되지 않는다.",
                    "target은 S-1 val 분자라 A-1이 이미 학습한 분자일 수 있다.",
                    "HOMO/LUMO는 GNN proxy 오차이며 DFT 검증 오차가 아니다.",
                    "paired_vs_prior는 두 arm 모두 평가 가능한 공통 target에서만 계산되며, "
                    "n_common_targets=0이면 diff/CI가 None이다.",
                    "descriptive 평균은 해당 arm 자신의 평가 가능 target 기준이라 arm마다 "
                    "모집단이 다를 수 있다 -- paired_vs_prior와 혼동하지 않는다."]
    return out


def verify_run_identity(run_dir, cfg_path, pins_path, targets_path, norm_path):
    """[P2] 분석 대상 run이 실제로 이 config/pins/targets/normalization으로 만들어졌는지
    launch_manifest.json과 대조. 불일치면 즉시 중단(잘못된 target/SD로 재분석 금지)."""
    lm_path = Path(run_dir) / "launch_manifest.json"
    if not lm_path.exists():
        raise RuntimeError(f"launch_manifest.json 없음: {lm_path} -- 이 run의 신원을 확인할 수 없음")
    lm = json.load(open(lm_path))
    checks = {
        "config_sha256": sha256_file(cfg_path), "pins_sha256": sha256_file(pins_path),
        "targets_sha256": sha256_file(targets_path),
    }
    for k, v in checks.items():
        if lm.get(k) != v:
            raise RuntimeError(f"run identity 불일치: launch_manifest.{k}={lm.get(k)} != 실제 {v}")
    return lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--normalization", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pins", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    lm = verify_run_identity(a.run_dir, a.config, a.pins, a.targets, a.normalization)
    cfg = json.load(open(a.config))
    targets_doc = json.load(open(a.targets))
    norm = json.load(open(a.normalization))
    sd = dict(zip(norm["cond_prop_names"], norm["std_cond7"]))
    all_target_ids = sorted(t["target_id"] for t in targets_doc["targets"])

    records, errors, started, corrupt_tail = load_completed(Path(a.run_dir) / "ledger.jsonl")
    res = analyze(records, errors, all_target_ids, targets_doc, cfg, sd, cfg["arms"],
                  cfg["repeats_per_target_mask_arm"])
    res["n_completed"] = len(records)
    res["n_started_not_completed"] = len(started - {r["attempt_id"] for r in records})
    res["ledger_corrupt_tail_line"] = corrupt_tail
    res["provenance"] = {
        "analyzer_script_sha256": sha256_file(__file__), "analyzed_at": time.time(),
        "run_dir": str(Path(a.run_dir).resolve()), "launch_manifest_run_id": lm.get("run_id"),
        "config_sha256": sha256_file(a.config), "pins_sha256": sha256_file(a.pins),
        "targets_sha256": sha256_file(a.targets), "normalization_sha256": sha256_file(a.normalization),
    }

    out = Path(a.out) if a.out else Path(a.run_dir) / "analysis.json"
    if out.exists():
        sys.exit(f"이미 존재: {out} (덮어쓰기 금지)")
    out.write_text(json.dumps(sanitize(res), indent=2, allow_nan=False))
    for mname, blk in res["per_mask"].items():
        print(f"\n=== mask={mname} (status={blk['status']}) ===")
        for arm, c in blk["counts_per_arm"].items():
            print(f"  {arm:<14} planned={c['planned']} completed={c['completed']} "
                  f"errored={c['errored']} missing={c['missing']}")
        if "struct5_std_err_single" in blk["summary"]:
            row = blk["summary"]["struct5_std_err_single"]
            for arm in cfg["arms"]:
                d = row[arm]
                print(f"    struct5_std_err_single[{arm}]: mean={d['mean_over_own_evaluable_targets']} "
                      f"(n_evaluable={d['n_targets_evaluable']}, missing={d['n_targets_missing']})")
    if corrupt_tail is not None:
        print(f"\n[경고] ledger 마지막 줄이 손상됨(무시하고 진행): {corrupt_tail[:80]}...")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
