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
import json
import math
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from pilot_common import ledger_state, read_ledger_safely, sha256_file  # noqa: E402

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

STRUCT5 = ["LogP", "TPSA", "HBA", "RotBonds", "AromaticRings"]
PROXY = ["HOMO", "LUMO"]
BOOT_SEED = 20260926
N_BOOT = 10000


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


def load_completed(ledger_path):
    """[astra_review_20260928.md P1-3, opus_review_20260928.md 1-2 수정] attempt_id별
    "최신 상태"만 집계한다(pilot_common.ledger_state). 이전 버전은 error 이벤트 개수를
    그대로 세어, "error -> 재시도 -> completed"로 끝난 attempt가 completed와 error
    양쪽에서 중복으로 잡혀 missing이 음수가 되는 버그가 있었다(두 리뷰 공통 반례).
    completed만 지표 계산에 쓰고, evaluation_failed(평가가 깨져 completed로 승격 못한
    attempt)는 별도로 돌려줘 신뢰 못 하는 데이터가 지표에 섞이지 않게 한다."""
    events, corrupt_tail = read_ledger_safely(ledger_path)
    state = ledger_state(events)
    latest = state["latest"]
    done = [ev for ev in latest.values() if ev["event"] == "completed"]
    eval_failed = [ev for ev in latest.values() if ev["event"] == "evaluation_failed"]
    errored = [ev for ev in latest.values() if ev["event"] == "error"]
    started = {ev["attempt_id"] for ev in events if ev["event"] == "started"}
    return done, eval_failed, errored, started, corrupt_tail, latest


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
    """자기 own 평가 가능 target(유한값)에서만 평균 -- 다른 arm과 무관한 기술 통계.
    [astra_review_20260928.md P2] ~isnan은 Inf를 유효 관측으로 센다(NaN만 걸러지고
    Inf는 그대로 평균에 들어가 지표를 오염시킬 수 있음) -- isfinite로 NaN/Inf 둘 다 제외."""
    vals = np.array([values_by_target.get(t, np.nan) for t in all_target_ids], dtype=float)
    ok = np.isfinite(vals)
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
    common_mask = np.isfinite(a) & np.isfinite(b)
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
    common_mask = np.all([np.isfinite(mats[a]) for a in s1_arms], axis=0)
    n_common = int(common_mask.sum())
    fully_missing_seeds = [a for a in s1_arms if not np.isfinite(mats[a]).any()]
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


def analyze(records, eval_failed, errors, latest, all_target_ids, targets_doc, cfg, sd, arms, repeats,
           include_auxiliary):
    """[astra_review_20260928.md P1-2/P1-3, opus_review_20260928.md 1-2/1-5 수정]
    records: completed만(필수 property 7개가 다 있는 attempt). eval_failed: 평가가 깨져
    completed로 승격 못한 attempt. latest: attempt_id -> 최신 terminal 이벤트(ledger_state)
    -- counts_per_arm을 여기서 직접 조회해서 계산한다(이전처럼 error 이벤트를 attempt_id
    문자열 파싱으로 다시 세지 않음 -- 재시도 성공 시 옛 error가 이중 집계돼 missing이
    음수가 되던 버그의 원인).

    [2026-09-30 advisor 재검토] eval_failed 중 reason="required_props_missing"인
    attempt는 생성/화학 판정(strict_valid/single_valid/smiles 등)은 정상적으로 끝났고
    GNN proxy 등 property 일부만 없는 것이다(opus 1-5의 "GNN만 계속 None" 시나리오가
    실제로 일어나도 그 자체가 반드시 evaluator 버그는 아님 -- qm_props.gnn_homo_lumo는
    원자 2개 미만/결합 0개인 분자에서 정상적으로 None을 반환하도록 설계돼 있음). 이런
    attempt를 validity/per_target_metrics 계산에서 통째로 빼면 strict_valid rate 같은
    "property 유무와 무관해야 할" 지표까지 GNN 성공 여부에 종속되게 바뀌어버린다 --
    사전 등록된 분석 계획에서 벗어나는 변화이므로, chem_records(완료 + 화학 정보가
    있는 evaluation_failed)를 validity/per_target_metrics 입력으로 쓴다(개별 property
    값은 per_target_metrics가 이미 None을 걸러내므로 실제로 없는 property만 그 지표
    계산에서 빠진다). reason="evaluate_exception"인 evaluation_failed는 애초에 chemistry
    정보 자체가 없으므로(evaluate_fn이 끝까지 못 감) 여전히 전부 제외한다.

    include_auxiliary: 이 run이 실제로 보조 마스크까지 실행했는지(launch_manifest 기준) --
    config의 전체 마스크 목록이 아니라 이 값으로 분석 대상 마스크를 정한다(opus 권고:
    주패널만 실행한 run에 미계획 보조 패널을 결측으로 잘못 표시하던 문제)."""
    tby = {t["target_id"]: t for t in targets_doc["targets"]}
    masks = {cfg["masks"]["primary"]["name"]: cfg["masks"]["primary"]["observed"]}
    if include_auxiliary:
        for m in cfg["masks"]["auxiliary"]:
            masks[m["name"]] = m["observed"]
    out = {"per_mask": {}, "errors": {"n": len(errors), "by_type": defaultdict(int)},
          "evaluation_failed": {"n": len(eval_failed), "by_reason": defaultdict(int)}}
    for e in errors:
        out["errors"]["by_type"][e["error_type"]] += 1
    out["errors"]["by_type"] = dict(out["errors"]["by_type"])
    for e in eval_failed:
        out["evaluation_failed"]["by_reason"][e.get("reason", "unknown")] += 1
    out["evaluation_failed"]["by_reason"] = dict(out["evaluation_failed"]["by_reason"])
    s1_arms = [a for a in arms if a.startswith("s1_")]

    chem_eval_failed = [e for e in eval_failed if e.get("reason") == "required_props_missing"]
    chem_records = records + chem_eval_failed
    out["evaluation_failed"]["n_with_chemistry_info_included_in_metrics"] = len(chem_eval_failed)
    out["evaluation_failed"]["n_excluded_no_chemistry_info"] = len(eval_failed) - len(chem_eval_failed)

    for mname, observed in masks.items():
        recs_mask = [r for r in chem_records if r["mask"] == mname]
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
            # attempt_id 형식: f"{target_id}|{mask}|r{repeat}|{arm}" (build_plan 참조).
            # latest는 attempt_id별 최신 상태 하나뿐이므로 아래 네 값은 항상 서로 배타적이고
            # 합이 planned를 넘지 않는다(재시도 이력을 다시 세는 문제가 구조적으로 불가능).
            planned_ids = [f"{tid}|{mname}|r{r}|{arm}" for tid in all_target_ids for r in range(repeats)]
            ev_kinds = [latest[aid]["event"] for aid in planned_ids if aid in latest]
            n_completed = ev_kinds.count("completed")
            n_eval_failed = ev_kinds.count("evaluation_failed")
            n_errored = ev_kinds.count("error")
            counts[arm] = {"planned": planned_per_arm, "completed": n_completed,
                          "evaluation_failed": n_eval_failed, "errored": n_errored,
                          "missing": planned_per_arm - n_completed - n_eval_failed - n_errored}
        status = "complete" if all(
            c["completed"] + c["evaluation_failed"] + c["errored"] == c["planned"]
            for c in counts.values()) else "partial"

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
                    "모집단이 다를 수 있다 -- paired_vs_prior와 혼동하지 않는다.",
                    "evaluation_failed 중 chemistry 정보(strict_valid 등)가 있는 attempt"
                    "(reason=required_props_missing)는 validity/구조 property 지표 계산에 "
                    "포함된다 -- 없는 개별 property만 그 지표에서 빠진다(예: GNN만 없으면 "
                    "mae_HOMO/LUMO만 그 attempt를 빼고, strict_valid rate에는 그대로 포함). "
                    "chemistry 정보 자체가 없는 evaluation_failed(reason=evaluate_exception)만 "
                    "전부 제외된다. ledger 집계(counts_per_arm의 completed/evaluation_failed)는 "
                    "이 포함 여부와 무관하게 원래 ledger 이벤트 그대로다 -- "
                    "out['evaluation_failed']에 포함/제외 개수가 따로 집계된다."]
    return out


def verify_run_identity(run_dir, cfg_path, pins_path, targets_path, norm_path, cfg):
    """[P2, astra_review_20260928.md P1-4/opus_review_20260928.md 1-3 수정] 분석 대상
    run이 실제로 이 config/pins/targets/normalization으로 만들어졌는지 launch_manifest.json과
    대조. 불일치면 즉시 중단(잘못된 target/SD로 재분석 금지).

    이전 버전은 함수 시그니처와 docstring이 normalization도 검증한다고 암시했지만 실제
    checks dict에는 normalization 항목이 없어 norm_path가 죽은 인자였다(존재하지 않는
    파일을 넘겨도 통과, 두 리뷰 공통 반례). normalization.json 자체는 launch_manifest에
    기록되지 않으므로, config_sha256이 이미 검증한 cfg 안에 사전 고정된
    s1_models.normalization_sha256(pilot_config.json은 PRE-REGISTERED라 생성 전 값)과
    대조한다 -- 이 값은 이 함수가 호출되기 전에 이미 config_sha256 검증을 통과한 cfg에서
    나오므로, cfg 자체를 위조하지 않는 한 우회할 수 없다."""
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
    expected_norm_sha = cfg.get("s1_models", {}).get("normalization_sha256")
    if not expected_norm_sha:
        raise RuntimeError("pilot_config.json에 s1_models.normalization_sha256이 없음 -- "
                           "normalization 신원을 검증할 사전 고정값이 없어 진행 불가")
    actual_norm_sha = sha256_file(norm_path)
    if actual_norm_sha != expected_norm_sha:
        raise RuntimeError(
            f"run identity 불일치: normalization 파일({norm_path}) SHA={actual_norm_sha} != "
            f"pilot_config.json에 사전 고정된 s1_models.normalization_sha256={expected_norm_sha}")
    return lm


def validate_normalization(norm):
    """[astra_review_20260928.md P1-4] property 순서/중복과 std의 양수/유한값을 검사한다
    -- struct5_std_err_* 지표가 이 값으로 직접 나눠지므로(per_target_metrics) 여기가
    잘못되면 주지표가 조용히 왜곡된다."""
    names = norm.get("cond_prop_names")
    stds = norm.get("std_cond7")
    if not names or not stds:
        raise RuntimeError("normalization.json에 cond_prop_names 또는 std_cond7이 없음")
    if len(names) != len(set(names)):
        raise RuntimeError(f"normalization.json의 cond_prop_names에 중복이 있음: {names}")
    if len(names) != len(stds):
        raise RuntimeError(
            f"normalization.json의 cond_prop_names({len(names)}개)와 std_cond7({len(stds)}개) 길이가 다름")
    sd = dict(zip(names, stds))
    missing = [p for p in STRUCT5 if p not in sd]
    if missing:
        raise RuntimeError(f"normalization.json에 STRUCT5 property가 빠짐: {missing}")
    bad = {p: sd[p] for p in STRUCT5
          if not (isinstance(sd[p], (int, float)) and math.isfinite(sd[p]) and sd[p] > 0)}
    if bad:
        raise RuntimeError(f"normalization.json의 std가 양수 유한값이 아님(0/음수/NaN/Inf 등): {bad}")
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--normalization", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pins", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    lm = verify_run_identity(a.run_dir, a.config, a.pins, a.targets, a.normalization, cfg)
    targets_doc = json.load(open(a.targets))
    norm = json.load(open(a.normalization))
    sd = validate_normalization(norm)
    all_target_ids = sorted(t["target_id"] for t in targets_doc["targets"])
    include_auxiliary = bool(lm.get("include_auxiliary", False))

    records, eval_failed, errors, started, corrupt_tail, latest = load_completed(
        Path(a.run_dir) / "ledger.jsonl")
    if corrupt_tail is not None:
        # [astra_review_20260928.md P2] runner는 손상된 ledger를 RuntimeError로 거부하는데
        # analyzer는 앞부분만으로 분석 파일을 쓰고 마지막에 경고만 출력했다(정책 불일치).
        # 정상 결과와 구분 없이 analysis.json이 만들어지면 이 경고를 못 본 소비자가 partial
        # 결과를 완전한 결과로 오인할 수 있으므로, runner와 동일하게 즉시 거부한다.
        sys.exit(f"ledger 마지막 줄 손상, 분석 거부: {Path(a.run_dir) / 'ledger.jsonl'} :: "
                 f"{corrupt_tail[:200]}")
    res = analyze(records, eval_failed, errors, latest, all_target_ids, targets_doc, cfg, sd,
                 cfg["arms"], cfg["repeats_per_target_mask_arm"], include_auxiliary)
    res["n_completed"] = len(records)
    res["n_started_not_completed"] = len(started - {r["attempt_id"] for r in records})
    res["include_auxiliary"] = include_auxiliary
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
                  f"evaluation_failed={c['evaluation_failed']} errored={c['errored']} "
                  f"missing={c['missing']}")
        if "struct5_std_err_single" in blk["summary"]:
            row = blk["summary"]["struct5_std_err_single"]
            for arm in cfg["arms"]:
                d = row[arm]
                print(f"    struct5_std_err_single[{arm}]: mean={d['mean_over_own_evaluable_targets']} "
                      f"(n_evaluable={d['n_targets_evaluable']}, missing={d['n_targets_missing']})")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
