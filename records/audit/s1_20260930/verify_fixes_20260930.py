"""[S-1 pilot] astra_review_20260928.md / opus_review_20260928.md의 confirmed-real
findings에 대한 CPU 전용 독립 반례/수정 검증 스크립트. records/audit/에 저장소 커밋용으로
둔다(astra_review_20260928.md §3가 /tmp에만 둔 검증 스크립트를 지적했으므로 -- 이번엔
저장소 안에 둔다). GPU/RDKit/실제 A-1/S-1 모델은 쓰지 않는다(run_attempts/analyze_pilot의
상태-기계 로직만 순수 함수로 검증).

각 테스트는 원래 리뷰의 반례가 이제 "수정된 동작"을 내는지 확인한다. exit 0 = 아래
assert가 전부 통과 = 리뷰에서 지적된 문제들이 해소됨.
"""
import json
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PILOT = _HERE.parent.parent.parent / "s1_size_predictor" / "pilot"
sys.path.insert(0, str(_PILOT))

from pilot_common import classify_props_failure, ledger_state, read_ledger_safely  # noqa: E402
import pilot_run  # noqa: E402
import analyze_pilot  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    PASS.append(bool(cond))
    if not cond:
        raise AssertionError(f"{name}: {detail}")


# ---------------------------------------------------------------------------
# 1) opus 1-5 / astra P1-1 첫 부분: 구조 5개 정상 + HOMO/LUMO만 None인 "부분 파손"을
#    이전 evaluator_looks_broken(전부 None일 때만 True)은 놓쳤다. classify_props_failure는
#    잡아야 한다.
def test_partial_gnn_breakage_detected():
    props = {"LogP": 1.0, "TPSA": 2.0, "HBA": 3.0, "RotBonds": 4.0, "AromaticRings": 5.0,
             "HOMO": None, "LUMO": None}
    cls = classify_props_failure(True, props)
    check("partial_gnn_breakage_applicable", cls["applicable"] is True)
    check("partial_gnn_breakage_struct_ok", cls["struct_failed"] is False, cls)
    check("partial_gnn_breakage_gnn_failed", cls["gnn_failed"] is True, cls)
    # 반대 방향: 구조 쪽만 깨져도 잡혀야 함
    props2 = {"LogP": None, "TPSA": 2.0, "HBA": 3.0, "RotBonds": 4.0, "AromaticRings": 5.0,
              "HOMO": 0.1, "LUMO": -0.1}
    cls2 = classify_props_failure(True, props2)
    check("partial_struct_breakage_detected", cls2["struct_failed"] is True and cls2["gnn_failed"] is False, cls2)
    # invalid 분자는 "판단 불가"(성공도 실패도 아님) -- streak를 건드리면 안 됨
    cls3 = classify_props_failure(False, None)
    check("invalid_molecule_not_applicable", cls3["applicable"] is False, cls3)
    # Inf도 실패로 잡아야 함(astra P2: ~isnan은 Inf를 유효로 침)
    props4 = {"LogP": float("inf"), "TPSA": 2.0, "HBA": 3.0, "RotBonds": 4.0, "AromaticRings": 5.0,
              "HOMO": 0.1, "LUMO": -0.1}
    cls4 = classify_props_failure(True, props4)
    check("inf_value_counts_as_failure", cls4["struct_failed"] is True, cls4)


# ---------------------------------------------------------------------------
# 2) astra P1-1 후반부 / opus 1-1: evaluate_fn이 예외를 던지면 이전 버전은 전체 run이
#    죽고 ledger에는 started만 남았다. 이제는 evaluation_failed로 기록되고 다음
#    attempt로 진행해야 한다.
def test_evaluate_exception_does_not_kill_run():
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d)
        attempts = [
            {"attempt_id": "t1|mask0|r0|arm0", "target_id": "t1", "mask": "mask0", "repeat": 0,
             "arm": "arm0", "size": 5, "denoise_key": 111},
            {"attempt_id": "t2|mask0|r0|arm0", "target_id": "t2", "mask": "mask0", "repeat": 0,
             "arm": "arm0", "size": 5, "denoise_key": 222},
        ]

        def gen(a):
            import numpy as np
            return np.zeros(a["size"], dtype=np.int64), np.zeros((a["size"], a["size"]), dtype=np.int64)

        def ev_raises(a, X, E):
            raise RuntimeError("evaluator exploded")

        counts = pilot_run.run_attempts(attempts, gen, ev_raises, out_dir,
                                        max_consecutive_evaluate_exceptions=10)
        check("run_did_not_crash", True)  # 여기 도달 자체가 "죽지 않았다"는 증거
        check("both_attempts_processed", counts["planned"] == 2 and counts["completed"] == 0, counts)
        check("both_marked_evaluation_failed", counts["evaluation_failed"] == 2, counts)
        events, corrupt = read_ledger_safely(out_dir / "ledger.jsonl")
        check("ledger_no_corruption", corrupt is None)
        kinds_t1 = [e["event"] for e in events if e["attempt_id"] == "t1|mask0|r0|arm0"]
        check("t1_has_generated_then_evaluation_failed", kinds_t1 == ["started", "generated", "evaluation_failed"],
             kinds_t1)


# ---------------------------------------------------------------------------
# 3) astra P1-2: 평가가 깨지면 completed로 안 올라가야 하고, resume은 이 attempt를
#    건너뛰지 않고 재평가 대상으로 남겨야 한다(재생성은 하지 않고 저장된 배열을 재사용).
def test_evaluation_failed_not_completed_and_reused_on_resume():
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d)
        attempts = [{"attempt_id": "t1|mask0|r0|arm0", "target_id": "t1", "mask": "mask0",
                    "repeat": 0, "arm": "arm0", "size": 5, "denoise_key": 111}]
        gen_calls = {"n": 0}

        def gen(a):
            import numpy as np
            gen_calls["n"] += 1
            return np.zeros(a["size"], dtype=np.int64), np.zeros((a["size"], a["size"]), dtype=np.int64)

        def ev_broken(a, X, E):
            return {"strict_valid": True, "single_valid": True, "frag_ratio": 1.0, "n_fragments": 1,
                    "requested_n_atoms": a["size"], "smiles": "C", "actual_heavy_atoms": 1,
                    "props": {"LogP": 1.0, "TPSA": 1.0, "HBA": 1.0, "RotBonds": 1.0,
                             "AromaticRings": 1.0, "HOMO": None, "LUMO": None}, "prop_errors": []}

        counts1 = pilot_run.run_attempts(attempts, gen, ev_broken, out_dir)
        check("first_pass_not_completed", counts1["completed"] == 0 and counts1["evaluation_failed"] == 1, counts1)
        check("generate_called_once", gen_calls["n"] == 1)

        def ev_fixed(a, X, E):
            return {"strict_valid": True, "single_valid": True, "frag_ratio": 1.0, "n_fragments": 1,
                    "requested_n_atoms": a["size"], "smiles": "C", "actual_heavy_atoms": 1,
                    "props": {"LogP": 1.0, "TPSA": 1.0, "HBA": 1.0, "RotBonds": 1.0,
                             "AromaticRings": 1.0, "HOMO": 0.1, "LUMO": -0.1}, "prop_errors": []}

        counts2 = pilot_run.run_attempts(attempts, gen, ev_fixed, out_dir)
        check("second_pass_completed", counts2["completed"] == 1, counts2)
        check("generate_not_called_again_on_retry", gen_calls["n"] == 1,
             "재평가는 저장된 배열을 재사용해야 하며 재생성하면 안 됨 (astra P1-1 권고)")
        events, _ = read_ledger_safely(out_dir / "ledger.jsonl")
        kinds = [e["event"] for e in events]
        check("ledger_sequence_matches_reuse_path",
             kinds == ["started", "generated", "evaluation_failed", "started", "completed"], kinds)


# ---------------------------------------------------------------------------
# 4) astra P1-3 / opus 1-2: error -> 재시도 -> completed로 끝나면 missing이 음수가
#    되던 버그. ledger_state는 최신 상태만 세므로 발생할 수 없어야 한다.
def test_missing_never_negative_after_retry_success():
    events = [
        {"event": "started", "attempt_id": "a", "time": 1},
        {"event": "error", "attempt_id": "a", "error_type": "RuntimeError", "time": 2},
        {"event": "started", "attempt_id": "a", "time": 3},
        {"event": "generated", "attempt_id": "a", "arrays": "x.npz", "arrays_sha256": "sha", "time": 4},
        {"event": "completed", "attempt_id": "a", "props": {}, "time": 5},
    ]
    state = ledger_state(events, planned_ids=["a"])
    n_completed = sum(1 for ev in state["latest"].values() if ev["event"] == "completed")
    n_errored = sum(1 for ev in state["latest"].values() if ev["event"] == "error")
    planned = 1
    missing = planned - n_completed - n_errored
    check("retry_then_success_counts_once", n_completed == 1 and n_errored == 0, (n_completed, n_errored))
    check("missing_not_negative", missing == 0, missing)


# ---------------------------------------------------------------------------
# 5) opus 1-3 / astra P1-4: normalization.json 신원 검증. 사전 고정 SHA와 다르면(또는
#    파일이 없으면) 거부해야 한다 -- 이전 버전은 norm_path가 죽은 인자라 아무 파일을
#    넘겨도(심지어 존재하지 않아도) 통과했다.
def test_verify_run_identity_checks_normalization():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cfg = {"s1_models": {"normalization_sha256": "a" * 64}}
        cfg_path = d / "cfg.json"
        cfg_path.write_text(json.dumps(cfg))
        pins_path = d / "pins.json"
        pins_path.write_text("{}")
        targets_path = d / "targets.json"
        targets_path.write_text("{}")
        run_dir = d / "run"
        run_dir.mkdir()
        lm = {"config_sha256": analyze_pilot.sha256_file(cfg_path),
              "pins_sha256": analyze_pilot.sha256_file(pins_path),
              "targets_sha256": analyze_pilot.sha256_file(targets_path)}
        (run_dir / "launch_manifest.json").write_text(json.dumps(lm))

        wrong_norm = d / "DOES_NOT_EXIST.json"
        try:
            analyze_pilot.verify_run_identity(run_dir, cfg_path, pins_path, targets_path, wrong_norm, cfg)
            rejected_missing = False
        except (RuntimeError, FileNotFoundError):
            rejected_missing = True
        check("nonexistent_normalization_rejected", rejected_missing,
             "이전 버전은 이 케이스를 identity 검사 자체에서 통과시켰음(astra 반례)")

        real_norm = d / "normalization.json"
        real_norm.write_text('{"x": 1}')  # SHA가 cfg의 사전 고정값(전부 'a')과 다름
        try:
            analyze_pilot.verify_run_identity(run_dir, cfg_path, pins_path, targets_path, real_norm, cfg)
            rejected_mismatch = False
        except RuntimeError:
            rejected_mismatch = True
        check("sha_mismatched_normalization_rejected", rejected_mismatch)


# ---------------------------------------------------------------------------
# 6) astra P2: descriptive_mean 등이 Inf를 유효 관측으로 세지 않아야 한다(~isnan -> isfinite).
def test_inf_excluded_from_descriptive_mean():
    vals = {"t1": 1.0, "t2": float("inf"), "t3": 3.0}
    out = analyze_pilot.descriptive_mean(["t1", "t2", "t3"], vals)
    check("inf_excluded_from_mean", out["n_targets_evaluable"] == 2, out)
    check("inf_excluded_mean_value", abs(out["mean_over_own_evaluable_targets"] - 2.0) < 1e-9, out)


# ---------------------------------------------------------------------------
# 7) [2026-09-30 advisor 재검토] qm_props.gnn_homo_lumo는 원자<2/결합 0개인 분자에서
#    정상적으로 None을 반환하도록 설계돼 있다 -- reason="required_props_missing"인
#    evaluation_failed는 chemistry 정보(strict_valid 등)가 있으므로 validity 계산에는
#    포함돼야 한다(반대로 reason="evaluate_exception"은 chemistry 정보 자체가 없어 제외).
def test_evaluation_failed_with_chemistry_info_counted_in_validity():
    targets_doc = {"targets": [{"target_id": "t1", "raw7": {}}, {"target_id": "t2", "raw7": {}}]}
    cfg = {"masks": {"primary": {"name": "m0", "observed": []}, "auxiliary": []},
          "arms": ["prior"]}  # "prior"는 paired_vs_baseline 비교에서 제외되는 baseline 이름
    sd = {}
    completed = [{"target_id": "t1", "mask": "m0", "arm": "prior", "strict_valid": True,
                 "single_valid": True, "requested_n_atoms": 5, "smiles": "C", "actual_heavy_atoms": 1,
                 "props": {k: 1.0 for k in pilot_run.PROP_ORDER}, "elapsed_s": 1.0}]
    eval_failed = [{"target_id": "t2", "mask": "m0", "arm": "prior", "strict_valid": True,
                   "single_valid": True, "requested_n_atoms": 5, "smiles": "O", "actual_heavy_atoms": 1,
                   "reason": "required_props_missing",
                   "props": {"LogP": 1.0, "TPSA": 1.0, "HBA": 1.0, "RotBonds": 1.0,
                             "AromaticRings": 1.0, "HOMO": None, "LUMO": None}, "elapsed_s": 1.0}]
    latest = {}  # counts_per_arm만 쓰므로 이 테스트에서는 비워도 됨(아래서 직접 안 읽음)
    for tid in ("t1", "t2"):
        latest[f"{tid}|m0|r0|prior"] = {"event": "completed" if tid == "t1" else "evaluation_failed"}
    res = analyze_pilot.analyze(completed, eval_failed, [], latest, ["t1", "t2"], targets_doc, cfg, sd,
                                ["prior"], 1, False)
    validity = res["per_mask"]["m0"]["validity"]["prior"]
    check("gnn_only_failure_still_counted_in_validity_denominator",
         validity["n_completed_total"] == 2, validity)
    check("both_strict_valid_counted", validity["pooled_strict_rate"] == 1.0, validity)
    check("evaluation_failed_breakdown_recorded",
         res["evaluation_failed"]["n_with_chemistry_info_included_in_metrics"] == 1, res["evaluation_failed"])


if __name__ == "__main__":
    test_partial_gnn_breakage_detected()
    test_evaluate_exception_does_not_kill_run()
    test_evaluation_failed_not_completed_and_reused_on_resume()
    test_missing_never_negative_after_retry_success()
    test_verify_run_identity_checks_normalization()
    test_inf_excluded_from_descriptive_mean()
    test_evaluation_failed_with_chemistry_info_counted_in_validity()
    print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
    sys.exit(0 if all(PASS) else 1)
