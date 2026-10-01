"""[S-1 pilot] 고정 A-1에서 초기 원자 수 정책만 바꿔 생성하는 실행기.
astra_review_20260926.md §2~§6 반영. 세 모드:

  plan  : 생성 없이 attempt 목록/크기 추첨만 만든다(CPU). 결정성 검사 포함.
  gates : sampling이 필요한 launch gate(같은 attempt 재현, arm 순서/무관한 RNG 소비에
          대한 불변, 강제 크기 연결)를 작은 규모로 실행하고 report를 저장한다.
  run   : gates 보고서(통과 + 같은 runner SHA)가 있어야만 실행. attempt별 ledger,
          원본 X/E 저장, 오류와 화학적 invalid 구분, 반복 오류 시 중단.

GPU는 GPU1만(CUDA_VISIBLE_DEVICES=1). 기존 출력 디렉터리 재사용 금지(resume은
--resume-from으로 명시할 때만, ledger의 completed attempt를 건너뜀).
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent.parent))
from pilot_common import (GNN_KEYS, MIN_ATOMS, N_CLASSES, PROP_ORDER, STRUCT_KEYS,
                          classify_props_failure, derive_key, draw_uniform, import_pinned,
                          ledger_state, mask_vector, read_ledger_safely, sha256_file,
                          size_from_uniform)


# ----------------------------------------------------------------------------- 계획
def build_plan(cfg, targets, size_dists, include_auxiliary):
    """모든 (mask, target, repeat)에 대해 U를 한 번 뽑고 각 arm의 크기를 inverse CDF로 결정.
    size_dists[(arm, target_id, mask_name)] = 길이 49 확률."""
    masks = [cfg["masks"]["primary"]] + (cfg["masks"]["auxiliary"] if include_auxiliary else [])
    master = cfg["rng"]["master_seed"]
    attempts = []
    for mk in masks:
        for t in sorted(targets["targets"], key=lambda x: x["target_id"]):
            tid = t["target_id"]
            for r in range(cfg["repeats_per_target_mask_arm"]):
                k_size = derive_key(master, tid, mk["name"], r, "size_uniform")
                k_den = derive_key(master, tid, mk["name"], r, "denoise")
                u = draw_uniform(k_size)
                for arm in cfg["arms"]:
                    size = size_from_uniform(size_dists[(arm, tid, mk["name"])], u)
                    attempts.append({
                        "attempt_id": f"{tid}|{mk['name']}|r{r}|{arm}",
                        "target_id": tid, "mask": mk["name"], "repeat": r, "arm": arm,
                        "u": u, "size": size, "size_key": k_size, "denoise_key": k_den,
                    })
    return attempts


def atomic_savez(path, **arrays):
    """[P2] .npz 저장을 fsync+atomic rename으로 하고 저장된 파일의 SHA를 반환한다
    (이전에는 ledger 텍스트만 fsync되고 배열 파일 자체는 durability 보장이 없었다)."""
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return sha256_file(path)


# ------------------------------------------------------------------ 핵심 루프(주입식)
def run_attempts(attempts, generate_fn, evaluate_fn, out_dir, max_consecutive_errors=3,
                 max_consecutive_struct_failures=3, max_consecutive_gnn_failures=3,
                 max_consecutive_evaluate_exceptions=3):
    """generate_fn(attempt)->(X_np, E_np), evaluate_fn(attempt, X_np, E_np)->dict.

    [astra_review_20260928.md P1-1/P1-2, opus_review_20260928.md 1-1/1-2/1-5 수정]
    ledger 이벤트는 started -> generated -> (completed | evaluation_failed), 또는
    started -> error(생성 자체 실패) 순서다. "completed"는 생성과 평가가 모두 성공하고
    필수 property 7개가 전부 유한값으로 있어야만 붙는다 -- 이전 버전은 평가가 깨져도
    먼저 completed를 쓰고 별도 evaluator_failure 이벤트를 덧붙여서, resume은 이 attempt를
    건너뛰고 analyzer는 별도 이벤트를 집계하지 않아 조용히 "완료"로 보였다.

    evaluate_fn 자체가 예외를 던지는 경우(이전 버전은 무방비 -- 개발자 본인이 재현했지만
    안 고친 버그)도 이제 잡아서 evaluation_failed로 기록하고 다음 attempt로 진행한다
    (전체 run이 죽지 않는다).

    구조 property(STRUCT_KEYS)와 GNN proxy(GNN_KEYS)의 연속 실패를 따로 추적해서(둘은
    독립된 코드 경로이므로 하나만 체계적으로 깨질 수 있음 -- opus 1-5), 그 중 하나만
    계속 실패해도 감지해 중단한다. 화학적으로 invalid한 분자(evaluator 판단 불가)나
    generate 실패는 이 streak를 건드리지 않는다 -- "성공도 실패도 아닌 무관한 사건"으로
    취급해야 진짜 evaluator 장애 연속 발생만 잡는다(Astra Q3: invalid를 성공으로 쳐서
    streak를 무조건 초기화하면 안 됨, 그렇다고 실패로 쳐도 안 됨).

    이미 이 attempt의 array가 생성된 적이 있으면(직전 시도가 evaluation_failed로
    끝났거나 evaluate 도중 프로세스가 죽은 경우) 재생성하지 않고 저장된 배열을 SHA
    확인 후 그대로 재평가한다(Astra P1-1 권고: "평가 실패 때문에 생성까지 반복하지
    않도록"). 반환: 요약 dict."""
    out_dir = Path(out_dir)
    (out_dir / "arrays").mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / "ledger.jsonl"
    events, corrupt_tail = read_ledger_safely(ledger_path)
    if corrupt_tail is not None:
        raise RuntimeError(f"ledger 마지막 줄 손상, 수동 확인 필요: {ledger_path} :: {corrupt_tail[:200]}")
    planned_ids = [a["attempt_id"] for a in attempts]
    state = ledger_state(events, planned_ids=planned_ids)
    if state["duplicate_completed_ids"]:
        raise RuntimeError(f"ledger 손상 의심: attempt_id별 completed 이벤트가 2개 이상: "
                           f"{state['duplicate_completed_ids']}")
    done_ids = {aid for aid, ev in state["latest"].items() if ev["event"] == "completed"}
    generated_records = state["generated"]

    # [astra_review_20260927.md P1-1 계승, 20260928 확장] 같은 attempt_id인데 내용이
    # 달라진 채 재실행돼도 조용히 받아들이지 않는다. completed/evaluation_failed/error
    # 최신 기록뿐 아니라 generated(재평가 대상) 기록도 대조한다.
    check_fields = ("target_id", "mask", "repeat", "arm", "size", "denoise_key")
    for a in attempts:
        for prev in (state["latest"].get(a["attempt_id"]), generated_records.get(a["attempt_id"])):
            if prev is None:
                continue
            for field in check_fields:
                if field in prev and prev.get(field) != a.get(field):
                    raise RuntimeError(
                        f"attempt {a['attempt_id']!r} 재실행 감지: 기록된 {field}={prev.get(field)!r} vs "
                        f"지금 {a.get(field)!r} -- 설정이 바뀐 채 같은 out_dir로 재실행하려는 것으로 보임")

    def log(ev):
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    counts = {"planned": len(attempts), "skipped_already_completed": 0, "completed": 0,
              "generate_errors": 0, "evaluation_failed": 0, "struct_failures": 0, "gnn_failures": 0}
    consecutive_err = 0
    consecutive_struct_fail = 0
    consecutive_gnn_fail = 0
    consecutive_eval_exc = 0
    aborted, abort_reason = False, None
    for a in attempts:
        if a["attempt_id"] in done_ids:
            counts["skipped_already_completed"] += 1
            continue
        t0 = time.time()
        reused = generated_records.get(a["attempt_id"])
        arr_name = arr_sha = None
        if reused is not None:
            arr_path = out_dir / "arrays" / reused["arrays"]
            if arr_path.exists() and sha256_file(arr_path) == reused.get("arrays_sha256"):
                with np.load(arr_path) as npz:
                    X, E = npz["X"], npz["E"]
                arr_name, arr_sha = reused["arrays"], reused["arrays_sha256"]
                log({"event": "started", **a, "time": time.time(), "reused_prior_generation": True})
            # 배열이 없거나 SHA가 안 맞으면 처음부터 다시 생성(아래 분기로 흘러감)
        if arr_name is None:
            log({"event": "started", **a, "time": time.time()})
            try:
                X, E = generate_fn(a)
            except Exception as e:   # 생성 자체의 OOM/소프트웨어 오류
                log({"event": "error", **a, "error_type": type(e).__name__,
                     "error": str(e)[:500], "trace": traceback.format_exc()[-1500:],
                     "elapsed_s": time.time() - t0})
                counts["generate_errors"] += 1
                consecutive_err += 1
                if consecutive_err >= max_consecutive_errors:
                    aborted, abort_reason = True, "consecutive_generate_errors"
                    break
                continue
            arr_name = f"{abs(hash_str(a['attempt_id']))}.npz"
            arr_sha = atomic_savez(out_dir / "arrays" / arr_name, X=X.astype(np.int8), E=E.astype(np.int8))
            log({"event": "generated", **a, "arrays": arr_name, "arrays_sha256": arr_sha,
                 "generate_elapsed_s": time.time() - t0, "time": time.time()})
        consecutive_err = 0

        try:
            result = evaluate_fn(a, X, E)
        except Exception as e:   # [astra P1-1/opus 1-1] evaluate_fn 예외로 run 전체가 죽던 문제
            log({"event": "evaluation_failed", **a, "arrays": arr_name, "arrays_sha256": arr_sha,
                 "elapsed_s": time.time() - t0, "reason": "evaluate_exception",
                 "error_type": type(e).__name__, "error": str(e)[:500],
                 "trace": traceback.format_exc()[-1500:]})
            counts["evaluation_failed"] += 1
            consecutive_eval_exc += 1
            if consecutive_eval_exc >= max_consecutive_evaluate_exceptions:
                aborted, abort_reason = True, "consecutive_evaluate_exceptions"
                break
            continue
        consecutive_eval_exc = 0

        cls = classify_props_failure(result.get("strict_valid"), result.get("props"))
        if cls["applicable"]:
            consecutive_struct_fail = consecutive_struct_fail + 1 if cls["struct_failed"] else 0
            consecutive_gnn_fail = consecutive_gnn_fail + 1 if cls["gnn_failed"] else 0
        # invalid 분자(applicable=False)는 두 streak 다 그대로 둔다(성공도 실패도 아님)

        if cls["applicable"] and (cls["struct_failed"] or cls["gnn_failed"]):
            counts["evaluation_failed"] += 1
            counts["struct_failures"] += int(cls["struct_failed"])
            counts["gnn_failures"] += int(cls["gnn_failed"])
            log({"event": "evaluation_failed", **a, "arrays": arr_name, "arrays_sha256": arr_sha,
                 "elapsed_s": time.time() - t0, "reason": "required_props_missing",
                 "props_check": cls, **result})
        else:
            log({"event": "completed", **a, "arrays": arr_name, "arrays_sha256": arr_sha,
                 "elapsed_s": time.time() - t0, "props_check": cls, **result})
            counts["completed"] += 1

        if consecutive_struct_fail >= max_consecutive_struct_failures:
            aborted, abort_reason = True, "consecutive_struct_property_failures"
            break
        if consecutive_gnn_fail >= max_consecutive_gnn_failures:
            aborted, abort_reason = True, "consecutive_gnn_property_failures"
            break

    events, corrupt_tail = read_ledger_safely(ledger_path)
    final_state = ledger_state(events, planned_ids=planned_ids)
    finished = {aid for aid, ev in final_state["latest"].items() if ev["event"] == "completed"}
    counts["unfinished_ids"] = [aid for aid in planned_ids if aid not in finished]
    counts["evaluation_failed_ids"] = [aid for aid, ev in final_state["latest"].items()
                                       if ev["event"] == "evaluation_failed"]
    counts["aborted"] = aborted
    counts["abort_reason"] = abort_reason
    counts["ledger_corrupt_tail"] = corrupt_tail
    return counts


def hash_str(s):
    import hashlib
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


def verify_resume_consistency(out_dir, attempts):
    """[astra_review_20260927.md P1-1, 20260928 확장] resume 시 기존 completed/
    evaluation_failed 최신 기록과 generated(재평가 대기 중인) 기록 전부를 지금 다시
    계산한 plan(attempts)과 대조한다. attempt_id가 사라졌거나, 같은 attempt_id인데
    target/mask/repeat/arm/size/denoise_key 중 하나라도 다르면(=설정이 바뀐 채 resume)
    즉시 거부한다. 참조하는 원본 배열 파일도 존재/SHA를 확인한다(Astra 반례:
    resume_changed_attempt)."""
    out_dir = Path(out_dir)
    events, corrupt = read_ledger_safely(out_dir / "ledger.jsonl")
    if corrupt is not None:
        raise RuntimeError(f"resume 거부: ledger 마지막 줄이 손상돼 있어 수동 확인이 먼저 필요함: {corrupt[:200]}")
    plan_by_id = {a["attempt_id"]: a for a in attempts}
    planned_ids = list(plan_by_id.keys())
    state = ledger_state(events, planned_ids=planned_ids)
    if state["duplicate_completed_ids"]:
        raise RuntimeError(f"resume 거부: attempt_id별 completed 이벤트가 2개 이상(ledger 손상 의심): "
                           f"{state['duplicate_completed_ids']}")
    fields = ("target_id", "mask", "repeat", "arm", "size", "denoise_key")
    records_to_check = list(state["latest"].items()) + list(state["generated"].items())
    for aid, ev in records_to_check:
        if aid not in plan_by_id:
            raise RuntimeError(f"resume 거부: 기존 기록 attempt {aid}가 현재 plan에 없음(설정이 바뀐 것으로 보임)")
        cur = plan_by_id[aid]
        for field in fields:
            if field in ev and ev.get(field) != cur.get(field):
                raise RuntimeError(
                    f"resume 거부: attempt {aid}의 {field}가 기록값({ev.get(field)!r})과 "
                    f"현재 plan값({cur.get(field)!r})에서 다름 -- 설정이 바뀐 채 resume 시도된 것으로 보임")
        if "arrays" not in ev:
            continue   # error 이벤트(생성 자체 실패)는 배열이 없음
        arr_path = out_dir / "arrays" / ev["arrays"]
        if not arr_path.exists():
            raise RuntimeError(f"resume 거부: attempt {aid}이 참조하는 배열 파일이 없음: {arr_path}")
        if "arrays_sha256" in ev and sha256_file(arr_path) != ev["arrays_sha256"]:
            raise RuntimeError(f"resume 거부: attempt {aid}의 배열 파일 SHA가 기록과 다름(손상 가능): {arr_path}")


# ------------------------------------------------------------------------- 로딩
def load_inputs(args):
    cfg_path = Path(args.config).resolve()
    pins_path = Path(args.pins).resolve()
    cfg = json.load(open(cfg_path))
    pins = json.load(open(pins_path))
    inputs_dir = Path(args.inputs_dir).resolve()
    manifest = json.load(open(inputs_dir / "inputs_manifest.json"))
    assert sha256_file(cfg_path) == manifest["config_sha256"], "config SHA가 입력 준비 시점과 다름"
    assert sha256_file(inputs_dir / "targets.json") == manifest["targets_json_sha256"]
    assert sha256_file(inputs_dir / "prior.json") == manifest["prior_json_sha256"]
    targets = json.load(open(inputs_dir / "targets.json"))
    prior = json.load(open(inputs_dir / "prior.json"))
    return cfg_path, pins_path, cfg, pins, inputs_dir, targets, prior


def size_predictor_module_sha():
    """[astra_review_20260928.md P2] S-1 체크포인트 SHA는 확인하지만 forward를 정의하는
    SizePredictor 소스 자체는 identity에 없었다 -- 소스가 바뀌면 완료 attempt는 (체크포인트가
    안 바뀌었으니) 그대로 통과하지만 미완료 attempt의 크기 계획은 달라질 수 있었다."""
    import size_predictor
    return sha256_file(size_predictor.__file__)


def load_s1(cfg, s1_root):
    from size_predictor import SizePredictor
    models = {}
    for tag, spec in cfg["s1_models"]["checkpoints"].items():
        matches = glob.glob(str(Path(s1_root) / spec["path"]))
        assert len(matches) == 1, f"{tag}: checkpoint {len(matches)}개 발견"
        assert sha256_file(matches[0]) == spec["sha256"], f"{tag}: checkpoint SHA 불일치"
        state = torch.load(matches[0], map_location="cpu", weights_only=False)
        m = SizePredictor()
        res = m.load_state_dict(state["model"], strict=True)
        assert not res.missing_keys and not res.unexpected_keys
        models[f"s1_{tag}"] = m.eval()
    return models


def build_size_dists(cfg, targets, prior, s1_models, include_auxiliary):
    masks = [cfg["masks"]["primary"]] + (cfg["masks"]["auxiliary"] if include_auxiliary else [])
    dists = {}
    for mk in masks:
        mv = torch.tensor(mask_vector(mk["observed"]), dtype=torch.float32).unsqueeze(0)
        for t in targets["targets"]:
            cond = torch.tensor(t["s1_normalized7"], dtype=torch.float32).unsqueeze(0)
            for arm in cfg["arms"]:
                if arm == "prior":
                    dists[(arm, t["target_id"], mk["name"])] = np.array(prior["probs"], dtype=np.float64)
                else:
                    with torch.no_grad():
                        p = torch.softmax(s1_models[arm](cond, mv), dim=-1)[0].double().numpy()
                    dists[(arm, t["target_id"], mk["name"])] = p
    return dists


# ---------------------------------------------------------------- A-1 생성/평가
class A1Runner:
    def __init__(self, pins, cfg, targets):
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1" or not torch.cuda.is_available(), \
            "GPU는 GPU1만 사용: CUDA_VISIBLE_DEVICES=1 필요"
        self.cfg, self.targets = cfg, targets
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        frozen = Path(pins["frozen_dir_on_server"])
        self.model3, self.train3, self.qm_props, self.c1 = import_pinned(frozen)
        for mod in (self.model3, self.train3, self.qm_props, self.c1):
            assert sha256_file(mod.__file__) == pins["imported_module_sha256"][mod.__name__], mod.__name__
        # [astra_review_20260927.md P1-2] 코드 SHA뿐 아니라 실제 로드되는 가중치/라이브러리
        # 버전도 실행 시점에 재검증한다(gate report의 기록만 신뢰하지 않는다).
        c1_reg_path = frozen / "c1_regressor.pt"
        assert sha256_file(c1_reg_path) == pins["files_sha256"]["c1_regressor.pt"], "c1_regressor.pt SHA 불일치"
        import rdkit
        assert rdkit.__version__ == pins["evaluator"]["rdkit_version"], \
            f"RDKit 버전 불일치: {rdkit.__version__} != {pins['evaluator']['rdkit_version']}"
        assert torch.__version__ == pins["evaluator"]["torch_version"], \
            f"torch 버전 불일치: {torch.__version__} != {pins['evaluator']['torch_version']}"
        sd = torch.load(frozen / "checkpoints_c4_a1/best.pt", map_location="cpu", weights_only=False)
        assert sha256_file(frozen / "checkpoints_c4_a1/best.pt") == pins["files_sha256"]["checkpoints_c4_a1/best.pt"]
        self.a1 = self.model3.MoleculeGraphDiffusion(sd["schedule.m_X"].clone(), sd["schedule.m_E"].clone())
        res = self.a1.load_state_dict(sd, strict=True)
        assert not res.missing_keys and not res.unexpected_keys
        self.a1.to(self.device).eval()
        self.guidance_w = cfg["frozen_a1"]["guidance_w"]
        self.tmap = {t["target_id"]: t for t in targets["targets"]}
        self.mask_obs = {cfg["masks"]["primary"]["name"]: cfg["masks"]["primary"]["observed"]}
        for m in cfg["masks"]["auxiliary"]:
            self.mask_obs[m["name"]] = m["observed"]

    def generate_raw(self, cond7, mask7, size, denoise_key):
        """순수 함수: (조건, 마스크, 크기, denoise seed) -> (X, E). arm/S-1 정보는 들어오지 않는다."""
        cond = torch.tensor(cond7, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask7, dtype=torch.float32).unsqueeze(0)
        torch.manual_seed(denoise_key)       # CPU + 보이는 모든 CUDA device의 전역 RNG
        with torch.no_grad():
            X, E = self.a1.sample(1, int(size), self.device, cond=cond, cond_mask=mask,
                                  guidance_w=self.guidance_w)
        return X[0].cpu().numpy(), E[0].cpu().numpy()

    def generate(self, a):
        t = self.tmap[a["target_id"]]
        return self.generate_raw(t["a1_normalized7"], mask_vector(self.mask_obs[a["mask"]]),
                                 a["size"], a["denoise_key"])

    def evaluate(self, a, X, E):
        from rdkit import Chem
        from rdkit.Chem import Crippen, rdMolDescriptors
        info = self.train3.analyze_molecule(torch.from_numpy(X.astype(np.int64)),
                                            torch.from_numpy(E.astype(np.int64)))
        res = {"strict_valid": bool(info["strict_valid"]), "single_valid": bool(info["single_valid"]),
               "frag_ratio": float(info["frag_ratio"]), "n_fragments": int(info["n_fragments"]),
               "requested_n_atoms": int(a["size"]), "smiles": None, "actual_heavy_atoms": None,
               "props": None, "prop_errors": []}
        mol = info["mol"]
        if info["strict_valid"] and mol is not None:   # mol is not None만으로 판단 금지(Astra §5)
            try:
                res["smiles"] = Chem.MolToSmiles(mol)
            except Exception as e:
                res["prop_errors"].append(f"smiles:{e}")
            res["actual_heavy_atoms"] = int(mol.GetNumAtoms())
            fns = {"LogP": Crippen.MolLogP, "TPSA": rdMolDescriptors.CalcTPSA,
                   "HBA": lambda m: float(rdMolDescriptors.CalcNumHBA(m)),
                   "RotBonds": lambda m: float(rdMolDescriptors.CalcNumRotatableBonds(m)),
                   "AromaticRings": lambda m: float(rdMolDescriptors.CalcNumAromaticRings(m))}
            props = {}
            for name, fn in fns.items():
                try:
                    props[name] = float(fn(mol))
                except Exception as e:
                    props[name] = None
                    res["prop_errors"].append(f"{name}:{e}")
            try:
                est = self.qm_props.gnn_homo_lumo(mol)
                props["HOMO"], props["LUMO"] = (est if est is not None else (None, None))
            except Exception as e:
                props["HOMO"] = props["LUMO"] = None
                res["prop_errors"].append(f"gnn:{e}")
            res["props"] = props
        return res


# ------------------------------------------------------------------- 모드 구현
def apply_determinism_settings():
    """[astra_review_20260928.md Q1] CUBLAS_WORKSPACE_CONFIG/deterministic algorithms/
    cudnn.benchmark/TF32를 CUDA 컨텍스트를 만드는 A1Runner() 호출보다 먼저 고정한다.
    gates와 run 양쪽에서 똑같이 호출해야 한다 -- gate에서만 걸면 gate가 검증한
    결정성이 실제 run에는 적용 안 되는 허점이 생긴다(2026-09-30 자체 재검토에서
    발견: 처음 구현은 mode_gates에만 넣었었음)."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def mode_plan(args, ctx):
    cfg, targets, prior, s1 = ctx["cfg"], ctx["targets"], ctx["prior"], ctx["s1"]
    dists = build_size_dists(cfg, targets, prior, s1, args.include_auxiliary)
    a1 = build_plan(cfg, targets, dists, args.include_auxiliary)
    a2 = build_plan(cfg, targets, dists, args.include_auxiliary)
    assert a1 == a2, "plan이 결정적이지 않음"
    ids = [a["attempt_id"] for a in a1]
    assert len(ids) == len(set(ids))
    summary = {}
    for arm in cfg["arms"]:
        s = [a["size"] for a in a1 if a["arm"] == arm]
        summary[arm] = {"n": len(s), "mean_size": float(np.mean(s)), "min": min(s), "max": max(s)}
    out = {"n_attempts": len(a1), "arm_size_summary": summary, "attempts": a1,
           "include_auxiliary": args.include_auxiliary}
    (ctx["out_dir"] / "plan.json").write_text(json.dumps(out, indent=2))
    print(f"plan: {len(a1)} attempts, deterministic OK")
    for arm, s in summary.items():
        print(f"  {arm:<14} mean_size={s['mean_size']:.1f} range=[{s['min']},{s['max']}]")
    return 0


def mode_gates(args, ctx):
    """[astra_review_20260927.md P1-3, astra_review_20260928.md P1-5/Q1,
    opus_review_20260928.md 1-6 수정] 직접 generate_raw()만 호출하던 이전 버전은 실제
    production 경로인 A1Runner.generate(attempt)를 한 번도 거치지 않고도 all_passed=true를
    낼 수 있었다. 이제:
      1. 결정성 설정(CUBLAS_WORKSPACE_CONFIG, deterministic algorithms, cudnn.benchmark=False,
         TF32 off)을 A1Runner 초기화(=CUDA 컨텍스트 생성) 전에 적용하고 report에 기록한다.
      2. generate() 호출 31회 전부를 순번/attempt/배열 SHA/경과시간/비교 대상과 함께
         실패해도 남긴다(이전 버전은 최초 4회만 저장하고 실패 시 아무것도 안 남았다).
      3. evaluate 스모크는 몽키패치된 analyze_molecule이 아니라 실제(수정 안 한)
         train3.analyze_molecule에, 손으로 인코딩한 X/E 텐서(에탄올=valid, 5결합
         탄소=invalid, 이원자 2조각=multi-fragment)를 직접 통과시켜 검증한다.
         필수 property 7개(PROP_ORDER)를 명시적으로 확인한다(빈 dict가 통과하던 반례 수정).
      4. S-1 forward -> softmax -> size_from_uniform 경로를 최소 1개 target/mask에 대해
         실제로 실행해 확률이 유한하고 합이 1에 가까운지, 뽑힌 크기가 지원 범위 안인지
         확인한다(gate가 이 경로를 전혀 실행하지 않던 문제)."""
    cfg, targets = ctx["cfg"], ctx["targets"]

    # 1) 결정성 설정 -- CUDA 컨텍스트를 만드는 A1Runner()보다 먼저.
    determinism_settings = apply_determinism_settings()

    r = A1Runner(ctx["pins"], cfg, targets)
    rep = []
    call_log = []
    arrays_dir = ctx["out_dir"] / "gate_arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    def write_report(crash=None):
        """[astra_review_20260928.md P1-5] 중간에 예외가 나도(call_generate가 실패를
        기록 후 re-raise하는 경우 등) 그때까지 쌓인 rep/call_log를 all_passed=false로
        디스크에 남긴다 -- 이전 버전은 report를 맨 끝에서만 썼기 때문에 gate가 크래시하면
        그 흔적이 전혀 안 남았다."""
        report = {"all_passed": (crash is None) and bool(rep) and all(x["passed"] for x in rep),
                  "crashed": crash is not None, "crash": crash,
                  "gates": rep, "determinism_settings": determinism_settings, "generate_calls": call_log,
                  "runner_sha256": sha256_file(__file__),
                  "common_sha256": sha256_file(_HERE / "pilot_common.py"),
                  "config_sha256": ctx["cfg_sha"], "pins_sha256": ctx["pins_sha"],
                  # [P1-2] gate report가 어떤 입력(targets/prior/plan)을 대상으로 통과했는지도
                  # 못박아, 다른 입력에 재사용되지 못하게 한다.
                  "targets_sha256": sha256_file(ctx["inputs_dir"] / "targets.json"),
                  "prior_sha256": sha256_file(ctx["inputs_dir"] / "prior.json"),
                  "inputs_manifest_sha256": sha256_file(ctx["inputs_dir"] / "inputs_manifest.json")}
        (ctx["out_dir"] / "generation_gate_report.json").write_text(json.dumps(report, indent=2))
        return report

    def chk(name, ok, detail=""):
        rep.append({"gate": name, "passed": bool(ok), "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")

    def call_generate(a, compare_to=None):
        """generate() 1회 호출을 순번과 함께 기록한다(2, "31회 중 4회분만 원본 보존"
        문제 수정) -- 성공/실패 무관하게 남는다."""
        seq = len(call_log) + 1
        t0 = time.time()
        entry = {"seq": seq, "attempt_id": a["attempt_id"], "compare_to": compare_to}
        try:
            X, E = r.generate(a)
        except Exception as e:
            entry.update(ok=False, error=f"{type(e).__name__}: {e}", elapsed_s=time.time() - t0)
            call_log.append(entry)
            raise
        arr_path = arrays_dir / f"{seq:03d}_{a['attempt_id'].replace('|', '_')}.npz"
        arr_sha = atomic_savez(arr_path, X=X.astype(np.int8), E=E.astype(np.int8))
        entry.update(ok=True, elapsed_s=time.time() - t0, arrays=arr_path.name, arrays_sha256=arr_sha,
                     shape=list(X.shape))
        call_log.append(entry)
        return X, E

    def make_attempt(t, mask_name, repeat, arm, size_override=None):
        tid = t["target_id"]
        k_den = derive_key(cfg["rng"]["master_seed"], tid, mask_name, repeat, "denoise")
        size = size_override if size_override is not None else 30
        return {"attempt_id": f"gate|{tid}|{mask_name}|r{repeat}|{arm}", "target_id": tid,
                "mask": mask_name, "repeat": repeat, "arm": arm, "size": size, "denoise_key": k_den}

    try:
        t0 = ctx["targets"]["targets"][0]
        t1 = ctx["targets"]["targets"][1]
        prim_name = cfg["masks"]["primary"]["name"]
        aux_name = cfg["masks"]["auxiliary"][0]["name"]
        r.mask_obs.setdefault(prim_name, cfg["masks"]["primary"]["observed"])
        all_arms = cfg["arms"]
        for t in (t0, t1):
            for mname in (prim_name, aux_name):
                # 실제 production 경로: generate(attempt) -- generate_raw() 우회 금지
                arm0 = all_arms[0]
                a1 = make_attempt(t, mname, 0, arm0, size_override=30)
                X1, E1 = call_generate(a1)
                # 무관한 RNG 소비를 사이에 끼워도 같은 attempt(같은 arm)는 같아야 한다
                _ = torch.rand(1000)
                if torch.cuda.is_available():
                    _ = torch.rand(1000, device="cuda")
                _ = torch.multinomial(torch.ones(10), 5)
                X2, E2 = call_generate(a1, compare_to="same_attempt")
                chk(f"same_attempt_reproducible_via_generate[{t['target_id']},{mname}]",
                    np.array_equal(X1, X2) and np.array_equal(E1, E2))
                # arm이 달라도 (target,mask,repeat,size,denoise_key)가 같으면 A-1 출력이 같아야
                # 한다(크기 정책 arm 간 차이는 "어떤 크기를 고르는지"에만 있고, 고른 뒤 생성
                # 자체는 arm과 무관해야 함) -- generate()를 arm 순서를 바꿔가며 호출해 상태 누수 확인
                for other_arm in all_arms[1:]:
                    a_other = make_attempt(t, mname, 0, other_arm, size_override=30)
                    Xo, Eo = call_generate(a_other, compare_to="same_attempt")
                    chk(f"forced_size_same_output_across_arms[{t['target_id']},{mname},{other_arm}]",
                        np.array_equal(X1, Xo) and np.array_equal(E1, Eo))
                # interleave: 다른 target/mask로 generate 한 번 끼운 뒤 원래 attempt 재호출해도 동일
                a_interleave = make_attempt(t1 if t is t0 else t0, aux_name if mname == prim_name else prim_name,
                                            1, all_arms[-1], size_override=28)
                _ = call_generate(a_interleave)
                X3, E3 = call_generate(a1, compare_to="same_attempt")
                chk(f"no_state_leak_after_interleaved_generate[{t['target_id']},{mname}]",
                    np.array_equal(X1, X3) and np.array_equal(E1, E3))

        # seed(반복 번호)가 다르면 출력이 달라야 한다(seed가 실제로 작동하는지) -- generate() 경유
        aA = make_attempt(t0, prim_name, 0, all_arms[0], size_override=30)
        aB = make_attempt(t0, prim_name, 1, all_arms[0], size_override=30)
        XA, EA = call_generate(aA)
        XB, EB = call_generate(aB, compare_to="different_repeat")
        chk("different_repeat_different_output_via_generate", not (np.array_equal(XA, XB) and np.array_equal(EA, EB)))
        # 크기가 바뀌면 출력 shape이 요청 크기와 일치
        aS = make_attempt(t0, prim_name, 0, all_arms[0], size_override=25)
        Xs, Es = call_generate(aS)
        chk("output_shape_matches_requested_size", Xs.shape == (25,) and Es.shape == (25, 25), f"{Xs.shape} {Es.shape}")
        chk("all_31_calls_recorded", len(call_log) == 31, f"actual={len(call_log)}")

        # 3) evaluate: 실제(수정 안 한) train3.analyze_molecule에 손으로 인코딩한 X/E를 통과시킨다.
        # 클래스 인코딩(model3.ATOMIC_NUM_TO_CLS): 1=C, 2=O, 3=N, ... / 결합 코드: 1=단일,2=이중,3=삼중.
        def fixture_result(name, X, E):
            res = r.evaluate({"size": len(X)}, np.array(X, dtype=np.int8), np.array(E, dtype=np.int8))
            return res

        # 에탄올(C-C-O, 단일결합 2개) -- 유효, 조각 1개
        eth_X = [1, 1, 2]
        eth_E = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        res_eth = fixture_result("ethanol", eth_X, eth_E)
        chk("fixture_ethanol_strict_and_single_valid",
            res_eth["strict_valid"] is True and res_eth["single_valid"] is True, detail=res_eth.get("smiles"))
        required_keys_present = (res_eth.get("props") is not None
                                 and all(k in res_eth["props"] for k in PROP_ORDER))
        required_keys_finite = (required_keys_present and all(
            res_eth["props"][k] is not None and np.isfinite(res_eth["props"][k]) for k in PROP_ORDER))
        chk("fixture_ethanol_all_7_required_props_present_and_finite",
            required_keys_present and required_keys_finite, detail=res_eth.get("props"))

        # 탄소 1개에 산소 5개를 전부 단일결합(5결합 탄소, 원자가 초과) -- invalid
        inv_X = [1, 2, 2, 2, 2, 2]
        inv_E = [[0, 1, 1, 1, 1, 1]] + [[1, 0, 0, 0, 0, 0],
                                        [1, 0, 0, 0, 0, 0],
                                        [1, 0, 0, 0, 0, 0],
                                        [1, 0, 0, 0, 0, 0],
                                        [1, 0, 0, 0, 0, 0]]
        res_inv = fixture_result("pentavalent_carbon", inv_X, inv_E)
        chk("fixture_invalid_valence_gives_strict_invalid", res_inv["strict_valid"] is False,
            detail=res_inv)

        # 서로 결합 없는 원자 2개(탄소, 산소) -- 조각 2개, frag_ratio=0.5 < 0.85 -> strict invalid
        frag_X = [1, 2]
        frag_E = [[0, 0], [0, 0]]
        res_frag = fixture_result("two_isolated_atoms", frag_X, frag_E)
        chk("fixture_multi_fragment_detected",
            res_frag["n_fragments"] == 2 and res_frag["strict_valid"] is False, detail=res_frag)

        # 4) S-1 forward -> softmax -> size_from_uniform 경로를 실제로 실행(gate가 이 경로를
        # 전혀 안 타던 문제, opus_review_20260928.md 1-6).
        dists = build_size_dists(cfg, targets, ctx["prior"], ctx["s1"], args.include_auxiliary)
        (arm0_check, tid0_check, mname0_check) = (all_arms[1], t0["target_id"], prim_name)  # arm[0]="prior"는 S-1 forward 없음
        probs = dists[(arm0_check, tid0_check, mname0_check)]
        probs_finite = bool(np.all(np.isfinite(probs)))
        probs_sum_ok = bool(abs(float(np.sum(probs)) - 1.0) < 1e-4)
        chk(f"s1_forward_softmax_finite_and_normalized[{arm0_check}]", probs_finite and probs_sum_ok,
            detail=f"sum={float(np.sum(probs))}")
        u_check = draw_uniform(derive_key(cfg["rng"]["master_seed"], tid0_check, mname0_check, 0, "size_uniform"))
        size_check = size_from_uniform(probs, u_check)
        chk("s1_size_from_uniform_in_support", MIN_ATOMS <= size_check <= MIN_ATOMS + N_CLASSES - 1,
            detail=size_check)

    except Exception as e:
        import traceback as _tb
        write_report(crash={"error_type": type(e).__name__, "error": str(e)[:500],
                             "trace": _tb.format_exc()[-2000:]})
        raise

    report = write_report()
    print(f"generation gates: {'ALL PASS' if report['all_passed'] else 'FAILED'}")
    return 0 if report["all_passed"] else 1


def mode_run(args, ctx):
    cfg, targets, prior, s1 = ctx["cfg"], ctx["targets"], ctx["prior"], ctx["s1"]
    gr = json.load(open(args.generation_gate_report))
    assert gr["all_passed"], "generation gate 미통과"
    assert gr["runner_sha256"] == sha256_file(__file__), "gate 이후 runner 코드가 바뀜"
    assert gr["common_sha256"] == sha256_file(_HERE / "pilot_common.py")
    assert gr["config_sha256"] == ctx["cfg_sha"] and gr["pins_sha256"] == ctx["pins_sha"]
    # [astra_review_20260928.md Q1, 2026-09-30 자체 재검토] gate가 검증한 결정성이 run에도
    # 그대로 적용돼야 gate의 재현성 검증이 run에 의미가 있다 -- A1Runner()보다 먼저 적용하고,
    # gate report에 기록된 설정과 정확히 같은지 확인한다(다르면 gate가 다른 설정을 검증한
    # 것이므로 재현성 보장이 깨짐).
    determinism_settings = apply_determinism_settings()
    assert gr.get("determinism_settings") == determinism_settings, (
        f"gate가 검증한 결정성 설정과 지금 run의 설정이 다름 -- gate의 재현성 검증이 "
        f"이 run에 적용 안 됨: gate={gr.get('determinism_settings')} run={determinism_settings}")
    # [astra_review_20260927.md P1-2] gate report가 지금 실행하려는 입력(targets/prior/
    # inputs_manifest)과 같은 대상에 대해 통과했는지도 확인한다 -- 다른 입력에 재사용 금지.
    inputs_dir = ctx["inputs_dir"]
    assert gr.get("targets_sha256") == sha256_file(inputs_dir / "targets.json"), \
        "gate report가 통과한 targets.json과 지금 입력이 다름"
    assert gr.get("prior_sha256") == sha256_file(inputs_dir / "prior.json"), \
        "gate report가 통과한 prior.json과 지금 입력이 다름"
    assert gr.get("inputs_manifest_sha256") == sha256_file(inputs_dir / "inputs_manifest.json"), \
        "gate report가 통과한 inputs_manifest.json과 지금 입력이 다름"
    dists = build_size_dists(cfg, targets, prior, s1, args.include_auxiliary)
    attempts = build_plan(cfg, targets, dists, args.include_auxiliary)
    size_dists_json = json.dumps({f"{k[0]}|{k[1]}|{k[2]}": v.tolist() for k, v in dists.items()})
    if ctx.get("resumed"):
        # [astra_review_20260927.md P1-1] 어떤 쓰기보다 먼저: 기존 completed record가
        # 지금 다시 계산한 plan과 attempt 단위로 정확히 일치하는지, 참조하는 배열 파일이
        # 실제로 존재하고 SHA가 맞는지 확인한다. 하나라도 다르면 즉시 거부.
        verify_resume_consistency(ctx["out_dir"], attempts)
        print("resume 신원 검증 통과: 기존 completed record가 현재 plan과 일치함")
        # [astra_review_20260928.md P2] 이전 버전은 resume 때마다 최초 plan.json/
        # size_dists.json을 지금 다시 계산한 값으로 덮어썼다 -- identity 검증을 통과했어도
        # 재계산이 최초 실행 때와 100% 같다는 보장은 이 파일들 자체가 서는 증거였는데,
        # 덮어쓰면 그 증거가 사라진다. 최초 실행의 파일은 그대로 두고 지금 값과 바이트
        # 단위로 같은지만 확인한다(다르면 identity 검증을 통과한 것 자체가 의심스러운
        # 상황이므로 진행하지 않는다).
        prev_plan = json.loads((ctx["out_dir"] / "plan.json").read_text())
        if prev_plan.get("attempts") != attempts:
            raise RuntimeError(
                "resume 거부: identity 검증은 통과했지만 재계산한 plan이 최초 plan.json과 다름 "
                "-- 결정성이 깨졌을 가능성, 수동 확인 필요")
        prev_size_dists = (ctx["out_dir"] / "size_dists.json").read_text()
        if json.loads(prev_size_dists) != json.loads(size_dists_json):
            raise RuntimeError(
                "resume 거부: identity 검증은 통과했지만 재계산한 size_dists가 최초 size_dists.json과 "
                "다름 -- 결정성이 깨졌을 가능성, 수동 확인 필요")
    else:
        (ctx["out_dir"] / "plan.json").write_text(json.dumps({"attempts": attempts}, indent=2))
        (ctx["out_dir"] / "size_dists.json").write_text(size_dists_json)
    r = A1Runner(ctx["pins"], cfg, targets)
    t0 = time.time()
    counts = run_attempts(attempts, r.generate, r.evaluate, ctx["out_dir"])
    counts["elapsed_total_s"] = time.time() - t0
    counts["include_auxiliary"] = args.include_auxiliary
    counts["generation_gate_report"] = str(Path(args.generation_gate_report).resolve())
    (ctx["out_dir"] / "run_summary.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps({k: v for k, v in counts.items()
                      if k not in ("unfinished_ids", "evaluation_failed_ids")}, indent=2))
    # [astra_review_20260928.md P1-2] "완료"는 unfinished_ids가 비었는지만으로 판단하면
    # 안 된다 -- evaluation_failed(평가가 깨져서 completed로 승격 못 한 attempt)가 남아
    # 있으면 unfinished_ids에는 안 잡히지만(terminal 이벤트는 있으니) 실제로는 재평가가
    # 필요한 상태다. aborted나 evaluation_failed가 하나라도 있으면 성공 종료(0)로 보지 않는다.
    if counts["aborted"]:
        return 3
    if counts["unfinished_ids"]:
        return 2
    if counts["evaluation_failed_ids"]:
        return 4
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["plan", "gates", "run"])
    parser.add_argument("--config", default=str(_HERE / "pilot_config.json"))
    parser.add_argument("--pins", default=str(_HERE / "pilot_launch_pins.json"))
    parser.add_argument("--inputs-dir", required=True)
    parser.add_argument("--s1-root", default=".")
    parser.add_argument("--out-root", default="./pilot_runs")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--generation-gate-report", default=None)
    parser.add_argument("--include-auxiliary", action="store_true",
                        help="보조 패널 포함(자원 예산 사용자 승인 필요, 주패널 결과를 보기 전에 결정)")
    args = parser.parse_args()

    from run_logger import RunLogger
    cfg_path = Path(args.config).resolve()
    pins_path = Path(args.pins).resolve()
    args.inputs_dir = str(Path(args.inputs_dir).resolve())
    args.s1_root = str(Path(args.s1_root).resolve())
    out_root = Path(args.out_root).resolve()
    if args.generation_gate_report:
        args.generation_gate_report = str(Path(args.generation_gate_report).resolve())
    run_logger = RunLogger(__file__, source_paths=(cfg_path, pins_path, _HERE / "pilot_common.py")).start()
    run_logger.record_arguments(vars(args))

    cfg_path_, pins_path_, cfg, pins, inputs_dir, targets, prior = load_inputs(args)
    identity = {
        "runner_sha256": sha256_file(__file__), "common_sha256": sha256_file(_HERE / "pilot_common.py"),
        "config_sha256": sha256_file(cfg_path), "pins_sha256": sha256_file(pins_path),
        "targets_sha256": sha256_file(inputs_dir / "targets.json"),
        "prior_sha256": sha256_file(inputs_dir / "prior.json"),
        "inputs_manifest_sha256": sha256_file(inputs_dir / "inputs_manifest.json"),
        "s1_checkpoint_sha256": {tag: spec["sha256"] for tag, spec in cfg["s1_models"]["checkpoints"].items()},
        "size_predictor_module_sha256": size_predictor_module_sha(),
        "include_auxiliary": args.include_auxiliary,
    }

    resumed = False
    if args.resume_from:
        out_dir = Path(args.resume_from).resolve()
        assert out_dir.exists() and args.mode == "run", "resume은 run 모드에서 기존 디렉터리만"
        lm_path = out_dir / "launch_manifest.json"
        assert lm_path.exists(), f"resume 거부: {lm_path} 없음 -- 이 디렉터리의 신원을 확인할 수 없음"
        prev = json.load(open(lm_path))
        mismatches = {k: (prev.get(k), v) for k, v in identity.items() if prev.get(k) != v}
        if mismatches:
            raise RuntimeError(
                "resume 거부: 현재 설정/입력이 이 run이 시작될 때와 다름(아래 필드 불일치, "
                "어떤 쓰기도 하지 않고 중단):\n" +
                "\n".join(f"  {k}: 기록={old!r} 현재={new!r}" for k, (old, new) in mismatches.items()))
        resumed = True
    else:
        out_dir = out_root / f"{args.mode}_{run_logger.run_id}"
        out_dir.mkdir(parents=True, exist_ok=False)

    # [astra_review_20260927.md P1-1] 동시 resume/실행을 lock으로 거부한다. 어떤 쓰기보다
    # 먼저 획득하고, 끝나면(정상/예외 무관) 반드시 해제한다.
    lock_path = out_dir / ".lock"
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(lock_fd, f"pid={os.getpid()} run_id={run_logger.run_id} time={time.time()}\n".encode())
        os.close(lock_fd)
    except FileExistsError:
        raise RuntimeError(
            f"resume/실행 거부: {lock_path}가 이미 존재함(다른 프로세스가 이 디렉터리를 쓰고 있거나 "
            "이전 실행이 비정상 종료됨). 정말 이전 실행이 끝났다면 수동으로 확인 후 lock 파일을 지울 것.")

    try:
        ctx = {"cfg": cfg, "pins": pins, "targets": targets, "prior": prior, "out_dir": out_dir,
               "cfg_sha": identity["config_sha256"], "pins_sha": identity["pins_sha256"],
               "inputs_dir": inputs_dir, "resumed": resumed,
               "s1": load_s1(cfg, args.s1_root)}
        manifest_body = {
            "mode": args.mode, "run_id": run_logger.run_id, **identity,
            "inputs_dir": str(inputs_dir), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "created": time.time(),
        }
        if resumed:
            # 기존 launch_manifest.json은 그대로 두고(최초 실행의 신원 기록 보존), 이번
            # resume 세션의 기록을 새 timestamp 파일로 별도 추가한다.
            (out_dir / f"resume_manifest_{run_logger.run_id}.json").write_text(
                json.dumps(manifest_body, indent=2))
        else:
            (out_dir / "launch_manifest.json").write_text(json.dumps(manifest_body, indent=2))
        code = {"plan": mode_plan, "gates": mode_gates, "run": mode_run}[args.mode](args, ctx)
        run_logger.finish("completed" if code == 0 else f"exit_{code}", out_dir=str(out_dir))
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass
    sys.exit(code)


if __name__ == "__main__":
    main()
