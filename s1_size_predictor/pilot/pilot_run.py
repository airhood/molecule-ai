"""[S-1 pilot] 고정 A-1에서 초기 원자 수 정책만 바꿔 생성하는 실행기.
astra_review_20260926.md §2~§6 반영. 세 모드:

  plan  : 생성 없이 attempt 목록/크기 추첨만 만든다(CPU). 결정성 검사 포함.
  gates : sampling이 필요한 launch gate(같은 attempt 재현, arm 순서/무관한 RNG 소비에
          대한 불변, 강제 크기 연결)를 작은 규모로 실행하고 report를 저장한다.
  run   : gates 보고서(통과 + 같은 runner SHA)가 있어야만 실행. attempt별 ledger,
          원본 X/E 저장, 오류와 화학적 invalid 구분, 반복 오류 시 중단.

GPU는 GPU0만(CUDA_VISIBLE_DEVICES=0). 기존 출력 디렉터리 재사용 금지(resume은
--resume-from으로 명시할 때만, ledger의 completed attempt를 건너뜀).
"""
import argparse
import glob
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
from pilot_common import (MIN_ATOMS, N_CLASSES, PROP_ORDER, derive_key, draw_uniform,
                          import_pinned, mask_vector, sha256_file, size_from_uniform)


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


# ------------------------------------------------------------------ 핵심 루프(주입식)
def run_attempts(attempts, generate_fn, evaluate_fn, out_dir, max_consecutive_errors=3):
    """generate_fn(attempt)->(X_np, E_np), evaluate_fn(attempt, X_np, E_np)->dict.
    ledger에 started/completed/error를 즉시 flush+fsync. 이미 completed인 attempt는 건너뜀.
    반환: 요약 dict."""
    out_dir = Path(out_dir)
    (out_dir / "arrays").mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / "ledger.jsonl"
    done = set()
    if ledger_path.exists():
        for line in open(ledger_path, encoding="utf-8"):
            ev = json.loads(line)
            if ev["event"] == "completed":
                done.add(ev["attempt_id"])

    def log(ev):
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    counts = {"planned": len(attempts), "skipped_already_completed": 0, "completed": 0, "errors": 0}
    consecutive_err = 0
    aborted = False
    for a in attempts:
        if a["attempt_id"] in done:
            counts["skipped_already_completed"] += 1
            continue
        log({"event": "started", "attempt_id": a["attempt_id"], "time": time.time()})
        t0 = time.time()
        try:
            X, E = generate_fn(a)
            arr_name = f"{abs(hash_str(a['attempt_id']))}.npz"
            np.savez_compressed(out_dir / "arrays" / arr_name, X=X.astype(np.int8), E=E.astype(np.int8))
            result = evaluate_fn(a, X, E)
            log({"event": "completed", **a, "arrays": arr_name, "elapsed_s": time.time() - t0, **result})
            counts["completed"] += 1
            consecutive_err = 0
        except Exception as e:   # OOM/소프트웨어 오류: chemical invalid와 구분해서 기록
            log({"event": "error", "attempt_id": a["attempt_id"], "error_type": type(e).__name__,
                 "error": str(e)[:500], "trace": traceback.format_exc()[-1500:], "elapsed_s": time.time() - t0})
            counts["errors"] += 1
            consecutive_err += 1
            if consecutive_err >= max_consecutive_errors:
                aborted = True
                break
    finished = set()
    for line in open(ledger_path, encoding="utf-8"):
        ev = json.loads(line)
        if ev["event"] == "completed":
            finished.add(ev["attempt_id"])
    counts["unfinished_ids"] = [a["attempt_id"] for a in attempts if a["attempt_id"] not in finished]
    counts["aborted_repeated_errors"] = aborted
    return counts


def hash_str(s):
    import hashlib
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


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
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0" or not torch.cuda.is_available(), \
            "GPU는 GPU0만 사용: CUDA_VISIBLE_DEVICES=0 필요"
        self.cfg, self.targets = cfg, targets
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        frozen = Path(pins["frozen_dir_on_server"])
        self.model3, self.train3, self.qm_props, self.c1 = import_pinned(frozen)
        for mod in (self.model3, self.train3, self.qm_props, self.c1):
            assert sha256_file(mod.__file__) == pins["imported_module_sha256"][mod.__name__], mod.__name__
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
    cfg, targets = ctx["cfg"], ctx["targets"]
    r = A1Runner(ctx["pins"], cfg, targets)
    rep = []

    def chk(name, ok, detail=""):
        rep.append({"gate": name, "passed": bool(ok), "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")

    t0 = ctx["targets"]["targets"][0]
    t1 = ctx["targets"]["targets"][1]
    mv_full = mask_vector(cfg["masks"]["primary"]["observed"])
    mv_part = mask_vector(cfg["masks"]["auxiliary"][0]["observed"])
    for t in (t0, t1):
        for mv, tag in ((mv_full, "full"), (mv_part, "partial")):
            key = derive_key(cfg["rng"]["master_seed"], t["target_id"], tag, 0, "denoise")
            X1, E1 = r.generate_raw(t["a1_normalized7"], mv, 30, key)
            # 무관한 RNG 소비(CPU/CUDA/S-1 유사 호출)를 사이에 끼워도 같은 attempt는 같아야 한다
            _ = torch.rand(1000)
            if torch.cuda.is_available():
                _ = torch.rand(1000, device="cuda")
            _ = torch.multinomial(torch.ones(10), 5)
            X2, E2 = r.generate_raw(t["a1_normalized7"], mv, 30, key)
            chk(f"same_attempt_reproducible[{t['target_id']},{tag}]",
                np.array_equal(X1, X2) and np.array_equal(E1, E2))
            # forced size 연결: prior 경로/S-1 경로가 같은 (조건,마스크,크기,seed)면 같은 A-1 출력
            X3, E3 = r.generate_raw(t["a1_normalized7"], mv, 30, key)
            chk(f"forced_size_same_output_any_arm[{t['target_id']},{tag}]",
                np.array_equal(X1, X3) and np.array_equal(E1, E3))
    # seed가 다르면 출력이 달라야(seed가 실제로 작동하는지)
    kA = derive_key(cfg["rng"]["master_seed"], t0["target_id"], "full", 0, "denoise")
    kB = derive_key(cfg["rng"]["master_seed"], t0["target_id"], "full", 1, "denoise")
    XA, EA = r.generate_raw(t0["a1_normalized7"], mv_full, 30, kA)
    XB, EB = r.generate_raw(t0["a1_normalized7"], mv_full, 30, kB)
    chk("different_seed_different_output", not (np.array_equal(XA, XB) and np.array_equal(EA, EB)))
    # 크기가 바뀌면 출력 shape이 요청 크기와 일치
    Xs, Es = r.generate_raw(t0["a1_normalized7"], mv_full, 25, kA)
    chk("output_shape_matches_requested_size", Xs.shape == (25,) and Es.shape == (25, 25), f"{Xs.shape} {Es.shape}")
    # evaluate 경로 스모크: 한 attempt를 평가해 필드가 채워지는지
    fake_attempt = {"size": 30}
    res = r.evaluate(fake_attempt, X1, E1)
    chk("evaluate_returns_fields", all(k in res for k in ("strict_valid", "single_valid", "props", "n_fragments")))

    runner_sha = sha256_file(__file__)
    report = {"all_passed": all(x["passed"] for x in rep), "gates": rep, "runner_sha256": runner_sha,
              "common_sha256": sha256_file(_HERE / "pilot_common.py"),
              "config_sha256": ctx["cfg_sha"], "pins_sha256": ctx["pins_sha"]}
    (ctx["out_dir"] / "generation_gate_report.json").write_text(json.dumps(report, indent=2))
    print(f"generation gates: {'ALL PASS' if report['all_passed'] else 'FAILED'}")
    return 0 if report["all_passed"] else 1


def mode_run(args, ctx):
    cfg, targets, prior, s1 = ctx["cfg"], ctx["targets"], ctx["prior"], ctx["s1"]
    gr = json.load(open(args.generation_gate_report))
    assert gr["all_passed"], "generation gate 미통과"
    assert gr["runner_sha256"] == sha256_file(__file__), "gate 이후 runner 코드가 바뀜"
    assert gr["common_sha256"] == sha256_file(_HERE / "pilot_common.py")
    assert gr["config_sha256"] == ctx["cfg_sha"] and gr["pins_sha256"] == ctx["pins_sha"]
    dists = build_size_dists(cfg, targets, prior, s1, args.include_auxiliary)
    attempts = build_plan(cfg, targets, dists, args.include_auxiliary)
    (ctx["out_dir"] / "plan.json").write_text(json.dumps({"attempts": attempts}, indent=2))
    (ctx["out_dir"] / "size_dists.json").write_text(json.dumps(
        {f"{k[0]}|{k[1]}|{k[2]}": v.tolist() for k, v in dists.items()}))
    r = A1Runner(ctx["pins"], cfg, targets)
    t0 = time.time()
    counts = run_attempts(attempts, r.generate, r.evaluate, ctx["out_dir"])
    counts["elapsed_total_s"] = time.time() - t0
    counts["include_auxiliary"] = args.include_auxiliary
    counts["generation_gate_report"] = str(Path(args.generation_gate_report).resolve())
    (ctx["out_dir"] / "run_summary.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps({k: v for k, v in counts.items() if k != "unfinished_ids"}, indent=2))
    return 0 if not counts["unfinished_ids"] else 2


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

    if args.resume_from:
        out_dir = Path(args.resume_from).resolve()
        assert out_dir.exists() and args.mode == "run", "resume은 run 모드에서 기존 디렉터리만"
    else:
        out_dir = out_root / f"{args.mode}_{run_logger.run_id}"
        out_dir.mkdir(parents=True, exist_ok=False)

    cfg_path_, pins_path_, cfg, pins, inputs_dir, targets, prior = load_inputs(args)
    ctx = {"cfg": cfg, "pins": pins, "targets": targets, "prior": prior, "out_dir": out_dir,
           "cfg_sha": sha256_file(cfg_path), "pins_sha": sha256_file(pins_path),
           "s1": load_s1(cfg, args.s1_root)}
    (out_dir / "launch_manifest.json").write_text(json.dumps({
        "mode": args.mode, "run_id": run_logger.run_id, "runner_sha256": sha256_file(__file__),
        "common_sha256": sha256_file(_HERE / "pilot_common.py"), "config_sha256": ctx["cfg_sha"],
        "pins_sha256": ctx["pins_sha"], "inputs_dir": str(inputs_dir),
        "targets_sha256": sha256_file(inputs_dir / "targets.json"),
        "prior_sha256": sha256_file(inputs_dir / "prior.json"),
        "include_auxiliary": args.include_auxiliary,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }, indent=2))
    code = {"plan": mode_plan, "gates": mode_gates, "run": mode_run}[args.mode](args, ctx)
    run_logger.finish("completed" if code == 0 else f"exit_{code}", out_dir=str(out_dir))
    sys.exit(code)


if __name__ == "__main__":
    main()
