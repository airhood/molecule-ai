"""[S-1 pilot] launch gate 검사 (astra_review_20260926.md §6). 생성/sampling은
실행하지 않는다 -- 파일 SHA, 실제 import된 모듈 SHA, A-1/S-1 strict load,
정규화 재현, mask 불변성(forward 1회)만 검사한다. RNG 재현성과 같은
attempt 재현 게이트는 sampling이 필요하므로 여기서 실행하지 않고
report에 not_run으로 명시한다.

하나라도 실패하면 exit code 1. 결과는 새 디렉터리의 gate_report.json에 저장.
"""
import argparse
import glob
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))          # size_predictor
sys.path.insert(0, str(_HERE.parent.parent))   # run_logger
from run_logger import RunLogger


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Report:
    def __init__(self):
        self.items = []

    def check(self, name, ok, detail=""):
        self.items.append({"gate": name, "passed": bool(ok), "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
        return bool(ok)

    @property
    def all_passed(self):
        # passed=None은 "실행 안 함(not run)" 표시 -- 통과/실패 판정에서 제외하되
        # 별도로 not_run 목록에 남긴다(None을 실패로 세면 통과한 게이트가
        # FAILED로 보고되는 오류가 생김, 2026-09-27 발견).
        return all(i["passed"] for i in self.items if i["passed"] is not None)

    @property
    def not_run(self):
        return [i["gate"] for i in self.items if i["passed"] is None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(_HERE / "pilot_config.json"))
    parser.add_argument("--pins", default=str(_HERE / "pilot_launch_pins.json"))
    parser.add_argument("--inputs-dir", required=True, help="prepare_pilot_inputs.py가 만든 run 디렉터리")
    parser.add_argument("--s1-root", default=".", help="checkpoints_v2_size_predictor_seed*/ 가 있는 디렉터리")
    parser.add_argument("--features-dir", default="./features_v2_fixed")
    parser.add_argument("--processed-dir", default="../data/processed_ext7")
    parser.add_argument("--out-root", default="./pilot_gate_reports")
    args = parser.parse_args()

    # 모든 경로를 chdir 전에 절대경로로 고정(qm_props가 cwd 기준 상대경로를 쓰므로)
    cfg_path = Path(args.config).resolve()
    pins_path = Path(args.pins).resolve()
    inputs_dir = Path(args.inputs_dir).resolve()
    s1_root = Path(args.s1_root).resolve()
    features_dir = Path(args.features_dir).resolve()
    processed = Path(args.processed_dir).resolve()
    out_root = Path(args.out_root).resolve()
    orig_cwd = Path.cwd()

    run_logger = RunLogger(__file__, source_paths=(cfg_path, pins_path)).start()
    run_logger.record_arguments(vars(args))
    out_dir = out_root / run_logger.run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    cfg = json.load(open(cfg_path))
    pins = json.load(open(pins_path))
    frozen = Path(pins["frozen_dir_on_server"])
    rep = Report()

    print("== G1 고정 파일 SHA ==")
    for rel, expected in pins["files_sha256"].items():
        p = frozen / rel
        actual = sha256_file(p) if p.exists() else "MISSING"
        rep.check(f"file_sha:{rel}", actual == expected, f"{actual[:16]}")
    rep.check("evaluator_pin_not_null", pins["evaluator"]["rdkit_version"] is not None
              and pins["files_sha256"]["c1_regressor.pt"] is not None)
    import rdkit
    rep.check("rdkit_version", rdkit.__version__ == pins["evaluator"]["rdkit_version"], rdkit.__version__)
    rep.check("torch_version", torch.__version__ == pins["evaluator"]["torch_version"], torch.__version__)
    rep.check("a1_ckpt_matches_config", pins["files_sha256"]["checkpoints_c4_a1/best.pt"]
              == cfg["frozen_a1"]["checkpoint_sha256"])
    rep.check("a1_model_matches_config", pins["files_sha256"]["model3.py"]
              == cfg["frozen_a1"]["model_source_sha256"])

    print("== G2 입력 산출물 일관성 ==")
    manifest = json.load(open(inputs_dir / "inputs_manifest.json"))
    rep.check("config_sha_matches_inputs_manifest", sha256_file(cfg_path) == manifest["config_sha256"])
    rep.check("prior_sha", sha256_file(inputs_dir / "prior.json") == manifest["prior_json_sha256"])
    rep.check("targets_sha", sha256_file(inputs_dir / "targets.json") == manifest["targets_json_sha256"])
    norm_sha = sha256_file(features_dir / "normalization.json")
    rep.check("normalization_sha", norm_sha == cfg["s1_models"]["normalization_sha256"] == manifest["normalization_sha256"])
    rep.check("a1_stats_sha", sha256_file(processed / "stats.json") == cfg["frozen_a1"]["stats_json_sha256"])
    prior = json.load(open(inputs_dir / "prior.json"))
    rep.check("prior_sum_1_and_49_classes", abs(sum(prior["probs"]) - 1) < 1e-9 and len(prior["probs"]) == 49)

    print("== G3 실제 import된 모듈 SHA (고정 디렉터리에서 import) ==")
    sys.path.insert(0, str(frozen))
    import model3
    import train3
    os.chdir(frozen)                      # qm_props가 'c1_regressor.pt'를 cwd 기준으로 로드
    import qm_props
    os.chdir(orig_cwd)
    import c1_prop_regressor
    for mod in (model3, train3, qm_props, c1_prop_regressor):
        name = mod.__name__
        actual = sha256_file(mod.__file__)
        rep.check(f"imported:{name}", actual == pins["imported_module_sha256"][name],
                  f"{mod.__file__} {actual[:16]}")
    rep.check("model3_from_frozen_dir", str(Path(model3.__file__).resolve().parent) == str(frozen))
    rep.check("train3.analyze_molecule_exists", hasattr(train3, "analyze_molecule"))

    print("== G4 A-1 strict load ==")
    sd = torch.load(frozen / "checkpoints_c4_a1/best.pt", map_location="cpu", weights_only=False)
    a1 = model3.MoleculeGraphDiffusion(sd["schedule.m_X"].clone(), sd["schedule.m_E"].clone())
    try:
        res = a1.load_state_dict(sd, strict=True)
        ok = (len(res.missing_keys) == 0 and len(res.unexpected_keys) == 0)
        rep.check("a1_strict_load", ok, f"missing={res.missing_keys} unexpected={res.unexpected_keys}")
    except Exception as e:
        rep.check("a1_strict_load", False, repr(e)[:200])
    n_params = sum(p.numel() for p in a1.parameters())
    rep.check("a1_param_count", n_params == 5754052, n_params)
    rep.check("a1_T_STEPS", model3.T_STEPS == cfg["frozen_a1"]["T_STEPS"], model3.T_STEPS)
    rep.check("a1_cond_props", list(model3.COND_PROP_NAMES) == ["HOMO", "LUMO", "LogP", "TPSA", "HBA", "RotBonds", "AromaticRings"]
              and list(model3.COND_PROP_INDICES) == [0, 1, 5, 6, 8, 9, 10])
    a1.eval()

    print("== G5 A-1 정규화 == QMugsDataset(stats.json) ==")
    from dataset2 import QMugsDataset
    ds = QMugsDataset(str(processed), split="val", max_samples=1)
    idx = model3.COND_PROP_INDICES
    stats = json.load(open(processed / "stats.json"))
    cols = list(stats.keys())
    a1_mean = np.array([stats[cols[i]]["mean"] for i in idx], dtype=np.float32)
    a1_std = np.array([stats[cols[i]]["std"] for i in idx], dtype=np.float32)
    rep.check("a1_norm_equals_dataset2", np.array_equal(ds.mean[idx].numpy(), a1_mean)
              and np.array_equal(ds.std[idx].numpy(), a1_std))
    targets = json.load(open(inputs_dir / "targets.json"))
    ok = True
    for t in targets["targets"]:
        raw7 = np.array([t["raw7"][n] for n in targets["prop_order"]], dtype=np.float32)
        ok &= np.array_equal(((raw7 - a1_mean) / a1_std).astype(np.float32),
                             np.array(t["a1_normalized7"], dtype=np.float32))
    rep.check("targets_a1_normalized_reproduce", ok, f"{len(targets['targets'])} targets")

    print("== G6 S-1 checkpoint SHA + strict load + mask 불변성 ==")
    from size_predictor import SizePredictor, COND_DIM
    s1_models = {}
    for tag, spec in cfg["s1_models"]["checkpoints"].items():
        matches = glob.glob(str(s1_root / spec["path"]))
        if len(matches) != 1:
            rep.check(f"s1_{tag}_unique_ckpt", False, f"{len(matches)} matches")
            continue
        rep.check(f"s1_{tag}_sha", sha256_file(matches[0]) == spec["sha256"])
        state = torch.load(matches[0], map_location="cpu", weights_only=False)
        m = SizePredictor()
        try:
            res = m.load_state_dict(state["model"], strict=True)
            rep.check(f"s1_{tag}_strict_load", len(res.missing_keys) == 0 and len(res.unexpected_keys) == 0)
        except Exception as e:
            rep.check(f"s1_{tag}_strict_load", False, repr(e)[:200])
            continue
        m.eval()
        s1_models[tag] = m
        g = torch.Generator().manual_seed(0)
        cond = torch.randn(256, COND_DIM, generator=g)
        mask = (torch.rand(256, COND_DIM, generator=g) > 0.5).float()
        garbage = torch.where(mask.bool(), cond, torch.full_like(cond, 123.0))
        with torch.no_grad():
            d = (m(cond, mask) - m(garbage, mask)).abs().max().item()
        rep.check(f"s1_{tag}_mask_invariance", d == 0.0, f"max|dlogit|={d}")
        with torch.no_grad():
            p_null = torch.softmax(m(torch.zeros(1, COND_DIM), torch.zeros(1, COND_DIM)), -1)[0].numpy()
        tv = 0.5 * np.abs(p_null - np.array(prior["probs"])).sum()
        # 정보용: 수학적으로 같아야 하는 게이트 아님(Astra §6-2). 값만 기록.
        print(f"        s1_{tag} all-null vs prior TV={tv:.5f} (informational)")
        rep.items.append({"gate": f"s1_{tag}_allnull_vs_prior_TV_informational", "passed": True, "detail": f"{tv:.6f}"})

    print("== G7 A-1 forward mask 불변성(sampling 아님, forward 1회) ==")
    torch.manual_seed(0)
    B, N = 4, 12
    X_t = torch.randint(1, model3.K_X, (B, N))
    E_t = torch.randint(0, model3.K_E, (B, N, N))
    node_mask = torch.ones(B, N, dtype=torch.bool)
    t = torch.randint(1, model3.T_STEPS + 1, (B,))
    cond = torch.randn(B, model3.COND_DIM)
    mask = torch.tensor([[1, 1, 0, 0, 1, 0, 1]] * B, dtype=torch.float32)
    garbage = torch.where(mask.bool(), cond, torch.full_like(cond, 123.0))
    with torch.no_grad():
        o1 = a1.net(X_t, E_t, node_mask, t, cond, mask)
        o2 = a1.net(X_t, E_t, node_mask, t, garbage, mask)
    diffs = [(a - b).abs().max().item() for a, b in zip(o1, o2) if isinstance(a, torch.Tensor)]
    rep.check("a1_forward_mask_invariance", max(diffs) == 0.0, f"max diff per output={diffs}")

    rep.items.append({"gate": "rng_reproducibility_same_attempt", "passed": None, "detail": "NOT RUN: sampling 필요 -- Astra 재검토/사용자 승인 후 별도 실행"})
    rep.items.append({"gate": "forced_size_prior_vs_s1_same_a1_output", "passed": None, "detail": "NOT RUN: sampling 필요"})

    report = {"all_non_generation_gates_passed": rep.all_passed, "not_run": rep.not_run,
              "launch_ready": False,
              "launch_ready_reason": "sampling 기반 게이트(rng_reproducibility 등)가 아직 not_run이므로 launch 준비 완료가 아님",
              "items": rep.items,
              "config_sha256": sha256_file(cfg_path), "pins_sha256": sha256_file(pins_path)}
    (out_dir / "gate_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n결과: 비생성 게이트 {'ALL PASS' if rep.all_passed else 'FAILED'} | not_run={rep.not_run} "
          f"-> {out_dir / 'gate_report.json'}")
    run_logger.finish("completed" if rep.all_passed else "gates_failed", out_dir=str(out_dir))
    sys.exit(0 if rep.all_passed else 1)


if __name__ == "__main__":
    main()
