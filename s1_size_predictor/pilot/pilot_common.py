"""[S-1 pilot] 공용 함수: SHA, RNG key 유도, 크기 추첨(inverse CDF), 고정 모듈 import,
ledger 상태 기계(runner/analyzer가 이벤트를 각자 해석하며 어긋나던 문제 -- astra_review_
20260928.md P1-2/P1-3, opus_review_20260928.md 1-1/1-2 공통 원인 -- 를 한 곳으로 모음).

pilot_config.json의 rng/size_sampling 규칙을 그대로 구현한다. Python의 hash()는
프로세스마다 달라지므로 쓰지 않는다.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

MIN_ATOMS, MAX_ATOMS = 2, 50
N_CLASSES = MAX_ATOMS - MIN_ATOMS + 1
PROP_ORDER = ["HOMO", "LUMO", "LogP", "TPSA", "HBA", "RotBonds", "AromaticRings"]
GNN_KEYS = ["HOMO", "LUMO"]
STRUCT_KEYS = ["LogP", "TPSA", "HBA", "RotBonds", "AromaticRings"]
assert set(GNN_KEYS + STRUCT_KEYS) == set(PROP_ORDER)

# ledger 이벤트 순서(정상 경로): started -> generated -> (completed | evaluation_failed)
# generate_fn 자체가 실패하면: started -> error (generated 없음)
TERMINAL_EVENTS = ("completed", "evaluation_failed", "error")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_key(master_seed, target_id, mask_name, repeat, purpose):
    """안정된 64bit 정수 key. purpose in {'size_uniform','denoise'}.
    arm은 key에 넣지 않는다 -> 같은 (target,mask,repeat)의 모든 arm이 같은 U와 같은
    denoising seed를 공유(pairing, pilot_config.json size_sampling.pairing)."""
    s = f"{master_seed}|{target_id}|{mask_name}|{repeat}|{purpose}"
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big") % (2 ** 63)


def draw_uniform(key):
    """크기 전용 CPU generator로 U in [0,1) 한 개. 전역 RNG를 건드리지 않는다."""
    g = torch.Generator(device="cpu")
    g.manual_seed(key)
    return float(torch.rand(1, generator=g, dtype=torch.float64).item())


def size_from_uniform(probs, u):
    """probs: 길이 49(원자수 2..50)의 확률. inverse CDF로 원자수 하나를 결정."""
    cdf = np.cumsum(np.asarray(probs, dtype=np.float64))
    idx = int(np.searchsorted(cdf, u, side="left"))
    idx = min(idx, N_CLASSES - 1)  # cdf[-1]이 부동소수점으로 1 미만일 때 끝 class로 클램프
    return idx + MIN_ATOMS


def mask_vector(mask_spec_observed):
    v = [1.0 if n in set(mask_spec_observed) else 0.0 for n in PROP_ORDER]
    return v


def read_ledger_safely(ledger_path):
    """runner/analyzer 공용. 마지막 줄이 중간에 끊겨 있어도(전원 차단 등) 그 앞까지는
    읽는다. 중간 줄이 손상되면 조용히 넘기지 않고 즉시 실패한다(자료 자체를 신뢰할 수
    없으므로). 이전에는 pilot_run.py와 analyze_pilot.py에 거의 같은 함수가 각각
    복붙돼 있어 두 정책이 갈라질 위험이 있었다(astra_review_20260928.md P2)."""
    if not Path(ledger_path).exists():
        return [], None
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
                raise
    return events, corrupt_tail


def ledger_state(events, planned_ids=None):
    """[astra_review_20260928.md P1-2/P1-3, opus_review_20260928.md 1-1/1-2] attempt_id별
    "최신 상태"와 전체 retry 이력을 분리해서 돌려준다. runner(resume skip 판단)와
    analyzer(집계) 둘 다 이 함수 하나로 상태를 읽어서, 두 쪽의 해석이 갈라지지 않게 한다.

    한 attempt_id는 started -> (error | generated -> (completed | evaluation_failed))
    순서로 재시도될 수 있다. "최신 상태"는 그 attempt_id의 가장 나중 terminal 이벤트
    (completed/evaluation_failed/error) 기준이며, 그보다 앞선 error/evaluation_failed는
    "history"로만 남고 최신 상태 집계에는 안 들어간다(Astra/Opus 공통 반례: error 후
    재시도로 completed가 되면 옛 error를 더는 세지 않아야 missing이 음수가 안 됨).

    반환:
      latest: {attempt_id: 그 id의 마지막 terminal 이벤트 dict}  (completed/evaluation_failed/error 중 하나)
      history: {attempt_id: [그 id의 모든 terminal 이벤트, 시간순]}
      generated: {attempt_id: 마지막 generated 이벤트}  (아직 평가 전이거나 평가 실패한 것도 포함 -- 재평가용)
      counts: {"completed": n, "evaluation_failed": n, "error": n, "pending": n}
              -- planned_ids가 주어지면 pending = planned - (completed+evaluation_failed+error 최신 상태 개수)
      unknown_ids: planned_ids에 없는데 ledger에 등장하는 attempt_id 목록(계획 밖 ID)
      duplicate_completed: 같은 attempt_id에 completed 이벤트가 2개 이상 있는 경우 목록
                           (정상적으로는 생기면 안 됨 -- run_attempts가 이미 completed인
                           attempt를 건너뛰므로. 생겼다면 ledger 손상/동시쓰기 의심.)
    """
    latest = {}
    history = {}
    generated = {}
    duplicate_completed_ids = set()
    for ev in events:
        aid = ev.get("attempt_id")
        if aid is None:
            continue
        if ev["event"] == "generated":
            generated[aid] = ev
        elif ev["event"] in TERMINAL_EVENTS:
            if ev["event"] == "completed" and aid in latest and latest[aid]["event"] == "completed":
                duplicate_completed_ids.add(aid)
            latest[aid] = ev
            history.setdefault(aid, []).append(ev)

    counts = {"completed": 0, "evaluation_failed": 0, "error": 0}
    for ev in latest.values():
        counts[ev["event"]] += 1
    unknown_ids = []
    if planned_ids is not None:
        planned_set = set(planned_ids)
        unknown_ids = [aid for aid in latest if aid not in planned_set]
        n_terminal_in_plan = sum(1 for aid in latest if aid in planned_set)
        counts["pending"] = len(planned_set) - n_terminal_in_plan
    return {
        "latest": latest, "history": history, "generated": generated, "counts": counts,
        "unknown_ids": unknown_ids, "duplicate_completed_ids": sorted(duplicate_completed_ids),
    }


def _key_ok(props, key):
    if key not in props:
        return False
    v = props[key]
    if v is None:
        return False
    try:
        return np.isfinite(v)
    except TypeError:
        return False


def classify_props_failure(strict_valid, props):
    """[astra_review_20260928.md P1-1, opus_review_20260928.md 1-5] 화학적으로 invalid한
    분자(정상적으로 props=None)와 evaluator가 일부만 조용히 망가진 경우(예: GNN proxy만
    항상 None, 구조 descriptor는 정상)를 구분한다. strict_valid=False면 애초에 평가
    대상이 아니므로 두 그룹 다 "판단 불가"(해당 없음)로 둔다 -- 이전 버전은 모든 값이
    None일 때만 감지해 "구조 5개 정상 + HOMO/LUMO만 None"을 놓쳤다(두 리뷰의 공통 반례).

    반환: {"applicable": bool, "struct_failed": bool, "struct_missing": [...],
           "gnn_failed": bool, "gnn_missing": [...]}
    applicable=False면 이 attempt는 evaluator 건강도 판단에 쓰지 않는다(invalid 분자).
    """
    if not strict_valid:
        return {"applicable": False, "struct_failed": False, "struct_missing": [],
                "gnn_failed": False, "gnn_missing": []}
    props = props or {}
    struct_missing = [k for k in STRUCT_KEYS if not _key_ok(props, k)]
    gnn_missing = [k for k in GNN_KEYS if not _key_ok(props, k)]
    return {"applicable": True, "struct_failed": len(struct_missing) > 0, "struct_missing": struct_missing,
            "gnn_failed": len(gnn_missing) > 0, "gnn_missing": gnn_missing}


def import_pinned(frozen_dir):
    """고정 디렉터리에서 model3/train3/qm_props를 import하고 (모듈들, 원래 cwd)를 반환.
    qm_props는 import 시 cwd 기준 상대경로 'c1_regressor.pt'를 읽고 sys.path를
    바꾸므로 cwd를 고정 디렉터리로 잠깐 옮겼다가 복원한다. train3를 qm_props보다
    먼저 import해야 qm_props의 sys.path 삽입이 train3 해석을 바꾸지 않는다."""
    frozen_dir = str(frozen_dir)
    orig = os.getcwd()
    sys.path.insert(0, frozen_dir)
    import model3
    import train3
    os.chdir(frozen_dir)
    try:
        import qm_props
    finally:
        os.chdir(orig)
    import c1_prop_regressor
    return model3, train3, qm_props, c1_prop_regressor
