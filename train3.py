import argparse
import copy
import io
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import QED
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataset2 import QMugsDataset
from model3 import ATOMIC_NUM_TO_CLS, K_E, K_X, MAX_ATOMS, MoleculeGraphDiffusion


RDLogger.DisableLog("rdApp.*")

# 2026-08-03: 방향족 클래스 제거(kekulize로 명시적 단일/이중 교대 변환).
# devlog.md/research.md 2026-08-03 참조.
_BOND_TYPE = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
}
_CLS_TO_ANUM = {v: k for k, v in ATOMIC_NUM_TO_CLS.items()}

# [V-1] 2026-08-31: review2.md 지적 -- GetDefaultValence()는 원소당 "기본" 원자가
# 하나만 준다(S=2, P=3, I=1). RDKit이 실제로 sanitize에서 허용하는 목록은
# GetValenceList()이고 최댓값을 써야 함(S=[2,4,6], P=[3,5], I=[1,3,5]).
# 술폰/술폰아미드(S 원자가 6), 인산염(P 원자가 5)은 약물형 분자에서 흔함.
# 실측: 구버전 표로 실제 학습 데이터를 채점하면 0.69%가 오탐 위반으로 집계됨
# (S 원자의 41%, P 원자의 100%가 오탐). index 0(padding)=0.
_PT = Chem.GetPeriodicTable()
_MAX_VALENCE = torch.zeros(K_X)
for _anum, _cls in ATOMIC_NUM_TO_CLS.items():
    _MAX_VALENCE[_cls] = max(_PT.GetValenceList(_anum))


class _Tee(io.TextIOBase):

    def __init__(self, stream, file_path):
        self._stream = stream
        self._f = open(file_path, "a", encoding="utf-8")

    def write(self, text):
        self._stream.write(text)
        self._f.write(text)
        return len(text)

    def flush(self):
        self._stream.flush()
        self._f.flush()

    def close(self):
        self._f.close()


class EMA:

    def __init__(self, model, decay = 0.9999):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for ema_p, p in zip(self.model.parameters(), model.parameters()):
                ema_p.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)


def tensors_to_mol(X, E):
    mol = Chem.RWMol()
    # MASK 클래스 없음(marginal transition) -> 패딩(0)만 제외, 나머지 1..K_X-1 전부 실제 원자
    real = ((X > 0) & (X < K_X)).nonzero(as_tuple=True)[0].tolist()
    if len(real) < 2:
        return None

    idx_map = {}
    for ai in real:
        anum = _CLS_TO_ANUM.get(int(X[ai]))
        if anum is None:
            return None
        idx_map[ai] = mol.AddAtom(Chem.Atom(anum))

    for i, ai in enumerate(real):
        for aj in real[i + 1:]:
            bc = int(E[ai, aj])
            # 0(padding/no bond)만 제외 -- MASK 클래스 없음
            if bc == 0:
                continue
            btype = _BOND_TYPE.get(bc)
            if btype is None:
                continue
            mol.AddBond(idx_map[ai], idx_map[aj], btype)

    try:
        Chem.SanitizeMol(mol)

        # 꼼수 방지: 고립 원자들의 단순 나열로 100% validity를 교란하는 모델 방어 필터
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            max_frag_size = max(len(f.GetAtoms()) for f in frags)
            if max_frag_size / len(mol.GetAtoms()) < 0.85:
                return None

        return mol.GetMol()
    except Exception:
        return None


def atom_valence_violations(X, E):
    """[R-1] Tensor 기반 원자가 위반 계산 -- RDKit SanitizeMol에 의존하지 않음
    (sanitize는 첫 실패에서 예외를 던져 위반 개수를 셀 수 없음).
    E의 클래스값이 곧 결합 차수(1=단일,2=이중,3=삼중)이므로 행 합이 그 원자가
    실제로 쓰고 있는 총 원자가다.

    반환: (위반 원자 수, 전체(패딩 제외) 원자 수)
    """
    real = X > 0
    n_atoms = int(real.sum().item())
    if n_atoms == 0:
        return 0, 0
    degree = E.float().sum(dim=-1)
    max_val = _MAX_VALENCE[X.clamp(min=0, max=K_X - 1)]
    violations = int(((degree > max_val) & real).sum().item())
    return violations, n_atoms


def analyze_molecule(X, E):
    """[R-1, V-3] 분자 하나에 대해 strict/largest-fragment/single validity와
    fragment 관련 통계를 계산.

    - strict: 현행 기준 그대로 유지 (SanitizeMol 통과 + 최대 조각 비율 >= 0.85,
      고립 원자 나열로 validity를 교란하는 꼼수 방지 필터 포함).
    - largest-fragment: 최대 조각만 추출해 그것만 sanitize (MOSES/DiGress 등
      문헌이 쓰는 관대한 정의, 논문 대비 비교용 -- [R-9]).
    - single: [V-3] review2.md 지적 -- 학습 데이터는 fragment 개수가 100% 1개인데
      strict는 15%까지 조각남을 허용해 학습 데이터 분포보다 관대하다. sanitize
      통과 + fragment 개수 == 1만 요구하는 더 엄격한(데이터 분포 기준) 지표.
    """
    real = ((X > 0) & (X < K_X)).nonzero(as_tuple=True)[0].tolist()
    empty = {"strict_valid": False, "frag_valid": False, "single_valid": False,
             "frag_ratio": 0.0, "n_fragments": 0, "mol": None}
    if len(real) < 2:
        return empty

    mol = Chem.RWMol()
    idx_map = {}
    for ai in real:
        anum = _CLS_TO_ANUM.get(int(X[ai]))
        if anum is None:
            return empty
        idx_map[ai] = mol.AddAtom(Chem.Atom(anum))

    for i, ai in enumerate(real):
        for aj in real[i + 1:]:
            bc = int(E[ai, aj])
            if bc == 0:
                continue
            btype = _BOND_TYPE.get(bc)
            if btype is None:
                continue
            mol.AddBond(idx_map[ai], idx_map[aj], btype)

    raw_mol = mol.GetMol()

    try:
        frags = Chem.GetMolFrags(raw_mol, asMols=True, sanitizeFrags=False)
    except Exception:
        frags = ()
    if not frags:
        return empty

    n_fragments = len(frags)
    largest = max(frags, key=lambda f: f.GetNumAtoms())
    frag_ratio = largest.GetNumAtoms() / len(real)

    strict_valid = False
    strict_mol = None
    try:
        cand = Chem.Mol(raw_mol)
        Chem.SanitizeMol(cand)
        if frag_ratio >= 0.85:
            strict_valid = True
            strict_mol = cand
    except Exception:
        pass

    single_valid = strict_valid and n_fragments == 1

    frag_valid = False
    frag_mol = None
    try:
        cand = Chem.Mol(largest)
        Chem.SanitizeMol(cand)
        frag_valid = True
        frag_mol = cand
    except Exception:
        pass

    return {
        "strict_valid": strict_valid,
        "frag_valid": frag_valid,
        "single_valid": single_valid,
        "frag_ratio": frag_ratio,
        "n_fragments": n_fragments,
        "mol": strict_mol if strict_mol is not None else frag_mol,
    }


# [V-3] 2026-08-31: review2.md 권고 -- 학습 데이터 실측 참조 분포(3,000분자 중
# sanitize 성공 2,868개 기준, 로컬 직접 측정으로 review2.md 수치와 대조 확인됨).
# "그럴듯함"은 단일 점수가 아니라 이 분포와의 거리로 진단할 것 -- 최적화 목표가
# 아니라 진단 용도.
# [review9.md] 2026-09-05: 위 3,000분자(청크0) 기반 상수는 표본이 작고
# 청크마다 흔들려(2.97~3.25) review8.md/review9.md에서 부정확함이 지적됨.
# val split n=2048(형식전하 미모델링으로 인한 sanitize 실패 65건 제외
# n=1983)로 재확정한 값으로 교체. REF_SINGLE_FRAGMENT_RATE도 1.0000이
# 아니라 0.9683 -- 실패 원인은 전부 4차 암모늄/피리디늄류(N 원자가 4,
# formal charge 미모델링) 였으며 atomViolRate의 floor(review2.md V-1)와
# 동일 원인. docs/review_followup_20260905b.md §1 참조.
REF_SINGLE_FRAGMENT_RATE = 0.9683
REF_RING_COUNT_MEAN = 3.6884     # val n=2048, strict-valid n=1983 기준 분자당 전체 고리 개수
REF_RING_SIZE_56_RATE = 0.9668   # 같은 표본, Sigma56/SigmaTotal (56ring/mol 상한 = 3.5658)
REF_DEGREE_MEAN = 2.185          # 1:15.7% 2:51.8% 3:30.9% 4:1.6%
REF_QED_MEAN = 0.5377
REF_QED_MEDIAN = 0.5397


def mol_diagnostics(mol):
    """[V-3] 분자 하나의 "그럴듯함" 진단 통계. strict-valid 분자에 대해서만 호출."""
    ri = mol.GetRingInfo()
    ring_sizes = [len(r) for r in ri.AtomRings()]
    degrees = [a.GetDegree() for a in mol.GetAtoms()]
    try:
        qed = QED.qed(mol)
    except Exception:
        qed = None
    return {
        "n_rings": ri.NumRings(),
        "ring_sizes": ring_sizes,
        "degrees": degrees,
        "qed": qed,
    }


def build_size_distribution(dataset, n_sample = 2000):
    sizes = []
    ci = 0
    while len(sizes) < n_sample and ci < len(dataset._chunk_names):
        for data in dataset._load_chunk(ci):
            n = int((data.z != 1).sum())
            if 0 < n <= MAX_ATOMS:
                sizes.append(n)
            if len(sizes) >= n_sample:
                break
        ci += 1
    return sizes


def run_epoch(model, loader, device, train, epoch, optimizer=None, ema=None):
    model.train(train)
    totals = {
        "loss": 0.0,
        "loss_x": 0.0,
        "loss_e": 0.0
    }

    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
            batch = batch.to(device)
            result = model(batch)

            if train:
                optimizer.zero_grad()
                result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
                optimizer.step()
                if ema is not None:
                    ema.update(model)

            for k in totals:
                totals[k] += result[k].item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate_validity(model, size_dist, n_eval, device):
    """[R-1] validity는 median 29원자에서 v^29로 수렴하는 곱셈 지표라 이 성능
    구간에서 해상도가 사실상 0(0.80과 0.95의 validity가 둘 다 0.000으로 찍힘).
    validity(strict/frag)에 더해 원자 단위 연속 지표(위반율)를 주 지표로 함께
    반환한다 -- 관측 개수가 n_eval(분자) -> 원자 수(약 30배)로 늘어나 훨씬
    이른 시점에 개선 추세를 볼 수 있다.
    """
    import random
    from collections import Counter

    model.eval()
    sizes = random.choices(size_dist, k=n_eval)
    size_counts = Counter(sizes)

    n_strict = 0
    n_frag = 0
    n_single = 0
    strict_mols = []
    total_atoms = 0
    total_violations = 0
    violations_per_mol = []
    frag_ratios = []
    n_fragments_list = []

    for n_atoms, count in size_counts.items():
        if n_atoms > MAX_ATOMS:
            continue
        X, E = model.sample(count, n_atoms, device)
        X_cpu, E_cpu = X.cpu(), E.cpu()
        del X, E
        torch.cuda.empty_cache()
        for i in range(count):
            xi, ei = X_cpu[i], E_cpu[i]

            n_v, n_a = atom_valence_violations(xi, ei)
            total_violations += n_v
            total_atoms += n_a
            violations_per_mol.append(n_v)

            info = analyze_molecule(xi, ei)
            frag_ratios.append(info["frag_ratio"])
            n_fragments_list.append(info["n_fragments"])
            if info["strict_valid"]:
                n_strict += 1
                strict_mols.append(info["mol"])
            if info["frag_valid"]:
                n_frag += 1
            if info["single_valid"]:
                n_single += 1

    if strict_mols:
        smiles = {Chem.MolToSmiles(m) for m in strict_mols}
        uniqueness = len(smiles) / len(strict_mols)
    else:
        uniqueness = 0.0

    # [V-3] strict-valid 분자들의 "그럴듯함" 분포 진단. 참조값(REF_*)과 나란히
    # 볼 것 -- 이 지표들은 최적화 목표가 아니라 진단용.
    ring_counts, ring_sizes_flat, degrees_flat, qed_vals = [], [], [], []
    for m in strict_mols:
        d = mol_diagnostics(m)
        ring_counts.append(d["n_rings"])
        ring_sizes_flat.extend(d["ring_sizes"])
        degrees_flat.extend(d["degrees"])
        if d["qed"] is not None:
            qed_vals.append(d["qed"])

    n_single_frag = sum(1 for n in n_fragments_list if n == 1)
    ring56 = sum(1 for s in ring_sizes_flat if s in (5, 6))

    return {
        "validity_strict": n_strict / n_eval,
        "validity_frag": n_frag / n_eval,
        "validity_single": n_single / n_eval,
        "uniqueness": uniqueness,
        "atom_violation_rate": (total_violations / total_atoms) if total_atoms > 0 else float("nan"),
        "mean_violations_per_mol": (sum(violations_per_mol) / len(violations_per_mol)) if violations_per_mol else float("nan"),
        "mean_frag_ratio": (sum(frag_ratios) / len(frag_ratios)) if frag_ratios else float("nan"),
        "single_fragment_rate": (n_single_frag / len(n_fragments_list)) if n_fragments_list else float("nan"),
        "mean_ring_count": (sum(ring_counts) / len(ring_counts)) if ring_counts else float("nan"),
        "ring_size_56_rate": (ring56 / len(ring_sizes_flat)) if ring_sizes_flat else float("nan"),
        "mean_degree": (sum(degrees_flat) / len(degrees_flat)) if degrees_flat else float("nan"),
        "mean_qed": (sum(qed_vals) / len(qed_vals)) if qed_vals else float("nan"),
    }


@torch.no_grad()
def diagnostic_fixed_t(model, val_loader, device, t_values=(10, 40, 75, 110), n_batches=4):
    """[R-2, V-2] 고정 t에서 q_sample -> forward 한 번만으로 x0 재구성 정확도를
    측정한다 (150-step 전체 샘플링 불필요, forward pass 몇 번이라 비용 거의 0).

    [V-2] 2026-08-31: review2.md 지적 -- 코사인 스케줄에서 t=140은 ac(alpha_bar)
    ≈0.0108로 원본 정보가 1%만 남아, 이 지점의 Bayes-최적 예측은 marginal
    분포 그 자체(argmax가 항상 no-bond)다. 즉 t=140에서 bond_acc=0.000은
    "실패"가 아니라 모델이 정확히 옳게 행동한다는 신호일 수 있어 argmax 정확도만
    으로는 이 구간에서 아무것도 측정하지 못한다. t 격자를 정보가 남아있는 구간
    {10,40,75,110}(ac≈0.99/0.85/0.49/0.13)으로 바꾸고, 포화되지 않는 확률 기반
    지표(참-결합/무결합 위치의 평균 P(bond), edge CE vs H(m_E)=0.3076 기준선)를
    추가한다.
    """
    H_M_E = 0.3076  # 마진 분포 m_E의 엔트로피 -- 상수 예측기 CE 하한
    model.eval()
    stats = {t: {"atom_c": 0, "atom_n": 0, "bond_c": 0, "bond_n": 0, "nobond_c": 0, "nobond_n": 0,
                  "p_bond_true_bond_sum": 0.0, "p_bond_true_nobond_sum": 0.0,
                  "edge_ce_sum": 0.0, "edge_ce_n": 0}
              for t in t_values}

    it = iter(val_loader)
    for _ in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        batch = batch.to(device)
        X0, E0, node_mask = model._to_dense(batch)
        B, N = X0.shape
        e_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        triu = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        em = e_mask & triu.unsqueeze(0)

        for t_val in t_values:
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            X_t = model.schedule.q_sample(X0, t, model.schedule.m_X)
            E_t = model._sym(model.schedule.q_sample(E0, t, model.schedule.m_E))
            X_t = torch.where(node_mask, X_t, torch.zeros_like(X_t))
            E_t = torch.where(e_mask, E_t, torch.zeros_like(E_t))

            x_logits, e_logits = model.net(X_t, E_t, node_mask, t)
            x_pred = x_logits.argmax(-1) + 1
            e_pred = e_logits.argmax(-1)
            e_probs = e_logits.softmax(-1)
            p_bond = 1.0 - e_probs[..., 0]  # P(no-bond 아님) = P(어떤 결합이든 존재)

            s = stats[t_val]
            s["atom_c"] += int(((x_pred == X0) & node_mask).sum().item())
            s["atom_n"] += int(node_mask.sum().item())

            bond_mask = em & (E0 > 0)
            nobond_mask = em & (E0 == 0)
            s["bond_c"] += int(((e_pred == E0) & bond_mask).sum().item())
            s["bond_n"] += int(bond_mask.sum().item())
            s["nobond_c"] += int(((e_pred == E0) & nobond_mask).sum().item())
            s["nobond_n"] += int(nobond_mask.sum().item())

            # [V-2] 확률 기반 지표 -- argmax와 달리 높은 t에서도 포화되지 않음
            s["p_bond_true_bond_sum"] += float(p_bond[bond_mask].sum().item())
            s["p_bond_true_nobond_sum"] += float(p_bond[nobond_mask].sum().item())
            if em.sum().item() > 0:
                ce = F.cross_entropy(e_logits[em], E0[em], reduction="sum")
                s["edge_ce_sum"] += float(ce.item())
                s["edge_ce_n"] += int(em.sum().item())

    result = {}
    for t_val, s in stats.items():
        result[t_val] = {
            "atom_acc":   s["atom_c"] / s["atom_n"] if s["atom_n"] > 0 else float("nan"),
            "bond_acc":   s["bond_c"] / s["bond_n"] if s["bond_n"] > 0 else float("nan"),
            "nobond_acc": s["nobond_c"] / s["nobond_n"] if s["nobond_n"] > 0 else float("nan"),
            # [V-2] true-bond 위치와 true-nobond 위치의 평균 P(bond) 차이.
            # argmax가 전부 no-bond로 포화돼도(예: t=140) 이 차이는 0이 아닐 수
            # 있음 -- 모델이 맥락을 실제로 구분해서 쓰는지 보여주는 연속 지표.
            "p_bond_true_bond":   s["p_bond_true_bond_sum"] / s["bond_n"] if s["bond_n"] > 0 else float("nan"),
            "p_bond_true_nobond": s["p_bond_true_nobond_sum"] / s["nobond_n"] if s["nobond_n"] > 0 else float("nan"),
            # [V-2] H(m_E)=0.3076보다 낮으면 상수 예측기(항상 marginal 출력)보다
            # 낫다는 뜻 -- 그 t에서 모델이 그래프 맥락으로부터 정보를 실제로 쓰고 있음.
            "edge_ce":     s["edge_ce_sum"] / s["edge_ce_n"] if s["edge_ce_n"] > 0 else float("nan"),
            "edge_ce_ref": H_M_E,
        }
    return result


def _print_eval_block(tag, metrics, diag):
    print(
        f"    [{tag}] valid(strict)={metrics['validity_strict']:.3f} "
        f"valid(frag)={metrics['validity_frag']:.3f} "
        f"valid(single)={metrics['validity_single']:.3f} "
        f"atomViolRate={metrics['atom_violation_rate']:.3f} "
        f"meanViol/mol={metrics['mean_violations_per_mol']:.2f} "
        f"fragRatio={metrics['mean_frag_ratio']:.3f} "
        f"uniq={metrics['uniqueness']:.3f}"
    )
    # [V-3] 생성 분포 vs 학습 데이터 참조 분포(REF_*) 나란히 출력. 진단용이지
    # 최적화 목표 아님.
    print(
        f"    [{tag}] singleFragRate={metrics['single_fragment_rate']:.3f}"
        f"(ref {REF_SINGLE_FRAGMENT_RATE:.3f}) "
        f"ringCount={metrics['mean_ring_count']:.2f}(ref {REF_RING_COUNT_MEAN:.2f}) "
        f"ring56Rate={metrics['ring_size_56_rate']:.3f}(ref {REF_RING_SIZE_56_RATE:.3f}) "
        f"degree={metrics['mean_degree']:.2f}(ref {REF_DEGREE_MEAN:.2f}) "
        f"QED={metrics['mean_qed']:.3f}(ref {REF_QED_MEAN:.3f})"
    )
    parts = " | ".join(
        f"t={t_val:<3} atom={d['atom_acc']:.3f} bond={d['bond_acc']:.3f} nobond={d['nobond_acc']:.3f} "
        f"P(b|bond)={d['p_bond_true_bond']:.3f} P(b|nobond)={d['p_bond_true_nobond']:.3f} "
        f"CE={d['edge_ce']:.3f}(ref {d['edge_ce_ref']:.3f})"
        for t_val, d in diag.items()
    )
    print(f"    [{tag}] {parts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir",  default="./data/processed")
    parser.add_argument("--save-dir",       default="./checkpoints_diffusion_marginal")
    parser.add_argument("--epochs",         type=int,   default=200)
    parser.add_argument("--batch-size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=1e-4)
    # [R-3] 0.9999(시간상수 10,000 step)는 QMugs 전체(155만개) 기준으로 튜닝된
    # 값이라, 소규모/스모크 런에서는 총 step 수가 시간상수에 못 미쳐 학습이
    # 끝나도 EMA에 랜덤 초기값이 상당량 남는 문제가 세 번째로 재발했음
    # (2026-07-26, 07-28, 08-05). 0.999(시간상수 1,000 step)로 낮춰 기본값을
    # 더 안전한 쪽으로 바꾸되, 실행 규모에 맞게 직접 계산해서 넘길 것.
    parser.add_argument("--ema-decay",      type=float, default=0.999)
    parser.add_argument("--save-every",     type=int,   default=10)
    parser.add_argument("--eval-every",     type=int,   default=10,
                        help="Validity evaluation interval (epochs). 0 = disable.")
    parser.add_argument("--n-eval",         type=int,   default=256,
                        help="Number of molecules to sample per validity evaluation.")
    parser.add_argument("--num-workers",    type=int,   default=0)
    parser.add_argument("--max-samples",    type=int,   default=None)
    parser.add_argument("--log-file",       default="./train_diffusion_marginal.log")
    parser.add_argument("--resume",         default=None,
                        help="체크포인트 경로 (.pt). model/ema/optimizer/scheduler/epoch 복원.")
    parser.add_argument("--init-weights",   default=None,
                        help="[G-2 파인튜닝] 순수 state_dict(.pt, best.pt 형식)만 로드 -- "
                             "optimizer/scheduler/epoch는 새로 시작. --resume과 동시 사용 금지.")
    parser.add_argument("--train-only",     default=None,
                        help="[G-5] 콤마로 구분된 부분문자열. 파라미터 이름에 하나라도 "
                             "포함되면 학습, 아니면 동결(requires_grad=False). "
                             "예: 'cond_mod' -> AdaLN 재주입 파라미터만 학습.")
    # [R-6] review7.md [Z-4] matched-budget A/B -- 모델 가중치 초기화/학습 중
    # 확률적 요소(q_sample 등)를 고정해 재현 가능하게 함.
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tee = _Tee(sys.stdout, args.log_file)
    sys.stdout = tee

    print(f"Device: {device}")
    print(f"Log: {args.log_file}")

    print("Loading datasets ...")
    train_set = QMugsDataset(
        args.processed_dir, split="train",
        max_samples=args.max_samples
    )
    val_set = QMugsDataset(
        args.processed_dir, split="val",
        max_samples=args.max_samples // 5 if args.max_samples else None
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    print(f"  Train: {len(train_set):,}  Val: {len(val_set):,}")

    print("Building size distribution ...")
    size_dist = build_size_distribution(train_set)
    print(f"  Size range: {min(size_dist)} ~ {max(size_dist)} atoms  "
          f"(median {sorted(size_dist)[len(size_dist)//2]})")

    # marginal transition noise용 클래스 분포 계산 (DiGress Sec 4.1).
    # inverse-frequency loss reweighting은 2026-07-26 ablation에서 효과 없음이
    # 확인됐고 논문에도 없는 기법이라 제거 -- 희소성 보정은 노이즈 자체(marginal
    # 분포로 수렴)가 담당하도록 함.
    print("Computing marginal class distributions (node/edge) ...")
    x_counts = torch.zeros(K_X, dtype=torch.float)
    n_weight_samples = min(5000, len(train_set))
    for i in range(n_weight_samples):
        data = train_set[i]
        for anum in data.z:
            cls_idx = ATOMIC_NUM_TO_CLS.get(int(anum), 0)
            x_counts[cls_idx] += 1
    x_counts[0] = 0.0  # padding은 marginal에 포함하지 않음
    m_X = x_counts / x_counts.sum().clamp(min=1.0)
    print(f"  m_X: {m_X.tolist()}")

    e_counts = torch.zeros(K_E, dtype=torch.float)
    n_edge_samples = min(1000, len(train_set))
    for i in range(n_edge_samples):
        data = train_set[i]
        btype = data.bond_type.argmax(dim=-1) + 1
        for bt in btype:
            e_counts[int(bt)] += 1
        n_atoms = len(data.z)
        total_possible = n_atoms * (n_atoms - 1)
        no_bond = total_possible - len(btype)
        e_counts[0] += no_bond
    m_E = e_counts / e_counts.sum().clamp(min=1.0)
    print(f"  m_E: {m_E.tolist()}")

    model = MoleculeGraphDiffusion(m_X, m_E).to(device)
    ema = EMA(model, decay=args.ema_decay)
    ema.model.to(device)

    best_val_loss = float('inf')
    start_epoch = 1

    if args.init_weights:
        assert not args.resume, "--init-weights와 --resume은 동시 사용 금지"
        sd = torch.load(args.init_weights, map_location=device)
        # [G-5] strict=False: cond_mod(AdaLN 재주입)처럼 새 아키텍처에서
        # 추가된 파라미터는 옛 체크포인트에 없음 -- zero-init 상태로 남겨둠
        # (GTLayer.__init__에서 이미 0으로 초기화됨, 항등변환이라 안전).
        missing, unexpected = model.load_state_dict(sd, strict=False)
        ema.model.load_state_dict(sd, strict=False)
        print(f"  [G-2] {args.init_weights}에서 가중치만 로드 (optimizer/scheduler/epoch는 새로 시작)")
        if missing:
            print(f"  [G-5] 새 아키텍처 파라미터(zero-init 유지): {missing}")
        if unexpected:
            print(f"  [경고] 체크포인트에만 있고 현재 모델엔 없는 키: {unexpected}")

    # [G-5] --train-only: 이름에 지정된 부분문자열이 포함된 파라미터만 학습.
    # zero-init으로 새로 추가된 conditioning 경로(cond_mod)만 움직이고
    # 이미 검증된 backbone은 그대로 둬서, 학습이 잘못될 때 backbone까지
    # 같이 흔들리는 위험을 없앤다.
    if args.train_only:
        substrs = args.train_only.split(",")
        n_trainable = 0
        for name, p in model.named_parameters():
            p.requires_grad = any(s in name for s in substrs)
            if p.requires_grad:
                n_trainable += p.numel()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"  [G-5] --train-only={args.train_only} -- 학습 파라미터 "
              f"{n_trainable:,} / 전체 {sum(p.numel() for p in model.parameters()):,}")
    else:
        trainable_params = model.parameters()

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float('inf'))
        print(f"  Resumed from epoch {ckpt['epoch']}  (best val loss: {best_val_loss:.4f})")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    print(f"\n{'Epoch':>6}  {'T-loss':>8} {'T-x':>8} {'T-e':>8}  "
          f"{'V-loss':>8} {'V-x':>8} {'V-e':>8}  Time")
    print("-" * 90)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_set.reshuffle_indices()

        train = run_epoch(model, train_loader, device, train=True, epoch=epoch, optimizer=optimizer, ema=ema)
        val = run_epoch(ema.model, val_loader, device, train=False, epoch=epoch)

        scheduler.step()

        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            torch.save(ema.model.state_dict(), save_dir / "best.pt")

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss
            }, save_dir / f"ckpt_epoch{epoch:04d}.pt")

        elapsed = time.time() - t0
        print(
            f"{epoch:>6}  "
            f"{train['loss']:>8.4f} {train['loss_x']:>8.4f} {train['loss_e']:>8.4f}  "
            f"{val['loss']:>8.4f} {val['loss_x']:>8.4f} {val['loss_e']:>8.4f}  "
            f"{elapsed:>6.1f}s"
        )

        # [R-3] raw model 지표를 EMA와 나란히 로깅 -- 2026-07-26에 이 비교가
        # EMA 오염 가설을 기각하는 데 결정적이었던 전례를 재사용.
        if args.eval_every > 0 and epoch % args.eval_every == 0:
            ema_metrics = evaluate_validity(ema.model, size_dist, args.n_eval, device)
            ema_diag = diagnostic_fixed_t(ema.model, val_loader, device)
            _print_eval_block("EMA", ema_metrics, ema_diag)

            raw_metrics = evaluate_validity(model, size_dist, args.n_eval, device)
            raw_diag = diagnostic_fixed_t(model, val_loader, device)
            _print_eval_block("RAW", raw_metrics, raw_diag)

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    sys.stdout = tee._stream
    tee.close()


if __name__ == "__main__":
    main()
