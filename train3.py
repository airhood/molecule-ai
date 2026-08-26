import argparse
import copy
import io
import sys
import time
from pathlib import Path

import torch
from rdkit import Chem, RDLogger
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

# [R-1] 원자가 위반 판정용 원소별 표준 원자가 (RDKit 실측: C4/N3/O2/S2/P3/할로겐1).
# index 0(padding)=0.
_PT = Chem.GetPeriodicTable()
_MAX_VALENCE = torch.zeros(K_X)
for _anum, _cls in ATOMIC_NUM_TO_CLS.items():
    _MAX_VALENCE[_cls] = _PT.GetDefaultValence(_anum)


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
    """[R-1] 분자 하나에 대해 strict/largest-fragment validity와 fragment 비율을 계산.

    - strict: 현행 기준 그대로 유지 (SanitizeMol 통과 + 최대 조각 비율 >= 0.85,
      고립 원자 나열로 validity를 교란하는 꼼수 방지 필터 포함).
    - largest-fragment: 최대 조각만 추출해 그것만 sanitize (MOSES/DiGress 등
      문헌이 쓰는 관대한 정의, 논문 대비 비교용 -- [R-9]).
    """
    real = ((X > 0) & (X < K_X)).nonzero(as_tuple=True)[0].tolist()
    if len(real) < 2:
        return {"strict_valid": False, "frag_valid": False, "frag_ratio": 0.0, "mol": None}

    mol = Chem.RWMol()
    idx_map = {}
    for ai in real:
        anum = _CLS_TO_ANUM.get(int(X[ai]))
        if anum is None:
            return {"strict_valid": False, "frag_valid": False, "frag_ratio": 0.0, "mol": None}
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
        return {"strict_valid": False, "frag_valid": False, "frag_ratio": 0.0, "mol": None}

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
        "frag_ratio": frag_ratio,
        "mol": strict_mol if strict_mol is not None else frag_mol,
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
    strict_mols = []
    total_atoms = 0
    total_violations = 0
    violations_per_mol = []
    frag_ratios = []

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
            if info["strict_valid"]:
                n_strict += 1
                strict_mols.append(info["mol"])
            if info["frag_valid"]:
                n_frag += 1

    if strict_mols:
        smiles = {Chem.MolToSmiles(m) for m in strict_mols}
        uniqueness = len(smiles) / len(strict_mols)
    else:
        uniqueness = 0.0

    return {
        "validity_strict": n_strict / n_eval,
        "validity_frag": n_frag / n_eval,
        "uniqueness": uniqueness,
        "atom_violation_rate": (total_violations / total_atoms) if total_atoms > 0 else float("nan"),
        "mean_violations_per_mol": (sum(violations_per_mol) / len(violations_per_mol)) if violations_per_mol else float("nan"),
        "mean_frag_ratio": (sum(frag_ratios) / len(frag_ratios)) if frag_ratios else float("nan"),
    }


@torch.no_grad()
def diagnostic_fixed_t(model, val_loader, device, t_values=(10, 75, 140), n_batches=4):
    """[R-2] 고정 t에서 q_sample -> forward 한 번만으로 x0 재구성 정확도를
    측정한다 (150-step 전체 샘플링 불필요, forward pass 몇 번이라 비용 거의 0).
    원자/결합/무결합 정확도를 분리해서, "맥락이 거의 없을 때(t 큼) 결합 쪽으로
    쏠리는 편향"이 marginal transition에서도 남아있는지 확인한다
    (devlog 2026-07-26 absorbing 진단 방법론을 marginal에 재적용).
    """
    model.eval()
    stats = {t: {"atom_c": 0, "atom_n": 0, "bond_c": 0, "bond_n": 0, "nobond_c": 0, "nobond_n": 0}
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

            s = stats[t_val]
            s["atom_c"] += int(((x_pred == X0) & node_mask).sum().item())
            s["atom_n"] += int(node_mask.sum().item())

            bond_mask = em & (E0 > 0)
            nobond_mask = em & (E0 == 0)
            s["bond_c"] += int(((e_pred == E0) & bond_mask).sum().item())
            s["bond_n"] += int(bond_mask.sum().item())
            s["nobond_c"] += int(((e_pred == E0) & nobond_mask).sum().item())
            s["nobond_n"] += int(nobond_mask.sum().item())

    result = {}
    for t_val, s in stats.items():
        result[t_val] = {
            "atom_acc":   s["atom_c"] / s["atom_n"] if s["atom_n"] > 0 else float("nan"),
            "bond_acc":   s["bond_c"] / s["bond_n"] if s["bond_n"] > 0 else float("nan"),
            "nobond_acc": s["nobond_c"] / s["nobond_n"] if s["nobond_n"] > 0 else float("nan"),
        }
    return result


def _print_eval_block(tag, metrics, diag):
    print(
        f"    [{tag}] valid(strict)={metrics['validity_strict']:.3f} "
        f"valid(frag)={metrics['validity_frag']:.3f} "
        f"atomViolRate={metrics['atom_violation_rate']:.3f} "
        f"meanViol/mol={metrics['mean_violations_per_mol']:.2f} "
        f"fragRatio={metrics['mean_frag_ratio']:.3f} "
        f"uniq={metrics['uniqueness']:.3f}"
    )
    parts = " | ".join(
        f"t={t_val:<3} atom={d['atom_acc']:.3f} bond={d['bond_acc']:.3f} nobond={d['nobond_acc']:.3f}"
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
    args = parser.parse_args()

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

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    start_epoch = 1

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
