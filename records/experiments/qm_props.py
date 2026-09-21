"""GNN 대리모델 물성 추정 + 요청 시 PySCF DFT 단일점 계산 + 전자밀도 시각화."""
import sys, os, time, io, base64
sys.path.insert(0, "/home/cbgpu/molecule-AI")

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False

from c1_prop_regressor import PropRegressor

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Cancelled(Exception):
    """progress_cb가 True를 반환하면(작업 취소 요청) 여기서 던져 SCF/그리드 계산
    루프 중간에 빠져나온다. progress_cb 자체가 실수로 예외를 던지는 경우와
    구분하기 위해 별도 return-값 컨벤션을 씀(버그 있는 콜백이 계산을 죽이지
    않도록 그 경우는 그냥 무시)."""
    pass

_reg_ckpt = torch.load("c1_regressor.pt", map_location=_DEVICE, weights_only=False)
_reg = PropRegressor().to(_DEVICE)
_reg.load_state_dict(_reg_ckpt["model"])
_reg.eval()
_VAL_MEAN = _reg_ckpt["val_mean"].to(_DEVICE)
_VAL_STD = _reg_ckpt["val_std"].to(_DEVICE)

_BT_IDX = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2,
           Chem.BondType.AROMATIC: 1}


def _mol_to_pyg(mol):
    z = [a.GetAtomicNum() for a in mol.GetAtoms()]
    if len(z) < 2:
        return None
    src, dst, bt = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        oh = [0.0, 0.0, 0.0]
        oh[_BT_IDX.get(b.GetBondType(), 0)] = 1.0
        src += [i, j]; dst += [j, i]; bt += [oh, oh]
    if not src:
        return None
    return Batch.from_data_list([Data(
        z=torch.tensor(z, dtype=torch.long),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        bond_type=torch.tensor(bt, dtype=torch.float32),
        num_nodes=len(z),
    )]).to(_DEVICE)


@torch.no_grad()
def gnn_homo_lumo(mol):
    """RDKit mol -> GNN 추정 (HOMO, LUMO) in Hartree. 실패 시 None."""
    batch = _mol_to_pyg(mol)
    if batch is None:
        return None
    pred = _reg(batch) * _VAL_STD.unsqueeze(0) + _VAL_MEAN.unsqueeze(0)
    return float(pred[0, 0]), float(pred[0, 1])


# ---------------------------------------------------------------- PySCF DFT (요청 시에만)
_LEVELS = {
    # QMugs 라벨은 wB97X-D/def2-SVP. 정확히 맞추면 느림(30원자 수분~).
    "qmugs": dict(xc="wb97xd", basis="def2-svp"),
    "fast":  dict(xc="b3lyp",   basis="6-31g*"),
    "veryfast": dict(xc="pbe",  basis="sto-3g"),
}


def _embed_3d(smiles, progress_cb=None):
    """SMILES -> RDKit ETKDGv3 임베딩 + MMFF 최적화. 반환: (mol_noH, atom_lines, charge) 또는 dict(error=...)."""
    def _p(**kw):
        if progress_cb:
            try:
                cancel = progress_cb(kw)
            except Exception:
                return
            if cancel:
                raise Cancelled("사용자 요청으로 취소됨")
    _p(stage="embed", msg="3D 구조 생성 중")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": "SMILES 파싱 실패"}
    m3 = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0
    if AllChem.EmbedMolecule(m3, params) < 0:
        if AllChem.EmbedMolecule(m3, useRandomCoords=True, randomSeed=1) < 0:
            return {"error": "3D 임베딩 실패 (DFT 불가)"}
    try:
        AllChem.MMFFOptimizeMolecule(m3, maxIters=500)
    except Exception:
        pass

    conf = m3.GetConformer()
    atom_lines = []
    coords = []
    for atom in m3.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        atom_lines.append(f"{atom.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}")
        coords.append((atom.GetSymbol(), p.x, p.y, p.z))
    charge = Chem.GetFormalCharge(m3)
    mol_block = Chem.MolToMolBlock(m3)
    return {"mol": mol, "atom_lines": atom_lines, "charge": charge, "coords": coords, "mol_block": mol_block}


def _run_scf(atom_lines, charge, level, t0, progress_cb=None):
    """공통 SCF 실행. 반환: (mf, cfg) 또는 dict(error=...)."""
    def _p(**kw):
        if progress_cb:
            try:
                cancel = progress_cb(kw)
            except Exception:
                return
            if cancel:
                raise Cancelled("사용자 요청으로 취소됨")
    cfg = _LEVELS.get(level, _LEVELS["fast"])
    _p(stage="setup", msg=f"{cfg['xc']}/{cfg['basis']} 세팅 (원자 {len(atom_lines)}개)",
       n_atoms=len(atom_lines))
    try:
        from pyscf import gto, dft
        lib_mol = gto.M(atom="\n".join(atom_lines), basis=cfg["basis"],
                        charge=charge, spin=0, verbose=0)
        mf = dft.RKS(lib_mol)
        mf.xc = cfg["xc"]
        mf.max_cycle = 100

        _cyc = {"n": 0}
        def _scf_cb(envs):
            _cyc["n"] += 1
            e = envs.get("e_tot")
            _p(stage="scf", cycle=_cyc["n"],
               energy=float(e) if e is not None else None,
               elapsed_s=round(time.time() - t0, 1))
        mf.callback = _scf_cb
        mf.kernel()
        return mf, cfg
    except Cancelled:
        raise
    except Exception as e:
        return {"error": f"DFT 계산 실패: {e}"}


def dft_homo_lumo(smiles, level="fast", n_threads=None, want_density=False, progress_cb=None):
    """SMILES -> 3D 임베딩 -> PySCF DFT 단일점 -> (HOMO, LUMO) in Hartree.
    want_density=True면 같은 SCF 수렴 결과 위에서 전자밀도(cube 파일 텍스트 +
    2D 슬라이스 이미지)까지 한 번에 계산한다(중복 SCF 방지 -- DFT와 전자구름을
    따로 요청하면 SCF가 두 번 돌아 낭비였던 문제 수정, 2026-09-14).
    progress_cb(dict) 가 주어지면 단계마다 호출된다(임베딩/SCF 사이클/전자밀도/완료).
    반환: dict(homo, lumo, gap, level, xc, basis, n_heavy, elapsed_s, mol_block,
    [density_img, cube_text]) 또는 dict(error=...)."""
    t0 = time.time()
    def _p(**kw):
        if progress_cb:
            try:
                cancel = progress_cb(kw)
            except Exception:
                return
            if cancel:
                raise Cancelled("사용자 요청으로 취소됨")
    if n_threads is None:
        n_threads = min(40, os.cpu_count() or 8)
    os.environ["OMP_NUM_THREADS"] = str(n_threads)

    embed = _embed_3d(smiles, progress_cb)
    if "error" in embed:
        return embed

    result = _run_scf(embed["atom_lines"], embed["charge"], level, t0, progress_cb)
    if isinstance(result, dict) and "error" in result:
        return result
    mf, cfg = result

    mo_e = mf.mo_energy
    occ = mf.mo_occ
    homo = float(max(mo_e[occ > 0]))
    lumo = float(min(mo_e[occ == 0]))

    out = {
        "homo": homo, "lumo": lumo, "gap": lumo - homo,
        "level": level, "xc": cfg["xc"], "basis": cfg["basis"],
        "n_heavy": embed["mol"].GetNumHeavyAtoms(),
        "mol_block": embed["mol_block"],
        "elapsed_s": round(time.time() - t0, 1),
    }
    if not want_density:
        return out

    _p(stage="density", msg="전자밀도 그리드 계산 중")
    try:
        from pyscf.tools import cubegen
        from pyscf.dft import numint
        import tempfile

        lib_mol = mf.mol
        dm = mf.make_rdm1()

        cc = cubegen.Cube(lib_mol, nx=45, ny=45, nz=45, margin=3.0)
        coords = cc.get_coords()
        ao = numint.eval_ao(lib_mol, coords)
        rho = numint.eval_rho(lib_mol, ao, dm)
        rho = rho.reshape(cc.nx, cc.ny, cc.nz)

        # 3Dmol.js에서 인터랙티브 isosurface로 회전해볼 수 있도록 표준 cube 텍스트도 생성
        with tempfile.NamedTemporaryFile(suffix=".cube", delete=False) as tf:
            cube_path = tf.name
        cc.write(rho, cube_path, comment="Electron density in real space (e/Bohr^3)")
        with open(cube_path) as f:
            cube_text = f.read()
        os.unlink(cube_path)

        # cc.xs/ys/zs는 0~1 fractional 좌표 -> box(diag=extent) 곱해 실공간(Bohr)으로 환산
        xs = cc.xs * cc.box[0, 0] + cc.boxorig[0]
        ys = cc.ys * cc.box[1, 1] + cc.boxorig[1]
        zs = cc.zs * cc.box[2, 2] + cc.boxorig[2]

        # 원자들의 z좌표 평균에 가장 가까운 그리드 슬라이스(분자 "평면" 근사, 2D 썸네일용)
        z_coords = np.array([c[3] for c in embed["coords"]])
        z_mean = float(z_coords.mean())
        z_idx = int(np.argmin(np.abs(zs - z_mean)))
        slice_rho = rho[:, :, z_idx]

        fig, ax = plt.subplots(figsize=(5, 5))
        levels = np.logspace(-3, 0.5, 25)
        cf = ax.contourf(xs, ys, np.clip(slice_rho.T, 1e-4, None), levels=levels,
                          norm=matplotlib.colors.LogNorm(), cmap="inferno")
        ax_x = [c[1] for c in embed["coords"] if abs(c[3] - z_mean) < 1.5]
        ax_y = [c[2] for c in embed["coords"] if abs(c[3] - z_mean) < 1.5]
        ax.scatter(ax_x, ax_y, s=18, c="cyan", edgecolors="white", linewidths=0.5, zorder=5)
        ax.set_xlabel("x (Bohr)")
        ax.set_ylabel("y (Bohr)")
        ax.set_title(f"전자밀도 슬라이스 (z≈{z_mean:.1f} Bohr 부근)")
        ax.set_aspect("equal")
        fig.colorbar(cf, ax=ax, label="ρ(r) (log scale)")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130)
        plt.close(fig)
        out["density_img"] = base64.b64encode(buf.getvalue()).decode()
        out["cube_text"] = cube_text
    except Cancelled:
        raise
    except Exception as e:
        out["density_error"] = f"전자밀도 계산 실패: {e}"
        return out

    _p(stage="esp", msg="정전기 퍼텐셜(ESP) 계산 중")
    try:
        from pyscf.tools import cubegen as _cg2
        from pyscf import gto as _gto2, df as _df2
        import tempfile as _tempfile2

        # 같은 grid(cc, coords) 재사용 -- MEP = 핵 쿨롱 기여 - 전자밀도 쿨롱 기여
        Vnuc = 0
        for i in range(lib_mol.natm):
            r = lib_mol.atom_coord(i)
            Z = lib_mol.atom_charge(i)
            rp = r - coords
            Vnuc += Z / np.einsum('xi,xi->x', rp, rp) ** 0.5
        Vele = np.empty_like(Vnuc)
        from pyscf import lib as _lib2
        for p0, p1 in _lib2.prange(0, Vele.size, 600):
            fakemol = _gto2.fakemol_for_charges(coords[p0:p1])
            ints = _df2.incore.aux_e2(lib_mol, fakemol)
            Vele[p0:p1] = np.einsum('ijp,ij->p', ints, dm)
        esp = (Vnuc - Vele).reshape(cc.nx, cc.ny, cc.nz)

        with _tempfile2.NamedTemporaryFile(suffix=".cube", delete=False) as tf:
            esp_path = tf.name
        cc.write(esp, esp_path, comment="Molecular electrostatic potential in real space (Hartree/e)")
        with open(esp_path) as f:
            out["esp_cube_text"] = f.read()
        os.unlink(esp_path)

        # 색 범위는 밀도 등고선(isoval=0.02) 근처에서만 계산 -- 핵 바로 옆
        # 특이점(값이 수백까지 치솟음)이나 먼 진공(값이 0에 가까움) 영역까지
        # 넣으면 실제 표면에 안 보이는 값들 때문에 범위가 왜곡됨.
        near_surface = esp[(rho > 0.01) & (rho < 0.04)]
        if near_surface.size > 20:
            lo, hi = float(np.percentile(near_surface, 3)), float(np.percentile(near_surface, 97))
        else:
            lo, hi = -0.05, 0.05
        if hi - lo < 1e-4:
            lo, hi = lo - 0.02, hi + 0.02
        out["esp_range"] = [lo, hi]
    except Cancelled:
        raise
    except Exception as e:
        out["esp_error"] = f"ESP 계산 실패: {e}"

    return out
