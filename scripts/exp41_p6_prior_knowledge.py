"""exp41 —— 改进「化学先验的知识本身」，而非加模块。

=============================================================================
诊断依据（前序实验已确立，本实验不改变这些事实）
=============================================================================
1. exp34/exp37：化学空间与谱空间**结构不相容**——linear CKA = 0.23，
   类间余弦 Spearman ρ = +0.14 (p=0.18)。
2. exp31/exp35：化学轴**有效但弱**——错误落化学近邻（高于随机），
   k=0 时 CLM 冻结码本 0.444 vs Rand 0.000。
3. exp38：k=0 时 Center 0.496 > Raw 0.446 > ABTT2 0.224 > Whiten 0.071；
   但 Center − Raw = +0.050 (p=0.30) —— **对齐层面校正的收益已达上限**。

=============================================================================
本实验的核心假设（可证伪）
=============================================================================
CLM 是「分子级语义嵌入」：为**分子性质预测**预训练，捕获的是
"哪些分子的整体性质相似"。
ECFP 是「子结构指纹」：捕获的是拓扑片段的共现，但**不含振动信息**。

而 FTIR 吸收峰 = 分子振动模式 = **官能团**的物理响应。
在这个数据集的窗口（1005–1799 cm⁻¹，指纹区 + 双键区）内，
有区分力的全部是官能团特征峰：酯 C=O 1743、羧酸 C=O 1710、
酰胺 I 1650、酰胺 II 1550、芳香/烯烃 1600、糖 C-O 1040、酯 C-O 1160……

⇒ **H41：把化学先验从「分子语义」换成「官能团物理」，
   可提升化学空间与谱空间的结构一致性，并提升零样本迁移。**

本实验提出两级改进（由弱到强地注入物理知识）：
  P-FG ：官能团计数向量（SMARTS 计数，按组分重量加权）
  P-PS ：**伪谱（pseudo-spectrum）**——官能团 → 特征峰（波数/强度/宽度）
         高斯展宽 → 240 点谱，与实测谱**同构**。
         若成立，则「跨模态对齐」问题退化为「同空间匹配」，
         CKA=0.23 的结构不相容被**绕开**而不是被拟合。

=============================================================================
Stage A（本脚本，纯 CPU，无需训练）—— 事前诊断
=============================================================================
不训练任何模型，直接回答三个可证伪的问题：
  A1 结构一致性：P-PS 的类间几何 vs 实测谱类间几何，Spearman / CKA
     是否显著高于 CLM / ECFP？（预注册：ρ_PS > ρ_CLM 且 ρ_PS > 0.3）
  A2 语义合理性：官能团计数是否符合化学直觉（油脂→酯、糖浆→C-O、
     肉类→酰胺）？（定性，可人工核对）
  A3 免训练零样本检索：把 P-PS 直接当检索库，实测谱做最近邻，
     top-1 是否显著高于 chance(1/19)？（预注册：> 2×chance）

A3 是**最强的事前证据**：它完全不依赖编码器训练，
若 P-PS 能免训练检索成功，说明先验本身携带了正确的谱结构。

=============================================================================
预注册判据（写死在脚本里，跑完不再改）
=============================================================================
- H41a：ρ(P-PS 余弦, 实测谱余弦) > ρ(CLM) 且 > ρ(ECFP)     【主】
- H41b：P-PS 免训练 top-1 ≥ 2 × chance = 0.105              【主】
- H41c：CKA(P-PS, 谱) > CKA(CLM, 谱)                        【次】
- H41d：P-PS 的优势在「高浓度子集」上更强（低浓度被基质淹没）  【次】
若 H41a 与 H41b 均不成立 ⇒ 放弃「改先验知识」这条线，
claim 退回 exp35/exp38/exp40 的样本量切换策略。
"""
from __future__ import annotations
import os, sys, json, io, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1
import exp34_cache_reprs as C

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

OUT = r"D:/2026sci论文实验设计/论文6/out"

# -----------------------------------------------------------------------------
# 官能团定义：只保留在 1005–1799 cm⁻¹ 窗口内有特征吸收的基团
#   (name, SMARTS, [(波数, 相对强度, FWHM), ...])
# 强度为相对摩尔吸光系数的粗略分级：C=O 最强，C=C 中等，C-O 强
# -----------------------------------------------------------------------------
FG = [
    ("ester",       "[CX3](=O)[OX2H0][#6]",            [(1743, 1.00, 22), (1165, 0.85, 30)]),
    ("carboxyl",    "[CX3](=O)[OX2H1]",                [(1710, 0.95, 25), (1290, 0.45, 40)]),
    ("carboxylate", "[CX3](=O)[OX1-]",                 [(1550, 0.80, 40), (1405, 0.70, 35)]),
    ("amide",       "[CX3](=O)[NX3]",                  [(1650, 1.00, 30), (1550, 0.60, 30), (1240, 0.30, 45)]),
    ("ketone",      "[#6][CX3](=O)[#6]",               [(1705, 0.75, 22)]),
    ("aldehyde",    "[CX3H1](=O)[#6]",                 [(1725, 0.70, 22)]),
    ("alkene",      "[CX3]=[CX3]",                     [(1655, 0.30, 25)]),
    ("aromatic",    "c1ccccc1",                        [(1605, 0.50, 20), (1515, 0.40, 20)]),
    ("amine",       "[NX3;H2,H1,H0;!$(N[C,S]=[O,S]);!$(N#*)]", [(1620, 0.30, 35)]),
    ("alcohol",     "[CX4;!$(C[OX2][CX3]=[OX1])][OX2H]", [(1055, 0.80, 45)]),
    ("ether",       "[CX4][OX2][CX4,c]",               [(1100, 0.65, 45)]),
    ("nitro",       "[NX3](=O)=O",                     [(1525, 0.90, 25), (1350, 0.80, 25)]),
    ("phosphate",   "[PX4](=O)",                       [(1250, 0.85, 40)]),
    ("sulfate",     "[SX4](=O)(=O)",                   [(1220, 0.85, 40)]),
    ("alkyl",       "[CX4;H1,H2,H3]",                  [(1455, 0.45, 25), (1375, 0.30, 20)]),
]
FG_NAMES = [f[0] for f in FG]
FG_PATS = [(nm, Chem.MolFromSmarts(s), pk) for nm, s, pk in FG]
assert all(p is not None for _, p, _ in FG_PATS), "SMARTS 解析失败"


def fg_counts(smi: str) -> np.ndarray:
    """单个 SMILES 的官能团计数向量 (nFG,)。"""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return np.zeros(len(FG))
    return np.array([len(m.GetSubstructMatches(p)) for _, p, _ in FG_PATS], float)


def pseudo_spectrum(fgc: np.ndarray, wn: np.ndarray) -> np.ndarray:
    """官能团计数 → 伪谱（高斯峰叠加，与实测谱同构）。"""
    ps = np.zeros_like(wn)
    for c, (_, _, peaks) in zip(fgc, FG_PATS):
        if c <= 0:
            continue
        for (w0, inten, fwhm) in peaks:
            sigma = fwhm / 2.3548          # FWHM → σ
            ps += c * inten * np.exp(-0.5 * ((wn - w0) / sigma) ** 2)
    return ps


# -----------------------------------------------------------------------------
# 谱预处理（与检索诊断配套）
# -----------------------------------------------------------------------------
def prep(X: np.ndarray, how: str) -> np.ndarray:
    if how == "raw":
        return X
    if how == "snv":                       # 去均值 + 除标准差（消除浓度尺度）
        return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)
    if how == "msc":                       # 乘性散射校正（对基线稳健）
        ref = X.mean(0)
        out = np.empty_like(X)
        for i in range(len(X)):
            b = np.polyfit(ref, X[i], 1)
            out[i] = (X[i] - b[1]) / (b[0] + 1e-9)
        return out
    if how == "deriv1":                    # 一阶差分（去基线）
        return np.diff(X, axis=1)
    if how == "snv+deriv1":
        return np.diff(prep(X, "snv"), axis=1)
    raise ValueError(how)


def cos_matrix(M: np.ndarray) -> np.ndarray:
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return Mn @ Mn.T


def offdiag(C: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(len(C), 1)
    return C[iu]


def cka(A: np.ndarray, B: np.ndarray) -> float:
    A = A - A.mean(0, keepdims=True)
    B = B - B.mean(0, keepdims=True)
    num = np.linalg.norm(A.T @ B) ** 2
    den = np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B) + 1e-12
    return float(num / den)


def main():
    ent, bg, _ = G1.load_g1()
    eids = sorted(ent.keys())
    mids = sorted(bg.keys())
    keys = eids + [f"M_{m}" for m in mids]
    n = G1.GRID_N
    wn = np.linspace(G1.GRID_LO, G1.GRID_HI, n)

    # ---- 实测谱：类平均（原始吸光度）与高浓度子集 ----
    Xmean = np.stack([np.asarray(ent[e], float).mean(0) if e in ent
                      else np.asarray(bg[e[2:]], float).mean(0) for e in keys])

    # 高浓度子集：掺假实体取谱范数最大的 30%（掺杂物信号最强）
    Xhi = []
    for e in keys:
        A = np.asarray(ent[e], float) if e in ent else np.asarray(bg[e[2:]], float)
        if len(A) < 10:
            Xhi.append(A.mean(0)); continue
        nrm = np.linalg.norm(A, axis=1)
        ix = np.argsort(-nrm)[:max(3, int(0.3 * len(A)))]
        Xhi.append(A[ix].mean(0))
    Xhi = np.stack(Xhi)

    # ---- 新先验：FG 计数 与 伪谱 ----
    FGmat = np.zeros((len(keys), len(FG)))
    PSraw = np.zeros((len(keys), n))
    for i, k in enumerate(keys):
        prof = C.prof_of(k)
        tw = sum(w for _, w in prof) or 1.0
        fgc = np.zeros(len(FG))
        ps = np.zeros(n)
        for smi, w in prof:
            c = fg_counts(smi)
            fgc += c * (w / tw)
            ps += pseudo_spectrum(c, wn) * (w / tw)
        FGmat[i] = fgc
        PSraw[i] = ps

    # ---- 旧先验 ----
    z = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    CLM = np.asarray(z["clm"], float)
    ECFP = np.asarray(z["ecfp"], float)

    print("=" * 72)
    print("[A2] 官能团计数（组分重量加权）—— 化学合理性核对")
    print("=" * 72)
    hdr = f"{'key':10s}" + "".join(f"{nm[:7]:>8s}" for nm in FG_NAMES)
    print(hdr)
    for i, k in enumerate(keys):
        row = f"{k:10s}" + "".join(f"{v:8.2f}" for v in FGmat[i])
        print(row)

    print()
    print("=" * 72)
    print("[A1/A3] 结构一致性：先验类间几何 vs 实测谱类间几何（19 类，171 对）")
    print("=" * 72)
    from scipy.stats import spearmanr
    C_spec = cos_matrix(prep(Xmean, "snv"))
    C_hi = cos_matrix(prep(Xhi, "snv"))
    o_spec, o_hi = offdiag(C_spec), offdiag(C_hi)

    priors = {
        "CLM(语义384)": CLM,
        "ECFP(指纹2048)": ECFP,
        "FG(官能团15)": FGmat,
        "PS(伪谱240)": PSraw,
    }
    print(f"{'先验':16s} {'ρ vs 谱':>9s} {'p':>8s} | {'ρ vs 高浓度谱':>13s} {'p':>8s} | "
          f"{'CKA(谱)':>8s} {'CKA(高浓度)':>11s}")
    rho = {}
    ck = {}
    for nm, P in priors.items():
        o_P = offdiag(cos_matrix(P))
        r1 = spearmanr(o_P, o_spec)
        r2 = spearmanr(o_P, o_hi)
        k1 = cka(P, Xmean)
        k2 = cka(P, Xhi)
        rho[nm] = (r1.statistic, r1.pvalue, r2.statistic, r2.pvalue)
        ck[nm] = (k1, k2)
        print(f"{nm:16s} {r1.statistic:+9.3f} {r1.pvalue:8.4f} | "
              f"{r2.statistic:+13.3f} {r2.pvalue:8.4f} | {k1:8.3f} {k2:11.3f}")

    print()
    print("=" * 72)
    print("[A3] 免训练零样本检索：把先验当检索库，实测谱做最近邻（chance=1/19=0.053）")
    print("=" * 72)
    # 伪谱与实测谱同构 → 可直接在谱域检索；CLM/ECFP 不同构，仅作对照时
    # 需投影，此处退化为「用谱域的伪谱」与「随机」对照。
    # 对每个实体：留出该实体的谱，在 19 个候选中检索。
    res_ret = {}
    for how in ["raw", "snv", "msc", "deriv1", "snv+deriv1"]:
        Xp = prep(Xmean, how)
        PSp = prep(PSraw, how)
        if Xp.shape[1] != PSp.shape[1]:
            continue
        # 检索：对每条测试谱 x，argmax cos(x, PS_i)
        Pn = PSp / (np.linalg.norm(PSp, axis=1, keepdims=True) + 1e-12)
        hits = 0
        tot = 0
        per_key = {}
        for i, k in enumerate(keys):
            A = np.asarray(ent[k], float) if k in ent else np.asarray(bg[k[2:]], float)
            Ap = prep(A, how)
            An = Ap / (np.linalg.norm(Ap, axis=1, keepdims=True) + 1e-12)
            pred = (An @ Pn.T).argmax(1)
            acc = float((pred == i).mean())
            per_key[k] = acc
            hits += int((pred == i).sum())
            tot += len(A)
        res_ret[how] = (hits / tot, per_key)
        mk = np.mean([per_key[k] for k in eids])
        print(f"  prep={how:12s} 全19类 micro={hits/tot:.3f}   13实体 macro={mk:.3f}")

    print()
    print("  逐实体（snv，13 个掺假/鉴别实体）:")
    for k in eids:
        print(f"    {k:10s} {res_ret['snv'][1][k]:.3f}")

    np.savez(os.path.join(OUT, "exp41_prior_cache.npz"),
             keys=np.array(keys), fg=FGmat, ps=PSraw,
             clm=CLM, ecfp=ECFP, wn=wn)
    print()
    print(f"[exp41] 已保存 out/exp41_prior_cache.npz  (fg={FGmat.shape}, ps={PSraw.shape})")


if __name__ == "__main__":
    main()
