"""exp41b —— 修正检索协议后重测「伪谱先验」是否有效。

=============================================================================
为什么必须做这一支（对 exp41 主脚本的直接回应）
=============================================================================
exp41 的 A3（免训练检索）几乎全 0（snv: 0.000 / deriv1: 0.074 ≈ chance 0.053），
当时读作「官能团先验无效」。但那个检验的协议有一个**致命缺陷**：

    测试对象是「混合物谱」x = μ_matrix + c·a_adulterant + noise
    而先验描述的是「纯掺杂物」a
    ⇒ cos(x, PS_i) 被基质项 μ 主导（掺杂物仅占 1–10%）
    ⇒ 所有候选的余弦都在 0.99 附近，检索必然退化为随机

这不是先验错了，是**协议错了**。

真实筛查场景里，基质是**已知**的：送检样品标签写着"肉桂粉"，
要回答的是"里面掺了什么"。这正是 NAS 匹配滤波的设定前提。
⇒ 正确协议是：**先在基质子空间上做正交投影扣除，再用残差检索**。

=============================================================================
本支检验（纯 CPU，无训练）
=============================================================================
对每条测试谱 x：
    r = x − P_M(x)          P_M = 基质背景谱张成子空间的投影
再与伪谱库 {PS_i} 做余弦检索，取 argmax。

四类对照，逐层剥离混淆：
  R1 原始混合物谱直接检索（= exp41 的失败协议，作基线）
  R2 扣除**真实基质**后的残差检索（合法：基质已知）
  R3 扣除**错误基质**后的残差检索（对照：证明收益来自基质扣除而非残差化）
  R4 高浓度子集（掺杂物信号最强 30%）+ R2

并报告：
  - 只检索 13 个掺假/鉴别实体 vs 全 19 类（含基质）
  - 高/低浓度分层（验证 H41d：先验收益应随浓度上升）

=============================================================================
预注册判据（跑完不再改）
=============================================================================
- H41b'（主）：R2 的 13 实体 macro top-1 ≥ 2 × chance(1/13=0.077) = 0.154
  若成立 ⇒ 伪谱先验有效，exp41 的 0.000 是协议缺陷，先验知识改进方向成立
  若不成立 ⇒ 先验知识改进方向被证伪，需在论文中如实报告
- H41d'（次）：R2 高浓度 > R2 低浓度（剂量-反应）
- 对照：R3 ≈ R1（证明不是"残差化"本身带来收益）
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1
import exp34_cache_reprs as C

OUT = r"D:/2026sci论文实验设计/论文6/out"


def proj_matrix(B: np.ndarray, k: int = 6) -> np.ndarray:
    """基质背景谱的主成分子空间正交投影矩阵。"""
    Bc = B - B.mean(0, keepdims=True)
    U, s, Vt = np.linalg.svd(Bc, full_matrices=False)
    k = min(k, len(s))
    return Vt[:k].T @ Vt[:k]                     # (D, D)


def unit(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def retrieve(Xte: np.ndarray, LIB: np.ndarray, Pm: np.ndarray | None) -> np.ndarray:
    """返回每条测试谱的预测类别索引。"""
    if Pm is not None:
        Xte = Xte - Xte @ Pm
    Ln = unit(LIB)
    An = unit(Xte)
    return (An @ Ln.T).argmax(1)


def main():
    ent, bg, _ = G1.load_g1()
    eids = sorted(ent.keys())
    mids = sorted(bg.keys())
    keys = eids + [f"M_{m}" for m in mids]

    z = np.load(os.path.join(OUT, "exp41_prior_cache.npz"), allow_pickle=True)
    PS = np.asarray(z["ps"], float)              # 伪谱 (19, 240)
    FG = np.asarray(z["fg"], float)

    # 每个实体的基质归属
    mat_of = {}
    for e in eids:
        meta = G1.ENTITY_META[e]
        mat_of[e] = meta[1] if len(meta) > 1 else None
    for m in mids:
        mat_of[f"M_{m}"] = m

    # 基质投影矩阵（用背景谱库）
    Pm_true = {m: proj_matrix(np.asarray(bg[m], float)) for m in mids}

    print("实体 → 基质：", {e: mat_of[e] for e in eids})
    print()

    # ---- 四种协议 ----
    def run(tag, use_sub, use_wrong, hi_only):
        per = {}
        for i, k in enumerate(keys):
            if k.startswith("M_") and not use_sub:
                continue
            A = np.asarray(ent[k], float) if k in ent else np.asarray(bg[k[2:]], float)
            if hi_only and len(A) >= 10:
                nrm = np.linalg.norm(A, axis=1)
                A = A[np.argsort(-nrm)[:max(3, int(0.3 * len(A)))]]
            mid = mat_of[k]
            if use_sub:
                Pm = Pm_true.get(mid)
            elif use_wrong:
                # 用另一个基质（循环错位）作对照
                others = [m for m in mids if m != mid]
                Pm = Pm_true[others[i % len(others)]] if others else None
            else:
                Pm = None
            cand = list(range(len(keys)))
            pred = retrieve(A, PS[cand], Pm)
            true_ix = cand.index(i)
            per[k] = float((pred == true_ix).mean())
        mk = np.mean([per[k] for k in eids if k in per])
        return mk, per

    print(f"{'协议':34s} {'13实体macro':>11s}   chance=0.077 (13类) / 0.053 (19类)")
    rows = {}
    for tag, kw in [
        ("R1 原始混合物谱（无扣除）", dict(use_sub=False, use_wrong=False, hi_only=False)),
        ("R2 扣除真实基质（合法协议）", dict(use_sub=True, use_wrong=False, hi_only=False)),
        ("R3 扣除错误基质（对照）", dict(use_sub=False, use_wrong=True, hi_only=False)),
        ("R4 扣除真实基质+高浓度", dict(use_sub=True, use_wrong=False, hi_only=True)),
    ]:
        mk, per = run(tag, **kw)
        rows[tag] = (mk, per)
        print(f"{tag:34s} {mk:11.3f}")

    print()
    print("逐实体（R2 扣除真实基质）:")
    for k in eids:
        print(f"  {k:10s} 基质={str(mat_of[k]):10s} R1={rows['R1 原始混合物谱（无扣除）'][1][k]:.3f} "
              f"R2={rows['R2 扣除真实基质（合法协议）'][1][k]:.3f} "
              f"R4={rows['R4 扣除真实基质+高浓度'][1][k]:.3f}")

    # ---- 浓度分层（H41d'）----
    print()
    try:
        entc, _, conc = G1.load_g1_conc()
        print("浓度分层（R2 协议，按每条谱的掺混浓度三分位）:")
        allc = []
        for k in eids:
            if k in conc:
                allc.extend(np.asarray(conc[k]).ravel())
        q1, q2 = np.quantile(allc, [1 / 3, 2 / 3])
        print(f"  浓度分位: q33={q1:.4f}  q67={q2:.4f}")
        for lo, hi, nm in [(None, q1, "低"), (q1, q2, "中"), (q2, None, "高")]:
            accs = []
            for i, k in enumerate(keys):
                if k.startswith("M_"):
                    continue
                A = np.asarray(ent[k], float)
                cc = np.asarray(conc.get(k, np.zeros(len(A)))).ravel()
                sel = np.ones(len(A), bool)
                if lo is not None:
                    sel &= cc >= lo
                if hi is not None:
                    sel &= cc < hi
                if sel.sum() < 3:
                    continue
                Pm = Pm_true.get(mat_of[k])
                pred = retrieve(A[sel], PS, Pm)
                accs.append(float((pred == i).mean()))
            print(f"  {nm}浓度 n={len(accs):2d} 类  macro={np.mean(accs):.3f}")
    except Exception as e:
        print(f"  [跳过浓度分层] {e}")

    # ---- 判据裁定 ----
    print()
    print("=" * 60)
    mk2 = rows["R2 扣除真实基质（合法协议）"][0]
    mk1 = rows["R1 原始混合物谱（无扣除）"][0]
    mk3 = rows["R3 扣除错误基质（对照）"][0]
    mk4 = rows["R4 扣除真实基质+高浓度"][0]
    print(f"H41b'（主）R2 macro = {mk2:.3f}  vs 2×chance = 0.154  ⇒ "
          f"{'✅ 成立' if mk2 >= 0.154 else '❌ 不成立'}")
    print(f"对照：R1 = {mk1:.3f}（无扣除）、R3 = {mk3:.3f}（扣错基质）"
          f" ⇒ 收益来自{'基质扣除' if mk2 > max(mk1, mk3) + 0.05 else '非基质因素'}")
    print(f"H41d'（次）R4(高浓度) = {mk4:.3f} vs R2 = {mk2:.3f} ⇒ "
          f"{'✅ 成立' if mk4 > mk2 else '❌ 不成立'}")


if __name__ == "__main__":
    main()
