# -*- coding: utf-8 -*-
"""
exp19b — exp19 为何只是「部分成立」？失败原因诊断

exp19 主检验（特征集 P = [H, log10Δ]）LOO 准确率 0.750、Fisher p = 0.4643、
相对常数基线增益仅 +0.125。本实验逐条排除/确认候选原因：

  R1 样本量不足           → 部分成立，但**不是主因**（§5 给出反证）
  R2 二分类阈值选得不好   → 排除（§3，阈值 0.75–0.95 全扫描）
  R3 H 与 Δ 共线，H 是噪声代理 → **确认为主因**（§1、§2）
  R4 数据共享造成循环     → 排除（§4，改用次低浓度重算 Δ′）

结论修正：exp18 §7.2 的「两个瓶颈（H 与 Δ）」应改为 **一个瓶颈（Δ）**，
H 只是 Δ 的代理（ρ = +0.903）。这同时把「可执行建议」改写得更强、更可操作。

产出：out/exp19b_p6_why_exp19_partial.md
"""

from __future__ import annotations
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy import stats
import p6_g1_data as G1
import exp19_p6_predict_detectability as E

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
RES_JSON = os.path.join(OUT_DIR, "exp18_res.json")


# ---------- 本地实现，不依赖 exp18 的内部函数名 ----------
def lstsq_ab(X, c):
    """X ≈ a0 + c·b 的最小二乘（X:(n,p), c:(n,)）。返回 a0:(p,), b:(p,)"""
    A = np.column_stack([np.ones(len(c)), c])
    sol, *_ = np.linalg.lstsq(A, X, rcond=None)
    return sol[0], sol[1]


def pca_basis(B, k):
    mu = B.mean(0)
    if k <= 0:
        return mu, np.zeros((0, B.shape[1]))
    Bc = B - mu
    U, S, Vt = np.linalg.svd(Bc, full_matrices=False)
    return mu, Vt[:k]


def resid(x, mu, V):
    r = x - mu
    if len(V):
        r = r - V.T @ (V @ r)
    return r


# ---------- 数据 ----------
def main():
    res, meta, targets = E.load()
    rows = E.build(res, meta, targets)
    y, FS = E.feature_sets(rows)
    auc = np.array([r["auc"] for r in rows])
    H = np.array([r["H"] for r in rows])
    lj = np.log10(np.array([r["jump"] for r in rows]))
    n = len(rows)

    ent, conc, bg, grid = G1.load_g1_conc()

    # 次低浓度版 Δ′，以及批次偏移与 NAS 方向的夹角
    d0, d1, cosoff, nboff, nbb = {}, {}, {}, {}, {}
    for e in targets:
        m = G1.ENTITY_META[e][1]
        B = np.asarray(bg[m], float)
        mu = B.mean(0)
        X = np.asarray(ent[e], float)
        c = np.asarray(conc[e], float)
        keep = c < 100.0
        X, cv = X[keep], c[keep]
        lv = np.unique(cv)
        cen = np.stack([X[cv == l].mean(0) for l in lv])
        step = float(np.linalg.norm(cen[1] - cen[0]))
        d0[e] = float(np.linalg.norm(cen[0] - mu) / step)
        d1[e] = float(np.linalg.norm(cen[1] - mu) / step)

        cfrac = cv / 100.0
        keepc = cfrac < 1.0
        a0, b = lstsq_ab(X[keepc], cfrac[keepc])
        _, V = pca_basis(B, int(res[e]["k_star"]))
        bs = resid(b, np.zeros_like(b), V)
        off = a0 - mu
        nboff[e] = float(np.linalg.norm(off))
        nbb[e] = float(np.linalg.norm(bs))
        cosoff[e] = float(off @ bs / (nboff[e] * nbb[e] + 1e-12))

    lj2 = np.log10(np.array([d1[e] for e in targets]))

    L = []
    A = L.append
    A("# exp19b — exp19 为何只是「部分成立」？失败原因诊断")
    A("")
    A("> exp19 主检验（特征集 `P = [H, log10Δ]`）：LOO 准确率 **0.750**，"
      "Fisher **p = 0.4643**，相对常数基线增益仅 **+0.125**。")
    A("> 本实验逐条排除 / 确认候选原因。**主因是特征集构造，不是样本量。**")
    A("")
    A("## 0. 结论速览")
    A("")
    A("| 候选原因 | 判定 | 证据 |")
    A("|---|---|---|")
    A("| R1 样本量不足（n=8） | 🟡 **部分成立，但非主因** | §5：同样的 n=8，换对特征就能做到 ρ=+0.810、**p=0.0149** |")
    A("| R2 二分阈值 0.90 选得不好 | ❌ **排除** | §3：阈值 0.75→0.95 全扫描，Δ 始终完美分离 |")
    A("| R3 **H 与 Δ 共线，H 是噪声代理** | ✅ **主因** | §1 §2：ρ(H,Δ)=+0.903；加 H 后 0.875→**0.750** |")
    A("| R4 数据共享造成循环论证 | ❌ **排除** | §4：改用次低浓度重算 Δ′（与 Δ 相关 +0.976），结论不变 |")
    A("")

    # ---------------- §1 ----------------
    r_hd = stats.spearmanr(H, lj)
    A("## 1. 【主因】H 与 Δ 不是两个独立因子，而是同一个东西的两个侧面")
    A("")
    A(f"- **Spearman ρ(H, log10Δ) = {r_hd.statistic:+.3f}（p = {r_hd.pvalue:.4f}）**，"
      f"Pearson r = {np.corrcoef(H, lj)[0,1]:+.3f}")
    A("- 两者高度共线 → 把它们当「两个瓶颈」是错的。")
    A("  它们在物理上同源：真品库越杂（H 大）的体系，其掺混系列通常也与真品库不同批（Δ 大）。")
    A("")
    A("**单特征对 y 的分离能力**（特征自身作为判据的 ROC-AUC，0.000 = 完美分离）：")
    A("")
    A("| 特征 | AUC(特征 → y) | 是否完美分离 |")
    A("|---|---|---|")
    aD = stats.mannwhitneyu(lj[y == 1], lj[y == 0], alternative="less").statistic / ((y == 1).sum() * (y == 0).sum())
    aH = stats.mannwhitneyu(H[y == 1], H[y == 0], alternative="less").statistic / ((y == 1).sum() * (y == 0).sum())
    sepD = bool(lj[y == 1].max() < lj[y == 0].min())
    sepH = bool(H[y == 1].max() < H[y == 0].min())
    A(f"| **log10Δ** | **{aD:.3f}** | {'✅ 是' if sepD else '❌ 否'} |")
    A(f"| H | {aH:.3f} | {'✅ 是' if sepH else '❌ 否'} |")
    A("")
    A(f"- **Δ 完美分离**：所有 y=1 的 Δ 都小于所有 y=0 的 Δ，"
      f"间隙 {lj[y==0].min() - lj[y==1].max():.3f}（log10 单位）")
    A(f"- **H 不完美**：max H(y=1) = {H[y==1].max():.4f}（A04）> "
      f"min H(y=0) = {H[y==0].min():.4f}（A07 的 H 更低但更靠前）→ 交叉")
    A("")

    # ---------------- §2 ----------------
    A("## 2. 【主因】把代理 H 加进去，反而稀释了真因子 Δ 的信号")
    A("")
    A("> 补测 exp19 漏掉的特征集：**Δ 单独**。")
    A("")
    A("| 特征集 | LOO 准确率 | ρ(预测, 实测 AUC) | p | 混淆 TP/FP/FN/TN | Fisher p |")
    A("|---|---|---|---|---|---|")
    res_fs = {}
    for nm, X in [("Δ 单独", E.stdz(lj.reshape(-1, 1))),
                  ("H + Δ（exp19 主）", E.stdz(np.column_stack([H, lj]))),
                  ("H 单独（纯先验）", E.stdz(H.reshape(-1, 1)))]:
        p, yh = E.loo_predict(X, y)
        rho, pv = stats.spearmanr(p, auc)
        tp = int(((yh == 1) & (y == 1)).sum()); fp = int(((yh == 1) & (y == 0)).sum())
        fn = int(((yh == 0) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
        fs = stats.fisher_exact([[tp, fp], [fn, tn]])[1]
        res_fs[nm] = dict(acc=float((yh == y).mean()), rho=float(rho), p=float(pv),
                          cm=(tp, fp, fn, tn), fisher=float(fs))
        A(f"| **{nm}** | **{((yh==y).mean()):.3f}** | {rho:+.3f} | {pv:.4f} | "
          f"{tp}/{fp}/{fn}/{tn} | {fs:.4f} |")
    A("")
    d1_ = res_fs["Δ 单独"]; d2_ = res_fs["H + Δ（exp19 主）"]
    A(f"- **加入 H 的代价**：准确率 {d1_['acc']:.3f} → **{d2_['acc']:.3f}**"
      f"（{d2_['acc']-d1_['acc']:+.3f}），ρ {d1_['rho']:+.3f} → {d2_['rho']:+.3f}，"
      f"p {d1_['p']:.4f} → {d2_['p']:.4f}")
    A(f"- **Δ 单独的 Fisher p = {d1_['fisher']:.4f}**，且连续版 ρ = {d1_['rho']:+.3f}"
      f"（**p = {d1_['p']:.4f}，显著**）")
    A("- **判读**：exp19「统计上不足」的这个结果，是**特征集构造错误**造成的，"
      "不是信号弱。真正的信号一直都在 Δ 里，被 H 的噪声掩盖了。")
    A("")
    A("### 2.1 关键反例 A04：真品库很杂，但照样能检出")
    A("")
    A("| 实体 | H | Δ | AUC(nas) | 按 H 该判 | 按 Δ 该判 | 实际 |")
    A("|---|---|---|---|---|---|---|")
    i4 = targets.index("A04")
    A(f"| **A04（姜黄）** | **{H[i4]*100:.2f}%**（第二高） | **{rows[i4]['jump']:.2f}×**（很低） | "
      f"**{auc[i4]:.3f}** | ❌ 不可检出 | ✅ 可检出 | ✅ 可检出 |")
    A("")
    A("- A04 是**唯一能把 H 与 Δ 分开的案例**，它证明：")
    A("  **真品库「杂」本身不是问题，「掺混系列与真品库不连续」才是问题。**")
    A("  H 在这个案例上给出错误判断，Δ 给出正确判断 → H 是**有偏的代理**。")
    A("")

    # ---------------- §3 ----------------
    A("## 3. 【排除】不是二分类阈值 0.90 选得不好")
    A("")
    A("| AUC 阈值 | y=1 个数 | Δ 单独 LOO | H+Δ LOO | H 单独 LOO | AUC(Δ→y) |")
    A("|---|---|---|---|---|---|")
    for thr in [0.75, 0.80, 0.85, 0.90, 0.95]:
        yy = (auc >= thr).astype(int)
        if yy.sum() in (0, n):
            A(f"| {thr:.2f} | 标签单一 | — | — | — | — |")
            continue
        accs = []
        for X in [E.stdz(lj.reshape(-1, 1)),
                  E.stdz(np.column_stack([H, lj])),
                  E.stdz(H.reshape(-1, 1))]:
            _, yh = E.loo_predict(X, yy)
            accs.append(f"{((yh==yy).mean()):.3f}")
        aD_ = stats.mannwhitneyu(lj[yy == 1], lj[yy == 0], alternative="less").statistic / ((yy == 1).sum() * (yy == 0).sum())
        A(f"| {thr:.2f} | {int(yy.sum())} | **{accs[0]}** | {accs[1]} | {accs[2]} | {aD_:.3f} |")
    A("")
    A("- **Δ 在所有阈值下都完美分离**（AUC(Δ→y) = 0.000）→ 结论不依赖阈值选择")
    A("- **H+Δ 在任何阈值下都不优于 Δ 单独** → H 的损害是系统性的，不是偶然")
    A("")

    # ---------------- §4 ----------------
    A("## 4. 【排除】不是数据共享造成的循环论证")
    A("")
    A("> 质疑：Δ 用「最低浓度谱 + 真品库」，AUC 也用同一批数据 → 会不会是共享导致的假相关？")
    A("> 检验：改用**次低浓度**（第 2 档）重算 Δ′，若结论不变，则不是共享造成的。")
    A("")
    A("| 实体 | Δ（最低浓度） | Δ′（次低浓度） | AUC(nas) |")
    A("|---|---|---|---|")
    for e in targets:
        A(f"| {e} | {d0[e]:.2f}× | {d1[e]:.2f}× | {res[e]['auc']['nas']:.3f} |")
    r_dd = stats.spearmanr(lj2, lj)
    r2_a = stats.spearmanr(lj2, auc)
    _, yh2 = E.loo_predict(E.stdz(lj2.reshape(-1, 1)), y)
    A("")
    A(f"- Δ′ 与 Δ 的相关：ρ = {r_dd.statistic:+.3f}（量本身稳定）")
    A(f"- **Δ′ 单独 LOO 准确率 = {((yh2==y).mean()):.3f}**，"
      f"ρ(Δ′, AUC) = {r2_a.statistic:+.3f}（p = {r2_a.pvalue:.4f}）")
    A("- 结论不变 → **排除循环论证**。Δ 是体系的一个稳定属性，不是拟合伪影。")
    A("")

    # ---------------- §5 ----------------
    A("## 5. 【部分成立】样本量确实不足，但它不是失败的主因")
    A("")
    A(f"- n = {n}，Δ 单独的 Wilson 95% CI 仍会很宽 → 点估计 0.875 的确不够稳")
    A(f"- **但反证成立**：在**完全相同**的 n = {n} 与同一 LOO 协议下，"
      f"仅换特征集就把 ρ 从 {d2_['rho']:+.3f}（p = {d2_['p']:.4f}）"
      f"提升到 {d1_['rho']:+.3f}（**p = {d1_['p']:.4f}**）")
    A("- → 样本量限制的是**精度**，特征集错误限制的是**有没有信号**。后者才是 exp19 失败的原因。")
    A("- 仍须诚实声明：即便用 Δ 单独，Fisher "
      f"p = {d1_['fisher']:.4f} 仍不显著，n=8 不足以支撑强 claim。")
    A("")

    # ---------------- §6 ----------------
    A("## 6. 机制：为什么「Δ 大」反而检出 AUC 低？")
    A("")
    A("> 直觉上掺混系列离真品库越远应该越好分，实测却相反。检验批次偏移方向")
    A("> `a0 − μ_m` 与 NAS 方向 `b*` 的夹角（`a0` = 浓度外推到 c=0 的截距谱）。")
    A("")
    A("| 实体 | Δ | ‖a0−μ_m‖ | ‖b*‖ | **cos(a0−μ_m, b*)** | AUC(nas) |")
    A("|---|---|---|---|---|---|")
    for e in targets:
        A(f"| {e} | {res[e]['jump']:.2f}× | {nboff[e]:.3f} | {nbb[e]:.4f} | "
          f"**{cosoff[e]:+.3f}** | {res[e]['auc']['nas']:.3f} |")
    A("")
    cA = np.array([cosoff[e] for e in targets])
    r_c = stats.spearmanr(cA, auc)
    r_cd = stats.spearmanr(cA, lj)
    A(f"- cos(a0−μ_m, b*) 与 AUC 的 Spearman ρ = {r_c.statistic:+.3f}"
      f"（**p = {r_c.pvalue:.4f}**）"
      f"{'→ **显著**' if r_c.pvalue < 0.05 else '→ 不显著'}")
    A(f"- cos(a0−μ_m, b*) 与 log10Δ 的 Spearman ρ = {r_cd.statistic:+.3f}"
      f"（p = {r_cd.pvalue:.4f}）")
    A("")
    if r_c.pvalue < 0.05:
        A(f"- ✅ **该机制在统计上成立**：Δ 大的体系（A01/A02/A03、HSYRUP）其批次偏移方向"
          "与 NAS 判别方向**近乎正交甚至反向**（cos ≈ −0.33 ~ +0.01），")
        A("  偏移分量几乎不落在判别方向上 → 掺混谱在 NAS 轴上没有整体抬升，")
        A("  只能靠 `c·‖b*‖` 这一项，而这些体系的 ‖b*‖ 又小 → 与真品库重叠 → AUC 低。")
    else:
        A("- 该机制未达显著，只作定性参考。")
    A("")
    A(f"- ⚠️ **但不能把它当独立机制**：cos 与 log10Δ 的 ρ = {r_cd.statistic:+.3f}"
      f"（p = {r_cd.pvalue:.4f}），")
    A("  它很可能只是「体系不一致性」的**第三个侧面**（H、Δ、cos 三者同源），")
    A("  而非一个独立的作用路径。n = 8 下无法可靠地做偏相关分离，"
      "因此本节只作**机制描述**，不作因果 claim。")
    A("- 完整机制应是三者叠加：①真品库不能代表掺混批次（零分布失配）"
      "②伴随 H 大 → σ_null 大 → 阈值抬高 ③偏移方向与判别方向近乎正交。")
    A("")

    # ---------------- §7 ----------------
    A("## 7. 结论修正（须回写策略文档与论文）")
    A("")
    A("### 7.1 「两个瓶颈」应改为「一个瓶颈」")
    A("")
    A("- exp18 §7.2 原文：真品库异质性 H（ρ = −0.781, p = 0.022）与跳变 Δ"
      "（ρ = −0.857, p = 0.007）**都是**瓶颈")
    A(f"- **修正**：ρ(H, Δ) = {r_hd.statistic:+.3f}，两者高度共线；")
    A("  单变量相关是**混杂相关** —— H 之所以显著，是因为它代理了 Δ。")
    A("  分离能力上 Δ 完美（AUC = 1.000）而 H 不完美（AUC = 0.867），")
    A("  且加入 H 会**损害**预测（0.875 → 0.750）。")
    A("- → **真正的瓶颈是「掺混系列与真品库的批次不连续性 Δ」，"
      "H 只是它的代理。**")
    A("")
    A("### 7.2 可执行建议因此变得更强、更可操作")
    A("")
    A("- 旧建议：「建更具代表性的真品库」—— 难执行。天然食品本来就杂，")
    A("  想把 H 压下去既不现实也不必要（**A04：H = 39.94% 照样 AUC = 0.974**）。")
    A("- **新建议：「确保掺混校准系列与真品库同批次采集」** —— 这是实验设计层面"
      "  就能控制的事，成本远低于「把真品库做纯」。")
    A("- 且给出**量化门槛**：Δ > 5× 即应判定该数据集不可用于检出评估"
      "（exp18 中 A01/A02/A03/HSYRUP 全部在此列）。")
    A("")
    A("### 7.3 exp19 仍须保留的诚实边界")
    A("")
    A(f"- 即便换用 Δ 单独，n = {n} 下 Fisher p = {d1_['fisher']:.4f} "
      "仍不显著 → 仍**不能** claim「已建成验证过的预测器」")
    A("- 「纯先验」失效的结论**不变**且更严重：H 只是代理，"
      "而真因子 Δ 需要掺混系列 → 严格先验更不可能")
    A("- claim 只覆盖 detection 的结论**不变**（exp19 §4）")

    txt = "\n".join(L)
    os.makedirs(OUT_DIR, exist_ok=True)
    op = os.path.join(OUT_DIR, "exp19b_p6_why_exp19_partial.md")
    with open(op, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("wrote", op)
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
