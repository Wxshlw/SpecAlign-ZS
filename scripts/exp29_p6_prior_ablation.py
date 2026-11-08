# -*- coding: utf-8 -*-
"""
exp29 — 化学先验的消融阶梯诊断（Prior Ablation Ladder）

背景（用户 2026-11-01 提出的硬问题）：
    论文 6 是「基于化学语义结构先验」的模型。要 claim 先验有用，消融必须成阶梯：
        A0 无先验 / 无字典      —— 不加任何化学信息能做到吗？
        A1 通用化学字典（ECFP） —— 普通字典够不够？
        A2 我们的先验（CLM）    —— 化学语言模型有增量吗？
        A3 融合（CLM ⊕ ECFP）   —— 两者是否互补？
    现状：exp14 只比了 A1 vs A2（统计不可区分），**缺 A0**，也**没做 A3 互补性检验**。

本脚本（零重训，纯重分析，基于 out/exp08_sims.npy）：
    Part 1  互补性主检验 H29a：融合臂 vs 最强单臂（聚类 bootstrap + Holm）
    Part 2  冗余性 H29b：union oracle / 分歧 McNemar —— 两者是否捕获不同样本
    Part 3  加权融合扫描：最优 α 落在中间（互补）还是端点（冗余）
    Part 4  top-k 曲线（k=1,3,5）：CLM 是否只在粗排上占优
    Part 5  逐折胜负表：CLM 赢在哪类掺杂物

预注册判据
    H29a 互补：  top1(CLM⊕ECFP) − max(top1(CLM), top1(ECFP)) > 0，Holm 单侧 p ≤ 0.05
                 ⇒ 成立即可 claim「化学语义提供 ECFP 未覆盖的结构信息」（保住化学语义招牌）
                 ⇒ 不成立 ⇒ 走 B 路线：放弃编码器 claim，只 claim「结构对齐 + 失效边界 + 预算式」
    H29b 冗余：  union oracle 显著高于各单臂 ⇒ 存在互补空间（必要非充分）
    α*：         若 α* 显著偏离 0/1 两端 ⇒ 互补；若 α*=0（纯 ECFP 最优）⇒ CLM 无贡献

⚠️ A0（无字典基线）无法由本脚本产出：它需要重训谱编码器（替换候选表示后重新对齐）。
   本脚本在报告末尾显式标注该缺口与补齐代价。
"""

import os
import json
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
N_BOOT = 20000
ARMS = ["Original", "Shuffle-Random", "Shuffle-Mismatch", "Random-Encoder", "Descriptor"]
MAIN = ["Original", "Descriptor"]          # CLM 对齐路线 / ECFP 通用字典路线


# ---------------------------------------------------------------- 工具
def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj


def topk_hits(sim, gt, k=1):
    """sim (n, C) → 前 k 命中 bool(n,)。gt 为该折真实候选列号。"""
    order = np.argsort(-sim, axis=1)
    return np.any(order[:, :k] == gt, axis=1)


def zrows(sim):
    mu = sim.mean(axis=1, keepdims=True)
    sd = sim.std(axis=1, keepdims=True) + 1e-12
    return (sim - mu) / sd


def rankrows(sim):
    """降序秩取负 → 越大越好（第一名 = 0，其余为负）。"""
    n, C = sim.shape
    order = np.argsort(-sim, axis=1)
    ranks = np.empty((n, C), dtype=np.float64)
    ranks[np.arange(n)[:, None], order] = np.arange(C)[None, :]
    return -ranks


def cluster_boot_delta(ho, hc, eids, n_boot=N_BOOT, seed=0):
    """fold 聚类 bootstrap：返回 (Δ, CI_lo, CI_hi, 单侧 p)。ho/hc: {eid: bool array}"""
    rng = np.random.default_rng(seed)
    so = np.array([ho[e].sum() for e in eids], float)
    sc = np.array([hc[e].sum() for e in eids], float)
    ns = np.array([len(ho[e]) for e in eids], float)
    obs = float(so.sum() - sc.sum()) / ns.sum()
    idxs = rng.integers(0, len(eids), size=(n_boot, len(eids)))
    bs = (so[idxs].sum(1) - sc[idxs].sum(1)) / ns[idxs].sum(1)
    centered = bs - bs.mean()          # 以 bootstrap 分布中心为零假设
    p = float((np.sum(centered <= obs) + 1) / (n_boot + 1))   # 单侧：Δ>0
    ci = np.percentile(bs, [2.5, 97.5])
    return obs, float(ci[0]), float(ci[1]), p


def mcnemar(b, c):
    """b/c 为不一致对计数，近似 McNemar（大样本正态）。"""
    from math import sqrt
    n = b + c
    if n == 0:
        return 1.0
    chi = (abs(b - c) - 1) ** 2 / n
    # 正态近似双尾 p
    from math import erfc
    z = sqrt(chi)
    return float(erfc(z / sqrt(2)))


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", default=os.path.join(OUT, "exp08_sims.npy"))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    S = np.load(args.sims, allow_pickle=True).item()
    eids = sorted(S["Original"].keys())
    n_fold = len(eids)
    chance = 1.0 / n_fold
    n_q = {e: np.asarray(S["Original"][e], float).shape[0] for e in eids}
    N_TOT = sum(n_q.values())
    print(f"[exp29] {n_fold} 折，{N_TOT} query，chance={chance:.4f}", flush=True)

    sims = {a: {e: np.asarray(S[a][e], float) for e in eids} for a in ARMS}

    # ---------- Part 0：单臂 top-1 基线 ----------
    base = {}
    for a in ARMS:
        h = {e: topk_hits(sims[a][e], i, 1) for i, e in enumerate(eids)}
        base[a] = {"hits": h,
                   "pooled": float(np.concatenate([h[e] for e in eids]).mean()),
                   "fold": {e: float(h[e].mean()) for e in eids}}

    # ---------- Part 1：融合主检验 ----------
    fuse_defs = {
        "Fuse-Z": lambda e: zrows(sims["Original"][e]) + zrows(sims["Descriptor"][e]),
        "Fuse-Rank": lambda e: rankrows(sims["Original"][e]) + rankrows(sims["Descriptor"][e]),
    }
    fuse = {}
    for name, fn in fuse_defs.items():
        h = {}
        for i, e in enumerate(eids):
            h[e] = topk_hits(fn(e), i, 1)
        fuse[name] = {"hits": h,
                       "pooled": float(np.concatenate([h[e] for e in eids]).mean()),
                       "fold": {e: float(h[e].mean()) for e in eids}}

    best_single = max(base["Original"]["pooled"], base["Descriptor"]["pooled"])
    best_name = "Original" if base["Original"]["pooled"] >= base["Descriptor"]["pooled"] else "Descriptor"

    tests = []
    for name in fuse_defs:
        d, lo, hi, p = cluster_boot_delta(fuse[name]["hits"], base[best_name]["hits"],
                                          eids, args.n_boot, seed=1)
        tests.append({"fuse": name, "vs": best_name, "delta": d, "ci": [lo, hi], "p_raw": p})
    padj = holm([t["p_raw"] for t in tests])
    for t, pa in zip(tests, padj):
        t["p_holm"] = float(pa)
        t["sig"] = bool(pa <= 0.05 and t["delta"] > 0)

    # ---------- Part 2：冗余性 / 分歧 ----------
    disc = {}
    for i, e in enumerate(eids):
        ho = base["Original"]["hits"][e]
        hd = base["Descriptor"]["hits"][e]
        disc[e] = {"b_only_clm": int(np.sum(ho & ~hd)), "c_only_ecfp": int(np.sum(~ho & hd)),
                   "both": int(np.sum(ho & hd)), "neither": int(np.sum(~ho & ~hd)),
                   "n": int(len(ho))}
    B = sum(d["b_only_clm"] for d in disc.values())
    C = sum(d["c_only_ecfp"] for d in disc.values())
    p_mcn = mcnemar(B, C)

    union_h = {e: base["Original"]["hits"][e] | base["Descriptor"]["hits"][e] for e in eids}
    union = {"pooled": float(np.concatenate([union_h[e] for e in eids]).mean()),
             "fold": {e: float(union_h[e].mean()) for e in eids}}
    inter_h = {e: base["Original"]["hits"][e] & base["Descriptor"]["hits"][e] for e in eids}
    inter = float(np.concatenate([inter_h[e] for e in eids]).mean())

    # ---------- Part 3：加权融合扫描 ----------
    alphas = np.round(np.arange(0, 1.01, 0.1), 2)
    scan = []
    for al in alphas:
        h = {}
        for i, e in enumerate(eids):
            z = al * zrows(sims["Original"][e]) + (1 - al) * zrows(sims["Descriptor"][e])
            h[e] = topk_hits(z, i, 1)
        scan.append({"alpha": float(al),
                     "pooled": float(np.concatenate([h[e] for e in eids]).mean())})
    a_star = max(scan, key=lambda d: d["pooled"])

    # ---------- Part 4：top-k ----------
    topk = {}
    for a in MAIN:
        for k in (1, 3, 5):
            h = {e: topk_hits(sims[a][e], i, k) for i, e in enumerate(eids)}
            topk[(a, k)] = float(np.concatenate([h[e] for e in eids]).mean())
    # 融合的 top-k
    for k in (1, 3, 5):
        h = {}
        for i, e in enumerate(eids):
            h[e] = topk_hits(zrows(sims["Original"][e]) + zrows(sims["Descriptor"][e]), i, k)
        topk[("Fuse-Z", k)] = float(np.concatenate([h[e] for e in eids]).mean())

    # ---------- 输出 ----------
    L = []
    A = L.append
    A("# exp29 — 化学先验的消融阶梯诊断（零重训 · 重分析 exp08_sims）\n")
    A(f"> {n_fold} 折 / {N_TOT} 个 query / chance = {chance:.4f}；融合与互补性检验，**未重训任何模型**。\n")

    A("## 0. 为什么要有这一层（消融阶梯）\n")
    A("要 claim「化学语义先验有用」，消融必须**成阶梯**，缺一层论证就断：\n")
    A("| 层 | 候选侧表示 | 回答什么问题 | 现状 |")
    A("|---|---|---|---|")
    A("| **A0** | 无化学信息（随机向量 / 纯光谱） | **不加字典能做到吗** | 🔴 **未做**（需重训） |")
    A("| **A1** | 通用化学字典（ECFP 指纹） | 普通字典够不够 | ✅ exp14 = 0.368 |")
    A("| **A2** | 我们的先验（CLM 对齐） | 化学语言模型有增量吗 | ✅ exp14 = 0.350 |")
    A("| **A3** | CLM ⊕ ECFP 融合 | 两者**互补**吗 | 🆕 本实验 |")
    A("")
    A("**exp14 的既有结论是 A2 与 A1 统计不可区分**（Δ=−0.018，CI 跨 0）⇒ 单看这一层，")
    A("「化学语义」无立足之地。A3 是**唯一能在不重训的前提下**救回它的检验：")
    A("若融合显著优于最强单臂，说明 CLM 带的是 ECFP 没有的信息（互补），而非更差的信息（冗余）。\n")

    A("## 1. 单臂基线（复算，与 exp14 一致）\n")
    A("| 臂 | pooled top-1 |")
    A("|---|---|")
    for a in ARMS:
        A(f"| {a} | {base[a]['pooled']:.3f} |")
    A(f"| **chance** | {chance:.3f} |")

    A("\n## 2. H29a 主检验：融合 vs 最强单臂\n")
    A(f"> 最强单臂 = **{best_name}**（{best_single:.3f}）；单侧聚类 bootstrap（fold 聚类，B={args.n_boot}），Holm 校正。\n")
    A("| 融合方式 | Δ vs 最强单臂 | 95% CI | raw p | Holm p | 判定 |")
    A("|---|---|---|---|---|---|")
    for t in tests:
        A(f"| {t['fuse']} | {t['delta']:+.4f} | [{t['ci'][0]:+.4f}, {t['ci'][1]:+.4f}] | "
          f"{t['p_raw']:.4f} | {t['p_holm']:.4f} | {'✅ 互补' if t['sig'] else '❌ 不成立'} |")
    A("")
    A(f"- Fuse-Z pooled = {fuse['Fuse-Z']['pooled']:.3f}，Fuse-Rank pooled = {fuse['Fuse-Rank']['pooled']:.3f}")
    A(f"- 参照：Original {base['Original']['pooled']:.3f} / Descriptor {base['Descriptor']['pooled']:.3f}\n")
    if any(t["sig"] for t in tests):
        A("**H29a 成立** ⇒ CLM 携带 ECFP 之外的结构信息，可 claim「化学语义提供互补信息」（不必单打赢）。")
    else:
        A("**H29a 不成立** ⇒ 无任何证据支持 CLM 的独立贡献 ⇒")
        A("**建议走 B 路线**：放弃编码器 claim，只 claim「结构表示与光谱的对齐 + 失效边界 + 检出限预算式」。")
    A("")

    A("## 3. H29b 冗余性：两者是否在解不同的题\n")
    A("| 量 | 值 | 含义 |")
    A("|---|---|---|")
    A(f"| CLM 独对 / ECFP 错（b） | {B} | CLM 独家贡献 |")
    A(f"| ECFP 独对 / CLM 错（c） | {C} | ECFP 独家贡献 |")
    A(f"| 两者都对 | {sum(d['both'] for d in disc.values())} | 重叠 |")
    A(f"| 两者都错 | {sum(d['neither'] for d in disc.values())} | 共同失效 |")
    A(f"| McNemar p（b vs c） | {p_mcn:.4g} | 谁更强 |")
    A(f"| **union oracle** | {union['pooled']:.3f} | 任一命中即算 ⇒ 互补空间上界 |")
    A(f"| intersection | {inter:.3f} | 两者都对 ⇒ 信息重叠度 |")
    A("")
    A("判读：**union 显著高于两个单臂 ⇒ 存在互补空间**（必要非充分；是否真能取到，看 H29a）。")
    A(f"union({union['pooled']:.3f}) − max 单臂({best_single:.3f}) = **{union['pooled']-best_single:+.3f}**\n")

    A("## 4. 加权融合扫描（α·CLM + (1−α)·ECFP）\n")
    A("| α | pooled top-1 |")
    A("|---|---|")
    for s in scan:
        mark = " ← α*" if s["alpha"] == a_star["alpha"] else ""
        A(f"| {s['alpha']:.1f} | {s['pooled']:.3f}{mark} |")
    A("")
    if a_star["alpha"] in (0.0, 1.0):
        A(f"**α\* = {a_star['alpha']:.1f}（端点）** ⇒ 最优解是纯用单一字典，**无互补证据**。")
    else:
        A(f"**α\* = {a_star['alpha']:.1f}（内部）** ⇒ 两端都不最优，存在互补；"
          f"但须以 H29a 的统计检验为准（点估计差异可能只是噪声）。")
    A("")

    A("## 5. top-k 曲线（CLM 是否只在粗排占优）\n")
    A("| 臂 | top-1 | top-3 | top-5 |")
    A("|---|---|---|---|")
    for a in MAIN + ["Fuse-Z"]:
        A(f"| {a} | {topk[(a,1)]:.3f} | {topk[(a,3)]:.3f} | {topk[(a,5)]:.3f} |")
    A("")

    A("## 6. 逐折明细\n")
    A("| fold | n | CLM(Original) | ECFP(Descriptor) | Fuse-Z | union | CLM独对 | ECFP独对 |")
    A("|---|---|---|---|---|---|---|---|")
    for e in eids:
        A(f"| {e} | {n_q[e]} | {base['Original']['fold'][e]:.3f} | {base['Descriptor']['fold'][e]:.3f} | "
          f"{fuse['Fuse-Z']['fold'][e]:.3f} | {union['fold'][e]:.3f} | {disc[e]['b_only_clm']} | {disc[e]['c_only_ecfp']} |")
    A("")

    A("## 7. ⚠️ 仍然缺失的一层：A0 无字典基线\n")
    A("本实验**无法**回答你提出的那个最根本的问题 ——")
    A("「**如果完全不加字典的模型就能做到我们能做到的事，那先验的优势何在？**」\n")
    A("原因：在本任务的检索范式下，**没有字典就没有候选集，top-1 无法定义**。")
    A("已有的 Shuffle-Random(0.023) / Shuffle-Mismatch(0.016) 只能说明「字典与谱的**对应关系**必须正确」，")
    A("不能说明「**化学信息本身**是否必需」—— 这两者是不同的命题。\n")
    A("补齐 A0 需要在同一协议下重训，候选侧换成：")
    A("1. **随机高斯向量**（匹配 CLM 嵌入的维度与范数）—— 回答「是否需要化学信息」；")
    A("2. **官能团 one-hot**（比 ECFP 更弱的字典）—— 回答「字典要多强才够」；")
    A("3. 谱侧**冻结** exp08 的编码器做零训练检索 —— 回答「检索时是否真的在读字典」。\n")
    A("代价：需要重跑 exp08 的对齐训练（13 折 × seeds）。**是否开跑，取决于 H29a 的结果**：")
    A("若 H29a 已不成立，A0 只会再补一刀，价值有限；若 H29a 成立，A0 是必须补的入场券。\n")

    md = "\n".join(L)
    p_md = os.path.join(OUT, "exp29_p6_prior_ablation.md")
    with open(p_md, "w", encoding="utf-8") as f:
        f.write(md)

    res = {
        "n_fold": n_fold, "n_query": N_TOT, "chance": chance,
        "base": {a: {"pooled": base[a]["pooled"], "fold": base[a]["fold"]} for a in ARMS},
        "fuse": {k: {"pooled": v["pooled"], "fold": v["fold"]} for k, v in fuse.items()},
        "h29a": tests, "best_single": best_name,
        "h29b": {"b": B, "c": C, "mcnemar_p": p_mcn, "union": union["pooled"],
                 "union_fold": union["fold"], "intersection": inter},
        "alpha_scan": scan, "alpha_star": a_star,
        "topk": {f"{a}_k{k}": v for (a, k), v in topk.items()},
        "discordance": disc,
    }
    p_json = os.path.join(OUT, "exp29_res.json")
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    print(md)
    print(f"\n[saved] {p_md}\n[saved] {p_json}")


if __name__ == "__main__":
    main()
