"""exp87b —— 变换扫描的配对检验与分层诊断

回答三个问题
------------
1. 固定同一个变换后，CLM 是否真的赢过每个对手？（配对 t，配对单元 = 候选对单元格）
2. |U|=5 的落后是否集中在含 A05/A06（纯基质谱退化实体）的判别对上？
3. ABTT 的收益是否与先验的各向异性程度相关？（能否写成机制）

口径
----
配对单元 = (tag, T|D) 单元格，每个单元格一个 auc_dis 差值 d = CLM − 对手。
主检验：逐单元格配对 t（n = 单元格数）。
保守检验：按 seed 聚类的三明治 SE（cluster-robust），t 分布 df = n_cluster − 1。
（memory 规定：不做折聚类 bootstrap；用按折聚类稳健 SE。）
"""
from __future__ import annotations
import os, sys, json, itertools
import numpy as np
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
DEG = {"A05", "A06"}          # 纯基质谱退化实体
PRIS = ["clm", "charbag", "physchem", "ecfp2048", "maccs"]


def load(nhu):
    d = json.load(open(os.path.join(OUT, f"exp87_nhu{nhu}.json"), encoding="utf-8"))
    return d


def cell_series(cells, p, tk, lam, kind="dis"):
    """返回 {(tag,T,D): v}"""
    return cells[f"{p}|{tk}|{lam:g}"][kind]


def paired(d1, d2, filt=None):
    """d1-d2 配对差；filt 作用于 cell 键。返回 (mean, t, p, n)"""
    keys = sorted(set(d1) & set(d2))
    if filt:
        keys = [k for k in keys if filt(k)]
    if len(keys) < 3:
        return float("nan"), float("nan"), float("nan"), len(keys)
    a = np.array([d1[k] for k in keys])
    b = np.array([d2[k] for k in keys])
    dif = a - b
    t, p = stats.ttest_1samp(dif, 0.0)
    return float(dif.mean()), float(t), float(p), len(keys)


def cluster_t(d1, d2, filt=None):
    """按 seed 聚类稳健 SE。cell 键形如 s{seed}c{combo}|T|D"""
    keys = sorted(set(d1) & set(d2))
    if filt:
        keys = [k for k in keys if filt(k)]
    if not keys:
        return float("nan"), float("nan"), float("nan"), 0
    dif = np.array([d1[k] - d2[k] for k in keys])
    cl = np.array([k.split("|")[0].split("c")[0] for k in keys])
    gs = sorted(set(cl))
    G = len(gs)
    if G < 2:
        return float(dif.mean()), float("nan"), float("nan"), G
    N = len(dif)
    # 组均值与组内离差 → 三明治
    gm = {g: dif[cl == g].mean() for g in gs}
    ns = {g: int((cl == g).sum()) for g in gs}
    num = sum(ns[g] ** 2 * (gm[g] - dif.mean()) ** 2 for g in gs)
    var = num / (N ** 2) * (G / (G - 1))
    se = np.sqrt(max(var, 1e-18))
    t = dif.mean() / se
    p = 2 * stats.t.sf(abs(t), df=G - 1)
    return float(dif.mean()), float(t), float(p), G


def main():
    lines = []
    W = lines.append

    for nhu in (3, 5):
        d = load(nhu)
        cells = d["cells"]
        agg = d["agg"]
        lams = d["meta"]["lams"]
        trs = d["meta"]["trs"]
        W(f"\n{'='*78}\n## |U|={nhu}   （单元格 n={agg['clm|raw|0.0001']['n_cell_dis']}）\n")

        # ---- 每个 (先验,变换) 的最优 λ（按 auc_dis）----
        best = {}
        for p in PRIS:
            for tk in trs:
                cand = [(agg[f"{p}|{tk}|{L:g}"]["auc_dis"], L) for L in lams]
                v, L = max(cand)
                best[(p, tk)] = (v, L)
        besttr = {p: max(trs, key=lambda t: best[(p, t)][0]) for p in PRIS}
        W("各先验最优变换： " + "  ".join(
            f"{p}→{besttr[p]}({best[(p,besttr[p])][0]:.4f})" for p in PRIS))

        # ---- 关键：固定同一个变换做公平比较 ----
        for tk in ["raw", "abtt1", "pca16"]:
            if tk not in trs:
                continue
            W(f"\n### 固定变换 = {tk}")
            W(f"{'先验':10s} {'auc_dis':>8s} {'λ':>7s} {'top1':>7s}   vs CLM: Δ(配对)  p(配对)  p(聚类)")
            ref_v, ref_L = best[("clm", tk)]
            ref = cell_series(cells, "clm", tk, ref_L)
            for p in PRIS:
                v, L = best[(p, tk)]
                t1 = agg[f"{p}|{tk}|{L:g}"]["top1"]
                if p == "clm":
                    W(f"{p:10s} {v:8.4f} {L:7g} {t1:7.3f}   （基准）")
                    continue
                cur = cell_series(cells, p, tk, L)
                md, t, pv, n = paired(ref, cur)
                mc, tc, pc, G = cluster_t(ref, cur)
                W(f"{p:10s} {v:8.4f} {L:7g} {t1:7.3f}   {md:+8.4f} (n={n:3d}) "
                  f"{pv:7.4f} {pc:7.4f} (G={G})")

        # ---- 诊断：|U|=5 的落后是否集中在含退化实体的对 ----
        if nhu == 5:
            W("\n### 诊断：按判别对是否含 A05/A06 分层（变换=raw，各取最优 λ）")
            for p in ["physchem", "charbag", "ecfp2048", "maccs"]:
                ref = cell_series(cells, "clm", "raw", best[("clm", "raw")][1])
                cur = cell_series(cells, p, "raw", best[(p, "raw")][1])
                for lab, f in [("含A05/A06", lambda k: bool(set(k.split("|")[1:]) & DEG)),
                               ("不含(clean)", lambda k: not (set(k.split("|")[1:]) & DEG))]:
                    md, t, pv, n = paired(ref, cur, f)
                    W(f"  {p:10s} {lab:12s} Δ={md:+.4f} p={pv:.4f} n={n}")

        # ---- 机制：ABTT 收益 vs 各向异性 ----
        if nhu == 3:
            W("\n### 机制：ABTT 收益 与 码本平均余弦（各向异性）的关系")
            diag = json.load(open(os.path.join(OUT, "exp81_prior_diag.json"),
                                  encoding="utf-8"))["diag"]
            xs, ys = [], []
            for p in PRIS:
                dlt = best[(p, "abtt1")][0] - best[(p, "raw")][0]
                W(f"  {p:10s} cos_mean={diag[p]['cos_mean']:+.3f}  "
                  f"ABTT收益={dlt:+.4f}")
                xs.append(diag[p]["cos_mean"]); ys.append(dlt)
            if len(xs) >= 3:
                r, pr = stats.pearsonr(xs, ys)
                W(f"  Pearson r={r:+.3f} p={pr:.3f}  (n={len(xs)}，仅描述，不作推断)")

    path = os.path.join(OUT, "exp87_prior_transform.md")
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[exp87b] 已写出 {path}")


if __name__ == "__main__":
    main()
