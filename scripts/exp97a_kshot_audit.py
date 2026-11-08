# -*- coding: utf-8 -*-
r"""exp97a —— k-shot 曲线的重复扫描审计：k 到底是不是 k。

【为什么要查这个】
MEMORY 的挂起项 B 说「k 曲线必须按独立样品组重定义」。原始实现（exp81b）是：

    te_ix, pool_ix = split_by_group(Xe, gid_e, rng)   # 按 gid 切分，正确
    tr_e_ix = pool_ix[:min(k, len(pool_ix))] if k else []   # ← 取前 k 条记录

切分是对的，但**取标样时按记录取、没按 gid 去重**。A05/A06/A07 是 50 个样品 × 5 次重复扫描
（250 条/实体），若 pool 的前 5 条来自同几个样品，则标称 k=5 的"5 个标样"实际只有 1–2 个独立样品。
后果：k 曲线的上升可能来自"多扫了同一样品几遍"，而不是"多知道了几个样品"。

【本脚本只做审计，不训练】
对每个实体、每个 k，模拟若干 seed 的 split_by_group，统计 pool[:k] 里的**唯一 gid 数**。
输出：标称 k → 实际独立样品数（均值/最小/最大）。

由此得到两个数：
  · 每个 k 档的"有效 k"（= 唯一 gid 数）
  · 该实体的最大可用 k（= 训练侧 gid 总数）

输出：out/exp97a_kshot_audit.json、out/exp97a_kshot_audit.md
"""
from __future__ import annotations
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "..", "out")

from exp71_p6_detection_fix import load_groups_cached
from exp68_p6_cond_scorer import split_by_group

KS = [1, 2, 3, 5, 10, 20, 30]
NSEED = 20

L = []
def w(s=""):
    L.append(s)
    print(s)


G = load_groups_cached()
eids = sorted(G.keys())

w("# exp97a · k-shot 曲线的重复扫描审计")
w()
w("实现位置：`exp81b_p6_benchmark_zoo.py:408` —— `tr_e_ix = pool_ix[:min(k,len(pool_ix))]`。")
w()
w("切分按 gid、**抽样不按 gid**。本表回答：标称给 k 个标样时，实际独立样品有几个。")
w()
w("---")
w()

# ---------------------------------------------------------------- §1 数据结构
w("## §1 每个实体的样品组结构")
w()
w("| 实体 | 条数 | 独立 gid | 条/gid | gid 内重复次数分布 |")
w("|---|---:|---:|---:|---|")
RES = {"struct": {}}
for e in eids:
    d = G[e]
    gid = np.asarray(d["gid"])
    u, c = np.unique(gid, return_counts=True)
    RES["struct"][e] = dict(n=len(gid), n_gid=len(u),
                            per_gid=(len(gid) / len(u)),
                            counts=c.tolist())
    dist = {}
    for x in c:
        dist[int(x)] = dist.get(int(x), 0) + 1
    dstr = ", ".join(f"{v} 个 gid×{k} 条" for k, v in sorted(dist.items())[:4])
    w(f"| {e} | {len(gid)} | {len(u)} | {len(gid)/len(u):.1f} | {dstr} |")
w()

# ---------------------------------------------------------------- §2 有效 k
w("## §2 标称 k → 实际独立样品数（%d seeds 模拟 exp81b 的抽取顺序）" % NSEED)
w()
hdr = "| 实体 | 可用 gid | " + " | ".join(f"k={k}" for k in KS) + " |"
w(hdr)
w("|---|---:|" + "---:|" * len(KS))
RES["eff"] = {}
for e in eids:
    d = G[e]
    X = np.asarray(d["X"], float)
    gid = np.asarray(d["gid"])
    row = []
    cell = {}
    for k in KS:
        vals = []
        for s in range(NSEED):
            rng = np.random.default_rng(s * 7919 + 13)
            te, pool = split_by_group(X, gid, rng)
            take = pool[:min(k, len(pool))] if k else []
            vals.append(len(np.unique(gid[take])) if len(take) else 0)
        mv = float(np.mean(vals))
        cell[k] = dict(mean=mv, min=int(np.min(vals)), max=int(np.max(vals)))
        mark = "" if abs(mv - k) < 0.01 else ("**" if mv < 0.7 * k else "*")
        row.append(f"{mark}{mv:.2f}{mark}")
    RES["eff"][e] = cell
    n_tr_gid = int(np.unique(gid).size * 0.7)
    w(f"| {e} | ~{n_tr_gid} | " + " | ".join(row) + " |")
w()
w("`**` = 实际独立样品数不足标称 k 的 70%；`*` = 不足 100%。")
w()

# ---------------------------------------------------------------- §3 判决
w("## §3 判决")
w()
bad = []
for e in eids:
    for k in KS:
        c = RES["eff"][e][k]
        if c["mean"] < 0.7 * k:
            bad.append((e, k, c["mean"]))
            break
if bad:
    w("| 实体 | 首个失效 k | 该 k 的实际独立样品数 | 结论 |")
    w("|---|---:|---:|---|")
    for e, k, mv in bad:
        w(f"| {e} | {k} | {mv:.2f} | 标称 k 不可信 |")
else:
    w("所有实体的 k 档都基本达到标称值 —— 该实现无重复扫描凑 k 的问题。")
w()

# ---------------------------------------------------------------- §4 光谱同质度
w("## §4 为什么修好 gid 也救不回来：样品之间本身就是同质的")
w()
w("k 的信息量取决于「换一个样品能看到多少新东西」。用 1−cos 衡量：")
w()
w("| 实体 | 同 gid 内 1−cos | 不同 gid 间 1−cos | 同/异 比值 | 判定 |")
w("|---|---:|---:|---:|---|")
RES["homo"] = {}
for e in eids:
    d = G[e]
    X = np.asarray(d["X"], float)
    gid = np.asarray(d["gid"])
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    ug = np.unique(gid)
    within, between = [], []
    for g in ug[:20]:
        ix = np.where(gid == g)[0]
        for i in range(min(4, len(ix))):
            for j in range(i + 1, min(4, len(ix))):
                within.append(1.0 - float(Xn[ix[i]] @ Xn[ix[j]]))
    first = [np.where(gid == g)[0][0] for g in ug[:20]]
    for i in range(len(first)):
        for j in range(i + 1, len(first)):
            between.append(1.0 - float(Xn[first[i]] @ Xn[first[j]]))
    wa, ba = float(np.mean(within)) if within else 0.0, float(np.mean(between))
    ratio = wa / ba if ba > 0 else float("nan")
    if ba < 5e-3:
        vd = "**样品间几乎同质 ⇒ k 无信息量**"
    elif ba < 5e-2:
        vd = "样品间差异弱 ⇒ k 效应小"
    else:
        vd = "样品间有差异 ⇒ k 有效"
    RES["homo"][e] = dict(within=wa, between=ba, ratio=ratio)
    w(f"| {e} | {wa:.5f} | {ba:.5f} | {ratio:.2f} | {vd} |")
w()
w("⇒ 若「不同 gid 间 1−cos」已经接近 0，那么**无论按 gid 抽 k 个还是按记录抽 k 条，"
  "标样都提供不了新信息** —— 这不是抽样实现的 bug，是数据本身没有 50 个互相独立的样品。")
w()

with open(os.path.join(OUT, "exp97a_kshot_audit.json"), "w", encoding="utf-8") as f:
    json.dump(RES, f, ensure_ascii=False, indent=1)
with open(os.path.join(OUT, "exp97a_kshot_audit.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("\n>>> out/exp97a_kshot_audit.{json,md}")
