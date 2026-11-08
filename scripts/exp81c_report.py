r"""exp81c —— exp81（算法动物园 × 先验动物园 × 先验×算法）汇总与出表。

读取 out/exp81_zoo_A01..A07.json（7 个实体并行跑出的分片），合并后输出：
  T1  算法动物园：23 个常见 ML/DL 基线 vs 本文（检出 / 识别 × k=0,5）
  T2  先验动物园：10 种先验 vs 随机码本（同一个岭回归头）
  T3  先验 × 算法：把本文先验装到 6 个普通 ML 算法上，逐算法 Δ_prior
  T3b 平滑头动物园：ridge/pls/krr/gp/knn1/mlp × clm/rand
"""
from __future__ import annotations
import os, sys, json, glob
import numpy as np
from scipy.stats import ttest_rel

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")
CLEAN = ["A01", "A02", "A03", "A04", "A07"]
ALL7 = ["A01", "A02", "A03", "A04", "A05", "A06", "A07"]
OURS = {"same-matrix": "ours-det", "old-heldout": "ours-id-m"}


def load(pattern="exp81_zoo_A*.json"):
    merged = {"B": {}, "P": {}, "C": {}, "O": {}}
    files = sorted(glob.glob(os.path.join(OUT, pattern)))
    assert files, f"未找到 {pattern}"
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        for bk in merged:
            src = d.get(bk, {})
            for k in src:
                for task in src[k]:
                    for arm in src[k][task]:
                        for e, vs in src[k][task][arm].items():
                            merged[bk].setdefault(k, {}).setdefault(
                                task, {}).setdefault(arm, {}).setdefault(e, []).extend(vs)
    print(f"[merge] {len(files)} 个分片: " + ", ".join(os.path.basename(f) for f in files))
    return merged


def per_entity(M, k, task, arm, pool, key="auc"):
    """返回 {entity: 该实体跨 seed/抽样 的均值}。"""
    src = M.get(str(k), {}).get(task, {}).get(arm, {})
    out = {}
    for e in pool:
        v = src.get(e)
        if v:
            out[e] = float(np.mean([q[key] for q in v]))
    return out


def cmp_paired(M, k, task, arm_a, arm_b, pool, key="auc"):
    a, b = per_entity(M, k, task, arm_a, pool, key), per_entity(M, k, task, arm_b, pool, key)
    es = sorted(set(a) & set(b))
    if len(es) < 2:
        return None
    va = np.array([a[e] for e in es]); vb = np.array([b[e] for e in es])
    return dict(da=float(va.mean()), db=float(vb.mean()),
                delta=float((va - vb).mean()), p=float(ttest_rel(va, vb).pvalue),
                n=len(es))


def fmt(v, p=None):
    if v is None:
        return "—"
    s = f"{v:.3f}"
    if p is not None:
        s += f"{'*' if p < 0.05 else ''}" + f" (p={p:.3g})"
    return s


def table_baselines(M, k, task, pool):
    rows = []
    ours = OURS[task]
    base = per_entity(M, k, task, ours, pool)
    arms = sorted(M.get(str(k), {}).get(task, {}).keys())
    for a in arms:
        v = per_entity(M, k, task, a, pool)
        if not v:
            continue
        c = cmp_paired(M, k, task, a, ours, pool)
        rows.append((a, np.mean(list(v.values())), c))
    rows.sort(key=lambda r: -r[1])
    return rows, ours


def main():
    M = load()
    L = []
    w = L.append
    ks = sorted({k for bk in M for k in M[bk]}, key=int)
    w("# exp81 三方横评结果\n")
    w(f"配对单元：逐实体（CLEAN 池 = {CLEAN}，n=5；ALL7 = n=7），配对 t 检验。")
    w("全部数字由 `out/exp81_zoo_A*.json` 重算，未使用任何缓存结论。\n")

    # ---------------- T1 算法动物园 ----------------
    w("## T1 算法动物园（23 个常见 ML/DL 基线，group 协议，AUC）\n")
    for pool, pname in [(CLEAN, "CLEAN(n=5)"), (ALL7, "ALL7(n=7)")]:
        w(f"### 池 = {pname}\n")
        for task, tname in [("same-matrix", "检出 same-matrix"),
                            ("old-heldout", "识别 old-heldout")]:
            for k in ks:
                rows, ours = table_baselines(M["B"], k, task, pool)
                if not rows:
                    continue
                w(f"**{tname}，k={k}**（本文臂 = `{ours}`）\n")
                w("| 排名 | 方法 | AUC | Δ vs 本文 | p |")
                w("|---:|---|---:|---:|---:|")
                for i, (a, v, c) in enumerate(rows, 1):
                    if c is None:
                        w(f"| {i} | `{a}` | {v:.3f} | — | — |")
                    else:
                        w(f"| {i} | `{a}` | {v:.3f} | {c['delta']:+.3f} | {c['p']:.3g} |")
                w("")
    # ---------------- T2 先验动物园 ----------------
    w("\n## T2 先验动物园（同一岭回归头，只换码本；rand384 = 5 次抽样平均）\n")
    for pool, pname in [(CLEAN, "CLEAN(n=5)")]:
        w(f"### 池 = {pname}\n")
        for task, tname in [("same-matrix", "检出"), ("old-heldout", "识别")]:
            suffix = "" if task == "same-matrix" else "-m"
            for k in ks:
                arms = [a for a in M["P"].get(str(k), {}).get(task, {})
                        if a.startswith("P:") and a.endswith(suffix)]
                if not arms:
                    continue
                w(f"**{tname}，k={k}**\n")
                w("| 先验 | AUC | Δ vs rand384 | p |")
                w("|---|---:|---:|---:|")
                rd = "P:rand384" + suffix
                vals = []
                for a in arms:
                    v = per_entity(M["P"], k, task, a, pool)
                    if not v:
                        continue
                    vals.append((a, np.mean(list(v.values()))))
                for a, v in sorted(vals, key=lambda r: -r[1]):
                    if a == rd:
                        w(f"| **{a}（阴性对照）** | {v:.3f} | — | — |")
                    else:
                        c = cmp_paired(M["P"], k, task, a, rd, pool)
                        w(f"| {a} | {v:.3f} | {c['delta']:+.3f} | {c['p']:.3g} |"
                          if c else f"| {a} | {v:.3f} | — | — |")
                w("")
    # ---------------- T3 先验 × 算法 ----------------
    w("\n## T3 把先验装到普通 ML/DL 算法上：逐算法 Δ_prior = clm − rand\n")
    for pool, pname in [(CLEAN, "CLEAN(n=5)")]:
        w(f"### 池 = {pname}（条件化打分器 g([z, S_T])，PCA-128 后拟合）\n")
        for task, tname in [("old-heldout", "识别"), ("same-matrix", "检出")]:
            for k in ks:
                arms = sorted(M["C"].get(str(k), {}).get(task, {}))
                algos = sorted({a.split("|")[0].replace("C:", "")
                                for a in arms})
                if not algos:
                    continue
                w(f"**{tname}，k={k}**\n")
                w("| 算法 | clm(本文先验) | rand(随机码本) | Δ_prior | p |")
                w("|---|---:|---:|---:|---:|")
                for an in algos:
                    ca, ra = f"C:{an}|clm", f"C:{an}|rand384"
                    if task == "old-heldout":
                        ca, ra = ca + "-m", ra + "-m"
                    c = cmp_paired(M["C"], k, task, ca, ra, pool)
                    if c is None:
                        continue
                    w(f"| {an} | {c['da']:.3f} | {c['db']:.3f} | "
                      f"{c['delta']:+.3f} | {c['p']:.3g} |")
                w("")
    # ---------------- T3b 平滑头 ----------------
    w("\n## T3b 平滑头动物园（S→Z 外推头，18 个老类对；目标与任务匹配）\n")
    for pool, pname in [(CLEAN, "CLEAN(n=5)")]:
        w(f"### 池 = {pname}\n")
        for task, tname in [("old-heldout", "识别(center 目标)"),
                            ("same-matrix", "检出(net 目标)")]:
            for k in ks:
                arms = sorted(a for a in M["P"].get(str(k), {}).get(task, {})
                              if a.startswith("H:"))
                if not arms:
                    continue
                w(f"**{tname}，k={k}**\n")
                w("| 头 | clm | rand | Δ_prior | p |")
                w("|---|---:|---:|---:|---:|")
                for a in arms:
                    parts = a.split("|")
                    if len(parts) < 3 or parts[1] != "clm":
                        continue
                    ra = f"{parts[0]}|rand384|{parts[2]}"
                    c = cmp_paired(M["P"], k, task, a, ra, pool)
                    if c is None:
                        continue
                    w(f"| {parts[0][2:]} | {c['da']:.3f} | {c['db']:.3f} | "
                      f"{c['delta']:+.3f} | {c['p']:.3g} |")
                w("")

    path = os.path.join(OUT, "exp81_zoo_report.md")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    print(f"[exp81c] 已写出 {path}")


if __name__ == "__main__":
    main()
