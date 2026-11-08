"""exp57 —— 论文主表生成器（纯 CPU，不训练）

把已经跑完的实验（exp49/50/51/53）合并成论文可直接用的表，并补上：
  1. **bootstrap 置信区间**：n=13 实体太少，点估计不足以支撑 claim，
     对 macro 指标做 2000 次实体级重抽的 percentile CI。
  2. **配对检验**：以实体为配对单元（n=13），不是以 seed 为单元。
  3. **Hedges' g** 效应量：p 值随 n 变化，效应量才说明"差多少"。

输出：out/paper_tables.md
"""
from __future__ import annotations
import os, json
import numpy as np
from scipy.stats import ttest_rel, ttest_1samp

OUT = r"D:/2026sci论文实验设计/论文6/out"
RNG = np.random.default_rng(20261102)
NBOOT = 2000


def load(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def per_entity(res, tag, arm, met):
    out = {}
    for e in res["meta"]["eids"]:
        v = [x[met] for x in res["acc"].get(tag, {}).get(arm, {}).get(e, [])
             if x.get(met) is not None]
        if v:
            out[e] = float(np.mean(v))
    return out


def boot_ci(x, nboot=NBOOT):
    x = np.asarray(x, float)
    if len(x) < 3:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(x), (nboot, len(x)))
    bs = x[idx].mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def hedges_g(d):
    d = np.asarray(d, float)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return float("nan")
    g = d.mean() / d.std(ddof=1)
    return float(g * (1 - 3 / (4 * n - 1)))     # 小样本校正


def cmp_rows(x_d, name, vs=0.5):
    """给一个 {实体: 值} 返回一行统计"""
    x = np.array(list(x_d.values()))
    lo, hi = boot_ci(x)
    p = ttest_1samp(x, vs).pvalue
    nm_disp = name if name.startswith("**") else f"`{name}`"
    return (f"| {nm_disp} | {x.mean():.3f} | [{lo:.3f}, {hi:.3f}] | "
            f"{x.min():.3f} | {x.std():.3f} | {p:.4f} | "
            f"{hedges_g(x - vs):+.2f} |")


def main():
    L = []
    w = L.append
    w("# 论文主表（exp57 汇总，实体级 bootstrap CI，n=13）\n")
    w("配对单元 = 13 个实体；CI = 2000 次实体级重抽的 percentile 区间；"
      "效应量 = Hedges' g（相对随机水平 0.5）。\n")

    # ---------------- 表 1：k=0 零样本筛查（主战场）----------------
    r49 = load("exp49_clean.json")
    r53 = load("exp53_zsl_k0.json")
    if r49 and r53:
        w("\n## 表 1｜k=0（一条纯品谱都没有）：定向筛查 AUC\n")
        w("| 方法 | macro | 95% CI | 最差实体 | 跨实体SD | vs 0.5 (p) | g |")
        w("|---|---|---|---|---|---|---|")
        rows = []
        # 本文
        rows.append(("**本文 ridge**", per_entity(r53, "0", "ridge-c", "auc")))
        # 真 ZSL 基线
        for a, nm in (("conse-c", "ConSE"), ("clip", "CLIP 式对比对齐"),
                      ("ale-c", "ALE"), ("eszsl-c", "ESZSL"), ("devise-c", "DEVISE")):
            d = per_entity(r53, "0", a, "auc")
            if d:
                rows.append((nm, d))
        # 经典外部基线
        for a, nm in (("svm-rbf", "SVM-RBF"), ("rf", "Random Forest"),
                      ("knn-cos", "谱库检索 kNN"), ("centroid-cos", "谱库检索 类中心"),
                      ("plsda", "PLS-DA")):
            d = per_entity(r49, "0", a, "auc")
            if d:
                rows.append((nm, d))
        # 零信息对照
        for a, nm in (("ridge-rand", "本文+零信息码本"), ("clip-rand", "CLIP+零信息码本")):
            d = per_entity(r53, "0", a, "auc")
            if d:
                rows.append((nm, d))
        for nm, d in rows:
            w(cmp_rows(d, nm))

        w("\n### 配对检验（本文 ridge − 各方法，n=13 实体）\n")
        w("| 对比 | Δ | p | Hedges' g |")
        w("|---|---|---|---|")
        ours = per_entity(r53, "0", "ridge-c", "auc")
        eids = sorted(ours)
        x = np.array([ours[e] for e in eids])
        pairs = []
        for a, nm in (("conse-c", "ConSE"), ("clip", "CLIP"), ("ale-c", "ALE"),
                      ("eszsl-c", "ESZSL"), ("devise-c", "DEVISE"),
                      ("ridge-rand", "零信息码本")):
            d = per_entity(r53, "0", a, "auc")
            if len(d) == len(eids):
                pairs.append((nm, np.array([d[e] for e in eids])))
        for a, nm in (("svm-rbf", "SVM-RBF"), ("rf", "RF"), ("knn-cos", "谱库 kNN"),
                      ("centroid-cos", "谱库 类中心"), ("plsda", "PLS-DA")):
            d = per_entity(r49, "0", a, "auc")
            if len(d) == len(eids):
                pairs.append((nm, np.array([d[e] for e in eids])))
        for nm, y in sorted(pairs, key=lambda z: -np.mean(x - z[1])):
            d = x - y
            w(f"| vs {nm} | {d.mean():+.3f} | {ttest_rel(x, y).pvalue:.4f} | "
              f"{hedges_g(d):+.2f} |")

    # ---------------- 表 2：k>0 ----------------
    if r49:
        w("\n## 表 2｜有少量纯品谱后（k=1/5/20）：macro top-1（识别）\n")
        w("| k | 本文最优臂 | 最强外部基线 | Δ | p |")
        w("|---|---|---|---|---|")
        EXT = ["plsda", "svm-rbf", "rf", "knn-cos", "centroid-cos"]
        for t in ("1", "5", "20"):
            if t not in r49["acc"]:
                continue
            ours = {}
            for a in ("ridge", "ncm", "hybrid"):
                d = per_entity(r49, t, a, "top1")
                if d:
                    ours[a] = d
            if not ours:
                continue
            ba = max(ours, key=lambda a: np.mean(list(ours[a].values())))
            ext = {a: per_entity(r49, t, a, "top1") for a in EXT}
            ext = {a: d for a, d in ext.items() if d}
            be = max(ext, key=lambda a: np.mean(list(ext[a].values())))
            eids = sorted(ours[ba])
            x = np.array([ours[ba][e] for e in eids])
            y = np.array([ext[be][e] for e in eids])
            w(f"| {t} | {ba} {x.mean():.3f} | {be} {y.mean():.3f} | "
              f"{np.mean(x - y):+.3f} | {ttest_rel(x, y).pvalue:.4f} |")

    # ---------------- 表 3：定向 vs 通用（exp50/51）----------------
    r51 = load("exp51_symmetry.json")
    if r51:
        w("\n## 表 3｜定向筛查 vs 通用异常检测：对称性检验（exp51，n=78 无序对）\n")
        w("通用 detectors 的评分**不依赖怀疑对象** ⇒ fwd + bwd ≡ 1.0 ⇒ min ≤ 0.5（数学恒等）。\n")
        w("| 方法 | min_dir | sum_dir | vs 0.5 (p) |")
        w("|---|---|---|---|")
        for a, nm in (("ridge", "**本文**"), ("rf", "RF"), ("svm-rbf", "SVM-RBF"),
                      ("knn-cos", "谱库 kNN"), ("centroid-cos", "谱库 类中心")):
            v = r51["acc"].get(a, {})
            if not v:
                continue
            mn = np.array([np.mean([d["min_dir"] for d in ds]) for ds in v.values()])
            sm = np.array([np.mean([d["sum_dir"] for d in ds]) for ds in v.values()])
            w(f"| {nm} `{a}` | {mn.mean():.3f} | {sm.mean():.3f} | "
              f"{ttest_1samp(mn, 0.5).pvalue:.2e} |")

    p = os.path.join(OUT, "paper_tables.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[exp57] 已写出 {p}")


if __name__ == "__main__":
    main()
