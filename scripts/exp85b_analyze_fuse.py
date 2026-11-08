"""exp85b —— 统一口径重算融合实验，并做「按组成距离分层」的机制检验

两处必须统一的统计口径
  1) 单元格定义：一个观测 = (cell, 候选对 T|D)。同一 cell 内同一对可能出现多次（不同随机码本 tag），
     先在该 (cell,pair) 上平均，再对 (cell,pair) 单元格等权平均。
     exp82/exp85 之前的 flat() 混用了「记录等权」，两者可差 0.05，必须统一。
  2) 配对检验的配对单元 = (cell, pair)，不是 pair 名。

机制检验（本脚本的重点）
  融合为什么在 |U|=3 有用、|U|=5 无用？假设：两个信息源强度接近时才互补。
  另测一条更本质的轴：**组成距离**。
     d_comp(T,D) = 1 − cos(physchem_T, physchem_D)      「分子式/元素组成」差异
     d_struct(T,D) = 1 − Tanimoto(ECFP2048_T, ECFP2048_D)「子结构」差异
     逐对回归  Δ = auc_dis(CLM) − auc_dis(physchem)  ~  d_comp
  预期：d_comp 越小（组成越接近，physchem 越失效），Δ 越正（CLM 相对越强）。
  若该斜率为负且显著，则「CLM 的价值在组成信息失效处」成立 —— 这才是可写、可预测的机制。
"""
from __future__ import annotations
import os, sys, json, itertools
import numpy as np
from scipy.stats import ttest_rel, linregress, pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")


def cells_of(d, arm, kind="dis", field="auc_dis", bucket="A"):
    """{(cell, key): value}；同 (cell,key) 内多次记录先平均。"""
    b = d.get(bucket)
    if not isinstance(b, dict) or arm not in b:
        b = d                      # exp85 把臂直接放在顶层（无 bucket 层）
    v = b.get(arm, {}).get(kind, {})
    out = {}
    for key, lst in v.items():
        tmp = {}
        for q in lst:
            tmp.setdefault(q["tag"], []).append(q[field])
        for tag, vs in tmp.items():
            out[(tag, key)] = float(np.mean(vs))
    return out


def mean_of(d, arm, kind="dis", field="auc_dis", bucket="A"):
    c = cells_of(d, arm, kind, field, bucket)
    return float(np.mean(list(c.values()))) if c else None


def paired(d, a1, a2, kind="dis", field="auc_dis", bucket="A"):
    A, B = cells_of(d, a1, kind, field, bucket), cells_of(d, a2, kind, field, bucket)
    ks = sorted(set(A) & set(B))
    if len(ks) < 3:
        return None
    x = np.array([A[k] for k in ks]); y = np.array([B[k] for k in ks])
    return dict(a=x.mean(), b=y.mean(), dd=float((x - y).mean()),
                p=float(ttest_rel(x, y).pvalue), n=len(ks),
                win=float(np.mean(x > y)))


def load(f):
    p = os.path.join(OUT, f)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def sec1_consistency():
    print("=" * 78)
    print("一、口径核验：exp85 的 single:physchem 必须与 exp82 的 physchem 同口径一致")
    print("=" * 78)
    pairs = [("exp82_dis.json", "exp85_nhu3.json", "|U|=3"),
             ("exp82_nhu5.json", "exp85_nhu5.json", "|U|=5")]
    for f82, f85, lbl in pairs:
        d82, d85 = load(f82), load(f85)
        if d82 is None or d85 is None:
            print(f"  {lbl} 缺文件，跳过"); continue
        a = mean_of(d82, "A:physchem|lam0.0001")
        b = mean_of(d85, "single:physchem")
        c = mean_of(d85, "score-g|a0")
        print(f"  {lbl}: exp82 physchem={a:.4f} | exp85 single={b:.4f} | score-g|a0={c:.4f}"
              f"   Δ(exp85−exp82)={b - a:+.4f}")
    print()


def sec2_fuse():
    print("=" * 78)
    print("二、融合 α 曲线（单元格口径，配对 vs physchem 单打）")
    print("=" * 78)
    for f, lbl in [("exp85_nhu3.json", "|U|=3"), ("exp85_nhu5.json", "|U|=5")]:
        d = load(f)
        if d is None:
            continue
        print(f"\n### {lbl}")
        singles = {p: mean_of(d, f"single:{p}") for p in
                   ["physchem", "clm", "charbag", "ecfp2048"]}
        t1 = {p: mean_of(d, f"single:{p}", "top1", "top1") for p in singles}
        print("  单打: " + " | ".join(
            f"{p} {singles[p]:.4f}/{t1[p]:.3f}" for p in singles))
        base = cells_of(d, "single:physchem")
        clmC = cells_of(d, "single:clm")
        print(f"  {'arm':20s} {'auc_dis':>8s} {'top1':>7s} {'Δvs最佳单打':>11s} {'p':>8s} {'胜率':>5s}")
        best_single = max(singles, key=lambda k: singles[k])
        for mode in ["concat"]:
            c = paired(d, mode, f"single:{best_single}")
            if c:
                print(f"  {mode:20s} {mean_of(d, mode):8.4f} "
                      f"{mean_of(d, mode, 'top1', 'top1'):7.3f} "
                      f"{c['dd']:+11.3f} {c['p']:8.3g} {c['win']:5.2f}")
        for fam in ["score-g", "score-c", "feat"]:
            rows = []
            for al in [round(x * .1, 1) for x in range(11)]:
                arm = f"{fam}|a{al:g}"
                a = mean_of(d, arm)
                if a is None:
                    continue
                c = paired(d, arm, f"single:{best_single}")
                rows.append((al, a, mean_of(d, arm, "top1", "top1"), c))
            if not rows:
                continue
            pk = max(rows, key=lambda r: r[1])
            for al, a, tt, c in rows:
                mark = "  <-- 峰值" if al == pk[0] else ""
                print(f"  {fam + '|a' + format(al, 'g'):20s} {a:8.4f} "
                      f"{(tt if tt is not None else float('nan')):7.3f} "
                      f"{c['dd']:+11.3f} {c['p']:8.3g} {c['win']:5.2f}{mark}")
        print()


def sec3_mechanism():
    print("=" * 78)
    print("三、机制：融合增益是否取决于「两个信息源强度差」")
    print("=" * 78)
    for f, lbl in [("exp85_nhu3.json", "|U|=3"), ("exp85_nhu5.json", "|U|=5")]:
        d = load(f)
        if d is None:
            continue
        ph = mean_of(d, "single:physchem"); cl = mean_of(d, "single:clm")
        gap = ph - cl
        improved = []
        for fam in ["score-g", "score-c", "feat"]:
            best = max((mean_of(d, f"{fam}|a{al:g}"), al)
                       for al in np.arange(0, 1.01, .1)
                       if mean_of(d, f"{fam}|a{al:g}") is not None)
            improved.append((fam, best[0], best[1]))
        print(f"  {lbl}: physchem={ph:.4f} clm={cl:.4f} 强度差={gap:+.4f} | "
              + " ; ".join(f"{f}峰值 {v:.4f}@α={a:g}" for f, v, a in improved))
        print(f"     → 融合峰值相对最强单打 "
              f"{max(ph, cl):.4f} 增益 {max(v for _, v, _ in improved) - max(ph, cl):+.4f}")
    print()


def sec4_distance():
    print("=" * 78)
    print("四、核心机制检验：按「组成距离 / 结构距离」分层，CLM 是否在组成信息失效处反超")
    print("=" * 78)
    z = np.load(os.path.join(OUT, "exp81_prior_cache.npz"), allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    PH = {k: np.asarray(z["physchem"][i], float) for i, k in enumerate(keys)}
    EF = {k: np.asarray(z["ecfp2048"][i], float) for i, k in enumerate(keys)}

    def d_cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(1 - (a @ b) / max(1e-12, na * nb))

    def d_tan(a, b):
        mn, mx = np.minimum(a, b).sum(), np.maximum(a, b).sum()
        return float(1 - mn / max(1e-12, mx))

    for f, lbl in [("exp82_dis.json", "|U|=3"), ("exp82_nhu5.json", "|U|=5"),
                   ("exp85_nhu3.json", "|U|=3(85)"), ("exp85_nhu5.json", "|U|=5(85)")]:
        d = load(f)
        if d is None:
            continue
        a_clm = mean_of(d, "A:clm|lam0.0001") if f.startswith("exp82") \
            else mean_of(d, "single:clm")
        a_ph = mean_of(d, "A:physchem|lam0.0001") if f.startswith("exp82") \
            else mean_of(d, "single:physchem")
        if a_clm is None or a_ph is None:
            continue
        c_clm = (cells_of(d, "A:clm|lam0.0001") if f.startswith("exp82")
                 else cells_of(d, "single:clm"))
        c_ph = (cells_of(d, "A:physchem|lam0.0001") if f.startswith("exp82")
                else cells_of(d, "single:physchem"))
        # 按 pair 名聚合（跨 cell 平均），与距离一一对应
        bypair = {}
        for (tag, key) in c_clm:
            if (tag, key) not in c_ph:
                continue
            T, D = key.split("|")
            bypair.setdefault(key, []).append(c_clm[(tag, key)] - c_ph[(tag, key)])
        rows = []
        for key, ds in bypair.items():
            T, D = key.split("|")
            if T not in PH or D not in PH:
                continue
            rows.append((key, d_cos(PH[T], PH[D]), d_tan(EF[T], EF[D]),
                         float(np.mean(ds)), len(ds)))
        if len(rows) < 5:
            print(f"  {lbl}: 候选对太少({len(rows)})，跳过"); continue
        dc = np.array([r[1] for r in rows]); ds = np.array([r[2] for r in rows])
        dd = np.array([r[3] for r in rows])
        print(f"\n  ### {lbl}  本文CLM={a_clm:.4f} physchem={a_ph:.4f} Δ={a_clm - a_ph:+.4f}"
              f"  (n_pair={len(rows)})")
        for nm, x in [("组成距离 d_comp", dc), ("结构距离 d_struct", ds)]:
            lr = linregress(x, dd)
            r, pr = pearsonr(x, dd)
            print(f"    Δ ~ {nm}: 斜率={lr.slope:+.4f} (p={lr.pvalue:.3g}), "
                  f"r={r:+.3f} (p={pr:.3g}), R²={lr.rvalue ** 2:.3f}")
        # 按组成距离中位分层
        med = float(np.median(dc))
        near = dd[dc <= med]; far = dd[dc > med]
        print(f"    组成接近 (d_comp≤{med:.3f}, n={len(near)}): Δ={near.mean():+.4f}")
        print(f"    组成远离 (d_comp> {med:.3f}, n={len(far)}): Δ={far.mean():+.4f}")
        if len(near) >= 3 and len(far) >= 3:
            print(f"    分层差 = {near.mean() - far.mean():+.4f}")
    print()


if __name__ == "__main__":
    sec1_consistency()
    sec2_fuse()
    sec3_mechanism()
    sec4_distance()
