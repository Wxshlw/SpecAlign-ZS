"""exp64 —— 零样本筛查的**可达性**能否由化学先验预先预测？

=============================================================================
动机（来自 exp63 的意外分层）
=============================================================================
exp63 `same-matrix` 口径（真筛查语义）下本文方法 tpr5 极端分化：
    A06(coffeeR) 0.974   A04(turmeric) 0.793   A05(oliveoil) 0.493
    A02 0.163   HSYRUP(honey) 0.111   A03 0.102   A01 0.035   A07(juice) 0.000

若这种分化**可由化学先验预测**，则「输给 RF」就从缺陷转为**可解释的性质**：
    ⇒ 零样本筛查的成败，在拿到任何光谱之前就能从 SMILES 判断。
    这正是 exp19（可检测性预测）与 exp23（ρ(z_off,AUC)=+0.9286）的第三块拼图，
    且是**本文独有的**：RF/SVM 的能力来自"见过这个类"，无法预先预测。

=============================================================================
设计（纯分析，重跑模型，复用 exp63 切分）
=============================================================================
对每个实体 T：
  化学侧（**只用 SMILES 库，不需要任何光谱样本**）
    d_chem   = 1 − cos(CLM_T, CLM_M(T))          掺杂物与基质的化学差异度
    d_nn     = 1 − cos(CLM_T, CLM_NN(T))         与最近邻掺杂物（老类）的差异度
    sep      = d_chem − d_nn                     双重分离裕度
  光谱侧（需要样本，仅作对照）
    z_off    exp23 的净信号强度（若可用）
  结果侧（exp63 `same-matrix` 口径）
    tpr5 / auc / pauc10

判据
  H64a：Spearman ρ(d_chem, tpr5) > 0 且显著
  H64b：**对比组** ρ(d_chem, rf_tpr5) 明显更小 ⇒ 预测力是化学先验独有，
        不是任务难度本身的可预测性
  H64c：加入 near-duplicate 稳健性（留一实体 jackknife）
"""
from __future__ import annotations
import os, sys, json, argparse, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = r"D:/2026sci论文实验设计/论文6/out"
EID2MID = {"A01": "cinnamon", "A02": "cinnamon", "A03": "cinnamon",
           "A04": "turmeric", "A05": "oliveoil", "A06": "coffeeR",
           "A07": "juice", "HSYRUP": "honey"}


def nrm(A):
    return A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="exp63_screen.json")
    args = ap.parse_args()

    with open(os.path.join(OUT, args.json), encoding="utf-8") as f:
        res = json.load(f)
    eids = res["meta"]["eids"]
    zc = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    T_all = np.asarray(zc["clm"], float)
    keys = [str(k) for k in zc["keys"]] if "keys" in zc.files else None
    if keys is None:
        import p6_g1_data as G1
        ent, conc, bg, _ = G1.load_g1_conc()
        keys = sorted(ent.keys()) + [f"M_{m}" for m in sorted(bg.keys())]
    ki = {k: i for i, k in enumerate(keys)}
    Tn = nrm(T_all)

    rows = []
    for e in eids:
        mid = EID2MID[e]
        cos_mid = float(Tn[ki[e]] @ Tn[ki[f"M_{mid}"]])
        sims = {o: float(Tn[ki[e]] @ Tn[ki[o]])
                for o in eids if o != e}
        nn_e = max(sims, key=sims.get)
        rec = dict(eid=e, mid=mid,
                   d_chem=1.0 - cos_mid,
                   d_nn=1.0 - sims[nn_e],
                   nn=nn_e, cos_mid=cos_mid,
                   sep=(1.0 - cos_mid) - (1.0 - sims[nn_e]))
        for arm, tag in (("margin-lse", "ours"), ("raw", "raw"),
                         ("rf", "rf"), ("svm-rbf", "svm"), ("rand", "rand")):
            for met in ("tpr5", "auc", "pauc10", "tpr20"):
                vs = [d["all"][met] for d in res["same-matrix"].get(arm, {}).get(e, [])]
                rec[f"{tag}_{met}"] = float(np.mean(vs)) if vs else float("nan")
        # 也存 old-heldout / all-matrix 便于稳健性检查
        for nm in ("old-heldout", "all-matrix"):
            vs = [d["all"]["tpr5"] for d in res[nm].get("margin-lse", {}).get(e, [])]
            rec[f"ours_{nm[:3]}_tpr5"] = float(np.mean(vs)) if vs else float("nan")
        rows.append(rec)

    from scipy.stats import spearmanr
    L = []
    w = L.append
    w("# exp64 —— 零样本筛查的可达性能否由化学先验预先预测？\n")
    w("化学侧只用 SMILES 库（`d_chem` = 1 − cos(CLM_T, CLM_基质)）；"
      "结果侧取 exp63 `same-matrix` 真筛查口径。\n")
    w("| 实体 T | 基质 | d_chem | 最近邻 | d_nn | 裕度 | 本文 tpr5 | 本文 auc | rf tpr5 | rf auc | rand tpr5 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: -x["d_chem"]):
        w(f"| {r['eid']} | {r['mid']} | {r['d_chem']:.3f} | {r['nn']} | "
          f"{r['d_nn']:.3f} | {r['sep']:+.3f} | {r['ours_tpr5']:.3f} | "
          f"{r['ours_auc']:.3f} | {r['rf_tpr5']:.3f} | {r['rf_auc']:.3f} | "
          f"{r['rand_tpr5']:.3f} |")

    w("\n## 判据\n")
    xs = np.array([r["d_chem"] for r in rows])
    for tgt, lab in (("ours_tpr5", "本文 margin-lse tpr5"),
                     ("ours_auc", "本文 margin-lse auc"),
                     ("ours_pauc10", "本文 margin-lse pauc10"),
                     ("rf_tpr5", "对照：rf tpr5"),
                     ("rf_auc", "对照：rf auc"),
                     ("svm_tpr5", "对照：svm tpr5"),
                     ("rand_tpr5", "对照：rand tpr5")):
        ys = np.array([r[tgt] for r in rows])
        ok = np.isfinite(xs) & np.isfinite(ys)
        rho, p = spearmanr(xs[ok], ys[ok])
        w(f"- ρ(d_chem, {lab}) = **{rho:+.3f}** (p={p:.4g}, n={ok.sum()})")

    # jackknife 稳健性
    w("\n### 留一实体稳健性（本文 tpr5）\n")
    ys = np.array([r["ours_tpr5"] for r in rows])
    rs = []
    for i in range(len(rows)):
        m = np.ones(len(rows), bool); m[i] = False
        rs.append(spearmanr(xs[m], ys[m]).statistic)
    rs = np.array(rs)
    w(f"- ρ 范围 [{rs.min():+.3f}, {rs.max():+.3f}]，中位 {np.median(rs):+.3f}，"
      f"全部同号 = {bool((rs > 0).all() or (rs < 0).all())}")
    worst = int(np.argmin(rs))
    w(f"- 最不利的留一：排除 **{rows[worst]['eid']}** 时 ρ={rs[worst]:+.3f}")

    # 分组对照
    hi = [r for r in rows if r["d_chem"] >= np.median(xs)]
    lo = [r for r in rows if r["d_chem"] < np.median(xs)]
    w(f"\n### 化学差异度高低两组（中位数切分）\n")
    w("| 组 | n | 本文 tpr5 | 本文 auc | rf tpr5 | 差值(本文−rf) |")
    w("|---|---|---|---|---|---|")
    for nm, g in (("高 d_chem", hi), ("低 d_chem", lo)):
        a = np.mean([r["ours_tpr5"] for r in g])
        b = np.mean([r["ours_auc"] for r in g])
        c = np.mean([r["rf_tpr5"] for r in g])
        w(f"| {nm} | {len(g)} | {a:.3f} | {b:.3f} | {c:.3f} | {a - c:+.3f} |")

    with open(os.path.join(OUT, "exp64_p6_predictive.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    with open(os.path.join(OUT, "exp64_predictive.json"), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("[exp64] 已写出 out/exp64_p6_predictive.md")


if __name__ == "__main__":
    main()
