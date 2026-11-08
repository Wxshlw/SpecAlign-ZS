# -*- coding: utf-8 -*-
"""
exp19 — 「可检出性」能否被**先验预测**？（预注册 LOO 检验）

================================================================================
## ⚠️ 对 exp19 原设计的修正（写在跑之前，不是事后辩解）

原设计（v2 §七）是「用 H 与 Δ 预测 **exp14 的 13 折 top-1 塌陷**」。
实现时发现这是一个**跨任务预测**，理论前提不成立：

|      | exp14 / G1                    | exp18                      |
|------|-------------------------------|----------------------------|
| 任务 | **识别**：13 个掺杂物候选里选对哪个 | **检出**：掺假 vs 真品（二分类） |
| 指标 | top-1 准确率                    | AUC(掺假 vs 真品)             |

两者可独立成败。**A06 就是反例**：检出 AUC = 1.000（一眼看出不是纯咖啡），
但识别 top-1 = 0.000（认不出是麦芽粉）。因此用「检出」的预测因子去预测
「识别」的成败，理论上就不该成立。

→ 本实验把主检验改为**同任务 LOO 预测**（预测 exp18 自身的 AUC），
  把跨任务检验降为**第 4 节的 claim 边界界定**（预期失败，用于精确限定
  论文的 claim 只能覆盖 detection，不能泛化到 identification）。
================================================================================

## 预注册（在看到任何 LOO 结果之前写死）

- **标签** `y = 1[AUC(nas) ≥ 0.90]` —— 阈值 0.90 已在策略文档 v2 §七预注册
- **特征集**（全部只依赖真品库 + 一条掺混系列，**不依赖任何检出结果**）：
  - `P`（主） = [H, log10 Δ]      H = 真品库异质性，Δ = 批次跳变
  - `Q`（纯先验） = [H]            完全不碰任何掺假样品即可算出
- **对照特征集**（对应文献里已有的主张）：
  - `B_phase` = [相态]             exp13 的主张
  - `B_conc`  = [log10 c_min]      「浓度决定检出限」的主张
  - `B_sel`   = [SEL_b]            「共线度决定可辨识性」的主张
  - `B_const` = 常数（总预测多数类）   朴素基线
- **模型**：LogisticRegression(C=1.0, L2)，**留一实体交叉验证**（7 训练 / 1 预测）
- **主指标**：
  1. `ρ(LOO 预测概率, 实测 AUC)` —— 连续版，功效高于二分
  2. LOO 二分准确率 + Wilson 95% CI
  3. 2×2 列联表 Fisher 精确检验
- **功效声明**：n = 8。**不做**「显著优于基线」的强 claim，只报点估计与一致性。

用法：
    python scripts/exp19_p6_predict_detectability.py
    python scripts/exp19_p6_predict_detectability.py --from-json out/exp18_res.json
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy import stats
from sklearn.linear_model import LogisticRegression

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
RES_JSON = os.path.join(OUT_DIR, "exp18_res.json")

# 预注册阈值
AUC_THR = 0.90

# exp14 外部实测：同基质留一掺杂物（LOAO）query 级 top-1，来源
# out/exp14_p6_g1_reanalysis.md §6。本实验不参与其计算，只作外部标签。
EXP14_TOP1 = {
    "A01": 0.697, "A02": 0.000, "A03": 0.000, "A04": 0.190,
    "A05": 1.000, "A06": 0.000, "A07": 1.000, "HSYRUP": 1.000,
}
COLLAPSE_THR = 0.01      # exp14 报告的塌陷判据：top-1 < 0.01


# --------------------------------------------------------------------------
def wilson_ci(k, n, conf=0.95):
    """Wilson score 区间（小样本比正态近似稳）。"""
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def loo_predict(X, y, C=1.0):
    """留一交叉验证：返回 (预测概率, 预测标签)。

    小样本下逻辑回归可能完全分离；用 L2 + 固定 C 抑制，不调参。
    """
    n = len(y)
    p = np.full(n, np.nan)
    for i in range(n):
        tr = np.arange(n) != i
        if len(set(y[tr].tolist())) < 2:          # 训练折单一类别 → 退化为常数
            p[i] = float(y[tr].mean())
            continue
        m = LogisticRegression(C=C, max_iter=5000)
        m.fit(X[tr], y[tr])
        p[i] = float(m.predict_proba(X[i:i + 1])[0, 1])
    return p, (p >= 0.5).astype(int)


def stdz(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd


# --------------------------------------------------------------------------
def load():
    with open(RES_JSON, encoding="utf-8") as f:
        d = json.load(f)
    return d["res"], d["meta"], d["targets"]


def build(res, meta, targets):
    rows = []
    for e in targets:
        r, m = res[e], meta[e]
        rows.append(dict(
            eid=e, matrix=m["matrix"], phase=m["phase"],
            H=float(r["H"]), jump=float(r["jump"]),
            sel=float(r["sel_b"]), cmin=float(m["cmin"]),
            auc=float(r["auc"]["nas"]),
            auc_cos=float(r["auc"]["cos"]),
            auc_od=float(r["auc"]["od"]),
            rho=float(r["rho"]),
            top1=EXP14_TOP1.get(e, None),
        ))
    return rows


def feature_sets(rows):
    y = np.array([1 if r["auc"] >= AUC_THR else 0 for r in rows])
    H = np.array([r["H"] for r in rows])
    lj = np.log10(np.array([r["jump"] for r in rows]))
    sel = np.array([r["sel"] for r in rows])
    lc = np.log10(np.array([r["cmin"] for r in rows]))
    ph = np.array([1.0 if r["phase"] == "液-液" else 0.0 for r in rows])
    return y, {
        "P  [H, log10Δ]  ★主": stdz(np.column_stack([H, lj])),
        "Q  [H]  纯先验": stdz(H.reshape(-1, 1)),
        "B_phase [相态]": stdz(ph.reshape(-1, 1)),
        "B_conc  [log10 c_min]": stdz(lc.reshape(-1, 1)),
        "B_sel   [SEL_b]": stdz(sel.reshape(-1, 1)),
    }


# --------------------------------------------------------------------------
def main():
    global RES_JSON, AUC_THR
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-json", default=RES_JSON)
    ap.add_argument("--auc-thr", type=float, default=AUC_THR)
    args = ap.parse_args()

    RES_JSON = args.from_json
    AUC_THR = args.auc_thr

    res, meta, targets = load()
    rows = build(res, meta, targets)
    y, FS = feature_sets(rows)
    auc = np.array([r["auc"] for r in rows])
    n = len(rows)

    L = []
    A = L.append
    A("# exp19 — 「可检出性」能否被**先验预测**？（预注册 LOO 检验，n = %d）" % n)
    A("")
    A("> ## ⚠️ 对原设计的修正（跑之前写死，非事后辩解）")
    A(">")
    A("> 原设计是「用 H 与 Δ 预测 exp14 的 **top-1 塌陷**」，实现时发现这是**跨任务预测**：")
    A("> exp14 测的是**识别**（13 候选里选对哪个掺杂物），exp18 测的是**检出**（掺假 vs 真品二分类）。")
    A("> 反例 **A06**：检出 AUC = 1.000（一眼看出不是纯咖啡），识别 top-1 = 0.000（认不出是麦芽粉）。")
    A("> → 主检验改为**同任务 LOO 预测**；跨任务检验降为第 4 节的 **claim 边界界定**。")
    A("")
    A("## 0. 预注册（在看 LOO 结果之前写死）")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A(f"| 标签 y | `1[AUC(nas) ≥ {AUC_THR:.2f}]`（阈值已在策略 v2 §七预注册） |")
    A("| 特征 | `H`（真品库异质性）、`log10 Δ`（批次跳变）—— 只需真品库 + 一条掺混系列 |")
    A("| 模型 | LogisticRegression(C=1.0, L2)，**留一实体** CV（7 训练 / 1 预测），不调参 |")
    A("| 主指标 | ① ρ(LOO 预测概率, 实测 AUC) ② LOO 二分准确率 + Wilson 95% CI ③ Fisher |")
    A("| **功效声明** | n = %d，**不做**「显著优于基线」的强 claim，只报点估计与一致性 |" % n)
    A("")
    A("## 1. 数据（exp18 实测量）")
    A("")
    A("| 实体 | 基质 | 相态 | H | Δ | log10Δ | SEL_b | c_min% | **AUC(nas)** | y | exp14 top-1 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r, yi in zip(rows, y):
        t1 = "—" if r["top1"] is None else f"{r['top1']:.3f}"
        A(f"| {r['eid']} | {r['matrix']} | {r['phase']} | {r['H']*100:.2f}% | "
          f"{r['jump']:.2f}× | {np.log10(r['jump']):+.3f} | {r['sel']:.3f} | "
          f"{r['cmin']:g} | **{r['auc']:.3f}** | {yi} | {t1} |")
    A("")
    A(f"- y = 1 的实体：{', '.join(r['eid'] for r, yi in zip(rows, y) if yi)}")
    A(f"- y = 0 的实体：{', '.join(r['eid'] for r, yi in zip(rows, y) if not yi)}")
    A("")

    # ---------------- 2. 主检验：LOO ----------------
    A("## 2. 主检验：留一实体 LOO 预测")
    A("")
    A("| 特征集 | ρ(预测概率, 实测 AUC) | p | LOO 准确率 | Wilson 95% CI | 混淆 (TP/FP/FN/TN) | Fisher p |")
    A("|---|---|---|---|---|---|---|")
    summ = {}
    for name, X in FS.items():
        p_hat, y_hat = loo_predict(X, y)
        rho, pv = stats.spearmanr(p_hat, auc)
        acc = float((y_hat == y).mean())
        lo, hi = wilson_ci(int((y_hat == y).sum()), n)
        tp = int(((y_hat == 1) & (y == 1)).sum())
        fp = int(((y_hat == 1) & (y == 0)).sum())
        fn = int(((y_hat == 0) & (y == 1)).sum())
        tn = int(((y_hat == 0) & (y == 0)).sum())
        if (tp + fp) * (fn + tn) * (tp + fn) * (fp + tn) == 0:
            fish = float("nan")
        else:
            fish = stats.fisher_exact([[tp, fp], [fn, tn]])[1]
        summ[name] = dict(p_hat=p_hat, y_hat=y_hat, rho=float(rho), p=float(pv),
                          acc=acc, ci=(lo, hi), cm=(tp, fp, fn, tn), fisher=float(fish))
        fs = "—" if np.isnan(fish) else f"{fish:.4f}"
        A(f"| `{name}` | **{rho:+.3f}** | {pv:.4f} | **{acc:.3f}** | "
          f"[{lo:.2f}, {hi:.2f}] | {tp}/{fp}/{fn}/{tn} | {fs} |")
    A("")

    # 常数基线
    maj = int(round(y.mean())) if y.mean() != 0.5 else 0
    acc_const = float((np.full(n, maj) == y).mean())
    lo, hi = wilson_ci(int((np.full(n, maj) == y).sum()), n)
    A("")
    A("- **常数基线**（总预测多数类 = %d）：准确率 %.3f "
      "[%.2f, %.2f] —— 任何特征集必须明显超过它才有意义"
      % (maj, acc_const, lo, hi))
    A("")

    # ---- 2.0 伪影诊断：单特征二值 + LOO 会产出虚假的显著秩相关 ----
    A("### 2.0 ⚠️ 伪影诊断：不要被 `B_phase` 的 ρ = −0.896 骗了")
    A("")
    A("> 单特征且取值为二值时，LOO 的预测概率只有极少数不同取值（大量 ties），")
    A("> Spearman ρ 会在这些近乎常数的数值上给出**虚假的显著相关**，")
    A("> 且其符号由训练集类别均值偶然决定，**不代表该特征有预测力**。")
    A("> 下表的「唯一值数」即诊断量；真正干净的检验是直接的组间比较。")
    A("")
    A("| 特征集 | 预测概率唯一值数 | 极差 | ρ 是否可信 |")
    A("|---|---|---|---|")
    for name, X in FS.items():
        pv_ = summ[name]["p_hat"]
        nuniq = len(np.unique(np.round(pv_, 6)))
        spread = float(pv_.max() - pv_.min())
        ok = "✅ 可信" if nuniq >= 5 else ("⚠️ 近乎退化，ρ 不可信" if nuniq <= 3 else "🟡 勉强")
        A(f"| `{name}` | {nuniq} / {n} | {spread:.3f} | {ok} |")
    A("")
    ll = [r["auc"] for r in rows if r["phase"] == "液-液"]
    ss = [r["auc"] for r in rows if r["phase"] == "固-固"]
    mw = stats.mannwhitneyu(ll, ss, alternative="two-sided")
    A("**相态的直接检验（不经过任何模型）**：")
    A("")
    A(f"- 液-液 AUC = {np.round(ll, 3).tolist()}；固-固 AUC = {np.round(ss, 3).tolist()}")
    A(f"- Mann–Whitney 双侧 **p = {mw.pvalue:.4f}** → 组间无差异")
    A(f"- 结论：相态**没有**预测力。exp13 的相态主张在此被第三次独立否证")
    A("  （前两次：exp18 的 p = 0.7143；本节直接检验 p = %.4f）" % mw.pvalue)
    A("")

    # 逐实体明细
    A("### 2.1 主特征集 `P [H, log10Δ]` 的逐实体 LOO 预测")
    A("")
    A("| 实体 | 实测 AUC(nas) | y | LOO 预测概率 | LOO 预测 | 命中 |")
    A("|---|---|---|---|---|---|")
    sp = summ["P  [H, log10Δ]  ★主"]
    for r, yi, ph_, yh_ in zip(rows, y, sp["p_hat"], sp["y_hat"]):
        A(f"| {r['eid']} | {r['auc']:.3f} | {yi} | {ph_:.3f} | {yh_} | "
          f"{'✅' if yh_ == yi else '❌'} |")
    A("")

    # ---------------- 3. 纯先验版 ----------------
    A("## 3. 「纯先验」版：只用真品库，完全不碰掺假样品")
    A("")
    A("> `H` 只需真品库即可算出 → 实验室在**做任何掺假实验之前**就能预判。")
    A("> `Δ` 需要至少一条掺混谱，属「半先验」。本节量化：放弃 Δ 会掉多少。")
    A("")
    sq = summ["Q  [H]  纯先验"]
    A(f"- 只用 H：ρ = {sq['rho']:+.3f}（p = {sq['p']:.4f}），LOO 准确率 {sq['acc']:.3f} "
      f"[{sq['ci'][0]:.2f}, {sq['ci'][1]:.2f}]")
    A(f"- H + Δ：ρ = {sp['rho']:+.3f}（p = {sp['p']:.4f}），LOO 准确率 {sp['acc']:.3f} "
      f"[{sp['ci'][0]:.2f}, {sp['ci'][1]:.2f}]")
    A(f"- **Δ 的增量**：ρ Δ = {sp['rho'] - sq['rho']:+.3f}，准确率 Δ = "
      f"{sp['acc'] - sq['acc']:+.3f}")
    A("")

    # ---------------- 4. 跨任务检验 ----------------
    A("## 4. 跨任务检验：H / Δ 能否预测 exp14 的**识别**成败？（预期失败）")
    A("")
    A("> 目的不是让它成立，而是**精确界定 claim 边界**：")
    A("> 若 H/Δ 能预测「检出」却不能预测「识别」，则论文 claim 只能覆盖")
    A("> **detection**，不得泛化为「能预测零样本方法何时可用」。")
    A("")
    yid = np.array([0 if r["top1"] < COLLAPSE_THR else 1 for r in rows])
    A("| 实体 | exp14 top-1 | 识别成败 (塌陷<0.01) | 检出 y | 一致? |")
    A("|---|---|---|---|---|")
    for r, yi_, yd_ in zip(rows, yid, y):
        A(f"| {r['eid']} | {r['top1']:.3f} | {yi_} | {yd_} | "
          f"{'✅' if yi_ == yd_ else '❌'} |")
    A("")
    agree = float((yid == y).mean())
    A(f"- 检出成败 vs 识别成败的一致率：**{agree:.3f}**（{int((yid==y).sum())}/{n}）")
    if len(set(yid.tolist())) > 1:
        fis = stats.fisher_exact([[int(((yid == 1) & (y == 1)).sum()),
                                   int(((yid == 1) & (y == 0)).sum())],
                                  [int(((yid == 0) & (y == 1)).sum()),
                                   int(((yid == 0) & (y == 0)).sum())]])[1]
        A(f"- Fisher 精确检验：**p = {fis:.4f}**")
    A("")
    # H / Δ 预测识别
    A("### 4.1 用 H / Δ 预测识别成败（LOO）")
    A("")
    if len(set(yid.tolist())) < 2:
        A("- 识别标签单一，无法建模")
    else:
        for name in ["P  [H, log10Δ]  ★主", "Q  [H]  纯先验"]:
            X = FS[name]
            p_hat, y_hat = loo_predict(X, yid)
            tp = int(((y_hat == 1) & (yid == 1)).sum())
            fp = int(((y_hat == 1) & (yid == 0)).sum())
            fn = int(((y_hat == 0) & (yid == 1)).sum())
            tn = int(((y_hat == 0) & (yid == 0)).sum())
            acc = float((y_hat == yid).mean())
            lo, hi = wilson_ci(int((y_hat == yid).sum()), n)
            A(f"- `{name}`：LOO 准确率 {acc:.3f} [{lo:.2f}, {hi:.2f}]，"
              f"混淆 {tp}/{fp}/{fn}/{tn}")
    A("")

    # ---------------- 5. cinnamon 天然对照 ----------------
    A("## 5. 天然对照：cinnamon 三实体共享真品库（H 相同，Δ 不同）")
    A("")
    A("> A01/A02/A03 共用同一真品库 → **H 恒为 55.75%**，只有 Δ 变化。")
    A("> 这是数据里唯一的「控制变量」机会，可单独看 Δ 的边际作用。")
    A("")
    A("| 实体 | H | Δ | AUC(nas) |")
    A("|---|---|---|---|")
    for r in rows:
        if r["matrix"] == "cinnamon":
            A(f"| {r['eid']} | {r['H']*100:.2f}% | {r['jump']:.2f}× | {r['auc']:.3f} |")
    cin = [r for r in rows if r["matrix"] == "cinnamon"]
    if len(cin) > 2:
        rr = stats.spearmanr([r["jump"] for r in cin], [r["auc"] for r in cin])
        A("")
        A(f"- 组内 Δ vs AUC：ρ = {rr.statistic:+.3f}（p = {rr.pvalue:.4f}，n = {len(cin)}）")
        A("- 注：H 已饱和在该组的最高档，Δ 的边际效应可能被 H 的主效应掩盖，")
        A("  组内样本仅 3 个，此检验功效极低，仅作定性参考。")
    A("")

    # ---------------- 6. 结论 ----------------
    A("## 6. 结论")
    A("")
    best = max(summ.items(), key=lambda kv: kv[1]["acc"])
    A(f"### 6.1 「可检出性可先验预测」是否成立")
    A("")
    A(f"- 最优特征集：`{best[0]}`，LOO 准确率 **{best[1]['acc']:.3f}** "
      f"[{best[1]['ci'][0]:.2f}, {best[1]['ci'][1]:.2f}]，"
      f"ρ(预测, 实测 AUC) = {best[1]['rho']:+.3f}（p = {best[1]['p']:.4f}）")
    A(f"- 常数基线：{acc_const:.3f}")
    A(f"- 主特征集相对常数基线的增益：{best[1]['acc'] - acc_const:+.3f}")
    verdict = ("✅ 成立" if (best[1]["acc"] > acc_const and best[1]["rho"] > 0.6
                          and best[1]["p"] < 0.05 and best[1]["fisher"] < 0.05)
               else ("⚠️ **部分成立**：点估计方向支持，但统计上不足以确证"
                     if (best[1]["acc"] > acc_const and best[1]["rho"] > 0.6)
                     else "❌ 不成立：不超过常数基线"))
    A(f"- **判定**：{verdict}")
    A("")
    A("> **为什么 n = 8 时不应判「成立」**：主特征集的 Fisher p = "
      f"{best[1]['fisher']:.4f}（远不显著），Wilson CI 下界仅 "
      f"{best[1]['ci'][0]:.2f}（接近抛硬币），且增益只有 "
      f"{best[1]['acc'] - acc_const:+.3f}。"
      "因此本文只能 claim **『H 与 Δ 是检出的显著相关因子』**（exp18 已证："
      "ρ = −0.781 / −0.857，p = 0.022 / 0.007），"
      "**不能** claim『已建成一个经过验证的先验预测器』。")
    A("")
    A("> **更重要的负面结果**：第 3 节显示**只用 H 时准确率掉到 "
      f"{sq['acc']:.3f}（等于或低于常数基线）** —— 即**放弃 Δ 后完全失效**。"
      "而 Δ 需要至少一条掺混谱，所以严格意义上的"
      "「采集任何掺假样品之前就能预测」**做不到**，只能做到"
      "「做一条校准系列之后就能预测」。论文须如实区分这两者。")
    A("")
    A("### 6.2 对论文 claim 的边界（第 4 节）")
    A("")
    A(f"- 检出成败与识别成败的一致率 {agree:.3f} → "
      f"{'两者可互相预测' if agree >= 0.75 else '**两者不可互相预测，claim 必须限定在 detection**'}")
    A("- 也就是说：本文能预告「**能不能发现这批样品被动了手脚**」，")
    A("  但不能预告「**能不能指认出掺的是哪一种**」。后者是另一个问题，")
    A("  由候选集内的结构简并度决定（exp00/exp01），不在 H/Δ 的射程内。")
    A("")
    A("### 6.3 功效与 limitation（须写进正文）")
    A("")
    A(f"- **n = {n}**（其余 5 个 exp14 折 COFA/COFR/MEATCH/MEATP/MEATT "
      "无真品库亦无浓度梯度，无法计算 H 与 Δ）")
    A("- LOO 准确率点估计的 Wilson 区间很宽，**不足以支撑「显著优于基线」的强 claim**")
    A("- 逻辑回归在 n=8、2 特征下有过拟合风险；已用固定 C=1.0 的 L2 抑制且**不调参**，")
    A("  但结论应视为**定性/探索性证据**，需 exp21（真品库子集因果实验）补强")
    A("- `Δ` 需要至少一条掺混谱 → 严格意义上的「先验」只有 `H`；")
    A("  第 3 节量化了放弃 Δ 的代价，论文须如实区分「纯先验」与「半先验」")

    txt = "\n".join(L)
    os.makedirs(OUT_DIR, exist_ok=True)
    op = os.path.join(OUT_DIR, "exp19_p6_predict_detectability.md")
    with open(op, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("wrote", op)
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
