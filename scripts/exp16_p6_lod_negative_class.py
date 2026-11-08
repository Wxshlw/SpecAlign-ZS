# -*- coding: utf-8 -*-
"""
exp16 — 零样本检出限（LOD_zs）的正确定义：候选池含「无掺假」负类

exp15 发现的致命缺陷
--------------------
在 13 类（全为掺杂物实体）候选池下，**纯基质谱（c=0，完全没掺假）的 top-1 已经是 1.000**，
且 margin 在 0–30% 全程不变（Δ<0.003）。这意味着：

    液-液体系的"高准确率"并非检出掺杂物，而是**模型在识别基质**。
    训练时「橄榄油基质的谱」总是配「葵花籽油」文本，于是认出基质 = 认出掺杂物。

根因：**候选池中没有「无掺假」这一负类**。在只含掺杂物的候选池里，
"是否掺假"这个真实检测问题被退化成了"掺了哪种"的多选一，而后者在
「一个基质恰好对应一个掺杂物」时完全等价于基质识别。

本实验的修正
------------
把 6 个纯基质文本并入检索候选（13 实体 + 6 基质 = 19 类）。此时：

- c = 0（纯基质谱）→ 应判为 **M_基质**（无掺假）；判为掺杂物 = **假阳性**
- c 足够大 → 应判为 **目标掺杂物实体**（检出）

于是 LOD_zs 得到**检测限的标准定义**：

    FPR(c=0) = P(判为掺杂物 | 无掺假)
    TPR(c)   = P(判为目标掺杂物 | 掺混浓度 c)
    LOD_zs   = 最小 c 使 TPR ≥ θ 且 TPR 显著高于 FPR（Fisher 精确检验 p<0.01）

θ 取 0.50 / 0.95 两档，与 analytical chemistry 的 LOD/LOQ 惯例对应。
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1          # noqa: E402
import exp08_p6_g1 as E8         # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
DEV = E8.DEV

TARGETS = ["A05", "A07", "HSYRUP", "A06"]
MATRIX_OF = {"A05": "oliveoil", "A07": "juice", "HSYRUP": "honey", "A06": "coffeeR"}
PHASE_OF = {"A05": "液-液", "A07": "液-液", "HSYRUP": "液-液", "A06": "固-固"}

GRID = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0]
ALPHA = 0.01


def fit_linear(X, c):
    A = np.stack([np.ones_like(c), c], axis=1)
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    return coef[0], coef[1]


def pooled_within_std(X, c):
    num = np.zeros(X.shape[1]); den = 0.0
    for lv in np.unique(c):
        m = (c == lv); Xi = X[m]
        if len(Xi) < 2:
            continue
        num = num + ((Xi - Xi.mean(0)) ** 2).sum(0)
        den += len(Xi) - 1
    if den <= 0:
        return np.full(X.shape[1], float(X.std() * 0.01))
    return np.sqrt(num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-syn", type=int, default=120)
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    ent, conc, bg, grid = G1.load_g1_conc()
    eids = sorted(ent.keys())
    mids = sorted(bg.keys())
    variants = E8.build_text_variants_g1(eids, mids)
    keys = eids + [f"M_{m}" for m in mids]
    T_all = np.stack([variants["Original"][k] for k in keys]).astype(np.float32)
    T_all = T_all / np.linalg.norm(T_all, axis=1, keepdims=True)
    T13 = T_all[:len(eids)]                       # 旧口径：仅 13 个掺杂物实体
    n_ent, n_all = len(eids), len(keys)
    print(f"[exp16] device={DEV}  候选池 19 类（13 掺杂物 + 6 纯基质）", flush=True)

    res = {}
    for fold, e_hold in enumerate(TARGETS):
        mat = MATRIX_OF[e_hold]
        mat_ix = n_ent + mids.index(mat)
        true_ix = eids.index(e_hold)

        eids_tr = [e for e in eids if e != e_hold]
        tr_ix = {e: i for i, e in enumerate(eids_tr)}
        parts = [ent[e] for e in eids_tr] + [bg[m] for m in mids]
        ys = np.concatenate([np.full(len(ent[e]), tr_ix[e]) for e in eids_tr] +
                            [np.full(len(bg[m]), len(eids_tr) + j) for j, m in enumerate(mids)])
        Xs = np.concatenate(parts).astype(np.float32)
        ix_seen = [keys.index(e) for e in eids_tr] + [keys.index(f"M_{m}") for m in mids]
        T_seen = T_all[ix_seen]

        # 线性混合模型（与 exp15 同）
        X = np.asarray(ent[e_hold], float)
        c = np.asarray(conc[e_hold], float) / 100.0
        a, b = fit_linear(X, c)
        sigma = pooled_within_std(X, c)
        rng = np.random.default_rng(2026 + fold)
        synth = {}
        for cp in GRID:
            base = a + (cp / 100.0) * b
            synth[cp] = base + rng.normal(0, 1, (args.n_syn, len(base))) * sigma

        tpr = {cp: [] for cp in GRID}      # 判为目标掺杂物
        hit13 = {cp: [] for cp in GRID}    # 旧口径 top-1（13 类）
        to_mat = {cp: [] for cp in GRID}   # 判为任何基质类（"无掺假"）
        wrong_ent = {cp: [] for cp in GRID}
        for seed in seeds:
            net = E8.train_cnn(Xs, ys, T_seen, args.epochs, seed * 1000 + fold)
            with torch.no_grad():
                z = E8.embed(net, np.asarray(np.concatenate(
                    [synth[cp] for cp in GRID]), dtype=np.float32))
            zt = torch.tensor(z, device=DEV)
            sim19 = (torch.nn.functional.normalize(zt, dim=1)
                     @ torch.nn.functional.normalize(
                         torch.tensor(T_all, dtype=torch.float32, device=DEV), dim=1).T)
            sim13 = (torch.nn.functional.normalize(zt, dim=1)
                     @ torch.nn.functional.normalize(
                         torch.tensor(T13, dtype=torch.float32, device=DEV), dim=1).T)
            p19 = sim19.cpu().numpy().argmax(1)
            p13 = sim13.cpu().numpy().argmax(1)
            off = 0
            for cp in GRID:
                blk19 = p19[off:off + args.n_syn]
                blk13 = p13[off:off + args.n_syn]
                tpr[cp].append(float((blk19 == true_ix).mean()))
                hit13[cp].append(float((blk13 == true_ix).mean()))
                to_mat[cp].append(float((blk19 >= n_ent).mean()))
                wrong_ent[cp].append(float(((blk19 < n_ent) & (blk19 != true_ix)).mean()))
                off += args.n_syn
            del net
            if DEV == "cuda":
                torch.cuda.empty_cache()

        res[e_hold] = dict(
            tpr={cp: float(np.mean(tpr[cp])) for cp in GRID},
            hit13={cp: float(np.mean(hit13[cp])) for cp in GRID},
            to_mat={cp: float(np.mean(to_mat[cp])) for cp in GRID},
            wrong_ent={cp: float(np.mean(wrong_ent[cp])) for cp in GRID},
            n=args.n_syn)
        print(f"[exp16] {e_hold}: c=0 → 判纯 {res[e_hold]['to_mat'][0.0]:.3f} / "
              f"假阳 {res[e_hold]['tpr'][0.0]:.3f}；"
              f"c=10% → TPR {res[e_hold]['tpr'][10.0]:.3f}", flush=True)

    # ================= 报告 =================
    L = ["# exp16 — 零样本检出限 LOD_zs（候选池含「无掺假」负类）", "",
         "> **为什么必须这样做**：exp15 证明，在只含掺杂物的 13 类候选池下，",
         "> 纯基质谱（c=0）的 top-1 已为 1.000 且 margin 全程不变 —— ",
         "> 模型识别的是**基质**而非掺杂物。加入 6 个纯基质文本作负类后，",
         "> 「是否掺假」才成为可判定的问题，LOD 才有意义。", "",
         f"> 协议：LOAO；候选池 19 类（13 掺杂物 + 6 纯基质）；{len(seeds)} seeds × "
         f"{args.epochs} epoch；每浓度 {args.n_syn} 条含噪合成谱（噪声=实测组内标准差）。", "",
         "## 1. 关键对照：13 类 vs 19 类候选池（c=0 纯基质谱）", "",
         "| 实体 | 相态 | 13 类 top-1（旧口径，虚高） | 19 类：判为「无掺假」 | 19 类：假阳（判为掺杂物） |",
         "|---|---|---|---|---|"]
    for e in TARGETS:
        r = res[e]
        L.append(f"| {e} | {PHASE_OF[e]} | {r['hit13'][0.0]:.3f} | "
                 f"{r['to_mat'][0.0]:.3f} | **{r['tpr'][0.0]:.3f}** |")
    L += ["", "旧口径把纯基质谱判成「检出掺假」的比例即**假阳性率**；",
          "它正是 exp15 中被误读为「100% 准确」的数字。", ""]

    L += ["## 2. 浓度–检出曲线（19 类候选池）", ""]
    for e in TARGETS:
        r = res[e]
        L += [f"### {e}（{G1.ENTITY_META[e][0]} / {G1.ENTITY_META[e][1]}，{PHASE_OF[e]}）", "",
              "| 浓度% | TPR（判为目标掺杂物） | 判为「无掺假」 | 判为其它掺杂物 | 13 类旧口径 top-1 |",
              "|---|---|---|---|---|"]
        for cp in GRID:
            L.append(f"| {cp} | {r['tpr'][cp]:.3f} | {r['to_mat'][cp]:.3f} | "
                     f"{r['wrong_ent'][cp]:.3f} | {r['hit13'][cp]:.3f} |")
        # LOD：TPR ≥ θ 且显著高于 c=0 假阳率
        fpr = r["tpr"][0.0]
        n0 = int(round(fpr * args.n_syn))
        lod = {}
        for theta in (0.50, 0.95):
            got = None
            for cp in GRID:
                if cp <= 0:
                    continue
                t = r["tpr"][cp]
                if t < theta:
                    continue
                n1 = int(round(t * args.n_syn))
                try:
                    p = stats.fisher_exact([[n1, args.n_syn - n1],
                                            [n0, args.n_syn - n0]], alternative="greater")[1]
                except Exception:
                    p = 1.0
                if p < ALPHA and all(r["tpr"][c2] >= theta - 0.05
                                     for c2 in GRID if c2 >= cp):
                    got = cp
                    break
            lod[theta] = got
        L += ["", f"- 假阳性率 FPR(c=0) = {fpr:.3f}",
              f"- **LOD_zs（TPR≥0.50）** = "
              f"{f'{lod[0.5]:g}%' if lod[0.5] else f'> {GRID[-1]:g}%（未达）'}",
              f"- **LOQ_zs（TPR≥0.95）** = "
              f"{f'{lod[0.95]:g}%' if lod[0.95] else f'> {GRID[-1]:g}%（未达）'}", ""]

    L += ["## 3. 汇总", "",
          "| 实体 | 相态 | FPR(c=0) | LOD_zs (TPR≥50%) | LOQ_zs (TPR≥95%) | 13 类旧口径 top-1 |",
          "|---|---|---|---|---|---|"]
    for e in TARGETS:
        r = res[e]
        fpr = r["tpr"][0.0]
        n0 = int(round(fpr * args.n_syn))
        out = {}
        for theta in (0.50, 0.95):
            got = None
            for cp in GRID:
                if cp <= 0:
                    continue
                t = r["tpr"][cp]
                if t < theta:
                    continue
                n1 = int(round(t * args.n_syn))
                try:
                    p = stats.fisher_exact([[n1, args.n_syn - n1],
                                            [n0, args.n_syn - n0]], alternative="greater")[1]
                except Exception:
                    p = 1.0
                if p < ALPHA and all(r["tpr"][c2] >= theta - 0.05
                                     for c2 in GRID if c2 >= cp):
                    got = cp
                    break
            out[theta] = got
        L.append(f"| {e} | {PHASE_OF[e]} | {fpr:.3f} | "
                 f"**{f'{out[0.5]:g}%' if out[0.5] else f'>{GRID[-1]:g}%'}** | "
                 f"{f'{out[0.95]:g}%' if out[0.95] else f'>{GRID[-1]:g}%'} | "
                 f"{r['hit13'][GRID[-1]]:.3f} |")

    L += ["", "## 4. 结论与对论文的影响", "",
          "1. **旧口径（13 类）的「高准确率」是假象**：纯基质谱即被判为掺假，",
          "   假阳性率见上表。此前 exp08/exp13 报告的液-液 1.000 需按此口径重新解释。",
          "2. **加入「无掺假」负类后**，零样本方法的检出限才第一次被真正测量；",
          "   该数值可与 exp03 的 PLS-R/PLS-DA 参考方法 LOD 直接对比（论文 Table）。",
          "3. **这是方法学警告**：在用掺假数据集评估零样本/检索式模型时，",
          "   候选池必须包含「未掺假」负类，否则评估退化为基质识别，",
          "   准确率会被系统性高估。此结论对领域具有普适价值。",
          "4. **limitation**：低浓度段仍为线性外推（exp15 已验证液-液线性性 cos≥0.995），",
          "   固-固体系（A06）线性叠加不成立，其外推数值不可用。", ""]

    md = "\n".join(L)
    with open(os.path.join(OUT, "exp16_p6_lod_negative_class.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
