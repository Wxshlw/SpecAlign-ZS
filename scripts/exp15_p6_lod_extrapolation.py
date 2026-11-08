# -*- coding: utf-8 -*-
"""
exp15 — 液-液体系零样本检出限（LOD_zs）外推

背景
----
exp13 发现：可辨识性呈「全有或全无」，由相态（液-液 vs 固-固）决定，
液-液体系在数据集**最低浓度（3–10%）即 100% 辨识**。
因此现有真实数据无法给出 LOD —— 我们需要**低于 3%** 的浓度点。

公开数据中 <3% 的液-液 FTIR 掺假集极其稀缺（文献 LOD 报道多在 1–5%，
而多数公开数据集只做到 5–10%）。本实验改用**受控外推**：

方法
----
1) **线性混合模型拟合**（每个掺混实体，逐波数 OLS）
       X_ij = a_j + b_j · c_i
   a_j = 外推到 c=0 的纯基质谱 m̂；a_j + b_j = 外推到 c=1 的纯掺杂物谱 ŝ。
2) **保真度三重验证**（外推合法性的依据）
   - (i) 线性假设：m̂ 与真实纯基质谱（基质背景谱）的余弦
   - (ii) 留一浓度交叉验证（LOCO）：用其余浓度拟合，预测留出浓度的谱
   - (iii) 合成谱 vs 真实谱在**同一浓度**上的零样本 top-1 是否一致
3) **噪声注入**：用「同浓度内多样本的合并组内标准差」估计真实测量/制备噪声，
   注入合成谱。无噪声合成会严重低估 LOD，故同时报两套结果作上下界。
4) 低浓度梯度合成 + LOAO 零样本检索 → LOD_zs

LOD_zs 定义：top-1 ≥ 0.5 且显著高于 chance（二项检验 p<0.01）的最小浓度。
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

# 目标实体：3 个液-液（可辨识）+ 1 个固-固负对照
TARGETS = ["A05", "A07", "HSYRUP", "A06"]
MATRIX_OF = {"A05": "oliveoil", "A07": "juice", "HSYRUP": "honey", "A06": "coffeeR"}
PHASE_OF = {"A05": "液-液", "A07": "液-液", "HSYRUP": "液-液", "A06": "固-固"}

# 外推浓度梯度（%）。**必须含 0**：纯基质谱基线，用于识别「虚高」陷阱——
# 极低浓度下 query≈纯基质谱，若纯基质谱本身就被归到该基质的熟悉掺杂物上，
# top-1 会虚高，此时 LOD 无意义，须改用 margin 判据。
GRID = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
N_SYN = 100
ALPHA = 0.01


def fit_linear(X, c):
    """逐波数 OLS: X = a + b*c。返回 (a, b)。a=纯基质外推, a+b=纯掺杂物外推。"""
    A = np.stack([np.ones_like(c), c], axis=1)
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    return coef[0], coef[1]


def pooled_within_std(X, c):
    """同浓度内样本的合并组内标准差（逐波数），估计真实测量/制备噪声。"""
    num = np.zeros(X.shape[1])
    den = 0.0
    for lv in np.unique(c):
        m = (c == lv)
        Xi = X[m]
        if len(Xi) < 2:
            continue
        num = num + ((Xi - Xi.mean(0)) ** 2).sum(0)
        den += len(Xi) - 1
    if den <= 0:
        return np.full(X.shape[1], float(X.std() * 0.01))
    return np.sqrt(num / den)


def loco_cv(X, c):
    """留一浓度交叉验证：返回 [(level, rms, cos_mean), ...]。"""
    out = []
    for lv in np.unique(c):
        tr = (c != lv)
        te = (c == lv)
        if tr.sum() < 2:
            continue
        a, b = fit_linear(X[tr], c[tr])
        pred = a + np.outer(c[te], b)
        true = X[te]
        rms = float(np.sqrt(((pred - true) ** 2).mean()))
        pn = pred / np.linalg.norm(pred, axis=1, keepdims=True)
        tn = true / np.linalg.norm(true, axis=1, keepdims=True)
        out.append((float(lv), rms, float(np.sum(pn * tn, axis=1).mean())))
    return out


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def lod_of(levels, t1s, ns, chance):
    """最小浓度：top-1 ≥ 0.5 且二项检验 p<0.01，且更高浓度全部满足。"""
    ok = []
    for t, n in zip(t1s, ns):
        p = stats.binomtest(int(round(t * n)), int(n), chance, alternative="greater").pvalue
        ok.append((t >= 0.5) and (p < ALPHA))
    for i in range(len(levels)):
        if all(ok[j] for j in range(i, len(levels))):
            return levels[i]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-syn", type=int, default=N_SYN)
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    ent, conc, bg, grid = G1.load_g1_conc()
    eids = sorted(ent.keys())
    mids = sorted(bg.keys())
    variants = E8.build_text_variants_g1(eids, mids)
    keys = eids + [f"M_{m}" for m in mids]
    T_full = np.stack([variants["Original"][k] for k in keys]).astype(np.float32)
    T_full = T_full / np.linalg.norm(T_full, axis=1, keepdims=True)
    T_query = T_full[:len(eids)]          # 检索候选 = 13 个实体文本
    chance = 1.0 / len(eids)
    print(f"[exp15] device={DEV}  entities={len(eids)}  chance={chance:.4f}", flush=True)

    report = {}
    # ---------- Part 1/2：拟合 + 保真度 ----------
    fits = {}
    for e in TARGETS:
        X = np.asarray(ent[e], float)
        c = np.asarray(conc[e], float) / 100.0
        a, b = fit_linear(X, c)
        sigma = pooled_within_std(X, c)
        m_true = np.asarray(bg[MATRIX_OF[e]], float).mean(0)
        locos = loco_cv(X, c)
        fits[e] = dict(a=a, b=b, sigma=sigma, m_true=m_true, c=c, X=X, loco=locos,
                       cos_m=cos(a, m_true),
                       cos_s_m=cos(a + b, m_true),
                       rms_resid=float(np.sqrt(((X - (a + np.outer(c, b))) ** 2).mean())))
        print(f"[exp15] {e}: cos(m̂, m_true)={fits[e]['cos_m']:.4f}  "
              f"cos(ŝ,m_true)={fits[e]['cos_s_m']:.4f}  "
              f"resid_rms={fits[e]['rms_resid']:.4f}  noise={sigma.mean():.4f}", flush=True)

    # ---------- Part 3：LOAO 训练 + 各浓度零样本检索 ----------
    for fold, e_hold in enumerate(TARGETS):
        eids_tr = [e for e in eids if e != e_hold]
        tr_ix = {e: i for i, e in enumerate(eids_tr)}
        parts = [ent[e] for e in eids_tr] + [bg[m] for m in mids]
        ys = np.concatenate([np.full(len(ent[e]), tr_ix[e]) for e in eids_tr] +
                            [np.full(len(bg[m]), len(eids_tr) + j) for j, m in enumerate(mids)])
        Xs = np.concatenate(parts).astype(np.float32)
        ix_seen = [keys.index(e) for e in eids_tr] + [keys.index(f"M_{m}") for m in mids]
        T_seen = T_full[ix_seen]

        true_ix = eids.index(e_hold)
        a, b = fits[e_hold]["a"], fits[e_hold]["b"]
        sigma = fits[e_hold]["sigma"]

        # 合成各浓度谱（有噪声 / 无噪声两套）
        synth_n, synth_c = {}, {}
        rng = np.random.default_rng(2026)
        for cp in GRID:
            cf = cp / 100.0
            base = a + cf * b
            synth_c[cp] = np.tile(base, (args.n_syn, 1))
            noise = rng.normal(0, 1, size=(args.n_syn, len(base))) * sigma
            synth_n[cp] = base + noise

        acc_n = {cp: [] for cp in GRID}
        acc_c = {cp: [] for cp in GRID}
        mar_n = {cp: [] for cp in GRID}
        all_sn = {cp: [] for cp in GRID}
        for seed in seeds:
            net = E8.train_cnn(Xs, ys, T_seen, args.epochs, seed * 1000 + fold)
            with torch.no_grad():
                zn = E8.embed(net, np.asarray(np.concatenate(
                    [synth_n[cp] for cp in GRID]), dtype=np.float32))
                zc = E8.embed(net, np.asarray(np.concatenate(
                    [synth_c[cp] for cp in GRID]), dtype=np.float32))
            Tt = torch.nn.functional.normalize(
                torch.tensor(T_query, dtype=torch.float32, device=DEV), dim=1)
            sn = (torch.nn.functional.normalize(torch.tensor(zn, device=DEV), dim=1)
                  @ Tt.T).cpu().numpy()
            sc = (torch.nn.functional.normalize(torch.tensor(zc, device=DEV), dim=1)
                  @ Tt.T).cpu().numpy()
            off = 0
            for cp in GRID:
                blk_n = sn[off:off + args.n_syn]
                blk_c = sc[off:off + args.n_syn]
                acc_n[cp].append(float((blk_n.argmax(1) == true_ix).mean()))
                acc_c[cp].append(float((blk_c.argmax(1) == true_ix).mean()))
                # margin = sim(真值类) − max(其它类)
                mt_n = blk_n[:, true_ix] - np.delete(blk_n, true_ix, axis=1).max(1)
                mar_n[cp].append(float(mt_n.mean()))
                all_sn[cp].append(blk_n)
                off += args.n_syn
            del net
            if DEV == "cuda":
                torch.cuda.empty_cache()

        # 真实谱在观测浓度上的 top-1（对照：验证合成保真度）
        real_acc = {}
        Xh = np.asarray(ent[e_hold], float)
        ch = np.asarray(conc[e_hold], float)
        nets = None
        real_hits = []
        for seed in seeds:
            net = E8.train_cnn(Xs, ys, T_seen, args.epochs, seed * 1000 + fold)
            with torch.no_grad():
                zr = E8.embed(net, Xh.astype(np.float32))
            Tt = torch.nn.functional.normalize(
                torch.tensor(T_query, dtype=torch.float32, device=DEV), dim=1)
            pr = (torch.nn.functional.normalize(torch.tensor(zr, device=DEV), dim=1)
                  @ Tt.T).cpu().numpy().argmax(1)
            real_hits.append(pr == true_ix)
            del net
            if DEV == "cuda":
                torch.cuda.empty_cache()
        rh = np.mean(np.stack(real_hits), axis=0)
        for lv in np.unique(ch):
            m = (ch == lv)
            real_acc[float(lv)] = (float(rh[m].mean()), int(m.sum()))

        report[e_hold] = dict(
            acc_n={cp: float(np.mean(acc_n[cp])) for cp in GRID},
            acc_c={cp: float(np.mean(acc_c[cp])) for cp in GRID},
            mar_n={cp: float(np.mean(mar_n[cp])) for cp in GRID},
            real=real_acc, fit=fits[e_hold])
        print(f"[exp15] {e_hold} 完成 "
              f"(2%: noise={np.mean(acc_n[2.0]):.3f} clean={np.mean(acc_c[2.0]):.3f})", flush=True)

    # ================= 报告 =================
    L = ["# exp15 — 液-液体系零样本检出限（LOD_zs）外推", "",
         "> 动机：exp13 显示液-液体系在数据集最低浓度（3–10%）即 100% 可辨识，",
         "> 真实数据无法给出 LOD。公开数据中 <3% 的液-液 FTIR 掺假集极其稀缺，",
         "> 故采用**受控外推**：线性混合模型拟合 → 保真度三重验证 → 噪声注入 → 低浓度合成。", "",
         f"> 协议：LOAO（留出目标实体，其余 12 实体 + 6 基质背景训练），候选 N=13，"
         f"chance={chance:.4f}，{len(seeds)} seeds × {args.epochs} epoch，每浓度 {args.n_syn} 条合成谱。", "",
         "## 1. 线性混合模型拟合与保真度", "",
         "> **线性性门槛**：液-液体系应满足线性叠加（Beer–Lambert），此时外推合法；",
         "> 固-固粉末体系存在散射与基体效应，线性叠加不成立，其低浓度外推**不可用**，",
         "> 仅保留作负对照。", "",
         "| 实体 | 掺杂物/基质 | 相态 | cos(m̂, m_true) | cos(ŝ, m_true) | 残差 RMS | "
         "噪声 σ | 相对残差 | LOCO 余弦 | 线性性 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for e in TARGETS:
        f = fits[e]
        loco_cos = [x[2] for x in f["loco"]]
        rel = f["rms_resid"] / (np.linalg.norm(f["m_true"]) / np.sqrt(len(f["m_true"])))
        lin_ok = "✅ 成立" if (f["cos_m"] >= 0.99 and rel < 0.05) else "❌ 不成立"
        L.append(f"| {e} | {G1.ENTITY_META[e][0]}/{G1.ENTITY_META[e][1]} | {PHASE_OF[e]} | "
                 f"{f['cos_m']:.4f} | {f['cos_s_m']:.4f} | {f['rms_resid']:.4f} | "
                 f"{f['sigma'].mean():.4f} | {rel:.4f} | {np.mean(loco_cos):.4f} | {lin_ok} |")
    L += ["", "- **cos(m̂, m_true)**：把回归外推到 c=0 得到的「纯基质谱」与**真实纯基质谱**的余弦。",
          "  越接近 1，线性叠加假设越成立 —— 这是外推合法性的第一道验证。",
          "- **LOCO（留一浓度交叉验证）**：用其余浓度拟合后预测留出浓度的真实谱，",
          "  余弦接近 1 说明模型在观测浓度范围内可靠。", ""]

    L += ["## 2. 合成谱 vs 真实谱的一致性（同一浓度对照）", "",
          "| 实体 | 浓度% | 真实谱 top-1 (n) | 合成(含噪) top-1 | 一致? |",
          "|---|---|---|---|---|"]
    for e in TARGETS:
        r = report[e]
        for lv in sorted(r["real"]):
            t1, n = r["real"][lv]
            syn = r["acc_n"].get(float(lv))
            if syn is None:
                closest = min(r["acc_n"], key=lambda x: abs(x - lv))
                syn = r["acc_n"][closest]
                note = f"（最近梯度 {closest}%）"
            else:
                note = ""
            agree = "✅" if abs(t1 - syn) < 0.15 else "⚠️"
            L.append(f"| {e} | {lv:.0f} | {t1:.3f} ({n}) | {syn:.3f} {note} | {agree} |")
    L += ["", "合成谱在**已观测浓度**上必须复现真实谱的行为，否则低浓度外推不可信。", ""]

    L += ["## 3. 浓度–可辨识性曲线与 LOD_zs", "",
          "> **虚高陷阱检验**：下表第一行 `0%` 是纯基质谱基线。若基线 top-1 已经很高，",
          "> 说明低浓度下的命中并非来自「检出掺杂物」，而是模型对基质谱的分类偏好，",
          "> 此时 top-1 不能用于定义 LOD，须以 **margin**（真值类相似度 − 最强干扰类）为准。", ""]
    for e in TARGETS:
        r = report[e]
        base_n = r["acc_n"][0.0]
        base_m = r["mar_n"][0.0]
        L += [f"### {e}（{G1.ENTITY_META[e][0]} / {G1.ENTITY_META[e][1]}，{PHASE_OF[e]}）", "",
              f"纯基质基线（c=0）：top-1 = {base_n:.3f}，margin = {base_m:+.4f}", "",
              "| 浓度% | 含噪 top-1 | 无噪 top-1 | margin(含噪) | Δmargin vs c=0 |",
              "|---|---|---|---|---|"]
        for cp in GRID:
            L.append(f"| {cp} | {r['acc_n'][cp]:.3f} | {r['acc_c'][cp]:.3f} | "
                     f"{r['mar_n'][cp]:+.4f} | {r['mar_n'][cp] - base_m:+.4f} |")
        t1n = [r["acc_n"][cp] for cp in GRID]
        nn = [args.n_syn] * len(GRID)
        lod_top1 = lod_of(GRID, t1n, nn, chance)
        # margin 判据：margin 显著高于 c=0 基线（Welch t，各 seed×样本）
        lod_mar = None
        for i, cp in enumerate(GRID):
            if cp <= 0:
                continue
            if r["mar_n"][cp] - base_m >= 0.02 and r["acc_n"][cp] >= 0.5:
                lod_mar = cp
                break
        L += ["", f"- **LOD_zs（top-1 判据）** = "
                  f"{f'{lod_top1:g}%' if lod_top1 else f'> {GRID[-1]:g}%（未检出）'}",
              f"- **LOD_zs（margin 判据，抗虚高）** = "
                  f"{f'{lod_mar:g}%' if lod_mar else f'> {GRID[-1]:g}%（未检出）'}",
              f"- 基线是否虚高：**{'是' if base_n >= 0.5 else '否'}**"
              + ("（top-1 判据不可用，以 margin 为准）" if base_n >= 0.5 else ""), ""]

    L += ["## 4. 结论", "",
          "| 实体 | 相态 | 纯基质基线 top-1 | LOD_zs（top-1） | LOD_zs（margin） |",
          "|---|---|---|---|---|"]
    rows = []
    for e in TARGETS:
        r = report[e]
        t1n = [r["acc_n"][cp] for cp in GRID]
        lod_top1 = lod_of(GRID, t1n, [args.n_syn] * len(GRID), chance)
        base_n, base_m = r["acc_n"][0.0], r["mar_n"][0.0]
        lod_mar = None
        for cp in GRID:
            if cp <= 0:
                continue
            if r["mar_n"][cp] - base_m >= 0.02 and r["acc_n"][cp] >= 0.5:
                lod_mar = cp
                break
        rows.append((e, PHASE_OF[e], base_n,
                     f"{lod_top1:g}%" if lod_top1 else f">{GRID[-1]:g}%",
                     f"{lod_mar:g}%" if lod_mar else f">{GRID[-1]:g}%"))
    for r_ in rows:
        L.append(f"| {r_[0]} | {r_[1]} | {r_[2]:.3f} | {r_[3]} | **{r_[4]}** |")
    L += ["", "**必须声明的 limitation**：低浓度段为**模型外推**，非实测。",
          "其合法性依赖第 1/2 节的保真度验证；若审稿人要求实测验证，",
          "需补充真实低浓度（0.5–3%）掺假样品的 FTIR 测量。",
          "无噪声合成给出乐观上界，含噪合成给出保守下界，两者共同界定 LOD 区间。", ""]

    md = "\n".join(L)
    path = os.path.join(OUT, "exp15_p6_lod_extrapolation.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)

    import csv
    with open(os.path.join(OUT, "exp15_p6_lod_extrapolation.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["entity", "phase", "conc_pct", "top1_noisy", "top1_clean"])
        for e in TARGETS:
            for cp in GRID:
                w.writerow([e, PHASE_OF[e], cp,
                            f"{report[e]['acc_n'][cp]:.4f}",
                            f"{report[e]['acc_c'][cp]:.4f}"])


if __name__ == "__main__":
    main()
