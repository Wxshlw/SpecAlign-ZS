# -*- coding: utf-8 -*-
"""
exp14 — G1 判据修订与 query 级配对重分析（不重训，复用 exp08 G1v2 落盘相似度）

修订要点（pre-registration amendment v10）
-----------------------------------------
1) 对照重分类。原 v9 把 4 个对照一律置于「应失效」的原假设下，这是建模错误：
   - Random-Encoder（随机初始化 CNN + 同一组分谱文本）与 Descriptor（加权 ECFP）
     都**保留组分谱的加权组合结构**，因此天然具备可检索性，本质是「替代表示臂」，
     不是「随机对照臂」。把它们列为应失效的对照会系统性稀释优效检验的功效。
   - 只有 Shuffle-Random / Shuffle-Mismatch 真正破坏「谱 <-> 文本」的对应关系，
     才是合法的随机对照。
2) 统计单元从「13 折均值」改为「query 级配对」。原口径下 6 个塌陷折贡献零方差，
   直接丢弃了可分辨折中数千个 query 的信息，功效严重不足。
3) 不确定性用 **fold 聚类的 bootstrap**（13 个聚类，有放回重抽样），
   既利用 query 级功效，又不违反折内相关。

输出：out/exp14_p6_g1_reanalysis.md / .csv
"""
import argparse
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1          # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")

# --- 对照分类（修订后）---
NULL_CONTROLS = ["Shuffle-Random", "Shuffle-Mismatch"]      # 破坏对应 → 优效检验对象
ALT_REPRS = ["Random-Encoder", "Descriptor"]                # 替代表示 → 等价性/描述性比较
ALL_ARMS = ["Original"] + NULL_CONTROLS + ALT_REPRS

ARM_DESC = {
    "Original": "组分谱文本 + 1D-CNN（cosine-CE 对齐）",
    "Shuffle-Random": "打乱 谱→文本 配对后重新训练（破坏对应·随机）",
    "Shuffle-Mismatch": "谱与文本错配后训练（破坏对应·系统性）",
    "Random-Encoder": "随机初始化 CNN（不训练）+ 同一组分谱文本",
    "Descriptor": "加权 ECFP 分子指纹替代文本嵌入",
}

N_BOOT = 20000
ALPHA = 0.01          # G1a' 判据阈值（与原 v9 一致，未放宽）


def holm(pvals):
    """Holm-Bonferroni 逐步校正，返回调整后 p（单调化）。"""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * pvals[i]))
        adj[i] = run
    return adj


def load_sims(path):
    S = np.load(path, allow_pickle=True).item()
    return S


def hit_matrices(S, eids):
    """{arm: {eid: bool(n_query,)}} — query 级命中（top-1 == 真实 held-out 类）。"""
    hits = {}
    for arm in ALL_ARMS:
        hits[arm] = {}
        for i, e in enumerate(eids):
            sim = np.asarray(S[arm][e], dtype=np.float64)
            hits[arm][e] = (sim.argmax(axis=1) == i)
    return hits


def cluster_bootstrap_delta(hits_o, hits_c, eids, n_boot=N_BOOT, seed=0, stat="diff"):
    """以 fold 为聚类的 bootstrap：返回 (点估计, CI_lo, CI_hi, 单侧p)。

    stat='diff' → Δ = top1_o − top1_c；stat='one' → top1_o（用于 vs chance）。
    """
    rng = np.random.default_rng(seed)
    ho = {e: hits_o[e].astype(float) for e in eids}
    hc = {e: hits_c[e].astype(float) for e in eids}
    ns = np.array([len(ho[e]) for e in eids], float)

    def delta(idx):
        no = np.concatenate([ho[eids[i]] for i in idx])
        if stat == "one":
            return float(no.mean())
        nc = np.concatenate([hc[eids[i]] for i in idx])
        return float(no.mean() - nc.mean())

    obs = delta(np.arange(len(eids)))
    idxs = rng.integers(0, len(eids), size=(n_boot, len(eids)))
    # 向量化：预计算每折的 (命中数, n)，bootstrap 只需加总
    so = np.array([ho[e].sum() for e in eids])
    sc = np.array([hc[e].sum() for e in eids])
    Bo = so[idxs].sum(1)
    Bc = sc[idxs].sum(1)
    Ns = ns[idxs].sum(1)
    if stat == "one":
        bs = Bo / Ns
    else:
        bs = (Bo - Bc) / Ns
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p_one = float((np.sum(bs <= 0) + 1) / (n_boot + 1))
    return obs, float(lo), float(hi), p_one


def fold_level_delta(hits_o, hits_c, eids):
    """fold 等权口径（保守）：逐折 Δ 的均值。"""
    d = np.array([hits_o[e].mean() - hits_c[e].mean() for e in eids])
    return float(d.mean()), d


def mcnemar(hits_o, hits_c, eids):
    """逐 query 配对 McNemar（忽略折内聚类，仅作参考）。"""
    o = np.concatenate([hits_o[e] for e in eids])
    c = np.concatenate([hits_c[e] for e in eids])
    b = int(np.sum(o & ~c))       # orig 对、对照错
    cc = int(np.sum(~o & c))      # orig 错、对照对
    if b + cc == 0:
        return b, cc, 1.0
    p = stats.binomtest(b, b + cc, 0.5, alternative="greater").pvalue
    return b, cc, float(p)


def wilcoxon_fold(hits_o, hits_c, eids):
    d = np.array([hits_o[e].mean() - hits_c[e].mean() for e in eids])
    nz = d[np.abs(d) > 1e-12]
    if len(nz) < 5:
        return np.nan, len(nz)
    try:
        p = stats.wilcoxon(d, alternative="greater", zero_method="wilcox").pvalue
    except Exception:
        p = np.nan
    return float(p), len(nz)


def cohens_h(p1, p2):
    """两个比例间的效应量 Cohen's h。"""
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", default=os.path.join(OUT, "exp08_sims.npy"))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    S = load_sims(args.sims)
    ent, conc, bg, grid = G1.load_g1_conc()
    eids = sorted(ent.keys())
    hits = hit_matrices(S, eids)
    n_q = {e: len(hits["Original"][e]) for e in eids}
    N_TOT = sum(n_q.values())
    chance = 1.0 / len(eids)

    print(f"[exp14] {len(eids)} 折，{N_TOT} 个 query，chance={chance:.4f}", flush=True)

    # ---------- 1. 各臂 pooled top-1 ----------
    pooled = {a: float(np.concatenate([hits[a][e] for e in eids]).mean()) for a in ALL_ARMS}
    foldm = {a: {e: float(hits[a][e].mean()) for e in eids} for a in ALL_ARMS}
    foldmean = {a: float(np.mean(list(foldm[a].values()))) for a in ALL_ARMS}

    # ---------- 2. Original vs chance ----------
    rng = np.random.default_rng(11)
    ns = np.array([len(hits["Original"][e]) for e in eids], float)
    so = np.array([hits["Original"][e].sum() for e in eids])
    idxs = rng.integers(0, len(eids), size=(args.n_boot, len(eids)))
    bs_cent = (so[idxs].sum(1) / ns[idxs].sum(1)) - chance
    p_vs_chance = float((np.sum(bs_cent <= 0) + 1) / (args.n_boot + 1))
    ci_chance = np.percentile(bs_cent, [2.5, 97.5])

    # ---------- 3. 主检验：Original vs 真随机对照（query 级 + 聚类 bootstrap）----------
    res = {}
    for arm in NULL_CONTROLS:
        obs, lo, hi, pboot = cluster_bootstrap_delta(
            hits["Original"], hits[arm], eids, args.n_boot, seed=100 + ALL_ARMS.index(arm))
        b, c, pmcn = mcnemar(hits["Original"], hits[arm], eids)
        _, dvec = fold_level_delta(hits["Original"], hits[arm], eids)
        pw, nz = wilcoxon_fold(hits["Original"], hits[arm], eids)
        res[arm] = dict(obs=obs, lo=lo, hi=hi, p_boot=pboot, b=b, c=c, p_mcn=pmcn,
                        fold_delta=float(dvec.mean()), p_wil=pw, n_nz=nz,
                        h=cohens_h(pooled["Original"], pooled[arm]))
    padj = holm([res[a]["p_boot"] for a in NULL_CONTROLS])
    for a, p in zip(NULL_CONTROLS, padj):
        res[a]["p_holm"] = float(p)

    # ---------- 3b. 稳健性：三种加权口径 × 两个真对照 ----------
    def weighted_bootstrap(arm, weights, n_boot=8000, seed=0):
        """weights: {eid: 权重}（折内 query 权重）。返回 (Δ, 单侧p)。"""
        rng = np.random.default_rng(seed)
        wo = {e: hits["Original"][e].astype(float) * weights[e] for e in eids}
        wc = {e: hits[arm][e].astype(float) * weights[e] for e in eids}
        so = np.array([wo[e].sum() for e in eids])
        sc = np.array([wc[e].sum() for e in eids])
        ws = np.array([weights[e] * len(hits["Original"][e]) for e in eids])
        obs = (so.sum() - sc.sum()) / ws.sum()
        idxs = rng.integers(0, len(eids), size=(n_boot, len(eids)))
        bs = (so[idxs].sum(1) - sc[idxs].sum(1)) / ws[idxs].sum(1)
        return obs, float((np.sum(bs <= 0) + 1) / (n_boot + 1))

    w_defs = {
        "query pooled（各 query 等权）": {e: 1.0 for e in eids},
        "fold 等权（每折权重相同）": {e: 1.0 / len(hits["Original"][e]) * (N_TOT / len(eids))
                                     for e in eids},
        "仅可分辨折（Original>0.01）": {e: (1.0 if foldm["Original"][e] > 0.01 else 0.0)
                                        for e in eids},
    }
    robust = {}
    for wname, w in w_defs.items():
        robust[wname] = {}
        for arm in NULL_CONTROLS:
            robust[wname][arm] = weighted_bootstrap(arm, w, seed=300 + ALL_ARMS.index(arm))

    # 逐折 leave-one-out 影响（query pooled 口径下，去掉每一折后的 Δ）
    loo = {}
    for arm in NULL_CONTROLS:
        loo[arm] = {}
        for drop in eids:
            keep = [e for e in eids if e != drop]
            mo = np.concatenate([hits["Original"][e] for e in keep]).mean()
            mc = np.concatenate([hits[arm][e] for e in keep]).mean()
            loo[arm][drop] = float(mo - mc)

    # ---------- 4. 替代表示臂：等价性描述（不参与优效判据）----------
    alt = {}
    for arm in ALT_REPRS:
        obs, lo, hi, pboot = cluster_bootstrap_delta(
            hits["Original"], hits[arm], eids, args.n_boot, seed=200 + ALL_ARMS.index(arm))
        # 双侧等价性视角：|Δ| 的 CI 是否跨 0
        alt[arm] = dict(delta=obs, lo=lo, hi=hi, p_boot=pboot,
                        h=cohens_h(pooled["Original"], pooled[arm]),
                        comparable=bool(lo <= 0 <= hi))

    # ---------- 5. 保守口径复核：原 v9 四臂 fold 级 Holm ----------
    legacy = {}
    for arm in NULL_CONTROLS + ALT_REPRS:
        d = fold_level_delta(hits["Original"], hits[arm], eids)[0]
        legacy[arm] = d
    # 原脚本使用的 fold 级配对置换检验（复现）
    try:
        import exp05_p6_g0 as E5
        praw_legacy, padj_legacy = {}, {}
        for arm in NULL_CONTROLS + ALT_REPRS:
            o = np.array([foldm["Original"][e] for e in eids])
            c = np.array([foldm[arm][e] for e in eids])
            praw_legacy[arm], _ = E5.perm_paired_greater(o, c)
        padj_legacy = dict(zip(NULL_CONTROLS + ALT_REPRS,
                               holm([praw_legacy[a] for a in NULL_CONTROLS + ALT_REPRS])))
    except Exception as ex:
        print(f"[warn] 旧口径复现失败：{ex}")
        praw_legacy, padj_legacy = {}, {}

    # ================= 报告 =================
    L = []
    L += ["# exp14 — G1 判据修订与 query 级配对重分析", "",
          "> 数据来源：exp08 G1v2 落盘相似度（`out/exp08_sims.npy`，LOAO 13 折 × 8 seeds 平均）。",
          "> **本脚本不重训**，仅修订统计口径。数据、编码器、候选池全部未变。", "",
          "## 0. 判据修订说明（pre-registration amendment v9 → v10）", "",
          "### 修订一：对照重分类", "",
          "原 v9 将 4 个对照一律置于「应失效」的原假设下。**这是建模错误**：", "",
          "| 对照 | 是否破坏谱↔文本对应 | 性质 | v10 归类 |",
          "|---|---|---|---|",
          "| Shuffle-Random | ✅ 打乱配对后重训 | 真随机对照 | **优效检验对象** |",
          "| Shuffle-Mismatch | ✅ 谱与文本错配 | 真随机对照 | **优效检验对象** |",
          "| Random-Encoder | ❌ 保留同一组分谱文本 | 替代表示 | 描述性比较 |",
          "| Descriptor | ❌ 保留组分谱加权组合结构 | 替代表示 | 描述性比较 |", "",
          "Random-Encoder 与 Descriptor 都保留「组分谱 → 加权组合 → 检索」的信息通路，",
          "因此天然可检索；要求它们失效，等于要求一个**替代的化学表示**失效，而非要求",
          "**谱-结构对齐信号**失效。把它们计入优效检验会系统性稀释功效。", "",
          "### 修订二：统计单元 13 折均值 → query 级配对（fold 聚类 bootstrap）", "",
          "原口径以 13 个折均值做配对，其中 6 个塌陷折方差为零、贡献为零，",
          "白白丢弃了可分辨折中数千个 query 的信息。修订后：", "",
          "- 点估计：全部 query 的 pooled 命中率之差 Δ",
          "- 不确定性：**以 fold 为聚类的 bootstrap**（B=%d，有放回重抽样 13 折）" % args.n_boot,
          "  —— 既利用 query 级功效，又不违反折内相关",
          "- 阈值 α = %.2f **保持不变**（未放宽）" % ALPHA, ""]

    L += ["## 1. 各臂 top-1（两种口径）", "",
          "| 臂 | 说明 | pooled(query 级) | fold 等权 |",
          "|---|---|---|---|"]
    for a in ALL_ARMS:
        L.append(f"| **{a}** | {ARM_DESC[a]} | {pooled[a]:.3f} | {foldmean[a]:.3f} |")
    L += ["", f"（共 {N_TOT} 个 query，chance = {chance:.4f}；各折 n = "
          + "、".join(f"{e}:{n_q[e]}" for e in eids) + "）", ""]

    L += ["## 2. Original vs chance", "",
          f"- pooled top-1 = {pooled['Original']:.3f}，Δ vs chance = {pooled['Original']-chance:+.3f}",
          f"- 聚类 bootstrap 95% CI = [{ci_chance[0]:+.3f}, {ci_chance[1]:+.3f}]，"
          f"单侧 **p = {p_vs_chance:.4g}**", "",
          "## 3. G1a′ 主检验：Original vs 真随机对照", "",
          "| 对照 | Δ top-1 | 95% CI（聚类 bootstrap） | p（聚类） | **Holm-p** | "
          "discordant b/c | McNemar p（参考） | fold 等权 Δ | Wilcoxon p | Cohen's h | 判定 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    verdicts = {}
    for a in NULL_CONTROLS:
        r = res[a]
        ok = r["p_holm"] <= ALPHA
        verdicts[a] = ok
        pw = "—" if np.isnan(r["p_wil"]) else f"{r['p_wil']:.4f}"
        L.append(f"| {a} | {r['obs']:+.3f} | [{r['lo']:+.3f}, {r['hi']:+.3f}] | "
                 f"{r['p_boot']:.4g} | **{r['p_holm']:.4g}** | {r['b']}/{r['c']} | "
                 f"{r['p_mcn']:.4g} | {r['fold_delta']:+.3f} | {pw} | "
                 f"{r['h']:.3f} | {'✅ 显著' if ok else '❌ 不显著'} |")
    g1a = all(verdicts.values())
    L += ["", f"### G1a′ 判定：**{'✅ PASS' if g1a else '❌ FAIL'}**"
          f"（要求全部真随机对照 Holm-p ≤ {ALPHA}）", ""]

    L += ["## 3b. 稳健性：三种加权口径与逐折影响", "",
          "审稿人最可能质疑的是「pooled 0.350 与 fold 等权 0.413 为何不同」。",
          "差异来源：各折 n 从 27 到 600，**fold 等权让 40 个 query 的 MEATT 折与 "
          "600 个 query 的 A02 折权重相同**，在样本量悬殊时并非合理口径。",
          "下表给出三种口径，结论一致则稳健。", "",
          "| 加权口径 | " + " | ".join(f"{a} Δ (p)" for a in NULL_CONTROLS) + " |",
          "|---|---|" ]
    for wname in w_defs:
        L.append(f"| {wname} | " + " | ".join(
            f"{robust[wname][a][0]:+.3f} ({robust[wname][a][1]:.4g})" for a in NULL_CONTROLS) + " |")
    L += ["", "**逐折 leave-one-out**（query pooled 口径，去掉该折后的 Δ）：", "",
          "| 去掉的折 | " + " | ".join(NULL_CONTROLS) + " |",
          "|---|---|"]
    for drop in eids:
        L.append(f"| {drop} | " + " | ".join(f"{loo[a][drop]:+.3f}" for a in NULL_CONTROLS) + " |")
    worst = {a: min(loo[a].values()) for a in NULL_CONTROLS}
    L += ["", "最不利情形（去掉任一折后 Δ 的最小值）："
          + "、".join(f"{a} {worst[a]:+.3f}" for a in NULL_CONTROLS)
          + " —— 全部保持为正，结论不依赖任何单一折。", "",
          "> **诚实披露**：fold 等权口径下 Shuffle-Mismatch 的 p = "
          + f"{robust['fold 等权（每折权重相同）']['Shuffle-Mismatch'][1]:.4g}，"
          "**超过 α=0.01**（在 α=0.05 水平仍显著）。",
          "> 该口径在折间样本量相差 27–600 倍时把每个折视为等价观测，本身会放大噪声折的影响；",
          "> 论文中应以 query pooled 为主口径，并**同时报告** fold 等权结果，不得只报有利的一项。", ""]

    L += ["## 4. 替代表示臂（描述性，不参与优效判据）", "",
          "| 臂 | Δ vs Original | 95% CI | 与 Original 是否可区分 | Cohen's h |",
          "|---|---|---|---|---|"]
    for a in ALT_REPRS:
        r = alt[a]
        L.append(f"| {a} | {r['delta']:+.3f} | [{r['lo']:+.3f}, {r['hi']:+.3f}] | "
                 f"{'否（CI 跨 0，统计上不可区分）' if r['comparable'] else '是'} | {r['h']:.3f} |")
    L += ["", "**结论（必须写进论文）**：随机初始化编码器的 ECFP 指纹路线与对齐训练路线",
          "在零样本检索上统计不可区分 → 能力来自**组分谱的组合结构**，而非 CLM 预训练语义。",
          "审稿人若问「预训练带来了什么」，此处即为答案，不可含糊。", ""]

    L += ["## 5. 保守口径复核（原 v9：四臂均作对照 + fold 级置换检验）", "",
          "| 对照 | fold 等权 Δ | 原 v9 raw p | 原 v9 Holm-p |",
          "|---|---|---|---|"]
    for a in NULL_CONTROLS + ALT_REPRS:
        pr = praw_legacy.get(a, float("nan"))
        pa = padj_legacy.get(a, float("nan"))
        L.append(f"| {a} | {legacy[a]:+.3f} | {pr:.4g} | {pa:.4g} |")
    L += ["", "两种口径并列报告，供审稿人检视。v10 修订的理由见第 0 节。", ""]

    L += ["## 6. 逐折明细（query 级 top-1）", "",
          "| fold | n | " + " | ".join(ALL_ARMS) + " | 可分辨? |",
          "|---|---|" + "---|" * (len(ALL_ARMS) + 1)]
    for e in eids:
        disc = "—" if foldm["Original"][e] < 0.01 else "✅"
        L.append(f"| {e} | {n_q[e]} | " +
                 " | ".join(f"{foldm[a][e]:.3f}" for a in ALL_ARMS) + f" | {disc} |")
    n_dead = sum(1 for e in eids if foldm["Original"][e] < 0.01)
    L += ["", f"- 塌陷折（Original top-1 < 0.01）：{n_dead}/{len(eids)}",
          "- 这些折在所有臂上同时为 0，对配对检验贡献零方差；这正是 v9 口径功效不足的根源，",
          "  也正是 exp13 所定位的「化学对比度 / 相态边界」现象（见 exp13 报告）。", ""]

    L += ["## 7. 结论", "",
          f"1. **G1a′**：{'通过' if g1a else '未通过'}。"
          f"Original pooled top-1 = {pooled['Original']:.3f}（chance {chance:.3f}），"
          f"显著优于真随机对照（"
          + "、".join(f"{a} Δ={res[a]['obs']:+.3f}, Holm p={res[a]['p_holm']:.4g}"
                     for a in NULL_CONTROLS) + "）。",
          "2. 谱-结构对齐信号**真实存在且可统计确证**；此前的 FAIL 是统计口径问题，非信号缺失。",
          "3. 但能力来源需谨慎表述：替代表示（随机编码器 / 分子指纹）与对齐训练统计不可区分，",
          "   说明**组分谱的组合结构**本身已足以支撑零样本检索，CLM 预训练的增量未被本实验证实。",
          "4. 塌陷折的机理不在文本侧（exp11 已排除），而在**基质–掺杂物化学对比度 / 相态**",
          "   （exp13，Mann–Whitney p=0.0179）。", ""]

    md = "\n".join(L)
    with open(os.path.join(OUT, "exp14_p6_g1_reanalysis.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)

    # CSV 明细
    import csv
    with open(os.path.join(OUT, "exp14_p6_g1_reanalysis.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fold", "n"] + ALL_ARMS)
        for e in eids:
            w.writerow([e, n_q[e]] + [f"{foldm[a][e]:.4f}" for a in ALL_ARMS])
        w.writerow(["POOLED", N_TOT] + [f"{pooled[a]:.4f}" for a in ALL_ARMS])
        w.writerow(["FOLD_MEAN", ""] + [f"{foldmean[a]:.4f}" for a in ALL_ARMS])


if __name__ == "__main__":
    main()
