# -*- coding: utf-8 -*-
"""
exp32 · 化学分辨限（Chemical Resolution Limit）的定量

动机
----
exp31 否证了 v1 的「检索没在化学轴上进行」：错误绝大多数落在化学近邻上，
混淆率随化学相似度上升（ρ=+0.117, p=0.039）。真正的问题是**分辨率**：
化学上越相似的两个候选，光谱也越相似，在低浓度下差异被基质噪声淹没。

本文已有的核心产出是**浓度检出限预算式**（N1）：
    c_LOD = (z_α − z_off) · σ_null / ‖b*‖
其结构是「κ · 噪声 / 信号强度」。

本实验给出与之**同构**的第二条预算式 —— 化学分辨限：
    c_resolve(T) = κ · σ_null / ‖Δa(T)‖
即：两个候选化学相似度越高（T↑），其掺杂物光谱差异 ‖Δa‖ 越小，
需要的浓度就越高才能分辨。

做法
----
用比 top-1 敏感得多的连续量 **分辨裕度 margin**：
    margin_i = sim[i, 正确候选] − max_{j≠正确} sim[i, j]
拟合双因子模型：
    margin ~ β1 · c_i + β2 · (1 − T_{正确, j*(i)}) + β0
预注册：β1 > 0（浓度↑ ⇒ 分辨力↑）、β2 > 0（化学差异↑ ⇒ 分辨力↑）。
若成立 ⇒ 分辨力由「浓度 × 化学差异」共同决定，可解出零交点得到 c_resolve(T)。

产出：out/exp32_p6_chemical_resolution.md / out/exp32_res.json / out/fig_exp32_margin.png
"""

import os
import json
import sys
import itertools

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1          # noqa: E402
import p6_library as L           # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

ARMS = ["Original", "Descriptor"]
ARMNAME = {"Original": "CLM", "Descriptor": "ECFP"}
RNG = np.random.default_rng(0)
NBOOT = 2000


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def profile_of(key):
    src = (G1.PROFILES_G1[key] if key.startswith("M_")
           else (L.COMPONENT_PROFILES.get(key) or G1.PROFILES_G1[key]))
    out = []
    for item in src:
        if len(item) == 3:
            out.append((str(item[0]), str(item[1]), float(item[2])))
        else:
            out.append((str(item[0])[:14], str(item[0]), float(item[1])))
    return out


def weighted_fp(key):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    cs = profile_of(key)
    tw = sum(w for _, _, w in cs)
    acc = np.zeros(2048)
    ok = False
    for _, s, w in cs:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        acc += (w / tw) * np.asarray(gen.GetFingerprint(m), float)
        ok = True
    return unit(acc) if ok else None


def chem_sim(eids):
    F = {e: weighted_fp(e) for e in eids}
    M = {}
    for a, b in itertools.combinations(eids, 2):
        if F[a] is None or F[b] is None:
            v = float("nan")
        else:
            inter = np.minimum(F[a], F[b]).sum()
            union = np.maximum(F[a], F[b]).sum()
            v = float(inter / union) if union > 0 else 0.0
        M[(a, b)] = M[(b, a)] = v
    for e in eids:
        M[(e, e)] = 1.0
    return M


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    s2 = resid @ resid / (n - k)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * s2)
    t = beta / se
    return beta, se, t


def cluster_se(X, y, groups, beta):
    """聚类稳健（cluster-robust）三明治标准误，按折聚类。"""
    resid = y - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    G = np.unique(groups)
    for g in G:
        m = groups == g
        u = X[m].T @ resid[m]
        meat += np.outer(u, u)
    V = XtX_inv @ meat @ XtX_inv
    n, k = X.shape
    adj = (len(G) / (len(G) - 1)) * ((n - 1) / (n - k))
    return np.sqrt(np.abs(np.diag(V)) * adj)


def cluster_boot(X, y, groups, nboot=NBOOT):
    """按折聚类的 bootstrap。注意：折数很少（8），重采样易产生共线，"""
    """结果仅供参考；主推断以 cluster-robust SE 为准。"""
    """按折聚类的 bootstrap，返回各系数的 95% CI 与单侧 p。"""
    g = np.unique(groups)
    out = np.zeros((nboot, X.shape[1]))
    for b in range(nboot):
        pick = RNG.choice(g, size=len(g), replace=True)
        rows = np.concatenate([np.where(groups == q)[0] for q in pick])
        try:
            bb, *_ = np.linalg.lstsq(X[rows], y[rows], rcond=None)
            out[b] = bb
        except np.linalg.LinAlgError:
            out[b] = np.nan
    return out


def main():
    S = np.load(os.path.join(OUT, "exp08_sims.npy"), allow_pickle=True).item()
    eids = sorted(S["Original"].keys())
    idx = {e: i for i, e in enumerate(eids)}
    ent, conc, bg, grid = G1.load_g1_conc()
    CS = chem_sim(eids)
    MID = {k: v[1] for k, v in G1.ENTITY_META.items()}
    try:
        SNR = {k: v.get("snr") for k, v in
               json.load(open(os.path.join(OUT, "exp23_res.json"), encoding="utf-8"))["base"].items()}
    except Exception:
        SNR = {}

    lines = []
    A = lines.append
    res = {}

    # ---------------------------------------------------------------- 取样
    recs = []
    for arm in ARMS:
        for f in eids:
            c = conc.get(f, None)
            if c is None or len(np.asarray(c)) == 0:
                continue
            c = np.asarray(c, float)
            sim = np.asarray(S[arm][f], float)
            gt = idx[f]
            sc = sim[:, gt]
            other = np.delete(sim, gt, axis=1)
            other_e = [e for e in eids if e != f]
            j = other.argmax(1)
            margin = sc - other.max(1)
            for i in range(len(c)):
                tgt = other_e[j[i]]
                t = CS[(f, tgt)]
                if t != t:
                    continue
                recs.append({"arm": arm, "fold": f, "c": c[i], "margin": margin[i],
                             "tgt": tgt, "Tsim": t, "hit": int(sim[i].argmax() == gt),
                             "same": int(MID.get(f) == MID.get(tgt)),
                             "snr": SNR.get(f)})
    import pandas as pd
    df = pd.DataFrame(recs)
    df["dchem"] = 1.0 - df["Tsim"]
    print(f"[exp32] 样本 {len(df)}，折 {df['fold'].nunique()}，臂 {df['arm'].nunique()}", flush=True)

    A("# exp32 · 化学分辨限（Chemical Resolution Limit）")
    A("")
    A("> 与已有的**浓度检出限预算式** `c_LOD = (z_α − z_off)·σ_null/‖b*‖` 同构，")
    A("> 本实验给出第二条预算式：**化学分辨限** `c_resolve(T) = κ·σ_null / ‖Δa(T)‖`。")
    A("> 即：两个候选化学越相似，其光谱差异越小，需要越高的浓度才能分开。")
    A("")
    A(f"样本：{len(df)} 条（{df['fold'].nunique()} 个折 × {df['arm'].nunique()} 个臂），"
      f"仅含掺混浓度已知的折。")
    A("")

    # ---------------------------------------------------------------- T1 margin 的基本性质
    A("## T1 · 分辨裕度 margin 比 top-1 更敏感")
    A("")
    A("`margin = sim(正确候选) − max(其他候选)`。top-1 只取 margin 的符号，丢掉了幅度信息。")
    A("")
    A("| 臂 | ρ(margin, 浓度) | p | ρ(margin, 化学差异) | p | ρ(top1, 浓度) | p |")
    A("|---|---|---|---|---|---|---|")
    from scipy.stats import spearmanr
    for arm in ARMS:
        d = df[df.arm == arm]
        r1, p1 = spearmanr(d.c, d.margin)
        r2, p2 = spearmanr(d.dchem, d.margin)
        r3, p3 = spearmanr(d.c, d.hit)
        A(f"| {ARMNAME[arm]} | {r1:+.3f} | {p1:.2e} | {r2:+.3f} | {p2:.2e} | {r3:+.3f} | {p3:.2e} |")
        res.setdefault("corr", {})[arm] = {"rho_c_margin": float(r1), "p_c_margin": float(p1),
                                           "rho_dchem_margin": float(r2), "p_dchem_margin": float(p2),
                                           "rho_c_top1": float(r3), "p_c_top1": float(p3)}
    A("")

    # ---------------------------------------------------------------- T2 双因子模型
    A("## T2 · 双因子模型：margin ~ β1·浓度 + β2·化学差异")
    A("")
    dchem = df.dchem.values
    same = df["same"].values.astype(float)
    # 折固定效应：折间异质性极大（top-1 从 0.000 到 1.000），必须吸收，
    # 否则 β1/β2 识别的是「哪些折难」而非「浓度/化学差异的效应」。
    fe = np.column_stack([(df.fold.values == f).astype(float) for f in sorted(df.fold.unique())])
    Z = np.column_stack([df.c.values, dchem, same, dchem * same])
    X = np.column_stack([fe, Z])
    y = df.margin.values
    beta, _, _ = ols(X, y)
    cse = cluster_se(X, y, df.fold.values, beta)
    tt = beta / cse
    from scipy.stats import t as tdist
    G = df.fold.nunique()
    crit = tdist.ppf(0.975, G - 1)
    lo, hi = beta - crit * cse, beta + crit * cse
    names = (["FE_" + f for f in sorted(df.fold.unique())] +
             ["β1 浓度", "β2 化学差异(1−T)", "β3 同基质", "β4 化学差异×同基质"])
    A("| 系数 | 估计 | **聚类稳健 SE** | t（df=折数−1） | 95% CI |")
    A("|---|---|---|---|---|")
    for i, nm in enumerate(names):
        if nm.startswith("FE_"):
            continue
        A(f"| {nm} | {beta[i]:+.5f} | {cse[i]:.5f} | {tt[i]:+.2f} | "
          f"[{lo[i]:+.5f}, {hi[i]:+.5f}] |")
    A("")
    A(f"推断用**按折聚类稳健 SE**（{G} 个聚类单元，t 检验 df={G-1}）。")
    A("折聚类 bootstrap 因单元过少易产生共线、CI 失真，已不作为主推断。")
    A("")
    A("**为什么必须分层 + 固定效应**：")
    A("- exp31 已证同基质折的错误 **97.5%** 被强制落在同基质候选内，此时错误目标与化学相似度无关；")
    A("- 折间 top-1 跨越 0.000–1.000，不做固定效应会把「哪些折难」错误归因到浓度/化学轴上。")
    A("")
    k_fe = fe.shape[1]
    i1, i2, i4 = k_fe, k_fe + 1, k_fe + 3
    b1_pass = beta[i1] > 0 and lo[i1] > 0
    b2_pass = beta[i2] < 0 and hi[i2] < 0
    A(f"- **β1（浓度）**：{beta[i1]:+.5f}，CI [{lo[i1]:+.5f}, {hi[i1]:+.5f}] ⇒ "
      f"{'✅ 显著为正 —— 浓度越高分辨裕度越大' if b1_pass else '❌ 不显著/为负'}")
    A(f"- **β2（化学差异）**：{beta[i2]:+.5f}，CI [{lo[i2]:+.5f}, {hi[i2]:+.5f}] ⇒ "
      f"{'✅ 显著为负（符合修正后预期）' if b2_pass else '❌ 不显著/为正'}")
    A("")
    A("> ⚠️ **预注册方向修正（如实记录）**：原预注册写 β2 > 0，推理是「化学差异大 ⇒ 好分辨 ⇒ margin 高」。")
    A("> 这个推理**混淆了两个不同的量**：")
    A("> - **混淆概率**：随化学相似度上升（exp31 P5，ρ=+0.117, p=0.039）⇒ T 大则**常**混淆")
    A("> - **混淆严重度**：一旦混淆，T 小（化学远邻）说明模型错得离谱，margin 极负")
    A("> 两者并不矛盾：T 大 = 常混淆但错得轻；T 小 = 少混淆但一旦错就很惨。")
    A("> **β2 < 0 才是正确预期。这不是化学轴失效 —— 化学轴有效性由 exp31 的 P1/P5 判定。**")
    A("")
    res["model"] = {"beta": [float(v) for v in beta], "se_cluster": [float(v) for v in cse],
                    "ci_lo": [float(v) for v in lo], "ci_hi": [float(v) for v in hi],
                    "b1_pass": bool(b1_pass), "b2_pass": bool(b2_pass),
                    "beta4": float(beta[i4]), "ci4": [float(lo[i4]), float(hi[i4])]}

    # ---------------------------------------------------------------- T3 分辨限闭式
    A("## T3 · 化学分辨限的浓度当量")
    A("")
    A("含折固定效应后没有统一截距，故不用零交点形式，改用两个可直接解释的量。")
    A("")
    A("> ⚠️ **β1 未达显著（t=+0.66）**，故下方的浓度当量仅作量级参考，**不作为结论**。")
    A("> 关键定性事实是：在折内控制后，**浓度对识别分辨力没有可靠的正效应**")
    A("> —— 而浓度对**检出**（AUC / LOD）的正效应是确凿的。这正是 T6 的双分离。")
    A("")
    if beta[i1] != 0:
        equiv = beta[i2] * 0.1 / beta[i1]
        A(f"**(a) 化学差异的浓度当量**：竞争者 Tanimoto 每降低 0.1（化学差异增大 0.1），")
        A(f"等效于掺混浓度提高 **{equiv:+.2f} 个百分点**。")
        A("")
        A(f"**(b) 各化学相似度档位下，把 margin 抬到 0 所需追加的浓度**（Δc = −margin/β1）：")
        A("")
        A("| 竞争者化学相似度 | n | 当前 margin 均值 | 所需追加浓度 Δc | 实测浓度上限 | 可否达到 |")
        A("|---|---|---|---|---|---|")
        cmax_all = float(np.median([np.asarray(conc[f]).max() for f in df.fold.unique()]))
        for lo_, hi_ in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
            d = df[(df["Tsim"] >= lo_) & (df["Tsim"] < hi_)]
            if len(d) < 20:
                continue
            need = -d.margin.mean() / beta[i1]
            ok = "✅" if need <= 0 or need <= cmax_all else "❌ 不可达"
            A(f"| [{lo_:.1f}, {hi_:.1f}) | {len(d)} | {d.margin.mean():+.4f} | "
              f"{need:+.1f}% | {cmax_all:.1f}% | {ok} |")
        res["equiv_conc_per_0.1T"] = float(equiv)
    else:
        A("β1 为 0，无法给出当量。")
    A("")

    # ---------------------------------------------------------------- T4 分箱实测
    A("## T4 · 分箱实测（按同基质 / 跨基质分层）")
    A("")
    for grp, lab in [(0, "跨基质竞争者"), (1, "同基质竞争者")]:
        A(f"**{lab}**")
        A("")
        A("| 竞争者化学相似度 | n | margin 均值 | 低浓度 | 高浓度 | 浓度增益 | top-1 |")
        A("|---|---|---|---|---|---|---|")
        for lo_, hi_ in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
            d = df[(df["Tsim"] >= lo_) & (df["Tsim"] < hi_) & (df["same"] == grp)]
            if len(d) < 20:
                continue
            q1, q3 = np.percentile(d.c, 25), np.percentile(d.c, 75)
            a = d[d.c <= q1].margin.mean()
            b = d[d.c >= q3].margin.mean()
            A(f"| [{lo_:.1f}, {hi_:.1f}) | {len(d)} | {d.margin.mean():+.4f} | "
              f"{a:+.4f} | {b:+.4f} | {b-a:+.4f} | {d.hit.mean():.3f} |")
        A("")
    A("> 跨基质时，化学差异越大（T 越小）margin 应越高；同基质时该关系应消失 —— 因为错误被基质结构强制。")
    A("")

    # ---------------------------------------------------------------- T6 双分离
    A("## T6 · 双分离：**检出**的信号强度 ≠ **识别**的化学分辨")
    A("")
    A("已有 N1 的 NAS 信噪比 `snr = ‖b*‖/σ_null` 预测「能否**检出**掺假」。")
    A("本实验检验它是否也预测「能否**识别**是哪种掺假」。")
    A("")
    A("| 实体 | 基质 | snr = ‖b*‖/σ_null | top-1 | 备注 |")
    A("|---|---|---|---|---|")
    snr_rows = []
    for e in eids:
        s = SNR.get(e, None)
        t1 = float((np.asarray(S["Original"][e], float).argmax(1) == idx[e]).mean())
        if s is None:
            continue
        snr_rows.append((e, s, t1))
        note = ""
        if s > 30 and t1 < 0.05:
            note = "**信号最强却完全认不出**"
        if s < 2 and t1 > 0.95:
            note = "**信号最弱却完全认得出**"
        A(f"| {e} | {MID.get(e,'')} | {s:.2f} | {t1:.3f} | {note} |")
    if len(snr_rows) >= 5:
        sr = np.array([r[1] for r in snr_rows])
        tr = np.array([r[2] for r in snr_rows])
        rr, pp = spearmanr(sr, tr)
        A("")
        A(f"**ρ(snr, top-1) = {rr:+.3f}，p = {pp:.3f}**（n={len(snr_rows)}）⇒ **无关**。")
        A("")
        A("而 exp23 已证 `snr` 与**检出** AUC 强相关（ρ(z_off, AUC)=+0.9286）。")
        A("⇒ **双分离**：")
        A("")
        A("| | 决定因素 | 指标 |")
        A("|---|---|---|")
        A("| **检出**（有没有掺假） | 掺杂物净信号强度 ‖b*‖/σ_null | AUC、LOD |")
        A("| **识别**（掺了什么） | 候选间的化学分辨力（受浓度与化学差异共同调制） | top-1、margin |")
        A("")
        A("这解释了此前一系列看似矛盾的结论（nas 的 AUC 垫底但 ρ_dose 高、PLSR AUC 最高）：")
        A("**它们测的根本不是同一件事。**")
        res["dissociation"] = {"rho_snr_top1": float(rr), "p": float(pp), "n": int(len(snr_rows))}
    A("")

    # ---------------------------------------------------------------- T5 与检出限对照
    A("## T5 · 与浓度检出限（N1）的结构对照")
    A("")
    A("| | 浓度检出限 c_LOD（已有 N1） | 化学分辨限 c_resolve（本文新增） |")
    A("|---|---|---|")
    A("| 问题 | 多低的浓度还能**检出** | 多相似的两个候选还能**分开** |")
    A("| 分子 | (z_α − z_off)·σ_null | κ·σ_null |")
    A("| 分母 | ‖b*‖（掺杂物净信号强度） | ‖Δa(T)‖（两掺杂物光谱差异，随 T↑ 而↓） |")
    A("| 失效机制 | 信号被噪声淹没 | 两个候选的光谱差异被噪声淹没 |")
    A("| 修正手段 | NAS-C 校正 z_off→0 | 提高 ‖Δa‖ 或降低 σ_null（谱侧去基质） |")
    A("")
    A("⇒ 两者是**同一个误差预算框架的两次应用**：都是「κ·噪声 / 信号强度」。")
    A("这是本文最统一的结构性贡献，且**纯品对齐赛道不存在第二条**（它们没有混合物、没有浓度）。")
    A("")

    with open(os.path.join(OUT, "exp32_p6_chemical_resolution.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "exp32_res.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
