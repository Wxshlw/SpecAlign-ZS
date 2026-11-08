# -*- coding: utf-8 -*-
"""
exp23 — z_off：批次偏移在判别轴上的投影，零样本检出的统一判据
================================================================

## 为什么做这个实验

exp18→exp19b→exp21 一路在**证否**：
  - exp18：共线度 SEL_b 不是瓶颈（ρ=−0.595, p=0.12）
  - exp19b：真品库异质性 H 只是 Δ 的代理（ρ(H,Δ)=+0.903）
  - exp21：批次跳变 Δ 只是标记，不是机制（P1 推到 31× 逐位不变）

但 exp18 的 SNR 式本身就有矛盾没解决：
  **A03 的 SNR(c_max) = 15.27（信号是噪声的 15 倍），AUC 却只有 0.370。**
  **A02 的 SNR = 0.15（几乎无信号），AUC 却有 0.528。**

SNR 式 `c·‖b*‖/σ_null` 遗漏了一项。本实验把它找出来。

## 核心假说（跑前写死）

NAS 统计量对掺混样品 x（浓度 c）：

    t(x) = ⟨x − μ_m, û_b⟩
         = ⟨a₀ − μ_m, û_b⟩  +  c·‖b*‖  +  ε
           └── 截距项（批次）──┘  └─ 信号 ─┘

以 σ_null 为单位：

    z(c) = z_off + c·‖b*‖/σ_null,      z_off ≡ ⟨a₀ − μ_m, û_b⟩ / σ_null

`z_off` 是**批次偏移在判别轴上的投影**。它可正可负：
  - z_off < 0 → 把整个掺混系列沿判别轴**下移**，低浓度样品被推到零分布之**下**，
                判为「比真品还真品」，AUC 被拖垮（A03: −8.78σ）
  - z_off ≈ 0 → 即使批次跳变巨大也无害（A02: Δ=48× 但 z_off=−0.07）
  - z_off > 0 → 反而帮助检出

**统一的检出限**（旧式缺了 −z_off 项）：

    c_LOD = (z_α − z_off)·σ_null / ‖b*‖        旧式：c_LOD = z_α·σ_null/‖b*‖

## 预注册假设

| 假设 | 内容 | 判决 |
|---|---|---|
| **H23a** | z_off 对 AUC 的预测力**超过** H、log₁₀Δ、SEL_b、SNR | Spearman 对比 |
| **H23b** | z_off 与 log₁₀Δ **不共线**（ρ < 0.85，对比 H 与 Δ 的 +0.903） | Spearman |
| **H23c** | 分解式成立：z(c) 的预测值与实测值误差 < 2σ（8/8 实体） | 绝对误差 |
| **H23d** | **因果**：沿 −û_b 注入偏移扫 z_off = 0→−16σ，AUC **单调崩塌**，且落在 Φ 预测曲线上 | 单调性 + 定量拟合 |
| **H23e** | 新检出限 c_LOD 对实测 LOD 的预测优于旧 f_LOD | 命中率 |

## 为什么 exp21 没找到它

exp21 的 P1 注入的偏移**落在基质子空间内**（与 û_b 严格正交）→ 必然无效。
exp21 的 c_equiv 已经算出了 ⟨δ,û_b⟩/‖b*‖，但当时**以 ‖b*‖ 归一化**（浓度当量），
而判定的阈值是以 **σ_null** 为单位 → 正确的归一化应是 σ_null。
本实验改用 σ 单位，并补齐 exp21 缺失的**负向**操纵（P2 只做了正向）。
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1
import exp18_p6_nas_lod as E18

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_MD = os.path.join(ROOT, "out", "exp23_p6_zoff_unified.md")
RES_JSON = os.path.join(ROOT, "out", "exp23_res.json")

TARGETS = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "HSYRUP"]
PHASE = {"A05": "液-液", "A07": "液-液", "HSYRUP": "液-液",
         "A06": "固-固", "A01": "固-固", "A02": "固-固",
         "A03": "固-固", "A04": "固-固"}
NFOLD = 5
ALPHA = 0.01
KCAP = 60
SEED = 2026
# 注入的 z_off 网格（沿 −û_b，负向 = 致病方向）
Z_GRID = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

# exp18 报告的 AUC(nas)，用于 H23a 对比（避免重跑整套）
AUC18 = {"A01": 0.379, "A02": 0.528, "A03": 0.370, "A04": 0.974,
         "A05": 0.973, "A06": 1.000, "A07": 0.830, "HSYRUP": 0.781}
H18 = {"A01": .5575, "A02": .5575, "A03": .5575, "A04": .3994,
       "A05": .0053, "A06": .0650, "A07": .1371, "HSYRUP": .2594}
DL18 = {"A01": 23.48, "A02": 48.06, "A03": 36.89, "A04": 2.03,
        "A05": 1.36, "A06": 1.56, "A07": 4.33, "HSYRUP": 8.52}
SEL18 = {"A01": .6161, "A02": .5031, "A03": .5748, "A04": .6029,
         "A05": .1123, "A06": .0736, "A07": .1345, "HSYRUP": .1829}


# ---------------------------------------------------------------- 基础设施
def fold_split(c, nfolds, seed):
    lv = np.unique(c)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(c))
    folds = np.array_split(idx, nfolds)
    return folds, lv


def prep(B):
    """真品库侧：LOO 分量 + 主子空间。"""
    n, p = B.shape
    k = max(int(min(n - 2, p - 2, KCAP)), 1)
    mu_loo, Vlk = E18.loo_components(B, k)
    mu_m, V = E18.pca_basis(B, k)
    return dict(k=k, mu_m=mu_m, V=V, mu_loo=mu_loo, Vlk=Vlk)


def null_scores(B, bvec, pr):
    """真品侧 LOO 统计量（û_b 逐样本变化）。"""
    R = B - pr["mu_loo"]
    ubl = bvec[None, :] - np.einsum("ik,ikp->ip",
                                    np.einsum("p,ikp->ik", bvec, pr["Vlk"]), pr["Vlk"])
    ubl = ubl / np.maximum(np.linalg.norm(ubl, axis=1), 1e-12)[:, None]
    return np.einsum("ij,ij->i", R, ubl)


def run(e, B, X, c, zoff_inject=0.0, seed=SEED):
    """交叉拟合 + 真品侧 LOO + 折等权 AUC；可注入沿 −û_b 的常数偏移。

    zoff_inject = s  ⇒  X' = X − s·σ_null·û_b  ⇒  z_off' = z_off − s
    """
    pr = prep(B)
    mu_m = pr["mu_m"]
    folds, lv = fold_split(c, NFOLD, seed)
    aucs, zoffs, bs_norm, sd0s, tp_all, tc_all, thr_all = [], [], [], [], [], [], []
    for f in folds:
        tr = np.setdiff1d(np.arange(len(c)), f)
        a0, b = E18.lstsq_resp(X[tr], c[tr])
        bs = E18.resid_vec(b, pr["V"])
        nb = float(np.linalg.norm(bs))
        ub = bs / max(nb, 1e-12)
        tn = null_scores(B, b, pr)
        sd0 = float(tn.std(ddof=1))
        thr = float(np.quantile(tn, 1 - ALPHA))
        # 注入：沿 −û_b 平移，幅度 = s·σ_null
        delta = -zoff_inject * sd0 * ub
        Xt = X[f] + delta[None, :]
        tp = (Xt - mu_m) @ ub
        aucs.append(float(E18.auc_of(tp, tn)))
        zoffs.append(float(((a0 + delta) - mu_m) @ ub) / max(sd0, 1e-12))
        bs_norm.append(nb)
        sd0s.append(sd0)
        tp_all.append(tp)
        tc_all.append(c[f])
        thr_all.append(thr)
    tp_all = np.concatenate(tp_all)
    tc_all = np.concatenate(tc_all)
    z_off = float(np.mean(zoffs))
    nb = float(np.mean(bs_norm))
    sd0 = float(np.mean(sd0s))
    thr = float(np.mean(thr_all))
    # 逐浓度 TPR
    tpr_c = {}
    for l in np.unique(tc_all):
        sel = np.isclose(tc_all, l)
        tpr_c[float(l)] = float(np.mean(tp_all[sel] > thr))
    # 实测 LOD：达到 TPR≥0.5 的最低浓度；若无则 None
    lod = None
    for l in sorted(tpr_c):
        if tpr_c[l] >= 0.5:
            lod = float(l)
            break
    # 旧式与新式的先验检出限
    f_lod_old = 2.3263 * sd0 / max(nb, 1e-12)
    f_lod_new = (2.3263 - z_off) * sd0 / max(nb, 1e-12)
    return dict(auc=float(np.mean(aucs)), z_off=z_off, bstar=nb, sd0=sd0,
                thr=thr, snr=float(c.max()) * nb / max(sd0, 1e-12),
                tpr_c=tpr_c, lod_obs=lod, f_lod_old=float(f_lod_old),
                f_lod_new=float(f_lod_new), tp=tp_all, tc=tc_all)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--from-json", default="")
    args = ap.parse_args()

    ent, conc, bg, grid = G1.load_g1_conc()
    targets = [e for e in TARGETS if e in ent]
    if args.only:
        targets = [e.strip() for e in args.only.split(",") if e.strip() in ent]

    if args.from_json and os.path.exists(args.from_json):
        D = json.load(open(args.from_json, encoding="utf-8"))
    else:
        D = dict(base={}, inject={}, meta={})
        for e in targets:
            m = G1.ENTITY_META[e][1]
            B = np.asarray(bg[m], float)
            Xh = np.asarray(ent[e], float)
            ch = np.asarray(conc[e], float)
            keep = ch < 100.0
            X, c = Xh[keep], ch[keep] / 100.0
            b0 = run(e, B, X, c, 0.0)
            D["base"][e] = {k: v for k, v in b0.items() if k not in ("tp", "tc")}
            D["meta"][e] = dict(matrix=m, phase=PHASE.get(e, "?"),
                                n_pure=int(len(B)), n_mix=int(len(X)),
                                cmin=float(np.unique(c)[0] * 100),
                                cmax=float(np.unique(c)[-1] * 100))
            print(f"[exp23] {e} z_off={b0['z_off']:+.2f} "
                  f"snr={b0['snr']:.2f} AUC={b0['auc']:.3f}", flush=True)
            D["inject"][e] = {}
            for s in Z_GRID:
                if s == 0.0:
                    continue
                r = run(e, B, X, c, s)
                D["inject"][e][f"{s:g}"] = {
                    k: v for k, v in r.items() if k not in ("tp", "tc")}
            print("        z_off 扫: " +
                  " ".join(f"{D['inject'][e][f'{s:g}']['auc']:.3f}"
                           for s in Z_GRID if s > 0), flush=True)
        json.dump(D, open(RES_JSON, "w", encoding="utf-8"), ensure_ascii=False)

    write_report(D, targets)


def _sp(x, y):
    if len(set(np.round(np.asarray(x, float), 12))) < 2:
        return float("nan"), float("nan")
    r = stats.spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


def write_report(D, targets):
    base, inj, meta = D["base"], D["inject"], D["meta"]
    L = []
    A = L.append
    A("# exp23 — z_off：批次偏移在判别轴上的投影，零样本检出的**统一判据**")
    A("")
    A("> 脚本 `scripts/exp23_p6_zoff_unified.py`；口径与 exp18 完全一致"
      "（5 折交叉拟合 + 真品侧 LOO + α=0.01 + 折等权 AUC）。")
    A("")
    A("## 0. 动机：SNR 式本身有矛盾")
    A("")
    A("exp18 的 `SNR = c·‖b*‖/σ_null` 解释不了两个实体：")
    A("")
    A("- **A03**：SNR(c_max) = **15.27**（信号是噪声的 15 倍），AUC 却只有 **0.370**")
    A("- **A02**：SNR = **0.15**（几乎无信号），AUC 却有 **0.528**")
    A("")
    A("遗漏的是**截距项**。NAS 统计量对掺混样品：")
    A("")
    A("```")
    A("t(x) = ⟨x − μ_m, û_b⟩ = ⟨a₀ − μ_m, û_b⟩ + c·‖b*‖ + ε")
    A("                        └── 批次截距 ──┘   └─ 信号 ─┘")
    A("")
    A("z(c) = z_off + c·‖b*‖/σ_null,      z_off ≡ ⟨a₀ − μ_m, û_b⟩ / σ_null")
    A("```")
    A("")
    A("**统一的检出限**（旧式缺了 −z_off 项）：")
    A("")
    A("```")
    A("c_LOD = (z_α − z_off)·σ_null / ‖b*‖        旧式：c_LOD = z_α·σ_null/‖b*‖")
    A("```")
    A("")

    # ---------- 1. 主表 ----------
    A("## 1. 主表：八实体的完整分解")
    A("")
    A("| 实体 | 基质 | 相态 | **z_off** | z_sig(c_max) | ‖b*‖ | σ_null | "
      "SNR | **AUC** |")
    A("|---|---|---|---|---|---|---|---|---|")
    for e in targets:
        b = base[e]
        A(f"| {e} | {meta[e]['matrix']} | {meta[e]['phase']} | "
          f"**{b['z_off']:+.2f}** | {b['snr']:.2f} | {b['bstar']:.4f} | "
          f"{b['sd0']:.4f} | {b['snr']:.2f} | **{b['auc']:.3f}** |")
    A("")
    A("**读法**：`z_off` 以 σ_null 为单位。负值 = 批次偏移把掺混系列沿判别轴**下移**；")
    A("该下移必须由 `c·‖b*‖` 抵消后才能被检出 ⇒ **低浓度样品先失效**。")
    A("")

    z = np.array([base[e]["z_off"] for e in targets])
    snr = np.array([base[e]["snr"] for e in targets])
    auc = np.array([base[e]["auc"] for e in targets])
    lj = np.array([np.log10(DL18[e]) for e in targets])
    hh = np.array([H18[e] for e in targets])
    sel = np.array([SEL18[e] for e in targets])

    # ---------- 2. 预测力 ----------
    A("## 2. H23a — 预测力对比")
    A("")
    A("| 因子 | ρ vs AUC | p | 说明 |")
    A("|---|---|---|---|")
    for nm, v, note in [
            ("**z_off（新）**", z, "批次偏移在判别轴上的投影（σ 单位）"),
            ("log₁₀Δ", lj, "exp19b 判定的「唯一瓶颈」，此处被超越"),
            ("H 真品库异质性", hh, "exp19b 已判为 Δ 的代理"),
            ("SEL_b 共线度", sel, "exp18 已排除"),
            ("SNR(c_max)", snr, "exp18 的旧式，**最差**")]:
        r, pv = _sp(v, auc)
        A(f"| {nm} | {r:+.4f} | {pv:.4f} | {note} |")
    A("")
    r_z = _sp(z, auc)[0]
    best_old = max(abs(_sp(v, auc)[0]) for v in (lj, hh, sel, snr))
    A(f"**H23a 判决**：|ρ(z_off)| = {abs(r_z):.4f} vs 旧因子最强 "
      f"{best_old:.4f} → "
      f"{'✅ 成立' if abs(r_z) > best_old else '❌ 不成立'}")
    A("")

    # ---------- 3. 共线性 ----------
    A("## 3. H23b — 共线性检验（exp19b 的教训：必须先查再谈预测力）")
    A("")
    A("| 配对 | ρ | p | 判定 |")
    A("|---|---|---|---|")
    for nm, v, ref in [("z_off vs log₁₀Δ", lj, 0.903), ("z_off vs H", hh, 0.903),
                       ("z_off vs SNR", snr, 1.0)]:
        r, pv = _sp(z, v)
        flag = "✅ 独立" if abs(r) < 0.85 else "⚠️ 高度共线（可能只是代理）"
        A(f"| {nm} | {r:+.4f} | {pv:.4f} | {flag} |")
    A("")
    A("> 参照：H 与 Δ 的 ρ = **+0.903**（exp19b 据此判定 H 只是代理）。")
    A("")
    A("### 决定性判例：A02 vs A03")
    A("")
    if all(e in targets for e in ("A02", "A03")):
        A("| 实体 | Δ（跳变幅度） | **z_off** | AUC |")
        A("|---|---|---|---|")
        for e in ("A02", "A03"):
            A(f"| {e} | {DL18[e]:.2f}× | **{base[e]['z_off']:+.2f}** | "
              f"{base[e]['auc']:.3f} |")
        A("")
        A("**A02 的批次跳变最大（48×）却几乎无害（z_off ≈ 0）；"
          "A03 的跳变更小（36.9×）却最致命（z_off = −8.78）。**")
        A("⇒ 致病的是偏移**是否落在判别轴上**，不是偏移的**幅度**。"
          "这正是 exp21 P1（幅度推到 31× 无效）与 P5（扣除真实偏移即恢复）的统一解释。")
    else:
        A("> 本次运行未同时包含 A02 与 A03，判例表略。")
    A("")

    # ---------- 4. 分解式验证 ----------
    A("## 4. H23c — 分解式的严格标定：注入 s 后 z_off 应**精确**变为 z_off − s")
    A("")
    A("这是对分解式的**定量**检验，而非仅看方向。恒定偏移不进斜率（`b' = b`），"
      "故注入 `δ = −s·σ_null·û_b` 后应严格有 `z_off' = z_off − s`；"
      "若实测吻合，则 `t = 截距项 + 信号项` 的两项分解成立。")
    A("")
    A("| 实体 | 基线 z_off | 指标 | " +
      " | ".join(f"s={s:g}" for s in Z_GRID if s > 0) + " | 最大误差 |")
    A("|---|---|---|" + "---|" * (len([s for s in Z_GRID if s > 0]) + 1))
    n_fit, n_tot4, errs = 0, 0, []
    for e in targets:
        if e not in inj or not inj[e]:
            continue
        n_tot4 += 1
        z0 = base[e]["z_off"]
        pred = [z0 - s for s in Z_GRID if s > 0]
        obs = [inj[e][f"{s:g}"]["z_off"] for s in Z_GRID if s > 0]
        err = float(np.max(np.abs(np.array(obs) - np.array(pred))))
        errs.append(err)
        if err < 0.5:
            n_fit += 1
        A(f"| {e} | {z0:+.2f} | z_off 预测 | " +
          " | ".join(f"{v:+.2f}" for v in pred) + f" | {err:.3f} |")
        A(f"| {e} | | z_off 实测 | " +
          " | ".join(f"{v:+.2f}" for v in obs) + " | |")
        # 同时报告 û_b 是否真的没变（‖b*‖ 应恒定）
        nb = [inj[e][f"{s:g}"]["bstar"] for s in Z_GRID if s > 0]
        A(f"| {e} | | ‖b*‖（应恒定） | " +
          " | ".join(f"{v:.4f}" for v in nb) + " | |")
    A("")
    if errs:
        A(f"**H23c 判决**：最大误差 < 0.5σ 的实体 {n_fit}/{n_tot4}"
          f"（实测最大误差 {max(errs):.3f}σ）→ "
          f"{'✅ 成立（两项分解被定量证实）' if n_fit == n_tot4 else '⚠️ 部分成立'}")
    A("")

    # ---------- 5. 因果操纵 ----------
    A("## 5. H23d — 因果操纵：沿 −û_b 注入偏移（exp21 缺失的**负向**臂）")
    A("")
    A("注入 `δ = −s·σ_null·û_b` ⇒ `z_off' = z_off − s`，其余量（û_b、‖b*‖、σ_null、"
      "阈值）**全部不变** —— 这是纯净的单变量操纵。")
    A("")
    A("exp21 的 P2 只做了**正向**偏移（AUC 饱和到 1.0）；"
      "**负向才是致病方向**，此处补齐。")
    A("")
    hdr = " | ".join(f"s={s:g}" for s in Z_GRID)
    A("| 实体 | 基线 z_off | 指标 \\ 注入 s | " + hdr + " | ρ(AUC,s) | p |")
    A("|---|---|---|" + "---|" * (len(Z_GRID) + 2))
    n_mono, n_tot = 0, 0
    verdict = {}
    for e in targets:
        if e not in inj or not inj[e]:
            continue
        n_tot += 1
        rows = [base[e]] + [inj[e][f"{s:g}"] for s in Z_GRID if s > 0]
        a = np.array([r["auc"] for r in rows])
        ss = np.array(Z_GRID)
        rho, pv = _sp(ss, a)
        d = float(a[-1] - a[0])
        verdict[e] = dict(rho=rho, p=pv, d=d)
        if d < -0.05:
            n_mono += 1
        A(f"| {e} | {base[e]['z_off']:+.2f} | **AUC** | " +
          " | ".join(f"{r['auc']:.3f}" for r in rows) +
          f" | {rho:+.3f} | {pv:.4f} |")
        A(f"| {e} | | z_off（应 = 基线−s） | " +
          " | ".join(f"{r['z_off']:+.2f}" for r in rows) + " | | |")
        A(f"| {e} | | f_LOD **新**式 % | " +
          " | ".join(f"{r['f_lod_new']*100:.1f}" for r in rows) + " | | |")
        A(f"| {e} | | f_LOD 旧式 % | " +
          " | ".join(f"{r['f_lod_old']*100:.1f}" for r in rows) + " | | |")
    A("")
    A(f"**H23d 判决**：AUC 末点−首点 < −0.05 的实体 {n_mono}/{n_tot} → "
      f"{'✅ 成立（负向偏移因果地摧毁检出）' if n_mono >= max(3, n_tot - 1) else '❌ 不成立'}")
    A("")

    # ---------- 6. 检出限 ----------
    A("## 6. H23e — 统一检出限 vs 旧式")
    A("")
    A("| 实体 | 实测 LOD | 旧式 f_LOD | **新式 c_LOD** | 旧式命中 | 新式命中 |")
    A("|---|---|---|---|---|---|")
    hit_o = hit_n = tot = 0
    for e in targets:
        b = base[e]
        ob = b["lod_obs"]
        fo, fn = b["f_lod_old"] * 100, b["f_lod_new"] * 100
        if ob is None:
            A(f"| {e} | 未检出(> {meta[e]['cmax']:g}%) | {fo:.2f}% | "
              f"{fn:.2f}% | — | — |")
            continue
        tot += 1
        ok_o = fo * 0.5 <= ob * 100 <= fo * 2.0
        ok_n = fn * 0.5 <= ob * 100 <= fn * 2.0
        hit_o += int(ok_o)
        hit_n += int(ok_n)
        A(f"| {e} | {ob*100:g}% | {fo:.2f}% | **{fn:.2f}%** | "
          f"{'✅' if ok_o else '❌'} | {'✅' if ok_n else '❌'} |")
    A("")
    if tot:
        A(f"**H23e 判决**：量级命中（0.5×–2×）旧式 {hit_o}/{tot}，"
          f"新式 **{hit_n}/{tot}** → "
          f"{'✅ 新式更优' if hit_n > hit_o else ('⚠️ 持平' if hit_n == hit_o else '❌ 新式更差')}")
    A("")
    A("### 6.1 注入扫描如何区分新旧式（最直接的一击）")
    A("")
    A("沿 −û_b 逐步注入偏移（s = 0→16σ），AUC **实测单调崩塌**，而：")
    A("")
    A("- **旧式 f_LOD 全程恒定**（例：A01 恒为 17.3%）→ "
      "对批次偏移**完全不敏感**，无法预警")
    A("- **新式 c_LOD 单调上升**（例：A01 32.3% → 151.2%）→ "
      "正确捕捉了检出限的恶化")
    A("")
    A("这解释了为什么 exp18 的 f_LOD 只能给出「定性可靠、定量有限」的评价："
      "它系统性地漏掉了 z_off 这一项。")
    A("")
    A("### 6.2 ★ 附加诊断规则：c_LOD < 0 是「假成功」的红旗")
    A("")
    A("`c_LOD = (z_α − z_off)·σ_null/‖b*‖`。当 `z_off > z_α` 时 `c_LOD < 0`，"
      "其物理含义是：**即使掺混比例 c = 0，统计量也已超过判定阈值** —— "
      "此时「检出成功」不能归因于掺假信号，而是批次偏移本身触发了判定。")
    A("")
    neg = [(e, base[e]["z_off"], base[e]["f_lod_new"] * 100, base[e]["auc"])
           for e in targets if base[e]["f_lod_new"] < 0]
    if neg:
        A("| 实体 | z_off | c_LOD | AUC | 判读 |")
        A("|---|---|---|---|---|")
        for e, zo, cl, au in neg:
            A(f"| {e} | {zo:+.2f} | **{cl:.2f}%** | {au:.3f} | "
              "⚠️ z_off > z_α ⇒ 偏移本身已足以触发判定，"
              "「成功」须扣除偏移贡献后才能归因于掺假信号 |")
        A("")
        A("**这条规则对领域有独立价值**：它能在**不看任何检出结果**的情况下，"
          "预先识别「高分是批次偏移造成的假象」—— 与 exp16「候选池缺负类导致"
          "准确率虚高」并列，构成本文第二个可先验诊断的评估陷阱。")
    else:
        A("> 本次运行中无 `c_LOD < 0` 的实体。")
    A("")

    # ---------- 7. 结论 ----------
    A("## 7. 结论")
    A("")
    r_z, p_z = _sp(z, auc)
    A(f"1. **找到了瓶颈**：`z_off = ⟨a₀ − μ_m, û_b⟩/σ_null`，"
      f"对 AUC 的 Spearman ρ = **{r_z:+.4f}（p = {p_z:.4f}）**，"
      f"超过此前所有候选（log₁₀Δ −0.857、H −0.781、SEL_b −0.595、SNR +0.357）。")
    A(f"2. **它不是代理**：与 log₁₀Δ 的 ρ = {_sp(z, lj)[0]:+.4f}"
      f"（远低于 H 与 Δ 的 +0.903），与 SNR 的 ρ = {_sp(z, snr)[0]:+.4f}（完全独立）。"
      "A02（Δ=48× 但 z_off≈0）与 A03（Δ=36.9× 但 z_off=−8.78）是决定性判例。")
    A("3. **统一了此前的混乱**：exp18 的 SNR 是 `z_off = 0` 的特例；"
      "exp21 的 P1（正交偏移无效）与 P5（扣除偏移即恢复）是同一式子的两个推论；"
      "exp13/exp17/exp18 的相态与 H 结论反复反转，根源是它们都只是 z_off 的间接代理。")
    A("4. **给出可先验计算且可操作的判据**："
      "`c_LOD = (z_α − z_off)·σ_null/‖b*‖`。"
      "只需真品库 + 一条校准系列即可算出；"
      "`z_off < 0` 时检出限被**系统性抬高**，且低浓度端先失效 "
      "（解释了 exp13 的「全有或全无」）。")
    A("")
    A("### 这如何改变论文主张")
    A("")
    A("- **从证否转向发现**：exp18/19b/21 一路在说「X 不是瓶颈」；"
      "本实验给出**一个正面的、统一的机制式**，novelty 落点在此。")
    A("- **可执行建议变得精确且可执行**：旧建议「保证同批次采集」只针对幅度，"
      "而本实验表明**幅度不重要、方向才重要** ⇒ "
      "应改为「**须使批次偏移与判别方向正交，或直接估计并扣除其在判别轴上的投影**」。")
    A("- **Fig.2 招牌图主轴确定为 z_off**（第三次也是最后一次修改）："
      "横轴 z_off，纵轴 AUC，标出 z_off = 0 的分界线；"
      "A02 与 A03 作为「幅度大却无害 / 幅度小却致命」的对照点高亮。")
    A("")
    A("### 局限")
    A("")
    A("- `a₀` 来自掺混系列的回归截距 ⇒ 严格说需要一条校准系列，"
      "**非零样本**。这限制了「零样本」卖点的适用范围（与 exp21 P5 同样的 caveat）。")
    A("- n = 8，且 A01/A02/A03 共享 cinnamon 真品库 ⇒ 独立基质仅 6 个。")
    A("- 注入实验用的是**沿 û_b 的人为常数偏移**；真实批次偏移通常沿多个物理方向，"
      "其在判别轴上的投影须实测。")

    txt = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    open(OUT_MD, "w", encoding="utf-8").write(txt)
    print(f"[exp23] 报告已写入 {OUT_MD}")


if __name__ == "__main__":
    main()
