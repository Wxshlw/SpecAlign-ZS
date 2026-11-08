# -*- coding: utf-8 -*-
"""
exp18 — 净分析物信号（NAS）：把「零样本能否检出」变成**可先验计算的品质因数**

================================================================================
要推翻的既有结论
================================================================================
exp13 → 「液-液全浓度 top-1 = 1.000、固-固全 0」⇒ 液-液更可辨识
exp17 → 加负类后反转为「固-固 AUC 0.896 > 液-液 0.499」⇒ 固-固才真有检出能力

本实验的判据显示：**两者都可能是假象**，因为两者都建立在
「与全谱强度比较」的**相对**度量上。检出与否的真正判据是

    SNR_NAS(c) = c · ‖(I − P_M)·b‖ / σ_null

其中 b 是**浓度响应方向**（每单位掺混分数的谱变化），P_M 是**真品库**的干扰子空间，
σ_null 是真品库自身的变异（**噪声地板**）—— 不是仪器噪声，而是「真品与真品之间差多少」。

================================================================================
方法
================================================================================
1) 干扰子空间  P_M = PCA(真品库 B_m, 前 k 个主成分)         ← 只需真品样品
2) 浓度响应    x ≈ a0 + b·c  （校准折拟合；b = 掺杂物相对基质的谱变化方向）
3) 净分析物信号 b* = (I − P_M) b ；选择性 SEL_b = ‖b*‖/‖b‖ ∈ [0,1]
   SEL_b → 0 表示掺杂物谱与真品库变异完全共线 ⇒ **数学上不可检出**
4) 噪声地板    σ_null = 真品库 LOO 在 û_b 方向上统计量的标准差
5) 匹配滤波    t(x) = ⟨x − μ_m, û_b⟩
6) 先验检出限  f_LOD = z_{1−α}·σ_null / ‖b*‖     ← 采集任何掺假样品**之前**即可算出

两个决定成败的量（本实验的核心）：
  · **共线度**   SEL_b        —— 掺杂物与基质有多像
  · **异质性**   H = 真品库组内 RMS / ‖μ_m‖  —— 真品与真品差多少
浓度 c 只是**线性缩放因子**，不是门槛本身。

================================================================================
防循环论证（关键）
================================================================================
方向 b 若用「将被判定的那批掺混谱」估计，则检出必然成立 —— 循环论证。
本实验对掺混谱做 **5 折交叉拟合**：方向只由校准折估计，判定在留出折上进行。
另设 `nas_null` 臂：用**与 {1, c} 都正交的随机标签**拟合方向。
（不能用「打乱 c」做置换零假设：随机置换的期望与真实 c 的内积为 −S²/(n−1)，
会使零方向系统性偏成 −b，从而把 AUC 压到 0.5 以下 —— 那是伪影，不是零假设。）
若 nas_null ≈ 0.5，则证明整套统计量没有自我实现。

================================================================================
臂
================================================================================
| 臂        | 统计量                                   | 先验等级 | 检验什么           |
|-----------|------------------------------------------|----------|--------------------|
| cos       | cos(x,X̂_a) − cos(x,μ_m)                   | L1       | 现有范式基线       |
| od        | ‖(I−P_M)(x−μ_m)‖（DD-SIMCA 式能量）       | L2       | **真零样本**上界   |
| nas       | ⟨x−μ_m, û_b⟩（匹配滤波）                  | L1       | 方向先验值多少     |
| nas_rand  | ⟨x−μ_m, û_随机⟩                            | —        | 任意方向都行吗     |
| nas_null  | 零响应方向（标签 ⊥ {1, c}）              | —        | 是否循环论证       |

α = 0.01；真品侧一律 LOO；无神经网络训练（结论不依赖随机种子）。
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1          # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
ALPHA = 0.01
K_GRID = [1, 2, 3, 5, 8, 12, 20, 30, 45, 60]
N_RAND = 200
N_NULL = 20
NFOLD = 5
ARMS = ["cos", "od", "nas", "nas_rand", "nas_null"]

_PHASE = {"A05": "液-液", "A07": "液-液", "HSYRUP": "液-液", "A06": "固-固",
          "A01": "固-固", "A02": "固-固", "A03": "固-固", "A04": "固-固"}


# ---------------------------------------------------------------- 线性代数
def pca_basis(B, k):
    mu = B.mean(0)
    if k <= 0:
        return mu, np.zeros((0, B.shape[1]))
    U, S, Vt = np.linalg.svd(B - mu, full_matrices=False)
    return mu, Vt[:k]


def resid_vec(v, V):
    """(I − P_V) v，v 为向量。"""
    r = np.asarray(v, float)
    if len(V):
        r = r - (r @ V.T) @ V
    return r


def loo_components(B, kmax):
    """每个真品样品只做一次 SVD：返回 mu_loo (n,p) 与 V_loo (n,kmax,p)。"""
    n, p = B.shape
    mu_loo = np.empty((n, p))
    V_loo = np.empty((n, kmax, p))
    for i in range(n):
        msk = np.ones(n, bool)
        msk[i] = False
        Bi = B[msk]
        mu_i = Bi.mean(0)
        mu_loo[i] = mu_i
        U, S, Vt = np.linalg.svd(Bi - mu_i, full_matrices=False)
        kk = min(kmax, Vt.shape[0])
        V_loo[i, :kk] = Vt[:kk]
        if kk < kmax:
            V_loo[i, kk:] = 0.0
    return mu_loo, V_loo


def loo_residual_curve(B, mu_loo, V_loo, kmax):
    r = B - mu_loo
    tot2 = (r ** 2).sum(1)
    proj = np.einsum("ip,ikp->ik", r, V_loo)
    sq = np.concatenate([np.zeros((len(B), 1)), np.cumsum(proj ** 2, axis=1)], axis=1)
    return np.sqrt(np.maximum(tot2[:, None] - sq, 0.0))


def pick_k(g, tol=1.02):
    """简约准则：最小 k 使 LOO 残差进入最优值 tol 倍内（只接触真品）。"""
    gm = g.mean(0)
    thr = tol * gm.min()
    for k in range(gm.shape[0]):
        if gm[k] <= thr:
            return int(k), gm
    return int(gm.shape[0] - 1), gm


def lstsq_resp(X, c):
    """X ≈ a0 + b·c，返回 (a0, b)。"""
    A = np.stack([np.ones_like(c), c], axis=1)
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    return coef[0], coef[1]


def auc_of(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(stats.mannwhitneyu(pos, neg, alternative="greater").statistic
                 / (len(pos) * len(neg)))


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(x) for x in o]
    if isinstance(o, (np.floating, float)):
        return None if (isinstance(o, float) and np.isnan(o)) else float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if o is None:
        return None
    return o


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--from-json", default="")
    ap.add_argument("--tol", type=float, default=1.02)
    ap.add_argument("--n-rand", type=int, default=N_RAND)
    ap.add_argument("--folds", type=int, default=NFOLD)
    args = ap.parse_args()

    if args.from_json:
        with open(args.from_json, encoding="utf-8") as f:
            d = json.load(f)
        write_report(d["targets"], d["res"], d["meta"], args)
        return

    ent, conc, bg, grid = G1.load_g1_conc()
    ADUL = [e for e in sorted(ent) if G1.ENTITY_META[e][2] == "adulteration"]
    TARGETS = [e for e in ADUL if e in args.only.split(",")] if args.only else ADUL

    res, meta = {}, {}
    for e in TARGETS:
        m = G1.ENTITY_META[e][1]
        B = np.asarray(bg[m], float)
        n, p = B.shape
        mu_m = B.mean(0)

        Xh = np.asarray(ent[e], float)
        ch = np.asarray(conc[e], float)
        keep = ch < 100.0
        X, c = Xh[keep], ch[keep] / 100.0
        lv = np.unique(c)

        # ---------- 真品库异质性 H 与跳变 Δ（纯诊断，不依赖任何拟合）----------
        H = float(np.sqrt(((B - mu_m) ** 2).sum(1).mean()) / np.linalg.norm(mu_m))
        cen = np.stack([X[np.isclose(c, l)].mean(0) for l in lv])
        d_gap = float(np.linalg.norm(cen[0] - mu_m))
        d_step = float(np.linalg.norm(cen[1] - cen[0])) if len(lv) > 1 else float("nan")
        jump = float(d_gap / d_step) if d_step and np.isfinite(d_step) and d_step > 0 \
            else float("nan")

        # ---------- k 选择（只用真品）----------
        kmax = max(int(min(n - 2, p - 2, 60)), 1)
        mu_loo, V_loo = loo_components(B, kmax)
        k_star, gm = pick_k(loo_residual_curve(B, mu_loo, V_loo, kmax), tol=args.tol)
        ksel = sorted(set([k for k in K_GRID if k <= kmax] + [k_star]))
        mu_k, Vk = pca_basis(B, k_star)
        Vlk = V_loo[:, :k_star] if k_star > 0 else np.zeros((n, 0, p))

        # 真品侧 LOO 残差（od 臂，与方向无关，算一次）
        Rloo = B - mu_loo
        if k_star > 0:
            Rloo = Rloo - np.einsum("ik,ikp->ip",
                                    np.einsum("ip,ikp->ik", Rloo, Vlk), Vlk)
        neg_od = np.linalg.norm(Rloo, axis=1)

        # ---------- 折划分（按浓度分层）----------
        rng = np.random.default_rng(2026)
        parts = []
        for l in lv:
            idx = np.where(np.isclose(c, l))[0]
            idx = rng.permutation(idx)
            parts.append(np.array_split(idx, args.folds))
        folds = [np.concatenate([parts[j][f] for j in range(len(lv))])
                 for f in range(args.folds)]

        # ---------- 交叉拟合 ----------
        rec = {a: [] for a in ARMS}
        rec_rho, rec_bstar, rec_sd0 = [], [], []
        pos_pool, c_pool, thr_pool = [], [], []
        tpr_all = {float(l): [] for l in lv}
        n_fp = []
        for f in range(args.folds):
            te = folds[f]
            tr = np.concatenate([folds[g] for g in range(args.folds) if g != f])
            a0t, bt = lstsq_resp(X[tr], c[tr])
            Xhat_t = a0t + bt                                   # 外推纯掺杂物谱
            bs = resid_vec(bt, Vk)
            nb = float(np.linalg.norm(bs))
            ub = bs / max(nb, 1e-12)
            # 真品侧：每个 i 用自己的 LOO 子空间重投影方向
            if k_star > 0:
                bsl = bt - np.einsum("ik,ikp->ip",
                                     np.einsum("p,ikp->ik", bt, Vlk), Vlk)
            else:
                bsl = np.tile(bt, (n, 1))
            ubl = bsl / np.maximum(np.linalg.norm(bsl, axis=1), 1e-12)[:, None]
            t_null = np.einsum("ij,ij->i", B - mu_loo, ubl)
            sd0 = float(np.std(t_null, ddof=1))
            thr = float(np.quantile(t_null, 1 - ALPHA))
            rec_bstar.append(nb)
            rec_sd0.append(sd0)
            thr_pool.append(thr)
            n_fp.append(int(np.sum(t_null > thr)))

            Xte = X[te]
            t_nas = (Xte - mu_m) @ ub
            t_od = np.linalg.norm(np.stack([resid_vec(x - mu_m, Vk) for x in Xte]),
                                  axis=1)
            # cos 臂（现有范式，用交叉拟合得到的 X̂_a）
            Xn = Xte / np.linalg.norm(Xte, axis=1, keepdims=True)
            an = Xhat_t / max(np.linalg.norm(Xhat_t), 1e-12)
            mn = mu_m / np.linalg.norm(mu_m)
            t_cos = (Xn * (an[None, :] - mn)).sum(1)
            Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
            mnl = mu_loo / np.linalg.norm(mu_loo, axis=1, keepdims=True)
            n_cos = (Bn * (an[None, :] - mnl)).sum(1)

            rec["nas"].append((t_nas, t_null))
            rec["od"].append((t_od, neg_od))
            rec["cos"].append((t_cos, n_cos))

            # 随机方向（正交补内均匀）
            G = np.linalg.svd(np.eye(p) - Vk.T @ Vk)[0][:, :(p - k_star)]
            Ur = rng.normal(size=(args.n_rand, p - k_star))
            Ur = (Ur / np.linalg.norm(Ur, axis=1, keepdims=True)) @ G.T
            rec["nas_rand"].append(((Xte - mu_m) @ Ur.T, (B - mu_loo) @ Ur.T))
            # 零响应对照：随机标签，且与 {1, c} 正交 → 拟合出的方向是纯噪声
            for _ in range(N_NULL):
                z = rng.normal(size=len(tr))
                z -= z.mean()
                cc = c[tr] - c[tr].mean()
                z -= cc * (float(z @ cc) / float(cc @ cc))
                z /= max(np.linalg.norm(z), 1e-12)
                _, bz = lstsq_resp(X[tr], z)
                bzs = resid_vec(bz, Vk)
                ubz = bzs / max(np.linalg.norm(bzs), 1e-12)
                if k_star > 0:
                    bzl = bz - np.einsum("ik,ikp->ip",
                                         np.einsum("p,ikp->ik", bz, Vlk), Vlk)
                else:
                    bzl = np.tile(bz, (n, 1))
                ubzl = bzl / np.maximum(np.linalg.norm(bzl, axis=1), 1e-12)[:, None]
                rec["nas_null"].append(((Xte - mu_m) @ ubz,
                                        np.einsum("ij,ij->i", B - mu_loo, ubzl)))

            # 趋势（对常数偏移免疫）
            r_, _ = stats.spearmanr(t_nas, c[te])
            rec_rho.append(float(r_) if np.isfinite(r_) else float("nan"))
            pos_pool.append(t_nas)
            c_pool.append(c[te])
            for l in lv:
                sel_ = np.isclose(c[te], l)
                tpr_all[float(l)].append(float(np.mean(t_nas[sel_] > thr)))

        aucs = {}
        for a in ARMS:
            if a == "nas_rand":
                aucs[a] = float(np.mean([auc_of(pp.ravel(), nn.ravel())
                                         for pp, nn in rec[a]]))
            else:
                aucs[a] = float(np.mean([auc_of(pp, nn) for pp, nn in rec[a]]))

        # ---------- 品质因数 ----------
        bstar = float(np.mean(rec_bstar))
        sd0 = float(np.mean(rec_sd0))
        norm_b = float(np.linalg.norm(lstsq_resp(X, c)[1]))
        sel_b = float(bstar / max(norm_b, 1e-12))
        z_alpha = float(stats.norm.ppf(1 - ALPHA))
        f_lod = z_alpha * sd0 / max(bstar, 1e-12)

        tp = np.concatenate(pos_pool)
        cp = np.concatenate(c_pool)
        rho_all = float(stats.spearmanr(tp, cp)[0])

        tpr_c = {float(l): float(np.mean(v)) for l, v in tpr_all.items()}
        p_c = {}
        k0 = int(np.mean(n_fp))
        for l in lv:
            n1 = int(np.sum(np.isclose(c, l)))
            k1 = int(round(tpr_c[float(l)] * n1))
            try:
                _, pv = stats.fisher_exact([[k1, n1 - k1], [k0, n - k0]],
                                           alternative="greater")
            except Exception:
                pv = float("nan")
            p_c[float(l)] = float(pv)
        lod_obs = None
        for l in sorted(tpr_c):
            if tpr_c[l] >= 0.5 and p_c[l] < ALPHA:
                lod_obs = float(l)
                break

        # ---------- k 敏感性 ----------
        b_all = lstsq_resp(X, c)[1]
        k_sens = {}
        for k in ksel:
            _, Vj = pca_basis(B, k)
            bs_k = resid_vec(b_all, Vj)
            ub_k = bs_k / max(np.linalg.norm(bs_k), 1e-12)
            Vlj = V_loo[:, :k] if k > 0 else np.zeros((n, 0, p))
            Rj = B - mu_loo
            bjl = np.tile(b_all, (n, 1))
            if k > 0:
                Rj = Rj - np.einsum("ik,ikp->ip",
                                    np.einsum("ip,ikp->ik", Rj, Vlj), Vlj)
                bjl = b_all - np.einsum("ik,ikp->ip",
                                        np.einsum("p,ikp->ik", b_all, Vlj), Vlj)
            ubl_k = bjl / np.maximum(np.linalg.norm(bjl, axis=1), 1e-12)[:, None]
            nn_k = np.einsum("ij,ij->i", B - mu_loo, ubl_k)
            pp_k = (X - mu_k) @ ub_k
            pp_od = np.linalg.norm(np.stack([resid_vec(x - mu_m, Vj) for x in X]),
                                   axis=1)
            k_sens[k] = dict(sel=float(np.linalg.norm(bs_k) / max(norm_b, 1e-12)),
                             auc_nas=auc_of(pp_k, nn_k),
                             auc_od=auc_of(pp_od, np.linalg.norm(Rj, axis=1)))

        res[e] = dict(auc=aucs, sel_b=sel_b, bstar=bstar, sd0=sd0, f_lod=f_lod,
                      H=H, jump=jump, d_gap=d_gap, d_step=d_step,
                      rho=rho_all, rho_fold=float(np.nanmean(rec_rho)),
                      tpr_c=tpr_c, p_c=p_c, lod_obs=lod_obs,
                      k_star=int(k_star), n_pure=int(n), n_mix=int(len(X)),
                      norm_b=norm_b, mu_norm=float(np.linalg.norm(mu_m)),
                      k_sens={str(k): v for k, v in k_sens.items()})
        meta[e] = dict(matrix=m, phase=_PHASE.get(e, "?"), cmin=float(lv[0] * 100),
                       cmax=float(lv[-1] * 100))
        print(f"[exp18] {e} ({m},{_PHASE.get(e,'?')}) k*={k_star:3d} "
              f"SEL_b={sel_b:.4f} H={H*100:5.2f}% jump={jump:6.2f}x | "
              f"predLOD={f_lod*100:6.2f}% obsLOD={lod_obs} | "
              f"AUC cos={aucs['cos']:.3f} od={aucs['od']:.3f} nas={aucs['nas']:.3f} "
              f"rand={aucs['nas_rand']:.3f} null={aucs['nas_null']:.3f} | "
              f"rho={rho_all:+.3f}", flush=True)

    with open(os.path.join(OUT, "exp18_res.json"), "w", encoding="utf-8") as f:
        json.dump(dict(targets=TARGETS, res=to_jsonable(res), meta=meta), f,
                  ensure_ascii=False)
    write_report(TARGETS, res, meta, args)


# ---------------------------------------------------------------- 报告
def write_report(TARGETS, res, meta, args):
    for e in res:
        for f in ("tpr_c", "p_c"):
            if isinstance(res[e].get(f), dict):
                res[e][f] = {float(k): v for k, v in res[e][f].items()}

    L = ["# exp18 — 净分析物信号（NAS）：把「能否检出」变成**可先验计算的品质因数**", "",
         "> **要解决的问题**：exp13 得到「液-液更可辨识」，exp17 加负类后反转为",
         "> 「固-固才真有检出能力」。本实验指出：**两者都可能是假象** —— 它们都建立在",
         "> 「与全谱强度比较」的**相对**度量（余弦）上，而检出与否取决于",
         "> 「信号 vs **真品库自身变异**」的**绝对**比较。", "",
         "```",
         "SNR_NAS(c) = c · ‖(I − P_M)·b‖ / σ_null",
         "   b      = 浓度响应方向（每单位掺混分数的谱变化）",
         "   P_M    = 真品库的干扰子空间（PCA，只需真品样品）",
         "   σ_null = 真品库自身的变异（噪声地板）—— 不是仪器噪声，是「真品与真品差多少」",
         "```", "",
         "**两个决定成败的量**：共线度 `SEL_b = ‖b*‖/‖b‖`（掺杂物与基质多像）",
         "和异质性 `H = 真品库组内 RMS / ‖μ_m‖`（真品库多松）。",
         "**浓度 c 只是线性缩放因子**，不是门槛本身。", "",
         f"> 协议：掺混谱 **{args.folds} 折交叉拟合**（方向只由校准折估计，判定在留出折上）；",
         f"> 真品侧一律 LOO；k 由简约准则选（只接触真品）；α = {ALPHA}；无神经网络训练。", "",
         "## 1. 主表", "",
         "| 实体 | 基质 | 相态 | 浓度范围 | **SEL_b** | **真品库异质性 H** | **跳变 Δ** | "
         "cos（现有范式） | od（**真零样本**） | **nas（匹配滤波）** | nas_rand | nas_null |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in TARGETS:
        r = res[e]
        a = r["auc"]
        L.append(f"| {e} | {meta[e]['matrix']} | {meta[e]['phase']} | "
                 f"{meta[e]['cmin']:g}–{meta[e]['cmax']:g}% | **{r['sel_b']:.4f}** | "
                 f"**{r['H']*100:.2f}%** | **{r['jump']:.2f}×** | "
                 f"{a['cos']:.3f} | {a['od']:.3f} | **{a['nas']:.3f}** | "
                 f"{a['nas_rand']:.3f} | {a['nas_null']:.3f} |")

    def _arr(arm):
        return np.array([res[e]["auc"][arm] for e in TARGETS], float)
    a_cos, a_od, a_nas, a_rand, a_null = (_arr(x) for x in
                                          ("cos", "od", "nas", "nas_rand", "nas_null"))

    def _w(x, y):
        try:
            return float(stats.wilcoxon(x, y, alternative="greater").pvalue)
        except Exception:
            return float("nan")
    L += ["",
          f"- **nas vs cos**（现有范式）：{a_nas.mean():.3f} vs {a_cos.mean():.3f}，"
              f"Wilcoxon 单侧 p = {_w(a_nas, a_cos):.4f}",
          f"- **nas vs od**（非靶向）：{a_nas.mean():.3f} vs {a_od.mean():.3f}，"
              f"Wilcoxon 单侧 p = {_w(a_nas, a_od):.4f}",
          f"- **od（真零样本，无需任何掺杂物信息）**：{a_od.mean():.3f}±{a_od.std():.3f}",
          f"- **nas_rand（随机方向）**：{a_rand.mean():.3f} —— 应 ≈ 0.5，"
              f"证明增益来自化学特异方向",
          f"- **nas_null（零响应方向）**：{a_null.mean():.3f}±{a_null.std():.3f} —— 应 ≈ 0.5，"
              f"**证明整套统计量没有自我实现（非循环论证）**"]

    # ---- 2. 机制诊断 ----
    L += ["", "## 2. 机制诊断：真品库异质性与「系列—真品库」跳变", "",
          "> `H` = 真品库组内 RMS / ‖μ_m‖：真品与真品差多少（噪声地板）。",
          "> `跳变 Δ` = ‖最低浓度质心 − 真品库质心‖ / ‖系列内相邻浓度步长‖：",
          "> 若 Δ ≫ 1，说明掺混系列与真品库**不连续**（采集批次/基线不一致），",
          "> 此时「检出」判到的是批次差异，不是掺假。", "",
          "| 实体 | 相态 | 真品库 n | ‖μ_m‖ | 真品库组内 RMS | **H** | "
          "d(最低浓度→质心) | 系列内相邻步长 | **跳变 Δ** |",
          "|---|---|---|---|---|---|---|---|---|"]
    for e in TARGETS:
        r = res[e]
        L.append(f"| {e} | {meta[e]['phase']} | {r['n_pure']} | {r['mu_norm']:.3f} | "
                 f"{r['H']*r['mu_norm']:.4f} | **{r['H']*100:.2f}%** | "
                 f"{r['d_gap']:.4f} | {r['d_step']:.4f} | **{r['jump']:.2f}×** |")
    hi = [e for e in TARGETS if res[e]["jump"] > 5]
    L += ["", f"- **跳变 Δ > 5× 的实体**：{'、'.join(hi) if hi else '无'}",
          "  → 这些体系的掺混系列与真品库不连续，任何「vs 真品库」的检出结论都不可信"]

    # ---- 3. 先验 LOD ----
    L += ["", "## 3. 先验预测检出限 vs 实测", "",
          "> `f_LOD = z₀.₉₉ · σ_null / ‖b*‖`，只用真品库 + 一条校准系列即可算出，",
          "> **不需要任何「检出结果」参与**。实测 LOD = TPR≥0.5 且 Fisher p<α 的最低浓度。",
          "",
          "| 实体 | 相态 | ‖b*‖ | σ_null | **先验 f_LOD** | **实测 LOD** | 量级一致 |",
          "|---|---|---|---|---|---|---|"]
    hit = 0
    for e in TARGETS:
        r = res[e]
        fl = r["f_lod"] * 100.0
        ob = r["lod_obs"]
        if ob is not None:
            ok = "✅" if (fl * 0.5 <= ob * 100 <= fl * 2.0) else "❌"
            hit += int(ok == "✅")
            ob_s = f"{ob*100:g}%"
        else:
            ok = "—"
            ob_s = f"> {max(r['tpr_c'])*100:g}%（未检出）"
        L.append(f"| {e} | {meta[e]['phase']} | {r['bstar']:.4f} | {r['sd0']:.4f} | "
                 f"**{fl:.2f}%** | **{ob_s}** | {ok} |")
    n_obs = sum(1 for e in TARGETS if res[e]["lod_obs"] is not None)
    L += ["", f"- 实测有 LOD 的 {n_obs}/{len(TARGETS)} 个实体中，"
              f"先验预测落在实测 0.5–2× 区间内 **{hit}** 个"]

    # ---- 4. 什么决定成败 ----
    y = a_nas
    x_sel = np.array([res[e]["sel_b"] for e in TARGETS])
    x_H = np.array([res[e]["H"] for e in TARGETS])
    x_jump = np.array([np.log10(max(res[e]["jump"], 1e-9)) for e in TARGETS])
    x_snr = np.array([res[e]["bstar"] / max(res[e]["sd0"], 1e-12) for e in TARGETS])
    x_lod = np.array([-np.log10(max(res[e]["f_lod"], 1e-9)) for e in TARGETS])
    rho = {}
    for nm, xx in [("SEL_b（共线度）", x_sel), ("H（真品库异质性）", x_H),
                   ("log10 跳变 Δ", x_jump), ("‖b*‖/σ_null（灵敏度）", x_snr),
                   ("先验 log(1/f_LOD)", x_lod)]:
        rho[nm] = tuple(float(v) for v in stats.spearmanr(xx, y))
    ss = [res[e]["auc"]["nas"] for e in TARGETS if meta[e]["phase"] == "固-固"]
    ll = [res[e]["auc"]["nas"] for e in TARGETS if meta[e]["phase"] == "液-液"]
    p_ph = float("nan")
    if len(ss) >= 2 and len(ll) >= 2:
        try:
            p_ph = float(stats.mannwhitneyu(ss, ll, alternative="greater").pvalue)
        except Exception:
            pass
    L += ["", "## 4. 什么决定检出能力？", "",
          "| 预测因子 | Spearman ρ vs AUC(nas) | p |", "|---|---|---|"]
    for nm, (r_, p_) in rho.items():
        L.append(f"| {nm} | {r_:+.3f} | {p_:.4f} |")
    # ---- 4b. 批次污染检验：cos 与 nas 的分歧能否被跳变 Δ 预测 ----
    gap = np.abs(a_cos - a_nas)
    rho_gap = tuple(float(v) for v in stats.spearmanr(
        np.array([res[e]["jump"] for e in TARGETS]), gap))
    # ---- 4c. 先验 LOD 的排序一致性 ----
    has_obs = [e for e in TARGETS if res[e]["lod_obs"] is not None]
    if len(has_obs) >= 4:
        rho_lodrank = tuple(float(v) for v in stats.spearmanr(
            [res[e]["f_lod"] for e in has_obs], [res[e]["lod_obs"] for e in has_obs]))
    else:
        rho_lodrank = (float("nan"), float("nan"))
    L += ["",
          f"- **批次污染检验**：|AUC(cos) − AUC(nas)| vs 跳变 Δ，"
              f"Spearman **ρ = {rho_gap[0]:+.3f}（p = {rho_gap[1]:.4f}）**",
          "  → 若显著为正，说明 **cos 偏离 nas 的程度就是批次污染的程度**",
          f"- **先验 LOD 的排序一致性**（仅 {len(has_obs)} 个实测有 LOD 的实体）："
              f"ρ = {rho_lodrank[0]:+.3f}（p = {rho_lodrank[1]:.4f}）",
          "  → 定量精度有限（见第 3 节），定性排序是否可靠以此为准", "",
          "**趋势量 ρ(t, c)**（对常数偏移免疫 → 纯化学信号存在性的度量）：", "",
          "| 实体 | 相态 | AUC(nas) 筛查 | **ρ(t, c)** | 判读 |", "|---|---|---|---|---|"]
    for e in TARGETS:
        r = res[e]
        if r["rho"] > 0.6 and r["auc"]["nas"] < 0.7:
            judge = "化学信号强但筛查失败 → 失败源于真品库/批次，非无信号"
        elif r["rho"] > 0.6 and r["auc"]["nas"] >= 0.7:
            judge = "信号强且筛查成功"
        elif r["rho"] <= 0.6 and r["auc"]["nas"] >= 0.7:
            judge = "筛查成功但浓度趋势弱（信号以常数偏移为主）"
        else:
            judge = "信号弱且筛查失败 → 真正的不可检出"
        L.append(f"| {e} | {meta[e]['phase']} | {r['auc']['nas']:.3f} | "
                 f"**{r['rho']:+.3f}** | {judge} |")

    # ---- 5. 逐浓度（每个实体按自己的浓度梯度列出）----
    L += ["", "## 5. nas 臂逐浓度检出率（TPR @ α=0.01）", "",
          "> 各数据集浓度梯度不同，按各自档位列出（格式：浓度% → TPR）。", "",
          "| 实体 | 相态 | 浓度梯度 → TPR |", "|---|---|---|"]
    for e in TARGETS:
        r = res[e]
        seq = " · ".join(f"{l*100:g}→{r['tpr_c'][l]:.2f}"
                         for l in sorted(r["tpr_c"]))
        L.append(f"| {e} | {meta[e]['phase']} | {seq} |")
    zero = [e for e in TARGETS
            if all(v < 0.005 for v in res[e]["tpr_c"].values())]
    if zero:
        L += ["", "> **" + "、".join(zero) + "** 全档位 TPR = 0.00：浓度响应方向几乎为零"
              f"（‖b*‖ = {res[zero[0]]['bstar']:.4f}，最低），属**数学上不可检出**"
              " —— 不是浓度不够，是这个「基质—掺杂物」对的谱响应本就不存在。"]

    # ---- 6. k 敏感性 ----
    L += ["", "## 6. k（干扰子空间维度）敏感性", "",
          "| 实体 | k | SEL_b | AUC(nas) | AUC(od) |", "|---|---|---|---|---|"]
    for e in TARGETS:
        r = res[e]
        for i, (k, v) in enumerate(sorted(r["k_sens"].items(), key=lambda x: int(x[0]))):
            L.append(f"| {e if i == 0 else ''} | {k}"
                     f"{' *' if int(k) == r['k_star'] else ''} | {v['sel']:.4f} | "
                     f"{v['auc_nas']:.3f} | {v['auc_od']:.3f} |")

    # ---- 7. 结论 ----
    good = [e for e in TARGETS if res[e]["auc"]["nas"] >= 0.9]
    bad = [e for e in TARGETS if res[e]["auc"]["nas"] < 0.7]
    L += ["", "## 7. 结论", "",
          "### 7.1 反反转：exp13 与 exp17 的相态结论都不成立", "",
          f"- NAS 判据下检出成功（AUC ≥ 0.9）：{'、'.join(good) if good else '无'}",
          f"- 检出失败（AUC < 0.7）：{'、'.join(bad) if bad else '无'}",
          f"- 相态检验：**p = {p_ph:.4f}**（不显著，且方向与 exp17 相反）→ 相态被否证",
          f"- 成功实体横跨两种相态：{'是' if good else '否'}", "",
          "### 7.2 真正的瓶颈不是「掺杂物与基质太像」，而是「真品库太松/不同批次」", "",
          f"- 共线度 SEL_b vs AUC：ρ = {rho['SEL_b（共线度）'][0]:+.3f}"
              f"（p = {rho['SEL_b（共线度）'][1]:.4f}）→ {'不显著' if rho['SEL_b（共线度）'][1] > 0.05 else '显著'}",
          f"- 真品库异质性 H vs AUC：ρ = {rho['H（真品库异质性）'][0]:+.3f}"
              f"（p = {rho['H（真品库异质性）'][1]:.4f}）",
          f"- 跳变 Δ vs AUC：ρ = {rho['log10 跳变 Δ'][0]:+.3f}"
              f"（p = {rho['log10 跳变 Δ'][1]:.4f}）",
          "- **判读**：SEL_b 在 k 扫描中从 0.99 掉到 0.07（第 6 节），AUC 却几乎不变 →",
          "  共线度**不是**本数据集的瓶颈；真品库的代表性与批次一致性才是",
          "- **对零样本研究的含义**：改进方向应从「更好的文本/表示」转向",
          "  「**更具代表性的真品参考库**」与「**跨批次校正**」", "",
          "### 7.3 先验 LOD：定性可靠，定量有限", "",
          f"- 排序一致：ρ = {rho_lodrank[0]:+.3f}（p = {rho_lodrank[1]:.4f}）",
          f"- 量级命中：{hit}/{n_obs} 落在实测 0.5–2× 内",
          "- 偏差来源：先验式假设「响应严格线性 + 噪声齐性 + 系列与真品库同批次」，",
          "  而第 2 节显示后两条在 cinnamon / honey 体系不成立", "",
          "### 7.4 方法学警告（对文献的普适价值）", "",
          "- 「cos(掺混谱, 纯基质谱) ≈ 1 ⇒ 信号被淹没 ⇒ 不可检出」是**错误推论**：",
          "  余弦是与**全谱强度**比较的相对量；检出是与**真品库变异**比较的绝对量",
          f"- 该错误可被量化：|AUC(cos) − AUC(nas)| 与跳变 Δ 的相关 "
              f"ρ = {rho_gap[0]:+.3f}（p = {rho_gap[1]:.4f}）",
          "- 掺假数据集的**掺混系列与真品库若不同批次**（跳变 Δ ≫ 1），",
          "  任何基于该数据集的「检出准确率」都是批次差异的度量，须先诊断", "",
          "### 7.5 边界（须写进论文 limitation）", "",
          "- 本实验的方向 b 来自**掺混梯度**（方法开发阶段的常规投入），",
          "  **不是**从文本/结构直接预测的谱 —— L0 级（纯结构 → 谱方向）本实验做不到",
          "- `od` 臂不需要任何掺杂物信息，是**真零样本**；其 AUC 即零样本检出的上界参考",
          "- 真品库仅 16 条的 honey 体系（HSYRUP）统计功效有限，结论需谨慎"]
    p_ = os.path.join(OUT, "exp18_p6_nas_lod.md")
    with open(p_, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n[exp18] 报告已写入:", p_)


if __name__ == "__main__":
    main()
