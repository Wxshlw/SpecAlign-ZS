# -*- coding: utf-8 -*-
"""exp91 —— 三任务 × 8 对齐：把「符号相反」放进同一张表

为什么要这个实验
----------------
exp90 证明：在**归因**口径上，8 个对齐臂无一显著改善，且 `mf`/`zq` 显著有害。
但 exp69（识别 margin +0.118, p=.013）与 exp73/77（检出 mf +0.132）报的是**正增益**。
三处口径不同（归因=成对差 |U| 候选；识别=one-vs-rest old-heldout；检出=one-vs-rest
同基质纯谱），因此"同一操作在不同任务上符号相反"目前**只能是推测**，不能进论文。

本实验把三者放到**同一次 stage1、同一份码本、同一个头部、同一批对齐臂**下：

    一个 (held-out 集 U, seed, rep) → 一次 stage1 → 一个候选分数矩阵
    对齐臂作用在**同一形状的分数矩阵**上 → 三种读出

读出（唯一的差别，也就是被研究的变量）：
  D 检出   s = 第 T 列；正例 = U 的掺混谱，负例 = **同基质纯谱**
  I 识别   s = 第 T 列；正例 = U 的掺混谱，负例 = **其他实体的掺混谱**
  A 归因   s = U 块内两列之差（auc_dis）/ U 块内 argmax（top-1）

★ 关键设计：对齐算子**一律作用在完整候选分数矩阵（老类 + U）上**，
  三个任务的矩阵完全相同、算子完全相同，只有读出不同。
  ⇒ 任何符号差异都只能来自任务本身，不可能来自口径。

对齐臂（与 exp90 完全一致，便于交叉验证）：
  A 标定类：lin（基线）/ cq（减候选均值）/ zq（除候选标准差）/ iqr / zC（老类分布标定）
  B 竞争类：margin（s_c − max_{j≠c} s_j）
  C 几何类：cosC（候选方向归一）/ mf（白化 Σ⁻¹）

口径固定：λ=1e-4（零样本不可调 λ）、group 分组、k=0。
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import itertools
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "论文8"))
import p6_g1_data as G1
from exp63_p6_screening_protocol import (stage1, embed, head_ridge_R,
                                         auc_of, tpr_at_fpr, pauc_std, EID2MID)
from exp68_p6_cond_scorer import split_by_group
from exp71_p6_detection_fix import load_groups_cached
from exp78_p6_attribution_proof import split_group_w
from exp81b_p6_benchmark_zoo import PRIORS, DEV

OUT = os.path.join(os.path.dirname(HERE), "out")
OLD_FRAC = 0.7
LAM = 1e-4

ARMS = ["lin", "cq", "zq", "iqr", "margin", "zC", "cosC", "mf1e4", "mf020", "mf060"]
GROUP = {
    "lin": "A_calib", "cq": "A_calib", "zq": "A_calib", "iqr": "A_calib",
    "zC": "A_calib",
    "margin": "B_compete",
    "cosC": "C_geometry",
    "mf1e4": "C_geometry", "mf020": "C_geometry", "mf060": "C_geometry",
}
# 白化收缩强度必须随类数调整：类数 8–10 时 1e-4 会使 Σ 近奇异。
# exp90 只用 1e-4（因此报"白化有害"），exp73/77 用 0.2–0.6（报"白化有效"）。
# 本实验三档并列，避免把"参数没调"当成"机制"。
SHRINKS = {"mf1e4": 1e-4, "mf020": 0.2, "mf060": 0.6}


def whiten_matrix(Zfit, ytr, shrink=1e-4):
    """类内散度 + shrinkage，返回 Σ⁻¹。"""
    rows = sorted({int(y) for y in ytr})
    d = Zfit.shape[1]
    S = np.zeros((d, d), float)
    n = 0
    means = {j: Zfit[ytr == j].mean(0) for j in rows}
    for j in rows:
        R = Zfit[ytr == j] - means[j]
        S += R.T @ R
        n += (ytr == j).sum()
    S /= max(1, n - len(rows))
    S = S + shrink * float(np.trace(S)) / d * np.eye(d)
    return np.linalg.inv(S)


def apply_arm(arm, L, muC=None, sdC=None):
    """L: [n, C] 候选分数矩阵。算子作用于**列方向**（候选间）。"""
    if arm == "lin":
        return L
    if arm == "cq":
        return L - L.mean(1, keepdims=True)
    if arm == "zq":
        return L / (L.std(1, keepdims=True) + 1e-12)
    if arm == "iqr":
        q75 = np.percentile(L, 75, axis=1, keepdims=True)
        q25 = np.percentile(L, 25, axis=1, keepdims=True)
        med = np.median(L, axis=1, keepdims=True)
        return (L - med) / (q75 - q25 + 1e-12)
    if arm == "margin":
        out = np.empty_like(L)
        for c in range(L.shape[1]):
            out[:, c] = L[:, c] - np.delete(L, c, axis=1).max(1)
        return out
    if arm == "zC":
        return (L - muC) / sdC
    raise ValueError(arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--nrep", type=int, default=4)
    ap.add_argument("--nhu", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--priors", default="clm,physchem,ecfp2048")
    ap.add_argument("--eids", default="")
    ap.add_argument("--tasks", default="DI A".replace(" ", ""))
    ap.add_argument("--out", default="exp91_t3_nhu3.json")
    args = ap.parse_args()
    pri_list = [x.strip() for x in args.priors.split(",") if x.strip()]

    ent, conc, bg, _ = G1.load_g1_conc()
    GR = load_groups_cached()
    eids = sorted(set(GR) & set(EID2MID))
    if args.eids:
        eids = [e for e in eids if e in args.eids.split(",")]
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]

    zc = np.load(os.path.join(OUT, "exp81_prior_cache.npz"), allow_pickle=True)
    assert list(zc["keys"]) == keys_all, "先验缓存 key 顺序不一致，禁止对齐！"
    S_pri = {n: np.asarray(zc[n], float) for n in PRIORS}
    dz = S_pri["clm"].shape[1]
    ki = {k: i for i, k in enumerate(keys_all)}

    # res[prior][arm][task][unit] = [records]
    res = {p: {a: {"D": {}, "I": {}, "A": {}} for a in ARMS} for p in pri_list}
    meta = dict(nhu=args.nhu, seeds=args.seeds, seed0=args.seed0, nrep=args.nrep,
                lam=LAM, priors=pri_list, arms=ARMS, group=GROUP,
                design="same stage1 / same codebook / same arms; only read-out differs",
                tasks=dict(D="detect: pos=adulterated U, neg=pure matrix of same matrix",
                           I="identify: pos=adulterated U, neg=other entities' spectra",
                           A="attribute: auc_dis T-D pair (both unseen) + top-1 over U"),
                note="alignment operator applied over FULL codebook columns in all tasks")

    combos = list(itertools.combinations(eids, args.nhu))
    print(f"[exp91] |U|={args.nhu} combos={len(combos)} reps={args.nrep} "
          f"seeds={args.seeds} priors={pri_list}", flush=True)

    for si in range(args.seed0, args.seed0 + args.seeds):
        rng0 = np.random.default_rng(si * 7919 + 13)
        cis = rng0.permutation(len(combos))[:args.nrep]
        for ci in cis:
            U = list(combos[ci])
            others = [e for e in sorted(ent) if e not in U]
            order = ([ki[e] for e in others]
                     + [ki[f"M_{m}"] for m in mids]
                     + [ki[u] for u in U])
            NEW0 = len(others) + len(mids)
            iU = {u: NEW0 + j for j, u in enumerate(U)}
            # ★ stage1 的化学对齐码本必须包含 U 的行（零样本下 U 的化学向量可用、标签不可用）。
            #   漏掉会显著削弱对齐，使归因从 0.87 掉到 0.78（已实测）。
            nrows = NEW0 + len(U)

            rng = np.random.default_rng(si * 104729 + ci * 37 + 7)
            Xte_U, Xtr_U = {}, {}
            for u in U:
                Xe = np.asarray(ent[u], float)
                te_ix, pool_ix = split_by_group(Xe, GR[u]["gid"], rng)
                Xte_U[u] = Xe[te_ix]
                Xtr_U[u] = Xe[pool_ix]          # k=0：不进入训练

            parts, ys = [], []
            old_ref, mid_ref = {}, {}
            for j, e in enumerate(others):
                Xo = np.asarray(ent[e], float)
                if e in GR:
                    a, b, _ = split_group_w(Xo, GR[e]["gid"], rng)
                else:                       # 无分组信息实体退化为随机划分
                    p1 = rng.permutation(len(Xo))
                    cut = max(2, int(round(OLD_FRAC * len(Xo))))
                    a, b = p1[:cut], p1[cut:]
                parts.append(Xo[a])
                ys.append(np.full(len(a), j))
                old_ref[e] = Xo[b]
            for j, m in enumerate(mids):
                Xo = np.asarray(bg[m], float)
                pp = rng.permutation(len(Xo))
                cut = max(2, int(round(OLD_FRAC * len(Xo))))
                parts.append(Xo[pp[:cut]])
                ys.append(np.full(cut, len(others) + j))
                mid_ref[m] = Xo[pp[cut:]]
            Xtr = np.concatenate(parts).astype(np.float32)
            ytr = np.concatenate(ys).astype(np.int64)
            rows = sorted({int(y) for y in ytr})
            tag = f"seed{si}_c{ci}"

            t0 = time.time()
            net, _ = stage1(Xtr, ytr, dz, nrows, args.steps, si * 1000 + ci,
                            S=S_pri["clm"][order])
            Zfit = embed(net, Xtr)
            Z_old = np.stack([Zfit[ytr == j].mean(0) for j in rows])
            Sin = {k: whiten_matrix(Zfit, ytr, shrink=v) for k, v in SHRINKS.items()}
            mu_z = Zfit.mean(0, keepdims=True)

            # ---- 三类 query 的嵌入（只算一次，所有先验共用）----
            Zpos = {u: embed(net, Xte_U[u]) for u in U}
            ZnegD = {u: embed(net, mid_ref[EID2MID[u]]) for u in U}
            ZnegI = embed(net, np.concatenate([old_ref[e] for e in others])
                          .astype(np.float32))
            print(f"  [{tag}] U={'-'.join(U)} stage1 {time.time()-t0:.1f}s "
                  f"nC={len(order)}", flush=True)

            for p in pri_list:
                Sp = S_pri[p][order]
                S_o = Sp[rows]
                mu = S_o.mean(0, keepdims=True)
                Wc = (Sp - mu) @ head_ridge_R(S_o, Z_old, LAM)   # [C, dz]
                Wn = Wc / (np.linalg.norm(Wc, axis=1, keepdims=True) + 1e-12)
                Wm = {k: Wc @ Sin[k].T for k in SHRINKS}
                muC = (Zfit @ Wc.T).mean(0)
                sdC = (Zfit @ Wc.T).std(0) + 1e-12

                def score(Zq, arm):
                    if arm == "cosC":
                        return Zq @ Wn.T
                    if arm in Wm:
                        return (Zq - mu_z) @ Wm[arm].T
                    return apply_arm(arm, Zq @ Wc.T, muC, sdC)

                Lpos = {u: None for u in U}
                for arm in ARMS:
                    Lp = {u: score(Zpos[u], arm) for u in U}
                    Ld = {u: score(ZnegD[u], arm) for u in U}
                    Li = score(ZnegI, arm)

                    # ---------- D 检出 / I 识别：第 T 列 one-vs-rest ----------
                    for u in U:
                        jT = iU[u]
                        s_p = Lp[u][:, jT]
                        s_d = Ld[u][:, jT]
                        s_i = Li[:, jT]
                        res[p][arm]["D"].setdefault(u, []).append(dict(
                            auc=round(float(auc_of(s_p, s_d)), 4),
                            tpr5=round(float(tpr_at_fpr(s_p, s_d, .05)), 4),
                            pauc10=round(float(pauc_std(s_p, s_d, .10)), 4),
                            n_pos=len(s_p), n_neg=len(s_d), tag=tag))
                        res[p][arm]["I"].setdefault(u, []).append(dict(
                            auc=round(float(auc_of(s_p, s_i)), 4),
                            tpr5=round(float(tpr_at_fpr(s_p, s_i, .05)), 4),
                            n_pos=len(s_p), n_neg=len(s_i), tag=tag))

                    # ---------- A 归因：U 块内成对差 + argmax ----------
                    Ub = [iU[u] for u in U]
                    for jT, jD in itertools.combinations(range(len(U)), 2):
                        T, D = U[jT], U[jD]
                        dT = Lp[T][:, Ub[jT]] - Lp[T][:, Ub[jD]]
                        dD = Lp[D][:, Ub[jT]] - Lp[D][:, Ub[jD]]
                        res[p][arm]["A"].setdefault(f"{T}|{D}", []).append(
                            dict(auc_dis=round(float(auc_of(dT, dD)), 4), tag=tag))
                    for j, u in enumerate(U):
                        L = Lp[u][:, Ub]
                        pred = np.array(U)[L.argmax(1)]
                        res[p][arm]["A"].setdefault(f"top1:{u}", []).append(
                            dict(top1=round(float(np.mean(pred == u)), 4),
                                 n=len(L), tag=tag))

    out = dict(meta=meta, res=res)
    path = os.path.join(OUT, args.out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[exp91] -> {path}", flush=True)


if __name__ == "__main__":
    main()
