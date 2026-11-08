# -*- coding: utf-8 -*-
"""exp90 —— 对齐消融：固定码本，只换「怎么比」

为什么要这个实验
----------------
exp82 轴 A 做了「固定对齐（岭回归头）、换码本」，证明结构类先验之间 0/15 显著（exp89）。
那是「换先验」的杠杆。
本实验做镜像：**固定码本、只换对齐**，测「换对齐」的杠杆。两者相减即得
「创新应归到哪一层」的定量依据。

对齐臂的分类（这是本实验的核心设计，不是随手堆操作）
----------------------------------------------------
A. 标定类（改变分数的尺度/零点）
   - `lin`     基线（原始分数）
   - `cq`      分数减 query 内均值            —— 对成对差**恒等**（平移被消掉）
   - `zq`      分数除 query 内标准差          —— **逐 query** 缩放 ⇒ 破坏跨 query 可比性
   - `iqr`     分数用 median/IQR 重标定       —— 同上
   ⇒ 预测：cq 恒等；zq/iqr 不是「无效」而是可能**有害**（smoke 已实测到方向反转）。
   - `zC`      用**训练集（老类谱）**上候选 c 的打分均值/标准差做标定
              —— 这是匹配滤波的标准做法，只跨候选改变尺度，不破坏 query 可比性。

B. 竞争结构类：
   - `margin`  归因语义的 margin：m_c = s_c − max_{j≠c} s_j
     ⇒ 对 argmax **恒等**（被选中的候选 m=0 恒为最大）；只可能改变成对判别。
     注意这与 exp69 的识别语义 margin 不同，二者不可混用。

C. 比较几何类（唯一能改变决策的）：
   - `cosC`    候选方向归一化 W_c/‖W_c‖
   - `mf`      白化匹配滤波 (z−μ)ᵀΣ⁻¹W_c

⚠ 已移除 `cosQ` / `cosB`：`exp63.embed()` 内置 `F.normalize`，query 嵌入已是单位范数，
两个臂数学上退化为恒等（smoke 实测差 0.00e+00），留着只会制造虚假的「无效」结论。

口径：定向归因（T、D 均未见），group 分组，λ 固定 1e-4（零样本不可调 λ）。
两指标并列：auc_dis（成对判别）与 top-1（候选集 argmax）——二者对对齐的敏感度不同，
缺一个会漏掉一半效应。
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
from exp63_p6_screening_protocol import (stage1, embed, head_ridge_R, nrm,
                                         auc_of, EID2MID)
from exp68_p6_cond_scorer import split_by_group
from exp71_p6_detection_fix import load_groups_cached
from exp78_p6_attribution_proof import split_group_w
from exp81b_p6_benchmark_zoo import PRIORS, DEV

OUT = os.path.join(os.path.dirname(HERE), "out")
OLD_FRAC = 0.7
LAM = 1e-4

ARMS = ["lin", "cq", "zq", "iqr", "margin", "zC", "cosC", "mf"]

GROUP = {
    "lin": "A_calib", "cq": "A_calib", "zq": "A_calib", "iqr": "A_calib",
    "zC": "A_calib",
    "margin": "B_compete",
    "cosC": "C_geometry", "mf": "C_geometry",
}


def whiten_matrix(Zfit, ytr, shrink=1e-4):
    """类内散度 + shrinkage，返回 Σ⁻¹。"""
    rows = sorted({int(y) for y in ytr})
    d = Zfit.shape[1]
    S = np.zeros((d, d), float)
    n = 0
    means = {}
    for j in rows:
        zj = Zfit[ytr == j]
        means[j] = zj.mean(0)
    for j in rows:
        zj = Zfit[ytr == j]
        R = zj - means[j]
        S += R.T @ R
        n += len(zj)
    S /= max(1, n - len(rows))
    tr = np.trace(S) / d
    S = S + shrink * tr * np.eye(d)
    return np.linalg.inv(S)


def transform(arm, L, Wc, Zte_u, Sin=None, mu_z=None):
    """L: [n_u, |U|] 原始分数矩阵；返回同形状的变换后矩阵。

    只允许使用 query 自身的分数 / 码本向量 / 训练统计量，不得触碰标签。
    """
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
            others = np.delete(L, c, axis=1)
            out[:, c] = L[:, c] - others.max(1)
        return out
    raise ValueError(arm)


def score_arms(Wc, Zte, U, Zfit=None, Sin=None, mu_z=None):
    """对每个码本，产出 {arm: {u: [n_u, |U|]}}。"""
    res = {arm: {} for arm in ARMS}
    # 候选级标定统计量：只用训练集（老类谱）在该候选方向上的打分分布
    if Zfit is not None:
        Sgrid = Zfit @ Wc.T
        muC = Sgrid.mean(0)
        sdC = Sgrid.std(0) + 1e-12
    for j, u in enumerate(U):
        L0 = Zte[u] @ Wc.T                    # [n_u, |U|]
        for arm in ["lin", "cq", "zq", "iqr", "margin"]:
            res[arm][u] = transform(arm, L0, Wc, Zte[u])
        res["zC"][u] = (L0 - muC) / sdC
        # ---- 几何类 ----
        Wn = Wc / (np.linalg.norm(Wc, axis=1, keepdims=True) + 1e-12)
        res["cosC"][u] = Zte[u] @ Wn.T
        if Sin is not None:
            Wm = Wc @ Sin.T
            res["mf"][u] = (Zte[u] - mu_z) @ Wm.T
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--nrep", type=int, default=4)
    ap.add_argument("--nhu", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--priors", default="clm,physchem,ecfp2048")
    ap.add_argument("--eids", default="")
    ap.add_argument("--out", default="exp90_align_nhu3.json")
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

    # 输出结构：res[prior][arm][kind][key] = [records]
    res = {p: {a: {"dis": {}, "top1": {}} for a in ARMS} for p in pri_list}
    meta = dict(nhu=args.nhu, seeds=args.seeds, seed0=args.seed0, nrep=args.nrep,
                lam=LAM, priors=pri_list, arms=ARMS, group=GROUP,
                note="fixed codebook, only alignment varies; auc_dis T/D both unseen")

    combos = list(itertools.combinations(eids, args.nhu))
    print(f"[exp90] |U|={args.nhu} combos={len(combos)} reps={args.nrep} "
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
            S_all = {p: S_pri[p][order] for p in pri_list}
            NEW0 = len(others) + len(mids)
            iu = {u: NEW0 + j for j, u in enumerate(U)}
            cand_ix = [iu[u] for u in U]
            nrows = NEW0 + len(U)

            rng = np.random.default_rng(si * 104729 + ci * 37 + 7)
            Xte_U, Xtr_U = {}, {}
            for u in U:
                Xe = np.asarray(ent[u], float)
                te_ix, pool_ix = split_by_group(Xe, GR[u]["gid"], rng)
                Xte_U[u] = Xe[te_ix]
                Xtr_U[u] = Xe[pool_ix]
            parts, ys = [], []
            for j, e in enumerate(others):
                Xo = np.asarray(ent[e], float)
                if e in GR:
                    a, b, _ = split_group_w(Xo, GR[e]["gid"], rng)
                else:
                    pp = rng.permutation(len(Xo))
                    cut = max(2, int(round(OLD_FRAC * len(Xo))))
                    a, b = pp[:cut], pp[cut:]
                parts.append(Xo[a])
                ys.append(np.full(len(a), j))
            for j, m in enumerate(mids):
                Xo = np.asarray(bg[m], float)
                pp = rng.permutation(len(Xo))
                cut = max(2, int(round(OLD_FRAC * len(Xo))))
                parts.append(Xo[pp[:cut]])
                ys.append(np.full(cut, len(others) + j))
            Xtr = np.concatenate(parts).astype(np.float32)
            ytr = np.concatenate(ys).astype(np.int64)
            rows = sorted({int(y) for y in ytr})
            tag = f"seed{si}_c{ci}"

            t0 = time.time()
            net, _ = stage1(Xtr, ytr, dz, nrows, args.steps, si * 1000 + ci,
                            S=S_all["clm"])
            Zfit = embed(net, Xtr)
            Z_old = np.stack([Zfit[ytr == j].mean(0) for j in rows])
            Zte = {u: embed(net, Xte_U[u]) for u in U}
            Sin = whiten_matrix(Zfit, ytr)
            mu_z = Zfit.mean(0, keepdims=True)
            print(f"  [{tag}] U={'-'.join(U)} stage1 {time.time()-t0:.1f}s", flush=True)

            for p in pri_list:
                Sp = S_all[p]
                S_o = Sp[rows]
                mu = S_o.mean(0, keepdims=True)
                Sc = Sp[cand_ix]
                Wc = (Sc - mu) @ head_ridge_R(S_o, Z_old, LAM)
                armL = score_arms(Wc, Zte, U, Zfit=Zfit, Sin=Sin, mu_z=mu_z)

                for arm in ARMS:
                    Ld = armL[arm]
                    # ---- auc_dis：成对判别 ----
                    for jT, jD in itertools.combinations(range(len(U)), 2):
                        T, D = U[jT], U[jD]
                        dT = Ld[T][:, jT] - Ld[T][:, jD]
                        dD = Ld[D][:, jT] - Ld[D][:, jD]
                        a = float(auc_of(dT, dD))
                        res[p][arm]["dis"].setdefault(f"{T}|{D}", []).append(
                            dict(auc_dis=a, sep=abs(a - .5) * 2, tag=tag))
                    # ---- top-1：候选集 argmax ----
                    for j, u in enumerate(U):
                        L = Ld[u]
                        pred = np.array(U)[L.argmax(1)]
                        res[p][arm]["top1"].setdefault(u, []).append(
                            dict(top1=float(np.mean(pred == u)), n=len(L), tag=tag))

    out = dict(meta=meta, res=res)
    path = os.path.join(OUT, args.out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[exp90] -> {path}", flush=True)


if __name__ == "__main__":
    main()
