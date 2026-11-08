# -*- coding: utf-8 -*-
"""exp92 —— 为什么加上语义先验，检出反而更差？

问题
----
同一批谱、同一个任务（同基质检出：正例=掺混谱，负例=同基质纯谱）：
    常规监督 rf   auc ≈ 0.814 (k=0) / 0.870 (k=5)
    本文化学先验   auc ≈ 0.633 (k=0) / 0.653 (k=5)
"先验没用"不能解释"更差"。必须定位到具体的损失来源。

本脚本把总损失拆成两项，并给出「为什么方向会偏」的机制证据
--------------------------------------------------------
    总损失 = 表示成本 + 方向成本
    表示成本 = auc(raw 谱上最优方向) − auc(z 上最优方向)
    方向成本 = auc(z 上最优方向)     − auc(z 上先验方向)

四个方向（全部单位化，都在同一 test 集上算，保证可比）
    b_x        原始谱空间的判别方向  mean(x_pos) − mean(x_neg)     ← 表示上界
    b_z        嵌入空间的判别方向    mean(z_pos) − mean(z_neg)     ← z 内的上界
    b_gen       数据驱动通用方向     老类净信号均值（k=0 合法，不用 T）
    w_T        化学先验方向          码本第 T 行（岭回归头外推）

机制诊断（回答「方向为什么偏」）
    H1 基质主导    w_T 与「基质中心方向 v_mat」的余弦应接近 1，
                   与检出方向 b_z 的余弦应接近 0（exp71 测到 +0.157）。
    H2 基质有害    把 w_T 沿 v_mat 的分量扣掉，只留残差 w_res，
                   auc(w_res) 应 **高于** auc(w_T)。若是，则基质分量不是"无用"而是"有害"。
    H3 白化放大基质 Σ⁻¹ 在同基质内被纯基质谱的极低类内方差主导，
                   cos(Σ⁻¹w_T, v_mat) 应高于 cos(w_T, v_mat)。

★ b_x / b_z 用到了 test 正例的均值 ⇒ 属于 oracle 上界，只用于诊断，不可作为方法报出。
"""
from __future__ import annotations
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "论文8"))
import p6_g1_data as G1
from exp63_p6_screening_protocol import (stage1, embed, head_ridge_R,
                                         auc_of, EID2MID)
from exp68_p6_cond_scorer import split_by_group
from exp71_p6_detection_fix import load_groups_cached
from exp78_p6_attribution_proof import split_group_w
from exp81b_p6_benchmark_zoo import PRIORS

OUT = os.path.join(os.path.dirname(HERE), "out")
OLD_FRAC = 0.7
LAM = 1e-4


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def inv_cov(X, y, shrink=1e-3):
    """类内残差协方差 + shrinkage 的逆。X: [n,d]，y: [n]。"""
    rows = sorted({int(t) for t in y})
    d = X.shape[1]
    S = np.zeros((d, d), float)
    n = 0
    for r in rows:
        Z = X[y == r]
        if len(Z) < 2:
            continue
        R = Z - Z.mean(0)
        S += R.T @ R
        n += len(Z) - 1
    S /= max(1, n)
    S += shrink * float(np.trace(S)) / d * np.eye(d)
    return np.linalg.inv(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--priors", default="clm,physchem")
    ap.add_argument("--eids", default="")
    ap.add_argument("--out", default="exp92_why_worse.json")
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
    assert list(zc["keys"]) == keys_all
    S_pri = {n: np.asarray(zc[n], float) for n in PRIORS}
    dz = S_pri["clm"].shape[1]
    ki = {k: i for i, k in enumerate(keys_all)}

    res = {p: {} for p in pri_list}
    print(f"[exp92] entities={eids} seeds={args.seeds} priors={pri_list}",
          flush=True)

    for e_hold in eids:
        mid_h = EID2MID[e_hold]
        others = [e for e in sorted(ent) if e != e_hold]
        order = ([ki[e] for e in others] + [ki[f"M_{m}"] for m in mids]
                 + [ki[e_hold]])
        NEW = len(others) + len(mids)
        iT = NEW
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed * 7919 + 13)
            Xe = np.asarray(ent[e_hold], float)
            te_ix, pool_ix = split_by_group(Xe, GR[e_hold]["gid"], rng)
            Xpos, Xneg = Xe[te_ix], np.asarray(bg[mid_h], float)
            pp = rng.permutation(len(Xneg))
            n_cut = max(5, int(round(0.3 * len(Xneg))))
            Xneg = Xneg[pp[:n_cut]]

            parts, ys = [], []
            mid_tr = {}
            for j, e in enumerate(others):
                Xo = np.asarray(ent[e], float)
                if e in GR:
                    a, b, _ = split_group_w(Xo, GR[e]["gid"], rng)
                else:                       # 无分组信息的实体退化为随机划分
                    p1 = rng.permutation(len(Xo))
                    cut = max(2, int(round(OLD_FRAC * len(Xo))))
                    a, b = p1[:cut], p1[cut:]
                parts.append(Xo[a])
                ys.append(np.full(len(a), j))
            for j, m in enumerate(mids):
                Xo = np.asarray(bg[m], float)
                p2 = rng.permutation(len(Xo))
                cut = max(2, int(round(OLD_FRAC * len(Xo))))
                parts.append(Xo[p2[:cut]])
                ys.append(np.full(cut, len(others) + j))
                mid_tr[m] = Xo[p2[:cut]]
            Xtr = np.concatenate(parts).astype(np.float32)
            ytr = np.concatenate(ys).astype(np.int64)
            rows = sorted({int(t) for t in ytr})

            net, _ = stage1(Xtr, ytr, dz, NEW, args.steps,
                            seed * 1000 + eids.index(e_hold),
                            S=S_pri["clm"][order])
            Zf = embed(net, Xtr)
            Zc = np.stack([Zf[ytr == j].mean(0) for j in rows])
            Zpos, Zneg = embed(net, Xpos), embed(net, Xneg)

            Sx = inv_cov(Xtr.astype(float), ytr)
            Sz = inv_cov(Zf, ytr)
            mu_x, mu_z = Xtr.mean(0), Zf.mean(0)

            # ---- 四个方向 ----
            b_x = unit(Xpos.mean(0) - Xneg.mean(0))
            b_z = unit(Zpos.mean(0) - Zneg.mean(0))
            # 数据驱动通用方向：老类净信号（k=0 合法）
            mid_c = {m: Zf[ytr == rows.index(len(others) + mids.index(m))].mean(0)
                     for m in mids}
            g = []
            for e in others:
                j = rows.index(others.index(e))
                zg = Zf[ytr == j].mean(0)
                mj = mid_c.get(EID2MID.get(e))
                g.append(zg - (mj if mj is not None else Zf.mean(0)))
            b_gen = unit(np.mean(g, 0))
            # 基质中心方向
            v_mat = unit(np.stack([mid_c[m] for m in mids]).mean(0) - Zf.mean(0))

            for p in pri_list:
                Sp = S_pri[p][order]
                S_o = Sp[rows]
                mu_s = S_o.mean(0, keepdims=True)
                Wc = (Sp - mu_s) @ head_ridge_R(S_o, Zc, LAM)
                w = unit(Wc[iT])
                w_mf = Wc[iT] @ Sz.T
                w_res = unit(w - (w @ v_mat) * v_mat)
                w_res_mf = w_res @ Sz.T

                def auc_dir(dvec, space):
                    if space == "x":
                        return float(auc_of((Xpos - mu_x) @ dvec,
                                            (Xneg - mu_x) @ dvec))
                    return float(auc_of((Zpos - mu_z) @ dvec,
                                        (Zneg - mu_z) @ dvec))

                rec = dict(
                    # 表示上界
                    auc_x=auc_dir(b_x, "x"),
                    auc_x_mf=auc_dir(Sx @ b_x, "x"),
                    auc_z=auc_dir(b_z, "z"),
                    auc_z_mf=auc_dir(Sz @ b_z, "z"),
                    # 方向候选
                    auc_gen=auc_dir(b_gen, "z"),
                    auc_gen_mf=auc_dir(Sz @ b_gen, "z"),
                    auc_prior=auc_dir(w, "z"),
                    auc_prior_mf=auc_dir(w_mf, "z"),
                    auc_res=auc_dir(w_res, "z"),
                    auc_res_mf=auc_dir(w_res_mf, "z"),
                    # 几何诊断
                    cos_p_bz=float(w @ b_z),
                    cos_pmf_bz=float(unit(w_mf) @ b_z),
                    cos_p_mat=float(w @ v_mat),
                    cos_pmf_mat=float(unit(w_mf) @ v_mat),
                    cos_gen_bz=float(b_gen @ b_z),
                )
                res[p].setdefault(e_hold, []).append(rec)
            print(f"  [{e_hold} s{seed}] x={rec['auc_x']:.3f}/{rec['auc_x_mf']:.3f} "
                  f"z={rec['auc_z']:.3f}/{rec['auc_z_mf']:.3f} "
                  f"gen={rec['auc_gen']:.3f} prior={rec['auc_prior']:.3f} "
                  f"res={rec['auc_res']:.3f} cos_p_bz={rec['cos_p_bz']:+.3f} "
                  f"cos_p_mat={rec['cos_p_mat']:+.3f}", flush=True)

    with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as f:
        json.dump(dict(meta=dict(seeds=args.seeds, priors=pri_list,
                                 lam=LAM, eids=eids,
                                 note="b_x/b_z/b_gen are oracle upper bounds "
                                      "(use test means); diagnosis only"),
                       res=res), f, ensure_ascii=False, indent=1)
    print(f"[exp92] -> {os.path.join(OUT, args.out)}", flush=True)


if __name__ == "__main__":
    main()
