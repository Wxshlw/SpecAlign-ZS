# -*- coding: utf-8 -*-
"""exp92b —— 检出为什么会「加了先验反而更差」：把损失全部归到方向

exp92 已排除两个假设
-------------------
  ✗ H1 表示损失：x_oracle 0.910 vs z_oracle 0.887 ⇒ 只丢 0.023。z 里信息是够的。
  ✗ H3 基质污染（均值方向版）：cos(w_T, v_mat) ≈ −0.04，与基质方向几乎正交。
  ⇒ 剩余全部损失来自 **方向**：z 内可达 0.887，化学先验方向只拿到 0.618–0.733。

本脚本回答「方向为什么错」，四条互斥解释各自给一个判据
--------------------------------------------------------
  E1 目标错配  先验头学的是 S → 类中心 Z_j（"这个分子长什么样"），
               而检出需要的是净信号方向 ΔZ_j = Z_j − Z_基质。
               判据：把目标换成 ΔZ 后，cos 与 AUC 是否显著回升（exp71 预测 cos 0.157→0.314）。
  E2 基质子空间污染  w_T 落在「基质中心张成的子空间」里的比例。
               判据：扣掉该子空间后的残差 AUC 是否高于 w_T 本身。
               （注意：exp92 用的是"基质均值方向"，过于粗糙，这里改用整个子空间。）
  E3 化学无贡献  化学码本换成随机码本、其余完全相同，AUC 是否不变。
               判据：rand ≈ chem ⇒ 增量来自目标定义，不来自化学。
  E4 数据方向本可更好  同一 z 上，用老类的观测净信号平均得到的**纯数据方向**能否
               显著超过化学先验方向。判据：auc(gen) − auc(prior)。
               （exp92 的 gen 实现有缺陷：把无基质信息的实体也平均进去了，本版修正。）

★ 所有 oracle 方向（b_z）用到 test 正例均值，仅作上界诊断，不作方法报出。
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


def orth_basis(M):
    """M: [k, d] → 行空间正交基 [d, r]。"""
    if len(M) == 0:
        return np.zeros((M.shape[1], 0))
    u, s, vt = np.linalg.svd(M, full_matrices=False)
    r = int((s > 1e-8 * s.max()).sum())
    return vt[:r].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--priors", default="clm,physchem")
    ap.add_argument("--out", default="exp92b_why_worse2.json")
    args = ap.parse_args()
    pri_list = [x.strip() for x in args.priors.split(",") if x.strip()]

    ent, conc, bg, _ = G1.load_g1_conc()
    GR = load_groups_cached()
    eids = sorted(set(GR) & set(EID2MID))
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]

    zc = np.load(os.path.join(OUT, "exp81_prior_cache.npz"), allow_pickle=True)
    assert list(zc["keys"]) == keys_all
    S_pri = {n: np.asarray(zc[n], float) for n in PRIORS}
    dz = S_pri["clm"].shape[1]
    ki = {k: i for i, k in enumerate(keys_all)}

    res = {p: {} for p in pri_list}
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
            Xpos = Xe[te_ix]
            Xn_all = np.asarray(bg[mid_h], float)
            pp = rng.permutation(len(Xn_all))
            Xneg = Xn_all[pp[:max(5, int(round(0.3 * len(Xn_all))))]]

            parts, ys = [], []
            mid_ids = {}
            for j, e in enumerate(others):
                Xo = np.asarray(ent[e], float)
                if e in GR:
                    a, b, _ = split_group_w(Xo, GR[e]["gid"], rng)
                else:
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
                mid_ids[m] = len(others) + j
            Xtr = np.concatenate(parts).astype(np.float32)
            ytr = np.concatenate(ys).astype(np.int64)
            rows = sorted({int(t) for t in ytr})

            net, _ = stage1(Xtr, ytr, dz, NEW, args.steps,
                            seed * 1000 + eids.index(e_hold),
                            S=S_pri["clm"][order])
            Zf = embed(net, Xtr)
            Zc = np.stack([Zf[ytr == j].mean(0) for j in rows])
            Zmid = {m: Zc[rows.index(mid_ids[m])] for m in mids}
            mu_mid = np.mean([Zmid[m] for m in mids], 0)
            Zpos, Zneg = embed(net, Xpos), embed(net, Xneg)
            mu_z = Zf.mean(0)
            b_z = unit(Zpos.mean(0) - Zneg.mean(0))

            # ---------- 净信号目标 dZ（exp73 定义）----------
            dZ = np.zeros_like(Zc)
            for ri, r in enumerate(rows):
                if r < len(others):
                    e = others[r]
                    m = EID2MID.get(e)
                    dZ[ri] = Zc[ri] - Zmid[m] if m in Zmid else Zc[ri] - mu_mid
                elif r < NEW:
                    dZ[ri] = 0.0
                else:
                    dZ[ri] = Zc[ri] - Zmid[mid_h]

            # ---------- 纯数据方向 b_gen（只用有基质信息的老实体）----------
            same, allm = [], []
            for j, e in enumerate(others):
                m = EID2MID.get(e)
                if m not in Zmid:
                    continue
                d = Zc[j] - Zmid[m]
                allm.append(d)
                if m == mid_h:
                    same.append(d)
            b_gen = unit(np.mean(same if len(same) >= 2 else allm, 0)) \
                if (same or allm) else np.zeros_like(b_z)

            # ---------- 基质子空间 ----------
            Vm = np.stack([Zmid[m] - mu_z for m in mids])
            Q = orth_basis(Vm)                      # [dz, r]
            Sgrid_x = None

            for p in pri_list:
                Sp = S_pri[p][order]
                S_o = Sp[rows]
                mu_s = S_o.mean(0, keepdims=True)

                def wvec(target, Sp_use):
                    W = (Sp_use - mu_s) @ head_ridge_R(S_o, target, LAM)
                    return unit(W[iT])

                w_z = wvec(Zc, Sp)                  # 目标=类中心（现用）
                w_dz = wvec(dZ, Sp)                 # 目标=净信号
                # 随机码本对照（码本换掉，目标与头不变）
                rng_r = np.random.default_rng(seed * 977 + 31)
                Sp_rnd = rng_r.normal(size=Sp.shape)
                Sp_rnd /= (np.linalg.norm(Sp_rnd, axis=1, keepdims=True) + 1e-12)
                Sr_o = Sp_rnd[rows]
                Wr = (Sp_rnd - Sr_o.mean(0, keepdims=True)) @ \
                    head_ridge_R(Sr_o, Zc, LAM)
                w_rnd = unit(Wr[iT])

                # 基质子空间投影
                coef = Q.T @ w_z
                frac_mat = float((coef ** 2).sum())
                w_perp = unit(w_z - Q @ coef)

                def auc_z(dvec):
                    return float(auc_of((Zpos - mu_z) @ dvec,
                                        (Zneg - mu_z) @ dvec))

                rec = dict(
                    auc_oracle_z=auc_z(b_z),
                    auc_gen=auc_z(b_gen),
                    auc_priorZ=auc_z(w_z),
                    auc_priorDZ=auc_z(w_dz),
                    auc_rand=auc_z(w_rnd),
                    auc_perp=auc_z(w_perp),
                    cos_Z_bz=float(w_z @ b_z),
                    cos_DZ_bz=float(w_dz @ b_z),
                    cos_gen_bz=float(b_gen @ b_z),
                    cos_rand_bz=float(w_rnd @ b_z),
                    frac_mat_subspace=frac_mat,
                    cos_Z_mat=float(unit(w_z) @ unit(Q @ (Q.T @ w_z)))
                    if Q.shape[1] else 0.0,
                )
                res[p].setdefault(e_hold, []).append(rec)
            print(f"  [{e_hold} s{seed}] oracle={rec['auc_oracle_z']:.3f} "
                  f"gen={rec['auc_gen']:.3f} Z={rec['auc_priorZ']:.3f} "
                  f"dZ={rec['auc_priorDZ']:.3f} rand={rec['auc_rand']:.3f} "
                  f"perp={rec['auc_perp']:.3f} fmat={rec['frac_mat_subspace']:.2f} "
                  f"cosZ={rec['cos_Z_bz']:+.3f} cosDZ={rec['cos_DZ_bz']:+.3f}",
                  flush=True)

    with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as f:
        json.dump(dict(meta=dict(seeds=args.seeds, priors=pri_list, lam=LAM,
                                 eids=eids,
                                 note="oracle directions use test means; "
                                      "diagnosis only, not a method"),
                       res=res), f, ensure_ascii=False, indent=1)
    print(f"[exp92b] -> {os.path.join(OUT, args.out)}", flush=True)


if __name__ == "__main__":
    main()
