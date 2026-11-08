# -*- coding: utf-8 -*-
"""
exp58 —— 诊断：为什么 A03 / A06 上「零信息随机码本」AUC 反而高于真实化学码本？

四类候选解释（预先列出，用数据裁决）：
  E1 码本几何：真实化学向量行间高度共线（都是有机分子），S S^T 病态
               ⇒ ridge 解出的未见类头 ≈ 老类头的线性组合，没给出"新方向"；
               随机高斯向量在高维下近似正交 ⇒ 天然是"好码本"(ECOC 已知结论)。
               若成立：offdiag_cos(S) 应 >> offdiag_cos(S_rand)，cond(S S^T) 应 >> 1。
  E2 化学向量本身错/退化：A03/A06 的 SMILES 缺失或与其他实体重复
               ⇒ 化学先验是"负信息"而非"无信息"。若成立：其 chem 行范数/相似度异常。
  E3 λ 选择错配：λ 在老类验证集上选，对未见类非最优。
               若成立：用未见类 oracle λ 扫，ridge-c 上限应能追上 ridge-rand。
  E4 rand 的高 AUC 是"通用异常检测"而非"定向筛查"（不是 bug，是任务口径不同）。
               若成立：ridge-rand 的 auc_dis（未见类 vs 另一个同样未见的干扰物）应 ≈ 0.5。

本脚本用纯测量回答 E1/E2/E3；E4 由 exp50/51 协议单独跑（见 exp58b）。
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1                      # noqa: E402
import exp53_p6_zsl_baselines as E53         # noqa: E402
import exp49_p6_clean_protocol as E49        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
DEV = E53.DEV


def offdiag_max(S, rows):
    """给定码本，返回指定行集合两两余弦的最大值/均值（去均值前后都要看）"""
    A = S[rows]
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    G = A @ A.T
    n = len(rows)
    iu = np.triu_indices(n, 1)
    return float(G[iu].max()), float(G[iu].mean()), G


def gram_stats(S, rows, lam):
    A = S[rows]
    G = A @ A.T
    w = np.linalg.eigvalsh(G)
    w = w[w > 1e-12]
    return float(np.linalg.cond(G + lam * np.eye(len(rows)))), float(w.max() / w.min())


def main():
    ent, bg = E53.load_data()
    eids = sorted(ent.keys())
    z = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    CLM = z["clm"].astype(np.float64)
    ki = {k: i for i, k in enumerate(keys)}

    print("=" * 78)
    print("E2 检查：19 个实体的化学向量是否退化 / 重复 / 范数异常")
    print("=" * 78)
    nrm = np.linalg.norm(CLM, axis=1)
    C = CLM / (nrm[:, None] + 1e-12)
    G = C @ C.T
    print(f"{'实体':<9}{'范数':>9}{'最近邻':>10}{'最近邻是谁':>12}{'次近邻':>9}")
    for i, k in enumerate(keys):
        g = G[i].copy(); g[i] = -2
        o1, o2 = np.argsort(g)[::-1][:2]
        print(f"{k:<9}{nrm[i]:9.3f}{g[o1]:10.4f}{keys[o1]:>12}{g[o2]:9.4f}")
    dup = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))
           if G[i, j] > 0.999]
    print("近乎重复的化学向量对 (cos>0.999):", dup if dup else "无")
    zero = [keys[i] for i in range(len(keys)) if nrm[i] < 1e-6]
    print("零向量:", zero if zero else "无")

    print()
    print("=" * 78)
    print("E1 检查：真实化学码本 vs 随机码本的几何（行间夹角 / Gram 条件数）")
    print("=" * 78)
    print(f"{'held-out':<9}{'chem 行间cos max':>17}{'rand 行间cos max':>17}"
          f"{'cond(S S^T) chem':>18}{'cond rand':>12}")
    rows_all = list(range(len(keys)))
    for e_hold in ["A03", "A06", "A01", "A07"]:
        if e_hold not in ki:
            continue
        others = [k for k in keys if k != e_hold]
        rows_old = [ki[k] for k in others]
        rng = np.random.default_rng(1000 + eids.index(e_hold))
        S_rand = E53.rand_like(CLM, rng)
        m1, a1, _ = offdiag_max(CLM, rows_old)
        m2, a2, _ = offdiag_max(S_rand, rows_old)
        c1, r1 = gram_stats(CLM, rows_old, 1.0)
        c2, r2 = gram_stats(S_rand, rows_old, 1.0)
        print(f"{e_hold:<9}{m1:17.4f}{m2:17.4f}{c1:18.1f}{c2:12.1f}")

    # held-out 实体自己的化学向量 vs 老类：共线程度
    print()
    print(f"{'held-out':<9}{'自身到老类max cos':>18}{'rand 版到老类max cos':>22}")
    for e_hold in ["A03", "A06", "A01", "A07"]:
        if e_hold not in ki:
            continue
        others = [k for k in keys if k != e_hold]
        rows_old = [ki[k] for k in others]
        rng = np.random.default_rng(1000 + eids.index(e_hold))
        S_rand = E53.rand_like(CLM, rng)
        i = ki[e_hold]
        a = CLM[i] / (np.linalg.norm(CLM[i]) + 1e-12)
        B = CLM[rows_old] / (np.linalg.norm(CLM[rows_old], axis=1, keepdims=True) + 1e-12)
        b = S_rand[i] / (np.linalg.norm(S_rand[i]) + 1e-12)
        Br = S_rand[rows_old] / (np.linalg.norm(S_rand[rows_old], axis=1,
                                                keepdims=True) + 1e-12)
        print(f"{e_hold:<9}{float((B @ a).max()):18.4f}{float((Br @ b).max()):22.4f}")

    # ---------------- E3：λ oracle 扫描（作弊上限） ----------------
    print()
    print("=" * 78)
    print("E3 检查：A03/A06 上 ridge-c 的 λ-oracle 上限（能否追上 ridge-rand）")
    print("需在真实协议下重跑 stage1，只跑 2 个实体 × 1 seed")
    print("=" * 78)
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    LAMS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

    for e_hold in ["A03", "A06"]:
        mids = sorted(bg.keys())
        others = [k for k in eids if k != e_hold]
        rng = np.random.default_rng(0)
        Xtr, ytr = [], []
        for j, e in enumerate(others):
            Xo = np.asarray(ent[e], np.float64)
            pp = rng.permutation(len(Xo))
            Xtr.append(Xo[pp]); ys = np.full(len(Xo), j)
            ytr.append(ys)
        for j, m in enumerate(mids):
            Xo = np.asarray(bg[m], np.float64)
            pp = rng.permutation(len(Xo))
            Xtr.append(Xo[pp]); ytr.append(np.full(len(Xo), len(others) + j))
        Xtr = np.vstack(Xtr).astype(np.float32)
        ytr = np.concatenate(ytr).astype(np.int64)
        rows_old = sorted(set(ytr.tolist()))
        nrows = len(rows_old)

        # 未见类测试集
        Xh = np.asarray(ent[e_hold], np.float64)
        ph = np.random.default_rng(999).permutation(len(Xh))
        nte = max(5, int(0.3 * len(Xh)))
        Xte_new = Xh[ph[:nte]].astype(np.float32)

        # 老类 held-out（干净协议：不进训练）
        Xte_old = []
        for j, e in enumerate(others):
            Xo = np.asarray(ent[e], np.float64)
            q = np.random.default_rng(j).permutation(len(Xo))
            Xte_old.append(Xo[q[-max(5, int(0.3 * len(Xo))):]])
        for m in mids:
            Xo = np.asarray(bg[m], np.float64)
            q = np.random.default_rng(len(others) + mids.index(m)).permutation(len(Xo))
            Xte_old.append(Xo[q[-max(5, int(0.3 * len(Xo))):]])
        Xte_old = np.vstack(Xte_old).astype(np.float32)

        NEW = nrows
        # 关键：S 必须与训练标签对齐（others→mids→e_hold），
        # **不能**直接取 CLM[:nrows]（others 已剔除 e_hold，行序错位）。
        S = np.asarray(CLM)[[ki[e] for e in others]
                            + [ki[f"M_{m}"] for m in mids]
                            + [ki[e_hold]]].astype(np.float32)
        S_rand = E53.rand_like(CLM, np.random.default_rng(
            1000 + eids.index(e_hold))).astype(np.float32)

        net, _ = E49.stage1(Xtr, ytr, CLM.shape[1], nrows, 400, 0, S=S, mode="chem")
        Zf = E53.embed(net, Xtr)
        Zn = E53.embed(net, Xte_new)
        Zo = E53.embed(net, Xte_old)
        Z_old = np.stack([Zf[ytr == j].mean(0) for j in rows_old])

        print(f"\n--- {e_hold} ---")
        print(f"{'λ':>10}{'ridge-c auc':>14}{'ridge-rand auc':>16}")
        best = (-1, None)
        for lam in LAMS:
            Wc = E53.head_ridge(S, rows_old, Z_old, lam)
            Wr = E53.head_ridge(S_rand, rows_old, Z_old, lam)
            ac = float(E49.auc_of(E53.lin_scores(Zn, Wc)[:, NEW],
                                  E53.lin_scores(Zo, Wc)[:, NEW]))
            ar = float(E49.auc_of(E53.lin_scores(Zn, Wr)[:, NEW],
                                  E53.lin_scores(Zo, Wr)[:, NEW]))
            print(f"{lam:10.4g}{ac:14.3f}{ar:16.3f}")
            if ac > best[0]:
                best = (ac, lam)
        print(f"  → ridge-c λ-oracle 上限 = {best[0]:.3f} (λ={best[1]:g})")

        # 未见类头与老类头的夹角：衡量"是否给出了新方向"
        for nm, Sx in (("chem", S), ("rand", S_rand)):
            for lam in (0.01, 1.0):
                W = E53.head_ridge(Sx, rows_old, Z_old, lam)
                wn = W[NEW] / (np.linalg.norm(W[NEW]) + 1e-12)
                Wo = W[rows_old] / (np.linalg.norm(W[rows_old], axis=1,
                                                   keepdims=True) + 1e-12)
                cs = Wo @ wn
                print(f"  {nm:<5} λ={lam:<6g} 未见类头到老类头 max cos = {cs.max():.3f}"
                      f"  (|w_new|/mean|w_old| = "
                      f"{np.linalg.norm(W[NEW]) / np.linalg.norm(W[rows_old], axis=1).mean():.3f})")
        del net
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
