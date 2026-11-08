# -*- coding: utf-8 -*-
"""
exp58b —— E4 裁决：随机码本的高 AUC 是「定向筛查」还是「通用异常检测」？

机制预测（预注册）：
  rand 码本各行近似正交 ⇒ 未见类头 w_T 是一条与所有老类头低相关的新方向
  ⇒ 任何不在库里的样本（**包括另一个未见化合物 D**）在这条方向上响应相似
  ⇒ auc_old(T vs 老类) 高，但 **auc_dis(T vs D) ≈ 0.5**。
  chem 码本 s_T 与 s_D 是不同分子的真实向量 ⇒ w_T、w_D 不同 ⇒ auc_dis 应 > 0.5。

判据：
  H58a：auc_old 上 ridge-rand ≥ ridge-c（复现 exp53 的异常，确认不是偶然）
  H58b：auc_dis 上 ridge-c > ridge-rand，且 ridge-rand 与 0.5 无显著差异
        ⇒ rand 的"高分"是通用 novelty detection，**不构成对定向筛查claim的威胁**
  H58c：对称性 —— rand 的 fwd+bwd ≈ 1.0（与 exp51 的外部方法同构）
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53_p6_zsl_baselines as E53         # noqa: E402
import exp49_p6_clean_protocol as E49        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
DEV = E53.DEV


def auc_of(pos, neg):
    return E49.auc_of(pos, neg)


def run_pair(T, D, ent, bg, eids, CLM, ki, seed, steps, lams, args):
    """留出 T 与 D 两个实体；老类=其余 11 实体 + 6 基质。"""
    others = [k for k in eids if k not in (T, D)]
    mids = sorted(bg.keys())
    rng = np.random.default_rng(seed * 101 + 3)

    Xtr, ytr, Xte_old = [], [], []
    for j, e in enumerate(others):
        Xo = np.asarray(ent[e], np.float64)
        q = rng.permutation(len(Xo))
        nte = max(5, int(0.3 * len(Xo)))
        Xte_old.append(Xo[q[:nte]])          # held-out，不进训练
        Xtr.append(Xo[q[nte:]]); ytr.append(np.full(len(Xo) - nte, j))
    for j, m in enumerate(mids):
        Xo = np.asarray(bg[m], np.float64)
        q = rng.permutation(len(Xo))
        nte = max(5, int(0.3 * len(Xo)))
        Xte_old.append(Xo[q[:nte]])
        Xtr.append(Xo[q[nte:]]); ytr.append(np.full(len(Xo) - nte, len(others) + j))
    Xtr = np.vstack(Xtr).astype(np.float32)
    ytr = np.concatenate(ytr).astype(np.int64)
    Xte_old = np.vstack(Xte_old).astype(np.float32)

    XT = np.asarray(ent[T], np.float64); XD = np.asarray(ent[D], np.float64)
    q1 = np.random.default_rng(seed * 7 + 1).permutation(len(XT))
    q2 = np.random.default_rng(seed * 7 + 2).permutation(len(XD))
    XT = XT[q1[:max(10, int(0.3 * len(XT)))]].astype(np.float32)
    XD = XD[q2[:max(10, int(0.3 * len(XD)))]].astype(np.float32)

    rows_old = sorted(set(ytr.tolist()))
    nrows = len(rows_old)
    rowT, rowD = nrows, nrows + 1            # 两个未见类的头所在行

    # 关键：训练标签 j 对应 others[j] / mids[j-len(others)]，
    # **不能**直接用 CLM[j]（others 已剔除 T、D，与 CLM 行序错位）。
    def name_of(j):
        return others[j] if j < len(others) else f"M_{mids[j - len(others)]}"

    S = np.zeros((nrows + 2, CLM.shape[1]), np.float32)
    for i, j in enumerate(rows_old):
        S[i] = CLM[ki[name_of(j)]]
    S[rowT] = CLM[ki[T]]
    S[rowD] = CLM[ki[D]]
    S_rand = E53.rand_like(S, np.random.default_rng(1000 + eids.index(T)))

    out = {}
    for mode in ((("chem", "free") if args.free else ("chem",))):
        net, _ = E49.stage1(Xtr, ytr, CLM.shape[1], nrows, steps, seed,
                            S=S, mode=mode)
        Zf = E53.embed(net, Xtr)
        Z_old = np.stack([Zf[ytr == j].mean(0) for j in rows_old])
        ZT = E53.embed(net, XT); ZD = E53.embed(net, XD)
        ZO = E53.embed(net, Xte_old)

        # λ 由老类验证集 macro recall 选（与 exp53 同准则）
        val_m = rng.random(len(Xtr)) < 0.2
        Zv, yv = Zf[val_m], ytr[val_m]
        for nm, Sx in ((f"ridge-{'c' if mode == 'chem' else 'f'}", S),
                       (("ridge-rand" if mode == "chem" else None), S_rand)):
            if nm is None:
                continue
            best, bv = lams[0], -1.0
            for lam in lams:
                v = E53.macro_val(Zv, E53.head_ridge(Sx, rows_old, Z_old, lam),
                                  yv, rows_old)
                if v > bv:
                    bv, best = v, lam
            W = E53.head_ridge(Sx, rows_old, Z_old, best)
            sT = E53.lin_scores(ZT, W)[:, rowT]
            sD = E53.lin_scores(XD if False else ZD, W)[:, rowT]
            sO = E53.lin_scores(ZO, W)[:, rowT]
            sD_own = E53.lin_scores(ZD, W)[:, rowD]
            sT_own = E53.lin_scores(ZT, W)[:, rowD]
            out[nm] = dict(
                auc_old=float(auc_of(sT, sO)),
                auc_dis=float(auc_of(sT, sD)),          # T 用 T 的头 vs D
                auc_dis_rev=float(auc_of(sD_own, sT_own)),  # D 用 D 的头 vs T
            )
        del net
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="A03,A06,A01,A07")
    ap.add_argument("--ndis", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lams", default="0.001,0.01,0.1,1")
    ap.add_argument("--free", action="store_true", help="也跑无化学编码器（在双重留出下会崩，默认关）")
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(",")]

    ent, bg = E53.load_data()
    eids = sorted(ent.keys())
    z = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    CLM = z["clm"].astype(np.float64)
    ki = {k: i for i, k in enumerate(keys)}

    tg = [t for t in args.targets.split(",") if t in eids]
    rows = []
    for T in tg:
        pool = [d for d in eids if d != T]
        Ds = pool[:args.ndis]
        for D in Ds:
            for seed in range(args.seeds):
                r = run_pair(T, D, ent, bg, eids, CLM, ki, seed, args.steps, lams, args)
                rows.append((T, D, seed, r))
                print(f"{T} vs {D} seed{seed}: " +
                      "  ".join(f"{k} old={v['auc_old']:.3f} dis={v['auc_dis']:.3f}"
                                f" rev={v['auc_dis_rev']:.3f}"
                                for k, v in r.items()), flush=True)

    print("\n" + "=" * 78)
    print("汇总（按臂平均）")
    print("=" * 78)
    print(f"{'臂':<12}{'auc_old':>10}{'auc_dis':>10}{'dis+rev':>10}{'n':>5}")
    from scipy.stats import ttest_1samp, ttest_rel
    store = {}
    ARMS = [a for a in ("ridge-c", "ridge-f", "ridge-rand") if a in rows[0][3]]
    for a in ARMS:
        o = np.array([r[a]["auc_old"] for _, _, _, r in rows])
        d = np.array([r[a]["auc_dis"] for _, _, _, r in rows])
        rv = np.array([r[a]["auc_dis_rev"] for _, _, _, r in rows])
        store[a] = (o, d, rv)
        p = ttest_1samp(d, 0.5).pvalue if len(d) > 1 else float("nan")
        print(f"{a:<12}{o.mean():10.3f}{d.mean():10.3f}"
              f"{(d + rv).mean():10.3f}{len(d):5d}   (auc_dis vs 0.5: p={p:.4f})")

    print("\n配对检验 (n=%d 对)：" % len(rows))
    for a in [x for x in ARMS if x != "ridge-c"]:
        for m, lbl in ((0, "auc_old"), (1, "auc_dis")):
            x, y = store["ridge-c"][m], store[a][m]
            p = ttest_rel(x, y).pvalue if len(x) > 1 else float("nan")
            print(f"  ridge-c − {a:<11} {lbl:<8} Δ={np.mean(x - y):+.3f}  p={p:.4f}")

    with open(os.path.join(OUT, "exp58b_rand_task.json"), "w",
              encoding="utf-8") as f:
        import json
        json.dump({"rows": [[t, d, s, r] for t, d, s, r in rows]}, f,
                  ensure_ascii=False)
    print("\n[exp58b] 已写出 out/exp58b_rand_task.json")


if __name__ == "__main__":
    main()
