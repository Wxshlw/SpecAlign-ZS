# -*- coding: utf-8 -*-
"""
exp58d —— 随机码本对照的**抽样方差**：A03/A06 上 rand > chem 是真实还是抽样噪声？

exp53 里 `S_rand = rand_like(S, rng(1000 + ei))` 写在 seed 循环**之外**
⇒ 3 个 seed 共用同一个随机码本 ⇒ 报出的 rand 数字是**单次抽样**，方差未知。
（A06 三个 seed 0.869/0.871/0.866 的"稳定"只是同一个码本的复现，不是稳健性。）

本实验：固定数据与编码器（与 exp53 同协议：fit/val 分离、steps、seed），
        只独立重抽 **N 个随机码本**，给出 AUC 分布。
判据 H58e：若 rand 的 AUC 分布跨度过宽（如 0.1~0.9），
          ⇒ exp53 中"A03/A06 上 rand 优于 chem"是抽样偶然，不构成科学结论。
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
NARROW = [1e-3, 1e-2, 1e-1, 1.0]
WIDE = [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="A03,A06,A01,A07")
    ap.add_argument("--nmc", type=int, default=20, help="独立随机码本个数")
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()

    ent, bg = E53.load_data()
    eids = sorted(ent.keys())
    mids = sorted(bg.keys())
    z = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    CLM = z["clm"].astype(np.float64)
    ki = {k: i for i, k in enumerate(keys)}
    NEW = 18

    print("每个实体：1 次编码器训练 + %d 个独立随机码本" % args.nmc)
    print("=" * 92)
    print(f"{'held':<8}{'chem(宽网格)':>13}{'chem(窄网格)':>13}"
          f"{'rand 均值':>11}{'rand 标准差':>11}{'rand min':>10}{'rand max':>10}"
          f"{'P(rand>chem)':>14}")

    summary = {}
    for e_hold in [t for t in args.targets.split(",") if t in eids]:
        ei = eids.index(e_hold)
        others = [e for e in sorted(ent.keys()) if e != e_hold]
        S = np.asarray(CLM)[[ki[e] for e in others]
                            + [ki[f"M_{m}"] for m in mids]
                            + [ki[e_hold]]].astype(np.float32)
        Xe = np.asarray(ent[e_hold], float)

        # 与 exp53 同：seed=0 的切分
        seed = 0
        rng = np.random.default_rng(seed * 7919 + 13)
        perm = rng.permutation(len(Xe))
        n_te = int(min(max(5, round(0.3 * len(Xe))), len(Xe) - 1))
        Xte_new = Xe[perm[:n_te]].astype(np.float32)
        parts, ys, refs = [], [], []
        for j, e in enumerate(others):
            Xo = np.asarray(ent[e], float)
            pp = rng.permutation(len(Xo))
            ntr = max(2, int(round(E49.OLD_FRAC_TRAIN * len(Xo))))
            parts.append(Xo[pp[:ntr]]); ys.append(np.full(ntr, j))
            refs.append(Xo[pp[ntr:]])
        for j, m in enumerate(mids):
            Xo = np.asarray(bg[m], float)
            pp = rng.permutation(len(Xo))
            ntr = max(2, int(round(E49.OLD_FRAC_TRAIN * len(Xo))))
            parts.append(Xo[pp[:ntr]])
            ys.append(np.full(ntr, len(others) + j))
            refs.append(Xo[pp[ntr:]])
        Xtr = np.concatenate(parts).astype(np.float32)
        ytr = np.concatenate(ys).astype(np.int64)
        Xte_old = np.concatenate(refs).astype(np.float32)
        rows_old = sorted({int(y) for y in ytr})

        rng_v = np.random.default_rng(seed * 31 + 7)
        val_mask = np.zeros(len(Xtr), bool)
        for j in rows_old:
            ix = rng_v.permutation(np.where(ytr == j)[0])
            n_val = min(max(3, int(0.2 * len(ix))), max(0, len(ix) - 1))
            val_mask[ix[:n_val]] = True
        Xfit, yfit = Xtr[~val_mask], ytr[~val_mask]
        Xval, yval = Xtr[val_mask], ytr[val_mask]
        val_rows = sorted({int(y) for y in yval})

        net, _ = E49.stage1(Xfit, yfit, CLM.shape[1], len(rows_old),
                            args.steps, seed * 1000 + ei, S=S, mode="chem")
        Z_fit = E53.embed(net, Xfit)
        Z_old = np.stack([Z_fit[yfit == j].mean(0) for j in rows_old])
        Zn = E53.embed(net, Xte_new)
        Zo = E53.embed(net, Xte_old)
        Zv = E53.embed(net, Xval)

        def auc_of(Sx, grid):
            best, bv = grid[0], -1.0
            for lam in grid:
                v = E53.macro_val(Zv, E53.head_ridge(Sx, rows_old, Z_old, lam),
                                  yval, val_rows)
                if v > bv:
                    bv, best = v, lam
            W = E53.head_ridge(Sx, rows_old, Z_old, best)
            return float(E49.auc_of(E53.lin_scores(Zn, W)[:, NEW],
                                    E53.lin_scores(Zo, W)[:, NEW])), best

        cw, lw = auc_of(S, WIDE)
        cn, _ = auc_of(S, NARROW)

        rands = []
        for m in range(args.nmc):
            Sr = E53.rand_like(S, np.random.default_rng(90000 + m * 977 + ei))
            a, _ = auc_of(Sr, WIDE)
            rands.append(a)
        rands = np.array(rands)
        pr = float((rands > cw).mean())
        summary[e_hold] = dict(chem_wide=cw, chem_narrow=cn,
                               rand_mean=float(rands.mean()),
                               rand_std=float(rands.std()),
                               rand_min=float(rands.min()),
                               rand_max=float(rands.max()), p_rand_better=pr)
        print(f"{e_hold:<8}{cw:13.3f}{cn:13.3f}{rands.mean():11.3f}"
              f"{rands.std():11.3f}{rands.min():10.3f}{rands.max():10.3f}"
              f"{pr:14.2f}")
        print(f"         λ*(chem宽)={lw:g}   rand 分位数[10,50,90]= "
              f"{np.round(np.percentile(rands, [10, 50, 90]), 3)}")
        del net
        torch.cuda.empty_cache()

    print("=" * 92)
    print("结论提示：若 rand 标准差 > 0.1，则 exp53 的单次抽样数字不可作为结论，")
    print("          必须报告 rand 的抽样分布（本表）而非点估计。")

    import json
    with open(os.path.join(OUT, "exp58d_rand_mc.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("\n[exp58d] 已写出 out/exp58d_rand_mc.json")


if __name__ == "__main__":
    main()
