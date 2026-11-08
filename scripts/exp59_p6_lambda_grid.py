# -*- coding: utf-8 -*-
"""
exp59 —— 修复：exp53 的 λ 网格 {1e-3,1e-2,1e-1,1} 未覆盖 A06 的最优点（实测 λ*=1e-4）

exp58 的 λ-oracle 扫描已证明：
  A06  ridge-c oracle = 0.972 @ λ=1e-4   （exp53 窄网格只报 0.618）
  A03  ridge-c oracle = 0.506 @ λ=1e-4   （仍低于 ridge-rand 0.735 ⇒ 真困难）

本实验把 λ 网格向小端扩展到 {1e-5 … 1}，在**与 exp53 完全相同的切分与选择准则**
（老类验证集 macro recall）下重跑，回答：
  H59a：扩展网格后 ridge-c 的 macro AUC 是否显著提升？A03/A06 是否翻盘？
  H59b：扩展网格后 ridge-rand 是否同步提升？（若同步提升则增益来自网格而非化学内容，
         必须保证 ridge-c − ridge-rand 的差不缩小才算真修复）
  H59c：选出 λ 的分布 —— 若大量落在 <1e-3，说明原网格系统性偏大。
"""
from __future__ import annotations

import os
import sys
import argparse
import json
import numpy as np
import torch
from scipy.stats import ttest_rel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53_p6_zsl_baselines as E53         # noqa: E402
import exp49_p6_clean_protocol as E49        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
DEV = E53.DEV

NARROW = [1e-3, 1e-2, 1e-1, 1.0]
WIDE = [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eids", default="")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--shots", default="0")
    args = ap.parse_args()

    ent, bg = E53.load_data()
    eids = sorted(ent.keys())
    if args.eids:
        eids = [e for e in args.eids.split(",") if e in eids]
    mids = sorted(bg.keys())
    z = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    CLM = z["clm"].astype(np.float64)
    ki = {k: i for i, k in enumerate(keys)}
    NEW = 18

    res = {"acc": {}, "lam": {}}
    for ei, e_hold in enumerate(eids):
        others = [e for e in sorted(ent.keys()) if e != e_hold]
        # 与 exp53 完全一致：others(12) + mids(6) + e_hold(1)
        S = np.asarray(CLM)[[ki[e] for e in others]
                            + [ki[f"M_{m}"] for m in mids]
                            + [ki[e_hold]]].astype(np.float32)
        S_rand = E53.rand_like(S, np.random.default_rng(1000 + eids.index(e_hold)))
        Xe = np.asarray(ent[e_hold], float)

        for k in (0,):
            tag = str(k)
            res["acc"].setdefault(tag, {})
            res["lam"].setdefault(tag, {})
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed * 7919 + 13)
                perm = rng.permutation(len(Xe))
                n_te = int(min(max(5, round(0.3 * len(Xe))), len(Xe) - 1))
                te_ix, pool_ix = perm[:n_te], perm[n_te:]
                Xte_new = Xe[te_ix].astype(np.float32)
                parts, ys, refs = [], [], []
                for j, e in enumerate(others):
                    Xo = np.asarray(ent[e], float)
                    pp = rng.permutation(len(Xo))
                    ntr = max(2, int(round(0.7 * len(Xo))))
                    parts.append(Xo[pp[:ntr]]); ys.append(np.full(ntr, j))
                    refs.append(Xo[pp[ntr:]])
                for j, m in enumerate(mids):
                    Xo = np.asarray(bg[m], float)
                    pp = rng.permutation(len(Xo))
                    ntr = max(2, int(round(0.7 * len(Xo))))
                    parts.append(Xo[pp[:ntr]])
                    ys.append(np.full(ntr, len(others) + j))
                    refs.append(Xo[pp[ntr:]])
                Xtr = np.vstack(parts).astype(np.float32)
                ytr = np.concatenate(ys).astype(np.int64)
                Xte_old = np.vstack(refs).astype(np.float32)
                rows_old = sorted(set(ytr.tolist()))

                # ---- 严格复刻 exp53：分层验证掩码，且验证集**不进**训练集 ----
                rng_v = np.random.default_rng(seed * 31 + 7)
                val_mask = np.zeros(len(Xtr), bool)
                for j in rows_old:
                    ix = rng_v.permutation(np.where(ytr == j)[0])
                    n_val = min(max(3, int(0.2 * len(ix))),
                                max(0, len(ix) - 1))
                    val_mask[ix[:n_val]] = True
                Xfit, yfit = Xtr[~val_mask], ytr[~val_mask]
                Xval, yval = Xtr[val_mask], ytr[val_mask]

                net, _ = E49.stage1(Xfit, yfit, CLM.shape[1], len(rows_old),
                                    args.steps, seed * 1000 + eids.index(e_hold),
                                    S=S, mode="chem")
                Z_fit = E53.embed(net, Xfit)
                Z_old = np.stack([Z_fit[yfit == j].mean(0) for j in rows_old])
                Zn = E53.embed(net, Xte_new)
                Zo = E53.embed(net, Xte_old)
                Zv = E53.embed(net, Xval)

                val_rows = sorted({int(y) for y in yval})

                # rand 臂做多次独立抽样（exp58d 已证明单次抽样方差 σ≈0.2）
                NMC_R = 5
                for nm, Sx, grid in (("ridge-c-narrow", S, NARROW),
                                     ("ridge-c-wide", S, WIDE)):
                    best, bv = grid[0], -1.0
                    for lam in grid:
                        v = E53.macro_val(Zv, E53.head_ridge(Sx, rows_old,
                                                             Z_old, lam),
                                          yval, val_rows)
                        if v > bv:
                            bv, best = v, lam
                    W = E53.head_ridge(Sx, rows_old, Z_old, best)
                    sn = E53.lin_scores(Zn, W)[:, NEW]
                    so = E53.lin_scores(Zo, W)[:, NEW]
                    res["acc"][tag].setdefault(nm, {}).setdefault(
                        e_hold, []).append(dict(
                            auc=round(float(E49.auc_of(sn, so)), 4),
                            top1=float(np.mean(
                                E53.lin_scores(Zn, W).argmax(1) == NEW)),
                            mrr=None))
                    res["lam"][tag].setdefault(nm, {}).setdefault(
                        e_hold, []).append(best)

                for nm, grid in (("ridge-rand-narrow", NARROW),
                                 ("ridge-rand-wide", WIDE)):
                    aucs = []
                    for m in range(NMC_R):
                        Sr = E53.rand_like(
                            S, np.random.default_rng(
                                90000 + m * 977 + eids.index(e_hold)))
                        best, bv = grid[0], -1.0
                        for lam in grid:
                            v = E53.macro_val(
                                Zv, E53.head_ridge(Sr, rows_old, Z_old, lam),
                                yval, val_rows)
                            if v > bv:
                                bv, best = v, lam
                        W = E53.head_ridge(Sr, rows_old, Z_old, best)
                        aucs.append(float(E49.auc_of(
                            E53.lin_scores(Zn, W)[:, NEW],
                            E53.lin_scores(Zo, W)[:, NEW])))
                    res["acc"][tag].setdefault(nm, {}).setdefault(
                        e_hold, []).append(dict(
                            auc=round(float(np.mean(aucs)), 4),
                            auc_std=round(float(np.std(aucs)), 4),
                            top1=None, mrr=None))

                del net
                torch.cuda.empty_cache()
        print(f"[{e_hold}] done", flush=True)

    with open(os.path.join(OUT, "exp59_lam_grid.json"), "w",
              encoding="utf-8") as f:
        json.dump({"meta": {"eids": eids, "seeds": args.seeds,
                            "narrow": NARROW, "wide": WIDE},
                   **res}, f, ensure_ascii=False)

    # ---------------- 报告 ----------------
    print("\n" + "=" * 84)
    print("扩展 λ 网格后（13 实体 × 3 seeds，k=0）")
    print("=" * 84)
    print(f"{'臂':<20}{'macro AUC':>11}{'worst':>9}{'vs narrow':>11}{'p':>9}")
    ARMS = ["ridge-c-narrow", "ridge-c-wide", "ridge-rand-narrow",
            "ridge-rand-wide"]

    def pe(a, m="auc"):
        return {e: np.mean([x[m] for x in res["acc"]["0"].get(a, {}).get(e, [])
                            if x[m] is not None])
                for e in eids if res["acc"]["0"].get(a, {}).get(e)}

    base = pe("ridge-c-narrow")
    for a in ARMS:
        d = pe(a)
        y = np.array([d[e] for e in eids])
        x = np.array([base[e] for e in eids])
        p = ttest_rel(y, x).pvalue
        print(f"{a:<20}{y.mean():11.3f}{y.min():9.3f}"
              f"{y.mean() - x.mean():+11.3f}{p:9.4f}")

    print("\n逐实体（关键看 A03 / A06）")
    print(f"{'实体':<9}{'c-narrow':>10}{'c-wide':>9}{'Δ':>8}"
          f"{'rand-narrow':>13}{'rand-wide':>11}{'c-wide − rand-wide':>20}")
    for e in eids:
        cn = pe("ridge-c-narrow")[e]; cw = pe("ridge-c-wide")[e]
        rn = pe("ridge-rand-narrow")[e]; rw = pe("ridge-rand-wide")[e]
        print(f"{e:<9}{cn:10.3f}{cw:9.3f}{cw - cn:+8.3f}{rn:13.3f}"
              f"{rw:11.3f}{cw - rw:+20.3f}")

    print("\n选出的 λ 分布（c-wide）")
    from collections import Counter
    for a in ("ridge-c-wide", "ridge-rand-wide", "ridge-c-narrow"):
        ls = [l for e in eids for l in res["lam"]["0"].get(a, {}).get(e, [])]
        c = Counter(ls)
        small = sum(v for k, v in c.items() if k < 1e-3)
        print(f"  {a:<18} λ<1e-3 占比 {small}/{len(ls)}  "
              f"分布: {dict(sorted(c.items()))}")

    print("\n配对检验 (n=%d)，H59b：增益是否只来自网格？" % len(eids))
    cw = np.array([pe("ridge-c-wide")[e] for e in eids])
    rw = np.array([pe("ridge-rand-wide")[e] for e in eids])
    cn = np.array([pe("ridge-c-narrow")[e] for e in eids])
    rn = np.array([pe("ridge-rand-narrow")[e] for e in eids])
    print(f"  ridge-c-wide − ridge-rand-wide : Δ={np.mean(cw - rw):+.3f} "
          f"p={ttest_rel(cw, rw).pvalue:.5f}")
    print(f"  ridge-c-narrow − ridge-rand-nar: Δ={np.mean(cn - rn):+.3f} "
          f"p={ttest_rel(cn, rn).pvalue:.5f}")
    print(f"  化学增益是否扩大: {(cw - rw).mean() - (cn - rn).mean():+.3f}")
    print("\n[exp59] 已写出 out/exp59_lam_grid.json")


if __name__ == "__main__":
    main()
