# -*- coding: utf-8 -*-
r"""exp99 —— 随机码本对照的「对等 readout」Monte Carlo（回应 Major 2）

背景
----
主表 / Fig.3 的结构 vs 随机对照用的是 ``A:rand384|lam0.0001``：
**一个**固定随机码本（out/exp81_prior_cache.npz 里的 rand384，19×384），
走与结构码本完全相同的岭回归读出，19 个候选对单位等权均值 = 0.2234。

exp82 里另有一个 ``A:rand384`` 臂（NRAND=10）均值 0.5109，但它是
**直接把随机向量投影到 Z 空间**，没有走 ridge / 中心化，与结构臂不对等，
因此不能拿来替换主对照。

审稿人质疑：若 auc_dis 在随机码本下的理论期望是 0.5，则 0.223 是
「单次抽样恰好反向」，+0.602 里混入了抽样运气。要回答这个问题，
必须让多个独立随机码本走**与结构臂完全相同**的读出流程。

本实验
------
精确复刻 exp82 轴 A 的数据加载、组合抽样、划分与 stage1：
  - si ∈ {0,1,2}, ci = default_rng(si*7919+13).permutation(n_combos)[:4]
  - stage1(..., seed=si*1000+ci, S=S_all["clm"])  （与 exp82 逐字一致）
  - 对每个 (si,ci)：
      * 6 个结构码本 -> struct[p][unit]
      * 缓存里那个固定 rand384 -> fixed_rand[unit]   ← 交叉验证，应复现 0.223
      * N 个独立随机码本 -> mc[m][unit]              ← 新证据
    全部走 ``(Sc-mu) @ head_ridge_R(S_o, Z_old, lam=1e-4)``

输出 out/exp99_rand_mc_matched.json + out/exp99_rand_mc_matched.md
"""
from __future__ import annotations
import os, sys, json, argparse, itertools, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "论文8"))
import p6_g1_data as G1
from exp63_p6_screening_protocol import (stage1, embed, head_ridge_R, nrm,
                                         auc_of, EID2MID)
from exp68_p6_cond_scorer import split_by_group
from exp71_p6_detection_fix import load_groups_cached
from exp78_p6_attribution_proof import split_group_w
from exp81b_p6_benchmark_zoo import PRIORS

OUT = os.path.join(os.path.dirname(HERE), "out")
OLD_FRAC = 0.7
STRUCT6 = ["clm", "ecfp384", "ecfp2048", "maccs", "avalon", "charbag"]
LAM = 1e-4


def unit_mean(d, key):
    """与 exp95 / fig3 完全一致的「单位内平均」原语。"""
    out = {}
    for u, recs in d.items():
        vs = [float(r[key]) for r in recs
              if isinstance(r, dict) and r.get(key) is not None and np.isfinite(r[key])]
        if vs:
            out[u] = float(np.mean(vs))
    return out


def paired_t(a, b):
    from scipy import stats
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks], float)
    t, p = stats.ttest_1samp(d, 0)
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(len(d))
    tc = stats.t.ppf(.975, len(d) - 1)
    return dict(n=len(d), d=float(d.mean()), sd=sd, t=float(t), p=float(p),
                ci=[float(d.mean() - tc * se), float(d.mean() + tc * se)],
                dz=float(d.mean() / sd) if sd > 0 else float("nan"),
                pts=d.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmc", type=int, default=30, help="独立随机码本个数")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--nrep", type=int, default=4)
    ap.add_argument("--nhu", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    ent, conc, bg, _ = G1.load_g1_conc()
    GR = load_groups_cached()
    eids = sorted(set(GR) & set(EID2MID))
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]
    zc = np.load(os.path.join(OUT, "exp81_prior_cache.npz"), allow_pickle=True)
    assert list(zc["keys"]) == keys_all, "先验缓存 key 顺序不一致，禁止对齐！"
    S_pri = {n: np.asarray(zc[n], float) for n in PRIORS}
    dz = S_pri["clm"].shape[1]
    ki = {k: i for i, k in enumerate(keys_all)}
    combos = list(itertools.combinations(eids, args.nhu))

    print(f"[exp99] nmc={args.nmc} seeds={args.seeds} nrep={args.nrep} "
          f"|U|={args.nhu} eids={eids} combos={len(combos)} dz={dz}", flush=True)

    cells = []
    seeds = min(args.seeds, 1) if args.smoke else args.seeds
    nrep = 1 if args.smoke else args.nrep
    nmc = 5 if args.smoke else args.nmc

    for si in range(seeds):
        rng0 = np.random.default_rng(si * 7919 + 13)
        cis = rng0.permutation(len(combos))[:nrep]
        for ci in cis:
            U = list(combos[ci])
            others = [e for e in sorted(ent) if e not in U]
            order = ([ki[e] for e in others]
                     + [ki[f"M_{m}"] for m in mids]
                     + [ki[u] for u in U])
            S_all = {p: S_pri[p][order] for p in PRIORS}
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
                parts.append(Xo[a]); ys.append(np.full(len(a), j))
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
            net, _ = stage1(Xtr, ytr, dz, nrows, args.steps,
                            si * 1000 + ci, S=S_all["clm"])
            Zfit = embed(net, Xtr)
            Z_old = np.stack([Zfit[ytr == j].mean(0) for j in rows])
            Zte = {u: embed(net, Xte_U[u]) for u in U}
            print(f"  [{tag}] U={'-'.join(U)} stage1 {time.time()-t0:.1f}s", flush=True)

            def scores_for(R):
                """R 已按 order 排列；结构臂与随机臂共用这一条读出流程。"""
                S_o = R[rows]
                mu = S_o.mean(0, keepdims=True)
                Sc = R[cand_ix]
                Wc = (Sc - mu) @ head_ridge_R(S_o, Z_old, LAM)
                out = {}
                for jT, jD in itertools.combinations(range(len(U)), 2):
                    T, D = U[jT], U[jD]
                    dT = Zte[T] @ Wc[jT] - Zte[T] @ Wc[jD]
                    dD = Zte[D] @ Wc[jT] - Zte[D] @ Wc[jD]
                    out[f"{T}|{D}"] = float(auc_of(dT, dD))
                return out

            struct = {p: scores_for(S_all[p]) for p in STRUCT6}
            fixed = scores_for(S_all["rand384"])
            mc = []
            for m in range(nmc):
                rr = np.random.default_rng(880000 + si * 100000 + ci * 1000 + m * 97 + 5)
                R = nrm(rr.normal(size=(len(keys_all), dz)).astype(np.float32))
                mc.append(scores_for(R))

            for unit in struct[STRUCT6[0]]:
                cells.append(dict(tag=tag, unit=unit,
                                  struct={p: struct[p][unit] for p in STRUCT6},
                                  fixed_rand=fixed[unit],
                                  mc=[x[unit] for x in mc]))
            del net
            torch.cuda.empty_cache()

    res = dict(META=dict(nmc=nmc, seeds=seeds, nrep=nrep, nhu=args.nhu,
                         steps=args.steps, lam=LAM, struct=STRUCT6,
                         note="matched readout: (Sc-mu) @ head_ridge_R(S_o,Z_old,lam)"),
               cells=cells)
    suffix = "_smoke" if args.smoke else ""
    jp = os.path.join(OUT, f"exp99_rand_mc_matched{suffix}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    print(f"[exp99] wrote {jp} | cells={len(cells)}")

    # ------------------------------------------------------------------ 分析
    from collections import defaultdict
    acc_s = defaultdict(list); acc_f = defaultdict(list); acc_m = defaultdict(list)
    for c in cells:
        acc_s[c["unit"]].append(float(np.mean(list(c["struct"].values()))))
        acc_f[c["unit"]].append(c["fixed_rand"])
        acc_m[c["unit"]].append(float(np.mean(c["mc"])))
    S = {u: float(np.mean(v)) for u, v in acc_s.items()}
    F = {u: float(np.mean(v)) for u, v in acc_f.items()}
    M = {u: float(np.mean(v)) for u, v in acc_m.items()}
    print("\n=== unit-equal means (n=%d units) ===" % len(S))
    print("  struct mean      = %.4f" % np.mean(list(S.values())))
    print("  fixed rand (1 draw, should reproduce 0.2234) = %.4f" % np.mean(list(F.values())))
    print("  MC rand (%d draws) = %.4f" % (nmc, np.mean(list(M.values()))))
    d_fixed = paired_t(S, F)
    d_mc = paired_t(S, M)
    print("\n  Delta vs fixed single draw : %+.4f  p=%.3g  CI=%s  n=%d" %
          (d_fixed["d"], d_fixed["p"], [round(x, 4) for x in d_fixed["ci"]], d_fixed["n"]))
    print("  Delta vs MC-matched null   : %+.4f  p=%.3g  CI=%s  n=%d" %
          (d_mc["d"], d_mc["p"], [round(x, 4) for x in d_mc["ci"]], d_mc["n"]))
    from scipy import stats as st
    rv = np.array(list(M.values()))
    tt, pp = st.ttest_1samp(rv, 0.5)
    print("  MC rand vs 0.5             : mean=%.4f sd=%.4f t=%.3f p=%.4g" %
          (rv.mean(), rv.std(ddof=1), tt, pp))
    # per-draw Delta distribution
    per_draw = []
    for m in range(nmc):
        dd = []
        for c in cells:
            sm = float(np.mean(list(c["struct"].values())))
            dd.append(sm - c["mc"][m])
        per_draw.append(float(np.mean(dd)))
    per_draw = np.array(per_draw)
    print("\n  per-draw Delta: mean=%.4f sd=%.4f min=%.4f max=%.4f" %
          (per_draw.mean(), per_draw.std(ddof=1), per_draw.min(), per_draw.max()))
    print("  fraction of draws giving Delta >= 0.602 = %.3f" % float((per_draw >= 0.602).mean()))
    print("  percentiles [5,25,50,75,95] = %s" % np.round(np.percentile(per_draw, [5,25,50,75,95]), 4))

    rep = dict(unit_equal=dict(struct=float(np.mean(list(S.values()))),
                               fixed_rand=float(np.mean(list(F.values()))),
                               mc_rand=float(np.mean(list(M.values()))),
                               n_units=len(S)),
               delta_fixed=d_fixed, delta_mc=d_mc,
               mc_vs_half=dict(t=float(tt), p=float(pp)),
               per_draw_delta=dict(mean=float(per_draw.mean()), sd=float(per_draw.std(ddof=1)),
                                   min=float(per_draw.min()), max=float(per_draw.max()),
                                   frac_ge_0602=float((per_draw >= 0.602).mean()),
                                   percentiles=np.percentile(per_draw, [5,25,50,75,95]).tolist()))
    rp = os.path.join(OUT, f"exp99_rand_mc_matched{suffix}_report.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"[exp99] wrote {rp}")


if __name__ == "__main__":
    main()
