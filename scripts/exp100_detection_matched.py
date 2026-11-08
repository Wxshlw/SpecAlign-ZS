# -*- coding: utf-8 -*-
r"""exp100 —— detection 的 readout-specific 边界：对等随机对照（回应 Major 4 / Minor 3）

审稿人 Major 4
--------------
「detection 的 prior null 可能是 readout（从 old-class 类中心 Zc 外推）mis-specified，
而非任务本质。应加一个 detection-specific readout。」

exp92b 已经做过一半
-------------------
它算了 ``w_dZ``（把 ridge 目标从 Zc 换成净信号 dZ = Z_j − Z_基质）：
AUC 0.618 -> 0.703（+0.085），但只与 ``w_rnd``（随机码本 + **Zc 目标**）比较，
两个臂的 readout 不对等；且随机码本只抽了 1 次（与 Major 2 同类问题）。

本实验补齐两点
--------------
1. 随机码本也走 **dZ 目标**（w_rnd_dZ），得到严格对等的 ``w_dZ − w_rnd_dZ``；
2. 随机码本抽 **NRAND=10** 次，报告单位等权均值与抽样分布（顺带回答 Minor 3）。

第一个 draw 刻意复用 exp92b 的 RNG（default_rng(seed*977+31)）以复现
``auc_rand = 0.6107``，作为流程正确性交叉验证。

输出 out/exp100_detection_matched.json + out/exp100_detection_matched_report.json
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch

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
NRAND = 10


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def paired_report(a, b, label):
    from scipy import stats
    ks = sorted(set(a) & set(b))
    x = np.array([a[k] - b[k] for k in ks], float)
    n = len(x); t, p = stats.ttest_1samp(x, 0)
    se = x.std(ddof=1) / np.sqrt(n); tc = stats.t.ppf(.975, n - 1)
    r = dict(label=label, n=n, d=float(x.mean()), t=float(t), p=float(p),
             ci=[float(x.mean() - tc * se), float(x.mean() + tc * se)],
             dz=float(x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 else None,
             pts=x.tolist())
    print("  %-46s n=%d d=%+.4f t=%+.3f p=%.4g CI=[%+.4f,%+.4f]" %
          (label, n, r['d'], r['t'], r['p'], r['ci'][0], r['ci'][1]))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--nrand", type=int, default=NRAND)
    ap.add_argument("--priors", default="clm,physchem")
    ap.add_argument("--smoke", action="store_true")
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

    cells = []
    hold_list = eids[:2] if args.smoke else eids
    nseed = 1 if args.smoke else args.seeds

    for e_hold in hold_list:
        mid_h = EID2MID[e_hold]
        others = [e for e in sorted(ent) if e != e_hold]
        order = ([ki[e] for e in others] + [ki[f"M_{m}"] for m in mids]
                 + [ki[e_hold]])
        NEW = len(others) + len(mids)
        iT = NEW
        for seed in range(nseed):
            rng = np.random.default_rng(seed * 7919 + 13)
            Xe = np.asarray(ent[e_hold], float)
            te_ix, pool_ix = split_by_group(Xe, GR[e_hold]["gid"], rng)
            Xpos = Xe[te_ix]
            Xn_all = np.asarray(bg[mid_h], float)
            pp = rng.permutation(len(Xn_all))
            Xneg = Xn_all[pp[:max(5, int(round(0.3 * len(Xn_all))))]]

            parts, ys, mid_ids = [], [], {}
            for j, e in enumerate(others):
                Xo = np.asarray(ent[e], float)
                if e in GR:
                    a, b, _ = split_group_w(Xo, GR[e]["gid"], rng)
                else:
                    p1 = rng.permutation(len(Xo))
                    cut = max(2, int(round(OLD_FRAC * len(Xo))))
                    a, b = p1[:cut], p1[cut:]
                parts.append(Xo[a]); ys.append(np.full(len(a), j))
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

            dZ = np.zeros_like(Zc)
            for ri, r in enumerate(rows):
                if r < len(others):
                    e = others[r]; m = EID2MID.get(e)
                    dZ[ri] = Zc[ri] - Zmid[m] if m in Zmid else Zc[ri] - mu_mid
                elif r < NEW:
                    dZ[ri] = 0.0
                else:
                    dZ[ri] = Zc[ri] - Zmid[mid_h]

            def wvec(target, Sx):
                Sx_o = Sx[rows]
                mu = Sx_o.mean(0, keepdims=True)
                W = (Sx - mu) @ head_ridge_R(Sx_o, target, LAM)
                return unit(W[iT])

            def auc_z(dvec):
                return float(auc_of((Zpos - mu_z) @ dvec,
                                    (Zneg - mu_z) @ dvec))

            # 随机码本：draw 0 复现 exp92b，其余独立
            ei = eids.index(e_hold)
            rands = []
            for m in range(args.nrand):
                if m == 0:
                    rr = np.random.default_rng(seed * 977 + 31)
                else:
                    rr = np.random.default_rng(770000 + seed * 10000 + ei * 100 + m)
                R = rr.normal(size=(len(order), dz))
                R /= (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
                rands.append(R)

            for p in pri_list:
                Sp = S_pri[p][order]
                out = dict(entity=e_hold, seed=seed, prior=p)
                for tname, target in (("Z", Zc), ("dZ", dZ)):
                    out[f"auc_struct_{tname}"] = auc_z(wvec(target, Sp))
                    out[f"auc_rand_{tname}"] = [auc_z(wvec(target, R)) for R in rands]
                cells.append(out)
            print("  [%s s%d] struct Z=%.3f dZ=%.3f | rand0 Z=%.3f dZ=%.3f" %
                  (e_hold, seed, out['auc_struct_Z'], out['auc_struct_dZ'],
                   out['auc_rand_Z'][0], out['auc_rand_dZ'][0]), flush=True)
            del net
            torch.cuda.empty_cache()

    jp = os.path.join(OUT, "exp100_detection_matched%s.json" % ("_smoke" if args.smoke else ""))
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(dict(META=dict(seeds=nseed, nrand=args.nrand, lam=LAM,
                                 priors=pri_list, eids=hold_list,
                                 note="draw0 RNG matches exp92b for cross-check"),
                       cells=cells), f, ensure_ascii=False)
    print("[exp100] wrote", jp)

    # ---------------------------------------------------------------- 分析
    from collections import defaultdict
    report = {}
    for p in pri_list:
        print("\n=== prior =", p, "===")
        for tname in ("Z", "dZ"):
            s_acc, r_acc = defaultdict(list), defaultdict(list)
            for c in cells:
                if c["prior"] != p:
                    continue
                s_acc[c["entity"]].append(c[f"auc_struct_{tname}"])
                r_acc[c["entity"]].append(float(np.mean(c[f"auc_rand_{tname}"])))
            S = {e: float(np.mean(v)) for e, v in s_acc.items()}
            R = {e: float(np.mean(v)) for e, v in r_acc.items()}
            print("  target=%-3s struct=%.4f rand(MC %d)=%.4f" %
                  (tname, np.mean(list(S.values())), args.nrand, np.mean(list(R.values()))))
            report[f"{p}_{tname}"] = paired_report(
                S, R, f"{p}: struct - MC-rand, target={tname}")
            report[f"{p}_{tname}_means"] = dict(
                struct=float(np.mean(list(S.values()))),
                rand=float(np.mean(list(R.values()))))
        # readout gain
        gz, gr = defaultdict(list), defaultdict(list)
        for c in cells:
            if c["prior"] != p:
                continue
            gz[c["entity"]].append(c["auc_struct_dZ"] - c["auc_struct_Z"])
            gr[c["entity"]].append(np.mean(c["auc_rand_dZ"]) - np.mean(c["auc_rand_Z"]))
        gz_m = {e: float(np.mean(v)) for e, v in gz.items()}
        gr_m = {e: float(np.mean(v)) for e, v in gr.items()}
        print("  readout gain dZ-Z (struct) = %+.4f | (MC rand) = %+.4f" %
              (np.mean(list(gz_m.values())), np.mean(list(gr_m.values()))))
        report[f"{p}_readout_gain"] = dict(
            struct=float(np.mean(list(gz_m.values()))),
            rand=float(np.mean(list(gr_m.values()))))

    rp = os.path.join(OUT, "exp100_detection_matched%s_report.json" % ("_smoke" if args.smoke else ""))
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("[exp100] wrote", rp)


if __name__ == "__main__":
    main()
