# -*- coding: utf-8 -*-
r"""exp97b —— 按独立样品组重算 k 曲线（修正 exp81b 的重复扫描凑 k）。

【问题来源】（exp97a 审计）
exp81b:408  `tr_e_ix = pool_ix[:min(k,len(pool_ix))]`
切分按 gid、抽样不按 gid。A05/A06/A07 是 50 个样品 × 5 条连续排列的重复扫描，
于是 pool[:2]..pool[:5] 的**唯一 gid 数恒为 1**，pool[:10] 只有 2，pool[:20] 只有 4。
⇒ 标称 k=5 的曲线点，实际只给了 1 个独立样品。

【本脚本做什么】
同一套 stage1 / 同一批 U / 同一 seed，并列两种抽取：
  rec  = pool[:k]            （复现 exp81b 的旧行为）
  gid  = 取 k 个**不同** gid，每个 gid 取一条（gid 内先平均后视为 1 个标样）
指标：
  top-1  = U 内 argmax 命中率（k=0 时用先验外推的中心，k>0 时用标样中心）
  auc_dis= 每对 (T,D) 的定向判别 AUC，对候选对等权后再对 combo 等权

k=0 走零样本（center = (S_c − μ_S) @ A），k>0 走标样（center = mean z_k − mean Z_old），
两条路的中心都在 z 空间，可直接比。

输出：out/exp97b_kcurve.{json,md}
"""
from __future__ import annotations
import os, sys, json, argparse, itertools, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "..", "out")

import p6_g1_data as G1
from exp63_p6_screening_protocol import (stage1, embed, head_ridge_R, auc_of, EID2MID)
from exp71_p6_detection_fix import load_groups_cached
from exp78_p6_attribution_proof import split_group_w
from exp68_p6_cond_scorer import split_by_group
from exp81b_p6_benchmark_zoo import DEV

OLD_FRAC = 0.7
KS = [0, 1, 2, 5, 10, 20]


def pick_gid(gid_pool, k):
    """取 k 个不同 gid，每个 gid 取首次出现的那条记录的索引。"""
    seen, idx = set(), []
    for i, g in enumerate(gid_pool):
        if g in seen:
            continue
        seen.add(g)
        idx.append(i)
        if len(idx) >= k:
            break
    return np.array(idx, int)


def evaluate(Zte, centers, U):
    """返回 (top1, auc_dis)。centers[u] 是 384 维中心。"""
    S = {u: Zte[u] @ centers[u] for u in U}          # 每个候选的自得分
    # top-1：对每个测试谱，比较它到各候选中心的得分
    hit, tot = 0, 0
    for u in U:
        M = np.stack([Zte[u] @ centers[c] for c in U], 1)   # (n_u, |U|)
        pred = np.argmax(M, 1)
        hit += int((pred == U.index(u)).sum())
        tot += len(pred)
    # auc_dis：每对 T,D
    aucs = []
    for T, D in itertools.combinations(U, 2):
        dT = Zte[T] @ (centers[T] - centers[D])
        dD = Zte[D] @ (centers[T] - centers[D])
        aucs.append(float(auc_of(dT, dD)))
    return (hit / tot if tot else float("nan")), (float(np.mean(aucs)) if aucs else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nhu", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--ks", default="0,1,2,5,10,20")
    ap.add_argument("--eids", default="", help="逗号分隔候选实体（空=全部）")
    ap.add_argument("--out", default="exp97b_kcurve.json")
    args = ap.parse_args()
    KS_ = [int(x) for x in args.ks.split(",")]

    ent, conc, bg, _ = G1.load_g1_conc()
    GR = load_groups_cached()
    eids = sorted(set(GR) & set(EID2MID))
    if args.eids:
        keep = [x.strip() for x in args.eids.split(",") if x.strip()]
        eids = [e for e in eids if e in keep]
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]

    zc = np.load(os.path.join(OUT, "exp81_prior_cache.npz"), allow_pickle=True)
    assert list(zc["keys"]) == keys_all, "先验缓存 key 顺序不一致，禁止对齐！"
    S_clm = np.asarray(zc["clm"], float)
    dz = S_clm.shape[1]

    combos = list(itertools.combinations(eids, args.nhu))
    recs = []
    t0 = time.time()
    for si in range(args.seed0, args.seed0 + args.seeds):
        rng0 = np.random.default_rng(si * 7919 + 13)
        cis = rng0.permutation(len(combos))[:args.reps]
        for ci in cis:
            U = list(combos[ci])
            others = [e for e in sorted(ent) if e not in U]
            rng = np.random.default_rng(si * 104729 + ci * 37 + 7)

            # ---- 划分（与 exp82 完全一致）----
            Xte_U, Xpool_U, gid_pool = {}, {}, {}
            for u in U:
                Xe = np.asarray(ent[u], float)
                te_ix, pool_ix = split_by_group(Xe, GR[u]["gid"], rng)
                Xte_U[u] = Xe[te_ix]
                Xpool_U[u] = Xe[pool_ix]
                gid_pool[u] = np.asarray(GR[u]["gid"])[pool_ix]

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
                parts.append(Xo[pp[:cut]]); ys.append(np.full(cut, len(others) + j))
            Xtr = np.concatenate(parts).astype(np.float32)
            ytr = np.concatenate(ys).astype(np.int64)
            rows = sorted({int(y) for y in ytr})

            # 与 exp82 完全一致的 stage1 口径：S 含 U 的行（只给先验、不给 U 的谱）
            order = ([sorted(ent).index(e) for e in others]
                     + [len(sorted(ent)) + mids.index(m) for m in mids]
                     + [sorted(ent).index(u) for u in U])
            S_full = S_clm[order]
            NEW0 = len(others) + len(mids)

            net, _ = stage1(Xtr, ytr, dz, len(S_full), args.steps,
                            si * 1000 + ci, S=S_full)
            Zfit = embed(net, Xtr)
            Z_old = np.stack([Zfit[ytr == j].mean(0) for j in rows])
            S_old = S_full[:NEW0]
            mu_S = S_old.mean(0, keepdims=True)
            z_mu = Z_old.mean(0)
            A = head_ridge_R(S_old, Z_old, args.lam)
            Zte = {u: embed(net, Xte_U[u]) for u in U}
            Zpool = {u: embed(net, Xpool_U[u]) for u in U}

            cand_idx = {u: sorted(ent).index(u) for u in U}
            tag = f"seed{si}_c{ci}"
            for k in KS_:
                # ---- 零样本：先验外推中心 ----
                if k == 0:
                    centers = {u: (S_clm[cand_idx[u]] - mu_S[0]) @ A for u in U}
                    t1, ad = evaluate(Zte, centers, U)
                    recs.append(dict(tag=tag, k=0, mode="prior", top1=t1,
                                     auc_dis=ad, eff_k=0, U="-".join(U)))
                    continue
                for mode in ("rec", "gid"):
                    centers, eff = {}, []
                    for u in U:
                        if mode == "rec":
                            ix = np.arange(min(k, len(Zpool[u])))
                        else:
                            ix = pick_gid(gid_pool[u], k)
                        eff.append(len(np.unique(gid_pool[u][ix])) if len(ix) else 0)
                        if len(ix) == 0:
                            centers[u] = (S_clm[cand_idx[u]] - mu_S[0]) @ A
                        else:
                            centers[u] = Zpool[u][ix].mean(0) - z_mu
                    t1, ad = evaluate(Zte, centers, U)
                    recs.append(dict(tag=tag, k=k, mode=mode, top1=t1, auc_dis=ad,
                                     eff_k=float(np.mean(eff)), U="-".join(U)))
            print(f"[{tag}] U={'-'.join(U)} {time.time()-t0:.0f}s", flush=True)

    # ---------------- 汇总：combo 等权 ----------------
    def agg(mode, k, field):
        by = {}
        for r in recs:
            if r["mode"] == mode and r["k"] == k:
                by.setdefault(r["tag"], []).append(r[field])
        return float(np.mean([np.mean(v) for v in by.values()])), len(by)

    L = []
    def w(s=""):
        L.append(s); print(s)

    w("# exp97b · 按独立样品组重算的 k 曲线")
    w()
    w(f"`|U|={args.nhu}`，{args.seeds} seeds × {args.reps} reps，lam={args.lam}，"
      f"stage1 {args.steps} steps。指标对 combo 等权。")
    w()
    w("`rec` = `pool[:k]`（exp81b 旧行为）；`gid` = k 个不同 gid、每个取一条。")
    w()
    w("| k | rec top-1 | gid top-1 | Δ | rec auc_dis | gid auc_dis | Δ | rec 实际独立样品 | gid 实际独立样品 |")
    w("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    RES = {}
    for k in KS_:
        if k == 0:
            t1, _ = agg("prior", 0, "top1")
            ad, _ = agg("prior", 0, "auc_dis")
            w(f"| 0 | {t1:.3f} | — | — | {ad:.3f} | — | — | 0 | 0 |")
            RES[0] = dict(top1=t1, auc_dis=ad)
            continue
        a1, n1 = agg("rec", k, "top1"); b1, _ = agg("gid", k, "top1")
        a2, _ = agg("rec", k, "auc_dis"); b2, _ = agg("gid", k, "auc_dis")
        ae, _ = agg("rec", k, "eff_k"); be, _ = agg("gid", k, "eff_k")
        w(f"| {k} | {a1:.3f} | {b1:.3f} | {b1-a1:+.3f} | {a2:.3f} | {b2:.3f} | {b2-a2:+.3f} "
          f"| {ae:.2f} | {be:.2f} |")
        RES[k] = dict(rec_top1=a1, gid_top1=b1, rec_auc=a2, gid_auc=b2,
                      rec_eff=ae, gid_eff=be)
    w()
    with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as f:
        json.dump(dict(res=RES, recs=recs, meta=vars(args)), f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "exp97b_kcurve.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n>>> out/exp97b_kcurve.{json,md}")


if __name__ == "__main__":
    main()
