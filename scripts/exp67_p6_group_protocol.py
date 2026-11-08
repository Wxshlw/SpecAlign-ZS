"""exp67 —— 按样品分组的协议（group-aware）：泄漏有多大？

背景（exp66 发现）：KAN 三件套（D1/D2/D3）与部分数据集是**重复扫描**：
  kan_oil  352 条 / 唯一 id 280，最近邻余弦中位 1.000，>0.9999 占 100%
  kan_coffee 405 / 394，93.3%
  kan_juice  413 / 59，99.5%（样品名 `11_1..11_5` = 样品 11 扫 5 次）
随机划分必然把同一样品的重复扫描分到 train/test 两侧 ⇒ 记忆而非泛化。

三种协议对比：
  P1 random      现状（泄漏）
  P2 group-split 同一样品 id 只能出现在 train 或 test 一侧
  P3 dedup       每个 id 合并为一条均值谱，再随机划分

量化「泄漏量」= P1 指标 − P2 指标。
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "论文8"))
import p6_g1_data as G1
import p6_data as P
from exp63_p6_screening_protocol import (stage1, embed, head_ridge_R, nrm,
                                         auc_of, tpr_at_fpr, pauc_std,
                                         macro_recall, EID2MID)
from exp66_p6_protocol_audit import dup_groups

OUT = os.path.join(os.path.dirname(HERE), "out")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OLD_FRAC = 0.7
SID2EID = {"D7w": "A01", "D7p": "A02", "D7d": "A03", "D8": "A04",
           "D1": "A05", "D2": "A06", "D3": "A07"}


def load_with_groups():
    """返回 {eid: dict(X=预处理后谱, y=浓度, gid=样品组号, raw_gid=原始 id)}"""
    sets = P.load_all(verbose=False)
    grid = G1._grid()
    out = {}
    for sid, D in sets.items():
        if sid not in SID2EID:
            continue
        Xg = G1._resample(np.asarray(D["X"], float),
                          np.asarray(D["axis"], float), grid)
        Xp = G1._prep(Xg)
        y = np.asarray(D["y"], float)
        eid = SID2EID[sid]
        g = D.get("groups", None)
        if g is None:
            g = np.arange(len(Xp))
        g = np.asarray(g, dtype=object)
        m = y > 0
        # 组号：用原始 id；缺失时用相似度聚类兜底
        ug = {v: i for i, v in enumerate(sorted(set(g[m].tolist())))}
        out[eid] = dict(X=Xp[m], y=y[m],
                        gid=np.array([ug[v] for v in g[m]]),
                        n_gid=len(ug))
    return out, sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.0001,0.001,0.01,0.1,1")
    ap.add_argument("--eids", default="")
    ap.add_argument("--shots", default="0,5")
    ap.add_argument("--out", default="exp67_group.json")
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(",")]
    shots = [None if s == "full" else int(s) for s in args.shots.split(",")]

    ent, conc, bg, _ = G1.load_g1_conc()
    G, _sets = load_with_groups()
    eids = sorted(set(G) & set(EID2MID))
    if args.eids:
        eids = [e for e in eids if e in args.eids.split(",")]
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]
    zc = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    T_all = np.asarray(zc["clm"], float)
    ki = {k: i for i, k in enumerate(keys_all)}
    dim = T_all.shape[1]

    res = {"meta": {}, "M": {}, "leak": {}}
    print(f"[exp67] device={DEV} 实体={eids} shots={shots} seeds={args.seeds}",
          flush=True)
    for e in eids:
        d = G[e]
        # 相似度兜底：同 id 的谱应当互近；报出来做交叉验证
        Xn = nrm(d["X"])
        S = Xn @ Xn.T
        same = []
        for gi in np.unique(d["gid"])[:40]:
            ix = np.where(d["gid"] == gi)[0]
            if len(ix) > 1:
                sub = S[np.ix_(ix, ix)]
                np.fill_diagonal(sub, np.nan)
                same.append(float(np.nanmedian(sub)))
        res["leak"][e] = dict(
            n=int(len(d["X"])), n_gid=int(d["n_gid"]),
            reps_per_sample=round(len(d["X"]) / max(d["n_gid"], 1), 2),
            same_gid_cos_med=round(float(np.median(same)), 4) if same else None,
            dup_by_sim=int(dup_groups(d["X"])[1]),
            dup_factor_sim=round(len(d["X"]) / max(dup_groups(d["X"])[1], 1), 2))
        print(f"[{e:>4}] n={len(d['X']):>4} 样品数={d['n_gid']:>4} "
              f"重复×{res['leak'][e]['reps_per_sample']} "
              f"同组cos={res['leak'][e]['same_gid_cos_med']} "
              f"相似聚类组={res['leak'][e]['dup_by_sim']}", flush=True)

    for e_hold in eids:
        others = [e for e in sorted(ent) if e != e_hold]
        Xe = np.asarray(ent[e_hold], float)
        mid_h = EID2MID[e_hold]
        gid = G[e_hold]["gid"]
        # 老类 / 基质的分组信息（有则用，无则用相似度聚类）
        gid_old = {e: (G[e]["gid"] if e in G else dup_groups(ent[e])[0])
                   for e in others}
        gid_mid = {m: dup_groups(bg[m])[0] for m in mids}

        S_all = T_all[[ki[e] for e in others]
                      + [ki[f"M_{m}"] for m in mids] + [ki[e_hold]]]
        NEW = len(others) + len(mids)

        for proto in ("random", "group", "dedup"):
            for k in shots:
                tag = "full" if k is None else str(k)
                for seed in range(args.seeds):
                    rng = np.random.default_rng(seed * 7919 + 13)
                    # ---- 划分 T 的 train/test ----
                    if proto == "random":
                        perm = rng.permutation(len(Xe))
                        n_te = int(min(max(5, round(0.3 * len(Xe))),
                                       len(Xe) - 1))
                        te_ix, pool_ix = perm[:n_te], perm[n_te:]
                        Xte_new, Xtr_e = Xe[te_ix], None
                    elif proto == "group":
                        ug = np.unique(gid)
                        ug_p = rng.permutation(ug)
                        n_te_g = max(1, int(round(0.3 * len(ug))))
                        te_g = set(ug_p[:n_te_g].tolist())
                        te_ix = np.where(np.isin(gid, list(te_g)))[0]
                        pool_ix = np.where(~np.isin(gid, list(te_g)))[0]
                        Xte_new = Xe[te_ix]
                    else:  # dedup
                        Xd, yd = [], []
                        for gi in np.unique(gid):
                            ix = np.where(gid == gi)[0]
                            Xd.append(Xe[ix].mean(0))
                        Xd = np.asarray(Xd, float)
                        pp = rng.permutation(len(Xd))
                        n_te = max(2, int(round(0.3 * len(Xd))))
                        Xte_new = Xd[pp[:n_te]]
                        pool_ix = pp[n_te:]

                    tr_e_ix = pool_ix[:min(k, len(pool_ix))] if k else []

                    # ---- 训练集（老类 + 基质 + T 的 k-shot）----
                    parts, ys, refs = [], [], []
                    mid_ref = {}
                    for j, e in enumerate(others):
                        Xo = np.asarray(ent[e], float)
                        g_o = gid_old[e]
                        pp = rng.permutation(len(Xo))
                        ntr = max(2, int(round(OLD_FRAC * len(Xo))))
                        parts.append(Xo[pp[:ntr]]); ys.append(np.full(ntr, j))
                        refs.append(Xo[pp[ntr:]])
                    for j, m in enumerate(mids):
                        Xo = np.asarray(bg[m], float)
                        pp = rng.permutation(len(Xo))
                        ntr = max(2, int(round(OLD_FRAC * len(Xo))))
                        parts.append(Xo[pp[:ntr]])
                        ys.append(np.full(ntr, len(others) + j))
                        refs.append(Xo[pp[ntr:]]); mid_ref[m] = Xo[pp[ntr:]]
                    nrows = NEW
                    if len(tr_e_ix):
                        parts.append(Xe[tr_e_ix])
                        ys.append(np.full(len(tr_e_ix), NEW)); nrows = NEW + 1
                    Xtr = np.concatenate(parts).astype(np.float32)
                    ytr = np.concatenate(ys).astype(np.int64)
                    Xte_old = np.concatenate(refs).astype(np.float32)

                    for n_name, Xn_neg in (("same-matrix", mid_ref[mid_h]),
                                           ("old-heldout", Xte_old)):
                        if len(Xn_neg) < 5:
                            continue
                        sc = StandardScaler().fit(Xtr)
                        Xtr_s, Xn_s, Xo_s = (sc.transform(Xtr),
                                             sc.transform(Xte_new),
                                             sc.transform(Xn_neg))
                        rf = RandomForestClassifier(300, random_state=0,
                                                    n_jobs=-1).fit(Xtr_s, ytr)
                        sv = SVC(kernel="rbf", C=10, gamma="scale",
                                 probability=True,
                                 random_state=0).fit(Xtr_s, ytr)
                        if len(tr_e_ix):
                            i_rf = list(rf.classes_).index(NEW)
                            i_sv = list(sv.classes_).index(NEW)
                            t = (("rf", rf.predict_proba(Xn_s)[:, i_rf],
                                  rf.predict_proba(Xo_s)[:, i_rf]),
                                 ("svm", sv.predict_proba(Xn_s)[:, i_sv],
                                  sv.predict_proba(Xo_s)[:, i_sv]))
                        else:
                            t = (("rf", 1 - rf.predict_proba(Xn_s).max(1),
                                  1 - rf.predict_proba(Xo_s).max(1)),
                                 ("svm", 1 - sv.predict_proba(Xn_s).max(1),
                                  1 - sv.predict_proba(Xo_s).max(1)))
                        for nm, sn, so in t:
                            res["M"].setdefault(proto, {}).setdefault(
                                tag, {}).setdefault(n_name, {}).setdefault(
                                nm, {}).setdefault(e_hold, []).append(
                                dict(auc=round(auc_of(sn, so), 4),
                                     tpr5=round(tpr_at_fpr(sn, so, 0.05), 4)))

                    # ---- 本文 ----
                    rng_v = np.random.default_rng(seed * 31 + 7)
                    val_mask = np.zeros(len(Xtr), bool)
                    for j in sorted({int(y) for y in ytr}):
                        ix = rng_v.permutation(np.where(ytr == j)[0])
                        nv = min(max(3, int(0.2 * len(ix))),
                                 max(0, len(ix) - 1))
                        val_mask[ix[:nv]] = True
                    Xfit, yfit = Xtr[~val_mask], ytr[~val_mask]
                    Xval, yval = Xtr[val_mask], ytr[val_mask]
                    net, _ = stage1(Xfit, yfit, dim, nrows, args.steps,
                                    seed * 1000 + eids.index(e_hold),
                                    S=S_all)
                    Zfit = embed(net, Xfit)
                    rows = sorted({int(y) for y in yfit})
                    Zc = np.stack([Zfit[yfit == j].mean(0) for j in rows])
                    S_old, mu_S = S_all[rows], S_all[rows].mean(0, keepdims=True)
                    lam_b, bv = lams[0], -1.0
                    for lam in lams:
                        Wc = (S_all - mu_S) @ head_ridge_R(S_old, Zc, lam)
                        v = macro_recall(net, Wc, Xval, yval, rows)
                        if v > bv:
                            bv, lam_b = v, lam
                    W = (S_all - mu_S) @ head_ridge_R(S_old, Zc, lam_b)
                    Zn = embed(net, Xte_new)
                    heads = {"zs": W[NEW]}
                    if len(tr_e_ix):
                        heads["proto"] = embed(net, Xe[tr_e_ix]).mean(0)
                    for n_name, Xn_neg in (("same-matrix", mid_ref[mid_h]),
                                           ("old-heldout", Xte_old)):
                        if len(Xn_neg) < 5:
                            continue
                        Zo = embed(net, Xn_neg)
                        for hn, w in heads.items():
                            res["M"].setdefault(proto, {}).setdefault(
                                tag, {}).setdefault(n_name, {}).setdefault(
                                hn, {}).setdefault(e_hold, []).append(
                                dict(auc=round(auc_of(Zn @ w, Zo @ w), 4),
                                     tpr5=round(tpr_at_fpr(Zn @ w, Zo @ w,
                                                           0.05), 4)))
        print(f"[{e_hold:>4}] done", flush=True)

    with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[exp67] 已写出 out/{args.out}")
    write_report(res, eids, args.out)


def write_report(res, eids, out_name):
    from scipy.stats import ttest_rel
    L = []
    w = L.append
    w("# exp67 —— 按样品分组的协议：随机划分泄漏了多少？\n")

    w("\n## 一、各数据集的重复结构\n")
    w("| 实体 | 谱数 | 唯一样品 id | 每样品重复 | 同 id 谱余弦中位 | 相似聚类组数 |")
    w("|---|---|---|---|---|---|")
    for e in eids:
        d = res["leak"].get(e, {})
        w(f"| {e} | {d.get('n')} | {d.get('n_gid')} | "
          f"×{d.get('reps_per_sample')} | {d.get('same_gid_cos_med')} | "
          f"{d.get('dup_by_sim')} |")

    def agg(proto, tag, n_name, arm, met):
        per = {}
        for e in eids:
            vs = [x[met] for x in res["M"].get(proto, {}).get(tag, {})
                  .get(n_name, {}).get(arm, {}).get(e, [])]
            if vs:
                per[e] = float(np.mean(vs))
        return per

    w("\n## 二、三种划分协议 × 各臂（auc，macro）\n")
    for n_name in ("same-matrix", "old-heldout"):
        for tag in ("0", "5"):
            w(f"\n### `{n_name}` k={tag}\n")
            w("| 臂 | random | group-split | dedup | 泄漏量(random−group) |")
            w("|---|---|---|---|---|")
            for arm in ("rf", "svm", "zs", "proto"):
                if arm == "proto" and tag == "0":
                    continue
                vals = {}
                for p in ("random", "group", "dedup"):
                    d = agg(p, tag, n_name, arm, "auc")
                    vals[p] = d
                if not vals["random"]:
                    continue
                com = sorted(set(vals["random"]) & set(vals["group"]))
                leak = (np.mean([vals["random"][e] - vals["group"][e]
                                 for e in com]) if com else float("nan"))
                w(f"| {arm} | {np.mean(list(vals['random'].values())):.3f} | "
                  f"{np.mean(list(vals['group'].values())):.3f} | "
                  f"{np.mean(list(vals['dedup'].values())):.3f} | "
                  f"{leak:+.3f} |")

    w("\n## 三、判据\n")
    for n_name in ("same-matrix", "old-heldout"):
        for tag in ("0", "5"):
            for arm in ("rf", "svm"):
                a = agg("random", tag, n_name, arm, "auc")
                b = agg("group", tag, n_name, arm, "auc")
                com = sorted(set(a) & set(b))
                if len(com) < 3:
                    continue
                x = np.array([a[e] for e in com]); y = np.array([b[e] for e in com])
                w(f"- **{arm}** `{n_name}` k={tag}: random {x.mean():.3f} → "
                  f"group {y.mean():.3f} (Δ={np.mean(y-x):+.3f}, "
                  f"p={ttest_rel(y,x).pvalue:.4g}, n={len(com)}) ⇒ "
                  f"{'存在重复扫描泄漏' if np.mean(x-y) > 0.05 else '无明显泄漏'}")
    # 分组协议下本文 vs RF
    for n_name in ("same-matrix", "old-heldout"):
        for tag in ("0", "5"):
            z = agg("group", tag, n_name, "zs", "auc")
            r = agg("group", tag, n_name, "rf", "auc")
            com = sorted(set(z) & set(r))
            if len(com) < 3:
                continue
            xz = np.array([z[e] for e in com]); xr = np.array([r[e] for e in com])
            w(f"- **group 协议下** 本文 zs {xz.mean():.3f} vs rf {xr.mean():.3f} "
              f"`{n_name}` k={tag} (Δ={np.mean(xz-xr):+.3f}, "
              f"p={ttest_rel(xz,xr).pvalue:.4g}, n={len(com)})")

    w("\n## 四、分实体明细（group 协议, same-matrix, k=5, auc）\n")
    w("| 实体 | 样品数 | 重复× | rf | svm | 本文 zs | 本文 proto |")
    w("|---|---|---|---|---|---|---|")
    for e in eids:
        row = []
        for arm in ("rf", "svm", "zs", "proto"):
            d = agg("group", "5", "same-matrix", arm, "auc")
            row.append(f"{d[e]:.3f}" if e in d else "—")
        w(f"| {e} | {res['leak'][e]['n_gid']} | "
          f"×{res['leak'][e]['reps_per_sample']} | " + " | ".join(row) + " |")

    with open(os.path.join(OUT, "exp67_p6_group_protocol.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
