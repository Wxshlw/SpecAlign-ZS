"""exp66 —— 协议审计：k>0 的 0.9+ 是泛化还是记忆？

三组相互独立的检验：
  A. 近重复泄漏：k-shot 抽出的训练谱与测试谱的最大余弦（如果 ≈1.0 ⇒ 记忆）
  B. 浓度不相交协议：训练只用低浓度段、测试只用高浓度段（逼外推）
  C. 可分性上界：类内 vs 类间相似度，判断"闭集分类平凡"是否数据集自带

外加 D：同一协议下补上「混合头」（α·化学外推 + (1−α)·真实样本中心），
        检验 k>0 的落后是化学先验之过，还是 exp63 没用上样本之过。
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import p6_g1_data as G1
from exp63_p6_screening_protocol import (stage1, embed, head_ridge_R, nrm,
                                         auc_of, tpr_at_fpr, pauc_std,
                                         macro_recall, EID2MID)

OUT = os.path.join(os.path.dirname(HERE), "out")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OLD_FRAC = 0.7


def dup_groups(X, tau=0.9999):
    """按 cos>tau 归并近重复组（贪心，够快）。"""
    Xn = nrm(np.asarray(X, float))
    labels = -np.ones(len(Xn), int)
    reps, c = [], 0
    for i in range(len(Xn)):
        if labels[i] >= 0:
            continue
        # 与已有代表比较
        if reps:
            s = Xn[i] @ np.asarray(reps).T
            j = int(np.argmax(s))
            if s[j] > tau:
                labels[i] = j
                continue
        reps.append(Xn[i]); labels[i] = c; c += 1
    return labels, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.0001,0.001,0.01,0.1,1")
    ap.add_argument("--eids", default="")
    ap.add_argument("--shots", default="5")
    ap.add_argument("--split", default="0.5")
    ap.add_argument("--out", default="exp66_audit.json")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.report_only:
        with open(os.path.join(OUT, args.out), encoding="utf-8") as f:
            res = json.load(f)
        eids = sorted(res["A_dup"])
        write_report(res, eids, args.out)
        return
    lams = [float(x) for x in args.lams.split(",")]
    shots = [None if s == "full" else int(s) for s in args.shots.split(",")]

    ent, conc, bg, _grid = G1.load_g1_conc()
    eids = [e for e in sorted(ent) if conc.get(e) is not None and e in EID2MID]
    if args.eids:
        eids = [e for e in eids if e in args.eids.split(",")]
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]
    zc = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    T_all = np.asarray(zc["clm"], float)
    ki = {k: i for i, k in enumerate(keys_all)}
    dim = T_all.shape[1]
    res = {"A_dup": {}, "B_conc": {}, "C_sep": {}, "D_head": {}}
    print(f"[exp66] device={DEV} 实体={eids} shots={shots} seeds={args.seeds}",
          flush=True)

    for e_hold in eids:
        Xe = np.asarray(ent[e_hold], float)
        ce = np.asarray(conc[e_hold], float)
        others = [e for e in eids if e != e_hold]
        mid_h = EID2MID[e_hold]

        # ---------- A. 近重复结构 ----------
        lab, ng = dup_groups(Xe)
        conc_u = np.unique(ce)
        # 每个浓度水平内的独立样品数
        per_conc = {}
        for c in conc_u:
            m = ce == c
            sub = lab[m]
            per_conc[float(c)] = int(len(set(sub.tolist())))
        Xn = nrm(Xe)
        Gmat = Xn @ Xn.T
        np.fill_diagonal(Gmat, -9)
        res["A_dup"][e_hold] = dict(
            n=int(len(Xe)), n_conc=int(len(conc_u)),
            n_group=int(ng), dup_factor=round(len(Xe) / max(ng, 1), 2),
            groups_per_conc=per_conc,
            within_med=float(np.median([Gmat[i][lab == lab[i]].max()
                                        for i in range(0, len(Xe), 7)]
                                       )) if ng > 1 else 1.0)

        # ---------- C. 可分性 ----------
        # 类内（同实体跨浓度）vs 类间（与其他实体/基质）
        win = [float(np.median(Xn[ce == c] @ Xn[ce == c].T))
               for c in conc_u if (ce == c).sum() > 2]
        Xin = nrm(np.asarray(bg[mid_h], float))
        betw_matrix = float(np.median(Xn @ Xin.T))
        betw_other = []
        for e2 in others[:4]:
            Xo2 = nrm(np.asarray(ent[e2], float))
            betw_other.append(float(np.median(Xn @ Xo2.T)))
        res["C_sep"][e_hold] = dict(
            within=round(float(np.mean(win)), 4) if win else None,
            vs_matrix=round(betw_matrix, 4),
            vs_other=round(float(np.mean(betw_other)), 4) if betw_other else None)

        # ---------- B/D. 两种划分协议 × 各臂 ----------
        for proto in ("random", "conc-disjoint"):
            for k in shots:
                tag = "full" if k is None else str(k)
                for seed in range(args.seeds):
                    rng = np.random.default_rng(seed * 7919 + 13)
                    if proto == "random":
                        perm = rng.permutation(len(Xe))
                        n_te = int(min(max(5, round(0.3 * len(Xe))),
                                       len(Xe) - 1))
                        te_ix, pool_ix = perm[:n_te], perm[n_te:]
                    else:
                        # 浓度不相交：测试高浓度段，训练低浓度段
                        q1, q3 = np.quantile(ce, [0.35, 0.65])
                        hi = np.where(ce >= q3)[0]
                        lo = np.where(ce <= q1)[0]
                        if len(hi) < 5 or len(lo) < 5:
                            continue
                        te_ix = rng.permutation(hi)[:max(5, int(0.3 * len(hi)))]
                        pool_ix = lo

                    Xte_new, conc_te = Xe[te_ix], ce[te_ix]
                    tr_e_ix = pool_ix[:min(k, len(pool_ix))] if k else []

                    # 训练集（老类 + 基质 + 可能的 T k-shot）
                    parts, ys, refs = [], [], []
                    mid_ref = {}
                    for j, e in enumerate(others):
                        Xo = np.asarray(ent[e], float)
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
                    NEW = len(others) + len(mids)
                    nrows = NEW
                    if len(tr_e_ix):
                        parts.append(Xe[tr_e_ix])
                        ys.append(np.full(len(tr_e_ix), NEW)); nrows = NEW + 1
                    Xtr = np.concatenate(parts).astype(np.float32)
                    ytr = np.concatenate(ys).astype(np.int64)
                    Xte_old = np.concatenate(refs).astype(np.float32)

                    # —— A 补充：k-shot 与测试的近重复度 ——
                    if len(tr_e_ix):
                        s = nrm(Xte_new) @ nrm(Xe[tr_e_ix]).T
                        res["A_dup"].setdefault(e_hold, {}).setdefault(
                            "kshot_maxcos", []).append(
                            dict(proto=proto, k=tag,
                                 med=round(float(np.median(s.max(1))), 4),
                                 frac_gt999=round(float(
                                     (s.max(1) > 0.999).mean()), 4)))

                    negs = {"same-matrix": mid_ref[mid_h],
                            "old-heldout": Xte_old}
                    for n_name, Xn_neg in negs.items():
                        if len(Xn_neg) < 5:
                            continue
                        sc = StandardScaler().fit(Xtr)
                        Xtr_s, Xn_s, Xo_s = (sc.transform(Xtr),
                                             sc.transform(Xte_new),
                                             sc.transform(Xn_neg))
                        rf = RandomForestClassifier(n_estimators=300,
                                                    random_state=0,
                                                    n_jobs=-1).fit(Xtr_s, ytr)
                        sv = SVC(kernel="rbf", C=10, gamma="scale",
                                 probability=True, random_state=0).fit(Xtr_s, ytr)
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
                            res["B_conc"].setdefault(proto, {}).setdefault(
                                tag, {}).setdefault(n_name, {}).setdefault(
                                nm, {}).setdefault(e_hold, []).append(
                                dict(auc=round(auc_of(sn, so), 4),
                                     tpr5=round(tpr_at_fpr(sn, so, 0.05), 4),
                                     pauc10=round(pauc_std(sn, so, 0.10), 4)))

                    # ---------- 本文：化学外推 + 混合头 ----------
                    S = T_all[[ki[e] for e in others]
                              + [ki[f"M_{m}"] for m in mids] + [ki[e_hold]]]
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
                                    seed * 1000 + eids.index(e_hold), S=S)
                    Zfit = embed(net, Xfit)
                    rows = sorted({int(y) for y in yfit})
                    Zc = np.stack([Zfit[yfit == j].mean(0) for j in rows])
                    S_old, mu_S = S[rows], S[rows].mean(0, keepdims=True)
                    lam_b, bv = lams[0], -1.0
                    for lam in lams:
                        Wc = (S - mu_S) @ head_ridge_R(S_old, Zc, lam)
                        v = macro_recall(net, Wc, Xval, yval, rows)
                        if v > bv:
                            bv, lam_b = v, lam
                    W = (S - mu_S) @ head_ridge_R(S_old, Zc, lam_b)
                    Zn = embed(net, Xte_new)

                    # —— E. 同空间对照：RF/SVM 直接跑在本文嵌入 z 上 ——
                    # 目的：控制"表示"变量，只留"分类头"变量。
                    #   若 rf_z ≈ rf_raw(0.99) 而本文 zs 低 ⇒ 病在分类头
                    #   若 rf_z 也低                      ⇒ 病在 stage1 表示
                    for n_name, Xn_neg in negs.items():
                        if len(Xn_neg) < 5:
                            continue
                        Zn_neg = embed(net, Xn_neg)
                        rfz = RandomForestClassifier(
                            n_estimators=300, random_state=0,
                            n_jobs=-1).fit(Zfit, yfit)
                        svz = SVC(kernel="rbf", C=10, gamma="scale",
                                  probability=True,
                                  random_state=0).fit(Zfit, yfit)
                        cent_z = np.stack([Zfit[yfit == j].mean(0)
                                           for j in rows])
                        Dnz = nrm(Zn) @ nrm(cent_z).T
                        Doz = nrm(Zn_neg) @ nrm(cent_z).T
                        if len(tr_e_ix):
                            i_rf = list(rfz.classes_).index(NEW)
                            i_sv = list(svz.classes_).index(NEW)
                            ic = list(rows).index(NEW)
                            tr = (("rf_z", rfz.predict_proba(Zn)[:, i_rf],
                                   rfz.predict_proba(Zn_neg)[:, i_rf]),
                                  ("svm_z", svz.predict_proba(Zn)[:, i_sv],
                                   svz.predict_proba(Zn_neg)[:, i_sv]),
                                  ("cent_z", Dnz[:, ic], Doz[:, ic]))
                        else:
                            tr = (("rf_z", 1 - rfz.predict_proba(Zn).max(1),
                                   1 - rfz.predict_proba(Zn_neg).max(1)),
                                  ("svm_z", 1 - svz.predict_proba(Zn).max(1),
                                   1 - svz.predict_proba(Zn_neg).max(1)),
                                  ("cent_z", 1 - Dnz.max(1), 1 - Doz.max(1)))
                        for nm, sn, so in tr:
                            res["B_conc"].setdefault(proto, {}).setdefault(
                                tag, {}).setdefault(n_name, {}).setdefault(
                                nm, {}).setdefault(e_hold, []).append(
                                dict(auc=round(auc_of(sn, so), 4),
                                     tpr5=round(tpr_at_fpr(sn, so, 0.05), 4),
                                     pauc10=round(pauc_std(sn, so, 0.10), 4)))

                    # 混合头：α·外推 + (1−α)·真实样本中心（仅 k>0 可用）
                    heads = {"zs": W[NEW]}
                    if len(tr_e_ix):
                        z_T = embed(net, Xe[tr_e_ix]).mean(0)
                        for al in (0.0, 0.25, 0.5, 0.75, 1.0):
                            heads[f"mix{al}"] = al * W[NEW] + (1 - al) * z_T
                        heads["proto"] = z_T
                    for n_name, Xn_neg in negs.items():
                        if len(Xn_neg) < 5:
                            continue
                        Zo = embed(net, Xn_neg)
                        for hn, w in heads.items():
                            sn = Zn @ w
                            so = Zo @ w
                            res["D_head"].setdefault(proto, {}).setdefault(
                                tag, {}).setdefault(n_name, {}).setdefault(
                                hn, {}).setdefault(e_hold, []).append(
                                dict(auc=round(auc_of(sn, so), 4),
                                     tpr5=round(tpr_at_fpr(sn, so, 0.05), 4)))
        print(f"[{e_hold:>7}] groups={res['A_dup'][e_hold]['n_group']:>4}"
              f"/{res['A_dup'][e_hold]['n']} "
              f"dup×{res['A_dup'][e_hold]['dup_factor']} "
              f"within={res['C_sep'][e_hold]['within']} "
              f"vs_matrix={res['C_sep'][e_hold]['vs_matrix']}", flush=True)

    with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[exp66] 已写出 out/{args.out}")
    write_report(res, eids, args.out)


def write_report(res, eids, out_name):
    from scipy.stats import ttest_rel
    L = []
    w = L.append
    w("# exp66 —— 协议审计：k>0 的 0.9+ 是泛化还是记忆？\n")

    w("\n## A. 近重复结构（每个实体内部）\n")
    w("| 实体 | 谱数 | 浓度水平 | 独立样品组(τ=0.9999) | 重复倍数 | 每浓度组数中位 |")
    w("|---|---|---|---|---|---|")
    for e in eids:
        d = res["A_dup"].get(e, {})
        gpc = d.get("groups_per_conc", {})
        w(f"| {e} | {d.get('n','—')} | {d.get('n_conc','—')} | "
          f"{d.get('n_group','—')} | ×{d.get('dup_factor','—')} | "
          f"{int(np.median(list(gpc.values()))) if gpc else '—'} |")

    w("\n### k-shot 训练谱 vs 测试谱的近重复度\n")
    w("| 实体 | 协议 | k | 最大余弦(中位) | >0.999 占比 |")
    w("|---|---|---|---|---|")
    for e in eids:
        for d in res["A_dup"].get(e, {}).get("kshot_maxcos", []):
            w(f"| {e} | {d['proto']} | {d['k']} | {d['med']} | "
              f"{d['frac_gt999']} |")

    w("\n## C. 可分性（余弦）\n")
    w("| 实体 | 类内(跨浓度) | vs 同基质纯谱 | vs 其他掺杂物 | 可分性余量 |")
    w("|---|---|---|---|---|")
    for e in eids:
        d = res["C_sep"].get(e, {})
        mar = (d["within"] - max(d["vs_matrix"], d["vs_other"] or 0)
               if d.get("within") is not None else None)
        w(f"| {e} | {d.get('within')} | {d.get('vs_matrix')} | "
          f"{d.get('vs_other')} | {round(mar,4) if mar is not None else '—'} |")

    def agg(sect, proto, tag, n_name, arm, met):
        per = {}
        for e in eids:
            vs = [x[met] for x in sect.get(proto, {}).get(tag, {})
                  .get(n_name, {}).get(arm, {}).get(e, [])]
            if vs:
                per[e] = float(np.mean(vs))
        return per

    w("\n## B. 浓度不相交协议（训练低浓度 / 测试高浓度）\n")
    w("| 口径 | k | 协议 | rf auc | rf tpr5 | svm auc | svm tpr5 |")
    w("|---|---|---|---|---|---|---|")
    for n_name in ("same-matrix", "old-heldout"):
        for tag in ("0", "5"):
            for proto in ("random", "conc-disjoint"):
                a = agg(res["B_conc"], proto, tag, n_name, "rf", "auc")
                b = agg(res["B_conc"], proto, tag, n_name, "rf", "tpr5")
                c = agg(res["B_conc"], proto, tag, n_name, "svm", "auc")
                d = agg(res["B_conc"], proto, tag, n_name, "svm", "tpr5")
                if not a:
                    continue
                w(f"| {n_name} | {tag} | {proto} | "
                  f"{np.mean(list(a.values())):.3f} | "
                  f"{np.mean(list(b.values())):.3f} | "
                  f"{np.mean(list(c.values())):.3f} | "
                  f"{np.mean(list(d.values())):.3f} |")
    # 判据：random → conc-disjoint 掉多少
    for n_name in ("same-matrix", "old-heldout"):
        for tag in ("0", "5"):
            a = agg(res["B_conc"], "random", tag, n_name, "rf", "tpr5")
            b = agg(res["B_conc"], "conc-disjoint", tag, n_name, "rf", "tpr5")
            com = sorted(set(a) & set(b))
            if len(com) >= 3:
                x = np.array([a[e] for e in com])
                y = np.array([b[e] for e in com])
                w(f"\n- **H66a** rf tpr5 `{n_name}` k={tag}: random "
                  f"{x.mean():.3f} → conc-disjoint {y.mean():.3f} "
                  f"(Δ={np.mean(y-x):+.3f}, p={ttest_rel(y,x).pvalue:.4g}, "
                  f"n={len(com)}) ⇒ "
                  f"{'存在浓度记忆' if np.mean(y-x) < -0.1 else '非浓度记忆'}")

    w("\n## E. 同嵌入空间对照（定位：病在「表示」还是「分类头」）\n")
    w("同一 stage1 嵌入 z 上，换不同分类头。\n")
    w("| 口径 | k | 协议 | 本文 ridge(化学外推) | rf_z | svm_z | cent_z | rf(原谱空间) |")
    w("|---|---|---|---|---|---|---|---|")
    for n_name in ("same-matrix", "old-heldout"):
        for tag in ("0", "5"):
            for proto in ("random", "conc-disjoint"):
                row = []
                for arm in ("rf_z", "svm_z", "cent_z", "rf"):
                    d = agg(res["B_conc"], proto, tag, n_name, arm, "auc")
                    row.append(f"{np.mean(list(d.values())):.3f}" if d else "—")
                z = agg(res["D_head"], proto, tag, n_name, "zs", "auc")
                if not z or row[0] == "—":
                    continue
                w(f"| {n_name} | {tag} | {proto} | "
                  f"{np.mean(list(z.values())):.3f} | " + " | ".join(row) + " |")
    for n_name in ("same-matrix", "old-heldout"):
        for tag in ("0", "5"):
            for proto in ("random", "conc-disjoint"):
                z = agg(res["D_head"], proto, tag, n_name, "zs", "auc")
                r = agg(res["B_conc"], proto, tag, n_name, "rf_z", "auc")
                c = agg(res["B_conc"], proto, tag, n_name, "cent_z", "auc")
                com = sorted(set(z) & set(r) & set(c))
                if len(com) < 3:
                    continue
                xz = np.array([z[e] for e in com])
                xr = np.array([r[e] for e in com])
                xc = np.array([c[e] for e in com])
                # 判据：非线性头(rf_z) vs 线性头(zs)。cent_z 同为线性，
                # 故以 |cent_z − zs| 小 且 rf_z 显著高 ⇒ 病在「线性投影头」形式。
                lin_gap = abs(np.mean(xc - xz))
                w(f"\n- **H66c** auc `{n_name}` k={tag} `{proto}`: 线性头 zs "
                  f"{xz.mean():.3f} / cent_z {xc.mean():.3f} "
                  f"(线性内差 {np.mean(xc-xz):+.3f}) | **非线性** rf_z "
                  f"{xr.mean():.3f} (Δvs zs={np.mean(xr-xz):+.3f}, "
                  f"p={ttest_rel(xr,xz).pvalue:.4g}, n={len(com)}) ⇒ "
                  + ("**病在「线性投影头」形式**（换非线性头即追上，"
                     "化学先验与表示均非瓶颈）"
                     if np.mean(xr - xz) > 0.1 and lin_gap < np.mean(xr - xz)
                     else "非线性头无优势 ⇒ 瓶颈不在头的线性/非线性"))

    w("\n## D. 混合头（k>0：化学外推 α·w_T + (1−α)·真实样本中心）\n")
    w("| 口径 | k | 协议 | zs(α=1) | mix0.75 | mix0.5 | mix0.25 | proto(α=0) | rf |")
    w("|---|---|---|---|---|---|---|---|---|")
    for n_name in ("same-matrix", "old-heldout"):
        for tag in ("5",):
            for proto in ("random", "conc-disjoint"):
                row = []
                for arm in ("zs", "mix0.75", "mix0.5", "mix0.25", "proto"):
                    d = agg(res["D_head"], proto, tag, n_name, arm, "auc")
                    row.append(f"{np.mean(list(d.values())):.3f}" if d else "—")
                r = agg(res["B_conc"], proto, tag, n_name, "rf", "auc")
                row.append(f"{np.mean(list(r.values())):.3f}" if r else "—")
                if row[0] == "—":
                    continue
                w(f"| {n_name} | {tag} | {proto} | " + " | ".join(row) + " |")
    for n_name in ("same-matrix", "old-heldout"):
        for proto in ("random", "conc-disjoint"):
            a = agg(res["D_head"], proto, "5", n_name, "zs", "auc")
            b = agg(res["D_head"], proto, "5", n_name, "proto", "auc")
            c = agg(res["B_conc"], proto, "5", n_name, "rf", "auc")
            com = sorted(set(a) & set(b) & set(c))
            if len(com) >= 3:
                xa = np.array([a[e] for e in com])
                xb = np.array([b[e] for e in com])
                xc = np.array([c[e] for e in com])
                w(f"\n- **H66b** auc `{n_name}` `{proto}` k=5: 纯外推 "
                  f"{xa.mean():.3f} → 纯样本中心 {xb.mean():.3f} "
                  f"(Δ={np.mean(xb-xa):+.3f}, p={ttest_rel(xb,xa).pvalue:.4g}) "
                  f"| rf {xc.mean():.3f} (Δ={np.mean(xb-xc):+.3f}, "
                  f"p={ttest_rel(xb,xc).pvalue:.4g}, n={len(com)})")

    with open(os.path.join(OUT, "exp66_p6_protocol_audit.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
