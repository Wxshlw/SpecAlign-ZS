"""exp70 —— k>0 混合头 + margin：把 k-shot 真实样本真正用起来

exp63 缺陷：k=5 时 5 条真实谱只进 ridge 回归当一行样本，w_T 仍纯化学外推。
exp66 D 补测过混合头，但那是**无 margin** 版本（0.730→0.750, n.s.）。
exp69 已证 margin 在 old-heldout 上 +0.118 (p=0.013)，故这里重测：
    w_T(α) = α · w_chem（SMILES 外推） + (1−α) · mean(z of k 条真实谱)
    score   = w_T(α)·z  −  max_{j≠T} w_j·z        （margin）
α=1 纯化学 ⇒ k=0 也能用；α<1 需 k>0。

同时测两种 margin：hard（减最大竞争者）与 LSE（τ 平滑，exp62 margin-lse）。
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
from exp63_p6_screening_protocol import (stage1, embed, head_ridge_R, nrm,
                                         auc_of, tpr_at_fpr, pauc_std,
                                         macro_recall, EID2MID)
from exp68_p6_cond_scorer import load_groups, split_by_group

OUT = os.path.join(os.path.dirname(HERE), "out")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OLD_FRAC = 0.7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.0001,0.001,0.01,0.1,1")
    ap.add_argument("--eids", default="")
    ap.add_argument("--shots", default="0,5")
    ap.add_argument("--alpha", default="1,0.75,0.5,0.25,0")
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--out", default="exp70_mix.json")
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(",")]
    shots = [None if s == "full" else int(s) for s in args.shots.split(",")]
    alphas = [float(x) for x in args.alpha.split(",")]

    ent, conc, bg, _ = G1.load_g1_conc()
    GR = load_groups()
    eids = sorted(set(GR) & set(EID2MID))
    if args.eids:
        eids = [e for e in eids if e in args.eids.split(",")]
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]
    zc = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    T_all = np.asarray(zc["clm"], float)
    ki = {k: i for i, k in enumerate(keys_all)}
    dz = T_all.shape[1]
    res = {"M": {}}
    print(f"[exp70] device={DEV} 实体={eids} shots={shots} "
          f"seeds={args.seeds} alpha={alphas}", flush=True)

    for e_hold in eids:
        others = [e for e in sorted(ent) if e != e_hold]
        Xe = np.asarray(ent[e_hold], float)
        gid_e = GR[e_hold]["gid"]
        mid_h = EID2MID[e_hold]
        S_all = T_all[[ki[e] for e in others]
                      + [ki[f"M_{m}"] for m in mids] + [ki[e_hold]]]
        NEW = len(others) + len(mids)
        iT = len(others)

        for k in shots:
            tag = "full" if k is None else str(k)
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed * 7919 + 13)
                te_ix, pool_ix = split_by_group(Xe, gid_e, rng)
                Xte_new = Xe[te_ix]
                tr_e_ix = pool_ix[:min(k, len(pool_ix))] if k else []

                old_ref, mid_ref = {}, {}
                parts, ys = [], []
                for j, e in enumerate(others):
                    Xo = np.asarray(ent[e], float)
                    pp = rng.permutation(len(Xo))
                    ntr = max(2, int(round(OLD_FRAC * len(Xo))))
                    old_ref[e] = Xo[pp[ntr:]]
                    parts.append(Xo[pp[:ntr]]); ys.append(np.full(ntr, j))
                for j, m in enumerate(mids):
                    Xo = np.asarray(bg[m], float)
                    pp = rng.permutation(len(Xo))
                    ntr = max(2, int(round(OLD_FRAC * len(Xo))))
                    parts.append(Xo[pp[:ntr]])
                    ys.append(np.full(ntr, len(others) + j))
                    mid_ref[m] = Xo[pp[ntr:]]
                nrows = NEW
                if len(tr_e_ix):
                    parts.append(Xe[tr_e_ix])
                    ys.append(np.full(len(tr_e_ix), NEW)); nrows = NEW + 1
                Xtr = np.concatenate(parts).astype(np.float32)
                ytr = np.concatenate(ys).astype(np.int64)
                Xte_old = np.concatenate(list(old_ref.values())
                                         + list(mid_ref.values())).astype(
                    np.float32)

                rng_v = np.random.default_rng(seed * 31 + 7)
                val_mask = np.zeros(len(Xtr), bool)
                for j in sorted({int(y) for y in ytr}):
                    ix = rng_v.permutation(np.where(ytr == j)[0])
                    nv = min(max(3, int(0.2 * len(ix))), max(0, len(ix) - 1))
                    val_mask[ix[:nv]] = True
                Xfit, yfit = Xtr[~val_mask], ytr[~val_mask]
                Xval, yval = Xtr[val_mask], ytr[val_mask]
                net, _ = stage1(Xfit, yfit, dz, nrows, args.steps,
                                seed * 1000 + eids.index(e_hold), S=S_all)
                Zfit = embed(net, Xfit)
                rows = sorted({int(y) for y in yfit})
                Zc = np.stack([Zfit[yfit == j].mean(0) for j in rows])
                S_o, mu_S = S_all[rows], S_all[rows].mean(0, keepdims=True)
                lam_b, bv = lams[0], -1.0
                for lam in lams:
                    Wc = (S_all - mu_S) @ head_ridge_R(S_o, Zc, lam)
                    v = macro_recall(net, Wc, Xval, yval, rows)
                    if v > bv:
                        bv, lam_b = v, lam
                W = (S_all - mu_S) @ head_ridge_R(S_o, Zc, lam_b)
                # 候选权重矩阵：老类 w_j + T 的 w_T
                W_cand = np.concatenate([W[:len(others)], W[NEW:NEW + 1]], 0)

                # k>0：真实样本中心
                if len(tr_e_ix):
                    z_T = embed(net, Xe[tr_e_ix]).mean(0)
                else:
                    z_T = None

                sc = StandardScaler().fit(Xtr)
                Xtr_s = sc.transform(Xtr)
                Xn_s = sc.transform(Xte_new)
                rf = RandomForestClassifier(300, random_state=0,
                                            n_jobs=-1).fit(Xtr_s, ytr)
                sv = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                         random_state=0).fit(Xtr_s, ytr)

                Zn = embed(net, Xte_new)
                for n_name, Xn_neg in (("same-matrix", mid_ref[mid_h]),
                                       ("old-heldout", Xte_old)):
                    if len(Xn_neg) < 5:
                        continue
                    Zo = embed(net, Xn_neg)
                    Xo_s = sc.transform(Xn_neg)
                    arms = {}
                    if len(tr_e_ix):
                        i_rf = list(rf.classes_).index(NEW)
                        i_sv = list(sv.classes_).index(NEW)
                        Pn, Po = rf.predict_proba(Xn_s), rf.predict_proba(Xo_s)
                        Qn, Qo = sv.predict_proba(Xn_s), sv.predict_proba(Xo_s)
                        arms["rf"] = (Pn[:, i_rf], Po[:, i_rf])
                        arms["rf-m"] = (Pn[:, i_rf]
                                        - np.delete(Pn, i_rf, 1).max(1),
                                        Po[:, i_rf]
                                        - np.delete(Po, i_rf, 1).max(1))
                        arms["svm"] = (Qn[:, i_sv], Qo[:, i_sv])
                    else:
                        arms["rf"] = (1 - rf.predict_proba(Xn_s).max(1),
                                      1 - rf.predict_proba(Xo_s).max(1))
                        arms["svm"] = (1 - sv.predict_proba(Xn_s).max(1),
                                       1 - sv.predict_proba(Xo_s).max(1))

                    def score_with(wT):
                        Wx = W_cand.copy()
                        Wx[iT] = wT
                        Ln = Zn @ Wx.T
                        Lo = Zo @ Wx.T
                        mxn = np.delete(Ln, iT, 1).max(1)
                        mxo = np.delete(Lo, iT, 1).max(1)
                        hard = (Ln[:, iT] - mxn, Lo[:, iT] - mxo)
                        lse_n = mxn + args.tau * np.log(np.exp(
                            (np.delete(Ln, iT, 1) - mxn[:, None])
                            / args.tau).sum(1))
                        lse_o = mxo + args.tau * np.log(np.exp(
                            (np.delete(Lo, iT, 1) - mxo[:, None])
                            / args.tau).sum(1))
                        lse = (Ln[:, iT] - lse_n, Lo[:, iT] - lse_o)
                        return dict(hard=hard, lse=lse)

                    for al in alphas:
                        if al < 1.0 - 1e-9 and z_T is None:
                            continue
                        wT = W[NEW] if al >= 1.0 - 1e-9 else \
                            al * W[NEW] + (1 - al) * z_T
                        d = score_with(wT)
                        for mn, (sn, so) in d.items():
                            nm = (f"a{al}-{mn}" if al < 1.0 - 1e-9
                                  else f"chem-{mn}")
                            arms[nm] = (sn, so)
                    for nm, (sn, so) in arms.items():
                        res["M"].setdefault(tag, {}).setdefault(
                            n_name, {}).setdefault(nm, {}).setdefault(
                            e_hold, []).append(
                            dict(auc=round(auc_of(sn, so), 4),
                                 tpr5=round(tpr_at_fpr(sn, so, 0.05), 4),
                                 tpr20=round(tpr_at_fpr(sn, so, 0.20), 4)))
        line = []
        for n_name in ("old-heldout", "same-matrix"):
            for arm in ("chem-hard", "a0.5-hard", "rf"):
                d = res["M"].get(tag, {}).get(n_name, {}).get(arm, {}).get(
                    e_hold)
                if d:
                    line.append(f"{n_name[:4]}/{arm}="
                                f"{np.mean([x['auc'] for x in d]):.3f}")
        print(f"[{e_hold:>4}] " + " ".join(line), flush=True)

    with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[exp70] 已写出 out/{args.out}")
    write_report(res, eids, args.out, alphas)


def write_report(res, eids, out_name, alphas):
    from scipy.stats import ttest_rel
    L = []
    w = L.append
    w("# exp70 —— k>0 混合头 + margin\n")
    w("`w_T(α) = α·w_chem(SMILES 外推) + (1−α)·mean(z of k 条真实谱)`，"
      "再取 margin `s_T − max_{j≠T} s_j`。\n")
    w("`chem-*` = α=1（**k=0 也可用**）；`a{α}-*` = 混合。`hard`/`lse` 两种 margin。\n")
    w("协议：按样品 id 分组。\n")

    def agg(tag, n_name, arm, met):
        per = {}
        for e in eids:
            vs = [x[met] for x in res["M"].get(tag, {}).get(n_name, {})
                  .get(arm, {}).get(e, [])]
            if vs:
                per[e] = float(np.mean(vs))
        return per

    arms = ["chem-hard", "chem-lse"] + \
           [f"a{a}-{m}" for a in alphas if a < 1.0
            for m in ("hard", "lse")] + ["rf", "rf-m", "svm"]
    for met in ("auc", "tpr5"):
        w(f"\n## {met}\n")
        w("| 口径 | k | " + " | ".join(f"`{a}`" for a in arms) + " |")
        w("|---|---|" + "---|" * len(arms))
        for n_name in ("old-heldout", "same-matrix"):
            for tag in ("0", "5"):
                if not res["M"].get(tag, {}).get(n_name):
                    continue
                cells = []
                for a in arms:
                    v = agg(tag, n_name, a, met)
                    cells.append(f"{np.mean(list(v.values())):.3f}"
                                 if v else "—")
                w(f"| {n_name} | {tag} | " + " | ".join(cells) + " |")

    w("\n## 判据（配对 t）\n")
    for n_name in ("old-heldout", "same-matrix"):
        for tag in ("0", "5"):
            for met in ("auc", "tpr5"):
                ch = agg(tag, n_name, "chem-hard", met)
                cl = agg(tag, n_name, "chem-lse", met)
                r = agg(tag, n_name, "rf", met)
                if not (ch and r):
                    continue
                com = sorted(set(ch) & set(r))
                xch = np.array([ch[e] for e in com])
                xr = np.array([r[e] for e in com])
                msg = (f"- **{n_name} k={tag} {met}**: chem-hard "
                       f"{xch.mean():.3f} vs rf {xr.mean():.3f} "
                       f"(Δ={np.mean(xch-xr):+.3f}, "
                       f"p={ttest_rel(np.array([ch[e] for e in com]), xr).pvalue:.4g})")
                if cl:
                    msg += (f" | chem-lse {np.mean(list(cl.values())):.3f}")
                for a in alphas:
                    if a >= 1.0:
                        continue
                    for m in ("hard", "lse"):
                        v = agg(tag, n_name, f"a{a}-{m}", met)
                        if not v:
                            continue
                        com2 = sorted(set(v) & set(r))
                        if len(com2) < 3:
                            continue
                        xv = np.array([v[e] for e in com2])
                        xr2 = np.array([r[e] for e in com2])
                        msg += (f" | a{a}-{m} {xv.mean():.3f} "
                                f"(Δvs rf={np.mean(xv-xr2):+.3f}, "
                                f"p={ttest_rel(xv, xr2).pvalue:.3g})")
                w(msg + f" (n={len(com)})")

    with open(os.path.join(OUT, "exp70_p6_mixhead.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
