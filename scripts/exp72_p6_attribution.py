"""exp72 —— 分离「负作用」与「零作用」：化学先验在检出上到底贡献了什么

exp71 已证：cos(w_T, 真实净信号 b*) 只有 +0.44，而把 ridge 回归目标换成净信号
ΔZ 后升到 +0.67 ⇒ 方向污染成立（M1）。但仍未回答用户的问题：
    化学先验在检出上是「有害」还是「没用」？

本实验用三组对照把二者分开：
  (1) rand 码本对照：只把 head 的 S 换成同形随机矩阵（stage1 完全不变）。
      若 chem ≈ rand ⇒ 化学先验在检出上是**零作用**，落后源于线性头容量；
      若 chem < rand ⇒ 才是真的**负作用**。
  (2) 数据驱动检出头：老类掺混谱(+) vs 纯基质谱(−) 学一个线性方向 d
      （逻辑回归 / LDA），**完全不用化学先验**。检出本质是「相对已知基质的
      偏离」，不是「投影到某个分子方向」。
  (3) 匹配滤波 / 马氏白化：用类内协方差 Σ 白化，压制基质样品间变异。
      s = w_T^netᵀ Σ⁻¹ (z − μ_基质)      ← 化学方向 + 数据白化（定向）
      s = −(z − μ_基质)ᵀ Σ⁻¹ (z − μ_基质) ← 纯数据（非定向）
协议：按样品 id 分组（exp67 定稿）。
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "论文8"))
import p6_g1_data as G1
from exp63_p6_screening_protocol import (stage1, embed, head_ridge_R, nrm,
                                         auc_of, tpr_at_fpr, macro_recall,
                                         EID2MID)
from exp68_p6_cond_scorer import split_by_group
from exp71_p6_detection_fix import load_groups_cached, cos, pick_lam

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
    ap.add_argument("--npc", type=int, default=8)
    ap.add_argument("--shri", type=float, default=0.2)   # 协方差收缩
    ap.add_argument("--out", default="exp72_attr.json")
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(",")]
    shots = [None if s == "full" else int(s) for s in args.shots.split(",")]

    ent, conc, bg, _ = G1.load_g1_conc()
    GR = load_groups_cached()
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
    print(f"[exp72] device={DEV} 实体={eids} shots={shots} seeds={args.seeds}",
          flush=True)

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

                old_tr, old_ref, mid_tr, mid_ref = {}, {}, {}, {}
                parts, ys = [], []
                for j, e in enumerate(others):
                    Xo = np.asarray(ent[e], float)
                    pp = rng.permutation(len(Xo))
                    ntr = max(2, int(round(OLD_FRAC * len(Xo))))
                    old_tr[e] = Xo[pp[:ntr]]; old_ref[e] = Xo[pp[ntr:]]
                    parts.append(Xo[pp[:ntr]]); ys.append(np.full(ntr, j))
                for j, m in enumerate(mids):
                    Xo = np.asarray(bg[m], float)
                    pp = rng.permutation(len(Xo))
                    ntr = max(2, int(round(OLD_FRAC * len(Xo))))
                    parts.append(Xo[pp[:ntr]])
                    ys.append(np.full(ntr, len(others) + j))
                    mid_tr[m] = Xo[pp[:ntr]]; mid_ref[m] = Xo[pp[ntr:]]
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

                Zmid_c = {}
                mid_stack = []
                for m in mids:
                    Zm = embed(net, np.asarray(mid_tr[m], float))
                    Zmid_c[m] = Zm.mean(0)
                    mid_stack.append(Zm - Zm.mean(0))
                mu_mid = np.mean([Zmid_c[m] for m in mids], 0)

                # ---------- 头：zs（类中心目标）与 net（净信号目标）----------
                dZ = np.zeros_like(Zc)
                for ri, r in enumerate(rows):
                    e_name = others[r] if r < len(others) else None
                    if r == NEW and len(tr_e_ix):
                        dZ[ri] = Zc[ri] - Zmid_c[mid_h]
                    elif r < len(others) and e_name in EID2MID:
                        dZ[ri] = Zc[ri] - Zmid_c[EID2MID[e_name]]
                    elif r < NEW:
                        dZ[ri] = 0.0
                    else:
                        dZ[ri] = Zc[ri] - mu_mid
                lam1 = pick_lam(S_o, Zc, lams, net, Xval, yval, rows,
                                mu_S, S_all)
                W = (S_all - mu_S) @ head_ridge_R(S_o, Zc, lam1)
                Wn = (S_all - mu_S) @ head_ridge_R(S_o, dZ, lam1)
                # ---------- rand 码本（只换 head 的码本，stage1 不变）----------
                rng_r = np.random.default_rng(seed * 977 + 5)
                S_rnd = rng_r.normal(size=S_all.shape).astype(np.float32)
                S_rnd = S_rnd / (np.linalg.norm(S_rnd, axis=1,
                                                keepdims=True) + 1e-9)
                S_ro = S_rnd[rows]
                mu_R = S_ro.mean(0, keepdims=True)
                Wr = (S_rnd - mu_R) @ head_ridge_R(S_ro, Zc, lam1)
                Wnr = (S_rnd - mu_R) @ head_ridge_R(S_ro, dZ, lam1)

                def cand(Wx):
                    return np.concatenate([Wx[:len(others)],
                                           Wx[NEW:NEW + 1]], 0)

                W_c, Wn_c = cand(W), cand(Wn)
                Wr_c, Wnr_c = cand(Wr), cand(Wnr)

                # ---------- 数据驱动检出头（无化学先验）----------
                Zold_blk, yb = [], []
                for j in rows:
                    if j < len(others):          # 老类掺混谱 = 正（有掺）
                        Zold_blk.append(Zfit[yfit == j])
                        yb.append(np.ones((yfit == j).sum()))
                    elif j < NEW:                # 纯基质谱 = 负（没掺）
                        Zold_blk.append(Zfit[yfit == j])
                        yb.append(np.zeros((yfit == j).sum()))
                Zb = np.concatenate(Zold_blk, 0)
                ybb = np.concatenate(yb)
                lr = LogisticRegression(max_iter=2000, C=1.0).fit(Zb, ybb)
                d_common = lr.coef_.ravel()
                d_common = d_common / (np.linalg.norm(d_common) + 1e-12)

                # ---------- 类内协方差白化 ----------
                R = np.concatenate([Zfit[yfit == j] - Zfit[yfit == j].mean(0)
                                    for j in rows], 0)
                Sw = R.T @ R / max(1, len(R))
                tr = float(np.trace(Sw)) / Sw.shape[0]
                Sig = (1 - args.shri) * Sw + args.shri * tr * np.eye(Sw.shape[0])
                Sig = Sig + 1e-6 * np.eye(Sig.shape[0])
                Sig_inv_wT = np.linalg.solve(Sig, Wn[NEW])
                Sig_inv_d = np.linalg.solve(Sig, d_common)
                try:
                    L = np.linalg.cholesky(Sig)
                    solve_maha = lambda Z: np.linalg.solve(
                        L, (np.asarray(Z, float) - Zmid_c[mid_h]).T)
                except np.linalg.LinAlgError:
                    L = None

                Zn = embed(net, Xte_new)
                sc = StandardScaler().fit(Xtr)
                Xtr_s = sc.transform(Xtr)
                Xn_s = sc.transform(Xte_new)
                rf = RandomForestClassifier(300, random_state=0,
                                            n_jobs=-1).fit(Xtr_s, ytr)
                sv = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                         random_state=0).fit(Xtr_s, ytr)

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
                        Pn = rf.predict_proba(Xn_s); Po = rf.predict_proba(Xo_s)
                        Qn = sv.predict_proba(Xn_s); Qo = sv.predict_proba(Xo_s)
                        arms["rf"] = (Pn[:, i_rf], Po[:, i_rf])
                        arms["rf-m"] = (Pn[:, i_rf] - np.delete(Pn, i_rf, 1).max(1),
                                        Po[:, i_rf] - np.delete(Po, i_rf, 1).max(1))
                        arms["svm"] = (Qn[:, i_sv], Qo[:, i_sv])
                    else:
                        Pn = rf.predict_proba(Xn_s); Po = rf.predict_proba(Xo_s)
                        Qn = sv.predict_proba(Xn_s); Qo = sv.predict_proba(Xo_s)
                        arms["rf"] = (1 - Pn.max(1), 1 - Po.max(1))
                        arms["svm"] = (1 - Qn.max(1), 1 - Qo.max(1))

                    for nm, Wc, Wr2 in (("zs", W_c, Wr_c),
                                        ("net", Wn_c, Wnr_c)):
                        Ln, Lo = Zn @ Wc.T, Zo @ Wc.T
                        arms[nm] = (Ln[:, iT], Lo[:, iT])
                        arms[nm + "-m"] = (
                            Ln[:, iT] - np.delete(Ln, iT, 1).max(1),
                            Lo[:, iT] - np.delete(Lo, iT, 1).max(1))
                        Rn, Ro = Zn @ Wr2.T, Zo @ Wr2.T
                        arms[nm + "-rand"] = (Rn[:, iT], Ro[:, iT])
                        arms[nm + "-rand-m"] = (
                            Rn[:, iT] - np.delete(Rn, iT, 1).max(1),
                            Ro[:, iT] - np.delete(Ro, iT, 1).max(1))

                    # 数据驱动检出头（无化学）
                    arms["common"] = (Zn @ d_common, Zo @ d_common)
                    # 匹配滤波（化学方向 + 白化），只对 same-matrix 有意义
                    if n_name == "same-matrix":
                        arms["mf-net"] = ((Zn - mu_mid) @ Sig_inv_wT,
                                          (Zo - mu_mid) @ Sig_inv_wT)
                        arms["mf-common"] = ((Zn - mu_mid) @ Sig_inv_d,
                                             (Zo - mu_mid) @ Sig_inv_d)
                        if L is not None:
                            tn = solve_maha(Zn); to = solve_maha(Zo)
                            arms["maha"] = (-np.sum(tn ** 2, 0),
                                            -np.sum(to ** 2, 0))
                        arms["common-orth"] = None
                    for nm in [x for x in list(arms) if arms[x] is None]:
                        arms.pop(nm)

                    for nm, (sn, so) in arms.items():
                        res["M"].setdefault(tag, {}).setdefault(
                            n_name, {}).setdefault(nm, {}).setdefault(
                            e_hold, []).append(
                            dict(auc=round(auc_of(sn, so), 4),
                                 tpr5=round(tpr_at_fpr(sn, so, 0.05), 4),
                                 tpr20=round(tpr_at_fpr(sn, so, 0.20), 4)))

        line = []
        for n_name in ("same-matrix", "old-heldout"):
            for arm in ("zs", "net", "net-rand", "common", "rf"):
                d = res["M"].get("0", {}).get(n_name, {}).get(arm, {}).get(
                    e_hold)
                if d:
                    line.append(f"{n_name[:4]}/{arm}="
                                f"{np.mean([x['auc'] for x in d]):.3f}")
        print(f"[{e_hold:>4}] " + " ".join(line), flush=True)

    with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[exp72] 已写出 out/{args.out}")
    write_report(res, eids)


def write_report(res, eids):
    from scipy.stats import ttest_rel
    L = []
    w = L.append
    w("# exp72 —— 化学先验在检出上：负作用还是零作用？\n")
    w("- `zs`/`net`：化学码本（ridge 目标分别为类中心 / 净信号 ΔZ）\n"
      "- `*-rand`：**只把 head 的码本换成同形随机矩阵**，stage1 与训练完全不变\n"
      "  ⇒ chem−rand 才是化学先验的真实增量（隔离变量）\n"
      "- `common`：数据驱动检出方向（老类掺混 vs 纯基质，逻辑回归），**无化学**\n"
      "- `mf-net`：匹配滤波 = w_T^netᵀ Σ⁻¹ (z−μ_基质)（化学方向 + 数据白化）\n"
      "- `maha`：纯数据马氏距离（非定向）\n")
    w("协议：按样品 id 分组。\n")

    def agg(tag, n_name, arm, met):
        per = {}
        for e in eids:
            vs = [x[met] for x in res["M"].get(tag, {}).get(n_name, {})
                  .get(arm, {}).get(e, [])]
            if vs:
                per[e] = float(np.mean(vs))
        return per

    for n_name in ("same-matrix", "old-heldout"):
        arms = (["zs", "zs-rand", "zs-m", "zs-rand-m", "net", "net-rand",
                 "net-m", "net-rand-m", "common", "mf-net", "mf-common",
                 "maha", "rf", "rf-m", "svm"]
                if n_name == "same-matrix" else
                ["zs", "zs-rand", "zs-m", "zs-rand-m", "net", "net-rand",
                 "net-m", "net-rand-m", "rf", "rf-m", "svm"])
        for met in ("auc", "tpr5"):
            w(f"\n## {n_name} · {met}\n")
            w("| k | " + " | ".join(f"`{a}`" for a in arms) + " |")
            w("|---|" + "---|" * len(arms))
            for tag in ("0", "5"):
                if not res["M"].get(tag, {}).get(n_name):
                    continue
                cells = []
                for a in arms:
                    v = agg(tag, n_name, a, met)
                    cells.append(f"{np.mean(list(v.values())):.3f}"
                                 if v else "—")
                w(f"| {tag} | " + " | ".join(cells) + " |")

    w("\n## 判据\n")
    for n_name in ("same-matrix", "old-heldout"):
        for tag in ("0", "5"):
            for met in ("auc", "tpr5"):
                for a, b in (("zs", "zs-rand"), ("net", "net-rand"),
                             ("zs-m", "zs-rand-m"), ("net-m", "net-rand-m")):
                    xa, xb = agg(tag, n_name, a, met), agg(tag, n_name, b, met)
                    com = sorted(set(xa) & set(xb))
                    if len(com) < 3:
                        continue
                    va = np.array([xa[e] for e in com])
                    vb = np.array([xb[e] for e in com])
                    w(f"- **{n_name} k={tag} {met}** {a} {va.mean():.3f} vs "
                      f"{b} {vb.mean():.3f} (Δ={np.mean(va-vb):+.3f}, "
                      f"p={ttest_rel(va, vb).pvalue:.3g}, n={len(com)})")
                r = agg(tag, n_name, "rf", met)
                if not r:
                    continue
                com = sorted(r)
                xr = np.array([r[e] for e in com])
                for a in ("zs", "net", "net-m", "common", "mf-net", "maha"):
                    v = agg(tag, n_name, a, met)
                    if len(set(v) & set(com)) < 3:
                        continue
                    xv = np.array([v[e] for e in com if e in v])
                    xc = np.array([r[e] for e in com if e in v])
                    w(f"  - vs rf({xr.mean():.3f}): {a} {xv.mean():.3f} "
                      f"(Δ={np.mean(xv-xc):+.3f}, "
                      f"p={ttest_rel(xv, xc).pvalue:.3g})")

    with open(os.path.join(OUT, "exp72_p6_attribution.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
