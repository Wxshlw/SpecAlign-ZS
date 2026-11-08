"""exp73 —— SpecAlign-ZS v4：双头解耦（检出头 = 匹配滤波 / 识别头 = margin）

已证事实链：
  exp71 cos(w_T, b*) = +0.157（几乎正交）⇒ ridge 目标 Z_j 里绝大部分是基质成分
                     改目标为净信号 ΔZ 后 cos 升到 +0.314，检出 auc 0.645→0.718
  exp72 化学先验在检出上 chem−rand = +0.157~+0.228，**恒为正** ⇒ 不是负作用
         白化（Σ⁻¹）是最强单件：same-matrix k=5 mf-net 0.854（vs zs 0.670）
  exp71 net 头在识别上 −0.208 (p=0.009) ⇒ 单头通吃必然互相伤害 ⇒ **必须双头解耦**

v4 架构（两任务两个头，各自最优，互不干扰）：
  识别头 = 化学外推 + margin（old-heldout）
  检出头 = 化学外推净信号方向 + 类内协方差白化（matched filter）（same-matrix）
白化：s = (z − μ_基质)ᵀ Σ⁻¹ w_T，Σ = 类内散度 + shrinkage·I
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
from exp71_p6_detection_fix import load_groups_cached, pick_lam

OUT = os.path.join(os.path.dirname(HERE), "out")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OLD_FRAC = 0.7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.0001,0.001,0.01,0.1,1")
    ap.add_argument("--shris", default="0.2,0.6")
    ap.add_argument("--eids", default="")
    ap.add_argument("--shots", default="0,5")
    ap.add_argument("--out", default="exp73_v4.json")
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(",")]
    shris = [float(x) for x in args.shris.split(",")]
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
    print(f"[exp73] device={DEV} 实体={eids} shots={shots} seeds={args.seeds}",
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
                for m in mids:
                    Zmid_c[m] = embed(net, np.asarray(mid_tr[m],
                                                      float)).mean(0)
                mu_mid = np.mean([Zmid_c[m] for m in mids], 0)

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
                W = (S_all - mu_S) @ head_ridge_R(S_o, Zc, lam1)     # 类中心目标
                Wn = (S_all - mu_S) @ head_ridge_R(S_o, dZ, lam1)    # 净信号目标
                rng_r = np.random.default_rng(seed * 977 + 5)
                S_rnd = rng_r.normal(size=S_all.shape).astype(np.float32)
                S_rnd /= (np.linalg.norm(S_rnd, axis=1, keepdims=True) + 1e-9)
                S_ro = S_rnd[rows]; mu_R = S_ro.mean(0, keepdims=True)
                Wnr = (S_rnd - mu_R) @ head_ridge_R(S_ro, dZ, lam1)  # 随机方向

                def cand(Wx):
                    return np.concatenate([Wx[:len(others)],
                                           Wx[NEW:NEW + 1]], 0)

                # 数据驱动通用方向（无化学）
                Zb, ybb = [], []
                for j in rows:
                    if j < NEW:
                        Zb.append(Zfit[yfit == j])
                        ybb.append(np.ones((yfit == j).sum()) if j < len(others)
                                   else np.zeros((yfit == j).sum()))
                Zb = np.concatenate(Zb, 0); ybb = np.concatenate(ybb)
                lr = LogisticRegression(max_iter=2000, C=1.0).fit(Zb, ybb)
                d_com = lr.coef_.ravel()
                d_com /= (np.linalg.norm(d_com) + 1e-12)

                # 类内协方差 + shrinkage
                R = np.concatenate([Zfit[yfit == j] - Zfit[yfit == j].mean(0)
                                    for j in rows], 0)
                Sw = R.T @ R / max(1, len(R))
                tr = float(np.trace(Sw)) / Sw.shape[0]

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

                    for shri in shris:
                        Sig = (1 - shri) * Sw + shri * tr * np.eye(Sw.shape[0])
                        Sig += 1e-6 * np.eye(Sig.shape[0])
                        tag_s = f"s{int(shri*10):02d}"
                        for hn, Wx in (("zs", W), ("net", Wn), ("rand", Wnr)):
                            Wc = cand(Wx)
                            Ln, Lo = Zn @ Wc.T, Zo @ Wc.T
                            if shri == shris[0]:        # raw 只记一次
                                arms[f"{hn}"] = (Ln[:, iT], Lo[:, iT])
                                arms[f"{hn}-m"] = (
                                    Ln[:, iT] - np.delete(Ln, iT, 1).max(1),
                                    Lo[:, iT] - np.delete(Lo, iT, 1).max(1))
                            # 匹配滤波：白化后的候选分数矩阵
                            An = np.linalg.solve(Sig, Wc.T).T
                            Fn = (Zn - mu_mid) @ An.T
                            Fo = (Zo - mu_mid) @ An.T
                            arms[f"mf-{hn}-{tag_s}"] = (Fn[:, iT], Fo[:, iT])
                            arms[f"mf-{hn}-m-{tag_s}"] = (
                                Fn[:, iT] - np.delete(Fn, iT, 1).max(1),
                                Fo[:, iT] - np.delete(Fo, iT, 1).max(1))
                        dc = np.linalg.solve(Sig, d_com)
                        arms[f"mf-common-{tag_s}"] = ((Zn - mu_mid) @ dc,
                                                      (Zo - mu_mid) @ dc)

                    for nm, (sn, so) in arms.items():
                        res["M"].setdefault(tag, {}).setdefault(
                            n_name, {}).setdefault(nm, {}).setdefault(
                            e_hold, []).append(
                            dict(auc=round(auc_of(sn, so), 4),
                                 tpr5=round(tpr_at_fpr(sn, so, 0.05), 4),
                                 tpr20=round(tpr_at_fpr(sn, so, 0.20), 4)))

        line = []
        for n_name in ("same-matrix", "old-heldout"):
            for arm in ("zs-m", "mf-net-s02", "mf-net-m-s02", "rf"):
                d = res["M"].get("0", {}).get(n_name, {}).get(arm, {}).get(
                    e_hold)
                if d:
                    line.append(f"{n_name[:4]}/{arm}="
                                f"{np.mean([x['auc'] for x in d]):.3f}")
        print(f"[{e_hold:>4}] " + " ".join(line), flush=True)

    with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[exp73] 已写出 out/{args.out}")
    write_report(res, eids, shris)


def write_report(res, eids, shris):
    from scipy.stats import ttest_rel
    L = []
    w = L.append
    w("# exp73 —— SpecAlign-ZS v4：双头解耦\n")
    w("| 头 | 任务 | 公式 |\n|---|---|---|\n"
      "| 识别头 | old-heldout（掺了哪种） | `s_T − max_{j≠T} s_j`（化学外推 + margin）|\n"
      "| 检出头 | same-matrix（有没有掺） | `(z − μ_基质)ᵀ Σ⁻¹ w_T^net`（化学净信号方向 + 白化）|\n")
    w("Σ = 类内散度 + shrinkage·(tr/d)·I；`s02`/`s06` = shrinkage 0.2 / 0.6。\n")

    def agg(tag, n_name, arm, met):
        per = {}
        for e in eids:
            vs = [x[met] for x in res["M"].get(tag, {}).get(n_name, {})
                  .get(arm, {}).get(e, [])]
            if vs:
                per[e] = float(np.mean(vs))
        return per

    TS = [f"s{int(s*10):02d}" for s in shris]
    ARMS = ["zs", "zs-m", "net", "net-m", "rand", "rand-m"]
    for t in TS:
        ARMS += [f"mf-zs-{t}", f"mf-zs-m-{t}", f"mf-net-{t}", f"mf-net-m-{t}",
                 f"mf-rand-{t}", f"mf-common-{t}"]
    ARMS += ["rf", "rf-m", "svm"]

    for n_name in ("same-matrix", "old-heldout"):
        for met in ("auc", "tpr5"):
            w(f"\n## {n_name} · {met}\n")
            w("| k | " + " | ".join(f"`{a}`" for a in ARMS) + " |")
            w("|---|" + "---|" * len(ARMS))
            for tag in ("0", "5"):
                if not res["M"].get(tag, {}).get(n_name):
                    continue
                cells = []
                for a in ARMS:
                    v = agg(tag, n_name, a, met)
                    cells.append(f"{np.mean(list(v.values())):.3f}"
                                 if v else "—")
                w(f"| {tag} | " + " | ".join(cells) + " |")

    w("\n## 判据\n")
    for n_name in ("same-matrix", "old-heldout"):
        for tag in ("0", "5"):
            for met in ("auc", "tpr5"):
                r = agg(tag, n_name, "rf", met)
                if not r:
                    continue
                com = sorted(r); xr = np.array([r[e] for e in com])
                out = [f"- **{n_name} k={tag} {met}** rf {xr.mean():.3f}"]
                for a in ("zs", "zs-m", "mf-net-s02", "mf-net-m-s02",
                          "mf-net-s06", "mf-zs-s02", "mf-common-s02",
                          "mf-rand-s02"):
                    v = agg(tag, n_name, a, met)
                    if len(set(v) & set(com)) < 3:
                        continue
                    xv = np.array([v[e] for e in com if e in v])
                    xc = np.array([r[e] for e in com if e in v])
                    out.append(f"{a} {xv.mean():.3f}(Δ={np.mean(xv-xc):+.3f},"
                               f"p={ttest_rel(xv, xc).pvalue:.3g})")
                w(" | ".join(out))
                for a, b in (("mf-net-s02", "mf-rand-s02"),
                             ("mf-net-m-s02", "mf-rand-m-s02")):
                    xa, xb = agg(tag, n_name, a, met), agg(tag, n_name, b, met)
                    com2 = sorted(set(xa) & set(xb))
                    if len(com2) < 3:
                        continue
                    va = np.array([xa[e] for e in com2])
                    vb = np.array([xb[e] for e in com2])
                    w(f"  - 化学增量 {a} − {b}: {np.mean(va-vb):+.3f} "
                      f"(p={ttest_rel(va, vb).pvalue:.3g}, n={len(com2)})")

    with open(os.path.join(OUT, "exp73_p6_v4_dualhead.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
