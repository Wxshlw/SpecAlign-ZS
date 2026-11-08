"""exp71 —— 诊断「化学先验为何在检出上有负作用」+ 改进方案验证

三个待验证机制（可证伪）：
  M1 方向污染：w_T 与真实净信号方向 b* = E[z_ad] − E[z_pure] 余弦 ≈ 0
               ⇒ 化学外推方向根本不指向「掺杂物净信号」
  M2 尺度挤占：score = z·w_T = (z_mtx·w_T) + (b·w_T)，
               基质项方差 ≫ 净信号项均值差 ⇒ 判别力被噪声埋掉
  M3 共线致 margin 失效：老类 w_j 相互 cos≈1 ⇒ 纯基质谱上所有 s_j 同涨同落，
               扣 max 后残留大噪声（exp58 共线问题的另一面）

四种改进：
  A1 zs-ctr  ：z 减去基质中心（一阶去基线）
  A2 zs-orth ：z 与 w_T 同时投影掉基质 PCA 子空间（OSC 式高阶去基线）
  A3 net     ：ridge 回归目标由 Z_old 改为净信号 ΔZ = Z_j − Z_matrix(j)
  A4 net-orth：A3 + A2
  参考臂：res-norm（残差范数，无化学先验）、res-dir（残差方向对齐）、rf / rf-m
协议：按样品 id 分组（exp67 定稿）。
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
                                         auc_of, tpr_at_fpr, macro_recall,
                                         EID2MID)
from exp68_p6_cond_scorer import load_groups, split_by_group

OUT = os.path.join(os.path.dirname(HERE), "out")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OLD_FRAC = 0.7
NPC = 8          # 基质子空间主成分数


def cos(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb + 1e-12)) if na > 0 and nb > 0 else 0.0


def pick_lam(S_o, Ztgt, lams, net, Xval, yval, rows, mu_S, S_all):
    """按验证集 macro-recall 选 λ（与 exp69 一致）。"""
    best, bv = lams[0], -1.0
    for lam in lams:
        W = (S_all - mu_S) @ head_ridge_R(S_o, Ztgt, lam)
        v = macro_recall(net, W, Xval, yval, rows)
        if v > bv:
            bv, best = v, lam
    return best


GCACHE = os.path.join(OUT, "g1_groups_cache.npz")


def load_groups_cached():
    """按样品 id 分组信息（p6_data.load_all 很慢，缓存到 out/）。"""
    if os.path.exists(GCACHE):
        z = np.load(GCACHE, allow_pickle=True)
        keys = set()
        for k in z.files:
            keys.add(k.rsplit("_", 1)[0])
        out = {}
        for e in sorted(keys):
            out[e] = dict(X=np.asarray(z[f"{e}_X"], float),
                          y=np.asarray(z[f"{e}_y"], float),
                          gid=np.asarray(z[f"{e}_gid"]))
        return out
    g = load_groups()
    np.savez_compressed(
        GCACHE, **{f"{e}_X": np.asarray(v["X"], float) for e, v in g.items()},
        **{f"{e}_y": np.asarray(v["y"], float) for e, v in g.items()},
        **{f"{e}_gid": np.asarray(v["gid"]) for e, v in g.items()})
    print(f"[exp71] 分组缓存已建 -> {GCACHE}", flush=True)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.0001,0.001,0.01,0.1,1")
    ap.add_argument("--eids", default="")
    ap.add_argument("--shots", default="0,5")
    ap.add_argument("--npc", type=int, default=NPC)
    ap.add_argument("--out", default="exp71_diag.json")
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
    res = {"M": {}, "D": {}}
    print(f"[exp71] device={DEV} 实体={eids} shots={shots} seeds={args.seeds}",
          flush=True)

    for e_hold in eids:
        others = [e for e in sorted(ent) if e != e_hold]
        Xe = np.asarray(ent[e_hold], float)
        gid_e = GR[e_hold]["gid"]
        mid_h = EID2MID[e_hold]
        S_all = T_all[[ki[e] for e in others]
                      + [ki[f"M_{m}"] for m in mids] + [ki[e_hold]]]
        NEW = len(others) + len(mids)
        cand_names = others + [e_hold]
        S_cand = T_all[[ki[e] for e in cand_names]]
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

                # ---- 基质中心与基质 PCA 子空间（只用训练基质谱） ----
                Zmid_c, mid_stack = {}, []
                for m in mids:
                    Zm = embed(net, np.asarray(mid_tr[m], float))
                    Zmid_c[m] = Zm.mean(0)
                    mid_stack.append(Zm - Zm.mean(0))
                Mall = np.concatenate(mid_stack, 0)
                U, Sv, Vt = np.linalg.svd(Mall, full_matrices=False)
                mu_mid = np.mean([Zmid_c[m] for m in mids], 0)
                # 投影基 = [基质平均中心方向, 基质 PCA 前 npc 个主成分]
                e0 = mu_mid / (np.linalg.norm(mu_mid) + 1e-12)
                Pm = np.concatenate([e0[None], Vt[:args.npc]], 0)
                Pm = Pm / (np.linalg.norm(Pm, axis=1, keepdims=True) + 1e-12)

                def orth(Z):
                    """投影掉「基质中心 + 基质变异子空间」，只留基质之外的成分。"""
                    Zc2 = np.asarray(Z, float)
                    return Zc2 - (Zc2 @ Pm.T) @ Pm

                # ---- 头 1：原 zs（目标 = 类中心嵌入）----
                lam1 = pick_lam(S_o, Zc, lams, net, Xval, yval, rows,
                                mu_S, S_all)
                W = (S_all - mu_S) @ head_ridge_R(S_o, Zc, lam1)
                # ---- 头 2：net（目标 = 净信号 ΔZ）----
                dZ = np.zeros_like(Zc)
                n_nomat = 0
                for ri, r in enumerate(rows):
                    e_name = others[r] if r < len(others) else None
                    if r == NEW and len(tr_e_ix):
                        dZ[ri] = Zc[ri] - Zmid_c[mid_h]   # T 行（k>0）
                    elif r < len(others) and e_name in EID2MID:
                        dZ[ri] = Zc[ri] - Zmid_c[EID2MID[e_name]]
                    elif r < NEW:
                        dZ[ri] = 0.0          # 纯基质行：净信号恒为零
                    else:
                        # 无基质对照的实体（COFA/MEATCH 等）：退化为减全局基质中心
                        dZ[ri] = Zc[ri] - mu_mid
                        n_nomat += 1
                Wn = (S_all - mu_S) @ head_ridge_R(S_o, dZ, lam1)

                wT, wTn = W[NEW], Wn[NEW]
                W_cand = np.concatenate([W[:len(others)], W[NEW:NEW + 1]], 0)
                Wn_cand = np.concatenate([Wn[:len(others)],
                                          Wn[NEW:NEW + 1]], 0)

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
                    if len(tr_e_ix):
                        i_rf = list(rf.classes_).index(NEW)
                        i_sv = list(sv.classes_).index(NEW)
                        Pn = rf.predict_proba(Xn_s); Po = rf.predict_proba(Xo_s)
                        Qn = sv.predict_proba(Xn_s); Qo = sv.predict_proba(Xo_s)
                        base = {"rf": (Pn[:, i_rf], Po[:, i_rf]),
                                "svm": (Qn[:, i_sv], Qo[:, i_sv]),
                                "rf-m": (Pn[:, i_rf] - np.delete(Pn, i_rf, 1).max(1),
                                         Po[:, i_rf] - np.delete(Po, i_rf, 1).max(1))}
                    else:
                        Pn = rf.predict_proba(Xn_s); Po = rf.predict_proba(Xo_s)
                        Qn = sv.predict_proba(Xn_s); Qo = sv.predict_proba(Xo_s)
                        base = {"rf": (1 - Pn.max(1), 1 - Po.max(1)),
                                "svm": (1 - Qn.max(1), 1 - Qo.max(1))}

                    arms = dict(base)
                    # zs 系列
                    Ln, Lo = Zn @ W_cand.T, Zo @ W_cand.T
                    arms["zs"] = (Ln[:, iT], Lo[:, iT])
                    arms["zs-m"] = (Ln[:, iT] - np.delete(Ln, iT, 1).max(1),
                                    Lo[:, iT] - np.delete(Lo, iT, 1).max(1))
                    Znc, Zoc = Zn - mu_mid, Zo - mu_mid
                    arms["zs-ctr"] = (Znc @ wT, Zoc @ wT)
                    Zno, Zoo = orth(Zn), orth(Zo)
                    wTo = wT - (wT @ Pm.T) @ Pm
                    arms["zs-orth"] = (Zno @ wTo, Zoo @ wTo)
                    # net 系列
                    Mn, Mo = Zn @ Wn_cand.T, Zo @ Wn_cand.T
                    arms["net"] = (Mn[:, iT], Mo[:, iT])
                    arms["net-m"] = (Mn[:, iT] - np.delete(Mn, iT, 1).max(1),
                                     Mo[:, iT] - np.delete(Mo, iT, 1).max(1))
                    arms["net-ctr"] = (Znc @ wTn, Zoc @ wTn)
                    wTno = wTn - (wTn @ Pm.T) @ Pm
                    arms["net-orth"] = (Zno @ wTno, Zoo @ wTno)
                    # 残差（无化学先验）与方向对齐
                    rn = np.linalg.norm(Zno, axis=1)
                    ro = np.linalg.norm(Zoo, axis=1)
                    arms["res-norm"] = (rn, ro)
                    if np.linalg.norm(wTno) > 1e-9:
                        u = wTno / np.linalg.norm(wTno)
                        arms["res-dir"] = (Zno @ u, Zoo @ u)
                        arms["res-comb"] = (rn * (Zno @ u), ro * (Zoo @ u))

                    for nm, (sn, so) in arms.items():
                        res["M"].setdefault(tag, {}).setdefault(
                            n_name, {}).setdefault(nm, {}).setdefault(
                            e_hold, []).append(
                            dict(auc=round(auc_of(sn, so), 4),
                                 tpr5=round(tpr_at_fpr(sn, so, 0.05), 4),
                                 tpr20=round(tpr_at_fpr(sn, so, 0.20), 4)))

                # ============ 机制诊断（只在 same-matrix 语义下） ============
                Zn_all = embed(net, Xte_new)
                Zp_all = embed(net, np.asarray(mid_ref[mid_h], float))
                bstar = Zn_all.mean(0) - Zp_all.mean(0)      # oracle 净信号
                d = {}
                d["cos_wT_bstar"] = round(cos(wT, bstar), 4)
                d["cos_wTn_bstar"] = round(cos(wTn, bstar), 4)
                d["cos_wT_mu_mid"] = round(cos(wT, mu_mid), 4)
                d["cos_wTn_mu_mid"] = round(cos(wTn, mu_mid), 4)
                # M2：基质项 vs 净信号项的尺度比
                sp = Zp_all @ wT; sa = Zn_all @ wT
                d["d_zs"] = round(float((sa.mean() - sp.mean())
                                        / (np.sqrt((sa.var() + sp.var()) / 2)
                                           + 1e-12)), 4)
                Zpo, Zno_ = orth(Zp_all), orth(Zn_all)
                sp2 = Zpo @ wTo; sa2 = Zno_ @ wTo
                d["d_zs_orth"] = round(float((sa2.mean() - sp2.mean())
                                             / (np.sqrt((sa2.var() + sp2.var()) / 2)
                                                + 1e-12)), 4)
                sp3 = Zp_all @ wTn; sa3 = Zn_all @ wTn
                d["d_net"] = round(float((sa3.mean() - sp3.mean())
                                         / (np.sqrt((sa3.var() + sp3.var()) / 2)
                                            + 1e-12)), 4)
                # 基质项波动 vs 净信号项均值差
                d["ratio_mtx_noise"] = round(float(
                    np.sqrt((sp.var() + (Zp_all @ wT).var()) / 2)
                    / (abs(sa.mean() - sp.mean()) + 1e-12)), 4)
                # M3：老类权重共线
                Wo = W[:len(others)]
                Cn = nrm(Wo) @ nrm(Wo).T
                np.fill_diagonal(Cn, -9)
                d["cos_wmax_old"] = round(float(Cn.max()), 4)
                d["cos_w_mean_old"] = round(float(Cn[Cn > -1].mean()), 4)
                # 老类净信号方向之间的共线（对照）
                d["nrm_wT"] = round(float(np.linalg.norm(wT)), 3)
                d["nrm_wTn"] = round(float(np.linalg.norm(wTn)), 3)
                res["D"].setdefault(tag, {}).setdefault(e_hold, []).append(d)

        dd = res["D"].get("0", {}).get(e_hold, [{}])[-1]
        print(f"[{e_hold:>4}] cos(wT,b*)={dd.get('cos_wT_bstar'):+.3f} "
              f"cos(wTnet,b*)={dd.get('cos_wTn_bstar'):+.3f} "
              f"d_zs={dd.get('d_zs'):+.2f} d_orth={dd.get('d_zs_orth'):+.2f} "
              f"d_net={dd.get('d_net'):+.2f} "
              f"noise/sig={dd.get('ratio_mtx_noise'):.2f}", flush=True)

    with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[exp71] 已写出 out/{args.out}")
    write_report(res, eids, args.out)


def write_report(res, eids, out_name):
    from scipy.stats import ttest_rel
    L = []
    w = L.append
    w("# exp71 —— 化学先验在「检出」上负作用的机制诊断与修复\n")
    w("## 一、机制诊断（k=0，oracle 净信号 b* = E[z_掺混] − E[z_纯基质]）\n")
    w("| 实体 | cos(w_T, b*) | cos(w_T^net, b*) | cos(w_T, μ_基质) | "
      "d(zs) | d(zs-orth) | d(net) | 基质噪声/信号 | 老类 w 最大 cos |")
    w("|---|---|---|---|---|---|---|---|---|")
    for e in eids:
        ds = res["D"].get("0", {}).get(e, [])
        if not ds:
            continue
        g = lambda k: np.mean([x.get(k, np.nan) for x in ds])
        w(f"| {e} | {g('cos_wT_bstar'):+.3f} | {g('cos_wTn_bstar'):+.3f} | "
          f"{g('cos_wT_mu_mid'):+.3f} | {g('d_zs'):+.2f} | "
          f"{g('d_zs_orth'):+.2f} | {g('d_net'):+.2f} | "
          f"{g('ratio_mtx_noise'):.2f} | {g('cos_wmax_old'):.3f} |")
    allv = [x for e in eids for x in res["D"].get("0", {}).get(e, [])]
    if allv:
        g = lambda k: np.mean([x.get(k, np.nan) for x in allv])
        w(f"| **macro** | **{g('cos_wT_bstar'):+.3f}** | "
          f"**{g('cos_wTn_bstar'):+.3f}** | {g('cos_wT_mu_mid'):+.3f} | "
          f"**{g('d_zs'):+.2f}** | **{g('d_zs_orth'):+.2f}** | "
          f"**{g('d_net'):+.2f}** | {g('ratio_mtx_noise'):.2f} | "
          f"{g('cos_wmax_old'):.3f} |")

    def agg(tag, n_name, arm, met):
        per = {}
        for e in eids:
            vs = [x[met] for x in res["M"].get(tag, {}).get(n_name, {})
                  .get(arm, {}).get(e, [])]
            if vs:
                per[e] = float(np.mean(vs))
        return per

    ARMS = ["zs", "zs-m", "zs-ctr", "zs-orth", "net", "net-ctr", "net-orth",
            "res-norm", "res-dir", "res-comb", "rf", "rf-m", "svm"]
    for met in ("auc", "tpr5"):
        w(f"\n## 二、{met}（group 协议，macro over {len(eids)} 实体）\n")
        w("| 口径 | k | " + " | ".join(f"`{a}`" for a in ARMS) + " |")
        w("|---|---|" + "---|" * len(ARMS))
        for n_name in ("same-matrix", "old-heldout"):
            for tag in ("0", "5"):
                if not res["M"].get(tag, {}).get(n_name):
                    continue
                cells = []
                for a in ARMS:
                    v = agg(tag, n_name, a, met)
                    cells.append(f"{np.mean(list(v.values())):.3f}"
                                 if v else "—")
                w(f"| {n_name} | {tag} | " + " | ".join(cells) + " |")

    w("\n## 三、判据（配对 t，vs rf）\n")
    for n_name in ("same-matrix", "old-heldout"):
        for tag in ("0", "5"):
            for met in ("auc", "tpr5"):
                r = agg(tag, n_name, "rf", met)
                if not r:
                    continue
                com = sorted(r)
                xr = np.array([r[e] for e in com])
                msg = [f"- **{n_name} k={tag} {met}**: rf {xr.mean():.3f}"]
                for a in ("zs", "zs-ctr", "zs-orth", "net", "net-orth",
                          "res-comb"):
                    v = agg(tag, n_name, a, met)
                    if len(set(v) & set(com)) < 3:
                        continue
                    xv = np.array([v[e] for e in com if e in v])
                    xc = np.array([r[e] for e in com if e in v])
                    msg.append(f"{a} {xv.mean():.3f} "
                               f"(Δ={np.mean(xv - xc):+.3f}, "
                               f"p={ttest_rel(xv, xc).pvalue:.3g})")
                w(" | ".join(msg) + f" (n={len(com)})")

    with open(os.path.join(OUT, "exp71_p6_detection_fix.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
