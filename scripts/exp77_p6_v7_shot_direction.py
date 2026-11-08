"""exp77 —— v7：k-shot 净信号方向（b̂）取代化学外推方向

决定性依据（exp71 已证）：
    cos(w_T, b*) = +0.157   ← 化学外推方向 与 真实净信号方向 几乎正交
    b* = E[z_掺混] − E[z_纯基质]
即：**化学先验给不出「检出的方向」**，它只给得出「识别的方向」。

但 k>0 时我们手上有 T 的 k 条真实谱，可以直接估计
    b̂ = mean(z_T^k) − μ_基质
这是 b* 的一致估计（不依赖任何化学先验）。k=0 时无此估计，只能退回 w_T。

⇒ **本文的主张因此变得清晰且可证**：
   k=0 化学先验是唯一方向来源（弱，但非零信息：chem−rand 恒正）
   k>0 只需 k 条真实谱即可拿到近乎 oracle 的方向（强）
  RF/SVM 拿到同样的 k 条却要学 20 类判别边界，样本效率更低

臂（same-matrix 检出为主，old-heldout 确认识别不被伤）：
  zs              化学外推（识别头）
  mf-net          化学净信号 + 白化（v4 最佳）
  mf-common       数据驱动通用方向 + 白化（无化学，k=0 可用）
  shot            b̂ 净信号方向（非白化），k>0
  mf-shot         b̂ + 白化，k>0                       ★ 本轮核心
  mf-shot-fuse    b̂ 与 w_T(net) 的方向融合，k>0
  mf-midnet       同基质其他老类的净信号平均方向（k=0 也可用）
  rf / svm        外部基线
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "论文8"))
import p6_g1_data as G1
from exp63_p6_screening_protocol import (stage1, embed, head_ridge_R, nrm,
                                         auc_of, tpr_at_fpr, macro_recall,
                                         EID2MID)
from exp68_p6_cond_scorer import split_by_group
from exp71_p6_detection_fix import load_groups_cached, pick_lam
from exp74_p6_fairness_audit import load_bg_gid, split_group
from exp75_p6_v5_universal_detector import whiten, metrics, zsc

OUT = os.path.join(os.path.dirname(HERE), "out")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OLD_FRAC = 0.7
SHRI = 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.0001,0.001,0.01,0.1,1")
    ap.add_argument("--eids", default="")
    ap.add_argument("--shots", default="0,5")
    ap.add_argument("--proto", default="group")
    ap.add_argument("--out", default="exp77_v7.json")
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(",")]
    shots = [int(s) for s in args.shots.split(",")]

    ent, conc, bg, _ = G1.load_g1_conc()
    GR = load_groups_cached()
    BG_GID = load_bg_gid(bg)
    eids = sorted(set(GR) & set(EID2MID))
    if args.eids:
        eids = [e for e in eids if e in args.eids.split(",")]
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]
    zc = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    T_all = np.asarray(zc["clm"], float)
    ki = {k: i for i, k in enumerate(keys_all)}
    dz = T_all.shape[1]
    res = {"M": {}, "COS": {}}
    print(f"[exp77] device={DEV} 实体={eids} shots={shots} proto={args.proto} "
          f"seeds={args.seeds}", flush=True)

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
            tag = str(k)
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed * 7919 + 13)
                te_ix, pool_ix = split_by_group(Xe, gid_e, rng)
                Xte_new = Xe[te_ix]
                tr_e_ix = pool_ix[:min(k, len(pool_ix))] if k else []

                parts, ys = [], []
                old_ref, mid_tr, mid_ref = {}, {}, {}
                for j, e in enumerate(others):
                    Xo = np.asarray(ent[e], float)
                    if args.proto == "group" and e in GR:
                        a, b, _ = split_group(Xo, GR[e]["gid"], rng)
                    else:
                        pp = rng.permutation(len(Xo))
                        cut = max(2, int(round(OLD_FRAC * len(Xo))))
                        a, b = pp[:cut], pp[cut:]
                    old_ref[e] = Xo[b]
                    parts.append(Xo[a]); ys.append(np.full(len(a), j))
                for j, m in enumerate(mids):
                    Xo = np.asarray(bg[m], float)
                    if args.proto == "group":
                        a, b, _ = split_group(Xo, BG_GID[m], rng)
                    else:
                        pp = rng.permutation(len(Xo))
                        cut = max(2, int(round(OLD_FRAC * len(Xo))))
                        a, b = pp[:cut], pp[cut:]
                    mid_ref[m] = Xo[b]; mid_tr[m] = Xo[a]
                    parts.append(Xo[a])
                    ys.append(np.full(len(a), len(others) + j))
                if len(tr_e_ix):
                    parts.append(Xe[tr_e_ix])
                    ys.append(np.full(len(tr_e_ix), NEW))
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
                nrows = NEW + 1 if len(tr_e_ix) else NEW
                net, _ = stage1(Xfit, yfit, dz, nrows, args.steps,
                                seed * 1000 + eids.index(e_hold), S=S_all)
                Zfit = embed(net, Xfit)
                rows = sorted({int(y) for y in yfit})
                Zc = np.stack([Zfit[yfit == j].mean(0) for j in rows])
                S_o, mu_S = S_all[rows], S_all[rows].mean(0, keepdims=True)
                Zmid_c = {m: embed(net, np.asarray(mid_tr[m], float)).mean(0)
                          for m in mids}
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
                W = (S_all - mu_S) @ head_ridge_R(S_o, Zc, lam1)
                Wn = (S_all - mu_S) @ head_ridge_R(S_o, dZ, lam1)

                R = np.concatenate([Zfit[yfit == j] - Zfit[yfit == j].mean(0)
                                    for j in rows], 0)
                Sw = R.T @ R / max(1, len(R))
                tr_ = float(np.trace(Sw)) / Sw.shape[0]
                Sig = (1 - SHRI) * Sw + SHRI * tr_ * np.eye(Sw.shape[0])
                Sig += 1e-6 * np.eye(Sig.shape[0])

                Zn = embed(net, Xte_new)
                # ---------- oracle 净信号方向 b*（仅用于诊断，不进得分） ----------
                b_star = Zn.mean(0) - embed(net, np.asarray(
                    mid_ref[mid_h], float)).mean(0)
                b_star /= (np.linalg.norm(b_star) + 1e-12)

                # ---------- k-shot 净信号方向 b̂（浓度加权最小二乘）----------
                # 光谱混合模型：x_i = μ_基质 + c_i·a + ε  ⇒  â = Σc_i(x_i−μ)/Σc_i²
                # 直接用均值估计会把低浓度谱的噪声当成信号，WLS 才是正确估计。
                muX_h = np.asarray(mid_tr[mid_h], float).mean(0)
                cc_all = conc.get(e_hold)
                b_hat = b_hat_w = None
                b_raw = b_raw_w = None
                if len(tr_e_ix):
                    cc = None
                    if cc_all is not None and np.isfinite(
                            np.asarray(cc_all, float)).all():
                        cc = np.asarray(cc_all, float)[tr_e_ix]
                        cc = cc / (np.nanmax(np.asarray(cc_all, float)) + 1e-9)
                    ZT = embed(net, Xe[tr_e_ix])
                    DZ = ZT - Zmid_c[mid_h]
                    DX = np.asarray(Xe[tr_e_ix], float) - muX_h
                    if cc is not None and (cc ** 2).sum() > 1e-9:
                        a_z = (cc[:, None] * DZ).sum(0) / (cc ** 2).sum()
                        a_x = (cc[:, None] * DX).sum(0) / (cc ** 2).sum()
                    else:
                        a_z, a_x = DZ.mean(0), DX.mean(0)
                    def _u(v):
                        nv = np.linalg.norm(v)
                        return v / nv if nv > 1e-12 else None
                    b_hat, b_hat_w = _u(DZ.mean(0)), _u(a_z)
                    b_raw, b_raw_w = _u(DX.mean(0)), _u(a_x)

                # 同基质其他老类的净信号平均方向（k=0 也可用）
                dmid = []
                for r in rows:
                    if r < len(others) and EID2MID.get(others[r]) == mid_h:
                        v = Zc[r] - Zmid_c[mid_h]
                        nv = np.linalg.norm(v)
                        if nv > 1e-12:
                            dmid.append(v / nv)
                d_midnet = np.mean(dmid, 0) if dmid else None
                if d_midnet is not None:
                    nn = np.linalg.norm(d_midnet)
                    d_midnet = (d_midnet / nn) if nn > 1e-12 else None

                res["COS"].setdefault(tag, {}).setdefault(e_hold, []).append(
                    dict(cos_chem=round(float(
                        Wn[NEW] @ b_star / (np.linalg.norm(Wn[NEW]) + 1e-12)),
                        4),
                        cos_shot=(round(float(b_hat @ b_star), 4)
                                  if b_hat is not None else None)))

                # ================= 原始谱空间「差谱」匹配滤波 =================
                # FTIR 掺假检测的经典标准操作：差谱 = mean(掺混谱) − mean(纯基质谱)
                # Beer-Lambert 下差谱就是掺杂物信号的线性估计。
                Xmid_h_tr = np.asarray(mid_tr[mid_h], float)
                muX_h = Xmid_h_tr.mean(0)
                Rx = np.concatenate(
                    [Xfit[yfit == j] - Xfit[yfit == j].mean(0)
                     for j in rows], 0)
                Swx = Rx.T @ Rx / max(1, len(Rx))
                trx = float(np.trace(Swx)) / Swx.shape[0]
                Sigx = (1 - SHRI) * Swx + SHRI * trx * np.eye(Swx.shape[0])
                Sigx += 1e-6 * np.eye(Sigx.shape[0])
                # k>0：真实差谱（b_raw / b_raw_w 已在上方按 WLS 算好）
                # k=0：同基质老类差谱均值 / 全老类差谱均值 / 化学预测差谱
                ds_mid, ds_all = [], []
                for j, e in enumerate(others):
                    if j not in rows:
                        continue
                    mm = EID2MID.get(e)
                    if mm is None:
                        continue
                    v = np.asarray(ent[e], float)[:0]      # 占位
                    Xe_tr = Xfit[yfit == j].mean(0)
                    Xm_tr = np.asarray(mid_tr[mm], float).mean(0)
                    d = Xe_tr - Xm_tr
                    nd = np.linalg.norm(d)
                    if nd < 1e-12:
                        continue
                    ds_all.append(d / nd)
                    if mm == mid_h:
                        ds_mid.append(d / nd)
                dsr_mid = np.mean(ds_mid, 0) if ds_mid else None
                dsr_all = np.mean(ds_all, 0) if ds_all else None
                for v in (dsr_mid, dsr_all):
                    if v is not None:
                        nv = np.linalg.norm(v)
                        if nv > 1e-12:
                            v /= nv
                # 化学预测差谱（ridge: S → 差谱，原始谱空间）
                dX = np.zeros((len(rows), Xfit.shape[1]))
                for ri, r in enumerate(rows):
                    if r == NEW and len(tr_e_ix):
                        dX[ri] = Xfit[yfit == r].mean(0) - muX_h
                    elif r < len(others) and others[r] in EID2MID:
                        mm = EID2MID[others[r]]
                        dX[ri] = Xfit[yfit == r].mean(0) - np.asarray(
                            mid_tr[mm], float).mean(0)
                Wdx = (S_all - mu_S) @ head_ridge_R(S_o, dX, lam1)

                from sklearn.preprocessing import StandardScaler
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
                    Wc = np.concatenate([W[:len(others)], W[NEW:NEW + 1]], 0)
                    Ln, Lo = Zn @ Wc.T, Zo @ Wc.T
                    arms["zs"] = (Ln[:, iT], Lo[:, iT])
                    arms["zs-m"] = (Ln[:, iT] - np.delete(Ln, iT, 1).max(1),
                                    Lo[:, iT] - np.delete(Lo, iT, 1).max(1))
                    Wcn = np.concatenate([Wn[:len(others)], Wn[NEW:NEW + 1]], 0)
                    An = np.linalg.solve(Sig, Wcn.T).T
                    arms["mf-net"] = ((Zn - mu_mid) @ An.T)[:, iT], \
                                     ((Zo - mu_mid) @ An.T)[:, iT]
                    # 通用方向（老类净信号均值）+ 白化
                    d_old = []
                    for r in rows:
                        if r < len(others) and others[r] in EID2MID:
                            v = Zc[r] - Zmid_c[EID2MID[others[r]]]
                            nv = np.linalg.norm(v)
                            if nv > 1e-12:
                                d_old.append(v / nv)
                    if d_old:
                        dcom = np.mean(d_old, 0)
                        dcom /= (np.linalg.norm(dcom) + 1e-12)
                        dc = np.linalg.solve(Sig, dcom)
                        arms["mf-common"] = ((Zn - mu_mid) @ dc,
                                             (Zo - mu_mid) @ dc)
                    if d_midnet is not None:
                        dm = np.linalg.solve(Sig, d_midnet)
                        arms["mf-midnet"] = ((Zn - mu_mid) @ dm,
                                             (Zo - mu_mid) @ dm)
                    # ★ k-shot 净信号方向
                    if b_hat is not None:
                        arms["shot"] = (Zn @ b_hat, Zo @ b_hat)
                        bs = np.linalg.solve(Sig, b_hat)
                        arms["mf-shot"] = ((Zn - mu_mid) @ bs,
                                           (Zo - mu_mid) @ bs)
                        wnet = Wn[NEW] / (np.linalg.norm(Wn[NEW]) + 1e-12)
                        bf = b_hat + wnet
                        bf /= (np.linalg.norm(bf) + 1e-12)
                        bfs = np.linalg.solve(Sig, bf)
                        arms["mf-shot-fuse"] = ((Zn - mu_mid) @ bfs,
                                                (Zo - mu_mid) @ bfs)
                    # ================= 原始谱差谱臂 =================
                    def ds_score(dirv, Xpos, Xneg):
                        d = np.asarray(dirv, float)
                        bs = np.linalg.solve(Sigx, d)
                        return ((Xpos - muX_h) @ bs, (Xneg - muX_h) @ bs)

                    if b_raw_w is not None:
                        arms["ds-wls"] = ds_score(b_raw_w, Xte_new, Xn_neg)
                        arms["ds-wls-nw"] = (Xte_new @ b_raw_w,
                                             Xn_neg @ b_raw_w)
                    if b_hat_w is not None:
                        bsw = np.linalg.solve(Sig, b_hat_w)
                        arms["mf-shot-wls"] = ((Zn - mu_mid) @ bsw,
                                               (Zo - mu_mid) @ bsw)
                    if b_raw is not None:
                        arms["ds-raw"] = ds_score(b_raw, Xte_new, Xn_neg)
                        arms["ds-raw-nw"] = (Xte_new @ b_raw, Xn_neg @ b_raw)
                    if dsr_mid is not None:
                        arms["ds-midnet-raw"] = ds_score(
                            dsr_mid, Xte_new, Xn_neg)
                    if dsr_all is not None:
                        arms["ds-common-raw"] = ds_score(
                            dsr_all, Xte_new, Xn_neg)
                    wdx = Wdx[NEW]
                    nw = np.linalg.norm(wdx)
                    if nw > 1e-12:
                        arms["ds-chem-raw"] = ds_score(
                            wdx / nw, Xte_new, Xn_neg)
                    # 差谱 ⊕ 化学（嵌入空间）融合
                    if b_raw is not None:
                        a1 = np.concatenate(arms["ds-raw"])
                        a2 = np.concatenate(arms["mf-net"])
                        nz = len(Xte_new)
                        arms["fuse-ds-mfnet"] = (zsc(a1)[:nz] + zsc(a2)[:nz],
                                                 zsc(a1)[nz:] + zsc(a2)[nz:])
                    # 外部
                    Pn = rf.predict_proba(Xn_s); Po = rf.predict_proba(Xo_s)
                    Qn = sv.predict_proba(Xn_s); Qo = sv.predict_proba(Xo_s)
                    if len(tr_e_ix):
                        i_rf = list(rf.classes_).index(NEW)
                        i_sv = list(sv.classes_).index(NEW)
                        arms["rf"] = (Pn[:, i_rf], Po[:, i_rf])
                        arms["svm"] = (Qn[:, i_sv], Qo[:, i_sv])
                    else:
                        arms["rf"] = (1 - Pn.max(1), 1 - Po.max(1))
                        arms["svm"] = (1 - Qn.max(1), 1 - Qo.max(1))

                    for nm, (sn, so) in arms.items():
                        res["M"].setdefault(tag, {}).setdefault(
                            n_name, {}).setdefault(nm, {}).setdefault(
                            e_hold, []).append(metrics(sn, so))

        line = []
        for arm in ("mf-shot", "mf-net", "rf"):
            d = res["M"].get("5", {}).get("same-matrix", {}).get(
                arm, {}).get(e_hold)
            if d:
                line.append(f"{arm}={np.mean([x['auc'] for x in d]):.3f}")
        print(f"[{e_hold:>4}] " + " ".join(line), flush=True)

    with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[exp77] 已写出 out/{args.out}")
    write_report(res, eids, shots)


SHOW = ["zs", "zs-m", "mf-net", "mf-common", "mf-midnet",
        "shot", "mf-shot", "mf-shot-fuse", "mf-shot-wls",
        "ds-raw", "ds-raw-nw", "ds-wls", "ds-wls-nw",
        "ds-midnet-raw", "ds-common-raw",
        "ds-chem-raw", "fuse-ds-mfnet", "rf", "svm"]
DS = ("ds-raw", "ds-raw-nw", "ds-wls", "ds-wls-nw", "ds-midnet-raw",
      "ds-common-raw", "ds-chem-raw", "fuse-ds-mfnet", "mf-shot-wls")


def write_report(res, eids, shots):
    from scipy.stats import ttest_rel
    L = []
    w = L.append
    w("# exp77 —— v7：k-shot 净信号方向 b̂ 取代化学外推方向\n")
    w("exp71 已证 cos(w_T, b*) = **+0.157**（化学方向 ⊥ 真实净信号方向）。\n"
      "k>0 时可直接用 k 条真实谱估计 b̂ = mean(z_T^k) − μ_基质，\n"
      "这是 b* 的一致估计，**不依赖化学先验**。\n")
    w("| 任务 | k | 方向来源 | 说明 |")
    w("|---|---|---|---|")
    w("| 检出 | 0 | `mf-net`（化学）/ `mf-common`（老类净信号均值） | 无 b̂ |")
    w("| 检出 | >0 | **`mf-shot`（b̂）** | k 条真实谱 |")
    w("| 识别 | 任意 | `zs-m`（化学外推 + margin） | 化学先验主场 |\n")

    def agg(tag, n_name, arm, met="auc"):
        per = {}
        for e in eids:
            vs = [x[met] for x in res["M"].get(tag, {}).get(
                n_name, {}).get(arm, {}).get(e, [])]
            if vs:
                per[e] = float(np.mean(vs))
        return per

    w("## 〇、方向质量：cos(·, b*)\n")
    w("| 实体 | k | cos(w_chem, b*) | cos(b̂, b*) |")
    w("|---|---|---|---|")
    for tag in [str(s) for s in shots]:
        cs, ss = [], []
        for e in eids:
            for d in res["COS"].get(tag, {}).get(e, []):
                cs.append(d["cos_chem"])
                if d["cos_shot"] is not None:
                    ss.append(d["cos_shot"])
        if cs:
            w(f"| **macro** | {tag} | {np.mean(cs):+.3f} | "
              f"{np.mean(ss):+.3f} |" if ss else
              f"| **macro** | {tag} | {np.mean(cs):+.3f} | — |")

    for n_name in ("same-matrix", "old-heldout"):
        for met in ("auc", "tpr5"):
            w(f"\n## {n_name} · {met}\n")
            w("| k | " + " | ".join(f"`{a}`" for a in SHOW) + " |")
            w("|---|" + "---|" * len(SHOW))
            for tag in [str(s) for s in shots]:
                row = []
                for a in SHOW:
                    d = agg(tag, n_name, a, met)
                    row.append(f"{np.mean(list(d.values())):.3f}" if d else "—")
                w(f"| {tag} | " + " | ".join(row) + " |")

    w("\n## 判据（same-matrix · auc，vs rf）\n")
    for tag in [str(s) for s in shots]:
        base = agg(tag, "same-matrix", "rf", "auc")
        for a in ("mf-net", "mf-common", "mf-midnet", "shot", "mf-shot",
                  "mf-shot-fuse") + DS:
            y = agg(tag, "same-matrix", a, "auc")
            com = sorted(set(base) & set(y))
            if len(com) < 3:
                continue
            xb = np.array([base[e] for e in com])
            yb = np.array([y[e] for e in com])
            p = ttest_rel(yb, xb).pvalue
            d = np.mean(yb - xb)
            flag = "✅**显著领先**" if (d > 0 and p < 0.05) else (
                "❌显著落后" if (d < 0 and p < 0.05) else "持平(n.s.)")
            w(f"- **k={tag}** rf {xb.mean():.3f} | `{a}` {yb.mean():.3f} "
              f"(Δ={d:+.3f}, p={p:.4g}, n={len(com)}) {flag}")

    w("\n## 分实体（same-matrix · auc）\n")
    w("| 实体 | " + " | ".join(
        f"{a}@k{s}" for s in [str(x) for x in shots]
        for a in ("rf", "mf-shot", "mf-net", "mf-common")) + " |")
    w("|---|" + "---|" * (4 * len(shots)) + "---|")
    for e in eids:
        cells = []
        for s in [str(x) for x in shots]:
            for a in ("rf", "mf-shot", "mf-net", "mf-common"):
                d = agg(s, "same-matrix", a, "auc").get(e)
                cells.append(f"{d:.3f}" if d is not None else "—")
        w(f"| {e} | " + " | ".join(cells) + " |")

    with open(os.path.join(OUT, "exp77_p6_v7_shot_direction.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("[exp77] 已写出 out/exp77_p6_v7_shot_direction.md")


if __name__ == "__main__":
    main()
