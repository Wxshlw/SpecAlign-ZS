r"""exp79 —— 先验-数据替代曲线：化学先验**在什么条件下**有用？

动机（用户要求）
--------------
k=5 时本文检出不如 RF/SVM。不能只写"我们输了"，要写成：
    「化学先验的适用区间是 k < k*，k ≥ k* 后数据信息已充分，先验被替代」
这是一个**可量化的科学边界**，而不是借口。

核心设计
--------
1. **k 扫描** k ∈ {0,1,2,3,5,10,20}，两个任务（检出 same-matrix / 识别 old-heldout）
2. **隔离对照 Δ_chem(k)**：只把分类头码本换成同形随机矩阵，stage1 与训练完全不动
       Δ_chem = chem − rand      ⇒ 化学先验的**净增量**
   假设：检出上 Δ_chem(k) 随 k 单调衰减并早早归零；识别上恒为正
3. **混合权重 α\* 扫描**（仅检出，k>0）：
       w(α) = unit( α·w_chem + (1−α)·b̂_data )
   α=1 纯化学先验，α=0 纯数据。α*(k) 单调下降 ⇒ 先验被数据替代的直接证据
4. **交叉点 k\***：本文最佳 vs rf 的差值 Δ_rf(k) 符号翻转处 ⇒ 适用区间的边界
5. **可自诊断的预测因子**：Δ_chem 与 cos(w_chem, b*) 的相关性
   —— 若相关强，则"先验是否有用"可在**见标样之前**预判

信息论解释（写进讨论）
--------------------
- 检出 = 1 bit（有没有掺）⇒ 少量样本即可给足 ⇒ k* 很小
- 识别 = log2(N) bit 且**结构化**（要说清是哪个分子）⇒ 数据难以替代化学轴
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
from exp68_p6_cond_scorer import split_by_group
from exp71_p6_detection_fix import load_groups_cached, pick_lam
from exp74_p6_fairness_audit import load_bg_gid, split_group
from exp75_p6_v5_universal_detector import whiten, metrics, zsc

OUT = os.path.join(os.path.dirname(HERE), "out")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OLD_FRAC = 0.7
SHRI = 0.2
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def unit(v):
    nv = np.linalg.norm(v)
    return v / nv if nv > 1e-12 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lams", default="0.0001,0.001,0.01,0.1,1")
    ap.add_argument("--eids", default="")
    ap.add_argument("--shots", default="0,1,2,5,10,20,50,100")
    ap.add_argument("--proto", default="group")
    ap.add_argument("--nrand", type=int, default=5,
                    help="随机码本抽样次数（单次抽样会碰巧赢）")
    ap.add_argument("--out", default="exp79_subst.json")
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

    res = {"M": {}, "DIAG": {}, "meta": dict(
        shots=shots, alphas=list(ALPHAS), nrand=args.nrand, seeds=args.seeds)}
    print(f"[exp79] device={DEV} 实体={eids} shots={shots} seeds={args.seeds} "
          f"nrand={args.nrand}", flush=True)

    for e_hold in eids:
        others = [e for e in sorted(ent) if e != e_hold]
        Xe = np.asarray(ent[e_hold], float)
        gid_e = GR[e_hold]["gid"]
        mid_h = EID2MID[e_hold]
        S_all = T_all[[ki[e] for e in others]
                      + [ki[f"M_{m}"] for m in mids] + [ki[e_hold]]]
        NEW = len(others) + len(mids)
        iT = len(others)
        # 化学独特性：T 与最相似老类的化学距离（exp64 的 d_chem）
        Sc = T_all[ki[e_hold]]
        Sold = T_all[[ki[e] for e in others]]
        cosc = nrm(Sold) @ nrm(Sc[None])[0]
        d_chem = float(1.0 - cosc.max())

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
                    if e in GR:
                        a, b, _ = split_group(Xo, GR[e]["gid"], rng)
                    else:
                        pp = rng.permutation(len(Xo))
                        cut = max(2, int(round(OLD_FRAC * len(Xo))))
                        a, b = pp[:cut], pp[cut:]
                    old_ref[e] = Xo[b]
                    parts.append(Xo[a]); ys.append(np.full(len(a), j))
                for j, m in enumerate(mids):
                    Xo = np.asarray(bg[m], float)
                    a, b, _ = split_group(Xo, BG_GID[m], rng)
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

                # 净信号目标 ΔZ_j（exp71 修复）
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
                A = head_ridge_R(S_o, Zc, lam1)
                An = head_ridge_R(S_o, dZ, lam1)
                W = (S_all - mu_S) @ A          # 识别头（类中心目标）
                Wn = (S_all - mu_S) @ An        # 检出头（净信号目标）
                w_chem_id, w_chem_det = W[NEW], Wn[NEW]
                Wcand_id = np.concatenate(      # 识别候选集：老类 + T
                    [W[:len(others)], W[NEW:NEW + 1]], 0)

                # 白化协方差
                R = np.concatenate([Zfit[yfit == j] - Zfit[yfit == j].mean(0)
                                    for j in rows], 0)
                Sw = R.T @ R / max(1, len(R))
                tr_ = float(np.trace(Sw)) / Sw.shape[0]
                Sig = (1 - SHRI) * Sw + SHRI * tr_ * np.eye(Sw.shape[0])
                Sig += 1e-6 * np.eye(Sig.shape[0])
                Si = np.linalg.inv(Sig)

                Zn = embed(net, Xte_new)
                # oracle 净信号方向（仅诊断）
                b_star = Zn.mean(0) - embed(
                    net, np.asarray(mid_ref[mid_h], float)).mean(0)
                snr = float(np.linalg.norm(b_star))
                b_star = unit(b_star)
                cos_chem = (float(w_chem_det @ b_star
                                  / (np.linalg.norm(w_chem_det) + 1e-12))
                            if b_star is not None else None)

                # k-shot 数据方向 b̂（浓度加权最小二乘）
                muX_h = np.asarray(mid_tr[mid_h], float).mean(0)
                b_hat = None
                if len(tr_e_ix):
                    cc_all = conc.get(e_hold)
                    ZT = embed(net, Xe[tr_e_ix])
                    DZ = ZT - Zmid_c[mid_h]
                    cc = None
                    if cc_all is not None:
                        ca = np.asarray(cc_all, float)
                        if np.isfinite(ca).all():
                            cc = ca[tr_e_ix] / (np.nanmax(ca) + 1e-9)
                    if cc is not None and (cc ** 2).sum() > 1e-9:
                        a_z = (cc[:, None] * DZ).sum(0) / (cc ** 2).sum()
                    else:
                        a_z = DZ.mean(0)
                    b_hat = unit(a_z)
                cos_shot = (float(b_hat @ b_star)
                            if (b_hat is not None and b_star is not None)
                            else None)

                res["DIAG"].setdefault(tag, {}).setdefault(e_hold, []).append(
                    dict(cos_chem=cos_chem, cos_shot=cos_shot, snr=snr,
                         d_chem=d_chem))

                # ---------------- 外部基线 ----------------
                sc = StandardScaler().fit(Xtr)
                Xtr_s = sc.transform(Xtr)
                rf = RandomForestClassifier(300, random_state=0,
                                            n_jobs=-1).fit(Xtr_s, ytr)
                # SVC 用 decision_function 代替 predict_proba：
                # probability=True 会内部跑 5-fold CV，是主要耗时瓶颈，
                # 而 decision_function 的单调变换对 AUC/TPR@FPR 等价。
                sv = SVC(kernel="rbf", C=10, gamma="scale",
                         random_state=0).fit(Xtr_s, ytr)
                sv_cls = list(sv.classes_)

                # 随机码本：与 S_all 同形、按行归一化，走**完全相同**的
                # (· − μ) @ A 流程 ⇒ 唯一差别是码本内容，隔离干净
                rng_r = np.random.default_rng(seed * 977 + 5)
                W_rand_id, W_rand_det = [], []
                for _ in range(args.nrand):
                    Rm = nrm(rng_r.normal(size=S_all.shape).astype(np.float32))
                    Rc = Rm - Rm[rows].mean(0, keepdims=True)
                    W_rand_id.append((Rc @ A)[NEW])
                    W_rand_det.append((Rc @ An)[NEW])
                W_rand_full = []
                for _ in range(args.nrand):
                    Rm = nrm(rng_r.normal(size=S_all.shape).astype(np.float32))
                    Rc = Rm - Rm[rows].mean(0, keepdims=True)
                    W_rand_full.append(Rc @ A)

                for n_name, Xn_neg in (("same-matrix", mid_ref[mid_h]),
                                       ("old-heldout", Xte_old)):
                    if len(Xn_neg) < 5:
                        continue
                    Zo = embed(net, Xn_neg)
                    Xn_s = sc.transform(Xte_new)
                    Xo_s = sc.transform(Xn_neg)
                    arms = {}

                    if n_name == "same-matrix":
                        Zc_n, Zc_o = Zn - mu_mid, Zo - mu_mid
                        chem = Si @ w_chem_det
                        arms["chem"] = (Zc_n @ chem, Zc_o @ chem)
                        # ⚠️ 随机码本必须**逐个算 AUC 再平均**，不能先平均分数：
                        # 随机方向的期望是零向量，分数先平均会退化成常数，
                        # 人为把 rand 压到远低于 0.5，使 Δ_chem 虚高。
                        for wr in W_rand_det:
                            v = Si @ wr
                            m = metrics(Zc_n @ v, Zc_o @ v)
                            res["M"].setdefault(tag, {}).setdefault(
                                n_name, {}).setdefault("rand", {}).setdefault(
                                e_hold, []).append(m)
                        # 纯数据方向（k>0）
                        if b_hat is not None:
                            v = Si @ b_hat
                            arms["data"] = (Zc_n @ v, Zc_o @ v)
                            for al in ALPHAS:
                                wm = unit(al * unit(w_chem_det)
                                          + (1 - al) * b_hat)
                                if wm is None:
                                    continue
                                v = Si @ wm
                                arms[f"mix{al:.2f}"] = (Zc_n @ v, Zc_o @ v)
                    else:
                        Ln, Lo = Zn @ Wcand_id.T, Zo @ Wcand_id.T
                        arms["chem"] = (Ln[:, iT], Lo[:, iT])
                        arms["chem-m"] = (
                            Ln[:, iT] - np.delete(Ln, iT, 1).max(1),
                            Lo[:, iT] - np.delete(Lo, iT, 1).max(1))
                        for Wr in W_rand_full:
                            Wcr = np.concatenate(
                                [Wr[:len(others)], Wr[NEW:NEW + 1]], 0)
                            Lr, Lro = Zn @ Wcr.T, Zo @ Wcr.T
                            res["M"].setdefault(tag, {}).setdefault(
                                n_name, {}).setdefault("rand", {}).setdefault(
                                e_hold, []).append(
                                metrics(Lr[:, iT], Lro[:, iT]))
                            res["M"].setdefault(tag, {}).setdefault(
                                n_name, {}).setdefault("rand-m", {}).setdefault(
                                e_hold, []).append(metrics(
                                    Lr[:, iT] - np.delete(Lr, iT, 1).max(1),
                                    Lro[:, iT] - np.delete(Lro, iT, 1).max(1)))
                        # 识别任务的 α 扫描：★ 必须**整个候选集同时混合**，
                        # 否则（只换 T 那一行）会造成几何不一致，人为压低 α=0。
                        #   α=1 → 全化学外推头（zs）
                        #   α=0 → 全数据中心头   （exp66 的 cent_z）
                        if NEW in rows:
                            mu_g = Zc.mean(0)
                            cand_r = [r for r in rows if r < len(others)]
                            iTl = len(cand_r)          # T 排在候选末位
                            for al in ALPHAS:
                                Wx = []
                                ok = True
                                for r in cand_r + [NEW]:
                                    uc, ud = unit(W[r]), unit(Zc[r] - mu_g)
                                    if uc is None or ud is None:
                                        ok = False
                                        break
                                    wm = unit(al * uc + (1 - al) * ud)
                                    Wx.append(wm if wm is not None else W[r])
                                if not ok:
                                    continue
                                Wx = np.stack(Wx)
                                Lx, Lxo = Zn @ Wx.T, Zo @ Wx.T
                                arms[f"mix{al:.2f}"] = (Lx[:, iTl], Lxo[:, iTl])
                                arms[f"mix{al:.2f}-m"] = (
                                    Lx[:, iTl] - np.delete(Lx, iTl, 1).max(1),
                                    Lxo[:, iTl] - np.delete(Lxo, iTl, 1).max(1))
                            # α=0（全数据中心）：单列，便于直接对比
                            Wd = np.stack([unit(Zc[r] - mu_g)
                                           for r in cand_r + [NEW]])
                            Ld, Ldo = Zn @ Wd.T, Zo @ Wd.T
                            arms["dat-all"] = (Ld[:, iTl], Ldo[:, iTl])
                            arms["dat-all-m"] = (
                                Ld[:, iTl] - np.delete(Ld, iTl, 1).max(1),
                                Ldo[:, iTl] - np.delete(Ldo, iTl, 1).max(1))

                    # RF / SVM：k=0 无 T 类 → 只能吐 1−maxprob；k>0 → P(T|x)
                    if k > 0:
                        arms["rf"] = (rf.predict_proba(Xn_s)[:, NEW],
                                      rf.predict_proba(Xo_s)[:, NEW])
                        ci = sv_cls.index(NEW) if NEW in sv_cls else None
                        if ci is not None:
                            arms["svm"] = (sv.decision_function(Xn_s)[:, ci],
                                           sv.decision_function(Xo_s)[:, ci])
                    else:
                        arms["rf"] = (1 - rf.predict_proba(Xn_s).max(1),
                                      1 - rf.predict_proba(Xo_s).max(1))
                        arms["svm"] = (-sv.decision_function(Xn_s).max(1),
                                       -sv.decision_function(Xo_s).max(1))

                    for nm, (sn, so) in arms.items():
                        m = metrics(np.asarray(sn, float),
                                    np.asarray(so, float))
                        res["M"].setdefault(tag, {}).setdefault(
                            n_name, {}).setdefault(nm, {}).setdefault(
                            e_hold, []).append(
                            dict(auc=m["auc"], tpr5=m["tpr5"]))

            dd = res["M"][tag].get("same-matrix", {})
            line = (f"k={k:<2} chem={np.mean([x['auc'] for x in dd.get('chem', {}).get(e_hold, [])]):.3f} "
                    if dd.get("chem", {}).get(e_hold) else f"k={k:<2} ")
            print(f"[{e_hold:>7}] {line}", flush=True)

    with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[exp79] 已写出 out/{args.out}")
    write_report(res, eids, shots)


def agg(res, tag, n_name, arm, met="auc", pool=None):
    per = {}
    src = res["M"].get(tag, {}).get(n_name, {}).get(arm, {})
    for e, vs in src.items():
        if pool is not None and e not in pool:
            continue
        if vs:
            per[e] = float(np.mean([x[met] for x in vs]))
    return per


def write_report(res, eids, shots):
    from scipy.stats import ttest_rel, pearsonr
    L = []
    w = L.append
    NEG = ("same-matrix", "old-heldout")
    TASK = {"same-matrix": "检出（掺混 vs 同基质纯谱）",
            "old-heldout": "识别（T vs 其他掺杂物）"}
    CLEAN = ["A01", "A02", "A03", "A04", "A07"]

    w("# exp79 —— 先验-数据替代曲线：化学先验在什么条件下有用？\n")
    w("**核心命题**：化学先验不是「一直有用」，而是**在 k < k\\* 时有用，"
      "k ≥ k\\* 后被数据替代**。\n")
    w("本实验把这条边界量化出来，使 k=5 的落后从「缺陷」变为「可预测的科学界限」。\n")
    w("- `chem`：化学先验码本　`rand`：同形随机码本（stage1/训练完全不动，"
      f"{res['meta']['nrand']} 次抽样取均值）\n")
    w("- `data`：k-shot 真实谱估出的净信号方向 b̂（不含任何化学信息）\n")
    w("- `mixα`：`unit(α·w_chem + (1−α)·b̂)`，α=1 纯先验，α=0 纯数据\n")
    w("- `Δ_chem = chem − rand`：**化学先验的净增量**（隔离对照）\n")
    w("- RF/SVM 在 k=0 时无 T 类，只能吐 `1−maxprob`；k>0 时输出 `P(T|x)`\n")

    # ---------- 一、k 扫描主表 ----------
    for n_name in NEG:
        SUF = "" if n_name == "same-matrix" else "-m"
        MIX = tuple(f"mix{a:.2f}{SUF}" for a in ALPHAS)
        POOL = (("chem", "chem-m", "dat-all", "dat-all-m")
                if n_name == "old-heldout" else ("chem",))
        w(f"\n## 一、k 扫描 —— {n_name}（{TASK[n_name]}）\n")
        arms = (["chem", "chem-m", "dat-all-m", "rand", "rand-m", "data",
                 "rf", "svm"]
                if n_name == "old-heldout"
                else ["chem", "rand", "data"] + [f"mix{a:.2f}" for a in ALPHAS]
                + ["rf", "svm"])
        w("| k | " + " | ".join(f"`{a}`" for a in arms)
          + " | Δ_chem | Δ_vs_rf |")
        w("|---|" + "---|" * len(arms) + "---|---|")
        curve = {}
        for k in shots:
            tag = str(k)
            row, vals = [], {}
            for a in arms:
                d = agg(res, tag, n_name, a)
                if d:
                    vals[a] = d
                    row.append(f"{np.mean(list(d.values())):.3f}")
                else:
                    row.append("—")
            # Δ_chem 用 clean 池
            dc = agg(res, tag, n_name, "chem", pool=CLEAN)
            dr = agg(res, tag, n_name, "rand", pool=CLEAN)
            if not dc:
                dc = agg(res, tag, n_name, "chem-m", pool=CLEAN)
                dr = agg(res, tag, n_name, "rand-m", pool=CLEAN)
            com = sorted(set(dc) & set(dr))
            if len(com) >= 3:
                xc = np.array([dc[e] for e in com])
                xr = np.array([dr[e] for e in com])
                dlt, p = float(np.mean(xc - xr)), ttest_rel(xc, xr).pvalue
                curve[k] = (dlt, p, xc.mean(), xr.mean())
                cell = f"**{dlt:+.3f}** (p={p:.3g})"
            else:
                curve[k] = None
                cell = "—"
            # Δ_vs_rf：本文该 k 下最佳臂
            best, bn = -1, None
            for a in POOL + MIX + (("data",) if n_name == "same-matrix" else ()):
                d = agg(res, tag, n_name, a, pool=CLEAN)
                if d and np.mean(list(d.values())) > best:
                    best, bn = np.mean(list(d.values())), a
            drf = agg(res, tag, n_name, "rf", pool=CLEAN)
            if bn and drf:
                com2 = sorted(set(agg(res, tag, n_name, bn, pool=CLEAN))
                              & set(drf))
                if len(com2) >= 3:
                    xb = np.array([agg(res, tag, n_name, bn, pool=CLEAN)[e]
                                   for e in com2])
                    xf = np.array([drf[e] for e in com2])
                    cell2 = (f"{np.mean(xb - xf):+.3f} (p="
                             f"{ttest_rel(xb, xf).pvalue:.3g}, {bn})")
                else:
                    cell2 = "—"
            else:
                cell2 = "—"
            w(f"| {k} | " + " | ".join(row) + f" | {cell} | {cell2} |")
        res["_curve_" + n_name] = {str(k): v for k, v in curve.items()}

        # ---------- α* 曲线（两个任务都输出，便于对照） ----------
        w("\n### α\\* —— 最优先验/数据混合权重随 k 的变化\n")
        w("α=1 纯化学先验，α=0 纯数据（检出=b̂ 净信号方向；识别=k-shot "
          "类中心方向）。α\\* 下降 = 先验被数据替代。\n")
        w("| k | " + " | ".join(f"α={a:.2f}" for a in ALPHAS)
          + " | α\\* | Δ(α\\*)−Δ(α=1) |")
        w("|---|" + "---|" * len(ALPHAS) + "---|---|")
        for k in shots:
            if k == 0:
                continue
            tag = str(k)
            vs = {}
            for a in ALPHAS:
                d = agg(res, tag, n_name, f"mix{a:.2f}{SUF}", pool=CLEAN)
                if d:
                    vs[a] = float(np.mean(list(d.values())))
            if not vs:
                continue
            bs = max(vs, key=vs.get)
            w(f"| {k} | " + " | ".join(
                f"{vs.get(a, float('nan')):.3f}" for a in ALPHAS)
                + f" | **{bs:.2f}** | {vs[bs] - vs.get(1.0, vs[bs]):+.3f} |")

        # ---------- 交叉点 ----------
        w(f"\n### 交叉点 k\\*（{n_name}）\n")
        w("`Δ_best−rf` 符号翻转处即先验/数据方法的优势交界。\n")
        w("| k | 本文最佳 | rf | Δ | p |")
        w("|---|---|---|---|---|")
        prev = None
        kstar = None
        for k in shots:
            tag = str(k)
            best, bn = -1, None
            for a in POOL + MIX + (("data",) if n_name == "same-matrix" else ()):
                d = agg(res, tag, n_name, a, pool=CLEAN)
                if d and np.mean(list(d.values())) > best:
                    best, bn = np.mean(list(d.values())), a
            drf = agg(res, tag, n_name, "rf", pool=CLEAN)
            if not bn or not drf:
                continue
            com = sorted(set(agg(res, tag, n_name, bn, pool=CLEAN)) & set(drf))
            if len(com) < 3:
                continue
            xb = np.array([agg(res, tag, n_name, bn, pool=CLEAN)[e]
                           for e in com])
            xf = np.array([drf[e] for e in com])
            dlt = float(np.mean(xb - xf))
            w(f"| {k} | {xb.mean():.3f} ({bn}) | {xf.mean():.3f} | "
              f"{dlt:+.3f} | {ttest_rel(xb, xf).pvalue:.3g} |")
            if prev is not None and prev > 0 >= dlt:
                kstar = k
            prev = dlt
        w(f"\n- **k\\* ≈ {kstar if kstar else '未出现（全程领先或全程落后）'}**"
          f"（clean 池 {CLEAN}）\n")

    # ---------- 二、判据 ----------
    w("\n## 二、判据\n")
    for n_name in NEG:
        ks = [k for k in shots if res["_curve_" + n_name].get(str(k))]
        if not ks:
            continue
        d0 = res["_curve_" + n_name].get("0")
        dmax = res["_curve_" + n_name].get(str(max(ks)))
        w(f"\n### {n_name}（{TASK[n_name]}）\n")
        if d0:
            w(f"- **H79a** k=0 时 Δ_chem = **{d0[0]:+.3f}** "
              f"(chem {d0[2]:.3f} vs rand {d0[3]:.3f}, p={d0[1]:.3g}) ⇒ "
              f"{'先验有效' if d0[0] > 0 else '先验无效'}\n")
        if dmax:
            w(f"- **H79b** k={max(ks)} 时 Δ_chem = **{dmax[0]:+.3f}** "
              f"(p={dmax[1]:.3g}) ⇒ "
              f"{'仍有效' if dmax[0] > 0 else '已被数据替代'}\n")
        if d0 and dmax:
            w(f"- **H79c 替代效应**：Δ_chem 从 {d0[0]:+.3f} → {dmax[0]:+.3f}，"
              f"衰减 {d0[0] - dmax[0]:+.3f} ⇒ "
              f"{'成立：样本替代先验' if dmax[0] < d0[0] else '不成立'}\n")

    # ---------- 三、可自诊断的预测因子 ----------
    w("\n## 三、预测因子：能否在见标样之前预判先验是否有用？\n")
    vs_by_k = {}
    for k in shots:
        tag = str(k)
        dc = agg(res, tag, "same-matrix", "chem", pool=CLEAN)
        dr = agg(res, tag, "same-matrix", "rand", pool=CLEAN)
        com = sorted(set(dc) & set(dr))
        dg = res["DIAG"].get(tag, {})
        xs, ys, sn = [], [], []
        for e in com:
            g = dg.get(e, [])
            if not g or g[0].get("cos_chem") is None:
                continue
            xs.append(np.mean([x["cos_chem"] for x in g]))
            ys.append(dc[e] - dr[e])
            sn.append(np.mean([x["snr"] for x in g]))
        if len(xs) >= 4:
            r1, p1 = pearsonr(xs, ys)
            r2, p2 = pearsonr(sn, ys)
            vs_by_k[k] = (r1, p1, r2, p2, len(xs))
            w(f"- k={k}: ρ(cos_chem, Δ_chem)=**{r1:+.3f}** (p={p1:.3g}) | "
              f"ρ(snr, Δ_chem)={r2:+.3f} (p={p2:.3g}) | n={len(xs)}")
    w("")
    w("**读法**：若 ρ(cos_chem, Δ_chem) 显著为正，则「化学先验是否有用」"
      "可由 `cos(w_chem, b*)` 在**拿到标样之前**预判——\n"
      "这使适用条件成为**可操作判据**而非事后解释。\n")

    # ---------- 五、为什么两个任务对化学先验的需求不同？ ----------
    w("\n## 五、为什么「检出」与「识别」对化学先验的需求不同？\n")
    w("这是回答「k=5 不如 RF」的关键：**不是先验失效，是两个任务要的信息"
      "不同**。\n")
    w("| 任务 | 要回答的问题 | 所需信息 | k-shot 数据方向能否替代先验 |")
    w("|---|---|---|---|")
    w("| 检出 | 有没有掺？（1 bit，是/否） | 只要「偏离纯基质」 | "
      "**能**：任一显著偏离方向都可判别 |")
    w("| 识别 | 掺的是**哪个**分子？（log₂N bit，且结构化） | "
      "「偏离**在哪条化学轴**上」 | **不能**：方向必须指名，数据方向无标签 |")
    w("")
    w("### 实证：把先验权重 α 从 1 降到 0，两个任务的反应相反\n")
    w("`Δ(α=1→α=0)` = 纯数据中心 − 纯化学先验。**负值 = 先验不可替代**。\n")
    w("| 任务 | k | α=0（纯数据） | α=1（纯化学） | Δ(α=1→α=0) | α\\* |")
    w("|---|---|---|---|---|---|")
    for n_name in NEG:
        SUF2 = "" if n_name == "same-matrix" else "-m"
        for k in shots:
            if k == 0:
                continue
            tag = str(k)
            v0 = agg(res, tag, n_name, f"mix{0.0:.2f}{SUF2}", pool=CLEAN)
            v1 = agg(res, tag, n_name, f"mix{1.0:.2f}{SUF2}", pool=CLEAN)
            if not v0 or not v1:
                continue
            com = sorted(set(v0) & set(v1))
            if len(com) < 3:
                continue
            x0 = np.array([v0[e] for e in com])
            x1 = np.array([v1[e] for e in com])
            vs = {}
            for a in ALPHAS:
                d = agg(res, tag, n_name, f"mix{a:.2f}{SUF2}", pool=CLEAN)
                if d:
                    vs[a] = float(np.mean(list(d.values())))
            bs = max(vs, key=vs.get) if vs else float("nan")
            w(f"| {n_name} | {k} | {x0.mean():.3f} | {x1.mean():.3f} | "
              f"**{np.mean(x0 - x1):+.3f}** | {bs:.2f} |")
    w("")
    w("**判读**：若检出的 Δ(α=1→α=0) ≈ 0 或为正、而识别的为显著负值，"
      "则「先验被替代」只发生在**检出**这一半——\n")
    w("识别那半先验**始终不可替代**。这正是「双分离」在先验层面的对应物。\n")

    # ---------- 四、诊断量 ----------
    w("\n## 四、方向质量诊断（cos 与 oracle 净信号 b\\*）\n")
    w("| 实体 | k | cos(w_chem, b\\*) | cos(b̂, b\\*) | ‖b\\*‖ | d_chem |")
    w("|---|---|---|---|---|---|")
    for e in eids:
        for k in (0, max(shots)):
            g = res["DIAG"].get(str(k), {}).get(e, [])
            if not g:
                continue
            cc = [x["cos_chem"] for x in g if x.get("cos_chem") is not None]
            cs = [x["cos_shot"] for x in g if x.get("cos_shot") is not None]
            sn = [x["snr"] for x in g]
            dc = [x["d_chem"] for x in g]
            w(f"| {e} | {k} | {np.mean(cc) if cc else float('nan'):+.3f} | "
              f"{np.mean(cs) if cs else float('nan'):+.3f} | "
              f"{np.mean(sn):.3f} | {np.mean(dc):.3f} |")

    with open(os.path.join(OUT, "exp79_p6_substitution.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("[exp79] 已写出 out/exp79_p6_substitution.md")


if __name__ == "__main__":
    main()
