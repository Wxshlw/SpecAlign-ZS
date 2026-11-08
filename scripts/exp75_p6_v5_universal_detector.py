"""exp75 —— SpecAlign-ZS v5：通用检出头（UD, Universal Detector）

核心洞察（本轮最重要的方法学发现）：
  **检出任务「这批基质有没有掺东西」不需要知道掺的是谁。**
  现有 zs/net/mf 头全部依赖 `w_T`（T 的化学方向），等于把「识别信息」
  硬塞进「检出判据」——而 exp71 已证 cos(w_T, b*) 仅 +0.157（几乎正交）。

UD 头：用**老类**掺混谱（正）vs 纯基质谱（负）训练一个判别器 g(z)，
  测试时直接对新掺杂物 T 打分。T 的谱一条没见过，k=0 与 k=5 同样可用。
  这正是「跨掺杂物泛化的异常检测」，是 NTS 筛查的真实设定。

对照基线：
  zs            化学外推（v1）
  mf-net        化学净信号 + 白化（v4 最佳）
  mf-common     数据驱动通用方向 + 白化（无化学）
  ud-lr/rf/gb   通用判别器（线性 / 非线性），嵌入空间
  ud-*-w        白化输入后再判别
  simca-q/t2    经典化学计量学（每基质 PCA 正交残差 / Hotelling T²）
  fuse          UD ⊕ 化学匹配滤波（z-score 融合）
外部：rf / svm（k=5 用 P(T|x)，k=0 用 1−maxprob）
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier)
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
from exp74_p6_fairness_audit import load_bg_gid, split_group

OUT = os.path.join(os.path.dirname(HERE), "out")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OLD_FRAC = 0.7
SHRI = 0.2


def whiten(Sigma, X, mu):
    """z_w = (X − mu) Σ^{-1/2}。"""
    lam, U = np.linalg.eigh(Sigma)
    lam = np.clip(lam, 1e-8, None)
    return (X - mu) @ (U @ np.diag(lam ** -0.5) @ U.T)


def metrics(sn, so):
    return dict(auc=round(float(auc_of(sn, so)), 4),
                tpr5=round(float(tpr_at_fpr(sn, so, 0.05)), 4),
                tpr20=round(float(tpr_at_fpr(sn, so, 0.20)), 4))


def zsc(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / (s + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.0001,0.001,0.01,0.1,1")
    ap.add_argument("--npc", type=int, default=8)
    ap.add_argument("--eids", default="")
    ap.add_argument("--shots", default="0,5")
    ap.add_argument("--proto", default="group")
    ap.add_argument("--out", default="exp75_v5.json")
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
    res = {"M": {}}
    print(f"[exp75] device={DEV} 实体={eids} shots={shots} proto={args.proto} "
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

                # 类内协方差（白化用）
                R = np.concatenate([Zfit[yfit == j] - Zfit[yfit == j].mean(0)
                                    for j in rows], 0)
                Sw = R.T @ R / max(1, len(R))
                tr = float(np.trace(Sw)) / Sw.shape[0]
                Sig = (1 - SHRI) * Sw + SHRI * tr * np.eye(Sw.shape[0])
                Sig += 1e-6 * np.eye(Sig.shape[0])

                # ---------- UD：老类掺混(正) vs 纯基质(负) ----------
                Zp, yp = [], []
                for j in rows:
                    if j == NEW:
                        continue                 # 不用 T 的谱，保持零样本
                    Zi = Zfit[yfit == j]
                    Zp.append(Zi)
                    yp.append(np.ones(len(Zi)) if j < len(others)
                              else np.zeros(len(Zi)))
                Zp = np.concatenate(Zp, 0); yp = np.concatenate(yp)
                ud_models = {
                    "ud-lr": LogisticRegression(max_iter=2000, C=1.0).fit(Zp, yp),
                    "ud-rf": RandomForestClassifier(300, random_state=0,
                                                    n_jobs=-1).fit(Zp, yp),
                }
                Zw = whiten(Sig, Zp, mu_mid)
                ud_models["ud-rf-w"] = RandomForestClassifier(
                    300, random_state=0, n_jobs=-1).fit(Zw, yp)
                ud_models["ud-lr-w"] = LogisticRegression(
                    max_iter=2000, C=1.0).fit(Zw, yp)

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

                    # 化学头
                    for hn, Wx in (("zs", W), ("net", Wn)):
                        Wc = np.concatenate([Wx[:len(others)],
                                             Wx[NEW:NEW + 1]], 0)
                        Ln, Lo = Zn @ Wc.T, Zo @ Wc.T
                        arms[hn] = (Ln[:, iT], Lo[:, iT])
                        arms[hn + "-m"] = (
                            Ln[:, iT] - np.delete(Ln, iT, 1).max(1),
                            Lo[:, iT] - np.delete(Lo, iT, 1).max(1))
                        An = np.linalg.solve(Sig, Wc.T).T
                        Fn = (Zn - mu_mid) @ An.T
                        Fo = (Zo - mu_mid) @ An.T
                        arms[f"mf-{hn}"] = (Fn[:, iT], Fo[:, iT])
                    # 数据驱动通用方向 + 白化
                    lr = LogisticRegression(max_iter=2000, C=1.0).fit(Zp, yp)
                    d_com = lr.coef_.ravel()
                    d_com /= (np.linalg.norm(d_com) + 1e-12)
                    dc = np.linalg.solve(Sig, d_com)
                    arms["mf-common"] = ((Zn - mu_mid) @ dc, (Zo - mu_mid) @ dc)

                    # UD 头
                    for nm, mdl in ud_models.items():
                        if nm.endswith("-w"):
                            Zq_n = whiten(Sig, Zn, mu_mid)
                            Zq_o = whiten(Sig, Zo, mu_mid)
                        else:
                            Zq_n, Zq_o = Zn, Zo
                        arms[nm] = (mdl.predict_proba(Zq_n)[:, 1],
                                    mdl.predict_proba(Zq_o)[:, 1])
                    # 融合：UD ⊕ 化学匹配滤波
                    sn_u, so_u = arms["ud-rf"]
                    sn_c, so_c = arms["mf-net"]
                    a = np.concatenate([sn_u, so_u]); b = np.concatenate(
                        [sn_c, so_c])
                    arms["fuse-ud-mfnet"] = (zsc(a)[:len(sn_u)] + zsc(b)[
                        :len(sn_c)],
                        zsc(a)[len(sn_u):] + zsc(b)[len(sn_c):])
                    sn_c2, so_c2 = arms["mf-zs"]
                    b2 = np.concatenate([sn_c2, so_c2])
                    arms["fuse-ud-mfzs"] = (zsc(a)[:len(sn_u)] + zsc(b2)[
                        :len(sn_c2)],
                        zsc(a)[len(sn_u):] + zsc(b2)[len(sn_c2):])

                    # 经典 SIMCA（原始谱空间，每基质 PCA）
                    Xm = np.asarray(mid_tr[mid_h], float)
                    mu_m = Xm.mean(0)
                    Xc0 = Xm - mu_m
                    U_, S_, Vt = np.linalg.svd(Xc0, full_matrices=False)
                    Pc = Vt[:min(args.npc, len(Vt) - 1)]
                    Pfull = Vt
                    def simca(X):
                        D = np.asarray(X, float) - mu_m
                        q = D - (D @ Pc.T) @ Pc
                        Q = np.linalg.norm(q, axis=1)
                        T2 = ((D @ Pc.T) / (S_[:len(Pc)] + 1e-9)) ** 2
                        return Q, T2.sum(1)
                    qn, tn = simca(Xte_new)
                    qo, to = simca(Xn_neg)
                    arms["simca-q"] = (qn, qo)
                    arms["simca-t2"] = (tn, to)
                    arms["simca-qt"] = (zsc(np.concatenate([qn, qo]))[:len(qn)]
                                        + zsc(np.concatenate([tn, to]))[
                                            :len(tn)],
                                        zsc(np.concatenate([qn, qo]))[len(qn):]
                                        + zsc(np.concatenate([tn, to]))[
                                            len(tn):])

                    # 外部基线
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
        for n_name in ("same-matrix", "old-heldout"):
            for arm in ("ud-rf", "mf-net", "rf"):
                d = res["M"].get("5", {}).get(n_name, {}).get(arm, {}).get(
                    e_hold)
                if d:
                    line.append(f"{n_name[:4]}/{arm}="
                                f"{np.mean([x['auc'] for x in d]):.3f}")
        print(f"[{e_hold:>4}] " + " ".join(line), flush=True)

    with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[exp75] 已写出 out/{args.out}")
    write_report(res, eids, shots)


SHOW = ["zs", "zs-m", "mf-zs", "mf-net", "mf-common",
        "ud-lr", "ud-rf", "ud-lr-w", "ud-rf-w",
        "fuse-ud-mfnet", "fuse-ud-mfzs",
        "simca-q", "simca-t2", "simca-qt", "rf", "svm"]


def write_report(res, eids, shots):
    from scipy.stats import ttest_rel
    L = []
    w = L.append
    w("# exp75 —— v5 通用检出头（UD）：检出不需要知道掺的是谁\n")
    w("**动机**：exp71 已证 cos(w_T, b*) 仅 +0.157 ⇒ 化学方向 ≠ 检出方向。\n"
      "而检出任务本身（掺混 vs 纯基质）**与掺杂物身份无关**，\n"
      "完全可以用老类训练、零样本迁移到新掺杂物。\n")
    w("- `ud-*`：老类掺混谱（正）vs 纯基质谱（负）训练的判别器，"
      "**测试实体 T 的谱一条未参与训练**，k=0 与 k=5 同一模型\n"
      "- `ud-*-w`：先把嵌入白化（Σ^{-1/2}）再判别\n"
      "- `simca-*`：经典化学计量学，每基质 PCA 正交残差 Q / Hotelling T²\n"
      "- `fuse-*`：UD 与化学匹配滤波的 z-score 融合\n")

    def agg(tag, n_name, arm, met="auc"):
        per = {}
        for e in eids:
            vs = [x[met] for x in res["M"].get(tag, {}).get(
                n_name, {}).get(arm, {}).get(e, [])]
            if vs:
                per[e] = float(np.mean(vs))
        return per

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

    w("\n## 判据\n")
    for tag in [str(s) for s in shots]:
        base = agg(tag, "same-matrix", "rf", "auc")
        for a in ("zs", "mf-net", "mf-common", "ud-lr", "ud-rf", "ud-rf-w",
                  "fuse-ud-mfnet", "simca-qt"):
            y = agg(tag, "same-matrix", a, "auc")
            com = sorted(set(base) & set(y))
            if len(com) < 3:
                continue
            xb = np.array([base[e] for e in com])
            yb = np.array([y[e] for e in com])
            p = ttest_rel(yb, xb).pvalue
            w(f"- **k={tag} auc** rf {xb.mean():.3f} | `{a}` {yb.mean():.3f} "
              f"(Δ={np.mean(yb - xb):+.3f}, p={p:.4g}, n={len(com)}) "
              f"{'✅ 显著领先' if (np.mean(yb-xb) > 0 and p < 0.05) else ''}"
              f"{'❌ 显著落后' if (np.mean(yb-xb) < 0 and p < 0.05) else ''}")
        base5 = agg(tag, "same-matrix", "rf", "tpr5")
        for a in ("mf-net", "ud-rf", "ud-rf-w", "fuse-ud-mfnet"):
            y = agg(tag, "same-matrix", a, "tpr5")
            com = sorted(set(base5) & set(y))
            if len(com) < 3:
                continue
            xb = np.array([base5[e] for e in com])
            yb = np.array([y[e] for e in com])
            p = ttest_rel(yb, xb).pvalue
            w(f"- **k={tag} tpr5** rf {xb.mean():.3f} | `{a}` {yb.mean():.3f} "
              f"(Δ={np.mean(yb - xb):+.3f}, p={p:.4g})")
    # 化学先验在 UD 融合里还有没有增量
    w("")
    for tag in [str(s) for s in shots]:
        u = agg(tag, "same-matrix", "ud-rf", "auc")
        f = agg(tag, "same-matrix", "fuse-ud-mfnet", "auc")
        com = sorted(set(u) & set(f))
        if len(com) >= 3:
            xu = np.array([u[e] for e in com]); xf = np.array([f[e] for e in com])
            w(f"- **k={tag} 化学增量**（fuse-ud-mfnet − ud-rf）："
              f"{np.mean(xf-xu):+.3f}, p={ttest_rel(xf, xu).pvalue:.4g}, "
              f"n={len(com)}")

    w("\n## 分实体明细（same-matrix · auc）\n")
    w("| 实体 | k=0 rf | k=0 ud-rf | k=0 mf-net | k=5 rf | k=5 ud-rf "
      "| k=5 mf-net |")
    w("|---|---|---|---|---|---|---|")
    for e in eids:
        def g(tag, arm):
            d = agg(tag, "same-matrix", arm, "auc").get(e)
            return f"{d:.3f}" if d is not None else "—"
        w(f"| {e} | {g('0','rf')} | {g('0','ud-rf')} | {g('0','mf-net')} "
          f"| {g(str(shots[-1]),'rf')} | {g(str(shots[-1]),'ud-rf')} "
          f"| {g(str(shots[-1]),'mf-net')} |")

    with open(os.path.join(OUT, "exp75_p6_v5_universal_detector.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("[exp75] 已写出 out/exp75_p6_v5_universal_detector.md")


if __name__ == "__main__":
    main()
