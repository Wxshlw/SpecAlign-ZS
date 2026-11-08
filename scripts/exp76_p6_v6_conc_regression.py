"""exp76 —— v6：浓度回归头（把检出从「二分类」升级为「掺入量估计」）

动机：
  exp75 发现 `ud-lr-w`（线性 + 白化）>> `ud-rf`（非线性）：
  **非线性容量帮助「记住已知掺杂物」，却损害「泛化到未知掺杂物」**。
  这与 exp66「检出需非线性」的结论方向相反，原因是 exp66 的 rf_z 是闭集
  20 类判别（含 T 的 k-shot），本质是记忆；而 UD 必须跨掺杂物泛化。

  但即便线性 UD 也只把「掺混 vs 纯」压成二分类，丢掉了最重要的信息：
  **掺了多少**。而训练数据里每条掺混谱都带浓度标签。

v6 核心：把检出重定义为**浓度回归**
    ĉ = f(z)，f 用老类掺混谱（y = 归一化浓度）+ 纯基质谱（y = 0）训练
    得分 = ĉ（直接就是掺入量估计）
  这是一次真正的语义升级：不只说「掺了没掺」，而是给出筛查最需要的量。
  且**完全零样本**（未用 T 的任何谱），k=0 与 k=5 同一模型。

臂：
  mf-net        v4 化学净信号 + 白化（参照）
  ud-lr-w       v5 线性 UD + 白化（参照）
  ud-lr-raw     原始谱空间线性 UD
  ridge-c       嵌入空间 ridge 浓度回归
  ridge-c-w     白化后 ridge 浓度回归
  plsr-raw      原始谱空间 PLS 浓度回归（化学计量学标准）
  fuse-c-mfnet  ridge-c-w ⊕ mf-net（z-score 融合，化学 + 数据）
  rf / svm      外部基线
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.cross_decomposition import PLSRegression
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.0001,0.001,0.01,0.1,1")
    ap.add_argument("--npls", type=int, default=10)
    ap.add_argument("--eids", default="")
    ap.add_argument("--shots", default="0,5")
    ap.add_argument("--proto", default="group")
    ap.add_argument("--out", default="exp76_v6.json")
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
    print(f"[exp76] device={DEV} 实体={eids} shots={shots} proto={args.proto} "
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
                ent_tr_ix = {}
                for j, e in enumerate(others):
                    Xo = np.asarray(ent[e], float)
                    if args.proto == "group" and e in GR:
                        a, b, _ = split_group(Xo, GR[e]["gid"], rng)
                    else:
                        pp = rng.permutation(len(Xo))
                        cut = max(2, int(round(OLD_FRAC * len(Xo))))
                        a, b = pp[:cut], pp[cut:]
                    old_ref[e] = Xo[b]; ent_tr_ix[e] = a
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
                Wn = (S_all - mu_S) @ head_ridge_R(S_o, dZ, lam1)

                R = np.concatenate([Zfit[yfit == j] - Zfit[yfit == j].mean(0)
                                    for j in rows], 0)
                Sw = R.T @ R / max(1, len(R))
                tr_ = float(np.trace(Sw)) / Sw.shape[0]
                Sig = (1 - SHRI) * Sw + SHRI * tr_ * np.eye(Sw.shape[0])
                Sig += 1e-6 * np.eye(Sig.shape[0])

                # ---------- 回归训练集：老类掺混(浓度) + 纯基质(0) ----------
                Xr, yr = [], []
                for e in others:
                    if e not in ent_tr_ix or conc.get(e) is None:
                        continue
                    ix = ent_tr_ix[e]
                    cc = np.asarray(conc[e], float)[ix]
                    mx = np.nanmax(np.asarray(conc[e], float))
                    if not np.isfinite(mx) or mx <= 0:
                        continue
                    Xr.append(Xe if False else np.asarray(ent[e], float)[ix])
                    yr.append(cc / mx)
                for m in mids:
                    Xr.append(np.asarray(mid_tr[m], float))
                    yr.append(np.zeros(len(mid_tr[m])))
                Xr = np.concatenate(Xr, 0).astype(np.float32)
                yr = np.concatenate(yr).astype(np.float64)
                Zr = embed(net, Xr)
                sc_r = StandardScaler().fit(Xr)
                Xr_s = sc_r.transform(Xr)

                ridge_c = Ridge(alpha=1.0).fit(Zr, yr)
                ridge_cw = Ridge(alpha=1.0).fit(whiten(Sig, Zr, mu_mid), yr)
                npls = min(args.npls, Xr_s.shape[1] - 1)
                pls = PLSRegression(n_components=npls).fit(Xr_s, yr)
                ud_raw = LogisticRegression(max_iter=2000, C=1.0).fit(
                    Xr_s, (yr > 0).astype(int))
                Zw_fit = whiten(Sig, Zr, mu_mid)
                ud_lrw = LogisticRegression(max_iter=2000, C=1.0).fit(
                    Zw_fit, (yr > 0).astype(int))

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
                    # 化学头（v4）
                    Wc = np.concatenate([Wn[:len(others)], Wn[NEW:NEW + 1]], 0)
                    An = np.linalg.solve(Sig, Wc.T).T
                    arms["mf-net"] = ((Zn - mu_mid) @ An.T)[:, iT], \
                                     ((Zo - mu_mid) @ An.T)[:, iT]
                    # v5 线性 UD
                    Zq_n = whiten(Sig, Zn, mu_mid)
                    Zq_o = whiten(Sig, Zo, mu_mid)
                    arms["ud-lr-w"] = (ud_lrw.predict_proba(Zq_n)[:, 1],
                                       ud_lrw.predict_proba(Zq_o)[:, 1])
                    arms["ud-lr-raw"] = (
                        ud_raw.predict_proba(sc_r.transform(Xte_new))[:, 1],
                        ud_raw.predict_proba(sc_r.transform(Xn_neg))[:, 1])
                    # v6 浓度回归
                    arms["ridge-c"] = (ridge_c.predict(Zn), ridge_c.predict(Zo))
                    arms["ridge-c-w"] = (ridge_cw.predict(Zq_n),
                                         ridge_cw.predict(Zq_o))
                    arms["plsr-raw"] = (
                        pls.predict(sc_r.transform(Xte_new)).ravel(),
                        pls.predict(sc_r.transform(Xn_neg)).ravel())
                    # 融合：浓度回归 ⊕ 化学匹配滤波
                    a1 = np.concatenate([arms["ridge-c-w"][0],
                                         arms["ridge-c-w"][1]])
                    a2 = np.concatenate([arms["mf-net"][0], arms["mf-net"][1]])
                    nz = len(arms["ridge-c-w"][0])
                    arms["fuse-c-mfnet"] = (zsc(a1)[:nz] + zsc(a2)[:nz],
                                            zsc(a1)[nz:] + zsc(a2)[nz:])
                    a3 = np.concatenate([arms["ud-lr-w"][0],
                                         arms["ud-lr-w"][1]])
                    arms["fuse-c-ud"] = (zsc(a1)[:nz] + zsc(a3)[:nz],
                                         zsc(a1)[nz:] + zsc(a3)[nz:])
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
        for arm in ("ridge-c-w", "mf-net", "rf"):
            d = res["M"].get("5", {}).get("same-matrix", {}).get(
                arm, {}).get(e_hold)
            if d:
                line.append(f"{arm}={np.mean([x['auc'] for x in d]):.3f}")
        print(f"[{e_hold:>4}] " + " ".join(line), flush=True)

    with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[exp76] 已写出 out/{args.out}")
    write_report(res, eids, shots)


SHOW = ["mf-net", "ud-lr-w", "ud-lr-raw", "ridge-c", "ridge-c-w", "plsr-raw",
        "fuse-c-mfnet", "fuse-c-ud", "rf", "svm"]


def write_report(res, eids, shots):
    from scipy.stats import ttest_rel
    L = []
    w = L.append
    w("# exp76 —— v6：把检出从「二分类」升级为「掺入量估计」\n")
    w("训练目标不再是「掺混 vs 纯」的 0/1，而是**归一化掺入浓度** "
      "（老类掺混谱 y=ĉ∈(0,1]，纯基质谱 y=0）。\n"
      "得分 = 预测浓度 ĉ。完全零样本（未用 T 的任何谱），k=0 与 k=5 同模型。\n")

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

    w("\n## 判据（same-matrix · auc，vs rf）\n")
    for tag in [str(s) for s in shots]:
        base = agg(tag, "same-matrix", "rf", "auc")
        for a in ("mf-net", "ud-lr-w", "ridge-c", "ridge-c-w", "plsr-raw",
                  "fuse-c-mfnet", "fuse-c-ud"):
            y = agg(tag, "same-matrix", a, "auc")
            com = sorted(set(base) & set(y))
            if len(com) < 3:
                continue
            xb = np.array([base[e] for e in com])
            yb = np.array([y[e] for e in com])
            p = ttest_rel(yb, xb).pvalue
            d = np.mean(yb - xb)
            flag = "✅显著领先" if (d > 0 and p < 0.05) else (
                "❌显著落后" if (d < 0 and p < 0.05) else "持平(n.s.)")
            w(f"- **k={tag}** rf {xb.mean():.3f} | `{a}` {yb.mean():.3f} "
              f"(Δ={d:+.3f}, p={p:.4g}, n={len(com)}) {flag}")
    w("")
    for tag in [str(s) for s in shots]:
        c = agg(tag, "same-matrix", "ridge-c-w", "auc")
        f = agg(tag, "same-matrix", "fuse-c-mfnet", "auc")
        com = sorted(set(c) & set(f))
        if len(com) >= 3:
            xc = np.array([c[e] for e in com]); xf = np.array([f[e] for e in com])
            w(f"- **k={tag} 化学增量**（fuse-c-mfnet − ridge-c-w）："
              f"{np.mean(xf - xc):+.3f}, p={ttest_rel(xf, xc).pvalue:.4g}")

    w("\n## 分实体（same-matrix · auc）\n")
    w("| 实体 | " + " | ".join(
        f"{a}@k{s}" for s in [str(x) for x in shots]
        for a in ("rf", "ridge-c-w", "mf-net")) + " |")
    w("|---|" + "---|" * (2 * len(shots)) + "---|")
    for e in eids:
        cells = []
        for s in [str(x) for x in shots]:
            for a in ("rf", "ridge-c-w", "mf-net"):
                d = agg(s, "same-matrix", a, "auc").get(e)
                cells.append(f"{d:.3f}" if d is not None else "—")
        w(f"| {e} | " + " | ".join(cells) + " |")

    with open(os.path.join(OUT, "exp76_p6_v6_conc_regression.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("[exp76] 已写出 out/exp76_p6_v6_conc_regression.md")


if __name__ == "__main__":
    main()
