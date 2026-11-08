r"""exp80 —— 非线性化分类头，但**保住零样本能力**

依据（exp79 C1 + exp66 E）
------------------------
- exp66：同一嵌入 z 上，**非线性头 rf_z 0.896 vs 线性头 zs 0.730**（Δ=+0.166, p=0.0029）
- exp79：k≥1 时化学外推头 ≡ 数据中心头（Δ≈0.000）⇒ 差距**与先验无关**，只与头的函数形式有关
⇒ 结论：**改函数形式是唯一剩余空间**

但有一个硬约束
--------------
exp66 的 rf_z 是用 **T 的 k 条标样**训练的闭集头 ⇒ **k=0 时它不存在**。
所以不能简单照搬。必须区分两条路：

  【零样本可用的非线性】 nonlinearity 不依赖 T 的样本
    N1 化学侧核化：核岭回归 S→Z（RBF 核），w_T = k(S_T,·)·α   ← 非线性在「化学→光谱」映射
    N2 光谱侧核化：RFF 特征映射 z→φ(z)，再接线性化学头         ← 非线性在「光谱→特征」
    N3 双侧核化：N1 + N2
  【仅 k>0 可用】 非线性来自 T 的样本所塑形的边界
    N4 rf_z / mlp_z（闭集非线性头，exp66 已证有效，作为**上界参照**）

⚠️ 已失败的两次前车之鉴（必须对照，避免重复踩坑）
    exp68 cond scorer：检出 +0.069(n.s.)，识别 **−0.228 (p=0.008 显著恶化)**
    exp75 UD 通用检出头：**线性 0.715 > 非线性 0.688**（非线性显著更差 p=0.009）
    ⇒ 机制：非线性容量帮「记住已知掺杂物」，损害「泛化到未知」
    ⇒ 本实验的重点判据就是：**N1/N2/N3 在 k=0 上是否也出现同样的泛化损害**
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel

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
from exp75_p6_v5_universal_detector import metrics

OUT = os.path.join(os.path.dirname(HERE), "out")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OLD_FRAC = 0.7
SHRI = 0.2
D_RFF = 256


def unit(v):
    nv = np.linalg.norm(v)
    return v / nv if nv > 1e-12 else None


def med_gamma(X, nsub=400):
    X = np.asarray(X, float)
    ix = np.random.default_rng(0).permutation(len(X))[:min(nsub, len(X))]
    Xs = X[ix]
    d2 = ((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1)
    m = float(np.median(d2[d2 > 0]))
    return 1.0 / (m + 1e-12)


def kr_fit(S_old, Z_old, lam, gamma):
    """核岭回归：Z ≈ K·alpha，K = rbf(S_old, S_old)。"""
    K = rbf_kernel(S_old, S_old, gamma=gamma)
    return np.linalg.solve(K + lam * np.eye(len(K)), Z_old)


def kr_pred(alpha, S_old, St, gamma):
    k = rbf_kernel(np.atleast_2d(St), S_old, gamma=gamma)
    return np.asarray(k @ alpha, float)[0]


def make_rff(d, D, gamma, seed):
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=np.sqrt(2 * gamma), size=(D, d))
    b = rng.uniform(0, 2 * np.pi, size=D)
    return W.astype(np.float32), b.astype(np.float32)


def rff(z, W, b):
    return (np.sqrt(2.0 / len(b)) * np.cos(np.asarray(z, float) @ W.T + b)
            ).astype(np.float32)


def whit(Sig):
    return np.linalg.inv(Sig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lams", default="0.0001,0.001,0.01,0.1,1")
    ap.add_argument("--eids", default="")
    ap.add_argument("--shots", default="0,5,20")
    ap.add_argument("--out", default="exp80_nonlin.json")
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
    print(f"[exp80] device={DEV} 实体={eids} shots={shots} seeds={args.seeds}",
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
                    a, b, _ = (split_group(Xo, GR[e]["gid"], rng) if e in GR
                               else (None, None, None))
                    if a is None:
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
                lam1 = pick_lam(S_o, Zc, lams, net, Xval, yval, rows,
                                mu_S, S_all)
                A = head_ridge_R(S_o, Zc, lam1)
                W = (S_all - mu_S) @ A
                Wcand = np.concatenate([W[:len(others)], W[NEW:NEW + 1]], 0)
                # 净信号目标（exp71 修复，与 exp79 的 `chem` 臂对齐）
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
                An = head_ridge_R(S_o, dZ, lam1)
                Wn = (S_all - mu_S) @ An

                # 白化（z 空间）
                R = np.concatenate([Zfit[yfit == j] - Zfit[yfit == j].mean(0)
                                    for j in rows], 0)
                Sw = R.T @ R / max(1, len(R))
                tr_ = float(np.trace(Sw)) / Sw.shape[0]
                Sig = (1 - SHRI) * Sw + SHRI * tr_ * np.eye(Sw.shape[0]) \
                    + 1e-6 * np.eye(Sw.shape[0])
                Si = np.linalg.inv(Sig)

                Zn = embed(net, Xte_new)
                # ---------- N1 化学侧核化 ----------
                gS = med_gamma(S_o - mu_S)
                alpha = kr_fit(S_o - mu_S, Zc - Zc.mean(0), lam1, gS)
                w_kr = kr_pred(alpha, S_o - mu_S, S_all[NEW] - mu_S[0], gS)
                # ---------- N2 光谱侧 RFF ----------
                gZ = med_gamma(Zfit[:800])
                Wr_, b_ = make_rff(dz, D_RFF, gZ, seed)
                Ph_old = rff(Zc, Wr_, b_)
                A2 = head_ridge_R(S_o, Ph_old, lam1)
                w_rff = (S_all[NEW] - mu_S[0]) @ A2
                # ---------- N3 双侧 ----------
                alpha3 = kr_fit(S_o - mu_S, Ph_old - Ph_old.mean(0), lam1, gS)
                w_krff = kr_pred(alpha3, S_o - mu_S,
                                 S_all[NEW] - mu_S[0], gS)

                # RFF 空间白化
                Phfit = rff(Zfit, Wr_, b_)
                R2 = np.concatenate([Phfit[yfit == j] - Phfit[yfit == j].mean(0)
                                     for j in rows], 0)
                Sw2 = R2.T @ R2 / max(1, len(R2))
                tr2 = float(np.trace(Sw2)) / Sw2.shape[0]
                Sig2 = (1 - SHRI) * Sw2 + SHRI * tr2 * np.eye(Sw2.shape[0]) \
                    + 1e-6 * np.eye(Sw2.shape[0])
                Si2 = np.linalg.inv(Sig2)
                mu_ph = Phfit.mean(0)

                # ---------- 外部基线 ----------
                sc = StandardScaler().fit(Xtr)
                Xtr_s = sc.transform(Xtr)
                rf = RandomForestClassifier(300, random_state=0,
                                            n_jobs=-1).fit(Xtr_s, ytr)
                sv = SVC(kernel="rbf", C=10, gamma="scale",
                         random_state=0).fit(Xtr_s, ytr)
                sv_cls = list(sv.classes_)
                # N4：闭集非线性头（k>0），在**本文嵌入 z** 上训练
                rf_z = mlp_z = None
                if k > 0:
                    rf_z = RandomForestClassifier(300, random_state=0,
                                                  n_jobs=-1).fit(Zfit, yfit)
                    mlp_z = _mlp(Zfit, yfit, dz, nrows, seed)

                for n_name, Xn_neg in (("same-matrix", mid_ref[mid_h]),
                                       ("old-heldout", Xte_old)):
                    if len(Xn_neg) < 5:
                        continue
                    Zo = embed(net, Xn_neg)
                    Xn_s = sc.transform(Xte_new)
                    Xo_s = sc.transform(Xn_neg)
                    arms = {}
                    Pn, Po = rff(Zn, Wr_, b_), rff(Zo, Wr_, b_)

                    if n_name == "same-matrix":
                        Zc_n, Zc_o = Zn - mu_mid, Zo - mu_mid
                        # 基线：线性化学外推
                        arms["zs"] = (Zn @ W[NEW], Zo @ W[NEW])
                        arms["zs-w"] = (Zc_n @ (Si @ W[NEW]),
                                        Zc_o @ (Si @ W[NEW]))
                        arms["zs-net"] = (Zn @ Wn[NEW], Zo @ Wn[NEW])
                        arms["zs-net-w"] = (Zc_n @ (Si @ Wn[NEW]),
                                            Zc_o @ (Si @ Wn[NEW]))
                        # N1 / N2 / N3（各含白化版）
                        arms["kr"] = (Zn @ w_kr, Zo @ w_kr)
                        arms["kr-w"] = (Zc_n @ (Si @ w_kr),
                                        Zc_o @ (Si @ w_kr))
                        arms["rff"] = (Pn @ w_rff, Po @ w_rff)
                        arms["rff-w"] = ((Pn - mu_ph) @ (Si2 @ w_rff),
                                         (Po - mu_ph) @ (Si2 @ w_rff))
                        arms["krff-w"] = ((Pn - mu_ph) @ (Si2 @ w_krff),
                                          (Po - mu_ph) @ (Si2 @ w_krff))
                    else:
                        Ln, Lo = Zn @ Wcand.T, Zo @ Wcand.T
                        arms["zs"] = (Ln[:, iT], Lo[:, iT])
                        arms["zs-m"] = (Ln[:, iT] - np.delete(Ln, iT, 1).max(1),
                                        Lo[:, iT] - np.delete(Lo, iT, 1).max(1))
                        Lk, Lko = Zn @ w_kr, Zo @ w_kr
                        arms["kr"] = (Lk, Lko)
                        Lr, Lro = Pn @ w_rff, Po @ w_rff
                        arms["rff"] = (Lr, Lro)
                        arms["krff"] = (Pn @ w_krff, Po @ w_krff)
                        # 下面 kr/rff/krff 是单列；margin 需整个候选矩阵，单独构造
                        Wk_c = np.concatenate(
                            [np.stack([kr_pred(alpha, S_o - mu_S,
                                               S_all[r] - mu_S[0], gS)
                                       for r in range(len(others))]),
                             w_kr[None]], 0)
                        Lk2, Lko2 = Zn @ Wk_c.T, Zo @ Wk_c.T
                        arms["kr-m"] = (
                            Lk2[:, iT] - np.delete(Lk2, iT, 1).max(1),
                            Lko2[:, iT] - np.delete(Lko2, iT, 1).max(1))
                        Wf_c = (S_all[np.array(list(range(len(others)))
                                               + [NEW])] - mu_S) @ A2
                        Lf2, Lfo2 = Pn @ Wf_c.T, Po @ Wf_c.T
                        arms["rff-m"] = (
                            Lf2[:, iT] - np.delete(Lf2, iT, 1).max(1),
                            Lfo2[:, iT] - np.delete(Lfo2, iT, 1).max(1))
                        Wkf_c = np.stack(
                            [kr_pred(alpha3, S_o - mu_S,
                                     S_all[r] - mu_S[0], gS)
                             for r in range(len(others))] + [w_krff], 0)
                        Lkf2, Lkfo2 = Pn @ Wkf_c.T, Po @ Wkf_c.T
                        arms["krff-m"] = (
                            Lkf2[:, iT] - np.delete(Lkf2, iT, 1).max(1),
                            Lkfo2[:, iT] - np.delete(Lkfo2, iT, 1).max(1))

                    # N4 闭集非线性头（k>0）
                    if k > 0 and rf_z is not None:
                        pn = rf_z.predict_proba(Zn)
                        po = rf_z.predict_proba(Zo)
                        ci = list(rf_z.classes_).index(NEW)
                        arms["rf_z"] = (pn[:, ci], po[:, ci])
                        if mlp_z is not None:
                            arms["mlp_z"] = _mlp_pred(mlp_z, Zn, Zo)
                    # 外部
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
                        sn = np.asarray(sn, float).ravel()
                        so = np.asarray(so, float).ravel()
                        m = metrics(sn, so)
                        res["M"].setdefault(tag, {}).setdefault(
                            n_name, {}).setdefault(nm, {}).setdefault(
                            e_hold, []).append(m)

            dd = res["M"][tag].get("same-matrix", {})
            if dd.get("zs", {}).get(e_hold):
                print(f"[{e_hold:>7}] k={k:<3} "
                      f"zs={np.mean([x['auc'] for x in dd['zs'][e_hold]]):.3f} "
                      f"kr={np.mean([x['auc'] for x in dd.get('kr', {}).get(e_hold, [])]):.3f} "
                      f"rff-w={np.mean([x['auc'] for x in dd.get('rff-w', {}).get(e_hold, [])]):.3f} "
                      f"rf={np.mean([x['auc'] for x in dd.get('rf', {}).get(e_hold, [])]):.3f}",
                      flush=True)

    with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[exp80] 已写出 out/{args.out}")
    write_report(res, eids, shots)


def _mlp(Ztr, ytr, d, nc, seed):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(d, 256), nn.ReLU(),
                      nn.Linear(256, nc)).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Zt = torch.tensor(Ztr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(ytr, dtype=torch.long, device=DEV)
    m.train()
    for _ in range(300):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(m(Zt), yt)
        loss.backward()
        opt.step()
    m.eval()
    return m


@torch.no_grad()
def _mlp_pred(m, Zn, Zo):
    a = torch.softmax(m(torch.tensor(Zn, dtype=torch.float32,
                                     device=DEV)), 1).cpu().numpy()
    b = torch.softmax(m(torch.tensor(Zo, dtype=torch.float32,
                                     device=DEV)), 1).cpu().numpy()
    return a[:, -1], b[:, -1]


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
    from scipy.stats import ttest_rel
    L = []
    w = L.append
    CLEAN = ["A01", "A02", "A03", "A04", "A07"]
    SHOW_D = ["zs", "zs-net", "zs-net-w", "kr", "kr-w", "rff", "rff-w",
              "krff-w", "rf_z", "mlp_z", "rf", "svm"]
    SHOW_I = ["zs", "zs-m", "kr", "kr-m", "rff", "rff-m", "krff", "krff-m",
              "rf_z", "mlp_z", "rf", "svm"]
    w("# exp80 —— 非线性化分类头（保住零样本能力）\n")
    w("**依据**：exp66 非线性头 rf_z 0.896 vs 线性 zs 0.730（+0.166, p=0.0029）；"
      "exp79 证明 k≥1 时化学外推 ≡ 数据中心（Δ≈0）⇒ 差距只与**函数形式**有关。\n")
    w("**硬约束**：rf_z 需 T 的标样 ⇒ k=0 不存在。故分两类：\n")
    w("- `kr`（化学侧核岭回归）/ `rff`（光谱侧 RFF）/ `krff`（双侧）"
      "：**零样本可用**，非线性不依赖 T 的样本\n")
    w("- `rf_z` / `mlp_z`：闭集非线性头，**仅 k>0**，作为上界参照\n")

    for n_name, SHOW in (("same-matrix", SHOW_D), ("old-heldout", SHOW_I)):
        w(f"\n## {n_name} · auc\n")
        w("| k | " + " | ".join(f"`{a}`" for a in SHOW) + " |")
        w("|---|" + "---|" * len(SHOW) + "|")
        for k in shots:
            tag = str(k)
            row = []
            for a in SHOW:
                d = agg(res, tag, n_name, a, pool=CLEAN)
                row.append(f"{np.mean(list(d.values())):.3f}" if d else "—")
            w(f"| {k} | " + " | ".join(row) + " |")

        w(f"\n## {n_name} · tpr@FPR5\n")
        w("| k | " + " | ".join(f"`{a}`" for a in SHOW) + " |")
        w("|---|" + "---|" * len(SHOW) + "|")
        for k in shots:
            tag = str(k)
            row = []
            for a in SHOW:
                d = agg(res, tag, n_name, a, met="tpr5", pool=CLEAN)
                row.append(f"{np.mean(list(d.values())):.3f}" if d else "—")
            w(f"| {k} | " + " | ".join(row) + " |")

    w("\n## 判据\n")
    for n_name in ("same-matrix", "old-heldout"):
        for k in shots:
            tag = str(k)
            # 基线取**最强线性臂**，避免拿弱基线衬托非线性
            base = agg(res, tag, n_name,
                       "zs-net-w" if n_name == "same-matrix" else "zs-m",
                       pool=CLEAN)
            for a in ("kr", "rff", "krff", "kr-w", "rff-w", "krff-w",
                      "kr-m", "rff-m", "krff-m"):
                y = agg(res, tag, n_name, a, pool=CLEAN)
                com = sorted(set(base) & set(y))
                if len(com) < 3:
                    continue
                xb = np.array([base[e] for e in com])
                yb = np.array([y[e] for e in com])
                dlt = float(np.mean(yb - xb))
                p = ttest_rel(yb, xb).pvalue
                flag = ("✅ 提升" if (dlt > 0.02 and p < 0.1)
                        else ("❌ 恶化" if dlt < -0.02 else "≈ 持平"))
                w(f"- **{n_name}** k={k} `{a}` vs 基线：{xb.mean():.3f} → "
                  f"{yb.mean():.3f} (Δ={dlt:+.3f}, p={p:.3g}) {flag}")
        # 与 rf 的最终对阵
        w("")
        for k in shots:
            tag = str(k)
            best, bn = -1, None
            for a in (("zs", "zs-w", "zs-net", "zs-net-w", "kr", "kr-w",
                       "rff", "rff-w", "krff-w")
                      if n_name == "same-matrix"
                      else ("zs-m", "kr-m", "rff-m", "krff-m")):
                d = agg(res, tag, n_name, a, pool=CLEAN)
                if d and np.mean(list(d.values())) > best:
                    best, bn = np.mean(list(d.values())), a
            drf = agg(res, tag, n_name, "rf", pool=CLEAN)
            if bn and drf:
                com = sorted(set(agg(res, tag, n_name, bn, pool=CLEAN))
                             & set(drf))
                if len(com) >= 3:
                    xb = np.array([agg(res, tag, n_name, bn, pool=CLEAN)[e]
                                   for e in com])
                    xf = np.array([drf[e] for e in com])
                    w(f"- **{n_name}** k={k} 本文最佳 `{bn}` {xb.mean():.3f} "
                      f"vs rf {xf.mean():.3f} (Δ={np.mean(xb - xf):+.3f}, "
                      f"p={ttest_rel(xb, xf).pvalue:.3g})")

    w("\n## 结论要点\n")
    w("- 若 `kr`/`rff`/`krff` 在 **k=0** 也提升 ⇒ 零样本非线性可行，直接改进主表。\n")
    w("- 若只在 k>0 提升、k=0 反而下降 ⇒ 印证 exp75 机制："
      "**非线性容量帮记忆、损泛化**，此时应写成科学界限而非硬改。\n")

    with open(os.path.join(OUT, "exp80_p6_nonlinear_head.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("[exp80] 已写出 out/exp80_p6_nonlinear_head.md")


if __name__ == "__main__":
    main()
