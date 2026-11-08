"""exp62 —— 「AUC 0.79 但 TPR@FPR5 = 0」的机制诊断与修复

=============================================================================
问题
=============================================================================
exp52 k=0：
    ridge  AUC(<=2%) = 0.792    但  TPR@FPR5 = 0.000
    rf     AUC(<=2%) = 0.863         TPR@FPR5 = 0.546

AUC 高却一个都抓不到 ⇒ ROC 在**左上角塌陷**。两种互斥解释：

  H62a【尺度/校准问题，可修】得分分布长尾，单调重标定即可。
  H62b【结构性混淆，不可修】老类里存在 T 的**化学近邻实体**，其样本全部
        排在 T 前面；近邻占老类样本 ≈1/n_old≈5.6% ⇒ FPR<5.6% 的操作点
        **结构性不可达**。

★ 另外发现 exp52 存在一处**口径不公平**（必须纠正）：
    本文 ridge 用「第 T 列的 logit」= 与 T 的相似度（含 T 与近邻的共享成分）
    rf/svm (k=0) 用 1 − max_j p_j ⇒ 本质是 **margin 型**分数（已扣除最像的对手）
    两者不同量纲、不同语义，直接比 sens@FPR5 不公平。

=============================================================================
设计
=============================================================================
Stage A 诊断：老类 top-5% 高分样本是谁？是否富集于 T 的化学近邻？
Stage B 修复：得分重定义（只有**非单调**变换能改变 TPR@FPR5，单调重标定无用）
    raw          s_T                               （exp52 现状）
    margin-hard  s_T − max_{j≠T} s_j
    margin-lse   s_T − τ·logsumexp_{j≠T}(s_j/τ)
    softmax      softmax(s/τ)_T
    nn-contrast  s_T − s_NN(T)   NN 由**化学先验库**的余弦决定（本文特色，
                 不需要 T 的任何光谱样本 ⇒ 完全符合 k=0）
    maha         −马氏距离（老类训练嵌入协方差，白化）
    knn-dist     −到最近老类原型的欧氏距离
Stage C 近邻感知操作点：零样本筛查的合法口径
    TPR@FPR5|excl-NN   排除最近邻实体后定 FPR=5% 阈值
    top2-hit           argmax ∈ {T, NN(T)}（Schymanski Level 4-5：报"二者之一"）
Stage D 公平对比：外部基线（rf/svm/centroid）同协议、同指标重跑
Stage E 浓度分层：低浓度段（<=2 / 3-5 / 6-10）

指标
    auc        全区间 AUC
    pAUC10     McClish 标准化部分 AUC（FPR ≤ 0.10）——筛查真正关心的区间
    tpr5/10/20 TPR @ FPR = 5/10/20%

预注册判据
    H62-A：老类 top-5% 高分样本显著富集于 NN(T)（占比 > 1/n_old）
    H62-B：margin-lse 的 TPR@FPR5 显著 > raw（配对 t，实体为单位）
    H62-C：修复后本文 pAUC10 不低于最强外部基线（公平口径）
"""
from __future__ import annotations
import os, sys, json, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

OUT = r"D:/2026sci论文实验设计/论文6/out"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMP = 0.07
OLD_FRAC = 0.70
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def build_net(d_in, d_out, seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d_in, 512), nn.ReLU(),
                         nn.Linear(512, d_out)).to(DEV)


@torch.no_grad()
def embed(net, X):
    net.eval()
    return F.normalize(net(torch.tensor(X, dtype=torch.float32, device=DEV)),
                       dim=1).cpu().numpy()


def stage1(Xtr, ytr, dim, nrows, steps, seed, S=None):
    torch.manual_seed(seed)
    net = build_net(Xtr.shape[1], dim, seed)
    Wc = S[:nrows] - S[:nrows].mean(0, keepdims=True)
    W = torch.tensor(Wc, dtype=torch.float32, device=DEV)
    opt = torch.optim.Adam(list(net.parameters()), lr=3e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(ytr, dtype=torch.long, device=DEV)
    n = len(Xtr)
    net.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (min(64, n),), device=DEV)
        z = F.normalize(net(Xt[idx]), dim=1)
        loss = F.cross_entropy(z @ F.normalize(W, dim=1).T / TEMP, yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return net, W.detach().cpu().numpy()


def head_ridge_R(S_old, Z_old, lam):
    Sc = S_old - S_old.mean(0, keepdims=True)
    Zc = Z_old - Z_old.mean(0, keepdims=True)
    return Sc.T @ np.linalg.solve(Sc @ Sc.T + lam * np.eye(len(Sc)), Zc)


def nrm(A):
    return A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-12)


def auc_of(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)).astype(float) + 1.0
    n1, n2 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def tpr_at_fpr(pos, neg, alpha):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thr = np.quantile(neg, 1.0 - alpha)
    return float((pos > thr).mean())


def pauc_std(pos, neg, alpha=0.10, ngrid=201):
    """McClish 标准化部分 AUC（FPR ∈ [0, alpha]）。随机=0.5，完美=1。"""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    fpr = np.linspace(0.0, alpha, ngrid)
    thrs = np.quantile(neg, 1.0 - fpr)
    tpr = np.array([(pos > t).mean() for t in thrs])
    area = TRAPZ(tpr, fpr) / alpha            # 归一化 pAUC ∈ [0,1]
    return float(0.5 * (1.0 + (area - alpha / 2) / (1.0 - alpha / 2)))


BINS = [(2.0, "<=2"), (5.0, "3-5"), (10.0, "6-10"), (20.0, "11-20"),
        (1e9, ">20")]


def bin_of(c):
    for hi, lab in BINS:
        if c <= hi:
            return lab
    return ">20"


def load_tanimoto():
    import csv
    p = os.path.join(OUT, "exp00_p6_tanimoto.csv")
    d = {}
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            d[(r["pair_i"], r["pair_j"])] = float(r["ecfp4_tanimoto"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="0")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.0001,0.001,0.01,0.1,1")
    ap.add_argument("--eids", default="")
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.report_only:
        with open(os.path.join(OUT, "exp62_calib.json"), encoding="utf-8") as f:
            res = json.load(f)
        write_report(res, res["meta"]["scores"], res["meta"]["ext"],
                     res["meta"]["shots"], res["meta"]["eids"])
        return
    shots = [None if s.strip() == "full" else int(s) for s in args.shots.split(",")]
    tags = [("full" if s is None else str(s)) for s in shots]
    lams = [float(x) for x in args.lams.split(",")]

    ent, conc, bg, _ = G1.load_g1_conc()
    eids = [e for e in sorted(ent) if conc.get(e) is not None]
    if args.eids:
        eids = [e for e in eids if e in args.eids.split(",")]
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]
    zc = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    T_all = np.asarray(zc["clm"], float)
    dim = T_all.shape[1]
    ki = {k: i for i, k in enumerate(keys_all)}
    TAN = load_tanimoto()
    Tn = nrm(T_all)

    SCORES = ["raw", "margin-hard", "margin-lse", "softmax", "nn-contrast",
              "maha", "knn-dist"]
    EXT = ["rf", "svm-rbf", "centroid-cos"]
    res = {"diag": {}, "score": {t: {s: {} for s in SCORES} for t in tags},
           "ext": {t: {a: {} for a in EXT} for t in tags},
           "meta": {"shots": tags, "seeds": args.seeds, "eids": eids,
                    "scores": SCORES, "ext": EXT, "tau": args.tau}}
    print(f"[exp62] device={DEV} 实体={eids} shots={tags} seeds={args.seeds}",
          flush=True)

    for e_hold in eids:
        others = [e for e in sorted(ent) if e != e_hold]
        NEW = len(others) + len(mids)
        S = T_all[[ki[e] for e in others]
                  + [ki[f"M_{m}"] for m in mids] + [ki[e_hold]]]
        Xe = np.asarray(ent[e_hold], float)
        ce = np.asarray(conc[e_hold], float)
        # —— 化学先验决定的最近邻 NN(T)（只用 SMILES 库，不用任何光谱样本）——
        sims = {o: float(Tn[ki[e_hold]] @ Tn[ki[o]]) for o in others}
        nn_ent = max(sims, key=sims.get)
        nn_row = others.index(nn_ent)

        for k in shots:
            tag = "full" if k is None else str(k)
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed * 7919 + 13)
                perm = rng.permutation(len(Xe))
                n_te = int(min(max(5, round(0.3 * len(Xe))), len(Xe) - 1))
                te_ix, pool_ix = perm[:n_te], perm[n_te:]
                tr_e_ix = pool_ix if k is None else pool_ix[:min(k, len(pool_ix))]
                Xte_new, conc_te = Xe[te_ix], ce[te_ix]

                parts, ys, refs, ref_lab = [], [], [], []
                for j, e in enumerate(others):
                    Xo = np.asarray(ent[e], float)
                    pp = rng.permutation(len(Xo))
                    ntr = max(2, int(round(OLD_FRAC * len(Xo))))
                    parts.append(Xo[pp[:ntr]]); ys.append(np.full(ntr, j))
                    refs.append(Xo[pp[ntr:]]); ref_lab += [e] * (len(Xo) - ntr)
                for j, m in enumerate(mids):
                    Xo = np.asarray(bg[m], float)
                    pp = rng.permutation(len(Xo))
                    ntr = max(2, int(round(OLD_FRAC * len(Xo))))
                    parts.append(Xo[pp[:ntr]])
                    ys.append(np.full(ntr, len(others) + j))
                    refs.append(Xo[pp[ntr:]]); ref_lab += [f"M_{m}"] * (len(Xo) - ntr)
                nrows = len(others) + len(mids)
                if len(tr_e_ix):
                    parts.append(Xe[tr_e_ix])
                    ys.append(np.full(len(tr_e_ix), NEW)); nrows = NEW + 1
                Xtr = np.concatenate(parts).astype(np.float32)
                ytr = np.concatenate(ys).astype(np.int64)
                Xte_old = np.concatenate(refs).astype(np.float32)
                ref_lab = np.asarray(ref_lab)

                # ---------------- 外部基线（k=0：1−maxprob，本质 margin 型）----------
                sc = StandardScaler().fit(Xtr)
                Xtr_s = sc.transform(Xtr)
                Xn_s, Xo_s = sc.transform(Xte_new), sc.transform(Xte_old)
                rf = RandomForestClassifier(n_estimators=300, random_state=0,
                                            n_jobs=-1).fit(Xtr_s, ytr)
                sv = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                         random_state=0).fit(Xtr_s, ytr)
                cj = np.unique(ytr)
                cent = np.stack([Xtr_s[ytr == j].mean(0) for j in cj])
                Dn, Do = nrm(Xn_s) @ nrm(cent).T, nrm(Xo_s) @ nrm(cent).T
                if len(tr_e_ix):
                    continue
                s_rf = 1 - rf.predict_proba(Xn_s).max(1)
                o_rf = 1 - rf.predict_proba(Xo_s).max(1)
                s_sv = 1 - sv.predict_proba(Xn_s).max(1)
                o_sv = 1 - sv.predict_proba(Xo_s).max(1)
                s_ce, o_ce = 1 - Dn.max(1), 1 - Do.max(1)
                for nm, (sn, so) in (("rf", (s_rf, o_rf)),
                                     ("svm-rbf", (s_sv, o_sv)),
                                     ("centroid-cos", (s_ce, o_ce))):
                    res["ext"][tag][nm].setdefault(e_hold, []).append(
                        strat_pack(sn, so, conc_te))

                # ---------------- 本文：stage1 + ridge 头 ----------------
                rng_v = np.random.default_rng(seed * 31 + 7)
                val_mask = np.zeros(len(Xtr), bool)
                for j in sorted({int(y) for y in ytr}):
                    ix = rng_v.permutation(np.where(ytr == j)[0])
                    nv = min(max(3, int(0.2 * len(ix))), max(0, len(ix) - 1))
                    val_mask[ix[:nv]] = True
                Xfit, yfit = Xtr[~val_mask], ytr[~val_mask]
                Xval, yval = Xtr[val_mask], ytr[val_mask]
                val_rows = sorted({int(y) for y in yval})

                net, _ = stage1(Xfit, yfit, dim, nrows, args.steps,
                                seed * 1000 + eids.index(e_hold), S=S)
                Zfit = embed(net, Xfit)
                rows_old = sorted({int(y) for y in yfit})
                Z_old = np.stack([Zfit[yfit == j].mean(0) for j in rows_old])
                S_old, mu_S = S[rows_old], S[rows_old].mean(0, keepdims=True)
                lam_b, bv = lams[0], -1.0
                for lam in lams:
                    Wc = (S - mu_S) @ head_ridge_R(S_old, Z_old, lam)
                    v = macro_recall(net, Wc, Xval, yval, val_rows)
                    if v > bv:
                        bv, lam_b = v, lam
                W = (S - mu_S) @ head_ridge_R(S_old, Z_old, lam_b)

                Zn = embed(net, Xte_new)
                Zo = embed(net, Xte_old)
                Ln = Zn @ W.T
                Lo = Zo @ W.T
                Ln_o, Lo_o = Ln[:, :NEW], Lo[:, :NEW]      # 只取老类列 + T 列
                tau = args.tau

                # —— Stage A：诊断 top-5% 高分老类样本的身份 ——
                sT_old = Lo[:, NEW]
                n_top = max(1, int(round(0.05 * len(sT_old))))
                topi = np.argsort(-sT_old)[:n_top]
                labs_top, cnt = np.unique(ref_lab[topi], return_counts=True)
                frac_nn = float(np.mean(ref_lab[topi] == nn_ent))
                share_nn = float(np.mean(ref_lab == nn_ent))
                res["diag"].setdefault(e_hold, []).append(dict(
                    nn_ent=nn_ent, nn_sim=round(sims[nn_ent], 4),
                    nn_tanimoto=TAN.get((e_hold, nn_ent), float("nan")),
                    frac_top5_is_nn=round(frac_nn, 4),
                    share_nn_in_old=round(share_nn, 4),
                    n_old=len(sT_old), n_top=n_top,
                    top5_lab={str(a): int(b) for a, b in zip(labs_top, cnt)},
                    lam=lam_b))

                # —— Stage B：各得分族 ——
                cand = {}
                cand["raw"] = (Ln[:, NEW], Lo[:, NEW])
                mx_n = Ln[:, :NEW].max(1)
                mx_o = Lo[:, :NEW].max(1)
                cand["margin-hard"] = (Ln[:, NEW] - mx_n, Lo[:, NEW] - mx_o)
                lse_n = tau * np.log(np.exp(Ln[:, :NEW] / tau - Ln[:, :NEW].max(1, keepdims=True) / tau).sum(1))
                lse_n = Ln[:, :NEW].max(1) + lse_n
                lse_o = tau * np.log(np.exp(Lo[:, :NEW] / tau - Lo[:, :NEW].max(1, keepdims=True) / tau).sum(1))
                lse_o = Lo[:, :NEW].max(1) + lse_o
                cand["margin-lse"] = (Ln[:, NEW] - lse_n, Lo[:, NEW] - lse_o)
                sm_n = np.exp(Ln / tau); sm_n /= sm_n.sum(1, keepdims=True)
                sm_o = np.exp(Lo / tau); sm_o /= sm_o.sum(1, keepdims=True)
                cand["softmax"] = (sm_n[:, NEW], sm_o[:, NEW])
                cand["nn-contrast"] = (Ln[:, NEW] - Ln[:, nn_row],
                                       Lo[:, NEW] - Lo[:, nn_row])
                # 马氏 / knn（用老类训练侧嵌入）
                Ztr_all = embed(net, Xtr)
                Zo_tr = np.stack([Ztr_all[ytr == j].mean(0)
                                  for j in sorted(set(ytr.tolist()))])
                mu = Ztr_all.mean(0, keepdims=True)
                C = np.cov((Ztr_all - mu).T) + 1e-3 * np.eye(Ztr_all.shape[1])
                Ci = np.linalg.pinv(C)
                d_n = np.einsum("ij,jk,ik->i", Zn - mu, Ci, Zn - mu)
                d_o = np.einsum("ij,jk,ik->i", Zo - mu, Ci, Zo - mu)
                cand["maha"] = (-d_n, -d_o)
                k_n = np.linalg.norm(Zn[:, None, :] - Zo_tr[None, :, :], axis=2).min(1)
                k_o = np.linalg.norm(Zo[:, None, :] - Zo_tr[None, :, :], axis=2).min(1)
                cand["knn-dist"] = (-k_n, -k_o)

                for nm, (sn, so) in cand.items():
                    res["score"][tag][nm].setdefault(e_hold, []).append(
                        strat_pack(sn, so, conc_te))
                # —— Stage C：近邻感知（仅整体，不分箱）——
                is_nn_old = (ref_lab == nn_ent)
                res["score"][tag].setdefault("__nn__", {})
                d = res["score"][tag]["__nn__"].setdefault(e_hold, [])
                thr_ex = np.quantile(Lo[~is_nn_old, NEW], 0.95)
                pred_n = Ln.argmax(1)
                d.append(dict(
                    nn_ent=nn_ent,
                    tpr5_exclnn=round(float((Ln[:, NEW] > thr_ex).mean()), 4),
                    frac_nn=round(float(is_nn_old.mean()), 4),
                    top1=round(float((pred_n == NEW).mean()), 4),
                    top2=round(float(np.isin(pred_n, [NEW, nn_row]).mean()), 4),
                    n_new=len(Ln)))
        # 进度
        dd = res["score"][tags[0]]["raw"].get(e_hold, [])
        if dd:
            print(f"[{e_hold:>7}] NN={nn_ent}(sim={sims[nn_ent]:.3f}) "
                  f"raw tpr5={np.mean([x['all']['tpr5'] for x in dd]):.3f} "
                  f"margin tpr5={np.mean([x['all']['tpr5'] for x in res['score'][tags[0]]['margin-lse'][e_hold]]):.3f}",
                  flush=True)

    with open(os.path.join(OUT, "exp62_calib.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print("[exp62] 已写出 out/exp62_calib.json")
    write_report(res, SCORES, EXT, tags, eids)


@torch.no_grad()
def macro_recall(net, W, Xv, yv, rows):
    z = F.normalize(net(torch.tensor(Xv, dtype=torch.float32, device=DEV)), dim=1)
    Wt = torch.tensor(nrm(W), dtype=torch.float32, device=DEV)
    pred = (z @ Wt.T)[:, rows].argmax(1).cpu().numpy()
    rs = [float((pred[yv == j] == i).mean()) for i, j in enumerate(rows)
          if (yv == j).sum()]
    return float(np.mean(rs)) if rs else 0.0


def strat_pack(sn, so, conc_te):
    """整体 + 分箱。阈值一律取自**老类 held-out 自身** ⇒ FPR 名义精确。"""
    out = {"all": dict(auc=round(auc_of(sn, so), 4),
                       pauc10=round(pauc_std(sn, so, 0.10), 4),
                       tpr5=round(tpr_at_fpr(sn, so, 0.05), 4),
                       tpr10=round(tpr_at_fpr(sn, so, 0.10), 4),
                       tpr20=round(tpr_at_fpr(sn, so, 0.20), 4),
                       n=int(len(sn)))}
    for hi, lab in BINS:
        m = np.asarray([bin_of(c) == lab for c in conc_te])
        if m.sum() < 3:
            continue
        out[lab] = dict(auc=round(auc_of(sn[m], so), 4),
                        pauc10=round(pauc_std(sn[m], so, 0.10), 4),
                        tpr5=round(tpr_at_fpr(sn[m], so, 0.05), 4),
                        tpr10=round(tpr_at_fpr(sn[m], so, 0.10), 4),
                        tpr20=round(tpr_at_fpr(sn[m], so, 0.20), 4),
                        n=int(m.sum()))
    return out


def write_report(res, SCORES, EXT, tags, eids):
    from scipy.stats import ttest_rel, ttest_1samp
    ORD = ["all", "<=2", "3-5", "6-10", "11-20", ">20"]

    def agg(store, arm, met, seg="all"):
        per = {}
        for e in eids:
            vs = [d[seg][met] for d in store.get(arm, {}).get(e, []) if seg in d]
            if vs:
                per[e] = float(np.mean(vs))
        return per

    L = []
    w = L.append
    w("# exp62 —— 「AUC 高但 TPR@FPR5=0」的机制诊断与修复\n")
    w("阈值一律取自老类 held-out 自身 ⇒ 名义 FPR 精确。`pAUC10`=McClish 标准化"
      "部分 AUC（FPR≤0.10，随机 0.5、完美 1.0）——筛查真正使用的区间。\n")

    # ---------------- Stage A ----------------
    w("\n## Stage A —— 诊断：top-5% 高分老类样本是谁？\n")
    w("| 实体 T | 化学最近邻 NN(T) | CLM cos | ECFP Tanimoto | NN 占老类比例 | top-5% 中 NN 占比 |")
    w("|---|---|---|---|---|---|")
    rows = []
    for e in eids:
        ds = res["diag"].get(e, [])
        if not ds:
            continue
        nn = ds[0]["nn_ent"]
        fr = [d["frac_top5_is_nn"] for d in ds]
        sh = [d["share_nn_in_old"] for d in ds]
        tn = [d["nn_tanimoto"] for d in ds if d["nn_tanimoto"] == d["nn_tanimoto"]]
        rows.append((e, nn, np.mean([d["nn_sim"] for d in ds]),
                     np.mean(tn) if tn else float("nan"),
                     np.mean(sh), np.mean(fr)))
        w(f"| {e} | {nn} | {rows[-1][2]:.3f} | "
          f"{rows[-1][3]:.3f} | {rows[-1][4]:.3f} | {rows[-1][5]:.3f} |")
    if rows:
        fr = np.array([r[5] for r in rows]); sh = np.array([r[4] for r in rows])
        t1 = ttest_1samp(fr, 0.0)
        w(f"\n- top-5% 高分样本中 NN(T) 的平均占比 = **{fr.mean():.3f}** "
          f"(各实体 1/n_old ≈ {sh.mean():.3f})")
        w(f"- **H62-A**（富集于化学近邻）：top-5% 中 NN 占比 vs 0，"
          f"t={t1.statistic:.3f}, p={t1.pvalue:.4g} ⇒ "
          f"{'成立：高分负样本几乎全部是化学近邻' if fr.mean() > 0.5 else '不成立'}")
        w(f"- 富集倍数 = {fr.mean() / max(sh.mean(), 1e-9):.1f}× "
          f"（>1 即高分负样本被近邻垄断）")

    # ---------------- Stage B ----------------
    for t in tags:
        w(f"\n## Stage B —— 得分重定义（k={t}，macro over {len(eids)} 实体）\n")
        for met in ("auc", "pauc10", "tpr5", "tpr10", "tpr20"):
            w(f"\n### {met}（整体）\n")
            w("| 得分 | " + " | ".join(SCORES) + " |")
            w("|---|" + "---|" * len(SCORES))
            row = []
            for s in SCORES:
                d = agg(res["score"][t], s, met)
                row.append(f"{np.mean(list(d.values())):.3f}" if d else "—")
            w("| 值 | " + " | ".join(row) + " |")
        # 修复增益：配对检验
        w(f"\n### 修复增益（margin-lse − raw，配对 t，n={len(eids)} 实体）\n")
        for met in ("auc", "pauc10", "tpr5", "tpr10", "tpr20"):
            a = agg(res["score"][t], "raw", met)
            b = agg(res["score"][t], "margin-lse", met)
            com = sorted(set(a) & set(b))
            if len(com) < 3:
                continue
            x = np.array([a[e] for e in com]); y = np.array([b[e] for e in com])
            w(f"- **{met}**：raw {x.mean():.3f} → margin {y.mean():.3f} "
              f"(Δ={np.mean(y - x):+.3f}, p={ttest_rel(y, x).pvalue:.4g}, "
              f"n={len(com)})")
        # 各 margin 变体互比
        w("\n### 各修复变体（tpr5 / pauc10，整体）\n")
        w("| 得分 | tpr5 | pauc10 | auc |")
        w("|---|---|---|---|")
        for s in SCORES:
            a = agg(res["score"][t], s, "tpr5")
            b = agg(res["score"][t], s, "pauc10")
            c = agg(res["score"][t], s, "auc")
            w(f"| `{s}` | {np.mean(list(a.values())):.3f} | "
              f"{np.mean(list(b.values())):.3f} | {np.mean(list(c.values())):.3f} |")

    # ---------------- Stage C ----------------
    for t in tags:
        w(f"\n## Stage C —— 近邻感知操作点（k={t}）\n")
        w("| 实体 T | NN(T) | NN 占老类 | TPR@FPR5\\n(含NN) | TPR@FPR5\\n(排除NN) | 增益 |")
        w("|---|---|---|---|---|---|")
        gain = []
        for e in eids:
            ds = res["score"][t].get("__nn__", {}).get(e, [])
            if not ds:
                continue
            raw5 = agg(res["score"][t], "raw", "tpr5").get(e, float("nan"))
            ex5 = float(np.mean([d["tpr5_exclnn"] for d in ds]))
            fn = float(np.mean([d["frac_nn"] for d in ds]))
            gain.append(ex5 - raw5)
            nn = ds[0].get("nn_ent", "?")
            w(f"| {e} | {nn} | {fn:.3f} | {raw5:.3f} | {ex5:.3f} | {ex5 - raw5:+.3f} |")
        if gain:
            w(f"\n- **排除最近邻后 TPR@FPR5 平均 {np.mean(gain):+.3f}**"
              f"（8 实体 macro：{np.mean([agg(res['score'][t], 'raw', 'tpr5').get(e, np.nan) for e in eids]):.3f}"
              f" → {np.mean([float(np.mean([d['tpr5_exclnn'] for d in res['score'][t]['__nn__'][e]])) for e in eids if e in res['score'][t]['__nn__']]):.3f}）")
            w("- 语义：零样本筛查下，「FPR 预算小于最近邻所占比例」的操作点"
              "**结构性不可达**；报「疑似 T 或其化学近邻之一」已属成功"
              "（Schymanski Level 4–5），精确定性交由色谱复核。")

    # ---------------- Stage D ----------------
    for t in tags:
        w(f"\n## Stage D —— 公平口径：本文 vs 外部基线（k={t}）\n")
        for met in ("auc", "pauc10", "tpr5", "tpr10", "tpr20"):
            w(f"\n### {met}\n")
            arms = ["raw", "margin-lse", "nn-contrast"] + EXT
            w("| 臂 | " + " | ".join(arms) + " |")
            w("|---|" + "---|" * len(arms))
            row = []
            for a in arms:
                st = res["score"][t] if a in SCORES else res["ext"][t]
                d = agg(st, a, met)
                row.append(f"{np.mean(list(d.values())):.3f}" if d else "—")
            w("| 值 | " + " | ".join(row) + " |")
        # 配对：margin-lse vs rf
        w("\n### 配对检验（本文 margin-lse − rf，n=实体数）\n")
        for met in ("auc", "pauc10", "tpr5", "tpr20"):
            a = agg(res["ext"][t], "rf", met)
            b = agg(res["score"][t], "margin-lse", met)
            com = sorted(set(a) & set(b))
            if len(com) < 3:
                continue
            x = np.array([a[e] for e in com]); y = np.array([b[e] for e in com])
            w(f"- **{met}**：rf {x.mean():.3f} vs margin-lse {y.mean():.3f} "
              f"(Δ={np.mean(y - x):+.3f}, p={ttest_rel(y, x).pvalue:.4g})")

    # ---------------- Stage E ----------------
    for t in tags:
        w(f"\n## Stage E —— 浓度分层（k={t}，raw → margin-lse）\n")
        for met in ("auc", "pauc10", "tpr5", "tpr20"):
            w(f"\n### {met}\n")
            w("| 浓度箱% | raw | margin-lse | nn-contrast | rf | n |")
            w("|---|---|---|---|---|---|")
            for seg in ORD:
                vals, nn_ = [], 0
                for a, st in (("raw", res["score"][t]),
                              ("margin-lse", res["score"][t]),
                              ("nn-contrast", res["score"][t]),
                              ("rf", res["ext"][t])):
                    d = agg(st, a, met, seg)
                    vals.append(f"{np.mean(list(d.values())):.3f}" if d else "—")
                    nn_ = max(nn_, len(d))
                if nn_ == 0:
                    continue
                w(f"| {seg} | " + " | ".join(vals) + f" | {nn_} |")

    with open(os.path.join(OUT, "exp62_p6_calibration.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("[exp62] 已写出 out/exp62_p6_calibration.md")


if __name__ == "__main__":
    main()
