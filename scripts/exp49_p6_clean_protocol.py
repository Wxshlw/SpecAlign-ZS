"""exp49 —— **干净协议**下的冷启动总对比（本文方法 × 外部基线，同一套数据切分）

=============================================================================
为什么必须重跑：exp44/45/47/48 的 AUC 存在**测试集泄漏**
=============================================================================
exp44/45/48 构造 AUC 负样本（"老类参照集"）时用的是：

    ref.append(Xo[rng.permutation(len(Xo))[:200]])     # Xo = ent[e] 的全部

而 `Xtr` 里放的正是 `ent[e]` 的**全部**。也就是说：
    AUC 的负样本 == 训练集里的样本本身（逐条重复）。

后果（实测）：
  - `knn-cos` 的异常分 = 1 − max_j cos(x, 训练样本_j)
    ⇒ 负样本（= 训练样本）恒为 0，正样本（新类，held-out）恒 > 0
    ⇒ **AUC = 1.000，13/13 实体、3/3 seeds 全中**。这是泄漏，不是能力。
  - 同理 `svm-rbf` 0.821 / `rf` 0.786 也被抬高。
  - 本文方法同样受影响（负样本不是"没见过"的老类样本）。

修复：老类/基质也做 **70/30 划分**——30% 抽出作为 `Xte_old`，
      并且**不进训练集**。这样正负样本都是 held-out，协议才成立。

=============================================================================
本实验的设计（统一协议 = 唯一的可比性来源）
=============================================================================
对每个 held-out 实体 e、每个 seed：
  - 新类 e：30% → Xte_new；剩下的 pool 取 k 条 → 训练集
  - 每个老类 / 基质：70% → 训练集，30% → Xte_old（**与训练集不相交**）
  - 所有方法（本文 + 外部）在同一个 Xtr / Xte_new / Xte_old 上评估

本文臂：
  `ridge`   纯化学外推  W_new = (s_new − μ_S)·A，A 由老类 ridge 闭式解
  `ncm`     z 空间最近类中心（k>0 才可定义）
  `hybrid`  α·化学外推 + (1−α)·类中心，α 由 pseudo-unseen 模拟选
  `rand`    随机码本（零化学信息对照，k=0 时唯一合法的无先验下限）
  `none`    自由学习头（k>0 才可定义）

外部基线（同一协议重跑）：
  `plsda` 化学计量学黄金标准（ncomp 3 折 CV 选）
  `svm-rbf` / `rf`  通用 ML
  `knn-cos` / `centroid-cos`  谱库检索（领域实际做法）
  k=0 时外部方法无法为新类建模 ⇒ top1 恒 0；auc 退为「异常分」口径（对它们有利）

预注册判据
----------
- H49a（核心）：k=0 时本文 `ridge` 的 **auc** ≥ 最好的外部异常检测法
      ⇒ 修复泄漏后，"先验不可替代"是否仍然成立？（可能不成立，照报）
- H49b（核心）：k=0 时本文 `ridge` 的 top1 > 0，而外部方法 = 0
      ⇒ 零样本**识别**上仍是唯一可行路径
- H49c：k∈{1,5,20} 时本文最优臂的 top1 ≥ `centroid-cos`
- H49d：`knn-cos` 在干净协议下的 auc 显著低于泄漏版的 1.000
      ⇒ 证实 exp45 的 1.000 是泄漏（这是本实验的自检项）
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

OUT = r"D:/2026sci论文实验设计/论文6/out"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMP = 0.07
OLD_FRAC_TRAIN = 0.70      # 老类进训练集的比例（其余作 Xte_old）


# --------------------------------------------------------------------------
# 本文模型
# --------------------------------------------------------------------------
def build_net(d_in, d_out, seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d_in, 512), nn.ReLU(),
                         nn.Linear(512, d_out)).to(DEV)


@torch.no_grad()
def embed(net, X):
    net.eval()
    return F.normalize(net(torch.tensor(X, dtype=torch.float32, device=DEV)),
                       dim=1).cpu().numpy()


def stage1(Xtr, ytr, dim, nrows, steps, seed, S=None, mode="chem"):
    torch.manual_seed(seed)
    net = build_net(Xtr.shape[1], dim, seed)
    params = list(net.parameters())
    if mode == "chem":
        Wc = S[:nrows] - S[:nrows].mean(0, keepdims=True)
        W = torch.tensor(Wc, dtype=torch.float32, device=DEV)
    else:
        W = nn.Parameter(torch.randn(nrows, dim, device=DEV) * 0.01)
        params = params + [W]
    opt = torch.optim.Adam(params, lr=3e-4)
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


def build_hybrid(W_chem, Z, y, rows, alpha):
    W = np.zeros_like(W_chem)
    wc = nrm(W_chem)
    for j in rows:
        ix = np.where(y == j)[0]
        if len(ix) == 0:
            W[j] = wc[j]; continue
        zj = Z[ix].mean(0)
        v = alpha * wc[j] + (1 - alpha) * zj / (np.linalg.norm(zj) + 1e-12)
        W[j] = v / (np.linalg.norm(v) + 1e-12)
    return W


@torch.no_grad()
def macro_recall(net, W, Xv, yv, rows):
    z = F.normalize(net(torch.tensor(Xv, dtype=torch.float32, device=DEV)), dim=1)
    Wt = torch.tensor(nrm(W), dtype=torch.float32, device=DEV)
    pred = (z @ Wt.T)[:, rows].argmax(1).cpu().numpy()
    rs = [float((pred[yv == j] == i).mean()) for i, j in enumerate(rows)
          if (yv == j).sum()]
    return float(np.mean(rs)) if rs else 0.0


@torch.no_grad()
def recall_of(net, W, Xv, yv, j, rows):
    z = F.normalize(net(torch.tensor(Xv, dtype=torch.float32, device=DEV)), dim=1)
    Wt = torch.tensor(nrm(W), dtype=torch.float32, device=DEV)
    pred = (z @ Wt.T)[:, rows].argmax(1).cpu().numpy()
    m = yv == j
    return float("nan") if m.sum() == 0 else float((pred[m] == rows.index(j)).mean())


@torch.no_grad()
def metrics(net, W, Xte_new, Xte_old, new_row=18):
    z_n = F.normalize(net(torch.tensor(Xte_new, dtype=torch.float32, device=DEV)), dim=1)
    z_o = F.normalize(net(torch.tensor(Xte_old, dtype=torch.float32, device=DEV)), dim=1)
    Wt = torch.tensor(nrm(W), dtype=torch.float32, device=DEV)
    lg_n = (z_n @ Wt.T).cpu().numpy()[:, :19]
    lg_o = (z_o @ Wt.T).cpu().numpy()[:, :19]
    top1 = float((lg_n.argmax(1) == new_row).mean())
    auc = auc_of(lg_n[:, new_row], lg_o[:, new_row])
    ranks = np.where(np.argsort(-lg_n, axis=1) == new_row)[1] + 1
    return dict(top1=top1, auc=auc, mrr=float((1.0 / ranks).mean()))


# --------------------------------------------------------------------------
# 外部基线
# --------------------------------------------------------------------------
def auc_of(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)).astype(float) + 1.0
    n1, n2 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def cos_to(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T


def fit_plsda(Xtr, ytr, ncomp):
    classes = np.unique(ytr)
    Y = np.zeros((len(ytr), len(classes)))
    for i, c in enumerate(classes):
        Y[ytr == c, i] = 1.0
    sc = StandardScaler().fit(Xtr)
    pls = PLSRegression(n_components=min(ncomp, Xtr.shape[1], len(ytr) - 1))
    pls.fit(sc.transform(Xtr), Y)
    return sc, pls, classes


def baselines(Xtr, ytr, Xte_new, Xte_old, NEW, ncomps=(5, 10, 20)):
    """返回 {方法名: {top1, auc}}。k=0（新类不在 ytr 里）时自动退为异常分口径。"""
    has_new = NEW in set(np.unique(ytr).tolist())
    out = {}
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    Xn_s, Xo_s = sc.transform(Xte_new), sc.transform(Xte_old)

    # PLS-DA -------------------------------------------------------------
    best_c, best_v = ncomps[0], -1.0
    idx_all = np.arange(len(Xtr_s))
    rng_c = np.random.default_rng(0); rng_c.shuffle(idx_all)
    folds = np.array_split(idx_all, 3)
    for c in ncomps:
        try:
            vs = []
            for f in folds:
                m = np.ones(len(Xtr_s), bool); m[f] = False
                s2, p2, cl2 = fit_plsda(Xtr_s[m], ytr[m], c)
                vs.append(float((cl2[p2.predict(s2.transform(Xtr_s[f])).argmax(1)]
                                 == ytr[f]).mean()))
            v = float(np.mean(vs))
        except Exception:
            v = -1.0
        if v > best_v:
            best_v, best_c = v, c
    s_p, pls, cls = fit_plsda(Xtr_s, ytr, best_c)
    P_n, P_o = pls.predict(s_p.transform(Xn_s)), pls.predict(s_p.transform(Xo_s))
    if has_new:
        top1 = float((cls[P_n.argmax(1)] == NEW).mean())
        auc = auc_of(P_n[:, list(cls).index(NEW)], P_o[:, list(cls).index(NEW)])
    else:
        top1, auc = 0.0, auc_of(-P_n.max(1), -P_o.max(1))
    out["plsda"] = dict(top1=top1, auc=auc)

    # SVM-RBF ------------------------------------------------------------
    sv = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
             random_state=0).fit(Xtr_s, ytr)
    Pn, Po = sv.predict_proba(Xn_s), sv.predict_proba(Xo_s)
    cn = list(sv.classes_)
    if has_new:
        top1 = float((sv.predict(Xn_s) == NEW).mean())
        auc = auc_of(Pn[:, cn.index(NEW)], Po[:, cn.index(NEW)])
    else:
        top1, auc = 0.0, auc_of(1 - Pn.max(1), 1 - Po.max(1))
    out["svm-rbf"] = dict(top1=top1, auc=auc)

    # Random Forest ------------------------------------------------------
    rf = RandomForestClassifier(n_estimators=300, random_state=0,
                                n_jobs=-1).fit(Xtr_s, ytr)
    Pn, Po = rf.predict_proba(Xn_s), rf.predict_proba(Xo_s)
    cn = list(rf.classes_)
    if has_new:
        top1 = float((rf.predict(Xn_s) == NEW).mean())
        auc = auc_of(Pn[:, cn.index(NEW)], Po[:, cn.index(NEW)])
    else:
        top1, auc = 0.0, auc_of(1 - Pn.max(1), 1 - Po.max(1))
    out["rf"] = dict(top1=top1, auc=auc)

    # 谱库检索 ------------------------------------------------------------
    Cn, Co = cos_to(Xn_s, Xtr_s), cos_to(Xo_s, Xtr_s)
    if has_new:
        top1 = float((ytr[Cn.argmax(1)] == NEW).mean())
        auc = auc_of(Cn[:, ytr == NEW].max(1), Co.max(1))
    else:
        top1, auc = 0.0, auc_of(1 - Cn.max(1), 1 - Co.max(1))
    out["knn-cos"] = dict(top1=top1, auc=auc)

    cj = np.unique(ytr)
    cent = np.stack([Xtr_s[ytr == j].mean(0) for j in cj])
    Cn2, Co2 = cos_to(Xn_s, cent), cos_to(Xo_s, cent)
    if has_new:
        top1 = float((cj[Cn2.argmax(1)] == NEW).mean())
        auc = auc_of(Cn2[:, list(cj).index(NEW)], Co2[:, list(cj).index(NEW)])
    else:
        top1, auc = 0.0, auc_of(1 - Cn2.max(1), 1 - Co2.max(1))
    out["centroid-cos"] = dict(top1=top1, auc=auc)
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="0,1,5,20")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.001,0.01,0.1,1")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--eids", default="")
    ap.add_argument("--nobase", action="store_true")
    args = ap.parse_args()
    shots = [None if s.strip() == "full" else int(s) for s in args.shots.split(",")]
    tags = [("full" if s is None else str(s)) for s in shots]
    lams = [float(x) for x in args.lams.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    ent, bg, _ = G1.load_g1()
    eids = sorted(ent.keys())
    if args.eids:
        eids = [e for e in eids if e in args.eids.split(",")]
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]
    zc = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    T_all = np.asarray(zc["clm"], float)
    dim = T_all.shape[1]
    ki = {k: i for i, k in enumerate(keys_all)}

    OURS = ["ridge", "ncm", "hybrid", "rand", "none"]
    BASE = ["plsda", "svm-rbf", "rf", "knn-cos", "centroid-cos"]
    arms = OURS + ([] if args.nobase else BASE)
    res = {"acc": {t: {a: {} for a in arms} for t in tags},
           "alpha": {}, "leak_check": {},
           "meta": {"shots": tags, "seeds": args.seeds, "eids": eids,
                    "alphas": alphas, "lams": lams, "dim": dim,
                    "old_frac_train": OLD_FRAC_TRAIN}}
    print(f"[exp49] device={DEV} shots={tags} seeds={args.seeds} "
          f"old_train_frac={OLD_FRAC_TRAIN}", flush=True)

    for e_hold in eids:
        others = [e for e in sorted(ent.keys()) if e != e_hold]
        S = np.asarray(T_all)[[ki[e] for e in others]
                              + [ki[f"M_{m}"] for m in mids] + [ki[e_hold]]]
        NEW = 18
        Xe = np.asarray(ent[e_hold], float)

        for k in shots:
            tag = "full" if k is None else str(k)
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed * 7919 + 13)
                # ---- 新类：30% 测试，pool 取 k ----
                perm = rng.permutation(len(Xe))
                n_te = int(min(max(5, round(0.3 * len(Xe))), len(Xe) - 1))
                te_ix, pool_ix = perm[:n_te], perm[n_te:]
                tr_e_ix = pool_ix if k is None else pool_ix[:min(k, len(pool_ix))]
                Xte_new = Xe[te_ix]

                # ---- 老类/基质：70% 训练，30% 作为 AUC 负样本（不相交）----
                parts, ys, refs = [], [], []
                for j, e in enumerate(others):
                    Xo = np.asarray(ent[e], float)
                    pp = rng.permutation(len(Xo))
                    ntr = max(2, int(round(OLD_FRAC_TRAIN * len(Xo))))
                    parts.append(Xo[pp[:ntr]]); ys.append(np.full(ntr, j))
                    refs.append(Xo[pp[ntr:]])
                for j, m in enumerate(mids):
                    Xo = np.asarray(bg[m], float)
                    pp = rng.permutation(len(Xo))
                    ntr = max(2, int(round(OLD_FRAC_TRAIN * len(Xo))))
                    parts.append(Xo[pp[:ntr]])
                    ys.append(np.full(ntr, len(others) + j))
                    refs.append(Xo[pp[ntr:]])
                nrows = len(others) + len(mids)
                if len(tr_e_ix):
                    parts.append(Xe[tr_e_ix])
                    ys.append(np.full(len(tr_e_ix), 18)); nrows = 19
                Xtr = np.concatenate(parts).astype(np.float32)
                ytr = np.concatenate(ys).astype(np.int64)
                Xte_old = np.concatenate(refs).astype(np.float32)

                # 自检：确认 Xte_old 不在 Xtr 里（逐行最近邻余弦应 < 1）
                lk = cos_to(Xte_old[:50], Xtr[:1000]).max(1)
                res["leak_check"].setdefault(e_hold, []).append(
                    float(np.median(lk)))

                # ---------------- 外部基线 ----------------
                if not args.nobase:
                    for m_, v in baselines(Xtr, ytr, Xte_new, Xte_old, NEW).items():
                        res["acc"][tag].setdefault(m_, {}).setdefault(
                            e_hold, []).append(
                            {kk: (None if not np.isfinite(vv) else round(float(vv), 4))
                             for kk, vv in v.items()})

                # ---------------- 本文 ----------------
                rng_v = np.random.default_rng(seed * 31 + 7)
                val_mask = np.zeros(len(Xtr), bool)
                rows_old = sorted({int(y) for y in ytr})
                for j in rows_old:
                    if j == 18 and len(tr_e_ix) == 0:
                        continue
                    ix = rng_v.permutation(np.where(ytr == j)[0])
                    n_val = min(max(3, int(0.2 * len(ix))), max(0, len(ix) - 1))
                    val_mask[ix[:n_val]] = True
                Xfit, yfit = Xtr[~val_mask], ytr[~val_mask]
                Xval, yval = Xtr[val_mask], ytr[val_mask]
                val_rows = sorted({int(y) for y in yval})

                net, W_free = stage1(Xfit, yfit, dim, nrows, args.steps,
                                     seed * 1000 + eids.index(e_hold),
                                     S=S, mode="chem")
                Zfit, Ztr = embed(net, Xfit), embed(net, Xtr)
                Z_old = np.stack([Zfit[yfit == j].mean(0) for j in rows_old])
                S_old, mu_S = S[rows_old], S[rows_old].mean(0, keepdims=True)

                lam_b, bv = lams[0], -1.0
                for lam in lams:
                    Wc = (S - mu_S) @ head_ridge_R(S_old, Z_old, lam)
                    v = macro_recall(net, Wc, Xval, yval, val_rows)
                    if v > bv:
                        bv, lam_b = v, lam
                W_chem = (S - mu_S) @ head_ridge_R(S_old, Z_old, lam_b)

                # 随机码本对照（零化学信息）
                rng_r = np.random.default_rng(seed * 977 + 5)
                Wr = rng_r.normal(size=W_chem.shape).astype(np.float32)
                Wr = Wr / (np.linalg.norm(Wr, axis=1, keepdims=True) + 1e-12)
                Wr = (S - mu_S) @ head_ridge_R(S_old, Z_old, lam_b) * 0 + Wr

                accs = {"ridge": metrics(net, W_chem, Xte_new, Xte_old, NEW),
                        "rand": metrics(net, Wr, Xte_new, Xte_old, NEW)}
                if len(tr_e_ix):
                    rows_fit = sorted({int(y) for y in yfit})
                    k_sim = max(1, len(tr_e_ix))
                    rng_s = np.random.default_rng(seed * 17 + 3)
                    cand = [j for j in rows_fit
                            if j != NEW and int((yfit == j).sum()) > k_sim]
                    if len(cand) > 6:
                        cand = [int(x) for x in rng_s.choice(cand, 6, replace=False)]
                    a_b, ab = 1.0, -1.0
                    for a in alphas:
                        recs = []
                        for j in cand:
                            W = build_hybrid(W_chem, Zfit, yfit, rows_fit, a)
                            ix = rng_s.permutation(np.where(yfit == j)[0])[:k_sim]
                            zj = Zfit[ix].mean(0)
                            v = (a * nrm(W_chem)[j]
                                 + (1 - a) * zj / (np.linalg.norm(zj) + 1e-12))
                            W[j] = v / (np.linalg.norm(v) + 1e-12)
                            rv = recall_of(net, W, Xval, yval, j, rows_fit)
                            if np.isfinite(rv):
                                recs.append(rv)
                        v = float(np.mean(recs)) if recs else 0.0
                        if v > ab:
                            ab, a_b = v, a
                    res["alpha"].setdefault(tag, {}).setdefault(
                        e_hold, []).append(float(a_b))
                    rows_all = sorted({int(y) for y in ytr})
                    accs["ncm"] = metrics(net, build_hybrid(
                        W_chem, Ztr, ytr, rows_all, 0.0), Xte_new, Xte_old, NEW)
                    accs["hybrid"] = metrics(net, build_hybrid(
                        W_chem, Ztr, ytr, rows_all, a_b), Xte_new, Xte_old, NEW)
                    accs["none"] = metrics(net, W_free, Xte_new, Xte_old, NEW)
                else:
                    accs["ncm"] = dict(top1=None, auc=None, mrr=None)
                    accs["hybrid"] = accs["ridge"]
                    accs["none"] = dict(top1=None, auc=None, mrr=None)

                for a, m_ in accs.items():
                    res["acc"][tag].setdefault(a, {}).setdefault(
                        e_hold, []).append(
                        {kk: (None if vv is None or not np.isfinite(vv)
                              else round(float(vv), 4)) for kk, vv in m_.items()})

        cells = []
        for t in tags:
            s = " ".join(
                f"{a}={np.nanmean([d['top1'] if d['top1'] is not None else np.nan for d in res['acc'][t][a].get(e_hold, [])]):.2f}"
                f"/{np.nanmean([d['auc'] if d['auc'] is not None else np.nan for d in res['acc'][t][a].get(e_hold, [])]):.2f}"
                for a in arms if res["acc"][t][a].get(e_hold))
            cells.append(f"{t}: {s}")
        print(f"[{e_hold:>7}] " + " | ".join(cells), flush=True)

    with open(os.path.join(OUT, "exp49_clean.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("[exp49] 已写出 out/exp49_clean.json")
    write_report(res, arms, tags, eids)


def write_report(res, arms, tags, eids):
    try:
        from scipy.stats import ttest_rel
        HAS = True
    except Exception:
        HAS = False

    def M(t, a, met="top1"):
        v = []
        for e in eids:
            d = [x[met] for x in res["acc"][t].get(a, {}).get(e, [])
                 if x[met] is not None]
            if d:
                v.append(float(np.mean(d)))
        return np.asarray(v, float)

    L = []
    w = L.append
    w("# exp49 —— 干净协议下的冷启动总对比（老类 70/30 划分，AUC 负样本与训练集不相交）\n")
    lk = [np.mean(v) for v in res["leak_check"].values()]
    w(f"**泄漏自检**：Xte_old 到 Xtr 的最近邻余弦中位数 = {np.mean(lk):.4f} "
      f"（若 ≈1.000 说明仍有泄漏）\n")
    w("\n单元格格式：**top1 / auc**。\n")
    for met in ("top1", "auc"):
        w(f"\n## {met}\n")
        w("| k | " + " | ".join(f"`{a}`" for a in arms) + " |")
        w("|---|" + "---|" * len(arms))
        for t in tags:
            row = []
            for a in arms:
                v = M(t, a, met)
                row.append(f"{np.nanmean(v):.3f}" if len(v) else "—")
            w(f"| {t} | " + " | ".join(row) + " |")

    w("\n## 判据\n")
    for t in tags:
        best_base, bb = "", -1.0
        for a in ("plsda", "svm-rbf", "rf", "knn-cos", "centroid-cos"):
            v = M(t, a, "auc")
            if len(v) and np.nanmean(v) > bb:
                bb, best_base = float(np.nanmean(v)), a
        o = M(t, "ridge", "auc")
        oa = float(np.nanmean(o)) if len(o) else float("nan")
        ot = M(t, "ridge", "top1")
        ott = float(np.nanmean(ot)) if len(ot) else float("nan")
        w(f"- **k={t}**：ridge auc {oa:.3f} vs 最强外部({best_base}) {bb:.3f}"
          f" ⇒ H49a {'成立' if oa >= bb else '**不成立**'}；"
          f"ridge top1 {ott:.3f} ⇒ H49b {'成立' if ott > 0 else '不成立'}")
        # 本文最优臂 vs centroid-cos（top1）
        cc = M(t, "centroid-cos", "top1")
        if len(cc):
            best_ours, bo = "", -1.0
            for a in ("ridge", "ncm", "hybrid"):
                v = M(t, a, "top1")
                if len(v) and np.nanmean(v) > bo:
                    bo, best_ours = float(np.nanmean(v)), a
            w(f"   - 本文最优({best_ours}) top1 {bo:.3f} vs centroid-cos "
              f"{np.nanmean(cc):.3f} ⇒ H49c {'成立' if bo >= np.nanmean(cc) else '不成立'}")
            if HAS:
                x, y = cc, M(t, best_ours, "top1")
                m = ~np.isnan(x) & ~np.isnan(y)
                if m.sum() > 1:
                    w(f"   - 配对 Δ={np.mean(y[m]) - np.mean(x[m]):+.3f}, "
                      f"p={ttest_rel(y[m], x[m]).pvalue:.4f}")
    w("\n## H49d：泄漏验证\n")
    w("exp45（泄漏协议）knn-cos auc = 1.000；本实验（干净协议）见上表 `knn-cos`。\n")

    with open(os.path.join(OUT, "exp49_p6_clean.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("[exp49] 已写出 out/exp49_p6_clean.md")


if __name__ == "__main__":
    main()
