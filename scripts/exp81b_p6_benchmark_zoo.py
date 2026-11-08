r"""exp81b —— 三方横评：算法动物园 × 先验动物园 × 先验×算法网格。

用户要求
--------
1. 只和 RF/SVM 比太少 → 至少 10+ 种常见机器学习 / 深度学习算法横评
2. 只和随机码本比太少 → 和**普通先验 / 普通语义**横评（ECFP / MACCS / Avalon /
   RDKit 描述符 / 元素-物化 / SMILES 字符袋 / one-hot）
3. 把本文先验**装到若干普通 ML/DL 算法上**，看增益是否算法无关（Δ_prior 逐算法）

三个轴
------
【轴1 B】算法动物园（23 个基线，全部用原始光谱，group 协议）
    多类/判别式（k>0 用 P(T)；k=0 用 1−maxprob 新奇度）：
      rf, extratrees, histgb, svm_rbf, svm_lin, logreg, knn, gnb, lda,
      mlp, cnn1d, adaboost, dtree, ncm, plsda
    单类/新奇检测（检出：队列=同基质纯谱；识别：队列=老类+基质）：
      ocsvm, iforest, lof, mcd, mahal, pca_resid, simca, knn_dist, ae1d
    表示对照：rf_z / svm_z（用本文 stage1 的 z，检验"赢在表示还是赢在头"）
【轴2 P】先验动物园（同一个岭回归头，只换码本；λ 逐先验按验证集选）
      clm(本文) / ecfp384 / ecfp2048 / maccs / avalon / rdkit / physchem /
      charbag / onehot / rand384
【轴3 H/C】先验 × 算法
    H：平滑头 S→Z（18 个老类对）ridge(本文) / pls / krr / gp / knn1(化学最近邻) / mlp
    C：条件化打分器 g([z, S_T])（可装任意算法）
       logreg / rf / histgb / svm_rbf / mlp / knn  ×  {clm, rand}
    ⇒ 逐算法 Δ_prior = clm − rand，检验"增益是否算法无关"

公平性
------
- 协议与 exp79 完全一致：group（按样品 id 分组）、OLD_FRAC=0.7、3 seeds
- 所有先验统一 384 维、行归一化、同一 profile 权重（exp81a）
- 基线全部只用原始光谱（不含本文 stage1），另给 rf_z/svm_z 作表示对照
- k=0 时基线不含 T 任何样本，与本文零样本口径严格对齐
"""
from __future__ import annotations
import os, sys, json, argparse, warnings, time, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
T0 = time.time()

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "论文8"))
import p6_g1_data as G1
from exp63_p6_screening_protocol import (stage1, embed, head_ridge_R, nrm,
                                         auc_of, tpr_at_fpr, EID2MID)
from exp68_p6_cond_scorer import split_by_group
from exp71_p6_detection_fix import load_groups_cached, pick_lam
from exp74_p6_fairness_audit import load_bg_gid, split_group
from exp75_p6_v5_universal_detector import metrics

from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              AdaBoostClassifier, IsolationForest)
from sklearn.svm import SVC, OneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, LocalOutlierFactor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.mixture import GaussianMixture

OUT = os.path.join(os.path.dirname(HERE), "out")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OLD_FRAC = 0.7
SHRI = 0.2
NRAND = 5          # 先验动物园：随机码本抽样次数
CRAND = 3          # 条件化打分器：随机码本抽样次数

PRIORS = ["clm", "ecfp384", "ecfp2048", "maccs", "avalon", "rdkit",
          "physchem", "charbag", "onehot", "rand384"]
MULTI = ["rf", "extratrees", "histgb", "svm_rbf", "svm_lin", "logreg",
         "knn", "gnb", "lda", "mlp", "cnn1d", "adaboost", "dtree", "ncm",
         "plsda"]
ONECLASS = ["ocsvm", "iforest", "lof", "mcd", "mahal", "pca_resid",
            "simca", "knn_dist", "ae1d"]
COND_ALGOS = ["logreg", "rf", "histgb", "svm_rbf", "mlp", "knn"]


# ---------------------------------------------------------------------------
# 深度学习组件
# ---------------------------------------------------------------------------
class CNN1D(nn.Module):
    def __init__(self, L, ncls):
        super().__init__()
        self.c1 = nn.Conv1d(1, 32, 9, padding=4)
        self.c2 = nn.Conv1d(32, 64, 9, padding=4)
        self.b1 = nn.BatchNorm1d(32)
        self.b2 = nn.BatchNorm1d(64)
        self.fc = nn.Linear(64 * (L // 4), ncls)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.max_pool1d(F.relu(self.b1(self.c1(x))), 2)
        x = F.max_pool1d(F.relu(self.b2(self.c2(x))), 2)
        return self.fc(x.flatten(1))


class AE1D(nn.Module):
    def __init__(self, L, hid=64, bot=16):
        super().__init__()
        self.e1 = nn.Linear(L, hid)
        self.e2 = nn.Linear(hid, bot)
        self.d1 = nn.Linear(bot, hid)
        self.d2 = nn.Linear(hid, L)

    def forward(self, x):
        h = F.relu(self.e1(x))
        b = F.relu(self.e2(h))
        return self.d2(F.relu(self.d1(b)))


def fit_cnn(X, y, ncls, epochs, seed, lr=1e-3):
    torch.manual_seed(seed)
    net = CNN1D(X.shape[1], ncls).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.tensor(X, dtype=torch.float32, device=DEV)
    yt = torch.tensor(y, dtype=torch.long, device=DEV)
    n = len(X)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            loss = F.cross_entropy(net(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


def cnn_score(net, X):
    with torch.no_grad():
        out = net(torch.tensor(X, dtype=torch.float32, device=DEV))
    return out.softmax(1).cpu().numpy()


def fit_ae(X, epochs, seed, hid=64, bot=16):
    torch.manual_seed(seed)
    net = AE1D(X.shape[1], hid, bot).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.tensor(X, dtype=torch.float32, device=DEV)
    n = len(X)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            loss = F.mse_loss(net(Xt[idx]), Xt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


def ae_score(net, X):
    with torch.no_grad():
        rec = net(torch.tensor(X, dtype=torch.float32, device=DEV))
        Xt = torch.tensor(X, dtype=torch.float32, device=DEV)
    return ((rec - Xt) ** 2).mean(1).cpu().numpy()


# ---------------------------------------------------------------------------
def unit(v):
    nv = np.linalg.norm(v)
    return v / nv if nv > 1e-12 else None


def zsc(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-9)


def safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        return None


# ---------------------------------------------------------------------------
def build_multiclass(seed):
    """多类/判别式基线工厂。"""
    f = {}
    f["rf"] = lambda: RandomForestClassifier(300, random_state=seed, n_jobs=-1)
    f["extratrees"] = lambda: ExtraTreesClassifier(300, random_state=seed, n_jobs=-1)
    f["histgb"] = lambda: HistGradientBoostingClassifier(random_state=seed)
    f["svm_rbf"] = lambda: SVC(kernel="rbf", C=10, gamma="scale", random_state=seed)
    f["svm_lin"] = lambda: SVC(kernel="linear", C=1, random_state=seed)
    f["logreg"] = lambda: LogisticRegression(max_iter=400, C=1.0,
                                             class_weight="balanced")
    f["knn"] = lambda: KNeighborsClassifier(5, metric="cosine")
    f["gnb"] = lambda: GaussianNB()
    f["lda"] = lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    f["mlp"] = lambda: MLPClassifier((256, 128), max_iter=400, random_state=seed)
    f["adaboost"] = lambda: AdaBoostClassifier(random_state=seed)
    f["dtree"] = lambda: DecisionTreeClassifier(random_state=seed, max_depth=12)
    return f


def multi_score(name, mdl, Xte, t_cls=None):
    """k>0：返回 P(T)；k=0：返回新奇度 1−maxprob。"""
    if name == "svm_rbf" or name == "svm_lin":
        D = mdl.decision_function(Xte)
        if t_cls is not None:
            return D[:, t_cls]
        return -D.max(1)
    if name == "ncm":
        C = mdl["C"]; Xn = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-12)
        S = Xn @ C.T
        if t_cls is not None:
            return S[:, t_cls]
        return 1 - S.max(1)
    if name == "plsda":
        T = mdl["pls"].transform(Xte)
        d = np.linalg.norm(T[:, None, :] - mdl["C"][None, :, :], axis=2)
        if t_cls is not None:
            return -d[:, t_cls]
        return d.min(1)
    P = mdl.predict_proba(Xte)
    if t_cls is not None:
        return P[:, t_cls]
    return 1 - P.max(1)


def fit_ncm(X, y, ncls):
    C = np.stack([X[y == j].mean(0) for j in range(ncls)])
    return {"C": C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)}


def fit_plsda(X, y, ncls):
    nc = max(2, min(10, ncls - 1, X.shape[1]))
    Y = np.eye(ncls)[y]
    pls = PLSRegression(n_components=nc).fit(X, Y)
    T = pls.transform(X)
    C = np.stack([T[y == j].mean(0) for j in range(ncls)])
    return {"pls": pls, "C": C}


def build_oneclass(seed, cohort):
    """单类 / 新奇检测（cohort = 只含"正常"样本的队列）。"""
    f = {}
    f["ocsvm"] = lambda: OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    f["iforest"] = lambda: IsolationForest(n_estimators=300, random_state=seed,
                                           n_jobs=-1)
    f["lof"] = lambda: LocalOutlierFactor(n_neighbors=20, novelty=True)
    f["mcd"] = lambda: GaussianMixture(1, covariance_type="full",
                                       random_state=seed)
    f["mahal"] = lambda: None            # 解析式，单独处理
    f["pca_resid"] = lambda: PCA(n_components=min(8, cohort.shape[1] - 1))
    f["simca"] = lambda: PCA(n_components=min(5, cohort.shape[1] - 1))
    f["knn_dist"] = lambda: NearestNeighbors(n_neighbors=min(6, len(cohort)))
    f["ae1d"] = lambda: None
    return f


def oc_score(name, mdl, Xte, cohort):
    if name == "ocsvm":
        return -mdl.decision_function(Xte).ravel()
    if name == "iforest":
        return -mdl.score_samples(Xte)
    if name == "lof":
        return -mdl.score_samples(Xte)
    if name == "mcd":                    # GMM 负对数似然
        return -mdl.score_samples(Xte)
    if name == "mahal":
        d = Xte - mdl["mu"]
        w = mdl["Winv"] @ d.T
        return (d * w.T).sum(1)
    if name in ("pca_resid", "simca"):
        Z = mdl.transform(Xte)
        R = mdl.inverse_transform(Z)
        q = ((Xte - R) ** 2).sum(1)
        if name == "simca":
            var = mdl.explained_variance_ + 1e-9
            t2 = ((Z ** 2) / var).sum(1)
            return zsc(q) + zsc(t2)
        return q
    if name == "knn_dist":
        d, _ = mdl.kneighbors(Xte)
        return d.mean(1)
    if name == "ae1d":
        return ae_score(mdl, Xte)
    return None


# ---------------------------------------------------------------------------
# 轴3：条件化打分器 g([z, S_T]) —— 可装任意 ML 算法
# ---------------------------------------------------------------------------
def build_cond_pairs(Zf, yf, S_all, rows, rng, nneg=1, max_n=6000):
    """正样本对 (z_i, S_其类)；负样本对 (z_i, S_随机他类)。"""
    ix = rng.permutation(len(Zf))[:max_n]
    rowset = np.asarray(sorted(rows))
    nr = len(rowset)
    Zp, Sp, yb = [], [], []
    for i in ix:
        j = int(yf[i])
        Zp.append(Zf[i]); Sp.append(S_all[j]); yb.append(1)
        for _ in range(nneg):
            jn = rowset[rng.integers(nr)]
            if jn == j:
                jn = rowset[(rng.integers(nr - 1) + (jn == rowset[0])) % nr]
                if jn == j:
                    continue
            Zp.append(Zf[i]); Sp.append(S_all[jn]); yb.append(0)
    return np.concatenate([np.stack(Zp), np.stack(Sp)], 1).astype(np.float32), \
        np.asarray(yb, np.int64)


def cond_algo(name, seed):
    if name == "logreg":
        return LogisticRegression(max_iter=1500, C=1.0)
    if name == "rf":
        return RandomForestClassifier(200, random_state=seed, n_jobs=-1,
                                      min_samples_leaf=2)
    if name == "histgb":
        return HistGradientBoostingClassifier(random_state=seed, max_iter=200)
    if name == "svm_rbf":
        return SVC(kernel="rbf", C=10, gamma="scale", random_state=seed)
    if name == "mlp":
        return MLPClassifier((128,), max_iter=200, random_state=seed)
    if name == "knn":
        return KNeighborsClassifier(5)
    raise ValueError(name)


def cond_score_matrix(mdl, Z, S_list, pca=None):
    """返回 (n, len(S_list))：每个候选与每个样本的匹配分。"""
    n, m = len(Z), len(S_list)
    P = np.empty((n, m), np.float32)
    for j in range(m):
        X = np.concatenate([Z, np.repeat(S_list[j][None, :], n, 0)], 1)
        if pca is not None:
            X = pca.transform(X)
        if hasattr(mdl, "predict_proba"):
            P[:, j] = mdl.predict_proba(X)[:, 1]
        else:
            P[:, j] = mdl.decision_function(X)
    return P


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lams", default="0.0001,0.001,0.01,0.1,1")
    ap.add_argument("--shots", default="0,5")
    ap.add_argument("--eids", default="")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--axis", default="all",
                    help="all | B(算法动物园) | P(先验动物园) | C(先验×算法)")
    ap.add_argument("--out", default="exp81_zoo.json")
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(",")]
    shots = [int(s) for s in args.shots.split(",")]
    do = {c: (args.axis == "all" or c in args.axis) for c in "BPC"}
    if do["C"]:
        do["P"] = True      # 条件化需要 clm / rand 码本

    ent, conc, bg, _ = G1.load_g1_conc()
    GR = load_groups_cached()
    BG_GID = load_bg_gid(bg)
    eids = sorted(set(GR) & set(EID2MID))
    if args.eids:
        eids = [e for e in eids if e in args.eids.split(",")]
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]
    ki = {k: i for i, k in enumerate(keys_all)}
    zc = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    dz = int(np.asarray(zc["clm"]).shape[1])
    PC = np.load(os.path.join(OUT, "exp81_prior_cache.npz"), allow_pickle=True)
    assert list(PC["keys"]) == keys_all
    PRI = {p: np.asarray(PC[p], float) for p in PRIORS}

    res = {"B": {}, "P": {}, "C": {}, "O": {}, "DIAG": {},
           "meta": dict(shots=shots, seeds=args.seeds, priors=PRIORS,
                        multi=MULTI, oneclass=ONECLASS, cond=COND_ALGOS,
                        epochs=args.epochs)}
    print(f"[exp81] device={DEV} 实体={eids} shots={shots} axis={args.axis}",
          flush=True)

    for e_hold in eids:
        others = [e for e in sorted(ent) if e != e_hold]
        Xe = np.asarray(ent[e_hold], float)
        gid_e = GR[e_hold]["gid"]
        mid_h = EID2MID[e_hold]
        ord_ix = [ki[e] for e in others] + [ki[f"M_{m}"] for m in mids] + [ki[e_hold]]
        S_pri = {p: PRI[p][ord_ix] for p in PRIORS}
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
                                         + list(mid_ref.values())).astype(np.float32)

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
                                seed * 1000 + eids.index(e_hold),
                                S=S_pri["clm"])
                Zfit = embed(net, Xfit)
                rows = sorted({int(y) for y in yfit})
                Zc = np.stack([Zfit[yfit == j].mean(0) for j in rows])
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

                R = np.concatenate([Zfit[yfit == j] - Zfit[yfit == j].mean(0)
                                    for j in rows], 0)
                Sw = R.T @ R / max(1, len(R))
                tr_ = float(np.trace(Sw)) / Sw.shape[0]
                Sig = (1 - SHRI) * Sw + SHRI * tr_ * np.eye(Sw.shape[0])
                Sig += 1e-6 * np.eye(Sig.shape[0])
                Si = np.linalg.inv(Sig)

                Zn = embed(net, Xte_new)

                # ---- 轴2 / 轴3H：同一岭回归头，只换先验 ----
                heads = {}
                if do["P"]:
                    for p in PRIORS:
                        # 随机码本必须多次抽样：单次抽样的方向是随机的，
                        # 会碰巧撞上净信号方向而虚高（exp78 已吃过这个亏）。
                        # 每个抽样**单独算 AUC 再平均**（先平均分数会退化成常数）。
                        draws = NRAND if p == "rand384" else 1
                        for di in range(draws):
                            S_all = (nrm(np.random.default_rng(
                                1000 * seed + di).normal(size=S_pri[p].shape))
                                     if p == "rand384" else S_pri[p])
                            S_o = S_all[rows]
                            mu_S = S_all[rows].mean(0, keepdims=True)
                            lam = pick_lam(S_o, Zc, lams, net, Xval, yval, rows,
                                           mu_S, S_all)
                            A = head_ridge_R(S_o, Zc, lam)
                            An = head_ridge_R(S_o, dZ, lam)
                            heads[p if draws == 1 else f"{p}#{di}"] = dict(
                                W=(S_all - mu_S) @ A,
                                Wn=(S_all - mu_S) @ An, lam=lam)
                # ---- 本文臂（clm + 白化）----
                if not do["P"]:
                    S_all = S_pri["clm"]
                    S_o, mu_S = S_all[rows], S_all[rows].mean(0, keepdims=True)
                    lam = pick_lam(S_o, Zc, lams, net, Xval, yval, rows,
                                   mu_S, S_all)
                    A = head_ridge_R(S_o, Zc, lam)
                    An = head_ridge_R(S_o, dZ, lam)
                    heads["clm"] = dict(W=(S_all - mu_S) @ A,
                                        Wn=(S_all - mu_S) @ An, lam=lam)

                # ---- 轴3H：平滑头动物园（S→Z，18 对；只 clm vs rand）----
                hzoo = {}
                if do["P"]:
                    lam = heads["clm"]["lam"]
                    for hn in ("ridge", "pls", "krr", "gp", "knn1", "mlp"):
                        for pp in ["clm"] + [f"rand384#{di}" for di in range(2)]:
                            p = pp.split("#")[0]
                            Sp = S_pri[p] if p == "clm" else nrm(
                                np.random.default_rng(1000 * seed + 77
                                                      + int(pp.split("#")[1])).normal(
                                    size=S_pri["clm"].shape))
                            So_, mu_ = Sp[rows], Sp[rows].mean(0, keepdims=True)
                            Sx = So_ - mu_
                            # 两个目标必须与任务匹配：
                            #   center（类中心）→ 识别；net（净信号 ΔZ）→ 检出
                            for Ztgt, Zraw in (("c", Zc), ("n", dZ)):
                                Zx = Zraw - Zraw.mean(0, keepdims=True)
                                try:
                                    if hn == "ridge":
                                        A_ = head_ridge_R(So_, Zraw, lam)
                                        f = lambda s, mu_=mu_, A_=A_: (s - mu_) @ A_
                                    elif hn == "pls":
                                        nc = max(2, min(8, len(rows) - 1))
                                        m = PLSRegression(n_components=nc).fit(Sx, Zx)
                                        f = lambda s, m=m, mu_=mu_: m.predict(s - mu_)
                                    elif hn == "krr":
                                        m = KernelRidge(kernel="rbf", alpha=0.1,
                                                        gamma=1.0 / max(1, Sx.shape[1]))
                                        m.fit(Sx, Zx)
                                        f = lambda s, m=m, mu_=mu_: m.predict(s - mu_)
                                    elif hn == "gp":
                                        m = GaussianProcessRegressor(
                                            kernel=RBF(1.0) + WhiteKernel(1e-3),
                                            normalize_y=True,
                                            random_state=0).fit(Sx, Zx)
                                        f = lambda s, m=m, mu_=mu_: m.predict(s - mu_)
                                    elif hn == "knn1":
                                        m = NearestNeighbors(1, metric="cosine").fit(Sx)
                                        def f(s, m=m, mu_=mu_, Zx=Zx):
                                            _, ix = m.kneighbors(s - mu_)
                                            return Zx[ix[:, 0]]
                                    else:
                                        mm = MLPRegressor((128,), max_iter=600,
                                                          random_state=0).fit(Sx, Zx)
                                        f = lambda s, m=mm, mu_=mu_: m.predict(s - mu_)
                                    hzoo[f"{hn}|{pp}|{Ztgt}"] = (f, Sp)
                                except Exception:
                                    pass

                # ---- 轴3C：条件化打分器 g([z, S_T]) × {clm, rand} ----
                conds = {}
                if do["C"]:
                    rng_c = np.random.default_rng(seed * 131 + 3)
                    # clm 1 次 + rand 多次抽样（随机码本单次抽样会碰巧赢）
                    pnames = ["clm"] + [f"rand384#{di}" for di in range(CRAND)]
                    for p in pnames:
                        pb = p.split("#")[0]
                        Sp = S_pri[pb] if p == "clm" else nrm(
                            np.random.default_rng(1000 * seed + 555
                                                  + int(p.split("#")[1])).normal(
                                size=S_pri["clm"].shape))
                        Xp, yb = build_cond_pairs(Zfit, yfit, Sp, rows, rng_c,
                                                  max_n=1200)
                        # 768 维原始特征 → 统一 PCA-128（加速，且对所有算法一致）
                        npc = min(128, Xp.shape[0] - 1, Xp.shape[1])
                        pca = PCA(n_components=npc).fit(Xp)
                        Xr = pca.transform(Xp)
                        for an in COND_ALGOS:
                            try:
                                m = cond_algo(an, seed).fit(Xr, yb)
                                conds[f"{an}|{p}"] = (m, pca, Sp)
                            except Exception:
                                pass
                    print("     [t] cond %.0fs" % (time.time() - T0), flush=True)

                # ---- 轴1B：算法动物园（原始光谱）----
                sc = StandardScaler().fit(Xtr)
                Xtr_s = sc.transform(Xtr)
                ncls = len(np.unique(ytr))
                t_cls = np.where(np.unique(ytr) == NEW)[0]
                t_cls = int(t_cls[0]) if len(t_cls) else None
                mfac = build_multiclass(seed)
                fitted = {}
                for nm in MULTI:
                    if nm == "cnn1d":
                        continue
                    t1 = time.time()
                    try:
                        if nm == "ncm":
                            fitted[nm] = fit_ncm(Xtr_s, ytr, ncls)
                        elif nm == "plsda":
                            fitted[nm] = fit_plsda(Xtr_s, ytr, ncls)
                        else:
                            fitted[nm] = mfac[nm]().fit(Xtr_s, ytr)
                    except Exception:
                        pass
                    print(f"     [t] fit {nm:12s} {time.time()-t1:6.1f}s", flush=True)
                cnn = safe(fit_cnn, Xtr_s, ytr, ncls, args.epochs, seed)
                ztr = embed(net, np.asarray(Xtr, float))
                zrf = safe(lambda: RandomForestClassifier(
                    300, random_state=seed, n_jobs=-1).fit(ztr, ytr))
                zsv = safe(lambda: SVC(kernel="rbf", C=10, gamma="scale",
                                       random_state=seed).fit(ztr, ytr))

                for n_name, Xn_neg in (("same-matrix", mid_ref[mid_h]),
                                       ("old-heldout", Xte_old)):
                    if len(Xn_neg) < 5:
                        continue
                    Zo = embed(net, np.asarray(Xn_neg, float))
                    Xn_s = sc.transform(Xte_new)
                    Xo_s = sc.transform(np.asarray(Xn_neg, float))
                    arms = {}
                    Zc_n, Zc_o = Zn - mu_mid, Zo - mu_mid

                    # ===== 本文臂 =====
                    hd = heads["clm"]
                    if n_name == "same-matrix":
                        v = Si @ hd["Wn"][NEW]
                        arms["ours-det"] = (Zc_n @ v, Zc_o @ v)
                    else:
                        Wcand = np.concatenate(
                            [hd["W"][:len(others)], hd["W"][NEW:NEW + 1]], 0)
                        Ln, Lo = Zn @ Wcand.T, Zo @ Wcand.T
                        arms["ours-id"] = (Ln[:, iT], Lo[:, iT])
                        arms["ours-id-m"] = (
                            Ln[:, iT] - np.delete(Ln, iT, 1).max(1),
                            Lo[:, iT] - np.delete(Lo, iT, 1).max(1))

                    # ===== 轴2：先验动物园 =====
                    # ⚠️ arms 是字典，随机码本的多次抽样必须用**唯一键**存放，
                    #    否则后面的抽样会覆盖前面的（写入 res 时再剥掉 #编号）
                    if do["P"]:
                        for hkey in heads:
                            p = hkey.split("#")[0]
                            hp = heads[hkey]
                            if n_name == "same-matrix":
                                v = Si @ hp["Wn"][NEW]
                                arms[f"P:{hkey}"] = (Zc_n @ v, Zc_o @ v)
                            else:
                                Wc = np.concatenate(
                                    [hp["W"][:len(others)], hp["W"][NEW:NEW + 1]], 0)
                                Lp, Lpo = Zn @ Wc.T, Zo @ Wc.T
                                arms[f"P:{hkey}"] = (Lp[:, iT], Lpo[:, iT])
                                arms[f"P:{hkey}-m"] = (
                                    Lp[:, iT] - np.delete(Lp, iT, 1).max(1),
                                    Lpo[:, iT] - np.delete(Lpo, iT, 1).max(1))

                    # ===== 轴3H：平滑头动物园 =====
                    if do["P"]:
                        for hkey, (f, Sp) in hzoo.items():
                            hn, p, Zt = hkey.split("|")
                            if (n_name == "same-matrix") != (Zt == "n"):
                                continue          # 目标必须与任务匹配
                            try:
                                w = f(Sp[NEW:NEW + 1])[0]
                                w = unit(w)
                                if w is None:
                                    continue
                                if n_name == "same-matrix":
                                    v = Si @ w
                                    arms[f"H:{hn}|{p}|{Zt}"] = (Zc_n @ v, Zc_o @ v)
                                else:
                                    arms[f"H:{hn}|{p}|{Zt}"] = (Zn @ w, Zo @ w)
                            except Exception:
                                pass

                    # ===== 轴3C：条件化打分器 =====
                    if do["C"]:
                        cand_ix = list(range(len(others))) + [NEW]
                        for key, md in conds.items():
                            an, p = key.split("|")
                            m, pca, Sp = md
                            Sl = Sp[cand_ix]
                            try:
                                Pm_n = cond_score_matrix(m, Zn, Sl, pca)
                                Pm_o = cond_score_matrix(m, Zo, Sl, pca)
                            except Exception:
                                continue
                            if n_name == "old-heldout":
                                arms[f"C:{an}|{p}"] = (Pm_n[:, iT], Pm_o[:, iT])
                                arms[f"C:{an}|{p}-m"] = (
                                    Pm_n[:, iT] - np.delete(Pm_n, iT, 1).max(1),
                                    Pm_o[:, iT] - np.delete(Pm_o, iT, 1).max(1))
                            else:
                                arms[f"C:{an}|{p}"] = (Pm_n[:, iT], Pm_o[:, iT])

                    # ===== 轴1B：算法动物园 =====
                    if do["B"]:
                        for nm, m in fitted.items():
                            try:
                                arms[f"B:{nm}"] = (
                                    multi_score(nm, m, Xn_s, t_cls),
                                    multi_score(nm, m, Xo_s, t_cls))
                            except Exception:
                                pass
                        if cnn is not None:
                            Pn = cnn_score(cnn, Xn_s)
                            if t_cls is not None:
                                arms["B:cnn1d"] = (Pn[:, t_cls],
                                                   cnn_score(cnn, Xo_s)[:, t_cls])
                            else:
                                arms["B:cnn1d"] = (1 - Pn.max(1),
                                                   1 - cnn_score(cnn, Xo_s).max(1))
                        if zrf is not None:
                            Zn_r = embed(net, np.asarray(Xte_new, float))
                            Pn = zrf.predict_proba(Zn_r)
                            Po = zrf.predict_proba(Zo)
                            arms["B:rf_z"] = (Pn[:, t_cls], Po[:, t_cls]) \
                                if t_cls is not None else (1 - Pn.max(1), 1 - Po.max(1))
                        if zsv is not None:
                            Dn = zsv.decision_function(Zn)
                            Do = zsv.decision_function(Zo)
                            arms["B:svm_z"] = (Dn[:, t_cls], Do[:, t_cls]) \
                                if t_cls is not None else (-Dn.max(1), -Do.max(1))
                        # 单类：检出用同基质纯谱队列，识别用老类+基质队列
                        cohort = np.asarray(mid_tr[mid_h], float) \
                            if n_name == "same-matrix" else Xtr
                        if len(cohort) > 10:
                            cohort_s = sc.transform(cohort)
                            ofac = build_oneclass(seed, cohort_s)
                            for nm in ONECLASS:
                                t1 = time.time()
                                try:
                                    if nm == "mahal":
                                        lw = LedoitWolf().fit(cohort_s)
                                        m = dict(mu=lw.location_,
                                                 Winv=np.linalg.pinv(lw.covariance_))
                                    elif nm == "ae1d":
                                        m = fit_ae(cohort_s, args.epochs, seed)
                                    else:
                                        m = ofac[nm]().fit(cohort_s)
                                    arms[f"B:{nm}"] = (oc_score(nm, m, Xn_s, cohort_s),
                                                       oc_score(nm, m, Xo_s, cohort_s))
                                except Exception:
                                    pass
                                print(f"     [t] oc {nm:10s} {time.time()-t1:6.1f}s",
                                      flush=True)

                    for nm, (sn, so) in arms.items():
                        if sn is None or so is None:
                            continue
                        sn = np.asarray(sn, float); so = np.asarray(so, float)
                        if not (np.isfinite(sn).all() and np.isfinite(so).all()):
                            continue
                        m = metrics(sn, so)
                        nm = re.sub(r"#\d+", "", nm)   # 随机码本多次抽样 → 同一个臂
                        bucket = ("B" if nm.startswith("B:") else
                                  "P" if nm.startswith(("P:", "H:")) else
                                  "O" if nm.startswith("ours") else "C")
                        res[bucket].setdefault(tag, {}).setdefault(
                            n_name, {}).setdefault(nm, {}).setdefault(
                            e_hold, []).append(dict(auc=m["auc"], tpr5=m["tpr5"]))

        print(f"[{e_hold:>7}] done", flush=True)

    with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[exp81] 已写出 out/{args.out}", flush=True)


if __name__ == "__main__":
    main()
