"""exp51 —— 定向筛查的**对称性检验**：把 exp50 的「原理上做不到」钉死

=============================================================================
exp50 已证：k=0 时本文 auc_dis=0.845，外部四个方法 0.53–0.57（全在随机）。
但 exp50 只跑了 (T→D) 单向。严格的审稿人会追问：

    「0.53 是不是只是因为你挑的 D 恰好和 T 难分？」

=============================================================================
本实验的判别逻辑（几乎无假设，纯结构性）
=============================================================================
对同一个**无序对 {X, Y}**，把两个方向都跑一遍：
    auc_dis(X→Y) = AUC( X 的样本 vs Y 的样本，用「X 定向评分」 )
    auc_dis(Y→X) = AUC( Y 的样本 vs X 的样本，用「Y 定向评分」 )

**通用异常检测的评分不依赖于你在怀疑谁**——它只有一个「离库距离」。
    记 a = 异常分。则
        auc_dis(X→Y) = P(a(X) > a(Y))
        auc_dis(Y→X) = P(a(Y) > a(X)) = 1 − P(a(X) > a(Y))
    ⇒ **两者之和恒等于 1.0，因此 min(两个方向) 恒 ≤ 0.5**。
      这是恒等式，不是实验结论——无论 X、Y 是谁都成立。

**本文的化学定向评分依赖于怀疑对象**（换一个怀疑分子就换一个头）：
    ⇒ 两个方向可以**同时 > 0.5**，min 可以 > 0.5。

★ 主统计量：min_dir = min(auc_dis(X→Y), auc_dis(Y→X))
   - 外部方法：由恒等式 min_dir ≤ 0.5（实测应 ≈ 0.5 − |Δ|）
   - 本文：若 min_dir 显著 > 0.5 ⇒ 定向识别能力成立

辅助统计量：sum_dir = 两者之和（外部恒 = 1，本文可 > 1）

=============================================================================
效率：{X,Y} 两个方向的训练集相同（都排除 X 和 Y）⇒ **训练一次，双向评估**
=============================================================================
分类头做成 19 行：11 个老实体 + 6 个基质 + X + Y
  行 17 = X 的化学外推头，行 18 = Y 的化学外推头
  ⇒ 一次前向即可同时得到两个方向的定向评分

预注册判据
----------
- H51a（主）：本文 `ridge` 的 min_dir 显著 > 0.5（单样本 t 检验，n = 无序对数）
- H51b（主）：所有外部方法的 min_dir ≤ 0.5，且 sum_dir ≈ 1.0（恒等式自检）
- H51c：本文 min_dir − 最强外部 min_dir ≥ 0.15，p < 0.01
"""
from __future__ import annotations
import os, sys, json, argparse, itertools, warnings
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


def cos_to(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T


@torch.no_grad()
def macro_recall(net, W, Xv, yv, rows):
    z = F.normalize(net(torch.tensor(Xv, dtype=torch.float32, device=DEV)), dim=1)
    Wt = torch.tensor(nrm(W), dtype=torch.float32, device=DEV)
    pred = (z @ Wt.T)[:, rows].argmax(1).cpu().numpy()
    rs = [float((pred[yv == j] == i).mean()) for i, j in enumerate(rows)
          if (yv == j).sum()]
    return float(np.mean(rs)) if rs else 0.0


@torch.no_grad()
def cols(net, W, X, c1, c2):
    z = F.normalize(net(torch.tensor(X, dtype=torch.float32, device=DEV)), dim=1)
    Wt = torch.tensor(nrm(W), dtype=torch.float32, device=DEV)
    lg = (z @ Wt.T).cpu().numpy()[:, :W.shape[0]]
    return lg[:, c1], lg[:, c2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.001,0.01,0.1,1")
    ap.add_argument("--maxpairs", type=int, default=0)
    ap.add_argument("--eids", default="")
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(",")]

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

    pairs = [tuple(sorted(p)) for p in itertools.combinations(eids, 2)]
    if args.maxpairs:
        rngp = np.random.default_rng(7)
        idx = rngp.choice(len(pairs), min(args.maxpairs, len(pairs)), replace=False)
        pairs = [pairs[int(i)] for i in sorted(idx)]

    ARMS = ["ridge", "rf", "svm-rbf", "knn-cos", "centroid-cos"]
    res = {"acc": {a: {} for a in ARMS},
           "meta": {"seeds": args.seeds, "n_pairs": len(pairs), "arms": ARMS,
                    "eids": eids}}
    print(f"[exp51] device={DEV} 无序对={len(pairs)} seeds={args.seeds}", flush=True)

    for pi, (X, Y) in enumerate(pairs):
        others = [e for e in eids if e not in (X, Y)]
        n_old = len(others) + len(mids)
        cX, cY = n_old, n_old + 1          # X→行17, Y→行18
        S = np.asarray(T_all)[[ki[e] for e in others]
                              + [ki[f"M_{m}"] for m in mids]
                              + [ki[X], ki[Y]]]
        XX, XY = np.asarray(ent[X], float), np.asarray(ent[Y], float)

        for seed in range(args.seeds):
            rng = np.random.default_rng(seed * 7919 + 13)
            # 两个实体各自 hold out 30% 作测试（比例一致，避免样本数不等造成偏差）
            nX = int(min(max(5, round(0.3 * len(XX))), len(XX) - 1))
            nY = int(min(max(5, round(0.3 * len(XY))), len(XY) - 1))
            pX = rng.permutation(len(XX)); pY = rng.permutation(len(XY))
            Xte_X, Xte_Y = XX[pX[:nX]], XY[pY[:nY]]

            parts, ys, refs = [], [], []
            for j, e in enumerate(others):
                Xo = np.asarray(ent[e], float)
                pp = rng.permutation(len(Xo))
                ntr = max(2, int(round(OLD_FRAC * len(Xo))))
                parts.append(Xo[pp[:ntr]]); ys.append(np.full(ntr, j))
                refs.append(Xo[pp[ntr:]])
            for j, m in enumerate(mids):
                Xo = np.asarray(bg[m], float)
                pp = rng.permutation(len(Xo))
                ntr = max(2, int(round(OLD_FRAC * len(Xo))))
                parts.append(Xo[pp[:ntr]])
                ys.append(np.full(ntr, len(others) + j))
                refs.append(Xo[pp[ntr:]])
            Xtr = np.concatenate(parts).astype(np.float32)
            ytr = np.concatenate(ys).astype(np.int64)

            # ---------- 外部基线（异常分，方向无关）----------
            sc = StandardScaler().fit(Xtr)
            Xtr_s = sc.transform(Xtr)
            sx, sy = sc.transform(Xte_X), sc.transform(Xte_Y)
            rf = RandomForestClassifier(n_estimators=300, random_state=0,
                                        n_jobs=-1).fit(Xtr_s, ytr)
            sv = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                     random_state=0).fit(Xtr_s, ytr)
            cent = np.stack([Xtr_s[ytr == j].mean(0)
                             for j in np.unique(ytr)])
            anom = {
                "rf": (1 - rf.predict_proba(sx).max(1),
                       1 - rf.predict_proba(sy).max(1)),
                "svm-rbf": (1 - sv.predict_proba(sx).max(1),
                            1 - sv.predict_proba(sy).max(1)),
                "knn-cos": (1 - cos_to(sx, Xtr_s).max(1),
                            1 - cos_to(sy, Xtr_s).max(1)),
                "centroid-cos": (1 - cos_to(sx, cent).max(1),
                                 1 - cos_to(sy, cent).max(1)),
            }
            for nm, (ax, ay) in anom.items():
                fwd = auc_of(ax, ay)      # X→Y
                bwd = auc_of(ay, ax)      # Y→X
                res["acc"][nm].setdefault(f"{X}|{Y}", []).append(dict(
                    fwd=round(float(fwd), 4), bwd=round(float(bwd), 4),
                    min_dir=round(float(min(fwd, bwd)), 4),
                    sum_dir=round(float(fwd + bwd), 4)))

            # ---------- 本文：训练一次，双向评估 ----------
            rng_v = np.random.default_rng(seed * 31 + 7)
            val_mask = np.zeros(len(Xtr), bool)
            for j in sorted({int(y) for y in ytr}):
                ix = rng_v.permutation(np.where(ytr == j)[0])
                nv = min(max(3, int(0.2 * len(ix))), max(0, len(ix) - 1))
                val_mask[ix[:nv]] = True
            Xfit, yfit = Xtr[~val_mask], ytr[~val_mask]
            Xval, yval = Xtr[val_mask], ytr[val_mask]
            val_rows = sorted({int(y) for y in yval})

            net, _ = stage1(Xfit, yfit, dim, n_old, args.steps,
                            seed * 1000 + pi, S=S)
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
            W_chem = (S - mu_S) @ head_ridge_R(S_old, Z_old, lam_b)
            # X→Y：X 的样本在「X 行」上的得分 vs Y 的样本在「X 行」上的得分
            sx_c, _ = cols(net, W_chem, Xte_X, cX, cY)
            sy_c, _ = cols(net, W_chem, Xte_Y, cX, cY)
            # Y→X：Y 的样本在「Y 行」上的得分 vs X 的样本在「Y 行」上的得分
            sy_c2, sx_c2 = cols(net, W_chem, Xte_Y, cY, cX)
            sx_c2b, _ = cols(net, W_chem, Xte_X, cY, cX)
            fwd = auc_of(sx_c, sy_c)          # X→Y
            bwd = auc_of(sy_c2, sx_c2b)       # Y→X
            res["acc"]["ridge"].setdefault(f"{X}|{Y}", []).append(dict(
                fwd=round(float(fwd), 4), bwd=round(float(bwd), 4),
                min_dir=round(float(min(fwd, bwd)), 4),
                sum_dir=round(float(fwd + bwd), 4)))

        if (pi + 1) % 10 == 0:
            print(f"  ...{pi + 1}/{len(pairs)} 对", flush=True)

    with open(os.path.join(OUT, "exp51_symmetry.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("[exp51] 已写出 out/exp51_symmetry.json")
    write_report(res, ARMS)


def write_report(res, ARMS):
    try:
        from scipy.stats import ttest_1samp, ttest_rel
        HAS = True
    except Exception:
        HAS = False

    def M(a, k):
        out = {}
        for p, lst in res["acc"].get(a, {}).items():
            v = [d[k] for d in lst if d[k] is not None]
            if v:
                out[p] = float(np.mean(v))
        return out

    L = []
    w = L.append
    w("# exp51 —— 定向筛查的**对称性检验**（无序对 {X,Y} 双向评估）\n")
    w("`fwd` = auc_dis(X→Y)，`bwd` = auc_dis(Y→X)，"
      "`min_dir` = min(fwd, bwd)，`sum_dir` = fwd + bwd。\n")
    w("★ **通用异常检测的评分不依赖怀疑对象** ⇒ fwd + bwd **恒等于 1.0**"
      " ⇒ min_dir **恒 ≤ 0.5**（恒等式，非实验结论）。\n")
    w("★ 本文的化学头随怀疑对象改变 ⇒ 两个方向可**同时 > 0.5** ⇒ min_dir 可 > 0.5。\n")

    w("\n## 主结果\n")
    w("| 方法 | min_dir 均值 | vs 0.5 (p) | sum_dir 均值 | fwd 均值 | bwd 均值 | n 对 |")
    w("|---|---|---|---|---|---|---|")
    base = {}
    for a in ARMS:
        d = M(a, "min_dir"); s = M(a, "sum_dir")
        f = M(a, "fwd"); b = M(a, "bwd")
        if not d:
            continue
        v = np.asarray(list(d.values()), float)
        p = ttest_1samp(v, 0.5).pvalue if (HAS and len(v) > 1) else float("nan")
        base[a] = (v, d)
        w(f"| `{a}` | {np.mean(v):.3f} | {p:.4f} | "
          f"{np.mean(list(s.values())):.3f} | {np.mean(list(f.values())):.3f} | "
          f"{np.mean(list(b.values())):.3f} | {len(v)} |")

    w("\n## 判据\n")
    rv, rd = base.get("ridge", (None, {}))
    if rv is not None:
        p = ttest_1samp(rv, 0.5).pvalue
        w(f"- **H51a**：ridge min_dir = {np.mean(rv):.3f}，vs 0.5 p={p:.6f} "
          f"⇒ {'成立' if np.mean(rv) > 0.5 and p < 0.05 else '不成立'}")
        frac = float((rv > 0.5).mean())
        w(f"   - min_dir > 0.5 的无序对占比 = {frac:.1%}（{int((rv > 0.5).sum())}/{len(rv)}）")
    for a in ARMS:
        if a == "ridge" or a not in base:
            continue
        v, d = base[a]
        w(f"   - **H51b** `{a}`：min_dir {np.mean(v):.3f}（应 ≤0.5），"
          f"sum_dir {np.mean(list(M(a, 'sum_dir').values())):.3f}（应 ≈1.0）")
    if rv is not None and HAS:
        for a in ARMS:
            if a == "ridge" or a not in base:
                continue
            v, d = base[a]
            common = sorted(set(rd) & set(d))
            if len(common) > 3:
                x = np.array([d[p_] for p_ in common])
                y = np.array([rd[p_] for p_ in common])
                w(f"   - **H51c** ridge − {a}: Δ={np.mean(y - x):+.3f}, "
                  f"p={ttest_rel(y, x).pvalue:.6f} (n={len(common)})")

    with open(os.path.join(OUT, "exp51_p6_symmetry.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("[exp51] 已写出 out/exp51_p6_symmetry.md")


if __name__ == "__main__":
    main()
