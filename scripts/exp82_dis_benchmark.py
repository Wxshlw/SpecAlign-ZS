"""exp82 —— 定向归因口径下的三轴基准（化学先验必须外推到未见分子）

为什么换口径：exp81 在 old-heldout 宏观 AUC（闭集 argmax）上发现 one-hot 反而最好，
随机码本也能到 0.92 —— 那是「记住类身份」就够的任务，检验不出先验的外推能力。
本实验把三轴整体搬到 exp78 的**定向归因**口径：
    U = 3 个未见实体（训练集完全不含）
    T, D ∈ U；判别函数 s_T − s_D
    auc_dis = AUC(s_T−s_D on T 谱, s_T−s_D on D 谱)，>0.5 才叫「指名正确」
    top-1   = 在 U 三候选中 argmax

三轴
  A 先验动物园：10 种先验 × 同一岭回归头（λ 网格，主口径 1e-4）
  B 算法动物园：25 个普通 ML/DL（原始光谱，k=0 不含 U 标样）
  C 先验 × 算法：条件化打分器 g([z, S_c]) × 6 算法 × 5 先验

公平性
  - 所有先验统一 384 维、行归一化（exp81a）
  - 随机码本多次抽样，每次单独算 auc 再平均
  - λ 主口径固定 1e-4（exp53/58d）；另报 λ-oracle 上界作诊断
  - 普通算法在 k=0 的打分 f(x)=1−maxprob(x) 不依赖候选集 ⇒ top-1 无输出（undefined），
    只报 auc_dis 并按「伪定向」标注
"""
from __future__ import annotations
import os, sys, json, argparse, itertools, time
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "论文8"))
import p6_g1_data as G1
from exp63_p6_screening_protocol import (stage1, embed, head_ridge_R, nrm,
                                         auc_of, EID2MID)
from exp68_p6_cond_scorer import split_by_group
from exp71_p6_detection_fix import load_groups_cached
from exp78_p6_attribution_proof import split_group_w
from exp81b_p6_benchmark_zoo import (
    build_multiclass, multi_score, fit_ncm, fit_plsda,
    build_oneclass, oc_score, fit_cnn, cnn_score, fit_ae, ae_score,
    build_cond_pairs, cond_algo, cond_score_matrix,
    MULTI, ONECLASS, COND_ALGOS, PRIORS, DEV)

OUT = os.path.join(os.path.dirname(HERE), "out")
OLD_FRAC = 0.7
COND_PRIORS = ["clm", "ecfp384", "physchem", "onehot", "rand384"]
NRAND = 10          # 随机码本定向判别抽样次数
H_ALGOS = ["ridge", "pls", "krr", "knn1", "mlp"]


def make_h(hn, Sx, Zx, lam):
    """把先验装到普通算法上：算法学 S(分子) → Z(谱表示) 的映射，再外推到候选。"""
    if hn == "ridge":
        A = head_ridge_R(Sx, Zx, lam)
        return lambda S: S @ A
    if hn == "pls":
        nc = max(2, min(8, len(Sx) - 1))
        m = PLSRegression(n_components=nc).fit(Sx, Zx)
        return lambda S: m.predict(S)
    if hn == "krr":
        m = KernelRidge(kernel="rbf", alpha=0.1,
                        gamma=1.0 / max(1, Sx.shape[1])).fit(Sx, Zx)
        return lambda S: m.predict(S)
    if hn == "knn1":
        m = NearestNeighbors(n_neighbors=1, metric="cosine").fit(Sx)

        def f(S, m=m, Zx=Zx):
            _, ix = m.kneighbors(S)
            return Zx[ix[:, 0]]
        return f
    if hn == "mlp":
        m = MLPRegressor(hidden_layer_sizes=(128,), max_iter=500,
                         random_state=0).fit(Sx, Zx)
        return lambda S: m.predict(S)
    if hn == "gp":
        m = GaussianProcessRegressor(
            kernel=RBF(1.0) + WhiteKernel(1e-3), normalize_y=True,
            random_state=0, n_restarts_optimizer=0).fit(Sx, Zx)
        return lambda S: m.predict(S)
    raise ValueError(hn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--nrep", type=int, default=4)
    ap.add_argument("--nhu", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lams", default="0.0001,0.001,0.01,0.1,1")
    ap.add_argument("--axes", default="ABC")
    ap.add_argument("--maxn", type=int, default=3000)
    ap.add_argument("--eids", default="")
    ap.add_argument("--priors", default="",
                    help="逗号分隔的先验子集（空=全部 PRIORS）")
    ap.add_argument("--fuse", action="store_true",
                    help="追加先验拼接臂（检验 CLM 与普通先验是否互补）")
    ap.add_argument("--out", default="exp82_dis.json")
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(",")]
    axes = set(args.axes.upper())

    ent, conc, bg, _ = G1.load_g1_conc()
    GR = load_groups_cached()
    eids = sorted(set(GR) & set(EID2MID))
    if args.eids:
        eids = [e for e in eids if e in args.eids.split(",")]
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]

    zc = np.load(os.path.join(OUT, "exp81_prior_cache.npz"), allow_pickle=True)
    assert list(zc["keys"]) == keys_all, "先验缓存 key 顺序不一致，禁止对齐！"
    S_pri = {n: np.asarray(zc[n], float) for n in PRIORS}
    dz = S_pri["clm"].shape[1]
    ki = {k: i for i, k in enumerate(keys_all)}

    # ---- 先验拼接臂：检验化学语义与普通指纹是否互补 ----
    pri_list = list(PRIORS) if not args.priors else [
        x.strip() for x in args.priors.split(",") if x.strip()]
    if args.fuse:
        FUSE = [("clm", "charbag"), ("clm", "ecfp2048"), ("clm", "physchem"),
                ("clm", "charbag", "ecfp2048"), ("charbag", "ecfp2048"),
                ("clm", "ecfp384", "maccs")]
        for combo in FUSE:
            nm = "+".join(combo)
            S_pri[nm] = np.concatenate([S_pri[c] for c in combo], 1)
            pri_list.append(nm)

    res = {"A": {}, "B": {}, "C": {}, "META": dict(
        seeds=args.seeds, seed0=args.seed0, nrep=args.nrep, nhu=args.nhu,
        lams=lams, priors=pri_list, cond_priors=COND_PRIORS,
        cond_algos=COND_ALGOS, multi=MULTI, oneclass=ONECLASS,
        axes="".join(sorted(axes)), note="auc_dis: T/D both unseen")}

    def add(bucket, arm, kind, key, rec):
        res[bucket].setdefault(arm, {}).setdefault(kind, {}) \
           .setdefault(key, []).append(rec)

    combos = list(itertools.combinations(eids, args.nhu))
    print(f"[exp82] device={DEV} eids={eids} |U|={args.nhu} 组合={len(combos)} "
          f"reps={args.nrep} seeds={args.seeds} axes={axes}", flush=True)

    for si in range(args.seed0, args.seed0 + args.seeds):
        rng0 = np.random.default_rng(si * 7919 + 13)
        cis = rng0.permutation(len(combos))[:args.nrep]
        for ci in cis:
            U = list(combos[ci])
            others = [e for e in sorted(ent) if e not in U]
            order = ([ki[e] for e in others]
                     + [ki[f"M_{m}"] for m in mids]
                     + [ki[u] for u in U])
            S_all = {p: S_pri[p][order] for p in pri_list}
            NEW0 = len(others) + len(mids)
            iu = {u: NEW0 + j for j, u in enumerate(U)}
            cand_ix = [iu[u] for u in U]
            nrows = NEW0 + len(U)

            # ---- 划分（与 exp78 完全一致）----
            rng = np.random.default_rng(si * 104729 + ci * 37 + 7)
            Xte_U, Xtr_U = {}, {}
            for u in U:
                Xe = np.asarray(ent[u], float)
                te_ix, pool_ix = split_by_group(Xe, GR[u]["gid"], rng)
                Xte_U[u] = Xe[te_ix]
                Xtr_U[u] = Xe[pool_ix]
            parts, ys = [], []
            for j, e in enumerate(others):
                Xo = np.asarray(ent[e], float)
                if e in GR:
                    a, b, _ = split_group_w(Xo, GR[e]["gid"], rng)
                else:
                    pp = rng.permutation(len(Xo))
                    cut = max(2, int(round(OLD_FRAC * len(Xo))))
                    a, b = pp[:cut], pp[cut:]
                parts.append(Xo[a]); ys.append(np.full(len(a), j))
            for j, m in enumerate(mids):
                Xo = np.asarray(bg[m], float)
                pp = rng.permutation(len(Xo))
                cut = max(2, int(round(OLD_FRAC * len(Xo))))
                parts.append(Xo[pp[:cut]])
                ys.append(np.full(cut, len(others) + j))
            Xtr = np.concatenate(parts).astype(np.float32)
            ytr = np.concatenate(ys).astype(np.int64)
            rows = sorted({int(y) for y in ytr})
            pairs = list(itertools.combinations(U, 2))
            tag = f"seed{si}_c{ci}"

            # ---- stage1（与先验无关，只训一次）----
            t0 = time.time()
            net, _ = stage1(Xtr, ytr, dz, nrows, args.steps, si * 1000 + ci,
                            S=S_all["clm"])
            Zfit = embed(net, Xtr)
            Z_old = np.stack([Zfit[ytr == j].mean(0) for j in rows])
            Zte = {u: embed(net, Xte_U[u]) for u in U}
            print(f"  [{tag}] U={'-'.join(U)} stage1 {time.time()-t0:.1f}s "
                  f"n_tr={len(Xtr)}", flush=True)

            # ================= 轴 A：先验动物园 =================
            if "A" in axes:
                for p in pri_list:
                    Sp = S_all[p]
                    S_o = Sp[rows]
                    mu = S_o.mean(0, keepdims=True)
                    Sc = Sp[cand_ix]
                    for lam in lams:
                        Wc = (Sc - mu) @ head_ridge_R(S_o, Z_old, lam)
                        arm = f"A:{p}|lam{lam:g}"
                        for jT, jD in itertools.combinations(range(len(U)), 2):
                            T, D = U[jT], U[jD]
                            dT = Zte[T] @ Wc[jT] - Zte[T] @ Wc[jD]
                            dD = Zte[D] @ Wc[jT] - Zte[D] @ Wc[jD]
                            a = float(auc_of(dT, dD))
                            add("A", arm, "dis", f"{T}|{D}",
                                dict(auc_dis=a, sep=abs(a - .5) * 2, tag=tag))
                        for j, u in enumerate(U):
                            L = Zte[u] @ Wc.T
                            pred = np.array(U)[L.argmax(1)]
                            add("A", arm, "top1", u,
                                dict(top1=float(np.mean(pred == u)),
                                     n=len(Zte[u]), tag=tag))
                # 随机码本（多次抽样，各自算 auc 再平均）
                for rr in range(NRAND):
                    r = np.random.default_rng(si * 100003 + ci * 977 + rr * 97 + 5)
                    R = nrm(r.normal(size=(len(keys_all), dz)).astype(np.float32))
                    Rc = R[cand_ix]
                    for jT, jD in itertools.combinations(range(len(U)), 2):
                        T, D = U[jT], U[jD]
                        dT = Zte[T] @ Rc[jT] - Zte[T] @ Rc[jD]
                        dD = Zte[D] @ Rc[jT] - Zte[D] @ Rc[jD]
                        a = float(auc_of(dT, dD))
                        add("A", "A:rand384", "dis", f"{T}|{D}",
                            dict(auc_dis=a, sep=abs(a - .5) * 2, tag=tag))
                    for j, u in enumerate(U):
                        L = Zte[u] @ Rc.T
                        pred = np.array(U)[L.argmax(1)]
                        add("A", "A:rand384", "top1", u,
                            dict(top1=float(np.mean(pred == u)),
                                 n=len(Zte[u]), tag=tag))

            # ================= 轴 B：算法动物园 =================
            if "B" in axes:
                sc = StandardScaler().fit(Xtr)
                Xtr_s = sc.transform(Xtr)
                TE_s = {u: sc.transform(Xte_U[u]) for u in U}
                cohort_s = Xtr_s[ytr >= len(others)].astype(np.float64)
                fitfac = build_multiclass(si)
                for nm in MULTI:
                    t1 = time.time()
                    try:
                        if nm == "ncm":
                            m = fit_ncm(Xtr_s, ytr, nrows)
                        elif nm == "plsda":
                            m = fit_plsda(Xtr_s, ytr, nrows)
                        elif nm == "cnn1d":
                            m = fit_cnn(Xtr_s, ytr, nrows, args.epochs, si)
                        else:
                            m = fitfac[nm]().fit(Xtr_s, ytr)

                        def score(X, nm=nm, m=m):
                            if nm == "cnn1d":
                                P = cnn_score(m, sc.transform(X))
                                return 1 - P[:, :nrows].max(1)
                            return multi_score(nm, m, sc.transform(X))
                        for T, D in pairs:
                            sT, sD = score(Xte_U[T]), score(Xte_U[D])
                            a = float(auc_of(sT, sD))
                            add("B", f"B:{nm}", "dis", f"{T}|{D}",
                                dict(auc_dis=a, sep=abs(a - .5) * 2,
                                     tag=tag, todo="pseudo_dis"))
                    except Exception as ex:
                        print(f"    B:{nm} skip {type(ex).__name__}: {ex}",
                              flush=True)
                ofac = build_oneclass(si, cohort_s)
                for nm in ONECLASS:
                    try:
                        if nm == "mahal":
                            lw = LedoitWolf().fit(cohort_s)
                            m = dict(mu=lw.location_,
                                     Winv=np.linalg.pinv(lw.covariance_))
                        elif nm == "ae1d":
                            m = fit_ae(cohort_s, args.epochs, si)
                        else:
                            m = ofac[nm]().fit(cohort_s)

                        def oscore(X, nm=nm, m=m):
                            return oc_score(nm, m, sc.transform(X), cohort_s)
                        for T, D in pairs:
                            a = float(auc_of(oscore(Xte_U[T]), oscore(Xte_U[D])))
                            add("B", f"B:{nm}", "dis", f"{T}|{D}",
                                dict(auc_dis=a, sep=abs(a - .5) * 2,
                                     tag=tag, todo="pseudo_dis"))
                    except Exception as ex:
                        print(f"    B:{nm} skip {type(ex).__name__}: {ex}",
                              flush=True)
                print(f"  [{tag}] 轴B 完成 {time.time()-t1:.1f}s", flush=True)

            # ================= 轴 C：先验 × 普通算法（算法作用于先验侧）=================
            # 算法学 S(分子) → Z(谱表示)，再对候选分子外推；ridge 即本文头，
            # 其余为"把同一先验装到普通算法上"的忠实实现。
            if "C" in axes:
                t2 = time.time()
                Zc_ = Z_old - Z_old.mean(0, keepdims=True)
                for p in pri_list:
                    Sp = S_all[p]
                    S_o = Sp[rows]
                    mu = S_o.mean(0, keepdims=True)
                    Sx, Sc = S_o - mu, Sp[cand_ix] - mu
                    for hn in H_ALGOS:
                        try:
                            f = make_h(hn, Sx, Zc_, lams[0])
                            Wc = np.asarray(f(Sc), float)
                            if Wc.shape != (len(U), dz) or not np.isfinite(Wc).all():
                                continue
                        except Exception as ex:
                            print(f"    C:{hn}|{p} skip {type(ex).__name__}: {ex}",
                                  flush=True)
                            continue
                        arm = f"C:{hn}|{p}"
                        for jT, jD in itertools.combinations(range(len(U)), 2):
                            T, D = U[jT], U[jD]
                            dT = Zte[T] @ Wc[jT] - Zte[T] @ Wc[jD]
                            dD = Zte[D] @ Wc[jT] - Zte[D] @ Wc[jD]
                            a = float(auc_of(dT, dD))
                            add("C", arm, "dis", f"{T}|{D}",
                                dict(auc_dis=a, sep=abs(a - .5) * 2, tag=tag))
                        for j, u in enumerate(U):
                            L = Zte[u] @ Wc.T
                            pred = np.array(U)[L.argmax(1)]
                            add("C", arm, "top1", u,
                                dict(top1=float(np.mean(pred == u)),
                                     n=len(Zte[u]), tag=tag))
                print(f"  [{tag}] 轴C 完成 {time.time()-t2:.1f}s", flush=True)

            # ================= 轴 D：条件化打分器（失效对照）=================
            if "D" in axes:
                t2 = time.time()
                for p in COND_PRIORS:
                    Sp = S_all[p]
                    Sc = Sp[cand_ix]
                    rc = np.random.default_rng(si * 131 + ci * 17 + 3)
                    Xp, yb = build_cond_pairs(Zfit, ytr, Sp, rows, rc,
                                              nneg=1, max_n=args.maxn)
                    for an in COND_ALGOS:
                        try:
                            m = cond_algo(an, si).fit(Xp, yb)
                            Pm = {u: cond_score_matrix(m, Zte[u], Sc) for u in U}
                        except Exception as ex:
                            print(f"    D:{an}|{p} skip {type(ex).__name__}",
                                  flush=True)
                            continue
                        arm = f"D:{an}|{p}"
                        for jT, jD in itertools.combinations(range(len(U)), 2):
                            T, D = U[jT], U[jD]
                            dT = Pm[T][:, jT] - Pm[T][:, jD]
                            dD = Pm[D][:, jT] - Pm[D][:, jD]
                            a = float(auc_of(dT, dD))
                            add("D", arm, "dis", f"{T}|{D}",
                                dict(auc_dis=a, sep=abs(a - .5) * 2, tag=tag))
                        for j, u in enumerate(U):
                            pred = np.array(U)[Pm[u].argmax(1)]
                            add("D", arm, "top1", u,
                                dict(top1=float(np.mean(pred == u)),
                                     n=len(Zte[u]), tag=tag))
                print(f"  [{tag}] 轴D 完成 {time.time()-t2:.1f}s", flush=True)

    path = os.path.join(OUT, args.out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    print(f"[exp82] 已写出 {path}", flush=True)


if __name__ == "__main__":
    main()
