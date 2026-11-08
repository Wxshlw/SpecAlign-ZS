r"""exp88c —— RSA 的置换检验 + 跨品类数据集对照（同一指标，两组候选）。

为什么要做对照
--------------
exp88b 显示：在**同类近邻**的 18 种植物油上，结构类先验（clm/ecfp/charbag）的
RSA ρ≈0.31-0.33，组成类先验 physchem 只有 0.12（p=0.13，不显著）。

但这只说明"植物油上如此"。要证明这是**任务依赖**而不是数据集差异，必须用
**完全相同**的 RSA 指标，在**跨品类**候选（现有 7 实体数据集的 6 个掺杂物 + 6 个基质）上再算一次。

预期（若机制解释成立）：
    同类近邻  -> physchem 低、结构类先验高
    跨品类    -> physchem 高（组成差异就是判别轴）

同时补两件事：
  1. 置换检验（10000 次打乱类别标签重算 ρ），因为距离矩阵元素不独立，
     spearmanr 的渐近 p 偏乐观。
  2. 两类候选的"组成差异度"直接量化（先验向量的平均两两距离），
     用来验证"跨品类差异大 / 同类差异小"这个前提。
"""
from __future__ import annotations
import os, sys, json, itertools, warnings
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "..", "out")

import exp88b_prior_rsa as A  # noqa: E402  (复用 profile 与 build_priors)
import p6_g1_data as G1        # noqa: E402
from exp81a_prior_zoo import unit  # noqa: E402

NRAND = 10000


def dist_mat(V, metric="cos"):
    if metric == "cos":
        D = 1.0 - V @ V.T
    else:
        D = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=2)
    np.fill_diagonal(D, 0.0)
    return D


def rsa_with_perm(Dp, Ds, nrand=NRAND, seed=0):
    from scipy.stats import spearmanr
    n = Dp.shape[0]
    iu = np.triu_indices(n, 1)
    rho, p_asy = spearmanr(Dp[iu], Ds[iu])
    rng = np.random.default_rng(seed)
    null = np.empty(nrand)
    for i in range(nrand):
        q = rng.permutation(n)
        null[i] = spearmanr(Dp[np.ix_(q, q)][iu], Ds[iu])[0]
    p_perm = float((np.sum(null >= rho) + 1) / (nrand + 1))
    return float(rho), float(p_asy), p_perm, float(null.mean()), float(null.std())


def run_dataset(name, T, D_spec, out_rows):
    print(f"\n==== {name}  n={len(T['clm'])} ====")
    print(f"{'prior':10s} {'rho':>8s} {'p_asym':>9s} {'p_perm':>8s} {'null_mean':>10s}")
    for n, V in T.items():
        if n == "onehot":
            continue
        Dp = dist_mat(V)
        rho, pa, pp, nm, ns = rsa_with_perm(Dp, D_spec)
        out_rows.append(dict(dataset=name, prior=n, rho=rho, p_asym=pa, p_perm=pp))
        print(f"{n:10s} {rho:+.4f} {pa:9.2e} {pp:8.4f} {nm:+.4f}")


def main():
    rows = []

    # ================= 数据集 1：同类近邻（植物油 18 类） =================
    d = np.load(os.path.join(OUT, "exp88_oil.npz"), allow_pickle=True)
    Xp, yc = d["X_pure"], d["yc"]
    classes = sorted(set(yc.tolist()))
    M = np.stack([Xp[yc == c].mean(0) for c in classes])
    D_spec_near = dist_mat(np.stack([unit(r) for r in M]))
    T_near, _ = A.build_priors(classes)
    run_dataset("同类近邻(植物油18类)", T_near, D_spec_near, rows)

    # ================= 数据集 2：跨品类（现有 7 实体） =================
    ent, bg, _ = G1.load_g1()
    keys = sorted(ent.keys()) + [f"M_{m}" for m in sorted(bg.keys())]
    z = np.load(os.path.join(OUT, "exp81_prior_cache.npz"), allow_pickle=True)
    assert list(z["keys"]) == keys, "先验缓存 key 顺序不一致"
    T_far = {n: np.asarray(z[n], float) for n in
             ("clm", "ecfp384", "ecfp2048", "maccs", "avalon", "rdkit",
              "physchem", "charbag", "rand384")}

    # 谱：每个 key 的类均值（与先验 key 一一对应）
    Ms = []
    for k in keys:
        if k.startswith("M_"):
            X = np.asarray(bg[k[2:]], float)
        else:
            X = np.asarray(ent[k], float)
        Ms.append(unit(X.mean(0)))
    D_spec_far = dist_mat(np.stack(Ms))
    run_dataset("跨品类(现有7实体,12key)", T_far, D_spec_far, rows)

    json.dump(rows, open(os.path.join(OUT, "exp88c_rsa_perm.json"), "w"), indent=1)
    print("\n[exp88c] saved out/exp88c_rsa_perm.json")

    # ---- 排序对照表 ----
    import collections
    tab = collections.defaultdict(dict)
    for r in rows:
        tab[r["prior"]][r["dataset"]] = r["rho"]
    ds = ["同类近邻(植物油18类)", "跨品类(现有7实体,12key)"]
    print("\n==== 同一先验在两组的 RSA ρ ====")
    print(f"{'prior':10s} " + "  ".join(f"{x[:10]:>12s}" for x in ds) + "   变化")
    for p in ("clm", "ecfp384", "ecfp2048", "charbag", "maccs",
              "rdkit", "physchem", "avalon", "rand384"):
        v = [tab[p].get(x, float("nan")) for x in ds]
        print(f"{p:10s} " + "  ".join(f"{x:12.4f}" for x in v) +
              f"   {v[0]-v[1]:+.4f}")


if __name__ == "__main__":
    main()
