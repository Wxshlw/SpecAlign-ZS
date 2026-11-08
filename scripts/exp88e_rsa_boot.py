r"""exp88e —— RSA 稳健性：类别 bootstrap + 合并同类油等级的敏感性分析。

两个必须排除的隐患
------------------
1. **橄榄油三兄弟稀释 ρ**
   纯油谱里 Class 1/2/3 都是橄榄油（Extra Virgin / Extra Light / Pure）。
   它们的化学组成（脂肪酸谱）几乎相同，我给的是**同一个 profile**，
   所以任何化学先验都无法把它们分开，但它们的谱有精炼差异 → 光谱距离 > 0。
   这会同时压低所有先验的 ρ，是共同噪声，但会压缩先验之间的差异。
   敏感性分析：合并 1/2/3 为一类（谱取均值），重算 → n=14。

2. **ρ 之间无法直接比大小**
   不同先验的 ρ 来自同一组距离对，彼此相关，不能当独立样本做 t 检验。
   做法：**按类别 bootstrap**（有放回重采样 14/18 个类，重算 ρ，1000 次），
   得到每个先验 ρ 的分布，再看 Δρ 的分布是否跨 0。
"""
from __future__ import annotations
import os, sys, json, itertools, warnings
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "..", "out")

import exp88b_prior_rsa as A  # noqa: E402
from exp81a_prior_zoo import unit  # noqa: E402

NB = 2000
STRUCT = ["clm", "ecfp384", "ecfp2048", "charbag"]
COMPOS = ["physchem", "rdkit", "maccs"]


def dmat(V):
    D = 1.0 - V @ V.T
    np.fill_diagonal(D, 0.0)
    return D


def rho(Dp, Ds):
    from scipy.stats import spearmanr
    n = Dp.shape[0]
    iu = np.triu_indices(n, 1)
    return float(spearmanr(Dp[iu], Ds[iu])[0])


def boot(Dp, Ds, n=NB, seed=0):
    rng = np.random.default_rng(seed)
    N = Dp.shape[0]
    out = np.empty(n)
    for i in range(n):
        q = rng.integers(0, N, N)
        if len(set(q.tolist())) < 3:
            q = rng.permutation(N)
        out[i] = rho(Dp[np.ix_(q, q)], Ds[np.ix_(q, q)])
    return out


def analyse(tag, T, D_spec):
    print(f"\n===== {tag}  n={len(T['clm'])} =====")
    Ds = {}
    for pn, V in T.items():
        if pn == "onehot":
            continue
        Ds[pn] = dmat(V)
    point = {p: rho(Ds[p], D_spec) for p in Ds}
    order = sorted(point, key=lambda p: -point[p])
    print(f"{'prior':10s} {'rho':>8s} {'boot95%CI':>18s}")
    B = {}
    for p in order:
        v = boot(Ds[p], D_spec, seed=hash(p) % 9973)
        B[p] = v
        print(f"{p:10s} {point[p]:+.4f}  [{np.percentile(v,2.5):+.3f},"
              f"{np.percentile(v,97.5):+.3f}]")

    # 结构与组成两类均值
    s_idx = [p for p in STRUCT if p in point]
    c_idx = [p for p in COMPOS if p in point]
    d_struct = np.mean([Bs for Bs in [B[p] for p in s_idx]], 0)
    d_comp = np.mean([B[p] for p in c_idx], 0)
    delta = d_struct - d_comp
    print(f"\n  结构类均值 ρ = {np.mean([point[p] for p in s_idx]):+.4f} "
          f"({','.join(s_idx)})")
    print(f"  组成类均值 ρ = {np.mean([point[p] for p in c_idx]):+.4f} "
          f"({','.join(c_idx)})")
    print(f"  Δ(结构−组成) = {np.mean(delta):+.4f}  "
          f"95%CI [{np.percentile(delta,2.5):+.4f},{np.percentile(delta,97.5):+.4f}]  "
          f"P(Δ>0) = {np.mean(delta>0):.4f}")

    # CLM vs physchem 逐对
    if "clm" in B and "physchem" in B:
        d = B["clm"] - B["physchem"]
        print(f"  Δ(clm−physchem) = {np.mean(d):+.4f}  "
              f"95%CI [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]  "
              f"P(Δ>0) = {np.mean(d>0):.4f}")
    return dict(point=point,
                ci={p: [float(np.percentile(B[p], 2.5)),
                        float(np.percentile(B[p], 97.5))] for p in B},
                struct_mean=float(np.mean([point[p] for p in s_idx])),
                comp_mean=float(np.mean([point[p] for p in c_idx])),
                delta_struct_comp=[float(np.mean(delta)),
                                   float(np.percentile(delta, 2.5)),
                                   float(np.percentile(delta, 97.5)),
                                   float(np.mean(delta > 0))])


def main():
    d = np.load(os.path.join(OUT, "exp88_oil.npz"), allow_pickle=True)
    Xp, yc = d["X_pure"], d["yc"]
    classes = sorted(set(yc.tolist()))

    # ---- 情形 1：原样 18 类 ----
    M = np.stack([Xp[yc == c].mean(0) for c in classes])
    D_spec = dmat(np.stack([unit(r) for r in M]))
    T, _ = A.build_priors(classes)
    r1 = analyse("情形1: 18 类（含橄榄油三兄弟各自成类）", T, D_spec)

    # ---- 情形 2：合并 1/2/3（橄榄油等级）→ 14 类 ----
    merge = {1: 1, 2: 1, 3: 1}
    yc2 = np.array([merge.get(c, c) for c in yc])
    cls2 = sorted(set(yc2.tolist()))
    M2 = np.stack([Xp[yc2 == c].mean(0) for c in cls2])
    D_spec2 = dmat(np.stack([unit(r) for r in M2]))
    # build_priors 需要 Class 号，用合并后的代表号重映射
    T2, _ = A.build_priors(cls2)
    r2 = analyse("情形2: 14 类（橄榄油 1/2/3 合并）", T2, D_spec2)

    json.dump(dict(case18=r1, case14=r2),
              open(os.path.join(OUT, "exp88e_boot.json"), "w"), indent=1)
    print("\n[exp88e] saved out/exp88e_boot.json")


if __name__ == "__main__":
    main()
