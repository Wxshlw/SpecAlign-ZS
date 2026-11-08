"""exp87 —— 零样本头的容量诊断：变换先验能不能赢？

起因
----
exp84/86 的结论是 CLM 在定向归因上打不赢 charbag（|U|=3）和 physchem（|U|=5）。
用户要求把"全面赢过 8 个先验 + 3 个算法族"做到，因此在换目标/换数据之前，
必须先把**所有"改良表示本身"的可能**一次性穷尽或排除，否则重训新先验是赌博。

数学事实（本实验要验证的对象）
--------------------------
`head_ridge_R` 是**对偶形式**：
    A = Sc^T (Sc Sc^T + λI)^{-1} Zc ,      Sc = S_old - mean(S_old)
    W_cand = (S_cand - mu) @ A
⇒ 打分只经由 **K = (S_cand - mu) @ Sc^T**（线性核，形状 [n_cand, n_old]）
   再乘一个 n_old×dz 的系数。

推论：当 n_old = 5~7（|U|=5 时老类仅 2 实体 + 3 基质）时，**任何 d 维先验的信息
都被完全压缩成 n_old 维核表示**。于是
  · 降维 (PCA 到 k 维)        —— 只要不丢候选间差异，理论上**不改变**结果
  · 去各向异性 (ABTT)          —— 同上，等价于改有效 λ
  · 逐维标准化 (std)           —— 同上
若本实验显示这些变换普遍 Δ≈0，则"换表示"这条路在本头下**结构性无效**，
真正的杠杆只剩「相似度模式」与「候选库构成」，资源应转移到那里。

变换（全部只用码本自身统计量；候选库是任务输入，部署时即已知 ⇒ 零样本合法）
------------------------------------------------------------------------
  raw      现状（384 维，行归一化）
  std      按列标准化（不降维）
  pca{k}   SVD 取前 k 个主方向（k=2,4,8,16）——保留候选间最大差异
  abtt{k}  减均值后去掉 top-k 主方向（NLP 去各向异性标准做法，Mu & Viswanath 2018）
  tanh     逐维 tanh 压缩（单调有界，检验幅度效应）

口径
----
与 exp86 一致：**单元格等权**（先按候选对平均跨 seed，再平均候选对）。
自检：raw 的 physchem |U|=3 必须 ≈ 0.8203，否则口径错了，结果作废。
"""
from __future__ import annotations
import os, sys, json, argparse, itertools, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "论文8"))
import p6_g1_data as G1
from exp63_p6_screening_protocol import (stage1, embed, head_ridge_R, nrm,
                                         auc_of, EID2MID)
from exp68_p6_cond_scorer import split_by_group
from exp71_p6_detection_fix import load_groups_cached
from exp78_p6_attribution_proof import split_group_w
from exp81b_p6_benchmark_zoo import PRIORS

OUT = os.path.join(os.path.dirname(HERE), "out")
OLD_FRAC = 0.7


def apply_tr(S, kind):
    """对码本矩阵 [n_keys, d] 施加无监督变换，返回同形状（行 L2 归一化）。"""
    if kind == "raw":
        return S
    mu = S.mean(0, keepdims=True)
    Xc = S - mu
    if kind == "std":
        sd = Xc.std(0, keepdims=True) + 1e-6
        return nrm(Xc / sd)
    if kind == "tanh":
        return nrm(np.tanh(Xc * 2.0))
    if kind.startswith("pca"):
        k = int(kind[3:])
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        return nrm(Xc @ Vt[:k].T)
    if kind.startswith("abtt"):
        k = int(kind[4:])
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        V = Vt[:k].T                      # [d, k]
        return nrm(Xc - (Xc @ V) @ V.T)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--nrep", type=int, default=4)
    ap.add_argument("--nhu", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.0001,0.01,0.1,1")
    ap.add_argument("--priors",
                    default="clm,charbag,physchem,ecfp2048,maccs")
    ap.add_argument("--trs", default="raw,std,pca2,pca4,pca8,pca16,abtt1,abtt2,tanh")
    ap.add_argument("--eids", default="",
                    help="逗号分隔的实体子集（空=全部）。用于 clean 池 A01-A04,A07")
    ap.add_argument("--out", default="exp87_nhu3.json")
    args = ap.parse_args()

    lams = [float(x) for x in args.lams.split(",")]
    trs = [x.strip() for x in args.trs.split(",") if x.strip()]
    pris = [x.strip() for x in args.priors.split(",") if x.strip()]

    ent, conc, bg, _ = G1.load_g1_conc()
    GR = load_groups_cached()
    eids = sorted(set(GR) & set(EID2MID))
    if args.eids:
        keep = set(x.strip() for x in args.eids.split(",") if x.strip())
        eids = [e for e in eids if e in keep]
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]

    zc = np.load(os.path.join(OUT, "exp81_prior_cache.npz"), allow_pickle=True)
    assert list(zc["keys"]) == keys_all, "先验缓存 key 顺序不一致，禁止对齐！"
    S_pri = {n: np.asarray(zc[n], float) for n in PRIORS}
    dz = S_pri["clm"].shape[1]
    ki = {k: i for i, k in enumerate(keys_all)}

    rec = {}   # rec[(prior,tr,lam)]['dis'][(tag,T,D)] = auc ; ['top1'][(tag,u)] = v
    def add(p, t, lam, kind, key, val):
        rec.setdefault((p, t, lam), {}).setdefault(kind, {})[key] = val

    combos = list(itertools.combinations(eids, args.nhu))
    print(f"[exp87] |U|={args.nhu} 组合={len(combos)} seeds={args.seeds} "
          f"reps={args.nrep} priors={len(pris)} trs={len(trs)}", flush=True)

    rank_info = []
    for si in range(args.seeds):
        rng0 = np.random.default_rng(si * 7919 + 13)
        cis = rng0.permutation(len(combos))[:args.nrep]
        for ci in cis:
            U = list(combos[ci])
            others = [e for e in sorted(ent) if e not in U]
            order = ([ki[e] for e in others] + [ki[f"M_{m}"] for m in mids]
                     + [ki[u] for u in U])
            NEW0 = len(others) + len(mids)
            cand_ix = [NEW0 + j for j in range(len(U))]
            nrows = NEW0 + len(U)

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
                parts.append(Xo[pp[:cut]]); ys.append(np.full(cut, len(others) + j))
            Xtr = np.concatenate(parts).astype(np.float32)
            ytr = np.concatenate(ys).astype(np.int64)
            rows = sorted({int(y) for y in ytr})
            tag = f"s{si}c{ci}"

            # stage1 与先验无关（固定用原始 CLM 对齐），所有变换复用同一个 Z
            S_clm_ord = S_pri["clm"][order]
            net, _ = stage1(Xtr, ytr, dz, nrows, args.steps,
                            si * 1000 + ci, S=S_clm_ord)
            Zfit = embed(net, Xtr)
            Z_old = np.stack([Zfit[ytr == j].mean(0) for j in rows])
            Zte = {u: embed(net, Xte_U[u]) for u in U}

            # 诊断：线性核的有效秩 = min(n_cand, n_old)
            rank_info.append(dict(tag=tag, n_old=len(rows),
                                  n_cand=len(cand_ix),
                                  rank=min(len(rows), len(cand_ix))))

            for p in pris:
                Sp0 = S_pri[p][order]
                for tk in trs:
                    Sp = apply_tr(Sp0, tk)
                    S_o = Sp[rows]
                    mu = S_o.mean(0, keepdims=True)
                    Sc = Sp[cand_ix]
                    for lam in lams:
                        Wc = (Sc - mu) @ head_ridge_R(S_o, Z_old, lam)
                        key3 = (p, tk, lam)
                        for jT, jD in itertools.combinations(range(len(U)), 2):
                            T, D = U[jT], U[jD]
                            dT = Zte[T] @ Wc[jT] - Zte[T] @ Wc[jD]
                            dD = Zte[D] @ Wc[jT] - Zte[D] @ Wc[jD]
                            add(p, tk, lam, "dis", f"{tag}|{T}|{D}",
                                float(auc_of(dT, dD)))
                        for j, u in enumerate(U):
                            L = Zte[u] @ Wc.T
                            pred = np.array(U)[L.argmax(1)]
                            add(p, tk, lam, "top1", f"{tag}|{u}",
                                float(np.mean(pred == u)))
            print(f"  [{tag}] U={'-'.join(U)} n_old={len(rows)} done", flush=True)

    # ---- 单元格等权聚合：先按候选对 / 候选实体平均，再平均 ----
    def cell_mean(d, nsplit):
        """键形如 tag|T|D 或 tag|u：先按后 nsplit 段聚合，再平均单元格。"""
        by = {}
        for k, v in d.items():
            parts = k.split("|")
            cell = "|".join(parts[-nsplit:])
            by.setdefault(cell, []).append(v)
        return float(np.mean([np.mean(v) for v in by.values()])), len(by)

    agg = {}
    for (p, tk, lam), d in rec.items():
        a, nc = cell_mean(d.get("dis", {}), 2)
        t1, nu = cell_mean(d.get("top1", {}), 1)
        agg[f"{p}|{tk}|{lam:g}"] = dict(auc_dis=a, top1=t1,
                                        n_cell_dis=nc, n_cell_top1=nu)

    path = os.path.join(OUT, args.out)
    cells = {f"{p}|{tk}|{lam:g}": dict(dis=d.get("dis", {}),
                                      top1=d.get("top1", {}))
             for (p, tk, lam), d in rec.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(meta=dict(nhu=args.nhu, seeds=args.seeds, nrep=args.nrep,
                                 lams=lams, trs=trs, priors=pris),
                       agg=agg, cells=cells, rank_info=rank_info),
                  f, ensure_ascii=False, indent=1)
    print(f"[exp87] 已写出 {path}", flush=True)

    # ---- 控制台摘要：每个先验取最优 λ，比较变换 ----
    print("\n=== auc_dis（每个 (先验,变换) 取最优 λ）===", flush=True)
    for p in pris:
        base = None
        line = []
        for tk in trs:
            best = max((agg[f"{p}|{tk}|{lam:g}"]["auc_dis"] for lam in lams))
            lam_b = max(lams, key=lambda L: agg[f"{p}|{tk}|{L:g}"]["auc_dis"])
            t1 = agg[f"{p}|{tk}|{lam_b:g}"]["top1"]
            if tk == "raw":
                base = best
            d = (best - base) if base is not None else 0.0
            line.append(f"{tk}:{best:.4f}(Δ{d:+.4f},t1={t1:.3f})")
        print(f"{p:10s} " + "  ".join(line), flush=True)


if __name__ == "__main__":
    main()
