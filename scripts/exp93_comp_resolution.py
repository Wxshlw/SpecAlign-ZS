"""exp93 —— 组成分辨率的因果消融（补「中间点」，替代缺失的第二套数据集）

要回答的问题
------------
用户方向：**什么时候适合语义先验、什么时候普通算法就够了。**
第一版设想是"经验对照表"，但现有两端都近乎平凡：
  - 跨品类（糖/肉桂/姜黄…）：组成差异极大 → physchem 赢（0.869）
  - 植物油互掺（exp88）：元素组成近乎相同 → physchem 崩（RSA ρ=0.123 n.s.）
中间点是空的。本实验用**受控消融**把中间点造出来，而不是再找第二套数据集
（IFR 四套数据全查过：水果泥二值、肉类是 3 个基质、咖啡二值、橄榄油按产地，
没有一套含"同基质 + ≥3 标注掺杂物"）。

两个操纵轴
----------
A. **候选间分辨率**（rank truncation，对所有先验同样施加）
   19 个 key 的先验矩阵中心化后 SVD，只保留前 r 个「候选间」方向：
       V_r = mean + U_r S_r V_r^T → 逐行单位化
   r = 1,2,3,4,5,8,full。只压缩候选之间的可分维度，不改平均朝向。
   ⇒ 检验「性能 = 候选间可分辨率」是否成立、斜率是否与先验类型有关。
   对所有先验同等施加 ⇒ "截断本身有害" 被控制住。

B. **组成块消融**（仅 physchem；25 维 = 13 元素计数 + 12 理化描述符）
   `elem`=只留元素计数 / `desc`=只留理化描述符 / `rfull`=完整。
   做法：完全复现 exp81a 的标准化 + 高斯随机投影，在**标准化原生空间**里把另一块置 0，
   再用同一投影矩阵投影。精确、无需改 pipeline。

判据（什么才算"有机制"）
-----------------------
若所有先验的 AUC(r) 曲线同斜率 ⇒ 只是"截断越狠越差"，无机制。
有机制 = **physchem 的斜率显著陡于结构类先验**，即它的优势完全来自候选间可分辨维数，
而结构类先验在低秩下仍保留判别信息。

口径沿用 exp82 轴 A：定向归因、group 分组、λ 固定 1e-4（零样本不可调 λ）、
auc_dis 与 top-1 并列。
"""
from __future__ import annotations
import os, sys, json, argparse, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "论文8"))
import p6_g1_data as G1
from exp63_p6_screening_protocol import (stage1, embed, head_ridge_R, auc_of, EID2MID)
from exp68_p6_cond_scorer import split_by_group
from exp71_p6_detection_fix import load_groups_cached
from exp78_p6_attribution_proof import split_group_w
from exp81b_p6_benchmark_zoo import PRIORS
import exp81a_prior_zoo as Z

OUT = os.path.dirname(HERE) and os.path.join(os.path.dirname(HERE), "out")
LAM = 1e-4
DIM = 384
N_ELEM = 13          # physchem 前 13 维 = 元素计数


def nrm_rows(V):
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def trunc_rank(V, r):
    """候选间秩截断：中心化 → 只留前 r 方向 → 加回均值 → 逐行单位化。"""
    mu = V.mean(0, keepdims=True)
    U, s, Vt = np.linalg.svd(V - mu, full_matrices=False)
    r = int(min(r, len(s)))
    return nrm_rows(mu + (U[:, :r] * s[:r]) @ Vt[:r])


def build_physchem_blocks(keys_all):
    """复现 exp81a 的 physchem（标准化 + 高斯投影），并给出 elem/desc 两块变体。

    ⚠ 投影矩阵必须复现 exp81a 的**抽取顺序**：该脚本用同一个 rng 先抽 rdkit
    (d_rd×384)、再抽 physchem (25×384)。只抽一次会拿到 rdkit 的矩阵。
    """
    all_smi = sorted({s for k in keys_all for s, _ in Z.prof_of(k)})
    raw_all = Z.mol_vectors(all_smi)
    raw = raw_all["physchem"]
    d_rd = raw_all["rdkit"][all_smi[0]].shape[0]

    nat = []
    for k in keys_all:
        cs = Z.prof_of(k)
        tw = sum(w for _, w in cs)
        nat.append(sum(raw[s] * (w / tw) for s, w in cs))
    nat = np.stack(nat)
    nat = (nat - nat.mean(0)) / (nat.std(0) + 1e-9)          # 与 exp81a 一致

    rng = np.random.default_rng(Z.HASH_SEED)
    rng.normal(size=(d_rd, DIM)) / np.sqrt(DIM)              # 先消耗掉 rdkit 的抽样
    proj = rng.normal(size=(nat.shape[1], DIM)) / np.sqrt(DIM)

    out = {"rebuilt": nrm_rows(nat @ proj)}
    for keep, sl in (("elem", slice(0, N_ELEM)),
                     ("desc", slice(N_ELEM, None))):
        v = np.zeros_like(nat)
        v[:, sl] = nat[:, sl]
        out[keep] = nrm_rows(v @ proj)

    # 最小充分描述符集：extra 的顺序（N_ELEM 起）为
    # [MolWt, HeavyAtoms, MolLogP, TPSA, HDon, HAcc, RotB, NumRings,
    #  RingCount, FracCSP3, AromaticRings, Heteroatoms]
    for nm, idx in (("mw", [0]),
                    ("mw_logp", [0, 2]),
                    ("mw_logp_tpsa", [0, 2, 3]),
                    ("mw_logp_tpsa_hv", [0, 1, 2, 3])):
        v = np.zeros_like(nat)
        cols = [N_ELEM + j for j in idx]
        v[:, cols] = nat[:, cols]
        out[nm] = nrm_rows(v @ proj)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--nrep", type=int, default=4)
    ap.add_argument("--nhu", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--priors", default="clm,physchem,charbag,ecfp2048")
    ap.add_argument("--ranks", default="1,2,3,4,5,8,0")      # 0 = full
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--out", default="exp93_comp_res.json")
    args = ap.parse_args()

    ent, conc, bg, _ = G1.load_g1_conc()
    GR = load_groups_cached()
    eids = sorted(set(GR) & set(EID2MID))
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]

    zc = np.load(os.path.join(OUT, "exp81_prior_cache.npz"), allow_pickle=True)
    assert list(zc["keys"]) == keys_all, "先验缓存 key 顺序不一致"
    S0 = {n: np.asarray(zc[n], float) for n in PRIORS}
    dz = S0["clm"].shape[1]
    ki = {k: i for i, k in enumerate(keys_all)}

    pri_list = [x.strip() for x in args.priors.split(",") if x.strip()]
    ranks = [int(x) for x in args.ranks.split(",")]

    S, vmeta = {}, {}
    for p in pri_list:
        for r in ranks:
            nm = f"{p}|r{r if r else 'full'}"
            S[nm] = trunc_rank(S0[p], r) if r else nrm_rows(S0[p])
            vmeta[nm] = dict(base=p, rank=r)

    if args.blocks and "physchem" in pri_list:
        print("[exp93] 重建 physchem 原生块 …", flush=True)
        blk = build_physchem_blocks(keys_all)
        # 自检：重建完整版必须≈缓存（否则投影/标准化没复现，elem/desc 两块不可信）
        cos_rb = float(np.mean(np.sum(blk["rebuilt"] * S0["physchem"], axis=1)))
        print(f"[exp93] 自检 cos(重建, 缓存) = {cos_rb:.6f}", flush=True)
        assert cos_rb > 0.999, f"physchem 重建失败 (cos={cos_rb:.4f})，块消融不可信"
        for keep in ("elem", "desc", "mw", "mw_logp", "mw_logp_tpsa", "mw_logp_tpsa_hv"):
            nm = f"physchem|{keep}"
            S[nm] = blk[keep]
            vmeta[nm] = dict(base="physchem", block=keep)
        res_selfcheck = dict(cos_rebuilt_vs_cache=cos_rb)

    pri_all = list(S)

    # 候选间有效秩（参与率）与 top 比例
    rank_info = {}
    for p in pri_list:
        Vc = S0[p] - S0[p].mean(0, keepdims=True)
        lam = np.linalg.svd(Vc, compute_uv=False) ** 2
        lam = lam / lam.sum()
        rank_info[p] = dict(pr=float(1.0 / np.sum(lam ** 2)),
                            top1=float(lam[0]), top3=float(lam[:3].sum()))

    res = {"META": dict(nhu=args.nhu, seeds=args.seeds, nrep=args.nrep, lam=LAM,
                        ranks=ranks, priors=pri_list, pri_all=pri_all,
                        variants=vmeta, rank_info=rank_info,
                        selfcheck=locals().get("res_selfcheck", {}),
                        note="exp93 组成分辨率因果消融；定向归因；λ固定")}

    combos = list(itertools.combinations(eids, args.nhu))
    print(f"[exp93] |U|={args.nhu} 组合={len(combos)} 变体={len(pri_all)} "
          f"reps={args.nrep} seeds={args.seeds}", flush=True)
    print("[exp93] 候选间有效秩(参与率): " +
          "  ".join(f"{p}={rank_info[p]['pr']:.2f}" for p in pri_list), flush=True)

    def add(arm, kind, key, rec):
        res.setdefault(arm, {}).setdefault(kind, {}).setdefault(key, []).append(rec)

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
            S_all = {p: S[p][order] for p in pri_all}
            # stage1 的化学对齐码本固定为 clm 原始（不受 --priors 影响，才能与 exp82/90/91 同条件）
            S_stage = nrm_rows(S0["clm"])[order]

            rng = np.random.default_rng(si * 104729 + ci * 37 + 7)
            Xte_U = {}
            for u in U:
                Xe = np.asarray(ent[u], float)
                te_ix, _ = split_by_group(Xe, GR[u]["gid"], rng)
                Xte_U[u] = Xe[te_ix]
            parts, ys = [], []
            for j, e in enumerate(others):
                Xo = np.asarray(ent[e], float)
                if e in GR:
                    a, b, _ = split_group_w(Xo, GR[e]["gid"], rng)
                else:
                    pp = rng.permutation(len(Xo))
                    cut = max(2, int(round(0.7 * len(Xo))))
                    a, b = pp[:cut], pp[cut:]
                parts.append(Xo[a]); ys.append(np.full(len(a), j))
            for j, m in enumerate(mids):
                Xo = np.asarray(bg[m], float)
                pp = rng.permutation(len(Xo))
                cut = max(2, int(round(0.7 * len(Xo))))
                parts.append(Xo[pp[:cut]]); ys.append(np.full(cut, len(others) + j))
            Xtr = np.concatenate(parts).astype(np.float32)
            ytr = np.concatenate(ys).astype(np.int64)
            rows = sorted({int(y) for y in ytr})
            tag = f"seed{si}_c{ci}"

            net, _ = stage1(Xtr, ytr, dz, nrows, args.steps, si * 1000 + ci,
                            S=S_stage)
            Zfit = embed(net, Xtr)
            Z_old = np.stack([Zfit[ytr == j].mean(0) for j in rows])
            Zte = {u: embed(net, Xte_U[u]) for u in U}

            for p in pri_all:
                Sp = S_all[p]
                S_o = Sp[rows]
                mu = S_o.mean(0, keepdims=True)
                Wc = (Sp[cand_ix] - mu) @ head_ridge_R(S_o, Z_old, LAM)
                for jT, jD in itertools.combinations(range(len(U)), 2):
                    T, D = U[jT], U[jD]
                    dT = Zte[T] @ Wc[jT] - Zte[T] @ Wc[jD]
                    dD = Zte[D] @ Wc[jT] - Zte[D] @ Wc[jD]
                    add(p, "dis", f"{T}|{D}", dict(auc_dis=float(auc_of(dT, dD)), tag=tag))
                for j, u in enumerate(U):
                    L = Zte[u] @ Wc.T
                    pred = np.array(U)[L.argmax(1)]
                    add(p, "top1", u, dict(top1=float(np.mean(pred == u)),
                                           n=len(Zte[u]), tag=tag))
            print(f"  [seed{si}_c{ci}] U={'-'.join(U)} n_tr={len(Xtr)}", flush=True)

    with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[exp93] 写入 {args.out}", flush=True)


if __name__ == "__main__":
    main()
