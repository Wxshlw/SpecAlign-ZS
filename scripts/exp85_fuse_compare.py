"""exp85 —— CLM × physchem 的三种融合方式对照（拼接 / 分数级 / 特征级）

背景
  exp82/84 已证：**特征拼接** concat(clm, physchem) 在 |U|=3 与 |U|=5 均不增益，
  且 |U|=5 时 0.774 < physchem 单打 0.856（Δ=−0.082）。
  机制：岭回归头行数 = 老类数（|U|=5 时仅 2 行），特征扩容 → n≪d → 稀释。

本实验检验两个「维度不增」的融合方式是否能把互补兑现：
  1) score  : s(T) = (1−α)·z[s_phys(T)] + α·z[s_clm(T)]，α ∈ [0,1]
              —— 等价于 log-linear 池化 / 「组成先验 × 结构似然」的贝叶斯组合
  2) feat   : S_fuse = nrm((1−α)·S_phys + α·S_clm)，再训单个岭回归头
              —— 特征空间加权（不是拼接），维度保持 384
  3) concat : 对照组（复现 exp82 的失败基线，保证同口径）

标准化口径（score 融合必须做，否则 α 无意义，两臂 w 的范数不同）
  z-global : 除以「老类样本在该候选头上的分数 std」的全局均值
  z-cand   : 每个候选 j 各自用老类样本分数 (mean_j, std_j) 标准化（per-candidate 校准）

口径：exp78/82 定向归因（U 全未见，T/D ∈ U，判别函数 s_T − s_D）
输出：out/exp85_fuse.json + 控制台汇总
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

OUT = os.path.join(os.path.dirname(HERE), "out")
OLD_FRAC = 0.7
ALPHAS = [round(x * 0.1, 1) for x in range(11)]      # 0,0.1,...,1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--nrep", type=int, default=4)
    ap.add_argument("--nhu", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--out", default="exp85_fuse.json")
    args = ap.parse_args()

    ent, conc, bg, _ = G1.load_g1_conc()
    GR = load_groups_cached()
    eids = sorted(set(GR) & set(EID2MID))
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]

    zc = np.load(os.path.join(OUT, "exp81_prior_cache.npz"), allow_pickle=True)
    assert list(zc["keys"]) == keys_all, "先验缓存 key 顺序不一致，禁止对齐！"
    S_pri = {n: np.asarray(zc[n], float) for n in
             ["clm", "physchem", "charbag", "ecfp2048"]}
    dz = S_pri["clm"].shape[1]
    ki = {k: i for i, k in enumerate(keys_all)}
    combos = list(itertools.combinations(eids, args.nhu))
    lam = args.lam

    res = {"META": dict(seeds=args.seeds, seed0=args.seed0, nrep=args.nrep,
                        nhu=args.nhu, lam=lam, alphas=ALPHAS,
                        modes=["score-g", "score-c", "feat", "concat"],
                        note="auc_dis: T/D both unseen; fuse(clm,physchem)")}
    cells = []

    def add(arm, kind, key, rec):
        res.setdefault(arm, {}).setdefault(kind, {}).setdefault(key, []).append(rec)

    print(f"[exp85] |U|={args.nhu} 组合={len(combos)} reps={args.nrep} "
          f"seeds={args.seeds} lam={lam:g}", flush=True)

    for si in range(args.seed0, args.seed0 + args.seeds):
        rng0 = np.random.default_rng(si * 7919 + 13)
        cis = rng0.permutation(len(combos))[:args.nrep]
        for ci in cis:
            U = list(combos[ci])
            others = [e for e in sorted(ent) if e not in U]
            order = ([ki[e] for e in others]
                     + [ki[f"M_{m}"] for m in mids]
                     + [ki[u] for u in U])
            NEW0 = len(others) + len(mids)
            iu = {u: NEW0 + j for j, u in enumerate(U)}
            cand_ix = [iu[u] for u in U]
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
                parts.append(Xo[pp[:cut]])
                ys.append(np.full(cut, len(others) + j))
            Xtr = np.concatenate(parts).astype(np.float32)
            ytr = np.concatenate(ys).astype(np.int64)
            rows = sorted({int(y) for y in ytr})
            tag = f"seed{si}_c{ci}"

            t0 = time.time()
            Sc_all = S_pri["clm"][order]
            net, _ = stage1(Xtr, ytr, dz, nrows, args.steps, si * 1000 + ci,
                            S=Sc_all)
            Zfit = embed(net, Xtr)
            Z_old = np.stack([Zfit[ytr == j].mean(0) for j in rows])
            Zte = {u: embed(net, Xte_U[u]) for u in U}
            print(f"  [{tag}] U={'-'.join(U)} stage1 {time.time()-t0:.1f}s", flush=True)

            # ---------- 单先验的候选分数矩阵（同 λ） ----------
            SP = {}
            for p in ["clm", "physchem", "charbag", "ecfp2048"]:
                Sp = S_pri[p][order]
                S_o = Sp[rows]
                mu = S_o.mean(0, keepdims=True)
                Wc = (Sp[cand_ix] - mu) @ head_ridge_R(S_o, Z_old, lam)
                SP[p] = dict(Wc=Wc,
                             L={u: Zte[u] @ Wc.T for u in U},
                             Lold=Z_old @ Wc.T)

            def record(arm, F_by_u):
                """F_by_u: {u: (n_u, |U|) 分数矩阵}"""
                for jT, jD in itertools.combinations(range(len(U)), 2):
                    T, D = U[jT], U[jD]
                    dT = F_by_u[T][:, jT] - F_by_u[T][:, jD]
                    dD = F_by_u[D][:, jT] - F_by_u[D][:, jD]
                    a = float(auc_of(dT, dD))
                    add(arm, "dis", f"{T}|{D}",
                        dict(auc_dis=a, tag=tag))
                for j, u in enumerate(U):
                    pred = np.array(U)[F_by_u[u].argmax(1)]
                    add(arm, "top1", u,
                        dict(top1=float(np.mean(pred == u)),
                             n=len(Zte[u]), tag=tag))

            # 单打基线
            for p in ["clm", "physchem", "charbag", "ecfp2048"]:
                record(f"single:{p}", {u: SP[p]["L"][u] for u in U})

            # ---------- 融合 ----------
            Lp, Lc = SP["physchem"], SP["clm"]
            # 尺度：老类参照分数的 std（全局 / 逐候选）
            sd_g_p = float(Lp["Lold"].std()) + 1e-12
            sd_g_c = float(Lc["Lold"].std()) + 1e-12
            mu_c_p = Lp["Lold"].mean(0, keepdims=True)
            sd_c_p = Lp["Lold"].std(0, keepdims=True) + 1e-12
            mu_c_c = Lc["Lold"].mean(0, keepdims=True)
            sd_c_c = Lc["Lold"].std(0, keepdims=True) + 1e-12

            for al in ALPHAS:
                # score-global
                Fg = {u: (1 - al) * (Lp["L"][u] / sd_g_p) + al * (Lc["L"][u] / sd_g_c)
                      for u in U}
                record(f"score-g|a{al:g}", Fg)
                # score-cand
                Fc = {u: (1 - al) * ((Lp["L"][u] - mu_c_p) / sd_c_p)
                      + al * ((Lc["L"][u] - mu_c_c) / sd_c_c) for u in U}
                record(f"score-c|a{al:g}", Fc)
                # feat：加权和（维度不增）
                Sf = nrm((1 - al) * S_pri["physchem"][order]
                         + al * S_pri["clm"][order])
                S_o = Sf[rows]
                mu = S_o.mean(0, keepdims=True)
                Wf = (Sf[cand_ix] - mu) @ head_ridge_R(S_o, Z_old, lam)
                record(f"feat|a{al:g}", {u: Zte[u] @ Wf.T for u in U})

            # concat 对照（与 exp82 一致）
            Sct = np.concatenate([S_pri["clm"][order],
                                  S_pri["physchem"][order]], 1)
            S_o = Sct[rows]
            mu = S_o.mean(0, keepdims=True)
            Wct = (Sct[cand_ix] - mu) @ head_ridge_R(S_o, Z_old, lam)
            record("concat", {u: Zte[u] @ Wct.T for u in U})

            cells.append(tag)

    res["META"]["cells"] = cells
    p = os.path.join(OUT, args.out)
    json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[exp85] 写出 {p}  ({len(cells)} cells)", flush=True)

    # ---------------- 汇总 ----------------
    def flat(arm, kind="dis", field="auc_dis"):
        v = res.get(arm, {}).get(kind, {})
        xs = [q[field] for lst in v.values() for q in lst]
        return float(np.mean(xs)) if xs else None

    def perf_cell(arm, kind="dis", field="auc_dis"):
        v = res.get(arm, {}).get(kind, {})
        out = {}
        for key, lst in v.items():
            tmp = {}
            for q in lst:
                tmp.setdefault(q["tag"], []).append(q[field])
            for t, vs in tmp.items():
                out[(t, key)] = float(np.mean(vs))
        return out

    from scipy.stats import ttest_rel

    def paired(a1, a2, kind="dis", field="auc_dis"):
        A, B = perf_cell(a1, kind, field), perf_cell(a2, kind, field)
        ks = sorted(set(A) & set(B))
        if len(ks) < 3:
            return None
        x = np.array([A[k] for k in ks]); y = np.array([B[k] for k in ks])
        return dict(a=x.mean(), b=y.mean(), dd=float((x - y).mean()),
                    p=float(ttest_rel(x, y).pvalue), n=len(ks))

    print("\n=== 单打基线（auc_dis / top1）===", flush=True)
    for p in ["physchem", "clm", "charbag", "ecfp2048"]:
        print(f"  {p:10s} auc_dis={flat(f'single:{p}'):.4f} "
              f"top1={flat(f'single:{p}','top1','top1'):.4f}", flush=True)

    print("\n=== 融合方式扫描（对 physchem 单打配对）===", flush=True)
    print(f"{'arm':22s} {'auc_dis':>8s} {'top1':>7s} | vs physchem: Δ / p", flush=True)
    for mode in ["concat"] + [f"{m}|a{al:g}" for al in ALPHAS for m in ["score-g", "score-c"]] \
            + [f"feat|a{al:g}" for al in ALPHAS]:
        a = flat(mode); t = flat(mode, "top1", "top1")
        if a is None:
            continue
        c = paired(mode, "single:physchem")
        print(f"{mode:22s} {a:8.4f} {t:7.4f} | {c['dd']:+.3f} p={c['p']:.3g}", flush=True)


if __name__ == "__main__":
    main()
