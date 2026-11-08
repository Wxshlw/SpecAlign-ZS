"""exp47 —— 混合头：把「化学外推」与「新类样本类中心」融合

=============================================================================
为什么必须做（exp46 暴露的设计缺陷，不是科学负结果）
=============================================================================
exp46 总表（macro top1）：
    k=0   本文 ridge-macro 0.106  vs 所有外部基线 0.000   ⇒ 我们独占（唯一可解）
    k=5   本文 0.445  vs centroid-cos 0.751              ⇒ **落后**
    k=20  本文 0.624  vs centroid-cos 0.813              ⇒ **落后**
逐实体：k=5 是 3 胜 8 负 2 平；在 COFR / HSYRUP / MEATP / MEATT / A05 上
       本文 top1 = 0.000，而最简的谱库检索 centroid-cos 能到 0.83–1.00。

**根因**：k>0 时新类**已经有样本**，但 exp44 的 ridge 臂只用化学向量外推
新类头 W_new = (s_new − μ)·A，**完全没有用这 k 个样本的类中心**。
而 `centroid-cos` 用的正是样本类中心 ⇒ 它在 k>0 时统治。
⇒ 这是**实现缺陷**（把可用的样本信息丢掉了），不是「先验无效」。

=============================================================================
修复：GZSL 标准混合头（Xian et al., TPAMI 2019 综述中的 hybrid 家族）
=============================================================================
    W_j = normalize( α · ŵ_chem,j  +  (1 − α) · z̄_j )
      ŵ_chem,j = 化学外推头（去均值化学向量经 ridge 投影），归一化
      z̄_j     = 该类样本在 z 空间的类中心，归一化
      α        = 1 纯化学（k=0 的唯一选择）
                 0 纯数据（退化为 z 空间最近类中心 ≈ centroid-cos）
⇒ 两端都被包含，α 由数据决定 ⇒ **天然的下限保证**：
   无论 α 取何值，都不会比两端中较差的那一端更差（在验证集意义下）。

★ **α 的选择准则（关键，必须模拟新类的少样本处境）**
   老类有几百个样本、新类只有 k 个。若在老类全量类中心上选 α，
   会系统性偏向数据端（因为老类的 z̄ 用全量算得更准）——正是 exp44 里
   γ 自动选成 0 的同一个陷阱。
   ⇒ 本实验用 **pseudo-unseen 模拟**：选 α 时，把**每个老类的类中心也只用
     k 个样本**构造，再在验证集上算 macro-recall。这样 α 是在
     「各类都只有 k 个样本」的处境下选出来的，与新类的真实处境一致。

=============================================================================
预注册判据（写死）
=============================================================================
- H47a（主）：k∈{5,20} 时 `hybrid`（模拟选 α）的 top1 ≥ `ncm`（α=0，纯数据）
             ⇒ 即便已有样本，化学外推仍有**边际贡献**（这是本文的核心主张）
- H47b（主）：k∈{5,20} 时 `hybrid` 的 top1 ≥ `ridge`（α=1，纯化学）
             ⇒ 融合不比纯化学差（下限保证）
- H47c（次）：k∈{5,20} 时 `hybrid` 的 top1 ≥ exp45 的 `centroid-cos`
             ⇒ 追平/超过最强的外部基线（把 exp46 的 8 负翻过来）
- H47d（次）：选出的 α 随 k 增大而**下降** ⇒ 模型自己学会「样本越多越信数据」
             （与 exp40 门控权重 Spearman ρ=−0.5 同向佐证）
- k=0 时 hybrid/ncm 不可定义（无样本），只报 ridge；这一点本身就是结论。
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

OUT = r"D:/2026sci论文实验设计/论文6/out"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMP = 0.07


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


def build_hybrid(W_chem, Z, y, rows, alpha, n_sim=None, rng=None):
    """分类头：α·化学外推 + (1−α)·样本类中心。
    n_sim：若给定，每个类的类中心只用 n_sim 个样本（模拟少样本处境，用于选 α）。"""
    W = np.zeros_like(W_chem)
    wc = nrm(W_chem)
    for j in rows:
        ix = np.where(y == j)[0]
        if n_sim is not None and len(ix) > n_sim:
            ix = rng.permutation(ix)[:n_sim]
        zj = Z[ix].mean(0)
        v = alpha * wc[j] + (1 - alpha) * zj / (np.linalg.norm(zj) + 1e-12)
        W[j] = v / (np.linalg.norm(v) + 1e-12)
    return W


@torch.no_grad()
def macro_recall(net, W, Xv, yv, rows):
    z = F.normalize(net(torch.tensor(Xv, dtype=torch.float32, device=DEV)), dim=1)
    Wt = torch.tensor(nrm(W), dtype=torch.float32, device=DEV)
    pred = (z @ Wt.T)[:, rows].argmax(1).cpu().numpy()
    rs = []
    for i, j in enumerate(rows):
        m = yv == j
        if m.sum():
            rs.append(float((pred[m] == i).mean()))
    return float(np.mean(rs)) if rs else 0.0


@torch.no_grad()
def metrics(net, W, Xte_new, Xte_old, new_row=18):
    z_n = F.normalize(net(torch.tensor(Xte_new, dtype=torch.float32, device=DEV)), dim=1)
    z_o = F.normalize(net(torch.tensor(Xte_old, dtype=torch.float32, device=DEV)), dim=1)
    Wt = torch.tensor(nrm(W), dtype=torch.float32, device=DEV)
    lg_n = (z_n @ Wt.T).cpu().numpy()[:, :19]
    lg_o = (z_o @ Wt.T).cpu().numpy()[:, :19]
    top1 = float((lg_n.argmax(1) == new_row).mean())
    pos, neg = lg_n[:, new_row], lg_o[:, new_row]
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)).astype(float) + 1.0
    n1, n2 = len(pos), len(neg)
    auc = float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))
    ranks = np.where(np.argsort(-lg_n, axis=1) == new_row)[1] + 1
    return dict(top1=top1, auc=auc, mrr=float((1.0 / ranks).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="0,5,20")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.001,0.01,0.1,1")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--eids", default="")
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

    arms = ["ridge", "ncm", "hybrid"] + [f"hybrid{a:g}" for a in alphas] + ["none"]
    res = {"acc": {t: {a: {} for a in arms} for t in tags},
           "alpha": {}, "meta": {"shots": tags, "seeds": args.seeds,
                                 "alphas": alphas, "lams": lams,
                                 "eids": eids, "dim": dim}}
    print(f"[exp47] device={DEV} shots={tags} seeds={args.seeds} "
          f"alphas={alphas} lams={lams}", flush=True)

    rng_g = np.random.default_rng(20240101)
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
                perm = rng.permutation(len(Xe))
                n_te = int(min(max(5, round(0.3 * len(Xe))), len(Xe) - 1))
                te_ix, pool_ix = perm[:n_te], perm[n_te:]
                tr_e_ix = pool_ix if k is None else pool_ix[:min(k, len(pool_ix))]
                Xte_new = Xe[te_ix]

                parts, ys = [], []
                for j, e in enumerate(others):
                    parts.append(np.asarray(ent[e], float)); ys.append(np.full(len(ent[e]), j))
                for j, m in enumerate(mids):
                    parts.append(np.asarray(bg[m], float)); ys.append(np.full(len(bg[m]), len(others) + j))
                nrows = len(others) + len(mids)
                if len(tr_e_ix):
                    parts.append(Xe[tr_e_ix]); ys.append(np.full(len(tr_e_ix), 18)); nrows = 19
                Xtr = np.concatenate(parts).astype(np.float32)
                ytr = np.concatenate(ys).astype(np.int64)

                ref = []
                for e in others:
                    Xo = np.asarray(ent[e], float)
                    ref.append(Xo[rng_g.permutation(len(Xo))[:min(200, len(Xo))]])
                for m in mids:
                    Xo = np.asarray(bg[m], float)
                    ref.append(Xo[rng_g.permutation(len(Xo))[:min(200, len(Xo))]])
                Xte_old = np.concatenate(ref).astype(np.float32)

                rng_v = np.random.default_rng(seed * 31 + 7)
                val_mask = np.zeros(len(Xtr), bool)
                rows_old = sorted({int(y) for y in ytr})
                for j in rows_old:
                    if j == 18 and len(tr_e_ix) == 0:
                        continue
                    ix = rng_v.permutation(np.where(ytr == j)[0])
                    val_mask[ix[:max(3, int(0.2 * len(ix)))]] = True
                Xfit, yfit = Xtr[~val_mask], ytr[~val_mask]
                Xval, yval = Xtr[val_mask], ytr[val_mask]
                val_rows = sorted({int(y) for y in yval})

                net, W_free = stage1(Xfit, yfit, dim, nrows, args.steps,
                                     seed * 1000 + eids.index(e_hold), S=S, mode="chem")
                Zfit = embed(net, Xfit)
                Ztr = embed(net, Xtr)
                Z_old = np.stack([Zfit[yfit == j].mean(0) for j in rows_old])
                S_old = S[rows_old]
                mu_S = S_old.mean(0, keepdims=True)

                # λ 由 macro-recall 选（沿用 exp44 的 P2 修正）
                lam_b, bv = lams[0], -1.0
                for lam in lams:
                    Wc = (S - mu_S) @ head_ridge_R(S_old, Z_old, lam)
                    v = macro_recall(net, Wc, Xval, yval, val_rows)
                    if v > bv:
                        bv, lam_b = v, lam
                W_chem = (S - mu_S) @ head_ridge_R(S_old, Z_old, lam_b)

                accs = {}
                if len(tr_e_ix):
                    # ---------- α 由 pseudo-unseen 模拟选 ----------
                    # 把每个老类的类中心也只用 k 个样本构造 ⇒ 复现新类的处境
                    rows_fit = sorted({int(y) for y in yfit})
                    k_sim = max(1, len(tr_e_ix))
                    a_b, ab = 1.0, -1.0
                    for a in alphas:
                        W_sim = build_hybrid(W_chem, Zfit, yfit, rows_fit,
                                             a, n_sim=k_sim,
                                             rng=np.random.default_rng(seed * 17 + 3))
                        v = macro_recall(net, W_sim, Xval, yval, val_rows)
                        if v > ab:
                            ab, a_b = v, a
                    res["alpha"].setdefault(tag, {}).setdefault(
                        e_hold, []).append(float(a_b))

                    rows_all = sorted({int(y) for y in ytr})
                    accs["ncm"] = metrics(net, build_hybrid(
                        W_chem, Ztr, ytr, rows_all, 0.0), Xte_new, Xte_old, NEW)
                    accs["hybrid"] = metrics(net, build_hybrid(
                        W_chem, Ztr, ytr, rows_all, a_b), Xte_new, Xte_old, NEW)
                    for a in alphas:
                        accs[f"hybrid{a:g}"] = metrics(net, build_hybrid(
                            W_chem, Ztr, ytr, rows_all, a), Xte_new, Xte_old, NEW)
                    accs["none"] = metrics(net, W_free, Xte_new, Xte_old, NEW)
                else:
                    # k=0：没有样本类中心 ⇒ 只有纯化学一条路，这正是结论本身
                    W0 = np.zeros_like(W_chem)
                    W0[:18] = nrm(W_chem)[:18]
                    W0[18] = nrm(W_chem)[18]
                    accs["ncm"] = dict(top1=float("nan"), auc=float("nan"),
                                       mrr=float("nan"))
                    accs["hybrid"] = metrics(net, W_chem, Xte_new, Xte_old, NEW)
                    for a in alphas:
                        accs[f"hybrid{a:g}"] = metrics(
                            net, W_chem, Xte_new, Xte_old, NEW)
                accs["ridge"] = metrics(net, W_chem, Xte_new, Xte_old, NEW)

                for a, m in accs.items():
                    res["acc"][tag].setdefault(a, {}).setdefault(
                        e_hold, []).append({kk: (None if np.isnan(vv)
                                                 else round(float(vv), 4))
                                            for kk, vv in m.items()})

        line = []
        for t in tags:
            s = " ".join(
                f"{a}={np.nanmean([d['top1'] if d['top1'] is not None else np.nan for d in res['acc'][t][a][e_hold]]):.2f}"
                f"/{np.nanmean([d['auc'] if d['auc'] is not None else np.nan for d in res['acc'][t][a][e_hold]]):.2f}"
                for a in arms if res["acc"][t][a].get(e_hold))
            ab = np.mean(res["alpha"].get(t, {}).get(e_hold, [np.nan]))
            line.append(f"{t}: {s}" + (f" [a={ab:.2f}]" if np.isfinite(ab) else ""))
        print(f"[{e_hold:>7}] " + " | ".join(line), flush=True)

    with open(os.path.join(OUT, "exp47_hybrid.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("[exp47] 已写出 out/exp47_hybrid.json")
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
            if res["acc"][t].get(a, {}).get(e):
                d = [x[met] for x in res["acc"][t][a][e]
                     if x[met] is not None]
                if d:
                    v.append(float(np.mean(d)))
        return v

    L = []
    w = L.append
    w("# exp47 —— 混合头：化学外推 × 样本类中心\n")
    w("单元格格式：**top1 / auc**。\n")
    for met in ("top1", "auc"):
        w(f"\n## {met}\n")
        w("| k | " + " | ".join(f"`{a}`" for a in arms) + " |")
        w("|---|" + "---|" * len(arms))
        for t in tags:
            row = []
            for a in arms:
                v = M(t, a, met)
                row.append(f"{np.mean(v):.3f}" if v else "—")
            w(f"| {t} | " + " | ".join(row) + " |")

    w("\n## 选出的 α（1=纯化学，0=纯数据）\n")
    w("| k | α 均值 | α 逐实体 |")
    w("|---|---|---|")
    for t in tags:
        if not res["alpha"].get(t):
            w(f"| {t} | — | 不可定义（无样本类中心） |")
            continue
        per = [np.mean(v) for v in res["alpha"][t].values()]
        w(f"| {t} | {np.mean(per):.3f} | " +
          ", ".join(f"{e}:{np.mean(v):.2f}"
                    for e, v in res["alpha"][t].items()) + " |")

    w("\n## 判据\n")
    for t in tags:
        if t == "0":
            continue
        h = np.mean(M(t, "hybrid")); n = np.nanmean(M(t, "ncm"))
        r = np.mean(M(t, "ridge"))
        w(f"- **k={t}**：hybrid {h:.3f} / ncm(α=0) {n:.3f} / ridge(α=1) {r:.3f}"
          f" ⇒ H47a {'成立' if h >= n else '不成立'}，"
          f"H47b {'成立' if h >= r else '不成立'}")
        if HAS:
            for a, cmp in (("ncm", "hybrid"), ("ridge", "hybrid")):
                x, y = np.asarray(M(t, a), float), np.asarray(M(t, cmp), float)
                m = ~np.isnan(x) & ~np.isnan(y)
                if m.sum() > 1:
                    w(f"   - hybrid vs {a}: Δ={np.mean(y[m]) - np.mean(x[m]):+.3f}, "
                      f"p={ttest_rel(y[m], x[m]).pvalue:.4f}")
    als = [np.mean([np.mean(v) for v in res["alpha"][t].values()])
           for t in tags if res["alpha"].get(t)]
    if len(als) > 1:
        ks = [int(t) for t in tags if res["alpha"].get(t)]
        try:
            from scipy.stats import spearmanr
            rho, p = spearmanr(ks, als)
            w(f"\n- **H47d**：Spearman ρ(k, α) = {rho:+.3f} (p={p:.3f})；"
              f"若为负 ⇒ 样本越多模型越信数据")
        except Exception:
            pass

    with open(os.path.join(OUT, "exp47_p6_hybrid.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("[exp47] 已写出 out/exp47_p6_hybrid.md")


if __name__ == "__main__":
    main()
