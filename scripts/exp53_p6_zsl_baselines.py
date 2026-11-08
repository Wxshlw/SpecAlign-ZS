"""exp53 —— **真正的 ZSL 基线**对比（审稿人必问：你跟"能做零样本"的方法比呢？）

=============================================================================
缺口
=============================================================================
exp45/49 的外部基线（PLS-DA / SVM / RF / 谱库检索）在 k=0 时**结构性不可解**：
它们没有新类的任何训练样本，就无法为新类建立决策面 ⇒ top1 恒 0。
这虽然是个真实的结论，但审稿人可以合理质疑：
    "你说经典方法做不到零样本 —— 那你跟**真正为 ZSL 设计的方法**比过吗？"

本实验补齐这一缺口，全部在 **exp49 的同一套干净切分**上评估。

=============================================================================
对比的方法（都是 ZSL 领域公认、且 k=0 时可解的）
=============================================================================
  `clip`     CLIP 式对比对齐（Radford 2021）：双塔 + 可学习化学→嵌入投影 P，
             InfoNCE/CE over 19 类。零样本打分 = cos(z, P·s_j)。
             ★ 独立训练自己的谱塔，是本实验给对手的最大算力。
  `devise`   DEVISE（Frome NIPS 2013）：学 M: 嵌入→语义，hinge rank loss，
             预测 = argmax_j cos(M x, s_j)。
  `eszsl`    ESZSL（Romera-Paredes ICML 2015）：闭式解
             W = (ZᵗZ+λI)⁻¹ Zᵗ Y S (SᵗS+γI)⁻¹，打分 = z·W s_j。
  `ale`      ALE（Akata CVPR 2015）：学双线性 W，结构化 hinge + L2，打分 = zᵗ W s_j。
  `conse`    ConSE（Norouzi ICLR 2014）：已见类概率凸组合出语义向量，
             ŝ = Σ_j p_j·s_j，再与所有类（含未见）比余弦。无超参、无需训练。
  `ridge`    本文（exp49 主张臂）：W = (S−μ_S)·A，A 由老类 ridge 闭式解。

每个方法都在**两个编码器**上跑，用来隔离"编码器"和"对齐方式"：
  `-c`  本文的 chem 编码器（stage1 用去均值化学向量作冻结头）
  `-f`  无化学条件的 free 编码器（stage1 头可学习，编码器本身不含先验）
⇒ 若本文在 `-f` 下仍领先，则优势来自**对齐方式**而非编码器。

因果对照（把化学语义换成零信息的随机码本，同形状同范数）：
  `clip-rand` / `eszsl-crand` / `ridge-rand`

=============================================================================
预注册判据（写死，跑前定）
=============================================================================
- H53a（主，风险最高）：k=0 时本文 `ridge-c` 的 **auc** ≥ 最好的真 ZSL 基线
        （最可能是 `clip`）。**若被超过，如实报告并改写 claim。**
- H53b（主）：在**同一编码器内**比较（`-c` 组内、`-f` 组内），本文 `ridge`
        的 top1/auc ≥ devise / eszsl / ale / conse ⇒ 对齐方式本身有贡献。
- H53c（因果）：`clip`-rand / `eszsl-crand` / `ridge-rand` 的 auc 显著低于其
        真化学版本 ⇒ 这些方法的增益同样来自**化学语义内容**。
- H53d（边界）：k=0 时所有真 ZSL 基线的 top1 均 > 0（它们"能"零样本），
        但本文是否最高。
"""
from __future__ import annotations
import os, sys, json, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p6_g1_data as G1
import exp49_p6_clean_protocol as E49

DEV = E49.DEV
OUT = E49.OUT
TEMP = 0.07
OLD_FRAC_TRAIN = E49.OLD_FRAC_TRAIN     # 0.70，与 exp49 完全一致

ARMS = ["ridge-c", "ridge-f", "ridge-rand",
        "eszsl-c", "eszsl-f", "eszsl-crand",
        "devise-c", "devise-f",
        "ale-c", "ale-f",
        "conse-c", "conse-f",
        "clip", "clip-rand"]


# --------------------------------------------------------------------------
# 打分 → 指标（所有方法统一）
# --------------------------------------------------------------------------
def scores_to_metrics(S_new, S_old, new_row=18):
    """S_new / S_old 均为 (n, 19) 的打分矩阵。"""
    top1 = float((S_new.argmax(1) == new_row).mean())
    auc = E49.auc_of(S_new[:, new_row], S_old[:, new_row])
    ranks = np.where(np.argsort(-S_new, axis=1) == new_row)[1] + 1
    return dict(top1=round(top1, 4),
                auc=(None if not np.isfinite(auc) else round(float(auc), 4)),
                mrr=round(float((1.0 / ranks).mean()), 4))


def macro_val(Zv, Wv, yv, rows):
    """老类验证集上的 macro recall（超参选择准则，与 exp49 一致）"""
    Wt = E49.nrm(Wv)[rows]
    pred = (Zv @ Wt.T).argmax(1)
    rs = [float((pred[yv == j] == i).mean()) for i, j in enumerate(rows)
          if (yv == j).sum()]
    return float(np.mean(rs)) if rs else 0.0


def lin_scores(Z, W):
    return Z @ E49.nrm(W).T


# --------------------------------------------------------------------------
# 各 ZSL 方法的「头」构造（都返回 19×dim 的 W，打分 = z·W_j）
# --------------------------------------------------------------------------
def head_ridge(S, rows_old, Z_old, lam):
    A = E49.head_ridge_R(S[rows_old], Z_old, lam)
    mu = S[rows_old].mean(0, keepdims=True)
    return (S - mu) @ A


def head_eszsl(S, rows_old, Zfit, yfit, lam):
    """ESZSL 闭式解：W = (ZᵗZ+λI)⁻¹ Zᵗ Y S (SᵗS+γI)⁻¹ ；这里取 γ=λ 以免双参数调优"""
    Sr = S[rows_old]
    n = len(yfit)
    Y = np.zeros((n, len(rows_old)))
    for i, j in enumerate(rows_old):
        Y[:, i] = (yfit == j).astype(float)
    d = Zfit.shape[1]
    A = np.linalg.solve(Zfit.T @ Zfit + lam * np.eye(d), Zfit.T @ (Y @ Sr))
    B = np.linalg.solve(Sr.T @ Sr + lam * np.eye(Sr.shape[1]), np.eye(Sr.shape[1]))
    V = A @ B                      # dim × a
    return S @ V.T


def train_devise(Zfit, yfit, S, rows_old, a, margin, steps=500, lr=1e-2):
    """DEVISE：学 M: 嵌入→语义（a×dim），hinge rank loss。返回 W = S @ M"""
    Zt = torch.tensor(Zfit, dtype=torch.float32, device=DEV)
    yt = torch.tensor(yfit, dtype=torch.long, device=DEV)
    St = torch.tensor(S, dtype=torch.float32, device=DEV)      # 19 × a
    Stn = F.normalize(St, dim=1)
    M = nn.Parameter(torch.eye(a, Zfit.shape[1], device=DEV) * 0.1
                     + 0.01 * torch.randn(a, Zfit.shape[1], device=DEV))
    opt = torch.optim.Adam([M], lr=lr)
    n = len(Zfit)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(128, n),), device=DEV)
        z = Zt[idx]
        proj = F.normalize(z @ M.T, dim=1)                     # (b, a)
        sim = proj @ Stn.T                                     # (b, 19)
        pos = sim.gather(1, yt[idx].unsqueeze(1))
        mask = torch.ones_like(sim)
        mask.scatter_(1, yt[idx].unsqueeze(1), 0.0)            # 掩掉真实类自身
        loss = (F.relu(margin + sim - pos) * mask).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (St @ M).cpu().numpy()


def train_ale(Zfit, yfit, S, rows_old, lam, margin=1.0, steps=500, lr=1e-2):
    """ALE：学 W (dim×a)，打分 zᵗ W s，结构化 hinge + L2"""
    Zt = torch.tensor(Zfit, dtype=torch.float32, device=DEV)
    yt = torch.tensor(yfit, dtype=torch.long, device=DEV)
    St = torch.tensor(S, dtype=torch.float32, device=DEV)
    dim, a = Zfit.shape[1], S.shape[1]
    W = nn.Parameter(torch.zeros(dim, a, device=DEV))
    opt = torch.optim.Adam([W], lr=lr)
    n = len(Zfit)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(128, n),), device=DEV)
        z = Zt[idx]                                    # (b, dim)
        comp = z @ W @ St.T                            # (b, 19)
        pos = comp.gather(1, yt[idx].unsqueeze(1))
        mask = torch.ones_like(comp)
        mask.scatter_(1, yt[idx].unsqueeze(1), 0.0)
        loss = (F.relu(margin + comp - pos) * mask).sum(1).mean() \
               + lam * (W ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return St.cpu().numpy() @ W.detach().cpu().numpy().T     # 19 × dim


def conse_scores(Z, S, rows_old, centers, tau):
    """ConSE：p_j ∝ exp(cos(z, z̄_j)/τ)；ŝ = Σ p_j s_j；打分 = cos(ŝ, s_all)"""
    Cn = E49.nrm(centers)                              # (nrows_old, dim)
    P = torch.softmax(torch.tensor(Z @ Cn.T, dtype=torch.float32) / tau, dim=1)
    P = P.numpy()
    S_hat = P @ S[rows_old]                            # (n, a)
    S_hat = S_hat / (np.linalg.norm(S_hat, axis=1, keepdims=True) + 1e-12)
    Sn = E49.nrm(S)
    return S_hat @ Sn.T                                # (n, 19)


def train_clip(Xtr, ytr, S, dim, seed, steps=600, lr=3e-4):
    """CLIP 式对比对齐：谱塔 + 可学习化学投影 P；CE over 19 类（未见类只在负样本里）"""
    torch.manual_seed(seed)
    net = E49.build_net(Xtr.shape[1], dim, seed)
    a = S.shape[1]
    P = nn.Linear(a, dim, bias=False).to(DEV)
    opt = torch.optim.Adam(list(net.parameters()) + list(P.parameters()), lr=lr)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(ytr, dtype=torch.long, device=DEV)
    St = torch.tensor(S, dtype=torch.float32, device=DEV)
    n = len(Xtr)
    net.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (min(64, n),), device=DEV)
        z = F.normalize(net(Xt[idx]), dim=1)
        W = F.normalize(P(St), dim=1)                  # 19 × dim
        loss = F.cross_entropy(z @ W.T / TEMP, yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        W = F.normalize(P(St), dim=1).cpu().numpy()
    net.eval()
    return net, W


def rand_like(S, rng):
    """零化学信息对照：同形状、同每行范数的随机高斯矩阵"""
    R = rng.standard_normal(S.shape)
    R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    return R * np.linalg.norm(S, axis=1, keepdims=True)


# --------------------------------------------------------------------------
@torch.no_grad()
def embed(net, X):
    net.eval()
    return F.normalize(net(torch.tensor(X, dtype=torch.float32, device=DEV)),
                       dim=1).cpu().numpy()


def load_data():
    """G1.load_g1 每次要解析 xlsx（~44s）。缓存成 npz 后复用。"""
    p = os.path.join(OUT, "g1_cache.npz")
    if os.path.exists(p):
        d = np.load(p, allow_pickle=True)
        ent = {k: d[f"e_{k}"] for k in d["eids"]}
        bg = {k: d[f"m_{k}"] for k in d["mids"]}
        return ent, bg
    ent, bg, _ = G1.load_g1()
    np.savez_compressed(
        p, eids=np.array(sorted(ent)), mids=np.array(sorted(bg)),
        **{f"e_{k}": np.asarray(v, float) for k, v in ent.items()},
        **{f"m_{k}": np.asarray(v, float) for k, v in bg.items()})
    return ent, bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="0,5,20")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.001,0.01,0.1,1")
    ap.add_argument("--margins", default="0.1,1.0")
    ap.add_argument("--taus", default="0.01,0.07,0.5")
    ap.add_argument("--eids", default="")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    shots = [None if s.strip() == "full" else int(s) for s in args.shots.split(",")]
    tags = [("full" if s is None else str(s)) for s in shots]
    lams = [float(x) for x in args.lams.split(",")]
    margins = [float(x) for x in args.margins.split(",")]
    taus = [float(x) for x in args.taus.split(",")]

    ent, bg = load_data()
    eids = sorted(ent.keys())
    if args.eids:
        eids = [e for e in eids if e in args.eids.split(",")]
    mids = sorted(bg.keys())
    keys_all = sorted(ent.keys()) + [f"M_{m}" for m in mids]
    zc = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    T_all = np.asarray(zc["clm"], float)
    dim = T_all.shape[1]
    ki = {k: i for i, k in enumerate(keys_all)}

    res = {"acc": {t: {a: {} for a in ARMS} for t in tags},
           "hyp": {},
           "meta": {"shots": tags, "seeds": args.seeds, "eids": eids,
                    "lams": lams, "dim": dim, "arms": ARMS,
                    "old_frac_train": OLD_FRAC_TRAIN}}
    print(f"[exp53] device={DEV} shots={tags} seeds={args.seeds} arms={len(ARMS)}",
          flush=True)

    for ei, e_hold in enumerate(eids):
        others = [e for e in sorted(ent.keys()) if e != e_hold]
        S = np.asarray(T_all)[[ki[e] for e in others]
                              + [ki[f"M_{m}"] for m in mids] + [ki[e_hold]]]
        NEW = 18
        S_rand = rand_like(S, np.random.default_rng(1000 + ei))
        Xe = np.asarray(ent[e_hold], float)

        for k in shots:
            tag = "full" if k is None else str(k)
            for seed in range(args.seeds):
                # ---------- 与 exp49 完全一致的切分 ----------
                rng = np.random.default_rng(seed * 7919 + 13)
                perm = rng.permutation(len(Xe))
                n_te = int(min(max(5, round(0.3 * len(Xe))), len(Xe) - 1))
                te_ix, pool_ix = perm[:n_te], perm[n_te:]
                tr_e_ix = pool_ix if k is None else pool_ix[:min(k, len(pool_ix))]
                Xte_new = Xe[te_ix]
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
                nrows = len(others) + len(mids)
                if len(tr_e_ix):
                    parts.append(Xe[tr_e_ix])
                    ys.append(np.full(len(tr_e_ix), 18)); nrows = 19
                Xtr = np.concatenate(parts).astype(np.float32)
                ytr = np.concatenate(ys).astype(np.int64)
                Xte_old = np.concatenate(refs).astype(np.float32)

                # ---------- 验证掩码 ----------
                rng_v = np.random.default_rng(seed * 31 + 7)
                val_mask = np.zeros(len(Xtr), bool)
                rows_old = sorted({int(y) for y in ytr})
                for j in rows_old:
                    ix = rng_v.permutation(np.where(ytr == j)[0])
                    n_val = min(max(3, int(0.2 * len(ix))), max(0, len(ix) - 1))
                    val_mask[ix[:n_val]] = True
                Xfit, yfit = Xtr[~val_mask], ytr[~val_mask]
                Xval, yval = Xtr[val_mask], ytr[val_mask]
                val_rows = sorted({int(y) for y in yval})

                # ---------- 两个编码器 ----------
                nets = {}
                for mode in ("chem", "free"):
                    nets[mode], _ = E49.stage1(
                        Xfit, yfit, dim, nrows, args.steps,
                        seed * 1000 + ei, S=S, mode=mode)
                Z = {}
                for mode in ("chem", "free"):
                    Z[mode] = dict(
                        fit=embed(nets[mode], Xfit),
                        val=embed(nets[mode], Xval),
                        new=embed(nets[mode], Xte_new),
                        old=embed(nets[mode], Xte_old))
                Z_old = {m: np.stack([Z[m]["fit"][yfit == j].mean(0)
                                      for j in rows_old]) for m in Z}
                centers = {m: Z_old[m] for m in Z}

                # ---------------- 各臂 ----------------
                def put(arm, Sn_, Sm_):
                    res["acc"][tag].setdefault(arm, {}).setdefault(
                        e_hold, []).append(
                        scores_to_metrics(Sn_, Sm_, NEW))

                # ridge（本文 + 无化学编码器 + 随机码本对照）
                for mode, arm in (("chem", "ridge-c"), ("free", "ridge-f")):
                    best, bv = lams[0], -1.0
                    for lam in lams:
                        W = head_ridge(S, rows_old, Z_old[mode], lam)
                        v = macro_val(Z[mode]["val"], W, yval, val_rows)
                        if v > bv:
                            bv, best = v, lam
                    W = head_ridge(S, rows_old, Z_old[mode], best)
                    put(arm, lin_scores(Z[mode]["new"], W),
                        lin_scores(Z[mode]["old"], W))
                best, bv = lams[0], -1.0
                for lam in lams:
                    W = head_ridge(S_rand, rows_old, Z_old["chem"], lam)
                    v = macro_val(Z["chem"]["val"], W, yval, val_rows)
                    if v > bv:
                        bv, best = v, lam
                W = head_ridge(S_rand, rows_old, Z_old["chem"], best)
                put("ridge-rand", lin_scores(Z["chem"]["new"], W),
                    lin_scores(Z["chem"]["old"], W))

                # ESZSL
                for mode, arm, Sx in (("chem", "eszsl-c", S),
                                      ("free", "eszsl-f", S),
                                      ("chem", "eszsl-crand", S_rand)):
                    best, bv = lams[0], -1.0
                    for lam in lams:
                        W = head_eszsl(Sx, rows_old, Z[mode]["fit"], yfit, lam)
                        v = macro_val(Z[mode]["val"], W, yval, val_rows)
                        if v > bv:
                            bv, best = v, lam
                    W = head_eszsl(Sx, rows_old, Z[mode]["fit"], yfit, best)
                    put(arm, lin_scores(Z[mode]["new"], W),
                        lin_scores(Z[mode]["old"], W))

                # DEVISE（margin 固定 1.0=原论文默认，避免网格爆炸）
                for mode, arm in (("chem", "devise-c"), ("free", "devise-f")):
                    W = train_devise(Z[mode]["fit"], yfit, S, rows_old,
                                     S.shape[1], 1.0, steps=max(150, args.steps // 2))
                    put(arm, lin_scores(Z[mode]["new"], W),
                        lin_scores(Z[mode]["old"], W))

                # ALE
                for mode, arm in (("chem", "ale-c"), ("free", "ale-f")):
                    bW, bv = None, -1.0
                    for lam in (0.01, 0.1):
                        W = train_ale(Z[mode]["fit"], yfit, S, rows_old, lam,
                                      steps=max(150, args.steps // 2))
                        v = macro_val(Z[mode]["val"], W, yval, val_rows)
                        if v > bv:
                            bv, bW = v, W
                    put(arm, lin_scores(Z[mode]["new"], bW),
                        lin_scores(Z[mode]["old"], bW))

                # ConSE
                for mode, arm in (("chem", "conse-c"), ("free", "conse-f")):
                    bv, bt = -1.0, taus[0]
                    for tau in taus:
                        sc = conse_scores(Z[mode]["val"], S, rows_old,
                                          centers[mode], tau)
                        pred = sc[:, val_rows].argmax(1)
                        rs = [float((pred[yval == j] == i).mean())
                              for i, j in enumerate(val_rows) if (yval == j).sum()]
                        v = float(np.mean(rs)) if rs else 0.0
                        if v > bv:
                            bv, bt = v, tau
                    put(arm, conse_scores(Z[mode]["new"], S, rows_old,
                                          centers[mode], bt),
                        conse_scores(Z[mode]["old"], S, rows_old,
                                     centers[mode], bt))

                # CLIP（独立塔）
                for arm, Sx in (("clip", S), ("clip-rand", S_rand)):
                    net_c, W = train_clip(Xtr, ytr, Sx, dim,
                                          seed * 1000 + ei, steps=args.steps)
                    zn = embed(net_c, Xte_new); zo = embed(net_c, Xte_old)
                    put(arm, lin_scores(zn, W), lin_scores(zo, W))
                    del net_c
                torch.cuda.empty_cache()

        # ---- 进度（逐实体从 json 重算，不累计）----
        line = [f"[{e_hold}]"]
        for t in tags:
            for a in ("ridge-c", "clip", "devise-c", "eszsl-c", "ale-c", "conse-c"):
                d = res["acc"][t].get(a, {}).get(e_hold, [])
                if d:
                    line.append(f"{t}:{a}={np.mean([x['auc'] for x in d if x['auc'] is not None]):.3f}"
                                f"/{np.mean([x['top1'] for x in d]):.3f}")
        print(" ".join(line), flush=True)

    SFX = args.suffix
    with open(os.path.join(OUT, f"exp53_zsl{SFX}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    print(f"[exp53] 已写出 out/exp53_zsl{SFX}.json", flush=True)
    write_report(res, tags, eids, SFX)


# --------------------------------------------------------------------------
def per_entity(res, t, a, met):
    out = {}
    for e in res["meta"]["eids"]:
        v = [x[met] for x in res["acc"][t].get(a, {}).get(e, [])
             if x[met] is not None]
        if v:
            out[e] = float(np.mean(v))
    return out


def write_report(res, tags, eids, SFX=""):
    from scipy.stats import ttest_rel, ttest_1samp
    L = []
    w = L.append
    w("# exp53 —— 真正的 ZSL 基线对比（DEVISE / ESZSL / ALE / ConSE / CLIP × 本文）\n")
    w("与 exp49 **同一套干净切分**（老类 70/30，30% 作 AUC 负样本且不进训练集）。")
    w("`-c` = 本文 chem 编码器，`-f` = 无化学条件的 free 编码器；"
      "同编码器内比较可隔离「对齐方式」。\n")

    for t in tags:
        w(f"\n## k={t}\n")
        for met in ("auc", "top1", "mrr"):
            w(f"\n### {met}（macro，13 实体等权）\n")
            w("| 臂 | " + " | ".join(eids) + " | **macro** |")
            w("|---|" + "---|" * (len(eids) + 1))
            rows = []
            for a in ARMS:
                d = per_entity(res, t, a, met)
                if not d:
                    continue
                cells = [f"{d[e]:.3f}" if e in d else "—" for e in eids]
                rows.append((a, np.mean(list(d.values())),
                             "| " + f"`{a}` | " + " | ".join(cells) +
                             f" | **{np.mean(list(d.values())):.3f}** |"))
            for _, m, line in sorted(rows, key=lambda x: -x[1]):
                w(line)

    w("\n## 判据\n")
    for t in tags:
        w(f"\n### k={t}")
        ours = per_entity(res, t, "ridge-c", "auc")
        if not ours:
            continue
        x = np.array([ours[e] for e in eids])
        w(f"- 本文 `ridge-c` auc = {x.mean():.3f}（vs 0.5：p={ttest_1samp(x, 0.5).pvalue:.4f}）")
        cands = [a for a in ARMS if a != "ridge-c" and len(per_entity(res, t, a, "auc"))]
        scored = sorted(((a, np.mean(list(per_entity(res, t, a, 'auc').values())))
                         for a in cands), key=lambda z: -z[1])
        for a, m in scored:
            y = np.array([per_entity(res, t, a, "auc")[e] for e in eids
                          if e in per_entity(res, t, a, "auc")])
            xx = np.array([ours[e] for e in eids if e in per_entity(res, t, a, "auc")])
            if len(xx) > 1:
                w(f"   - vs `{a}` ({m:.3f}): Δ={np.mean(xx - y):+.3f} "
                  f"p={ttest_rel(xx, y).pvalue:.4f} "
                  f"{'✅ 本文领先' if np.mean(xx - y) > 0 and ttest_rel(xx, y).pvalue < 0.05 else ''}"
                  f"{'❌ 本文落后' if np.mean(xx - y) < 0 and ttest_rel(xx, y).pvalue < 0.05 else ''}")
        best_a, best_m = scored[0]
        w(f"- **H53a**：最好的真 ZSL 基线 = `{best_a}` ({best_m:.3f})"
          f" ⇒ {'✅ 成立（本文 ≥ 最好基线）' if x.mean() >= best_m else '❌ 不成立，本文被超过，须改写 claim'}")
        # H53b：同编码器组内
        for grp, ourname, mem in (("-c 组", "ridge-c", ["devise-c", "eszsl-c", "ale-c", "conse-c"]),
                                  ("-f 组", "ridge-f", ["devise-f", "eszsl-f", "ale-f", "conse-f"])):
            od = per_entity(res, t, ourname, "auc")
            if not od:
                continue
            wins = []
            for a in mem:
                ad = per_entity(res, t, a, "auc")
                if not ad:
                    continue
                d = np.mean([od[e] for e in ad]) - np.mean(list(ad.values()))
                wins.append(f"`{a}`{d:+.3f}")
            w(f"   - H53b {grp}：" + "，".join(wins))
        # H53c 因果
        for real, rnd in (("ridge-c", "ridge-rand"), ("clip", "clip-rand"),
                          ("eszsl-c", "eszsl-crand")):
            a1 = per_entity(res, t, real, "auc"); a2 = per_entity(res, t, rnd, "auc")
            if a1 and a2:
                kk = [e for e in a1 if e in a2]
                d = np.array([a1[e] - a2[e] for e in kk])
                w(f"   - H53c `{real}` − `{rnd}`(零化学信息): Δ={d.mean():+.3f} "
                  f"p={ttest_rel([a1[e] for e in kk], [a2[e] for e in kk]).pvalue:.4f}")

    p = os.path.join(OUT, f"exp53_p6_zsl_baselines{SFX}.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[exp53] 已写出 {p}", flush=True)


if __name__ == "__main__":
    main()
