"""exp37 · 跨模态结构一致性（CKA）能否预测「化学先验该不该加」？

背景：为什么必须做这个实验
--------------------------
exp34 的闭集结果是一个**负结果**，而且带强异质性：

    all19 (n_te=741)      Δ(CLM-init − None) = +0.031 (n=5) … −0.049 (n=20)，全不显著
    within-cinnamon(360)  Δ 从 +0.030 恶化到 **−0.341 (p=0.008)**，样本越多越有害
    within-meat (24)      Δ 为正但功效不足（n_te=24，SE≈0.10）
    within-coffeeI(11)    天花板效应，无信息

「加先验有时有害」必须有一个**可事前计算**的解释，否则论文只能停在经验观察。
本实验检验的核心假设：

    **化学先验的增益由「化学空间」与「谱空间」的类间几何结构一致性决定。**
    结构一致 → 先验是正确归纳偏置 → 增益为正；
    结构错配 → 先验把分类头锚在错误的几何上 → 增益转负。

实测（19 个候选全集）：Spearman ρ(化学余弦, 谱余弦) = +0.14，linear CKA = 0.23
⇒ 两个空间的**二阶关系几乎不相关**，这正是直接把化学嵌入当分类头会失效的根因。

设计
----
1. 从 19 个候选中随机抽取 m ∈ {5,6,7,8} 类的子集（保证 ≥10 个类对，使结构指标可估）
2. 对每个子集事前计算两个结构一致性指标：
     ρ_struct = Spearman(化学类间余弦, 谱类间余弦)   （秩相关，尺度无关）
     CKA      = linear CKA(化学嵌入, 类平均谱)        （Kornblith et al., ICML 2019）
3. 按 CKA 分位数**分层抽样**出 N 个子集，保证覆盖从「错配」到「一致」的完整区间
4. 每个子集跑 4 个臂：None（无先验）/ Init（先验初始化后微调）/
   Anchor0.1（软锚定）/ Center（冻结去均值先验）
5. 以**子集**为单位，检验 Δ 与结构一致性的关系

预注册判据
----------
  H37a（主）：Δ_Init = Acc(Init) − Acc(None) 与 CKA 显著正相关（Pearson p<0.05）
  H37b（主）：存在 CKA 阈值，低于该阈值时 Δ_Init 的均值为**负**
             ⇒ 证明「何时该加先验」可由事前指标判定
  H37c（次）：软锚定 Anchor 的 Δ 对 CKA 的**依赖斜率小于** Init
             ⇒ 证明对齐方式能降低对结构一致性的依赖（对齐方法的价值）
"""
from __future__ import annotations
import os, sys, json, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp34_p6_closedset_prior as E34
import exp08_p6_g1 as E8

OUT = r"D:/2026sci论文实验设计/论文6/out"
TEMP = 0.07
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- 结构指标
def offdiag(C):
    iu = np.triu_indices(len(C), 1)
    return C[iu]


def cos_sim(M):
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    return Mn @ Mn.T


def linear_cka(A, B):
    """linear CKA（Kornblith et al., ICML 2019）；A,B 形状 (n, d)，按样本（此处=类别）对齐。"""
    A = np.asarray(A, float); B = np.asarray(B, float)
    A = A - A.mean(0, keepdims=True)
    B = B - B.mean(0, keepdims=True)
    num = np.linalg.norm(A.T @ B) ** 2
    den = np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)
    return float(num / den) if den > 0 else float("nan")


def struct_indices(T, Xc):
    """T: (m, d_chem) 化学嵌入；Xc: (m, d_spec) 类平均谱。"""
    from scipy.stats import spearmanr
    Cc, Cs = cos_sim(T), cos_sim(Xc)
    a, b = offdiag(Cc), offdiag(Cs)
    rho = spearmanr(a, b).statistic if len(a) >= 4 else float("nan")
    return {"rho_struct": float(rho), "cka": linear_cka(T, Xc)}


# ---------------------------------------------------------------- 训练（与 exp34 同构）
def train_arm(Xtr, ytr, W0, kind, dim, ncls, epochs, seed, lam=0.0):
    torch.manual_seed(seed)
    net = E8.SpecCNN(dim).to(DEV)
    Tt = None if W0 is None else torch.tensor(np.asarray(W0, dtype=np.float32), device=DEV)
    if kind in ("learn", "init", "anchor"):
        head = nn.Linear(dim, ncls).to(DEV)
        if kind != "learn":
            with torch.no_grad():
                head.weight.copy_(Tt); head.bias.zero_()
        W = head.weight
        params = list(net.parameters()) + list(head.parameters())
    else:  # frozen
        W = Tt
        params = list(net.parameters())
        head = None
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(ytr, dtype=torch.long, device=DEV)
    gen = torch.Generator().manual_seed(seed)
    n = len(Xt)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            z = F.normalize(net(Xt[idx]), dim=1)
            loss = F.cross_entropy(z @ F.normalize(W, dim=1).T / TEMP, yt[idx])
            if kind == "anchor" and lam > 0:
                loss = loss + lam * F.mse_loss(F.normalize(W, dim=1), F.normalize(Tt, dim=1))
            loss.backward(); opt.step()
    return net, (head.weight.detach() if head is not None else W)


@torch.no_grad()
def accuracy(net, W, Xte, yte):
    net.eval()
    z = F.normalize(net(torch.tensor(Xte, dtype=torch.float32, device=DEV)), dim=1)
    pred = (z @ F.normalize(W, dim=1).T).argmax(1).cpu().numpy()
    return float((pred == yte).mean())


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subsets", type=int, default=48)
    ap.add_argument("--n-cand", type=int, default=400, help="先抽这么多候选子集用于 CKA 分层")
    ap.add_argument("--sizes", default="8,9,10,11")
    ap.add_argument("--pool", default="entities", choices=["all19", "entities"],
                    help="抽子集的候选池。entities = 只从 13 个掺假实体抽（排除 6 个"
                         "基质背景）；把背景算进去会让任务过于简单（实测 None 已到 0.93~0.99，"
                         "Δ 被天花板压缩到 0，检验不出结构一致性的效应）。")
    ap.add_argument("--n-shot", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--repr", default="clm", choices=["clm", "ecfp"])
    args = ap.parse_args()

    print(f"[exp37] device = {DEV}  repr = {args.repr}", flush=True)
    Xs, keys, eids, mids = E34.build_all()
    key_index = {k: i for i, k in enumerate(keys)}
    zc = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    assert [str(x) for x in zc["keys"]] == keys
    T_all = np.asarray(zc[args.repr], dtype=np.float32)
    dim = T_all.shape[1]

    # 类平均谱（结构指标的「谱侧」，可事前计算，不依赖任何训练）
    Xc = np.stack([np.asarray(Xs[k], float).mean(0) for k in keys]).astype(np.float32)
    # 每个类的样本数（用于可行性检查）
    sizes_arr = np.array([len(Xs[k]) for k in keys])

    sizes = [int(x) for x in args.sizes.split(",")]
    need = args.n_shot + 5          # 训练 n 个 + 至少 5 个测试
    base_ok = (np.array([not k.startswith("M_") for k in keys]) if args.pool == "entities"
               else np.ones(len(keys), dtype=bool))
    for need in (args.n_shot + 5, args.n_shot + 2):
        feasible = base_ok & (sizes_arr >= need)
        pool = [i for i in range(len(keys)) if feasible[i]]
        if len(pool) >= max(sizes):
            break
    print(f"[exp37] 候选池 = {args.pool}；样本数 min/median/max = "
          f"{sizes_arr.min()}/{int(np.median(sizes_arr))}/{sizes_arr.max()}；"
          f"可行类（≥{need}）= {len(pool)}", flush=True)

    # ------------------------------------------------ 1. 抽候选子集并算结构指标
    rng = np.random.default_rng(20261102)
    cand = []
    for _ in range(args.n_cand):
        m = int(rng.choice(sizes))
        if len(pool) < m:
            continue
        ix = rng.choice(pool, size=m, replace=False)
        ix = np.sort(ix)
        T = T_all[ix]
        s = struct_indices(T, Xc[ix])
        if np.isnan(s["cka"]) or np.isnan(s["rho_struct"]):
            continue
        cand.append((ix, s["cka"], s["rho_struct"]))
    print(f"[exp37] 候选子集 {len(cand)} 个；CKA 范围 "
          f"{min(c[1] for c in cand):.3f} ~ {max(c[1] for c in cand):.3f}", flush=True)

    # ------------------------------------------------ 2. 按 CKA 分位数分层抽取
    cand.sort(key=lambda c: c[1])
    N = min(args.n_subsets, len(cand))
    picks = [int(round(i * (len(cand) - 1) / max(N - 1, 1))) for i in range(N)]
    subsets = [cand[p] for p in picks]
    print(f"[exp37] 分层抽取 {len(subsets)} 个子集，CKA "
          f"{subsets[0][1]:.3f} ~ {subsets[-1][1]:.3f}", flush=True)

    arms = ["None", "Init", f"Anchor{args.lam:g}", "Center"]
    res = {"acc": [], "diag": [], "meta": {"repr": args.repr, "n_shot": args.n_shot,
                                           "seeds": args.seeds, "steps": args.steps,
                                           "lam": args.lam, "arms": arms, "dim": dim}}

    for si, (ix, cka, rho) in enumerate(subsets):
        ks = [keys[i] for i in ix]
        ncls = len(ks)
        parts, ys = [], []
        for j, i in enumerate(ix):
            parts.append(Xs[keys[i]]); ys.append(np.full(len(Xs[keys[i]]), j))
        X = np.concatenate(parts).astype(np.float32)
        y = np.concatenate(ys).astype(np.int64)

        T = T_all[ix]
        Tc = (T - T.mean(0, keepdims=True)).astype(np.float32)

        rec = {"subset": int(si), "keys": ks, "cka": float(cka), "rho": float(rho),
               "acc": {a: [] for a in arms}}
        for seed in range(args.seeds):
            tr_pool, te = E34.stratified_split(y, ncls, seed=seed, train_frac=0.7)
            r2 = np.random.default_rng(seed * 7919 + 13)
            tr = E34.subsample(tr_pool, y, ncls, args.n_shot, r2)
            Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
            spe = max(1, int(np.ceil(len(Xtr) / 256)))
            epochs = int(np.ceil(args.steps / spe))
            for a in arms:
                if a == "None":
                    kind, W0, lam = "learn", None, 0.0
                elif a == "Init":
                    kind, W0, lam = "init", Tc, 0.0
                elif a == "Center":
                    kind, W0, lam = "frozen", Tc, 0.0
                else:
                    kind, W0, lam = "anchor", Tc, args.lam
                net, W = train_arm(Xtr, ytr, W0, kind, dim, ncls, epochs, seed * 1000 + 7, lam=lam)
                rec["acc"][a].append(float(accuracy(net, W, Xte, yte)))
        rec["nte"] = len(te)
        res["acc"].append(rec)
        if (si + 1) % 4 == 0 or si == 0:
            line = " ".join(f"{a}={np.mean(rec['acc'][a]):.3f}" for a in arms)
            print(f"  [{si:3d}] m={ncls} cka={cka:.3f} rho={rho:+.2f} | {line}", flush=True)

    write_report(res, arms, args)
    with open(os.path.join(OUT, f"exp37_res_{args.repr}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[exp37] 已写出 out/exp37_p6_cka_{args.repr}.md", flush=True)


# ---------------------------------------------------------------- 报告
def write_report(res, arms, args):
    from scipy.stats import pearsonr, spearmanr, ttest_1samp
    A = []
    w = lambda s: A.append(s)
    recs = res["acc"]
    cka = np.array([r["cka"] for r in recs])
    rho = np.array([r["rho"] for r in recs])
    acc = {a: np.array([np.mean(r["acc"][a]) for r in recs]) for a in arms}
    base = acc["None"]

    w(f"# exp37 · 跨模态结构一致性（CKA）能否预测「化学先验该不该加」（{args.repr.upper()}）")
    w("")
    w("> **要解决的问题**：exp34 在闭集上是一个带强异质性的负结果 ——")
    w("> `all19` 增益不显著，`within-cinnamon` 显著有害（Δ=−0.341, p=0.008）。")
    w("> 本文检验：这个异质性可由**事前可计算的**跨模态结构一致性解释并预测。")
    w("")
    w(f"- 子集数 {len(recs)}，类别数 m ∈ {{{args.sizes}}}，每类训练样本 {args.n_shot}，"
      f"seeds {args.seeds}，{args.steps} 步预算")
    w("- 子集按 **CKA 分位数分层**抽取，保证覆盖从「结构错配」到「结构一致」的完整区间")
    w("")

    w("## 主结果：Δ 与结构一致性的关系")
    w("")
    w("| 臂 | Δ 均值（相对 None） | 与 CKA 的 Pearson r | p | 与 ρ_struct 的 Spearman r | p |")
    w("|---|---|---|---|---|---|")
    corrs = {}
    for a in arms:
        if a == "None":
            continue
        d = acc[a] - base
        r1, p1 = pearsonr(cka, d)
        r2, p2 = spearmanr(rho, d)
        corrs[a] = {"pearson_cka": float(r1), "p_cka": float(p1),
                    "spearman_rho": float(r2), "p_rho": float(p2), "d_mean": float(d.mean())}
        w(f"| {a} | {d.mean():+.4f} | {r1:+.3f} | {p1:.4f} | {r2:+.3f} | {p2:.4f} |")
    w("")

    w("## 分箱：Δ 随 CKA 的变化（关键图的数据表）")
    w("")
    qs = np.quantile(cka, [0, 0.25, 0.5, 0.75, 1.0])
    w("| CKA 分箱 | 区间 | 子集数 | " + " | ".join(a for a in arms if a != "None") + " |")
    w("|---|---|---|" + "---|" * (len(arms) - 1))
    bins = {}
    for b in range(4):
        m = (cka >= qs[b]) & (cka <= qs[b + 1]) if b == 3 else (cka >= qs[b]) & (cka < qs[b + 1])
        if m.sum() == 0:
            continue
        cells = [f"{np.mean((acc[a] - base)[m]):+.4f}" for a in arms if a != "None"]
        bins[f"Q{b+1}"] = {"lo": float(qs[b]), "hi": float(qs[b + 1]), "n": int(m.sum()),
                           **{a: float(np.mean((acc[a] - base)[m])) for a in arms if a != "None"}}
        w(f"| Q{b+1} | [{qs[b]:.2f}, {qs[b+1]:.2f}) | {int(m.sum())} | " + " | ".join(cells) + " |")
    w("")

    w("## 预注册判据裁定")
    w("")
    ci = corrs.get("Init", {})
    r1, p1 = ci.get("pearson_cka", float("nan")), ci.get("p_cka", float("nan"))
    h37a = (r1 > 0) and (p1 < 0.05)
    w(f"- **H37a**（Δ_Init 与 CKA 正相关）：r = {r1:+.3f}，p = {p1:.4f} ⇒ "
      f"{'✅ 成立' if h37a else '❌ 不成立'}")
    lo_m = cka <= qs[1]
    if lo_m.sum() >= 3:
        d_low = (acc["Init"] - base)[lo_m]
        ts, pv = ttest_1samp(d_low, 0.0)
        h37b = (d_low.mean() < 0) and (pv < 0.05)
        w(f"- **H37b**（低 CKA 区 Δ_Init 显著为负）：Q1 内 Δ = {d_low.mean():+.4f}，"
          f"单样本 t p = {pv:.4f}（n={int(lo_m.sum())}）⇒ {'✅ 成立' if h37b else '❌ 不成立'}")
    else:
        h37b = False
        w("- **H37b**：低 CKA 区子集不足，无法裁定")
    an = corrs.get(f"Anchor{args.lam:g}", {})
    h37c = False
    if an and "Init" in corrs:
        # 用分箱极差（max−min）作为「对 CKA 的依赖强度」
        def rng_(a):
            v = [bins[b][a] for b in bins if a in bins[b]]
            return max(v) - min(v) if len(v) >= 2 else float("nan")
        r_init, r_anch = rng_("Init"), rng_(f"Anchor{args.lam:g}")
        h37c = (not np.isnan(r_init)) and (not np.isnan(r_anch)) and (r_anch < r_init)
        w(f"- **H37c**（软锚定降低对 CKA 的依赖）：分箱极差 Init = {r_init:.4f} vs "
          f"Anchor = {r_anch:.4f} ⇒ {'✅ 成立' if h37c else '❌ 不成立'}")
    w("")

    w("## 解读")
    w("")
    if h37a:
        w("结构一致性**可以事前预测**化学先验的增益：CKA 越高，先验越可能带来正增益。")
        w("这把「要不要加先验」从一个玄学选择变成一个**可计算的判据**。")
    else:
        w("未观察到 CKA 与先验增益的显著关系 —— 需改用其他结构指标或在更大子集上重估。")
    w("")
    w("**方法论含义**：化学先验不是「加了就好」。它只在化学空间的类间几何与谱空间")
    w("的类间几何相容时才是正确归纳偏置；否则它把分类头锚在一个结构错配的空间上，")
    w("**主动损害**判别能力。这正是「怎么对齐」必须被认真对待的原因。")
    w("")

    txt = "\n".join(A)
    with open(os.path.join(OUT, f"exp37_p6_cka_{args.repr}.md"), "w", encoding="utf-8") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
