"""exp36 · 化学先验的「对齐方式」消融：不是加不加先验，而是怎么对齐。

动机（来自 exp34 第一批结果，all19, n=5）
------------------------------------------
    None（可学习随机头）        0.744
    CLM 冻结（原始嵌入）        0.604   ← 比无先验低 14 个点
    CLM 冻结（去均值）          0.709   ← 缺口收窄到 3.5 个点
    CLM 初始化后微调            0.775   ← 高于基线
⇒ 化学先验「降低准确度」不是先验无效，而是**各向异性未校正**导致分类头几何退化。
   本实验把「对齐」作为唯一自变量，系统比较 9 种对齐策略。

对齐策略（唯一变量 = 分类头 W 与化学嵌入 T 的关系）
------------------------------------------------
  A. 无先验       None        nn.Linear 随机初始化、可学习
  B. 随机码本     Rand        冻结随机高斯（近似正交，零化学信息对照）
  C. 直接冻结     Raw         冻结 T（原始，未校正）
  D. 去均值       Center      冻结 T − mean(T)（All-but-the-Top 首步）
  E. 白化         Whiten      冻结 阻尼 ZCA 白化后的 T（去均值 + 各方向等方差）
  F. 去顶主成分   ABTT2       冻结 去均值 − top-2 主成分（All-but-the-Top 完整版）
  G. 先验初始化   Init        以 Center 初始化、随后微调
  H. 软锚定       Anchor(λ)   可学习头 + λ·‖Ŵ − T̂‖²（先验当正则，λ 可调）
  I. 对比对齐     AlignCE(μ)  可学习头 + μ·CE(z, T̂)（把对齐写成辅助对比损失）

对齐质量诊断（每个臂的分类头 W 在训练前的静态几何）
--------------------------------------------------
  cos_off     W 行两两余弦均值       —— 各向异性强度（越接近 0 越好）
  eff_rank90  去均值后 90% 能量所需秩 —— 分类头能否张满类别空间（越接近 ncls 越好）
  gini        预测分布的基尼系数     —— hubness（越接近 0 越均匀，无枢纽类）

预注册判据
----------
  H36a（主）：校正类对齐（Center/Whiten/ABTT2）显著优于未校正（Raw）
  H36b（主）：对齐质量指标 cos_off 与下游 Δ(vs None) **负相关**（Pearson r < 0 且 p<0.05）
             ⇒ 证明「对齐质量可测量，且能预测下游增益」
  H36c（次）：软锚定 Anchor 存在**中间最优 λ**（过大过小都更差）
             ⇒ 证明「加先验的方式需要一个强度旋钮」，而非全冻结/全自由二选一
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


# ---------------------------------------------------------------- 对齐变换
def center(T):
    T = np.asarray(T, dtype=np.float32)
    return (T - T.mean(0, keepdims=True)).astype(np.float32)


def whiten(T, eps=1e-2):
    """阻尼 ZCA 白化：去均值后把各主成分方向压缩到等方差。

    Tc = U diag(S) Vᵀ  ⇒  Ŵ = U diag(S/√(S²+eps)) Vᵀ
    eps 防止近零奇异值放大噪声（19 个候选在 384 维中必然秩亏）。
    """
    Tc = np.asarray(center(T), dtype=np.float64)
    U, S, Vt = np.linalg.svd(Tc, full_matrices=False)
    scale = S / np.sqrt(S ** 2 + eps)
    return (U * scale @ Vt).astype(np.float32)


def abtt(T, D=2):
    """All-but-the-Top：去均值后剔除 top-D 主成分（Mu & Viswanath, ICLR 2018）。"""
    Tc = np.asarray(center(T), dtype=np.float64)
    U, S, Vt = np.linalg.svd(Tc, full_matrices=False)
    D = min(D, len(S))
    return (Tc - (U[:, :D] * S[:D]) @ Vt[:D]).astype(np.float32)


# ---------------------------------------------------------------- 诊断
def align_diag(W, ncls):
    """分类头的静态几何质量：各向异性、有效秩。"""
    W = np.asarray(W, dtype=np.float64)
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    off = (Wn @ Wn.T)[~np.eye(len(W), dtype=bool)]
    Wc = W - W.mean(0, keepdims=True)
    s = np.linalg.svd(Wc, compute_uv=False)
    if s.sum() < 1e-12:
        erank = 0
    else:
        erank = int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), 0.90) + 1)
    return {"cos_off": float(off.mean()), "eff_rank90": int(erank),
            "eff_rank_frac": float(erank / max(ncls, 1))}


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64)).ravel()
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


# ---------------------------------------------------------------- 训练
def train_arm(Xtr, ytr, T, kind, dim, ncls, epochs, seed, lam=0.0, mu=0.0):
    """kind: learn | frozen | init | anchor | alignce"""
    torch.manual_seed(seed)
    net = E8.SpecCNN(dim).to(DEV)
    Tt = None if T is None else torch.tensor(np.asarray(T, dtype=np.float32), device=DEV)

    if kind in ("learn", "init", "anchor", "alignce"):
        head = nn.Linear(dim, ncls).to(DEV)
        if kind != "learn":
            with torch.no_grad():
                head.weight.copy_(Tt)
                head.bias.zero_()
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
                # 软锚定：把可学习分类头拉向化学嵌入方向（均在单位球面上比较，避免尺度混淆）
                loss = loss + lam * F.mse_loss(F.normalize(W, dim=1), F.normalize(Tt, dim=1))
            elif kind == "alignce" and mu > 0:
                # 对比对齐：冻结化学码本上的辅助对比损失 = 把对齐写成显式的检索目标
                loss = loss + mu * F.cross_entropy(z @ F.normalize(Tt, dim=1).T / TEMP, yt[idx])
            loss.backward(); opt.step()
    return net, (head.weight.detach() if head is not None else W)


@torch.no_grad()
def evaluate(net, W, Xte, yte):
    net.eval()
    z = F.normalize(net(torch.tensor(Xte, dtype=torch.float32, device=DEV)), dim=1)
    Wn = F.normalize(W, dim=1)
    pred = (z @ Wn.T).argmax(1).cpu().numpy()
    cnt = np.bincount(pred, minlength=Wn.shape[0]).astype(float)
    return float((pred == yte).mean()), gini(cnt)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="5,full")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lams", default="0.01,0.1,1")
    ap.add_argument("--mus", default="0.5")
    ap.add_argument("--repr", default="clm", choices=["clm", "ecfp"])
    args = ap.parse_args()

    shots = []
    for s in args.shots.split(","):
        shots.append(None if s.strip() == "full" else int(s))
    tags = [("full" if s is None else str(s)) for s in shots]

    print(f"[exp36] device = {DEV}  repr = {args.repr}", flush=True)
    Xs, keys, eids, mids = E34.build_all()
    key_index = {k: i for i, k in enumerate(keys)}
    tasks = E34.build_tasks(Xs, keys, eids)
    print(f"[exp36] 任务 {len(tasks)} 套: " + ", ".join(f"{nm}({len(ks)})" for nm, _, ks in tasks),
          flush=True)

    zc = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    assert [str(x) for x in zc["keys"]] == keys, "缓存 key 顺序不符"
    T_all = np.asarray(zc[args.repr], dtype=np.float32)
    dim = T_all.shape[1]

    lams = [float(x) for x in args.lams.split(",")]
    mus = [float(x) for x in args.mus.split(",")]

    def rand_W(n, d, seed):
        return np.random.default_rng(4000 + seed).normal(size=(n, d)).astype(np.float32)

    def arm_spec(ks):
        """返回 [(名字, kind, W生成函数或None, lam, mu)]，W 生成依赖 ks 与 seed。"""
        T = np.asarray(T_all)[[key_index[k] for k in ks]]
        Tc = center(T)
        out = [("None", "learn", lambda s: None, 0.0, 0.0),
               ("Rand", "frozen", lambda s: rand_W(len(ks), dim, s), 0.0, 0.0),
               ("Raw", "frozen", lambda s: T, 0.0, 0.0),
               ("Center", "frozen", lambda s: Tc, 0.0, 0.0),
               ("Whiten", "frozen", lambda s: whiten(T), 0.0, 0.0),
               ("ABTT2", "frozen", lambda s: abtt(T, 2), 0.0, 0.0),
               ("Init", "init", lambda s: Tc, 0.0, 0.0)]
        for l in lams:
            out.append((f"Anchor{l:g}", "anchor", lambda s: Tc, l, 0.0))
        for m in mus:
            out.append((f"AlignCE{m:g}", "alignce", lambda s: Tc, 0.0, m))
        # 软锚定/对比对齐的 W 目标仍是 Tc；Anchor 的初始化用 Tc，与 Init 一致以便比较
        return out

    res = {"acc": {}, "gini": {}, "diag": {}, "nte": {},
           "meta": {"repr": args.repr, "shots": tags, "seeds": args.seeds,
                    "steps": args.steps, "dim": dim, "lams": lams, "mus": mus,
                    "tasks": [nm for nm, _, _ in tasks]}}

    for task_name, (X, y), ks in tasks:
        ncls = len(ks)
        specs = arm_spec(ks)
        names = [s[0] for s in specs]
        res["acc"][task_name] = {t: {a: [] for a in names} for t in tags}
        res["gini"][task_name] = {t: {a: [] for a in names} for t in tags}
        res["diag"][task_name] = {}
        res["nte"][task_name] = {}
        for name, kind, wfun, lam, mu in specs:
            W0 = wfun(0)
            if W0 is None:
                res["diag"][task_name][name] = {"cos_off": float("nan"),
                                                "eff_rank90": ncls,
                                                "eff_rank_frac": 1.0}
            else:
                res["diag"][task_name][name] = align_diag(W0, ncls)
        for n in shots:
            tag = "full" if n is None else str(n)
            for seed in range(args.seeds):
                tr_pool, te = E34.stratified_split(y, ncls, seed=seed)
                res["nte"][task_name][tag] = len(te)
                rng = np.random.default_rng(seed * 7919 + 13)
                tr = E34.subsample(tr_pool, y, ncls, n, rng)
                Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
                spe = max(1, int(np.ceil(len(Xtr) / 256)))
                epochs = int(np.ceil(args.steps / spe))
                if n is None or n > 20:
                    epochs = max(epochs, 60)
                res["meta"].setdefault("epochs", {}).setdefault(task_name, {})[tag] = epochs
                for name, kind, wfun, lam, mu in specs:
                    net, W = train_arm(Xtr, ytr, wfun(seed), kind, dim, ncls, epochs,
                                       seed * 1000 + 7, lam=lam, mu=mu)
                    acc, g = evaluate(net, W, Xte, yte)
                    res["acc"][task_name][tag][name].append(float(acc))
                    res["gini"][task_name][tag][name].append(float(g))
            line = "  ".join(f"{a}={np.mean(res['acc'][task_name][tag][a]):.3f}" for a in names)
            print(f"[{task_name[:12]:>12} n={tag:>4}] {line}", flush=True)

    write_report(res, tasks, tags, args)
    with open(os.path.join(OUT, f"exp36_res_{args.repr}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[exp36] 已写出 out/exp36_p6_alignment_{args.repr}.md", flush=True)


# ---------------------------------------------------------------- 报告
def write_report(res, tasks, tags, args):
    from scipy.stats import pearsonr, ttest_rel
    A = []
    w = lambda s: A.append(s)
    task_names = [nm for nm, _, _ in tasks]
    within = [t for t in task_names if t.startswith("within-")]
    names = list(res["acc"][task_names[0]][tags[0]].keys())

    def pooled(tn, t, a, key="acc"):
        num = np.zeros(args.seeds); den = 0.0
        for x in tn:
            num = num + np.array(res[key][x][t][a]) * res["nte"][x][t]
            den += res["nte"][x][t]
        return num / max(den, 1e-9)

    w(f"# exp36 · 化学先验的「对齐方式」消融（表示 = {args.repr.upper()}）")
    w("")
    w("> 唯一自变量 = **分类头与化学嵌入的对齐方式**；编码器结构、训练样本、步数预算全部一致。")
    w("> 核心论点：**先验本身有效，问题出在对齐** —— 未校正的各向异性会让分类头几何退化。")
    w("")
    w("## 主结果（准确率）")
    w("")
    for scope, tn in [("all19", ["all19"])] + ([("within 合并", within)] if within else []):
        w(f"### {scope}")
        w("")
        w("| n | " + " | ".join(names) + " |")
        w("|---|" + "---|" * len(names))
        for t in tags:
            vals = [np.mean(pooled(tn, t, a)) for a in names]
            w(f"| {t} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
        w("")

    w("## 对齐质量诊断（训练前分类头的静态几何）")
    w("")
    w("| 任务 | 臂 | 行间余弦均值 cos_off ↓ | 有效秩(90%能量) / 类别数 ↑ |")
    w("|---|---|---|---|")
    for tn in task_names:
        ncls = len([k for k in res["diag"][tn]])
        for a in names:
            d = res["diag"][tn][a]
            w(f"| {tn} | {a} | {d['cos_off']:+.3f} | {d['eff_rank90']} ({d['eff_rank_frac']:.2f}) |")
    w("")

    w("## Δ 相对无先验基线 None")
    w("")
    for scope, tn in [("all19", ["all19"])] + ([("within 合并", within)] if within else []):
        w(f"### {scope}")
        w("")
        w("| n | " + " | ".join(a for a in names if a != "None") + " |")
        w("|---|" + "---|" * (len(names) - 1))
        for t in tags:
            base = pooled(tn, t, "None")
            cells = [f"{np.mean(pooled(tn, t, a) - base):+.4f}" for a in names if a != "None"]
            w(f"| {t} | " + " | ".join(cells) + " |")
        w("")

    # ------------------------------------------------- 判据
    w("## 预注册判据裁定")
    w("")
    # H36a：校正 vs 未校正
    for scope, tn in [("all19", ["all19"])] + ([("within 合并", within)] if within else []):
        for t in tags:
            raw = pooled(tn, t, "Raw")
            for a in ["Center", "Whiten", "ABTT2"]:
                d = pooled(tn, t, a) - raw
                ts, pv = ttest_rel(pooled(tn, t, a), raw)
                ok = d.mean() > 0 and pv < 0.05
                w(f"- **H36a** {scope} n={t}：{a} − Raw = {d.mean():+.4f}，p={pv:.4f} ⇒ "
                  f"{'✅' if ok else '❌'}")
    w("")
    # H36b：cos_off 与 Δ 的相关
    rows = []
    for tn in task_names:
        for t in tags:
            base = np.array(res["acc"][tn][t]["None"])
            for a in names:
                if a == "None":
                    continue
                d = np.array(res["acc"][tn][t][a]) - base
                rows.append((res["diag"][tn][a]["cos_off"],
                             res["diag"][tn][a]["eff_rank_frac"],
                             d.mean(), f"{tn}/{t}/{a}"))
    X = np.array([[r[0], r[1]] for r in rows])
    Y = np.array([r[2] for r in rows])
    ok_mask = ~np.isnan(X[:, 0])
    if ok_mask.sum() >= 5 and np.std(X[ok_mask, 0]) > 0:
        r1, p1 = pearsonr(X[ok_mask, 0], Y[ok_mask])
    else:
        r1, p1 = float("nan"), float("nan")
    if ok_mask.sum() >= 5 and np.std(X[ok_mask, 1]) > 0:
        r2, p2 = pearsonr(X[ok_mask, 1], Y[ok_mask])
    else:
        r2, p2 = float("nan"), float("nan")
    w(f"- **H36b**：Δ(vs None) ~ cos_off 的 Pearson r = {r1:+.3f}（p={p1:.4f}，"
      f"n={int(ok_mask.sum())} 个 臂×任务×样本档）⇒ {'✅ 负相关成立' if (r1 < 0 and p1 < 0.05) else '❌ 不成立'}")
    w(f"- **H36b'**：Δ ~ 有效秩占比 的 Pearson r = {r2:+.3f}（p={p2:.4f}）⇒ "
      f"{'✅ 正相关成立' if (r2 > 0 and p2 < 0.05) else '❌ 不成立'}")
    w("")
    # H36c：软锚定的中间最优 λ
    anch = [a for a in names if a.startswith("Anchor")]
    if len(anch) >= 3:
        w("### H36c：软锚定强度 λ 是否存在中间最优")
        w("")
        for scope, tn in [("all19", ["all19"])] + ([("within 合并", within)] if within else []):
            for t in tags:
                vals = [(float(a.replace("Anchor", "")), np.mean(pooled(tn, t, a))) for a in anch]
                s = "，".join(f"λ={l:g}:{v:.4f}" for l, v in vals)
                best = max(vals, key=lambda x: x[1])[0]
                inner = (best > min(l for l, _ in vals)) and (best < max(l for l, _ in vals))
                w(f"- {scope} n={t}：{s} ⇒ 最优 λ={best:g}（"
                  f"{'✅ 中间最优' if inner else '❌ 在端点'}）")
        w("")
    w("## 解读要点")
    w("")
    w("1. **Raw → Center/Whiten/ABTT 的跃迁**就是「对齐」的价值：先验没换，只是把嵌入空间")
    w("   校正到可分辨的几何，缺口就从 −14 个点收窄到 −3.5 个点甚至转正。")
    w("2. **软锚定 λ** 给出连续旋钮：λ→∞ 退化为冻结（Raw/Center），λ=0 退化为 Init。")
    w("   若存在中间最优 λ，说明「加先验」既不是全冻结也不是全自由。")
    w("3. **对比对齐 AlignCE** 把对齐写成显式检索损失，可与软锚定直接对比，")
    w("   回答「在参数空间对齐」还是「在损失层面对齐」更有效。")
    w("")

    txt = "\n".join(A)
    with open(os.path.join(OUT, f"exp36_p6_alignment_{args.repr}.md"), "w", encoding="utf-8") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
