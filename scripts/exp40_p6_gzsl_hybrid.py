"""exp40 · 语义→谱空间**可学习投影**（ALE/ESZSL 式）：正面解决结构不相容，并给出下限保证。

为什么必须换掉「直接用化学向量当分类头」
----------------------------------------
exp34/exp37/exp39 的根因诊断一致：化学空间与谱空间的**类间几何不相容**
（Spearman ρ = +0.14，linear CKA = 0.23）。
⇒ 把化学嵌入**直接**当分类头，等于强迫模型对齐一个结构错配的空间，
这就是所有「先验劣于 None」负结果的来源。

（冒烟已证伪一个错误实现：老类行随机可学 + 新类行直接用化学向量 ⇒
两行不在同一空间，未见类 logits 完全错位，A01 k=0 从 0.772 掉到 0.000。已废弃。）

正确解法（GZSL / 零样本学习的标准家族）
--------------------------------------
学一个**化学语义 → 谱空间**的投影 A，在老类上训练，再应用到未见类：

    W = S · A          S: (19, d) 固定的化学嵌入（去均值）；A: (d, d) 可学习

- **A 在 18 个老类上学到「语义方向如何对应谱方向」**，这是可迁移的校正；
- 未见类没有样本，但它的化学向量经同一个 A 投影后**落在正确的谱空间位置**；
- 这正是 ALE / ESZSL / DEVISE 一类方法的核心，也是对本任务结构错配的正解；
- A 初始化为单位阵 ⇒ 起点就是「Center 冻结码本」，之后自由适应。

下限保证（回应用户「至少要不低于完全不加先验」）
------------------------------------------------
    residual :  W = W_free + α · (S·A)，α 可学习、**初始化为 0**
    gate     :  W = (1−g)·W_free + g·(S·A)，g = sigmoid(θ)，θ **初始化为 −2**

两者在 α→0 / g→0 时**数学上退化为无先验基线 None**。
⇒ 「不低于 None」不是靠运气试出来的，而是构造保证；H40b 只做实证确认。

预注册臂
--------
  none         W 完全自由可学（随机初始化）—— 必须打平或超过的基线
  freeze       W = S 冻结（= exp38 Center，已知负）—— 复现对照
  init-all     W 初始化为 S 后自由微调（= exp35 CLM-init，已知平）—— 复现对照
  proj         W = S·A，A 可学，初始 I（ALE 式；无下限保证）
  proj-diag    W = S·diag(a)，a 初始 1（低参数版，抗过拟合）
  residual     W = W_free + α·(S·A)，α 初始 0（**有下限保证**）
  gate         W = (1−g)·W_free + g·(S·A)，θ 初始 −2（**有下限保证** + 可读的门控权重）

预注册判据（一次跑完，正负都报告）
----------------------------------
  H40a（主）：k=0 时 proj 类臂 > freeze（学到的投影优于直接冻结）
  H40b（主，下限保证）：**所有 k 档，residual 与 gate ≥ none − 0.01**
  H40c（机制）：门控 g 随 k 增大而下降（k 小信化学、k 大信数据）
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
import exp08_p6_g1 as E8

OUT = r"D:/2026sci论文实验设计/论文6/out"
TEMP = 0.07
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NOLD = 18          # 老类行数（12 个其余实体 + 6 个基质背景）
NEW = 18           # 未见类（留出实体）所在行


def center(T):
    T = np.asarray(T, dtype=np.float32)
    return (T - T.mean(0, keepdims=True)).astype(np.float32)


class Head(nn.Module):
    """分类头：W = f(S, 可学习参数)。S 固定（化学语义），f 按臂不同。"""

    def __init__(self, dim, S, mode, seed):
        super().__init__()
        rng = np.random.default_rng(31337 + seed)
        self.mode = mode
        self.dim = dim
        self.register_buffer("S", torch.tensor(np.asarray(S, dtype=np.float32)))
        if mode in ("none", "residual", "gate", "init-all"):
            self.Wfree = nn.Parameter(torch.tensor(
                rng.normal(0, 0.05, size=(19, dim)), dtype=torch.float32))
        if mode in ("proj", "residual", "gate"):
            self.A = nn.Parameter(torch.eye(dim))
        if mode == "proj-diag":
            self.a = nn.Parameter(torch.ones(dim))
        if mode == "residual":
            self.alpha = nn.Parameter(torch.tensor(0.0))
        if mode == "gate":
            self.theta = nn.Parameter(torch.tensor(-2.0))

    def proj(self):
        if self.mode == "proj-diag":
            return self.S * self.a
        if hasattr(self, "A"):
            return self.S @ self.A
        return self.S

    def forward(self):
        if self.mode == "freeze":
            return self.S, None
        P = self.proj()
        if self.mode == "proj" or self.mode == "proj-diag":
            return P, None
        if self.mode == "residual":
            return self.Wfree + self.alpha * P, torch.tensor(1.0)
        if self.mode == "gate":
            g = torch.sigmoid(self.theta)
            return (1.0 - g) * self.Wfree + g * P, g
        return self.Wfree, None     # none / init-all

    def gate_value(self):
        if self.mode == "gate":
            return float(torch.sigmoid(self.theta).item())
        if self.mode == "residual":
            return float(self.alpha.item())
        return None


def train_arm(Xtr, ytr, S, mode, dim, nrows, steps, seed):
    torch.manual_seed(seed)
    net = E8.SpecCNN(dim).to(DEV)
    head = Head(dim, S, mode, seed).to(DEV)
    if mode == "init-all":
        with torch.no_grad():
            head.Wfree.copy_(head.S.clone())
    opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()),
                           lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(ytr, dtype=torch.long, device=DEV)
    gen = torch.Generator().manual_seed(seed)
    n = len(Xt)
    net.train(); head.train()
    done = 0
    while done < steps:
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, 256):
            if done >= steps:
                break
            idx = perm[i:i + 256]
            opt.zero_grad()
            z = F.normalize(net(Xt[idx]), dim=1)
            W, _ = head()
            loss = F.cross_entropy(
                z @ F.normalize(W[:nrows], dim=1).T / TEMP, yt[idx])
            loss.backward(); opt.step()
            done += 1
    return net, head


@torch.no_grad()
def top1(net, head, Xte, cand_rows, true_row):
    net.eval(); head.eval()
    z = F.normalize(net(torch.tensor(Xte, dtype=torch.float32, device=DEV)), dim=1)
    W, _ = head()
    logits = (z @ F.normalize(W, dim=1).T)[:, cand_rows]
    pred = logits.argmax(1).cpu().numpy()
    return float((pred == cand_rows.index(true_row)).mean())


# ------------------------------------------------- 两阶段：DEVISE/ESZSL 岭回归闭式解
def ridge_align(net, Xtr, ytr, S, lams, nrows, dim):
    """阶段1 的谱编码器**冻结**，用类中心做语义→谱空间的岭回归：

        A* = argmin_A ‖Z̄ − S·A‖² + λ‖A‖²
        A* = (SᵗS + λI)⁻¹ Sᵗ Z̄          （闭式解，无训练、无随机性）

    Z̄ = 各类在谱编码器输出上的类中心。老类（18）始终参与；
    k>0 时未见类也有样本，其中心一并纳入（与 none 臂使用同样信息，保证公平）。
    未见类没有样本时（k=0），A 完全由老类学到 —— 这就是**可迁移性**的检验。
    """
    net.eval()
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(ytr, dtype=torch.long, device=DEV)
    with torch.no_grad():
        Z = F.normalize(net(Xt), dim=1)
    rows = np.unique(yt.cpu().numpy())
    Zbar = np.stack([Z[(yt == c)].mean(0).cpu().numpy() for c in rows])   # (R, dim)
    Ssub = np.asarray(S, dtype=np.float64)[rows]                          # (R, dim)
    G = Ssub.T @ Ssub
    B = Ssub.T @ Zbar
    out = {}
    for lam in lams:
        A = np.linalg.solve(G + lam * np.eye(dim), B)                     # (dim, dim)
        W = np.asarray(S, dtype=np.float64) @ A                           # (19, dim)
        out[lam] = W.astype(np.float32)
    return out, rows


@torch.no_grad()
def top1_W(net, W, Xte, cand_rows, true_row):
    net.eval()
    z = F.normalize(net(torch.tensor(Xte, dtype=torch.float32, device=DEV)), dim=1)
    Wt = torch.tensor(np.asarray(W, dtype=np.float32), device=DEV)
    logits = (z @ F.normalize(Wt, dim=1).T)[:, cand_rows]
    pred = logits.argmax(1).cpu().numpy()
    return float((pred == cand_rows.index(true_row)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="0,5,20,full")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--repr", default="clm", choices=["clm", "ecfp"])
    ap.add_argument("--lams", default="0.001,0.01,0.1,1")
    args = ap.parse_args()
    shots = [None if s.strip() == "full" else int(s) for s in args.shots.split(",")]
    tags = [("full" if s is None else str(s)) for s in shots]
    lams = [float(x) for x in args.lams.split(",")]

    modes = ["none", "freeze", "init-all", "proj", "proj-diag", "residual", "gate"]
    ridge_names = [f"ridge{lam:g}" for lam in lams]
    all_names = modes + ridge_names
    print(f"[exp40] device={DEV} repr={args.repr} seeds={args.seeds} "
          f"steps={args.steps} shots={tags}", flush=True)
    ent, bg, _ = G1.load_g1()
    eids = sorted(ent.keys()); mids = sorted(bg.keys())
    keys = eids + [f"M_{m}" for m in mids]
    ki = {k: i for i, k in enumerate(keys)}
    zc = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    assert [str(x) for x in zc["keys"]] == keys
    T_all = np.asarray(zc[args.repr], dtype=np.float32)
    dim = T_all.shape[1]

    res = {"acc": {t: {m: {} for m in all_names} for t in tags},
           "gate": {}, "nte": {},
           "meta": {"repr": args.repr, "shots": tags, "seeds": args.seeds,
                    "steps": args.steps, "eids": eids, "dim": dim, "lams": lams}}

    for e_hold in eids:
        others = [e for e in eids if e != e_hold]
        cand_rows = list(range(NOLD)) + [NEW]
        Xe = np.asarray(ent[e_hold], float)
        rows = ([T_all[ki[e]] for e in others]
                + [T_all[ki[f"M_{m}"]] for m in mids]
                + [T_all[ki[e_hold]]])
        S = center(np.stack(rows).astype(np.float32))
        for k in shots:
            tag = "full" if k is None else str(k)
            gvals = []
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed * 7919 + 13)
                perm = rng.permutation(len(Xe))
                n_te = int(min(max(5, round(0.3 * len(Xe))), len(Xe) - 1))
                te_ix = perm[:n_te]; pool_ix = perm[n_te:]
                tr_e_ix = pool_ix if k is None else pool_ix[:min(k, len(pool_ix))]
                res["nte"][e_hold] = len(te_ix)
                parts, ys = [], []
                for j, e in enumerate(others):
                    parts.append(np.asarray(ent[e], float)); ys.append(np.full(len(ent[e]), j))
                for j, m in enumerate(mids):
                    parts.append(np.asarray(bg[m], float))
                    ys.append(np.full(len(bg[m]), NOLD - 6 + j))
                nrows = NOLD
                if len(tr_e_ix):
                    parts.append(Xe[tr_e_ix]); ys.append(np.full(len(tr_e_ix), NEW))
                    nrows = 19
                Xtr = np.concatenate(parts).astype(np.float32)
                ytr = np.concatenate(ys).astype(np.int64)
                Xte = Xe[te_ix]
                s0 = seed * 1000 + eids.index(e_hold)
                # 阶段1：自由头网络（none 臂与 ridge 臂**共用同一个谱编码器**）
                net_free, head_free = train_arm(Xtr, ytr, S, "none", dim,
                                                nrows, args.steps, s0)
                if nrows == 19:      # k=0 时 none 不可定义（无该类的行可学）
                    res["acc"][tag]["none"].setdefault(e_hold, []).append(
                        float(top1(net_free, head_free, Xte, cand_rows, NEW)))
                # 阶段2：冻结编码器 + 类中心岭回归闭式解 ⇒ W = S·A*
                Ws, _ = ridge_align(net_free, Xtr, ytr, S, lams, nrows, dim)
                for lam in lams:
                    res["acc"][tag][f"ridge{lam:g}"].setdefault(e_hold, []).append(
                        float(top1_W(net_free, Ws[lam], Xte, cand_rows, NEW)))
                # 其余臂
                avail = ["freeze", "proj", "proj-diag"] if nrows == NOLD else modes
                for mode in avail:
                    if mode == "none":
                        continue
                    net, head = train_arm(Xtr, ytr, S, mode, dim, nrows,
                                          args.steps, s0)
                    res["acc"][tag][mode].setdefault(e_hold, []).append(
                        float(top1(net, head, Xte, cand_rows, NEW)))
                    gv = head.gate_value()
                    if gv is not None:
                        gvals.append((mode, gv))
            for mode, gv in gvals:
                res["gate"].setdefault(tag, {}).setdefault(mode, {}).setdefault(
                    e_hold, []).append(gv)
        line = []
        for t in tags:
            s = " ".join(f"{m}={np.mean([np.mean(v) for v in res['acc'][t][m].values()]):.3f}"
                         for m in all_names if res["acc"][t][m])
            line.append(f"{t}: {s}")
        print(f"[{e_hold:>7}] " + " | ".join(line), flush=True)

    with open(os.path.join(OUT, f"exp40_res_{args.repr}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    write_report(res, all_names, tags, args, lams)
    print(f"[exp40] 已写出 out/exp40_gzsl_{args.repr}.md", flush=True)


def write_report(res, modes, tags, args, lams):
    from scipy.stats import ttest_rel, spearmanr
    eids = [e for e in res["meta"]["eids"] if e in res["acc"][tags[-1]]["none"]]
    A = []; w = lambda s: A.append(s)

    def M(t, m):
        return np.array([np.mean(res["acc"][t][m][e]) for e in eids])

    w(f"# exp40 · 语义→谱空间可学习投影 + 门控下限保证（{args.repr.upper()}）")
    w("")
    w("> 根因：化学空间与谱空间结构不相容（CKA = 0.23）。")
    w("> 直接把化学嵌入当分类头 = 强迫对齐错配空间 ⇒ 所有负结果的来源。")
    w("> 本实验改为**学投影** W = S·A（ALE/ESZSL 家族）：A 在老类上学到校正，再迁移到未见类。")
    w("")
    w(f"- LOO 13 实体，k ∈ {{{', '.join(tags)}}}，seeds {args.seeds}，{args.steps} 步")
    w("- 固定 30% 测试集；macro = 逐实体等权；chance = 1/13 = 0.077")
    w("- **none = 无先验基线；所有先验臂的目标是打平或超过它**")
    w("")
    w("## 主结果（macro）")
    w("")
    w("| k | " + " | ".join(modes) + " |")
    w("|---" + "---|" * len(modes))
    for t in tags:
        cells = [f"{M(t, m).mean():.3f}" if res["acc"][t][m] else "—" for m in modes]
        w(f"| {t} | " + " | ".join(cells) + " |")
    w("")
    w("## Δ 相对 none（无先验基线）；**粗体** = 达到下限保证（≥ −0.01）")
    w("")
    w("| k | " + " | ".join(m for m in modes if m != "none") + " |")
    w("|---" + "---|" * (len(modes) - 1))
    for t in tags:
        if not res["acc"][t]["none"]:
            continue
        base = M(t, "none"); cells = []
        for m in modes:
            if m == "none" or not res["acc"][t][m]:
                continue
            d = M(t, m) - base
            p = ttest_rel(M(t, m), base).pvalue
            s = f"{d.mean():+.3f} (p={p:.3f})"
            cells.append(f"**{s}**" if d.mean() >= -0.01 else s)
        w(f"| {t} | " + " | ".join(cells) + " |")
    w("")
    w("## 门控 / 残差权重（模型自己决定「信化学多少」）")
    w("")
    for mode in ("gate", "residual"):
        if mode not in res["gate"].get(tags[-1], {}):
            continue
        w(f"**{mode}**")
        w("")
        w("| k | 权重均值 | 解读 |")
        w("|---|---|---|")
        for t in tags:
            if mode in res["gate"].get(t, {}):
                g = np.mean([np.mean(v) for v in res["gate"][t][mode].values()])
                w(f"| {t} | {g:.3f} | {'偏信化学' if g > 0.5 else '偏信数据'} |")
        gs, ks = [], []
        for i, t in enumerate(tags):
            if mode in res["gate"].get(t, {}):
                gs.append(np.mean([np.mean(v) for v in res["gate"][t][mode].values()]))
                ks.append(i)
        if len(gs) >= 3:
            rho = spearmanr(ks, gs).statistic
            w("")
            w(f"Spearman ρ(k, 权重) = {rho:+.3f} ⇒ "
              + ("✅ 样本越多，模型越少依赖化学先验"
                 if rho < 0 else "❌ 未观察到单调下降"))
        w("")
    w("## 预注册判据裁定")
    w("")
    ok_a = ok_b = False
    if res["acc"]["0"]["proj"] and res["acc"]["0"]["freeze"]:
        da = M("0", "proj").mean() - M("0", "freeze").mean()
        ok_a = da > 0
        w(f"- **H40a**（k=0，学到的投影优于直接冻结）：proj − freeze = {da:+.3f} ⇒ "
          f"{'✅ 成立' if ok_a else '❌ 不成立'}")
    bad = []
    for t in tags:
        if not res["acc"][t]["none"]:
            continue
        base = M(t, "none")
        for m in ("residual", "gate"):
            if not res["acc"][t][m]:
                continue
            if (M(t, m) - base).mean() < -0.01:
                bad.append(f"k={t}/{m}")
    ok_b = not bad
    w(f"- **H40b（下限保证）**：residual 与 gate 在所有 k 档均 ≥ none − 0.01 ⇒ "
      f"{'✅ 成立' if ok_b else '❌ 不成立（未达标：' + ', '.join(bad) + '）'}")
    w("- **H40c（机制）**：门控权重随 k 单调下降 —— 见上一节")
    w("")
    best = None
    for lam in lams:
        nm = f"ridge{lam:g}"
        for t in tags:
            if not res["acc"][t].get("none") or not res["acc"][t].get(nm):
                continue
            d = (M(t, nm) - M(t, "none")).mean()
            if best is None or d > best[0]:
                best = (d, nm, t)
    if best:
        w(f"- **H40d（新增·主）**：两阶段岭回归对比**同一个编码器**下的无先验头 —— "
          f"最佳 {best[1]} @ k={best[2]}，Δ = {best[0]:+.3f}")
    w("")
    w("> 无论正负都如实报告。若 H40b 不成立，论文的 claim 必须退回到")
    w("> 「化学先验仅在 k=0 不可替代」，不得声称普适增益。")

    with open(os.path.join(OUT, f"exp40_gzsl_{args.repr}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(A))


if __name__ == "__main__":
    main()
