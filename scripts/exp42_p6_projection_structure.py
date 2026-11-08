"""exp42 —— 投影 A 的**结构约束**（对 exp40 发现的定向改进，非堆臂）。

=============================================================================
诊断依据（全部来自前序实验，本实验不重测）
=============================================================================
1. exp34/38：硬对齐（化学嵌入直接当分类头）**有害**
   —— k=full 冻结 Center − None = −0.251 (p=0.014)
2. exp41/41b：物理先验（官能团伪谱）在**原始谱域**系统性反相关
   （正确配对余弦 −0.114 < 错误配对 −0.098）⇒ 先验只能在**嵌入域**用
3. exp38：对齐层面的校正（去均值/白化/ABTT）收益已到顶
   （Center − Raw = +0.050, p=0.30）
4. **exp40（本轮的直接触发）**：`proj`（W = S·A，A 可学）在 k=0 稳定领先
   freeze 约 1.4×（A01 0.763 vs 0.550；A02 0.381 vs 0.275；A05 0.156 vs 0.111）
   ⇒ **学一个投影是唯一成立的对齐方式**

=============================================================================
本实验要改什么（有理论依据的定向改进）
=============================================================================
exp40 的 A 是**全秩**：A ∈ R^{384×384}，约 **14.7 万参数**。
而可用来学 A 的老类只有 **18 个**。
⇒ 参数/样本比 ≈ 8000 : 1，**必然过拟合**。
这是可检验的诊断，不是猜测：若成立，则约束 A 的结构应当提升泛化。

三个有理论依据的约束（由强到弱）：
  ★ `orth`    **正交 Procrustes 闭式解**  R = U Vᵗ，其中 SVD(SᵗZ̄) = UΣVᵗ
              —— **零可学参数**，且**保内积结构**：
                 (S R)(S R)ᵗ = S Sᵗ，即化学空间的相似结构被完整保留。
                 若两空间只差一个旋转，这是**理论最优解**。
                 直接针对 CKA=0.23 的「结构不相容」：不是拟合它，是**旋转对齐**它。
  ★ `lowrank` 低秩 A = U Vᵗ，秩 r ∈ {2,4,8}
              —— 参数量从 147k 降到 384·2r（r=4 时仅 3k，**降 50 倍**）
  ★ `orth-ft` 正交闭式解作初始化，再用极少步微调（兼得结构与数据适配）

=============================================================================
协议设计（关键：隔离变量 + 降本一个数量级）
=============================================================================
**两阶段，所有臂共用同一个阶段-1 谱编码器**：
  阶段1：训 net + 自由分类头（= exp38 的 `none` 臂），得到谱编码器 net
  阶段2：**冻结 net**，计算各训练类的谱嵌入类中心 Z̄；各臂只用
          {S, Z̄} 构造分类头 W，不再改动谱编码器
⇒ 唯一自变量就是「分类头怎么由化学先验构造」——正是要隔离的效应。
⇒ 训练量从 O(臂数) 降到 O(1)（阶段2 多为闭式解或极少步），
   13 实体 × 3 档 × 3 seeds 只需 ~117 次阶段1 训练。

固定 30% held-out 测试集（沿用 exp35 修正版），k 从剩余 70% 抽取。
分类头恒 19 列（13 实体 + 6 基质背景）。

=============================================================================
预注册判据（写死，跑完不再改）
=============================================================================
- **H42a（主）**：k=0 时 `orth` > `center`（冻结去均值码本）
  依据：exp41 证明先验只能旋转对齐；exp38 测得 center 是冻结族最优（0.496）
- **H42b（主）**：`orth` ≥ `proj`（全秩可学 A）
  若成立 ⇒ 过拟合诊断成立，结构约束有效（**本实验的核心主张**）
- **H42c（次）**：低秩 r=4/8 ≥ 全秩 A（同向佐证过拟合诊断）
- **H42d（次）**：k=full 时 `gate` ≥ `none`（下限保证：先验无用时 g→0 自动退回）
- 报告：所有 λ / r 全部列出，不挑最优；负结果照报
"""
from __future__ import annotations
import os, sys, json, io, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1
import exp34_cache_reprs as C

OUT = r"D:/2026sci论文实验设计/论文6/out"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMP = 0.07


# --------------------------------------------------------------------------- #
def center(T):
    return T - T.mean(0, keepdims=True)


def build_net(d_in, d_out, seed):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(d_in, 512), nn.ReLU(),
        nn.Linear(512, d_out),
    ).to(DEV)


@torch.no_grad()
def embed(net, X):
    net.eval()
    z = net(torch.tensor(X, dtype=torch.float32, device=DEV))
    return F.normalize(z, dim=1).cpu().numpy()


def stage1(Xtr, ytr, dim, nrows, steps, seed, S=None, mode="free"):
    """训练谱编码器，返回 (net, W)。

    mode 决定**分类头的形态**，它决定 z 空间学成什么样：
      - "free"：头随机可学 ⇒ z 是纯判别空间，与化学空间无关
                （本实验用它作对照，预期一切化学对齐在此失效）
      - "chem"：头冻结为去均值化学向量 ⇒ net 被牵引到化学语义空间
                （DEVISE / ESZSL 的标准做法；阶段2 的对齐在此才有意义）
      - "proj"：头 = S[:nrows]·A，A 与 net 联合可学 ⇒ z 被牵引到 S·A 空间
    """
    torch.manual_seed(seed)
    net = build_net(Xtr.shape[1], dim, seed)
    params = list(net.parameters())

    if mode == "chem":
        assert S is not None, "mode=chem 需要传入化学向量 S"
        Wc = S[:nrows] - S[:nrows].mean(0, keepdims=True)
        W = torch.tensor(Wc, dtype=torch.float32, device=DEV)   # 冻结
    elif mode == "proj":
        assert S is not None, "mode=proj 需要传入化学向量 S"
        A = nn.Parameter(torch.eye(dim, device=DEV))
        params = params + [A]
    else:                                                        # free
        W = nn.Parameter(torch.randn(nrows, dim, device=DEV) * 0.01)
        params = params + [W]

    opt = torch.optim.Adam(params, lr=3e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(ytr, dtype=torch.long, device=DEV)
    n = len(Xtr)
    net.train()
    for s in range(steps):
        idx = torch.randint(0, n, (min(64, n),), device=DEV)
        z = F.normalize(net(Xt[idx]), dim=1)
        if mode == "proj":
            Wc = S[:nrows] - S[:nrows].mean(0, keepdims=True)
            Wt = torch.tensor(Wc, dtype=torch.float32, device=DEV) @ A
        else:
            Wt = W
        loss = F.cross_entropy(z @ F.normalize(Wt, dim=1).T / TEMP, yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    if mode == "proj":
        Wc = S[:nrows] - S[:nrows].mean(0, keepdims=True)
        W = (torch.tensor(Wc, dtype=torch.float32, device=DEV) @ A).detach()
    return net, W.detach().cpu().numpy()


# --------------------------------------------------------------------------- #
#  阶段2：由 {S, Z̄} 构造分类头 W 的各种方式
# --------------------------------------------------------------------------- #
# 说明：以下函数一律返回投影矩阵 A，再由调用方统一做 W = (S − μ_S)·A，
# 保证 19 行（含未见类）与老类使用**同一个**中心化，避免中心不一致造成的错位。
def cka(A, B):
    """Linear CKA（Kornblith et al., ICML 2019）。用于回答核心诊断：
    谱编码器学到的 z 空间，是否真的含有化学空间的结构。"""
    A = np.asarray(A, float) - np.asarray(A, float).mean(0, keepdims=True)
    B = np.asarray(B, float) - np.asarray(B, float).mean(0, keepdims=True)
    num = float(np.linalg.norm(A.T @ B) ** 2)
    den = float(np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B))
    return num / den if den > 0 else 0.0


def head_orth_R(S_old, Z_old):
    """正交 Procrustes 闭式解：R = U Vᵗ, SVD(S_oldᵗ Z_old) = UΣVᵗ。零可学参数，
    且保内积结构：(SR)(SR)ᵗ = SSᵗ ⇒ 化学空间的相似结构被完整保留。"""
    Sc = S_old - S_old.mean(0, keepdims=True)
    Zc = Z_old - Z_old.mean(0, keepdims=True)
    M = Sc.T @ Zc
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def head_ridge_R(S_old, Z_old, lam):
    """岭回归闭式解（DEVISE / ESZSL 家族）。用等价恒等式避免 384×384 求逆病态。"""
    Sc = S_old - S_old.mean(0, keepdims=True)
    Zc = Z_old - Z_old.mean(0, keepdims=True)
    return Sc.T @ np.linalg.solve(Sc @ Sc.T + lam * np.eye(len(Sc)), Zc)


def head_lowrank_R(S_old, Z_old, r):
    """岭回归解再做秩-r 截断（截断 SVD）⇒ 参数/容量大幅下降，抑制过拟合。"""
    A = head_ridge_R(S_old, Z_old, 1e-2)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    r = min(r, len(s))
    return (U[:, :r] * s[:r]) @ Vt[:r]


def train_A(net, Xtr, ytr, S_all, W_init, steps, seed, dim):
    """在冻结的谱编码器上，只训练分类头（从 W_init 出发）。"""
    torch.manual_seed(seed)
    W = nn.Parameter(torch.tensor(W_init, dtype=torch.float32, device=DEV))
    opt = torch.optim.Adam([W], lr=1e-3)
    with torch.no_grad():
        Z = F.normalize(net(torch.tensor(Xtr, dtype=torch.float32, device=DEV)), dim=1)
    yt = torch.tensor(ytr, dtype=torch.long, device=DEV)
    n = len(Xtr)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(64, n),), device=DEV)
        loss = F.cross_entropy(Z[idx] @ F.normalize(W, dim=1).T / TEMP, yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach().cpu().numpy()


def head_gate(net, Xtr, ytr, S_all, S_old, Z_old, W_free, mu_S, steps, seed):
    """门控融合 W = (1−g)·W_free + g·(S·A_orth)，θ 初始 −2 ⇒ g→0 时退化为 none。"""
    W_prior = (S_all - mu_S) @ head_orth_R(S_old, Z_old)
    Wp = torch.tensor(W_prior, dtype=torch.float32, device=DEV)
    Wf = _pad19(W_free)
    theta = nn.Parameter(torch.tensor(-2.0, device=DEV))
    opt = torch.optim.Adam([theta], lr=1e-2)
    with torch.no_grad():
        Z = F.normalize(net(torch.tensor(Xtr, dtype=torch.float32, device=DEV)), dim=1)
    yt = torch.tensor(ytr, dtype=torch.long, device=DEV)
    n = len(Xtr)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(64, n),), device=DEV)
        g = torch.sigmoid(theta)
        W = (1 - g) * Wf + g * Wp
        loss = F.cross_entropy(Z[idx] @ F.normalize(W, dim=1).T / TEMP, yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    g = float(torch.sigmoid(theta).item())
    return ((1 - g) * Wf + g * Wp).cpu().numpy(), g


# --------------------------------------------------------------------------- #
@torch.no_grad()
def _pad19(W):
    """k=0 时自由头只有 18 行（未见类无训练样本），补零行到 19 行后再评估，
    与 exp38 的处理保持一致。"""
    Wt = torch.tensor(np.asarray(W), dtype=torch.float32, device=DEV)
    if Wt.shape[0] < 19:
        Wt = torch.cat([Wt, torch.zeros(19 - Wt.shape[0], Wt.shape[1], device=DEV)], 0)
    return Wt


@torch.no_grad()
def pred_hist(net, W, Xte, cand_rows):
    """预测分布诊断：若某一类被预测的比例过高，说明 z 空间塌陷（恒定预测），
    此时的高 top-1 是假象，不能作为证据。返回最大类占比与熵。"""
    z = F.normalize(net(torch.tensor(Xte, dtype=torch.float32, device=DEV)), dim=1)
    logits = (z @ F.normalize(_pad19(W), dim=1).T)[:, cand_rows]
    pred = logits.argmax(1).cpu().numpy()
    p = np.bincount(pred, minlength=len(cand_rows)) / len(pred)
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(p)))
    return [float(p.max()), ent]


@torch.no_grad()
def top1(net, W, Xte, cand_rows, true_row):
    z = F.normalize(net(torch.tensor(Xte, dtype=torch.float32, device=DEV)), dim=1)
    logits = (z @ F.normalize(_pad19(W), dim=1).T)[:, cand_rows]
    pred = logits.argmax(1).cpu().numpy()
    return float((pred == cand_rows.index(true_row)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="0,20,full")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--steps2", type=int, default=300)
    ap.add_argument("--ranks", default="2,4,8")
    ap.add_argument("--lams", default="0.01,0.1")
    ap.add_argument("--s1", default="chem,free",
                    help="阶段1 编码器训练时分类头的形态：chem=冻结化学头（net 被牵引到"
                         "化学空间，DEVISE 标准做法）；free=自由头（纯判别空间，作对照）；"
                         "proj=头=S·A 联合可学。可多选，逗号分隔。")
    ap.add_argument("--repr", default="clm", choices=["clm", "ecfp"])
    args = ap.parse_args()
    s1_modes = [m.strip() for m in args.s1.split(",") if m.strip()]
    shots = [None if s.strip() == "full" else int(s) for s in args.shots.split(",")]
    tags = [("full" if s is None else str(s)) for s in shots]
    ranks = [int(x) for x in args.ranks.split(",")]
    lams = [float(x) for x in args.lams.split(",")]

    print(f"[exp42] device={DEV} repr={args.repr} shots={tags} seeds={args.seeds} "
          f"steps={args.steps}/{args.steps2} ranks={ranks}", flush=True)

    ent, bg, _ = G1.load_g1()
    eids = sorted(ent.keys()); mids = sorted(bg.keys())
    keys = eids + [f"M_{m}" for m in mids]
    z = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    T_all = np.asarray(z[args.repr if args.repr != "ecfp" else "ecfp384"], float)
    dim = T_all.shape[1]
    ki = {k: i for i, k in enumerate(keys)}
    print(f"[exp42] dim={dim} n_keys={len(keys)}", flush=True)

    base_arms = (["none", "center", "orth", "orth-ft", "proj", "ridge-cv", "lr-cv"]
                 + [f"orth+lr{r}" for r in ranks]
                 + [f"ridge{lam:g}" for lam in lams]
                 + ["gate"])
    arms = [f"{s1}:{a}" for s1 in s1_modes for a in base_arms]
    res = {"acc": {t: {a: {} for a in arms} for t in tags},
           "gate": {}, "cka": {s1: {t: {} for t in tags} for s1 in s1_modes},
           "meta": {"repr": args.repr, "shots": tags, "seeds": args.seeds,
                    "steps": args.steps, "eids": eids, "dim": dim,
                    "ranks": ranks, "lams": lams, "s1": s1_modes,
                    "base_arms": base_arms}}

    for e_hold in eids:
        # ---- 协议严格对齐 exp38：老类全量，留出类固定放在索引 18 ----
        others = [e for e in eids if e != e_hold]
        # 化学向量按 [others(12), mids(6), e_hold] 重排，使未见类恒在最后一行
        S = np.asarray(T_all)[[ki[e] for e in others]
                              + [ki[f"M_{m}"] for m in mids]
                              + [ki[e_hold]]]
        cand_rows = list(range(len(others) + len(mids))) + [18]
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
                Xte_e = Xe[te_ix]

                parts, ys = [], []
                for j, e in enumerate(others):
                    parts.append(np.asarray(ent[e], float))
                    ys.append(np.full(len(ent[e]), j))
                for j, m in enumerate(mids):
                    parts.append(np.asarray(bg[m], float))
                    ys.append(np.full(len(bg[m]), len(others) + j))
                nrows = len(others) + len(mids)          # 18
                if len(tr_e_ix):
                    parts.append(Xe[tr_e_ix])
                    ys.append(np.full(len(tr_e_ix), 18))
                    nrows = 19
                Xtr = np.concatenate(parts).astype(np.float32)
                ytr = np.concatenate(ys).astype(np.int64)

                # ---------- 阶段 1：训谱编码器（所有臂共用）----------
                # 老类内部再划 80/20 作 fit/val：val **只用于选 λ 与秩 r**，
                # 不接触未见类 ⇒ 超参选择不泄漏测试信息，避免「挑最好的 λ」。
                rows_old = sorted({int(y) for y in ytr})
                rng_v = np.random.default_rng(seed * 31 + 7)
                val_mask = np.zeros(len(Xtr), bool)
                for j in rows_old:
                    if j == 18 and len(tr_e_ix) == 0:
                        continue
                    ix = rng_v.permutation(np.where(ytr == j)[0])
                    val_mask[ix[:max(3, int(0.2 * len(ix)))]] = True
                Xfit, yfit = Xtr[~val_mask], ytr[~val_mask]
                Xval, yval = Xtr[val_mask], ytr[val_mask]
                val_rows = sorted({int(y) for y in yval})

                for s1 in s1_modes:
                  net, W_free = stage1(Xfit, yfit, dim, nrows, args.steps,
                                       seed * 1000 + eids.index(e_hold),
                                       S=S, mode=s1)
                  Ztr = embed(net, Xtr)
                  Zfit = embed(net, Xfit)

                  # 老类（有训练样本的类）的谱中心与化学向量；
                  # 未见类（索引 18）在 k=0 时无训练样本，只能由 S 外推
                  Z_old = np.stack([Zfit[yfit == j].mean(0) for j in rows_old])
                  S_old = S[rows_old]
                  mu_S = S_old.mean(0, keepdims=True)   # 中心化须统一用老类均值
                  # ★ 核心诊断：z 空间是否真的含有化学结构。
                  #   若 CKA≈0，则阶段2 的任何对齐（ridge/orth/…）在数学上无意义。
                  res["cka"][s1][tag].setdefault(e_hold, []).append(cka(Z_old, S_old))
                  accs = {}

                  def W_of(A):
                    return (S - mu_S) @ A               # 全 19 行共用同一中心化

                  def val_acc(W):
                    """在**老类**验证集上的 top-1（用于自动选 λ / r）。"""
                    z = F.normalize(net(torch.tensor(Xval, dtype=torch.float32,
                                                     device=DEV)), dim=1)
                    Wt = F.normalize(_pad19(W), dim=1)
                    pred = (z @ Wt.T)[:, val_rows].argmax(1).cpu().numpy()
                    want = np.searchsorted(val_rows, yval)
                    return float((pred == want).mean())

                  # ---- 超参由老类验证集自动选择（不挑、不泄漏未见类）----
                  cand_lam = [lam for lam in lams]
                  best_lam, best_v = cand_lam[0], -1.0
                  for lam in cand_lam:
                    v = val_acc(W_of(head_ridge_R(S_old, Z_old, lam)))
                    if v > best_v:
                        best_v, best_lam = v, lam
                  cand_r = list(ranks)
                  best_r, best_vr = cand_r[0], -1.0
                  for r in cand_r:
                    v = val_acc(W_of(head_lowrank_R(S_old, Z_old, r)))
                    if v > best_vr:
                        best_vr, best_r = v, r
                  res.setdefault("cv", {}).setdefault(tag, {}).setdefault(
                      e_hold, []).append([best_lam, best_r])

                  # 冻结基线：去均值化学码本（exp38 的 center）
                  accs["center"] = top1(net, S - mu_S, Xte_e, cand_rows, NEW)

                  R = head_orth_R(S_old, Z_old)
                  accs["orth"] = top1(net, W_of(R), Xte_e, cand_rows, NEW)
                  accs["orth-ft"] = top1(net, train_A(net, Xtr, ytr, S, W_of(R),
                                                      args.steps2, seed + 7, dim),
                                        Xte_e, cand_rows, NEW)
                  accs["proj"] = top1(net, train_A(net, Xtr, ytr, S, W_of(np.eye(dim)),
                                                   args.steps2, seed + 11, dim),
                                      Xte_e, cand_rows, NEW)
                  for r in ranks:
                    accs[f"orth+lr{r}"] = top1(net, W_of(head_lowrank_R(S_old, Z_old, r)),
                                               Xte_e, cand_rows, NEW)
                  for lam in lams:
                    accs[f"ridge{lam:g}"] = top1(net, W_of(head_ridge_R(S_old, Z_old, lam)),
                                                 Xte_e, cand_rows, NEW)
                  # ★ 自动选择版（论文主张的臂）：λ 与 r 由老类验证集选，不人工挑
                  accs["ridge-cv"] = top1(net, W_of(head_ridge_R(S_old, Z_old, best_lam)),
                                          Xte_e, cand_rows, NEW)
                  accs["lr-cv"] = top1(net, W_of(head_lowrank_R(S_old, Z_old, best_r)),
                                       Xte_e, cand_rows, NEW)
                  Wg, gv = head_gate(net, Xtr, ytr, S, S_old, Z_old, W_free, mu_S,
                                     args.steps2, seed + 13)
                  accs["gate"] = top1(net, Wg, Xte_e, cand_rows, NEW)
                  if len(tr_e_ix):                      # k=0 时 none 不可定义
                    accs["none"] = top1(net, W_free, Xte_e, cand_rows, NEW)
                    # 预测分布诊断：识别「恒定预测」假象
                    res.setdefault("predhist", {}).setdefault(tag, {}).setdefault(
                        "none", []).append(pred_hist(net, W_free, Xte_e, cand_rows))

                  for a, v in accs.items():
                    res["acc"][tag][f"{s1}:{a}"].setdefault(e_hold, []).append(float(v))
                  res.setdefault("gate", {}).setdefault(f"{s1}:gate", {}).setdefault(
                      tag, {}).setdefault(e_hold, []).append(gv)


        # ⚠ 打印**当前实体自身的 seeds 均值**，不是累计 macro。
        # 累计均值会让逐实体的 0 被前面实体的高分稀释，制造「多实体一致」的假象
        # （已在 exp35 / exp40 / exp42 上连续误读三次，务必保持逐实体打印）。
        line = []
        for t in tags:
            s = " ".join(f"{a.replace('chem:', '').replace('free:', 'f_')}="
                         f"{np.mean(res['acc'][t][a][e_hold]):.3f}"
                         for a in arms if res["acc"][t][a].get(e_hold))
            line.append(f"{t}: {s}")
        cks = " ".join(f"cka_{s1}={np.mean(res['cka'][s1].get(tags[0], {}).get(e_hold, [0])):.3f}"
                       for s1 in s1_modes)
        print(f"[{e_hold:>7}] " + " | ".join(line) + f"  || {cks}", flush=True)

    with open(os.path.join(OUT, "exp42_res.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    write_report(res, arms, tags, args)


def write_report(res, arms, tags, args):
    from scipy.stats import ttest_rel
    eids = res["meta"]["eids"]
    A = []
    w = A.append
    w("# exp42 —— 投影 A 的结构约束 + 阶段1 协议诊断\n")
    s1s = res["meta"]["s1"]
    PRI = s1s[0]                      # 主张的协议（通常为 chem）

    def P(a, s1=None):
        return f"{s1 or PRI}:{a}"

    # ---- 核心诊断：z 空间是否含化学结构 ----
    w("## 诊断 A：谱编码器学到的 z 空间是否含有化学结构（linear CKA）\n")
    w("CKA(Z̄_old, S_old) —— 阶段2 的任何对齐只有在它显著 > 0 时才有意义。\n")
    w("| 阶段1 头形态 | " + " | ".join(f"k={t}" for t in tags) + " |")
    w("|---|" + "---|" * len(tags))
    for s1 in s1s:
        row = []
        for t in tags:
            v = [np.mean(x) for x in res["cka"][s1].get(t, {}).values()]
            row.append(f"{np.mean(v):.3f}" if v else "—")
        w(f"| `{s1}` | " + " | ".join(row) + " |")
    w("")

    w("## 主结果（macro，13 实体等权）\n")
    for s1 in s1s:
        sub = [a for a in arms if a.startswith(s1 + ":")]
        w(f"### 阶段1 = `{s1}`\n")
        w("| 臂 | " + " | ".join(tags) + " |")
        w("|---|" + "---|" * len(tags))
        for a in sub:
            row = [f"{np.mean([np.mean(res['acc'][t][a][e]) for e in res['acc'][t][a]]):.3f}"
                   if res["acc"][t][a] else "—" for t in tags]
            w(f"| {a.split(':', 1)[1]} | " + " | ".join(row) + " |")
        w("")
    usable = [a for a in arms if a.startswith(PRI + ":")
              and any(res["acc"][t][a] for t in tags)]

    def M(t, a, grp=None):
        g = grp or [e for e in eids if e in res["acc"][t][a]]
        return np.array([np.mean(res["acc"][t][a][e]) for e in g])

    w("## 配对检验（以实体为单元）\n")
    for t in tags:
        base = P("center") if t == "0" else P("none")
        if not res["acc"][t][base]:
            continue
        w(f"### k={t}　基线 = `{base}`\n")
        w("| 臂 | Δ | p |")
        w("|---|---|---|")
        b = M(t, base)
        for a in usable:
            if a == base or not res["acc"][t][a]:
                continue
            m = M(t, a)
            if len(m) != len(b):
                continue
            d = m - b
            p = ttest_rel(m, b).pvalue
            w(f"| {a.split(':', 1)[1]} | {d.mean():+.3f} | {p:.4f} |")
        w("")

    w("## 预注册判据裁定\n")
    def get(t, a):
        return np.mean([np.mean(res["acc"][t][a][e]) for e in res["acc"][t][a]]) \
            if res["acc"][t][a] else None
    o0, c0 = get("0", P("orth")), get("0", P("center"))
    if o0 is not None and c0 is not None:
        w(f"- **H42a（主）** k=0：`orth`({o0:.3f}) > `center`({c0:.3f}) ⇒ "
          f"{'✅ 成立' if o0 > c0 else '❌ 不成立'}")
    for t in tags:
        if t == "0":
            continue
        pr = get(t, P("proj"))
        o = get(t, P("orth"))
        if pr is not None and o is not None:
            w(f"- **H42b（主）** k={t}：`orth`({o:.3f}) ≥ `proj`({pr:.3f}) ⇒ "
              f"{'✅ 成立' if o >= pr else '❌ 不成立'}")
    for r in res["meta"]["ranks"]:
        for t in tags:
            lr = get(t, P(f"orth+lr{r}")); pr = get(t, P("proj"))
            if lr is not None and pr is not None:
                w(f"- **H42c** k={t} r={r}：低秩({lr:.3f}) vs 全秩 proj({pr:.3f}) ⇒ "
                  f"{'✅ 优于' if lr > pr else '❌ 不优于'}")
    # H42d：阶段1 协议诊断 —— chem 须显著优于 free
    for t in tags:
        vc = np.mean([np.mean(x) for x in res["cka"]["chem"].get(t, {}).values()]) \
            if res["cka"].get("chem", {}).get(t) else None
        vf = np.mean([np.mean(x) for x in res["cka"]["free"].get(t, {}).values()]) \
            if res["cka"].get("free", {}).get(t) else None
        if vc is not None and vf is not None:
            w(f"- **H42d（协议）** k={t}：CKA chem({vc:.3f}) > free({vf:.3f}) ⇒ "
              f"{'✅ 成立' if vc > vf else '❌ 不成立'}"
              "　（若不成立，则阶段2 的一切对齐都无意义）")
    w("")
    w("## 门控值 g（下限保证的机制证据）\n")
    for key in res["gate"]:
        for t in tags:
            if t in res["gate"][key]:
                gs = [np.mean(v) for v in res["gate"][key][t].values()]
                w(f"- {key} k={t}: g 均值 = {np.mean(gs):.3f}"
                  "（g→0 表示模型主动放弃先验）")
    w("")
    path = os.path.join(OUT, "exp42_p6_projection_structure.md")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(A))
    print(f"[exp42] 报告 → {path}", flush=True)


if __name__ == "__main__":
    main()
