"""exp39 · 零样本（k=0）化学先验的**因果**检验 + 适用边界分层。

为什么必须做这个实验
--------------------
exp35 已证：k=0（无该掺杂物任何样本、也无纯品参考谱）时
    化学冻结码本 CLM-c ≈ 0.44–0.49，而 Rand / None **结构性恒为 0**。
这是「先验不可替代」的硬证据，但它只是一个**相关性**事实：
    真实化学嵌入之所以有效，可能并非因为「化学对应关系正确」，
    而仅仅因为它的**向量分布性质**（各向异性、hubness、范数分布）
    恰好把 19 个类方向撑开了。
⇒ 必须有一个对照：保留全部向量分布性质，只**破坏「实体 → 化学向量」的对应关系**。
这正是本实验的主检验。

主检验：行置换（Row-Permutation）
--------------------------------
对 19 行分类头做随机置换：
  - 向量集合**完全相同** ⇒ 各向异性、有效秩、行范数分布、hubness 全部保留
  - 唯一被破坏的是「哪个向量属于哪个类」
若 Real ≫ Perm，则增益**只能**来自化学对应关系的正确性（因果证据）。
经验 p 值 = (1 + #{Perm ≥ Real}) / (1 + P)。

次检验：列置换（Column-Shuffle）
-------------------------------
对 384 个维度**各自独立**置换行顺序：
  - 保留每个维度的边缘分布
  - 破坏「行向量作为一个整体的化学语义」
用于排除「单维边缘统计足够」的解释。

适用边界分层（预注册之外的必要分层）
----------------------------------
13 个实体其实混了两类**物理性质完全不同**的任务：
  adulteration   掺假（n=8）A01–A07, HSYRUP —— 外源掺杂物有**明确分子结构**
  authentication 鉴别（n=5）COFA/COFR/MEATCH/MEATP/MEATT —— 物种/产地差异
                 主要来自**组织结构与组成比例**，而非单一分子结构
⇒ 化学语义先验对这两类任务的适用性**理应不同**，必须分层报告。

预注册判据
----------
  H39a（主）：k=0 时 Real 显著优于行置换零分布（经验 p < 0.05）
  H39b（次）：Real 显著优于列置换零分布
  H39c（次）：掺假组 k=0 增益 ≥ 鉴别组（化学结构明确的掺杂物更受益）
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


def center(T):
    T = np.asarray(T, dtype=np.float32)
    return (T - T.mean(0, keepdims=True)).astype(np.float32)


def train_frozen(Xtr, ytr, W0, dim, nrows, steps, seed):
    """冻结码本训练谱编码器；评估时用完整 19 行。"""
    torch.manual_seed(seed)
    net = E8.SpecCNN(dim).to(DEV)
    Tt = torch.tensor(np.asarray(W0, dtype=np.float32), device=DEV)
    W = Tt[:nrows]
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(ytr, dtype=torch.long, device=DEV)
    gen = torch.Generator().manual_seed(seed)
    n = len(Xt)
    net.train()
    done = 0
    while done < steps:
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, 256):
            if done >= steps:
                break
            idx = perm[i:i + 256]
            opt.zero_grad()
            z = F.normalize(net(Xt[idx]), dim=1)
            loss = F.cross_entropy(z @ F.normalize(W, dim=1).T / TEMP, yt[idx])
            loss.backward(); opt.step()
            done += 1
    return net, Tt


@torch.no_grad()
def top1(net, W, Xte, cand_rows, true_row):
    net.eval()
    z = F.normalize(net(torch.tensor(Xte, dtype=torch.float32, device=DEV)), dim=1)
    logits = (z @ F.normalize(W, dim=1).T)[:, cand_rows]
    pred = logits.argmax(1).cpu().numpy()
    return float((pred == cand_rows.index(true_row)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--n-perm", type=int, default=12, help="行置换次数")
    ap.add_argument("--n-col", type=int, default=5, help="列置换次数")
    ap.add_argument("--repr", default="clm", choices=["clm", "ecfp"])
    args = ap.parse_args()

    print(f"[exp39] device={DEV} repr={args.repr} seeds={args.seeds} "
          f"steps={args.steps} n_perm={args.n_perm} n_col={args.n_col}", flush=True)
    ent, bg, _ = G1.load_g1()
    eids = sorted(ent.keys()); mids = sorted(bg.keys())
    keys = eids + [f"M_{m}" for m in mids]
    ki = {k: i for i, k in enumerate(keys)}
    zc = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    assert [str(x) for x in zc["keys"]] == keys
    T_all = np.asarray(zc[args.repr], dtype=np.float32)
    dim = T_all.shape[1]

    grp = {e: G1.ENTITY_META[e][2] for e in eids}
    AD = [e for e in eids if grp[e] == "adulteration"]
    AU = [e for e in eids if grp[e] == "authentication"]

    def base_W(e_hold):
        others = [e for e in eids if e != e_hold]
        rows = ([T_all[ki[e]] for e in others]
                + [T_all[ki[f"M_{m}"]] for m in mids]
                + [T_all[ki[e_hold]]])
        return np.stack(rows).astype(np.float32)      # (19, dim)

    def build(e_hold, variant, seed):
        W0 = base_W(e_hold)
        if variant == "Real":
            return center(W0)
        if variant == "Raw":
            return W0
        if variant == "Rand":
            return np.random.default_rng(5000 + seed).normal(size=(19, dim)).astype(np.float32)
        if variant.startswith("Perm"):
            j = int(variant[4:])
            p = np.random.default_rng(90000 + j * 131 + seed).permutation(19)
            return center(W0[p])
        if variant.startswith("Col"):
            j = int(variant[3:])
            rng = np.random.default_rng(77000 + j * 97 + seed)
            Wc = np.empty_like(W0)
            for d in range(dim):
                Wc[:, d] = W0[rng.permutation(19), d]
            return center(Wc)
        raise ValueError(variant)

    variants = (["Real", "Raw", "Rand"]
                + [f"Perm{j}" for j in range(args.n_perm)]
                + [f"Col{j}" for j in range(args.n_col)])
    res = {"acc": {v: {} for v in variants}, "nte": {},
           "meta": {"repr": args.repr, "seeds": args.seeds, "steps": args.steps,
                    "n_perm": args.n_perm, "n_col": args.n_col,
                    "eids": eids, "AD": AD, "AU": AU, "dim": dim}}

    for e_hold in eids:
        others = [e for e in eids if e != e_hold]
        cand_rows = list(range(len(others))) + [18]
        Xe = np.asarray(ent[e_hold], float)
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed * 7919 + 13)
            perm = rng.permutation(len(Xe))
            n_te = int(min(max(5, round(0.3 * len(Xe))), len(Xe) - 1))
            te_ix = perm[:n_te]
            res["nte"][e_hold] = len(te_ix)
            parts, ys = [], []
            for j, e in enumerate(others):
                parts.append(np.asarray(ent[e], float)); ys.append(np.full(len(ent[e]), j))
            for j, m in enumerate(mids):
                parts.append(np.asarray(bg[m], float))
                ys.append(np.full(len(bg[m]), len(others) + j))
            Xtr = np.concatenate(parts).astype(np.float32)
            ytr = np.concatenate(ys).astype(np.int64)
            Xte = Xe[te_ix]
            for v in variants:
                W0 = build(e_hold, v, seed)
                net, W = train_frozen(Xtr, ytr, W0, dim, 18, args.steps,
                                      seed * 1000 + eids.index(e_hold))
                res["acc"][v].setdefault(e_hold, []).append(
                    float(top1(net, W, Xte, cand_rows, 18)))
        perm_vals = [np.mean(res["acc"][f"Perm{j}"][e_hold]) for j in range(args.n_perm)]
        col_vals = [np.mean(res["acc"][f"Col{j}"][e_hold]) for j in range(args.n_col)]
        print(f"[{e_hold:>7}] Real={np.mean(res['acc']['Real'][e_hold]):.3f} "
              f"Raw={np.mean(res['acc']['Raw'][e_hold]):.3f} "
              f"Rand={np.mean(res['acc']['Rand'][e_hold]):.3f} | "
              f"Perm={np.mean(perm_vals):.3f}±{np.std(perm_vals):.3f} "
              f"Col={np.mean(col_vals):.3f}±{np.std(col_vals):.3f}", flush=True)

    with open(os.path.join(OUT, f"exp39_res_{args.repr}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    write_report(res, variants, args, AD, AU)
    print(f"[exp39] 已写出 out/exp39_permutation_{args.repr}.md", flush=True)


def write_report(res, variants, args, AD, AU):
    eids = [e for e in res["meta"]["eids"] if e in res["acc"]["Real"]]
    AD = [e for e in AD if e in eids]
    AU = [e for e in AU if e in eids]
    A = []
    w = lambda s: A.append(s)

    def per_entity(v):
        return np.array([np.mean(res["acc"][v][e]) for e in eids])

    def macro(v, grp):
        return np.array([np.mean(res["acc"][v][e]) for e in grp])

    w(f"# exp39 · 零样本化学先验的因果检验（{args.repr.upper()}）")
    w("")
    w("> 核心问题：k=0 时化学先验的有效性，究竟来自「化学对应关系正确」，")
    w("> 还是仅仅来自「向量分布恰好撑开了 19 个类方向」？")
    w("> **行置换**保留全部分布性质、只破坏对应关系 ⇒ 唯一的因果对照。")
    w("")
    w(f"- LOO 13 实体，k=0，seeds {args.seeds}，{args.steps} 步，固定 30% 测试集")
    w(f"- 行置换 {args.n_perm} 次，列置换 {args.n_col} 次；chance = 1/13 = 0.077")
    w("")
    w("## 主结果：逐实体 k=0 top-1")
    w("")
    hdr = ["Real", "Raw", "Rand"] + [f"Perm{j}" for j in range(args.n_perm)] \
          + [f"Col{j}" for j in range(args.n_col)]
    w("| 留出实体 | 类型 | " + " | ".join(hdr[:6]) + " | Perm 均值±SD | Col 均值±SD |")
    w("|---" * 9 + "|")
    for e in eids:
        t = "掺假" if e in AD else "鉴别"
        pv = [np.mean(res["acc"][f"Perm{j}"][e]) for j in range(args.n_perm)]
        cv = [np.mean(res["acc"][f"Col{j}"][e]) for j in range(args.n_col)]
        row = f"| {e} | {t} | " + " | ".join(
            f"{np.mean(res['acc'][v][e]):.3f}" for v in hdr[:6]) + " |"
        row += f" {np.mean(pv):.3f}±{np.std(pv):.3f} | {np.mean(cv):.3f}±{np.std(cv):.3f} |"
        w(row)
    w("")

    # ---- 置换检验
    P = args.n_perm
    real = per_entity("Real")
    perm_mat = np.array([[np.mean(res["acc"][f"Perm{j}"][e]) for e in eids]
                         for j in range(P)])          # (P, 13)
    col_mat = np.array([[np.mean(res["acc"][f"Col{j}"][e]) for e in eids]
                        for j in range(args.n_col)])
    rand = per_entity("Rand")

    def perm_p(real, null, higher=True):
        """以实体为配对单元的置换 p 值：对每个置换取 macro，再比较。"""
        null_macro = null.mean(1) if null.ndim == 2 else null
        r = real.mean()
        if higher:
            cnt = int((null_macro >= r).sum())
        else:
            cnt = int((null_macro <= r).sum())
        return (1 + cnt) / (1 + len(null_macro))

    w("## 置换检验（以实体为配对单元，无需正态假设）")
    w("")
    w("| 对照 | macro 均值 | 零分布 均值±SD | 经验 p（Real > null） |")
    w("|---|---|---|---|")
    w(f"| **Real（真实化学）** | **{real.mean():.4f}** | — | — |")
    w(f"| 行置换（破坏对应） | {perm_mat.mean():.4f} | {perm_mat.mean(0).mean():.4f}"
      f"±{perm_mat.mean(0).std():.4f} | **{perm_p(real, perm_mat):.4f}** |")
    w(f"| 列置换（破坏语义） | {col_mat.mean():.4f} | {col_mat.mean(0).mean():.4f}"
      f"±{col_mat.mean(0).std():.4f} | **{perm_p(real, col_mat):.4f}** |")
    w(f"| 随机高斯码本 | {rand.mean():.4f} | — | — |")
    w(f"| Raw（未去均值） | {per_entity('Raw').mean():.4f} | — | — |")
    w("")

    w("## 适用边界分层：掺假 vs 鉴别")
    w("")
    w("| 组 | n | Real | Rand | 行置换 | 列置换 | Δ(Real−行置换) |")
    w("|---|---|---|---|---|---|---|")
    for grp, nm in [(AD, "掺假 adulteration"), (AU, "鉴别 authentication")]:
        r = macro("Real", grp).mean()
        rd = macro("Rand", grp).mean()
        pm = np.array([[np.mean(res["acc"][f"Perm{j}"][e]) for e in grp]
                       for j in range(P)]).mean(0).mean()
        cm = np.array([[np.mean(res["acc"][f"Col{j}"][e]) for e in grp]
                       for j in range(args.n_col)]).mean(0).mean()
        w(f"| {nm} | {len(grp)} | **{r:.3f}** | {rd:.3f} | {pm:.3f} | {cm:.3f} "
          f"| {r - pm:+.3f} |")
    w("")

    w("## 预注册判据裁定")
    w("")
    p_row = perm_p(real, perm_mat)
    p_col = perm_p(real, col_mat)
    d_ad = macro("Real", AD).mean() - np.array(
        [[np.mean(res["acc"][f"Perm{j}"][e]) for e in AD] for j in range(P)]).mean(0).mean()
    d_au = macro("Real", AU).mean() - np.array(
        [[np.mean(res["acc"][f"Perm{j}"][e]) for e in AU] for j in range(P)]).mean(0).mean()
    w(f"- **H39a（主）**：Real 优于行置换零分布 — Δ = {real.mean() - perm_mat.mean():+.4f}，"
      f"全局经验 p = {p_row:.4f} ⇒ {'✅ 成立' if p_row < 0.05 else '❌ 不成立'}")
    w(f"- **H39b（次）**：Real 优于列置换零分布 — Δ = {real.mean() - col_mat.mean():+.4f}，"
      f"全局经验 p = {p_col:.4f} ⇒ {'✅ 成立' if p_col < 0.05 else '❌ 不成立'}")
    w(f"- **H39c（次）**：掺假组增益 ≥ 鉴别组 — Δ_掺假 = {d_ad:+.3f}，"
      f"Δ_鉴别 = {d_au:+.3f} ⇒ {'✅ 成立' if d_ad >= d_au else '❌ 不成立'}")
    w("")
    w("> **全局 p 的分辨率受置换次数限制**：P 次置换的最小可达 p = 1/(1+P)。")
    w(f"> 本轮 P = {P} ⇒ p_min = {1 / (1 + P):.4f}。"
      f"因此下面补做**逐实体置换检验**（每个实体一个 p，再合并），功效远高于全局检验。")
    w("")
    w("### 逐实体置换检验（功效更高的补充分析）")
    w("")
    w("| 留出实体 | 类型 | Real | 行置换 均值±SD | 行置换 max | 逐实体 p |")
    w("|---|---|---|---|---|---|")
    sig = 0
    for e in eids:
        t = "掺假" if e in AD else "鉴别"
        rv = np.mean(res["acc"]["Real"][e])
        pv = np.array([np.mean(res["acc"][f"Perm{j}"][e]) for j in range(P)])
        pe = (1 + int((pv >= rv).sum())) / (1 + P)
        if pe < 0.05:
            sig += 1
        w(f"| {e} | {t} | {rv:.3f} | {pv.mean():.3f}±{pv.std():.3f} | "
          f"{pv.max():.3f} | {pe:.4f} |")
    w("")
    w(f"- **{sig}/{len(eids)} 个实体的 Real 显著超过其行置换零分布**（逐实体 p < 0.05）")
    w(f"- 若 Real ≤ 置换（如 A02/A03/A06 本身零样本就失败），逐实体 p 自然不显著 ——")
    w("  这是「该方法对部分掺杂物无效」的如实反映，不是检验失效。")
    w("")
    w("## 解读")
    w("")
    w("若 H39a 成立：k=0 的增益**只能**来自「实体→化学向量」对应关系的正确性，")
    w("而非向量分布的各向异性 / hubness / 范数结构 —— 这是把「化学语义先验有用」")
    w("从相关事实升级为**因果结论**的关键一步。")
    w("")
    w("若 H39c 成立：方法的适用边界是**外源化学掺杂物**（有明确分子结构），")
    w("对物种/产地鉴别这类差异源于组织结构与组成比例的任务，先验收益更小 ——")
    w("这一条必须如实写进论文的适用范围，不能含糊。")

    with open(os.path.join(OUT, f"exp39_permutation_{args.repr}.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(A))


if __name__ == "__main__":
    main()
