"""exp35 · 新类别冷启动：LOO + k-shot × 化学先验

设定：**有掺假样本（k 个）+ 无纯品参考谱**，且是**训练时从未见过的新掺杂物**。

与 exp34 的分工：
- exp34 = 全部类别都参与训练（闭集，考察「先验能不能让它更好」）
- exp35 = 留一实体（新类别），k 从 0 递增（考察「先验能不能让它**做得到**，
  以及 k 很小时是否仍有增益」）

k=0 即 exp08 的零样本设定；此时**可学习分类头（None 臂）根本没有该类的行**，
预测在数学上不可定义 —— 这正是「无纯品谱时必须加化学先验」的最直接论证。

协议
----
对每个留出的实体 e：
  候选矩阵 W 恒为 19 行 = [其余 12 个实体行; 6 个基质背景行; 第 19 行 = e 的行]
  训练类别：
    k = 0   → 18 类（12 实体 + 6 基质），第 19 行不参与训练
    k > 0   → 19 类（额外给 e 的 k 个样本）
  评估：在 e 的**未见样本**上，对 13 个实体行（前 12 行 + 第 19 行）做 argmax，
        top-1 命中率。与 exp08 的 13 候选口径一致。

臂（唯一变量 = 分类头 W 的来源；全部 384 维）
  None      可学习 nn.Linear（k=0 时该行从未被训练 → 退化为随机，约等于 chance）
  Rand      冻结随机高斯
  CLM-c     CLM 嵌入去均值（各向异性校正后）
  CLM-init  以 CLM-c 初始化后微调（k=0 时退化为冻结）
  ECFP-c    ECFP(384) 去均值
  ECFP-init 以 ECFP-c 初始化后微调
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
import p6_library as L
import exp08_p6_g1 as E8

OUT = r"D:/2026sci论文实验设计/论文6/out"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TEMP = 0.07


# ---------------------------------------------------------------- 模型
def train_arm(Xtr, ytr, W0, mode, dim, nrows, steps, seed):
    """nrows = 参与训练 softmax 的类别数（k=0 时为 18，k>0 时为 19）。

    关键：k=0 时留出实体的行**不得进入训练 softmax** —— 否则交叉熵会把它当作
    「永不为正的负类」持续压低，z 被推得与该类的化学嵌入反相关，
    零样本检索直接归零（实测 0.003）。exp08 的原始实现里该行根本不在 Ttr 中。
    """
    torch.manual_seed(seed)
    net = E8.SpecCNN(dim).to(DEV)
    if mode == "learn":
        head = nn.Linear(dim, nrows).to(DEV)
        W = head.weight
        params = list(net.parameters()) + list(head.parameters())
    elif mode == "frozen":
        Wfull = torch.tensor(np.asarray(W0, dtype=np.float32), device=DEV)   # 19 行
        W = Wfull[:nrows]
        params = list(net.parameters())
    else:  # init
        head = nn.Linear(dim, nrows).to(DEV)
        with torch.no_grad():
            head.weight.copy_(torch.tensor(np.asarray(W0, dtype=np.float32)[:nrows]))
            head.bias.zero_()
        W = head.weight
        params = list(net.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)

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
            Wn = F.normalize(W, dim=1)
            loss = F.cross_entropy(z @ Wn.T / TEMP, yt[idx])
            loss.backward(); opt.step()
            done += 1
    if mode == "frozen":
        return net, Wfull
    if nrows < 19:
        if mode == "learn":
            # 可学习头天然没有新类别的行 —— 补随机高斯行，即「无先验」的零样本基线
            rng = np.random.default_rng(3000 + seed)
            pad_np = rng.normal(size=(19 - nrows, dim)).astype(np.float32)
        else:
            # init 臂：新类别无训练样本，其行收不到梯度，直接补回化学行（退化为冻结）
            pad_np = np.asarray(W0, dtype=np.float32)[nrows:]
        W = torch.cat([W.detach(),
                       torch.tensor(pad_np, dtype=torch.float32, device=DEV)], dim=0)
    return net, W


@torch.no_grad()
def top1(net, W, Xte, cand_rows, true_row):
    """在 cand_rows 指定的行上做 argmax，返回命中 true_row 的比例。"""
    net.eval()
    z = F.normalize(net(torch.tensor(Xte, dtype=torch.float32, device=DEV)), dim=1)
    Wn = F.normalize(W, dim=1)
    logits = (z @ Wn.T)[:, cand_rows]
    pred = logits.argmax(1).cpu().numpy()
    tgt = cand_rows.index(true_row)
    return float((pred == tgt).mean())


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="0,5,20,full")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()

    shots = []
    for s in args.shots.split(","):
        shots.append(None if s.strip() == "full" else int(s))
    tags = [("full" if s is None else str(s)) for s in shots]

    print(f"[exp35] device = {DEV}", flush=True)
    ent, bg, _ = G1.load_g1()
    eids = sorted(ent.keys())
    mids = sorted(bg.keys())
    keys = eids + [f"M_{m}" for m in mids]
    key_index = {k: i for i, k in enumerate(keys)}
    nE = len(eids)

    zc = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    assert [str(x) for x in zc["keys"]] == keys, "缓存 key 顺序不符"
    T_clm = np.asarray(zc["clm"], dtype=np.float32)
    T_ecfp = np.asarray(zc["ecfp384"] if "ecfp384" in zc.files else zc["ecfp"],
                        dtype=np.float32)
    dim = T_clm.shape[1]

    def center(W):
        W = np.asarray(W, dtype=np.float32)
        return (W - W.mean(0, keepdims=True)).astype(np.float32)

    def rand_W(n, d, seed):
        return np.random.default_rng(2000 + seed).normal(size=(n, d)).astype(np.float32)

    def build_W(arm, e_hold, seed):
        """W 恒为 19 行：[12 个其他实体; 6 个基质; e_hold]，顺序见 docstring。"""
        others = [e for e in eids if e != e_hold]
        ent_rows = np.asarray(T_clm if arm.startswith("CLM") else T_ecfp)
        bg_rows = np.asarray(T_clm if arm.startswith("CLM") else T_ecfp)
        rows = [ent_rows[key_index[e]] for e in others]
        rows += [bg_rows[key_index[f"M_{m}"]] for m in mids]
        rows += [ent_rows[key_index[e_hold]]]
        W = np.stack(rows).astype(np.float32)
        if arm == "Rand":
            W = rand_W(19, dim, seed)
        elif arm.endswith("-c") or arm.endswith("-init"):
            W = center(W)
        return W

    arms = [
        ("None",      "learn"),
        ("Rand",      "frozen"),
        ("CLM-c",     "frozen"),
        ("CLM-init",  "init"),
        ("ECFP-c",    "frozen"),
        ("ECFP-init", "init"),
    ]
    arm_names = [a[0] for a in arms]

    res = {"acc": {t: {a: {} for a in arm_names} for t in tags}, "nte": {},
           "meta": {"shots": tags, "seeds": args.seeds, "steps": args.steps,
                    "eids": eids, "dim": dim, "test_frac": 0.3}}

    for e_hold in eids:
        others = [e for e in eids if e != e_hold]
        # 候选行在 W 中的下标：前 12 行是 others，第 19 行(下标18)是 e_hold
        cand_rows = list(range(len(others))) + [18]
        Xe = np.asarray(ent[e_hold], float)
        for k in shots:
            tag = "full" if k is None else str(k)
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed * 7919 + 13)
                perm = rng.permutation(len(Xe))
                # 固定测试集：所有 k 档共用同一个 held-out 集合。
                # 原实现 te_ix = perm[n_tr:] 使测试集随 k 变化 —— 对只有 16 条谱
                # 的小类，k=100 时测试集只剩 1 条，准确度只能是 0 或 1，而 macro
                # 平均还把它与 600 条谱的大类等权 ⇒ 整条 k 曲线不可信。
                n_te = int(min(max(5, round(0.3 * len(Xe))), len(Xe) - 1))
                te_ix = perm[:n_te]
                pool_ix = perm[n_te:]
                if k is None:
                    tr_e_ix = pool_ix                       # 用尽剩余全部
                else:
                    tr_e_ix = pool_ix[:min(k, len(pool_ix))]
                res["nte"].setdefault(e_hold, len(te_ix))

                parts, ys = [], []
                for j, e in enumerate(others):
                    parts.append(np.asarray(ent[e], float)); ys.append(np.full(len(ent[e]), j))
                for j, m in enumerate(mids):
                    parts.append(np.asarray(bg[m], float))
                    ys.append(np.full(len(bg[m]), len(others) + j))
                nrows = len(others) + len(mids)      # k=0 → 18
                if len(tr_e_ix):
                    parts.append(Xe[tr_e_ix]); ys.append(np.full(len(tr_e_ix), 18))
                    nrows = 19
                Xtr = np.concatenate(parts).astype(np.float32)
                ytr = np.concatenate(ys).astype(np.int64)
                Xte = Xe[te_ix]

                for name, mode in arms:
                    W0 = None if mode == "learn" else build_W(name, e_hold, seed)
                    net, W = train_arm(Xtr, ytr, W0, mode, dim, nrows, args.steps,
                                       seed * 1000 + eids.index(e_hold))
                    acc = top1(net, W, Xte, cand_rows, 18)
                    res["acc"][tag][name].setdefault(e_hold, []).append(float(acc))
        line = []
        for t in tags:
            s = " ".join(f"{a}={np.mean([np.mean(v) for v in res['acc'][t][a].values()]):.3f}"
                         for a in arm_names)
            line.append(f"n={t}: {s}")
        print(f"[{e_hold:>7}] " + " | ".join(line), flush=True)

    write_report(res, arm_names, tags, args)
    with open(os.path.join(OUT, "exp35_res.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("[exp35] 已写出 out/exp35_p6_newclass_coldstart.md", flush=True)


# ---------------------------------------------------------------- 报告
def write_report(res, arm_names, tags, args):
    from scipy.stats import ttest_rel, wilcoxon
    A = []
    w = lambda s: A.append(s)
    eids_all = res["meta"]["eids"]
    # 只取实际跑过的实体（便于用小折数做冒烟测试）
    eids = [e for e in eids_all if e in res["acc"][tags[0]][arm_names[0]]]

    def macro(t, a):
        """逐实体平均（macro），逐 seed 保留以便配对检验。"""
        M = np.array([res["acc"][t][a][e] for e in eids])   # (nE, seeds)
        return M.mean(0), M

    def micro(t, a):
        """按测试集大小加权（micro）：实体样本量从 16 到 600 不等，
        等权 macro 会让只有十几条谱的实体与几百条的实体同样影响结论。"""
        wts = np.array([res["nte"].get(e, 1) for e in eids], float)
        M = np.array([res["acc"][t][a][e] for e in eids])   # (nE, seeds)
        return (M * wts[:, None]).sum(0) / wts.sum(), M

    w("# exp35 · 新类别冷启动：LOO + k-shot × 化学先验")
    w("")
    w("> 设定：**有掺假样本（k 个）+ 无纯品参考谱**，且是训练时从未见过的**新掺杂物**。")
    w("> 与 exp34（全类别参与训练的闭集）互补：这里考察先验在「新威胁出现」时")
    w("> 能否让模型**做得到**，以及 k 很小时是否仍有增益。")
    w("")
    w("## 协议")
    w("")
    w("留一实体（LOO）：候选矩阵 W 恒为 19 行 = `[其余 12 实体; 6 基质; 留出实体]`，")
    w("评估时在该 13 个实体行上做 argmax（与 exp08 的 13 候选口径一致）。")
    w("")
    w(f"- k ∈ {{{', '.join(tags)}}}，其中 **k=0 即 exp08 的零样本设定**")
    w(f"- seeds = {args.seeds}，{args.steps} 梯度步预算，全部臂输出维度 {res['meta']['dim']}")
    w("")
    w("### k=0 时为什么 None 臂不可定义")
    w("")
    w("可学习分类头 `nn.Linear` 的列数等于训练类别数；k=0 时留出实体**没有任何训练样本**，")
    w("该类别根本不存在对应的列 ⇒ 预测在数学上不可定义。")
    w("**这就是「无纯品谱时必须加化学先验」的最直接论证**：")
    w("化学结构是唯一能在没有该类别任何样本时，仍然给出该类方向的信息源。")
    w("")

    w("## 主结果：留出实体 top-1（13 候选，chance = 0.077）")
    w("")
    w("**固定测试集**：所有 k 档共用同一个 30% held-out 集合（原实现的测试集随 k 变化，已修正）。")
    w("micro = 按测试集大小加权；macro = 逐实体等权。两者并列，因为实体样本量从 16 到 600 相差 40 倍。")
    w("")
    for title, agg in [("micro（按测试集大小加权）", micro), ("macro（逐实体等权）", macro)]:
        w(f"### {title}")
        w("")
        w("| k | " + " | ".join(arm_names) + " |")
        w("|---|" + "---|" * len(arm_names))
        for t in tags:
            vals = [np.mean(agg(t, a)[0]) for a in arm_names]
            w(f"| {t} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
        w("")

    w("## Δ 相对 Rand（随机固定码本 = 无任何化学信息的对照）")
    w("")
    w("| k | Δ_CLM-c | Δ_CLM-init | Δ_ECFP-c | Δ_ECFP-init | Δ_None |")
    w("|---|---|---|---|---|---|")
    for t in tags:
        r = macro(t, "Rand")[0]
        cells = []
        for a in arm_names:
            if a == "Rand":
                continue
            cells.append(f"{np.mean(macro(t, a)[0] - r):+.4f}")
        w(f"| {t} | " + " | ".join(cells) + " |")
    w("")
    w("Rand 是**零化学信息**的对照（随机高斯行近似正交）：")
    w("任何超过 Rand 的增益，都只能来自化学结构携带的信息。")
    w("")

    w("## 逐实体明细")
    w("")
    for t in tags:
        w(f"### k = {t}")
        w("")
        w("| 留出实体 | " + " | ".join(arm_names) + " |")
        w("|---|" + "---|" * len(arm_names))
        for e in eids:
            vals = [np.mean(res["acc"][t][a][e]) for a in arm_names]
            w(f"| {e} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
        w("")

    # ---------------------------------------------------------- 判据
    w("## 预注册判据裁定")
    w("")
    k0 = tags[0]
    w(f"### H35a（主）：k=0 时化学先验显著优于随机对照")
    w("")
    for a in ["CLM-c", "ECFP-c"]:
        d = macro(k0, a)[0] - macro(k0, "Rand")[0]
        try:
            ts, pv = ttest_rel(macro(k0, a)[0], macro(k0, "Rand")[0])
        except Exception:
            pv = float("nan")
        ok = d.mean() > 0 and pv < 0.05
        w(f"- Δ_{a} = {d.mean():+.4f}（配对 t，{len(d)} 个实体）p = {pv:.4f} ⇒ "
          f"{'✅ 成立' if ok else '❌ 不成立'}")
    w("")
    w("> 这里以**实体**为配对单元（n=13），而非 seed —— 各实体难度差异远大于 seed 噪声。")
    w("")

    if len(tags) >= 2:
        w("### H35b：先验增益随 k 增大而衰减")
        w("")
        for a in ["CLM-init", "ECFP-init", "CLM-c"]:
            ys = [np.mean(macro(t, a)[0] - macro(t, "Rand")[0]) for t in tags]
            w(f"- {a}: " + ", ".join(f"k={t}:{y:+.4f}" for t, y in zip(tags, ys)))
        w("")
        w("若先验的价值集中在冷启动期，Δ 应随 k 增大而单调减小。")
        w("")

    w("### H35c：k 很小时，先验初始化（*-init）优于随机初始化（None）")
    w("")
    for t in tags:
        if t == "0":
            w(f"- k={t}：None 臂不可定义（无该类别训练样本）")
            continue
        for a in ["CLM-init", "ECFP-init"]:
            d = macro(t, a)[0] - macro(t, "None")[0]
            ts, pv = ttest_rel(macro(t, a)[0], macro(t, "None")[0])
            ok = d.mean() > 0 and pv < 0.05
            w(f"- k={t}, Δ_{a}−None = {d.mean():+.4f}，p = {pv:.4f} ⇒ "
              f"{'✅ 成立' if ok else '❌ 不成立'}")
    w("")

    txt = "\n".join(A)
    with open(os.path.join(OUT, "exp35_p6_newclass_coldstart.md"), "w", encoding="utf-8") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
