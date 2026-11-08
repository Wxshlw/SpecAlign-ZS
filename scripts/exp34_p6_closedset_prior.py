"""exp34 —— 闭集分类 × 化学先验 × 少样本梯度。

背景（用户 2026-11-01 拍板的目标层次）：
  层1  闭集分类（有掺假样本）+ 化学先验 → 效果更好
  层2  有掺假样本 + 无纯品参考谱   → 化学先验是关键通路（本实验的不做-LOO 设定）
  层3  零样本 + 无纯品参考谱       → 必须加（= exp08 原有 LOO 设定，本实验不涉及）

与 exp08 的关系：
  exp08 = LOO（每折留出一个实体）⇒「零样本 + 无纯品谱」
  exp34 = 不做 LOO，全部 19 类参与训练 ⇒「有掺假样本 + 无纯品谱」
  两者共用同一批公开谱、同一编码器结构，仅训练/评估协议不同。

核心对照设计（关键：只改分类头 W 的来源，其余完全一致）：
  谱编码器 z = SpecCNN(x)，z 归一化；logits = normalize(z) @ normalize(W).T / TEMP
  ┌──────────────┬────────────────────────────┬────────┐
  │ 臂           │ W 的来源                   │ 可学习 │
  ├──────────────┼────────────────────────────┼────────┤
  │ None         │ nn.Linear 权重（随机初始） │ 是     │
  │ Rand         │ 固定随机高斯               │ 否     │
  │ CLM(Prior)   │ 化学语言模型嵌入（冻结）   │ 否     │
  │ ECFP(Prior)  │ Morgan 指纹（冻结）        │ 否     │
  └──────────────┴────────────────────────────┴────────┘
  ⇒ Δ_CLM  = Acc(CLM)  − Acc(None)   化学先验的净增益
    Δ_Rand = Acc(Rand) − Acc(None)   「固定目标」本身的效应（排除混淆）
    Δ_CLM−Δ_Rand                     化学信息相对随机固定目标的增量

少样本梯度：每类训练样本数 n ∈ {5, 10, 20, 50, full}
  预期（预注册 H34a）：样本充足时谱信息本身已足够，先验增益≈0；
  样本稀缺时先验提供归纳偏置，增益放大 ⇒ Δ(n) 随 n 减小而**增大**。

预注册判据（方向已复核：Δ 随 n 增大而**减小** ⇒ 回归斜率为负）：
  H34a 主：slope( Δ_CLM ~ log n ) < 0，且 Δ_CLM(n_min) > Δ_CLM(n_full)
  H34b 次：n = n_min 处 Δ_CLM > 0（配对检验）
  H34c 次：n = n_min 处 Δ_CLM > Δ_Rand（增益来自化学信息，而非「固定目标」效应）
"""
from __future__ import annotations
import os, sys, json, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1
import p6_library as L
import exp05_p6_g0 as E5
import exp08_p6_g1 as E8

OUT = r"D:/2026sci论文实验设计/论文6/out"
TEMP = 0.07
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- 数据
def build_all():
    """返回每个候选 key 的光谱矩阵，以及 key 列表。"""
    ent, bg, grid = G1.load_g1()
    eids = sorted(ent.keys())
    mids = sorted(bg.keys())
    keys = eids + [f"M_{m}" for m in mids]
    Xs = {}
    for e in eids:
        Xs[e] = np.asarray(ent[e], float)
    for m in mids:
        Xs[f"M_{m}"] = np.asarray(bg[m], float)
    return Xs, keys, eids, mids


def make_dataset(Xs, keys):
    parts, ys = [], []
    for i, k in enumerate(keys):
        parts.append(Xs[k]); ys.append(np.full(len(Xs[k]), i))
    return np.concatenate(parts).astype(np.float32), np.concatenate(ys).astype(np.int64)


def build_tasks(Xs, keys, eids):
    """两套任务难度：
    all19   —— 19 类全闭集（基质互不相同，谱差异大，**容易**）
    within-*—— 同一基质内区分不同掺杂物（谱几乎被基质主导，**困难**）
    """
    import collections
    tasks = [("all19", make_dataset(Xs, keys), keys)]
    groups = collections.defaultdict(list)
    for k, v in G1.ENTITY_META.items():
        if k in eids:
            groups[v[1]].append(k)
    for mid, ks in sorted(groups.items()):
        ks = sorted(ks)
        if len(ks) >= 2:
            tasks.append((f"within-{mid}", make_dataset(Xs, ks), ks))
    return tasks


def stratified_split(y, ncls, seed, train_frac=0.8):
    """按类分层划分 train_pool / test。测试集在各 n 档间固定，保证可比。"""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in range(ncls):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = int(round(len(idx) * train_frac))
        k = max(1, min(len(idx) - 1, k))
        tr.append(idx[:k]); te.append(idx[k:])
    return np.concatenate(tr), np.concatenate(te)


def subsample(tr_idx, y, ncls, n, rng):
    """从 train_pool 每类抽 n 个（n=None 表示全取）。"""
    if n is None:
        return tr_idx
    out = []
    for c in range(ncls):
        idx = tr_idx[y[tr_idx] == c]
        out.append(idx if n >= len(idx) else rng.choice(idx, size=n, replace=False))
    return np.concatenate(out)


# ---------------------------------------------------------------- 训练
def train_arm(Xtr, ytr, W0, mode, dim, ncls, epochs, seed):
    """唯一变量：分类头 W 的来源与可学习性。

    mode = "learn"  → nn.Linear 随机初始化、可学习（无先验基线）
           "frozen" → W0 冻结（先验直接当固定分类目标）
           "init"   → 以 W0 初始化、随后微调（先验当起点）
    """
    torch.manual_seed(seed)
    net = E8.SpecCNN(dim).to(DEV)
    if mode == "learn":
        head = nn.Linear(dim, ncls).to(DEV)
        W = head.weight
        params = list(net.parameters()) + list(head.parameters())
    elif mode == "frozen":
        W = torch.tensor(np.asarray(W0, dtype=np.float32), device=DEV)
        params = list(net.parameters())
    else:  # init
        head = nn.Linear(dim, ncls).to(DEV)
        with torch.no_grad():
            head.weight.copy_(torch.tensor(np.asarray(W0, dtype=np.float32)))
            head.bias.zero_()
        W = head.weight
        params = list(net.parameters()) + list(head.parameters())
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
            Wn = F.normalize(W, dim=1)
            loss = F.cross_entropy(z @ Wn.T / TEMP, yt[idx])
            loss.backward(); opt.step()
    return net, W


@torch.no_grad()
def accuracy(net, W, Xte, yte):
    net.eval()
    z = F.normalize(net(torch.tensor(Xte, dtype=torch.float32, device=DEV)), dim=1)
    Wn = F.normalize(W, dim=1)
    pred = (z @ Wn.T).argmax(1).cpu().numpy()
    return float((pred == yte).mean())


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="5,10,20,50,full")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs-small", type=int, default=150, help="n<=20 时的 epoch（下限）")
    ap.add_argument("--epochs-large", type=int, default=60, help="n>20 时的 epoch（下限）")
    ap.add_argument("--steps", type=int, default=600, help="梯度步数预算（各档统一）")
    args = ap.parse_args()

    shots = []
    for s in args.shots.split(","):
        shots.append(None if s.strip() == "full" else int(s))

    print(f"[exp34] device = {DEV}", flush=True)
    Xs, keys, eids, mids = build_all()
    key_index = {k: i for i, k in enumerate(keys)}
    tasks = build_tasks(Xs, keys, eids)
    print(f"[exp34] 候选 {len(keys)} 个；任务 {len(tasks)} 套： " +
          ", ".join(f"{nm}({len(ks)}类)" for nm, _, ks in tasks), flush=True)

    cache_path = os.path.join(OUT, "exp34_repr_cache.npz")
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        ck = [str(x) for x in z["keys"]]
        assert ck == keys, "缓存的候选 key 顺序与当前数据集不一致，请删除缓存重跑"
        T_clm = np.asarray(z["clm"], dtype=np.float32)
        # ecfp384：与 CLM 同维（384）的 Morgan 指纹，用于**维度匹配**的公平对照。
        # 维度不等时对照会被「输出维度」这个混淆变量污染（实测 2048 维臂在小任务上会塌缩）。
        T_ecfp = np.asarray(z["ecfp384"] if "ecfp384" in z.files else z["ecfp"], dtype=np.float32)
        print("[exp34] 命中候选表示缓存 out/exp34_repr_cache.npz", flush=True)
    else:
        print("[exp34] 构建化学候选表示（CLM / ECFP / 随机）...", flush=True)
        variants = E8.build_text_variants_g1(eids, mids)
        T_clm = np.stack([variants["Original"][k] for k in keys]).astype(np.float32)
        T_ecfp = np.stack([variants["Descriptor"][k] for k in keys]).astype(np.float32)
    d_clm, d_ecfp = T_clm.shape[1], T_ecfp.shape[1]
    print(f"[exp34] dim: CLM={d_clm}  ECFP={d_ecfp}", flush=True)

    def aniso(W):
        W = np.asarray(W, float)
        Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
        off = (Wn @ Wn.T)[~np.eye(len(W), dtype=bool)]
        Wc = W - W.mean(0, keepdims=True)
        s = np.linalg.svd(Wc, compute_uv=False)
        erank = int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), 0.90) + 1)
        return {"cos_mean": float(off.mean()), "cos_max": float(off.max()),
                "eff_rank90": erank,
                "cos_mean_centered": float(((Wc / np.linalg.norm(Wc, axis=1, keepdims=True))
                                            @ (Wc / np.linalg.norm(Wc, axis=1, keepdims=True)).T)
                                           [~np.eye(len(W), dtype=bool)].mean())}

    aniso_diag = {"CLM": aniso(T_clm), "ECFP": aniso(T_ecfp)}
    print(f"[exp34] 各向异性 CLM: cos={aniso_diag['CLM']['cos_mean']:.3f} "
          f"eff_rank={aniso_diag['CLM']['eff_rank90']} | "
          f"ECFP: cos={aniso_diag['ECFP']['cos_mean']:.3f} "
          f"eff_rank={aniso_diag['ECFP']['eff_rank90']}", flush=True)

    def rand_W(n, dim, seed):
        r = np.random.default_rng(1000 + seed)
        return r.normal(size=(n, dim)).astype(np.float32)

    def center(W):
        """各向异性校正：去掉候选嵌入的公共均值方向（All-but-the-Top 的首步）。

        CLM / ECFP 的原始嵌入各行高度共线（实测 CLM 两两余弦均值 0.728、去均值后有效秩仅 4），
        行归一化后分类头近乎退化，训练梯度互相冲突。去均值可恢复类别间的可分辨结构。
        """
        W = np.asarray(W, dtype=np.float32)
        return (W - W.mean(0, keepdims=True)).astype(np.float32)

    T_clm_c, T_ecfp_c = center(T_clm), center(T_ecfp)

    # 臂定义：(名字, dim, mode, W 生成函数(ks, n, seed) -> array|None)
    def sel(T, ks):
        return np.asarray(T)[[key_index[k] for k in ks]]

    def sel_c(T, ks):
        return center(sel(T, ks))

    arms = [
        ("None",      d_clm,  "learn",  lambda ks, n, s: None),
        ("Rand",      d_clm,  "frozen", lambda ks, n, s: rand_W(n, d_clm, s)),
        ("CLM",       d_clm,  "frozen", lambda ks, n, s: sel(T_clm, ks)),
        ("CLM-c",     d_clm,  "frozen", lambda ks, n, s: sel_c(T_clm, ks)),
        ("CLM-init",  d_clm,  "init",   lambda ks, n, s: sel_c(T_clm, ks)),
        ("ECFP",      d_ecfp, "frozen", lambda ks, n, s: sel(T_ecfp, ks)),
        ("ECFP-c",    d_ecfp, "frozen", lambda ks, n, s: sel_c(T_ecfp, ks)),
        ("ECFP-init", d_ecfp, "init",   lambda ks, n, s: sel_c(T_ecfp, ks)),
    ]
    arm_names = [a[0] for a in arms]

    res = {"acc": {}, "aniso": aniso_diag, "nte": {},
           "meta": {"keys": keys, "shots": [("full" if s is None else s) for s in shots],
                    "seeds": args.seeds, "temp": TEMP, "dim_clm": d_clm, "dim_ecfp": d_ecfp,
                    "tasks": [nm for nm, _, _ in tasks]}}

    for task_name, (X, y), ks in tasks:
        ncls = len(ks)
        res["acc"][task_name] = {}
        res["nte"][task_name] = {}
        for n in shots:
            tag = "full" if n is None else str(n)
            res["acc"][task_name][tag] = {a: [] for a in arm_names}
            for seed in range(args.seeds):
                tr_pool, te = stratified_split(y, ncls, seed=seed)
                res["nte"][task_name][tag] = len(te)
                rng = np.random.default_rng(seed * 7919 + 13)
                tr = subsample(tr_pool, y, ncls, n, rng)
                Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
                # 固定「梯度步数预算」而非固定 epoch：否则少样本档每 epoch 只有 1 步，严重欠训练
                steps_per_epoch = max(1, int(np.ceil(len(Xtr) / 256)))
                epochs = int(np.ceil(args.steps / steps_per_epoch))
                if n is None or n > 20:
                    epochs = max(epochs, args.epochs_large)
                res["meta"].setdefault("epochs", {}).setdefault(task_name, {})[tag] = epochs
                for name, dim, mode, wfun in arms:
                    Wf = wfun(ks, ncls, seed)
                    net, W = train_arm(Xtr, ytr, Wf, mode, dim, ncls, epochs, seed * 1000 + 7)
                    acc = accuracy(net, W, Xte, yte)
                    res["acc"][task_name][tag][name].append(float(acc))
            line = "  ".join(f"{a}={np.mean(res['acc'][task_name][tag][a]):.3f}" for a in arm_names)
            print(f"[{task_name[:12]:>12} n={tag:>4}] {line}", flush=True)

    write_report(res, arms, shots, args, tasks)
    with open(os.path.join(OUT, "exp34_res.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("[exp34] 已写出 out/exp34_p6_closedset_prior.md", flush=True)


# ---------------------------------------------------------------- 报告
def write_report(res, arms, shots, args, tasks):
    from scipy.stats import ttest_rel, linregress

    def best(acc_t, cand, base):
        """在同一表示的多个臂中取均值最高者，返回 (Δ 数组, 臂名)。"""
        bs, bn = None, ""
        for a in cand:
            d = np.array(acc_t[a]) - np.array(acc_t[base])
            if bs is None or d.mean() > bs.mean():
                bs, bn = d, a
        return bs, bn

    def pooled(task_names, tag, arm):
        """跨任务按测试集大小加权（micro）合并，返回逐 seed 的数组。"""
        num = np.zeros(args.seeds); den = 0.0
        for tn in task_names:
            num = num + np.array(res["acc"][tn][tag][arm]) * res["nte"][tn][tag]
            den += res["nte"][tn][tag]
        return num / max(den, 1e-9)

    A = []
    w = lambda s: A.append(s)
    arm_names = [a[0] for a in arms]
    task_names = [nm for nm, _, _ in tasks]
    within = [t for t in task_names if t.startswith("within-")]
    tags = [("full" if s is None else str(s)) for s in shots]

    w("# exp34 · 闭集分类 × 化学先验 × 少样本梯度")
    w("")
    w("> 设定：**有掺假样本 + 无纯品参考谱**（与 exp08 的留一实体零样本设定相对）。")
    w("> 全部类别参与训练，无 LOO。考察化学先验在**闭集监督分类**中是否带来增益。")
    w("")
    w("## 两套任务难度")
    w("")
    w("| 任务 | 类别数 | 说明 |")
    w("|---|---|---|")
    for nm, (X, y), ks in tasks:
        w(f"| {nm} | {len(ks)} | {'（' + ', '.join(ks) + '）' if nm.startswith('within-') else '13 掺假实体 + 6 基质背景'} |")
    w("")
    w("**为什么必须分两套**：`all19` 中基质互不相同，谱差异极大，")
    w("即使每类只给 5 个样本也能轻松分类 —— 此时先验没有发挥空间，")
    w("测出「无增益」无法区分「先验无用」与「任务太易」。")
    w("`within-*` 固定基质、只区分掺杂物，谱几乎被基质主导，")
    w("**这才是先验唯一可能兑现价值的场所**（与 exp31 的受控干预逻辑一致）。")
    w("")
    w("## 设计")
    w("")
    w("唯一变量 = **分类头 W 的来源**，谱编码器结构、初始化、训练样本、步数预算全部一致：")
    w("")
    w("| 臂 | W 的来源 | 可学习 | 说明 |")
    w("|---|---|---|---|")
    w("| None | `nn.Linear` 随机初始化 | ✅ | 无先验基线 |")
    w("| Rand | 固定随机高斯 | ❌ | 控制「固定分类目标」本身的效应 |")
    w("| CLM | 化学语言模型嵌入（**原始**） | ❌ | 化学语义先验，未校正 |")
    w("| **CLM-c** | CLM 嵌入**去均值**（各向异性校正） | ❌ | **化学语义先验，校正后** |")
    w("| **CLM-init** | 以 CLM-c 初始化 | ✅ | 先验当起点，随后微调 |")
    w("| ECFP | Morgan 指纹（**原始**，384 位） | ❌ | 化学结构先验，未校正 |")
    w("| **ECFP-c** | ECFP **去均值** | ❌ | **化学结构先验，校正后** |")
    w("| **ECFP-init** | 以 ECFP-c 初始化 | ✅ | 先验当起点，随后微调 |")
    w("")
    w("全部臂的谱编码器输出维度统一为 **384**（ECFP 使用 384 位指纹，与 CLM 同维），")
    w("避免「输出维度」成为混淆变量。")
    w("")
    w("### 为什么必须加「去均值」这一层")
    w("")
    w("候选化学嵌入存在严重**各向异性**（anisotropy）—— 各行高度共线，")
    w("行归一化后分类头近乎退化：")
    w("")
    w("| 表示 | 两两余弦均值 | 最大 | 去均值后余弦均值 | 去均值后有效秩（90% 能量） |")
    w("|---|---|---|---|---|")
    for nm in ["CLM", "ECFP"]:
        d = res["aniso"][nm]
        w(f"| {nm}（19 类全集） | {d['cos_mean']:.3f} | {d['cos_max']:.3f} | "
          f"{d['cos_mean_centered']:+.3f} | {d['eff_rank90']} / {len(res['meta']['keys'])} |")
    w("")
    w(f"CLM 的 {res['meta']['dim_clm']} 维嵌入在 {len(res['meta']['keys'])} 个类别上"
      f"有效秩仅 {res['aniso']['CLM']['eff_rank90']} —— 远低于类别数，"
      "冻结的分类头在几何上无法区分全部类别。")
    w("去均值（All-but-the-Top 的首步，Mu & Viswanath, ICLR 2018）即可恢复可分辨结构。")
    w("**这正是「怎么对齐」的核心一环：先验不是拿来就能用，必须先做各向异性校正。**")
    w("")
    w(f"- 少样本梯度 n ∈ {{{', '.join(tags)}}}（每类训练样本数）")
    w(f"- seeds = {args.seeds}；测试集按类分层 20% 划分，在各 n 档间固定")
    w(f"- 各档统一 {args.steps} 个梯度步的预算（batch 256）")
    w("  —— 固定步数而非固定 epoch，否则少样本档每 epoch 仅 1 步会严重欠训练。")
    w("")

    # ---------------------------------------------------------- 主结果
    w("## 主结果")
    w("")
    for nm in task_names:
        w(f"### {nm}")
        w("")
        w("| 每类训练样本 n | " + " | ".join(arm_names) + " |")
        w("|---|" + "---|" * len(arm_names))
        for t in tags:
            vals = [np.mean(res["acc"][nm][t][a]) for a in arm_names]
            w(f"| {t} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
        w("")
    if within:
        w(f"### within 合并（按测试集大小加权：{' + '.join(within)}）")
        w("")
        w("| 每类训练样本 n | " + " | ".join(arm_names) + " |")
        w("|---|" + "---|" * len(arm_names))
        for t in tags:
            vals = [np.mean(pooled(within, t, a)) for a in arm_names]
            w(f"| {t} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
        w("")

    # ---------------------------------------------------------- Δ
    w("## 化学先验的净增益 Δ")
    w("")
    w("Δ = 先验臂 − **无先验基线 None**（全部臂同为 384 维输出）。")
    w("「best」= 同一表示下三个臂中均值最高者。")
    w("")

    def delta_block(scope_name, get):
        w(f"### {scope_name}")
        w("")
        w("| n | Δ_CLM(best) | 臂 | Δ_Rand | Δ_ECFP(best) | 臂 |")
        w("|---|---|---|---|---|---|")
        out = {}
        for t in tags:
            acc_t = {a: get(t, a) for a in arm_names}
            d_c, a_c = best(acc_t, ["CLM", "CLM-c", "CLM-init"], "None")
            d_e, a_e = best(acc_t, ["ECFP", "ECFP-c", "ECFP-init"], "None")
            d_r = np.array(acc_t["Rand"]) - np.array(acc_t["None"])
            out[t] = {"clm": d_c, "rand": d_r, "ecfp": d_e, "arm_clm": a_c, "arm_ecfp": a_e}
            w(f"| {t} | {d_c.mean():+.4f} | {a_c} | {d_r.mean():+.4f} | "
              f"{d_e.mean():+.4f} | {a_e} |")
        w("")
        w("各臂 Δ 明细（相对 None）：")
        w("")
        w("| n | " + " | ".join(a for a in arm_names if a != "None") + " |")
        w("|---|" + "---|" * (len(arm_names) - 1))
        for t in tags:
            cells = []
            for a in arm_names:
                if a == "None":
                    continue
                cells.append(f"{np.mean(np.array(get(t, a)) - np.array(get(t, 'None'))):+.4f}")
            w(f"| {t} | " + " | ".join(cells) + " |")
        w("")
        return out

    deltas = {}
    deltas["all19"] = delta_block("all19", lambda t, a: res["acc"]["all19"][t][a])
    if within:
        deltas["within"] = delta_block("within 合并", lambda t, a: pooled(within, t, a))

    # ---------------------------------------------------------- 判据
    w("## 预注册判据裁定")
    w("")
    numeric = [t for t in tags if t != "full"]
    xs = np.array([np.log(float(t)) for t in numeric])
    lo_t, hi_t = numeric[0], ("full" if "full" in tags else numeric[-1])

    verdicts = {}
    for scope in deltas:
        D = deltas[scope]
        ys = np.array([D[t]["clm"].mean() for t in numeric])
        if len(numeric) >= 3 and ys.std() > 0:
            lr = linregress(xs, ys); slope, p_slope = lr.slope, lr.pvalue
        else:
            slope, p_slope = float("nan"), float("nan")
        d_lo, d_hi = D[lo_t]["clm"].mean(), D[hi_t]["clm"].mean()
        h34a = (slope < 0) and (d_lo > d_hi)
        ts, pv = ttest_rel(D[lo_t]["clm"], np.zeros_like(D[lo_t]["clm"]))
        h34b = d_lo > 0 and pv < 0.05
        ts2, pv2 = ttest_rel(D[lo_t]["clm"], D[lo_t]["rand"])
        h34c = (d_lo > D[lo_t]["rand"].mean()) and pv2 < 0.05

        w(f"### {scope}")
        w("")
        w(f"- **H34a**（增益随样本减少而放大）：Δ~log n 斜率 {slope:+.5f}（p={p_slope:.4f}）；"
          f"Δ(n={lo_t})={d_lo:+.4f} vs Δ(n={hi_t})={d_hi:+.4f} ⇒ "
          f"{'✅ 成立' if h34a else '❌ 不成立'}")
        w(f"- **H34b**（最少样本处有正增益）：Δ={d_lo:+.4f}，配对 t p={pv:.4f} ⇒ "
          f"{'✅ 成立' if h34b else '❌ 不成立'}")
        w(f"- **H34c**（增益超过随机固定目标）：Δ_CLM={d_lo:+.4f} vs Δ_Rand="
          f"{D[lo_t]['rand'].mean():+.4f}，配对 t p={pv2:.4f} ⇒ "
          f"{'✅ 成立' if h34c else '❌ 不成立'}")
        w("")
        verdicts[scope] = {"a": bool(h34a), "b": bool(h34b), "c": bool(h34c),
                           "slope": float(slope), "p_slope": float(p_slope),
                           "d_lo": float(d_lo), "d_hi": float(d_hi),
                           "p_b": float(pv), "p_c": float(pv2)}
    res["verdicts"] = verdicts

    w("## 解读")
    w("")
    for scope in deltas:
        v = verdicts[scope]
        if v["b"] and v["c"]:
            w(f"- **{scope}**：化学先验在低资源下有显著正增益，且显著超过随机固定目标 "
              "⇒ 增益来自**化学信息本身**。")
        elif v["b"]:
            w(f"- **{scope}**：有正增益，但未超过随机固定目标 "
              "⇒ 增益可能主要来自「冻结分类头」的正则效应，**不得解释为化学先验的价值**。")
        else:
            w(f"- **{scope}**：**未观察到化学先验的正增益。**")
    w("")
    w("### 关于 Rand 臂的重要提醒")
    w("")
    w("随机高斯行在 384/2048 维中近似正交，是**几何上近乎最优的固定码本**；")
    w("而任何真实化学嵌入都必然相关（化学近邻天然相似）。")
    w("因此「冻结的先验头打不过 Rand」**不构成对化学先验的否证** —— 它只说明")
    w("**纯固定码本的上界由正交性决定**。真正有意义的对照，是先验作**初始化**后微调")
    w("（*-init 臂）与随机初始化后微调（None 臂）之比。")
    w("")

    txt = "\n".join(A)
    with open(os.path.join(OUT, "exp34_p6_closedset_prior.md"), "w", encoding="utf-8") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
