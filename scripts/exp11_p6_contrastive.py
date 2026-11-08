# -*- coding: utf-8 -*-
"""exp11 —— 论文6 Step 3｜Degeneracy-aware contrastive alignment（表示学习修法）。

动机（exp09 已量化）：
  ChemBERTa 文本嵌入空间严重各向异性——19 个类向量两两余弦均值 0.728，PC1 解释方差 68.3%。
  加权组分谱在这种空间里几乎坍缩成一点（鸡–猪文本余弦 0.9996），
  导致 G1 的姊妹实体折（A02/A03、COFA、MEATCH/MEATP）全部塌陷、配对检验全是零贡献。

本实验做"学习式"修法，四臂同源消融（同一数据、同一折、同一 seeds）：
  baseline   : cosine-CE，谱塔直接对齐冻结 CLM 文本（= exp08 的 Original 臂）
  ce_proj    : cosine-CE + 可学习文本投影头 + 可学习温度（控制"投影头"这一混淆变量）
  clip       : CLIP 式**对称** InfoNCE（实例级 spec→text + 类级双向）
  clip_da    : clip + degeneracy-aware 硬负样本加权（姊妹对加权 → 主动拉开简并）

Degeneracy-aware 加权（本实验的核心，无泄漏）：
  负样本 j 对锚 i 的 logit 加 log w_ij，
      w_ij = 1 + λ · max(0, S_ij − τ) / (1 − τ)
  其中 S 是**冻结 CLM 空间**的类间余弦（训练前即可算，只用文本，不用任何谱/标签/测试信息）。
  文本上越像的负样本，惩罚越大 → 直接对抗各向异性造成的hubness 塌陷。

参考：Wang & Isola ICML'20（alignment/uniformity）、CLIP ICML'21（对称 InfoNCE）、
      SimCSE NeurIPS'21、Radovanović JMLR'10（hubness）。
"""
from __future__ import annotations
import os, sys, csv, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1
import exp05_p6_g0 as E5
import exp08_p6_g1 as E8
from exp09_p6_anisotropy import metrics, CORPUS, embed_smiles

OUT = r"D:/2026sci论文实验设计/论文6/out"
ARMS = ["baseline", "abtt", "ce_proj", "clip", "clip_da"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 60
BATCH = 256
TEMP_INIT = 0.07
PROJ_DIM = 256
TAU_DEG_Q = 0.75    # 硬负阈值 τ：取冻结空间类间余弦的分位数（自适应，见 build_logw）
LAM_DEG = 1.0       # 硬负强度

SIBLING_PAIRS = [("MEATCH", "MEATP"), ("MEATCH", "MEATT"), ("MEATP", "MEATT"),
                 ("COFA", "COFR"), ("A01", "A02"), ("A03", "A06"), ("A04", "A06")]


# ----------------------------------------------------------------------------
# 模型
# ----------------------------------------------------------------------------
class TextProj(nn.Module):
    """冻结 CLM 嵌入 → 可学习投影头（梯度只流经这里）。"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(),
                                 nn.Linear(in_dim, out_dim))

    def forward(self, x):
        return self.mlp(x)


def build_logw(S, tau=None, lam=LAM_DEG):
    """degeneracy-aware 硬负权重矩阵（log 域），对角为 0。S: 类间余弦。

    τ 自适应：各向异性空间的余弦普遍很高（本项目 mean-cos≈0.73），
    用固定阈值会让几乎所有负样本都被加权，退化成全局 logit 平移。
    故默认取非对角余弦的 75 分位，只对"相对最像"的负样本加权。
    """
    S = np.asarray(S, float)
    off = S[~np.eye(len(S), dtype=bool)]
    if tau is None:
        tau = float(np.quantile(off, TAU_DEG_Q))
    W = 1.0 + lam * np.clip(S - tau, 0.0, None) / max(1.0 - tau, 1e-6)
    np.fill_diagonal(W, 1.0)
    return np.log(W).astype(np.float32), tau


def abtt_transform(T, k=None):
    """All-but-the-top（Mu & Viswanath, ICLR'18）：用**纯结构分子语料**拟合公共方向并扣除。

    语料只含氨基酸/脂肪酸/糖/生物碱的 SMILES，不含任何谱、标签或候选池样本，
    因此该变换对 held-out 实体同样成立（这是它相对"学习式投影头"的关键优势：
    投影头只在 seen 类上训练，对 held-out 类是外推）。
    k=None → 在 0..6 中选使候选池 mean-cos 最小者（无监督超参选择）。
    """
    emb = embed_smiles(CORPUS)
    C = np.stack([E5._unit(emb[s]) for s in CORPUS])
    mu = C.mean(0, keepdims=True)
    U, sv, Vt = np.linalg.svd(C - mu, full_matrices=False)

    def rm(X, kk):
        X = X - mu
        for j in range(kk):
            X = X - (X @ Vt[j:j + 1].T) @ Vt[j:j + 1]
        return X / np.linalg.norm(X, axis=1, keepdims=True)

    T = np.asarray(T, float)
    if k is None:
        ks = list(range(0, 7))
        k = min(ks, key=lambda kk: metrics(rm(T, kk), "x")["cos_mean"])
    return rm(T, k).astype(np.float32), k


def train_joint(Xs, ys, T_seen, logW_seen, arm, epochs, seed):
    """返回 (spec_net, text_fn, temp)。text_fn: np.ndarray -> np.ndarray(投影后, 未归一化)。"""
    torch.manual_seed(seed)
    in_dim = T_seen.shape[1]
    out_dim = in_dim if arm in ("baseline", "abtt") else PROJ_DIM

    spec = E8.SpecCNN(out_dim).to(DEV)
    if arm in ("baseline", "abtt"):
        text_net = None
    else:
        text_net = TextProj(in_dim, out_dim).to(DEV)
    # 温度固定为 baseline 同值：臂间唯一差异就是"投影头 / 对称损失 / 硬负"，
    # 避免可学习 logit-scale 跑到 clamp 边界造成训练震荡（smoke 中已观察到 align 转负）。
    params = list(spec.parameters()) + (list(text_net.parameters()) if text_net else [])
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)

    Xt = torch.tensor(Xs, dtype=torch.float32, device=DEV)
    yt = torch.tensor(ys, device=DEV)
    Traw = torch.tensor(T_seen, dtype=torch.float32, device=DEV)
    LW = torch.tensor(logW_seen, dtype=torch.float32, device=DEV)
    gen = torch.Generator().manual_seed(seed)
    n = len(Xt)

    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            z = F.normalize(spec(Xt[idx]), dim=1)                     # (B,d)
            if text_net is None:
                t = F.normalize(Traw, dim=1)
            else:
                t = F.normalize(text_net(Traw), dim=1)                # (C,d)
            logits = z @ t.T / TEMP_INIT                               # (B,C)

            if arm in ("baseline", "abtt", "ce_proj"):
                loss = F.cross_entropy(logits, yt[idx])
            else:
                # 实例级 spec→text（硬负加权）
                loss_i2t = F.cross_entropy(logits + LW[yt[idx]], yt[idx])
                # 类级对称 InfoNCE：batch 内每类取谱质心，与文本双向对齐
                uniq, inv = torch.unique(yt[idx], return_inverse=True)
                cnt = torch.zeros(len(uniq), device=DEV).index_add_(
                    0, inv, torch.ones_like(inv).float())
                M = torch.zeros(len(uniq), z.shape[1], device=DEV).index_add_(0, inv, z)
                zc = F.normalize(M / cnt.clamp(min=1).unsqueeze(1), dim=1)
                t_u = t[uniq]
                clog = (zc @ t_u.T) / TEMP_INIT                        # (C',C')
                tgt = torch.arange(len(uniq), device=DEV)
                LW_u = LW[uniq][:, uniq]
                loss_c2t = F.cross_entropy(clog + LW_u, tgt)
                loss_t2c = F.cross_entropy(clog.T + LW_u.T, tgt)
                loss = loss_i2t + 0.5 * (loss_c2t + loss_t2c)
            loss.backward()
            opt.step()

    def text_fn(T_np):
        if text_net is None:
            return torch.tensor(T_np, dtype=torch.float32)
        with torch.no_grad():
            return text_net(torch.tensor(T_np, dtype=torch.float32, device=DEV)).cpu()
    return spec, text_fn


@torch.no_grad()
def eval_fold(spec, text_fn, Xh, T_np, true_ix):
    z = F.normalize(spec(torch.tensor(Xh, dtype=torch.float32, device=DEV)), dim=1)
    t = F.normalize(text_fn(T_np).to(DEV), dim=1)
    sim = (z @ t.T).cpu().numpy()
    return float((sim.argmax(1) == true_ix).mean()), sim


@torch.no_grad()
def rep_quality(spec, text_fn, Xs, ys, T_full_np, ix_seen):
    """返回 (alignment, uniformity, 投影后 mean-cos, PC1)。ix_seen: 本折训练用到的类索引。"""
    t_all = F.normalize(text_fn(T_full_np).to(DEV), dim=1)            # (19,d)
    Tu = t_all.cpu().numpy()
    m = metrics(Tu.astype(np.float64), "proj")
    z = F.normalize(spec(torch.tensor(Xs, dtype=torch.float32, device=DEV)), dim=1)
    cls = torch.tensor(ys, device=DEV)
    C = len(ix_seen)
    cnt = torch.zeros(C, device=DEV).index_add_(0, cls, torch.ones_like(cls).float())
    M = torch.zeros(C, z.shape[1], device=DEV).index_add_(0, cls, z)
    zc = F.normalize(M / cnt.clamp(min=1).unsqueeze(1), dim=1)
    t_seen = t_all[list(ix_seen)]                                     # 与 zc 逐行对应
    align = float(torch.diagonal(zc @ t_seen.T).mean().cpu())
    return align, m["uniform"], m["cos_mean"], m["pc1"]


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2, help="种子数（默认 2=快速版；8=完整版）")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--folds", type=int, default=0, help=">0 时只跑前 N 折（smoke）")
    ap.add_argument("--tag", default="", help="报告后缀")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    arms = args.arms.split(",")
    tag = args.tag or f"s{len(seeds)}"

    ent, bg, grid = G1.load_g1()
    eids = sorted(ent.keys())
    mids = sorted(bg.keys())
    keys = eids + [f"M_{m}" for m in mids]
    print(f"[exp11] device={DEV} | {len(eids)} entities + {len(mids)} matrices | "
          f"seeds={seeds} | arms={arms}", flush=True)

    V = E8.build_text_variants_g1(eids, mids)
    T_full = np.stack([V["Original"][k] for k in keys]).astype(np.float32)   # (19,384)
    T_full = T_full / np.linalg.norm(T_full, axis=1, keepdims=True)
    S_full = T_full @ T_full.T
    LOGW_FULL, TAU = build_logw(S_full)
    ZERO = np.zeros_like(LOGW_FULL)
    n_hard = int((LOGW_FULL > 1e-6).sum() // 2)
    print(f"[exp11] 冻结 CLM 空间：mean-cos={metrics(T_full.astype(float),'raw')['cos_mean']:.4f} "
          f"PC1={metrics(T_full.astype(float),'raw')['pc1']:.3%} | "
          f"硬负 τ=q{int(TAU_DEG_Q*100)}={TAU:.4f}，共 {n_hard} 对 "
          f"(max log-w={LOGW_FULL.max():.3f})", flush=True)

    # 零训练等距化（ABTT）：对 held-out 文本同样生效，作为"学习式"臂的对照
    T_abtt, K_ABTT = abtt_transform(T_full)
    m_ab = metrics(T_abtt.astype(np.float64), "abtt")
    print(f"[exp11] ABTT(k={K_ABTT})：mean-cos {metrics(T_full.astype(float),'raw')['cos_mean']:.4f}"
          f" → {m_ab['cos_mean']:.4f}，uniformity {metrics(T_full.astype(float),'raw')['uniform']:.3f}"
          f" → {m_ab['uniform']:.3f}（零训练，语料 {len(CORPUS)} 个分子，无标签无谱）", flush=True)

    results = {a: {} for a in arms}          # {arm: {(eid,seed): top1}}
    reps = {a: {} for a in arms}             # {arm: {eid: (align, uniform, cos_mean, pc1)}}
    simstore = {a: {} for a in arms}
    pair_cos = {a: {} for a in arms}         # {arm: {eid: {(p,q): cos}}}
    folds = eids if args.folds <= 0 else eids[:args.folds]

    for arm in arms:
        for fold, e_hold in enumerate(folds):
            eids_tr = [e for e in eids if e != e_hold]
            tr_ix = {e: i for i, e in enumerate(eids_tr)}
            parts = [ent[e] for e in eids_tr] + [bg[m] for m in mids]
            ys = np.concatenate([np.full(len(ent[e]), tr_ix[e]) for e in eids_tr] +
                                [np.full(len(bg[m]), len(eids_tr) + j) for j, m in enumerate(mids)])
            Xs = np.concatenate(parts)
            ix_seen = [keys.index(e) for e in eids_tr] + [keys.index(f"M_{m}") for m in mids]
            SRC = T_abtt if arm == "abtt" else T_full
            T_seen = SRC[ix_seen]
            LW_seen = (LOGW_FULL if arm == "clip_da" else ZERO)[np.ix_(ix_seen, ix_seen)]
            Xh = ent[e_hold]
            true_ix = eids.index(e_hold)
            accs, sims, rq = [], [], []
            for seed in seeds:
                spec, text_fn = train_joint(Xs, ys, T_seen, LW_seen, arm, args.epochs,
                                            seed * 1000 + fold)
                acc, sim = eval_fold(spec, text_fn, Xh, SRC[:len(eids)], true_ix)
                accs.append(acc); sims.append(sim)
                rq.append(rep_quality(spec, text_fn, Xs, ys, SRC, ix_seen))
                del spec
                if DEV == "cuda":
                    torch.cuda.empty_cache()
            for s, a in zip(seeds, accs):
                results[arm][(e_hold, s)] = a
            simstore[arm][e_hold] = np.mean(sims, axis=0)
            reps[arm][e_hold] = tuple(np.mean([r[i] for r in rq]) for i in range(4))
            with torch.no_grad():
                Tp = F.normalize(text_fn(torch.tensor(SRC[:len(eids)], dtype=torch.float32)).to(DEV),
                                 dim=1).cpu().numpy()
            pair_cos[arm][e_hold] = {(p[0], p[1]): float(Tp[eids.index(p[0])] @ Tp[eids.index(p[1])])
                                     for p in SIBLING_PAIRS}
            print(f"[{arm}] {e_hold}: top1={np.mean(accs):.3f} "
                  f"align={reps[arm][e_hold][0]:.3f} unif={reps[arm][e_hold][1]:.3f} "
                  f"(n_q={len(Xh)})", flush=True)

    if len(folds) == len(eids):
        np.save(os.path.join(OUT, f"exp11_sims_{tag}.npy"), simstore, allow_pickle=True)
        write_report(eids, keys, arms, results, reps, simstore, T_full, seeds, tag,
                     T_abtt, K_ABTT, pair_cos)
    else:
        print(f"[smoke] 只跑 {len(folds)} 折，跳过报告", flush=True)


# ----------------------------------------------------------------------------
def write_report(eids, keys, arms, results, reps, simstore, T_full, seeds, tag,
                 T_abtt=None, K_ABTT=0, pair_cos=None):
    chance = 1.0 / len(eids)
    fold_top1 = {a: {e: np.mean([results[a][(e, s)] for s in seeds]) for e in eids} for a in arms}
    mean_top1 = {a: float(np.mean(list(fold_top1[a].values()))) for a in arms}

    # 逐折配对置换检验：各臂 vs baseline
    p_raw, verdict = {}, {}
    for a in arms:
        if a == "baseline":
            continue
        o = [fold_top1[a][e] for e in eids]
        c = [fold_top1["baseline"][e] for e in eids]
        p_raw[a], _ = E5.perm_paired_greater(o, c)
        verdict[a] = p_raw[a] <= 0.05

    # 表示质量（各臂逐折均值）
    rep_mean = {a: tuple(np.mean([reps[a][e][i] for e in eids]) for i in range(4)) for a in arms}
    m0 = metrics(T_full.astype(np.float64), "raw")

    lines = [f"# exp11 Degeneracy-aware contrastive alignment（{len(seeds)} seeds）", "",
             "> 同源多臂消融：同一 LOAO 13 折、同一 seeds、同一数据与编码器骨架。",
             "> 文本侧 = 冻结 ChemBERTa 组分谱嵌入 → **可学习投影头**（或零训练 ABTT）；谱侧 = 1D-CNN。",
             "> 硬负权重由**冻结 CLM 空间的类间余弦**决定（训练前可算，未使用任何谱/标签/测试信息）。",
             "> 温度固定 τ=0.07，臂间唯一差异＝文本变换方式 / 损失形式。", "",
             "## 1. 主表（LOAO top-1）", "",
             "| 臂 | 损失 | 投影头 | top-1 均值 | vs baseline 配对置换 p |", "|---|---|---|---|---|"]
    desc = {"baseline": "cosine-CE（=exp08 Original）",
            "abtt": "cosine-CE（ABTT 文本，零训练）", "ce_proj": "cosine-CE",
            "clip": "对称 InfoNCE", "clip_da": "对称 InfoNCE + 简并硬负加权"}
    proj = {"baseline": "—", "abtt": "—（k=%d 主成分扣除）" % K_ABTT,
            "ce_proj": "✅", "clip": "✅", "clip_da": "✅"}
    for a in arms:
        p = "—" if a == "baseline" else f"{p_raw[a]:.4f}{'✅' if verdict[a] else ''}"
        lines.append(f"| **{a}** | {desc[a]} | {proj[a]} | {mean_top1[a]:.3f} | {p} |")

    lines += ["", f"chance = {chance:.3f}；配对检验为单侧符号翻转精确置换（n=13 折）。", "",
              "## 2. 表示质量（Wang & Isola ICML'20 口径）", "",
              "| 臂 | alignment↑ | uniformity↓ | 文本 mean-cos↓ | PC1 解释方差↓ |", "|---|---|---|---|---|"]
    lines.append(f"| 冻结 CLM（无投影） | — | {m0['uniform']:.3f} | {m0['cos_mean']:.3f} | {m0['pc1']:.1%} |")
    for a in arms:
        al, un, cm, pc = rep_mean[a]
        lines.append(f"| {a} | {al:.3f} | {un:.3f} | {cm:.3f} | {pc:.1%} |")

    lines += ["", "## 3. 简并对在**各臂文本空间**的实际余弦（核心证据）", "",
              "| 对 | 冻结 CLM | " + " | ".join(a for a in arms if a != "baseline") + " |",
              "|---|" + "---|" * len(arms)]
    for p in SIBLING_PAIRS:
        i, j = keys.index(p[0]), keys.index(p[1])
        c0 = float(T_full[i] @ T_full[j])
        cab = "—" if T_abtt is None else f"{float(T_abtt[i] @ T_abtt[j]):.4f}"
        row = f"| {p[0]}–{p[1]} | {c0:.4f} | "
        if "abtt" in arms:
            row += " | ".join(
                (cab if a == "abtt"
                 else (f"{np.mean([pair_cos[a][e][p] for e in pair_cos[a]]):.4f}"
                       if pair_cos and a in pair_cos and pair_cos[a] else "—"))
                for a in arms if a != "baseline") + " |"
        else:
            row += " | ".join(
                (f"{np.mean([pair_cos[a][e][p] for e in pair_cos[a]]):.4f}"
                 if pair_cos and a in pair_cos and pair_cos[a] else "—")
                for a in arms if a != "baseline") + " |"
        lines.append(row)
    lines.append("（学习式臂为每个 fold 训练后文本空间的值，取 13 折均值；"
                 "ABTT 为确定性零训练变换，对所有折相同。）")

    lines += ["", "## 4. 逐折明细（top-1，seed 均值）", "",
              "| fold | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for e in eids:
        lines.append(f"| {e} | " + " | ".join(f"{fold_top1[a][e]:.3f}" for a in arms) + " |")

    best = max(arms, key=lambda a: mean_top1[a])
    lines += ["", "## 5. 结论", "",
              f"- 最佳臂：**{best}**（top-1 = {mean_top1[best]:.3f}，baseline = {mean_top1['baseline']:.3f}）"]
    if best != "baseline":
        gain = mean_top1[best] - mean_top1["baseline"]
        lines.append(f"- 相对 baseline 提升 {gain:+.3f}；配对置换 p={p_raw[best]:.4f} "
                     f"（{'显著' if verdict[best] else '不显著'}）")
    lines.append("- 若 `clip_da` > `clip` > `ce_proj` > `baseline` 单调成立，则增益可分解为"
                 "「投影头（等距化）」+「对称对比」+「简并硬负」三部分，可作为论文第二创新点的完整消融链")

    md = os.path.join(OUT, f"exp11_p6_contrastive{'_' + tag if tag else ''}.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(OUT, f"exp11_p6_contrastive{'_' + tag if tag else ''}.csv"),
              "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "eid", "seed", "top1"])
        for a in arms:
            for (e, s), v in results[a].items():
                w.writerow([a, e, s, v])
    print("\n".join(lines), flush=True)
    print(f"[written] {md}", flush=True)


if __name__ == "__main__":
    main()
