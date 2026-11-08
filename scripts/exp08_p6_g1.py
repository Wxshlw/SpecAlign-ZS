"""exp08 —— 论文6 Step 2｜G1 zero-shot 主门（v9 §六，前置①–④ 已满足）。

设计（exp06 裁定）：
  - 候选 = 13 实体（全公开真实谱）：A01–A07 掺混 + HSYRUP 蜂蜜糖浆 + 3 肉种 + 2 咖啡种
  - LOAO：每折留出一个实体；训练 = 其余 12 实体 query 谱 + 6 矩阵背景谱
    （含 held-out 实体所在矩阵的纯谱 → 消除 exp05 的跨基质 OOD 塌陷，前置④）
  - 编码器升级（前置②）：1D-CNN（4 卷积块 + GAP），cosine-CE 对齐
  - 五档 shuffle 套件（前置③，判据不放宽）
  - 候选规模扫描 N ∈ {2,3,5,10,13}（sim 矩阵 post-hoc 子集扫描）

判据（v9，单向收紧）：
  G1a：Original top-1 显著高于随机（1/13≈0.077）与全部对照（逐折配对置换 + Holm）
  G1b/H2：实体 top-1 与候选简并度 Spearman 负相关
"""
from __future__ import annotations
import os, sys, csv, random, warnings
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1
import p6_library as L
import exp05_p6_g0 as E5

OUT = r"D:/2026sci论文实验设计/论文6/out"
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]   # G1v2：2 → 8（降低配对检验方差）
EPOCHS = 60
TEMP = 0.07
SCAN_NS = [2, 3, 5, 10, 13]
R_SUB = 30

rng = random.Random(0)
np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[exp08] device = {DEV}", flush=True)


# ----------------------------------------------------------------------------
# 文本变体（13 实体 + 6 矩阵背景同构生成）
# ----------------------------------------------------------------------------
def build_text_variants_g1(eids, mids):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    tok, mdl = E5._load_clm(random_init=False)

    def prof_of(key):
        if key.startswith("M_"):
            return [(s, w) for _, s, w in G1.PROFILES_G1[key]]
        return [(s, w) for _, s, w in (L.COMPONENT_PROFILES.get(key) or G1.PROFILES_G1[key])]

    keys = eids + [f"M_{m}" for m in mids]
    all_smi = sorted({s for k in keys for s, _ in prof_of(k)})

    def embed_batch(strs, model):
        outs = {}
        with torch.no_grad():
            B = 16
            uniq = sorted(set(strs))
            for i in range(0, len(uniq), B):
                batch = uniq[i:i + B]
                inputs = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
                h = model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
                for s, vv in zip(batch, emb.numpy()):
                    outs[s] = vv
        return outs

    def prof_embed(k, lookup):
        cs = prof_of(k); tw = sum(w for _, w in cs)
        return E5._unit(sum(lookup[s] * (w / tw) for s, w in cs))

    variants = {}
    emb_pre = embed_batch(all_smi, mdl)
    variants["Original"] = {k: prof_embed(k, emb_pre) for k in keys}

    der = {eids[i]: eids[(i + 1) % len(eids)] for i in range(len(eids))}   # 仅实体错位
    variants["Shuffle-Mismatch"] = {k: prof_embed(der.get(k, k), emb_pre) for k in keys}

    charset = "ACDEFGHIKLMNPQRSTVWY0123456789@#%&*+=[]()"
    rstr = {k: "".join(rng.choice(charset) for _ in range(48)) for k in keys}
    emb_r = embed_batch(list(rstr.values()), mdl)
    variants["Shuffle-Random"] = {k: E5._unit(emb_r[s]) for k, s in rstr.items()}

    _, mdl_r = E5._load_clm(random_init=True)
    emb_re = embed_batch(all_smi, mdl_r)
    variants["Random-Encoder"] = {k: prof_embed(k, emb_re) for k in keys}

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp_of = {s: np.asarray(gen.GetFingerprint(Chem.MolFromSmiles(s)), float) for s in all_smi}
    def prof_fp(k):
        cs = prof_of(k); tw = sum(w for _, w in cs)
        return E5._unit(sum(fp_of[s] * (w / tw) for s, w in cs))
    variants["Descriptor"] = {k: prof_fp(k) for k in keys}
    return variants


# ----------------------------------------------------------------------------
# 1D-CNN 编码器
# ----------------------------------------------------------------------------
class SpecCNN(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.GELU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(),
            nn.Linear(512, out_dim))

    def forward(self, x):
        return self.net(x.unsqueeze(1))


def train_cnn(Xs, ys, Ttr, epochs, seed):
    """minibatch 训练（batch 256 × 60 epoch）；GPU 可用时自动用 CUDA。"""
    torch.manual_seed(seed)
    net = SpecCNN(Ttr.shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xs, dtype=torch.float32, device=DEV); yt = torch.tensor(ys, device=DEV)
    Tt = nn.functional.normalize(torch.tensor(Ttr, dtype=torch.float32, device=DEV), dim=1)
    gen = torch.Generator().manual_seed(seed)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            z = nn.functional.normalize(net(Xt[idx]), dim=1)
            loss = nn.functional.cross_entropy(z @ Tt.T / TEMP, yt[idx])
            loss.backward(); opt.step()
    return net


@torch.no_grad()
def embed(net, X):
    return nn.functional.normalize(net(torch.tensor(X, dtype=torch.float32, device=DEV)), dim=1).cpu().numpy()


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    ent, bg, grid = G1.load_g1()
    eids = sorted(ent.keys())
    mids = sorted(bg.keys())
    nE = len(eids)
    print(f"[G1] {nE} candidates: {eids}", flush=True)
    print(f"[G1] {len(mids)} matrix backgrounds: {mids}", flush=True)
    variants = build_text_variants_g1(eids, mids)

    results = {}     # {variant: {(eid,seed): top1}}
    simstore = {}    # {variant: {eid: sim(n_query,13)}}（seed 平均）
    for v in E5.VARIANTS:
        T_all = np.stack([variants[v][e] for e in eids]).astype(np.float32)     # (13,dim)
        T_mat = np.stack([variants[v][f"M_{m}"] for m in mids]).astype(np.float32)  # (6,dim)
        results[v] = {}
        simstore[v] = {}
        for fold, e_hold in enumerate(eids):
            eids_tr = [e for e in eids if e != e_hold]
            tr_ix = {e: i for i, e in enumerate(eids_tr)}
            parts = [ent[e] for e in eids_tr] + [bg[m] for m in mids]
            ys = ([np.full(len(ent[e]), tr_ix[e]) for e in eids_tr] +
                  [np.full(len(bg[m]), len(eids_tr) + j) for j, m in enumerate(mids)])
            Xs = np.concatenate(parts)
            ys = np.concatenate(ys)
            Ttr = np.concatenate([T_all[[eids.index(e) for e in eids_tr]], T_mat])
            Xh = ent[e_hold]
            sims = []
            for seed in SEEDS:
                net = train_cnn(Xs, ys, Ttr, EPOCHS, seed * 1000 + fold)
                sim = embed(net, Xh) @ T_all.T           # (n_query, 13)
                sims.append(sim)
                pred = sim.argmax(1)                     # 修正：原 argsort(1).argsort(1)[:,0] 算的是第 0 类排名
                results[v][(e_hold, seed)] = float((pred == eids.index(e_hold)).mean())
            simstore[v][e_hold] = np.mean(sims, axis=0)
            print(f"[{v}] {e_hold}: top1={np.mean([results[v][(e_hold,s)] for s in SEEDS]):.3f} "
                  f"(n_q={len(Xh)})", flush=True)

    np.save(os.path.join(OUT, "exp08_sims.npy"), simstore, allow_pickle=True)
    write_report(eids, ent, variants, results, simstore)


# ----------------------------------------------------------------------------
# 统计与报告
# ----------------------------------------------------------------------------
def subset_scan(sim_by_fold, eids, N, R=R_SUB, seed=0):
    """规模扫描：每折抽 R 个含真值的 N 候选子集，top-1 均值。"""
    r = np.random.default_rng(seed)
    accs = []
    for e_hold, sim in sim_by_fold.items():
        ix_true = eids.index(e_hold)
        for _ in range(R):
            others = [i for i in range(len(eids)) if i != ix_true]
            sub = list(r.choice(others, N - 1, replace=False)) + [ix_true]
            pred = sim[:, sub].argmax(1)
            accs.append(float((np.array(sub)[pred] == ix_true).mean()))
    return float(np.mean(accs))


def write_report(eids, ent, variants, results, simstore):
    pairs = [(e, s) for e in eids for s in SEEDS]
    chance = 1.0 / len(eids)
    fold_top1 = {v: {e: np.mean([results[v][(e, s)] for s in SEEDS]) for e in eids} for v in E5.VARIANTS}
    mean_top1 = {v: float(np.mean(list(fold_top1[v].values()))) for v in E5.VARIANTS}

    p_raw, obs = {}, {}
    for v in E5.VARIANTS[1:]:
        o = [fold_top1["Original"][e] for e in eids]
        c = [fold_top1[v][e] for e in eids]
        p_raw[v], obs[v] = E5.perm_paired_greater(o, c)
    p_adj = E5.holm([p_raw[v] for v in E5.VARIANTS[1:]])
    verdict = {v: p <= 0.01 for v, p in zip(E5.VARIANTS[1:], p_adj)}   # G1a: p<0.01

    # 规模扫描（Original）
    scan = {}
    for N in SCAN_NS:
        scan[N] = subset_scan(simstore["Original"], eids, N)

    # H2：简并度 vs top-1
    T0 = np.stack([variants["Original"][e] for e in eids])
    S = T0 @ T0.T
    np.fill_diagonal(S, -1)
    degen_clm = S.max(1)
    from scipy.stats import spearmanr
    top1_e = [fold_top1["Original"][e] for e in eids]
    rho_c, p_c = spearmanr(degen_clm, top1_e)

    lines = ["# exp08 G1 zero-shot 主门（13 实体，全公开真实数据）", "",
             f"> LOAO 13 折 × {len(SEEDS)} seeds；候选 N=13（chance={chance:.3f}）；",
             "> 编码器=1D-CNN；训练含 6 矩阵背景谱（消除跨基质 OOD）；公共窗 1005–1799 cm⁻¹。",
             "> **G1v2**：seeds 2→8；文本先验按文献种属/品种成分差异细化"
             "（鹅肌肽/肌肽、胡芦巴碱、硬脂酸等标志物），非拟合测试集。", "",
             "## 主表（LOAO top-1）", "",
             "| 变体 | top-1 均值 | vs Original Holm-p (p<0.01) | 判定 |", "|---|---|---|---|"]
    lines.append(f"| **Original（组分谱+CNN）** | {mean_top1['Original']:.3f} | — | — |")
    for i, v in enumerate(E5.VARIANTS[1:]):
        lines.append(f"| {v} | {mean_top1[v]:.3f} | {p_adj[i]:.4f} (raw {p_raw[v]:.4f}) | "
                     f"{'✅ 显著低于 Original' if verdict[v] else '❌ 不显著'} |")
    lines += ["", "## G1a 判定", ""]
    g1a_pass = (mean_top1["Original"] > chance) and all(verdict.values())
    lines.append(f"- Original top-1 = {mean_top1['Original']:.3f} vs chance {chance:.3f} → "
                 f"{'高于随机' if mean_top1['Original'] > chance else '未高于随机'}；"
                 f"对照拉开：{sum(verdict.values())}/4 显著（Holm 后 p<0.01）")
    lines.append(f"- **G1a {'✅ PASS' if g1a_pass else '❌ FAIL'}**")

    lines += ["", "## 候选规模扫描（Original，含真值随机子集 ×30）", "",
              "| N | top-1 |", "|---|---|"]
    for N in SCAN_NS:
        lines.append(f"| {N} | {scan[N]:.3f} |")

    lines += ["", "## G1b/H2：简并度 vs zero-shot top-1", "",
              f"- CLM 文本空间简并度（max cos to others）与 top-1：Spearman ρ = {rho_c:.3f}（p={p_c:.4f}，n=13）",
              f"- 期望负相关（H2）；{'方向正确' if rho_c < 0 else '方向不符'}；"
              f"{'显著' if p_c < 0.05 else 'n=13 探索性证据'}"]

    lines += ["", "## 逐折明细（top-1，seed 均值）", "",
              "| fold | " + " | ".join(E5.VARIANTS) + " |", "|---|" + "---|" * len(E5.VARIANTS)]
    for e in eids:
        lines.append(f"| {e} | " + " | ".join(f"{fold_top1[v][e]:.3f}" for v in E5.VARIANTS) + " |")

    with open(os.path.join(OUT, "exp08_p6_g1.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(OUT, "exp08_p6_g1.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "eid", "seed", "top1"])
        for v in E5.VARIANTS:
            for (e, s), val in results[v].items():
                w.writerow([v, e, s, val])
    print("\n".join(lines), flush=True)
    print("[written] exp08_p6_g1.md / .csv / exp08_sims.npy", flush=True)


if __name__ == "__main__":
    main()
