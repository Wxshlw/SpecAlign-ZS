"""exp05 —— 论文6 Step 0｜G0 实体打乱探针（全篇生死门，v9 §六收紧判据）。

五档变体（判据单向收紧，2026-10-20）：
  1. Original          组分谱表示（成分 SMILES 的 CLM 嵌入加权平均）——本文主张
  2. Shuffle-Random    随机字符串替换 SMILES         → Original 须显著更优
  3. Shuffle-Mismatch  实体↔组分谱 整体错位指派      → Original 须显著更优
  4. Random-Encoder    冻结随机初始化同构 CLM         → Original 须显著更优（否则=随机平滑插值）
  5. Descriptor        ECFP4 指纹加权平均（不经 CLM） → Original 须显著更优（否则 CLM 无增量）

探针任务：leave-one-entity-out（LOEO）零样本检索。
  - 6 个 seen 实体训练谱编码器（MLP，余弦交叉熵）对齐到文本嵌入；
  - 留出实体：全部谱 → 在 7 个候选文本嵌入中检索，top-1 / top-3 / MRR；
  - 留出实体的文本嵌入由冻结 CLM 计算（训练中不可见）——度量语义可迁移性。

统计：逐 fold 配对（n=7），单侧符号翻转置换检验，Holm 校正 4 组比较；
分层报告：简并组（A01/A05 亚油酸对、A04/A06 麦芽糖对）vs 干净组（A02/A03/A07）。

数据口径：仅公共 7 集（作者 2026-10-30 授权，自采三套已排除）。
判据：Original 显著优于全部 4 对照（Holm 后 p<0.05）→ PASS。
"""
from __future__ import annotations
import os, sys, csv, json, random, warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, r"D:/2026sci论文实验设计/论文6/scripts")
import p6_data as P
import p6_library as L

OUT = r"D:/2026sci论文实验设计/论文6/out"
SEED = 0
GRID_N = 1200
TEMP = 0.07
EPOCHS = 400
VARIANTS = ["Original", "Shuffle-Random", "Shuffle-Mismatch", "Random-Encoder", "Descriptor"]
# ChemBERTa 本地副本（hf-mirror 手动下载，config 与权重自洽：3 层 / 384 维，"77M"=预训练分子数）
CLM_DIR = r"D:/2026sci论文实验设计/论文6/models/chemberta"

rng = random.Random(SEED)
np.random.seed(SEED)


# ----------------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------------
def load_matrix_data():
    """公共 7 集 → 统一网格 + SNV+SG 预处理。返回 ({sid: dict}, grid)。"""
    from scipy.signal import savgol_filter
    sets = P.load_all(verbose=False)
    pub = {sid: D for sid, D in sets.items() if sid not in {"S1", "S2", "S3"}}
    lo = max(D["axis"].min() for D in pub.values())
    hi = min(D["axis"].max() for D in pub.values())
    grid = np.linspace(lo + 0.5, hi - 0.5, GRID_N)
    out = {}
    for sid, D in pub.items():
        axis = np.asarray(D["axis"], float)
        X = np.asarray(D["X"], float)
        Xg = np.stack([np.interp(grid, axis, row) for row in X])
        mu = Xg.mean(1, keepdims=True); sd = Xg.std(1, keepdims=True)
        Xz = (Xg - mu) / (sd + 1e-9)
        Xz = savgol_filter(Xz, 15, 3, deriv=1, axis=1)
        out[sid] = dict(X=Xz.astype(np.float32), y=np.asarray(D["y"], float))
    return out, grid


SID2EID = {"D7w": "A01", "D7p": "A02", "D7d": "A03", "D8": "A04", "D1": "A05", "D2": "A06", "D3": "A07"}


# ----------------------------------------------------------------------------
# 文本侧（5 变体）
# ----------------------------------------------------------------------------
def _load_tokenizer():
    """ChemBERTa = RoBERTa BPE，本地目录加载（读 vocab/merges/tokenizer_config/special_tokens_map）。"""
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained(CLM_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _load_clm(random_init: bool, seed: int = SEED):
    """返回 (tokenizer, RobertaModel, 已加载预训练权重?)。随机初始化臂用同 config + 固定种子。"""
    import torch
    from transformers import RobertaConfig, RobertaModel
    cfg = RobertaConfig.from_json_file(os.path.join(CLM_DIR, "config.json"))
    if random_init:
        torch.manual_seed(seed)
        mdl = RobertaModel(cfg, add_pooling_layer=False)
    else:
        mdl = RobertaModel(cfg, add_pooling_layer=False)
        sd = torch.load(os.path.join(CLM_DIR, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        sd = {k[len("roberta."):]: v for k, v in sd.items()
              if k.startswith("roberta.") and not k.endswith("position_ids")}
        mdl.load_state_dict(sd, strict=True)
    mdl.eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    return _load_tokenizer(), mdl


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def build_text_variants(eids):
    """返回 {variant: {eid: np.ndarray}}。"""
    import torch
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    profiles = L.COMPONENT_PROFILES
    def comps(eid):
        return [(s, w) for _, s, w in profiles[eid]]

    all_smiles = sorted({s for e in eids for s, _ in comps(e)})

    # ---------- CLM（预训练，单实例） ----------
    tok, mdl = _load_clm(random_init=False)
    with torch.no_grad():
        enc_emb = {}
        B = 16
        for i in range(0, len(all_smiles), B):
            batch = all_smiles[i:i + B]
            inputs = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = mdl(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            for s, v in zip(batch, emb.numpy()):
                enc_emb[s] = v
    """del mdl 延后到 Shuffle-Random 之后（该块复用同一模型）"""

    def prof_embed(eid, lookup):
        cs = comps(eid)
        tw = sum(w for _, w in cs)
        return _unit(sum(lookup[s] * (w / tw) for s, w in cs))

    variants = {}
    variants["Original"] = {e: prof_embed(e, enc_emb) for e in eids}

    # Shuffle-Mismatch：7 元循环移位（derangement），整体指派 donor 的组分谱
    der = {eids[i]: eids[(i + 1) % len(eids)] for i in range(len(eids))}
    variants["Shuffle-Mismatch"] = {e: prof_embed(der[e], enc_emb) for e in eids}

    # Shuffle-Random：随机字符串（固定 seed），同 tokenizer 嵌入
    charset = "ACDEFGHIKLMNPQRSTVWY0123456789@#%&*+=[]()"
    rand_str = {e: "".join(rng.choice(charset) for _ in range(48)) for e in eids}
    emb_rand = {}
    B = 8
    rlist = [rand_str[e] for e in eids]
    with torch.no_grad():
        for i in range(0, len(rlist), B):
            batch = rlist[i:i + B]
            inputs = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = mdl(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            for e, v in zip(eids[i:i + B], emb.numpy()):
                emb_rand[e] = v
    del mdl
    variants["Shuffle-Random"] = {e: _unit(emb_rand[e]) for e in eids}

    # ---------- Random-Encoder：随机初始化同构 CLM（单实例、冻结、固定种子） ----------
    _, mdl_r = _load_clm(random_init=True)
    flat = [s for e in eids for s, _ in comps(e)]
    re_emb = {}
    with torch.no_grad():
        for i in range(0, len(flat), B):
            batch = flat[i:i + B]
            inputs = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = mdl_r(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            for s, v in zip(batch, emb.numpy()):
                re_emb[s] = v
    del mdl_r
    variants["Random-Encoder"] = {e: prof_embed(e, re_emb) for e in eids}

    # ---------- Descriptor：ECFP4 加权平均（不经 CLM） ----------
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp_of = {}
    for s in all_smiles:
        m = Chem.MolFromSmiles(s)
        fp_of[s] = np.asarray(gen.GetFingerprint(m), float)
    def prof_fp(eid):
        cs = comps(eid); tw = sum(w for _, w in cs)
        return _unit(sum(fp_of[s] * (w / tw) for s, w in cs))
    variants["Descriptor"] = {e: prof_fp(e) for e in eids}

    return variants


# ----------------------------------------------------------------------------
# 谱编码器（MLP）+ LOEO
# ----------------------------------------------------------------------------
def train_and_eval(X_by_eid, text_emb, eids, out_dim=None):
    """LOEO 主循环。返回 {eid: dict(top1,top3,mrr)}。out_dim 缺省取文本嵌入实际维度。"""
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    T_all = np.stack([text_emb[e] for e in eids]).astype(np.float32)   # (7,dim)
    out_dim = out_dim or T_all.shape[1]
    eid_ix = {e: i for i, e in enumerate(eids)}
    res = {}
    for fold, e_hold in enumerate(eids):
        eids_tr = [e for e in eids if e != e_hold]
        tr_ix = {e: i for i, e in enumerate(eids_tr)}          # seen 索引 0..5
        Xs = np.concatenate([X_by_eid[e] for e in eids_tr])
        ys = np.concatenate([[tr_ix[e]] * len(X_by_eid[e]) for e in eids_tr])
        Xh = X_by_eid[e_hold]
        Ttr = T_all[[eid_ix[e] for e in eids_tr]]                      # (6,dim)

        torch.manual_seed(SEED + fold)
        net = torch.nn.Sequential(
            torch.nn.Linear(Xs.shape[1], 256), torch.nn.GELU(),
            torch.nn.Linear(256, 256), torch.nn.GELU(),
            torch.nn.Linear(256, out_dim))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        Xt = torch.tensor(Xs); yt = torch.tensor(ys)
        Tt = torch.tensor(Ttr)
        Tt = torch.nn.functional.normalize(Tt, dim=1)
        for _ in range(EPOCHS):
            opt.zero_grad()
            z = torch.nn.functional.normalize(net(Xt), dim=1)
            logits = z @ Tt.T / TEMP
            loss = torch.nn.functional.cross_entropy(logits, yt)
            loss.backward(); opt.step()
        with torch.no_grad():
            zh = torch.nn.functional.normalize(net(torch.tensor(Xh)), dim=1)
            Ta = torch.nn.functional.normalize(torch.tensor(T_all), dim=1)
            sim = (zh @ Ta.T).numpy()                                   # (n_h,7)
        # 修正 bug：原 `(-sim).argsort(1).argsort(1)[:, 0]` 计算的是第 0 类（A01）的排名，
        # 而非真实类排名。正确做法：预测列 = argmax，与真实类列比对。
        pred = sim.argmax(1)
        top1 = float((pred == eid_ix[e_hold]).mean())
        order = (-sim).argsort(1)
        rank_true = np.array([np.where(order[q] == eid_ix[e_hold])[0][0] for q in range(len(Xh))]) + 1
        top3 = float((rank_true <= 3).mean())
        mrr = float((1.0 / rank_true).mean())
        res[e_hold] = dict(top1=top1, top3=top3, mrr=mrr, n=len(Xh))
        print(f"  fold {e_hold}: top1={top1:.3f} top3={top3:.3f} mrr={mrr:.3f} (n={len(Xh)})", flush=True)
    return res


# ----------------------------------------------------------------------------
# 统计
# ----------------------------------------------------------------------------
def perm_paired_greater(orig, ctrl):
    """单侧符号翻转置换：H1 = mean(orig-ctrl) > 0。返回精确 p。"""
    d = np.asarray(orig, float) - np.asarray(ctrl, float)
    obs = d.mean()
    n = len(d)
    best = 0
    cnt = 0
    for mask in range(1 << n):
        s = sum(d[i] * (1 if (mask >> i) & 1 else -1) for i in range(n))
        if s / n >= obs - 1e-12:
            cnt += 1
    return cnt / (1 << n), obs


def holm(pvals):
    """Holm-Bonferroni。返回校正后 p 列表（与输入同序）。"""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, pvals[i] * (m - rank)))
        adj[i] = running
    return adj


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    eids = sorted(L.ACTIVE_EIDS)
    print(f"[G0] active entities: {eids}", flush=True)
    data, grid = load_matrix_data()
    X_by_eid = {SID2EID[sid]: D["X"] for sid, D in data.items() if sid in SID2EID}
    assert set(X_by_eid) == set(eids)

    print("[G0] building text variants (ChemBERTa via hf-mirror) ...", flush=True)
    variants = build_text_variants(eids)

    results = {}
    for v in VARIANTS:
        print(f"[G0] variant = {v}", flush=True)
        results[v] = train_and_eval(X_by_eid, variants[v], eids)

    # ---- 汇总 ----
    folds = eids
    top1 = {v: [results[v][e]["top1"] for e in folds] for v in VARIANTS}
    chance = 1.0 / len(eids)

    p_raw, diffs = {}, {}
    for v in VARIANTS[1:]:
        p_raw[v], diffs[v] = perm_paired_greater(top1["Original"], top1[v])
    p_adj = holm([p_raw[v] for v in VARIANTS[1:]])
    verdict = {v: p <= 0.05 for v, p in zip(VARIANTS[1:], p_adj)}

    # 分层：简并组 vs 干净组
    DEGEN = {"A01", "A05", "A04", "A06"}
    CLEAN = {"A02", "A03", "A07"}
    layer = {}
    for grp, members in [("degenerate", DEGEN), ("clean", CLEAN)]:
        layer[grp] = {v: float(np.mean([results[v][e]["top1"] for e in members if e in folds]))
                      for v in VARIANTS}

    # ---- 报告 ----
    lines = ["# exp05 G0 实体打乱探针（v9 §六 收紧判据，五档）", "",
             f"> 数据口径：仅公共 7 集（作者 2026-10-30 授权）；LOEO 零样本检索；候选数 = 7；随机水平 = 1/7 ≈ {chance:.3f}",
             "> 预处理：统一网格 [600, 3996] cm⁻¹ / 1200 点 + SNV + SG 一阶导。", "",
             "## 主表（LOEO top-1 / top-3 / MRR）", "",
             "| 变体 | top-1 | top-3 | MRR | vs Original Holm-p | 判定 |", "|---|---|---|---|---|---|"]
    mean = lambda v, k: float(np.mean([results[v][e][k] for e in folds]))
    lines.append(f"| **Original（组分谱）** | {mean('Original','top1'):.3f} | {mean('Original','top3'):.3f} | "
                 f"{mean('Original','mrr'):.3f} | — | — |")
    for i, v in enumerate(VARIANTS[1:]):
        lines.append(f"| {v} | {mean(v,'top1'):.3f} | {mean(v,'top3'):.3f} | {mean(v,'mrr'):.3f} | "
                     f"{p_adj[i]:.4f} (raw {p_raw[v]:.4f}) | {'✅ Original 显著更优' if verdict[v] else '❌ 不显著'} |")
    lines += ["", "## 逐 fold 明细（top-1）", "", "| fold | " + " | ".join(VARIANTS) + " |",
              "|---|" + "---|" * len(VARIANTS)]
    for e in folds:
        lines.append(f"| {e} | " + " | ".join(f"{results[v][e]['top1']:.3f}" for v in VARIANTS) + " |")
    lines += ["", "## 分层（简并组 vs 干净组，top-1 均值）", "",
              "| 层 | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS),
              f"| 简并组（A01/A05 亚油酸对、A04/A06 麦芽糖对） | "
              + " | ".join(f"{layer['degenerate'][v]:.3f}" for v in VARIANTS) + " |",
              f"| 干净组（A02/A03/A07） | "
              + " | ".join(f"{layer['clean'][v]:.3f}" for v in VARIANTS) + " |"]

    # ---- 判定（v9 退出路径） ----
    n_pass = sum(verdict.values())
    lines += ["", "## G0 判定（v9 单向收紧判据）", ""]
    if n_pass == 4:
        lines.append("- ✅ **G0 PASS**：Original 显著优于全部 4 对照（Holm 后 p<0.05）→ 化学语义贡献成立，进 G1。")
    elif verdict.get("Shuffle-Random") and verdict.get("Shuffle-Mismatch") and not (
            verdict.get("Random-Encoder") and verdict.get("Descriptor")):
        lines.append("- 🟡 **CLM 先验无增量**：打乱对照过但 Random-Encoder/Descriptor 未拉开 → "
                     "主结论改写为「结构描述的参数化（而非 CLM 本身）决定 zero-shot 上限」，claim 下移仍可发。")
    else:
        lines.append("- 🔴 **文本侧零语义贡献**：连打乱对照都未拉开 → 走 v9 退出路径（撤销编号/并入论文5/改投属性条件化二区）。")
    lines.append(f"- 分层注意：简并组（{sorted(DEGEN)}）与干净组（{sorted(CLEAN)}）分别报告；"
                 "简并组 top-1 塌陷属『不可辨识性』而非语义缺失，不影响 G0 主判定（判据分层原则）。")
    lines.append("- ⚠️ n=7 fold 的置换检验为探针级证据；正式显著性 Machinery 在 G1 候选集规模扫描下重建。")

    with open(os.path.join(OUT, "exp05_p6_g0.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with open(os.path.join(OUT, "exp05_p6_g0.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "eid", "n", "top1", "top3", "mrr"])
        for v in VARIANTS:
            for e in folds:
                r = results[v][e]
                w.writerow([v, e, r["n"], r["top1"], r["top3"], r["mrr"]])

    print("\n".join(lines), flush=True)
    print("[written] exp05_p6_g0.md / exp05_p6_g0.csv", flush=True)


if __name__ == "__main__":
    main()
