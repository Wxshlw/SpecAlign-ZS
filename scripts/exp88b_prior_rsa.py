r"""exp88b —— 外部数据集：同类近邻候选上，8 个先验的结构保真度（RSA）。

为什么需要这个实验
------------------
exp87 的结论是：CLM 打不过 physchem / charbag，**不是**因为 CLM 差，
而是因为这批候选（糖/肉桂/姜黄/橄榄油/咖啡/橙汁）之间**差异就是组成差异**，
physchem / charbag 这类组成统计量天生占优，化学语义没有可兑现的对象。

要检验这个解释，必须换一批"组成相近、结构不同"的候选。
公开数据集 Mendeley 10.17632/ctgg7k4m5g（CC BY 4.0）提供 17 种食用植物油：
Almond / Canola / Corn / Grapeseed / Safflower / Sunflower / Walnut / Flaxseed …
它们都是甘油三酯，元素组成与分子量几乎相同，差别在**脂肪酸链长与不饱和度**。

指标：Representational Similarity Analysis（RSA）
    先验距离矩阵 D_prior (17×17)  vs  真实光谱距离矩阵 D_spec (17×17, 由纯油 ATR-FTIR 类均值算)
    报 Spearman ρ。ρ 越高 = 该先验越能还原"这批分子在光谱上有多像"。

这个指标**不用标签、不用训练、不受掺混浓度影响**，是对"先验保真度"最干净的测量。

另附：3 个掺杂物（Almond/Canola/Corn）的局部诊断——最难的同类近邻三角。
"""
from __future__ import annotations
import os, sys, json, itertools, warnings
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "..", "out")
CLM_DIR = os.path.join(HERE, "..", "models", "chemberta")

from exp81a_prior_zoo import mol_vectors, unit, hash_fold  # noqa: E402

# ---------------------------------------------------------------------------
# 脂肪酸组成（g/100 g total fatty acids）。来源：USDA FoodData Central 典型值，
# 同一油种跨文献变动 ±2-3 g；此处取中位典型值。
#   列 = C16:0, C18:0, C18:1, C18:2, C18:3
# ---------------------------------------------------------------------------
FA_SMILES = {
    "C16:0": "CCCCCCCCCCCCCCCC(=O)O",
    "C18:0": "CCCCCCCCCCCCCCCCCC(=O)O",
    "C18:1": r"CCCCCCCC\C=C/CCCCCCCC(=O)O",
    "C18:2": r"CCCCC\C=C/C\C=C/CCCCCCCC(=O)O",
    "C18:3": r"CC\C=C/C\C=C/C\C=C/CCCCCCCC(=O)O",
    "C22:1": r"CCCCCCCCCC\C=C/CCCCCCCCC(=O)O",   # erucic，芥花油特征
}
FA_ORDER = ["C16:0", "C18:0", "C18:1", "C18:2", "C18:3"]

# oil name -> (C16:0, C18:0, C18:1, C18:2, C18:3)
OIL_FA = {
    "olive":     (11.0, 2.0, 75.0, 8.0, 0.8),
    "avocado":   (15.0, 2.0, 63.0, 12.0, 0.8),
    "peanut":    (9.0, 3.0, 45.0, 32.0, 0.5),
    "corn":      (11.0, 2.0, 27.0, 55.0, 1.0),
    "grapeseed": (7.0, 3.0, 16.0, 71.0, 0.2),
    "safflower": (5.5, 2.5, 13.0, 77.0, 0.3),
    "hazelnut":  (5.0, 2.5, 80.0, 12.0, 0.2),
    "almond":    (6.2, 1.8, 69.0, 21.0, 0.1),
    "canola":    (3.5, 1.5, 61.0, 20.0, 9.0),
    "sesame":    (9.0, 5.5, 41.0, 43.0, 0.3),
    "sunflower": (5.5, 4.5, 20.0, 68.0, 0.2),
    "walnut":    (7.0, 2.0, 18.0, 58.0, 12.0),
    "flaxseed":  (5.0, 4.0, 20.0, 15.0, 55.0),
    "soybean":   (11.0, 4.0, 23.0, 53.0, 8.0),
    "vegoil":    (10.0, 2.0, 30.0, 50.0, 5.0),
}

# 纯油 Class（ATRPure3）-> 组份定义
#   单油 -> [(oil, 1.0)]；调和油 -> [(oil, frac), ...]
CLASS_DEF = {
    1:  [("olive", 1.0)],                      # Extra Virgin Olive
    2:  [("olive", 1.0)],                      # Extra Light Olive（精炼，脂肪酸谱近似）
    3:  [("olive", 1.0)],                      # Pure Olive
    4:  [("avocado", 1.0)],
    5:  [("peanut", 1.0)],
    6:  [("corn", 1.0)],
    7:  [("grapeseed", 1.0)],
    8:  [("safflower", 1.0)],
    9:  [("hazelnut", 1.0)],
    11: [("almond", 1.0)],
    12: [("canola", 1.0)],
    13: [("avocado", 1 / 3), ("flaxseed", 1 / 3), ("olive", 1 / 3)],
    14: [("sesame", 1.0)],
    15: [("canola", 0.5), ("vegoil", 0.5)],
    16: [("vegoil", 1.0)],
    17: [("canola", 1 / 3), ("sunflower", 1 / 3), ("soybean", 1 / 3)],
    18: [("sunflower", 1.0)],
    19: [("walnut", 1.0)],
}


def profile_of_class(c):
    """返回 [(SMILES, weight)]，权重 = 该类中所有油按混合比例摊到脂肪酸上的质量分数。"""
    acc = {k: 0.0 for k in FA_ORDER}
    for oil, frac in CLASS_DEF[c]:
        fa = OIL_FA[oil]
        for k, v in zip(FA_ORDER, fa):
            acc[k] += frac * v
    tot = sum(acc.values())
    return [(FA_SMILES[k], acc[k] / tot) for k in FA_ORDER if acc[k] / tot > 1e-6]


# ---------------------------------------------------------------------------
def clm_embed(smiles_list, batch=16):
    """冻结 ChemBERTa mean-pooling（与 exp05/exp34 完全同配方）。"""
    import torch
    from transformers import GPT2Tokenizer, RobertaConfig, RobertaModel
    tok = GPT2Tokenizer.from_pretrained(CLM_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = RobertaConfig.from_json_file(os.path.join(CLM_DIR, "config.json"))
    mdl = RobertaModel(cfg, add_pooling_layer=False)
    sd = torch.load(os.path.join(CLM_DIR, "pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    sd = {k[len("roberta."):]: v for k, v in sd.items()
          if k.startswith("roberta.") and not k.endswith("position_ids")}
    mdl.load_state_dict(sd, strict=True)
    mdl.eval()
    out = {}
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch):
            b = smiles_list[i:i + batch]
            inp = tok(b, padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = mdl(**inp).last_hidden_state
            m = inp["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1)
            for s, v in zip(b, e.numpy()):
                out[s] = v.astype(float)
    return out


def build_priors(classes):
    """{prior_name: (n_class, 384)}，与 exp81a 同一配方（加权 profile → 统一 384 维 → 行归一）。"""
    DIM = 384
    profs = {c: profile_of_class(c) for c in classes}
    all_smi = sorted({s for p in profs.values() for s, _ in p})
    print(f"[exp88b] {len(classes)} 类, {len(all_smi)} 个唯一脂肪酸 SMILES")

    raw = mol_vectors(all_smi)
    raw["clm"] = clm_embed(all_smi)

    def prof_native(name):
        out = []
        for c in classes:
            cs = profs[c]
            tw = sum(w for _, w in cs)
            out.append(sum(raw[name][s] * (w / tw) for s, w in cs))
        return np.stack(out)

    native = {n: prof_native(n) for n in raw}
    for n in ("rdkit", "physchem"):
        v = native[n]
        native[n] = (v - v.mean(0)) / (v.std(0) + 1e-9)

    rng = np.random.default_rng(0)
    proj = {n: rng.normal(size=(native[n].shape[1], DIM)) / np.sqrt(DIM)
            for n in ("rdkit", "physchem")}

    T = {}
    for n, V in native.items():
        if n == "ecfp384":
            T[n] = np.stack([unit(v) for v in V])
        elif n in ("rdkit", "physchem"):
            T[n] = np.stack([unit(v @ proj[n]) for v in V])
        else:
            T[n] = np.stack([unit(hash_fold(v)) for v in V])
    T["onehot"] = np.eye(len(classes), DIM)
    T["rand384"] = np.stack([unit(v) for v in rng.normal(size=(len(classes), DIM))])
    return T, native


def dist_mat(V, metric="cos"):
    if metric == "cos":
        C = V @ V.T
        D = 1.0 - C
    else:
        D = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=2)
    np.fill_diagonal(D, 0.0)
    return D


def spearman(a, b):
    from scipy.stats import spearmanr
    return spearmanr(a, b)


def main():
    d = np.load(os.path.join(OUT, "exp88_oil.npz"), allow_pickle=True)
    Xp, yc = d["X_pure"], d["yc"]
    classes = sorted(set(yc.tolist()))
    print(f"[exp88b] 纯油 {Xp.shape}, 类别 {len(classes)}")

    # ---- 真实光谱结构：类均值谱的两两 cos 距离 ----
    M = np.stack([Xp[yc == c].mean(0) for c in classes])
    Ms = np.stack([unit(r) for r in M])          # 形状相似度对强度不敏感
    D_spec = dist_mat(Ms)
    iu = np.triu_indices(len(classes), 1)
    ds = D_spec[iu]

    T, native = build_priors(classes)
    rows = []
    for n, V in T.items():
        Dp = dist_mat(V)
        rho, p = spearman(Dp[iu], ds)
        rows.append(dict(prior=n, rho=float(rho), p=float(p),
                         dim=int(native[n].shape[1]) if n in native else int(V.shape[1])))
    rows.sort(key=lambda r: -r["rho"])
    print("\n==== RSA：先验距离 vs 光谱距离（17 类纯油）====")
    print(f"{'prior':10s} {'native_dim':>10s} {'spearman':>9s} {'p':>9s}")
    for r in rows:
        print(f"{r['prior']:10s} {r['dim']:10d} {r['rho']:+.4f} {r['p']:.2e}")

    # ---- 局部：3 个掺杂物三角 ----
    idx = {c: i for i, c in enumerate(classes)}
    tri = [6, 11, 12]  # Corn, Almond, Canola
    print("\n==== 三个掺杂物（Corn/Almond/Canola）的成对距离 ====")
    print(f"{'prior':10s} " + "  ".join(f"{classes[a]}-{classes[b]}"
                                        for a, b in itertools.combinations(tri, 2)))
    for n, V in T.items():
        Dp = dist_mat(V)
        print(f"{n:10s} " + "  ".join(f"{Dp[idx[a], idx[b]]:14.4f}"
                                      for a, b in itertools.combinations(tri, 2)))
    print(f"{'SPECTRUM':10s} " + "  ".join(f"{D_spec[idx[a], idx[b]]:14.4f}"
                                          for a, b in itertools.combinations(tri, 2)))

    json.dump(dict(rsa=rows, tri=[int(x) for x in tri],
                   tri_classes=[classes[i] for i in tri]),
              open(os.path.join(OUT, "exp88b_rsa.json"), "w"), indent=1)
    print("\n[exp88b] saved out/exp88b_rsa.json")


if __name__ == "__main__":
    main()
