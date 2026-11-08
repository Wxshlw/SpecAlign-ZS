# -*- coding: utf-8 -*-
"""exp09 —— 文本嵌入空间各向异性诊断 + 无训练等距化修正（all-but-the-top / whitening）。

参考文献（本脚本度量口径的来源）：
  - Ethayarajh 2019 (EMNLP)          ：预训练 LM 表征各向异性
  - Mu & Viswanath 2018 (ICLR)       ：All-but-the-top（去前 k 个主成分）
  - Su et al. 2021 (ICLR)            ：Whitening 句子表征
  - Wang & Isola 2020 (ICML)         ：alignment / uniformity 度量
  - Radovanovic et al. 2010 (JMLR)   ：hubness（k-occurrence skewness）

只做诊断与"零训练"修法的可行性验证，不改动 G1 判据。
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p6_g1_data as G1
import p6_library as L
import exp05_p6_g0 as E5
import exp08_p6_g1 as E8

# --- 更宽的"分子语料"：用于无偏地拟合主成分方向（只用化学结构，不碰任何标签/谱） ---
AA = ["NCC(=O)O", "CC(N)C(=O)O", "CC(C)C(N)C(=O)O", "CC(C)CC(N)C(=O)O", "CC[C@H](C)C(N)C(=O)O",
      "N[C@@H](CO)C(=O)O", "C[C@H](O)[C@H](N)C(=O)O", "N[C@@H](CS)C(=O)O", "CSCCC[C@H](N)C(=O)O",
      "N[C@@H](CC(=O)O)C(=O)O", "N[C@@H](CCC(=O)O)C(=O)O", "N[C@@H](CC(N)=O)C(=O)O",
      "NC(=O)CCC(N)C(=O)O", "NCCCC[C@H](N)C(=O)O", "NC(=N)NCCC[C@H](N)C(=O)O",
      "N[C@@H](Cc1c[nH]cn1)C(=O)O", "N[C@@H](Cc1ccccc1)C(=O)O", "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
      "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O", "OC(=O)[C@@H]1CCCN1"]
FA = ["CCCC(=O)O", "CCCCCC(=O)O", "CCCCCCCC(=O)O", "CCCCCCCCCC(=O)O", "CCCCCCCCCCCC(=O)O",
      "CCCCCCCCCCCCCC(=O)O", "CCCCCCCCCCCCCCCC(=O)O", "CCCCCCCCCCCCCCCCCC(=O)O",
      "CCCCCCCCC=CCCCCCCCC(=O)O", "CCCCC=CCC=CCCCCCCCCC(=O)O", "CC=CCC=CCC=CCCCCCCCCC(=O)O",
      "CCCCCCCCCCCCCCCCCCC(=O)O", "CCCCCCCCCCCC=CCCCCCCCCC(=O)O"]
OTHER = ["OC[C@H]1O[C@H](O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@H](O)[C@@H]1O",
         "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O", "OC[C@@]1(O)O[C@H](CO)[C@@H](O)[C@H]1O",
         "O=C(O)CC(O)C(=O)O", "OC(=O)CC(O)(CC(=O)O)C(=O)O", "CC(=O)O", "CC(O)C(=O)O",
         "O=C(O)C(O)C(O)C(=O)O", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "C[n+]1cccc(c1)C(=O)[O-]",
         "O=CC=CC1=CC=CC=C1", "O=C1OC2=CC=CC=C2C=C1", "COc1cc(C=CC(=O)CC(=O)C=Cc2ccc(O)c(OC)c2)ccc1O"]
CORPUS = sorted(set(AA + FA + OTHER))


def embed_smiles(smis):
    tok, mdl = E5._load_clm(random_init=False)
    outs = {}
    with __import__("torch").no_grad():
        B = 16
        for i in range(0, len(smis), B):
            batch = smis[i:i + B]
            inputs = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = mdl(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            for s, v in zip(batch, emb.numpy()):
                outs[s] = v
    return outs


def metrics(T, tag):
    """T: (n,d) 单位向量。返回各向异性/均匀性度量。"""
    n = len(T)
    S = T @ T.T
    off = S[~np.eye(n, dtype=bool)]
    cos_mean = float(off.mean())
    # 均匀性 uniform = log E_{i!=j} exp(-2 * (1 - cos))   （Wang & Isola 2020，t=2）
    d2 = 2.0 * (1.0 - off)
    uniform = float(np.log(np.mean(np.exp(-d2))))
    # PCA 谱：第一主成分解释方差占比（各向异性的直接证据）
    Xc = T - T.mean(0, keepdims=True)
    ev = np.linalg.svd(Xc, compute_uv=False) ** 2
    ev = ev / ev.sum()
    return dict(tag=tag, n=n, cos_mean=cos_mean, cos_max=float(off.max()),
                pc1=float(ev[0]), pc1_3=float(ev[:3].sum()), uniform=uniform)


def pairwise_max_cos(T, names):
    S = T @ T.T
    np.fill_diagonal(S, -1)
    return {names[i]: (float(S[i].max()), names[int(S[i].argmax())]) for i in range(len(names))}


def main():
    ent, bg, grid = G1.load_g1()
    eids = sorted(ent.keys()); mids = sorted(bg.keys())
    variants = E8.build_text_variants_g1(eids, mids)
    keys = eids + [f"M_{m}" for m in mids]
    T0 = np.stack([variants["Original"][k] for k in keys]).astype(np.float64)
    T0 = T0 / np.linalg.norm(T0, axis=1, keepdims=True)
    names = keys

    print("=== 1) 原始空间（G1v2 先验）===")
    m0 = metrics(T0, "raw")
    print(f"  n={m0['n']}  mean-cos={m0['cos_mean']:.4f}  max-cos={m0['cos_max']:.4f}")
    print(f"  PC1 解释方差={m0['pc1']:.3%}   前3主成分={m0['pc1_3']:.3%}   uniformity={m0['uniform']:.4f}")
    print("  诊断：PC1 占比越高 → 所有向量共享一个巨大公共分量（经典各向异性）")

    # --- 用宽分子语料拟合"公共方向"（无监督、无标签、无谱） ---
    print(f"\n=== 2) 用分子语料拟合公共方向（{len(CORPUS)} 个分子，纯结构，非候选池）===")
    emb = embed_smiles(CORPUS)
    C = np.stack([E5._unit(emb[s]) for s in CORPUS]).astype(np.float64)
    mu = C.mean(0, keepdims=True)
    U, sv, Vt = np.linalg.svd(C - mu, full_matrices=False)
    print(f"  语料嵌入 PC1 解释方差 = {(sv**2 / (sv**2).sum())[0]:.3%}")

    def remove_topk(T, k):
        """All-but-the-top：去掉均值方向 + 前 k 个主成分。"""
        X = T - mu
        for j in range(k):
            X = X - (X @ Vt[j:j + 1].T) @ Vt[j:j + 1]
        return X / np.linalg.norm(X, axis=1, keepdims=True)

    print("\n=== 3) all-but-the-top 修正效果（候选池 19 个向量）===")
    print(f"  {'k':>3s} {'mean-cos':>9s} {'max-cos':>9s} {'PC1':>8s} {'uniformity':>11s}")
    best = None
    for k in range(0, 7):
        Tk = remove_topk(T0, k)
        mk = metrics(Tk, f"abtt{k}")
        print(f"  {k:3d} {mk['cos_mean']:9.4f} {mk['cos_max']:9.4f} {mk['pc1']:8.3%} {mk['uniform']:11.4f}")
        if best is None or mk["cos_mean"] < best[1]["cos_mean"]:
            best = (k, mk)

    print("\n=== 4) 修正前后 姊妹实体 文本余弦对照 ===")
    pm0 = pairwise_max_cos(T0, names)
    Tk = remove_topk(T0, best[0])
    pmk = pairwise_max_cos(Tk, names)
    for pair in [("MEATCH", "MEATP"), ("MEATCH", "MEATT"), ("MEATP", "MEATT"),
                 ("COFA", "COFR"), ("A01", "A02"), ("A03", "A06"), ("A04", "A06"), ("A05", "A02")]:
        i, j = names.index(pair[0]), names.index(pair[1])
        c0 = float(T0[i] @ T0[j]); ck = float(Tk[i] @ Tk[j])
        print(f"  {pair[0]:7s}–{pair[1]:7s}  raw={c0:.4f}  abtt(k={best[0]})={ck:.4f}   Δ={c0-ck:+.4f}")

    print("\n=== 5) 修正后各实体的最近邻（是否仍塌向姊妹）===")
    for e in eids:
        i = names.index(e)
        print(f"  {e:8s} raw最近邻={pm0[e][1]:7s}({pm0[e][0]:.4f})   "
              f"abtt最近邻={pmk[e][1]:7s}({pmk[e][0]:.4f})")

    print("\n=== 结论口径 ===")
    print("  以上均为零训练、零标签、零谱的无监督变换，可直接作为 G1 的一个消融臂（Original+ABTT）。")
    print("  若需真正的表示学习修法 → CLIP 式对称 InfoNCE 联合微调 + 姊妹对硬负样本（见 exp10）。")


if __name__ == "__main__":
    main()
