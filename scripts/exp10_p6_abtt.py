# -*- coding: utf-8 -*-
"""exp10 —— 文本空间等距化（all-but-the-top）消融臂。

动机（exp09 诊断）：CLM 文本嵌入空间高度各向异性（候选池两两余弦均值 0.728、
PC1 解释方差 68.3%），加权组分谱在该空间近乎坍缩 → 检索退化为"全选最热门方向"。

修法（零训练、零标签、零谱）：用 46 个分子的纯结构语料估计公共方向
（语料均值 + 前 k 个主成分），从实体/矩阵文本嵌入中扣除，再单位化。

对照：exp08（G1v2）的 Original 臂逐折 top-1。本实验只作消融，
**不改动 G1 预注册判据**；统计口径同源（paired 符号翻转 + Holm），仅作参考。
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")
sys.path.insert(0, HERE)

import torch
import p6_g1_data as G1
import exp05_p6_g0 as E5
import exp08_p6_g1 as E8
from exp09_p6_anisotropy import CORPUS, embed_smiles, metrics

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
K_PC = 1          # exp09 诊断出的最优 k（mean-cos 最低）
EPOCHS = E8.EPOCHS
DEV = E8.DEV


def fit_abtt(k=K_PC):
    """用分子语料拟合：均值方向 + 前 k 个主成分。返回 (mu, Vt[:k])。"""
    emb = embed_smiles(CORPUS)
    C = np.stack([E5._unit(emb[s]) for s in CORPUS]).astype(np.float64)
    mu = C.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(C - mu, full_matrices=False)
    return mu, Vt[:k]


def apply_abtt(T, mu, Vt):
    X = T - mu
    for j in range(len(Vt)):
        X = X - (X @ Vt[j:j + 1].T) @ Vt[j:j + 1]
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def main():
    ent, bg, grid = G1.load_g1()
    eids = sorted(ent.keys()); mids = sorted(bg.keys())
    variants = E8.build_text_variants_g1(eids, mids)
    keys = eids + [f"M_{m}" for m in mids]

    mu, Vt = fit_abtt()
    print(f"[exp10] ABTT: corpus={len(CORPUS)} mols, k={K_PC}, device={DEV}", flush=True)

    def pack(v):
        T = np.stack([variants[v][k] for k in keys]).astype(np.float64)
        return apply_abtt(T, mu, Vt)          # (19, dim) 顺序 = keys

    results = {}
    for arm in ["Original", "ABTT"]:
        Tall = (np.stack([variants["Original"][k] for k in keys]).astype(np.float32)
                if arm == "Original" else pack("Original"))
        T_ent = Tall[:len(eids)]
        T_mat = Tall[len(eids):]
        results[arm] = {}
        for fold, e_hold in enumerate(eids):
            eids_tr = [e for e in eids if e != e_hold]
            tr_ix = {e: i for i, e in enumerate(eids_tr)}
            parts = [ent[e] for e in eids_tr] + [bg[m] for m in mids]
            ys = ([np.full(len(ent[e]), tr_ix[e]) for e in eids_tr] +
                  [np.full(len(bg[m]), len(eids_tr) + j) for j, m in enumerate(mids)])
            Xs = np.concatenate(parts); ys = np.concatenate(ys)
            idx_tr = [keys.index(e) for e in eids_tr] + [len(eids) + j for j in range(len(mids))]
            Ttr = Tall[idx_tr]
            Xh = ent[e_hold]
            accs = []
            for seed in SEEDS:
                net = E8.train_cnn(Xs, ys, Ttr, EPOCHS, seed * 1000 + fold)
                sim = E8.embed(net, Xh) @ T_ent.T
                accs.append(float((sim.argmax(1) == eids.index(e_hold)).mean()))
            results[arm][e_hold] = float(np.mean(accs))
            print(f"[{arm}] {e_hold}: top1={results[arm][e_hold]:.3f} (n_q={len(Xh)})", flush=True)

    # ---- 报告 ----
    o = np.array([results["Original"][e] for e in eids])
    a = np.array([results["ABTT"][e] for e in eids])
    p_raw, _ = E5.perm_paired_greater(a, o)          # H1：ABTT > Original
    lines = ["# exp10 消融：文本空间等距化（ABTT）", "",
             f"> 与 exp08 G1v2 完全同源（{len(SEEDS)} seeds、1D-CNN、6 矩阵背景、13 折 LOAO）；",
             "> 只把文本嵌入换成扣除语料公共方向后的版本。零训练/零标签/零谱，不触碰预注册判据。", "",
             "## 逐折 top-1", "", "| fold | Original | ABTT(等距化) | Δ |", "|---|---|---|---|"]
    for e in eids:
        lines.append(f"| {e} | {results['Original'][e]:.3f} | {results['ABTT'][e]:.3f} | "
                     f"{results['ABTT'][e] - results['Original'][e]:+.3f} |")
    lines += ["", f"- 均值：Original **{o.mean():.3f}** → ABTT **{a.mean():.3f}**"
                  f"（chance={1/len(eids):.3f}）",
              f"- 配对符号翻转检验（ABTT > Original）：raw p = {p_raw:.4f}（探索性，非预注册门）", "",
              "## 解读口径", "",
              "- 等距化是**无监督**修正：公共方向由 46 个分子的纯结构语料估计，不含任何谱或标签；",
              "- 若姊妹对（鸡/火鸡等化学上近乎同组分者）仍不分离，属 H2 的不可约简并度，应如实报告。"]
    with open(os.path.join(OUT, "exp10_p6_abtt.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("[written] exp10_p6_abtt.md", flush=True)


if __name__ == "__main__":
    main()
