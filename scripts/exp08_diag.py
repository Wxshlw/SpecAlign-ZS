# -*- coding: utf-8 -*-
"""exp08 事后诊断：零分折塌向哪里 + Original 文本简并结构（不重训）。"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import p6_g1_data as G1
import exp08_p6_g1 as E8

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
simstore = np.load(os.path.join(OUT, "exp08_sims.npy"), allow_pickle=True).item()

ent, bg, grid = G1.load_g1()
eids = sorted(ent.keys())
variants = E8.build_text_variants_g1(eids, sorted(bg.keys()))

print("=== 1) 各折预测分布（Original，13 类 argmax 直方图）===")
for e_hold in eids:
    sim = simstore["Original"][e_hold]
    pred = sim.argmax(1)
    top1 = float((pred == eids.index(e_hold)).mean())
    if top1 < 0.10:
        uniq, cnt = np.unique(pred, return_counts=True)
        dist = ", ".join(f"{eids[u]}:{c}" for u, c in sorted(zip(uniq, cnt), key=lambda t: -t[1])[:4])
        print(f"  {e_hold:8s} top1={top1:.3f}  塌向 → {dist}")

print("\n=== 2) Original 文本两两余弦（T0@T0.T，对角线置 -1）===")
T0 = np.stack([variants["Original"][e] for e in eids]).astype(np.float32)
T0n = T0 / np.linalg.norm(T0, axis=1, keepdims=True)
S = T0n @ T0n.T
np.fill_diagonal(S, -1)
for i, e in enumerate(eids):
    j = int(S[i].argmax())
    print(f"  {e:8s} max-cos={S[i, j]:.4f} vs {eids[j]}")
mx = S.max()
print(f"  全局最高非对角余弦 = {mx:.4f}")

print("\n=== 3) 各变体每折 top1 全表 ===")
hdr = "  " + " ".join(f"{e:>7s}" for e in eids)
for v in E8.E5.VARIANTS:
    row = []
    for e in eids:
        sim = simstore[v][e]
        row.append(f"{float((sim.argmax(1) == eids.index(e)).mean()):7.3f}")
    print(f"{v:16s}{hdr}")
    print(f"{'':16s}" + " ".join(row))
