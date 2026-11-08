"""exp43b —— 定位 exp42「center 在 k=0 高、k=5 崩为 0」的反常跳变。

exp42 日志（A01，chem 协议）：
    k=0 : center=0.685  orth=0.630  ridge0.01=0.761
    k=5 : center=0.000  orth=0.798  ridge0.01=0.828
`center` 与 `orth` 的头只差一个正交旋转 R，且 R 保范数，两者**不应差 0.8**。
而 exp43 已排除三个候选解释：
    × 测试集太小  —— A01 有 600 样本，n_te=180，粒度 0.006
    × 中心化漂移  —— k=5 时 μ_S 含新类一行，d_new 仅缩放 18/19，归一化后不变
    × 化学向量退化—— ‖Δs‖/‖s‖=0.48，不是噪声主导
⇒ 剩下的可能：**恒定预测假象**（z 空间塌陷 ⇒ argmax 由 W 行的范数/偏置决定，
  与化学语义无关），或 **W 行范数极度不均**导致 cosine 头退化。

本脚本只跑 A01 一个实体、单 seed，把中间量全部打印出来做**人工判读**：
  1. 各类 W 行范数（center / orth）——是否有行趋近 0
  2. 测试样本的预测分布 top-5 ——是否恒定预测某一类
  3. z 与 W 各行的余弦均值矩阵 ——新类行的余弦是否真的最大
  4. k=0 与 k=5 的 z 空间是否同一个（CKA 对比）
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1
import exp42_p6_projection_structure as E42

OUT = r"D:/2026sci论文实验设计/论文6/out"
DEV = E42.DEV
TEMP = E42.TEMP


def main():
    ent, bg, _ = G1.load_g1()
    eids = sorted(ent.keys())
    mids = sorted(bg.keys())
    keys = eids + [f"M_{m}" for m in mids]
    z = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    T_all = np.asarray(z["clm"], float)
    dim = T_all.shape[1]
    ki = {k: i for i, k in enumerate(keys)}

    e_hold = "A01"
    others = [e for e in eids if e != e_hold]
    S = np.asarray(T_all)[[ki[e] for e in others]
                          + [ki[f"M_{m}"] for m in mids]
                          + [ki[e_hold]]]
    NEW = 18
    Xe = np.asarray(ent[e_hold], float)
    names = others + [f"M_{m}" for m in mids] + [e_hold]

    for k in (0, 5):
        tag = "k=0" if k == 0 else f"k={k}"
        rng = np.random.default_rng(0 * 7919 + 13)
        perm = rng.permutation(len(Xe))
        n_te = int(min(max(5, round(0.3 * len(Xe))), len(Xe) - 1))
        te_ix, pool_ix = perm[:n_te], perm[n_te:]
        tr_e_ix = pool_ix if k is None else pool_ix[:min(k, len(pool_ix))]
        Xte = Xe[te_ix]

        parts, ys = [], []
        for j, e in enumerate(others):
            parts.append(np.asarray(ent[e], float)); ys.append(np.full(len(ent[e]), j))
        for j, m in enumerate(mids):
            parts.append(np.asarray(bg[m], float)); ys.append(np.full(len(bg[m]), len(others) + j))
        nrows = len(others) + len(mids)
        if len(tr_e_ix):
            parts.append(Xe[tr_e_ix]); ys.append(np.full(len(tr_e_ix), 18)); nrows = 19
        Xtr = np.concatenate(parts).astype(np.float32)
        ytr = np.concatenate(ys).astype(np.int64)

        rng_v = np.random.default_rng(0 * 31 + 7)
        val_mask = np.zeros(len(Xtr), bool)
        rows_old = sorted({int(y) for y in ytr})
        for j in rows_old:
            if j == 18 and len(tr_e_ix) == 0:
                continue
            ix = rng_v.permutation(np.where(ytr == j)[0])
            val_mask[ix[:max(3, int(0.2 * len(ix)))]] = True
        Xfit, yfit = Xtr[~val_mask], ytr[~val_mask]

        net, W_free = E42.stage1(Xfit, yfit, dim, nrows, 600, 0, S=S, mode="chem")
        Zfit = E42.embed(net, Xfit)
        Z_old = np.stack([Zfit[yfit == j].mean(0) for j in rows_old])
        S_old = S[rows_old]
        mu_S = S_old.mean(0, keepdims=True)

        W_center = S - mu_S
        W_orth = (S - mu_S) @ E42.head_orth_R(S_old, Z_old)

        print()
        print("=" * 92)
        print(f"[{tag}] nrows={nrows}  n_te={n_te}  "
              f"CKA(Z_old,S_old)={E42.cka(Z_old, S_old):.3f}")
        print("=" * 92)

        for nm, W in (("center", W_center), ("orth", W_orth)):
            nrm = np.linalg.norm(W, axis=1)
            print(f"\n-- {nm} --  W 行范数: min={nrm.min():.3f} max={nrm.max():.3f} "
                  f"新类行(#18)={nrm[18]:.3f} 中位数={np.median(nrm):.3f}")
            order = np.argsort(nrm)[:4]
            print(f"   范数最小的 4 行: " +
                  ", ".join(f"{names[j]}={nrm[j]:.3f}" for j in order))

            zt = F.normalize(net(torch.tensor(Xte, dtype=torch.float32, device=DEV)), dim=1).detach()
            Wt = F.normalize(torch.tensor(W, dtype=torch.float32, device=DEV), dim=1)
            logits = (zt @ Wt.T)[:, :19]
            pred = logits.argmax(1).cpu().numpy()
            cnt = np.bincount(pred, minlength=19) / len(pred)
            top = np.argsort(cnt)[::-1][:5]
            print(f"   预测分布 top5: " +
                  ", ".join(f"{names[j]}={cnt[j]:.3f}" for j in top))
            acc = float((pred == NEW).mean())
            print(f"   新类召回 = {acc:.3f}")

            # 测试样本 z 与各行 W 的平均余弦，看新类行是否真的最大
            Cm = (zt @ Wt.T).cpu().numpy().mean(0)
            o2 = np.argsort(Cm)[::-1][:5]
            print(f"   z·W 均值 top5: " +
                  ", ".join(f"{names[j]}={Cm[j]:+.3f}" for j in o2) +
                  f"  | 新类行 = {Cm[18]:+.3f} (排名 {int((Cm > Cm[18]).sum()) + 1}/19)")

    # k=0 与 k=5 的 z 空间是否同一个
    print()
    print("=" * 92)
    print("注：若 center 的「z·W 均值 top5」里新类行排名靠后，则 0.685 是"
          "**恒定预测假象**；若排名靠前却仍判错，则是对齐方向错了。")


if __name__ == "__main__":
    main()
