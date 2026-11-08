# -*- coding: utf-8 -*-
"""
exp58c —— 为什么 A06 上「带化学条件的编码器」反而差于「无化学编码器」？
          （A03 不存在此现象：chem 0.589 ≈ free 0.590）

假设 H58d：stage1 把 z 空间训练成按**老类化学相似度**排布。
  若未见类的**化学向量**与某老类极相似（A06↔A04 cos=0.984），
  但其**谱**实际很可分（free 编码器能到 0.889），
  则 chem 编码器会把它的谱压到与该老类重叠的 z 区域 ⇒ 自伤。
  预测：cos_z(未见类中心, 最近老类中心) 在 chem 下 **高于** free 下。

同时给出 A03（两者相当）作为对照：A03 应该是**谱本身就混淆**，
   ⇒ chem/free 下 cos_z 都很高 ⇒ 换编码器救不了 ⇒ 那是"码本共线"的问题。
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53_p6_zsl_baselines as E53         # noqa: E402
import exp49_p6_clean_protocol as E49        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")


def cos_to_nearest(v, M):
    """v 到 M 各行余弦的最大值"""
    vn = v / (np.linalg.norm(v) + 1e-12)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return float((Mn @ vn).max())


def main():
    ent, bg = E53.load_data()
    eids = sorted(ent.keys())
    mids = sorted(bg.keys())
    z = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    CLM = z["clm"].astype(np.float64)
    ki = {k: i for i, k in enumerate(keys)}

    print("=" * 96)
    print("三个空间中「未见类 → 最近老类」的相似度（越高越混淆）")
    print("=" * 96)
    print(f"{'held':<8}{'化学空间':>10}{'谱空间':>10}"
          f"{'z空间(chem)':>13}{'z空间(free)':>13}{'Δ chem−free':>13}"
          f"{'判定':>18}")

    rec = {}
    for e_hold in eids:
        others = [k for k in eids if k != e_hold]
        rng = np.random.default_rng(0)
        Xtr, ytr = [], []
        for j, e in enumerate(others):
            Xo = np.asarray(ent[e], np.float64)
            q = rng.permutation(len(Xo)); nte = max(5, int(0.3 * len(Xo)))
            Xtr.append(Xo[q[nte:]]); ytr.append(np.full(len(Xo) - nte, j))
        for j, m in enumerate(mids):
            Xo = np.asarray(bg[m], np.float64)
            q = rng.permutation(len(Xo)); nte = max(5, int(0.3 * len(Xo)))
            Xtr.append(Xo[q[nte:]]); ytr.append(np.full(len(Xo) - nte,
                                                        len(others) + j))
        Xtr = np.vstack(Xtr).astype(np.float32)
        ytr = np.concatenate(ytr).astype(np.int64)
        rows_old = sorted(set(ytr.tolist()))

        Xh = np.asarray(ent[e_hold], np.float64)
        q = np.random.default_rng(999).permutation(len(Xh))
        Xte = Xh[q[:max(5, int(0.3 * len(Xh)))]].astype(np.float32)

        # 关键：训练标签 j 对应 others[j] / mids[j-len(others)]，
        # **不能**直接用 CLM[j]（others 已剔除 e_hold，与 CLM 行序错位）。
        def name_of(j):
            return others[j] if j < len(others) else f"M_{mids[j - len(others)]}"

        # ---- 化学空间 ----
        c_chem = cos_to_nearest(CLM[ki[e_hold]],
                                np.stack([CLM[ki[name_of(j)]] for j in rows_old]))

        # ---- 谱空间（原始）----
        sh = Xte.mean(0)
        Sp = np.stack([np.asarray(ent[name_of(j)] if j < len(others)
                                  else bg[mids[j - len(others)]],
                                  np.float64).mean(0)
                       for j in rows_old])
        c_spec = cos_to_nearest(sh, Sp)

        # ---- z 空间：chem / free 两种编码器 ----
        # S 必须与训练标签对齐：others(12) + mids(6) + e_hold
        S = np.asarray(CLM)[[ki[name_of(j)] for j in rows_old]
                            + [ki[e_hold]]].astype(np.float32)
        cz = {}
        for mode in ("chem", "free"):
            net, _ = E49.stage1(Xtr, ytr, CLM.shape[1], len(rows_old),
                                400, 0, S=S, mode=mode)
            Zf = E53.embed(net, Xtr)
            Zi_old = np.stack([Zf[ytr == j].mean(0) for j in rows_old])
            zh = E53.embed(net, Xte).mean(0)
            cz[mode] = cos_to_nearest(zh, Zi_old)
            del net
        torch.cuda.empty_cache()

        d = cz["chem"] - cz["free"]
        verdict = ("chem 自伤" if d > 0.05 else
                   "chem 有益" if d < -0.05 else "无差")
        rec[e_hold] = (c_chem, c_spec, cz["chem"], cz["free"], d)
        print(f"{e_hold:<8}{c_chem:10.4f}{c_spec:10.4f}"
              f"{cz['chem']:13.4f}{cz['free']:13.4f}{d:+13.4f}{verdict:>18}")

    print()
    print("=" * 96)
    print("相关性：chem 自伤程度 Δ 与 exp53 的 (ridge-f − ridge-c) 是否同向？")
    print("=" * 96)
    import json
    from scipy.stats import spearmanr
    r = json.load(open(os.path.join(OUT, "exp53_zsl_k0.json"), encoding="utf-8"))

    def pe(a, m):
        return {e: np.mean([x[m] for x in r["acc"]["0"].get(a, {}).get(e, [])
                            if x[m] is not None])
                for e in eids if r["acc"]["0"].get(a, {}).get(e)}

    fc = pe("ridge-f", "auc")
    cc = pe("ridge-c", "auc")
    rc = pe("ridge-rand", "auc")
    dd = np.array([rec[e][4] for e in eids])
    gap = np.array([fc[e] - cc[e] for e in eids])
    gap2 = np.array([rc[e] - cc[e] for e in eids])
    rho, p = spearmanr(dd, gap)
    print(f"  Spearman ρ(Δ cos_z,  ridge-f − ridge-c) = {rho:+.3f}  p={p:.4f}")
    rho2, p2 = spearmanr(dd, gap2)
    print(f"  Spearman ρ(Δ cos_z,  ridge-rand − ridge-c) = {rho2:+.3f}  p={p2:.4f}")

    print()
    print(f"{'held':<8}{'ridge-c':>9}{'ridge-f':>9}{'ridge-rand':>11}"
          f"{'f−c':>9}{'rand−c':>9}")
    for e in sorted(eids, key=lambda x: -(fc[x] - cc[x])):
        print(f"{e:<8}{cc[e]:9.3f}{fc[e]:9.3f}{rc[e]:11.3f}"
              f"{fc[e] - cc[e]:+9.3f}{rc[e] - cc[e]:+9.3f}")

    with open(os.path.join(OUT, "exp58c_encoder.json"), "w",
              encoding="utf-8") as f:
        json.dump({e: list(map(float, v)) for e, v in rec.items()}, f,
                  ensure_ascii=False)
    print("\n[exp58c] 已写出 out/exp58c_encoder.json")


if __name__ == "__main__":
    main()
