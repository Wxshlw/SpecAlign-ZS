r"""exp88a —— 外部数据集（植物油互掺 ATR-FTIR）解析与结构探索。

目的
----
现有 7 实体数据集的 6 个掺杂物全是**不同品类**（糖/肉桂/姜黄/橄榄油/咖啡/橙汁），
判别轴就是"组成"，因此 physchem / charbag 这类组成统计量天生占优。
exp87 已证明：这是 CLM 打不过它们的结构性原因，换表示/换对齐都救不回来。

本数据集（Mendeley 10.17632/ctgg7k4m5g, CC BY 4.0）提供的是**同类近邻**掺杂物：
    EVOO（特级初榨橄榄油）掺 Almond（杏仁油）/ Canola（芥花油）/ Corn（玉米油）
三者都是植物油，脂肪酸组成相近（C18:1 / C18:2 / C18:3 之别），
元素组成与 MW 几乎相同。⇒ 这是检验"physchem 优势是否只在跨品类任务成立"的判决性数据。

输出
----
out/exp88_oil.npz：
    X_adul (n, w)   掺混谱；y (n,) 掺杂物类别；conc (n,) 掺杂物百分数；
    gid (n,)        配方组 id（同一配方 3 条重复 → 同组，用于 group 划分）
    X_pure (m, w)   纯油谱；yc (m,) 纯油类别号；wn (w,) 波数
    names           类别名
另出诊断 json。
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
DAT = os.path.join(HERE, "..", "data_ext")

ADUL = os.path.join(DAT, "ATRAdulteration3.csv")
PURE = os.path.join(DAT, "ATRPure3.csv")
KEY = os.path.join(DAT, "OilClassKey.csv")


def main():
    a = pd.read_csv(ADUL)
    comp_cols = ["EVOO", "Almond", "Canola", "Corn"]
    comp = a[comp_cols].to_numpy(float)
    sp_cols = [c for c in a.columns if c not in comp_cols]
    wn = np.array([float(c) for c in sp_cols])
    X = a[sp_cols].to_numpy(np.float32)

    # 掺杂物类别 = 唯一非零的掺杂物列（EVOO 是基质）
    adulter_cols = ["Almond", "Canola", "Corn"]
    ac = a[adulter_cols].to_numpy(float)
    nz = (ac > 0).sum(1)
    assert (nz == 1).all(), f"存在多掺杂物样本: {np.where(nz != 1)[0]}"
    y_lab = np.array(adulter_cols)[ac.argmax(1)]
    conc = ac.max(1)

    # 配方组：组成向量完全相同的样本为同一配方（3 条重复）
    comp_key = [tuple(r) for r in comp.astype(int)]
    uniq = {}
    gid = np.zeros(len(a), int)
    for i, k in enumerate(comp_key):
        if k not in uniq:
            uniq[k] = len(uniq)
        gid[i] = uniq[k]

    # 纯油谱
    p = pd.read_csv(PURE)
    yc = p["Class"].to_numpy(int)
    Xp = p.drop(columns=["Class"]).to_numpy(np.float32)
    key = pd.read_csv(KEY)
    cname = {int(r["Class Number"]): str(r["Class Name"]) for _, r in key.iterrows()}

    # 波数对齐检查
    wn_p = np.array([float(c) for c in p.columns if c != "Class"])
    same_grid = (len(wn_p) == len(wn)) and np.allclose(wn_p, wn)
    if not same_grid:
        lo = max(wn.min(), wn_p.min()); hi = min(wn.max(), wn_p.max())
        print(f"[warn] 纯油谱网格不同: pure={len(wn_p)} adul={len(wn)} "
              f"重叠区 {lo:.1f}-{hi:.1f}", flush=True)

    print(f"[exp88a] 掺混 {X.shape}  配方组 {len(uniq)}  纯油 {Xp.shape}")
    print(f"  波数 {wn.min():.1f}-{wn.max():.1f} cm-1, {len(wn)} 点")
    print("  掺杂物分布:", {c: int((y_lab == c).sum()) for c in adulter_cols})
    for c in adulter_cols:
        v = conc[y_lab == c]
        print(f"    {c:7s} conc {v.min():.0f}-{v.max():.0f}%  中位 {np.median(v):.0f}"
              f"  唯一浓度 {sorted(set(v.astype(int)))}")
    print("  共同重叠浓度区间检查:")
    rng = {c: (conc[y_lab == c].min(), conc[y_lab == c].max()) for c in adulter_cols}
    lo = max(v[0] for v in rng.values()); hi = min(v[1] for v in rng.values())
    print(f"    三类共同重叠 = [{lo:.0f}, {hi:.0f}]%  " +
          ("⚠️ 空/极窄，必须按浓度分层" if hi - lo < 10 else "可用"))

    print("  纯油类别:")
    for k in sorted(set(yc)):
        print(f"    {k:2d} {cname[k]:35s} n={int((yc==k).sum())}")

    # 类内重复度诊断：同配方 3 条重复的余弦
    def unit(v):
        n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return v / n
    Xu = unit(X)
    rep_cos = []
    for g in np.unique(gid):
        ix = np.where(gid == g)[0]
        if len(ix) > 1:
            C = Xu[ix] @ Xu[ix].T
            iu = np.triu_indices(len(ix), 1)
            rep_cos.append(C[iu].mean())
    print(f"  同配方重复余弦均值 = {np.mean(rep_cos):.6f} (n_rep组={len(rep_cos)})")

    np.savez(os.path.join(OUT, "exp88_oil.npz"),
             X_adul=X, y=y_lab, conc=conc, gid=gid, wn=wn,
             X_pure=Xp, yc=yc, names=np.array(adulter_cols),
             pure_names=np.array([cname[k] for k in sorted(set(yc))]))
    json.dump(dict(n_adul=int(len(X)), n_pure=int(len(Xp)), n_group=int(len(uniq)),
                   wn=[float(wn.min()), float(wn.max()), int(len(wn))],
                   conc_range=rng, overlap=[float(lo), float(hi)],
                   rep_cos=float(np.mean(rep_cos))),
              open(os.path.join(OUT, "exp88_oil_diag.json"), "w"), indent=1)
    print("[exp88a] saved out/exp88_oil.npz")


if __name__ == "__main__":
    main()
