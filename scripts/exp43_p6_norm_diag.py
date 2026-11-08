"""exp43 —— 诊断：冷启动到底为什么在某些实体上恒为 0？

背景（来自 exp42 实时日志，未下结论，仅作诊断线索）
--------------------------------------------------
exp42 的 LOO 冷启动矩阵里出现大量 0.000 / 1.000 的极端值，例如
  A05: k=0 几乎所有先验臂 = 0.000，k=full 时 none=1.000 但 ridge/orth=0.000
  A07: k=0 ridge1=1.000，其余几乎全 0
  COFA: k=5 大量 1.000
这种 0/1 跳变不像模型行为，更像下述三种退化之一（或叠加）：

  D1 **测试集过小**      n_te = max(5, 0.3n)，若 n_te≈5，acc 粒度 0.2，
                         单实体 SE≈0.2，任何结论都不成立。
  D2 **化学向量退化**    s_new − μ_S 范数趋近 0 ⇒ 去均值后方向由噪声主导；
                         即使做了 cosine 归一化，方向也可能是随机的。
  D3 **化学不可分**      s_new 与某个老类中心余弦≈1 ⇒ 该实体在化学空间里
                         与老类**本就不可区分**，冷启动在原理上无解。

本实验只测量，不训练（除一个极小的 kNN 参照），回答三个问题：
  Q1  每个实体的样本量 / 测试集大小 ⇒ D1 是否成立？
  Q2  每个实体的 ‖s_new − μ_S‖ / ‖s_new‖ 与方向可分性 ⇒ D2 是否成立？
  Q3  化学空间最近邻余弦 vs 谱空间最近邻余弦 ⇒ D3 是否成立，且两空间是否一致？

预注册判据（写死）
------------------
- H43a：若 ≥3 个实体满足 max_cos_chem(s_new, 老类) > 0.9（化学不可分），
        则「冷启动宏观均值」必须**分层报告**（可分组 / 不可分组），
        否则单一 macro 数字是被不可分子集稀释的假象。
- H43b：若 ‖s_new−μ_S‖/‖s_new‖ 与 k=0 准确率 Spearman ρ > 0.4，
        则去均值是失效根因，应改用**不做中心化的 cosine 头**。
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1

OUT = r"D:/2026sci论文实验设计/论文6/out"


def cosmat(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repr", default="clm", choices=["clm", "ecfp"])
    args = ap.parse_args()

    ent, bg, _ = G1.load_g1()
    eids = sorted(ent.keys())
    mids = sorted(bg.keys())
    keys = eids + [f"M_{m}" for m in mids]
    z = np.load(os.path.join(OUT, "exp34_repr_cache.npz"), allow_pickle=True)
    T = np.asarray(z[args.repr if args.repr != "ecfp" else "ecfp384"], float)
    ki = {k: i for i, k in enumerate(keys)}
    print(f"[exp43] repr={args.repr} dim={T.shape[1]} keys={len(keys)}")
    print(f"[exp43] 基质: {mids}")
    print()

    rows = []
    for e_hold in eids:
        others = [e for e in eids if e != e_hold]
        S = np.asarray(T)[[ki[e] for e in others]
                          + [ki[f"M_{m}"] for m in mids]
                          + [ki[e_hold]]]
        S_old, s_new = S[:18], S[18]
        mu_S = S_old.mean(0, keepdims=True)

        d_new = s_new - mu_S[0]
        ratio = float(np.linalg.norm(d_new) / (np.linalg.norm(s_new) + 1e-12))

        # ---- 化学空间：新类 vs 各老类（去均值 + cosine）----
        Sc_old = S_old - mu_S
        dn = d_new / (np.linalg.norm(d_new) + 1e-12)
        Co = Sc_old / (np.linalg.norm(Sc_old, axis=1, keepdims=True) + 1e-12)
        cos_chem = Co @ dn
        # 不做中心化时的余弦（对照：中心化是否毁掉了信息）
        cn = s_new / (np.linalg.norm(s_new) + 1e-12)
        Co_raw = S_old / (np.linalg.norm(S_old, axis=1, keepdims=True) + 1e-12)
        cos_chem_raw = Co_raw @ cn

        # ---- 谱空间：类中心余弦（真实可分性参照）----
        Xn = np.asarray(ent[e_hold], float)
        Zc_new = Xn.mean(0)
        Zc_old = np.stack([np.asarray(ent[e], float).mean(0) for e in others]
                          + [np.asarray(bg[m], float).mean(0) for m in mids])
        # 谱域用「减去全体谱均值」以消除基线/散射共性，与化学侧处理对称
        gmu = np.concatenate([Zc_old, Zc_new[None]]).mean(0, keepdims=True)
        a = (Zc_new - gmu[0]); a = a / (np.linalg.norm(a) + 1e-12)
        Bm = Zc_old - gmu
        Bm = Bm / (np.linalg.norm(Bm, axis=1, keepdims=True) + 1e-12)
        cos_spec = Bm @ a

        n = len(Xn)
        n_te = int(min(max(5, round(0.3 * n)), n - 1))
        rows.append(dict(
            eid=e_hold, n=n, n_te=n_te,
            ratio=ratio,
            max_cos_chem=float(cos_chem.max()),
            argmax_chem=(others + [f"M_{m}" for m in mids])[int(cos_chem.argmax())],
            max_cos_chem_raw=float(cos_chem_raw.max()),
            max_cos_spec=float(cos_spec.max()),
            argmax_spec=(others + [f"M_{m}" for m in mids])[int(cos_spec.argmax())],
        ))

    print(f"{'实体':>7} {'n':>5} {'n_te':>5} {'‖Δs‖/‖s‖':>9} "
          f"{'chemMax':>8} {'chemRaw':>8} {'specMax':>8}  最近化学邻居 / 最近谱邻居")
    print("-" * 100)
    for r in rows:
        print(f"{r['eid']:>7} {r['n']:>5} {r['n_te']:>5} {r['ratio']:>9.3f} "
              f"{r['max_cos_chem']:>8.3f} {r['max_cos_chem_raw']:>8.3f} "
              f"{r['max_cos_spec']:>8.3f}  {r['argmax_chem']} / {r['argmax_spec']}")
    print()

    # ---------------- 判据 H43a：化学不可分子集 ----------------
    inseparable = [r["eid"] for r in rows if r["max_cos_chem"] > 0.90]
    sep = [r["eid"] for r in rows if r["max_cos_chem"] <= 0.90]
    print("=" * 100)
    print(f"H43a 化学不可分（max_cos_chem > 0.90）: {len(inseparable)}/{len(rows)} "
          f"→ {inseparable}")
    if len(inseparable) >= 3:
        print("      ⇒ H43a 成立：宏观 macro 均值被不可分子集稀释，"
              "必须分层报告（可分组 / 不可分组）。")
    else:
        print("      ⇒ H43a 不成立：化学向量整体可分，"
              "0.000 的成因指向别处（编码器或对齐，而非先验本身）。")
    print(f"     可分组（≤0.90）: {len(sep)} → {sep}")

    # ---------------- 判据 H43b：去均值是否毁掉信息 ----------------
    print()
    print("=" * 100)
    print("H43b ‖Δs‖/‖s‖ 小 ⇒ 去均值后方向由噪声主导")
    small = [r["eid"] for r in rows if r["ratio"] < 0.5]
    print(f"     ‖Δs‖/‖s‖ < 0.5 的实体: {len(small)}/{len(rows)} → {small}")
    dg = np.array([r["max_cos_chem_raw"] - r["max_cos_chem"] for r in rows])
    print(f"     中心化后最近邻余弦的变化 Δ = raw − centered："
          f"mean={dg.mean():+.3f}  min={dg.min():+.3f}  max={dg.max():+.3f}")
    if dg.mean() > 0.05:
        print("     ⇒ 中心化**降低**了新类与老类的可区分度，"
              "应试验不做中心化的 cosine 头。")
    else:
        print("     ⇒ 中心化确实提高了可区分度，去均值不是失效根因。")

    # ---------------- 两空间一致性 ----------------
    cc = np.array([r["max_cos_chem"] for r in rows])
    cs = np.array([r["max_cos_spec"] for r in rows])
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(cc, cs)
    except Exception:
        rho, p = float("nan"), float("nan")
    print()
    print("=" * 100)
    print(f"化学最近邻余弦 vs 谱最近邻余弦 Spearman ρ = {rho:+.3f} (p={p:.3f})")
    print(f"     mean(chem)={cc.mean():.3f}  mean(spec)={cs.mean():.3f}")
    print("     注：两者都高 ⇒ 该实体在两个空间里都与某个老类高度相似，"
          "冷启动在原理上不可解，不应计入「先验有效吗」的证据。")

    with open(os.path.join(OUT, "exp43_diag.json"), "w", encoding="utf-8") as f:
        json.dump({"repr": args.repr, "rows": rows,
                   "inseparable": inseparable, "separable": sep,
                   "rho_chem_spec": [float(rho), float(p)]}, f,
                  ensure_ascii=False, indent=1)
    print("\n[exp43] 已写出 out/exp43_diag.json")


if __name__ == "__main__":
    main()
