"""exp65 —— 数据集健康度审计：RF 的 tpr5=1.000 是泛化还是记忆？

=============================================================================
触发
=============================================================================
exp63 k=5：rf / svm-rbf 的 TPR@FPR5 = **1.000**（A01 等实体）。
完美分类在真实掺假数据上极不寻常 ⇒ 必须排除两种数据病：

  P1【技术重复】同一份样品重复扫描多次，被随机划分切进 train 与 test
     ⇒ 模型只需记忆，tpr5=1.000 是必然，毫无意义。
  P2【近重复泄漏】同一浓度水平的 50 条谱互为近似副本，随机划分使
     训练侧与测试侧共享同一浓度的副本 ⇒ 浓度维度上的插值而非外推。

=============================================================================
设计（三层，从便宜到决定性）
=============================================================================
L1 结构审计
    每个实体：n、浓度水平数、每水平条数、浓度集合
L2 近重复审计（**关键**）
    同浓度组内 vs 跨浓度组的余弦相似度分布
    单链接代理分组（tau=0.9999/0.999/0.995/0.99）⇒ 估计"独立样品数"
    若 独立样品数 << 谱数 ⇒ 存在严重技术重复
L3 决定性检验：浓度**不相交**划分（严格外推协议）
    现状协议：随机划分（训练/测试共享浓度 ⇒ 插值）
    严格协议：训练只用浓度子集 C_tr，测试用不相交的 C_te ⇒ 必须外推
    若严格协议下 rf 从 1.000 崩塌 ⇒ 证明现状是记忆，不是泛化
"""
from __future__ import annotations
import os, sys, json, argparse, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6_g1_data as G1

OUT = r"D:/2026sci论文实验设计/论文6/out"
EID2MID = {"A01": "cinnamon", "A02": "cinnamon", "A03": "cinnamon",
           "A04": "turmeric", "A05": "oliveoil", "A06": "coffeeR",
           "A07": "juice", "HSYRUP": "honey"}


def nrm(A):
    Az = A - A.mean(axis=1, keepdims=True)
    return Az / (np.linalg.norm(Az, axis=1, keepdims=True) + 1e-12)


def proxy_groups(X, taus=(0.9999, 0.999, 0.995, 0.99, 0.95)):
    """余弦相似度单链接代理分组：估计独立样品数（复刻 p5/p6_data 口径）。"""
    S = nrm(X) @ nrm(X).T
    n = len(X)
    out = {}
    for tau in taus:
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        ii, jj = np.where(np.triu(S >= tau, 1))
        for a, b in zip(ii, jj):
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[ra] = rb
        out[tau] = len({find(i) for i in range(n)})
    return out


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    ent, conc, bg, _ = G1.load_g1_conc()
    eids = [e for e in sorted(ent) if conc.get(e) is not None]

    L = []
    w = L.append
    w("# exp65 —— 数据集健康度审计：RF 的 tpr5=1.000 是泛化还是记忆？\n")
    w("预处理口径与评估脚本一致：SNV + SG(9,2,deriv=1)，公共谱窗 240 点。\n")

    # ---------------- L1 结构 ----------------
    w("\n## L1 结构审计\n")
    w("| 实体 | 谱数 | 浓度水平数 | 每水平条数 | 浓度集合 |")
    w("|---|---|---|---|---|")
    struct = {}
    for e in eids:
        X = np.asarray(ent[e], float)
        c = np.asarray(conc[e], float)
        lv, cnt = np.unique(c, return_counts=True)
        struct[e] = dict(n=len(X), n_lv=len(lv),
                         per=dict(zip(lv.tolist(), cnt.tolist())))
        per = ", ".join(f"{v:g}×{n}" for v, n in zip(lv, cnt))
        w(f"| {e} | {len(X)} | {len(lv)} | {cnt.min()}–{cnt.max()} | "
          f"{np.array2string(lv, precision=0, max_line_width=200)} |")

    # ---------------- L2 近重复 ----------------
    w("\n## L2 近重复审计（**决定性**）\n")
    w("同浓度组内 vs 跨浓度组的余弦相似度；代理分组估计「独立样品数」。\n")
    w("| 实体 | 谱数 | 组内相似度<br>中位数 | 组内<br>>0.999 占比 | 跨浓度<br>中位数 | "
      "独立样品数<br>τ=0.9999 | τ=0.999 | τ=0.99 | 重复倍数 |")
    w("|---|---|---|---|---|---|---|---|---|")
    summary = {}
    for e in eids:
        X = np.asarray(ent[e], float)
        c = np.asarray(conc[e], float)
        Xn = nrm(X)
        S = Xn @ Xn.T
        lv = np.unique(c)
        within, cross = [], []
        for v in lv:
            idx = np.where(c == v)[0]
            if len(idx) > 1:
                sub = S[np.ix_(idx, idx)]
                iu = np.triu_indices(len(idx), 1)
                within.append(sub[iu])
            oth = np.where(c != v)[0]
            if len(oth):
                cross.append(S[np.ix_(idx, oth)].ravel())
        within = np.concatenate(within) if within else np.array([np.nan])
        cross = np.concatenate(cross) if cross else np.array([np.nan])
        pg = proxy_groups(X)
        dup = len(X) / max(pg[0.9999], 1)
        summary[e] = dict(n=len(X), within_med=float(np.median(within)),
                          within_hi=float(np.mean(within > 0.999)),
                          cross_med=float(np.median(cross)),
                          pg={str(k): int(v) for k, v in pg.items()},
                          dup=float(dup))
        w(f"| {e} | {len(X)} | {np.median(within):.4f} | "
          f"{np.mean(within > 0.999):.2f} | {np.median(cross):.4f} | "
          f"{pg[0.9999]} | {pg[0.999]} | {pg[0.99]} | **{dup:.1f}×** |")

    # ---------------- L3 划分协议对比 ----------------
    w("\n## L3 划分协议：随机划分是否造成浓度泄漏？\n")
    w("现状协议为**随机划分**：同一浓度的副本会同时落进 train 与 test。\n")
    w("| 实体 | 随机划分下<br>测试浓度被训练集覆盖的比例 | 严格(浓度不相交)<br>可划分的测试浓度数 |")
    w("|---|---|---|")
    for e in eids:
        c = np.asarray(conc[e], float)
        lv = np.unique(c)
        # 随机划分：k=5 训练，测试 30%，抽样 200 次估计覆盖比例
        rng = np.random.default_rng(0)
        n = len(c)
        n_te = int(round(0.3 * n))
        cov = []
        for _ in range(200):
            perm = rng.permutation(n)
            te, pool = perm[:n_te], perm[n_te:]
            tr = pool[:5]
            cov.append(len(set(c[te]) & set(c[tr])) / max(len(set(c[te])), 1))
        cov = float(np.mean(cov))
        w(f"| {e} | {cov:.2f} | {len(lv)} |")

    w("\n## 判据\n")
    hi = [e for e in eids if summary[e]["within_hi"] > 0.5]
    dup_hi = [e for e in eids if summary[e]["dup"] >= 3]
    w(f"- **P1 技术重复**：组内相似度 >0.999 的占比 >0.5 的实体 = "
      f"{hi if hi else '无'}")
    w(f"- **P1' 重复倍数 ≥3×** 的实体 = {dup_hi if dup_hi else '无'}")
    w(f"- 平均重复倍数 = "
      f"{np.mean([summary[e]['dup'] for e in eids]):.1f}× "
      f"（1× 表示全部独立）")
    w("\n> 若存在技术重复/近重复，则 exp63 中 rf/svm 的 tpr5=1.000 应解释为"
      "**记忆同一浓度的副本**，而非泛化能力；必须用浓度不相交协议重测。")

    with open(os.path.join(OUT, "exp65_p6_data_audit.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(OUT, "exp65_audit.json"), "w",
              encoding="utf-8") as f:
        json.dump({"struct": struct, "summary": summary}, f,
                  ensure_ascii=False, indent=1, default=str)
    print("\n".join(L))
    print("[exp65] 已写出 out/exp65_p6_data_audit.md")


if __name__ == "__main__":
    main()
